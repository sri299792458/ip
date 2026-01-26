# Instant Policy Deployment: Step-by-Step Setup Guide

This guide walks through deploying Instant Policy on a **brand new system** from scratch. It covers hardware setup, software installation, calibration, and first run.

---

## Prerequisites

### Hardware Required
- **Robot**: UR5e with Robotiq 2F-85 gripper
- **Cameras**: Intel RealSense D405 or D435 (1-2 cameras)
- **Compute**: Linux workstation with NVIDIA GPU (CUDA required for XMem++)
- **Network**: Ethernet connection to robot (same subnet)

### Software Required
- Ubuntu 22.04 or 24.04
- NVIDIA drivers + CUDA 11.8+
- Python 3.10+
- Conda or Mamba

---

## Step 1: Network Setup

### 1.1 Configure Robot IP
On the UR teach pendant, go to **Settings → Network**:
- IP address: `10.33.55.90`
- Subnet mask: `255.255.255.0`
- Default gateway: `10.33.55.1`

### 1.2 Configure Workstation IP
```bash
# Set static IP on same subnet
sudo ip addr add 10.33.55.100/24 dev enp1s0f0
sudo ip link set enp1s0f0 up
```

### 1.3 Verify Connectivity
```bash
ping 10.33.55.90
```

---

## Step 2: Install Dependencies

### 2.1 Create Conda Environment from environment.yml
The Instant Policy repo includes a complete `environment.yml` with all dependencies:

```bash
cd /path/to/instant_policy
conda env create -f environment.yml
conda activate ip_env
```

This installs:
- Python 3.10
- PyTorch 2.2.0 with CUDA 11.8
- PyTorch Geometric (pyg) + cluster/scatter
- PyTorch Lightning
- NumPy, SciPy, Open3D, and more

### 2.2 Install PyG-lib (required for graph operations)
```bash
pip install pyg-lib -f https://data.pyg.org/whl/torch-2.2.0+cu118.html
```

### 2.3 Register the `ip` Package
```bash
cd /path/to/instant_policy
pip install -e .
```

> **Note**: This only registers the package name so you can `import ip` from anywhere.
> It shows "0 packages installed" because `setup.py` has no dependencies listed -
> all actual dependencies come from `environment.yml` (already installed in Step 2.1).

### 2.4 Install ur_rtde (for robot control)
```bash
pip install ur_rtde
```

### 2.5 Install Intel RealSense SDK + Viewer + Python bindings (Conda-local, shared-machine safe)
This flow keeps everything inside your conda env.
- Installs under `$CONDA_PREFIX`
- No DKMS / kernel modules
- No `/usr/local` install and no `/etc/udev` edits
- `realsense-viewer` is required for verification

#### 2.5.1 System build + GUI dependencies (safe global install)
```bash
sudo apt-get update
sudo apt-get install -y \
  git cmake build-essential pkg-config \
  libusb-1.0-0-dev libudev-dev libssl-dev \
  libgtk-3-dev libglfw3-dev \
  libgl1-mesa-dev libglu1-mesa-dev
```

#### 2.5.2 Build librealsense (RSUSB) into your conda env (includes viewer + tools + python modules)
```bash
conda activate ip_env

mkdir -p ~/src
cd ~/src
git clone https://github.com/IntelRealSense/librealsense.git
cd librealsense
git checkout v2.54.2

rm -rf build
mkdir build && cd build

cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$CONDA_PREFIX" \
  -DFORCE_RSUSB_BACKEND=ON \
  -DBUILD_EXAMPLES=ON \
  -DBUILD_GRAPHICAL_EXAMPLES=ON \
  -DBUILD_PYTHON_BINDINGS=ON \
  -DPYTHON_EXECUTABLE="$(which python)"

make -j"$(nproc)"
make install
```

#### 2.5.3 If build fails with uint64_t / `<cstdint>` (Ubuntu 24.04+ sometimes)
If you see errors mentioning `uint64_t` not declared in:
`third-party/rsutils/include/rsutils/version.h`, apply:
```bash
cd ~/src/librealsense
sed -i '/^#pragma once/a#include <cstdint>' third-party/rsutils/include/rsutils/version.h

cd build
make -j"$(nproc)"
make install
```

#### 2.5.4 Make `pyrealsense2` importable in the env
`make install` places Python extension modules into `$CONDA_PREFIX/OFF/`. Copy them into your env’s site-packages:
```bash
conda activate ip_env
SITEPKG="$(python -c 'import site; print(site.getsitepackages()[0])')"

cp -av "$CONDA_PREFIX/OFF/pyrealsense2"*.so* "$SITEPKG/"
cp -av "$CONDA_PREFIX/OFF/pybackend2"*.so* "$SITEPKG/" 2>/dev/null || true
cp -av "$CONDA_PREFIX/OFF/pyrsutils"*.so*   "$SITEPKG/" 2>/dev/null || true
```

#### 2.5.5 Ensure the env finds the correct runtime libs
```bash
conda activate ip_env

mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"
cat > "$CONDA_PREFIX/etc/conda/activate.d/realsense.sh" <<'EOF'
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
EOF

conda deactivate
conda activate ip_env
```

#### 2.5.6 Verify installation (viewer + CLI + Python)
```bash
conda activate ip_env

# 1) CLI sees devices
rs-enumerate-devices

# 2) Viewer launches (must work)
realsense-viewer

# 3) Python bindings import and detect cameras
python -c "import pyrealsense2 as rs; print('OK', rs.__version__); print('devices', len(rs.context().query_devices()))"
```

#### 2.5.7 Notes on permissions (shared systems)
If `rs-enumerate-devices` or `realsense-viewer` shows no device detected but cameras exist (e.g., visible under `/dev/video*`), you likely lack access to USB/video nodes.
Preferred shared-machine fix (admin-managed):
- Add your user to `plugdev` and `video`
- Avoid global udev rules unless the admin wants a standardized policy

#### 2.5.8 Optional sanity check
Confirm you are using the env-local binaries:
```bash
command -v realsense-viewer
ldd "$(which realsense-viewer)" | grep librealsense
```


### 2.6 Install hotkey listener (demo control)
```bash
pip install pynput
```

---

## Step 3: Download Model Weights

### 3.1 Instant Policy Pre-trained Weights
Use the official download script (requires `gdown`):

```bash
cd /path/to/instant_policy

# Install gdown if not already installed
pip install gdown

# Download pre-trained model
bash ip/scripts/download_weights.sh
```

This downloads from Google Drive to a `weights/` folder containing:
- `config.pkl` - Model configuration
- `model.pt` - Pre-trained weights

Alternatively, download manually from:
https://drive.google.com/drive/folders/1hfyQ0DhZ8sCLrrH7dmE4WLMIibxZGVpI

### 3.2 SAM Weights (for initial segmentation)
```bash
mkdir -p checkpoints/sam
cd checkpoints/sam

# Download SAM ViT-B (default for Instant Policy)
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```
Install the SAM Python package:
```bash
pip install "segment-anything @ git+https://github.com/facebookresearch/segment-anything.git"
```

### 3.3 XMem Weights (for video object tracking)
```bash
mkdir -p checkpoints/xmem
cd checkpoints/xmem

# Download XMem checkpoint (from official hkchengrex/XMem repo)
wget https://github.com/hkchengrex/XMem/releases/download/v1.0/XMem.pth
```

---

## Step 4: Setup XMem++ (XMem2) Repository

XMem++ (XMem2) must be cloned and available on the Python path.

> **Note**: The code expects the directory to be named `XMem2-main`.

```bash
cd /path/to/instant_policy

# Clone XMem2 from the correct repo (mbzuai-metaverse, NOT hkchengrex)
git clone https://github.com/mbzuai-metaverse/XMem2.git XMem2-main

# Put the XMem weights in XMem2-main/saves/
mkdir -p XMem2-main/saves
cp checkpoints/xmem/XMem.pth XMem2-main/saves/

# Verify structure:
# instant_policy/
#   ├── XMem2-main/
#   │   ├── model/
#   │   ├── inference/
#   │   ├── saves/
#   │   │   └── XMem.pth
#   │   └── ...
#   └── ip/
```

---

## Step 5: Camera Calibration

### 5.0 Hardware + Calibration Summary (Current Setup)

**Hardware Configuration**
- Robot gripper: Robotiq 2F-85
- Gripper TCP offset (Z): `0.162 m` (162 mm)
  - Set this in **UR Installation → General → TCP**.
- Camera 1 serial: `f1380660`
- Camera 2 serial: `f1371463`
- ArUco marker:
  - Dictionary: `DICT_6X6_50`
  - Tag ID: `5`
  - Physical size: `0.05 m` (50 mm)

**World Tag Measurements (Robot Base Frame)**
Measured by touching the ArUco marker corners with the closed gripper (TCP: Gripper, Feature: Base):
- Top-Left (TL): x = `-0.4646`, y = `0.5282`, z = `-0.2563`
- Top-Right (TR): x = `-0.4661`, y = `0.5283`, z = `-0.3048`
- Bottom-Left (BL): x = `-0.5143`, y = `0.5289`, z = `-0.2552`

Calculated `world_tag.json` (T_world_tag):
```
[[-0.0309, -0.9994,  0.0141, -0.4902]
 [ 0.0021,  0.0141,  0.9999,  0.5286]
 [-0.9995,  0.0309,  0.0016, -0.2800]
 [ 0.0000,  0.0000,  0.0000,  1.0000]]
```

**Camera Extrinsics (World → Camera)**
Command used:
```bash
python calibrate_realsense_aruco.py \
  --serial f1380660 \
  --serial f1371463 \
  --tag-dict DICT_6X6_50 \
  --tag-id 5 \
  --tag-size 0.05 \
  --world-tag-matrix world_tag.json
```

Calibration results (saved to `realsense_T_world_camera.json`):

Camera 1 (`f1380660`)
- Position (approx): X = `-0.44 m`, Y = `1.17 m`, Z = `0.086 m`
- Quality: `30 samples`, `0.21 px` reprojection error (excellent)
```
[[ 0.9952,  0.0067, -0.0978, -0.4402]
 [-0.0729, -0.6154, -0.7849,  1.1759]
 [-0.0654,  0.7882, -0.6119,  0.0865]
 [ 0.0000,  0.0000,  0.0000,  1.0000]]
```

Camera 2 (`f1371463`)
- Position (approx): X = `-0.51 m`, Y = `1.50 m`, Z = `-0.012 m`
- Quality: `30 samples`, `0.05 px` reprojection error (excellent)
```
[[ 0.9978,  0.0556, -0.0352, -0.5160]
 [-0.0283, -0.1209, -0.9923,  1.5065]
 [-0.0595,  0.9911, -0.1190, -0.0123]
 [ 0.0000,  0.0000,  0.0000,  1.0000]]
```

**File Locations**
- Tag calibration: `ip/deployment/world_tag.json`
- Camera calibration: `ip/deployment/calibration_outputs/realsense_T_world_camera.json`

**Validation (Click‑to‑World vs TCP)**
Example validation using `validate_click_point.py`:
```
Pixel (347,334) depth 0.7290 m
Camera point: [0.0237, 0.1144, 0.7290] m
World point : [-0.4871, 0.5316, -0.2710] m
TCP pose   : [-0.4901, 0.5291, -0.2786, -1.1912, -1.2087, -1.2095]
Delta (TCP - point): [-0.0030, -0.0024, -0.0077] m
```
This shows ~3 mm XY error and ~8 mm Z error, which is within expected RealSense depth noise.

### 5.1 Get Camera Serial Numbers
```bash
# List connected RealSense cameras
rs-enumerate-devices | grep "Serial Number"
```

---

## Step 6: Gripper Setup

### 6.1 Connect Gripper
The Robotiq 2F-85 connects via the UR tool connector. The deployment uses TCP socket communication on port 63352.

### 6.2 Enable URCap
On the UR teach pendant:
1. Go to **Installation → URCaps**
2. Enable the Robotiq gripper URCap
3. The gripper should now be accessible on the robot's IP at port 63352

### 6.3 Test Gripper Connection
```python
from ip.deployment.ur.robotiq_gripper import RobotiqGripper

gripper = RobotiqGripper(host="10.33.55.90", port=63352)
gripper.connect()
gripper.activate()
print("Gripper activated!")
gripper.open()
gripper.close()
gripper.disconnect()
```

---

## Step 7: Robot Setup

### 7.1 Test RTDE Connection
```python
import rtde_receive
import rtde_control

rtde_r = rtde_receive.RTDEReceiveInterface("10.33.55.90")
print("TCP Pose:", rtde_r.getActualTCPPose())

rtde_c = rtde_control.RTDEControlInterface("10.33.55.90")
print("RTDE Control connected!")
```
**Note**: If RTDE control fails, ensure the robot is in **Remote Control** and the motors are **ON** (brakes released).

---

## Step 8: Create Deployment Configuration

Create a configuration file or modify `ip/deployment.py`:

```python
import numpy as np
from ip.deployment.config import (
    DeploymentConfig,
    CameraConfig,
    SegmentationConfig,
    GripperConfig,
    RTDEControlConfig,
)

# Camera transforms (from Step 5)
T_world_camera_cam1 = np.array([
    [0.9952, 0.0067, -0.0978, -0.4402],
    [-0.0729, -0.6154, -0.7849, 1.1759],
    [-0.0654, 0.7882, -0.6119, 0.0865],
    [0.0, 0.0, 0.0, 1.0],
])
T_world_camera_cam2 = np.array([
    [0.9978, 0.0556, -0.0352, -0.5160],
    [-0.0283, -0.1209, -0.9923, 1.5065],
    [-0.0595, 0.9911, -0.1190, -0.0123],
    [0.0, 0.0, 0.0, 1.0],
])

config = DeploymentConfig(
    # Cameras
    camera_configs=[
        CameraConfig(
            serial="f1380660",
            T_world_camera=T_world_camera_cam1,
            width=640,
            height=480,
            fps=30,
            align_to_color=True,
        ),
        CameraConfig(
            serial="f1371463",
            T_world_camera=T_world_camera_cam2,
            width=640,
            height=480,
            fps=30,
            align_to_color=True,
        ),
    ],
   
    # Robot
    robot_ip="10.33.55.90",
   
    # Model
    model_path="./checkpoints/ip",
    num_demos=2,
    num_traj_wp=10,
    num_diffusion_iters=4,
   
    # Segmentation
    segmentation=SegmentationConfig(
        enable=True,
        backend="xmem",
        sam_checkpoint_path="./checkpoints/sam/sam_vit_b_01ec64.pth",
        xmem_checkpoint_path="./checkpoints/xmem/XMem.pth",
        xmem_init_with_sam=True,
    ),
   
    # Gripper
    gripper=GripperConfig(
        enable=True,
        host=None,  # Uses robot_ip
        port=63352,
        open_position=0,
        closed_position=255,
    ),
   
    # Control
    rtde=RTDEControlConfig(
        control_mode="moveL",  # or "servoL"
        move_speed=0.1,
        move_acceleration=0.5,
    ),
   
    # Safety (adjust for your workspace)
    safety=None,  # Uses defaults, or provide SafetyLimits
   
    # Execution
    execute_until_grip_change=True,
    device="cuda:0",
)
```

---

## Step 9: Collect Demonstrations

### 9.1 Start Demo Collection
```bash
python -m ip.deployment --collect-demo --demo-out demos/task1_demo1.pkl
```

### 9.2 Recording Process
1. Move robot to start position
2. Press **ENTER** to begin (robot enters freedrive mode)
3. Kinesthetically demonstrate the task
4. Use `o` to open and `c` to close the gripper while recording
5. Press `q` or **ESC** to stop recording

### 9.3 Collect Multiple Demos
Repeat for 2-5 demonstrations per task:
```bash
python -m ip.deployment --collect-demo --demo-out demos/task1_demo2.pkl
python -m ip.deployment --collect-demo --demo-out demos/task1_demo3.pkl
```

---

## Step 10: Run Deployment

### 10.1 Single Demo
```bash
python -m ip.deployment --demo demos/task1_demo1.pkl
```

### 10.2 Multiple Demos
```bash
python -m ip.deployment --demo demos/task1_demo1.pkl demos/task1_demo2.pkl
```

### 10.3 Python API
```python
from ip.deployment.orchestrator import InstantPolicyDeployment
import pickle

# Load demos
demos = []
for path in ["demos/task1_demo1.pkl", "demos/task1_demo2.pkl"]:
    with open(path, "rb") as f:
        demos.append(pickle.load(f))

# Run deployment
deployment = InstantPolicyDeployment(config)
success = deployment.run(demos, max_steps=100)
print("Deployment success:", success)
```

---

## Debug Pipeline (recommended for every new setup)

### D1. Debug segmentation masks (live)
Use this to verify SAM/XMem++ masks look correct before collecting demos.
```bash
python -m ip.deployment.debug_segmentation
```
Outputs images to `ip/deployment/debug_outputs`.

### D2. Debug XMem++ tracking (live, manual seed)
This verifies tracking over time (objects must move).
```bash
python -m ip.deployment.debug_xmem_tracking --frames 120 --sleep 0.2 --out-dir ip/deployment/debug_outputs
```

### D3. Inspect a collected demo (stats + visual PLYs)
This checks the saved demo itself (point clouds + poses + grips).
```bash
python -m ip.deployment.debug_demo \
  --demo demos/task1_demo1.pkl \
  --save-frames \
  --save-ee \
  --out-dir ip/deployment/debug_outputs \
  --workspace-min -0.9008 0.2936 -0.4819 \
  --workspace-max -0.1227 0.5751 0.5293
```
This writes a few PLYs (world + EE frame) into `ip/deployment/debug_outputs` for visual inspection.

### D3.1 Play back demo frames (Viser, web-based)
This plays the demo point clouds in a browser-based viewer.
```bash
pip install viser
python -m ip.deployment.view_demo_pcds --demo demos/task1_demo1.pkl --stride 2 --fps 20
```
Open the URL shown in the terminal (default is `http://localhost:8080`).

Optional TCP overlay:
```bash
python -m ip.deployment.view_demo_pcds --demo demos/task1_demo1.pkl --show-tcp
```
Policy-view (what the model actually sees: EE frame + subsampled points):
```bash
python -m ip.deployment.view_demo_pcds --demo demos/task1_demo1.pkl --policy-view --use-config
```
Auto-fit view + axes (useful if clouds drift out of view):
```bash
python -m ip.deployment.view_demo_pcds --demo demos/task1_demo1.pkl --policy-view --use-config --auto-fit --show-axes
```

### D4. Manual-seed workflow (if SAM is unreliable)
```bash
python -m ip.deployment --manual-seed --demo demos/task1_demo1.pkl
```
Manual seeding lets you select multiple objects per camera and bypass SAM initialization.

---

## Step 11: Troubleshooting

### Issue: "RTDE connection failed"
- Verify robot IP is reachable (`ping`)
- Ensure "External Control" program is running on teach pendant
- Check firewall settings on workstation

### Issue: "Gripper did not activate"
- Verify Robotiq URCap is installed and enabled
- Check gripper cable connection
- Try power-cycling the gripper

### Issue: "No valid point clouds captured"
- Verify cameras are connected (`rs-enumerate-devices`)
- Check camera serial numbers in config
- Ensure depth stream is working (`realsense-viewer`)

### Issue: "XMem++ requires CUDA"
- XMem++ only runs on GPU
- Verify CUDA is installed: `nvidia-smi`
- Ensure PyTorch sees CUDA: `torch.cuda.is_available()`

### Issue: "Robot moves in wrong direction"
- **Most common cause**: Incorrect `T_world_camera`
- Verify camera calibration
- Visualize point cloud in world frame to debug

### Issue: "Safety limit exceeded"
- Adjust `SafetyLimits` workspace bounds
- Increase `max_translation` / `max_rotation` if needed (carefully)

---

## Step 12: Best Practices

### Demo Collection Tips
- Demonstrate slowly and smoothly
- Keep gripper motions intentional (pause before open/close)
- Collect 3-5 demos with slight variations
- Include both successful and recovery motions

### Execution Tips
- Start with low speeds (`move_speed=0.05`)
- Keep hand near E-stop during testing
- Test on simple tasks first (e.g., pick-and-place)

### Safety Checklist
- [ ] E-stop accessible
- [ ] Workspace bounds configured
- [ ] Per-step limits set
- [ ] Tested in freedrive before autonomous

---

## Quick Reference: File Locations

| Item              | Path                                                |
| ----------------- | --------------------------------------------------- |
| Deployment config | `ip/deployment.py` or custom script                 |
| SAM checkpoint    | `checkpoints/sam/sam_vit_b_01ec64.pth`              |
| XMem++ checkpoint | `checkpoints/xmem/XMem.pth`                         |
| IP model          | `checkpoints/ip/{config.pkl, model.pt}`             |
| XMem++ source     | `XMem2-main/` (sibling to `ip/`)                    |
| Demos             | `demos/*.pkl`                                       |
