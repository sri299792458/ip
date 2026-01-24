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
1. On the UR teach pendant, go to **Settings → Network**
2. Set a static IP (e.g., `192.168.1.102`)
3. Note the subnet mask (typically `255.255.255.0`)

### 1.2 Configure Workstation IP
```bash
# Set static IP on same subnet
sudo ip addr add 192.168.1.100/24 dev eth0
```

### 1.3 Verify Connectivity
```bash
ping 192.168.1.102
```

---

## Step 2: Install Dependencies

### 2.1 Create Conda Environment
```bash
conda create -n ip python=3.10 -y
conda activate ip
```

### 2.2 Install PyTorch with CUDA
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 2.3 Install ur_rtde
```bash
pip install ur_rtde
```

### 2.4 Install RealSense SDK
```bash
# Install librealsense2
sudo apt-get install librealsense2-dkms librealsense2-utils librealsense2-dev

# Install Python bindings
pip install pyrealsense2
```

### 2.5 Install Open3D (for point cloud processing)
```bash
pip install open3d
```

### 2.6 Install Instant Policy
```bash
cd /path/to/instant_policy
pip install -e .
```

### 2.7 Install hotkey listener (demo control)
```bash
pip install pynput
```

---

## Step 3: Download Model Weights

### 3.1 SAM Weights (for initial segmentation)
```bash
mkdir -p checkpoints/sam
cd checkpoints/sam

# Download SAM ViT-B
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

### 3.2 XMem++ Weights (for video tracking)
```bash
mkdir -p checkpoints/xmem
cd checkpoints/xmem

# Download XMem++ checkpoint from the official repo
# See: https://github.com/hkchengrex/XMem2
wget https://github.com/hkchengrex/XMem2/releases/download/v1.0/XMem2.pth
```

### 3.3 Instant Policy Checkpoint
```bash
# Your trained Instant Policy model should be in:
# checkpoints/instant_policy/
#   ├── config.pkl
#   └── model.pt
```

---

## Step 4: Setup XMem++ Repository

XMem++ must be cloned and available on the Python path:

```bash
cd /path/to/instant_policy
git clone https://github.com/hkchengrex/XMem2.git XMem2-main

# Verify structure:
# instant_policy/
#   ├── XMem2-main/
#   │   ├── model/
#   │   ├── inference/
#   │   └── ...
#   └── ip/
```

---

## Step 5: Camera Calibration

### 5.1 Get Camera Serial Numbers
```bash
# List connected RealSense cameras
rs-enumerate-devices | grep "Serial Number"
```

### 5.2 Compute Camera-to-World Transform

You need `T_world_camera` (4x4 matrix) for each camera. Options:

**Option A: Hand-Eye Calibration**
1. Mount camera rigidly
2. Use a calibration pattern (e.g., ChArUco board)
3. Move robot to known poses, capture images
4. Compute transform using OpenCV or similar

**Option B: Manual Measurement**
1. Measure camera position relative to robot base
2. Measure camera orientation (may require AprilTags)
3. Construct 4x4 transform matrix

**Example Transform**:
```python
import numpy as np
from scipy.spatial.transform import Rotation

# Camera at [0.5, 0.0, 0.8] meters, looking down at 45°
position = np.array([0.5, 0.0, 0.8])
rotation = Rotation.from_euler('xyz', [-135, 0, 0], degrees=True)

T_world_camera = np.eye(4)
T_world_camera[:3, :3] = rotation.as_matrix()
T_world_camera[:3, 3] = position
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

gripper = RobotiqGripper(host="192.168.1.102", port=63352)
gripper.connect()
gripper.activate()
print("Gripper activated!")
gripper.open()
gripper.close()
gripper.disconnect()
```

---

## Step 7: Robot Setup

### 7.1 Load External Control Program
On the UR teach pendant:
1. Create a new program
2. Add **URCap → External Control** node
3. Set the host IP to your workstation (`192.168.1.100`)
4. Save as `external_control.urp`

### 7.2 Run the Program
Press **Play** on the teach pendant. The robot is now ready to receive RTDE commands.

### 7.3 Test RTDE Connection
```python
import rtde_receive
import rtde_control

rtde_r = rtde_receive.RTDEReceiveInterface("192.168.1.102")
print("TCP Pose:", rtde_r.getActualTCPPose())

rtde_c = rtde_control.RTDEControlInterface("192.168.1.102")
print("RTDE Control connected!")
```

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

# Camera transform (from Step 5)
T_world_camera = np.array([
    [1, 0, 0, 0.5],
    [0, -1, 0, 0.0],
    [0, 0, -1, 0.8],
    [0, 0, 0, 1],
])

config = DeploymentConfig(
    # Cameras
    camera_configs=[
        CameraConfig(
            serial="123456789",  # From Step 5.1
            T_world_camera=T_world_camera,
            width=640,
            height=480,
            fps=30,
            align_to_color=True,
        ),
    ],
    
    # Robot
    robot_ip="192.168.1.102",
    
    # Model
    model_path="./checkpoints/instant_policy",
    num_demos=2,
    num_traj_wp=10,
    num_diffusion_iters=4,
    
    # Segmentation
    segmentation=SegmentationConfig(
        enable=True,
        backend="xmem",
        sam_checkpoint_path="./checkpoints/sam/sam_vit_b_01ec64.pth",
        xmem_checkpoint_path="./checkpoints/xmem/XMem2.pth",
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
| XMem++ checkpoint | `checkpoints/xmem/XMem2.pth`                        |
| IP model          | `checkpoints/instant_policy/{config.pkl, model.pt}` |
| XMem++ source     | `XMem2-main/` (sibling to `ip/`)                    |
| Demos             | `demos/*.pkl`                                       |
