# Instant Policy Deployment Guide (RTDE, no ROS)

Complete deployment instructions for a lab computer running Ubuntu 25 with a UR5e.

---

## Prerequisites

- **OS**: Ubuntu 25 (ROS1 Noetic not required)
- **GPU**: NVIDIA GPU with CUDA support
- **Robot**: UR5e with Robotiq 2F-85 gripper
- **Cameras**: Intel RealSense D405/D435 or D415
- **Network**: PC and robot on same subnet

---

## 1. System Dependencies

```bash
sudo apt update
sudo apt install -y build-essential python3-dev
```

### RealSense SDK (librealsense)

```bash
sudo apt install -y librealsense2-dkms librealsense2-utils librealsense2-dev
```

---

## 2. Create Conda Environment

```bash
cd ~/Desktop/ip/instant_policy

# Create environment from yml (preferred)
conda env create -f environment.yml
conda activate ip_env

# Install PyG CUDA extensions
pip install pyg-lib -f https://data.pyg.org/whl/torch-2.2.0+cu118.html

# Install Instant Policy package
pip install -e .
```

---

## 3. Install Deployment Dependencies

```bash
conda activate ip_env

# UR RTDE
pip install ur_rtde

# RealSense Python bindings
pip install pyrealsense2

# Segment Anything Model (SAM)
pip install git+https://github.com/facebookresearch/segment-anything.git

# XMem++ dependencies
pip install progressbar2
```

---

## 4. Download Model Weights

### 4.1 Instant Policy Checkpoint

```bash
cd ~/Desktop/ip/instant_policy
mkdir -p checkpoints

# Required files:
#   checkpoints/model.pt
#   checkpoints/config.pkl
```

### 4.2 SAM Weights

```bash
mkdir -p weights
cd weights

# Download SAM ViT-B checkpoint
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

### 4.3 XMem++ Weights

```bash
cd ~/Desktop/ip/XMem2-main
mkdir -p saves

# Download XMem++ checkpoint
gdown 1QoChoCkFWMl93k0MknLcCEJMLQnCNhSm -O saves/XMem.pth
```

---

## 5. Robot Setup (UR5e)

- Set the robot to **Remote Control** mode.
- Confirm RTDE is enabled on the UR controller.
- Verify the robot IP (default often `192.168.1.102`).
- Ensure the Robotiq gripper is connected (port 63352).

---

## 6. Camera Calibration

You must obtain `T_world_camera` for each RealSense camera.

### 6.1 Find Camera Serials

```bash
rs-enumerate-devices | grep Serial
```

### 6.2 Calibration Options

1. AprilTag or ArUco calibration (recommended).
2. Hand-eye calibration (easy_handeye or equivalent).
3. Manual measurement (only for coarse setups).

The transform `T_world_camera` must be a 4x4 matrix that transforms points from
camera frame to the UR base frame.

---

## 7. Configure Deployment

Edit `instant_policy/ip/deployment.py`:

```python
def _build_default_config() -> DeploymentConfig:
    camera_configs = [
        CameraConfig(
            serial="ACTUAL_SERIAL_1",
            T_world_camera=np.array([
                [1, 0, 0, 0.5],
                [0, 1, 0, 0.0],
                [0, 0, 1, 0.7],
                [0, 0, 0, 1.0]
            ]),
        ),
        CameraConfig(
            serial="ACTUAL_SERIAL_2",
            T_world_camera=np.array([...]),
        ),
    ]

    config = DeploymentConfig(camera_configs=camera_configs)
    config.model_path = "./checkpoints"
    config.robot_ip = "192.168.1.102"

    # Enable XMem++ segmentation (SAM seeding required)
    config.segmentation.enable = True
    config.segmentation.backend = "xmem"
    config.segmentation.xmem_checkpoint_path = "../XMem2-main/saves/XMem.pth"
    config.segmentation.sam_checkpoint_path = "./weights/sam_vit_b_01ec64.pth"

    return config
```

---

## 8. Run Deployment

### 8.1 Collect a Demo

```bash
python -m ip.deployment --collect-demo --demo-out demo.pkl
```

### 8.2 Run Inference

```bash
python -m ip.deployment --demo demo.pkl
```

---

## 9. Troubleshooting

Common issues:

1) **RTDE connection failure**
   - Check robot IP and network.
   - Ensure the robot is in Remote Control mode.

2) **Robotiq connection failure**
   - Verify port 63352 is reachable.
   - Ensure gripper is powered and enabled.

3) **Empty point cloud**
   - Check camera serials and USB3 connection.
   - Confirm depth streams are enabled.

4) **Segmentation missing**
   - Confirm SAM and XMem weights paths.
   - Ensure CUDA is available for XMem++.
