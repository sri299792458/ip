# Instant Policy Deployment Guide (ROS + Zeus)

Complete deployment instructions for a lab computer running Ubuntu 20.04 with ROS Noetic and UR5e robots.

---

## Prerequisites

- **OS**: Ubuntu 20.04 LTS (required for ROS Noetic)
- **GPU**: NVIDIA GPU with CUDA 11.8 support
- **Robot**: UR5e with Robotiq 2F-85 gripper
- **Cameras**: Intel RealSense D415/D435
- **ROS**: Noetic with MoveIt installed

---

## 1. Install System Dependencies

```bash
# NVIDIA Driver + CUDA 11.8
sudo apt update
sudo apt install -y nvidia-driver-535 nvidia-cuda-toolkit

# ROS Noetic dependencies
sudo apt install -y python3-rosdep python3-rosinstall python3-rosinstall-generator \
    python3-wstool build-essential python3-catkin-tools

# RealSense SDK
sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-key F6E65AC044F831AC80A06380C8B3A55A6F3EFCDE
sudo add-apt-repository "deb https://librealsense.intel.com/Debian/apt-repo $(lsb_release -cs) main"
sudo apt update
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

## 3. Install Deployment-Specific Dependencies

```bash
conda activate ip_env

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

## 5. ROS + Zeus Setup

### 5.1 Build Zeus Workspace

```bash
# Source ROS
source /opt/ros/noetic/setup.bash

# Build zeus-master as a catkin workspace
cd ~/Desktop/ip/zeus-master
catkin_make

# Source the workspace
source devel/setup.bash
```

### 5.2 Configure ROS Environment

Add to `~/.bashrc`:

```bash
source /opt/ros/noetic/setup.bash
source ~/Desktop/ip/zeus-master/devel/setup.bash
export ROS_MASTER_URI=http://localhost:11311
```

---

## 6. Camera Calibration

You must obtain `T_world_camera` for each RealSense camera.

### 6.1 Find Camera Serials

```bash
rs-enumerate-devices | grep Serial
```

### 6.2 Calibration Options

1. Hand-eye calibration with AprilTags.
2. MoveIt calibration using `moveit_calibration` package.
3. Manual measurement for coarse setups.

The transform `T_world_camera` must be a 4x4 matrix that transforms points from
camera frame to MoveIt world frame.

---

## 7. Configure Deployment

Edit `ip/deployment.py`:

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
            ])
        ),
        CameraConfig(
            serial="ACTUAL_SERIAL_2",
            T_world_camera=np.array([...])
        ),
    ]

    config = DeploymentConfig(camera_configs=camera_configs)
    config.model_path = "./checkpoints"

    # Enable XMem++ segmentation
    config.segmentation.enable = True
    config.segmentation.backend = "xmem"
    config.segmentation.xmem_checkpoint_path = "../XMem2-main/saves/XMem.pth"
    config.segmentation.sam_checkpoint_path = "./weights/sam_vit_b_01ec64.pth"

    # Arm selection: "lightning" (left) or "thunder" (right)
    config.arm = "lightning"
    return config
```

---

## 8. Launch Procedure

### Terminal 1: ROS Core + MoveIt

```bash
roscore
```

### Terminal 2: Zeus bringup

```bash
cd ~/Desktop/ip/zeus-master
source devel/setup.bash
roslaunch zeus_bringup ur5e.launch
```

---

## 9. Run Deployment

### 9.1 Collect a Demo

```bash
python -m ip.deployment --collect-demo --demo-out demo.pkl
```

### 9.2 Run Inference

```bash
python -m ip.deployment --demo demo.pkl
```

---

## 10. Troubleshooting

Common issues:

1) **MoveIt planning fails**
   - Verify MoveIt is receiving robot state.
   - Check collisions in the planning scene.

2) **Gripper state missing**
   - Ensure Robotiq driver is publishing `Robotiq2FGripperRobotInput`.

3) **Empty point cloud**
   - Check camera serials and USB3 connection.
   - Confirm depth streams are enabled.

4) **Segmentation missing**
   - Confirm SAM and XMem weights paths.
   - Ensure CUDA is available for XMem++.
