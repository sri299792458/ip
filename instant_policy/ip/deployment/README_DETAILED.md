# Instant Policy Deployment: Exhaustive Technical Walkthrough

This document provides a comprehensive, **file-by-file** technical walkthrough of the Instant Policy deployment module. It is designed for engineers who need to understand, modify, or deploy the Instant Policy model on real UR5e hardware.

> [!IMPORTANT]
> **Key Architectural Note**: This deployment stack is **ROS-free**. It uses `ur_rtde` for robot control, `pyrealsense2` for camera capture, and direct socket communication for the Robotiq gripper.

---

## Overview

The deployment module enables real-world execution of Instant Policy on UR5e robots with Intel RealSense cameras and Robotiq grippers. It handles:
- **Perception**: RGB-D capture, XMem++ segmentation, point cloud generation
- **State**: Robot pose and gripper state via RTDE
- **Control**: Motion execution with safety gating
- **Demo Collection**: Kinesthetic teaching for in-context learning

---

## Directory Structure

```
ip/deployment/
├── __init__.py           # Package exports
├── config.py             # Configuration dataclasses
├── orchestrator.py       # Main deployment loop
├── control/
│   ├── __init__.py
│   ├── action_executor.py   # Safety-gated action execution
│   └── ur_rtde_control.py   # RTDE motion commands
├── demo/
│   ├── __init__.py
│   └── demo_collector.py    # Kinesthetic demo capture
├── perception/
│   ├── __init__.py
│   ├── realsense_perception.py  # RealSense RGB-D capture
│   ├── sam_segmentation.py      # SAM segmenter
│   └── xmem_segmentation.py     # XMem++ video tracker
├── state/
│   ├── __init__.py
│   └── ur_rtde_state.py     # RTDE state reading
└── ur/
    ├── __init__.py
    └── robotiq_gripper.py   # Robotiq socket driver
```

---

## 1. Configuration: `config.py` (77 lines)

**Purpose**: Defines all configuration dataclasses for the deployment system.

### Dataclasses

| Dataclass            | Description                                                                                    |
| -------------------- | ---------------------------------------------------------------------------------------------- |
| `CameraConfig`       | Per-camera settings: serial, `T_world_camera` (4x4), resolution, FPS, depth-to-color alignment |
| `SegmentationConfig` | Segmentation backend (`"sam"` or `"xmem"`), checkpoint paths, SAM parameters                   |
| `GripperConfig`      | Gripper host/port, open/closed positions (0-255), speed & force                                |
| `RTDEControlConfig`  | Control frequency (500Hz), mode (`"moveL"` or `"servoL"`), motion parameters                   |
| `DeploymentConfig`   | Main config combining all above + model path, num_demos, num_diffusion_iters, etc.             |

### Key Defaults

```python
robot_ip: str = "192.168.1.102"
num_demos: int = 2
num_traj_wp: int = 10  # Waypoints per demo
num_diffusion_iters: int = 4
pcd_num_points: int = 2048
```

---

## 2. Main Orchestrator: `orchestrator.py` (191 lines)

**Purpose**: The main deployment class that coordinates perception, state, control, and model inference.

### Class: `InstantPolicyDeployment`

**Constructor Flow**:
1. Build `SAMSegmenter` or `XMemOnlineSegmenter` based on config
2. Initialize `RealSensePerception` with camera configs and segmenter
3. Connect `RobotiqGripper` via socket (port 63352)
4. Connect `URRTDEControl` and `URRTDEState`
5. Build `ActionExecutor` with safety limits
6. Load `GraphDiffusion` model from checkpoint

**Key Methods**:

| Method                         | Description                                                                                  |
| ------------------------------ | -------------------------------------------------------------------------------------------- |
| `_load_model(model_path, ...)` | Loads `GraphDiffusion` checkpoint, sets `batch_size=1`, reinitializes graphs for N demos     |
| `_prepare_demos(demos)`        | Converts raw demos to model format via `sample_to_cond_demo()`. Pads if fewer than required. |
| `run(demos, max_steps)`        | Main inference loop (see below)                                                              |
| `_horizon_until_grip_change()` | Finds execution horizon before gripper state changes                                         |

### Main Loop (`run()`)

```python
for k in range(max_steps):
    # 1. Get current state
    T_w_e = self.state.get_T_w_e()
    grip = 1.0 if self.state.get_gripper_state() >= 0.5 else 0.0
    
    # 2. Capture world-frame point cloud
    pcd_w = self.perception.capture_pcd_world(use_segmentation=True)
    
    # 3. Transform to EE frame and subsample
    pcd_ee = transform_pcd(subsample_pcd(pcd_w, 2048), np.linalg.inv(T_w_e))
    
    # 4. Build model input
    full_sample = {"demos": prepared_demos, "live": {"obs": [pcd_ee], ...}}
    data = save_sample(full_sample, None)
    
    # 5. Cache demo embeddings (step 0 only)
    if k == 0:
        self._demo_embds, self._demo_pos = self.model.model.get_demo_scene_emb(data)
    
    # 6. Compute live embeddings
    data.live_scene_node_embds, data.live_scene_node_pos = self.model.model.get_live_scene_emb(data)
    
    # 7. Run diffusion inference
    actions, grips = self.model.test_step(data, 0)
    
    # 8. Execute actions with safety gating
    success, steps, error = self.executor.execute_actions(actions, grips, T_w_e, horizon)
```

---

## 3. Perception: `perception/`

### 3.1 `realsense_perception.py` (175 lines)

**Purpose**: RealSense RGB-D capture and point cloud generation.

**Class: `RealSensePerception`**

**Key Methods**:

| Method                                            | Description                                                                         |
| ------------------------------------------------- | ----------------------------------------------------------------------------------- |
| `__init__(camera_configs, segmenter, voxel_size)` | Starts RealSense pipelines for each camera, extracts intrinsics `K` and depth scale |
| `capture_pcd_world(use_segmentation)`             | Main capture function (see below)                                                   |
| `_voxel_downsample(points, voxel_size)`           | Open3D voxel downsampling                                                           |

**Capture Pipeline**:
```
For each camera:
  1. Wait for frames
  2. Align depth to color (if enabled)
  3. Get depth (meters) and color (RGB)
  4. Get segmentation mask (if enabled)
  5. Apply mask to depth: depth *= mask
  6. Back-project to camera frame: xyz_c = inv(K) @ depth
  7. Transform to world: xyz_w = T_world_camera @ xyz_c
  
Concatenate all camera clouds -> optional voxel downsample -> return
```

**Helper Function: `_get_xyz(depth_m, K)`**
- Standard pinhole back-projection
- Returns `[3, N]` points in camera frame

---

### 3.2 `xmem_segmentation.py` (149 lines)

**Purpose**: XMem++ video object segmentation with SAM seeding.

**Class: `XMemOnlineSegmenter`**

**Architecture**:
- One `InferenceCore` per camera (multi-camera support)
- SAM seeds the first frame, then XMem++ tracks

**Key Methods**:

| Method                                 | Description                                                        |
| -------------------------------------- | ------------------------------------------------------------------ |
| `segment_camera(rgb, camera_index)`    | Main entry point. Initializes with SAM on first call, then tracks. |
| `_initialize(camera_index, rgb, mask)` | Puts SAM mask into XMem++ permanent memory                         |
| `_track(camera_index, rgb)`            | Runs `InferenceCore.step()`, returns binary mask                   |
| `_prepare_image(rgb)`                  | Converts to normalized tensor                                      |
| `_prepare_mask(mask)`                  | Converts mask to float tensor                                      |

**Initialization Flow**:
```
1. First frame: SAM segments image -> binary mask
2. Clear XMem++ memory
3. Set labels = [1] (single object)
4. Put (image, mask) into permanent memory
5. Mark camera as initialized
```

---

### 3.3 `sam_segmentation.py` (referenced)

**Purpose**: Segment Anything Model (SAM) wrapper for single-image segmentation.

**Class: `SAMSegmenter`**
- Uses `SamAutomaticMaskGenerator`
- Parameters: `points_per_side`, `pred_iou_thresh`, `stability_score_thresh`, `min_mask_region_area`
- `select_largest=True`: Returns only the largest connected component

---

## 4. State: `state/ur_rtde_state.py` (44 lines)

**Purpose**: Reads robot state via RTDE.

**Class: `URRTDEState`**

| Method                | Description                                                                               |
| --------------------- | ----------------------------------------------------------------------------------------- |
| `connect(robot_ip)`   | Creates `RTDEReceiveInterface`                                                            |
| `get_T_w_e()`         | Returns 4x4 EE pose in world/base frame. Converts axis-angle to rotation matrix.          |
| `get_gripper_state()` | Returns normalized gripper position [0, 1] via `RobotiqGripper.get_position_normalized()` |

**Pose Conversion**:
```python
pose = rtde.getActualTCPPose()  # [x, y, z, rx, ry, rz]
T[:3, 3] = pose[:3]  # position
T[:3, :3] = Rotation.from_rotvec(pose[3:]).as_matrix()  # rotation
```

---

## 5. Control: `control/`

### 5.1 `ur_rtde_control.py` (78 lines)

**Purpose**: Sends motion commands via RTDE.

**Class: `URRTDEControl`**

| Method                                       | Description                                                                   |
| -------------------------------------------- | ----------------------------------------------------------------------------- |
| `connect(robot_ip, control_config)`          | Creates `RTDEControlInterface` at configured frequency                        |
| `execute_pose(T_w_e)`                        | Converts 4x4 to UR pose `[x, y, z, rx, ry, rz]`, executes `moveL` or `servoL` |
| `execute_gripper(command)`                   | Opens or closes gripper based on command > 0.5                                |
| `enable_freedrive()` / `disable_freedrive()` | For kinesthetic teaching                                                      |

**Control Modes**:
- **`moveL`**: Blocking, robust, slower
- **`servoL`**: Non-blocking streaming, faster, requires tuning (`servo_time`, `servo_lookahead`, `servo_gain`)

---

### 5.2 `action_executor.py` (61 lines)

**Purpose**: Safety-gated action execution.

**Dataclass: `SafetyLimits`**

| Parameter         | Default             | Description               |
| ----------------- | ------------------- | ------------------------- |
| `workspace_min`   | `[0.2, -0.4, 0.05]` | XYZ lower bounds (meters) |
| `workspace_max`   | `[0.7, 0.4, 0.5]`   | XYZ upper bounds (meters) |
| `max_translation` | `0.01`              | 1 cm per step             |
| `max_rotation`    | `3°`                | 3 degrees per step        |

**Class: `ActionExecutor`**

| Method                                                    | Description                                                   |
| --------------------------------------------------------- | ------------------------------------------------------------- |
| `execute_actions(actions, grips, T_w_e_initial, horizon)` | Executes up to `horizon` actions with safety checks           |
| `_check_safety(T_prev, T_next)`                           | Validates workspace bounds, translation limit, rotation limit |

**Action Composition**:
```python
# Actions are cumulative relative to inference-time pose
T_w_e_target_j = T_w_e_initial @ actions[j]
```

---

## 6. Demo Collection: `demo/demo_collector.py` (64 lines)

**Purpose**: Kinesthetic demonstration capture.

**Class: `DemoCollector`**

| Method                                      | Description                                         |
| ------------------------------------------- | --------------------------------------------------- |
| `collect_kinesthetic(task_name, rate_hz)`   | Enables freedrive, captures at 10Hz, stops on ENTER |
| `prepare_for_model(raw_demo, num_traj_wp)`  | Calls `sample_to_cond_demo()`                       |
| `save_demo(demo, path)` / `load_demo(path)` | Pickle serialization                                |

**Collection Loop**:
```python
# At 10Hz:
pcd_w = perception.capture_pcd_world()
T_w_e = state.get_T_w_e()
grip = 1.0 if state.get_gripper_state() >= 0.5 else 0.0
frames["pcds"].append(pcd_w)
frames["T_w_es"].append(T_w_e)
frames["grips"].append(grip)
```

---

## 7. Gripper: `ur/robotiq_gripper.py` (116 lines)

**Purpose**: Robotiq 2F-85 gripper control via socket.

**Class: `RobotiqGripper`**

**Protocol**: ASCII commands over TCP port 63352.
- `SET VAR VALUE\n` -> `ack`
- `GET VAR\n` -> `VAR value`

**Variables**:

| Variable | Description                      |
| -------- | -------------------------------- |
| `ACT`    | Activation (0=reset, 1=activate) |
| `GTO`    | Go to command                    |
| `POS`    | Position (0=open, 255=closed)    |
| `SPE`    | Speed (0-255)                    |
| `FOR`    | Force (0-255)                    |
| `STA`    | Status (3=active)                |

**Key Methods**:

| Method                         | Description                                     |
| ------------------------------ | ----------------------------------------------- |
| `connect()`                    | Opens TCP socket                                |
| `activate()`                   | Reset -> activate -> verify status=3            |
| `move(position, speed, force)` | Sets SPE, FOR, POS, GTO=1                       |
| `open()` / `close()`           | Convenience wrappers                            |
| `get_position_normalized()`    | Returns [0, 1] based on open/closed calibration |

---

## 8. Data Flow Summary

```
┌─────────────────────────────────────────────────────────────────────┐
│                        INITIALIZATION                                │
├─────────────────────────────────────────────────────────────────────┤
│  1. Load DeploymentConfig                                           │
│  2. Build XMemOnlineSegmenter (loads SAM + XMem++ checkpoints)      │
│  3. Start RealSense pipelines (extract K, depth_scale)             │
│  4. Connect RobotiqGripper (socket port 63352)                      │
│  5. Connect RTDE Control + Receive                                  │
│  6. Load GraphDiffusion model                                       │
│  7. Prepare demos via sample_to_cond_demo()                         │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                         MAIN LOOP                                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────┐   ┌─────────────────┐   ┌──────────────────────┐  │
│  │  RealSense  │──►│  XMem++ Mask    │──►│  Back-project        │  │
│  │  RGB-D      │   │  (per camera)   │   │  depth * mask → xyz_c│  │
│  └─────────────┘   └─────────────────┘   └──────────────────────┘  │
│                                                      │               │
│                                                      ▼               │
│  ┌─────────────┐   ┌─────────────────┐   ┌──────────────────────┐  │
│  │  RTDE       │──►│  T_w_e (4x4)    │   │  xyz_w = T_world_cam │  │
│  │  Receive    │   │  from rotvec    │   │          @ xyz_c     │  │
│  └─────────────┘   └─────────────────┘   └──────────────────────┘  │
│         │                  │                         │               │
│         ▼                  ▼                         ▼               │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │  xyz_ee = inv(T_w_e) @ xyz_w   →   subsample to 2048 points    ││
│  └─────────────────────────────────────────────────────────────────┘│
│                                │                                     │
│                                ▼                                     │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │  save_sample() → GraphDiffusion.test_step() → actions, grips   ││
│  └─────────────────────────────────────────────────────────────────┘│
│                                │                                     │
│                                ▼                                     │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │  ActionExecutor: safety check → RTDE moveL/servoL → gripper    ││
│  └─────────────────────────────────────────────────────────────────┘│
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 9. Key Constraints (from Instant Policy paper)

| Constraint            | Value               | Enforced By                           |
| --------------------- | ------------------- | ------------------------------------- |
| Point cloud frame     | EE frame            | `transform_pcd(pcd_w, inv(T_w_e))`    |
| Point cloud size      | 2048 points         | `subsample_pcd()`                     |
| Demo waypoints        | 10                  | `sample_to_cond_demo(num_traj_wp=10)` |
| Prediction horizon    | 8                   | Model config                          |
| Per-step translation  | ≤ 1 cm              | `SafetyLimits.max_translation`        |
| Per-step rotation     | ≤ 3°                | `SafetyLimits.max_rotation`           |
| Action interpretation | Cumulative relative | `T_target = T_initial @ action`       |

---

## 10. Usage

### Collect Demos
```bash
python -m ip.deployment --collect-demo --demo-out demo.pkl
```

### Run Deployment
```bash
python -m ip.deployment --demo demo.pkl
```

### Python API
```python
from ip.deployment.config import DeploymentConfig, CameraConfig
from ip.deployment.orchestrator import InstantPolicyDeployment

config = DeploymentConfig(
    camera_configs=[CameraConfig(serial="...", T_world_camera=T)],
    robot_ip="192.168.1.102",
    model_path="./checkpoints",
)

deployment = InstantPolicyDeployment(config)
deployment.run(demos=[raw_demo_dict])
```
