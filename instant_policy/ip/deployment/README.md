Instant Policy Deployment (Zeus + RealSense + XMem++)
=====================================================

This document explains the deployment code from first principles. It starts by
defining what Instant Policy requires at test time, enumerates what Zeus and
camera-utils already provide, and then shows exactly how the deployment code
bridges the gaps. Nothing is assumed: every coordinate frame, tensor shape,
transform, and policy decision is spelled out.

Table of contents
-----------------
1. First principles: the Instant Policy test-time contract
2. What Zeus already provides (and how we use it)
3. What camera-utils already provides (and how we use it)
4. Data contract and sample structure
5. Coordinate frames and transforms
6. Perception pipeline (capture -> mask -> point cloud -> fuse)
7. Segmentation: XMem++ tracking and SAM seeding
8. State estimation and gripper mapping
9. Control and safety gating
10. Demo collection and conversion
11. Model inference loop and execution policy
12. Initialization and runtime sequence diagram
13. Shapes, units, and normalization
14. Configuration reference
15. Setup and run instructions
16. Debug checklist and failure modes
17. File map


1. First principles: the Instant Policy test-time contract
----------------------------------------------------------
Instant Policy is an in-context imitation model. It does not update weights at
test time. Instead, you provide a small set of demonstrations (context) and a
current observation, and it predicts a short horizon of actions.

The model is trained under strict assumptions. Deployment must reproduce those
assumptions as closely as possible.

1.1 Required observation inputs (per time step)
- Segmented point cloud Pt of the scene, expressed in the end-effector frame E.
- End-effector pose in world/base frame W: T_w_e (4x4).
- Gripper state sg in {0, 1} (0 = closed, 1 = open).

1.2 Required demonstration context
Context consists of N demos, each downsampled to L waypoints:
- Each waypoint has a point cloud in EE frame, a T_w_e in world frame, and
  a gripper state in {0, 1}.

1.3 Action outputs
- A sequence of relative SE(3) transforms, actions[j], each in EE frame.
  Each actions[j] represents the transform from the current pose at inference
  time to the future pose at step j (i.e., cumulative relative to the initial
  pose for that inference step).
- A sequence of gripper commands, grips[j], in {-1, 1}.

1.4 Critical rules (must be correct)
- Point clouds must be expressed in EE frame.
- Actions are relative to the pose at inference time. For step j:
  T_w_e_target_j = T_w_e_initial @ actions[j]

1.5 Training distribution constraints (paper)
- Per-step translation: approx 0.01 m.
- Per-step rotation: approx 3 degrees.
- Point clouds are downsampled to 2048 points.
- Demos are downsampled to L = 10 waypoints.
- Prediction horizon T = 8.

The deployment code enforces these constraints wherever possible.


2. What Zeus already provides (and how we use it)
-------------------------------------------------
Zeus is the lab's UR5e stack (zeus-master). It already provides reliable ROS,
MoveIt, and Robotiq integration. We reuse it instead of re-implementing robot
control. This keeps safety behavior consistent with the lab stack.

2.1 Relevant Zeus modules
- `zeus-master/common/move_group_interface.py`
  - `pose_from_transform(T)` -> ROS Pose
  - `transform_from_pose(pose)` -> 4x4 matrix
  - `plan_cartesian_path(waypoints, move_group)`
  - `execute_plan(plan, move_group)`
  - `open_gripper(left=True)` / `close_gripper(left=True)`
  - `gripper_wait()` for blocking gripper feedback

2.2 What we wrap
In `instant_policy/ip/deployment/state/zeus_state.py`:
- We call `get_current_pose()` on the MoveIt group.
- We convert to 4x4 with `transform_from_pose`.

In `instant_policy/ip/deployment/control/zeus_control.py`:
- We convert a 4x4 target to a Pose with `pose_from_transform`.
- We plan a cartesian path and execute it.
- We open/close the Robotiq gripper using existing methods.

Deployment-specific guard:
- We construct `MoveGroupInterface(enable_right=False)` when using the left arm
  so the right arm is not homed or commanded during initialization.

2.3 Important Zeus assumptions
- `MoveGroupInterface` initializes ROS internally.
- It homes both arms at construction time.
- MoveIt planning is blocking (not 10 Hz real-time).
- The base frame used by MoveIt is the "world" for this pipeline.

We do not assume any additional control features beyond these.


3. What camera-utils already provides (and how we use it)
---------------------------------------------------------
We use direct RealSense capture via `pyrealsense2` for low latency, then reuse
utility functions from zeus-master for intrinsics and back-projection.

3.1 Relevant modules
- `zeus-master/perception/scripts/publish_d405.py`
  - `get_intrinsics(stream_profile)`
  - `get_K(intrinsics)` -> 3x3 camera matrix
- `zeus-master/common/ros_pointcloud.py`
  - `_get_xyz(depth_m, K)` -> [3, H*W] XYZ points

3.2 What we reuse
In `instant_policy/ip/deployment/perception/zeus_perception.py`:
- We use `get_intrinsics` and `get_K` to build K.
- We use `_get_xyz` to back-project depth to XYZ.

3.3 What we do not assume
- We do not assume ROS image topics.
- We do not assume `publish_d405.py` is running.
- We do not assume a fixed depth scale. We query it from the device.


4. Data contract and sample structure
-------------------------------------
We keep the model data format identical to the original Instant Policy code.

4.1 Raw demo format (input)
`demo = { "pcds": [], "T_w_es": [], "grips": [] }`

- `pcds[i]`: [Ni, 3] point cloud in world frame, meters.
- `T_w_es[i]`: [4, 4] EE pose in world frame.
- `grips[i]`: gripper state in {0, 1}.

4.2 Condensed demo format (model context)
`sample_to_cond_demo(demo, num_traj_wp=10)` produces:

- `obs`: list of length L (L=10).
  Each entry is a [2048, 3] point cloud in EE frame.
- `grips`: list of length L, {0, 1}.
- `T_w_es`: list of length L, each [4, 4].

The conversion:
- Selects waypoints based on motion and gripper changes.
- Converts each pcd to EE frame at that waypoint.

4.3 Live sample format (model input at runtime)
We build `full_sample`:
- `full_sample["demos"] = list of condensed demos`
- `full_sample["live"]` contains:
  - `obs`: [pcd_ee] with shape [2048, 3]
  - `grips`: [current_grip]
  - `T_w_es`: [T_w_e]
  - `actions`: [T_w_e repeated pred_horizon times]
  - `actions_grip`: [zeros(pred_horizon)]

`save_sample(full_sample, None)` converts this into a graph input for the model.


5. Coordinate frames and transforms
-----------------------------------
We use three frames:
- W: world/base frame (MoveIt base).
- E: end-effector frame.
- C: camera frame.

Transforms:
- `T_w_e`: transform from E to W.
- `T_world_camera`: transform from C to W.

5.1 Point cloud transforms
Depth -> camera XYZ:
  xyz_c = _get_xyz(depth_m, K)

Camera -> world:
  xyz_w = T_world_camera * xyz_c

World -> EE:
  xyz_e = inv(T_w_e) * xyz_w

5.2 Action composition
Actions are relative to the pose at inference time (T_w_e_initial):
  T_w_e_target_j = T_w_e_initial @ actions[j]
When executing sequentially, you move from target j-1 to target j, so the
per-step delta is the difference between these consecutive targets.

5.3 Units
- Distances: meters.
- Rotation: radians internally, degrees only for documentation.
- Depth scale: meters per depth unit from RealSense.


6. Perception pipeline (capture -> mask -> point cloud -> fuse)
---------------------------------------------------------------
Implemented in `perception/zeus_perception.py`.

6.1 Capture
For each camera:
- Start a RealSense pipeline with color + depth.
- Optionally align depth to color (`rs.align`).
- Extract color and depth arrays.
- Convert depth to meters using the device depth scale.

6.2 Masking
If segmentation is enabled:
- Obtain a binary mask (0/1).
- Multiply depth by mask.
- If mask shape does not match depth, ignore it.

6.3 Back-projection
- Use `_get_xyz(depth_m, K)` to compute XYZ in camera frame.
- Filter invalid points:
  - finite coordinates
  - z > 0

6.4 Transform to world
- Apply `T_world_camera` to each point.

6.5 Fuse
- Concatenate all camera clouds.
- Optional voxel downsample.
- Later, subsample to 2048 points for the model.


7. Segmentation: XMem++ tracking and SAM seeding
------------------------------------------------
We follow the paper: SAM seeds a tracker, then tracking runs each frame.

7.1 XMem++ in-process (recommended)
Implemented in `perception/xmem_segmentation.py` as `XMemOnlineSegmenter`.

Initialization per camera:
- First frame: run SAM to get a mask.
- Put that mask into XMem++ permanent memory.
- Mark camera as initialized.

Tracking:
- For each new frame, call `InferenceCore.step`.
- Output is a probability mask with background + objects.
- We take argmax and return a binary mask (foreground vs background).

Assumptions:
- Single-object mode (we treat "all objects of interest" as one foreground).
- CUDA is required, and this integration expects `cuda:0`.
- SAM checkpoint is required unless external masks are provided.

7.2 XMem++ via ROS mask topics
Implemented as `XMemMaskSubscriber`.

- Subscribe to one ROS image topic per camera.
- Convert masks to binary arrays.
- Used when `mask_topics` is set in config.

7.3 Why SAM is required by default
XMem++ must be seeded with an initial mask. The paper seeds with SAM.
If no external masks are provided, we require SAM weights.


8. State estimation and gripper mapping
---------------------------------------
Implemented in `state/zeus_state.py`.

8.1 End-effector pose
- `pose = move_group.get_current_pose().pose`
- `T_w_e = transform_from_pose(pose)`

8.2 Gripper state
- Subscribe to `Robotiq2FGripperRobotInput`.
- `gPO` in [0..255] maps to `grip = gPO / 255`.
- If stale or missing, return default 0.5.

Why a subscriber:
- Avoids blocking calls and keeps the loop responsive.


9. Control and safety gating
----------------------------
Implemented in `control/zeus_control.py` and `control/action_executor.py`.

9.1 Control (MoveIt cartesian planning)
- Convert target 4x4 to Pose.
- Plan a cartesian path.
- Execute plan if fraction >= threshold.

Limitations:
- Blocking; not guaranteed 10 Hz.
- For real-time control, use MoveIt Servo or RTDE.

9.2 Safety gating
We enforce:
- Workspace bounds.
- Per-step translation <= 1 cm.
- Per-step rotation <= 3 degrees.

The model is trained with these limits, so this keeps execution in-distribution.

Per-step vs horizon:
- These are per-step limits.
- Over 8 steps, total displacement can accumulate.


10. Demo collection and conversion
----------------------------------
Implemented in `demo/demo_collector.py`.

Process:
- Prompt operator to start.
- Optionally enable freedrive (if supported).
- At 10 Hz:
  - capture world-frame point cloud
  - record T_w_e
  - record gripper state
- Stop on user input.

Conversion:
- `sample_to_cond_demo` extracts L=10 waypoints.
- Converts each pcd to EE frame at that waypoint.

Why 10 Hz:
- Matches the paper's real-world setup.
- Balances segmentation cost and trajectory density.


11. Model inference loop and execution policy
---------------------------------------------
Implemented in `orchestrator.py`.

11.1 Load model
- Load `GraphDiffusion` checkpoint.
- Set `num_demos`, `num_diffusion_iters`, batch_size=1.
- Reinitialize graphs.

11.2 Prepare demos
- Convert raw demos with `sample_to_cond_demo`.
- If fewer demos than required, duplicate the last demo.

11.3 Per-step loop
1) Capture state: `T_w_e`, `grip`.
2) Capture perception: `pcd_w`.
3) Convert to EE frame and subsample to 2048 points.
4) Build `full_sample` and call `save_sample`.
5) Cache demo embeddings on step 0.
6) Compute live embeddings.
7) Run diffusion inference.
8) Execute actions until gripper state changes (paper recommendation).
   Each target is computed as T_w_e_initial @ actions[j].

11.4 Gripper command mapping
Model outputs in {-1, 1}:
  command = (grip + 1) / 2


12. Initialization and runtime sequence diagram
------------------------------------------------
Initialization:
1) Build DeploymentConfig.
2) Initialize segmentation backend (SAM, XMem++).
3) Initialize RealSense pipelines.
4) Initialize MoveGroupInterface.
5) Initialize state and control wrappers.
6) Load GraphDiffusion model and config.

Runtime loop (single iteration):

  [RealSense] -> RGB-D -> [Segmentation] -> mask
        |                                   |
        v                                   v
  depth + K -> xyz_c -> T_world_camera -> xyz_w
        |                                   |
        +-------------- fuse ---------------+
                        |
                        v
                   pcd_w -> pcd_ee
                        |
                        v
                   save_sample -> model
                        |
                        v
            actions (relative SE3), grips
                        |
                        v
             safety gate -> MoveIt execute


13. Shapes, units, and normalization
------------------------------------
Key tensors (typical shapes):
- `pcd_w`: [N, 3], float32, meters (world frame)
- `pcd_ee`: [2048, 3], float32, meters (EE frame)
- `T_w_e`: [4, 4], float32
- `actions`: [8, 4, 4], float32 (relative transforms)
- `grips`: [8], float32 in {-1, 1}

Normalization in `save_sample`:
- `current_grip` is mapped from [0,1] to [-1,1].
- `actions_grip` is mapped similarly.
- Point clouds are NOT normalized; they are fed in meters.


14. Configuration reference
---------------------------
See `config.py`.

CameraConfig:
- `serial`: RealSense serial string.
- `T_world_camera`: 4x4 camera-to-world transform.
- `width`, `height`, `fps`: capture settings.
- `align_to_color`: align depth to color.

SegmentationConfig:
- `enable`: enable segmentation.
- `backend`: "sam" or "xmem".
- `sam_checkpoint_path`: SAM weights (required to seed XMem++).
- `xmem_checkpoint_path`: XMem++ weights (required for XMem++).
- `xmem_init_with_sam`: seed XMem++ with SAM.
- `mask_topics`: optional ROS mask topics.
- `mask_threshold`: threshold for float masks.
- `points_per_side`, `pred_iou_thresh`, etc: SAM parameters.

DeploymentConfig:
- `camera_configs`: list of CameraConfig.
- `model_path`: Instant Policy checkpoint directory.
- `num_demos`, `num_traj_wp`, `num_diffusion_iters`.
- `pcd_num_points`: default 2048.
- `pcd_voxel_size`: optional downsample size.
- `execute_until_grip_change`: True by default.
- `arm`: "lightning" or "thunder".


15. Setup and run instructions
------------------------------
1) Prepare calibration:
   - Obtain `T_world_camera` for each RealSense.
   - Ensure it is in the same world frame as MoveIt.

2) Configure deployment:
   - Edit `instant_policy/ip/deployment.py`:
     - Set camera serials and `T_world_camera`.
     - Set SAM and XMem checkpoints.

3) Collect demos:
   - `python -m ip.deployment --collect-demo --demo-out demo.pkl`

4) Run deployment:
   - `python -m ip.deployment --demo demo.pkl`


16. Debug checklist and failure modes
-------------------------------------
Common failure modes:

1) Robot moves in wrong direction:
   - `T_world_camera` is wrong or inverted.
   - `T_w_e` frame does not match MoveIt world.

2) Jitter or stalling:
   - Poor segmentation (mask includes robot/table).
   - Point cloud not in EE frame.

3) No segmentation / empty mask:
   - SAM checkpoint missing.
   - XMem++ not initialized (no seed mask).

4) Controller too slow:
   - MoveIt planning is blocking.
   - Use MoveIt Servo or RTDE for streaming.

5) Gripper oscillation:
   - Ensure execute-until-grip-change is enabled.

Sanity checks:
- Visualize pcd in world and EE frame.
- Print action magnitudes per step.
- Confirm gripper topic updates.


17. File map
------------
- `instant_policy/ip/deployment.py`: entrypoint.
- `instant_policy/ip/deployment/orchestrator.py`: main loop.
- `instant_policy/ip/deployment/config.py`: config dataclasses.
- `instant_policy/ip/deployment/perception/zeus_perception.py`: RealSense capture.
- `instant_policy/ip/deployment/perception/sam_segmentation.py`: SAM wrapper.
- `instant_policy/ip/deployment/perception/xmem_segmentation.py`: XMem++.
- `instant_policy/ip/deployment/state/zeus_state.py`: EE pose + gripper state.
- `instant_policy/ip/deployment/control/zeus_control.py`: cartesian motion.
- `instant_policy/ip/deployment/control/action_executor.py`: safety checks.
- `instant_policy/ip/deployment/demo/demo_collector.py`: demo capture.
