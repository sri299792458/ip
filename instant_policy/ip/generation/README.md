# Pseudo-Demo Generation (Instant Policy)

Single source of truth for pseudo-data generation in this repo.

This module implements the pseudo-demonstration pipeline used to train Instant Policy-style models:
- sample ShapeNet tabletop scenes,
- sample object-centric pseudo-task waypoints,
- synthesize gripper trajectories,
- render segmented point clouds,
- package data for training (`steps` or `trajectory` format).

## Scope

- Code root: `instant_policy/ip/generation/`
- Main CLI: `python -m ip.scripts.generate_pseudo_demos`
- Mesh utility: `python -m ip.scripts.build_robotiq_mesh`

## Code Map

- `scene_builder.py`: ShapeNet mesh loading, normalization, scene population.
- `waypoint_sampler.py`: pseudo-task specification and biased skill sampling.
- `trajectory_interpolator.py`: interpolation + fixed spacing resampling.
- `augmentation.py`: disturbance and gripper-noise augmentation.
- `renderer.py`: PyRender depth-camera observation rendering.
- `pseudo_demo_generator.py`: end-to-end task/demo generation and storage.
- `config.py`: generation defaults.

## Paper-Aligned Defaults

These are the defaults we treat as paper-canonical for pseudo generation:
- two objects on a plane (`num_objects_range=(2,2)`)
- object metric scale prior `0.20..0.30 m` (`object_scale_range`)
- pseudo-task waypoints `2..6` (`num_waypoints_range`)
- biased/random mix `50/50` (`bias_prob=0.5`)
- interpolation spacing `1cm / 3deg` (`trans_spacing=0.01`, `rot_spacing_deg=3.0`)
- translation interpolation methods `linear/cubic`; rotation interpolation uses quaternion Slerp
- disturbance `30%` (`disturbance_prob=0.3`)
- gripper-noise `10%` (`gripper_noise_prob=0.1`)
- Robotiq 2F-85 mesh required (`--gripper_mesh_path`)
- attach on open->closed and detach on closed->open
- front-only jaw-capture attach (`attach_capture_min_points=3`), no full-mesh distance gate
- gripper mesh frame canonicalization to policy-origin frame (`z=0.088 m` from URDF base/flange frame)
- three depth cameras rendered via PyRender (RLBench-style rig offsets + intrinsics, 128x128)
- debug video rendering uses a separate high-res visual renderer (`640x640`) and does not change training observations

Reference files:
- generation constants: `instant_policy/ip/generation/config.py`
- behavior implementation: `instant_policy/ip/generation/pseudo_demo_generator.py`
- global paper quickref: `instant_policy/ip/paper.md`

## Data Contracts

Raw generated demo (world frame):

```python
{
  "pcds":   [P_t_world],
  "T_w_es": [T_w_e_t],
  "grips":  [g_t],  # RLBench convention: 1=open, 0=closed
}
```

### `steps` storage

- Output files: `data_*.pt`
- One file per live timestep (large file count at scale)
- Created via `ip.utils.data_proc.save_sample`

### `trajectory` storage

- Output files: `task_*.pt`
- One file per pseudo-task
- Stores demos plus precomputed context waypoints (`cond`)
- Read at train-time by `ip.utils.trajectory_dataset.TrajectoryDataset`

Use `trajectory` for large-scale continuous generation.

## Generation Flow

1. Build a base scene from ShapeNet meshes.
2. Sample pseudo-task waypoint specs (object-relative).
3. Generate multiple demos per task by varying start pose and scene perturbation.
4. Interpolate and resample trajectory at fixed spacing.
5. Attach/detach object on gripper state transitions using front-only jaw-capture gating.
6. Render object point clouds from depth cameras using clean gripper transitions (rigid attachment).
7. Apply 10% gripper-state corruption to stored labels (not to attachment dynamics).
8. Store in chosen format (`steps` or `trajectory`).

## Gripper Mesh Policy

- `--gripper_mesh_path` is required.
- Use `build_robotiq_mesh` to create a metric Robotiq 2F-85 mesh.
- We use a fixed jaw state mesh (usually `open`) for proximity checks.
- This is sufficient for paper-style pseudo-data; full finger articulation simulation is not required.
- Loaded mesh is translated to the policy-origin frame so waypoint/contact sampling matches the same convention used by deployment/model inputs (`flange/base -> policy-origin = 0.088 m`).
- Contact is event-based:
  - close transition attempts object-centric attach first (if waypoint targets an object),
  - attach succeeds only if enough object points lie in the front jaw-capture region (`attach_capture_min_points`, thin-object robustness),
  - backside points (behind policy-origin `z`) are excluded from attach eligibility,
  - for non-object-centric waypoints, best candidate object is selected from the same gating rule,
  - open transition detaches.
- Gripper label noise (`gripper_noise_prob`) is applied after render/simulation, so grasped-object motion remains rigid during pseudo trajectory synthesis.

## Commands

Build mesh:

```bash
python -m ip.scripts.build_robotiq_mesh \
  --out /scratch/.../robotiq_2f85_collision_open.obj \
  --source collision \
  --state open
```

Minimal generation:

```bash
python -m ip.scripts.generate_pseudo_demos \
  --shapenet_path /scratch.global/$USER/ShapeNetCore.v2 \
  --save_dir /scratch/.../pseudo/test \
  --num_tasks 10 \
  --storage_format trajectory \
  --gripper_mesh_path /scratch/.../robotiq_2f85_collision_open.obj
```

Headless rendering:

```bash
export PYOPENGL_PLATFORM=egl
```

Optional RGB frame dumps:

```bash
python -m ip.scripts.generate_pseudo_demos \
  --shapenet_path /scratch.global/$USER/ShapeNetCore.v2 \
  --save_dir /scratch/.../pseudo/test \
  --num_tasks 1 \
  --storage_format trajectory \
  --gripper_mesh_path /scratch/.../robotiq_2f85_collision_open.obj \
  --save_renders \
  --render_dir /scratch/.../pseudo/renders
```

Video export only (no frame PNGs):

```bash
python -m ip.scripts.generate_pseudo_demos \
  --shapenet_path /scratch.global/$USER/ShapeNetCore.v2 \
  --save_dir /scratch/.../pseudo/test \
  --num_tasks 1 \
  --storage_format trajectory \
  --gripper_mesh_path /scratch/.../robotiq_2f85_collision_open.obj \
  --render_make_videos \
  --render_video_dir /scratch/.../pseudo/videos
```

Render one video per task category (debug):

```bash
for SKILL in random grasp pick_place pull push; do
  python -m ip.scripts.generate_pseudo_demos \
    --shapenet_path /scratch.global/$USER/ShapeNetCore.v2 \
    --save_dir /scratch/.../pseudo_debug/$SKILL/tasks \
    --num_tasks 1 \
    --storage_format trajectory \
    --force_skill "$SKILL" \
    --gripper_mesh_path /scratch/.../robotiq_2f85_collision_open.obj \
    --render_make_videos \
    --render_video_dir /scratch/.../pseudo_debug/$SKILL/videos \
    --render_video_fps 15
done
```

## Ring Buffer Guidance

Paper requires continuous generation with replacement; exact buffer size is an implementation choice.

Important:
- for `steps` storage, `buffer_size` means number of `data_*.pt` step files
- for `trajectory` storage, `buffer_size` means number of `task_*.pt` task files

Recommended starting point for trajectory mode:
- `buffer_size=8192`
- `num_shards=1`
- if using `--fill_buffer`: set `num_tasks=buffer_size` (bounded one-pass fill)
- if running as a continuous producer without `--fill_buffer`: large `num_tasks` is acceptable
- in MSI usage, prefer `apptainer/train_instant_policy.slurm` which bootstraps/train in one flow

`pcd_storage_dtype` guidance:
- start with `float32` for maximal fidelity
- move to `float16` only if disk/I/O becomes a bottleneck
- matrix inversions use `T_w_e` in float32 in dataset loading path

## Visual Behavior Notes

If RGB renders look static, this can still be correct:
- object-only point clouds are rendered
- objects move only when attachment is active
- training signal is still present because observations are transformed into the end-effector frame and paired with `T_w_e` + `grips`

## Troubleshooting

- Empty point clouds: generation fails fast by design; check camera setup/workspace coverage.
- Headless render failure: set `PYOPENGL_PLATFORM=egl`.
- Storage explosion: use `trajectory` mode + ring buffer.
- Unexpected no-motion visuals: verify gripper state transitions and attachment behavior.
- If the gripper appears to pass through objects frequently: verify mesh frame canonicalization and front-only jaw-capture settings (`attach_capture_min_points`).

## Dependencies

Required:
- `numpy`, `scipy`, `trimesh`, `pyrender`, `open3d`, `torch`

Optional:
- `imageio`, `imageio-ffmpeg` (video export)
