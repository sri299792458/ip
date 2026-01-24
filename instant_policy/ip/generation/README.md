# Pseudo Demonstration Generation (Instant Policy)

This module implements the pseudo-demonstration generator described in the
Instant Policy paper (Appendix D/E). It creates "arbitrary but consistent"
trajectories in simulation using ShapeNet objects, renders segmented point
clouds, and converts them into the training format expected by Instant Policy.

This README is a lab-meeting ready technical summary with code references,
design rationale, and operational guidance for large-scale runs.

## TL;DR

- Pseudo demos are synthetic trajectories of a gripper moving between
  object-centric waypoints.
- Each "pseudo-task" is defined by a shared set of waypoints sampled on/near
  the same objects, with multiple demos per task created by perturbing object
  poses and the gripper start pose.
- Observations are segmented point clouds (objects only) rendered with PyRender,
  and gripper state is provided via `T_w_e` and binary open/close labels.
- A fixed-size ring buffer is recommended for storage (paper-style continuous
  generation). Static storage of 100k tasks is infeasible due to file count and
  size.

## Code Layout

Core generation:
- `instant_policy/ip/generation/scene_builder.py`
- `instant_policy/ip/generation/waypoint_sampler.py`
- `instant_policy/ip/generation/trajectory_interpolator.py`
- `instant_policy/ip/generation/augmentation.py`
- `instant_policy/ip/generation/renderer.py`
- `instant_policy/ip/generation/pseudo_demo_generator.py`

CLI utilities:
- `instant_policy/ip/scripts/generate_pseudo_demos.py`
- `instant_policy/ip/scripts/merge_pseudo_demos.py`
- `instant_policy/ip/scripts/plot_pseudo_sample.py`
- `instant_policy/ip/scripts/animate_pseudo_demo.py`
- `instant_policy/ip/scripts/render_to_video.py`

## Data Format (What Gets Stored)

Each generated demonstration is stored as a list of steps:

```
demo_sample = {
  "pcds":   [P_t_world],   # point clouds in world frame (objects only)
  "T_w_es": [T_w_e_t],     # gripper pose in world frame (SE(3))
  "grips":  [g_t],         # gripper open/close (0/1)
}
```

Training samples are built by `sample_to_cond_demo` and `sample_to_live`:
- Point clouds are transformed into the end-effector frame using `inv(T_w_e)`.
- Gripper nodes are created from `T_w_e` and `grips` (no gripper points needed).

Files saved are `data_*.pt` (one file per live timestep), created by
`ip.utils.data_proc.save_sample`.

Important: The world-frame render can look static if no object attaches.
Motion is still encoded because point clouds are transformed into the
end-effector frame for training.

## Trajectory Storage (task_*.pt)

To avoid per-timestep files, you can store whole pseudo-tasks and sample
timesteps on the fly during training.

Generator flag:
```
--storage_format trajectory
```

Each file stores one pseudo-task:
```
task_data = {
  "task_id": int,
  "demos": [
    {
      "pcds":   [P_t_world],
      "T_w_es": [T_w_e_t],
      "grips":  [g_t],
      "cond":   {  # precomputed context demo
        "obs":   [P_k_ee],
        "grips": [g_k],
        "T_w_es": [T_w_e_k],
      },
    },
    ...
  ],
}
```

Training uses `ip.utils.trajectory_dataset.TrajectoryDataset` to:
- choose a live demo and timestep,
- build actions for `pred_horizon`,
- assemble the same `Data` object as `save_sample`.

This cuts storage dramatically (no `data_*.pt` explosion) and supports a
task-level ring buffer.

## Pipeline Summary (Step-by-Step)

1) Scene population (ShapeNet)
   - Sample 2 objects by default.
   - Normalize mesh scale and center.
   - Place on a plane in workspace bounds.
   - Code: `scene_builder.py`

2) Pseudo-task definition (waypoint specs)
   - Sample 2-6 waypoints on/near objects.
   - 50% biased sampling toward grasp/pick-place/open/close, 50% random.
   - Waypoints are stored object-relative (semantic consistency).
   - Code: `waypoint_sampler.py`

3) Trajectory synthesis
   - Sample a starting gripper pose.
   - Resolve object-relative waypoints to world frame.
   - Interpolate between waypoints (linear/cubic/spherical).
   - Resample to uniform spacing (1 cm, 3 deg).
   - Code: `trajectory_interpolator.py`

4) Augmentation
   - 30% disturbance + recovery segment.
   - 10% random gripper state flips.
   - Per-step small pose noise.
   - Code: `augmentation.py`

5) Attachment simulation
   - When gripper closes near an object, attach it rigidly to the gripper.
   - Object pose follows gripper until release.
   - Code: `pseudo_demo_generator.py`

6) Rendering (PyRender)
   - 3 depth cameras by default.
   - Object-only point clouds (segmented).
   - Optional RGB render dump for visualization.
   - Code: `renderer.py`

7) Convert to training format
   - Use `sample_to_cond_demo` and `sample_to_live`.
   - Save with `save_sample`.
   - Code: `ip.utils.data_proc`

## Alignment with Paper

Appendix D/E claims:
- ShapeNet object sampling.
- 2-6 object-centric waypoints.
- Linear/cubic/spherical interpolation.
- Uniform spacing (1 cm, 3 deg).
- 3 depth cameras (PyRender).
- 50% biased skills, 50% random.
- 30% disturbance, 10% gripper noise.

This implementation follows those claims. Notable practical clarifications:
- Objects are normalized and scaled to metric sizes (5-15 cm by default).
- Gripper feasibility is not enforced (matching the paper's "no dynamics").
- Visual renders optionally include a gripper marker for clarity. This does not
  affect training point clouds.

## Biased Skill Definitions (Exact Behavior)

The generator uses four explicit skill modes (plus random). These are the only
task "categories" defined in code.

1) grasp
   - Sample a surface point on a random object.
   - Pre-grasp waypoint: gripper open, 5–10 cm offset along approach direction.
   - Grasp waypoint: gripper closed at the surface.
   - Lift waypoint: gripper closed, +5–15 cm in z.
   - Code: `ip/generation/waypoint_sampler.py::_grasp_waypoints`.

2) pick_place
   - Same pre-grasp/grasp/lift as above.
   - Place waypoint: sample a surface point on the other object (if present),
     otherwise a random tabletop pose.
   - Release waypoint: same pose, gripper open.
   - Code: `ip/generation/waypoint_sampler.py::_pick_place_waypoints`.

3) open
   - Pre-grasp (open) → grasp (closed) → pull 5–20 cm along local +x axis.
   - Code: `ip/generation/waypoint_sampler.py::_open_waypoints`.

4) close
   - Pre-grasp (open) → grasp (closed) → push 5–20 cm along local −x axis.
   - Code: `ip/generation/waypoint_sampler.py::_close_waypoints`.

5) random (50% of samples)
   - 2–6 object-centric waypoints with random gripper flips.
   - Code: `ip/generation/waypoint_sampler.py::_random_waypoints`.

Note: Object motion in renders only appears when attachment is triggered (gripper
closes near an object). Otherwise the world-frame render can look static.

### Why These Four Were Used in the Paper

Appendix D states bias sampling uses grasping, pick-and-place, opening, and
closing, with the remaining 50% random. The intent is to cover broad interaction
primitives while relying on random trajectories and in-context demos to supply
task-specific patterns (e.g., pushing) at test time.

## Configuration Reference

All parameters live in `ip/generation/config.py` as `GenerationConfig`.

Key fields:
- `shapenet_path`: root directory containing synset folders.
- `num_tasks`: number of pseudo-tasks to generate.
- `num_demos_per_task`: range, default (3, 5).
- `num_waypoints_range`: default (2, 6).
- `bias_prob`: default 0.5.
- `trans_spacing`: default 0.01 (1 cm).
- `rot_spacing_deg`: default 3.0 (3 degrees).
- `disturbance_prob`: default 0.3.
- `gripper_noise_prob`: default 0.1.
- `attach_radius`: distance to attach on close.
- `cameras`: list of 3 cameras around workspace.
- `save_renders`: dump RGB frames to disk.
- `render_stride`: save every Nth frame.
- `buffer_size`: ring buffer size (via CLI).
- `pcd_storage_dtype`: `"float32"` (default) or `"float16"` for stored world-frame point clouds (trajectory format).

## CLI Usage

Minimal run:
```bash
python -m ip.scripts.generate_pseudo_demos \
  --shapenet_path /scratch/.../shapenet \
  --save_dir /scratch/.../pseudo_demos/test \
  --num_tasks 3
```

Headless GPU rendering:
```bash
export PYOPENGL_PLATFORM=egl
```

Save RGB renders:
```bash
python -m ip.scripts.generate_pseudo_demos \
  --shapenet_path /scratch/.../shapenet \
  --save_dir /scratch/.../pseudo_demos/test \
  --num_tasks 1 \
  --save_renders \
  --render_dir /scratch/.../pseudo_demos/renders
```

Stitch saved frames into videos:
```bash
python -m ip.scripts.render_to_video \
  --render_dir /scratch/.../pseudo_demos/renders \
  --out_dir /scratch/.../pseudo_demos/videos
```

## Ring Buffer Mode (Paper-Style Continuous Generation)

Static storage is infeasible at scale. Each sample is ~0.83 MB in our tests
and each task produced ~865 samples (~0.7 GB per task). A static 100k-task
dataset would be ~70 TB and ~86M files.

Instead, use a fixed-size ring buffer and overwrite continuously.

Example: 250k-sample buffer (~210 GB), 8 shards:
```bash
python -m ip.scripts.generate_pseudo_demos \
  --shapenet_path /scratch/.../shapenet \
  --save_dir /scratch/.../pseudo_demos/buffer \
  --num_tasks 999999 \
  --storage_format steps \
  --buffer_size 250000 \
  --num_shards 8 \
  --shard_id 0 \
  --fill_buffer
```

Repeat with `--shard_id 1..7`. Remove `--fill_buffer` for continuous overwrite.

Trajectory ring buffer (per-task files):
```bash
python -m ip.scripts.generate_pseudo_demos \
  --shapenet_path /scratch/.../shapenet \
  --save_dir /scratch/.../pseudo_demos/task_buffer \
  --num_tasks 999999 \
  --storage_format trajectory \
  --buffer_size 250000 \
  --num_shards 8 \
  --shard_id 0 \
  --fill_buffer
```

Train with:
```bash
python -m ip.train \
  --data_path_train /scratch/.../pseudo_demos/task_buffer \
  --data_path_val /scratch/.../pseudo_demos/task_buffer \
  --data_format trajectory
```

## How Task Information Is Encoded (No Labels)

There are no explicit task labels. Task identity is implicit in:
- Object geometry (point clouds).
- Demonstration trajectories (`T_w_e`, `grips`).
- Relative motions between objects and gripper in EE frame.

The graph uses `T_w_e` and gripper states to place gripper nodes and
propagate context information.

## Visual Debugging Notes

Why a demo might look static in RGB:
- Only objects are rendered.
- Objects only move if attached during a grasp.

The training signal is still present because:
- Point clouds are transformed into the end-effector frame each step.
- Gripper nodes are defined from `T_w_e` and `grips`.

If you want visible motion in RGB, enable renders and include a gripper marker
(enabled in the generator's render path).

## Dependencies

Required:
- `numpy`, `scipy`, `trimesh`, `pyrender`, `open3d`, `torch`

Optional (videos):
- `imageio`, `imageio-ffmpeg`

## ShapeNet Layout (Expected)

```
shapenet_root/
  02691156/
    <model_id>/
      models/
        model_normalized.obj
```

The loader searches for `model.obj` or `model_normalized.obj`.

## Troubleshooting

- PyRender fails on headless: set `PYOPENGL_PLATFORM=egl`.
- Empty point clouds: fallback uses surface samples; check camera config.
- No visible motion in RGB: gripper marker or forced grasp segment.
- Storage blow-up: use ring buffer and increase spacing to reduce timesteps.

## Suggested Sanity Checks (Before Full Run)

1) Data integrity (NaNs, action magnitudes).
2) Visual check of 2-3 tasks with renders.
3) Throughput and storage estimate from a small shard.
4) Short training smoke test to confirm loss decreases.

## Known Limitations

- No physics or collision checking (by design).
- Object-only point clouds (no table/robot).
- Gripper feasibility not enforced.
- Distribution may differ from real robot scenes.

These align with the paper's assumptions and can be refined if needed.
