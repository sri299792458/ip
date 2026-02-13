# Instant Policy Apptainer (MSI)

Apptainer container for running Instant Policy + RLBench on MSI (with optional VNC GUI).

## Build

```bash
cd ~/ip/apptainer
./build_instant_policy.sh
```

This creates `instant_policy.sif` (symlinked from `/scratch.global/$USER/ips`).

## Run (VNC)

Request a GPU node first, then:

```bash
cd ~/ip/apptainer
./run_instant_policy_vnc.sh python eval.py --task_name=plate_out --num_demos=2 --num_rollouts=10
```

Connect from your laptop:

```bash
ssh -L 5900:<node>:5900 <user>@agate.msi.umn.edu
```

Then open a VNC viewer to `localhost:5900`.

## Run (No VNC, Xvfb only)

For training jobs where you do not need interactive GUI streaming:

```bash
cd ~/ip/apptainer
./run_instant_policy.sh python train_language.py --help
```

## Single SLURM Pipeline (Pseudo + Train)

Use only this script:

```bash
cd ~/ip/apptainer
sbatch train_instant_policy.slurm
```

What this one job does:
- ensures Robotiq mesh exists (`MESH_PATH`)
- builds validation pseudo set if missing (`VAL_DATA_DIR`)
- runs continuous ring overwrite generation during training
- waits for a minimal train-data threshold, then starts training
- starts training (or resumes) in the same job

Resume across 24h jobs (same run name):
```bash
cd ~/ip/apptainer
sbatch --export=ALL,RUN_NAME=my_run,RECORD=1,AUTO_RESUME=1 train_instant_policy.slurm
```

Optional explicit checkpoint resume:
```bash
cd ~/ip/apptainer
sbatch --export=ALL,RUN_NAME=my_run,RECORD=1,RESUME_CKPT=/workspace/data/runs_policy/my_run/last.pt train_instant_policy.slurm
```

Important defaults for training script:
- ShapeNet path: auto-resolved (`/workspace/data/shapenet` preferred, else `/workspace/data/ShapeNetCore.v2`)
- train data: `/workspace/data/pseudo_ring/task_buffer`
- val data: `/workspace/data/pseudo_ring/val`
- format: `steps` (`data_*.pt`)
- model path: `/workspace/data/checkpoints`
- W&B logging: enabled by default (`USE_WANDB=1`, `WANDB_MODE=online` unless overridden)

You can override any script variable with `sbatch --export=ALL,KEY=VALUE,...`.

Example overrides:
```bash
sbatch --export=ALL,RUN_NAME=ring_bs32,BATCH_SIZE=32 train_instant_policy.slurm
sbatch --export=ALL,RUN_NAME=ring_bs32,RECORD=1,AUTO_RESUME=1,WANDB_ID=<existing_wandb_id>,WANDB_RESUME=allow train_instant_policy.slurm
sbatch --export=ALL,SHAPENET_PATH=/workspace/data/shapenet train_instant_policy.slurm
sbatch --export=ALL,TRAIN_BUFFER_SIZE=50000,DEMOS_PER_TASK_MIN=3,DEMOS_PER_TASK_MAX=5 train_instant_policy.slurm
sbatch --export=ALL,VAL_FORCE_REBUILD=1,VAL_NUM_TASKS=300 train_instant_policy.slurm
sbatch --export=ALL,NUM_ITERS_OVERRIDE=50000,RECORD=0,USE_WANDB=0 train_instant_policy.slurm
```

Resume defaults:
- default: `AUTO_RESUME=1` (starts fresh automatically when no checkpoint exists)
- explicit resume: pass `RESUME_CKPT=...` (or keep `AUTO_RESUME=1` with same `RUN_NAME`)

Core pseudo-data knobs:
- `TRAIN_BUFFER_SIZE` (default `8192`)
- `TRAIN_START_MIN_ITEMS` (default `512`)
- `TRAIN_START_TIMEOUT_SEC` (default `7200`)
- `GEN_CHUNK_TASKS` / `GEN_TASK_START` (continuous single-producer behavior)
- `TRAIN_SAMPLE_CACHE_SIZE` (default `2048`; per-worker LRU sample cache for `data_*.pt`)
- `DEMOS_PER_TASK_MIN` / `DEMOS_PER_TASK_MAX` (default `2/4`)
- `PCD_DTYPE` (default `float16`)
- `VAL_NUM_TASKS` (default `100`)
- `VAL_TASK_START` (default `200000000`)

## Data Units (What Is a File / Sample / Task / Batch)

This section is the ground-truth contract for `steps` mode (`data_*.pt`), which is the only
data mode used by `train_instant_policy.slurm`.

### 1) Raw generated demo trajectory (before step packing)

Each generated demo trajectory has:
- `pcds[t]`: object-only point cloud in world frame (downsampled to `2048` points)
- `T_w_es[t]`: end-effector pose in world frame
- `grips[t]`: binary gripper state (`1=open`, `0=closed`)

### 2) What one `data_<k>.pt` file stores

A single `data_*.pt` is one training step sample (one live timestep + action horizon), with:
- context demo tensors (same for all timesteps from the same pseudo sample):
  - `pos_demos`: concatenated context waypoint point clouds
  - `graps_demos`: context gripper states at waypoints
  - `demo_T_w_es`: context waypoint poses
  - `batch_demos`: point-to-waypoint/demo index map
- live-step tensors (change per file):
  - `pos_obs`: current live observation point cloud in end-effector frame
  - `current_grip`: current gripper state
  - `T_w_e`: current end-effector pose
  - `actions`: next `pred_horizon` relative SE(3) actions (default horizon `8`)
  - `actions_grip`: next `pred_horizon` gripper states

Important:
- In `steps` mode, context is duplicated across many files (all live timesteps of the same sample),
  which increases file count and storage.

### 3) What is "one task" in generation

In pseudo generation:
- one pseudo-task samples one scene + one waypoint spec
- then generates `D` demos (`D` in `[DEMOS_PER_TASK_MIN, DEMOS_PER_TASK_MAX]`, default `2..4`)
- each demo can become the live trajectory once, with others as context
- each live trajectory contributes one `data_*.pt` per live timestep

So files per pseudo-task are:
- `files_per_task = sum(live_timesteps_of_each_live_demo)`
- not equal to 1

This is why `VAL_NUM_TASKS=100` can produce many more than 100 files.
Example from your run:
- `16507` files in `val` for `100` tasks means about `165` step-samples per task on average.

- per-demo live timestep count is set by trajectory geometry plus interpolation density
  (`trans_spacing` / `rot_spacing_deg`), then summed across demos in the task.
- with current defaults, interpolation targets roughly `1 cm` translation spacing and `3 deg` rotation spacing,
  so longer/curvier motions produce more step files.

### 4) What is one training sample

One training sample is exactly one file:
- `data_<idx>.pt`
- loaded by `RunningDataset.__getitem__`

### 5) What batch size means here

`BATCH_SIZE` in training means:
- number of `data_*.pt` files consumed per optimizer step
- not number of pseudo-tasks
- not number of trajectories

With `drop_last=True`, steps per epoch are:
- `steps_per_epoch = floor(train_items / batch_size)`

Example with ring size `TRAIN_BUFFER_SIZE=8192`:
- `batch_size=16` -> `512` steps/epoch
- `batch_size=64` -> `128` steps/epoch

### 6) Why per-file size can look large

`data_*.pt` includes both:
- live-step tensors
- duplicated context tensors

So sizes like `~846K` per file and multi-GB val/train dirs are expected in `steps` mode.
This is a representation choice for training simplicity and online ring replacement behavior.

### 7) What happens near the end of a trajectory (`pred_horizon=8`)

For a live index `t`, targets are built for `t+1 ... t+8`.
- if `t+j` exists: target pose is relative transform `inv(T_w_e[t]) @ T_w_e[t+j]`, target grip is `grips[t+j]`
- if `t+j` is out of range: target pose is identity (`I_4`), target grip is the final grip (`grips[-1]`)

So if a very late timestep is selected, part (or all) of the 8-step target window is padding.

## Key Runtime Behavior (Why It Works)

The run script keeps the runtime CoppeliaSim path aligned with how PyRep was built:

- PyRep is built against `/opt/CoppeliaSim`.
- The run script uses `/opt/CoppeliaSim` by default and adds `--writable-tmpfs`,
  so GUI state writes do not crash.
- Host env is isolated with `--cleanenv --no-home`.

This avoids Bullet segfaults caused by path or library mismatches.

W&B auth note:
- The run scripts use `--no-home`, so host home is isolated.
- If `~/.netrc` exists on host, the script now auto-binds it to `/workspace/data/.netrc`
  so W&B login works inside container without extra steps.

## Environment Variables

- `PROJECT_DIR` (default: repo root containing this `apptainer/` folder)
- `INSTANT_POLICY_DIR` (default: `$PROJECT_DIR/instant_policy`)
- `DATA_DIR` (default: `/scratch.global/$USER/ips`)
- `RLBENCH_DISPLAY` (default: `:1`)
- `RLBENCH_VNC_PORT` (default: `5900`)
- `RLBENCH_ENABLE_VNC` (default: `1`; set `0` for Xvfb-only mode)
- `COPPELIASIM_USE_COPY=1` to use a writable copy at `/workspace/data/.coppeliasim`

## Troubleshooting

VNC is blank:
- Make sure you are tunneled to the correct node and port.
- Try a fresh display/port:
  ```bash
  RLBENCH_DISPLAY=2 RLBENCH_VNC_PORT=5901 ./run_instant_policy_vnc.sh python eval.py --task_name=plate_out --num_demos=2 --num_rollouts=10
  ```

GUI segfaults:
- Use the default `/opt/CoppeliaSim` path (do not set `COPPELIASIM_USE_COPY=1`).
- The script already enables `--writable-tmpfs` and creates `~/.CoppeliaSim`.

Update popup blocks the sim:
- The run script writes CoppeliaSim user settings with:
  `doNotShowUpdateCheckMessage=1`, `suppressStartupDialogs=1`, `noVersionCheck=1`
  in `~/.CoppeliaSim/usrset.txt` (and also updates `$COPPELIASIM_ROOT/system/usrset.txt`).
- This is the setting path CoppeliaSim reads on Linux when `HOME` is set.

GPU missing:
- Check allocation: `nvidia-smi`
- Ensure you are on a GPU node.

Transformers cache warning:
- The `TRANSFORMERS_CACHE` warning is harmless. `HF_HOME` is already set.
