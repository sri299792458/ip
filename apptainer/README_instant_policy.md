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
- bootstraps train ring buffer to minimal target size (`TRAIN_DATA_DIR`)
- runs continuous ring overwrite generation during training
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
- `MIN_BOOTSTRAP_TASKS` (default `512`, backward-compatible alias)
- `MIN_BOOTSTRAP_ITEMS` (default `MIN_BOOTSTRAP_TASKS`)
- `GEN_NUM_SHARDS` / `GEN_CHUNK_TASKS` (continuous producer parallelism)
- `DEMOS_PER_TASK_MIN` / `DEMOS_PER_TASK_MAX` (default `3/3`)
- `PCD_DTYPE` (default `float16`)
- `VAL_NUM_TASKS` (default `100`)
- `VAL_TASK_START` (default `200000000`)

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
- `FORCE_SOFTWARE_RENDERING=1` to force Mesa software rendering (use only if needed)

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
