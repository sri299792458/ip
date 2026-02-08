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

## Pseudo-Demo Ring Buffer Pipeline (Trajectory Format)

This is the paper-style setup: continuously regenerate pseudo-tasks into a
fixed-size task buffer and train from `task_*.pt`.

1) Fill the sharded ring buffer once:
```bash
cd ~/ip/apptainer
sbatch generate_pseudo_buffer.slurm
```

Default behavior:
- 8 shards (`--array=0-7%4`), each writing a disjoint index range.
- trajectory storage (`task_*.pt`) in `/scratch.global/$USER/ips/pseudo_ring/task_buffer`.
- ring fill mode enabled (`FILL_BUFFER=1`) until each shard wraps once.

2) Start continuous overwrite generation (after initial fill):
```bash
cd ~/ip/apptainer
sbatch --export=ALL,FILL_BUFFER=0,APPEND=1 generate_pseudo_buffer.slurm
```

3) Build a fixed validation pseudo set:
```bash
cd ~/ip/apptainer
sbatch generate_pseudo_val.slurm
```

4) Train policy on trajectory data:
```bash
cd ~/ip/apptainer
sbatch train_instant_policy.slurm
```

Important defaults for training script:
- train data: `/workspace/data/pseudo_ring/task_buffer`
- val data: `/workspace/data/pseudo_ring/val`
- format: `trajectory`
- model path: `/workspace/data/checkpoints`

You can override any script variable with `sbatch --export=ALL,KEY=VALUE,...`.

Example overrides:
```bash
sbatch --export=ALL,BUFFER_SIZE=300000,NUM_SHARDS=8 generate_pseudo_buffer.slurm
sbatch --export=ALL,VAL_NUM_TASKS=500 generate_pseudo_val.slurm
sbatch --export=ALL,RUN_NAME=ring_bs32,BATCH_SIZE=32 train_instant_policy.slurm
```

## Key Runtime Behavior (Why It Works)

The run script keeps the runtime CoppeliaSim path aligned with how PyRep was built:

- PyRep is built against `/opt/CoppeliaSim`.
- The run script uses `/opt/CoppeliaSim` by default and adds `--writable-tmpfs`,
  so GUI state writes do not crash.
- Host env is isolated with `--cleanenv --no-home`.

This avoids Bullet segfaults caused by path or library mismatches.

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
