# Instant Policy Apptainer (MSI)

Apptainer container for running Instant Policy + RLBench with CoppeliaSim GUI over VNC on MSI.

## Build

```bash
cd ~/ips/apptainer
./build_instant_policy.sh
```

This creates `instant_policy.sif` (symlinked from `/scratch.global/$USER/ips`).

## Run (VNC)

Request a GPU node first, then:

```bash
cd ~/ips/apptainer
./run_instant_policy_vnc.sh python eval.py --task_name=plate_out --num_demos=2 --num_rollouts=10
```

Connect from your laptop:

```bash
ssh -L 5900:<node>:5900 <user>@agate.msi.umn.edu
```

Then open a VNC viewer to `localhost:5900`.

## Key Runtime Behavior (Why It Works)

The run script keeps the runtime CoppeliaSim path aligned with how PyRep was built:

- PyRep is built against `/opt/CoppeliaSim`.
- The run script uses `/opt/CoppeliaSim` by default and adds `--writable-tmpfs`,
  so GUI state writes do not crash.
- Host env is isolated with `--cleanenv --no-home`.

This avoids Bullet segfaults caused by path or library mismatches.

## Environment Variables

- `PROJECT_DIR` (default: `$HOME/ips`)
- `INSTANT_POLICY_DIR` (default: `$PROJECT_DIR/instant_policy`)
- `DATA_DIR` (default: `/scratch.global/$USER/ips`)
- `RLBENCH_DISPLAY` (default: `:1`)
- `RLBENCH_VNC_PORT` (default: `5900`)
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
- The run script disables update checks in user and install configs:
  `~/.CoppeliaSim/system/usrset.txt`, `~/CoppeliaSim/system/usrset.txt`,
  `~/.config/CoppeliaSim/system/usrset.txt`, and `$COPPELIASIM_ROOT/system/usrset.txt`.

GPU missing:
- Check allocation: `nvidia-smi`
- Ensure you are on a GPU node.

Transformers cache warning:
- The `TRANSFORMERS_CACHE` warning is harmless. `HF_HOME` is already set.
