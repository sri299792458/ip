#!/bin/bash
# Benchmark generator throughput for render backend mode:
# - FORCE_SOFTWARE_RENDERING=0 (GPU GL path)
# - FORCE_SOFTWARE_RENDERING=1 (software GL path)
#
# Runs both cases on the same node/job to minimize variance.
#
# Usage:
#   cd ~/ip
#   sbatch apptainer/benchmark_generator_render_mode.slurm
#
# Optional overrides (sbatch --export=ALL,...):
#   DATA_DIR=/scratch.global/$USER/ips
#   SHAPENET_PATH=/workspace/data/shapenet
#   BENCH_TASKS=16
#   DEMOS_PER_TASK_MIN=2
#   DEMOS_PER_TASK_MAX=4
#   PCD_DTYPE=float16
#   BENCH_MODES=gpu,sw   (or just gpu / just sw)

#SBATCH -A kdesingh
#SBATCH -p msigpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32gb
#SBATCH -t 01:00:00
#SBATCH --job-name=ip_genbench
#SBATCH --output=/scratch.global/%u/ips/logs/ip_genbench_%j.out
#SBATCH --error=/scratch.global/%u/ips/logs/ip_genbench_%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"

resolve_runner() {
  if [ -n "${RUNNER:-}" ] && [ -x "$RUNNER" ]; then
    printf "%s\n" "$RUNNER"
    return 0
  fi
  local candidates=()
  if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    candidates+=("$SLURM_SUBMIT_DIR/apptainer/run_instant_policy.sh")
    candidates+=("$SLURM_SUBMIT_DIR/run_instant_policy.sh")
  fi
  candidates+=("$PWD/apptainer/run_instant_policy.sh")
  candidates+=("$PWD/run_instant_policy.sh")
  candidates+=("$SCRIPT_DIR/run_instant_policy.sh")
  local c
  for c in "${candidates[@]}"; do
    if [ -x "$c" ]; then
      printf "%s\n" "$c"
      return 0
    fi
  done
  return 1
}

RUNNER="$(resolve_runner || true)"
if [ ! -x "$RUNNER" ]; then
  echo "ERROR: runner not found or not executable: ${RUNNER:-<unset>}"
  exit 1
fi

DATA_DIR=${DATA_DIR:-/scratch.global/$USER/ips}
SHAPENET_PATH=${SHAPENET_PATH:-}
INDEX_PATH=${INDEX_PATH:-/workspace/data/pseudo_ring/shapenet_index.json}
MESH_PATH=${MESH_PATH:-/workspace/data/assets/robotiq_2f85_collision_open.obj}

# Keep this small by default: benchmark should measure relative throughput quickly,
# not produce a large dataset.
BENCH_TASKS=${BENCH_TASKS:-16}
BENCH_SEED=${BENCH_SEED:-1337}
DEMOS_PER_TASK_MIN=${DEMOS_PER_TASK_MIN:-2}
DEMOS_PER_TASK_MAX=${DEMOS_PER_TASK_MAX:-4}
PCD_DTYPE=${PCD_DTYPE:-float16}
BENCH_MODES=${BENCH_MODES:-gpu,sw}

DISPLAY_BASE=${DISPLAY_BASE:-$((200 + (${SLURM_JOB_ID:-0} % 200)))}
VNC_BASE=${VNC_BASE:-$((20000 + (${SLURM_JOB_ID:-0} % 10000)))}

BENCH_ROOT=${BENCH_ROOT:-/workspace/data/bench_gen_render/${SLURM_JOB_ID:-manual}}

map_workspace_to_host() {
  local p="$1"
  if [[ "$p" == /workspace/data/* ]]; then
    printf "%s\n" "$DATA_DIR/${p#/workspace/data/}"
  else
    printf "%s\n" "$p"
  fi
}

if [ -z "$SHAPENET_PATH" ]; then
  if [ -d "$DATA_DIR/shapenet" ]; then
    SHAPENET_PATH="/workspace/data/shapenet"
  elif [ -d "$DATA_DIR/ShapeNetCore.v2" ]; then
    SHAPENET_PATH="/workspace/data/ShapeNetCore.v2"
  else
    SHAPENET_PATH="/workspace/data/shapenet"
  fi
fi

HOST_SHAPENET="$(map_workspace_to_host "$SHAPENET_PATH")"
HOST_BENCH_ROOT="$(map_workspace_to_host "$BENCH_ROOT")"
HOST_MESH_PATH="$(map_workspace_to_host "$MESH_PATH")"
HOST_INDEX_PATH="$(map_workspace_to_host "$INDEX_PATH")"

if [ ! -d "$HOST_SHAPENET" ]; then
  echo "ERROR: ShapeNet path not found: $HOST_SHAPENET"
  echo "       Set SHAPENET_PATH explicitly if needed."
  exit 1
fi

mkdir -p "$DATA_DIR/logs" "$HOST_BENCH_ROOT" "$(dirname "$HOST_MESH_PATH")" "$(dirname "$HOST_INDEX_PATH")"

if [ ! -f "$HOST_MESH_PATH" ]; then
  echo "Mesh not found, building: $HOST_MESH_PATH"
  "$RUNNER" "set -euo pipefail
python -m ip.scripts.build_robotiq_mesh --out \"$MESH_PATH\" --source collision --state open"
fi

SUMMARY_CSV="$HOST_BENCH_ROOT/summary.csv"
echo "mode,force_software,elapsed_s,num_tasks,num_items,tasks_per_s,items_per_s,render_platform,gl_vendor,gl_renderer" > "$SUMMARY_CSV"

run_case() {
  local mode="$1"          # gpu|sw
  local force_sw="$2"      # 0|1
  local disp="$3"
  local vnc="$4"

  local case_dir="$BENCH_ROOT/$mode"
  local host_case_dir
  host_case_dir="$(map_workspace_to_host "$case_dir")"
  rm -rf "$host_case_dir"
  mkdir -p "$host_case_dir"

  local case_log="$DATA_DIR/logs/ip_genbench_${SLURM_JOB_ID:-manual}_${mode}.log"
  local t0 t1 dt items tps ips
  t0="$(date +%s)"

  FORCE_SOFTWARE_RENDERING="$force_sw" \
  RLBENCH_DISPLAY="$disp" \
  RLBENCH_VNC_PORT="$vnc" \
  "$RUNNER" "set -euo pipefail
export PYOPENGL_PLATFORM=\${PYOPENGL_PLATFORM:-egl}
python - <<'PY'
import os
try:
    import pyrender
    from OpenGL import GL
    r = pyrender.OffscreenRenderer(64, 64)
    plat = type(getattr(r, '_platform', None)).__name__
    def dec(x):
        if isinstance(x, (bytes, bytearray)):
            return x.decode('utf-8', errors='ignore')
        return str(x)
    vendor = dec(GL.glGetString(GL.GL_VENDOR))
    renderer = dec(GL.glGetString(GL.GL_RENDERER))
    print(f\"[render_probe] FORCE_SOFTWARE_RENDERING={os.environ.get('FORCE_SOFTWARE_RENDERING', '')}\")
    print(f\"[render_probe] PYOPENGL_PLATFORM={os.environ.get('PYOPENGL_PLATFORM', '')}\")
    print(f\"[render_probe] platform={plat}\")
    print(f\"[render_probe] vendor={vendor}\")
    print(f\"[render_probe] renderer={renderer}\")
    r.delete()
except Exception as exc:
    print(f\"[render_probe] ERROR: {exc}\")
PY
python -m ip.scripts.generate_pseudo_demos \
  --shapenet_path \"$SHAPENET_PATH\" \
  --shapenet_index_path \"$INDEX_PATH\" \
  --save_dir \"$case_dir\" \
  --num_tasks $BENCH_TASKS \
  --seed $BENCH_SEED \
  --storage_format steps \
  --pcd_storage_dtype $PCD_DTYPE \
  --num_demos_per_task $DEMOS_PER_TASK_MIN $DEMOS_PER_TASK_MAX \
  --gripper_mesh_path \"$MESH_PATH\" \
  --attach_capture_min_points 3" > "$case_log" 2>&1

  t1="$(date +%s)"
  dt=$((t1 - t0))
  items="$(find "$host_case_dir" -maxdepth 1 -type f -name 'data_*.pt' | wc -l | tr -d ' ')"
  tps="$(awk -v t="$BENCH_TASKS" -v d="$dt" 'BEGIN {if (d>0) printf "%.6f", t/d; else print "0"}')"
  ips="$(awk -v i="$items" -v d="$dt" 'BEGIN {if (d>0) printf "%.6f", i/d; else print "0"}')"

  local plat vendor renderer
  plat="$(grep -m1 '^\[render_probe\] platform=' "$case_log" | sed 's/.*platform=//')"
  vendor="$(grep -m1 '^\[render_probe\] vendor=' "$case_log" | sed 's/.*vendor=//')"
  renderer="$(grep -m1 '^\[render_probe\] renderer=' "$case_log" | sed 's/.*renderer=//')"

  echo "$mode,$force_sw,$dt,$BENCH_TASKS,$items,$tps,$ips,\"$plat\",\"$vendor\",\"$renderer\"" >> "$SUMMARY_CSV"

  echo "[$mode] elapsed_s=$dt num_tasks=$BENCH_TASKS num_items=$items tasks_per_s=$tps items_per_s=$ips"
  echo "[$mode] render_platform=$plat"
  echo "[$mode] gl_vendor=$vendor"
  echo "[$mode] gl_renderer=$renderer"
  echo "[$mode] log=$case_log"
}

echo "Node=$(hostname)"
echo "Runner=$RUNNER"
echo "ShapeNet host path=$HOST_SHAPENET"
echo "Bench root host path=$HOST_BENCH_ROOT"
echo "BENCH_TASKS=$BENCH_TASKS demos=$DEMOS_PER_TASK_MIN..$DEMOS_PER_TASK_MAX pcd_dtype=$PCD_DTYPE"
echo "BENCH_MODES=$BENCH_MODES"
echo "Starting generation-only backend benchmark..."

if [[ ",$BENCH_MODES," == *",gpu,"* ]]; then
  run_case gpu 0 "$((DISPLAY_BASE + 1))" "$((VNC_BASE + 1))"
fi
if [[ ",$BENCH_MODES," == *",sw,"* ]]; then
  run_case sw 1 "$((DISPLAY_BASE + 2))" "$((VNC_BASE + 2))"
fi

echo "----- Summary CSV -----"
cat "$SUMMARY_CSV"

# Print slowdown/speedup ratio for quick read.
awk -F',' '
NR==1 {next}
$1=="gpu" {gpu_tps=$6; gpu_ips=$7}
$1=="sw"  {sw_tps=$6;  sw_ips=$7}
END{
  if (gpu_tps>0 && sw_tps>0) {
    printf "tasks_per_s_ratio(sw/gpu)=%.4f\n", sw_tps/gpu_tps
  }
  if (gpu_ips>0 && sw_ips>0) {
    printf "items_per_s_ratio(sw/gpu)=%.4f\n", sw_ips/gpu_ips
  }
}' "$SUMMARY_CSV"

echo "Benchmark complete."
