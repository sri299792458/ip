#!/bin/bash
# Unified MSI entrypoint:
# 1) Ensure val set + minimal train bootstrap
# 2) Start sharded continuous pseudo-demo generation in background
# 3) Run training (resume-friendly across 24h jobs)
#
# Re-submit this same script for each 24h window.
# Generator shard logs are written to:
#   /scratch.global/$USER/ips/logs/ip_gen_${SLURM_JOB_ID}_shard*.log
#
# Defaults are chosen for one clean pipeline (no offline full fill):
# - Bootstrap ring to target item count (default: TRAIN_BUFFER_SIZE)
# - Continuous sharded generation (single shard by default for 1-GPU runs)
# - Ring buffer overwrite behavior via --buffer_size
#SBATCH -A kdesingh
#SBATCH -p msigpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64gb
#SBATCH -t 24:00:00
#SBATCH --job-name=ip_train
#SBATCH --output=/scratch.global/%u/ips/logs/ip_train_%j.out
#SBATCH --error=/scratch.global/%u/ips/logs/ip_train_%j.err

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
DATA_DIR=${DATA_DIR:-/scratch.global/$USER/ips}

# Paths inside container.
TRAIN_DATA_DIR=${TRAIN_DATA_DIR:-/workspace/data/pseudo_ring/task_buffer}
VAL_DATA_DIR=${VAL_DATA_DIR:-/workspace/data/pseudo_ring/val}
SHAPENET_PATH=${SHAPENET_PATH:-}
MESH_PATH=${MESH_PATH:-/workspace/data/assets/robotiq_2f85_collision_open.obj}
INDEX_PATH=${INDEX_PATH:-/workspace/data/pseudo_ring/shapenet_index.json}

MODEL_PATH=${MODEL_PATH:-/workspace/data/checkpoints}
MODEL_NAME=${MODEL_NAME:-model.pt}
SAVE_ROOT=${SAVE_ROOT:-/workspace/data/runs_policy}
RUN_NAME=${RUN_NAME:-ring_train_${SLURM_JOB_ID:-manual}}

# Training settings (paper-aligned path only).
DATA_FORMAT=steps
BATCH_SIZE=${BATCH_SIZE:-16}

RECORD=${RECORD:-1}
FINE_TUNE=${FINE_TUNE:-0}
COMPILE_MODELS=${COMPILE_MODELS:-1}
USE_WANDB=${USE_WANDB:-1}
AUTO_RESUME=${AUTO_RESUME:-1}
RESUME_CKPT=${RESUME_CKPT:-}
WANDB_ID=${WANDB_ID:-}
WANDB_RESUME=${WANDB_RESUME:-allow}
EXTRA_TRAIN_ARGS=${EXTRA_TRAIN_ARGS:-}
NUM_ITERS_OVERRIDE=${NUM_ITERS_OVERRIDE:-}
TRAIN_NUM_WORKERS=${TRAIN_NUM_WORKERS:-8}
TRAIN_PERSISTENT_WORKERS=${TRAIN_PERSISTENT_WORKERS:-1}
TRAIN_PREFETCH_FACTOR=${TRAIN_PREFETCH_FACTOR:-4}
TRAIN_VAL_CHECK_INTERVAL=${TRAIN_VAL_CHECK_INTERVAL:-20000}
TRAIN_LOG_EVERY_N_STEPS=${TRAIN_LOG_EVERY_N_STEPS:-500}
TRAIN_DEVICES=${TRAIN_DEVICES:-1}
TRAIN_STRATEGY=${TRAIN_STRATEGY:-auto}

# Ring/generation settings.
TRAIN_BUFFER_SIZE=${TRAIN_BUFFER_SIZE:-8192}
# Backward-compatible alias:
# - Old env used MIN_BOOTSTRAP_TASKS.
# - New name reflects count of files in active DATA_FORMAT.
MIN_BOOTSTRAP_TASKS=${MIN_BOOTSTRAP_TASKS:-512}
MIN_BOOTSTRAP_ITEMS=${MIN_BOOTSTRAP_ITEMS:-$TRAIN_BUFFER_SIZE}
TRAIN_GEN_SEED=${TRAIN_GEN_SEED:-1337}
DEMOS_PER_TASK_MIN=${DEMOS_PER_TASK_MIN:-3}
DEMOS_PER_TASK_MAX=${DEMOS_PER_TASK_MAX:-3}
PCD_DTYPE=${PCD_DTYPE:-float16}

# Profiling/telemetry settings (for 1h decision runs).
TELEMETRY_INTERVAL_SEC=${TELEMETRY_INTERVAL_SEC:-30}
TELEMETRY_TASK_COUNT_EVERY=${TELEMETRY_TASK_COUNT_EVERY:-10}

# Parallel shard workers (continuous producer).
ENABLE_PARALLEL_GENERATORS=${ENABLE_PARALLEL_GENERATORS:-1}
GEN_NUM_SHARDS=${GEN_NUM_SHARDS:-1}
GEN_CHUNK_TASKS=${GEN_CHUNK_TASKS:-256}
GEN_TASK_START_BASE=${GEN_TASK_START_BASE:-1000000}
GEN_TASK_START_STRIDE=${GEN_TASK_START_STRIDE:-100000000}
GEN_SEED_STRIDE=${GEN_SEED_STRIDE:-1000003}
GEN_FORCE_SOFTWARE_RENDERING=${GEN_FORCE_SOFTWARE_RENDERING:-0}
DISPLAY_BASE=${DISPLAY_BASE:-$((200 + (${SLURM_JOB_ID:-0} % 200)))}
VNC_BASE=${VNC_BASE:-$((20000 + (${SLURM_JOB_ID:-0} % 10000)))}

# Validation set settings (fixed split for reproducibility).
VAL_BUILD_IF_MISSING=${VAL_BUILD_IF_MISSING:-1}
VAL_FORCE_REBUILD=${VAL_FORCE_REBUILD:-0}
VAL_NUM_TASKS=${VAL_NUM_TASKS:-100}
VAL_TASK_START=${VAL_TASK_START:-200000000}
VAL_SEED=${VAL_SEED:-424242}
DEBUG_VAL_DIAGNOSTICS=${DEBUG_VAL_DIAGNOSTICS:-0}

if [ ! -x "$RUNNER" ]; then
  echo "ERROR: runner not found or not executable: ${RUNNER:-<unset>}"
  echo "Tried candidates under:"
  echo "  SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-<unset>}"
  echo "  PWD=$PWD"
  echo "  SCRIPT_DIR=$SCRIPT_DIR"
  exit 1
fi

map_workspace_to_host() {
  local p="$1"
  if [[ "$p" == /workspace/data/* ]]; then
    printf "%s\n" "$DATA_DIR/${p#/workspace/data/}"
  else
    printf "%s\n" "$p"
  fi
}

DATA_FILE_GLOB='data_*.pt'
GEN_STORAGE_FORMAT='steps'

count_data_files() {
  local d="$1"
  if [ ! -d "$d" ]; then
    echo 0
    return
  fi
  find "$d" -maxdepth 1 -type f -name "$DATA_FILE_GLOB" | wc -l | tr -d ' '
}

debug_val_diagnostics() {
  local stage="$1"
  if [ "$DEBUG_VAL_DIAGNOSTICS" != "1" ]; then
    return
  fi

  echo "[VAL_DEBUG][$stage] DATA_DIR=$DATA_DIR"
  echo "[VAL_DEBUG][$stage] HOST_VAL_DIR=$HOST_VAL_DIR"
  echo "[VAL_DEBUG][$stage] VAL_DATA_DIR=$VAL_DATA_DIR"
  if [ -d "$HOST_VAL_DIR" ]; then
    local host_count
    host_count="$(count_data_files "$HOST_VAL_DIR")"
    echo "[VAL_DEBUG][$stage] host ${DATA_FILE_GLOB} count=$host_count"
    find "$HOST_VAL_DIR" -maxdepth 1 -type f -name "$DATA_FILE_GLOB" | sort | head -n 20
  else
    echo "[VAL_DEBUG][$stage] host val dir missing"
  fi

  local dbg_display dbg_vnc
  dbg_display=$((DISPLAY_BASE + 99))
  dbg_vnc=$((VNC_BASE + 99))
  RLBENCH_DISPLAY=$dbg_display RLBENCH_VNC_PORT=$dbg_vnc "$RUNNER" "set -euo pipefail
python - <<'PY'
import glob, os
p = \"$VAL_DATA_DIR\"
pattern = \"$DATA_FILE_GLOB\"
matches = sorted(glob.glob(os.path.join(p, pattern)))
print('[VAL_DEBUG][$stage] container cwd=', os.getcwd())
print('[VAL_DEBUG][$stage] container path=', p)
print('[VAL_DEBUG][$stage] container exists=', os.path.isdir(p))
print('[VAL_DEBUG][$stage] container count=', len(matches))
print('[VAL_DEBUG][$stage] container sample=', matches[:20])
parent = os.path.dirname(p.rstrip('/')) or '/'
print('[VAL_DEBUG][$stage] container parent=', parent)
try:
    print('[VAL_DEBUG][$stage] container parent entries=', sorted(os.listdir(parent))[:50])
except Exception as exc:
    print('[VAL_DEBUG][$stage] container parent list error=', repr(exc))
PY"
}

read_cpu_tot_idle() {
  awk '/^cpu / {idle=$5+$6; total=0; for (i=2; i<=9; i++) total+=$i; print total, idle; exit}' /proc/stat
}

MONITOR_PID=""
TRAINER_PID=""
TELEMETRY_LOG_CSV=""
start_system_monitor() {
  TELEMETRY_LOG_CSV="$DATA_DIR/logs/ip_hw_${SLURM_JOB_ID:-manual}.csv"
  cat > "$TELEMETRY_LOG_CSV" <<'EOF'
ts_unix,elapsed_s,cpu_pct,mem_used_gb,mem_total_gb,gpu_util_pct,gpu_mem_util_pct,gpu_mem_used_mb,gpu_mem_total_mb,gpu_power_w,gpu_temp_c,trainer_pid,trainer_rss_mb,trainer_cpu_pct,gen_alive,gen_rss_mb,gen_cpu_pct,train_items
EOF

  (
    set +e
    local start_now
    start_now="$(date +%s)"
    local sample_idx=0
    local prev_total prev_idle
    read -r prev_total prev_idle < <(read_cpu_tot_idle)

    while true; do
      sleep "$TELEMETRY_INTERVAL_SEC"
      sample_idx=$((sample_idx + 1))

      local now elapsed
      now="$(date +%s)"
      elapsed=$((now - start_now))

      local total idle dt_total dt_idle cpu_pct
      read -r total idle < <(read_cpu_tot_idle)
      dt_total=$((total - prev_total))
      dt_idle=$((idle - prev_idle))
      if [ "$dt_total" -gt 0 ]; then
        cpu_pct="$(awk -v t="$dt_total" -v i="$dt_idle" 'BEGIN {printf "%.2f", (100.0*(t-i))/t}')"
      else
        cpu_pct=""
      fi
      prev_total="$total"
      prev_idle="$idle"

      local mem_total_kb mem_avail_kb mem_used_kb mem_total_gb mem_used_gb
      mem_total_kb="$(awk '/MemTotal:/ {print $2; exit}' /proc/meminfo)"
      mem_avail_kb="$(awk '/MemAvailable:/ {print $2; exit}' /proc/meminfo)"
      mem_used_kb=$((mem_total_kb - mem_avail_kb))
      mem_total_gb="$(awk -v kb="$mem_total_kb" 'BEGIN {printf "%.2f", kb/1048576.0}')"
      mem_used_gb="$(awk -v kb="$mem_used_kb" 'BEGIN {printf "%.2f", kb/1048576.0}')"

      local gpu_line gpu_util gpu_mem_util gpu_mem_used gpu_mem_total gpu_power gpu_temp
      gpu_line="$(nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu --format=csv,noheader,nounits 2>/dev/null | head -n1 | tr -d ' ')"
      if [ -n "$gpu_line" ]; then
        IFS=',' read -r gpu_util gpu_mem_util gpu_mem_used gpu_mem_total gpu_power gpu_temp <<< "$gpu_line"
      else
        gpu_util=""
        gpu_mem_util=""
        gpu_mem_used=""
        gpu_mem_total=""
        gpu_power=""
        gpu_temp=""
      fi

      local trainer_pid trainer_rss_mb trainer_cpu_pct
      trainer_pid=""
      trainer_rss_mb=""
      trainer_cpu_pct=""
      if [ -n "${TRAINER_PID:-}" ] && kill -0 "$TRAINER_PID" 2>/dev/null; then
        trainer_pid="$TRAINER_PID"
        read -r trainer_rss_mb trainer_cpu_pct < <(
          ps -p "$TRAINER_PID" -o rss=,%cpu= 2>/dev/null | awk 'NF {printf "%.2f %.2f\n", $1/1024.0, $2; exit}'
        )
      fi

      local gen_alive gen_rss_mb gen_cpu_pct
      gen_alive=0
      gen_rss_mb=""
      gen_cpu_pct=""
      if [ "${#GEN_PIDS[@]}" -gt 0 ]; then
        local gen_pid_csv
        gen_pid_csv="$(IFS=,; echo "${GEN_PIDS[*]}")"
        if [ -n "$gen_pid_csv" ]; then
          gen_alive="$(ps -p "$gen_pid_csv" -o pid= 2>/dev/null | wc -l | tr -d ' ')"
          read -r gen_rss_mb gen_cpu_pct < <(
            ps -p "$gen_pid_csv" -o rss=,%cpu= 2>/dev/null | awk 'NF {rss+=$1; cpu+=$2} END {if (NR>0) printf "%.2f %.2f\n", rss/1024.0, cpu}'
          )
        fi
      fi

      local train_items
      train_items=""
      if [ $((sample_idx % TELEMETRY_TASK_COUNT_EVERY)) -eq 0 ]; then
        train_items="$(count_data_files "$HOST_TRAIN_DIR")"
      fi

      printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" \
        "$now" "$elapsed" "$cpu_pct" "$mem_used_gb" "$mem_total_gb" \
        "$gpu_util" "$gpu_mem_util" "$gpu_mem_used" "$gpu_mem_total" "$gpu_power" "$gpu_temp" \
        "$trainer_pid" "$trainer_rss_mb" "$trainer_cpu_pct" "$gen_alive" "$gen_rss_mb" "$gen_cpu_pct" "$train_items" \
        >> "$TELEMETRY_LOG_CSV"
    done
  ) &
  MONITOR_PID="$!"
  echo "hardware telemetry log: $TELEMETRY_LOG_CSV (interval=${TELEMETRY_INTERVAL_SEC}s)"
}

stop_system_monitor() {
  if [ -n "${MONITOR_PID:-}" ] && kill -0 "$MONITOR_PID" 2>/dev/null; then
    kill "$MONITOR_PID" 2>/dev/null || true
    wait "$MONITOR_PID" 2>/dev/null || true
  fi
}

# Resolve ShapeNet path robustly.
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
HOST_TRAIN_DIR="$(map_workspace_to_host "$TRAIN_DATA_DIR")"
HOST_VAL_DIR="$(map_workspace_to_host "$VAL_DATA_DIR")"
HOST_MODEL_DIR="$(map_workspace_to_host "$MODEL_PATH")"
HOST_MESH_PATH="$(map_workspace_to_host "$MESH_PATH")"
HOST_INDEX_PATH="$(map_workspace_to_host "$INDEX_PATH")"
HOST_SAVE_ROOT="$(map_workspace_to_host "$SAVE_ROOT")"

if [ ! -d "$HOST_SHAPENET" ]; then
  echo "ERROR: ShapeNet path not found on host: $HOST_SHAPENET"
  echo "       Tried SHAPENET_PATH=$SHAPENET_PATH"
  echo "       Common choices under DATA_DIR ($DATA_DIR): shapenet or ShapeNetCore.v2"
  echo "       Example: sbatch --export=ALL,SHAPENET_PATH=/workspace/data/shapenet train_instant_policy.slurm"
  exit 1
fi

if [ "$FINE_TUNE" = "1" ] && [ ! -d "$HOST_MODEL_DIR" ]; then
  echo "ERROR: model path not found on host (fine-tune enabled): $HOST_MODEL_DIR"
  exit 1
fi

mkdir -p "$DATA_DIR/logs"
mkdir -p "$(dirname "$HOST_MESH_PATH")" "$(dirname "$HOST_INDEX_PATH")" "$HOST_TRAIN_DIR" "$HOST_VAL_DIR" "$HOST_SAVE_ROOT"

start_ts=$(date +%s)
echo "Starting unified pipeline on $(hostname)"
echo "run_name=$RUN_NAME format=$DATA_FORMAT batch_size=$BATCH_SIZE"
echo "shapenet_path=$SHAPENET_PATH (host: $HOST_SHAPENET)"
echo "auto_resume=$AUTO_RESUME use_wandb=$USE_WANDB"
echo "compile_models=$COMPILE_MODELS devices=$TRAIN_DEVICES strategy=$TRAIN_STRATEGY"
echo "num_workers=$TRAIN_NUM_WORKERS prefetch=$TRAIN_PREFETCH_FACTOR val_interval=$TRAIN_VAL_CHECK_INTERVAL"
echo "ring_buffer_size=$TRAIN_BUFFER_SIZE min_bootstrap_items=$MIN_BOOTSTRAP_ITEMS"
echo "parallel_generators=$ENABLE_PARALLEL_GENERATORS num_shards=$GEN_NUM_SHARDS chunk_tasks=$GEN_CHUNK_TASKS"
echo "data_file_glob=$DATA_FILE_GLOB storage_format=$GEN_STORAGE_FORMAT gen_force_software=$GEN_FORCE_SOFTWARE_RENDERING"
if [ "$DEBUG_VAL_DIAGNOSTICS" = "1" ]; then
  echo "DEBUG_VAL_DIAGNOSTICS=1"
fi
if [ -n "$NUM_ITERS_OVERRIDE" ]; then
  echo "num_iters_override=$NUM_ITERS_OVERRIDE"
fi
if [ -n "$RESUME_CKPT" ]; then
  echo "resume_ckpt=$RESUME_CKPT"
fi

# Ensure gripper mesh exists.
if [ ! -f "$HOST_MESH_PATH" ]; then
  echo "Mesh not found, building: $HOST_MESH_PATH"
  "$RUNNER" "set -euo pipefail
python -m ip.scripts.build_robotiq_mesh --out \"$MESH_PATH\" --source collision --state open"
fi

# Build/rebuild validation set if needed.
debug_val_diagnostics "before_val_build"
VAL_COUNT="$(count_data_files "$HOST_VAL_DIR")"
if [ "$VAL_FORCE_REBUILD" = "1" ]; then
  echo "VAL_FORCE_REBUILD=1, clearing validation files (${DATA_FILE_GLOB}) in $HOST_VAL_DIR"
  find "$HOST_VAL_DIR" -maxdepth 1 -type f -name "$DATA_FILE_GLOB" -delete
  VAL_COUNT=0
fi
if [ "$VAL_BUILD_IF_MISSING" = "1" ] && [ "$VAL_COUNT" -eq 0 ]; then
  echo "Building validation set..."
  "$RUNNER" "set -euo pipefail
export PYOPENGL_PLATFORM=\${PYOPENGL_PLATFORM:-egl}
python -m ip.scripts.generate_pseudo_demos \
  --shapenet_path \"$SHAPENET_PATH\" \
  --shapenet_index_path \"$INDEX_PATH\" \
  --save_dir \"$VAL_DATA_DIR\" \
  --num_tasks $VAL_NUM_TASKS \
  --task_start $VAL_TASK_START \
  --seed $VAL_SEED \
  --storage_format $GEN_STORAGE_FORMAT \
  --pcd_storage_dtype $PCD_DTYPE \
  --num_demos_per_task $DEMOS_PER_TASK_MIN $DEMOS_PER_TASK_MAX \
  --gripper_mesh_path \"$MESH_PATH\""
fi
debug_val_diagnostics "after_val_build"

# Minimal bootstrap only: enough to avoid empty train dataset.
TRAIN_COUNT="$(count_data_files "$HOST_TRAIN_DIR")"
if [ "$MIN_BOOTSTRAP_ITEMS" -gt "$TRAIN_BUFFER_SIZE" ]; then
  echo "WARNING: MIN_BOOTSTRAP_ITEMS=$MIN_BOOTSTRAP_ITEMS exceeds TRAIN_BUFFER_SIZE=$TRAIN_BUFFER_SIZE; clamping."
  MIN_BOOTSTRAP_ITEMS="$TRAIN_BUFFER_SIZE"
fi
if [ "$TRAIN_COUNT" -lt "$MIN_BOOTSTRAP_ITEMS" ]; then
  echo "Bootstrapping train ring to target=$MIN_BOOTSTRAP_ITEMS from current=$TRAIN_COUNT"
  BOOTSTRAP_ATTEMPT=0
  while [ "$TRAIN_COUNT" -lt "$MIN_BOOTSTRAP_ITEMS" ]; do
    BOOTSTRAP_ATTEMPT=$((BOOTSTRAP_ATTEMPT + 1))
    if [ "$BOOTSTRAP_ATTEMPT" -gt 8 ]; then
      echo "ERROR: bootstrap did not reach target after $BOOTSTRAP_ATTEMPT attempts (current=$TRAIN_COUNT target=$MIN_BOOTSTRAP_ITEMS)."
      exit 1
    fi
    "$RUNNER" "set -euo pipefail
export PYOPENGL_PLATFORM=\${PYOPENGL_PLATFORM:-egl}
python -m ip.scripts.generate_pseudo_demos \
  --shapenet_path \"$SHAPENET_PATH\" \
  --shapenet_index_path \"$INDEX_PATH\" \
  --save_dir \"$TRAIN_DATA_DIR\" \
  --num_tasks $TRAIN_BUFFER_SIZE \
  --task_start $TRAIN_COUNT \
  --append \
  --seed $TRAIN_GEN_SEED \
  --storage_format $GEN_STORAGE_FORMAT \
  --pcd_storage_dtype $PCD_DTYPE \
  --buffer_size $MIN_BOOTSTRAP_ITEMS \
  --fill_buffer \
  --num_shards 1 \
  --shard_id 0 \
  --num_demos_per_task $DEMOS_PER_TASK_MIN $DEMOS_PER_TASK_MAX \
  --gripper_mesh_path \"$MESH_PATH\""
    TRAIN_COUNT="$(count_data_files "$HOST_TRAIN_DIR")"
    echo "  bootstrap attempt=$BOOTSTRAP_ATTEMPT train_items=$TRAIN_COUNT target=$MIN_BOOTSTRAP_ITEMS"
  done
fi

VAL_COUNT="$(count_data_files "$HOST_VAL_DIR")"
TRAIN_COUNT="$(count_data_files "$HOST_TRAIN_DIR")"
echo "val_items=$VAL_COUNT train_items=$TRAIN_COUNT"
debug_val_diagnostics "before_train"
if [ "$VAL_COUNT" -eq 0 ]; then
  echo "ERROR: validation set is empty at $HOST_VAL_DIR"
  exit 1
fi
if [ "$TRAIN_COUNT" -eq 0 ]; then
  echo "ERROR: training ring buffer is empty at $HOST_TRAIN_DIR"
  exit 1
fi

ACTIVE_BUFFER_SIZE="$TRAIN_COUNT"
REQUESTED_GEN_WORKERS="$GEN_NUM_SHARDS"
GEN_WORKERS=0
if [ "$REQUESTED_GEN_WORKERS" -gt 0 ]; then
  GEN_WORKERS="$REQUESTED_GEN_WORKERS"
  if [ "$GEN_WORKERS" -gt "$ACTIVE_BUFFER_SIZE" ]; then
    GEN_WORKERS="$ACTIVE_BUFFER_SIZE"
  fi
fi
echo "active_ring_size=$ACTIVE_BUFFER_SIZE requested_gen_workers=$REQUESTED_GEN_WORKERS gen_workers=$GEN_WORKERS display_base=$DISPLAY_BASE"

GEN_PIDS=()
start_generators() {
  if [ "$ENABLE_PARALLEL_GENERATORS" != "1" ]; then
    echo "Parallel generators disabled."
    return
  fi
  if [ "$GEN_WORKERS" -le 0 ]; then
    echo "Parallel generators disabled because gen_workers=$GEN_WORKERS"
    return
  fi

  echo "Starting $GEN_WORKERS sharded generator workers..."
  local shard
  for shard in $(seq 0 $((GEN_WORKERS - 1))); do
    local shard_seed=$((TRAIN_GEN_SEED + shard * GEN_SEED_STRIDE))
    local shard_task_start=$((GEN_TASK_START_BASE + shard * GEN_TASK_START_STRIDE))
    local shard_log="$DATA_DIR/logs/ip_gen_${SLURM_JOB_ID:-manual}_shard${shard}.log"

    SHARD_DISPLAY=$((DISPLAY_BASE + 1 + shard))
    SHARD_VNC_PORT=$((VNC_BASE + 1 + shard))
    RLBENCH_DISPLAY=$SHARD_DISPLAY RLBENCH_VNC_PORT=$SHARD_VNC_PORT FORCE_SOFTWARE_RENDERING=$GEN_FORCE_SOFTWARE_RENDERING "$RUNNER" "set -euo pipefail
export PYOPENGL_PLATFORM=\${PYOPENGL_PLATFORM:-egl}
TASK_START=$shard_task_start
while true; do
  T0=\$(date +%s)
  python -m ip.scripts.generate_pseudo_demos \
    --shapenet_path \"$SHAPENET_PATH\" \
    --shapenet_index_path \"$INDEX_PATH\" \
    --save_dir \"$TRAIN_DATA_DIR\" \
    --num_tasks $GEN_CHUNK_TASKS \
    --task_start \$TASK_START \
    --append \
    --seed $shard_seed \
    --storage_format $GEN_STORAGE_FORMAT \
    --pcd_storage_dtype $PCD_DTYPE \
    --buffer_size $ACTIVE_BUFFER_SIZE \
    --num_shards $GEN_WORKERS \
    --shard_id $shard \
    --num_demos_per_task $DEMOS_PER_TASK_MIN $DEMOS_PER_TASK_MAX \
    --gripper_mesh_path \"$MESH_PATH\"
  TASK_START=\$((TASK_START + $GEN_CHUNK_TASKS))
  T1=\$(date +%s)
  echo \"[shard $shard] ts=\$T1 chunk=$GEN_CHUNK_TASKS dt=\$((T1 - T0))s next_task_start=\$TASK_START\"
done" >"$shard_log" 2>&1 &

    local gen_pid="$!"
    GEN_PIDS+=("$gen_pid")
    echo "  shard=$shard pid=$gen_pid log=$shard_log seed=$shard_seed start_task=$shard_task_start"
  done
}

stop_generators() {
  if [ "${#GEN_PIDS[@]}" -eq 0 ]; then
    return
  fi
  echo "Stopping ${#GEN_PIDS[@]} generator workers..."
  kill "${GEN_PIDS[@]}" 2>/dev/null || true
  wait "${GEN_PIDS[@]}" 2>/dev/null || true
}
cleanup_all() {
  stop_generators
  stop_system_monitor
}
trap cleanup_all EXIT INT TERM

start_system_monitor
start_generators

ITER_OVERRIDE_ARGS=""
if [ -n "$NUM_ITERS_OVERRIDE" ]; then
  ITER_OVERRIDE_ARGS="--num_iters_override $NUM_ITERS_OVERRIDE"
fi

RESUME_ARGS=""
if [ -n "$RESUME_CKPT" ]; then
  RESUME_ARGS="--resume_ckpt_path \"$RESUME_CKPT\""
elif [ "$AUTO_RESUME" = "1" ]; then
  HOST_RUN_DIR="$HOST_SAVE_ROOT/$RUN_NAME"
  if [ -d "$HOST_RUN_DIR" ] && find "$HOST_RUN_DIR" -maxdepth 1 -type f -name '*.pt' | grep -q .; then
    RESUME_ARGS="--auto_resume"
  else
    echo "AUTO_RESUME=1 but no checkpoint found in $HOST_RUN_DIR; starting fresh."
  fi
fi

WANDB_ARGS=""
if [ "$USE_WANDB" = "1" ]; then
  WANDB_ARGS="--wandb_resume $WANDB_RESUME"
  if [ -n "$WANDB_ID" ]; then
    WANDB_ARGS="$WANDB_ARGS --wandb_id \"$WANDB_ID\""
  fi
fi

train_start_ts=$(date +%s)
TRAIN_DISPLAY=${TRAIN_DISPLAY:-$DISPLAY_BASE}
TRAIN_VNC_PORT=${TRAIN_VNC_PORT:-$VNC_BASE}
RLBENCH_DISPLAY=$TRAIN_DISPLAY RLBENCH_VNC_PORT=$TRAIN_VNC_PORT "$RUNNER" "set -euo pipefail
export IP_DEBUG_PATHS=$DEBUG_VAL_DIAGNOSTICS
mkdir -p \"$SAVE_ROOT\"
python /workspace/instant_policy/ip/train.py \
  --run_name \"$RUN_NAME\" \
  --record $RECORD \
  --use_wandb $USE_WANDB \
  --fine_tune $FINE_TUNE \
  --compile_models $COMPILE_MODELS \
  --model_path \"$MODEL_PATH\" \
  --model_name \"$MODEL_NAME\" \
  --data_path_train \"$TRAIN_DATA_DIR\" \
  --data_path_val \"$VAL_DATA_DIR\" \
  --batch_size $BATCH_SIZE \
  --num_workers $TRAIN_NUM_WORKERS \
  --persistent_workers $TRAIN_PERSISTENT_WORKERS \
  --prefetch_factor $TRAIN_PREFETCH_FACTOR \
  --val_check_interval $TRAIN_VAL_CHECK_INTERVAL \
  --log_every_n_steps $TRAIN_LOG_EVERY_N_STEPS \
  --devices $TRAIN_DEVICES \
  --strategy \"$TRAIN_STRATEGY\" \
  --save_path \"$SAVE_ROOT\" \
  $RESUME_ARGS \
  $WANDB_ARGS \
  $ITER_OVERRIDE_ARGS \
  $EXTRA_TRAIN_ARGS" &
TRAINER_PID="$!"

train_rc=0
wait "$TRAINER_PID" || train_rc=$?
TRAINER_PID=""
if [ "$train_rc" -ne 0 ]; then
  echo "ERROR: training process exited with code $train_rc"
  exit "$train_rc"
fi
train_end_ts=$(date +%s)
echo "Training wall time: $((train_end_ts - train_start_ts)) seconds"

end_ts=$(date +%s)
echo "Finished unified pipeline in $((end_ts - start_ts)) seconds"
