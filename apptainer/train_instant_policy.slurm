#!/bin/bash
# Unified MSI entrypoint:
# 1) Ensure val set
# 2) Start continuous pseudo-demo generation in background
# 3) Wait for minimal train data, then run training (resume-friendly across 24h jobs)
#
# Re-submit this same script for each 24h window.
# Generator log is written to:
#   /scratch.global/$USER/ips/logs/ip_gen_${SLURM_JOB_ID}.log
#
# Defaults are chosen for one clean pipeline (no explicit bootstrap):
# - Continuous single generator process (1-GPU stable default)
# - Ring buffer overwrite behavior via --buffer_size
# - Train starts after a minimal number of files exist
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
TRAIN_NUM_WORKERS=${TRAIN_NUM_WORKERS:-8}
TRAIN_PERSISTENT_WORKERS=${TRAIN_PERSISTENT_WORKERS:-1}
TRAIN_PREFETCH_FACTOR=${TRAIN_PREFETCH_FACTOR:-4}
TRAIN_SAMPLE_CACHE_SIZE=${TRAIN_SAMPLE_CACHE_SIZE:-2048}
TRAIN_VAL_CHECK_INTERVAL=${TRAIN_VAL_CHECK_INTERVAL:-20000}
TRAIN_LOG_EVERY_N_STEPS=${TRAIN_LOG_EVERY_N_STEPS:-500}

# Ring/generation settings.
TRAIN_BUFFER_SIZE=${TRAIN_BUFFER_SIZE:-8192}
TRAIN_GEN_SEED=${TRAIN_GEN_SEED:-1337}
DEMOS_PER_TASK_MIN=${DEMOS_PER_TASK_MIN:-2}
DEMOS_PER_TASK_MAX=${DEMOS_PER_TASK_MAX:-4}
PCD_DTYPE=${PCD_DTYPE:-float16}
TRAIN_START_MIN_ITEMS=${TRAIN_START_MIN_ITEMS:-512}
TRAIN_START_TIMEOUT_SEC=${TRAIN_START_TIMEOUT_SEC:-7200}

# Profiling/telemetry settings (for 1h decision runs).
TELEMETRY_INTERVAL_SEC=${TELEMETRY_INTERVAL_SEC:-30}
TELEMETRY_TASK_COUNT_EVERY=${TELEMETRY_TASK_COUNT_EVERY:-10}

# Continuous producer settings.
GEN_CHUNK_TASKS=${GEN_CHUNK_TASKS:-256}
GEN_TASK_START=${GEN_TASK_START:-1000000}
DISPLAY_BASE=${DISPLAY_BASE:-$((200 + (${SLURM_JOB_ID:-0} % 200)))}
VNC_BASE=${VNC_BASE:-$((20000 + (${SLURM_JOB_ID:-0} % 10000)))}

# Validation set settings (fixed split for reproducibility).
VAL_BUILD_IF_MISSING=${VAL_BUILD_IF_MISSING:-1}
VAL_FORCE_REBUILD=${VAL_FORCE_REBUILD:-0}
VAL_NUM_TASKS=${VAL_NUM_TASKS:-100}
VAL_TASK_START=${VAL_TASK_START:-200000000}
VAL_SEED=${VAL_SEED:-424242}

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

read_cpu_tot_idle() {
  awk '/^cpu / {idle=$5+$6; total=0; for (i=2; i<=9; i++) total+=$i; print total, idle; exit}' /proc/stat
}

MONITOR_PID=""
TRAINER_PID=""
TELEMETRY_LOG_CSV=""
TRAINER_PID_FILE=""
GEN_PIDS_FILE=""
start_system_monitor() {
  TELEMETRY_LOG_CSV="$DATA_DIR/logs/ip_hw_${SLURM_JOB_ID:-manual}.csv"
  TRAINER_PID_FILE="$DATA_DIR/logs/ip_trainer_${SLURM_JOB_ID:-manual}.pid"
  GEN_PIDS_FILE="$DATA_DIR/logs/ip_gen_${SLURM_JOB_ID:-manual}.pids"
  : > "$TRAINER_PID_FILE"
  : > "$GEN_PIDS_FILE"
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
      local tp=""
      if [ -f "$TRAINER_PID_FILE" ]; then
        tp="$(tr -d ' \n\r\t' < "$TRAINER_PID_FILE" 2>/dev/null || true)"
      fi
      if [ -n "$tp" ] && kill -0 "$tp" 2>/dev/null; then
        trainer_pid="$tp"
        read -r trainer_rss_mb trainer_cpu_pct < <(
          ps -p "$tp" -o rss=,%cpu= 2>/dev/null | awk 'NF {printf "%.2f %.2f\n", $1/1024.0, $2; exit}'
        )
      fi

      local gen_alive gen_rss_mb gen_cpu_pct
      gen_alive=0
      gen_rss_mb=""
      gen_cpu_pct=""
      local gen_pid_csv=""
      if [ -f "$GEN_PIDS_FILE" ]; then
        gen_pid_csv="$(tr '\n' ',' < "$GEN_PIDS_FILE" | sed 's/,$//' 2>/dev/null || true)"
      fi
      if [ -n "$gen_pid_csv" ]; then
        gen_alive="$(ps -p "$gen_pid_csv" -o pid= 2>/dev/null | wc -l | tr -d ' ')"
        read -r gen_rss_mb gen_cpu_pct < <(
          ps -p "$gen_pid_csv" -o rss=,%cpu= 2>/dev/null | awk 'NF {rss+=$1; cpu+=$2} END {if (NR>0) printf "%.2f %.2f\n", rss/1024.0, cpu}'
        )
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
  [ -n "${TRAINER_PID_FILE:-}" ] && rm -f "$TRAINER_PID_FILE" 2>/dev/null || true
  [ -n "${GEN_PIDS_FILE:-}" ] && rm -f "$GEN_PIDS_FILE" 2>/dev/null || true
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
echo "compile_models=$COMPILE_MODELS"
echo "num_workers=$TRAIN_NUM_WORKERS prefetch=$TRAIN_PREFETCH_FACTOR val_interval=$TRAIN_VAL_CHECK_INTERVAL"
echo "sample_cache_size=$TRAIN_SAMPLE_CACHE_SIZE"
echo "ring_buffer_size=$TRAIN_BUFFER_SIZE train_start_min_items=$TRAIN_START_MIN_ITEMS timeout_s=$TRAIN_START_TIMEOUT_SEC"
echo "generator=single chunk_tasks=$GEN_CHUNK_TASKS"
echo "data_file_glob=$DATA_FILE_GLOB storage_format=$GEN_STORAGE_FORMAT"
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

VAL_COUNT="$(count_data_files "$HOST_VAL_DIR")"
TRAIN_COUNT="$(count_data_files "$HOST_TRAIN_DIR")"
echo "val_items=$VAL_COUNT train_items=$TRAIN_COUNT"
if [ "$VAL_COUNT" -eq 0 ]; then
  echo "ERROR: validation set is empty at $HOST_VAL_DIR"
  exit 1
fi
if [ "$TRAIN_START_MIN_ITEMS" -lt 1 ]; then
  TRAIN_START_MIN_ITEMS=1
fi
if [ "$TRAIN_START_MIN_ITEMS" -gt "$TRAIN_BUFFER_SIZE" ]; then
  echo "WARNING: TRAIN_START_MIN_ITEMS=$TRAIN_START_MIN_ITEMS exceeds TRAIN_BUFFER_SIZE=$TRAIN_BUFFER_SIZE; clamping."
  TRAIN_START_MIN_ITEMS="$TRAIN_BUFFER_SIZE"
fi

ACTIVE_BUFFER_SIZE="$TRAIN_BUFFER_SIZE"
echo "active_ring_size=$ACTIVE_BUFFER_SIZE generator_workers=1 display_base=$DISPLAY_BASE"

GEN_PIDS=()
start_generators() {
  echo "Starting single generator worker..."
  local gen_log="$DATA_DIR/logs/ip_gen_${SLURM_JOB_ID:-manual}.log"
  local gen_display=$((DISPLAY_BASE + 1))
  local gen_vnc_port=$((VNC_BASE + 1))
  local gen_seed="$TRAIN_GEN_SEED"
  RLBENCH_DISPLAY=$gen_display RLBENCH_VNC_PORT=$gen_vnc_port "$RUNNER" "set -euo pipefail
export PYOPENGL_PLATFORM=\${PYOPENGL_PLATFORM:-egl}
TASK_START=$GEN_TASK_START
while true; do
  T0=\$(date +%s)
  python -m ip.scripts.generate_pseudo_demos \
    --shapenet_path \"$SHAPENET_PATH\" \
    --shapenet_index_path \"$INDEX_PATH\" \
    --save_dir \"$TRAIN_DATA_DIR\" \
    --num_tasks $GEN_CHUNK_TASKS \
    --task_start \$TASK_START \
    --append \
    --seed $gen_seed \
    --storage_format $GEN_STORAGE_FORMAT \
    --pcd_storage_dtype $PCD_DTYPE \
    --buffer_size $ACTIVE_BUFFER_SIZE \
    --num_demos_per_task $DEMOS_PER_TASK_MIN $DEMOS_PER_TASK_MAX \
    --gripper_mesh_path \"$MESH_PATH\"
  TASK_START=\$((TASK_START + $GEN_CHUNK_TASKS))
  T1=\$(date +%s)
  echo \"[generator] ts=\$T1 chunk=$GEN_CHUNK_TASKS dt=\$((T1 - T0))s next_task_start=\$TASK_START\"
done" >"$gen_log" 2>&1 &

  local gen_pid="$!"
  GEN_PIDS+=("$gen_pid")
  if [ -n "${GEN_PIDS_FILE:-}" ]; then
    printf "%s\n" "$gen_pid" >> "$GEN_PIDS_FILE"
  fi
  echo "  pid=$gen_pid log=$gen_log seed=$gen_seed start_task=$GEN_TASK_START"
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

wait_for_train_data() {
  local target="$1"
  local timeout="$2"
  local interval=20
  local waited=0
  local current=0

  while true; do
    current="$(count_data_files "$HOST_TRAIN_DIR")"
    if [ "$current" -ge "$target" ]; then
      echo "Train data ready: train_items=$current target=$target"
      break
    fi
    if [ "$waited" -ge "$timeout" ]; then
      echo "ERROR: timed out waiting for train data (have=$current target=$target timeout=${timeout}s)."
      exit 1
    fi
    echo "Waiting for train data: have=$current target=$target waited=${waited}s"
    sleep "$interval"
    waited=$((waited + interval))
  done
}

wait_for_train_data "$TRAIN_START_MIN_ITEMS" "$TRAIN_START_TIMEOUT_SEC"

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
  --sample_cache_size $TRAIN_SAMPLE_CACHE_SIZE \
  --val_check_interval $TRAIN_VAL_CHECK_INTERVAL \
  --log_every_n_steps $TRAIN_LOG_EVERY_N_STEPS \
  --save_path \"$SAVE_ROOT\" \
  $RESUME_ARGS \
  $WANDB_ARGS" &
TRAINER_PID="$!"
if [ -n "${TRAINER_PID_FILE:-}" ]; then
  printf "%s\n" "$TRAINER_PID" > "$TRAINER_PID_FILE"
fi

train_rc=0
wait "$TRAINER_PID" || train_rc=$?
TRAINER_PID=""
if [ -n "${TRAINER_PID_FILE:-}" ]; then
  : > "$TRAINER_PID_FILE"
fi
if [ "$train_rc" -ne 0 ]; then
  echo "ERROR: training process exited with code $train_rc"
  exit "$train_rc"
fi
train_end_ts=$(date +%s)
echo "Training wall time: $((train_end_ts - train_start_ts)) seconds"

end_ts=$(date +%s)
echo "Finished unified pipeline in $((end_ts - start_ts)) seconds"
