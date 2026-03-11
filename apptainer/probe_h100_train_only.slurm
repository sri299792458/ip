#!/bin/bash
# Trainer-only H100 probe:
# 1. Prefill a local dataset on the allocated H100.
# 2. Stop generation entirely.
# 3. Run train_h100.py against the static local dataset.

#SBATCH -A kdesingh
#SBATCH -p msigpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=220G
#SBATCH --tmp=300G
#SBATCH -t 06:00:00
#SBATCH --job-name=ip_h100_train_only
#SBATCH --output=ip_h100_train_only_%j.out
#SBATCH --error=ip_h100_train_only_%j.err

set -Eeuo pipefail
trap 'rc=$?; echo "ERROR: ${BASH_SOURCE[0]}:${LINENO}: ${BASH_COMMAND}" >&2; exit "$rc"' ERR

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd -P)"

PROJECT_DIR="${PROJECT_DIR:-$REPO_ROOT}"
INSTANT_POLICY_DIR="${INSTANT_POLICY_DIR:-$PROJECT_DIR/instant_policy}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-$SCRIPT_DIR/instant_policy.sif}"

STATIC_DATA_DIR="${STATIC_DATA_DIR:-/scratch.global/$USER/ips}"
SHAPENET_PATH="${SHAPENET_PATH:-$STATIC_DATA_DIR/shapenet}"
INDEX_PATH="${INDEX_PATH:-$SHAPENET_PATH/index.json}"
SCENE_ENCODER_PATH="${SCENE_ENCODER_PATH:-$STATIC_DATA_DIR/checkpoints/scene_encoder.pt}"
MESH_PATH="${MESH_PATH:-$STATIC_DATA_DIR/assets/robotiq_2f85_collision_open.obj}"
RUNS_ROOT="${RUNS_ROOT:-$STATIC_DATA_DIR/runs_policy}"

BATCH_SIZE="${BATCH_SIZE:-16}"
TRAIN_NUM_WORKERS="${TRAIN_NUM_WORKERS:-16}"
PROBE_STEPS="${PROBE_STEPS:-2000}"
PREFILL_MIN_ITEMS="${PREFILL_MIN_ITEMS:-$((256 * BATCH_SIZE))}"
PREFILL_CHUNK_TASKS="${PREFILL_CHUNK_TASKS:-128}"
SAVE_EVERY="${SAVE_EVERY:-$((PROBE_STEPS + 1))}"
RUN_NAME="${RUN_NAME:-ip_train_only_b${BATCH_SIZE}_${SLURM_JOB_ID:-manual}}"

NUM_DEMOS_PER_TASK=3
PCD_DTYPE=float16
TRAIN_GEN_SEED=1337
GEN_TASK_START=2000000

if [ ! -d /scratch.local ]; then
  echo "ERROR: /scratch.local is unavailable; this job expects MSI local scratch." >&2
  exit 1
fi

resolve_existing_dir() {
  local path="$1"
  local label="$2"
  if [ ! -d "$path" ]; then
    echo "ERROR: required directory not found for $label: $path" >&2
    exit 1
  fi
  (cd "$path" && pwd -P)
}

resolve_existing_file() {
  local path="$1"
  local label="$2"
  if [ ! -f "$path" ]; then
    echo "ERROR: required file not found for $label: $path" >&2
    exit 1
  fi
  local dir
  dir="$(dirname "$path")"
  if [ ! -d "$dir" ]; then
    echo "ERROR: parent directory not found for $label: $dir" >&2
    exit 1
  fi
  dir="$(cd "$dir" && pwd -P)"
  printf "%s/%s\n" "$dir" "$(basename "$path")"
}

resolve_path_with_existing_parent() {
  local path="$1"
  local label="$2"
  local dir
  dir="$(dirname "$path")"
  if [ ! -d "$dir" ]; then
    echo "ERROR: parent directory not found for $label: $dir" >&2
    exit 1
  fi
  dir="$(cd "$dir" && pwd -P)"
  printf "%s/%s\n" "$dir" "$(basename "$path")"
}

resolve_container_runtime() {
  if command -v apptainer >/dev/null 2>&1; then
    printf "apptainer\n"
    return
  fi
  if command -v singularity >/dev/null 2>&1; then
    printf "singularity\n"
    return
  fi
  echo "ERROR: neither apptainer nor singularity is available in PATH" >&2
  exit 1
}

PROJECT_DIR="$(resolve_existing_dir "$PROJECT_DIR" "PROJECT_DIR")"
INSTANT_POLICY_DIR="$(resolve_existing_dir "$INSTANT_POLICY_DIR" "INSTANT_POLICY_DIR")"
mkdir -p "$STATIC_DATA_DIR"
STATIC_DATA_DIR="$(resolve_existing_dir "$STATIC_DATA_DIR" "STATIC_DATA_DIR")"
mkdir -p "$RUNS_ROOT"
RUNS_ROOT="$(resolve_existing_dir "$RUNS_ROOT" "RUNS_ROOT")"
SHAPENET_PATH="$(resolve_existing_dir "$SHAPENET_PATH" "SHAPENET_PATH")"
SCENE_ENCODER_PATH="$(resolve_existing_file "$SCENE_ENCODER_PATH" "SCENE_ENCODER_PATH")"
MESH_PATH="$(resolve_existing_file "$MESH_PATH" "MESH_PATH")"
INDEX_PATH="$(resolve_path_with_existing_parent "$INDEX_PATH" "INDEX_PATH")"
CONTAINER_IMAGE="$(resolve_existing_file "$CONTAINER_IMAGE" "CONTAINER_IMAGE")"
CONTAINER_RUNTIME="$(resolve_container_runtime)"

HOT_DATA_DIR="/scratch.local/$USER/ip_train_only_${RUN_NAME}_${SLURM_JOB_ID:-manual}"
HOT_TRAIN_DIR="$HOT_DATA_DIR/train_prefill"
mkdir -p "$HOT_TRAIN_DIR"

cleanup_all() {
  rm -rf "$HOT_DATA_DIR" 2>/dev/null || true
}
trap cleanup_all EXIT INT TERM

find "$HOT_TRAIN_DIR" -maxdepth 1 -type f -name 'data_*.pt' -delete

HOST_NETRC="${HOME}/.netrc"
NETRC_BIND=""
if [ -f "$HOST_NETRC" ]; then
  NETRC_BIND="--bind $HOST_NETRC:/workspace/hot/.netrc"
fi

SHAPENET_BIND_HOST=""
SHAPENET_BIND_CONTAINER=""
EXTRA_BINDS=""
if [[ "$SHAPENET_PATH" == "$STATIC_DATA_DIR"/* || "$SHAPENET_PATH" == "$STATIC_DATA_DIR" ]]; then
  :
else
  SHAPENET_BIND_HOST="$SHAPENET_PATH"
  SHAPENET_BIND_CONTAINER="/workspace/shapenet"
  EXTRA_BINDS="--bind $SHAPENET_BIND_HOST:$SHAPENET_BIND_CONTAINER"
fi

to_container_path() {
  local host_path="$1"
  if [ "$host_path" = "$STATIC_DATA_DIR" ]; then
    printf "/workspace/static\n"
    return
  fi
  if [[ "$host_path" == "$STATIC_DATA_DIR"/* ]]; then
    printf "/workspace/static/%s\n" "${host_path#"$STATIC_DATA_DIR"/}"
    return
  fi
  if [ "$host_path" = "$HOT_DATA_DIR" ]; then
    printf "/workspace/hot\n"
    return
  fi
  if [[ "$host_path" == "$HOT_DATA_DIR"/* ]]; then
    printf "/workspace/hot/%s\n" "${host_path#"$HOT_DATA_DIR"/}"
    return
  fi
  if [ -n "$SHAPENET_BIND_HOST" ]; then
    if [ "$host_path" = "$SHAPENET_BIND_HOST" ]; then
      printf "%s\n" "$SHAPENET_BIND_CONTAINER"
      return
    fi
    if [[ "$host_path" == "$SHAPENET_BIND_HOST"/* ]]; then
      printf "%s/%s\n" "$SHAPENET_BIND_CONTAINER" "${host_path#"$SHAPENET_BIND_HOST"/}"
      return
    fi
  fi
  echo "ERROR: unmapped host path: $host_path" >&2
  return 1
}

container_exec() {
  local cmd="$1"
  "$CONTAINER_RUNTIME" exec --nv --cleanenv --no-home \
    $NETRC_BIND \
    $EXTRA_BINDS \
    --bind "$INSTANT_POLICY_DIR:/workspace/instant_policy" \
    --bind "$STATIC_DATA_DIR:/workspace/static" \
    --bind "$HOT_DATA_DIR:/workspace/hot" \
    --pwd /workspace/instant_policy/ip \
    "$CONTAINER_IMAGE" \
    bash -lc "
      export HOME=/workspace/hot
      export XDG_CACHE_HOME=/workspace/hot/.cache
      export HF_HOME=/workspace/hot/.cache/huggingface
      export MPLCONFIGDIR=/workspace/hot/.cache/matplotlib
      mkdir -p /workspace/hot/.cache/huggingface /workspace/hot/.cache/matplotlib
      if [ -f /workspace/hot/.netrc ]; then
        chmod 600 /workspace/hot/.netrc || true
      fi
      export PYTHONPATH=/workspace/instant_policy:\$PYTHONPATH
      $cmd
    "
}

SHAPENET_CONTAINER_PATH="$(to_container_path "$SHAPENET_PATH")"
INDEX_CONTAINER_PATH="$(to_container_path "$INDEX_PATH")"
SCENE_ENCODER_CONTAINER_PATH="$(to_container_path "$SCENE_ENCODER_PATH")"
MESH_CONTAINER_PATH="$(to_container_path "$MESH_PATH")"
RUNS_ROOT_CONTAINER_PATH="$(to_container_path "$RUNS_ROOT")"
HOT_TRAIN_CONTAINER_PATH="$(to_container_path "$HOT_TRAIN_DIR")"

HOST_RUN_DIR="$RUNS_ROOT/$RUN_NAME"
mkdir -p "$HOST_RUN_DIR"
HOST_WANDB_ID_FILE="$HOST_RUN_DIR/wandb_id.txt"
if [ -s "$HOST_WANDB_ID_FILE" ]; then
  WANDB_ID="$(tr -d ' \n\r\t' < "$HOST_WANDB_ID_FILE")"
else
  WANDB_ID="$(tr '[:upper:]' '[:lower:]' < /proc/sys/kernel/random/uuid | tr -d '-' | cut -c1-8)"
  printf "%s\n" "$WANDB_ID" > "$HOST_WANDB_ID_FILE"
fi

count_data_files() {
  find "$HOT_TRAIN_DIR" -maxdepth 1 -type f -name 'data_*.pt' | wc -l | tr -d ' '
}

echo "Node: $(hostname)"
echo "run_name=$RUN_NAME"
echo "batch_size=$BATCH_SIZE num_workers=$TRAIN_NUM_WORKERS"
echo "probe_steps=$PROBE_STEPS"
echo "prefill_min_items=$PREFILL_MIN_ITEMS prefill_chunk_tasks=$PREFILL_CHUNK_TASKS"
echo "hot_data_dir=$HOT_DATA_DIR"

TRAIN_COUNT="$(count_data_files)"
TASK_START="$GEN_TASK_START"
while [ "$TRAIN_COUNT" -lt "$PREFILL_MIN_ITEMS" ]; do
  echo "Prefill chunk: have=$TRAIN_COUNT target=$PREFILL_MIN_ITEMS task_start=$TASK_START num_tasks=$PREFILL_CHUNK_TASKS"
  container_exec "
    export PYOPENGL_PLATFORM=egl
    python -m ip.scripts.generate_pseudo_demos \
      --shapenet_path \"$SHAPENET_CONTAINER_PATH\" \
      --shapenet_index_path \"$INDEX_CONTAINER_PATH\" \
      --save_dir \"$HOT_TRAIN_CONTAINER_PATH\" \
      --num_tasks $PREFILL_CHUNK_TASKS \
      --task_start $TASK_START \
      --append \
      --seed $TRAIN_GEN_SEED \
      --storage_format steps \
      --pcd_storage_dtype $PCD_DTYPE \
      --num_demos_per_task $NUM_DEMOS_PER_TASK $NUM_DEMOS_PER_TASK \
      --gripper_mesh_path \"$MESH_CONTAINER_PATH\"
  "
  TASK_START=$((TASK_START + PREFILL_CHUNK_TASKS))
  TRAIN_COUNT="$(count_data_files)"
done

echo "Prefill complete: train_items=$TRAIN_COUNT"
echo "Starting trainer-only probe..."
container_exec "
  python /workspace/instant_policy/ip/train_h100.py \
    --run_name \"$RUN_NAME\" \
    --train_dir \"$HOT_TRAIN_CONTAINER_PATH\" \
    --save_root \"$RUNS_ROOT_CONTAINER_PATH\" \
    --scene_encoder_path \"$SCENE_ENCODER_CONTAINER_PATH\" \
    --batch_size $BATCH_SIZE \
    --num_workers $TRAIN_NUM_WORKERS \
    --save_every $SAVE_EVERY \
    --wandb_id \"$WANDB_ID\" \
    --num_iters_override $PROBE_STEPS
"
