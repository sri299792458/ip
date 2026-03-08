#!/bin/bash
# Minimal single-H100 streaming job:
# - ShapeNet/checkpoints/runs stay on shared scratch
# - validation is copied once to local scratch
# - training samples stream into a bounded local ring buffer

#SBATCH -A kdesingh
#SBATCH -p msigpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=220G
#SBATCH --tmp=300G
#SBATCH -t 24:00:00
#SBATCH --job-name=ip_h100_stream
#SBATCH --output=ip_h100_stream_%j.out
#SBATCH --error=ip_h100_stream_%j.err

set -euo pipefail

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
VAL_SHARED_DIR="${VAL_SHARED_DIR:-$STATIC_DATA_DIR/pseudo/val}"
RUNS_ROOT="${RUNS_ROOT:-$STATIC_DATA_DIR/runs_policy}"
LOG_ROOT="${LOG_ROOT:-$STATIC_DATA_DIR/logs}"

RUN_NAME="${RUN_NAME:-ip_h100}"
BATCH_SIZE="${BATCH_SIZE:-32}"
TRAIN_NUM_WORKERS="${TRAIN_NUM_WORKERS:-16}"
TRAIN_BUFFER_SIZE="${TRAIN_BUFFER_SIZE:-$((256 * BATCH_SIZE))}"
SAVE_EVERY="${SAVE_EVERY:-50000}"
NUM_ITERS_OVERRIDE="${NUM_ITERS_OVERRIDE:-}"
SKIP_VAL="${SKIP_VAL:-0}"

NUM_DEMOS_PER_TASK=3
GEN_CHUNK_TASKS=128
PCD_DTYPE=float16
TRAIN_GEN_SEED=1337
GEN_TASK_START=1000000
START_TIMEOUT_SEC=7200
MIN_START_BATCHES=16
MIN_START_ITEMS_FLOOR=256
TRAIN_START_MIN_ITEMS=$((MIN_START_BATCHES * BATCH_SIZE))
if [ "$TRAIN_START_MIN_ITEMS" -lt "$MIN_START_ITEMS_FLOOR" ]; then
  TRAIN_START_MIN_ITEMS="$MIN_START_ITEMS_FLOOR"
fi
if [ "$TRAIN_BUFFER_SIZE" -lt "$TRAIN_START_MIN_ITEMS" ]; then
  echo "ERROR: TRAIN_BUFFER_SIZE=$TRAIN_BUFFER_SIZE must be >= derived start threshold $TRAIN_START_MIN_ITEMS"
  exit 1
fi

if [ ! -d /scratch.local ]; then
  echo "ERROR: /scratch.local is unavailable; this job expects MSI local scratch."
  exit 1
fi

resolve_existing_dir() {
  local path="$1"
  local label="$2"
  if [ ! -d "$path" ]; then
    echo "ERROR: required directory not found for $label: $path"
    exit 1
  fi
  (cd "$path" && pwd -P)
}

resolve_existing_file() {
  local path="$1"
  local label="$2"
  if [ ! -f "$path" ]; then
    echo "ERROR: required file not found for $label: $path"
    exit 1
  fi
  local dir
  dir="$(dirname "$path")"
  if [ ! -d "$dir" ]; then
    echo "ERROR: parent directory not found for $label: $dir"
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
    echo "ERROR: parent directory not found for $label: $dir"
    exit 1
  fi
  dir="$(cd "$dir" && pwd -P)"
  printf "%s/%s\n" "$dir" "$(basename "$path")"
}

PROJECT_DIR="$(resolve_existing_dir "$PROJECT_DIR" "PROJECT_DIR")"
INSTANT_POLICY_DIR="$(resolve_existing_dir "$INSTANT_POLICY_DIR" "INSTANT_POLICY_DIR")"
mkdir -p "$STATIC_DATA_DIR"
STATIC_DATA_DIR="$(resolve_existing_dir "$STATIC_DATA_DIR" "STATIC_DATA_DIR")"
if [ "$SKIP_VAL" != "1" ]; then
  VAL_SHARED_DIR="$(resolve_existing_dir "$VAL_SHARED_DIR" "VAL_SHARED_DIR")"
fi
mkdir -p "$RUNS_ROOT"
RUNS_ROOT="$(resolve_existing_dir "$RUNS_ROOT" "RUNS_ROOT")"
mkdir -p "$LOG_ROOT"
LOG_ROOT="$(resolve_existing_dir "$LOG_ROOT" "LOG_ROOT")"
SHAPENET_PATH="$(resolve_existing_dir "$SHAPENET_PATH" "SHAPENET_PATH")"
SCENE_ENCODER_PATH="$(resolve_existing_file "$SCENE_ENCODER_PATH" "SCENE_ENCODER_PATH")"
MESH_PATH="$(resolve_existing_file "$MESH_PATH" "MESH_PATH")"
INDEX_PATH="$(resolve_path_with_existing_parent "$INDEX_PATH" "INDEX_PATH")"
CONTAINER_IMAGE="$(resolve_existing_file "$CONTAINER_IMAGE" "CONTAINER_IMAGE")"

required_dirs=("$INSTANT_POLICY_DIR" "$STATIC_DATA_DIR" "$SHAPENET_PATH")
if [ "$SKIP_VAL" != "1" ]; then
  required_dirs+=("$VAL_SHARED_DIR")
fi
for path in "${required_dirs[@]}"; do
  if [ ! -d "$path" ]; then
    echo "ERROR: required directory not found: $path"
    exit 1
  fi
done
for path in "$CONTAINER_IMAGE" "$SCENE_ENCODER_PATH" "$MESH_PATH"; do
  if [ ! -f "$path" ]; then
    echo "ERROR: required file not found: $path"
    exit 1
  fi
done
if [ "$SKIP_VAL" != "1" ]; then
  if ! find "$VAL_SHARED_DIR" -maxdepth 1 -type f -name 'data_*.pt' | grep -q .; then
    echo "ERROR: validation directory is empty: $VAL_SHARED_DIR"
    exit 1
  fi
fi

HOT_DATA_DIR="/scratch.local/$USER/ip_${RUN_NAME}_${SLURM_JOB_ID:-manual}"
HOT_TRAIN_DIR="$HOT_DATA_DIR/train_ring"
HOT_VAL_DIR="$HOT_DATA_DIR/val"
mkdir -p "$HOT_TRAIN_DIR" "$HOT_VAL_DIR"

GEN_PID=""
cleanup_all() {
  if [ -n "$GEN_PID" ] && kill -0 "$GEN_PID" 2>/dev/null; then
    kill "$GEN_PID" 2>/dev/null || true
    wait "$GEN_PID" 2>/dev/null || true
  fi
  rm -rf "$HOT_DATA_DIR" 2>/dev/null || true
}
trap cleanup_all EXIT INT TERM

if [ "$SKIP_VAL" != "1" ]; then
  if command -v rsync >/dev/null 2>&1; then
    rsync -a --delete "$VAL_SHARED_DIR"/ "$HOT_VAL_DIR"/
  else
    find "$HOT_VAL_DIR" -mindepth 1 -maxdepth 1 -exec rm -rf {} +
    cp -a "$VAL_SHARED_DIR"/. "$HOT_VAL_DIR"/
  fi
fi
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
  apptainer exec --nv --cleanenv --no-home \
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
HOT_VAL_CONTAINER_PATH=""
if [ "$SKIP_VAL" != "1" ]; then
  HOT_VAL_CONTAINER_PATH="$(to_container_path "$HOT_VAL_DIR")"
fi

HOST_RUN_DIR="$RUNS_ROOT/$RUN_NAME"
mkdir -p "$HOST_RUN_DIR"
WANDB_ID=""
HOST_WANDB_ID_FILE="$HOST_RUN_DIR/wandb_id.txt"
if [ -s "$HOST_WANDB_ID_FILE" ]; then
  WANDB_ID="$(tr -d ' \n\r\t' < "$HOST_WANDB_ID_FILE")"
else
  WANDB_ID="$(tr -dc 'a-z0-9' </dev/urandom | head -c 8)"
  printf "%s\n" "$WANDB_ID" > "$HOST_WANDB_ID_FILE"
fi
NUM_ITERS_OVERRIDE_ARG=""
if [ -n "$NUM_ITERS_OVERRIDE" ]; then
  NUM_ITERS_OVERRIDE_ARG="--num_iters_override $NUM_ITERS_OVERRIDE"
fi
VAL_ARG=""
if [ "$SKIP_VAL" != "1" ]; then
  VAL_ARG="--val_dir \"$HOT_VAL_CONTAINER_PATH\""
fi

GEN_SCRIPT="$HOT_DATA_DIR/generate_stream.sh"
cat > "$GEN_SCRIPT" <<EOF
#!/bin/bash
set -euo pipefail
export PYOPENGL_PLATFORM=egl
TASK_START=$GEN_TASK_START
while true; do
  python -m ip.scripts.generate_pseudo_demos \\
    --shapenet_path "$SHAPENET_CONTAINER_PATH" \\
    --shapenet_index_path "$INDEX_CONTAINER_PATH" \\
    --save_dir "$HOT_TRAIN_CONTAINER_PATH" \\
    --num_tasks $GEN_CHUNK_TASKS \\
    --task_start \$TASK_START \\
    --append \\
    --seed $TRAIN_GEN_SEED \\
    --storage_format steps \\
    --pcd_storage_dtype $PCD_DTYPE \\
    --buffer_size $TRAIN_BUFFER_SIZE \\
    --num_demos_per_task $NUM_DEMOS_PER_TASK $NUM_DEMOS_PER_TASK \\
    --gripper_mesh_path "$MESH_CONTAINER_PATH"
  TASK_START=\$((TASK_START + $GEN_CHUNK_TASKS))
done
EOF
chmod +x "$GEN_SCRIPT"

TRAIN_SCRIPT="$HOT_DATA_DIR/train_stream.sh"
cat > "$TRAIN_SCRIPT" <<EOF
#!/bin/bash
set -euo pipefail
python /workspace/instant_policy/ip/train_h100.py \\
  --run_name "$RUN_NAME" \\
  --train_dir "$HOT_TRAIN_CONTAINER_PATH" \\
  --save_root "$RUNS_ROOT_CONTAINER_PATH" \\
  --scene_encoder_path "$SCENE_ENCODER_CONTAINER_PATH" \\
  --batch_size $BATCH_SIZE \\
  --num_workers $TRAIN_NUM_WORKERS \\
  --save_every $SAVE_EVERY \\
  --wandb_id "$WANDB_ID" \\
  $VAL_ARG \\
  $NUM_ITERS_OVERRIDE_ARG
EOF
chmod +x "$TRAIN_SCRIPT"

echo "Node: $(hostname)"
echo "run_name=$RUN_NAME"
echo "batch_size=$BATCH_SIZE num_workers=$TRAIN_NUM_WORKERS"
echo "buffer_size=$TRAIN_BUFFER_SIZE start_min_items=$TRAIN_START_MIN_ITEMS"
echo "save_every=$SAVE_EVERY"
echo "hot_data_dir=$HOT_DATA_DIR"

container_exec "bash /workspace/hot/$(basename "$GEN_SCRIPT")" > "$LOG_ROOT/ip_gen_${RUN_NAME}_${SLURM_JOB_ID:-manual}.log" 2>&1 &
GEN_PID="$!"

count_data_files() {
  find "$HOT_TRAIN_DIR" -maxdepth 1 -type f -name 'data_*.pt' | wc -l | tr -d ' '
}

waited=0
while true; do
  TRAIN_COUNT="$(count_data_files)"
  if [ "$TRAIN_COUNT" -ge "$TRAIN_START_MIN_ITEMS" ]; then
    echo "Train data ready: train_items=$TRAIN_COUNT target=$TRAIN_START_MIN_ITEMS"
    break
  fi
  if [ "$waited" -ge "$START_TIMEOUT_SEC" ]; then
    echo "ERROR: timed out waiting for train data (have=$TRAIN_COUNT target=$TRAIN_START_MIN_ITEMS)"
    exit 1
  fi
  echo "Waiting for train data: have=$TRAIN_COUNT target=$TRAIN_START_MIN_ITEMS waited=${waited}s"
  sleep 20
  waited=$((waited + 20))
done

container_exec "bash /workspace/hot/$(basename "$TRAIN_SCRIPT")"
