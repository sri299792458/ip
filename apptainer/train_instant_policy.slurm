#!/bin/bash
# Single entrypoint for MSI: bootstrap pseudo data (if needed) + train/resume.
# Re-submit this same script for each 24h window.
#SBATCH -A kdesingh
#SBATCH -p msigpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
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

# Training settings.
DATA_FORMAT=${DATA_FORMAT:-trajectory}
BATCH_SIZE=${BATCH_SIZE:-16}
NUM_POINTS=${NUM_POINTS:-2048}
SUBSAMPLE_LIVE=${SUBSAMPLE_LIVE:-0}
LIVE_SPACING_TRANS=${LIVE_SPACING_TRANS:-0.01}
LIVE_SPACING_ROT=${LIVE_SPACING_ROT:-3.0}

RECORD=${RECORD:-1}
FINE_TUNE=${FINE_TUNE:-0}
COMPILE_MODELS=${COMPILE_MODELS:-0}
USE_WANDB=${USE_WANDB:-1}
AUTO_RESUME=${AUTO_RESUME:-0}
RESUME_CKPT=${RESUME_CKPT:-}
WANDB_ID=${WANDB_ID:-}
WANDB_RESUME=${WANDB_RESUME:-allow}

EXTRA_TRAIN_ARGS=${EXTRA_TRAIN_ARGS:-}
NUM_ITERS_OVERRIDE=${NUM_ITERS_OVERRIDE:-}

# Pseudo-data bootstrap settings.
BOOTSTRAP_IF_EMPTY=${BOOTSTRAP_IF_EMPTY:-1}
BOOTSTRAP_FILL_BUFFER=${BOOTSTRAP_FILL_BUFFER:-1}
TRAIN_BUFFER_SIZE=${TRAIN_BUFFER_SIZE:-8192}
# Default bootstrap budget is bounded. With --fill_buffer enabled (default),
# generation stops once the ring is filled.
TRAIN_NUM_TASKS=${TRAIN_NUM_TASKS:-$TRAIN_BUFFER_SIZE}
TRAIN_GEN_SEED=${TRAIN_GEN_SEED:-1337}
DEMOS_PER_TASK_MIN=${DEMOS_PER_TASK_MIN:-3}
DEMOS_PER_TASK_MAX=${DEMOS_PER_TASK_MAX:-3}
PCD_DTYPE=${PCD_DTYPE:-float16}

# Optional per-job refresh before training.
REFRESH_BEFORE_TRAIN=${REFRESH_BEFORE_TRAIN:-0}
REFRESH_NUM_TASKS=${REFRESH_NUM_TASKS:-0}
REFRESH_SEED=${REFRESH_SEED:-223344}

# Validation-set bootstrap settings.
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

# Resolve ShapeNet path robustly to avoid folder-name mismatches.
if [ -z "$SHAPENET_PATH" ]; then
  if [ -d "$DATA_DIR/shapenet" ]; then
    SHAPENET_PATH="/workspace/data/shapenet"
  elif [ -d "$DATA_DIR/ShapeNetCore.v2" ]; then
    SHAPENET_PATH="/workspace/data/ShapeNetCore.v2"
  else
    # Keep a sane fallback for error messaging below.
    SHAPENET_PATH="/workspace/data/shapenet"
  fi
fi

count_task_files() {
  local d="$1"
  if [ ! -d "$d" ]; then
    echo 0
    return
  fi
  find "$d" -maxdepth 1 -type f -name 'task_*.pt' | wc -l | tr -d ' '
}

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

start=$(date +%s)
echo "Starting unified pipeline on $(hostname)"
echo "run_name=$RUN_NAME format=$DATA_FORMAT batch_size=$BATCH_SIZE"
echo "shapenet_path=$SHAPENET_PATH (host: $HOST_SHAPENET)"
echo "auto_resume=$AUTO_RESUME"
if [ -n "$NUM_ITERS_OVERRIDE" ]; then
  echo "num_iters_override=$NUM_ITERS_OVERRIDE"
fi
if [ -n "$RESUME_CKPT" ]; then
  echo "resume_ckpt=$RESUME_CKPT"
fi
echo "use_wandb=$USE_WANDB"
echo "train_buffer_size=$TRAIN_BUFFER_SIZE train_num_tasks=$TRAIN_NUM_TASKS fill_buffer=$BOOTSTRAP_FILL_BUFFER"

BOOTSTRAP_NUM_TASKS="$TRAIN_NUM_TASKS"
if [ "$BOOTSTRAP_FILL_BUFFER" = "1" ] && [ "$BOOTSTRAP_NUM_TASKS" -lt "$TRAIN_BUFFER_SIZE" ]; then
  echo "Adjusting TRAIN_NUM_TASKS to TRAIN_BUFFER_SIZE for fill_buffer bootstrap."
  BOOTSTRAP_NUM_TASKS="$TRAIN_BUFFER_SIZE"
fi

# Ensure gripper mesh exists.
if [ ! -f "$HOST_MESH_PATH" ]; then
  echo "Mesh not found, building: $HOST_MESH_PATH"
  "$RUNNER" "set -euo pipefail
python -m ip.scripts.build_robotiq_mesh --out \"$MESH_PATH\" --source collision --state open"
fi

# Build/rebuild validation set if needed.
VAL_COUNT="$(count_task_files "$HOST_VAL_DIR")"
if [ "$VAL_FORCE_REBUILD" = "1" ]; then
  echo "VAL_FORCE_REBUILD=1, clearing validation tasks in $HOST_VAL_DIR"
  find "$HOST_VAL_DIR" -maxdepth 1 -type f -name 'task_*.pt' -delete
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
  --storage_format trajectory \
  --pcd_storage_dtype $PCD_DTYPE \
  --num_demos_per_task $DEMOS_PER_TASK_MIN $DEMOS_PER_TASK_MAX \
  --gripper_mesh_path \"$MESH_PATH\""
fi

# Bootstrap train ring buffer if empty, or top-up if partially filled.
TRAIN_COUNT="$(count_task_files "$HOST_TRAIN_DIR")"
BOOT_MODE=""
if [ "$BOOTSTRAP_IF_EMPTY" = "1" ]; then
  if [ "$TRAIN_COUNT" -eq 0 ]; then
    BOOT_MODE="empty"
  elif [ "$BOOTSTRAP_FILL_BUFFER" = "1" ] && [ "$TRAIN_COUNT" -lt "$TRAIN_BUFFER_SIZE" ]; then
    BOOT_MODE="topup"
  fi
fi
if [ -n "$BOOT_MODE" ]; then
  EXTRA_BOOT_FLAGS=""
  APPEND_BOOT_FLAG=""
  if [ "$BOOT_MODE" = "empty" ]; then
    echo "Training buffer empty, bootstrapping..."
  else
    echo "Training buffer partially filled ($TRAIN_COUNT/$TRAIN_BUFFER_SIZE), topping up..."
    APPEND_BOOT_FLAG="--append"
  fi
  if [ "$BOOTSTRAP_FILL_BUFFER" = "1" ]; then
    EXTRA_BOOT_FLAGS="$EXTRA_BOOT_FLAGS --fill_buffer"
  fi
  "$RUNNER" "set -euo pipefail
export PYOPENGL_PLATFORM=\${PYOPENGL_PLATFORM:-egl}
python -m ip.scripts.generate_pseudo_demos \
  --shapenet_path \"$SHAPENET_PATH\" \
  --shapenet_index_path \"$INDEX_PATH\" \
  --save_dir \"$TRAIN_DATA_DIR\" \
  --num_tasks $BOOTSTRAP_NUM_TASKS \
  --seed $TRAIN_GEN_SEED \
  --storage_format trajectory \
  --pcd_storage_dtype $PCD_DTYPE \
  --buffer_size $TRAIN_BUFFER_SIZE \
  --num_shards 1 \
  --shard_id 0 \
  --num_demos_per_task $DEMOS_PER_TASK_MIN $DEMOS_PER_TASK_MAX \
  --gripper_mesh_path \"$MESH_PATH\" \
  $APPEND_BOOT_FLAG \
  $EXTRA_BOOT_FLAGS"
fi

# Optional pre-train refresh pass.
if [ "$REFRESH_BEFORE_TRAIN" = "1" ] && [ "$REFRESH_NUM_TASKS" -gt 0 ]; then
  echo "Refreshing training ring buffer before training..."
  "$RUNNER" "set -euo pipefail
export PYOPENGL_PLATFORM=\${PYOPENGL_PLATFORM:-egl}
python -m ip.scripts.generate_pseudo_demos \
  --shapenet_path \"$SHAPENET_PATH\" \
  --shapenet_index_path \"$INDEX_PATH\" \
  --save_dir \"$TRAIN_DATA_DIR\" \
  --num_tasks $REFRESH_NUM_TASKS \
  --seed $REFRESH_SEED \
  --append \
  --storage_format trajectory \
  --pcd_storage_dtype $PCD_DTYPE \
  --buffer_size $TRAIN_BUFFER_SIZE \
  --num_shards 1 \
  --shard_id 0 \
  --num_demos_per_task $DEMOS_PER_TASK_MIN $DEMOS_PER_TASK_MAX \
  --gripper_mesh_path \"$MESH_PATH\""
fi

VAL_COUNT="$(count_task_files "$HOST_VAL_DIR")"
TRAIN_COUNT="$(count_task_files "$HOST_TRAIN_DIR")"
echo "val_tasks=$VAL_COUNT train_tasks=$TRAIN_COUNT"
if [ "$VAL_COUNT" -eq 0 ]; then
  echo "ERROR: validation set is empty at $HOST_VAL_DIR"
  exit 1
fi
if [ "$TRAIN_COUNT" -eq 0 ]; then
  echo "ERROR: training ring buffer is empty at $HOST_TRAIN_DIR"
  exit 1
fi

SUBSAMPLE_ARGS=""
if [ "$SUBSAMPLE_LIVE" = "1" ]; then
  SUBSAMPLE_ARGS="--subsample_live --live_spacing_trans $LIVE_SPACING_TRANS --live_spacing_rot $LIVE_SPACING_ROT"
fi

ITER_OVERRIDE_ARGS=""
if [ -n "$NUM_ITERS_OVERRIDE" ]; then
  ITER_OVERRIDE_ARGS="--num_iters_override $NUM_ITERS_OVERRIDE"
fi

RESUME_ARGS=""
if [ -n "$RESUME_CKPT" ]; then
  RESUME_ARGS="--resume_ckpt_path \"$RESUME_CKPT\""
elif [ "$AUTO_RESUME" = "1" ]; then
  RESUME_ARGS="--auto_resume"
fi

WANDB_ARGS=""
if [ "$USE_WANDB" = "1" ]; then
  WANDB_ARGS="--wandb_resume $WANDB_RESUME"
  if [ -n "$WANDB_ID" ]; then
    WANDB_ARGS="$WANDB_ARGS --wandb_id \"$WANDB_ID\""
  fi
fi

"$RUNNER" "set -euo pipefail
mkdir -p \"$SAVE_ROOT\"
if [ \"$USE_WANDB\" = \"1\" ]; then
  export WANDB_MODE=\${WANDB_MODE:-online}
fi
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
  --data_format \"$DATA_FORMAT\" \
  --batch_size $BATCH_SIZE \
  --num_points $NUM_POINTS \
  --save_path \"$SAVE_ROOT\" \
  $RESUME_ARGS \
  $WANDB_ARGS \
  $ITER_OVERRIDE_ARGS \
  $SUBSAMPLE_ARGS \
  $EXTRA_TRAIN_ARGS"

end=$(date +%s)
echo "Finished unified pipeline in $((end - start)) seconds"
