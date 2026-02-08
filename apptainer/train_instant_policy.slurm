#!/bin/bash
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

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
  if [ -d "$SLURM_SUBMIT_DIR/apptainer" ]; then
    SCRIPT_DIR="$SLURM_SUBMIT_DIR/apptainer"
  else
    SCRIPT_DIR="$SLURM_SUBMIT_DIR"
  fi
else
  SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
fi

RUNNER=${RUNNER:-$SCRIPT_DIR/run_instant_policy.sh}
DATA_DIR=${DATA_DIR:-/scratch.global/$USER/ips}

TRAIN_DATA_DIR=${TRAIN_DATA_DIR:-/workspace/data/pseudo_ring/task_buffer}
VAL_DATA_DIR=${VAL_DATA_DIR:-/workspace/data/pseudo_ring/val}
MODEL_PATH=${MODEL_PATH:-/workspace/data/checkpoints}
MODEL_NAME=${MODEL_NAME:-model.pt}
SAVE_ROOT=${SAVE_ROOT:-/workspace/data/runs_policy}
RUN_NAME=${RUN_NAME:-ring_train_${SLURM_JOB_ID:-manual}}

DATA_FORMAT=${DATA_FORMAT:-trajectory}
BATCH_SIZE=${BATCH_SIZE:-16}
NUM_POINTS=${NUM_POINTS:-2048}
SUBSAMPLE_LIVE=${SUBSAMPLE_LIVE:-0}
LIVE_SPACING_TRANS=${LIVE_SPACING_TRANS:-0.01}
LIVE_SPACING_ROT=${LIVE_SPACING_ROT:-3.0}

RECORD=${RECORD:-1}
FINE_TUNE=${FINE_TUNE:-0}
COMPILE_MODELS=${COMPILE_MODELS:-0}
USE_WANDB=${USE_WANDB:-0}

EXTRA_TRAIN_ARGS=${EXTRA_TRAIN_ARGS:-}

if [ ! -x "$RUNNER" ]; then
  echo "ERROR: runner not found or not executable: $RUNNER"
  exit 1
fi

if [[ "$TRAIN_DATA_DIR" == /workspace/data/* ]]; then
  HOST_TRAIN_DIR="$DATA_DIR/${TRAIN_DATA_DIR#/workspace/data/}"
  if [ ! -d "$HOST_TRAIN_DIR" ]; then
    echo "ERROR: training data path not found on host: $HOST_TRAIN_DIR"
    exit 1
  fi
fi

if [[ "$VAL_DATA_DIR" == /workspace/data/* ]]; then
  HOST_VAL_DIR="$DATA_DIR/${VAL_DATA_DIR#/workspace/data/}"
  if [ ! -d "$HOST_VAL_DIR" ]; then
    echo "ERROR: validation data path not found on host: $HOST_VAL_DIR"
    exit 1
  fi
fi

if [[ "$MODEL_PATH" == /workspace/data/* ]]; then
  HOST_MODEL_DIR="$DATA_DIR/${MODEL_PATH#/workspace/data/}"
  if [ ! -d "$HOST_MODEL_DIR" ]; then
    echo "ERROR: model path not found on host: $HOST_MODEL_DIR"
    exit 1
  fi
fi

mkdir -p "$DATA_DIR/logs"

start=$(date +%s)
echo "Starting training on $(hostname)"
echo "run_name=$RUN_NAME format=$DATA_FORMAT batch_size=$BATCH_SIZE"

SUBSAMPLE_ARGS=""
if [ "$SUBSAMPLE_LIVE" = "1" ]; then
  SUBSAMPLE_ARGS="--subsample_live --live_spacing_trans $LIVE_SPACING_TRANS --live_spacing_rot $LIVE_SPACING_ROT"
fi

"$RUNNER" "set -euo pipefail
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
  --data_format \"$DATA_FORMAT\" \
  --batch_size $BATCH_SIZE \
  --num_points $NUM_POINTS \
  --save_path \"$SAVE_ROOT\" \
  $SUBSAMPLE_ARGS \
  $EXTRA_TRAIN_ARGS"

end=$(date +%s)
echo "Finished training job in $((end - start)) seconds"
