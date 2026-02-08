#!/bin/bash
#SBATCH -A kdesingh
#SBATCH -p msigpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH -t 24:00:00
#SBATCH --array=0-7%4
#SBATCH --job-name=ip_pseudo_buf
#SBATCH --output=/scratch.global/%u/ips/logs/ip_pseudo_buf_%A_%a.out
#SBATCH --error=/scratch.global/%u/ips/logs/ip_pseudo_buf_%A_%a.err

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
SHAPENET_REL=${SHAPENET_REL:-ShapeNetCore.v2}
SHAPENET_PATH=${SHAPENET_PATH:-/workspace/data/$SHAPENET_REL}
MESH_PATH=${MESH_PATH:-/workspace/data/assets/robotiq_2f85_collision_open.obj}
SAVE_DIR=${SAVE_DIR:-/workspace/data/pseudo_ring/task_buffer}
INDEX_PATH=${INDEX_PATH:-/workspace/data/pseudo_ring/shapenet_index.json}

NUM_SHARDS=${NUM_SHARDS:-8}
SHARD_ID=${SHARD_ID:-${SLURM_ARRAY_TASK_ID:-0}}
BUFFER_SIZE=${BUFFER_SIZE:-250000}
NUM_TASKS=${NUM_TASKS:-999999}

BASE_SEED=${BASE_SEED:-1337}
SEED_STRIDE=${SEED_STRIDE:-10000000}
SHARD_SEED=$((BASE_SEED + SHARD_ID * SEED_STRIDE))

FILL_BUFFER=${FILL_BUFFER:-1}
APPEND=${APPEND:-0}
RANDOMIZE_NUM_DEMOS=${RANDOMIZE_NUM_DEMOS:-0}
PCD_DTYPE=${PCD_DTYPE:-float32}

if [ ! -x "$RUNNER" ]; then
  echo "ERROR: runner not found or not executable: $RUNNER"
  exit 1
fi

# Validate host availability for /workspace/data/* defaults.
if [[ "$SHAPENET_PATH" == /workspace/data/* ]]; then
  HOST_SHAPENET="$DATA_DIR/${SHAPENET_PATH#/workspace/data/}"
  if [ ! -d "$HOST_SHAPENET" ]; then
    echo "ERROR: ShapeNet path not found on host: $HOST_SHAPENET"
    exit 1
  fi
fi

mkdir -p "$DATA_DIR/logs"
mkdir -p "$DATA_DIR/assets"
mkdir -p "$DATA_DIR/pseudo_ring"

EXTRA_FLAGS=""
if [ "$RANDOMIZE_NUM_DEMOS" = "1" ]; then
  EXTRA_FLAGS="$EXTRA_FLAGS --randomize_num_demos"
fi
if [ "$APPEND" = "1" ]; then
  EXTRA_FLAGS="$EXTRA_FLAGS --append"
fi
if [ "$FILL_BUFFER" = "1" ]; then
  EXTRA_FLAGS="$EXTRA_FLAGS --fill_buffer"
fi

start=$(date +%s)
echo "Starting pseudo buffer shard $SHARD_ID/$NUM_SHARDS on $(hostname)"
echo "seed=$SHARD_SEED buffer_size=$BUFFER_SIZE num_tasks=$NUM_TASKS fill_buffer=$FILL_BUFFER append=$APPEND"

"$RUNNER" "set -euo pipefail
export PYOPENGL_PLATFORM=\${PYOPENGL_PLATFORM:-egl}
mkdir -p \$(dirname \"$MESH_PATH\") \"$SAVE_DIR\"
if [ ! -f \"$MESH_PATH\" ]; then
  python -m ip.scripts.build_robotiq_mesh --out \"$MESH_PATH\" --source collision --state open
fi
python -m ip.scripts.generate_pseudo_demos \
  --shapenet_path \"$SHAPENET_PATH\" \
  --shapenet_index_path \"$INDEX_PATH\" \
  --save_dir \"$SAVE_DIR\" \
  --num_tasks $NUM_TASKS \
  --seed $SHARD_SEED \
  --storage_format trajectory \
  --pcd_storage_dtype $PCD_DTYPE \
  --buffer_size $BUFFER_SIZE \
  --num_shards $NUM_SHARDS \
  --shard_id $SHARD_ID \
  --gripper_mesh_path \"$MESH_PATH\" \
  $EXTRA_FLAGS"

end=$(date +%s)
echo "Finished pseudo buffer shard $SHARD_ID in $((end - start)) seconds"
