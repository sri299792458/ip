#!/bin/bash
#SBATCH -A kdesingh
#SBATCH -p msigpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH -t 06:00:00
#SBATCH --job-name=ip_pseudo_val
#SBATCH --output=/scratch.global/%u/ips/logs/ip_pseudo_val_%j.out
#SBATCH --error=/scratch.global/%u/ips/logs/ip_pseudo_val_%j.err

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
VAL_SAVE_DIR=${VAL_SAVE_DIR:-/workspace/data/pseudo_ring/val}
INDEX_PATH=${INDEX_PATH:-/workspace/data/pseudo_ring/shapenet_index.json}

VAL_NUM_TASKS=${VAL_NUM_TASKS:-200}
VAL_TASK_START=${VAL_TASK_START:-200000000}
VAL_SEED=${VAL_SEED:-424242}
PCD_DTYPE=${PCD_DTYPE:-float32}

if [ ! -x "$RUNNER" ]; then
  echo "ERROR: runner not found or not executable: $RUNNER"
  exit 1
fi

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

start=$(date +%s)
echo "Generating fixed validation pseudo set on $(hostname)"
echo "val_num_tasks=$VAL_NUM_TASKS task_start=$VAL_TASK_START seed=$VAL_SEED"

"$RUNNER" "set -euo pipefail
export PYOPENGL_PLATFORM=\${PYOPENGL_PLATFORM:-egl}
mkdir -p \$(dirname \"$MESH_PATH\") \"$VAL_SAVE_DIR\"
if [ ! -f \"$MESH_PATH\" ]; then
  python -m ip.scripts.build_robotiq_mesh --out \"$MESH_PATH\" --source collision --state open
fi
python -m ip.scripts.generate_pseudo_demos \
  --shapenet_path \"$SHAPENET_PATH\" \
  --shapenet_index_path \"$INDEX_PATH\" \
  --save_dir \"$VAL_SAVE_DIR\" \
  --num_tasks $VAL_NUM_TASKS \
  --task_start $VAL_TASK_START \
  --seed $VAL_SEED \
  --storage_format trajectory \
  --pcd_storage_dtype $PCD_DTYPE \
  --gripper_mesh_path \"$MESH_PATH\""

end=$(date +%s)
echo "Finished validation pseudo set in $((end - start)) seconds"
