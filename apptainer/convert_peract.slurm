#!/bin/bash
#SBATCH -A kdesingh
#SBATCH -p msigpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH -t 00:30:00
#SBATCH --array=0-17%4
#SBATCH --job-name=ip_peract_conv
#SBATCH --output=/scratch.global/%u/ips/logs/ip_peract_conv_%A_%a.out
#SBATCH --error=/scratch.global/%u/ips/logs/ip_peract_conv_%A_%a.err

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

REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
INSTANT_POLICY_DIR=${INSTANT_POLICY_DIR:-$REPO_ROOT/instant_policy}
PROJECT_DIR=${PROJECT_DIR:-$REPO_ROOT}
DATA_DIR=${DATA_DIR:-/scratch.global/$USER/ips}
RUNNER=${RUNNER:-$SCRIPT_DIR/run_instant_policy_vnc.sh}

NUM_DEMOS=${NUM_DEMOS:-2}
NUM_WAYPOINTS=${NUM_WAYPOINTS:-10}
PRED_HORIZON=${PRED_HORIZON:-8}
NUM_POINTS=${NUM_POINTS:-2048}
MASK_THRESH=${MASK_THRESH:-60}
CAMERAS=${CAMERAS:-front,left_shoulder,right_shoulder}
NUM_EPISODES=${NUM_EPISODES:-}
APPEND=${APPEND:-0}

TASKS=(
  open_drawer
  slide_block_to_color_target
  sweep_to_dustpan_of_size
  meat_off_grill
  turn_tap
  put_item_in_drawer
  close_jar
  reach_and_drag
  stack_blocks
  light_bulb_in
  put_money_in_safe
  place_wine_at_rack_location
  put_groceries_in_cupboard
  place_shape_in_shape_sorter
  push_buttons
  insert_onto_square_peg
  stack_cups
  place_cups
)

if [ ! -x "$RUNNER" ]; then
  echo "ERROR: run_instant_policy_vnc.sh not found at $RUNNER"
  exit 1
fi

if [ ! -f "$DATA_DIR/checkpoints/scene_encoder.pt" ]; then
  echo "ERROR: scene_encoder.pt not found at $DATA_DIR/checkpoints/scene_encoder.pt"
  exit 1
fi

mkdir -p "$DATA_DIR/logs"

TASK=${TASKS[$SLURM_ARRAY_TASK_ID]}
ZIP_PATH="/workspace/data/rlbench/train/${TASK}.zip"
SAVE_DIR="/workspace/data/peract/train/${TASK}"

EPISODE_ARGS=()
if [ -n "$NUM_EPISODES" ]; then
  EPISODE_ARGS+=(--num_episodes "$NUM_EPISODES")
fi

APPEND_ARGS=()
if [ "$APPEND" = "1" ]; then
  APPEND_ARGS+=(--append)
fi

export PROJECT_DIR INSTANT_POLICY_DIR DATA_DIR

start=$(date +%s)
echo "Starting conversion for $TASK on $(hostname)"

"$RUNNER" "python /workspace/instant_policy/ip/scripts/convert_rlbench_peract.py \
  --zip_path $ZIP_PATH \
  --save_dir $SAVE_DIR \
  --num_demos $NUM_DEMOS \
  --num_waypoints_demo $NUM_WAYPOINTS \
  --pred_horizon $PRED_HORIZON \
  --num_points $NUM_POINTS \
  --mask_thresh $MASK_THRESH \
  --cameras $CAMERAS \
  --compute_embeddings \
  --scene_encoder_path /workspace/data/checkpoints/scene_encoder.pt \
  --device cuda \
  ${EPISODE_ARGS[@]} \
  ${APPEND_ARGS[@]}"

end=$(date +%s)
echo "Finished $TASK in $((end - start)) seconds"
