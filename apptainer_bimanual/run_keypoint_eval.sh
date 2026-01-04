#!/bin/bash
# Run keypoint-based IP evaluation in the RLBench container
# Usage: ./run_keypoint_eval.sh [options]
# 
# This evaluates single-arm Instant Policy with different demo contexts:
# - real: Standard demos collected from RLBench
# - sparse: Only first/last waypoints from real demos, interpolated
# - keypoint: Fully synthesized from button position (simulating VLM output)
#
# Examples:
#   ./run_keypoint_eval.sh --demo_type real --num_rollouts 10
#   ./run_keypoint_eval.sh --demo_type keypoint --num_rollouts 10
#   ./run_keypoint_eval.sh --demo_type all --num_rollouts 5

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Default arguments
DEMO_TYPE="${DEMO_TYPE:-all}"
NUM_ROLLOUTS="${NUM_ROLLOUTS:-5}"
NUM_DEMOS="${NUM_DEMOS:-2}"
TASK_NAME="${TASK_NAME:-push_button}"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --demo_type) DEMO_TYPE="$2"; shift 2 ;;
        --num_rollouts) NUM_ROLLOUTS="$2"; shift 2 ;;
        --num_demos) NUM_DEMOS="$2"; shift 2 ;;
        --task_name) TASK_NAME="$2"; shift 2 ;;
        --headless) HEADLESS="--headless"; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "==================================="
echo "Keypoint-to-IP Evaluation"
echo "==================================="
echo "Task: $TASK_NAME"
echo "Demo type: $DEMO_TYPE"
echo "Num demos: $NUM_DEMOS"
echo "Num rollouts: $NUM_ROLLOUTS"
echo "==================================="

# Run via the RLBench container
"$SCRIPT_DIR/run_rlbench_vnc.sh" python -m external.ip.eval_keypoint_ip \
    --task_name "$TASK_NAME" \
    --demo_type "$DEMO_TYPE" \
    --num_demos "$NUM_DEMOS" \
    --num_rollouts "$NUM_ROLLOUTS" \
    $HEADLESS
