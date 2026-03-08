#!/bin/bash

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
LOG_ROOT="${LOG_ROOT:-$STATIC_DATA_DIR/logs}"

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
mkdir -p "$STATIC_DATA_DIR" "$RUNS_ROOT" "$LOG_ROOT"
STATIC_DATA_DIR="$(resolve_existing_dir "$STATIC_DATA_DIR" "STATIC_DATA_DIR")"
RUNS_ROOT="$(resolve_existing_dir "$RUNS_ROOT" "RUNS_ROOT")"
LOG_ROOT="$(resolve_existing_dir "$LOG_ROOT" "LOG_ROOT")"
SHAPENET_PATH="$(resolve_existing_dir "$SHAPENET_PATH" "SHAPENET_PATH")"
SCENE_ENCODER_PATH="$(resolve_existing_file "$SCENE_ENCODER_PATH" "SCENE_ENCODER_PATH")"
MESH_PATH="$(resolve_existing_file "$MESH_PATH" "MESH_PATH")"
INDEX_PATH="$(resolve_path_with_existing_parent "$INDEX_PATH" "INDEX_PATH")"
CONTAINER_IMAGE="$(resolve_existing_file "$CONTAINER_IMAGE" "CONTAINER_IMAGE")"
CONTAINER_RUNTIME="$(resolve_container_runtime)"

TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/ip_preflight.XXXXXX")"
cleanup() {
  rm -rf "$TMP_DIR" 2>/dev/null || true
}
trap cleanup EXIT

echo "Preflight resolved paths"
echo "  PROJECT_DIR=$PROJECT_DIR"
echo "  INSTANT_POLICY_DIR=$INSTANT_POLICY_DIR"
echo "  CONTAINER_IMAGE=$CONTAINER_IMAGE"
echo "  CONTAINER_RUNTIME=$CONTAINER_RUNTIME"
echo "  STATIC_DATA_DIR=$STATIC_DATA_DIR"
echo "  SHAPENET_PATH=$SHAPENET_PATH"
echo "  INDEX_PATH=$INDEX_PATH"
echo "  SCENE_ENCODER_PATH=$SCENE_ENCODER_PATH"
echo "  MESH_PATH=$MESH_PATH"
echo "  RUNS_ROOT=$RUNS_ROOT"
echo "  LOG_ROOT=$LOG_ROOT"

"$CONTAINER_RUNTIME" exec --cleanenv --no-home \
  --bind "$INSTANT_POLICY_DIR:/workspace/instant_policy" \
  --bind "$STATIC_DATA_DIR:/workspace/static" \
  --bind "$TMP_DIR:/workspace/hot" \
  --pwd /workspace/instant_policy/ip \
  "$CONTAINER_IMAGE" \
  bash -lc '
    export HOME=/workspace/hot
    export XDG_CACHE_HOME=/workspace/hot/.cache
    export HF_HOME=/workspace/hot/.cache/huggingface
    export MPLCONFIGDIR=/workspace/hot/.cache/matplotlib
    mkdir -p /workspace/hot/.cache/huggingface /workspace/hot/.cache/matplotlib
    export PYTHONPATH=/workspace/instant_policy:$PYTHONPATH
    python /workspace/instant_policy/ip/train_h100.py --help >/dev/null
    python -m ip.scripts.generate_pseudo_demos --help >/dev/null
    python - <<'"'"'PY'"'"'
import lightning
import torch
import torch_geometric
import ip
print("CONTAINER_IMPORT_OK")
PY
  '

echo "PREFLIGHT_OK"
