#!/bin/bash
#
# Build script for Instant Policy Apptainer container
# Designed for UMN MSI cluster (agate)
#
# Usage:
#   ./build_instant_policy.sh
#
# Build time: ~20-30 minutes
# Output: instant_policy.sif (~8-10 GB)
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd -P)"
DEF_FILE="$SCRIPT_DIR/instant_policy.def"

echo "================================"
echo "Instant Policy Container Builder"
echo "================================"
echo ""

# Load CUDA module (required for GPU support during build)
echo "Loading CUDA 11.8 module..."
module load cuda/11.8.0-gcc-7.2.0-xqzqlf2

# Set up Apptainer cache directories
# Use scratch.global to avoid filling up home directory quota
if [ -d "/scratch.global/$USER" ]; then
    export APPTAINER_CACHEDIR="/scratch.global/$USER/.apptainer_cache"
    export APPTAINER_TMPDIR="/scratch.global/$USER/.apptainer_tmp"
    echo "Using scratch.global for cache: $APPTAINER_CACHEDIR"
else
    export APPTAINER_CACHEDIR="$HOME/.apptainer_cache"
    export APPTAINER_TMPDIR="$HOME/.apptainer_tmp"
    echo "Using home directory for cache: $APPTAINER_CACHEDIR"
fi

# Create cache directories
mkdir -p "$APPTAINER_CACHEDIR"
mkdir -p "$APPTAINER_TMPDIR"

# Output location in scratch (override with OUTPUT_DIR if needed)
OUTPUT_DIR="${OUTPUT_DIR:-/scratch.global/$USER/ips}"
if [[ "$OUTPUT_DIR" != /* ]]; then
    OUTPUT_DIR="$REPO_ROOT/$OUTPUT_DIR"
fi
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd -P)"

# Check if definition file exists
if [ ! -f "$DEF_FILE" ]; then
    echo "ERROR: instant_policy.def not found in $SCRIPT_DIR"
    exit 1
fi

echo ""
echo "Build configuration:"
echo "  Repo root: $REPO_ROOT"
echo "  Definition file: $DEF_FILE"
echo "  Output file: $OUTPUT_DIR/instant_policy.sif"
echo "  Cache directory: $APPTAINER_CACHEDIR"
echo "  Temp directory: $APPTAINER_TMPDIR"
echo ""
echo "Starting build..."
echo "This will take approximately 20-30 minutes."
echo ""

# Build the container with --fakeroot (no root privileges needed)
# --force overwrites existing .sif file
apptainer build --fakeroot --force "$OUTPUT_DIR/instant_policy.sif" "$DEF_FILE"

# Create symlink in current directory for convenience
ln -sf "$OUTPUT_DIR/instant_policy.sif" "$SCRIPT_DIR/instant_policy.sif"

echo ""
echo "================================"
echo "Build completed successfully!"
echo "================================"
echo ""
echo "Container image: $OUTPUT_DIR/instant_policy.sif"
echo "Symlink created: $SCRIPT_DIR/instant_policy.sif -> $OUTPUT_DIR/instant_policy.sif"
echo "Size: $(du -h "$OUTPUT_DIR/instant_policy.sif" | cut -f1)"
echo ""
echo "Next steps:"
echo "  1. Review run_instant_policy_vnc.sh and adjust paths if needed"
echo "  2. Submit a job using the run script"
echo "  3. Connect via VNC to visualize simulations"
echo ""
echo "For detailed usage instructions, see README_instant_policy.md"
