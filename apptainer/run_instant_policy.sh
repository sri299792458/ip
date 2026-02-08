#!/bin/bash
#
# Run Instant Policy container without VNC (Xvfb only).
#
# Usage: ./run_instant_policy.sh [command]
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export RLBENCH_ENABLE_VNC=0

exec "$SCRIPT_DIR/run_instant_policy_vnc.sh" "$@"
