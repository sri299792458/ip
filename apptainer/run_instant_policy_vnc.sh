#!/bin/bash
#
# Run Instant Policy container with VNC on MSI cluster
#
# Usage: ./run_instant_policy_vnc.sh [command]
# Example: ./run_instant_policy_vnc.sh python eval.py --task_name=reach_target --num_demos=2
#

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$HOME/ips}"
INSTANT_POLICY_DIR="${INSTANT_POLICY_DIR:-$PROJECT_DIR/instant_policy}"
DATA_DIR="${DATA_DIR:-/scratch.global/$USER/ips}"
CONTAINER_IMAGE="$SCRIPT_DIR/instant_policy.sif"
DISPLAY_NUM="${RLBENCH_DISPLAY:-1}"
VNC_PORT="${RLBENCH_VNC_PORT:-5900}"

# Validation
[ ! -f "$CONTAINER_IMAGE" ] && echo "ERROR: Container not found: $CONTAINER_IMAGE" && exit 1
[ ! -d "$INSTANT_POLICY_DIR" ] && echo "ERROR: instant_policy not found: $INSTANT_POLICY_DIR" && exit 1

mkdir -p "$DATA_DIR/checkpoints"

# ============================================
# Library bindings (Rocky Linux -> Ubuntu)
# ============================================
NVIDIA_LIB_DIR="/usr/lib64"
[ ! -d "$NVIDIA_LIB_DIR" ] && NVIDIA_LIB_DIR="/usr/lib/x86_64-linux-gnu"

BIND_LIBS=""
for lib in libGLX_nvidia.so libEGL_nvidia.so libnvidia-glcore.so libnvidia-tls.so \
           libnvidia-glsi.so libGLdispatch.so libOpenGL.so libGLX.so libEGL.so; do
    found=$(find $NVIDIA_LIB_DIR -name "${lib}*" 2>/dev/null | head -1)
    [ -n "$found" ] && BIND_LIBS="$BIND_LIBS --bind $found:/usr/lib/x86_64-linux-gnu/$(basename $found)"
done

[ -d "/usr/share/vulkan/icd.d" ] && BIND_LIBS="$BIND_LIBS --bind /usr/share/vulkan/icd.d:/usr/share/vulkan/icd.d"

# ============================================
# Cleanup stale processes
# ============================================
cleanup_stale_processes() {
    for pattern in "Xvfb :$DISPLAY_NUM" "x11vnc.*-rfbport $VNC_PORT" "fluxbox"; do
        pid=$(ps -u "$USER" -o pid= -o args= 2>/dev/null | awk "/$pattern/{print \$1; exit}")
        [ -n "$pid" ] && kill -9 "$pid" 2>/dev/null || true
    done
    rm -f "/tmp/.X${DISPLAY_NUM}-lock" "/tmp/.X11-unix/X${DISPLAY_NUM}"
}

echo "==================================="
echo "Instant Policy + VNC"
echo "Node: $(hostname)"
echo "VNC: ssh -L $VNC_PORT:$(hostname):$VNC_PORT $USER@agate.msi.umn.edu"
echo "==================================="

cleanup_stale_processes

CMD="${@:-bash}"

# Run container (isolate from host env/home for stability)
apptainer exec --nv --cleanenv --no-home --writable-tmpfs \
    $BIND_LIBS \
    --bind "$INSTANT_POLICY_DIR:/workspace/instant_policy" \
    --bind "$DATA_DIR:/workspace/data" \
    --pwd /workspace/instant_policy/ip \
    "$CONTAINER_IMAGE" \
    bash -c "
        # Redirect caches and HOME to writable location
        export HOME=/workspace/data
        export XDG_CACHE_HOME=/workspace/data/.cache
        export TRANSFORMERS_CACHE=/workspace/data/.cache/huggingface
        export HF_HOME=/workspace/data/.cache/huggingface
        export MPLCONFIGDIR=/workspace/data/.cache/matplotlib
        mkdir -p /workspace/data/.cache/huggingface /workspace/data/.cache/matplotlib /workspace/data/.fluxbox /workspace/data/.CoppeliaSim

        # Ensure Qt runtime dir exists (prevents XDG_RUNTIME_DIR warnings/crashes)
        if [ -z \"\${XDG_RUNTIME_DIR:-}\" ] || [ ! -d \"\$XDG_RUNTIME_DIR\" ]; then
            export XDG_RUNTIME_DIR=/tmp/runtime-\$(id -u)
            mkdir -p \"\$XDG_RUNTIME_DIR\"
            chmod 700 \"\$XDG_RUNTIME_DIR\"
        fi

        # Start Xvfb + VNC
        export DISPLAY=:$DISPLAY_NUM
        Xvfb :$DISPLAY_NUM -screen 0 1280x1024x24 &
        sleep 2
        x11vnc -display :$DISPLAY_NUM -forever -nopw -rfbport $VNC_PORT -noxdamage -nowf &
        sleep 1
        fluxbox &
        sleep 1

        echo 'VNC server running on port $VNC_PORT'
        nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'WARNING: No GPU'

        # CoppeliaSim setup
        COPPELIASIM_ROOT=/opt/CoppeliaSim
        if [ \"\${COPPELIASIM_USE_COPY:-0}\" = \"1\" ]; then
            if [ ! -d /workspace/data/.coppeliasim ]; then
                echo 'Copying CoppeliaSim...'
                cp -r /opt/CoppeliaSim /workspace/data/.coppeliasim
                rm -f /workspace/data/.coppeliasim/libsimExtCustomUI.so
                rm -f /workspace/data/.coppeliasim/libsimExtCodeEditor.so
            fi
            # Avoid libstdc++/libgcc conflicts with conda (common CoppeliaSim crash source)
            rm -f /workspace/data/.coppeliasim/libstdc++.so.6 /workspace/data/.coppeliasim/libgcc_s.so.1
            COPPELIASIM_ROOT=/workspace/data/.coppeliasim
        fi

        export COPPELIASIM_ROOT=\$COPPELIASIM_ROOT
        export LD_LIBRARY_PATH=\$COPPELIASIM_ROOT:\$LD_LIBRARY_PATH
        export QT_QPA_PLATFORM_PLUGIN_PATH=\$COPPELIASIM_ROOT

        # Prefer system libstdc++/libgcc to avoid conda conflicts with CoppeliaSim
        SYS_LIBSTDCPP=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
        SYS_LIBGCC=/usr/lib/x86_64-linux-gnu/libgcc_s.so.1
        if [ -f \"\$SYS_LIBSTDCPP\" ] && [ -f \"\$SYS_LIBGCC\" ]; then
            if [ -n \"\${LD_PRELOAD:-}\" ]; then
                export LD_PRELOAD=\"\$SYS_LIBSTDCPP:\$SYS_LIBGCC:\$LD_PRELOAD\"
            else
                export LD_PRELOAD=\"\$SYS_LIBSTDCPP:\$SYS_LIBGCC\"
            fi
        fi

        # Clear any host-provided software GL overrides
        unset LIBGL_ALWAYS_SOFTWARE
        unset MESA_GL_VERSION_OVERRIDE

        # Optional: force software rendering (set FORCE_SOFTWARE_RENDERING=1)
        if [ \"\${FORCE_SOFTWARE_RENDERING:-0}\" = \"1\" ]; then
            export MESA_GL_VERSION_OVERRIDE=3.3
            export LIBGL_ALWAYS_SOFTWARE=1
        fi

        # instant_policy from host
        export PYTHONPATH=/workspace/instant_policy:\$PYTHONPATH

        # Checkpoints symlink
        [ -d /workspace/instant_policy/ip/checkpoints ] && [ ! -L /workspace/instant_policy/ip/checkpoints ] && rm -rf /workspace/instant_policy/ip/checkpoints
        [ ! -e /workspace/instant_policy/ip/checkpoints ] && ln -sf /workspace/data/checkpoints /workspace/instant_policy/ip/checkpoints

        cd /workspace/instant_policy/ip
        $CMD
    "

echo "Container exited"
