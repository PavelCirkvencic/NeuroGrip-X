#!/usr/bin/env bash
# Source this file from the repository root before building or testing:
#
#   cd NeuroGrip-X
#   source scripts/env.sh
#
# It must work from a completely clean shell (env -i).  Order matters:
#   1. repo root
#   2. activate the project venv
#   3. source ROS 2 Humble under `set +u` (ROS setup scripts use unset vars)
#   4. source the pinned EUFS overlay if it has been built
#   5. source this workspace overlay if it has been built
#   6. only then enable `set -u`
#
# The venv site-packages are prepended to PYTHONPATH exactly once so the
# system-Python colcon/pytest runner can import torch/numpy.  The caller's
# existing PYTHONPATH is preserved.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [ ! -f "$REPO_ROOT/.venv/bin/activate" ]; then
    echo "error: .venv is missing; run scripts/bootstrap_env.sh first" >&2
    return 1 2>/dev/null || exit 1
fi

# shellcheck disable=SC1091
source "$REPO_ROOT/.venv/bin/activate"

set +u
# shellcheck disable=SC1091
source /opt/ros/humble/setup.bash

# Local, sudo-free dependency prefix (e.g. geodesy/pyproj extracted from .deb
# by scripts/bootstrap_eufs.sh when the system package is unavailable).
EUFS_LOCAL_PREFIX="$REPO_ROOT/external/deps/local_prefix"
if [ -d "$EUFS_LOCAL_PREFIX/opt/ros/humble" ]; then
    export AMENT_PREFIX_PATH="$EUFS_LOCAL_PREFIX/opt/ros/humble${AMENT_PREFIX_PATH:+:$AMENT_PREFIX_PATH}"
    export CMAKE_PREFIX_PATH="$EUFS_LOCAL_PREFIX/opt/ros/humble${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
    export LD_LIBRARY_PATH="$EUFS_LOCAL_PREFIX/opt/ros/humble/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
for local_site in \
    "$EUFS_LOCAL_PREFIX/opt/ros/humble/local/lib/python3.10/dist-packages" \
    "$EUFS_LOCAL_PREFIX/usr/lib/python3/dist-packages"; do
    if [ -d "$local_site" ]; then
        export PYTHONPATH="$local_site${PYTHONPATH:+:$PYTHONPATH}"
    fi
done

if [ -f "$REPO_ROOT/external/eufs_ws/install/setup.bash" ]; then
    # shellcheck disable=SC1091
    source "$REPO_ROOT/external/eufs_ws/install/setup.bash"
fi

if [ -f "$REPO_ROOT/install/setup.bash" ]; then
    # shellcheck disable=SC1091
    source "$REPO_ROOT/install/setup.bash"
fi
set -u

export EUFS_MASTER="$REPO_ROOT/external/eufs_ws"

VENV_SITE="$VIRTUAL_ENV/lib/python3.10/site-packages"
case ":${PYTHONPATH:-}:" in
    *":$VENV_SITE:"*) ;;
    *) export PYTHONPATH="$VENV_SITE${PYTHONPATH:+:$PYTHONPATH}" ;;
esac

# ROS-free model definitions live in learning/neurogrip and are shared by the
# trainer, offline evaluator and deployed inference wrapper.
LEARNING_ROOT="$REPO_ROOT/learning"
case ":${PYTHONPATH:-}:" in
    *":$LEARNING_ROOT:"*) ;;
    *) export PYTHONPATH="$LEARNING_ROOT${PYTHONPATH:+:$PYTHONPATH}" ;;
esac

echo "NeuroGrip-X environment ready (venv=$VIRTUAL_ENV, ROS_DISTRO=${ROS_DISTRO:-unset}, EUFS_MASTER=$EUFS_MASTER)"
