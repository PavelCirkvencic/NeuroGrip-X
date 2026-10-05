#!/usr/bin/env bash
# Create/refresh the workspace Python environment.  Idempotent.
#
#   bash scripts/bootstrap_env.sh
#
# The virtual environment uses --system-site-packages so ROS 2 Humble stays
# importable.  Its site-packages are also re-exported from activate so that the
# system-Python colcon pytest runner can import torch/numpy (see docs/environment.md).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [ ! -d "$REPO_ROOT/.venv" ]; then
    echo "Creating .venv with --system-site-packages"
    python3 -m venv --system-site-packages "$REPO_ROOT/.venv"
fi

# shellcheck disable=SC1091
source "$REPO_ROOT/.venv/bin/activate"
python -m pip install -r "$REPO_ROOT/requirements.txt"

MARKER="# NeuroGrip-X: expose venv site-packages to system-Python colcon tests"
if ! grep -qF "$MARKER" "$REPO_ROOT/.venv/bin/activate"; then
    cat >> "$REPO_ROOT/.venv/bin/activate" <<'EOF'

# NeuroGrip-X: expose venv site-packages to system-Python colcon tests
export PYTHONPATH="$VIRTUAL_ENV/lib/python3.10/site-packages${PYTHONPATH:+:$PYTHONPATH}"
EOF
    echo "Patched .venv/bin/activate with PYTHONPATH export"
fi

echo "Environment ready. Use: source scripts/env.sh"
