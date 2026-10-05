#!/usr/bin/env bash
# Local CI gate: build, lint, unit tests, preflight and worktree hygiene.
#
#   bash scripts/ci_check.sh
#
# Safe for a machine without a MATLAB license (preflight only inspects the
# installation) and without a display (no Gazebo GUI is started).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

bash "$REPO_ROOT/scripts/run_tests.sh"

echo "== preflight =="
# GitHub-style CI hosts may not have MATLAB; treat it as advisory here.
# `run_tests.sh` executes in a child shell, so source the workspace overlay
# again before asking `ros2 pkg list` to verify the local packages.
# shellcheck disable=SC1091
source "$REPO_ROOT/scripts/env.sh"
"$REPO_ROOT/.venv/bin/python" \
    "$REPO_ROOT/scripts/preflight_check.py" --allow-missing-matlab

echo "== worktree hygiene =="
git diff --check
echo "Local CI passed."
