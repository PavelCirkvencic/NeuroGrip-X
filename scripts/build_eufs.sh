#!/usr/bin/env bash
# Build the pinned EUFS workspace.  Exits non-zero on any build error.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# shellcheck disable=SC1091
source "$REPO_ROOT/scripts/env.sh"

mkdir -p "$REPO_ROOT/runs/eufs_v1/bootstrap"
LOG="$REPO_ROOT/runs/eufs_v1/bootstrap/eufs_build.log"

cd "$REPO_ROOT/external/eufs_ws"
colcon build --symlink-install --cmake-args -DBUILD_TESTING=OFF 2>&1 | tee "$LOG"
echo "EUFS build finished; log: $LOG"
