#!/usr/bin/env bash
# EUFS Sim 2 runtime smoke gate (headless).
#
# Launches the pinned backend, runs the rclpy checker and only ever signals the
# process group it created.  Prints EUFS_SMOKE_PASS on success.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# shellcheck disable=SC1091
source "$REPO_ROOT/scripts/env.sh"
set +u

OUT_DIR="$REPO_ROOT/runs/eufs_v1/bootstrap"
mkdir -p "$OUT_DIR"
LOG="$OUT_DIR/eufs_smoke.log"
SUMMARY="$OUT_DIR/eufs_smoke_summary.json"
TRACK="$REPO_ROOT/external/eufs_ws/src/map_lib/maps/tracks/small_track.csv"

setsid ros2 launch neurogrip_bringup eufs_backend.launch.py \
    headless:=true track:="$TRACK" seed:=1 > "$LOG" 2>&1 &
LAUNCH_PID=$!
PGID="$(ps -o pgid= -p "$LAUNCH_PID" 2>/dev/null | tr -d ' ')"
if [ -z "$PGID" ]; then
    echo "error: could not determine launch process group" >&2
    exit 1
fi

cleanup() {
    kill -TERM -- "-$PGID" 2>/dev/null || true
    sleep 3
    kill -KILL -- "-$PGID" 2>/dev/null || true
}
trap cleanup EXIT

# Wait for the backend to advertise odometry before running the checker.
for _ in $(seq 1 60); do
    if ros2 topic list 2>/dev/null | grep -qx "/odom"; then
        break
    fi
    sleep 1
done
if ! ros2 topic list 2>/dev/null | grep -qx "/odom"; then
    echo "SMOKE_FAIL: /odom never appeared (see $LOG)" | tee -a "$LOG"
    echo "EUFS_SMOKE_FAIL"
    exit 1
fi

python "$REPO_ROOT/scripts/smoke_eufs_check.py" --output "$SUMMARY"
CHECK_EXIT=$?

if [ "$CHECK_EXIT" -ne 0 ]; then
    echo "EUFS_SMOKE_FAIL"
    echo "Log: $LOG"
    exit 1
fi

echo "EUFS_SMOKE_PASS"
echo "Summary: $SUMMARY"
