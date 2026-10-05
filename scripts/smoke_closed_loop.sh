#!/usr/bin/env bash
# Conservative C0 Python MPC closed-loop smoke on small_track.
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO_ROOT"
source "$REPO_ROOT/scripts/env.sh"; set +u
OUT="$REPO_ROOT/runs/eufs_v1/controllers/python_smoke/C0_FIXED"; mkdir -p "$OUT"
TRACK="$REPO_ROOT/artifacts/tracks/small_track_reference.npz"
setsid ros2 launch neurogrip_bringup eufs_backend.launch.py headless:=true \
  track:="$REPO_ROOT/external/eufs_ws/src/map_lib/maps/tracks/small_track.csv" > "$OUT/eufs.log" 2>&1 &
EUFS_PID=$!; EUFS_PGID="$(ps -o pgid= -p "$EUFS_PID" | tr -d ' ')"
cleanup(){ for g in "${CTRL_PGID:-}" "${EUFS_PGID:-}"; do [ -n "$g" ] && kill -TERM -- "-$g" 2>/dev/null; done; sleep 3; for g in "${CTRL_PGID:-}" "${EUFS_PGID:-}"; do [ -n "$g" ] && kill -KILL -- "-$g" 2>/dev/null; done; }
trap cleanup EXIT
for _ in $(seq 1 45); do ros2 topic list 2>/dev/null | grep -qx /odom && break; sleep 1; done
ros2 topic list 2>/dev/null | grep -qx /odom || { echo "CLOSED_LOOP_FAIL: no odom"; exit 1; }
setsid ros2 launch neurogrip_bringup closed_loop.launch.py track_npz:="$TRACK" controller:=C0_FIXED target_speed:=3.0 > "$OUT/controller.log" 2>&1 &
CTRL_PID=$!; CTRL_PGID="$(ps -o pgid= -p "$CTRL_PID" | tr -d ' ')"
sleep 6
CMD_PUBS="$(ros2 topic info /cmd 2>/dev/null | grep -oP 'Publisher count:\s*\K[0-9]+')"
FIRST_PROGRESS="$(python3 "$REPO_ROOT/scripts/read_tracking_progress.py" --seconds 3)"
sleep 13
LAST_PROGRESS="$(python3 "$REPO_ROOT/scripts/read_tracking_progress.py" --seconds 3)"
echo "cmd_publishers=$CMD_PUBS"
echo "first_progress=$FIRST_PROGRESS"
echo "last_progress=$LAST_PROGRESS"
[ "${CMD_PUBS:-0}" = "1" ] || { echo "CLOSED_LOOP_FAIL: /cmd publishers=$CMD_PUBS"; exit 1; }
python3 - "$FIRST_PROGRESS" "$LAST_PROGRESS" <<'PY'
import sys
try:
    first = float(sys.argv[1]); last = float(sys.argv[2])
except Exception:
    print("CLOSED_LOOP_FAIL: could not parse progress"); raise SystemExit(1)
if last <= first:
    print(f"CLOSED_LOOP_FAIL: progress did not advance ({first} -> {last})"); raise SystemExit(1)
print(f"progress_advanced={first:.3f}->{last:.3f}")
PY
[ $? -eq 0 ] || exit 1
echo "PYTHON_C0_SMOKE_PASS"
