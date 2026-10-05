#!/usr/bin/env bash
# Run one recorded closed-loop episode: EUFS + scenario + controller + recorder.
# Usage: run_episode.sh <controller> <scenario_id> <seed> <front> <rear> <t_start> <front2> <rear2> [duration]
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO_ROOT"
source "$REPO_ROOT/scripts/env.sh"; set +u
CTRL="${1:?controller}"; SID="${2:?scenario}"; SEED="${3:-1}"
F1="${4:-1.0}"; R1="${5:-1.0}"; TS="${6:--1}"; F2="${7:-1.0}"; R2="${8:-1.0}"
DURATION="${9:-62}"
OUT="$REPO_ROOT/runs/eufs_v1/$SID/$CTRL"; mkdir -p "$OUT"
TRACK="$REPO_ROOT/artifacts/tracks/small_track_reference.npz"
CSV="$OUT/episode.csv"; rm -f "$CSV" "$OUT/episode.metadata.json"
fail(){ echo "EPISODE_FAIL: $1"; exit 1; }
setsid ros2 launch neurogrip_bringup eufs_backend.launch.py headless:=true \
  track:="$REPO_ROOT/external/eufs_ws/src/map_lib/maps/tracks/small_track.csv" > "$OUT/eufs.log" 2>&1 &
EUFS_PID=$!; EUFS_PGID="$(ps -o pgid= -p "$EUFS_PID" | tr -d ' ')"
cleanup(){ for g in "${REC_PGID:-}" "${CTRL_PGID:-}" "${SM_PGID:-}" "${EUFS_PGID:-}"; do [ -n "$g" ] && kill -TERM -- "-$g" 2>/dev/null; done; sleep 3; for g in "${REC_PGID:-}" "${CTRL_PGID:-}" "${SM_PGID:-}" "${EUFS_PGID:-}"; do [ -n "$g" ] && kill -KILL -- "-$g" 2>/dev/null; done; }
trap cleanup EXIT
for _ in $(seq 1 45); do ros2 topic list 2>/dev/null | grep -qx /odom && break; sleep 1; done
ros2 topic list 2>/dev/null | grep -qx /odom || fail "no odom"
setsid ros2 run neurogrip_control scenario_manager --ros-args \
  -p front_grip_scale:=$F1 -p rear_grip_scale:=$R1 \
  -p transition_start_s:=$TS -p transition_front_grip_scale:=$F2 -p transition_rear_grip_scale:=$R2 \
  > "$OUT/scenario.log" 2>&1 &
SM_PID=$!; SM_PGID="$(ps -o pgid= -p "$SM_PID" | tr -d ' ')"
sleep 2
setsid ros2 launch neurogrip_bringup closed_loop.launch.py \
  track_npz:="$TRACK" controller:=$CTRL target_speed:="${TARGET_SPEED:-3.0}" > "$OUT/controller.log" 2>&1 &
CTRL_PID=$!; CTRL_PGID="$(ps -o pgid= -p "$CTRL_PID" | tr -d ' ')"
sleep 4
setsid python experiments_v2/record_episode.py --output "$CSV" --scenario-id "$SID" \
  --controller "$CTRL" --seed "$SEED" > "$OUT/recorder.log" 2>&1 &
REC_PID=$!; REC_PGID="$(ps -o pgid= -p "$REC_PID" | tr -d ' ')"
echo "episode running: $SID/$CTRL (${DURATION}s)"
sleep "$DURATION"
kill -TERM -- "-$REC_PGID" 2>/dev/null || true
sleep 2
ROWS="$(wc -l < "$CSV" 2>/dev/null || echo 0)"
[ "$ROWS" -gt 100 ] || fail "only $ROWS rows recorded"
echo "EPISODE_DONE rows=$((ROWS-1)) csv=$CSV"
