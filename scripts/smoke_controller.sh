#!/usr/bin/env bash
# Generalised Python controller smoke (C0_FIXED / C1_ORACLE_MU / C2_NEUROGRIP).
#
# Usage: bash scripts/smoke_controller.sh C1_ORACLE_MU [front_scale] [rear_scale]
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO_ROOT"
source "$REPO_ROOT/scripts/env.sh"; set +u

CONTROLLER="${1:-C0_FIXED}"
FRONT="${2:-1.0}"
REAR="${3:-1.0}"
TARGET_SPEED="${TARGET_SPEED:-3.0}"
OUT="$REPO_ROOT/runs/eufs_v1/controllers/python_smoke/$CONTROLLER"
mkdir -p "$OUT"
TRACK="$REPO_ROOT/artifacts/tracks/small_track_reference.npz"
CSV="$OUT/tracking.csv"
rm -f "$CSV"

fail(){ echo "CONTROLLER_FAIL: $1"; echo "PYTHON_${CONTROLLER}_FAIL"; exit 1; }

setsid ros2 launch neurogrip_bringup eufs_backend.launch.py headless:=true \
  track:="$REPO_ROOT/external/eufs_ws/src/map_lib/maps/tracks/small_track.csv" > "$OUT/eufs.log" 2>&1 &
EUFS_PID=$!; EUFS_PGID="$(ps -o pgid= -p "$EUFS_PID" | tr -d ' ')"
cleanup(){ for g in "${EXTRA_PGIDS:-}" "${CTRL_PGID:-}" "${EUFS_PGID:-}"; do [ -n "$g" ] && kill -TERM -- "-$g" 2>/dev/null; done; sleep 3; for g in "${EXTRA_PGIDS:-}" "${CTRL_PGID:-}" "${EUFS_PGID:-}"; do [ -n "$g" ] && kill -KILL -- "-$g" 2>/dev/null; done; }
trap cleanup EXIT

for _ in $(seq 1 45); do ros2 topic list 2>/dev/null | grep -qx /odom && break; sleep 1; done
ros2 topic list 2>/dev/null | grep -qx /odom || fail "no odom"

setsid ros2 run neurogrip_control scenario_manager --ros-args \
  -p front_grip_scale:=$FRONT -p rear_grip_scale:=$REAR \
  -p transition_start_s:=-1.0 > "$OUT/scenario.log" 2>&1 &
SM_PID=$!; EXTRA_PGIDS="$(ps -o pgid= -p "$SM_PID" | tr -d ' ')"
sleep 2
setsid ros2 launch neurogrip_bringup closed_loop.launch.py \
  track_npz:="$TRACK" controller:=$CONTROLLER target_speed:=$TARGET_SPEED \
  > "$OUT/controller.log" 2>&1 &
CTRL_PID=$!; CTRL_PGID="$(ps -o pgid= -p "$CTRL_PID" | tr -d ' ')"
sleep 4
CMD_PUBS="$(ros2 topic info /cmd 2>/dev/null | grep -oP 'Publisher count:\s*\K[0-9]+')"
[ "${CMD_PUBS:-0}" = "1" ] || fail "/cmd publishers=$CMD_PUBS"

timeout 58 ros2 topic echo --csv /neurogrip/tracking_state > "$CSV" 2>/dev/null &
ECHO_PID=$!
wait "$ECHO_PID" 2>/dev/null || true
ROWS="$(wc -l < "$CSV")"
[ "$ROWS" -gt 50 ] || fail "too few tracking rows ($ROWS)"

python3 - "$CSV" "$CONTROLLER" "$FRONT" "$REAR" "$OUT/summary.json" <<'PY'
import csv, json, sys, math
rows=[]
with open(sys.argv[1]) as handle:
    for line in handle:
        parts=[p for p in line.strip().split(',') if p!='']
        if len(parts)>=9:
            try:
                rows.append([float(v) for v in parts[-9:]])
            except ValueError:
                pass
if not rows:
    print("CONTROLLER_FAIL: no numeric rows"); raise SystemExit(1)
e_y=[r[2] for r in rows]
progress=[r[8] for r in rows]
max_progress=max(progress)
rms=math.sqrt(sum(v*v for v in e_y)/len(e_y))
summary={"controller":sys.argv[2],"front_scale":float(sys.argv[3]),"rear_scale":float(sys.argv[4]),
         "rows":len(rows),"rms_lateral_error_m":rms,"max_abs_lateral_error_m":max(abs(v) for v in e_y),
         "max_progress":max_progress,"completed_lap":max_progress>=0.999}
json.dump(summary,open(sys.argv[5],"w"),indent=2)
print(json.dumps(summary))
if max_progress < 0.999:
    print(f"CONTROLLER_FAIL: lap not completed (max_progress={max_progress:.3f})"); raise SystemExit(1)
PY
[ $? -eq 0 ] || { echo "PYTHON_${CONTROLLER}_FAIL"; exit 1; }
echo "PYTHON_${CONTROLLER}_LAP_PASS"
