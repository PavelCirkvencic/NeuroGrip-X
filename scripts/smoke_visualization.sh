#!/usr/bin/env bash
# Gazebo Fortress visual frontend smoke gate.
#
# Builds the visual world, runs EUFS + Gazebo + sync + recorder, drives a short
# arc, checks entity motion and camera fps, and encodes a 10 s smoke video.
# Only signals the process groups it created.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
# shellcheck disable=SC1091
source "$REPO_ROOT/scripts/env.sh"
set +u

OUT="$REPO_ROOT/runs/eufs_v1/visual"
mkdir -p "$OUT"
FRAMES="$OUT/frames"
rm -rf "$FRAMES"; mkdir -p "$FRAMES"
WORLD="$OUT/neurogrip_visual.sdf"
MESH="$REPO_ROOT/external/eufs_ws/src/eufs_sim2/eufs_racecar/car/meshes/ads-dv.dae"
TRACK="$REPO_ROOT/artifacts/tracks/small_track_visual.sdf"
# ``headless`` is the default because it is safe for CI-like runs.  Some
# hybrid-GPU laptops cannot create an EGL pbuffer even though their normal X11
# OpenGL path works; ``NEUROGRIP_VISUAL_MODE=gui`` is the explicit diagnostic
# fallback for that machine.  Neither mode affects EUFS dynamics or metrics.
VISUAL_MODE="${NEUROGRIP_VISUAL_MODE:-headless}"

fail() { echo "VISUAL_FAIL: $1"; echo "GAZEBO_VISUAL_SYNC_FAIL"; exit 1; }

python -m neurogrip_visualization.build_world --track-sdf "$TRACK" --mesh "$MESH" --output "$WORLD" --world-name neurogrip_visual --entity-name formula_student_visual >/dev/null || fail "world build failed"
ign sdf -k "$WORLD" >/dev/null 2>&1 || fail "generated world is not valid SDF"
grep -q 'ads-dv.dae' "$WORLD" || fail "mesh not referenced in world"

# EUFS backend
setsid ros2 launch neurogrip_bringup eufs_backend.launch.py headless:=true track:="$REPO_ROOT/external/eufs_ws/src/map_lib/maps/tracks/small_track.csv" > "$OUT/eufs.log" 2>&1 &
EUFS_PID=$!
EUFS_PGID="$(ps -o pgid= -p "$EUFS_PID" 2>/dev/null | tr -d ' ')"
# Gazebo needs an active renderer for its camera sensor.  ``-s`` starts a
# server-only process and therefore exposes the camera topic without producing
# images.  Headless rendering keeps the test GUI-free but retains OGRE2.
case "$VISUAL_MODE" in
    headless)
        setsid ign gazebo -r --headless-rendering "$WORLD" --force-version 6 > "$OUT/gazebo.log" 2>&1 &
        ;;
    gui)
        setsid ign gazebo -r "$WORLD" --force-version 6 > "$OUT/gazebo.log" 2>&1 &
        ;;
    *)
        fail "unknown NEUROGRIP_VISUAL_MODE=$VISUAL_MODE (use headless or gui)"
        ;;
esac
GZ_PID=$!
GZ_PGID="$(ps -o pgid= -p "$GZ_PID" 2>/dev/null | tr -d ' ')"
# Bridges: camera image + SetEntityPose service
setsid ros2 run ros_gz_bridge parameter_bridge \
    '/camera/image@sensor_msgs/msg/Image[ignition.msgs.Image' \
    '/world/neurogrip_visual/set_pose@ros_gz_interfaces/srv/SetEntityPose' > "$OUT/bridge.log" 2>&1 &
BR_PID=$!
BR_PGID="$(ps -o pgid= -p "$BR_PID" 2>/dev/null | tr -d ' ')"

cleanup() {
    for pgid in "${PUB_PGID:-}" "${REC_PGID:-}" "${SYNC_PGID:-}" "${BR_PGID:-}" "${GZ_PGID:-}" "${EUFS_PGID:-}"; do
        if [ -n "$pgid" ]; then
            kill -TERM -- "-$pgid" 2>/dev/null || true
        fi
    done
    sleep 3
    for pgid in "${PUB_PGID:-}" "${REC_PGID:-}" "${SYNC_PGID:-}" "${BR_PGID:-}" "${GZ_PGID:-}" "${EUFS_PGID:-}"; do
        if [ -n "$pgid" ]; then
            kill -KILL -- "-$pgid" 2>/dev/null || true
        fi
    done
}
trap cleanup EXIT

for _ in $(seq 1 45); do
    ros2 topic list 2>/dev/null | grep -qx "/odom" && break
    sleep 1
done
ros2 topic list 2>/dev/null | grep -qx "/odom" || fail "no /odom from EUFS"
for _ in $(seq 1 30); do
    ros2 topic list 2>/dev/null | grep -qx "/camera/image" && break
    sleep 1
done
ros2 topic list 2>/dev/null | grep -qx "/camera/image" || fail "no /camera/image (rendering?)"

setsid ros2 run neurogrip_visualization visual_sync_node --ros-args \
    -p world_name:=neurogrip_visual -p entity_name:=formula_student_visual > "$OUT/sync.log" 2>&1 &
SYNC_PID=$!
SYNC_PGID="$(ps -o pgid= -p "$SYNC_PID" 2>/dev/null | tr -d ' ')"
setsid ros2 run neurogrip_visualization video_recorder_node --ros-args \
    -p output_dir:="$FRAMES" -p max_frames:=300 > "$OUT/recorder.log" 2>&1 &
REC_PID=$!
REC_PGID="$(ps -o pgid= -p "$REC_PID" 2>/dev/null | tr -d ' ')"
sleep 3

# Drive a short arc so the visual entity must move.
setsid ros2 topic pub -r 20 /cmd ackermann_msgs/msg/AckermannDriveStamped \
    "{drive: {acceleration: 1.5, steering_angle: 0.1}}" > /dev/null 2>&1 &
PUB_PID=$!
PUB_PGID="$(ps -o pgid= -p "$PUB_PID" 2>/dev/null | tr -d ' ')"
sleep 8
kill -TERM -- "-$PUB_PGID" 2>/dev/null || true

SYNC_SENT="$(timeout 3 ros2 topic echo --once /neurogrip/visual_diagnostics 2>/dev/null | grep -oE '"sent": [0-9]+' | grep -oE '[0-9]+')"
FRAME_COUNT="$(ls "$FRAMES" | wc -l)"
HZ="$(timeout 6 ros2 topic hz /camera/image 2>/dev/null | grep -m1 'average rate' | sed -E 's/.*average rate: ([0-9.]+).*/\1/')"

[ -n "$SYNC_SENT" ] && [ "$SYNC_SENT" -gt 0 ] || fail "visual sync sent no poses"
[ -n "$FRAME_COUNT" ] && [ "$FRAME_COUNT" -gt 0 ] || fail "no camera frames recorded"
awk -v hz="${HZ:-0}" 'BEGIN{exit !(hz>=25)}' || fail "camera fps below 25: ${HZ:-none}"

# Encode a short smoke video from the recorded frames (10 s worth at 30 fps).
ffmpeg -y -framerate 30 -i "$FRAMES/frame_%06d.ppm" -t 10 -pix_fmt yuv420p -c:v libx264 "$OUT/visual_smoke.mp4" >/dev/null 2>&1 || fail "ffmpeg encode failed"
[ -s "$OUT/visual_smoke.mp4" ] || fail "smoke video missing"

echo "CAMERA_HZ=$HZ"
echo "FRAMES=$FRAME_COUNT"
echo "SYNC_SENT=$SYNC_SENT"
echo "GAZEBO_VISUAL_SYNC_PASS"
