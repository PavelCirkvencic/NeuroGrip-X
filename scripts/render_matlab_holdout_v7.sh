#!/usr/bin/env bash
# Render the accepted MATLAB-actuated v7 holdout from immutable telemetry.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
# shellcheck disable=SC1091
source "$REPO_ROOT/scripts/env.sh"
set +u

RUN_ROOT="runs/eufs_v1/final_holdout_matlab_actuated_v7"
SCENARIO="matlabrelease_trackdrive_balanced_seed0933"
LEFT="$RUN_ROOT/${SCENARIO}__C0_FIXED__seed0933/episode.csv"
RIGHT="$RUN_ROOT/${SCENARIO}__C2_NEUROGRIP__seed0933/episode.csv"
REPORT="artifacts/reports/final_holdout_matlab_actuated_v7/benchmark_summary.json"
OUTPUT="${1:-artifacts/videos/neurogrip_x_matlab_holdout_v7.mp4}"
TEMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TEMP_DIR"' EXIT
BASE_VIDEO="$TEMP_DIR/base.mp4"
SCOPE_VIDEO="$TEMP_DIR/simulink_grip_scope.avi"

test -f "$LEFT" || { echo "missing frozen C0 episode: $LEFT" >&2; exit 2; }
test -f "$RIGHT" || { echo "missing frozen C2 episode: $RIGHT" >&2; exit 2; }
test -f "$REPORT" || { echo "missing accepted benchmark report: $REPORT" >&2; exit 2; }

python experiments_v2/render_telemetry_video.py \
    --left-csv "$LEFT" \
    --right-csv "$RIGHT" \
    --track-npz artifacts/tracks/trackdrive_reference.npz \
    --left-label "Standard" \
    --right-label "NeuroGripX" \
    --output "$BASE_VIDEO" \
    --width 1920 --height 1080 --fps 30 --playback-rate 2.5 \
    --intro-title "MATLAB-in-the-loop grip-aware MPC" \
    --intro-subtitle "ROS 2 + EUFS DynamicBicycle | Same 43.2 km/h cap | Paired sensor noise" \
    --intro-duration-s 2.5 \
    --outro-title "5.17% lower mean lap time" \
    --outro-line "All 6 hash-frozen holdout laps completed - zero collisions or boundary exits" \
    --outro-line "Representative run: 30.70 s fixed-mu vs 28.94 s NeuroGrip-X (5.73% faster)" \
    --outro-line "MATLAB active-set MPC: zero active fallback, rejection, or deadline events" \
    --outro-highlight "Around the Red Bull Ring, that would mean about 3.39 seconds." \
    --outro-highlight-detail "Scale reference: official 1:05.619 F1 lap record (Carlos Sainz, 2020)" \
    --outro-highlight-2 "Around the Nordschleife, that would mean about 16.52 seconds." \
    --outro-highlight-detail-2 "Scale reference: 5:19.546 outright record (Porsche 919 Hybrid Evo, 2018)" \
    --outro-duration-s 5.0 \
    --overwrite

/opt/MATLAB/R2026a/bin/matlab -batch \
    "cd('$REPO_ROOT'); addpath(fullfile(pwd,'matlab')); render_grip_scope_video('$RIGHT','$SCOPE_VIDEO',5.0,30);"

BASE_DURATION="$(ffprobe -v error -show_entries format=duration \
    -of default=noprint_wrappers=1:nokey=1 "$BASE_VIDEO")"
OVERLAY_START="$(awk -v duration="$BASE_DURATION" 'BEGIN { printf "%.6f", duration - 5.0 }')"
ffmpeg -y -loglevel error \
    -i "$BASE_VIDEO" -i "$SCOPE_VIDEO" \
    -filter_complex \
    "[1:v]scale=870:690:force_original_aspect_ratio=decrease,setpts=PTS-STARTPTS+${OVERLAY_START}/TB[scope];[0:v][scope]overlay=980:210:eof_action=pass:enable='gte(t,${OVERLAY_START})'[video]" \
    -map "[video]" -map 0:a? -c:v libx264 -preset slow -crf 17 \
    -pix_fmt yuv420p -movflags +faststart -c:a copy "$OUTPUT"

POSTER="${OUTPUT%.*}_poster.png"
ffmpeg -y -loglevel error -ss 16.8 -i "$OUTPUT" -frames:v 1 "$POSTER"
echo "MATLAB_HOLDOUT_VIDEO_CREATED output=$OUTPUT poster=$POSTER"
