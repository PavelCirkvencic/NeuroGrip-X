#!/usr/bin/env python3
"""Render an honest Formula Student telemetry replay from EUFS episode CSVs.

This renderer is deliberately *not* a physics simulator. It uses recorded
world pose when telemetry schema 3 provides it, otherwise reconstructing pose
from the independent track projection and recorded tracking errors. Every
displayed trajectory, speed, grip value and error comes from the supplied
episode CSV rather than from a decorative animation.

The output is intended for fast internal review and a later, clearly labelled
``Telemetry replay — EUFS DynamicBicycle`` clip.  It must never be presented as
a Gazebo camera recording.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class Episode:
    """Continuous, lap-truncated measurements required by the replay."""

    label: str
    time_s: np.ndarray
    progress: np.ndarray
    lateral_error_m: np.ndarray
    heading_error_rad: np.ndarray
    speed_mps: np.ndarray
    front_grip: np.ndarray
    rear_grip: np.ndarray
    x_m: np.ndarray | None = None
    y_m: np.ndarray | None = None
    yaw_rad: np.ndarray | None = None


@dataclass(frozen=True)
class LapTiming:
    """Monotonic full-lap progress and inverse passage-time lookup."""

    sample_progress: np.ndarray
    passage_progress: np.ndarray
    passage_time_s: np.ndarray


def parse_arguments() -> argparse.Namespace:
    """Parse two recorded episodes and the immutable reference path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left-csv", type=Path, required=True)
    parser.add_argument("--right-csv", type=Path, required=True)
    parser.add_argument("--track-npz", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--left-label", default="C0 - Fixed MPC")
    parser.add_argument("--right-label", default="C2 - NeuroGrip-X")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--playback-rate", type=float, default=4.0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--max-duration-s", type=float, default=0.0)
    parser.add_argument("--intro-title", default="")
    parser.add_argument("--intro-subtitle", default="")
    parser.add_argument("--intro-duration-s", type=float, default=0.0)
    parser.add_argument("--outro-title", default="")
    parser.add_argument("--outro-line", action="append", default=[])
    parser.add_argument("--outro-highlight", default="")
    parser.add_argument("--outro-highlight-detail", default="")
    parser.add_argument("--outro-highlight-2", default="")
    parser.add_argument("--outro-highlight-detail-2", default="")
    parser.add_argument("--outro-duration-s", type=float, default=0.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def measured_interval(csv_path: Path) -> tuple[float | None, float | None]:
    """Read the explicit measured-lap interval from the companion event log."""
    event_path = csv_path.with_suffix(".events.jsonl")
    if not event_path.is_file():
        return None, None
    start = None
    finish = None
    for line in event_path.read_text(encoding="utf-8").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSON event in {event_path}") from error
        if event.get("event") == "experiment_start" and start is None:
            start = float(event["sim_time_s"])
        if event.get("event") == "lap_complete" and finish is None:
            finish = float(event["sim_time_s"])
    return start, finish


def load_episode(path: Path, label: str) -> Episode:
    """Strictly load one finite, monotonic episode without hiding bad rows."""
    required = (
        "time_s", "lap_progress", "e_y_m", "e_psi_rad", "v_x_mps",
        "grip_front", "grip_rear",
    )
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        rows = list(reader)
    if len(rows) < 100:
        raise ValueError(f"{path}: at least 100 telemetry samples are required")
    values: dict[str, np.ndarray] = {}
    for name in required:
        try:
            values[name] = np.asarray([float(row[name]) for row in rows], dtype=float)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"{path}: invalid required column {name}") from error
    raw_pose_fields = {"x_m", "y_m", "yaw_rad"}
    raw_pose = None
    if raw_pose_fields <= fieldnames:
        try:
            raw_pose = {
                name: np.asarray([float(row[name]) for row in rows], dtype=float)
                for name in raw_pose_fields
            }
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"{path}: invalid recorded raw pose") from error
        values.update(raw_pose)
    if not all(np.all(np.isfinite(value)) for value in values.values()):
        raise ValueError(f"{path}: non-finite telemetry cannot be rendered")
    if np.any(np.diff(values["time_s"]) <= 0.0):
        raise ValueError(f"{path}: timestamps must be strictly increasing")
    started_at, completed_at = measured_interval(path)
    if started_at is not None or completed_at is not None:
        lower = values["time_s"][0] if started_at is None else started_at
        upper = values["time_s"][-1] if completed_at is None else completed_at
        keep = (values["time_s"] >= lower - 1e-9) & (values["time_s"] <= upper + 1e-9)
        values = {name: value[keep] for name, value in values.items()}
    if len(values["time_s"]) < 100:
        raise ValueError(f"{path}: no usable data before lap completion")
    return Episode(
        label=label,
        time_s=values["time_s"] - values["time_s"][0],
        progress=np.mod(values["lap_progress"], 1.0),
        lateral_error_m=values["e_y_m"],
        heading_error_rad=values["e_psi_rad"],
        speed_mps=values["v_x_mps"],
        front_grip=values["grip_front"],
        rear_grip=values["grip_rear"],
        x_m=values.get("x_m"),
        y_m=values.get("y_m"),
        yaw_rad=values.get("yaw_rad"),
    )


def periodic_interpolate(values: np.ndarray, progress: np.ndarray) -> np.ndarray:
    """Interpolate one periodic reference array at fractional lap progress."""
    source = np.linspace(0.0, 1.0, len(values), endpoint=False)
    padded_progress = np.r_[source, 1.0]
    padded_values = np.r_[values, values[0]]
    return np.interp(np.mod(progress, 1.0), padded_progress, padded_values)


def reconstruct_pose(
    track: dict[str, np.ndarray], episode: Episode
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Use recorded v3 pose, with reconstruction only for obsolete v2 traces."""
    if (
        episode.x_m is not None
        and episode.y_m is not None
        and episode.yaw_rad is not None
    ):
        return episode.x_m, episode.y_m, episode.yaw_rad
    reference_x = periodic_interpolate(track["x"], episode.progress)
    reference_y = periodic_interpolate(track["y"], episode.progress)
    reference_yaw = periodic_interpolate(np.unwrap(track["yaw"]), episode.progress)
    x = reference_x - episode.lateral_error_m * np.sin(reference_yaw)
    y = reference_y + episode.lateral_error_m * np.cos(reference_yaw)
    return x, y, reference_yaw + episode.heading_error_rad


def interpolate_episode(episode: Episode, at_s: float) -> dict[str, float]:
    """Sample one continuous telemetry episode at a relative replay time."""
    return {
        "progress": float(np.interp(at_s, episode.time_s, episode.progress)),
        "e_y": float(np.interp(at_s, episode.time_s, episode.lateral_error_m)),
        "e_psi": float(np.interp(at_s, episode.time_s, episode.heading_error_rad)),
        "speed": float(np.interp(at_s, episode.time_s, episode.speed_mps)),
        "front": float(np.interp(at_s, episode.time_s, episode.front_grip)),
        "rear": float(np.interp(at_s, episode.time_s, episode.rear_grip)),
    }


def build_lap_timing(episode: Episode) -> LapTiming:
    """Build a monotonic zero-to-one lap clock from wrapped track progress."""
    unwrapped = np.unwrap(episode.progress * 2.0 * np.pi) / (2.0 * np.pi)
    unwrapped = np.maximum.accumulate(unwrapped - unwrapped[0])
    if unwrapped[-1] < 0.90:
        raise ValueError(f"{episode.label}: telemetry does not contain a full lap")
    sample_progress = unwrapped / unwrapped[-1]
    keep = np.r_[True, np.diff(sample_progress) > 1e-9]
    passage_progress = sample_progress[keep]
    passage_time_s = episode.time_s[keep]
    if passage_progress[-1] < 1.0 - 1e-9:
        passage_progress = np.r_[passage_progress, 1.0]
        passage_time_s = np.r_[passage_time_s, episode.time_s[-1]]
    return LapTiming(sample_progress, passage_progress, passage_time_s)


def same_distance_interval(
    left: Episode,
    right: Episode,
    left_timing: LapTiming,
    right_timing: LapTiming,
    at_s: float,
) -> tuple[bool, float]:
    """Return the current leader and causal time gap at the trailer's progress."""
    left_now = min(max(at_s, 0.0), float(left.time_s[-1]))
    right_now = min(max(at_s, 0.0), float(right.time_s[-1]))
    left_progress = float(
        np.interp(left_now, left.time_s, left_timing.sample_progress)
    )
    right_progress = float(
        np.interp(right_now, right.time_s, right_timing.sample_progress)
    )
    common_progress = min(left_progress, right_progress)
    left_passage = float(
        np.interp(
            common_progress,
            left_timing.passage_progress,
            left_timing.passage_time_s,
        )
    )
    right_passage = float(
        np.interp(
            common_progress,
            right_timing.passage_progress,
            right_timing.passage_time_s,
        )
    )
    if abs(left_progress - right_progress) <= 1e-9:
        left_leads = left_passage <= right_passage
    else:
        left_leads = left_progress > right_progress
    return left_leads, abs(left_passage - right_passage)


def layout_transform(track: dict[str, np.ndarray], width: int, height: int):
    """Return a world-to-pixel map with a little space for labels and panels."""
    left_x = track["x"] - track["left_width"] * np.sin(track["yaw"])
    left_y = track["y"] + track["left_width"] * np.cos(track["yaw"])
    right_x = track["x"] + track["right_width"] * np.sin(track["yaw"])
    right_y = track["y"] - track["right_width"] * np.cos(track["yaw"])
    x_values, y_values = np.r_[left_x, right_x], np.r_[left_y, right_y]
    span_x, span_y = np.ptp(x_values), np.ptp(y_values)
    scale = min((width - 56.0) / max(span_x, 1.0), (height - 190.0) / max(span_y, 1.0))
    center_x, center_y = float(np.mean(x_values)), float(np.mean(y_values))

    def world_to_pixel(x: np.ndarray | float, y: np.ndarray | float) -> np.ndarray:
        points = np.column_stack((np.atleast_1d(x), np.atleast_1d(y))).astype(float)
        pixels = np.empty_like(points)
        pixels[:, 0] = (points[:, 0] - center_x) * scale + width / 2.0
        pixels[:, 1] = (center_y - points[:, 1]) * scale + height / 2.0 + 34.0
        return np.rint(pixels).astype(np.int32)

    return world_to_pixel, scale


def draw_text(
    frame: np.ndarray,
    text: str,
    origin: tuple[int, int],
    scale: float,
    color: tuple[int, int, int],
    thickness: int = 1,
) -> None:
    """Draw compact antialiased text with a dark outline for readability."""
    cv2.putText(
        frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale,
        (12, 16, 24), thickness + 3, cv2.LINE_AA,
    )
    cv2.putText(
        frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale,
        color, thickness, cv2.LINE_AA,
    )


def draw_information_card(
    width: int,
    height: int,
    eyebrow: str,
    title: str,
    lines: list[str],
    accent: tuple[int, int, int],
    highlight: str = "",
    highlight_detail: str = "",
    highlight_2: str = "",
    highlight_detail_2: str = "",
) -> np.ndarray:
    """Draw an explicit, evidence-labelled title or result card."""
    frame = np.full((height, width, 3), (14, 21, 32), dtype=np.uint8)
    cv2.rectangle(frame, (0, 0), (width, 7), accent, -1)
    cv2.rectangle(frame, (54, 88), (width - 54, height - 88), (21, 31, 46), -1)
    cv2.rectangle(frame, (54, 88), (width - 54, height - 88), accent, 2)
    draw_text(frame, "NEUROGRIP-X", (86, 145), 0.95, (245, 247, 250), 2)
    draw_text(frame, eyebrow.upper(), (86, 198), 0.52, accent, 2)
    title_scale = 1.02 if len(title) <= 34 else 0.82
    draw_text(frame, title, (86, 286), title_scale, (245, 247, 250), 2)
    baseline = 376
    for line in lines:
        draw_text(frame, line, (92, baseline), 0.58, (214, 224, 235), 1)
        baseline += 54
    if highlight:
        draw_text(
            frame,
            highlight,
            (86, baseline + 32),
            0.66,
            (245, 247, 250),
            2,
        )
    if highlight_detail:
        draw_text(
            frame,
            highlight_detail,
            (86, baseline + 68),
            0.40,
            accent,
            1,
        )
    if highlight_2:
        draw_text(
            frame,
            highlight_2,
            (86, baseline + 124),
            0.66,
            (245, 247, 250),
            2,
        )
    if highlight_detail_2:
        draw_text(
            frame,
            highlight_detail_2,
            (86, baseline + 160),
            0.40,
            accent,
            1,
        )
    draw_text(
        frame,
        "EUFS Sim 2 telemetry replay - simulation only",
        (86, height - 142),
        0.48,
        (159, 180, 203),
        1,
    )
    draw_text(
        frame,
        "Every path and metric is reconstructed from recorded episode telemetry.",
        (86, height - 110),
        0.42,
        (159, 180, 203),
        1,
    )
    return frame


def draw_cone(frame: np.ndarray, pixel: np.ndarray, color: tuple[int, int, int]) -> None:
    """Draw a small shaded cone rather than a bare track marker."""
    x, y = int(pixel[0]), int(pixel[1])
    triangle = np.asarray(
        [[x, y - 6], [x - 4, y + 5], [x + 4, y + 5]], dtype=np.int32
    )
    cv2.fillConvexPoly(frame, triangle, color, lineType=cv2.LINE_AA)
    cv2.polylines(frame, [triangle], True, (28, 30, 34), 1, cv2.LINE_AA)


def draw_formula_student_car(
    frame: np.ndarray,
    position: np.ndarray,
    yaw_rad: float,
    color: tuple[int, int, int],
    scale_px: float,
) -> None:
    """Draw a recognisable top-down Formula Student silhouette at one pose."""
    # x points forward in car coordinates, y points left.  The visual contains
    # front/rear wings, exposed wheels and a narrow monocoque.
    car = np.asarray([
        [1.35, 0.00], [0.98, 0.27], [0.47, 0.24], [0.23, 0.42],
        [-0.72, 0.42], [-0.92, 0.21], [-1.18, 0.21], [-1.18, -0.21],
        [-0.92, -0.21], [-0.72, -0.42], [0.23, -0.42], [0.47, -0.24],
        [0.98, -0.27],
    ]) * scale_px
    rotation = np.asarray(
        [
            [math.cos(yaw_rad), -math.sin(yaw_rad)],
            [math.sin(yaw_rad), math.cos(yaw_rad)],
        ]
    )
    # Pixel y points down, so transform world yaw into the image coordinate system.
    image_rotation = np.asarray(
        [
            [rotation[0, 0], rotation[0, 1]],
            [-rotation[1, 0], -rotation[1, 1]],
        ]
    )
    body = car @ image_rotation.T + position
    body = np.rint(body).astype(np.int32)
    cv2.fillConvexPoly(frame, body, color, lineType=cv2.LINE_AA)
    cv2.polylines(frame, [body], True, (240, 244, 248), 1, cv2.LINE_AA)
    # Wheels are rectangles, black with a light rim, positioned at the two axles.
    wheel_positions = (
        (0.62, 0.50),
        (0.62, -0.50),
        (-0.72, 0.50),
        (-0.72, -0.50),
    )
    for longitudinal, lateral in wheel_positions:
        wheel = np.asarray([
            [longitudinal + 0.20, lateral + 0.10], [longitudinal + 0.20, lateral - 0.10],
            [longitudinal - 0.20, lateral - 0.10], [longitudinal - 0.20, lateral + 0.10],
        ]) * scale_px
        wheel = np.rint(wheel @ image_rotation.T + position).astype(np.int32)
        cv2.fillConvexPoly(frame, wheel, (18, 20, 24), lineType=cv2.LINE_AA)
        cv2.polylines(frame, [wheel], True, (180, 190, 200), 1, cv2.LINE_AA)


def draw_panel(
    frame: np.ndarray,
    episode: Episode,
    sampled: dict[str, float],
    elapsed_s: float,
    color: tuple[int, int, int],
) -> None:
    """Draw status values measured from the CSV, never inferred from pixels."""
    height, width = frame.shape[:2]
    panel_right = width // 2 - 18
    vertical_offset = int(round(height * 0.12))
    cv2.rectangle(
        frame,
        (18, height - 142 - vertical_offset),
        (panel_right, height - 18 - vertical_offset),
        (20, 28, 42),
        -1,
    )
    cv2.rectangle(
        frame,
        (18, height - 142 - vertical_offset),
        (panel_right, height - 18 - vertical_offset),
        color,
        1,
    )
    draw_text(
        frame,
        episode.label,
        (34, height - 112 - vertical_offset),
        0.64,
        color,
        2,
    )
    draw_text(
        frame,
        f"t  {elapsed_s:4.1f} s",
        (34, height - 78 - vertical_offset),
        0.39, (235, 239, 245),
    )
    draw_text(
        frame, f"speed  {sampled['speed'] * 3.6:4.1f} km/h",
        (168, height - 78 - vertical_offset), 0.39, (235, 239, 245),
    )
    draw_text(
        frame, f"lateral error  {sampled['e_y'] * 100:+5.1f} cm",
        (34, height - 50 - vertical_offset), 0.39, (235, 239, 245),
    )
    draw_text(
        frame, f"grip  F {sampled['front']:.2f}  R {sampled['rear']:.2f}",
        (244, height - 50 - vertical_offset), 0.39, (235, 239, 245),
    )
    heading = f"heading error  {math.degrees(sampled['e_psi']):+4.1f} deg"
    draw_text(
        frame,
        heading,
        (34, height - 25 - vertical_offset),
        0.39,
        (235, 239, 245),
    )


def draw_interval_tower(
    frame: np.ndarray,
    left: Episode,
    right: Episode,
    left_leads: bool,
    gap_s: float,
    left_color: tuple[int, int, int],
    right_color: tuple[int, int, int],
) -> None:
    """Draw an F1-style interval left of centre and centred vertically."""
    height, width = frame.shape[:2]
    tower_width, tower_height = 360, 136
    centre_x = int(round(width * 0.45))
    centre_y = height // 2
    x0 = centre_x - tower_width // 2
    y0 = centre_y - tower_height // 2
    x1, y1 = x0 + tower_width, y0 + tower_height
    background = (16, 20, 28)
    row_background = (23, 29, 40)
    border = (74, 84, 99)
    cv2.rectangle(frame, (x0, y0), (x1, y1), background, -1)
    cv2.rectangle(frame, (x0, y0), (x1, y1), border, 1)
    draw_text(
        frame,
        "SAME-DISTANCE INTERVAL  |  1 Hz",
        (x0 + 15, y0 + 25),
        0.39,
        (176, 190, 207),
        1,
    )

    leader = left if left_leads else right
    trailer = right if left_leads else left
    leader_color = left_color if left_leads else right_color
    trailer_color = right_color if left_leads else left_color
    for row_index, (position, episode, color) in enumerate(
        ((1, leader, leader_color), (2, trailer, trailer_color))
    ):
        row_top = y0 + 34 + row_index * 48
        cv2.rectangle(
            frame,
            (x0 + 8, row_top),
            (x1 - 8, row_top + 42),
            row_background,
            -1,
        )
        cv2.rectangle(
            frame,
            (x0 + 8, row_top),
            (x0 + 48, row_top + 42),
            color,
            -1,
        )
        draw_text(
            frame,
            str(position),
            (x0 + 22, row_top + 29),
            0.58,
            (248, 249, 251),
            2,
        )
        draw_text(
            frame,
            episode.label,
            (x0 + 61, row_top + 29),
            0.53,
            (242, 245, 249),
            2,
        )
        if position == 2:
            draw_text(
                frame,
                f"+{gap_s:.3f} s",
                (x1 - 112, row_top + 29),
                0.48,
                (206, 215, 226),
                1,
            )


def draw_view(
    track: dict[str, np.ndarray],
    episode: Episode,
    at_s: float,
    dimensions: tuple[int, int],
    color: tuple[int, int, int],
) -> np.ndarray:
    """Render one controller in an independent full-track camera view."""
    width, height = dimensions
    frame = np.full((height, width, 3), (42, 82, 53), dtype=np.uint8)  # grass
    world_to_pixel, scale = layout_transform(track, width, height)
    left_x = track["x"] - track["left_width"] * np.sin(track["yaw"])
    left_y = track["y"] + track["left_width"] * np.cos(track["yaw"])
    right_x = track["x"] + track["right_width"] * np.sin(track["yaw"])
    right_y = track["y"] - track["right_width"] * np.cos(track["yaw"])
    left, right = world_to_pixel(left_x, left_y), world_to_pixel(right_x, right_y)
    surface = np.vstack((left, right[::-1]))
    cv2.fillPoly(frame, [surface], (49, 53, 61), lineType=cv2.LINE_AA)
    cv2.polylines(frame, [left], True, (220, 125, 35), 2, cv2.LINE_AA)
    cv2.polylines(frame, [right], True, (35, 220, 240), 2, cv2.LINE_AA)
    center = world_to_pixel(track["x"], track["y"])
    cv2.polylines(frame, [center], True, (145, 154, 165), 1, cv2.LINE_AA)
    stride = max(1, len(left) // 34)
    for point in left[::stride]:
        draw_cone(frame, point, (215, 105, 28))  # BGR: blue
    for point in right[::stride]:
        draw_cone(frame, point, (30, 210, 245))  # BGR: yellow

    x, y, yaw = reconstruct_pose(track, episode)
    sampled = interpolate_episode(episode, at_s)
    index = int(np.searchsorted(episode.time_s, at_s, side="right") - 1)
    index = int(np.clip(index, 0, len(episode.time_s) - 1))
    tail_start = max(0, index - 80)
    tail = world_to_pixel(x[tail_start:index + 1], y[tail_start:index + 1])
    cv2.rectangle(frame, (0, 0), (width, 65), (13, 20, 32), -1)
    draw_text(frame, episode.label, (18, 30), 0.65, (245, 247, 250), 2)
    draw_text(frame, "EUFS telemetry replay", (18, 54), 0.40, (169, 188, 210))
    draw_panel(frame, episode, sampled, at_s, color)
    # The vehicle and its recent path deliberately sit above the information
    # card: the telemetry stays visible, but the actual recorded car pose is
    # never hidden when it drives through the lower part of the map.
    if len(tail) > 1:
        cv2.polylines(frame, [tail], False, color, 2, cv2.LINE_AA)
    position = world_to_pixel(x[index], y[index])[0].astype(float)
    draw_formula_student_car(frame, position, float(yaw[index]), color, max(10.0, scale * 0.55))
    return frame


def encoder(output: Path, width: int, height: int, fps: int) -> subprocess.Popen:
    """Launch an H.264 encoder receiving BGR frames through stdin."""
    return subprocess.Popen(
        [
            "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo",
            "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-r", str(fps),
            "-i", "-", "-an", "-c:v", "libx264", "-preset", "medium",
            "-crf", "19", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            str(output),
        ],
        stdin=subprocess.PIPE,
    )


def main() -> int:
    """Create one side-by-side video from two genuinely recorded episodes."""
    arguments = parse_arguments()
    if arguments.fps < 10 or arguments.fps > 60:
        raise ValueError("--fps must be between 10 and 60")
    if arguments.playback_rate <= 0.0:
        raise ValueError("--playback-rate must be positive")
    if arguments.intro_duration_s < 0.0 or arguments.outro_duration_s < 0.0:
        raise ValueError("intro/outro duration must be non-negative")
    if arguments.intro_duration_s > 0.0 and not arguments.intro_title:
        raise ValueError("--intro-title is required when --intro-duration-s is positive")
    if arguments.outro_duration_s > 0.0 and not arguments.outro_title:
        raise ValueError("--outro-title is required when --outro-duration-s is positive")
    if arguments.width < 800 or arguments.height < 480 or arguments.width % 2:
        raise ValueError("width must be even and at least 800; height must be at least 480")
    output = arguments.output.expanduser().resolve()
    if output.exists() and not arguments.overwrite:
        raise FileExistsError(f"refusing to overwrite {output}; pass --overwrite")
    output.parent.mkdir(parents=True, exist_ok=True)
    track_file = arguments.track_npz.expanduser().resolve()
    track = {
        name: np.asarray(value, dtype=float)
        for name, value in np.load(track_file).items()
    }
    required_track = {"x", "y", "yaw", "left_width", "right_width"}
    if not required_track <= track.keys():
        missing = sorted(required_track - track.keys())
        raise ValueError(f"{track_file}: reference path lacks {missing}")
    left = load_episode(arguments.left_csv.expanduser().resolve(), arguments.left_label)
    right = load_episode(arguments.right_csv.expanduser().resolve(), arguments.right_label)
    left_timing = build_lap_timing(left)
    right_timing = build_lap_timing(right)
    # Continue until both cars finish. np.interp and the pose index clamp hold
    # the earlier finisher at the line, making the actual lap-time gap visible.
    duration = max(float(left.time_s[-1]), float(right.time_s[-1]))
    if arguments.max_duration_s > 0.0:
        duration = min(duration, arguments.max_duration_s)
    replay_frames = max(
        1, int(math.floor(duration * arguments.fps / arguments.playback_rate))
    )
    intro_frames = int(round(arguments.intro_duration_s * arguments.fps))
    outro_frames = int(round(arguments.outro_duration_s * arguments.fps))
    frames = intro_frames + replay_frames + outro_frames
    half_width, height = arguments.width // 2, arguments.height
    process = encoder(output, arguments.width, height, arguments.fps)
    try:
        assert process.stdin is not None
        for frame_index in range(frames):
            if frame_index < intro_frames:
                frame = draw_information_card(
                    arguments.width,
                    height,
                    "Context-conditioned physical dynamics for safe MPC",
                    arguments.intro_title,
                    [arguments.intro_subtitle] if arguments.intro_subtitle else [],
                    (218, 75, 199),
                )
            elif frame_index >= intro_frames + replay_frames:
                frame = draw_information_card(
                    arguments.width,
                    height,
                    "Frozen final-holdout result",
                    arguments.outro_title,
                    arguments.outro_line,
                    (93, 205, 232),
                    arguments.outro_highlight,
                    arguments.outro_highlight_detail,
                    arguments.outro_highlight_2,
                    arguments.outro_highlight_detail_2,
                )
            else:
                replay_index = frame_index - intro_frames
                at_s = min(
                    duration, replay_index * arguments.playback_rate / arguments.fps
                )
                left_frame = draw_view(
                    track, left, at_s, (half_width, height), (43, 88, 226)
                )
                right_frame = draw_view(
                    track, right, at_s, (half_width, height), (218, 75, 199)
                )
                frame = np.hstack((left_frame, right_frame))
                cv2.line(
                    frame, (half_width, 0), (half_width, height),
                    (18, 23, 34), 3, cv2.LINE_AA,
                )
                cv2.rectangle(frame, (0, 0), (arguments.width, 2), (93, 205, 232), -1)
                replay_video_s = replay_index / arguments.fps
                interval_second = math.floor(replay_video_s)
                interval_at_s = min(
                    duration, interval_second * arguments.playback_rate
                )
                left_leads, gap_s = same_distance_interval(
                    left,
                    right,
                    left_timing,
                    right_timing,
                    interval_at_s,
                )
                draw_interval_tower(
                    frame,
                    left,
                    right,
                    left_leads,
                    gap_s,
                    (43, 88, 226),
                    (218, 75, 199),
                )
            process.stdin.write(frame.tobytes())
    finally:
        if process.stdin is not None:
            process.stdin.close()
        return_code = process.wait()
    if return_code != 0 or not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError("ffmpeg failed to create the telemetry replay")
    print(
        f"TELEMETRY_REPLAY_CREATED output={output} frames={frames} "
        f"duration_s={frames / arguments.fps:.2f} source_duration_s={duration:.2f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
