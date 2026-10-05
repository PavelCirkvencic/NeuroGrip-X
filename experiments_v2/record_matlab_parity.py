#!/usr/bin/env python3
"""Record exact read-only C2 controller decisions for MATLAB replay."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

FRAME_SCHEMA = 3
FRAME_LENGTH = 155


def matrix_fields(prefix: str, rows: int, columns: int) -> list[str]:
    """Return row-major matrix column names matching NumPy ``ravel``."""
    return [
        f"{prefix}_{row + 1}{column + 1}"
        for row in range(rows)
        for column in range(columns)
    ]


FRAME_FIELDS = [
    "schema_version",
    "control_time_s",
    "sample_time_s",
    "raw_front_grip",
    "raw_rear_grip",
    "raw_front_grip_std",
    "raw_rear_grip_std",
    "raw_front_grip_error_q90",
    "raw_rear_grip_error_q90",
    "filtered_front_grip",
    "filtered_rear_grip",
    "held_front_grip_std",
    "held_rear_grip_std",
    "held_front_grip_error_q90",
    "held_rear_grip_error_q90",
    "conservative_front_grip",
    "conservative_rear_grip",
    "effective_grip",
    "profile_utilisation",
    "grip_error_margin_scale",
    "target_speed_mps",
    "v_x_mps",
    "lap_progress",
    "maximum_speed_mps",
    "minimum_speed_mps",
    "acceleration_limit_mps2",
    "braking_limit_mps2",
    "model_blend",
    "previous_steering_rad",
    "e_y_m",
    "e_psi_rad",
    "v_y_mps",
    "yaw_rate_rps",
    *matrix_fields("raw_a", 2, 2),
    *matrix_fields("raw_b", 2, 1),
    *matrix_fields("tracking_a", 4, 4),
    *matrix_fields("tracking_b", 4, 2),
    *[f"curvature_preview_{index:02d}_1pm" for index in range(30)],
    *[f"left_corridor_{index:02d}_m" for index in range(30)],
    *[f"right_corridor_{index:02d}_m" for index in range(30)],
    "first_move_rad",
    "solution_valid",
]
if len(FRAME_FIELDS) != FRAME_LENGTH:
    raise RuntimeError("MATLAB parity recorder field contract has wrong length")


def sha256_file(path: Path) -> str:
    """Return immutable capture identity."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def current_git_commit() -> str:
    """Return source revision without making recorder shutdown brittle."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, check=False, text=True
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


class MatlabParityRecorder(Node):
    """Persist the versioned parity topic without publishing any command."""

    def __init__(self, output: Path):
        super().__init__("neurogrip_matlab_parity_recorder")
        self.output = output.expanduser().resolve()
        if self.output.exists():
            raise FileExistsError(f"refusing to overwrite parity capture: {self.output}")
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.file_handle = self.output.open("x", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.file_handle, fieldnames=FRAME_FIELDS)
        self.writer.writeheader()
        self.rows = 0
        self.invalid_frames = 0
        self.first_control_time_s: float | None = None
        self.last_control_time_s: float | None = None
        self.create_subscription(
            Float64MultiArray,
            "/neurogrip/matlab_parity_frame",
            self.on_frame,
            50,
        )
        self.create_timer(1.0, self.file_handle.flush)
        self.get_logger().info(f"recording MATLAB parity frames -> {self.output}")

    def on_frame(self, message: Float64MultiArray) -> None:
        """Accept only the finite, fixed-length schema-3 frame."""
        values = [float(value) for value in message.data]
        if (
            len(values) != FRAME_LENGTH
            or int(values[0]) != FRAME_SCHEMA
            or not all(math.isfinite(value) for value in values)
        ):
            self.invalid_frames += 1
            return
        self.writer.writerow(dict(zip(FRAME_FIELDS, values, strict=True)))
        self.rows += 1
        control_time_s = values[1]
        if self.first_control_time_s is None:
            self.first_control_time_s = control_time_s
        self.last_control_time_s = control_time_s

    def close(self) -> None:
        """Close the append-only CSV and write its provenance sidecar."""
        self.file_handle.flush()
        self.file_handle.close()
        metadata = {
            "schema_version": FRAME_SCHEMA,
            "frame_length": FRAME_LENGTH,
            "topic": "/neurogrip/matlab_parity_frame",
            "read_only": True,
            "rows": self.rows,
            "invalid_frames": self.invalid_frames,
            "first_control_time_s": self.first_control_time_s,
            "last_control_time_s": self.last_control_time_s,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": current_git_commit(),
            "output_sha256": sha256_file(self.output),
        }
        self.output.with_suffix(".metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


def main() -> int:
    """Record until interrupted by the owning development episode."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    arguments, ros_arguments = parser.parse_known_args()
    rclpy.init(args=ros_arguments)
    node = MatlabParityRecorder(arguments.output)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
