#!/usr/bin/env python3
"""Record the numeric command-mux safety status for actuation provenance."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

STATUS_SCHEMA = 1
STATUS_LENGTH = 8
STATUS_FIELDS = [
    "schema_version",
    "time_s",
    "source_id",
    "python_age_s",
    "matlab_receipt_age_s",
    "matlab_control_age_s",
    "fallback_count",
    "reject_count",
]


def sha256_file(path: Path) -> str:
    """Return immutable output identity."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class CommandMuxStatusRecorder(Node):
    """Persist valid fixed-length mux status without publishing anything."""

    def __init__(self, output: Path):
        super().__init__("neurogrip_command_mux_status_recorder")
        self.output = output.expanduser().resolve()
        if self.output.exists():
            raise FileExistsError(
                f"refusing to overwrite mux trace: {self.output}"
            )
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.file_handle = self.output.open("x", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(
            self.file_handle, fieldnames=STATUS_FIELDS
        )
        self.writer.writeheader()
        self.rows = 0
        self.invalid_rows = 0
        self.create_subscription(
            Float64MultiArray,
            "/neurogrip/command_mux_status",
            self.on_status,
            100,
        )
        self.create_timer(1.0, self.file_handle.flush)

    def on_status(self, message: Float64MultiArray) -> None:
        """Write only a finite schema-1 status row."""
        values = [float(value) for value in message.data]
        valid = (
            len(values) == STATUS_LENGTH
            and int(values[0]) == STATUS_SCHEMA
            and all(math.isfinite(value) for value in values)
            and int(values[2]) in {0, 1, 2, 3, 4}
        )
        if not valid:
            self.invalid_rows += 1
            return
        self.writer.writerow(dict(zip(STATUS_FIELDS, values, strict=True)))
        self.rows += 1

    def close(self) -> None:
        """Close the CSV and write compact provenance metadata."""
        self.file_handle.flush()
        self.file_handle.close()
        metadata = {
            "schema_version": STATUS_SCHEMA,
            "topic": "/neurogrip/command_mux_status",
            "rows": self.rows,
            "invalid_rows": self.invalid_rows,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "output_sha256": sha256_file(self.output),
        }
        self.output.with_suffix(".metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def main() -> int:
    """Record until interrupted by the owning development run."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    arguments, ros_arguments = parser.parse_known_args()
    rclpy.init(args=ros_arguments)
    node = CommandMuxStatusRecorder(arguments.output)
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
