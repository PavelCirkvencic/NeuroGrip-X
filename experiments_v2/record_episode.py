#!/usr/bin/env python3
"""Record one traceable EUFS episode using measured state and actuation only."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from eufs_msgs.msg import ConeWithColorProbabilityArray, WheelSpeedsStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from sensor_msgs.msg import Imu
from std_msgs.msg import Float64MultiArray, String

from neurogrip_control.body_velocity import yaw_from_quaternion
from neurogrip_control.track_safety import TrackEnvelope

TRACKING_SCHEMA = 1
FIELDS = [
    "time_s", "x_m", "y_m", "yaw_rad", "v_x_mps", "v_y_mps",
    "yaw_rate_rps", "e_y_m", "e_psi_rad", "steering_applied_rad",
    "steering_safe_rad", "acceleration_command_mps2", "a_x_mps2",
    "a_y_mps2", "curvature_1pm", "d_kappa_rad_s", "lap_progress",
    "grip_front", "grip_rear", "left_width_m", "right_width_m",
    "boundary_margin_m", "cone_collision_count", "controller_solution_valid",
    "controller_solve_time_ms", "controller_max_slack_m",
    "controller_first_move_rad", "controller_boundary_margin_m",
    "controller_diagnostics_age_s", "experiment_started",
    "cumulative_progress", "target_speed_mps", "diagnostic_speed_mps",
    "profile_grip", "profile_lateral_utilisation", "estimated_front_grip",
    "estimated_rear_grip",
]


def sha256_file(path: Path) -> str:
    """Return the SHA-256 of a file, used for immutable run provenance."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def current_git_commit() -> str:
    """Return the source revision without making recorder startup brittle."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, check=False, text=True
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


class EpisodeRecorder(Node):
    """Write rows only when all core sampled signals are fresh and coherent."""

    def __init__(self, arguments: argparse.Namespace):
        super().__init__("neurogrip_episode_recorder")
        self.output = Path(arguments.output).expanduser().resolve()
        if self.output.exists():
            raise FileExistsError(f"refusing to overwrite existing episode: {self.output}")
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.scenario_id, self.controller, self.seed = (
            arguments.scenario_id,
            arguments.controller,
            arguments.seed,
        )
        self.provenance = self.load_provenance(arguments.provenance_file)
        self.track_path = Path(arguments.track_npz).expanduser().resolve()
        self.track_envelope = TrackEnvelope.from_npz(self.track_path)
        self.vehicle_half_width_m = float(arguments.vehicle_half_width_m)
        self.vehicle_half_length_m = float(arguments.vehicle_half_length_m)
        self.tracking: list[float] | None = None
        self.controller_diagnostics: list[float] | None = None
        self.safe_command: AckermannDriveStamped | None = None
        self.wheel_speeds: WheelSpeedsStamped | None = None
        self.imu: Imu | None = None
        self.grip_front: float | None = None
        self.grip_rear: float | None = None
        self.cone_collision_count = 0
        self.events_path = self.output.with_suffix(".events.jsonl")
        self.events_handle = self.events_path.open("x", encoding="utf-8")
        self.rows = 0
        self.skipped = {"missing": 0, "tracking_schema": 0, "tracking_stale": 0}
        self.first_sim_time_s: float | None = None
        self.last_sim_time_s: float | None = None
        self.writer_handle = self.output.open("x", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.writer_handle, fieldnames=FIELDS)
        self.writer.writeheader()

        self.create_subscription(Odometry, "/odom", self.on_odom, 50)
        self.create_subscription(
            WheelSpeedsStamped, "/ros_can/wheel_speeds", self.on_wheel_speeds, 50
        )
        self.create_subscription(Imu, "/imu/data", self.on_imu, 50)
        self.create_subscription(
            Float64MultiArray, "/neurogrip/tracking_state", self.on_tracking, 50
        )
        self.create_subscription(
            Float64MultiArray,
            "/neurogrip/controller_diagnostics",
            self.on_controller_diagnostics,
            50,
        )
        self.create_subscription(
            ConeWithColorProbabilityArray,
            "/plugin/cone_collision_tracker/colliding_cones",
            self.on_collisions,
            10,
        )
        self.create_subscription(
            AckermannDriveStamped, "/neurogrip/command_safe", self.on_safe_command, 50
        )
        status_qos = QoSProfile(depth=10, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(
            String, "/neurogrip/scenario_status", self.on_event, status_qos
        )
        self.create_subscription(
            String, "/neurogrip/controller_status", self.on_event, status_qos
        )
        self.create_subscription(
            String, "/neurogrip/actuation_status", self.on_event, 10
        )
        self.create_timer(1.0, self.flush)
        self.get_logger().info(
            f"recording {self.scenario_id} ({self.controller}) -> {self.output}"
        )

    @staticmethod
    def load_provenance(path: str) -> dict:
        """Load runner-created immutable provenance before simulation begins."""
        if not path:
            return {}
        provenance_path = Path(path).expanduser().resolve()
        if not provenance_path.is_file():
            raise FileNotFoundError(f"provenance file missing: {provenance_path}")
        return json.loads(provenance_path.read_text(encoding="utf-8"))

    @staticmethod
    def stamp_s(message) -> float:
        """Convert a standard ROS header stamp to seconds."""
        stamp = message.header.stamp
        return stamp.sec + stamp.nanosec * 1e-9

    def write_event(self, payload: dict) -> None:
        """Persist ordered status events independently of the 50 Hz telemetry."""
        payload = dict(payload)
        payload.setdefault("recorded_at_sim_s", self.get_clock().now().nanoseconds / 1e9)
        self.events_handle.write(json.dumps(payload, sort_keys=True) + "\n")
        self.events_handle.flush()

    def on_tracking(self, message: Float64MultiArray) -> None:
        """Cache only the documented v1 tracking contract."""
        values = list(message.data)
        if len(values) != 10 or int(values[0]) != TRACKING_SCHEMA:
            self.tracking = None
            self.skipped["tracking_schema"] += 1
            return
        self.tracking = values

    def on_safe_command(self, message: AckermannDriveStamped) -> None:
        """Cache commanded-safe acceleration separately from measured steering."""
        self.safe_command = message

    def on_controller_diagnostics(self, message: Float64MultiArray) -> None:
        """Cache documented legacy or physics-profile MPC diagnostics."""
        values = list(message.data)
        if len(values) == 9 and int(values[0]) == 1:
            self.controller_diagnostics = [*values, *([-1.0] * 6)]
        elif len(values) == 15 and int(values[0]) == 2:
            self.controller_diagnostics = values

    def on_collisions(self, message: ConeWithColorProbabilityArray) -> None:
        """Record the authoritative current EUFS collision count."""
        self.cone_collision_count = len(message.cones)

    def on_wheel_speeds(self, message: WheelSpeedsStamped) -> None:
        """Cache EUFS patched wheel telemetry containing actual steering."""
        self.wheel_speeds = message

    def on_imu(self, message: Imu) -> None:
        """Cache acceleration measurements used by the causal AI history."""
        self.imu = message

    def on_event(self, message: String) -> None:
        """Persist scenario and controller events; update confirmed grip only."""
        try:
            payload = json.loads(message.data)
        except json.JSONDecodeError:
            self.write_event({"event": "invalid_json", "raw": message.data})
            return
        self.write_event(payload)
        if payload.get("confirmed"):
            self.grip_front = payload.get("readback_front")
            self.grip_rear = payload.get("readback_rear")

    def on_odom(self, message: Odometry) -> None:
        """Write one row per coherent odometry timestamp."""
        if (
            self.tracking is None
            or self.safe_command is None
            or self.wheel_speeds is None
            or self.imu is None
        ):
            self.skipped["missing"] += 1
            return
        time_s = self.stamp_s(message)
        tracking_time_s = float(self.tracking[1])
        if abs(time_s - tracking_time_s) > 0.04:
            self.skipped["tracking_stale"] += 1
            return
        pose, twist = message.pose.pose, message.twist.twist
        yaw = yaw_from_quaternion(
            pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w
        )
        left_width, right_width, boundary_margin = self.track_envelope.boundary_margin(
            float(self.tracking[9]),
            float(self.tracking[2]),
            float(self.tracking[3]),
            self.vehicle_half_width_m,
            self.vehicle_half_length_m,
        )
        diagnostics = self.controller_diagnostics
        if diagnostics is None:
            diagnostics = [
                2.0, time_s, -1.0, -1.0, -1.0, 0.0, boundary_margin,
                0.0, 0.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
            ]
        diagnostics_age = max(0.0, time_s - float(diagnostics[1]))
        self.writer.writerow(
            {
                "time_s": time_s,
                "x_m": pose.position.x,
                "y_m": pose.position.y,
                "yaw_rad": yaw,
                "v_x_mps": twist.linear.x,
                "v_y_mps": twist.linear.y,
                "yaw_rate_rps": twist.angular.z,
                "e_y_m": self.tracking[2],
                "e_psi_rad": self.tracking[3],
                "steering_applied_rad": self.wheel_speeds.speeds.steering,
                "steering_safe_rad": self.safe_command.drive.steering_angle,
                "acceleration_command_mps2": self.safe_command.drive.acceleration,
                "a_x_mps2": self.imu.linear_acceleration.x,
                "a_y_mps2": self.imu.linear_acceleration.y,
                "curvature_1pm": self.tracking[7],
                "d_kappa_rad_s": self.tracking[8],
                "lap_progress": self.tracking[9],
                "grip_front": self.grip_front,
                "grip_rear": self.grip_rear,
                "left_width_m": left_width,
                "right_width_m": right_width,
                "boundary_margin_m": boundary_margin,
                "cone_collision_count": self.cone_collision_count,
                "controller_solution_valid": diagnostics[2],
                "controller_solve_time_ms": diagnostics[3],
                "controller_max_slack_m": diagnostics[4],
                "controller_first_move_rad": diagnostics[5],
                "controller_boundary_margin_m": diagnostics[6],
                "controller_diagnostics_age_s": diagnostics_age,
                "experiment_started": diagnostics[7],
                "cumulative_progress": diagnostics[8],
                "target_speed_mps": diagnostics[9],
                "diagnostic_speed_mps": diagnostics[10],
                "profile_grip": diagnostics[11],
                "profile_lateral_utilisation": diagnostics[12],
                "estimated_front_grip": diagnostics[13],
                "estimated_rear_grip": diagnostics[14],
            }
        )
        self.rows += 1
        self.first_sim_time_s = time_s if self.first_sim_time_s is None else self.first_sim_time_s
        self.last_sim_time_s = time_s

    def flush(self) -> None:
        """Flush both append-only outputs while the run is in progress."""
        self.writer_handle.flush()
        self.events_handle.flush()

    def close(self) -> None:
        """Close all handles and write final metadata without guessing success."""
        self.flush()
        self.writer_handle.close()
        self.events_handle.close()
        metadata = {
            "schema_version": 4,
            "scenario_id": self.scenario_id,
            "controller": self.controller,
            "seed": self.seed,
            "rows": self.rows,
            "skipped_rows": self.skipped,
            "first_sim_time_s": self.first_sim_time_s,
            "last_sim_time_s": self.last_sim_time_s,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": current_git_commit(),
            "output": str(self.output),
            "output_sha256": sha256_file(self.output),
            "events": str(self.events_path),
            "events_sha256": sha256_file(self.events_path),
            "track_npz": str(self.track_path),
            "track_npz_sha256": sha256_file(self.track_path),
            "vehicle_half_width_m": self.vehicle_half_width_m,
            "vehicle_half_length_m": self.vehicle_half_length_m,
            "provenance": self.provenance,
        }
        self.output.with_suffix(".metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


def main() -> int:
    """Run the recorder until the owning episode runner stops it."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--scenario-id", required=True)
    parser.add_argument("--controller", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--provenance-file", default="")
    parser.add_argument("--track-npz", required=True)
    parser.add_argument("--vehicle-half-width-m", type=float, default=0.70)
    parser.add_argument("--vehicle-half-length-m", type=float, default=1.30)
    arguments, ros_arguments = parser.parse_known_args()
    rclpy.init(args=ros_arguments)
    node = EpisodeRecorder(arguments)
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
