"""Record vehicle state, IMU data, and commands into a raw CSV dataset."""

import csv
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import rclpy
import yaml
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu
from std_msgs.msg import String

from neurogrip_control.body_velocity import (
    CausalBodyVelocityEstimator,
    yaw_from_quaternion,
)


class StateLogger(Node):
    """Logs the latest odometry, IMU, and command into a CSV file."""

    def __init__(self):
        super().__init__("state_logger")

        self.declare_parameter("vehicle_name", "vehicle_blue")
        self.declare_parameter("output_dir", "data/raw")
        self.declare_parameter("sample_rate_hz", 50.0)
        self.declare_parameter("stop_on_completion", True)
        self.declare_parameter("scenario_manifest_path", "")
        self.declare_parameter("require_single_odom_publisher", True)
        self.declare_parameter("excitation_profile", "unknown")
        self.declare_parameter("imu_topic", "")

        vehicle_name = self.get_parameter("vehicle_name").value
        output_dir = Path(self.get_parameter("output_dir").value).expanduser()
        sample_rate_hz = self.get_parameter("sample_rate_hz").value
        self.stop_on_completion = self.get_parameter("stop_on_completion").value
        scenario_manifest_path = self.get_parameter("scenario_manifest_path").value
        self.require_single_odom_publisher = self.get_parameter(
            "require_single_odom_publisher"
        ).value
        self.excitation_profile = self.get_parameter("excitation_profile").value
        configured_imu_topic = self.get_parameter("imu_topic").value
        self.imu_topic = configured_imu_topic or (
            f"/model/{vehicle_name}/imu_filtered"
        )

        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.output_path = output_dir / f"run_{timestamp}.csv"
        self.metadata_path = output_dir / f"run_{timestamp}.metadata.json"
        self.recording_started_at_utc = datetime.now(timezone.utc).isoformat()
        self.vehicle_name = vehicle_name
        self.odometry_topic = f"/model/{vehicle_name}/odometry"
        self.sample_rate_hz = sample_rate_hz
        self.scenario_manifest = self.load_scenario_manifest(scenario_manifest_path)
        self.git_commit = self.read_git_commit()

        self.file = self.output_path.open("w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(
            self.file,
            fieldnames=[
                "time_s",
                "odom_imu_dt_s",
                "cmd_age_s",
                "x_m",
                "y_m",
                "z_m",
                "qx",
                "qy",
                "qz",
                "qw",
                "v_x_mps",
                "v_y_mps",
                "v_z_mps",
                "body_v_x_mps",
                "body_v_y_mps",
                "yaw_rate_rps",
                "imu_w_x_rps",
                "imu_w_y_rps",
                "imu_w_z_rps",
                "imu_a_x_mps2",
                "imu_a_y_mps2",
                "imu_a_z_mps2",
                "cmd_linear_x_mps",
                "cmd_angular_z_rps",
                "excitation_phase",
            ],
        )
        self.writer.writeheader()

        self.latest_odom = None
        self.latest_imu = None
        self.latest_command = Twist()
        self.last_command_received_ns = None
        self.last_written_odom_stamp_ns = None
        self.dropped_out_of_order_samples = 0
        self.row_count = 0
        self.current_phase = "unlabeled"
        self.recording_complete = False
        self.stop_reason = "interrupted_or_manual_stop"
        self.grip_transition_status = None
        self.body_velocity_estimator = CausalBodyVelocityEstimator()

        self.create_subscription(
            Odometry,
            self.odometry_topic,
            self.odom_callback,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Imu,
            self.imu_topic,
            self.imu_callback,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Twist,
            f"/model/{vehicle_name}/cmd_vel",
            self.command_callback,
            10,
        )
        self.create_subscription(
            String,
            "/neurogrip/excitation_phase",
            self.phase_callback,
            10,
        )
        self.create_subscription(
            String,
            "/neurogrip/grip_transition_status",
            self.grip_status_callback,
            10,
        )

        self.create_timer(1.0 / sample_rate_hz, self.write_sample)

        self.get_logger().info(
            f"Recording {sample_rate_hz:.1f} Hz data to {self.output_path}"
        )
        if self.scenario_manifest is None:
            self.get_logger().warning(
                "No scenario manifest attached. This run is suitable for "
                "debugging, but not a leakage-safe benchmark dataset."
            )
        else:
            self.get_logger().info(
                "Scenario provenance: "
                f"{self.scenario_manifest['scenario_id']} "
                f"({self.scenario_manifest['sha256'][:12]}...)."
            )

    @staticmethod
    def read_git_commit():
        """Return the current Git commit when the logger starts in a repository."""
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            check=False,
            text=True,
            timeout=2,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"

    @staticmethod
    def load_scenario_manifest(manifest_path_value):
        """Load scenario provenance without exposing its labels as ROS data."""
        if not manifest_path_value:
            return None

        manifest_path = Path(manifest_path_value).expanduser().resolve()
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Scenario manifest does not exist: {manifest_path}"
            )

        content = manifest_path.read_bytes()
        document = yaml.safe_load(content)
        if not isinstance(document, dict):
            raise ValueError(f"Scenario manifest is not a YAML mapping: {manifest_path}")
        scenario_id = document.get("scenario_id")
        scenario = document.get("scenario")
        if not isinstance(scenario_id, str) or not isinstance(scenario, dict):
            raise ValueError(
                "Scenario manifest must contain string scenario_id and mapping scenario."
            )

        return {
            "path": str(manifest_path),
            "sha256": hashlib.sha256(content).hexdigest(),
            "scenario_id": scenario_id,
            "scenario": scenario,
            "label_visibility": document.get("label_visibility", "unspecified"),
        }

    def odom_callback(self, message):
        """Cache the latest Gazebo odometry message."""
        self.latest_odom = message

    def imu_callback(self, message):
        """Cache the latest IMU message."""
        self.latest_imu = message

    def command_callback(self, message):
        """Cache the latest command and its receipt time."""
        self.latest_command = message
        self.last_command_received_ns = self.get_clock().now().nanoseconds

    def grip_status_callback(self, message):
        """Cache the grip transition provenance published by the scheduler."""
        try:
            self.grip_transition_status = json.loads(message.data)
        except json.JSONDecodeError:
            self.grip_transition_status = {
                "expected": True,
                "applied": False,
                "detail": "invalid_status_json",
            }
            self.get_logger().error("Grip transition status was not valid JSON.")

    def grip_transition_expected(self):
        """Return whether the attached scenario defines a grip transition."""
        if not isinstance(self.scenario_manifest, dict):
            return False
        scenario = self.scenario_manifest.get("scenario")
        return isinstance(scenario, dict) and isinstance(
            scenario.get("grip_transition"), dict
        )

    def grip_transition_applied(self):
        """Return whether a successful transition was actually confirmed."""
        status = self.grip_transition_status
        return bool(isinstance(status, dict) and status.get("applied") is True)

    def phase_callback(self, message):
        """Track automated-driver phase labels for later dataset filtering."""
        self.current_phase = message.data
        if self.stop_on_completion and message.data == "complete":
            self.recording_complete = True
            if self.grip_transition_expected() and not self.grip_transition_applied():
                self.stop_reason = "grip_transition_failed"
                self.get_logger().error(
                    "Scenario expected a grip transition but it was not applied; "
                    "marking the run as failed."
                )
            else:
                self.stop_reason = "excitation_complete"
                self.get_logger().info("Excitation completed; closing the recording.")

    @staticmethod
    def stamp_to_ns(stamp):
        """Convert a ROS time stamp into nanoseconds."""
        return stamp.sec * 1_000_000_000 + stamp.nanosec

    def write_sample(self):
        """Write one row for every new odometry sample."""
        if self.latest_odom is None or self.latest_imu is None:
            return

        if self.require_single_odom_publisher:
            publisher_count = len(
                self.get_publishers_info_by_topic(self.odometry_topic)
            )
            if publisher_count != 1:
                self.stop_reason = f"invalid_odometry_publisher_count_{publisher_count}"
                self.recording_complete = True
                self.get_logger().error(
                    "Refusing to record mixed simulator data: "
                    f"{self.odometry_topic} has {publisher_count} publishers."
                )
                return

        odom_stamp_ns = self.stamp_to_ns(self.latest_odom.header.stamp)
        imu_stamp_ns = self.stamp_to_ns(self.latest_imu.header.stamp)

        if self.last_written_odom_stamp_ns is not None:
            if odom_stamp_ns == self.last_written_odom_stamp_ns:
                return
            if odom_stamp_ns < self.last_written_odom_stamp_ns:
                self.dropped_out_of_order_samples += 1
                if self.dropped_out_of_order_samples % 100 == 1:
                    self.get_logger().warning(
                        "Dropping out-of-order odometry samples; "
                        "check that only one simulator is publishing."
                    )
                return

        self.last_written_odom_stamp_ns = odom_stamp_ns

        if self.last_command_received_ns is None:
            cmd_age_s = -1.0
        else:
            cmd_age_s = max(
                0.0,
                (self.get_clock().now().nanoseconds - self.last_command_received_ns)
                / 1_000_000_000.0,
            )

        pose = self.latest_odom.pose.pose
        twist = self.latest_odom.twist.twist
        imu = self.latest_imu

        yaw_rad = yaw_from_quaternion(
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        )
        body_v_x_mps, body_v_y_mps = self.body_velocity_estimator.update(
            odom_stamp_ns / 1_000_000_000.0,
            pose.position.x,
            pose.position.y,
            yaw_rad,
        )

        self.writer.writerow(
            {
                "time_s": odom_stamp_ns / 1_000_000_000.0,
                "odom_imu_dt_s": (odom_stamp_ns - imu_stamp_ns) / 1_000_000_000.0,
                "cmd_age_s": cmd_age_s,
                "x_m": pose.position.x,
                "y_m": pose.position.y,
                "z_m": pose.position.z,
                "qx": pose.orientation.x,
                "qy": pose.orientation.y,
                "qz": pose.orientation.z,
                "qw": pose.orientation.w,
                "v_x_mps": twist.linear.x,
                "v_y_mps": twist.linear.y,
                "v_z_mps": twist.linear.z,
                "body_v_x_mps": "" if body_v_x_mps is None else body_v_x_mps,
                "body_v_y_mps": "" if body_v_y_mps is None else body_v_y_mps,
                "yaw_rate_rps": twist.angular.z,
                "imu_w_x_rps": imu.angular_velocity.x,
                "imu_w_y_rps": imu.angular_velocity.y,
                "imu_w_z_rps": imu.angular_velocity.z,
                "imu_a_x_mps2": imu.linear_acceleration.x,
                "imu_a_y_mps2": imu.linear_acceleration.y,
                "imu_a_z_mps2": imu.linear_acceleration.z,
                "cmd_linear_x_mps": self.latest_command.linear.x,
                "cmd_angular_z_rps": self.latest_command.angular.z,
                "excitation_phase": self.current_phase,
            }
        )

        self.row_count += 1
        if self.row_count % 50 == 0:
            self.file.flush()

        if self.recording_complete:
            self.file.flush()

    def destroy_node(self):
        """Flush and close the CSV file when the node stops."""
        if not self.file.closed:
            self.file.flush()
            self.file.close()

        metadata = {
            "schema_version": 1,
            "csv_path": str(self.output_path),
            "vehicle_name": self.vehicle_name,
            "sample_rate_hz": self.sample_rate_hz,
            "rows": self.row_count,
            "dropped_out_of_order_samples": self.dropped_out_of_order_samples,
            "recording_started_at_utc": self.recording_started_at_utc,
            "recording_finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "termination_reason": self.stop_reason,
            "require_single_odom_publisher": self.require_single_odom_publisher,
            "excitation_profile": self.excitation_profile,
            "git_commit": self.git_commit,
            "scenario_manifest": self.scenario_manifest,
            "grip_transition_status": self.grip_transition_status,
        }
        with self.metadata_path.open("w", encoding="utf-8") as metadata_file:
            json.dump(metadata, metadata_file, indent=2)
            metadata_file.write("\n")

        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = StateLogger()

    try:
        while rclpy.ok() and not node.recording_complete:
            rclpy.spin_once(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
