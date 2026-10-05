"""
Relay Gazebo IMU through a deterministic corruption model.

Topic contract:

* ``ros_gz_bridge`` publishes the raw Gazebo IMU on ``/model/<vehicle>/imu``.
* this relay subscribes to that topic and publishes the corrupted stream on
  ``/model/<vehicle>/imu_filtered``;
* ``state_logger`` defaults to reading ``/model/<vehicle>/imu_filtered``.

When no scenario noise is configured the relay is a transparent pass-through,
so the rest of the pipeline is unchanged.
"""

from __future__ import annotations

from pathlib import Path

import rclpy
import yaml
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu

from neurogrip_control.imu_corruption import ImuCorruption


class ImuRelay(Node):
    """Republish IMU messages with optional deterministic corruption."""

    def __init__(self):
        super().__init__("imu_relay")

        self.declare_parameter("vehicle_name", "vehicle_blue")
        self.declare_parameter("input_topic", "")
        self.declare_parameter("output_topic", "")
        self.declare_parameter("scenario_manifest_path", "")

        vehicle_name = self.get_parameter("vehicle_name").value
        input_topic = self.get_parameter("input_topic").value or (
            f"/model/{vehicle_name}/imu"
        )
        output_topic = self.get_parameter("output_topic").value or (
            f"/model/{vehicle_name}/imu_filtered"
        )
        manifest_path = self.get_parameter("scenario_manifest_path").value

        imu_noise, seed = self.load_noise_config(manifest_path)
        self.corruption = ImuCorruption.from_manifest(imu_noise, seed)
        if self.corruption.enabled:
            self.get_logger().info(
                "IMU corruption active: "
                f"angular_stddev={self.corruption.angular_velocity_stddev_rps:.4f}rad/s "
                f"linear_stddev={self.corruption.linear_acceleration_stddev_mps2:.4f}m/s^2 "
                f"dropout={self.corruption.dropout_probability:.3f} seed={seed}."
            )
        else:
            self.get_logger().info("IMU corruption disabled; relaying unchanged.")

        self.publisher = self.create_publisher(Imu, output_topic, qos_profile_sensor_data)
        self.create_subscription(
            Imu, input_topic, self.on_imu, qos_profile_sensor_data
        )
        self.dropped = 0

    @staticmethod
    def load_noise_config(manifest_path: str) -> tuple[dict, int]:
        """Return (imu_noise mapping, scenario seed) from a scenario manifest."""
        if not manifest_path:
            return {}, 0
        path = Path(manifest_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Scenario manifest does not exist: {path}")
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        scenario = document.get("scenario", {}) if isinstance(document, dict) else {}
        return scenario.get("imu_noise") or {}, int(scenario.get("seed", 0))

    def on_imu(self, message: Imu) -> None:
        """Apply corruption and republish, or drop the sample."""
        if self.corruption.dropout():
            self.dropped += 1
            return
        angular, linear = self.corruption.corrupt_vectors(
            (
                message.angular_velocity.x,
                message.angular_velocity.y,
                message.angular_velocity.z,
            ),
            (
                message.linear_acceleration.x,
                message.linear_acceleration.y,
                message.linear_acceleration.z,
            ),
        )
        (
            message.angular_velocity.x,
            message.angular_velocity.y,
            message.angular_velocity.z,
        ) = angular
        (
            message.linear_acceleration.x,
            message.linear_acceleration.y,
            message.linear_acceleration.z,
        ) = linear
        self.publisher.publish(message)


def main(args=None):
    """Run the IMU relay."""
    rclpy.init(args=args)
    node = ImuRelay()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
