"""Expose EUFS measured steering through a MATLAB-compatible ROS contract."""

from __future__ import annotations

import math

import rclpy
from eufs_msgs.msg import WheelSpeedsStamped
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

APPLIED_STEERING_SCHEMA = 1.0


def stamp_s(message: WheelSpeedsStamped) -> float:
    """Return the source simulation timestamp in seconds."""
    stamp = message.header.stamp
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def encode_applied_steering(message: WheelSpeedsStamped) -> list[float]:
    """
    Encode one finite wheel telemetry message in the versioned flat schema.

    Wire order: ``[schema, sim_time_s, measured_steering_rad]``.  This is a
    diagnostic, one-way compatibility topic: its source is wheel telemetry,
    never a command topic, and it is deliberately not an actuator interface.
    """
    time_s = stamp_s(message)
    steering_rad = float(message.speeds.steering)
    if not math.isfinite(time_s) or time_s < 0.0:
        raise ValueError("wheel telemetry has an invalid source timestamp")
    if not math.isfinite(steering_rad):
        raise ValueError("wheel telemetry has a non-finite steering angle")
    return [APPLIED_STEERING_SCHEMA, time_s, steering_rad]


class AppliedSteeringBridge(Node):
    """Publish measured steering as a standard ROS message for MATLAB only."""

    def __init__(self) -> None:
        super().__init__("neurogrip_applied_steering_bridge")
        self.declare_parameter("input_topic", "/ros_can/wheel_speeds")
        self.declare_parameter("output_topic", "/neurogrip/applied_steering_flat")
        self.publisher = self.create_publisher(
            Float64MultiArray, self.get_parameter("output_topic").value, 50
        )
        self.create_subscription(
            WheelSpeedsStamped,
            self.get_parameter("input_topic").value,
            self.on_wheel_speeds,
            50,
        )

    def on_wheel_speeds(self, message: WheelSpeedsStamped) -> None:
        """Relay valid measured steering without modifying vehicle state."""
        try:
            encoded = encode_applied_steering(message)
        except ValueError as error:
            self.get_logger().warning(f"dropping invalid measured steering: {error}")
            return
        output = Float64MultiArray()
        output.data = encoded
        self.publisher.publish(output)


def main(args=None) -> None:
    """Run the read-only wheel telemetry bridge."""
    rclpy.init(args=args)
    node = AppliedSteeringBridge()
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
