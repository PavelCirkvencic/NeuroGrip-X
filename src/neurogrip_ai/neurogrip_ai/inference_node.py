"""
Publish learned state predictions and health diagnostics only.

This node is intentionally non-actuating: it never publishes a command, never
writes to ``cmd_vel`` and cannot bypass the command guard.  It exists to prove
that a trained fixed model can run in the ROS graph inside a real-time budget
before any context-adaptive or controller work begins.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import rclpy
from geometry_msgs.msg import Twist, TwistStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import String

from neurogrip_ai.model_runner import TransitionModelRunner


class NeuroGripInferenceNode(Node):
    """Run the fixed MLP baseline as a read-only ROS observer."""

    def __init__(self):
        super().__init__("neurogrip_inference")

        self.declare_parameter("checkpoint_path", "")
        self.declare_parameter("vehicle_name", "vehicle_blue")
        self.declare_parameter("publish_rate_hz", 20.0)
        self.declare_parameter("max_input_age_s", 0.5)
        self.declare_parameter("device", "cpu")

        checkpoint_path = self.get_parameter("checkpoint_path").value
        vehicle_name = self.get_parameter("vehicle_name").value
        publish_rate_hz = float(self.get_parameter("publish_rate_hz").value)
        self.max_input_age_s = float(self.get_parameter("max_input_age_s").value)
        device = self.get_parameter("device").value

        if not checkpoint_path:
            raise ValueError("checkpoint_path parameter is required.")
        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint does not exist: {self.checkpoint_path}")

        self.runner = TransitionModelRunner.from_checkpoint(
            self.checkpoint_path, device=device
        )

        self.latest_odometry: Odometry | None = None
        self.latest_command: Twist | None = None
        self.odometry_received_at: float | None = None
        self.command_received_at: float | None = None

        self.prediction_publisher = self.create_publisher(
            TwistStamped, "/neurogrip/predicted_state", 10
        )
        self.health_publisher = self.create_publisher(
            String, "/neurogrip/ai_health", 10
        )

        self.create_subscription(
            Odometry,
            f"/model/{vehicle_name}/odometry",
            self.odometry_callback,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Twist,
            f"/model/{vehicle_name}/cmd_vel",
            self.command_callback,
            10,
        )
        self.create_timer(1.0 / publish_rate_hz, self.infer)

        self.get_logger().info(
            f"Loaded fixed MLP baseline from {self.checkpoint_path} on {device}."
        )

    def odometry_callback(self, message: Odometry) -> None:
        """Cache the latest odometry message and its receipt time."""
        self.latest_odometry = message
        self.odometry_received_at = time.monotonic()

    def command_callback(self, message: Twist) -> None:
        """Cache the latest applied command and its receipt time."""
        self.latest_command = message
        self.command_received_at = time.monotonic()

    def inputs_are_fresh(self, now: float) -> bool:
        """Return whether both odometry and command are recent enough."""
        if self.latest_odometry is None or self.latest_command is None:
            return False
        if self.odometry_received_at is None or self.command_received_at is None:
            return False
        return (
            now - self.odometry_received_at <= self.max_input_age_s
            and now - self.command_received_at <= self.max_input_age_s
        )

    def publish_health(self, healthy: bool, detail: str, latency_ms: float) -> None:
        """Publish a compact JSON health diagnostic."""
        message = String()
        message.data = json.dumps(
            {
                "healthy": healthy,
                "detail": detail,
                "latency_ms": round(latency_ms, 4),
                "checkpoint": self.checkpoint_path.name,
                "model": "fixed_mlp_state_transition",
            }
        )
        self.health_publisher.publish(message)

    def infer(self) -> None:
        """Predict the next body-frame state; never publish a command."""
        now = time.monotonic()
        if not self.inputs_are_fresh(now):
            self.publish_health(False, "stale_or_missing_inputs", 0.0)
            return

        odometry = self.latest_odometry
        command = self.latest_command
        twist = odometry.twist.twist
        started = time.perf_counter()
        predicted_v_x, predicted_yaw_rate = self.runner.predict_state(
            twist.linear.x,
            twist.angular.z,
            command.linear.x,
            command.angular.z,
        )
        latency_ms = (time.perf_counter() - started) * 1_000.0

        message = TwistStamped()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = "base_link"
        message.twist.linear.x = predicted_v_x
        # Lateral velocity is not modeled (degenerate on the low-speed plant).
        message.twist.linear.y = 0.0
        message.twist.angular.z = predicted_yaw_rate
        self.prediction_publisher.publish(message)
        self.publish_health(True, "ok", latency_ms)


def main(args=None):
    """Run the read-only NeuroGrip-X inference node."""
    rclpy.init(args=args)
    node = NeuroGripInferenceNode()

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
