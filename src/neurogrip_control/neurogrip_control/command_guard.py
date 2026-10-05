"""Bound and watchdog vehicle commands before they reach Gazebo."""

from __future__ import annotations

from pathlib import Path

import rclpy
import yaml
from geometry_msgs.msg import Twist
from rclpy.node import Node

from neurogrip_control.steering_actuator import SteeringActuator


class CommandGuard(Node):
    """Publish the only bounded, watchdog-protected command to the vehicle."""

    def __init__(self):
        super().__init__("command_guard")

        self.declare_parameter("vehicle_name", "vehicle_blue")
        self.declare_parameter("input_topic", "/neurogrip/command_raw")
        self.declare_parameter("publish_rate_hz", 50.0)
        self.declare_parameter("timeout_s", 0.25)
        self.declare_parameter("max_linear_x_mps", 0.80)
        self.declare_parameter("max_angular_z_rps", 0.45)
        self.declare_parameter("scenario_manifest_path", "")
        self.declare_parameter("steering_delay_s", 0.0)
        self.declare_parameter("steering_gain", 1.0)

        vehicle_name = self.get_parameter("vehicle_name").value
        input_topic = self.get_parameter("input_topic").value
        publish_rate_hz = self.get_parameter("publish_rate_hz").value
        self.timeout_s = self.get_parameter("timeout_s").value
        self.max_linear_x_mps = self.get_parameter("max_linear_x_mps").value
        self.max_angular_z_rps = self.get_parameter("max_angular_z_rps").value
        scenario_manifest_path = self.get_parameter("scenario_manifest_path").value
        steering_delay_s = self.get_parameter("steering_delay_s").value
        steering_gain = self.get_parameter("steering_gain").value

        manifest_actuator = self.load_actuator_parameters(scenario_manifest_path)
        if manifest_actuator is not None:
            steering_delay_s, steering_gain = manifest_actuator
        self.actuator = SteeringActuator(
            delay_s=float(steering_delay_s), gain=float(steering_gain)
        )

        self.output_topic = f"/model/{vehicle_name}/cmd_vel"
        self.last_command = Twist()
        self.last_command_received_ns: int | None = None
        self.watchdog_active = True

        self.create_subscription(Twist, input_topic, self.command_callback, 10)
        self.command_publisher = self.create_publisher(Twist, self.output_topic, 10)
        self.create_timer(1.0 / publish_rate_hz, self.publish_safe_command)

        self.get_logger().info(
            f"Guarding {input_topic} -> {self.output_topic} at {publish_rate_hz:.1f} Hz"
        )
        self.get_logger().info(
            "Steering actuator: "
            f"delay={self.actuator.delay_s:.3f}s gain={self.actuator.gain:.3f}"
        )

    @staticmethod
    def load_actuator_parameters(manifest_path: str):
        """Read delay/gain from a scenario manifest, if one was provided."""
        if not manifest_path:
            return None
        path = Path(manifest_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Scenario manifest does not exist: {path}")
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        scenario = document.get("scenario", {}) if isinstance(document, dict) else {}
        delay_s = float(scenario.get("steering_delay_s", 0.0))
        gain = float(scenario.get("steering_gain", 1.0))
        return delay_s, gain

    @staticmethod
    def clamp(value: float, limit: float) -> float:
        """Clamp one scalar command to the symmetric hard limit."""
        return max(-limit, min(limit, value))

    def bounded_command(self, message: Twist) -> Twist:
        """Return a copy of ``message`` clamped to the hard actuator limits."""
        bounded = Twist()
        bounded.linear.x = self.clamp(message.linear.x, self.max_linear_x_mps)
        bounded.angular.z = self.clamp(message.angular.z, self.max_angular_z_rps)
        return bounded

    def command_is_fresh(self, now_ns: int) -> bool:
        """Return whether the last raw command is inside the watchdog window."""
        if self.last_command_received_ns is None:
            return False
        age_s = (now_ns - self.last_command_received_ns) / 1_000_000_000.0
        return age_s <= self.timeout_s

    def safe_command(self, now_ns: int) -> Twist:
        """Return the bounded command to apply, or zero when it has gone stale."""
        if self.command_is_fresh(now_ns):
            return self.last_command
        return Twist()

    def command_callback(self, message: Twist) -> None:
        """Store a bounded raw command and refresh the watchdog timer."""
        self.last_command = self.bounded_command(message)
        self.last_command_received_ns = self.get_clock().now().nanoseconds

    def apply_actuator(self, now_s: float, command: Twist) -> Twist:
        """Apply steering delay/gain and re-clamp the actuator command."""
        applied = Twist()
        applied.linear.x = command.linear.x
        applied.angular.z = self.clamp(
            self.actuator.update(now_s, command.angular.z), self.max_angular_z_rps
        )
        return applied

    def publish_safe_command(self) -> None:
        """Publish a fresh bounded command or zero if it has gone stale."""
        now_ns = self.get_clock().now().nanoseconds
        if self.command_is_fresh(now_ns):
            if self.watchdog_active:
                self.watchdog_active = False
                self.get_logger().info("Fresh command received; watchdog released.")
            command = self.apply_actuator(
                now_ns / 1_000_000_000.0, self.safe_command(now_ns)
            )
        else:
            if not self.watchdog_active:
                self.watchdog_active = True
                self.get_logger().warning("Command timeout; publishing zero command.")
            # Safety stop must not wait for the actuator delay: clear the delay
            # history and emit an immediate zero command.
            self.actuator.reset()
            command = Twist()

        self.command_publisher.publish(command)

    def destroy_node(self):
        """Guarantee a final stop command during shutdown."""
        self.command_publisher.publish(Twist())
        return super().destroy_node()


def main(args=None):
    """Run the command guard."""
    rclpy.init(args=args)
    node = CommandGuard()

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
