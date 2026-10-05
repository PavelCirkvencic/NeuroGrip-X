"""Single-authority Ackermann command guard and actuator model."""

from __future__ import annotations

import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from rclpy.node import Node

from neurogrip_control.steering_actuator import SteeringActuator


class AckermannGuard(Node):
    """Clamp, watchdog and rate-limit the command sent to EUFS."""

    def __init__(self):
        super().__init__("ackermann_guard")
        self.declare_parameter("input_topic", "/neurogrip/command_candidate")
        self.declare_parameter("output_topic", "/cmd")
        self.declare_parameter("publish_rate_hz", 50.0)
        self.declare_parameter("timeout_s", 0.10)
        self.declare_parameter("max_steering_rad", 0.37)
        self.declare_parameter("max_acceleration_mps2", 5.2)
        self.declare_parameter("max_steering_rate_rad_s", 2.5)
        self.declare_parameter("steering_delay_s", 0.0)
        self.declare_parameter("steering_gain", 1.0)

        self.timeout_s = float(self.get_parameter("timeout_s").value)
        self.max_steering = float(self.get_parameter("max_steering_rad").value)
        self.max_acceleration = float(self.get_parameter("max_acceleration_mps2").value)
        self.max_rate = float(self.get_parameter("max_steering_rate_rad_s").value)
        self.actuator = SteeringActuator(
            delay_s=float(self.get_parameter("steering_delay_s").value),
            gain=float(self.get_parameter("steering_gain").value),
        )
        self.last_command = AckermannDriveStamped()
        self.last_received_ns = None
        self.last_applied_steering = 0.0
        self.watchdog_active = False

        self.publisher = self.create_publisher(
            AckermannDriveStamped, self.get_parameter("output_topic").value, 10
        )
        self.safe_publisher = self.create_publisher(
            AckermannDriveStamped, "/neurogrip/command_safe", 10
        )
        self.create_subscription(
            AckermannDriveStamped, self.get_parameter("input_topic").value, self.on_command, 10
        )
        self.create_timer(1.0 / float(self.get_parameter("publish_rate_hz").value), self.publish)

    def on_command(self, message: AckermannDriveStamped) -> None:
        """Store the candidate command."""
        self.last_command = message
        self.last_received_ns = self.get_clock().now().nanoseconds

    def now_s(self) -> float:
        """Return the ROS clock in seconds."""
        return self.get_clock().now().nanoseconds / 1_000_000_000.0

    def publish(self) -> None:
        """Publish the safe command or a zero command when stale."""
        now_ns = self.get_clock().now().nanoseconds
        fresh = (
            self.last_received_ns is not None
            and (now_ns - self.last_received_ns) / 1e9 <= self.timeout_s
        )
        command = AckermannDriveStamped()
        command.header.stamp = self.get_clock().now().to_msg()
        if fresh:
            requested = max(
                -self.max_steering,
                min(self.max_steering, self.last_command.drive.steering_angle),
            )
            target_steering = self.actuator.update(self.now_s(), requested)
            if not self.actuator.delay_s:
                # Rate limit still applies even without transport delay.
                step = self.max_rate * (1.0 / 50.0)
                target_steering = max(
                    self.last_applied_steering - step,
                    min(self.last_applied_steering + step, target_steering),
                )
            applied = max(-self.max_steering, min(self.max_steering, target_steering))
            self.last_applied_steering = applied
            command.drive.steering_angle = applied
            command.drive.acceleration = max(
                -self.max_acceleration,
                min(self.max_acceleration, self.last_command.drive.acceleration),
            )
            if self.watchdog_active:
                self.watchdog_active = False
        else:
            if not self.watchdog_active:
                self.get_logger().warning("command timeout; publishing safe stop")
            self.watchdog_active = True
            self.actuator.reset()
            self.last_applied_steering = 0.0
            command.drive.acceleration = 0.0
            command.drive.steering_angle = 0.0
        self.publisher.publish(command)
        self.safe_publisher.publish(command)


def main(args=None):
    """Run the guard."""
    rclpy.init(args=args)
    node = AckermannGuard()
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
