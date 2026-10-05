"""Publish a deterministic command sequence for system-identification runs."""

from __future__ import annotations

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from std_msgs.msg import String

from neurogrip_control.excitation_schedule import EXCITATION_PROFILES, ExcitationSchedule


class ExcitationDriver(Node):
    """Drive the simulated vehicle through a deterministic excitation sequence."""

    def __init__(self):
        super().__init__("excitation_driver")

        self.declare_parameter("vehicle_name", "vehicle_blue")
        self.declare_parameter("publish_rate_hz", 20.0)
        self.declare_parameter("profile_name", "baseline_v1")

        publish_rate_hz = self.get_parameter("publish_rate_hz").value
        self.profile_name = self.get_parameter("profile_name").value
        if self.profile_name not in EXCITATION_PROFILES:
            available = ", ".join(sorted(EXCITATION_PROFILES))
            raise ValueError(
                f"Unknown profile_name '{self.profile_name}'. Available: {available}"
            )
        self.schedule = ExcitationSchedule(EXCITATION_PROFILES[self.profile_name])

        self.command_publisher = self.create_publisher(
            Twist,
            "/neurogrip/command_raw",
            10,
        )
        self.phase_publisher = self.create_publisher(
            String,
            "/neurogrip/excitation_phase",
            10,
        )

        self.start_time_s: float | None = None

        self.create_timer(1.0 / publish_rate_hz, self.publish_command)
        self.get_logger().info(
            f"Starting '{self.profile_name}' excitation run "
            f"({self.schedule.total_duration_s:.1f} s, "
            f"{len(self.schedule.phases)} phases)."
        )

    def now_s(self) -> float:
        """Return the current ROS clock in seconds."""
        return self.get_clock().now().nanoseconds / 1_000_000_000.0

    def publish_phase_name(self, name: str) -> None:
        """Publish a phase label once when the active command changes."""
        message = String()
        message.data = name
        self.phase_publisher.publish(message)

    def publish_twist(self, linear_x_mps: float, angular_z_rps: float) -> None:
        """Publish one velocity command."""
        message = Twist()
        message.linear.x = linear_x_mps
        message.angular.z = angular_z_rps
        self.command_publisher.publish(message)

    def publish_command(self) -> None:
        """Advance the schedule and publish the active command."""
        if self.start_time_s is None:
            self.start_time_s = self.now_s()
            first = self.schedule.phases[0]
            self.publish_phase_name(first.name)
            self.get_logger().info(f"Phase 1/{len(self.schedule.phases)}: {first.name}")

        elapsed_s = self.now_s() - self.start_time_s
        state = self.schedule.advance(elapsed_s)

        if state.completed_now:
            self.publish_phase_name("complete")
            self.get_logger().info("Excitation run complete; publishing zero command.")
        elif state.entered_new_phase:
            self.publish_phase_name(state.phase_name)
            self.get_logger().info(
                f"Phase {state.phase_index + 1}/{len(self.schedule.phases)}: "
                f"{state.phase_name}"
            )

        self.publish_twist(state.linear_x_mps, state.angular_z_rps)

    def destroy_node(self):
        """Command a stop before ROS destroys this publisher."""
        self.publish_twist(0.0, 0.0)
        return super().destroy_node()


def main(args=None):
    """Run the excitation driver."""
    rclpy.init(args=args)
    node = ExcitationDriver()

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
