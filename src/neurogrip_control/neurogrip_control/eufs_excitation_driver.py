"""Safe 50 Hz Ackermann excitation for identifying EUFS lateral dynamics."""

from __future__ import annotations

import math

import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from std_msgs.msg import String


class EufsExcitationDriver(Node):
    """Publish a deterministic ramp/chirp/PRBS profile through the command guard."""

    def __init__(self):
        super().__init__("neurogrip_eufs_excitation")
        self.declare_parameter("profile", "system_id_v2")
        self.declare_parameter("publish_rate_hz", 50.0)
        self.profile = str(self.get_parameter("profile").value)
        if self.profile not in {"system_id_v2", "high_speed_v3"}:
            raise ValueError("profile must be system_id_v2 or high_speed_v3")
        self.start_time_s: float | None = None
        self.complete = False
        self.command_publisher = self.create_publisher(
            AckermannDriveStamped, "/neurogrip/command_candidate", 10
        )
        status_qos = QoSProfile(depth=10, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.status_publisher = self.create_publisher(
            String, "/neurogrip/controller_status", status_qos
        )
        self.create_timer(1.0 / float(self.get_parameter("publish_rate_hz").value), self.step)

    def now_s(self) -> float:
        """Return ROS simulation time in seconds."""
        return self.get_clock().now().nanoseconds / 1e9

    def command_for_elapsed(self, elapsed_s: float) -> tuple[float, float, bool]:
        """Return bounded acceleration/steering and whether this profile has ended."""
        if self.profile == "high_speed_v3":
            return self.high_speed_command(elapsed_s)
        if elapsed_s < 3.0:
            return 2.5, 0.0, False
        if elapsed_s < 21.0:
            local = elapsed_s - 3.0
            frequency = 0.2 + (1.5 - 0.2) * local / 18.0
            return 0.10, 0.26 * math.sin(2.0 * math.pi * frequency * local), False
        if elapsed_s < 33.0:
            index = int((elapsed_s - 21.0) / 0.40)
            return 0.0, 0.22 if index % 2 == 0 else -0.22, False
        if elapsed_s < 36.0:
            return -1.5, 0.0, False
        return 0.0, 0.0, True

    @staticmethod
    def high_speed_command(elapsed_s: float) -> tuple[float, float, bool]:
        """Excite tyre dynamics around 8–13 m/s without a track controller."""
        if elapsed_s < 4.0:
            return 4.0, 0.0, False
        if elapsed_s < 24.0:
            local = elapsed_s - 4.0
            phase = 2.0 * math.pi * (
                0.12 * local + 0.5 * ((0.55 - 0.12) / 20.0) * local**2
            )
            return 1.20, 0.105 * math.sin(phase), False
        if elapsed_s < 38.0:
            index = int((elapsed_s - 24.0) / 0.70)
            return 1.00, 0.085 if index % 2 == 0 else -0.085, False
        if elapsed_s < 43.0:
            return -4.0, 0.0, False
        return 0.0, 0.0, True

    def step(self) -> None:
        """Publish one 50 Hz command and one explicit completion event at the end."""
        if self.start_time_s is None:
            self.start_time_s = self.now_s()
            status = String()
            status.data = (
                '{"controller":"EXCITATION","event":"excitation_start",'
                f'"sim_time_s":{self.start_time_s:.9f}'
                "}"
            )
            self.status_publisher.publish(status)
        elapsed_s = self.now_s() - self.start_time_s
        acceleration, steering, finished = self.command_for_elapsed(elapsed_s)
        command = AckermannDriveStamped()
        command.header.stamp = self.get_clock().now().to_msg()
        command.drive.acceleration, command.drive.steering_angle = acceleration, steering
        self.command_publisher.publish(command)
        if finished and not self.complete:
            self.complete = True
            status = String()
            status.data = (
                '{"controller":"EXCITATION","event":"excitation_complete",'
                f'"sim_time_s":{self.now_s():.9f}'
                "}"
            )
            self.status_publisher.publish(status)


def main(args=None) -> None:
    """Run the phase-complete open-loop excitation driver."""
    rclpy.init(args=args)
    node = EufsExcitationDriver()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
