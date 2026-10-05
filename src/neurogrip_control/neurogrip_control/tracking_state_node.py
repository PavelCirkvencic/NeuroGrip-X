"""Publish the versioned 50 Hz tracking-state contract independently of control."""

from __future__ import annotations

import math

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

from neurogrip_control.track_projection import TrackProjector

TRACKING_SCHEMA = 1


class TrackingStateNode(Node):
    """Project `/odom` onto one immutable reference-track artifact."""

    def __init__(self):
        super().__init__("neurogrip_tracking_state")
        self.declare_parameter("track_npz", "")
        path = self.get_parameter("track_npz").value
        if not path:
            raise ValueError("track_npz parameter is required")
        self.projector = TrackProjector.from_npz(path)
        self.previous_index: int | None = None
        self.publisher = self.create_publisher(Float64MultiArray, "/neurogrip/tracking_state", 10)
        self.create_subscription(Odometry, "/odom", self.on_odom, 50)

    def on_odom(self, message: Odometry) -> None:
        """Project a complete EUFS pose and publish the fixed ten-value schema."""
        pose, twist = message.pose.pose, message.twist.twist
        yaw = math.atan2(
            2.0
            * (
                pose.orientation.w * pose.orientation.z
                + pose.orientation.x * pose.orientation.y
            ),
            1.0 - 2.0 * (pose.orientation.y**2 + pose.orientation.z**2),
        )
        projected = self.projector.project(
            pose.position.x, pose.position.y, yaw, self.previous_index
        )
        stamp = message.header.stamp
        time_s = stamp.sec + stamp.nanosec * 1e-9
        # EUFS vehicle-state odometry already publishes body-frame longitudinal
        # and lateral velocity.  Re-differentiating world pose here suppressed
        # the small but physically important lateral-velocity signal.
        body_vx = float(twist.linear.x)
        body_vy = float(twist.linear.y)
        self.previous_index = projected["index"]
        state = Float64MultiArray()
        state.data = [
            float(TRACKING_SCHEMA), float(time_s), projected["e_y_m"], projected["e_psi_rad"],
            float(body_vy), float(twist.angular.z), float(body_vx),
            projected["curvature_1pm"], float(body_vx * projected["curvature_1pm"]),
            projected["lap_progress"],
        ]
        self.publisher.publish(state)


def main(args=None) -> None:
    """Run the tracking state adapter."""
    rclpy.init(args=args)
    node = TrackingStateNode()
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
