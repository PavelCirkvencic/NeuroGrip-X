"""
Sync EUFS odometry into a Gazebo Fortress visual entity.

EUFS is the only physics authority; this node only calls the Gazebo
``set_pose`` service with the odom pose.  It rejects out-of-order state, keeps
at most one outstanding request (dropping frames instead of queueing) and
publishes diagnostics.  It can never slow down EUFS or a controller.
"""

from __future__ import annotations

import json
import time

import rclpy
from geometry_msgs.msg import Pose
from nav_msgs.msg import Odometry
from rclpy.node import Node
from ros_gz_interfaces.msg import Entity
from ros_gz_interfaces.srv import SetEntityPose
from std_msgs.msg import String


class VisualSyncNode(Node):
    """Publish EUFS odometry poses to a Gazebo model."""

    def __init__(self):
        super().__init__("neurogrip_visual_sync")

        self.declare_parameter("world_name", "neurogrip_visual")
        self.declare_parameter("entity_name", "formula_student_visual")
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("max_update_rate_hz", 30.0)
        self.declare_parameter("diagnostics_topic", "/neurogrip/visual_diagnostics")

        world_name = self.get_parameter("world_name").value
        self.entity_name = self.get_parameter("entity_name").value
        self.max_period_s = 1.0 / float(
            self.get_parameter("max_update_rate_hz").value
        )

        self.client = self.create_client(
            SetEntityPose, f"/world/{world_name}/set_pose"
        )
        self.diagnostics_publisher = self.create_publisher(
            String, self.get_parameter("diagnostics_topic").value, 10
        )
        self.create_subscription(
            Odometry, self.get_parameter("odom_topic").value, self.on_odometry, 50
        )
        self.create_timer(1.0, self.publish_diagnostics)

        self.last_odom_stamp_ns = -1
        self.last_send_time = 0.0
        self.outstanding = None
        self.sent = 0
        self.failed = 0
        self.dropped_busy = 0
        self.dropped_out_of_order = 0
        self.latencies_ms: list[float] = []

    def on_odometry(self, message: Odometry) -> None:
        """Convert and forward one odometry pose when not busy."""
        stamp_ns = message.header.stamp.sec * 1_000_000_000 + message.header.stamp.nanosec
        if stamp_ns <= self.last_odom_stamp_ns:
            self.dropped_out_of_order += 1
            return
        self.last_odom_stamp_ns = stamp_ns

        now = time.monotonic()
        if now - self.last_send_time < self.max_period_s:
            return
        if self.outstanding is not None and not self.outstanding.done():
            self.dropped_busy += 1
            return
        if not self.client.service_is_ready():
            return

        request = SetEntityPose.Request()
        entity = Entity()
        entity.name = self.entity_name
        entity.type = Entity.MODEL
        request.entity = entity
        request.pose = Pose()
        request.pose.position.x = message.pose.pose.position.x
        request.pose.position.y = message.pose.pose.position.y
        request.pose.position.z = message.pose.pose.position.z
        request.pose.orientation = message.pose.pose.orientation

        self.last_send_time = now
        self.sent += 1
        future = self.client.call_async(request)
        self.outstanding = future
        future.add_done_callback(self.on_response)

    def on_response(self, future) -> None:
        """Record latency and success of one service call."""
        try:
            response = future.result()
        except Exception as error:  # noqa: BLE001 - diagnostics only
            self.failed += 1
            self.get_logger().warning(f"set_pose failed: {error}")
            return
        if response is None or not response.success:
            self.failed += 1
            return
        self.latencies_ms.append((time.monotonic() - self.last_send_time) * 1000.0)
        if len(self.latencies_ms) > 200:
            del self.latencies_ms[:100]

    def publish_diagnostics(self) -> None:
        """Publish sync diagnostics once per second."""
        mean_latency = (
            sum(self.latencies_ms) / len(self.latencies_ms)
            if self.latencies_ms
            else 0.0
        )
        message = String()
        message.data = json.dumps(
            {
                "sent": self.sent,
                "failed": self.failed,
                "dropped_busy": self.dropped_busy,
                "dropped_out_of_order": self.dropped_out_of_order,
                "mean_latency_ms": round(mean_latency, 3),
                "entity": self.entity_name,
            }
        )
        self.diagnostics_publisher.publish(message)


def main(args=None):
    """Run the visual sync node."""
    rclpy.init(args=args)
    node = VisualSyncNode()
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
