"""Set the EUFS Sim 2 track CSV through the ``/eufs_sim2/file`` service."""

from __future__ import annotations

import time
from pathlib import Path

import rclpy
from eufs_msgs.srv import SetString
from rclpy.node import Node


class SetTrackNode(Node):
    """Wait for the file service, load the track and verify success."""

    def __init__(self):
        super().__init__("neurogrip_set_track")
        self.declare_parameter("track_path", "")
        self.declare_parameter("service_name", "/eufs_sim2/file")
        self.declare_parameter("timeout_s", 60.0)

        self.track_path = (
            Path(self.get_parameter("track_path").value).expanduser().resolve()
        )
        self.service_name = self.get_parameter("service_name").value
        self.timeout_s = float(self.get_parameter("timeout_s").value)
        if not self.track_path.is_file():
            raise FileNotFoundError(f"track CSV does not exist: {self.track_path}")

        self.client = self.create_client(SetString, self.service_name)
        self.started_at = time.monotonic()
        self.request_sent = False
        self.succeeded = False
        self.create_timer(0.2, self.try_call)

    def try_call(self) -> None:
        """Send the request once the service is available."""
        if self.request_sent:
            return
        if time.monotonic() - self.started_at > self.timeout_s:
            self.get_logger().error(
                f"service {self.service_name} not available after {self.timeout_s}s"
            )
            rclpy.shutdown()
            return
        if not self.client.service_is_ready():
            return

        request = SetString.Request()
        request.data = str(self.track_path)
        future = self.client.call_async(request)
        future.add_done_callback(self.on_response)
        self.request_sent = True

    def on_response(self, future) -> None:
        """Verify the service response and shut down the helper."""
        try:
            response = future.result()
        except Exception as error:  # noqa: BLE001 - report and fail clearly
            self.get_logger().error(f"track service call failed: {error}")
            rclpy.shutdown()
            return
        if response.success:
            self.succeeded = True
            self.get_logger().info(f"track loaded: {self.track_path}")
        else:
            self.get_logger().error(
                f"track load rejected: {response.message or 'no message'}"
            )
        rclpy.shutdown()


def main(args=None) -> int:
    """Run the track helper and return a non-zero code on failure."""
    rclpy.init(args=args)
    node = SetTrackNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        succeeded = node.succeeded
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0 if succeeded else 1


if __name__ == "__main__":
    raise SystemExit(main())
