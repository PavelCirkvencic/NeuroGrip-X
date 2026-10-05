"""Record a camera image stream to timestamped frames for offline encoding."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String


class VideoRecorderNode(Node):
    """Save camera frames and publish recording diagnostics."""

    def __init__(self):
        super().__init__("neurogrip_video_recorder")
        self.declare_parameter("image_topic", "/camera/image")
        self.declare_parameter("output_dir", "runs/eufs_v1/visual/frames")
        self.declare_parameter("diagnostics_topic", "/neurogrip/video_diagnostics")
        self.declare_parameter("max_frames", 0)

        self.output_dir = Path(self.get_parameter("output_dir").value).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_frames = int(self.get_parameter("max_frames").value)
        self.frames = 0
        self.last_stamp_ns = -1

        self.create_subscription(
            Image, self.get_parameter("image_topic").value, self.on_image, 10
        )
        self.diagnostics_publisher = self.create_publisher(
            String, self.get_parameter("diagnostics_topic").value, 10
        )
        self.create_timer(1.0, self.publish_diagnostics)

    def on_image(self, message: Image) -> None:
        """Write one PPM frame with a monotonic index."""
        stamp_ns = message.header.stamp.sec * 1_000_000_000 + message.header.stamp.nanosec
        if stamp_ns <= self.last_stamp_ns:
            return
        self.last_stamp_ns = stamp_ns
        if self.max_frames and self.frames >= self.max_frames:
            return
        if message.encoding not in ("rgb8", "bgr8") or message.height == 0:
            return
        row_bytes = int(message.step)
        minimum_row_bytes = int(message.width) * 3
        required_bytes = int(message.height) * row_bytes
        if row_bytes < minimum_row_bytes or len(message.data) < required_bytes:
            self.get_logger().warning("dropping malformed image row stride")
            return
        rows = np.frombuffer(message.data, dtype=np.uint8, count=required_bytes).reshape(
            message.height, row_bytes
        )
        pixels = rows[:, :minimum_row_bytes].reshape(message.height, message.width, 3)
        # PPM is RGB by definition; Gazebo can legitimately bridge BGR images.
        if message.encoding == "bgr8":
            pixels = pixels[:, :, ::-1]
        path = self.output_dir / f"frame_{self.frames:06d}.ppm"
        header = f"P6\n{message.width} {message.height}\n255\n".encode()
        path.write_bytes(header + pixels.tobytes())
        self.frames += 1

    def publish_diagnostics(self) -> None:
        """Publish the frame count once per second."""
        message = String()
        message.data = json.dumps({"frames": self.frames})
        self.diagnostics_publisher.publish(message)


def main(args=None):
    """Run the recorder."""
    rclpy.init(args=args)
    node = VideoRecorderNode()
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
