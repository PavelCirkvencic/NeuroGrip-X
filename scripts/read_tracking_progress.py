#!/usr/bin/env python3
"""Print the last lap_progress value from /neurogrip/tracking_state."""

from __future__ import annotations

import argparse
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray


class Reader(Node):
    """Cache the latest tracking state."""

    def __init__(self):
        super().__init__("tracking_progress_reader")
        self.latest = None
        self.create_subscription(Float64MultiArray, "/neurogrip/tracking_state", self.on_state, 10)

    def on_state(self, message: Float64MultiArray) -> None:
        """Store the message."""
        self.latest = list(message.data)


def main() -> int:
    """Read for a fixed duration and print the final progress value."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=3.0)
    arguments = parser.parse_args()
    rclpy.init()
    node = Reader()
    deadline = time.monotonic() + arguments.seconds
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.05)
    node.destroy_node()
    rclpy.shutdown()
    if node.latest is None:
        print("NONE")
        return 1
    print(f"{node.latest[-1]:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
