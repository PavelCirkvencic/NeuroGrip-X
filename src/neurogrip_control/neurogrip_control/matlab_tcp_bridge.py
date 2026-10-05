"""Transport MATLAB control frames over a localhost-only TCP connection."""

from __future__ import annotations

import math
import socket

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

CONTROL_FRAME_LENGTH = 125
CANDIDATE_LENGTH = 5


def encode_control_frame(values: list[float] | tuple[float, ...]) -> bytes:
    """Encode one finite schema-1 control frame as newline-delimited CSV."""
    numeric = [float(value) for value in values]
    if len(numeric) != CONTROL_FRAME_LENGTH:
        raise ValueError(f"control frame must contain {CONTROL_FRAME_LENGTH} values")
    if any(not math.isfinite(value) for value in numeric):
        raise ValueError("control frame contains NaN or Inf")
    if numeric[0] != 1.0:
        raise ValueError("unsupported control frame schema")
    return (",".join(format(value, ".17g") for value in numeric) + "\n").encode(
        "ascii"
    )


def decode_candidate(line: bytes) -> list[float]:
    """Decode one finite five-value candidate without making safety decisions."""
    try:
        values = [float(field) for field in line.decode("ascii").strip().split(",")]
    except (UnicodeDecodeError, ValueError) as error:
        raise ValueError("candidate is not numeric ASCII CSV") from error
    if len(values) != CANDIDATE_LENGTH:
        raise ValueError(f"candidate must contain {CANDIDATE_LENGTH} values")
    if any(not math.isfinite(value) for value in values):
        raise ValueError("candidate contains NaN or Inf")
    return values


class MatlabTcpBridge(Node):
    """Bridge flat ROS messages to MATLAB without joining MATLAB to ROS DDS."""

    def __init__(self):
        super().__init__("neurogrip_matlab_tcp_bridge")
        self.declare_parameter("host", "127.0.0.1")
        self.declare_parameter("port", 55980)
        self.declare_parameter("poll_rate_hz", 500.0)
        host = str(self.get_parameter("host").value)
        port = int(self.get_parameter("port").value)
        poll_rate_hz = float(self.get_parameter("poll_rate_hz").value)
        if host not in {"127.0.0.1", "localhost"}:
            raise ValueError("MATLAB bridge host must remain localhost-only")
        if not 1 <= port <= 65535 or poll_rate_hz <= 0.0:
            raise ValueError("invalid MATLAB TCP bridge configuration")

        self._socket = socket.create_connection((host, port), timeout=10.0)
        self._socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._socket.setblocking(False)
        self._pending: bytes | None = None
        self._outgoing = bytearray()
        self._incoming = bytearray()
        self._request_in_flight = False
        self._closed = False
        self._invalid_responses = 0

        self._publisher = self.create_publisher(
            Float64MultiArray,
            "/neurogrip/command_candidate/matlab_flat",
            10,
        )
        self.create_subscription(
            Float64MultiArray,
            "/neurogrip/matlab_control_frame",
            self.on_control_frame,
            1,
        )
        self.create_timer(1.0 / poll_rate_hz, self.poll_socket)
        self.get_logger().info(f"MATLAB TCP bridge connected to {host}:{port}")

    def on_control_frame(self, message: Float64MultiArray) -> None:
        """Retain the newest complete frame when no partial write is active."""
        try:
            encoded = encode_control_frame(list(message.data))
        except ValueError as error:
            self.get_logger().warning(f"rejected outbound control frame: {error}")
            return
        # Maintain one request in flight. While MATLAB solves it, retain only
        # the newest not-yet-sent frame so simulator speed cannot form a queue.
        self._pending = encoded

    def poll_socket(self) -> None:
        """Progress nonblocking writes and publish all complete responses."""
        if self._closed:
            return
        if (
            self._pending is not None
            and not self._outgoing
            and not self._request_in_flight
        ):
            self._outgoing.extend(self._pending)
            self._pending = None
        if self._outgoing and not self._request_in_flight:
            try:
                sent = self._socket.send(self._outgoing)
                del self._outgoing[:sent]
                if not self._outgoing:
                    self._request_in_flight = True
            except BlockingIOError:
                pass
            except OSError as error:
                self.fail_transport(error)
                return

        while True:
            try:
                packet = self._socket.recv(65536)
            except BlockingIOError:
                break
            except OSError as error:
                self.fail_transport(error)
                return
            if not packet:
                self.fail_transport(ConnectionError("MATLAB closed the TCP connection"))
                return
            self._incoming.extend(packet)

        while b"\n" in self._incoming:
            raw_line, _, remainder = self._incoming.partition(b"\n")
            self._incoming[:] = remainder
            try:
                values = decode_candidate(raw_line)
            except ValueError as error:
                self._invalid_responses += 1
                self.get_logger().warning(f"rejected MATLAB TCP response: {error}")
                continue
            message = Float64MultiArray()
            message.data = values
            self._publisher.publish(message)
            self._request_in_flight = False

    def fail_transport(self, error: Exception) -> None:
        """Close a failed connection; downstream watchdogs then fail closed."""
        if self._closed:
            return
        self._closed = True
        self.get_logger().error(f"MATLAB TCP transport failed: {error}")
        try:
            self._socket.close()
        except OSError:
            pass

    def destroy_node(self):
        """Close the localhost transport before ROS node destruction."""
        if not self._closed:
            self._closed = True
            try:
                self._socket.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            self._socket.close()
        return super().destroy_node()


def main(args=None):
    """Run the transport-only MATLAB TCP bridge."""
    rclpy.init(args=args)
    node = MatlabTcpBridge()
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
