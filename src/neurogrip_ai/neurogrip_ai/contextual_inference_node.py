"""
Read-only contextual Koopman (N1) inference node.

It keeps a causal history buffer, publishes its prediction, the context
embedding and the local A/B/bias model through ``neurogrip_interfaces``, and
reports latency, deadline misses and health.  It never publishes a command and
cannot bypass the command guard.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from neurogrip_control.body_velocity import (
    CausalBodyVelocityEstimator,
    yaw_from_quaternion,
)
from neurogrip_interfaces.msg import DynamicsModel, SafetyStatus, VehicleState
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu

from neurogrip_ai.contextual_runner import ContextualModelRunner


class ContextualInferenceNode(Node):
    """Run the N1 model as a non-actuating ROS observer."""

    def __init__(self):
        super().__init__("neurogrip_contextual_inference")

        self.declare_parameter("checkpoint_path", "")
        self.declare_parameter("model_root", "learning")
        self.declare_parameter("vehicle_name", "vehicle_blue")
        self.declare_parameter("publish_rate_hz", 20.0)
        self.declare_parameter("max_input_age_s", 0.5)
        self.declare_parameter("deadline_ms", 20.0)
        self.declare_parameter("device", "cpu")
        self.declare_parameter("uncertainty_radius", 0.0)
        self.declare_parameter("model_version", 1)

        checkpoint_path = self.get_parameter("checkpoint_path").value
        model_root = self.get_parameter("model_root").value
        vehicle_name = self.get_parameter("vehicle_name").value
        publish_rate_hz = float(self.get_parameter("publish_rate_hz").value)
        self.max_input_age_s = float(self.get_parameter("max_input_age_s").value)
        self.deadline_ms = float(self.get_parameter("deadline_ms").value)
        self.uncertainty_radius = float(
            self.get_parameter("uncertainty_radius").value
        )
        self.model_version = int(self.get_parameter("model_version").value)

        if not checkpoint_path:
            raise ValueError("checkpoint_path parameter is required.")
        self.runner = ContextualModelRunner(
            Path(checkpoint_path), Path(model_root), device=self.get_parameter("device").value
        )

        self.body_velocity = CausalBodyVelocityEstimator()
        self.latest_odometry = None
        self.latest_imu = None
        self.latest_command = Twist()
        self.odometry_received_at = None
        self.imu_received_at = None
        self.command_received_at = None
        self.inference_count = 0
        self.deadline_misses = 0

        self.vehicle_state_publisher = self.create_publisher(
            VehicleState, "/neurogrip/vehicle_state", 10
        )
        self.dynamics_publisher = self.create_publisher(
            DynamicsModel, "/neurogrip/dynamics_model", 10
        )
        self.safety_publisher = self.create_publisher(
            SafetyStatus, "/neurogrip/safety_status", 10
        )
        self.prediction_publisher = self.create_publisher(
            Twist, "/neurogrip/n1_predicted_state", 10
        )

        self.create_subscription(
            Odometry,
            f"/model/{vehicle_name}/odometry",
            self.odometry_callback,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Imu,
            f"/model/{vehicle_name}/imu_filtered",
            self.imu_callback,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Twist,
            f"/model/{vehicle_name}/cmd_vel",
            self.command_callback,
            10,
        )
        self.create_timer(1.0 / publish_rate_hz, self.infer)
        self.get_logger().info(
            f"N1 contextual inference loaded ({self.runner.context_mode}) on "
            f"{self.runner.device}; history={self.runner.config.history_steps}."
        )

    def odometry_callback(self, message: Odometry) -> None:
        """Cache odometry and update the causal body-velocity estimate."""
        self.latest_odometry = message
        self.odometry_received_at = time.monotonic()

    def imu_callback(self, message: Imu) -> None:
        """Cache the filtered IMU sample."""
        self.latest_imu = message
        self.imu_received_at = time.monotonic()

    def command_callback(self, message: Twist) -> None:
        """Cache the applied command."""
        self.latest_command = message
        self.command_received_at = time.monotonic()

    def inputs_are_fresh(self, now: float) -> bool:
        """Return whether all inputs are recent enough."""
        if None in (
            self.odometry_received_at,
            self.imu_received_at,
            self.command_received_at,
        ):
            return False
        return (
            now - self.odometry_received_at <= self.max_input_age_s
            and now - self.imu_received_at <= self.max_input_age_s
            and now - self.command_received_at <= self.max_input_age_s
        )

    def publish_safety(
        self, healthy: bool, state: str, detail: str, ood: float, latency_ms: float
    ) -> None:
        """Publish one safety/health status."""
        message = SafetyStatus()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = "base_link"
        message.healthy = healthy
        message.state = state
        message.detail = detail
        message.uncertainty_radius = float(self.uncertainty_radius)
        message.ood_score = float(ood)
        message.deadline_miss_rate = float(
            self.deadline_misses / max(self.inference_count, 1)
        )
        message.latency_ms = float(latency_ms)
        self.safety_publisher.publish(message)

    def infer(self) -> None:
        """Append one causal sample and publish the prediction and model."""
        now = time.monotonic()
        if not self.inputs_are_fresh(now):
            self.publish_safety(False, "stale", "missing_or_stale_inputs", 0.0, 0.0)
            return

        odometry = self.latest_odometry
        pose = odometry.pose.pose
        yaw_rad = yaw_from_quaternion(
            pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w
        )
        body_v_x, _ = self.body_velocity.update(
            odometry.header.stamp.sec + odometry.header.stamp.nanosec * 1e-9,
            pose.position.x,
            pose.position.y,
            yaw_rad,
        )
        v_x_mps = body_v_x if body_v_x is not None else odometry.twist.twist.linear.x
        yaw_rate = self.latest_imu.angular_velocity.z
        self.runner.append(
            v_x_mps, yaw_rate, self.latest_command.linear.x, self.latest_command.angular.z
        )

        started = time.perf_counter()
        result = self.runner.predict()
        latency_ms = (time.perf_counter() - started) * 1_000.0
        if result is None:
            self.publish_safety(False, "warming_up", "insufficient_history", 0.0, latency_ms)
            return

        self.inference_count += 1
        if latency_ms > self.deadline_ms:
            self.deadline_misses += 1

        state_message = VehicleState()
        state_message.header.stamp = self.get_clock().now().to_msg()
        state_message.header.frame_id = "base_link"
        state_message.v_x_mps = float(v_x_mps)
        state_message.v_y_mps = 0.0
        state_message.yaw_rate_rps = float(yaw_rate)
        state_message.yaw_rad = float(yaw_rad)
        state_message.source = 1
        self.vehicle_state_publisher.publish(state_message)

        predicted_v_x, predicted_yaw = result["predicted_state"]
        prediction = Twist()
        prediction.linear.x = float(predicted_v_x)
        prediction.angular.z = float(predicted_yaw)
        self.prediction_publisher.publish(prediction)

        model_message = DynamicsModel()
        model_message.header.stamp = self.get_clock().now().to_msg()
        model_message.header.frame_id = "base_link"
        model_message.model_version = self.model_version
        model_message.latent_dim = self.runner.config.latent_dim
        model_message.input_dim = self.runner.config.input_dim
        model_message.rank = self.runner.config.rank
        model_message.context = result["context"].astype(float).tolist()
        model_message.a_matrix = result["a_matrix"].astype(float).flatten().tolist()
        model_message.b_matrix = result["b_matrix"].astype(float).flatten().tolist()
        model_message.bias = result["bias"].astype(float).tolist()
        model_message.latency_ms = float(latency_ms)
        model_message.spectral_radius = float(result["spectral_radius"])
        self.dynamics_publisher.publish(model_message)

        ood_score = float(np.linalg.norm(result["context"]))
        self.publish_safety(True, "ok", "uncalibrated_ood_norm", ood_score, latency_ms)

        if self.inference_count % 200 == 0:
            self.get_logger().info(
                f"N1 inference #{self.inference_count}: latency={latency_ms:.2f}ms "
                f"spectral_radius={result['spectral_radius']:.3f} "
                f"ood_norm={ood_score:.3f} deadline_miss_rate="
                f"{self.deadline_misses / self.inference_count:.4f}"
            )


def main(args=None):
    """Run the read-only contextual inference node."""
    rclpy.init(args=args)
    node = ContextualInferenceNode()

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
