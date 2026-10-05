"""Read-only NeuroGrip-X learned-dynamics ensemble runtime for C2 MPC."""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections import deque
from pathlib import Path

import numpy as np
import rclpy
import torch
from ackermann_msgs.msg import AckermannDriveStamped
from eufs_msgs.msg import WheelSpeedsStamped
from nav_msgs.msg import Odometry
from neurogrip_interfaces.msg import DynamicsModel, SafetyStatus, VehicleState
from rclpy.node import Node
from sensor_msgs.msg import Imu
from std_msgs.msg import Float64MultiArray

from neurogrip_ai.physics_residual import (
    FEATURE_NAMES,
    PhysicsResidualConfig,
    PhysicsResidualNet,
    physical_prediction,
    stable_enough,
)

TRACKING_SCHEMA = 1


def sha256_file(path: Path) -> str:
    """Return the SHA-256 checksum used for artifact identity validation."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def header_stamp_s(message) -> float:
    """Convert a standard ROS message header timestamp into seconds."""
    return message.header.stamp.sec + message.header.stamp.nanosec * 1e-9


class PhysicalEnsemble:
    """Load a hash-linked residual or Koopman ensemble and emit physical Jacobians."""

    def __init__(self, manifest_path: Path, nominal_fit: Path):
        # These compact recurrent models are slower and less deterministic when
        # PyTorch starts a large thread pool. One CPU thread keeps inference
        # inside its deadline on this laptop and makes latency reproducible.
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        self.manifest_path = manifest_path.expanduser().resolve()
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.manifest_schema = int(self.manifest.get("schema_version", -1))
        if self.manifest_schema not in (1, 2, 3):
            raise ValueError("unsupported C2 ensemble manifest schema")
        if tuple(self.manifest.get("feature_names", ())) != FEATURE_NAMES:
            raise ValueError("ensemble feature order does not match deployed v2 contract")
        nominal_sha = sha256_file(nominal_fit.expanduser().resolve())
        if self.manifest.get("nominal_fit_sha256") != nominal_sha:
            raise ValueError("ensemble nominal-fit hash does not match controller nominal fit")
        self.artifact_sha256 = sha256_file(self.manifest_path)
        if self.manifest_schema == 1:
            self.model_family = "contextual_physics_residual_gru"
            self.dynamics_schema = 2
            self.config = PhysicsResidualConfig(**self.manifest["architecture"])
            model_type = PhysicsResidualNet
        elif self.manifest_schema == 2:
            if self.manifest.get("model_family") != "contextual_neural_koopman_v2":
                raise ValueError("schema-2 manifest is not a neural Koopman model")
            from neurogrip.config import NeuroGripConfig
            from neurogrip.koopman_v2 import PhysicalKoopmanModel

            self.model_family = "contextual_neural_koopman_v2"
            self.dynamics_schema = 3
            self.config = NeuroGripConfig(**self.manifest["architecture"])
            model_type = PhysicalKoopmanModel
        else:
            if self.manifest.get("model_family") != "contextual_neural_koopman_grip_v3":
                raise ValueError("schema-3 manifest is not a grip-aware neural Koopman model")
            from neurogrip.config import NeuroGripConfig
            from neurogrip.koopman_grip import GripAwareKoopmanModel

            self.model_family = "contextual_neural_koopman_grip_v3"
            self.dynamics_schema = 4
            self.config = NeuroGripConfig(**self.manifest["architecture"])
            model_type = GripAwareKoopmanModel
        self.models = []
        self.mean: np.ndarray | None = None
        self.scale: np.ndarray | None = None
        for member in self.manifest.get("members", []):
            checkpoint_path = self.manifest_path.parent / member["path"]
            if not checkpoint_path.is_file() or sha256_file(checkpoint_path) != member["sha256"]:
                raise ValueError(f"ensemble member hash mismatch: {checkpoint_path}")
            bundle = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            if bundle.get("nominal_fit_sha256") != nominal_sha:
                raise ValueError("member nominal fit differs from ensemble manifest")
            if self.manifest_schema == 1:
                model = model_type(PhysicsResidualConfig(**bundle["config"]))
            else:
                model = model_type(type(self.config)(**bundle["config"]))
            model.load_state_dict(bundle["state_dict"])
            model.eval()
            member_mean = np.asarray(bundle["feature_mean"], dtype=np.float32)
            member_scale = np.asarray(bundle["feature_scale"], dtype=np.float32)
            if self.mean is None:
                self.mean, self.scale = member_mean, member_scale
            elif not (
                np.allclose(self.mean, member_mean)
                and np.allclose(self.scale, member_scale)
            ):
                raise ValueError("ensemble members use different feature statistics")
            self.models.append(model)
        if len(self.models) != 3:
            raise ValueError("C2 requires exactly three ensemble members")
        calibration = self.manifest.get("calibration", {})
        self.conformal_radius = float(calibration["conformal_radius"])
        self.context_mean = np.asarray(calibration["context_mean"], dtype=float)
        self.context_covariance_inverse = np.asarray(
            calibration["context_covariance_inverse"], dtype=float
        )
        self.ood_threshold = float(calibration["ood_threshold"])
        grip_q90 = calibration.get("grip_absolute_error_q90", [0.0, 0.0])
        self.grip_error_q90 = np.asarray(grip_q90, dtype=float).reshape(-1)
        if self.manifest_schema == 3 and self.grip_error_q90.shape != (2,):
            raise ValueError("grip-aware calibration requires two error quantiles")
        if not (
            math.isfinite(self.conformal_radius)
            and self.conformal_radius > 0.0
            and np.all(np.isfinite(self.context_covariance_inverse))
            and math.isfinite(self.ood_threshold)
        ):
            raise ValueError("calibration artifact is incomplete or non-finite")
        self._warm_up()

    def _warm_up(self) -> None:
        """Pay PyTorch's one-time kernel setup cost before simulation starts."""
        history = np.zeros((self.config.history_steps, self.config.feature_dim), dtype=float)
        matrix_a = np.eye(2, dtype=float)
        matrix_b = np.zeros((2, 1), dtype=float)
        self.predict(history, matrix_a, matrix_b, np.zeros(2), 0.0)

    def predict(
        self,
        history: np.ndarray,
        nominal_a: np.ndarray,
        nominal_b: np.ndarray,
        state: np.ndarray,
        steering: float,
    ) -> dict:
        """Return mean A/B, ensemble predictive std, context and OOD score."""
        normalized = (history - self.mean) / self.scale
        history_tensor = torch.as_tensor(normalized[None], dtype=torch.float32)
        a_tensor = torch.as_tensor(nominal_a[None], dtype=torch.float32)
        b_tensor = torch.as_tensor(nominal_b[None], dtype=torch.float32)
        state_tensor = torch.as_tensor(state[None], dtype=torch.float32)
        steering_tensor = torch.tensor([steering], dtype=torch.float32)
        matrices_a, matrices_b, contexts, predictions, grips = [], [], [], [], []
        if self.manifest_schema == 1:
            with torch.no_grad():
                for model in self.models:
                    matrix_a, matrix_b, context = model(history_tensor, a_tensor, b_tensor)
                    prediction = physical_prediction(
                        matrix_a, matrix_b, state_tensor, steering_tensor
                    )
                    matrices_a.append(matrix_a[0].numpy())
                    matrices_b.append(matrix_b[0].numpy())
                    contexts.append(context[0].numpy())
                    predictions.append(prediction[0].numpy())
        else:
            for model in self.models:
                matrix_a, matrix_b, context, prediction = model.physical_jacobians(
                    history_tensor,
                    state_tensor,
                    steering_tensor,
                    a_tensor,
                    b_tensor,
                )
                matrices_a.append(matrix_a.detach().numpy())
                matrices_b.append(matrix_b.detach().numpy())
                contexts.append(context.detach().numpy())
                predictions.append(prediction.detach().numpy())
                if self.manifest_schema == 3:
                    grip = model.grip_from_context(context.reshape(1, -1))[0]
                    grips.append(grip.detach().numpy())
        context = np.mean(np.asarray(contexts), axis=0)
        residual = context - self.context_mean
        ood_score = float(residual @ self.context_covariance_inverse @ residual)
        prediction_stack = np.asarray(predictions)
        result = {
            "a": np.mean(np.asarray(matrices_a), axis=0),
            "b": np.mean(np.asarray(matrices_b), axis=0),
            "context": context,
            "ensemble_std": float(np.linalg.norm(np.std(prediction_stack, axis=0))),
            "ood_score": ood_score,
        }
        if grips:
            grip_stack = np.asarray(grips)
            result.update(
                {
                    "grip": np.mean(grip_stack, axis=0),
                    "grip_std": np.std(grip_stack, axis=0),
                    "grip_error_q90": self.grip_error_q90.copy(),
                }
            )
        return result


class PhysicsResidualNode(Node):
    """Build causal telemetry history and publish only read-only model messages."""

    def __init__(self):
        super().__init__("neurogrip_physics_residual")
        self.declare_parameter("ensemble_manifest", "")
        self.declare_parameter("nominal_fit", "runs/eufs_v1/models/nominal_fit.json")
        self.declare_parameter("max_input_age_s", 0.06)
        self.declare_parameter("deadline_ms", 18.0)
        self.declare_parameter("inference_period_s", 0.05)
        manifest_value = self.get_parameter("ensemble_manifest").value
        if not manifest_value:
            raise ValueError("ensemble_manifest parameter is required")
        nominal_path = Path(self.get_parameter("nominal_fit").value).expanduser().resolve()
        self.ensemble = PhysicalEnsemble(Path(manifest_value), nominal_path)
        self.nominal = json.loads(nominal_path.read_text(encoding="utf-8"))
        self.max_input_age_s = float(self.get_parameter("max_input_age_s").value)
        self.deadline_ms = float(self.get_parameter("deadline_ms").value)
        self.inference_period_s = float(
            self.get_parameter("inference_period_s").value
        )
        if self.inference_period_s < 0.02:
            raise ValueError("inference_period_s must be at least 0.02 s")
        self.history: deque[np.ndarray] = deque(maxlen=self.ensemble.config.history_steps)
        self.last_sample_stamp: float | None = None
        self.last_inference_stamp: float | None = None
        self.latest_odom: Odometry | None = None
        self.latest_wheels: WheelSpeedsStamped | None = None
        self.latest_command: AckermannDriveStamped | None = None
        self.latest_imu: Imu | None = None
        self.latest_tracking: list[float] | None = None
        self.inference_count = 0
        self.deadline_misses = 0
        self.last_health_detail = ""
        self.latencies_ms: list[float] = []
        self.model_publisher = self.create_publisher(
            DynamicsModel, "/neurogrip/dynamics_model", 10
        )
        self.flat_publisher = self.create_publisher(
            Float64MultiArray, "/neurogrip/dynamics_flat", 10
        )
        self.state_publisher = self.create_publisher(
            VehicleState, "/neurogrip/vehicle_state", 10
        )
        self.safety_publisher = self.create_publisher(
            SafetyStatus, "/neurogrip/safety_status", 10
        )
        self.create_subscription(Odometry, "/odom", self.on_odom, 50)
        self.create_subscription(
            WheelSpeedsStamped, "/ros_can/wheel_speeds", self.on_wheels, 50
        )
        self.create_subscription(
            AckermannDriveStamped, "/neurogrip/command_safe", self.on_command, 50
        )
        self.create_subscription(Imu, "/imu/data", self.on_imu, 50)
        self.create_subscription(
            Float64MultiArray, "/neurogrip/tracking_state", self.on_tracking, 50
        )
        self.create_timer(0.02, self.step)
        # Warm both SciPy's ZOH discretisation and the real-shape neural path.
        # Without this, the first on-track inference pays one-time setup costs
        # inside the controller's real-time window.
        warm_a, warm_b = self.nominal_matrices(3.0)
        warm_history = np.zeros(
            (self.ensemble.config.history_steps, self.ensemble.config.feature_dim),
            dtype=float,
        )
        for _ in range(2):
            self.ensemble.predict(warm_history, warm_a, warm_b, np.zeros(2), 0.0)

    def on_odom(self, message: Odometry) -> None:
        """Cache newest ground-truth vehicle state."""
        self.latest_odom = message

    def on_wheels(self, message: WheelSpeedsStamped) -> None:
        """Cache actual applied steering from EUFS patched wheel telemetry."""
        self.latest_wheels = message

    def on_command(self, message: AckermannDriveStamped) -> None:
        """Cache safe acceleration command used in the causal feature vector."""
        self.latest_command = message

    def on_imu(self, message: Imu) -> None:
        """Cache IMU acceleration features."""
        self.latest_imu = message

    def on_tracking(self, message: Float64MultiArray) -> None:
        """Accept only the documented tracking schema required for timestamp checks."""
        values = list(message.data)
        self.latest_tracking = (
            values
            if len(values) == 10 and int(values[0]) == TRACKING_SCHEMA
            else None
        )

    def publish_safety(
        self, healthy: bool, state: str, detail: str, ood: float, latency: float
    ) -> None:
        """Publish model health used inside the controller's fixed hard limits."""
        message = SafetyStatus()
        message.header.stamp = self.get_clock().now().to_msg()
        message.healthy, message.state, message.detail = healthy, state, detail
        message.uncertainty_radius = float(self.ensemble.conformal_radius)
        message.ood_score, message.latency_ms = float(ood), float(latency)
        message.deadline_miss_rate = float(
            self.deadline_misses / max(self.inference_count, 1)
        )
        self.safety_publisher.publish(message)
        if detail != self.last_health_detail:
            self.get_logger().info(
                f"runtime health={state} detail={detail or 'none'} latency={latency:.2f} ms"
            )
            self.last_health_detail = detail

    def nominal_matrices(self, vx_mps: float) -> tuple[np.ndarray, np.ndarray]:
        """Compute the matching discrete ZOH nominal bicycle model at current speed."""
        from scipy.signal import cont2discrete

        vx = max(vx_mps, 1.0)
        mass = float(self.nominal["mass_kg"])
        inertia = float(self.nominal.get("yaw_inertia_kgm2", 172.44))
        wheelbase = float(self.nominal["wheelbase_m"])
        front_fraction = float(self.nominal["front_fraction"])
        front = float(self.nominal["front_cornering_stiffness_n_rad"])
        rear = float(self.nominal["rear_cornering_stiffness_n_rad"])
        lf, lr = front_fraction * wheelbase, (1.0 - front_fraction) * wheelbase
        matrix_a = np.array(
            [
                [
                    -(front + rear) / (mass * vx),
                    -vx - (lf * front - lr * rear) / (mass * vx),
                ],
                [
                    -(lf * front - lr * rear) / (inertia * vx),
                    -(lf**2 * front + lr**2 * rear) / (inertia * vx),
                ],
            ]
        )
        matrix_b = np.array([[front / mass], [lf * front / inertia]])
        discrete_a, discrete_b, _, _, _ = cont2discrete(
            (matrix_a, matrix_b, np.eye(2), np.zeros((2, 1))), 0.02
        )
        return np.asarray(discrete_a), np.asarray(discrete_b)

    def inputs_valid(self) -> tuple[bool, str, float]:
        """Validate availability, freshness and matching timestamp for one causal sample."""
        inputs = (
            self.latest_odom,
            self.latest_wheels,
            self.latest_command,
            self.latest_imu,
            self.latest_tracking,
        )
        if any(value is None for value in inputs):
            return False, "missing_input", 0.0
        stamp = header_stamp_s(self.latest_odom)
        now = self.get_clock().now().nanoseconds / 1e9
        for label, message in (
            ("odom", self.latest_odom),
            ("wheel_speeds", self.latest_wheels),
            ("safe_command", self.latest_command),
            ("imu", self.latest_imu),
        ):
            age = now - header_stamp_s(message)
            if not 0.0 <= age <= self.max_input_age_s:
                return False, f"stale_{label}", stamp
        if abs(float(self.latest_tracking[1]) - stamp) > 0.04:
            return False, "tracking_timestamp_mismatch", stamp
        if self.last_sample_stamp is not None and stamp <= self.last_sample_stamp:
            return False, "non_monotonic_or_duplicate", stamp
        if self.last_sample_stamp is not None and stamp - self.last_sample_stamp > 0.06:
            self.history.clear()
            self.last_sample_stamp = stamp
            return False, "history_gap", stamp
        return True, "", stamp

    def step(self) -> None:
        """Append one sample and publish C2 A/B only once the causal buffer is full."""
        valid, reason, stamp = self.inputs_valid()
        if not valid:
            # A 50 Hz timer and 50 Hz odometry are not phase locked. Seeing the
            # same odometry stamp twice is normal and must not create an unsafe
            # state or reset the learned-model hold in the controller.
            if reason == "non_monotonic_or_duplicate":
                return
            self.publish_safety(False, "stale", reason, 0.0, 0.0)
            return
        odom = self.latest_odom
        wheels = self.latest_wheels
        command = self.latest_command
        imu = self.latest_imu
        twist = odom.twist.twist
        body_vx = float(twist.linear.x)
        body_vy = float(twist.linear.y)
        feature = np.asarray([
            body_vx, body_vy, twist.angular.z, wheels.speeds.steering,
            command.drive.acceleration, imu.linear_acceleration.x, imu.linear_acceleration.y,
        ], dtype=np.float32)
        if not np.all(np.isfinite(feature)):
            self.publish_safety(False, "error", "non_finite_input", 0.0, 0.0)
            return
        self.history.append(feature)
        self.last_sample_stamp = stamp
        if len(self.history) < self.ensemble.config.history_steps:
            self.publish_safety(False, "warming_up", "insufficient_history", 0.0, 0.0)
            return
        # Preserve the 50 Hz causal history used during training, but evaluate
        # the comparatively expensive ensemble at 20 Hz. The controller holds
        # the last validated model for at most 200 ms, so 50 ms publications
        # retain ample freshness while freeing CPU for the 50 Hz MPC QP.
        if (
            self.last_inference_stamp is not None
            and stamp - self.last_inference_stamp < self.inference_period_s - 1e-6
        ):
            return
        self.last_inference_stamp = stamp
        started = time.perf_counter()
        nominal_a, nominal_b = self.nominal_matrices(float(feature[0]))
        result = self.ensemble.predict(
            np.stack(self.history), nominal_a, nominal_b, feature[1:3], float(feature[3])
        )
        latency = (time.perf_counter() - started) * 1000.0
        self.inference_count += 1
        self.latencies_ms.append(float(latency))
        if latency > self.deadline_ms:
            self.deadline_misses += 1
        stable = stable_enough(result["a"])
        healthy = stable and result["ood_score"] <= self.ensemble.ood_threshold
        reason = "" if healthy else ("unstable_matrix" if not stable else "ood_threshold")
        message = DynamicsModel()
        message.header.stamp = odom.header.stamp
        message.header.frame_id = "base_footprint"
        message.schema_version = self.ensemble.dynamics_schema
        message.artifact_sha256 = self.ensemble.artifact_sha256
        message.sample_time_s = 0.02
        message.state_dim, message.input_dim = 2, 1
        message.state_names = ["v_y_mps", "yaw_rate_rps"]
        message.input_names = ["steering_angle_rad"]
        message.a_matrix = result["a"].ravel().tolist()
        message.b_matrix = result["b"].ravel().tolist()
        message.context = result["context"].tolist()
        message.ensemble_std = float(result["ensemble_std"])
        message.conformal_radius = float(self.ensemble.conformal_radius)
        message.ood_score = float(result["ood_score"])
        message.latency_ms = float(latency)
        if self.ensemble.dynamics_schema == 4:
            message.estimated_front_grip = float(result["grip"][0])
            message.estimated_rear_grip = float(result["grip"][1])
            message.front_grip_std = float(result["grip_std"][0])
            message.rear_grip_std = float(result["grip_std"][1])
            message.front_grip_error_q90 = float(result["grip_error_q90"][0])
            message.rear_grip_error_q90 = float(result["grip_error_q90"][1])
        else:
            message.estimated_front_grip = 1.0
            message.estimated_rear_grip = 1.0
        message.valid = healthy
        message.invalid_reason = reason
        self.model_publisher.publish(message)
        flat = Float64MultiArray()
        flat_values = [
            float(self.ensemble.dynamics_schema),
            0.02,
            float(healthy),
            *message.a_matrix,
            *message.b_matrix,
            message.ensemble_std,
            message.conformal_radius,
            message.ood_score,
            message.latency_ms,
        ]
        if self.ensemble.dynamics_schema == 4:
            flat_values.extend(
                [
                    message.estimated_front_grip,
                    message.estimated_rear_grip,
                    message.front_grip_std,
                    message.rear_grip_std,
                    message.front_grip_error_q90,
                    message.rear_grip_error_q90,
                ]
            )
        flat.data = flat_values
        self.flat_publisher.publish(flat)
        state = VehicleState()
        state.header.stamp = odom.header.stamp
        state.header.frame_id = "base_footprint"
        state.v_x_mps = float(feature[0])
        state.v_y_mps = float(feature[1])
        state.yaw_rate_rps = float(feature[2])
        state.a_x_mps2 = float(feature[5])
        state.a_y_mps2 = float(feature[6])
        state.steering_applied_rad = float(feature[3])
        state.acceleration_command_mps2, state.source = float(feature[4]), 0
        self.state_publisher.publish(state)
        self.publish_safety(
            healthy,
            "ok" if healthy else "out_of_distribution",
            reason,
            result["ood_score"],
            latency,
        )

    def destroy_node(self):
        """Report runtime latency distribution before a normal launch teardown."""
        if self.latencies_ms:
            latencies = np.asarray(self.latencies_ms, dtype=float)
            self.get_logger().info(
                "C2 runtime summary: samples=%d median=%.2fms p95=%.2fms misses=%d"
                % (
                    len(latencies),
                    float(np.median(latencies)),
                    float(np.percentile(latencies, 95.0)),
                    self.deadline_misses,
                )
            )
        return super().destroy_node()


def main(args=None) -> None:
    """Run the non-actuating physics-residual inference node."""
    rclpy.init(args=args)
    node = PhysicsResidualNode()
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
