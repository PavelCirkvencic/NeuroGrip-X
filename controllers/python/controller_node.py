#!/usr/bin/env python3
"""Fair C0/C1/C2 Python lateral MPC controller.

Every benchmark controller shares the same track, costs, safety limits,
longitudinal scheduler and nominal physical prior.  Only the lateral model
given to the QP changes: fixed nominal (C0), a labelled scalar oracle (C1), or
a fresh physical model published by the read-only NeuroGrip AI node (C2).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import rclpy  # noqa: E402
from ackermann_msgs.msg import AckermannDriveStamped  # noqa: E402
from eufs_msgs.msg import ConeWithColorProbabilityArray, WheelSpeedsStamped  # noqa: E402
from nav_msgs.msg import Odometry  # noqa: E402
from neurogrip_interfaces.msg import DynamicsModel  # noqa: E402
from neurogrip_control.track_safety import SafetyLapGate, TrackEnvelope  # noqa: E402
from rclpy.node import Node  # noqa: E402
from rclpy.qos import DurabilityPolicy, QoSProfile  # noqa: E402
from std_msgs.msg import Float64, Float64MultiArray, String  # noqa: E402

from bicycle_model import (  # noqa: E402
    VehicleParameters,
    build_tracking_plant,
    build_tracking_plant_from_discrete,
)
from longitudinal_pi import LongitudinalPI  # noqa: E402
from mpc_qp import LateralMpc  # noqa: E402
from reference_preview import ReferencePreview  # noqa: E402
from grip_policy import (  # noqa: E402
    conservative_grip_policy,
    grip_is_observable,
    update_filtered_grip,
)
from speed_profile import profile_from_npz, tracking_safety_speed_scale  # noqa: E402

TRACKING_SCHEMA = 1
DYNAMICS_SCHEMAS = (2, 3, 4)
DIAGNOSTICS_SCHEMA = 2
MATLAB_PARITY_SCHEMA = 3
MATLAB_PARITY_FRAME_LENGTH = 155
MATLAB_CONTROL_SCHEMA = 1
MATLAB_CONTROL_FRAME_LENGTH = 125


def apply_startup_speed_envelope(
    target_speed_mps: float,
    *,
    experiment_started: bool,
    cumulative_progress: float,
    cap_mps: float,
    end_progress: float,
) -> float:
    """Apply the controller-neutral launch cap until the first sector ends."""
    in_startup_sector = (
        not experiment_started or cumulative_progress < end_progress
    )
    return min(target_speed_mps, cap_mps) if in_startup_sector else target_speed_mps


def build_matlab_control_frame(
    *,
    control_time_s: float,
    sample_time_s: float,
    vx_mps: float,
    previous_steering_rad: float,
    state: np.ndarray,
    a_tracking: np.ndarray,
    b_tracking: np.ndarray,
    curvature_preview: np.ndarray,
    left_corridor_preview: np.ndarray,
    right_corridor_preview: np.ndarray,
    first_move_rad: float,
    solution_valid: bool,
) -> list[float]:
    """Pack the exact QP inputs used for continuous MATLAB candidate output."""
    arrays = {
        "state": (state, 4),
        "a_tracking": (a_tracking, 16),
        "b_tracking": (b_tracking, 8),
        "curvature_preview": (curvature_preview, 30),
        "left_corridor_preview": (left_corridor_preview, 30),
        "right_corridor_preview": (right_corridor_preview, 30),
    }
    flattened: dict[str, np.ndarray] = {}
    for name, (value, expected_size) in arrays.items():
        flat = np.asarray(value, dtype=float).reshape(-1)
        if flat.size != expected_size or np.any(~np.isfinite(flat)):
            raise ValueError(f"{name} must contain {expected_size} finite values")
        flattened[name] = flat
    scalars = np.asarray(
        [
            control_time_s,
            sample_time_s,
            vx_mps,
            previous_steering_rad,
            first_move_rad,
        ],
        dtype=float,
    )
    if np.any(~np.isfinite(scalars)):
        raise ValueError("MATLAB control scalars must be finite")
    values = [
        float(MATLAB_CONTROL_SCHEMA),
        float(control_time_s),
        float(sample_time_s),
        float(vx_mps),
        float(previous_steering_rad),
        *flattened["state"],
        *flattened["a_tracking"],
        *flattened["b_tracking"],
        *flattened["curvature_preview"],
        *flattened["left_corridor_preview"],
        *flattened["right_corridor_preview"],
        float(first_move_rad),
        float(bool(solution_valid)),
    ]
    if len(values) != MATLAB_CONTROL_FRAME_LENGTH:
        raise RuntimeError("internal MATLAB control frame length mismatch")
    return [float(value) for value in values]


def build_matlab_parity_frame(
    *,
    control_time_s: float,
    sample_time_s: float,
    raw_grip: np.ndarray,
    raw_grip_std: np.ndarray,
    raw_grip_error_q90: np.ndarray,
    filtered_grip: np.ndarray,
    held_grip_std: np.ndarray,
    held_grip_error_q90: np.ndarray,
    conservative_grip: np.ndarray,
    effective_grip: float,
    profile_utilisation: float,
    grip_error_margin_scale: float,
    target_speed_mps: float,
    vx_mps: float,
    lap_progress: float,
    maximum_speed_mps: float,
    minimum_speed_mps: float,
    acceleration_limit_mps2: float,
    braking_limit_mps2: float,
    model_blend: float,
    previous_steering_rad: float,
    state: np.ndarray,
    raw_a_lateral: np.ndarray,
    raw_b_lateral: np.ndarray,
    a_tracking: np.ndarray,
    b_tracking: np.ndarray,
    curvature_preview: np.ndarray,
    left_corridor_preview: np.ndarray,
    right_corridor_preview: np.ndarray,
    first_move_rad: float,
    solution_valid: bool,
) -> list[float]:
    """Pack one exact, read-only controller decision for MATLAB replay."""
    arrays = {
        "raw_grip": (raw_grip, 2),
        "raw_grip_std": (raw_grip_std, 2),
        "raw_grip_error_q90": (raw_grip_error_q90, 2),
        "filtered_grip": (filtered_grip, 2),
        "held_grip_std": (held_grip_std, 2),
        "held_grip_error_q90": (held_grip_error_q90, 2),
        "conservative_grip": (conservative_grip, 2),
        "state": (state, 4),
        "raw_a_lateral": (raw_a_lateral, 4),
        "raw_b_lateral": (raw_b_lateral, 2),
        "a_tracking": (a_tracking, 16),
        "b_tracking": (b_tracking, 8),
        "curvature_preview": (curvature_preview, 30),
        "left_corridor_preview": (left_corridor_preview, 30),
        "right_corridor_preview": (right_corridor_preview, 30),
    }
    flattened: dict[str, np.ndarray] = {}
    for name, (value, expected_size) in arrays.items():
        flat = np.asarray(value, dtype=float).reshape(-1)
        if flat.size != expected_size or np.any(~np.isfinite(flat)):
            raise ValueError(f"{name} must contain {expected_size} finite values")
        flattened[name] = flat
    scalars = np.asarray(
        [
            control_time_s,
            sample_time_s,
            effective_grip,
            profile_utilisation,
            grip_error_margin_scale,
            target_speed_mps,
            vx_mps,
            lap_progress,
            maximum_speed_mps,
            minimum_speed_mps,
            acceleration_limit_mps2,
            braking_limit_mps2,
            model_blend,
            previous_steering_rad,
            first_move_rad,
        ],
        dtype=float,
    )
    if np.any(~np.isfinite(scalars)):
        raise ValueError("MATLAB parity scalars must be finite")
    values = [
        float(MATLAB_PARITY_SCHEMA),
        float(control_time_s),
        float(sample_time_s),
        *flattened["raw_grip"],
        *flattened["raw_grip_std"],
        *flattened["raw_grip_error_q90"],
        *flattened["filtered_grip"],
        *flattened["held_grip_std"],
        *flattened["held_grip_error_q90"],
        *flattened["conservative_grip"],
        float(effective_grip),
        float(profile_utilisation),
        float(grip_error_margin_scale),
        float(target_speed_mps),
        float(vx_mps),
        float(lap_progress),
        float(maximum_speed_mps),
        float(minimum_speed_mps),
        float(acceleration_limit_mps2),
        float(braking_limit_mps2),
        float(model_blend),
        float(previous_steering_rad),
        *flattened["state"],
        *flattened["raw_a_lateral"],
        *flattened["raw_b_lateral"],
        *flattened["a_tracking"],
        *flattened["b_tracking"],
        *flattened["curvature_preview"],
        *flattened["left_corridor_preview"],
        *flattened["right_corridor_preview"],
        float(first_move_rad),
        float(bool(solution_valid)),
    ]
    if len(values) != MATLAB_PARITY_FRAME_LENGTH:
        raise RuntimeError("internal MATLAB parity frame length mismatch")
    return [float(value) for value in values]


def blend_models(
    nominal_a: np.ndarray,
    nominal_b: np.ndarray,
    learned_a: np.ndarray,
    learned_b: np.ndarray,
    learned_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Shrink a learned local model toward the common physical prior."""
    fraction = float(learned_fraction)
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("learned model fraction must be in [0, 1]")
    return (
        nominal_a + fraction * (learned_a - nominal_a),
        nominal_b + fraction * (learned_b - nominal_b),
    )


def load_nominal_parameters(path: str) -> VehicleParameters:
    """Load the single physical nominal model used by every controller."""
    fit_path = _Path(path).expanduser().resolve()
    if not fit_path.is_file():
        raise FileNotFoundError(f"nominal fit missing: {fit_path}")
    fit = json.loads(fit_path.read_text(encoding="utf-8"))
    names = ("front_cornering_stiffness_n_rad", "rear_cornering_stiffness_n_rad")
    if any(name not in fit for name in names):
        raise ValueError(f"{fit_path} does not contain front/rear stiffness")
    front_stiffness, rear_stiffness = (float(fit[name]) for name in names)
    if not all(
        math.isfinite(value) and value > 0.0
        for value in (front_stiffness, rear_stiffness)
    ):
        raise ValueError("nominal stiffnesses must be finite and positive")
    return VehicleParameters(
        mass_kg=float(fit.get("mass_kg", 300.0)),
        wheelbase_m=float(fit.get("wheelbase_m", 1.53)),
        front_axle_fraction=float(fit.get("front_fraction", 0.5)),
        front_cornering_stiffness_n_rad=front_stiffness,
        rear_cornering_stiffness_n_rad=rear_stiffness,
    )


def dynamic_model_is_usable(
    message: DynamicsModel | None, now_s: float, max_age_s: float = 0.08
) -> bool:
    """Accept only a fresh physical v2 model with valid dimensions/values."""
    if message is None or not message.valid or message.schema_version not in DYNAMICS_SCHEMAS:
        return False
    if message.state_dim != 2 or message.input_dim != 1:
        return False
    if len(message.a_matrix) != 4 or len(message.b_matrix) != 2:
        return False
    stamp_s = message.header.stamp.sec + message.header.stamp.nanosec * 1e-9
    values = [*message.a_matrix, *message.b_matrix, message.sample_time_s]
    if not all(math.isfinite(float(value)) for value in values):
        return False
    if abs(float(message.sample_time_s) - 0.02) > 1e-6:
        return False
    return 0.0 <= now_s - stamp_s <= max_age_s


def grip_model_is_usable(
    message: DynamicsModel | None, now_s: float, max_age_s: float = 0.20
) -> bool:
    """Accept only a fresh schema-4 model with bounded per-axle grip outputs."""
    if not dynamic_model_is_usable(message, now_s, max_age_s):
        return False
    assert message is not None
    if message.schema_version != 4:
        return False
    values = (
        message.estimated_front_grip,
        message.estimated_rear_grip,
        message.front_grip_std,
        message.rear_grip_std,
        message.front_grip_error_q90,
        message.rear_grip_error_q90,
    )
    return (
        all(math.isfinite(float(value)) for value in values)
        and 0.35 <= message.estimated_front_grip <= 1.30
        and 0.35 <= message.estimated_rear_grip <= 1.30
        and min(values[2:]) >= 0.0
    )


def uncertainty_speed_cap(
    baseline_mps: float,
    adaptive_max_mps: float,
    model: DynamicsModel | None,
    now_s: float,
) -> tuple[float, float]:
    """Return a bounded C2 speed cap and its calibrated confidence in [0, 1].

    The fixed baseline is always available. C2 may use the extra envelope only
    while the newest model is fresh and valid; ensemble disagreement consumes
    the conformal uncertainty budget continuously instead of acting as a
    brittle on/off switch.
    """
    baseline = float(baseline_mps)
    maximum = max(baseline, float(adaptive_max_mps))
    if not dynamic_model_is_usable(model, now_s):
        return baseline, 0.0
    assert model is not None
    radius = float(model.conformal_radius)
    disagreement = float(model.ensemble_std)
    if not math.isfinite(radius) or radius <= 0.0 or not math.isfinite(disagreement):
        return baseline, 0.0
    confidence = float(np.clip(1.0 - max(disagreement, 0.0) / radius, 0.0, 1.0))
    return baseline + confidence * (maximum - baseline), confidence


class ControllerNode(Node):
    """Track one reference lap and publish a command candidate at 50 Hz."""

    def __init__(self, arguments: argparse.Namespace):
        super().__init__("neurogrip_python_controller")
        self.declare_parameter("track_npz", arguments.track_npz)
        self.declare_parameter("controller_id", arguments.controller)
        self.declare_parameter("target_speed_max_mps", arguments.target_speed)
        self.declare_parameter("target_speed_min_mps", arguments.target_speed_min)
        self.declare_parameter(
            "c2_adaptive_speed_max_mps", arguments.c2_adaptive_speed_max
        )
        self.declare_parameter("curvature_speed_gain", 6.0)
        self.declare_parameter("scalar_mu_utilisation", 0.78)
        self.declare_parameter("fixed_mu", arguments.fixed_mu)
        self.declare_parameter("neurogrip_utilisation", 0.95)
        self.declare_parameter("grip_error_margin_scale", 0.50)
        self.declare_parameter("fallback_mu", 0.65)
        self.declare_parameter("profile_acceleration_limit_mps2", 3.0)
        self.declare_parameter("profile_braking_limit_mps2", 2.5)
        self.declare_parameter("startup_speed_cap_mps", 8.0)
        self.declare_parameter("startup_speed_cap_end_progress", 0.12)
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("wheel_speeds_topic", "/ros_can/wheel_speeds")
        self.declare_parameter(
            "candidate_topic", "/neurogrip/command_candidate/python"
        )
        self.declare_parameter("lap_count", arguments.laps)
        self.declare_parameter("nominal_fit", arguments.nominal_fit)
        # The learned local model is normally refreshed at 50 Hz. A 200 ms
        # bound absorbs a short calibration/OOD rejection burst while staying
        # short relative to vehicle dynamics; a longer gap reverts to C0.
        self.declare_parameter("dynamics_max_age_s", 0.20)
        self.declare_parameter("c2_model_blend", 0.02)
        self.declare_parameter("vehicle_half_width_m", 0.70)
        self.declare_parameter("vehicle_half_length_m", 1.30)

        self.controller_id = self.get_parameter("controller_id").value
        if self.controller_id not in {"C0_FIXED", "C1_ORACLE_MU", "C2_NEUROGRIP"}:
            raise ValueError(f"unsupported controller: {self.controller_id}")
        self.target_speed_max = float(self.get_parameter("target_speed_max_mps").value)
        self.target_speed_min = float(self.get_parameter("target_speed_min_mps").value)
        requested_adaptive_max = float(
            self.get_parameter("c2_adaptive_speed_max_mps").value
        )
        self.c2_adaptive_speed_max = (
            self.target_speed_max
            if requested_adaptive_max < 0.0
            else max(self.target_speed_max, requested_adaptive_max)
        )
        self.curvature_gain = float(self.get_parameter("curvature_speed_gain").value)
        self.scalar_mu_utilisation = float(
            self.get_parameter("scalar_mu_utilisation").value
        )
        self.fixed_mu = float(self.get_parameter("fixed_mu").value)
        self.neurogrip_utilisation = float(
            self.get_parameter("neurogrip_utilisation").value
        )
        self.grip_error_margin_scale = float(
            self.get_parameter("grip_error_margin_scale").value
        )
        self.fallback_mu = float(self.get_parameter("fallback_mu").value)
        self.profile_acceleration_limit = float(
            self.get_parameter("profile_acceleration_limit_mps2").value
        )
        self.profile_braking_limit = float(
            self.get_parameter("profile_braking_limit_mps2").value
        )
        self.startup_speed_cap = float(
            self.get_parameter("startup_speed_cap_mps").value
        )
        self.startup_speed_cap_end_progress = float(
            self.get_parameter("startup_speed_cap_end_progress").value
        )
        if not (
            self.target_speed_min <= self.startup_speed_cap <= self.target_speed_max
            and 0.0 <= self.startup_speed_cap_end_progress <= 0.25
        ):
            raise ValueError("invalid shared startup speed envelope")
        if not (
            0.0 < self.scalar_mu_utilisation <= 1.0
            and 0.0 < self.neurogrip_utilisation <= 1.0
            and 0.0 <= self.grip_error_margin_scale <= 1.0
            and 0.35 <= self.fixed_mu <= 1.30
            and 0.35 <= self.fallback_mu <= 1.30
        ):
            raise ValueError("invalid speed-profile grip policy")
        self.lap_count = int(self.get_parameter("lap_count").value)
        self.dynamics_max_age_s = float(self.get_parameter("dynamics_max_age_s").value)
        self.c2_model_blend = float(self.get_parameter("c2_model_blend").value)
        if not 0.0 <= self.c2_model_blend <= 1.0:
            raise ValueError("c2_model_blend must be in [0, 1]")
        self.nominal_parameters = load_nominal_parameters(
            self.get_parameter("nominal_fit").value
        )
        self.lap_gate = SafetyLapGate(
            TrackEnvelope.from_npz(self.get_parameter("track_npz").value),
            vehicle_half_width_m=float(
                self.get_parameter("vehicle_half_width_m").value
            ),
            vehicle_half_length_m=float(
                self.get_parameter("vehicle_half_length_m").value
            ),
        )
        self.reference_preview = ReferencePreview.from_npz(
            self.get_parameter("track_npz").value
        )
        self.track_npz = str(self.get_parameter("track_npz").value)
        self.sample_time_s = float(arguments.sample_time)

        self.mpc = LateralMpc(horizon=arguments.horizon)
        self.mpc_warm_start: np.ndarray | None = None
        self.longitudinal = LongitudinalPI()
        self.applied_steering = 0.0
        self.latest_odom: Odometry | None = None
        self.latest_tracking: list[float] | None = None
        self.latest_dynamics: DynamicsModel | None = None
        # A single invalid OOD publication must not make the controller switch
        # models at 50 Hz.  Keep only the most recent *validated* model and
        # permit it for the same short freshness interval used everywhere else.
        # Persistent invalid/missing publications still fall back to C0 after
        # 200 ms; this is a bounded hold, never reuse of an old learned model.
        self.last_valid_dynamics: DynamicsModel | None = None
        self.oracle_mu = 1.0
        self.done = False
        self.collision_count = 0
        self.consecutive_solver_failures = 0
        self.c2_fallback_count = 0
        self.c2_was_in_fallback = False
        self.model_confidence = 0.0
        # The pre-observability lateral plant and speed policy must share one
        # conservative prior. Starting the plant at 1.0 while the speed policy
        # used fallback_mu produced an avoidable model switch and steering
        # oscillation before the learned grip estimate became observable.
        self.filtered_front_grip = self.fallback_mu
        self.filtered_rear_grip = self.fallback_mu
        self.filtered_front_std = 0.0
        self.filtered_rear_std = 0.0
        self.front_grip_error_q90 = 0.0
        self.rear_grip_error_q90 = 0.0
        self.grip_estimate_ready = False
        self.grip_filter_alpha = 0.05
        self.speed_profile = None
        self.speed_profile_key: tuple[float, float] | None = None
        self.profile_grip = self.fallback_mu
        self.profile_utilisation = self.scalar_mu_utilisation
        self.profile_front_grip = self.fallback_mu
        self.profile_rear_grip = self.fallback_mu

        self.candidate_publisher = self.create_publisher(
            AckermannDriveStamped,
            str(self.get_parameter("candidate_topic").value),
            10,
        )
        status_qos = QoSProfile(depth=10, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.status_publisher = self.create_publisher(
            String, "/neurogrip/controller_status", status_qos
        )
        self.diagnostics_publisher = self.create_publisher(
            Float64MultiArray, "/neurogrip/controller_diagnostics", 10
        )
        self.matlab_parity_publisher = self.create_publisher(
            Float64MultiArray, "/neurogrip/matlab_parity_frame", 10
        )
        self.matlab_control_publisher = self.create_publisher(
            Float64MultiArray, "/neurogrip/matlab_control_frame", 10
        )
        self.create_subscription(
            Odometry, self.get_parameter("odom_topic").value, self.on_odom, 50
        )
        self.create_subscription(
            Float64MultiArray, "/neurogrip/tracking_state", self.on_tracking, 50
        )
        self.create_subscription(
            WheelSpeedsStamped,
            self.get_parameter("wheel_speeds_topic").value,
            self.on_wheel_speeds,
            50,
        )
        self.create_subscription(
            ConeWithColorProbabilityArray,
            "/plugin/cone_collision_tracker/colliding_cones",
            self.on_collisions,
            10,
        )
        if self.controller_id == "C1_ORACLE_MU":
            self.create_subscription(Float64, "/neurogrip/oracle_mu", self.on_mu, 10)
        if self.controller_id == "C2_NEUROGRIP":
            self.create_subscription(
                DynamicsModel, "/neurogrip/dynamics_model", self.on_dynamics, 10
            )
        self.create_timer(arguments.sample_time, self.control_step)
        self.get_logger().info(
            f"{self.controller_id}: target={self.target_speed_max:.1f} m/s; "
            f"shared nominal Cf={self.nominal_parameters.front_cornering_stiffness_n_rad:.1f}, "
            f"Cr={self.nominal_parameters.rear_cornering_stiffness_n_rad:.1f}"
        )

    def on_odom(self, message: Odometry) -> None:
        """Cache the newest EUFS odometry sample."""
        self.latest_odom = message

    def on_wheel_speeds(self, message: WheelSpeedsStamped) -> None:
        """Use measured steering, never requested steering, for the rate constraint."""
        self.applied_steering = float(message.speeds.steering)

    def on_collisions(self, message: ConeWithColorProbabilityArray) -> None:
        """Cache the current authoritative EUFS cone-collision count."""
        self.collision_count = len(message.cones)

    def on_tracking(self, message: Float64MultiArray) -> None:
        """Cache the independent, versioned track-projection state."""
        values = list(message.data)
        self.latest_tracking = (
            values
            if len(values) == 10 and int(values[0]) == TRACKING_SCHEMA
            else None
        )

    def on_mu(self, message: Float64) -> None:
        """Cache the C1-only scalar ground-truth oracle."""
        self.oracle_mu = float(message.data)

    def on_dynamics(self, message: DynamicsModel) -> None:
        """Cache diagnostic input and retain only fresh-schema candidates."""
        self.latest_dynamics = message
        if (
            message.valid
            and message.schema_version in DYNAMICS_SCHEMAS
            and message.state_dim == 2
            and message.input_dim == 1
            and len(message.a_matrix) == 4
            and len(message.b_matrix) == 2
            and abs(float(message.sample_time_s) - 0.02) <= 1e-6
        ):
            self.last_valid_dynamics = message
            if message.schema_version == 4 and self.latest_tracking is not None:
                vx = float(self.latest_tracking[6])
                observable = grip_is_observable(
                    vx,
                    float(self.latest_tracking[5]),
                    self.applied_steering,
                )
                if observable:
                    alpha = self.grip_filter_alpha
                    self.filtered_front_grip, self.filtered_rear_grip = (
                        update_filtered_grip(
                            self.filtered_front_grip,
                            self.filtered_rear_grip,
                            float(message.estimated_front_grip),
                            float(message.estimated_rear_grip),
                            alpha,
                        )
                    )
                    self.filtered_front_std = float(message.front_grip_std)
                    self.filtered_rear_std = float(message.rear_grip_std)
                    self.front_grip_error_q90 = float(
                        message.front_grip_error_q90
                    )
                    self.rear_grip_error_q90 = float(message.rear_grip_error_q90)
                    self.grip_estimate_ready = True

    def publish_event(self, event: str, **details: object) -> None:
        """Publish a timestamped lifecycle event that the recorder can persist."""
        message = String()
        message.data = json.dumps(
            {
                "event": event,
                "controller": self.controller_id,
                "sim_time_s": self.get_clock().now().nanoseconds / 1e9,
                **details,
            },
            sort_keys=True,
        )
        self.status_publisher.publish(message)

    def model_for_step(self, vx_mps: float) -> tuple[np.ndarray, np.ndarray, bool]:
        """Return the selected tracking plant and whether C2 used safe fallback."""
        parameters = self.nominal_parameters
        if self.controller_id == "C0_FIXED":
            parameters = VehicleParameters(
                **{
                    **parameters.__dict__,
                    "front_cornering_stiffness_n_rad": (
                        parameters.front_cornering_stiffness_n_rad * self.fixed_mu
                    ),
                    "rear_cornering_stiffness_n_rad": (
                        parameters.rear_cornering_stiffness_n_rad * self.fixed_mu
                    ),
                }
            )
        if self.controller_id == "C1_ORACLE_MU":
            mu = float(np.clip(self.oracle_mu, 0.35, 1.30))
            parameters = VehicleParameters(
                **{
                    **parameters.__dict__,
                    "front_cornering_stiffness_n_rad": (
                        parameters.front_cornering_stiffness_n_rad * mu
                    ),
                    "rear_cornering_stiffness_n_rad": (
                        parameters.rear_cornering_stiffness_n_rad * mu
                    ),
                }
            )
        if self.controller_id == "C2_NEUROGRIP":
            now_s = self.get_clock().now().nanoseconds / 1e9
            adaptive_parameters = VehicleParameters(
                **{
                    **parameters.__dict__,
                    "front_cornering_stiffness_n_rad": (
                        parameters.front_cornering_stiffness_n_rad
                        * self.filtered_front_grip
                    ),
                    "rear_cornering_stiffness_n_rad": (
                        parameters.rear_cornering_stiffness_n_rad
                        * self.filtered_rear_grip
                    ),
                }
            )
            nominal_a, nominal_b = build_tracking_plant(vx_mps, adaptive_parameters)
            if dynamic_model_is_usable(
                self.last_valid_dynamics, now_s, self.dynamics_max_age_s
            ):
                model = self.last_valid_dynamics
                assert model is not None  # narrowed by dynamic_model_is_usable
                matrix_a = np.asarray(model.a_matrix, dtype=float).reshape(2, 2)
                matrix_b = np.asarray(model.b_matrix, dtype=float).reshape(2, 1)
                learned_a, learned_b = build_tracking_plant_from_discrete(
                    vx_mps, matrix_a, matrix_b, float(model.sample_time_s)
                )
                blended_a, blended_b = blend_models(
                    nominal_a,
                    nominal_b,
                    learned_a,
                    learned_b,
                    self.c2_model_blend,
                )
                return blended_a, blended_b, False
            self.c2_fallback_count += 1
            if self.c2_fallback_count == 1:
                self.get_logger().warning(
                    "C2 dynamics unavailable; using safe C0 nominal fallback"
                )
            return nominal_a, nominal_b, True
        return (*build_tracking_plant(vx_mps, parameters), False)

    def speed_policy(self, now_s: float) -> tuple[float, float, float, float]:
        """Return friction, utilisation and axle estimates without hidden C2 inputs."""
        if self.controller_id == "C0_FIXED":
            return (
                self.fixed_mu,
                self.scalar_mu_utilisation,
                self.fixed_mu,
                self.fixed_mu,
            )
        if self.controller_id == "C1_ORACLE_MU":
            scalar = float(np.clip(self.oracle_mu, 0.35, 1.30))
            return scalar, self.scalar_mu_utilisation, scalar, scalar
        if not self.grip_estimate_ready:
            return (
                self.fallback_mu,
                self.scalar_mu_utilisation,
                self.fallback_mu,
                self.fallback_mu,
            )
        grip, front, rear = conservative_grip_policy(
            self.filtered_front_grip,
            self.filtered_rear_grip,
            self.filtered_front_std,
            self.filtered_rear_std,
            self.front_grip_error_q90,
            self.rear_grip_error_q90,
            self.grip_error_margin_scale,
        )
        return grip, self.neurogrip_utilisation, front, rear

    def target_speed_for_progress(self, progress: float, now_s: float) -> float:
        """Evaluate a periodically feasible speed profile shared by all controllers."""
        grip, utilisation, front, rear = self.speed_policy(now_s)
        key = (round(grip, 2), round(utilisation, 3))
        if self.speed_profile is None or key != self.speed_profile_key:
            self.speed_profile = profile_from_npz(
                self.track_npz,
                friction_coefficient=grip,
                lateral_utilisation=utilisation,
                maximum_speed_mps=self.target_speed_max,
                minimum_speed_mps=self.target_speed_min,
                acceleration_limit_mps2=self.profile_acceleration_limit,
                braking_limit_mps2=self.profile_braking_limit,
            )
            self.speed_profile_key = key
        self.profile_grip = grip
        self.profile_utilisation = utilisation
        self.profile_front_grip = front
        self.profile_rear_grip = rear
        return self.speed_profile.at_progress(progress)

    def control_step(self) -> None:
        """Project, solve the QP and publish exactly one candidate command."""
        if self.done or self.latest_odom is None or self.latest_tracking is None:
            return
        tracking = self.latest_tracking
        progress = float(tracking[9])
        gate = self.lap_gate.update(
            lap_progress=progress,
            lateral_error_m=float(tracking[2]),
            heading_error_rad=float(tracking[3]),
            speed_mps=float(tracking[6]),
            collision_count=self.collision_count,
        )
        if gate.event == "experiment_start":
            self.publish_event(
                "experiment_start",
                boundary_margin_m=gate.boundary_margin_m,
            )
            if self.controller_id == "C2_NEUROGRIP":
                self.publish_event(
                    "c2_grip_policy",
                    shared_maximum_speed_mps=self.target_speed_max,
                    lateral_utilisation=self.neurogrip_utilisation,
                    uncertainty_rule="estimate - ensemble_std - margin_scale*q90_error",
                    grip_error_margin_scale=self.grip_error_margin_scale,
                    model_blend=self.c2_model_blend,
                )
        elif gate.event == "safety_violation":
            self.publish_event(
                "safety_violation",
                reason=gate.reason,
                boundary_margin_m=gate.boundary_margin_m,
                collision_count=self.collision_count,
                cumulative_progress=gate.cumulative_progress,
            )
            self.done = True
            self.publish_zero()
            return
        elif gate.event == "lap_complete":
            self.get_logger().info(f"safety-valid lap {gate.laps_completed} completed")
            self.publish_event(
                "lap_complete",
                lap=gate.laps_completed,
                safety_valid=True,
                boundary_margin_m=gate.boundary_margin_m,
            )
            if gate.laps_completed >= self.lap_count:
                self.done = True
                self.publish_zero()
                return

        vx = max(float(tracking[6]), 1.0)
        curvature_preview, left_corridor, right_corridor = (
            self.reference_preview.horizon(
                progress,
                vx,
                self.sample_time_s,
                self.mpc.horizon,
                body_clearance_m=0.80,
                preview_lead_m=0.80,
            )
        )
        target_speed = self.target_speed_for_progress(
            progress,
            self.get_clock().now().nanoseconds / 1e9,
        )
        target_speed = apply_startup_speed_envelope(
            target_speed,
            experiment_started=gate.experiment_started,
            cumulative_progress=gate.cumulative_progress,
            cap_mps=self.startup_speed_cap,
            end_progress=self.startup_speed_cap_end_progress,
        )
        target_speed *= tracking_safety_speed_scale(
            float(tracking[2]), float(tracking[3])
        )
        target_speed = max(self.target_speed_min, target_speed)
        acceleration = self.longitudinal.update(target_speed, float(tracking[6]), 0.02)
        matrix_a, matrix_b, c2_fallback = self.model_for_step(vx)
        solution = self.mpc.solve(
            x0=np.array(
                [
                    float(tracking[2]),
                    float(tracking[3]),
                    float(tracking[4]),
                    float(tracking[5]),
                ]
            ),
            curvature_preview=curvature_preview,
            vx_mps=vx,
            a_aug=matrix_a,
            b_aug=matrix_b,
            sample_time_s=self.nominal_parameters.sample_time_s,
            previous_delta_rad=self.applied_steering,
            left_corridor_preview=left_corridor,
            right_corridor_preview=right_corridor,
            warm_start=self.mpc_warm_start,
        )
        if solution.valid:
            # Shift the previous optimum by one control interval. The final
            # steering/slack samples are held, providing a feasible, nearby
            # initial point for the newly constructed 50 Hz OSQP problem.
            steering_warm = np.r_[
                solution.steering_sequence[1:], solution.steering_sequence[-1]
            ]
            slack_warm = np.r_[solution.slack[1:], solution.slack[-1]]
            self.mpc_warm_start = np.r_[steering_warm, slack_warm]
        self.consecutive_solver_failures = (
            0 if solution.valid else self.consecutive_solver_failures + 1
        )
        self.publish_diagnostics(solution, gate, target_speed, float(tracking[6]))
        self.publish_matlab_parity_frame(
            solution=solution,
            c2_fallback=c2_fallback,
            target_speed_mps=target_speed,
            vx_mps=vx,
            state=np.asarray(
                [tracking[2], tracking[3], tracking[4], tracking[5]], dtype=float
            ),
            a_tracking=matrix_a,
            b_tracking=matrix_b,
            curvature_preview=curvature_preview,
            left_corridor_preview=left_corridor,
            right_corridor_preview=right_corridor,
            lap_progress=progress,
        )
        self.publish_matlab_control_frame(
            solution=solution,
            vx_mps=vx,
            state=np.asarray(
                [tracking[2], tracking[3], tracking[4], tracking[5]], dtype=float
            ),
            a_tracking=matrix_a,
            b_tracking=matrix_b,
            curvature_preview=curvature_preview,
            left_corridor_preview=left_corridor,
            right_corridor_preview=right_corridor,
        )
        if self.consecutive_solver_failures >= 10:
            self.publish_event(
                "controller_failure",
                reason="ten_consecutive_invalid_qp_solutions",
                solver_status=solution.status,
            )
            self.done = True
            self.publish_zero()
            return
        command = AckermannDriveStamped()
        command.header.stamp = self.get_clock().now().to_msg()
        command.drive.steering_angle = float(solution.first_move_rad)
        command.drive.acceleration = float(acceleration)
        self.candidate_publisher.publish(command)

        if self.controller_id == "C2_NEUROGRIP":
            if c2_fallback and not self.c2_was_in_fallback:
                self.publish_event("c2_nominal_fallback", count=self.c2_fallback_count)
            elif not c2_fallback and self.c2_was_in_fallback:
                model = self.last_valid_dynamics
                self.publish_event(
                    "c2_model_active",
                    dynamics_schema=(None if model is None else int(model.schema_version)),
                    artifact_sha256=(None if model is None else model.artifact_sha256),
                )
            self.c2_was_in_fallback = c2_fallback

    def publish_matlab_parity_frame(
        self,
        *,
        solution,
        c2_fallback: bool,
        target_speed_mps: float,
        vx_mps: float,
        state: np.ndarray,
        a_tracking: np.ndarray,
        b_tracking: np.ndarray,
        curvature_preview: np.ndarray,
        left_corridor_preview: np.ndarray,
        right_corridor_preview: np.ndarray,
        lap_progress: float,
    ) -> None:
        """Publish exact C2 decision inputs for read-only MATLAB replay."""
        model = self.last_valid_dynamics
        if (
            self.controller_id != "C2_NEUROGRIP"
            or c2_fallback
            or not self.grip_estimate_ready
            or model is None
            or model.schema_version != 4
        ):
            return
        message = Float64MultiArray()
        message.data = build_matlab_parity_frame(
            control_time_s=self.get_clock().now().nanoseconds / 1e9,
            sample_time_s=self.sample_time_s,
            raw_grip=np.asarray(
                [model.estimated_front_grip, model.estimated_rear_grip]
            ),
            raw_grip_std=np.asarray([model.front_grip_std, model.rear_grip_std]),
            raw_grip_error_q90=np.asarray(
                [model.front_grip_error_q90, model.rear_grip_error_q90]
            ),
            filtered_grip=np.asarray(
                [self.filtered_front_grip, self.filtered_rear_grip]
            ),
            held_grip_std=np.asarray(
                [self.filtered_front_std, self.filtered_rear_std]
            ),
            held_grip_error_q90=np.asarray(
                [self.front_grip_error_q90, self.rear_grip_error_q90]
            ),
            conservative_grip=np.asarray(
                [self.profile_front_grip, self.profile_rear_grip]
            ),
            effective_grip=self.profile_grip,
            profile_utilisation=self.profile_utilisation,
            grip_error_margin_scale=self.grip_error_margin_scale,
            target_speed_mps=target_speed_mps,
            vx_mps=vx_mps,
            lap_progress=lap_progress,
            maximum_speed_mps=self.target_speed_max,
            minimum_speed_mps=self.target_speed_min,
            acceleration_limit_mps2=self.profile_acceleration_limit,
            braking_limit_mps2=self.profile_braking_limit,
            model_blend=self.c2_model_blend,
            previous_steering_rad=self.applied_steering,
            state=state,
            raw_a_lateral=np.asarray(model.a_matrix),
            raw_b_lateral=np.asarray(model.b_matrix),
            a_tracking=a_tracking,
            b_tracking=b_tracking,
            curvature_preview=curvature_preview,
            left_corridor_preview=left_corridor_preview,
            right_corridor_preview=right_corridor_preview,
            first_move_rad=solution.first_move_rad,
            solution_valid=solution.valid,
        )
        self.matlab_parity_publisher.publish(message)

    def publish_matlab_control_frame(
        self,
        *,
        solution,
        vx_mps: float,
        state: np.ndarray,
        a_tracking: np.ndarray,
        b_tracking: np.ndarray,
        curvature_preview: np.ndarray,
        left_corridor_preview: np.ndarray,
        right_corridor_preview: np.ndarray,
    ) -> None:
        """Publish the actual QP input even during nominal C2 fallback."""
        message = Float64MultiArray()
        message.data = build_matlab_control_frame(
            control_time_s=self.get_clock().now().nanoseconds / 1e9,
            sample_time_s=self.sample_time_s,
            vx_mps=vx_mps,
            previous_steering_rad=self.applied_steering,
            state=state,
            a_tracking=a_tracking,
            b_tracking=b_tracking,
            curvature_preview=curvature_preview,
            left_corridor_preview=left_corridor_preview,
            right_corridor_preview=right_corridor_preview,
            first_move_rad=solution.first_move_rad,
            solution_valid=solution.valid,
        )
        self.matlab_control_publisher.publish(message)

    def publish_diagnostics(
        self, solution, gate, target_speed_mps: float, measured_speed_mps: float
    ) -> None:
        """Publish a compact versioned record of every MPC decision."""
        maximum_slack = (
            float(np.max(solution.slack)) if solution.slack.size else -1.0
        )
        message = Float64MultiArray()
        message.data = [
            float(DIAGNOSTICS_SCHEMA),
            self.get_clock().now().nanoseconds / 1e9,
            float(solution.valid),
            float(solution.solve_time_ms),
            maximum_slack,
            float(solution.first_move_rad),
            float(gate.boundary_margin_m),
            float(gate.experiment_started),
            float(gate.cumulative_progress),
            float(target_speed_mps),
            float(measured_speed_mps),
            float(self.profile_grip),
            float(self.profile_utilisation),
            float(self.profile_front_grip),
            float(self.profile_rear_grip),
        ]
        self.diagnostics_publisher.publish(message)

    def publish_zero(self) -> None:
        """Publish a final stop candidate after the requested number of laps."""
        command = AckermannDriveStamped()
        command.header.stamp = self.get_clock().now().to_msg()
        self.candidate_publisher.publish(command)


def parse_arguments(argv=None) -> argparse.Namespace:
    """Parse non-ROS arguments; ROS parameters remain launch-overridable."""
    parser = argparse.ArgumentParser(description="NeuroGrip Python controller node")
    parser.add_argument("--track-npz", required=True)
    parser.add_argument(
        "--controller",
        default="C0_FIXED",
        choices=["C0_FIXED", "C1_ORACLE_MU", "C2_NEUROGRIP"],
    )
    parser.add_argument("--nominal-fit", default="runs/eufs_v1/models/nominal_fit.json")
    parser.add_argument("--target-speed", type=float, default=5.0)
    parser.add_argument("--target-speed-min", type=float, default=1.2)
    parser.add_argument("--fixed-mu", type=float, default=1.0)
    parser.add_argument("--c2-adaptive-speed-max", type=float, default=-1.0)
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--sample-time", type=float, default=0.02)
    parser.add_argument("--laps", type=int, default=1)
    return parser.parse_known_args(argv)[0]


def main(argv=None) -> int:
    """Run until interrupted or the requested lap count completes."""
    rclpy.init()
    node = ControllerNode(parse_arguments(argv))
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
