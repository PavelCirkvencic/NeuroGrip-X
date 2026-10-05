"""Unit tests for C0/C1/C2 fair-model and C2 message acceptance rules."""

import json

from neurogrip_interfaces.msg import DynamicsModel

import numpy as np
import pytest

from controller_node import (
    MATLAB_CONTROL_FRAME_LENGTH,
    MATLAB_PARITY_FRAME_LENGTH,
    apply_startup_speed_envelope,
    blend_models,
    build_matlab_control_frame,
    build_matlab_parity_frame,
    dynamic_model_is_usable,
    grip_model_is_usable,
    load_nominal_parameters,
    tracking_safety_speed_scale,
    uncertainty_speed_cap,
)
from grip_policy import (
    conservative_grip_policy,
    grip_is_observable,
    update_filtered_grip,
)


def valid_model(stamp_s: float) -> DynamicsModel:
    """Return the smallest valid schema-2 physical C2 model message."""
    message = DynamicsModel()
    message.valid, message.schema_version = True, 2
    message.state_dim, message.input_dim, message.sample_time_s = 2, 1, 0.02
    message.a_matrix, message.b_matrix = [0.9, 0.0, 0.0, 0.8], [0.1, 0.2]
    message.header.stamp.sec = int(stamp_s)
    message.header.stamp.nanosec = int((stamp_s % 1.0) * 1e9)
    return message


def test_c2_requires_fresh_physical_schema_two_model():
    """C2 must reject stale, malformed or wrong-schema model publications."""
    message = valid_model(10.0)
    assert dynamic_model_is_usable(message, 10.04)
    assert not dynamic_model_is_usable(message, 10.20)
    message.schema_version = 1
    assert not dynamic_model_is_usable(message, 10.04)


def test_c2_accepts_schema_three_koopman_physical_jacobian():
    """The controller consumes the Koopman model only through its physical Jacobian."""
    message = valid_model(10.0)
    message.schema_version = 3
    assert dynamic_model_is_usable(message, 10.04)


def test_grip_policy_requires_fresh_bounded_schema_four_output():
    """Only schema 4 carries usable causal per-axle grip estimates."""
    message = valid_model(10.0)
    message.schema_version = 4
    message.estimated_front_grip = 0.78
    message.estimated_rear_grip = 0.66
    message.front_grip_std = 0.02
    message.rear_grip_std = 0.03
    message.front_grip_error_q90 = 0.06
    message.rear_grip_error_q90 = 0.13
    assert grip_model_is_usable(message, 10.04)
    message.estimated_rear_grip = 1.5
    assert not grip_model_is_usable(message, 10.04)


def test_all_controllers_can_load_the_same_nominal_fit(tmp_path):
    """The shared fit parser prevents the old C0/C1/C2 nominal mismatch."""
    path = tmp_path / "nominal.json"
    path.write_text(
        json.dumps(
            {
                "mass_kg": 300.0,
                "wheelbase_m": 1.53,
                "front_fraction": 0.5,
                "front_cornering_stiffness_n_rad": 25000.0,
                "rear_cornering_stiffness_n_rad": 26000.0,
            }
        )
    )
    parameters = load_nominal_parameters(str(path))
    assert parameters.front_cornering_stiffness_n_rad == 25000.0
    assert parameters.rear_cornering_stiffness_n_rad == 26000.0


def test_c2_model_correction_is_shrunk_toward_physics_prior():
    """Runtime adaptation must apply only its declared bounded authority."""
    nominal_a, nominal_b = np.eye(2), np.ones((2, 1))
    learned_a, learned_b = 3.0 * np.eye(2), 3.0 * np.ones((2, 1))
    matrix_a, matrix_b = blend_models(
        nominal_a, nominal_b, learned_a, learned_b, 0.25
    )
    assert np.array_equal(matrix_a, 1.5 * np.eye(2))
    assert np.array_equal(matrix_b, 1.5 * np.ones((2, 1)))


def test_uncertainty_speed_cap_is_bounded_and_fails_to_baseline():
    """Only a fresh calibrated model may unlock the adaptive speed envelope."""
    message = valid_model(10.0)
    message.conformal_radius = 0.04
    message.ensemble_std = 0.01
    cap, confidence = uncertainty_speed_cap(3.0, 5.0, message, 10.04)
    assert confidence == 0.75
    assert cap == 4.5
    assert uncertainty_speed_cap(3.0, 5.0, message, 10.20) == (3.0, 0.0)
    message.valid = False
    assert uncertainty_speed_cap(3.0, 5.0, message, 10.04) == (3.0, 0.0)


def test_tracking_safety_governor_is_shared_smooth_and_bounded():
    """Nominal tracking is untouched; large error commands recovery speed."""
    assert tracking_safety_speed_scale(0.10, 0.04) == 1.0
    assert 0.45 < tracking_safety_speed_scale(0.30, 0.08) < 1.0
    assert tracking_safety_speed_scale(0.80, 0.40) == 0.45
    assert tracking_safety_speed_scale(-0.80, -0.40) == 0.45


def test_schema4_grip_filter_gate_and_uncertainty_policy_are_explicit():
    """Golden-export helpers must remain the exact runtime policy."""
    assert not grip_is_observable(5.99, 0.30, 0.05)
    assert not grip_is_observable(8.0, 0.10, 0.05)
    assert not grip_is_observable(8.0, 0.20, 0.009)
    assert grip_is_observable(8.0, 0.20, 0.01)
    front, rear = update_filtered_grip(1.0, 1.0, 0.70, 0.80, 0.05)
    assert front == 0.985
    assert rear == 0.99
    effective, conservative_front, conservative_rear = conservative_grip_policy(
        front, rear, 0.02, 0.03, 0.06, 0.10, 0.50
    )
    assert conservative_front == pytest.approx(0.935)
    assert conservative_rear == pytest.approx(0.91)
    assert effective == pytest.approx(conservative_rear)


def test_matlab_parity_frame_freezes_exact_controller_decision():
    """The read-only MATLAB capture has a fixed finite schema and row order."""
    frame = build_matlab_parity_frame(
        control_time_s=42.0,
        sample_time_s=0.02,
        raw_grip=np.array([0.78, 0.66]),
        raw_grip_std=np.array([0.02, 0.03]),
        raw_grip_error_q90=np.array([0.06, 0.13]),
        filtered_grip=np.array([0.80, 0.70]),
        held_grip_std=np.array([0.02, 0.03]),
        held_grip_error_q90=np.array([0.06, 0.13]),
        conservative_grip=np.array([0.75, 0.60]),
        effective_grip=0.60,
        profile_utilisation=0.98,
        grip_error_margin_scale=0.50,
        target_speed_mps=8.5,
        vx_mps=8.0,
        lap_progress=0.42,
        maximum_speed_mps=12.0,
        minimum_speed_mps=4.0,
        acceleration_limit_mps2=3.0,
        braking_limit_mps2=2.5,
        model_blend=0.0,
        previous_steering_rad=0.01,
        state=np.arange(4, dtype=float),
        raw_a_lateral=np.arange(4, dtype=float),
        raw_b_lateral=np.arange(2, dtype=float),
        a_tracking=np.arange(16, dtype=float).reshape(4, 4),
        b_tracking=np.arange(8, dtype=float).reshape(4, 2),
        curvature_preview=np.linspace(0.0, 0.1, 30),
        left_corridor_preview=np.linspace(1.0, 2.0, 30),
        right_corridor_preview=np.linspace(1.5, 2.5, 30),
        first_move_rad=-0.04,
        solution_valid=True,
    )
    assert len(frame) == MATLAB_PARITY_FRAME_LENGTH == 155
    assert frame[:3] == [3.0, 42.0, 0.02]
    assert frame[3:9] == pytest.approx([0.78, 0.66, 0.02, 0.03, 0.06, 0.13])
    assert frame[-2:] == [-0.04, 1.0]
    with pytest.raises(ValueError, match="curvature_preview"):
        build_matlab_parity_frame(
            control_time_s=42.0,
            sample_time_s=0.02,
            raw_grip=np.ones(2),
            raw_grip_std=np.ones(2),
            raw_grip_error_q90=np.ones(2),
            filtered_grip=np.ones(2),
            held_grip_std=np.ones(2),
            held_grip_error_q90=np.ones(2),
            conservative_grip=np.ones(2),
            effective_grip=1.0,
            profile_utilisation=0.9,
            grip_error_margin_scale=0.5,
            target_speed_mps=8.0,
            vx_mps=8.0,
            lap_progress=0.4,
            maximum_speed_mps=12.0,
            minimum_speed_mps=4.0,
            acceleration_limit_mps2=3.0,
            braking_limit_mps2=2.5,
            model_blend=0.0,
            previous_steering_rad=0.0,
            state=np.ones(4),
            raw_a_lateral=np.ones(4),
            raw_b_lateral=np.ones(2),
            a_tracking=np.ones((4, 4)),
            b_tracking=np.ones((4, 2)),
            curvature_preview=np.ones(29),
            left_corridor_preview=np.ones(30),
            right_corridor_preview=np.ones(30),
            first_move_rad=0.0,
            solution_valid=True,
        )


def test_matlab_control_frame_is_complete_during_nominal_fallback():
    """The actuation frame contains every QP input without neural-only fields."""
    frame = build_matlab_control_frame(
        control_time_s=42.0,
        sample_time_s=0.02,
        vx_mps=8.0,
        previous_steering_rad=0.01,
        state=np.arange(4, dtype=float),
        a_tracking=np.arange(16, dtype=float).reshape(4, 4),
        b_tracking=np.arange(8, dtype=float).reshape(4, 2),
        curvature_preview=np.linspace(0.0, 0.1, 30),
        left_corridor_preview=np.linspace(1.0, 2.0, 30),
        right_corridor_preview=np.linspace(1.5, 2.5, 30),
        first_move_rad=-0.04,
        solution_valid=True,
    )
    assert len(frame) == MATLAB_CONTROL_FRAME_LENGTH == 125
    assert frame[:5] == [1.0, 42.0, 0.02, 8.0, 0.01]
    assert frame[-2:] == [-0.04, 1.0]


def test_startup_speed_envelope_is_shared_and_bounded():
    """The launch sector cap releases only after its configured progress."""
    arguments = {"cap_mps": 8.0, "end_progress": 0.12}
    assert apply_startup_speed_envelope(
        11.0,
        experiment_started=False,
        cumulative_progress=0.0,
        **arguments,
    ) == 8.0
    assert apply_startup_speed_envelope(
        11.0,
        experiment_started=True,
        cumulative_progress=0.119,
        **arguments,
    ) == 8.0
    assert apply_startup_speed_envelope(
        11.0,
        experiment_started=True,
        cumulative_progress=0.12,
        **arguments,
    ) == 11.0
