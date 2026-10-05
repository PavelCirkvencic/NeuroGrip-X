"""Tests for the nominal bicycle model and the lateral MPC QP."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.signal import cont2discrete

from bicycle_model import (
    DEFAULT_PARAMETERS,
    VehicleParameters,
    build_tracking_plant_from_discrete,
    build_tracking_plant,
    continuous_lateral_matrices,
    discretize_zoh,
)
from mpc_qp import LateralMpc


def test_model_dimensions_and_units():
    """Continuous and discrete matrices have the physical shape."""
    a_cont, b_cont = continuous_lateral_matrices(8.0)
    assert a_cont.shape == (2, 2)
    assert b_cont.shape == (2, 1)
    a_aug, b_aug = build_tracking_plant(8.0)
    assert a_aug.shape == (4, 4)
    assert b_aug.shape == (4, 2)


def test_physical_discrete_model_lifts_to_same_tracking_plant():
    """C2 must consume physical discrete A/B in exactly the nominal format."""
    parameters = VehicleParameters()
    a_cont, b_cont = continuous_lateral_matrices(8.0, parameters)
    ad, bd = discretize_zoh(a_cont, b_cont, parameters.sample_time_s)
    expected_a, expected_b = build_tracking_plant(8.0, parameters)
    actual_a, actual_b = build_tracking_plant_from_discrete(
        8.0, ad, bd, parameters.sample_time_s
    )
    assert np.allclose(actual_a, expected_a)
    assert np.allclose(actual_b, expected_b)


def test_zoh_matches_scipy_reference():
    """Our ZOH result equals a direct cont2discrete call."""
    a_cont, b_cont = continuous_lateral_matrices(10.0)
    ad, bd = discretize_zoh(a_cont, b_cont, 0.02)
    ref_a, ref_b, _, _, _ = cont2discrete(
        (a_cont, b_cont, np.eye(2), np.zeros((2, 1))), 0.02
    )
    assert np.allclose(ad, ref_a, atol=1e-12)
    assert np.allclose(bd, ref_b, atol=1e-12)


def test_positive_steering_gives_positive_yaw_authority():
    """Positive steering must produce positive yaw acceleration/rate."""
    _, b_cont = continuous_lateral_matrices(8.0)
    assert b_cont[1, 0] > 0.0


def test_lower_cornering_stiffness_reduces_authority():
    """A lower front stiffness reduces the steering-to-yaw gain."""
    _, b_strong = continuous_lateral_matrices(
        8.0, VehicleParameters(front_cornering_stiffness_n_rad=50000.0)
    )
    _, b_weak = continuous_lateral_matrices(
        8.0, VehicleParameters(front_cornering_stiffness_n_rad=20000.0)
    )
    assert b_weak[1, 0] < b_strong[1, 0]


def test_lateral_model_rejects_low_speed():
    """The lateral model is undefined below 1 m/s."""
    with pytest.raises(ValueError):
        continuous_lateral_matrices(0.5)


def _solve(x0, curvature=None, previous=0.0, horizon=10):
    a_aug, b_aug = build_tracking_plant(8.0)
    mpc = LateralMpc(horizon=horizon)
    preview = np.zeros(horizon) if curvature is None else curvature
    return mpc.solve(
        x0=np.array(x0, dtype=float),
        curvature_preview=preview,
        vx_mps=8.0,
        a_aug=a_aug,
        b_aug=b_aug,
        sample_time_s=DEFAULT_PARAMETERS.sample_time_s,
        previous_delta_rad=previous,
    )


def test_straight_line_zero_equilibrium():
    """On a straight line with zero error the first move is ~0."""
    solution = _solve([0.0, 0.0, 0.0, 0.0])
    assert solution.valid
    assert abs(solution.first_move_rad) < 1e-4


def test_constraints_are_respected():
    """Steering box, rate and soft corridor are never violated by >1e-6."""
    solution = _solve([1.0, 0.1, 0.0, 0.0], previous=0.0)
    assert solution.valid
    steering = solution.steering_sequence
    assert np.max(np.abs(steering)) <= 0.37 + 1e-6
    assert abs(steering[0] - 0.0) <= 2.5 * 0.02 + 1e-6
    rates = np.abs(np.diff(steering))
    assert np.max(rates) <= 2.5 * 0.02 + 1e-6
    lateral = np.abs(solution.predicted_states[:, 0])
    assert np.max(lateral) <= 1.5 + np.max(solution.slack) + 1e-6


def test_rate_limit_against_previous_move():
    """The first move cannot jump from the previous steering command."""
    solution = _solve([0.5, 0.0, 0.0, 0.0], previous=0.3)
    assert abs(solution.first_move_rad - 0.3) <= 2.5 * 0.02 + 1e-6


def test_nonfinite_input_triggers_safe_fallback():
    """Non-finite state returns the rate-safe fallback, not NaN."""
    solution = _solve([np.nan, 0.0, 0.0, 0.0], previous=0.2)
    assert solution.valid is False
    assert solution.first_move_rad == pytest.approx(0.2)
    assert solution.status == "nonfinite_input"


def test_first_move_is_deterministic():
    """Identical inputs produce an identical first move."""
    first = _solve([1.0, 0.1, 0.0, 0.0])
    second = _solve([1.0, 0.1, 0.0, 0.0])
    assert first.first_move_rad == pytest.approx(second.first_move_rad, abs=1e-12)


def test_recursive_prediction_matrices_match_direct_definition():
    """The real-time recurrence must preserve the exact finite-horizon model."""
    horizon = 8
    a_aug, b_aug = build_tracking_plant(5.0)
    mpc = LateralMpc(horizon=horizon)
    actual_f, actual_g, actual_h = mpc._prediction_matrices(a_aug, b_aug, 0.02)
    expected_f = np.zeros_like(actual_f)
    expected_g = np.zeros_like(actual_g)
    expected_h = np.zeros_like(actual_h)
    for step in range(horizon):
        row = slice(4 * step, 4 * (step + 1))
        expected_f[row] = np.linalg.matrix_power(a_aug, step + 1)
        for inner in range(step + 1):
            power = np.linalg.matrix_power(a_aug, step - inner)
            expected_g[row, inner] = power @ b_aug[:, 0]
            expected_h[row, inner] = power @ b_aug[:, 1]
    assert np.allclose(actual_f, expected_f, atol=1e-14)
    assert np.allclose(actual_g, expected_g, atol=1e-14)
    assert np.allclose(actual_h, expected_h, atol=1e-14)
