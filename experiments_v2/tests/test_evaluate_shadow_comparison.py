"""Regression tests for non-actuating MATLAB/Python shadow diagnostics."""

import numpy as np
import pytest

from experiments_v2.evaluate_shadow_comparison import nearest_episode_indices, summarize


def episode():
    return {
        "time_s": np.asarray([10.00, 10.02, 10.04, 10.06]),
        "steering_applied_rad": np.asarray([0.00, 0.01, 0.02, 0.03]),
        "steering_safe_rad": np.asarray([0.01, 0.02, 0.03, 0.04]),
        "e_y_m": np.zeros(4),
        "e_psi_rad": np.zeros(4),
        "v_x_mps": np.ones(4) * 4.5,
        "d_kappa_rad_s": np.zeros(4),
    }


def shadow():
    return {
        "time_s": np.asarray([10.001, 10.041, 10.061]),
        "shadow_steering_rad": np.asarray([0.011, 0.031, 0.37]),
        "applied_steering_rad": np.asarray([0.00, 0.02, 0.03]),
        "e_y_m": np.zeros(3),
        "e_psi_rad": np.zeros(3),
        "v_y_mps": np.zeros(3),
        "yaw_rate_rps": np.zeros(3),
        "v_x_mps": np.ones(3) * 4.5,
        "d_kappa_rad_s": np.zeros(3),
        "lateral_spectral_radius": np.asarray([0.90, 0.92, 0.95]),
    }


def test_nearest_matching_rejects_out_of_tolerance_shadow_rows():
    indices, deltas = nearest_episode_indices(
        episode()["time_s"], np.asarray([10.001, 10.20]), 0.025
    )
    assert indices.tolist() == [0]
    assert deltas.tolist() == pytest.approx([0.001])


def test_summary_reports_difference_and_saturation_without_performance_claim():
    report = summarize(episode(), shadow(), 0.025)
    assert report["matched_samples"] == 3
    assert report["unmatched_shadow_samples"] == 0
    assert report["shadow_minus_python_mae_rad"] == pytest.approx((0.001 + 0.001 + 0.33) / 3)
    assert report["shadow_saturation_fraction"] == pytest.approx(1 / 3)
    assert report["python_saturation_fraction"] == 0.0
    assert report["shadow_lastmove_minus_episode_applied_mae_rad"] == pytest.approx(0.0)
    assert report["max_shadow_lateral_spectral_radius"] == pytest.approx(0.95)
