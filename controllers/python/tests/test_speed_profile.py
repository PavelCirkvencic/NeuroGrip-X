"""Tests for the periodic friction-limited speed planner."""

from pathlib import Path

import numpy as np

from speed_profile import build_speed_profile, effective_axle_grip, profile_from_npz

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_lateral_limit_obeys_mu_g_relation():
    """A constant-radius track is capped by sqrt(mu*g/kappa)."""
    s = np.arange(0.0, 100.0, 1.0)
    profile = build_speed_profile(s, 100.0, np.full(100, 0.1), 0.8, 0.9, 30.0, 1.0, 5.2, 5.2)
    expected = np.sqrt(0.8 * 0.9 * 9.81 / 0.1)
    assert np.allclose(profile.speed_mps, expected)


def test_lower_grip_never_increases_target_speed():
    """Reducing grip cannot increase any point of the periodic profile."""
    s = np.arange(0.0, 50.0, 0.5)
    curvature = 0.01 + 0.15 * np.sin(2.0 * np.pi * s / 50.0) ** 2
    high = build_speed_profile(s, 50.0, curvature, 1.0, 0.9, 20.0, 2.0, 4.0, 5.0)
    low = build_speed_profile(s, 50.0, curvature, 0.6, 0.9, 20.0, 2.0, 4.0, 5.0)
    assert np.all(low.speed_mps <= high.speed_mps + 1e-12)


def test_axle_grip_penalises_imbalance():
    """The weaker axle is the safe lateral-capacity bottleneck."""
    assert effective_axle_grip(0.70, 0.90) < effective_axle_grip(0.80, 0.80)
    assert effective_axle_grip(0.70, 0.90) == 0.70
    assert effective_axle_grip(0.90, 0.70) == 0.70


def test_trackdrive_profile_reaches_realistic_formula_student_speed():
    """The long FSUK reference contains fast and friction-limited sections."""
    profile = profile_from_npz(
        REPO_ROOT / "artifacts/tracks/trackdrive_reference.npz",
        friction_coefficient=0.8,
        lateral_utilisation=0.85,
        maximum_speed_mps=16.0,
        minimum_speed_mps=4.5,
        acceleration_limit_mps2=5.2,
        braking_limit_mps2=5.2,
    )
    assert np.max(profile.speed_mps) >= 12.0
    assert np.quantile(profile.speed_mps, 0.5) > 7.0
    assert np.min(profile.speed_mps) >= 4.5
