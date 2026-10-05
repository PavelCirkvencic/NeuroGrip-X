"""Tests for the deterministic IMU corruption model."""

from __future__ import annotations

import pytest

from neurogrip_control.imu_corruption import ImuCorruption


def test_disabled_model_is_transparent():
    """A default model changes nothing."""
    model = ImuCorruption()
    assert not model.enabled
    assert model.dropout() is False
    angular, linear = model.corrupt_vectors((0.1, 0.2, 0.3), (1.0, 2.0, 3.0))
    assert angular == (0.1, 0.2, 0.3)
    assert linear == (1.0, 2.0, 3.0)


def test_units_are_independent():
    """Angular noise must not perturb the linear acceleration channel."""
    model = ImuCorruption(angular_velocity_stddev_rps=0.1, seed=3)
    angular, linear = model.corrupt_vectors((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    assert any(abs(value) > 0.0 for value in angular)
    assert linear == (0.0, 0.0, 0.0)


def test_linear_noise_is_independent():
    """Linear noise must not perturb the angular velocity channel."""
    model = ImuCorruption(linear_acceleration_stddev_mps2=0.5, seed=4)
    angular, linear = model.corrupt_vectors((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    assert angular == (0.0, 0.0, 0.0)
    assert any(abs(value) > 0.0 for value in linear)


def test_bias_is_applied_per_axis():
    """A constant bias shifts the corresponding axis."""
    model = ImuCorruption(angular_velocity_bias_rps=(0.1, -0.2, 0.0))
    angular, linear = model.corrupt_vectors((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    assert angular == pytest.approx((0.1, -0.2, 0.0))
    assert linear == (0.0, 0.0, 0.0)


def test_noise_is_reproducible_from_seed():
    """The same seed reproduces the same corrupted stream."""
    first = ImuCorruption(angular_velocity_stddev_rps=0.05, seed=7)
    second = ImuCorruption(angular_velocity_stddev_rps=0.05, seed=7)
    for _ in range(5):
        assert first.corrupt_vectors((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)) == (
            second.corrupt_vectors((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        )


def test_random_walk_accumulates():
    """A random walk drifts the reading over successive samples."""
    model = ImuCorruption(angular_velocity_random_walk_rps=0.01, seed=5)
    first, _ = model.corrupt_vectors((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    second, _ = model.corrupt_vectors((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    assert first != (0.0, 0.0, 0.0)
    assert second != first


def test_full_dropout_drops_everything():
    """Dropout probability 1 drops every sample."""
    model = ImuCorruption(dropout_probability=1.0, seed=0)
    assert all(model.dropout() for _ in range(10))


def test_from_manifest_maps_legacy_isotropic_stddev():
    """Legacy manifests apply the old stddev to both units."""
    model = ImuCorruption.from_manifest({"stddev": 0.05}, seed=1)
    assert model.angular_velocity_stddev_rps == pytest.approx(0.05)
    assert model.linear_acceleration_stddev_mps2 == pytest.approx(0.05)


def test_from_manifest_prefers_per_unit_keys():
    """Modern manifests use the explicit per-unit scales."""
    model = ImuCorruption.from_manifest(
        {
            "stddev": 0.05,
            "angular_velocity_stddev_rps": 0.02,
            "linear_acceleration_stddev_mps2": 0.40,
            "angular_velocity_bias_rps": [0.01, 0.0, -0.01],
        },
        seed=2,
    )
    assert model.angular_velocity_stddev_rps == pytest.approx(0.02)
    assert model.linear_acceleration_stddev_mps2 == pytest.approx(0.40)
    assert model.angular_velocity_bias_rps == pytest.approx((0.01, 0.0, -0.01))


def test_invalid_parameters_are_rejected():
    """Negative noise or out-of-range dropout are configuration errors."""
    with pytest.raises(ValueError):
        ImuCorruption(angular_velocity_stddev_rps=-0.1)
    with pytest.raises(ValueError):
        ImuCorruption(linear_acceleration_stddev_mps2=-0.1)
    with pytest.raises(ValueError):
        ImuCorruption(dropout_probability=1.5)
