"""Unit tests for the read-only wheel-to-MATLAB compatibility contract."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from neurogrip_control.applied_steering_bridge import encode_applied_steering


def wheel_message(time_s: float, steering_rad: float):
    """Build the small WheelSpeedsStamped surface used by the pure encoder."""
    seconds = int(time_s)
    nanoseconds = round((time_s - seconds) * 1e9)
    return SimpleNamespace(
        header=SimpleNamespace(
            stamp=SimpleNamespace(sec=seconds, nanosec=nanoseconds)
        ),
        speeds=SimpleNamespace(steering=steering_rad),
    )


def test_encoder_preserves_source_stamp_and_measured_steering():
    """The flat wire contract is versioned and does not alter the input value."""
    encoded = encode_applied_steering(wheel_message(12.34, -0.051))
    assert encoded == pytest.approx([1.0, 12.34, -0.051])


@pytest.mark.parametrize("time_s, steering_rad", [(-1.0, 0.0), (1.0, float("nan"))])
def test_encoder_rejects_invalid_source_values(time_s, steering_rad):
    """Malformed telemetry cannot be forwarded into the shadow-controller gate."""
    with pytest.raises(ValueError):
        encode_applied_steering(wheel_message(time_s, steering_rad))
