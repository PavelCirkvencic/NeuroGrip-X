"""Pure schema-4 NeuroGrip filtering and conservative axle policy."""

from __future__ import annotations

import math

import numpy as np

from speed_profile import effective_axle_grip


def grip_is_observable(
    vx_mps: float, yaw_rate_rps: float, applied_steering_rad: float
) -> bool:
    """Match the runtime excitation gate used before accepting grip updates."""
    values = (vx_mps, yaw_rate_rps, applied_steering_rad)
    if not all(math.isfinite(float(value)) for value in values):
        return False
    lateral_excitation_mps2 = abs(float(vx_mps) * float(yaw_rate_rps))
    return (
        float(vx_mps) >= 6.0
        and lateral_excitation_mps2 >= 1.0
        and abs(float(applied_steering_rad)) >= 0.01
    )


def update_filtered_grip(
    filtered_front: float,
    filtered_rear: float,
    estimated_front: float,
    estimated_rear: float,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Apply the exact scalar EMA used by the C2 runtime controller."""
    values = (
        filtered_front,
        filtered_rear,
        estimated_front,
        estimated_rear,
        alpha,
    )
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("grip filter values must be finite")
    if not 0.0 < float(alpha) <= 1.0:
        raise ValueError("grip filter alpha must be in (0, 1]")
    return (
        float(filtered_front)
        + float(alpha) * (float(estimated_front) - float(filtered_front)),
        float(filtered_rear)
        + float(alpha) * (float(estimated_rear) - float(filtered_rear)),
    )


def conservative_grip_policy(
    filtered_front: float,
    filtered_rear: float,
    front_std: float,
    rear_std: float,
    front_error_q90: float,
    rear_error_q90: float,
    margin_scale: float = 0.50,
) -> tuple[float, float, float]:
    """Return effective/front/rear grip after calibrated uncertainty margins."""
    values = (
        filtered_front,
        filtered_rear,
        front_std,
        rear_std,
        front_error_q90,
        rear_error_q90,
        margin_scale,
    )
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("grip policy values must be finite")
    if min(float(value) for value in values[2:6]) < 0.0:
        raise ValueError("grip uncertainty values must be non-negative")
    if not 0.0 <= float(margin_scale) <= 1.0:
        raise ValueError("grip margin scale must be in [0, 1]")
    front_margin = float(front_std) + float(margin_scale) * float(front_error_q90)
    rear_margin = float(rear_std) + float(margin_scale) * float(rear_error_q90)
    front = float(np.clip(float(filtered_front) - front_margin, 0.35, 1.30))
    rear = float(np.clip(float(filtered_rear) - rear_margin, 0.35, 1.30))
    return effective_axle_grip(front, rear), front, rear
