"""
Causal body-frame velocity estimation from pose samples.

The Gazebo odometry twist reports zero lateral velocity for the Ackermann
model, which would make the lateral state degenerate.  This module derives the
full body-frame velocity from consecutive world poses instead.

Frame and sign conventions:

* world frame is ENU (x east, y north, z up), yaw is CCW around +z;
* body frame is ISO 8855 / SAE (x forward, y left, z up);
* units are metres, seconds, radians and m/s.

The estimator is strictly causal: it uses a backward difference between the
current and previous pose, so the offline preprocessing and a live ROS node can
share exactly the same code without using future samples.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, sin


def yaw_from_quaternion(x: float, y: float, z: float, w: float) -> float:
    """Return the ENU yaw angle [rad] from a quaternion."""
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return atan2(siny_cosp, cosy_cosp)


def normalize_angle(angle_rad: float) -> float:
    """Wrap an angle to (-pi, pi]."""
    from math import fmod, pi

    wrapped = fmod(angle_rad + pi, 2.0 * pi)
    if wrapped <= 0.0:
        wrapped += 2.0 * pi
    return wrapped - pi


@dataclass
class CausalBodyVelocityEstimator:
    """Backward-difference body-frame velocity estimator."""

    max_gap_s: float = 0.10
    min_dt_s: float = 1.0e-6
    _previous_time_s: float | None = None
    _previous_x_m: float | None = None
    _previous_y_m: float | None = None
    _previous_yaw_rad: float | None = None

    def reset(self) -> None:
        """Forget the previous pose, e.g. after a simulator reset."""
        self._previous_time_s = None
        self._previous_x_m = None
        self._previous_y_m = None
        self._previous_yaw_rad = None

    def update(
        self, time_s: float, x_m: float, y_m: float, yaw_rad: float
    ) -> tuple[float | None, float | None]:
        """Return ``(v_x_body, v_y_body)`` or ``(None, None)`` when undefined."""
        if self._previous_time_s is None:
            self._remember(time_s, x_m, y_m, yaw_rad)
            return (None, None)

        elapsed_s = time_s - self._previous_time_s
        previous_x_m = self._previous_x_m
        previous_y_m = self._previous_y_m
        previous_yaw_rad = self._previous_yaw_rad
        self._remember(time_s, x_m, y_m, yaw_rad)

        if elapsed_s <= self.min_dt_s or elapsed_s > self.max_gap_s:
            # Non-monotonic clock, duplicate stamp or a recording gap: the
            # backward difference would be meaningless, so report invalid.
            return (None, None)

        velocity_world_x = (x_m - previous_x_m) / elapsed_s
        velocity_world_y = (y_m - previous_y_m) / elapsed_s
        # The chord between two samples points along the *midpoint* heading, so
        # rotating by the causal midpoint yaw removes the O(omega*dt) bias.
        midpoint_yaw = previous_yaw_rad + 0.5 * normalize_angle(yaw_rad - previous_yaw_rad)
        cos_yaw = cos(midpoint_yaw)
        sin_yaw = sin(midpoint_yaw)
        velocity_body_x = velocity_world_x * cos_yaw + velocity_world_y * sin_yaw
        velocity_body_y = -velocity_world_x * sin_yaw + velocity_world_y * cos_yaw
        return (velocity_body_x, velocity_body_y)

    def _remember(self, time_s: float, x_m: float, y_m: float, yaw_rad: float) -> None:
        """Store the current pose as the next previous pose."""
        self._previous_time_s = time_s
        self._previous_x_m = x_m
        self._previous_y_m = y_m
        self._previous_yaw_rad = yaw_rad
