"""Longitudinal PI speed controller with anti-windup and saturation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LongitudinalPI:
    """PI controller producing acceleration in [-limit, limit]."""

    kp: float = 1.2
    ki: float = 0.6
    acceleration_limit_mps2: float = 5.2
    integral_limit: float = 5.0
    _integral: float = 0.0

    def reset(self) -> None:
        """Reset the integral term."""
        self._integral = 0.0

    def update(self, target_speed_mps: float, measured_speed_mps: float, dt_s: float) -> float:
        """Return the acceleration command."""
        error = target_speed_mps - measured_speed_mps
        candidate_integral = self._integral + error * dt_s
        candidate_integral = max(-self.integral_limit, min(self.integral_limit, candidate_integral))
        command = self.kp * error + self.ki * candidate_integral
        if abs(command) <= self.acceleration_limit_mps2:
            self._integral = candidate_integral
        else:
            command = max(-self.acceleration_limit_mps2, min(self.acceleration_limit_mps2, command))
        return command


def target_speed_from_curvature(
    curvature_1pm: float, v_max_mps: float, v_min_mps: float, gain: float
) -> float:
    """Shared target-speed scheduler used by C0/C1/C2 (future-speed free)."""
    target = v_max_mps / (1.0 + gain * abs(curvature_1pm))
    return max(v_min_mps, min(v_max_mps, target))
