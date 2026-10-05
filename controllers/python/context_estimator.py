"""Online scalar grip-scale estimator from the yaw-rate model residual."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GripEstimator:
    """Estimate a scalar lateral grip scale with a forgetting-factor RLS."""

    initial_scale: float = 1.0
    min_scale: float = 0.3
    max_scale: float = 1.3
    forgetting: float = 0.995
    regularization: float = 10.0
    enabled: bool = False

    def __post_init__(self) -> None:
        self.scale = self.initial_scale
        self.covariance = self.regularization
        self.ready = False

    def update(self, regressor: float, measurement: float) -> float:
        """Update the grip estimate from one yaw-acceleration sample."""
        if abs(regressor) < 1e-3 or not self.enabled:
            return self.scale
        gain = self.covariance * regressor / (
            self.forgetting + regressor * self.covariance * regressor
        )
        innovation = measurement - self.scale * regressor
        self.scale = max(
            self.min_scale, min(self.max_scale, self.scale + gain * innovation)
        )
        self.covariance = (self.covariance - gain * regressor * self.covariance) / self.forgetting
        self.covariance = max(1e-4, min(1e4, self.covariance))
        self.ready = True
        return self.scale

    def reset(self, scale: float | None = None) -> None:
        """Reset to the initial or a provided scale."""
        self.scale = self.initial_scale if scale is None else scale
        self.covariance = self.regularization
        self.ready = False
