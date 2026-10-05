"""
Deterministic IMU corruption model (noise, bias, random walk and dropout).

Angular velocity and linear acceleration have independent units and therefore
independent noise scales:

* ``angular_velocity_stddev_rps`` -- rad/s
* ``linear_acceleration_stddev_mps2`` -- m/s^2

An optional constant per-axis bias and an optional Gaussian random walk are
supported.  The model is seeded from the scenario, so the same scenario
reproduces the same corrupted stream, and its parameters never appear as ROS
labels.

Legacy manifests that only define a single isotropic ``stddev`` are mapped to
both scales for backward compatibility (documented schema migration).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field


def _as_triplet(values, default: float) -> tuple[float, float, float]:
    """Return a validated three-element float tuple."""
    if values is None:
        return (default, default, default)
    triplet = tuple(float(value) for value in values)
    if len(triplet) != 3:
        raise ValueError("bias must have exactly three components.")
    return triplet


@dataclass
class ImuCorruption:
    """Apply deterministic noise, bias, random walk and message dropout."""

    angular_velocity_stddev_rps: float = 0.0
    linear_acceleration_stddev_mps2: float = 0.0
    dropout_probability: float = 0.0
    seed: int = 0
    angular_velocity_bias_rps: tuple[float, float, float] = (0.0, 0.0, 0.0)
    linear_acceleration_bias_mps2: tuple[float, float, float] = (0.0, 0.0, 0.0)
    angular_velocity_random_walk_rps: float = 0.0
    linear_acceleration_random_walk_mps2: float = 0.0
    _rng: random.Random = field(init=False, repr=False)
    _angular_walk: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0], repr=False)
    _linear_walk: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0], repr=False)

    def __post_init__(self) -> None:
        if self.angular_velocity_stddev_rps < 0.0:
            raise ValueError("angular_velocity_stddev_rps must be non-negative.")
        if self.linear_acceleration_stddev_mps2 < 0.0:
            raise ValueError("linear_acceleration_stddev_mps2 must be non-negative.")
        if self.angular_velocity_random_walk_rps < 0.0:
            raise ValueError("angular random walk must be non-negative.")
        if self.linear_acceleration_random_walk_mps2 < 0.0:
            raise ValueError("linear random walk must be non-negative.")
        if not 0.0 <= self.dropout_probability <= 1.0:
            raise ValueError("dropout probability must be within [0, 1].")
        self.angular_velocity_bias_rps = _as_triplet(
            self.angular_velocity_bias_rps, 0.0
        )
        self.linear_acceleration_bias_mps2 = _as_triplet(
            self.linear_acceleration_bias_mps2, 0.0
        )
        self._rng = random.Random(self.seed)

    @classmethod
    def from_manifest(cls, imu_noise: dict | None, seed: int) -> "ImuCorruption":
        """
        Build a corruption model from a manifest ``imu_noise`` mapping.

        Accepts the modern per-unit keys and the legacy isotropic ``stddev``.
        """
        imu_noise = imu_noise or {}
        legacy_stddev = float(imu_noise.get("stddev", 0.0))
        return cls(
            angular_velocity_stddev_rps=float(
                imu_noise.get("angular_velocity_stddev_rps", legacy_stddev)
            ),
            linear_acceleration_stddev_mps2=float(
                imu_noise.get("linear_acceleration_stddev_mps2", legacy_stddev)
            ),
            dropout_probability=float(imu_noise.get("dropout_probability", 0.0)),
            seed=seed,
            angular_velocity_bias_rps=imu_noise.get("angular_velocity_bias_rps"),
            linear_acceleration_bias_mps2=imu_noise.get(
                "linear_acceleration_bias_mps2"
            ),
            angular_velocity_random_walk_rps=float(
                imu_noise.get("angular_velocity_random_walk_rps", 0.0)
            ),
            linear_acceleration_random_walk_mps2=float(
                imu_noise.get("linear_acceleration_random_walk_mps2", 0.0)
            ),
        )

    @property
    def enabled(self) -> bool:
        """Return whether this model changes anything at all."""
        return any(
            value > 0.0
            for value in (
                self.angular_velocity_stddev_rps,
                self.linear_acceleration_stddev_mps2,
                self.dropout_probability,
                self.angular_velocity_random_walk_rps,
                self.linear_acceleration_random_walk_mps2,
            )
        ) or any(self.angular_velocity_bias_rps) or any(
            self.linear_acceleration_bias_mps2
        )

    def dropout(self) -> bool:
        """Return True when this sample must be dropped."""
        if self.dropout_probability <= 0.0:
            return False
        return self._rng.random() < self.dropout_probability

    def _gaussian(self, stddev: float) -> float:
        """Return one Gaussian sample, or zero when the scale is disabled."""
        if stddev <= 0.0:
            return 0.0
        return self._rng.gauss(0.0, stddev)

    def corrupt_vectors(
        self,
        angular_velocity: tuple[float, float, float],
        linear_acceleration: tuple[float, float, float],
    ) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        """Return corrupted (angular_velocity, linear_acceleration) tuples."""
        angular = []
        for axis, value in enumerate(angular_velocity):
            self._angular_walk[axis] += self._gaussian(
                self.angular_velocity_random_walk_rps
            )
            angular.append(
                value
                + self.angular_velocity_bias_rps[axis]
                + self._angular_walk[axis]
                + self._gaussian(self.angular_velocity_stddev_rps)
            )
        linear = []
        for axis, value in enumerate(linear_acceleration):
            self._linear_walk[axis] += self._gaussian(
                self.linear_acceleration_random_walk_mps2
            )
            linear.append(
                value
                + self.linear_acceleration_bias_mps2[axis]
                + self._linear_walk[axis]
                + self._gaussian(self.linear_acceleration_stddev_mps2)
            )
        return tuple(angular), tuple(linear)
