"""Physics-based periodic speed profiles for Formula Student tracks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

GRAVITY_MPS2 = 9.81


def tracking_safety_speed_scale(
    lateral_error_m: float, heading_error_rad: float
) -> float:
    """Reduce speed only after tracking leaves the shared nominal envelope."""
    severity = max(
        abs(float(lateral_error_m)) / 0.25,
        abs(float(heading_error_rad)) / 0.10,
    )
    if severity <= 1.0:
        return 1.0
    return float(np.clip(1.0 - 0.45 * (severity - 1.0), 0.45, 1.0))


@dataclass(frozen=True)
class SpeedProfile:
    """Periodic arc-length speed target after lateral and acceleration limits."""

    progress: np.ndarray
    speed_mps: np.ndarray
    lateral_limit_mps2: float
    maximum_speed_mps: float

    def at_progress(self, lap_progress: float) -> float:
        """Interpolate the periodic target at fractional lap progress."""
        query = float(lap_progress) % 1.0
        source = np.r_[self.progress, 1.0]
        values = np.r_[self.speed_mps, self.speed_mps[0]]
        return float(np.interp(query, source, values))


def effective_axle_grip(front: float, rear: float) -> float:
    """Return the conservative axle bottleneck for lateral speed planning.

    A harmonic or arithmetic average can command more lateral acceleration
    than the weaker axle can generate. That is especially unsafe under front
    grip loss, where the car understeers even though the rear axle still has
    reserve. The lateral MPC still receives separate front/rear stiffness;
    this scalar is used only by the shared curvature speed planner.
    """
    front_value = float(np.clip(front, 0.35, 1.30))
    rear_value = float(np.clip(rear, 0.35, 1.30))
    return min(front_value, rear_value)


def build_speed_profile(
    s_m: np.ndarray,
    length_m: float,
    curvature_1pm: np.ndarray,
    friction_coefficient: float,
    lateral_utilisation: float,
    maximum_speed_mps: float,
    minimum_speed_mps: float,
    acceleration_limit_mps2: float,
    braking_limit_mps2: float,
    passes: int = 4,
) -> SpeedProfile:
    """Build a periodic friction-limited profile with forward/backward passes."""
    s = np.asarray(s_m, dtype=float).reshape(-1)
    curvature = np.asarray(curvature_1pm, dtype=float).reshape(-1)
    if len(s) < 3 or len(s) != len(curvature):
        raise ValueError("speed-profile arrays must have equal non-trivial length")
    if np.isclose(s[-1], length_m, atol=1e-6):
        s = s[:-1]
        curvature = curvature[:-1]
    if np.any(np.diff(s) <= 0.0) or not length_m > s[-1]:
        raise ValueError("arc length must increase and remain below periodic length")
    scalars = (
        friction_coefficient,
        lateral_utilisation,
        maximum_speed_mps,
        minimum_speed_mps,
        acceleration_limit_mps2,
        braking_limit_mps2,
    )
    if not all(np.isfinite(value) and value > 0.0 for value in scalars):
        raise ValueError("speed-profile limits must be finite and positive")
    if minimum_speed_mps > maximum_speed_mps or passes < 1:
        raise ValueError("invalid speed bounds or pass count")

    lateral_limit = friction_coefficient * lateral_utilisation * GRAVITY_MPS2
    curve_limit = np.sqrt(lateral_limit / np.maximum(np.abs(curvature), 1e-5))
    speed = np.clip(curve_limit, minimum_speed_mps, maximum_speed_mps)
    segment = np.diff(np.r_[s, length_m])

    # Repeated periodic passes propagate braking needs backward across the
    # start/finish boundary and acceleration capability forward after corners.
    for _ in range(passes):
        for index in range(len(speed) - 1, -1, -1):
            following = (index + 1) % len(speed)
            allowed = np.sqrt(
                max(speed[following] ** 2 + 2.0 * braking_limit_mps2 * segment[index], 0.0)
            )
            speed[index] = min(speed[index], allowed)
        for index in range(len(speed)):
            previous = (index - 1) % len(speed)
            distance = segment[previous]
            allowed = np.sqrt(
                max(speed[previous] ** 2 + 2.0 * acceleration_limit_mps2 * distance, 0.0)
            )
            speed[index] = min(speed[index], allowed)

    return SpeedProfile(
        progress=s / float(length_m),
        speed_mps=speed,
        lateral_limit_mps2=float(lateral_limit),
        maximum_speed_mps=float(maximum_speed_mps),
    )


def profile_from_npz(
    path: str | Path,
    friction_coefficient: float,
    lateral_utilisation: float,
    maximum_speed_mps: float,
    minimum_speed_mps: float,
    acceleration_limit_mps2: float,
    braking_limit_mps2: float,
) -> SpeedProfile:
    """Load a versioned track artifact and build its physical speed profile."""
    with np.load(Path(path).expanduser().resolve()) as data:
        return build_speed_profile(
            s_m=data["s"],
            length_m=float(np.asarray(data["length"]).reshape(-1)[0]),
            curvature_1pm=data["curvature"],
            friction_coefficient=friction_coefficient,
            lateral_utilisation=lateral_utilisation,
            maximum_speed_mps=maximum_speed_mps,
            minimum_speed_mps=minimum_speed_mps,
            acceleration_limit_mps2=acceleration_limit_mps2,
            braking_limit_mps2=braking_limit_mps2,
        )
