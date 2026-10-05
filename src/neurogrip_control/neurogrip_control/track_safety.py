"""Provide pure track-envelope and safety-valid lap-completion logic."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class TrackEnvelope:
    """Periodic centreline widths used for conservative footprint clearance."""

    progress: np.ndarray
    left_width_m: np.ndarray
    right_width_m: np.ndarray

    @classmethod
    def from_npz(cls, path: str | Path) -> "TrackEnvelope":
        """Load and validate the immutable reference-track envelope."""
        with np.load(Path(path).expanduser().resolve()) as data:
            required = {"s", "length", "left_width", "right_width"}
            missing = required - set(data.files)
            if missing:
                raise ValueError(f"track envelope is missing arrays: {sorted(missing)}")
            length_m = float(np.asarray(data["length"]).reshape(-1)[0])
            s_m = np.asarray(data["s"], dtype=float).reshape(-1)
            left = np.asarray(data["left_width"], dtype=float).reshape(-1)
            right = np.asarray(data["right_width"], dtype=float).reshape(-1)
        if length_m <= 0.0 or not np.isfinite(length_m):
            raise ValueError("track length must be finite and positive")
        if not (len(s_m) == len(left) == len(right) and len(s_m) >= 3):
            raise ValueError("track envelope arrays must have equal non-trivial length")
        if np.any(np.diff(s_m) <= 0.0):
            raise ValueError("track arc length must be strictly increasing")
        if not np.all(np.isfinite(np.r_[s_m, left, right])):
            raise ValueError("track envelope contains NaN or Inf")
        if np.any(left <= 0.0) or np.any(right <= 0.0):
            raise ValueError("track widths must be positive")
        return cls(np.clip(s_m / length_m, 0.0, 1.0), left, right)

    def widths_at(self, lap_progress: float) -> tuple[float, float]:
        """Interpolate left and right widths at periodic lap progress."""
        progress = float(lap_progress) % 1.0
        source = np.r_[self.progress, 1.0]
        left = np.r_[self.left_width_m, self.left_width_m[0]]
        right = np.r_[self.right_width_m, self.right_width_m[0]]
        return float(np.interp(progress, source, left)), float(
            np.interp(progress, source, right)
        )

    def boundary_margin(
        self,
        lap_progress: float,
        lateral_error_m: float,
        heading_error_rad: float,
        vehicle_half_width_m: float,
        vehicle_half_length_m: float,
    ) -> tuple[float, float, float]:
        """Return widths and conservative oriented-rectangle body clearance."""
        left, right = self.widths_at(lap_progress)
        heading = abs(float(heading_error_rad))
        projected_half_width = (
            vehicle_half_width_m * abs(np.cos(heading))
            + vehicle_half_length_m * abs(np.sin(heading))
        )
        error = float(lateral_error_m)
        available = left - error if error >= 0.0 else right + error
        return left, right, float(available - projected_half_width)


@dataclass(frozen=True)
class LapGateUpdate:
    """One deterministic safety/lap-state update returned to the controller."""

    event: str | None
    reason: str | None
    left_width_m: float
    right_width_m: float
    boundary_margin_m: float
    experiment_started: bool
    cumulative_progress: float
    laps_completed: int


class SafetyLapGate:
    """Require a full forward, in-bounds and collision-free lap."""

    def __init__(
        self,
        envelope: TrackEnvelope,
        vehicle_half_width_m: float = 0.70,
        vehicle_half_length_m: float = 1.30,
        maximum_heading_error_rad: float = np.pi / 2.0,
        heading_violation_samples: int = 10,
        minimum_start_speed_mps: float = 0.1,
        minimum_lap_progress: float = 0.90,
    ):
        if vehicle_half_width_m <= 0.0 or vehicle_half_length_m <= 0.0:
            raise ValueError("vehicle half dimensions must be positive")
        if heading_violation_samples < 1:
            raise ValueError("heading_violation_samples must be positive")
        self.envelope = envelope
        self.vehicle_half_width_m = float(vehicle_half_width_m)
        self.vehicle_half_length_m = float(vehicle_half_length_m)
        self.maximum_heading_error_rad = float(maximum_heading_error_rad)
        self.heading_violation_samples = int(heading_violation_samples)
        self.minimum_start_speed_mps = float(minimum_start_speed_mps)
        self.minimum_lap_progress = float(minimum_lap_progress)
        self.previous_progress: float | None = None
        self.experiment_started = False
        self.cumulative_progress = 0.0
        self.laps_completed = 0
        self.heading_violation_count = 0
        self.terminal = False

    def update(
        self,
        lap_progress: float,
        lateral_error_m: float,
        heading_error_rad: float,
        speed_mps: float,
        collision_count: int = 0,
    ) -> LapGateUpdate:
        """Advance the gate and emit at most one lifecycle event."""
        progress = float(lap_progress) % 1.0
        left, right, margin = self.envelope.boundary_margin(
            progress,
            lateral_error_m,
            heading_error_rad,
            self.vehicle_half_width_m,
            self.vehicle_half_length_m,
        )
        event = None
        reason = None
        if self.previous_progress is None:
            self.previous_progress = progress
            return self._result(event, reason, left, right, margin)

        raw_delta = progress - self.previous_progress
        forward_crossing = self.previous_progress > 0.90 and progress < 0.10
        if raw_delta < -0.5:
            progress_delta = raw_delta + 1.0
        elif raw_delta > 0.5:
            progress_delta = raw_delta - 1.0
        else:
            progress_delta = raw_delta
        self.previous_progress = progress

        forward_heading = abs(float(heading_error_rad)) < self.maximum_heading_error_rad
        if (
            not self.experiment_started
            and forward_crossing
            and forward_heading
            and float(speed_mps) >= self.minimum_start_speed_mps
            and margin >= 0.0
            and collision_count == 0
        ):
            self.experiment_started = True
            self.cumulative_progress = 0.0
            event = "experiment_start"
        elif self.experiment_started and not self.terminal:
            self.cumulative_progress += progress_delta

        if self.experiment_started and not self.terminal:
            if collision_count > 0:
                event, reason = "safety_violation", "cone_collision"
                self.terminal = True
            elif margin < 0.0:
                event, reason = "safety_violation", "track_boundary"
                self.terminal = True
            else:
                if forward_heading:
                    self.heading_violation_count = 0
                else:
                    self.heading_violation_count += 1
                if self.heading_violation_count >= self.heading_violation_samples:
                    event, reason = "safety_violation", "wrong_way_or_spin"
                    self.terminal = True
                elif (
                    forward_crossing
                    and self.cumulative_progress >= self.minimum_lap_progress
                ):
                    self.laps_completed += 1
                    event = "lap_complete"
                    self.cumulative_progress = 0.0

        return self._result(event, reason, left, right, margin)

    def _result(
        self,
        event: str | None,
        reason: str | None,
        left: float,
        right: float,
        margin: float,
    ) -> LapGateUpdate:
        """Build the immutable state snapshot returned from ``update``."""
        return LapGateUpdate(
            event=event,
            reason=reason,
            left_width_m=left,
            right_width_m=right,
            boundary_margin_m=margin,
            experiment_started=self.experiment_started,
            cumulative_progress=self.cumulative_progress,
            laps_completed=self.laps_completed,
        )
