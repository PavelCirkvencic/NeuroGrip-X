"""Generate periodic curvature and track-corridor previews for the MPC."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ReferencePreview:
    """Immutable arc-length parameterised reference data."""

    s_m: np.ndarray
    length_m: float
    curvature_1pm: np.ndarray
    left_width_m: np.ndarray
    right_width_m: np.ndarray

    @classmethod
    def from_npz(cls, path: str | Path) -> "ReferencePreview":
        """Load arrays required by the controller's finite horizon."""
        with np.load(Path(path).expanduser().resolve()) as data:
            required = {"s", "length", "curvature", "left_width", "right_width"}
            missing = required - set(data.files)
            if missing:
                raise ValueError(f"reference preview missing arrays: {sorted(missing)}")
            result = cls(
                s_m=np.asarray(data["s"], dtype=float).reshape(-1),
                length_m=float(np.asarray(data["length"]).reshape(-1)[0]),
                curvature_1pm=np.asarray(data["curvature"], dtype=float).reshape(-1),
                left_width_m=np.asarray(data["left_width"], dtype=float).reshape(-1),
                right_width_m=np.asarray(data["right_width"], dtype=float).reshape(-1),
            )
        sizes = {
            len(result.s_m),
            len(result.curvature_1pm),
            len(result.left_width_m),
            len(result.right_width_m),
        }
        if len(sizes) != 1 or next(iter(sizes)) < 3:
            raise ValueError("reference arrays must have equal non-trivial length")
        if result.length_m <= 0.0 or np.any(np.diff(result.s_m) <= 0.0):
            raise ValueError("reference arc length must be positive and increasing")
        if not np.all(
            np.isfinite(
                np.r_[
                    result.s_m,
                    result.curvature_1pm,
                    result.left_width_m,
                    result.right_width_m,
                ]
            )
        ):
            raise ValueError("reference preview contains NaN or Inf")
        return result

    def horizon(
        self,
        lap_progress: float,
        speed_mps: float,
        sample_time_s: float,
        steps: int,
        body_clearance_m: float,
        preview_lead_m: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return future curvature and left/right centre-clearance limits."""
        if (
            steps < 1
            or sample_time_s <= 0.0
            or body_clearance_m <= 0.0
            or preview_lead_m < 0.0
        ):
            raise ValueError("preview dimensions and timing must be positive")
        distance = preview_lead_m + (
            max(float(speed_mps), 0.0)
            * sample_time_s
            * np.arange(1, steps + 1)
        )
        query_s = (float(lap_progress) * self.length_m + distance) % self.length_m
        curvature = np.interp(query_s, self.s_m, self.curvature_1pm)
        left = np.interp(query_s, self.s_m, self.left_width_m) - body_clearance_m
        right = np.interp(query_s, self.s_m, self.right_width_m) - body_clearance_m
        return curvature, np.maximum(left, 0.05), np.maximum(right, 0.05)
