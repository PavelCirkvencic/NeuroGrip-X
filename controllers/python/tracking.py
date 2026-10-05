"""Track projection from a reference artifact into the MPC state."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def wrap_angle(angle: float) -> float:
    """Wrap an angle to (-pi, pi]."""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


@dataclass
class TrackProjector:
    """Project the car onto the nearest centerline sample (causal)."""

    x: np.ndarray
    y: np.ndarray
    yaw: np.ndarray
    curvature: np.ndarray
    s: np.ndarray
    search_window: int = 40

    @property
    def length(self) -> float:
        """Closed centerline length [m]."""
        return float(self.s[-1])

    @classmethod
    def from_npz(cls, path) -> "TrackProjector":
        """Load a reference artifact produced by build_track.py."""
        data = np.load(path)
        return cls(
            x=np.asarray(data["x"]),
            y=np.asarray(data["y"]),
            yaw=np.asarray(data["yaw"]),
            curvature=np.asarray(data["curvature"]),
            s=np.asarray(data["s"]),
        )

    def project(self, x_m: float, y_m: float, yaw_rad: float, previous_index: int | None) -> dict:
        """Return tracking state at the nearest reference sample."""
        count = len(self.x)
        if previous_index is None:
            candidates = np.arange(count)
        else:
            offsets = np.arange(-self.search_window, self.search_window + 1)
            candidates = (previous_index + offsets) % count
        delta = np.column_stack([x_m - self.x[candidates], y_m - self.y[candidates]])
        distances = np.hypot(delta[:, 0], delta[:, 1])
        best = int(candidates[int(np.argmin(distances))])
        normal = np.array([-np.sin(self.yaw[best]), np.cos(self.yaw[best])])
        e_y = float(delta[np.argmin(distances)] @ normal)
        e_psi = float(wrap_angle(yaw_rad - self.yaw[best]))
        return {
            "index": best,
            "e_y_m": e_y,
            "e_psi_rad": e_psi,
            "curvature_1pm": float(self.curvature[best]),
            "s_m": float(self.s[best]),
            "lap_progress": float(self.s[best] / self.length),
        }
