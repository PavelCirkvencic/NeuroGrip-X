"""Causal periodic projection of EUFS pose onto a versioned reference track."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def wrap_angle(angle_rad: float) -> float:
    """Wrap a heading error to [-pi, pi)."""
    return float((angle_rad + np.pi) % (2.0 * np.pi) - np.pi)


@dataclass
class TrackProjector:
    """Nearest-reference projector with a local window and reset-safe fallback."""

    x: np.ndarray
    y: np.ndarray
    yaw: np.ndarray
    curvature: np.ndarray
    s: np.ndarray
    closed_length_m: float | None = None
    search_window: int = 40
    teleport_distance_m: float = 4.0

    @classmethod
    def from_npz(cls, path: str) -> "TrackProjector":
        """Load the exact reference artifact shared by controllers and MATLAB."""
        with np.load(path) as data:
            length_m = float(np.asarray(data["length"]).reshape(-1)[0])
            arrays = {
                name: np.asarray(data[name], dtype=float).reshape(-1)
                for name in ("x", "y", "yaw", "curvature", "s")
            }
        # The artifact retains an exact periodic endpoint for MATLAB and plot
        # compatibility. It is geometrically identical to index zero and must
        # not be a second projection candidate with progress exactly 1.0.
        if np.isclose(arrays["s"][-1], length_m):
            arrays = {name: values[:-1] for name, values in arrays.items()}
        return cls(**arrays, closed_length_m=length_m)

    @property
    def length_m(self) -> float:
        """Return the closed centreline length in metres."""
        return float(self.closed_length_m or self.s[-1])

    def project(self, x_m: float, y_m: float, yaw_rad: float, previous: int | None) -> dict:
        """Return signed tracking error and progress for the current pose."""
        if previous is None:
            candidates = np.arange(len(self.x))
        else:
            offsets = np.arange(-self.search_window, self.search_window + 1)
            candidates = (previous + offsets) % len(self.x)
        distances = np.hypot(x_m - self.x[candidates], y_m - self.y[candidates])
        local_best = int(np.argmin(distances))
        if previous is not None and distances[local_best] > self.teleport_distance_m:
            candidates = np.arange(len(self.x))
            distances = np.hypot(x_m - self.x, y_m - self.y)
            local_best = int(np.argmin(distances))
        index = int(candidates[local_best])
        delta = np.asarray([x_m - self.x[index], y_m - self.y[index]])
        normal = np.asarray([-np.sin(self.yaw[index]), np.cos(self.yaw[index])])
        return {
            "index": index,
            "e_y_m": float(delta @ normal),
            "e_psi_rad": wrap_angle(yaw_rad - float(self.yaw[index])),
            "curvature_1pm": float(self.curvature[index]),
            "lap_progress": float(self.s[index] / self.length_m),
        }
