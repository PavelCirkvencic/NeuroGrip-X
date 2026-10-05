"""
Nominal lateral bicycle model and augmented tracking plant (physical units).

Continuous lateral model at longitudinal speed ``vx`` (vx >= 1.0 m/s):

    v_y_dot = -(Cf+Cr)/(m*vx) * v_y + (-vx - (lf*Cf-lr*Cr)/(m*vx)) * r + Cf/m * delta
    r_dot   = -(lf*Cf-lr*Cr)/(Iz*vx) * v_y - (lf^2*Cf+lr^2*Cr)/(Iz*vx) * r + lf*Cf/Iz * delta

ZOH discretisation uses ``scipy.signal.cont2discrete`` so Python and MATLAB can
use the same plant.  The controller uses the augmented state
``[e_y, e_psi, v_y, r]`` with MV ``delta`` and MD ``d_kappa = vx * curvature``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import cont2discrete


@dataclass(frozen=True)
class VehicleParameters:
    """Physical vehicle parameters (SI units)."""

    mass_kg: float = 300.0
    yaw_inertia_kgm2: float = 172.44
    wheelbase_m: float = 1.53
    front_axle_fraction: float = 0.5
    front_cornering_stiffness_n_rad: float = 50000.0
    rear_cornering_stiffness_n_rad: float = 55000.0
    sample_time_s: float = 0.02

    @property
    def front_length_m(self) -> float:
        """Distance from CG to front axle."""
        return self.front_axle_fraction * self.wheelbase_m

    @property
    def rear_length_m(self) -> float:
        """Distance from CG to rear axle."""
        return (1.0 - self.front_axle_fraction) * self.wheelbase_m


DEFAULT_PARAMETERS = VehicleParameters()


def continuous_lateral_matrices(
    vx_mps: float, parameters: VehicleParameters = DEFAULT_PARAMETERS
) -> tuple[np.ndarray, np.ndarray]:
    """Return continuous (A, B) for x = [v_y, r] and input delta."""
    if vx_mps < 1.0:
        raise ValueError("lateral model is only defined for vx >= 1.0 m/s")
    m = parameters.mass_kg
    iz = parameters.yaw_inertia_kgm2
    lf = parameters.front_length_m
    lr = parameters.rear_length_m
    cf = parameters.front_cornering_stiffness_n_rad
    cr = parameters.rear_cornering_stiffness_n_rad

    a = np.array(
        [
            [-(cf + cr) / (m * vx_mps), -vx_mps - (lf * cf - lr * cr) / (m * vx_mps)],
            [-(lf * cf - lr * cr) / (iz * vx_mps), -(lf**2 * cf + lr**2 * cr) / (iz * vx_mps)],
        ]
    )
    b = np.array([[cf / m], [lf * cf / iz]])
    return a, b


def discretize_zoh(
    a: np.ndarray, b: np.ndarray, sample_time_s: float
) -> tuple[np.ndarray, np.ndarray]:
    """Zero-order-hold discretisation matching scipy.signal.cont2discrete."""
    discrete_a, discrete_b, _, _, _ = cont2discrete((a, b, np.eye(2), np.zeros((2, 1))), sample_time_s)
    return np.asarray(discrete_a), np.asarray(discrete_b)


def build_tracking_plant(
    vx_mps: float, parameters: VehicleParameters = DEFAULT_PARAMETERS
) -> tuple[np.ndarray, np.ndarray]:
    """Return augmented (A_aug, B_aug) with state [e_y, e_psi, v_y, r].

    ``B_aug`` columns are [MV delta, MD d_kappa]; the heading error integrates
    ``-d_kappa`` exactly as in the plan (ADR-v2-002).
    """
    a_cont, b_cont = continuous_lateral_matrices(vx_mps, parameters)
    ad, bd = discretize_zoh(a_cont, b_cont, parameters.sample_time_s)
    return build_tracking_plant_from_discrete(
        vx_mps, ad, bd, parameters.sample_time_s
    )


def build_tracking_plant_from_discrete(
    vx_mps: float,
    lateral_a: np.ndarray,
    lateral_b: np.ndarray,
    sample_time_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Lift a physical discrete lateral model into the MPC tracking model.

    This is the only conversion used by C2.  ``lateral_a`` and ``lateral_b``
    are deliberately physical, discrete-time matrices for ``[v_y, r]`` and
    steering angle, respectively.  It prevents a normalised latent Koopman
    operator from accidentally being consumed by the controller.
    """
    if vx_mps < 1.0:
        raise ValueError("tracking plant is only defined for vx >= 1.0 m/s")
    ad = np.asarray(lateral_a, dtype=float)
    bd = np.asarray(lateral_b, dtype=float).reshape(2, 1)
    if ad.shape != (2, 2) or not np.all(np.isfinite(ad)) or not np.all(np.isfinite(bd)):
        raise ValueError("lateral model must contain finite A[2,2] and B[2,1]")
    if not np.isfinite(sample_time_s) or sample_time_s <= 0.0:
        raise ValueError("sample_time_s must be positive and finite")
    ts = sample_time_s
    a_aug = np.array(
        [
            [1.0, ts * vx_mps, ts, 0.0],
            [0.0, 1.0, 0.0, ts],
            [0.0, 0.0, ad[0, 0], ad[0, 1]],
            [0.0, 0.0, ad[1, 0], ad[1, 1]],
        ]
    )
    b_aug = np.array(
        [
            [0.0, 0.0],
            [0.0, -ts],
            [bd[0, 0], 0.0],
            [bd[1, 0], 0.0],
        ]
    )
    return a_aug, b_aug
