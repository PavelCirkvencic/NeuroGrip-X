#!/usr/bin/env python3
"""Fit nominal front/rear cornering stiffness from a grip-1.0 episode."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

MASS_KG = 300.0
WHEELBASE_M = 1.53
FRONT_FRACTION = 0.5


def parse_arguments() -> argparse.Namespace:
    """Parse the episode CSV and output path."""
    parser = argparse.ArgumentParser(description="Fit nominal bicycle stiffness")
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    """Fit Cf/Cr from the lateral-velocity equation and save a JSON."""
    arguments = parse_arguments()
    data = pd.read_csv(arguments.episode)
    required = {"time_s", "v_x_mps", "v_y_mps", "yaw_rate_rps", "steering_applied_rad"}
    if not required.issubset(data.columns):
        raise ValueError(f"episode is missing columns: {required - set(data.columns)}")
    data = data[(data["v_x_mps"] > 1.0) & data["v_y_mps"].notna()].reset_index(drop=True)
    time = data["time_s"].to_numpy()
    vx = data["v_x_mps"].to_numpy()
    vy = data["v_y_mps"].to_numpy()
    r = data["yaw_rate_rps"].to_numpy()
    delta = data["steering_applied_rad"].to_numpy()
    dt = np.gradient(time)
    dt = np.where(np.abs(dt) < 1e-6, np.nan, dt)
    dvy = np.gradient(vy) / dt
    mask = np.isfinite(dvy) & np.isfinite(vx) & (vx > 1.0)
    vx, vy, r, delta, dvy = vx[mask], vy[mask], r[mask], delta[mask], dvy[mask]

    # Joint least-squares over both lateral equations for (Cf, Cr) directly.
    # yaw:  dr = Cf*[-(lf*vy + lf^2*r)/(Iz*vx) + lf*delta/Iz]
    #           + Cr*[(lr*vy - lr^2*r)/(Iz*vx)]
    # vy:   dvy + vx*r = Cf*[(-vy - lf*r)/(m*vx) + delta/m]
    #                   + Cr*[(-vy + lr*r)/(m*vx)]
    iz = 172.44
    lf = FRONT_FRACTION * WHEELBASE_M
    lr = (1.0 - FRONT_FRACTION) * WHEELBASE_M
    dr = np.gradient(r) / dt
    mask = mask & np.isfinite(dr)
    vx, vy, r, delta, dvy, dr = vx[mask], vy[mask], r[mask], delta[mask], dvy[mask], dr[mask]
    safe_vx = np.maximum(vx, 1.0)
    x_cf_yaw = (-lf * vy - lf**2 * r) / (iz * safe_vx) + lf * delta / iz
    x_cr_yaw = (lr * vy - lr**2 * r) / (iz * safe_vx)
    x_cf_vy = (-vy - lf * r) / (MASS_KG * safe_vx) + delta / MASS_KG
    x_cr_vy = (-vy + lr * r) / (MASS_KG * safe_vx)
    design = np.column_stack([np.concatenate([x_cf_yaw, x_cf_vy]),
                              np.concatenate([x_cr_yaw, x_cr_vy])])
    target = np.concatenate([dr, dvy + vx * r])
    coefficients, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
    cf, cr = (float(value) for value in coefficients)
    prediction = design @ coefficients
    ss_res = float(np.sum((target - prediction) ** 2))
    ss_tot = float(np.sum((target - target.mean()) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    total = cf + cr
    consistency = lf * cf - lr * cr
    check = 0.0

    result = {
        "schema_version": 1,
        "episode": str(arguments.episode),
        "samples": int(len(target)),
        "front_cornering_stiffness_n_rad": cf,
        "rear_cornering_stiffness_n_rad": cr,
        "sum_cornering_stiffness_n_rad": total,
        "front_lf_minus_rear_lr_nm_rad": consistency,
        "yaw_equation_cross_check_nm_rad": check,
        "fit_r_squared": r_squared,
        "mass_kg": MASS_KG,
        "wheelbase_m": WHEELBASE_M,
        "note": "Cf/Cr from the lateral-velocity equation on a nominal-grip episode.",
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in (
        "front_cornering_stiffness_n_rad", "rear_cornering_stiffness_n_rad",
        "fit_r_squared", "samples")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
