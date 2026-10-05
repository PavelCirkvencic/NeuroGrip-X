#!/usr/bin/env python3
"""Export deterministic schema-4 policy/model vectors for MATLAB parity."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONTROLLER_ROOT = REPOSITORY_ROOT / "controllers" / "python"
sys.path.insert(0, str(CONTROLLER_ROOT))

from bicycle_model import VehicleParameters, build_tracking_plant  # noqa: E402
from grip_policy import (  # noqa: E402
    conservative_grip_policy,
    grip_is_observable,
    update_filtered_grip,
)
from mpc_qp import LateralMpc  # noqa: E402
from speed_profile import (  # noqa: E402
    build_speed_profile,
    tracking_safety_speed_scale,
)


def file_sha256(path: Path) -> str:
    """Return a reproducible source hash for golden-vector provenance."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def nominal_parameters() -> tuple[VehicleParameters, str]:
    """Load exactly the same physical prior as the runtime controller."""
    path = REPOSITORY_ROOT / "runs" / "eufs_v1" / "models" / "nominal_fit.json"
    fit = json.loads(path.read_text(encoding="utf-8"))
    parameters = VehicleParameters(
        mass_kg=float(fit.get("mass_kg", 300.0)),
        wheelbase_m=float(fit.get("wheelbase_m", 1.53)),
        front_axle_fraction=float(fit.get("front_fraction", 0.5)),
        front_cornering_stiffness_n_rad=float(
            fit["front_cornering_stiffness_n_rad"]
        ),
        rear_cornering_stiffness_n_rad=float(
            fit["rear_cornering_stiffness_n_rad"]
        ),
    )
    return parameters, file_sha256(path)


def policy_vectors() -> list[dict[str, object]]:
    """Exercise threshold boundaries, EMA memory and axle asymmetry."""
    filtered_front = 1.0
    filtered_rear = 1.0
    front_std = rear_std = front_q90 = rear_q90 = 0.0
    ready = False
    vectors: list[dict[str, object]] = []
    for index in range(40):
        vx = 5.5 + 0.18 * index
        yaw_rate = 0.10 + 0.012 * (index % 9)
        steering = (0.008 if index % 7 == 0 else 0.012 + 0.001 * (index % 4))
        estimated_front = float(np.clip(0.98 - 0.012 * index, 0.58, 1.1))
        estimated_rear = float(np.clip(0.96 - 0.006 * index + 0.03 * np.sin(index), 0.62, 1.1))
        candidate_front_std = 0.010 + 0.001 * (index % 5)
        candidate_rear_std = 0.014 + 0.0015 * (index % 4)
        candidate_front_q90 = 0.045 + 0.002 * (index % 6)
        candidate_rear_q90 = 0.060 + 0.003 * (index % 5)
        observable = grip_is_observable(vx, yaw_rate, steering)
        if observable:
            filtered_front, filtered_rear = update_filtered_grip(
                filtered_front,
                filtered_rear,
                estimated_front,
                estimated_rear,
                0.05,
            )
            front_std = candidate_front_std
            rear_std = candidate_rear_std
            front_q90 = candidate_front_q90
            rear_q90 = candidate_rear_q90
            ready = True
        effective, conservative_front, conservative_rear = conservative_grip_policy(
            filtered_front,
            filtered_rear,
            front_std,
            rear_std,
            front_q90,
            rear_q90,
            0.50,
        )
        vectors.append(
            {
                "index": index,
                "vx_mps": vx,
                "yaw_rate_rps": yaw_rate,
                "applied_steering_rad": steering,
                "estimated_front_grip": estimated_front,
                "estimated_rear_grip": estimated_rear,
                "front_grip_std": candidate_front_std,
                "rear_grip_std": candidate_rear_std,
                "front_grip_error_q90": candidate_front_q90,
                "rear_grip_error_q90": candidate_rear_q90,
                "observable": observable,
                "expected_ready": ready,
                "expected_filtered_front_grip": filtered_front,
                "expected_filtered_rear_grip": filtered_rear,
                "expected_held_front_std": front_std,
                "expected_held_rear_std": rear_std,
                "expected_held_front_error_q90": front_q90,
                "expected_held_rear_error_q90": rear_q90,
                "expected_front_grip": conservative_front,
                "expected_rear_grip": conservative_rear,
                "expected_effective_grip": effective,
            }
        )
    return vectors


def safety_vectors() -> list[dict[str, float]]:
    """Cover both nominal and saturated tracking-recovery branches."""
    pairs = [
        (0.0, 0.0),
        (0.10, 0.04),
        (0.25, 0.10),
        (0.30, 0.08),
        (-0.30, -0.08),
        (0.80, 0.40),
        (-0.80, -0.40),
    ]
    return [
        {
            "lateral_error_m": lateral,
            "heading_error_rad": heading,
            "expected_scale": tracking_safety_speed_scale(lateral, heading),
        }
        for lateral, heading in pairs
    ]


def speed_profile_vectors() -> list[dict[str, object]]:
    """Export full periodic profiles, not only a few interpolation points."""
    s_m = np.array([0.0, 4.0, 9.0, 15.0, 22.0, 30.0, 39.0, 49.0, 60.0, 72.0])
    curvature = np.array([0.0, 0.03, 0.08, 0.12, 0.04, -0.02, -0.10, -0.06, 0.01, 0.0])
    cases = []
    for case_id, grip, utilisation in (
        ("nominal", 1.00, 0.75),
        ("front_limited", 0.68, 0.98),
        ("low_grip", 0.50, 0.95),
    ):
        profile = build_speed_profile(
            s_m=s_m,
            length_m=85.0,
            curvature_1pm=curvature,
            friction_coefficient=grip,
            lateral_utilisation=utilisation,
            maximum_speed_mps=12.0,
            minimum_speed_mps=4.0,
            acceleration_limit_mps2=3.0,
            braking_limit_mps2=2.5,
            passes=4,
        )
        queries = np.array([-0.05, 0.0, 0.13, 0.51, 0.92, 1.07])
        cases.append(
            {
                "id": case_id,
                "s_m": s_m.tolist(),
                "length_m": 85.0,
                "curvature_1pm": curvature.tolist(),
                "friction_coefficient": grip,
                "lateral_utilisation": utilisation,
                "maximum_speed_mps": 12.0,
                "minimum_speed_mps": 4.0,
                "acceleration_limit_mps2": 3.0,
                "braking_limit_mps2": 2.5,
                "passes": 4,
                "expected_progress": profile.progress.tolist(),
                "expected_speed_mps": profile.speed_mps.tolist(),
                "expected_lateral_limit_mps2": profile.lateral_limit_mps2,
                "queries": queries.tolist(),
                "expected_query_speed_mps": [
                    profile.at_progress(float(query)) for query in queries
                ],
            }
        )
    return cases


def model_and_move_vectors(policy: list[dict[str, object]]) -> list[dict[str, object]]:
    """Export current 30-step/2.5-rad-s C2 matrices and OSQP first moves."""
    nominal, _ = nominal_parameters()
    mpc = LateralMpc(horizon=30, delta_rate_max_rad_s=2.5)
    vectors: list[dict[str, object]] = []
    selected = [item for item in policy if item["observable"]][-20:]
    for case_index, item in enumerate(selected):
        vx = max(float(item["vx_mps"]), 6.0)
        parameters = VehicleParameters(
            mass_kg=nominal.mass_kg,
            yaw_inertia_kgm2=nominal.yaw_inertia_kgm2,
            wheelbase_m=nominal.wheelbase_m,
            front_axle_fraction=nominal.front_axle_fraction,
            front_cornering_stiffness_n_rad=(
                nominal.front_cornering_stiffness_n_rad
                * float(item["expected_filtered_front_grip"])
            ),
            rear_cornering_stiffness_n_rad=(
                nominal.rear_cornering_stiffness_n_rad
                * float(item["expected_filtered_rear_grip"])
            ),
            sample_time_s=0.02,
        )
        matrix_a, matrix_b = build_tracking_plant(vx, parameters)
        x0 = np.array(
            [
                0.04 * np.sin(0.7 * case_index),
                0.018 * np.cos(0.5 * case_index),
                0.03 * np.sin(0.3 * case_index),
                0.08 * np.cos(0.4 * case_index),
            ]
        )
        phase = np.linspace(0.0, 1.2, 30)
        curvature = 0.025 * np.sin(phase + 0.2 * case_index) + 0.008
        previous = 0.012 * np.sin(0.4 * case_index)
        solution = mpc.solve(
            x0=x0,
            curvature_preview=curvature,
            vx_mps=vx,
            a_aug=matrix_a,
            b_aug=matrix_b,
            sample_time_s=0.02,
            previous_delta_rad=previous,
        )
        if not solution.valid:
            raise RuntimeError(f"golden MPC case {case_index} failed: {solution.status}")
        vectors.append(
            {
                "id": f"schema4_case_{case_index:02d}",
                "vx_mps": vx,
                "filtered_front_grip": float(item["expected_filtered_front_grip"]),
                "filtered_rear_grip": float(item["expected_filtered_rear_grip"]),
                "a_tracking": matrix_a.tolist(),
                "b_tracking": matrix_b.tolist(),
                "state": x0.tolist(),
                "curvature_preview_1pm": curvature.tolist(),
                "previous_steering_rad": previous,
                "python_first_move_rad": solution.first_move_rad,
            }
        )
    return vectors


def export_c0_smoke_golden(path: Path) -> None:
    """Refresh the small legacy C0 smoke artifact with current rate limits."""
    nominal, nominal_hash = nominal_parameters()
    cases = [
        ("zero_straight", 4.5, [0.0, 0.0, 0.0, 0.0], 0.0, 0.0),
        ("small_signal", 4.5, [0.005, 0.0, 0.0, 0.0], 0.0, 0.0),
        ("positive_tracking", 4.5, [0.2, 0.05, 0.0, 0.0], 0.0, 0.0),
        ("negative_tracking", 4.5, [-0.35, -0.08, 0.03, -0.02], 0.0, 0.0),
        ("curved_entry", 4.5, [0.1, 0.03, 0.02, 0.01], 0.04, -0.02),
    ]
    controller = LateralMpc(horizon=20, delta_rate_max_rad_s=2.5)
    vectors = []
    for case_id, vx, state, curvature, previous in cases:
        matrix_a, matrix_b = build_tracking_plant(vx, nominal)
        solution = controller.solve(
            x0=np.asarray(state, dtype=float),
            curvature_preview=np.full(20, curvature),
            vx_mps=vx,
            a_aug=matrix_a,
            b_aug=matrix_b,
            sample_time_s=0.02,
            previous_delta_rad=previous,
        )
        if not solution.valid:
            raise RuntimeError(f"C0 smoke golden failed for {case_id}: {solution.status}")
        vectors.append(
            {
                "id": case_id,
                "vx_mps": vx,
                "state": state,
                "curvature_1pm": curvature,
                "previous_steering_rad": previous,
                "python_first_move_rad": solution.first_move_rad,
            }
        )
    payload = {
        "schema_version": 2,
        "source": "controllers/python/mpc_qp.py LateralMpc / OSQP",
        "nominal_fit_sha256": nominal_hash,
        "sample_time_s": 0.02,
        "mpc_horizon": 20,
        "steering_rate_limit_rad_s": 2.5,
        "vectors": vectors,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    """Write one stable artifact consumed by MATLAB parity tests."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY_ROOT / "matlab" / "data" / "schema4_policy_golden_v1.json",
    )
    parser.add_argument(
        "--c0-output",
        type=Path,
        default=REPOSITORY_ROOT / "matlab" / "data" / "python_c0_first_move_golden_v1.json",
    )
    arguments = parser.parse_args()
    policy = policy_vectors()
    _, nominal_hash = nominal_parameters()
    payload = {
        "schema_version": 1,
        "source": "NeuroGrip-X Python release controller policy",
        "sample_time_s": 0.02,
        "mpc_horizon": 30,
        "steering_rate_limit_rad_s": 2.5,
        "grip_filter_alpha": 0.05,
        "grip_error_margin_scale": 0.50,
        "nominal_fit_sha256": nominal_hash,
        "source_sha256": {
            name: file_sha256(CONTROLLER_ROOT / name)
            for name in (
                "grip_policy.py",
                "speed_profile.py",
                "bicycle_model.py",
                "mpc_qp.py",
            )
        },
        "policy_vectors": policy,
        "tracking_safety_vectors": safety_vectors(),
        "speed_profile_vectors": speed_profile_vectors(),
        "model_and_move_vectors": model_and_move_vectors(policy),
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    export_c0_smoke_golden(arguments.c0_output)
    print(f"SCHEMA4_GOLDEN_EXPORT_PASS {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
