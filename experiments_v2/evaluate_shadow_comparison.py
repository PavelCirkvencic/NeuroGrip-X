#!/usr/bin/env python3
"""Compare read-only MATLAB Adaptive MPC shadow moves with Python C2 telemetry.

This evaluator is diagnostic only.  It joins MATLAB's shadow trace to the
existing v2 episode by simulation timestamp and reports agreement, saturation
and timing coverage.  It neither creates ROS nodes nor authorizes a controller
performance claim.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

EPISODE_FIELDS = {
    "time_s",
    "steering_applied_rad",
    "steering_safe_rad",
    "e_y_m",
    "e_psi_rad",
    "v_x_mps",
    "d_kappa_rad_s",
}
SHADOW_FIELDS = {
    "time_s",
    "shadow_steering_rad",
    "applied_steering_rad",
    "e_y_m",
    "e_psi_rad",
    "v_y_mps",
    "yaw_rate_rps",
    "v_x_mps",
    "d_kappa_rad_s",
    "lateral_spectral_radius",
}
STEERING_LIMIT_RAD = 0.37


def parse_arguments() -> argparse.Namespace:
    """Parse explicit, non-destructive input/output paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-csv", type=Path, required=True)
    parser.add_argument("--shadow-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--max-time-delta-s", type=float, default=0.025)
    parser.add_argument("--min-samples", type=int, default=20)
    return parser.parse_args()


def load_csv(path: Path, required_fields: set[str]) -> dict[str, np.ndarray]:
    """Load a numeric trace strictly, preserving every supplied sample."""
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        missing = required_fields - fieldnames
        if missing:
            raise ValueError(f"{path}: missing fields {sorted(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path}: contains no samples")
    result: dict[str, np.ndarray] = {}
    for field in required_fields:
        try:
            result[field] = np.asarray([float(row[field]) for row in rows], dtype=float)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"{path}: invalid numeric {field}") from error
    if any(not np.all(np.isfinite(values)) for values in result.values()):
        raise ValueError(f"{path}: contains NaN or Inf")
    if np.any(np.diff(result["time_s"]) <= 0.0):
        raise ValueError(f"{path}: timestamps are not strictly increasing")
    return result


def nearest_episode_indices(
    episode_time_s: np.ndarray, shadow_time_s: np.ndarray, max_time_delta_s: float
) -> tuple[np.ndarray, np.ndarray]:
    """Match each shadow sample to at most one nearest 50 Hz episode row."""
    indices, _, deltas = matching_indices(
        episode_time_s, shadow_time_s, max_time_delta_s
    )
    return indices, deltas


def matching_indices(
    episode_time_s: np.ndarray, shadow_time_s: np.ndarray, max_time_delta_s: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return aligned episode indices, shadow indices and their time deltas."""
    positions = np.searchsorted(episode_time_s, shadow_time_s, side="left")
    right = np.clip(positions, 0, len(episode_time_s) - 1)
    left = np.clip(positions - 1, 0, len(episode_time_s) - 1)
    choose_left = np.abs(shadow_time_s - episode_time_s[left]) <= np.abs(
        shadow_time_s - episode_time_s[right]
    )
    indices = np.where(choose_left, left, right)
    deltas = np.abs(shadow_time_s - episode_time_s[indices])
    keep = deltas <= max_time_delta_s
    return indices[keep], np.flatnonzero(keep), deltas[keep]


def summarize(
    episode: dict[str, np.ndarray], shadow: dict[str, np.ndarray], max_time_delta_s: float
) -> dict:
    """Return transparent diagnostic statistics; no metric is a race result."""
    indices, shadow_indices, deltas = matching_indices(
        episode["time_s"], shadow["time_s"], max_time_delta_s
    )
    if len(indices) == 0:
        raise ValueError("no shadow samples match the episode within the time tolerance")

    python_move = episode["steering_safe_rad"][indices]
    shadow_move = shadow["shadow_steering_rad"][shadow_indices]
    difference = shadow_move - python_move
    nonzero = (np.abs(python_move) > 1e-4) & (np.abs(shadow_move) > 1e-4)
    return {
        "schema_version": 1,
        "matched_samples": int(len(indices)),
        "unmatched_shadow_samples": int(len(shadow["time_s"]) - len(indices)),
        "max_time_delta_s": float(np.max(deltas)),
        "mean_abs_time_delta_s": float(np.mean(deltas)),
        "shadow_minus_python_mae_rad": float(np.mean(np.abs(difference))),
        "shadow_minus_python_p95_abs_rad": float(np.percentile(np.abs(difference), 95)),
        "shadow_lastmove_minus_episode_applied_mae_rad": float(
            np.mean(
                np.abs(
                    shadow["applied_steering_rad"][shadow_indices]
                    - episode["steering_applied_rad"][indices]
                )
            )
        ),
        "shadow_saturation_fraction": float(
            np.mean(np.abs(shadow_move) >= STEERING_LIMIT_RAD - 1e-8)
        ),
        "python_saturation_fraction": float(
            np.mean(np.abs(python_move) >= STEERING_LIMIT_RAD - 1e-8)
        ),
        "nonzero_sign_agreement_fraction": (
            None
            if not np.any(nonzero)
            else float(np.mean(np.sign(shadow_move[nonzero]) == np.sign(python_move[nonzero])))
        ),
        "max_shadow_lateral_spectral_radius": float(
            np.max(shadow["lateral_spectral_radius"][shadow_indices])
        ),
    }


def main() -> int:
    """Write a deterministic diagnostic JSON and concise terminal summary."""
    arguments = parse_arguments()
    if not 0.0 < arguments.max_time_delta_s <= 0.1:
        raise ValueError("--max-time-delta-s must be in (0, 0.1]")
    if arguments.min_samples < 1:
        raise ValueError("--min-samples must be positive")
    episode = load_csv(arguments.episode_csv.expanduser().resolve(), EPISODE_FIELDS)
    shadow = load_csv(arguments.shadow_csv.expanduser().resolve(), SHADOW_FIELDS)
    report = summarize(episode, shadow, arguments.max_time_delta_s)
    if report["matched_samples"] < arguments.min_samples:
        raise ValueError(
            f"only {report['matched_samples']} timestamp matches; need {arguments.min_samples}"
        )
    output = arguments.output_json.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        "SHADOW_COMPARISON_SUMMARY "
        f"matched={report['matched_samples']} "
        f"mae_rad={report['shadow_minus_python_mae_rad']:.6f} "
        f"shadow_sat={report['shadow_saturation_fraction']:.3f} "
        f"python_sat={report['python_saturation_fraction']:.3f} "
        f"lastmove_mae_rad={report['shadow_lastmove_minus_episode_applied_mae_rad']:.3e}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
