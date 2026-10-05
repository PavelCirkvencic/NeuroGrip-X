#!/usr/bin/env python3
"""Recalibrate a frozen Koopman ensemble on hashed development closed-loop runs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "experiments_v2"))

from evaluate_koopman_v2 import load_ensemble  # noqa: E402
from train_physics_residual import FEATURE_NAMES, nominal_matrices  # noqa: E402


def portable_path(path: Path) -> str:
    """Return a repository-relative path when the file is in this checkout."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(resolved)

FIELD_MAP = {
    "v_x_mps": "v_x_mps",
    "v_y_mps": "v_y_mps",
    "yaw_rate_rps": "yaw_rate_rps",
    "steering_applied_rad": "steering_applied_rad",
    "acceleration_command_mps2": "acceleration_command_mps2",
    "a_x_mps2": "a_x_mps2",
    "a_y_mps2": "a_y_mps2",
}


def parse_arguments() -> argparse.Namespace:
    """Parse a frozen source manifest and explicit development episodes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ensemble-manifest", type=Path, required=True)
    parser.add_argument("--nominal-fit", type=Path, required=True)
    parser.add_argument("--episode", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    """Return a complete file digest for calibration provenance."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_episode(path: Path) -> dict[str, np.ndarray]:
    """Read finite, continuous schema-v3 telemetry needed by the model."""
    required = ["time_s", *FIELD_MAP.values()]
    values = {name: [] for name in required}
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            for name in required:
                values[name].append(float(row[name]))
    result = {name: np.asarray(data, dtype=np.float32) for name, data in values.items()}
    if len(result["time_s"]) < 100 or not all(
        np.all(np.isfinite(data)) for data in result.values()
    ):
        raise ValueError(f"invalid calibration episode: {path}")
    dt = np.diff(result["time_s"])
    if np.any(dt <= 0.0) or not 49.0 <= 1.0 / np.median(dt) <= 51.0:
        raise ValueError(f"calibration episode is not continuous 50 Hz data: {path}")
    return result


def episode_windows(episode: dict[str, np.ndarray], fit: dict):
    """Build causal one-step windows without changing any model weights."""
    features = np.column_stack([episode[FIELD_MAP[name]] for name in FEATURE_NAMES])
    histories, states, steering, targets, matrices_a, matrices_b = [], [], [], [], [], []
    for index in range(24, len(features) - 1):
        histories.append(features[index - 24:index + 1])
        states.append(features[index, 1:3])
        steering.append(features[index, 3])
        targets.append(features[index + 1, 1:3])
        matrix_a, matrix_b = nominal_matrices(features[index, 0], fit)
        matrices_a.append(matrix_a)
        matrices_b.append(matrix_b)
    return tuple(
        np.asarray(value, dtype=np.float32)
        for value in (histories, states, steering, targets, matrices_a, matrices_b)
    )


def main() -> int:
    """Write a new manifest whose only changed values are calibrated thresholds."""
    arguments = parse_arguments()
    source = arguments.ensemble_manifest.expanduser().resolve()
    nominal_fit = arguments.nominal_fit.expanduser().resolve()
    manifest, models, mean, scale = load_ensemble(source)
    fit = json.loads(nominal_fit.read_text(encoding="utf-8"))
    all_contexts, all_predictions, all_targets = [], [], []
    provenance = []
    for requested in arguments.episode:
        episode_path = requested.expanduser().resolve()
        windows = episode_windows(read_episode(episode_path), fit)
        history, state, steering, target, matrix_a, matrix_b = windows
        normalized = torch.as_tensor((history - mean) / scale, dtype=torch.float32)
        state_tensor = torch.as_tensor(state)
        steering_tensor = torch.as_tensor(steering[:, None])
        a_tensor, b_tensor = torch.as_tensor(matrix_a), torch.as_tensor(matrix_b)
        member_contexts, member_predictions = [], []
        with torch.no_grad():
            for model in models:
                output = model(
                    normalized,
                    state_tensor,
                    steering_tensor,
                    a_tensor,
                    b_tensor,
                )
                member_contexts.append(output.context.numpy())
                member_predictions.append(output.state_rollout[:, 0].numpy())
        all_contexts.append(np.mean(np.stack(member_contexts), axis=0))
        all_predictions.append(np.stack(member_predictions))
        all_targets.append(target)
        metadata = episode_path.with_suffix(".metadata.json")
        provenance.append(
            {
                "episode_csv": portable_path(episode_path),
                "episode_csv_sha256": sha256(episode_path),
                "metadata_sha256": sha256(metadata) if metadata.is_file() else None,
            }
        )
    contexts = np.concatenate(all_contexts, axis=0)
    member_predictions = np.concatenate(all_predictions, axis=1)
    targets = np.concatenate(all_targets, axis=0)
    ensemble_prediction = member_predictions.mean(axis=0)
    errors = np.linalg.norm(ensemble_prediction - targets, axis=1)
    disagreement = np.linalg.norm(np.std(member_predictions, axis=0), axis=1)
    calibration = manifest["calibration"]
    center = np.asarray(calibration["context_mean"], dtype=float)
    inverse = np.asarray(calibration["context_covariance_inverse"], dtype=float)
    residual = contexts - center
    ood_scores = np.sum((residual @ inverse) * residual, axis=1)
    updated = json.loads(json.dumps(manifest))
    updated["parent_manifest_sha256"] = sha256(source)
    updated["calibration"] = {
        **calibration,
        "conformal_radius": float(np.quantile(errors, 0.90)),
        "ood_threshold": float(
            max(float(calibration["ood_threshold"]), np.quantile(ood_scores, 0.99))
        ),
        "closed_loop_development": {
            "episodes": provenance,
            "sample_count": int(len(errors)),
            "one_step_error_q90": float(np.quantile(errors, 0.90)),
            "ensemble_disagreement_q90": float(np.quantile(disagreement, 0.90)),
            "ood_score_q99": float(np.quantile(ood_scores, 0.99)),
            "weights_updated": False,
        },
    }
    output = arguments.output.expanduser().resolve()
    if output.parent != source.parent:
        raise ValueError("recalibrated manifest must stay beside its member checkpoints")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite calibration artifact: {output}")
    output.write_text(json.dumps(updated, indent=2, sort_keys=True) + "\n")
    print(
        "KOOPMAN_CLOSED_LOOP_CALIBRATED "
        f"radius={updated['calibration']['conformal_radius']:.6f} "
        f"ood={updated['calibration']['ood_threshold']:.3f} output={output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
