#!/usr/bin/env python3
"""Compare the learned physical ensemble with nominal multistep rollouts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src/neurogrip_ai"))

from neurogrip_ai.physics_residual import (  # noqa: E402
    PhysicsResidualConfig,
    PhysicsResidualNet,
)
from train_physics_residual import HORIZONS, load_splits  # noqa: E402


def parse_arguments() -> argparse.Namespace:
    """Parse immutable data/model artifacts and report destination."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--nominal-fit", type=Path, required=True)
    parser.add_argument("--ensemble-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2048)
    return parser.parse_args()


def load_ensemble(path: Path) -> tuple[list[PhysicsResidualNet], np.ndarray, np.ndarray]:
    """Load the three immutable checkpoints and shared normalization."""
    manifest = json.loads(path.read_text(encoding="utf-8"))
    models, mean, scale = [], None, None
    for member in manifest["members"]:
        bundle = torch.load(path.parent / member["path"], map_location="cpu", weights_only=False)
        model = PhysicsResidualNet(PhysicsResidualConfig(**bundle["config"]))
        model.load_state_dict(bundle["state_dict"])
        model.eval()
        models.append(model)
        member_mean = np.asarray(bundle["feature_mean"], dtype=np.float32)
        member_scale = np.asarray(bundle["feature_scale"], dtype=np.float32)
        if mean is None:
            mean, scale = member_mean, member_scale
        elif not (np.array_equal(mean, member_mean) and np.array_equal(scale, member_scale)):
            raise ValueError("ensemble members disagree on normalization")
    if len(models) != 3 or mean is None or scale is None:
        raise ValueError("expected exactly three complete ensemble members")
    return models, mean, scale


def ensemble_matrices(
    models: list[PhysicsResidualNet],
    history: np.ndarray,
    nominal_a: np.ndarray,
    nominal_b: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate all members in bounded batches and return mean A/B."""
    matrices_a, matrices_b = [], []
    with torch.no_grad():
        for start in range(0, len(history), batch_size):
            stop = min(start + batch_size, len(history))
            history_tensor = torch.as_tensor(
                (history[start:stop] - mean) / scale, dtype=torch.float32
            )
            a_tensor = torch.as_tensor(nominal_a[start:stop], dtype=torch.float32)
            b_tensor = torch.as_tensor(nominal_b[start:stop], dtype=torch.float32)
            member_outputs = [model(history_tensor, a_tensor, b_tensor) for model in models]
            matrices_a.append(
                torch.stack([output[0] for output in member_outputs]).mean(0).numpy()
            )
            matrices_b.append(
                torch.stack([output[1] for output in member_outputs]).mean(0).numpy()
            )
    return np.concatenate(matrices_a), np.concatenate(matrices_b)


def rollout(
    matrix_a: np.ndarray,
    matrix_b: np.ndarray,
    state: np.ndarray,
    commands: np.ndarray,
) -> np.ndarray:
    """Roll one fixed local model to every declared evaluation horizon."""
    predicted = state.copy()
    outputs, horizon_index = [], 0
    for step in range(1, max(HORIZONS) + 1):
        predicted = (
            np.einsum("nij,nj->ni", matrix_a, predicted)
            + matrix_b[:, :, 0] * commands[:, step - 1, None]
        )
        if step == HORIZONS[horizon_index]:
            outputs.append(predicted.copy())
            horizon_index += 1
    return np.stack(outputs, axis=1)


def split_report(windows, models, mean, scale, batch_size: int) -> dict:
    """Return per-horizon physical errors and learned reductions for one split."""
    learned_a, learned_b = ensemble_matrices(
        models,
        windows.history,
        windows.nominal_a,
        windows.nominal_b,
        mean,
        scale,
        batch_size,
    )
    predictions = {
        "nominal": rollout(
            windows.nominal_a, windows.nominal_b, windows.state, windows.commands
        ),
        "learned": rollout(
            learned_a, learned_b, windows.state, windows.commands
        ),
    }
    report = {"windows": len(windows.history), "horizons": {}}
    for index, horizon in enumerate(HORIZONS):
        metrics = {}
        for name, prediction in predictions.items():
            error = prediction[:, index] - windows.targets[:, index]
            metrics[name] = {
                "v_y_rmse_mps": float(np.sqrt(np.mean(error[:, 0] ** 2))),
                "yaw_rate_rmse_rps": float(np.sqrt(np.mean(error[:, 1] ** 2))),
                "vector_rmse": float(np.sqrt(np.mean(np.sum(error**2, axis=1)))),
            }
        nominal = metrics["nominal"]["vector_rmse"]
        learned = metrics["learned"]["vector_rmse"]
        metrics["relative_reduction_percent"] = 100.0 * (nominal - learned) / nominal
        report["horizons"][str(horizon)] = metrics
    return report


def main() -> int:
    """Evaluate validation/calibration without touching closed-loop holdouts."""
    arguments = parse_arguments()
    fit = json.loads(arguments.nominal_fit.expanduser().resolve().read_text())
    splits = load_splits(
        arguments.run_root.expanduser().resolve(),
        arguments.split_manifest.expanduser().resolve(),
        fit,
    )
    manifest_path = arguments.ensemble_manifest.expanduser().resolve()
    models, mean, scale = load_ensemble(manifest_path)
    report = {
        "schema_version": 1,
        "ensemble_manifest": str(manifest_path),
        "splits": {
            name: split_report(
                splits[name], models, mean, scale, arguments.batch_size
            )
            for name in ("validation", "calibration")
        },
    }
    output = arguments.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
