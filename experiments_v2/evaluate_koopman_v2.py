#!/usr/bin/env python3
"""Evaluate contextual neural Koopman rollouts against the nominal bicycle model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "learning"))
sys.path.insert(0, str(REPO_ROOT / "experiments_v2"))

from neurogrip.config import NeuroGripConfig  # noqa: E402
from neurogrip.koopman_v2 import PhysicalKoopmanModel  # noqa: E402
from train_physics_residual import (  # noqa: E402
    HORIZONS,
    load_splits,
    tensors,
)


def parse_arguments() -> argparse.Namespace:
    """Parse immutable artifacts and report output."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--nominal-fit", type=Path, required=True)
    parser.add_argument("--ensemble-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_ensemble(path: Path):
    """Load only a hash-linked schema-2 Koopman ensemble."""
    import hashlib

    manifest = json.loads(path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != 2
        or manifest.get("model_family") != "contextual_neural_koopman_v2"
    ):
        raise ValueError("manifest is not a contextual neural Koopman v2 ensemble")
    models, mean, scale = [], None, None
    for member in manifest["members"]:
        checkpoint = path.parent / member["path"]
        digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        if digest != member["sha256"]:
            raise ValueError(f"checkpoint hash mismatch: {checkpoint}")
        bundle = torch.load(checkpoint, map_location="cpu", weights_only=False)
        config = NeuroGripConfig(**bundle["config"])
        model = PhysicalKoopmanModel(config)
        model.load_state_dict(bundle["state_dict"])
        model.eval()
        models.append(model)
        member_mean = np.asarray(bundle["feature_mean"], dtype=np.float32)
        member_scale = np.asarray(bundle["feature_scale"], dtype=np.float32)
        if mean is None:
            mean, scale = member_mean, member_scale
        elif not (np.allclose(mean, member_mean) and np.allclose(scale, member_scale)):
            raise ValueError("ensemble members use different normalization")
    if len(models) != 3:
        raise ValueError("evaluation requires exactly three members")
    return manifest, models, mean, scale


def nominal_rollout(values) -> np.ndarray:
    """Roll the fixed physical model with the exact same measured commands."""
    _, state, commands, _, matrix_a, matrix_b = values
    predicted = state
    selected = []
    for step in range(max(HORIZONS)):
        predicted = (
            torch.bmm(matrix_a, predicted.unsqueeze(-1)).squeeze(-1)
            + torch.bmm(
                matrix_b, commands[:, step].reshape(-1, 1, 1)
            ).squeeze(-1)
        )
        if step + 1 in HORIZONS:
            selected.append(predicted)
    return torch.stack(selected, dim=1).cpu().numpy()


def split_report(models, values) -> dict:
    """Return horizon-wise vector RMSE and stability for one untouched split."""
    member_rollouts, maximum_radii = [], []
    with torch.no_grad():
        for model in models:
            output = model(values[0], values[1], values[2], values[4], values[5])
            indices = torch.as_tensor([h - 1 for h in HORIZONS])
            member_rollouts.append(
                output.state_rollout.index_select(1, indices).cpu().numpy()
            )
            maximum_radii.append(
                float(torch.linalg.eigvals(output.koopman).abs().max().cpu())
            )
    learned = np.mean(np.stack(member_rollouts), axis=0)
    nominal = nominal_rollout(values)
    target = values[3].cpu().numpy()
    result = {"maximum_koopman_spectral_radius": max(maximum_radii), "horizons": {}}
    for index, horizon in enumerate(HORIZONS):
        learned_rmse = float(
            np.sqrt(np.mean(np.sum((learned[:, index] - target[:, index]) ** 2, axis=1)))
        )
        nominal_rmse = float(
            np.sqrt(np.mean(np.sum((nominal[:, index] - target[:, index]) ** 2, axis=1)))
        )
        result["horizons"][str(horizon)] = {
            "learned_vector_rmse": learned_rmse,
            "nominal_vector_rmse": nominal_rmse,
            "relative_reduction_percent": 100.0
            * (nominal_rmse - learned_rmse)
            / nominal_rmse,
        }
    return result


def main() -> int:
    """Evaluate validation/calibration splits without touching a final holdout."""
    arguments = parse_arguments()
    manifest_path = arguments.ensemble_manifest.expanduser().resolve()
    manifest, models, mean, scale = load_ensemble(manifest_path)
    fit = json.loads(arguments.nominal_fit.expanduser().read_text(encoding="utf-8"))
    splits = load_splits(
        arguments.run_root.expanduser().resolve(), arguments.split_manifest, fit
    )
    reports = {}
    for name in ("validation", "calibration"):
        values = tensors(splits[name], mean, scale, torch.device("cpu"))
        reports[name] = split_report(models, values)
    required = [
        reports[split]["horizons"][str(horizon)]["relative_reduction_percent"]
        for split in reports
        for horizon in (5, 10, 20)
    ]
    report = {
        "schema_version": 1,
        "model_family": manifest["model_family"],
        "ensemble_manifest": str(manifest_path),
        "splits": reports,
        "offline_gate": {
            "required_minimum_reduction_percent": 15.0,
            "required_horizons": [5, 10, 20],
            "passed": min(required) >= 15.0,
            "minimum_observed_reduction_percent": min(required),
        },
    }
    output = arguments.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["offline_gate"], sort_keys=True))
    return 0 if report["offline_gate"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
