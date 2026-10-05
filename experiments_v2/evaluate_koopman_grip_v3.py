#!/usr/bin/env python3
"""Audit held-out dynamics and per-axle grip estimates for schema-3 artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "learning"))
sys.path.insert(0, str(REPO_ROOT / "src/neurogrip_ai"))
sys.path.insert(0, str(REPO_ROOT / "experiments_v2"))

from neurogrip.config import NeuroGripConfig  # noqa: E402
from neurogrip.koopman_grip import GripAwareKoopmanModel  # noqa: E402
from train_koopman_grip_v3 import load_splits, outputs, tensors  # noqa: E402


def portable_path(path: Path) -> str:
    """Return a repository-relative path when the file is in this checkout."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def parse_arguments() -> argparse.Namespace:
    """Parse immutable model, data and report locations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, action="append", required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--nominal-fit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_models(manifest_path: Path) -> tuple[dict, list[GripAwareKoopmanModel]]:
    """Load exactly the hash-linked deployment ensemble on CPU."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != 3
        or manifest.get("model_family") != "contextual_neural_koopman_grip_v3"
    ):
        raise ValueError("expected a schema-3 grip-aware Koopman manifest")
    models = []
    for member in manifest.get("members", []):
        bundle = torch.load(
            manifest_path.parent / member["path"], map_location="cpu", weights_only=False
        )
        model = GripAwareKoopmanModel(NeuroGripConfig(**bundle["config"]))
        model.load_state_dict(bundle["state_dict"])
        model.eval()
        models.append(model)
    if len(models) != 3:
        raise ValueError("evaluation requires the three-member deployment ensemble")
    return manifest, models


def error_summary(prediction: np.ndarray, target: np.ndarray) -> dict:
    """Return interpretable front/rear absolute-error statistics."""
    error = prediction - target
    absolute = np.abs(error)
    return {
        "samples": int(len(target)),
        "target_front_mean": float(np.mean(target[:, 0])),
        "target_rear_mean": float(np.mean(target[:, 1])),
        "estimated_front_mean": float(np.mean(prediction[:, 0])),
        "estimated_rear_mean": float(np.mean(prediction[:, 1])),
        "front_mae": float(np.mean(absolute[:, 0])),
        "rear_mae": float(np.mean(absolute[:, 1])),
        "front_rmse": float(np.sqrt(np.mean(error[:, 0] ** 2))),
        "rear_rmse": float(np.sqrt(np.mean(error[:, 1] ** 2))),
        "front_absolute_error_q90": float(np.quantile(absolute[:, 0], 0.90)),
        "rear_absolute_error_q90": float(np.quantile(absolute[:, 1], 0.90)),
    }


def main() -> int:
    """Write an auditable acceptance report without modifying model weights."""
    arguments = parse_arguments()
    manifest_path = arguments.manifest.expanduser().resolve()
    manifest, models = load_models(manifest_path)
    fit = json.loads(arguments.nominal_fit.expanduser().read_text(encoding="utf-8"))
    splits = load_splits(
        [path.expanduser().resolve() for path in arguments.run_root],
        arguments.split_manifest.expanduser().resolve(),
        fit,
    )
    first_bundle = torch.load(
        manifest_path.parent / manifest["members"][0]["path"],
        map_location="cpu",
        weights_only=False,
    )
    mean = np.asarray(first_bundle["feature_mean"], dtype=np.float32)
    scale = np.asarray(first_bundle["feature_scale"], dtype=np.float32)
    report = {
        "schema_version": 1,
        "model_family": manifest["model_family"],
        "manifest": portable_path(manifest_path),
        "splits": {},
    }
    acceptance_values = []
    for split_name, windows in splits.items():
        values = tensors(windows, mean, scale, torch.device("cpu"))
        _, _, member_grip = outputs(models, values)
        prediction = member_grip.mean(axis=0)
        target = windows.grip
        split_report = error_summary(prediction, target)
        scenarios = {}
        identifiers = np.asarray(windows.scenario_ids)
        for scenario_id in sorted(set(windows.scenario_ids)):
            mask = identifiers == scenario_id
            scenario_report = error_summary(prediction[mask], target[mask])
            regimes = {}
            for pair in np.unique(target[mask], axis=0):
                regime_mask = mask & np.all(np.isclose(target, pair, atol=1e-6), axis=1)
                label = f"front_{pair[0]:.2f}_rear_{pair[1]:.2f}"
                regimes[label] = error_summary(prediction[regime_mask], target[regime_mask])
            scenario_report["regimes"] = regimes
            scenarios[scenario_id] = scenario_report
        split_report["scenarios"] = scenarios
        report["splits"][split_name] = split_report
        if split_name in {"validation", "calibration"}:
            acceptance_values.extend(
                [split_report["front_mae"], split_report["rear_mae"]]
            )
    threshold = 0.08
    report["acceptance"] = {
        "per_axle_mae_limit": threshold,
        "maximum_validation_or_calibration_mae": float(max(acceptance_values)),
        "passed": bool(max(acceptance_values) <= threshold),
        "note": "Simulation grip labels are evaluation targets, never runtime inputs.",
    }
    output = arguments.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        "KOOPMAN_GRIP_ACCEPTANCE_%s max_mae=%.5f report=%s"
        % (
            "PASS" if report["acceptance"]["passed"] else "FAIL",
            report["acceptance"]["maximum_validation_or_calibration_mae"],
            output,
        )
    )
    return 0 if report["acceptance"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
