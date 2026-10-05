#!/usr/bin/env python3
"""Calibrate uncertainty for a trained NeuroGrip-X context model.

Validation windows are the dedicated calibration split; the locked test split
is only used to *evaluate* coverage.  The script writes a ``calibration.json``
with the radius, measured coverage and an OOD score summary.  It never mutates
the model or increases any control command.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from neurogrip.config import NeuroGripConfig
from neurogrip.data import build_split_windows, load_catalog, run_frames_by_split
from neurogrip.model import NeuroGripModel
from neurogrip.uncertainty import (
    conformal_quantile,
    empirical_coverage,
    mahalanobis_scores,
    normalized_scores,
    per_horizon_sigma,
)


def parse_arguments() -> argparse.Namespace:
    """Parse model, catalog and calibration options."""
    parser = argparse.ArgumentParser(description="Calibrate NeuroGrip-X uncertainty.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("runs/models/scenario_neurogrip_latest")
    )
    parser.add_argument("--alpha", type=float, default=0.10)
    return parser.parse_args()


def git_commit() -> str:
    """Return the current Git commit, or 'unknown'."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False, timeout=2
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def load_model(checkpoint_path: Path, device: torch.device):
    """Load a NeuroGrip checkpoint and return (model, config, statistics)."""
    bundle = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = NeuroGripConfig(**bundle["config"])
    model = NeuroGripModel(config, context_mode=bundle["context_mode"]).to(device)
    model.load_state_dict(bundle["state_dict"])
    model.eval()
    statistics = {key: np.asarray(value) for key, value in bundle["statistics"].items()}
    return model, config, statistics


def predict(batch, model, statistics, device):
    """Return physical rollout predictions, targets and context embeddings."""
    state_mean, state_scale = statistics["state_mean"], statistics["state_scale"]
    cmd_mean, cmd_scale = statistics["cmd_mean"], statistics["cmd_scale"]
    history_mean = np.concatenate([state_mean, cmd_mean])
    history_scale = np.concatenate([state_scale, cmd_scale])
    with torch.no_grad():
        history = torch.tensor(
            (batch.history - history_mean) / history_scale, dtype=torch.float32, device=device
        )
        current = torch.tensor(
            (batch.current_state - state_mean) / state_scale, dtype=torch.float32, device=device
        )
        commands = torch.tensor(
            (batch.future_commands - cmd_mean) / cmd_scale, dtype=torch.float32, device=device
        )
        output = model(history, current, commands)
    predicted = output.rollout.cpu().numpy() * state_scale + state_mean
    contexts = output.context.cpu().numpy()
    return predicted, batch.target_states, contexts


def main() -> int:
    """Compute conformal calibration and OOD statistics and save them."""
    arguments = parse_arguments()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config, statistics = load_model(arguments.model.expanduser().resolve(), device)

    catalog = load_catalog(arguments.dataset_manifest)
    grouped = run_frames_by_split(catalog)
    split_batches = build_split_windows(grouped, config)

    splits = ("train", "calibration", "development_test")
    predictions = {}
    contexts = {}
    for split in splits:
        predictions[split], _, contexts[split] = predict(
            split_batches[split], model, statistics, device
        )
    train_targets = split_batches["train"].target_states
    calibration_targets = split_batches["calibration"].target_states
    development_targets = split_batches["development_test"].target_states

    # Raw uncertainty scale comes from *train* residuals; the conformal radius
    # is calibrated on a disjoint calibration split; the development test and
    # (only in the final benchmark) the sealed test are evaluation only.
    sigma = per_horizon_sigma(predictions["train"], train_targets)
    calibration_scores = normalized_scores(
        predictions["calibration"], calibration_targets, sigma
    )
    radius = conformal_quantile(calibration_scores, arguments.alpha)

    calibration_coverage, calibration_width = empirical_coverage(
        predictions["calibration"], calibration_targets, sigma, radius
    )
    development_coverage, development_width = empirical_coverage(
        predictions["development_test"], development_targets, sigma, radius
    )

    train_ood = mahalanobis_scores(contexts["train"], contexts["train"])
    calibration_ood = mahalanobis_scores(contexts["train"], contexts["calibration"])
    development_ood = mahalanobis_scores(
        contexts["train"], contexts["development_test"]
    )
    threshold = float(np.quantile(train_ood, 0.95))

    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_path": str(arguments.model.expanduser().resolve()),
        "model_sha256": hashlib.sha256(
            arguments.model.expanduser().resolve().read_bytes()
        ).hexdigest(),
        "catalog_path": str(arguments.dataset_manifest.expanduser().resolve()),
        "catalog_sha256": hashlib.sha256(
            arguments.dataset_manifest.expanduser().resolve().read_bytes()
        ).hexdigest(),
        "git_commit": git_commit(),
        "alpha": arguments.alpha,
        "target_coverage": 1.0 - arguments.alpha,
        "scale_fit_split": "train",
        "calibration_split": "calibration",
        "evaluation_split": "development_test",
        "sealed_test_used": False,
        "conformal_radius": radius,
        "sigma_by_horizon": sigma.tolist(),
        "coverage": {
            "calibration": calibration_coverage,
            "development_test": development_coverage,
            "development_test_interval_width": development_width,
            "calibration_interval_width": calibration_width,
        },
        "ood": {
            "train_threshold_p95": threshold,
            "calibration_fraction_above": float((calibration_ood > threshold).mean()),
            "development_fraction_above": float(
                (development_ood > threshold).mean()
            ),
            "development_mean_distance": float(development_ood.mean()),
            "limitations": (
                "Split conformal gives marginal coverage under exchangeability; "
                "abrupt OOD transitions can break that interpretation. The OOD "
                "score is a context-distance heuristic, not a validated detector."
            ),
        },
    }
    output_dir = arguments.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "calibration.json"
    output_path.write_text(json.dumps(report, indent=2, default=float) + "\n", encoding="utf-8")

    print(f"Conformal radius ({1 - arguments.alpha:.0%}): {radius:.4f}")
    print(f"Calibration coverage: {calibration_coverage:.3f}")
    print(
        f"Development-test coverage: {development_coverage:.3f} "
        f"(width {development_width:.5f})"
    )
    print(
        "Development-test OOD fraction above train p95: "
        f"{report['ood']['development_fraction_above']:.3f}"
    )
    print(f"Calibration: {output_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError) as error:
        print(f"Calibration failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error
