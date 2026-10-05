#!/usr/bin/env python3
"""Train and calibrate the genuine contextual neural Koopman C2 ensemble."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "learning"))
sys.path.insert(0, str(REPO_ROOT / "src/neurogrip_ai"))
sys.path.insert(0, str(REPO_ROOT / "experiments_v2"))

from neurogrip.config import NeuroGripConfig  # noqa: E402
from neurogrip.koopman_v2 import PhysicalKoopmanModel  # noqa: E402
from neurogrip_ai.physics_residual import FEATURE_NAMES  # noqa: E402
from train_physics_residual import (  # noqa: E402
    HORIZONS,
    SEEDS,
    load_splits,
    sha256_path,
    standardization,
    tensors,
)


def parse_arguments() -> argparse.Namespace:
    """Parse immutable data inputs and bounded training settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--nominal-fit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=8e-4)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def model_config() -> NeuroGripConfig:
    """Return the versioned deployable model dimensions."""
    return NeuroGripConfig(
        state_dim=2,
        input_dim=1,
        history_steps=25,
        context_dim=8,
        latent_extra=6,
        rank=4,
        hidden_dim=48,
        rollout_steps=20,
        history_feature_dim=len(FEATURE_NAMES),
    )


def rollout_loss(model: PhysicalKoopmanModel, values, indices=None):
    """Combine physical rollout, Koopman consistency and stability losses."""
    history, state, commands, targets, nominal_a, nominal_b = values
    if indices is not None:
        history, state, commands, targets, nominal_a, nominal_b = (
            value[indices] for value in values
        )
    output = model(history, state, commands, nominal_a, nominal_b)
    horizon_indices = torch.as_tensor(
        [horizon - 1 for horizon in HORIZONS], device=history.device
    )
    predicted_states = output.state_rollout.index_select(1, horizon_indices)
    physical_loss = torch.mean((predicted_states - targets) ** 2)

    batch, horizon_count, _ = targets.shape
    repeated_context = output.context[:, None, :].expand(
        -1, horizon_count, -1
    )
    target_lift = model.lift_state(
        targets.reshape(-1, 2), repeated_context.reshape(-1, model.config.context_dim)
    ).reshape(batch, horizon_count, model.config.latent_dim)
    predicted_lift = output.latent_rollout.index_select(1, horizon_indices)
    koopman_consistency = torch.mean((predicted_lift - target_lift) ** 2)

    radius = torch.linalg.eigvals(output.koopman).abs().amax(dim=1)
    stability = torch.relu(radius - 1.02).pow(2).mean()
    physical_delta = output.koopman[:, :2, :2] - nominal_a
    input_delta = output.input_matrix[:, :2, :] - nominal_b
    residual_prior = physical_delta.pow(2).mean() + input_delta.pow(2).mean()
    total = (
        physical_loss
        + 0.05 * koopman_consistency
        + 0.20 * stability
        + 1e-3 * residual_prior
    )
    return total, physical_loss, output


def evaluate(model: PhysicalKoopmanModel, values) -> tuple[float, float]:
    """Return deterministic validation objective and physical rollout loss."""
    model.eval()
    with torch.no_grad():
        total, physical, _ = rollout_loss(model, values)
    return float(total.cpu()), float(physical.cpu())


def train_member(seed, train_values, validation_values, arguments, config, device):
    """Train one independently initialized ensemble member."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = PhysicalKoopmanModel(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=arguments.learning_rate, weight_decay=1e-5
    )
    best, best_epoch, best_state, stale = float("inf"), 0, None, 0
    count = train_values[0].shape[0]
    generator = torch.Generator().manual_seed(seed)
    for epoch in range(1, arguments.epochs + 1):
        model.train()
        permutation = torch.randperm(count, generator=generator, device=device)
        for start in range(0, count, arguments.batch_size):
            indices = permutation[start:start + arguments.batch_size]
            loss, _, _ = rollout_loss(model, train_values, indices)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        validation_loss, physical_loss = evaluate(model, validation_values)
        if validation_loss < best:
            best, best_epoch, stale = validation_loss, epoch, 0
            best_physical = physical_loss
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
        else:
            stale += 1
            if stale >= arguments.patience:
                break
    if best_state is None:
        raise RuntimeError("training produced no finite validation checkpoint")
    model.load_state_dict(best_state)
    return model, {
        "seed": seed,
        "best_epoch": best_epoch,
        "validation_loss": best,
        "validation_physical_loss": best_physical,
    }


def ensemble_outputs(models, values):
    """Return mean contexts, one-step predictions and local physical Jacobians."""
    contexts, predictions, matrices_a, matrices_b = [], [], [], []
    history, state, commands, _, nominal_a, nominal_b = values
    for model in models:
        model.eval()
        with torch.no_grad():
            output = model(history, state, commands[:, :1], nominal_a, nominal_b)
        contexts.append(output.context.cpu().numpy())
        predictions.append(output.state_rollout[:, 0].cpu().numpy())
        # The physical state is copied into the lift; autograd Jacobians include
        # the learned observable path. Calibration can use the direct physical
        # block cheaply, while deployment computes exact per-sample Jacobians.
        matrices_a.append(output.koopman[:, :2, :2].cpu().numpy())
        matrices_b.append(output.input_matrix[:, :2, :].cpu().numpy())
    return (
        np.mean(np.stack(contexts), axis=0),
        np.stack(predictions),
        np.mean(np.stack(matrices_a), axis=0),
        np.mean(np.stack(matrices_b), axis=0),
    )


def main() -> int:
    """Train three seeds and write a hash-linked schema-2 Koopman ensemble."""
    arguments = parse_arguments()
    output = arguments.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    fit_path = arguments.nominal_fit.expanduser().resolve()
    fit = json.loads(fit_path.read_text(encoding="utf-8"))
    splits = load_splits(
        arguments.run_root.expanduser().resolve(), arguments.split_manifest, fit
    )
    mean, scale = standardization(splits["train"])
    device = torch.device(arguments.device)
    config = model_config()
    values = {
        name: tensors(split, mean, scale, device) for name, split in splits.items()
    }
    models, members, reports = [], [], []
    for seed in SEEDS:
        model, report = train_member(
            seed,
            values["train"],
            values["validation"],
            arguments,
            config,
            device,
        )
        checkpoint = output / f"member_seed{seed}.pt"
        torch.save(
            {
                "model_family": "contextual_neural_koopman_v2",
                "config": config.as_dict(),
                "state_dict": model.cpu().state_dict(),
                "feature_mean": mean,
                "feature_scale": scale,
                "nominal_fit_sha256": sha256_path(fit_path),
                "seed": seed,
            },
            checkpoint,
        )
        models.append(model.cpu())
        members.append(
            {"seed": seed, "path": checkpoint.name, "sha256": sha256_path(checkpoint)}
        )
        reports.append(report)

    train_context, _, _, _ = ensemble_outputs(models, values["train"])
    calibration_context, calibration_member_predictions, _, _ = ensemble_outputs(
        models, values["calibration"]
    )
    ensemble_prediction = calibration_member_predictions.mean(axis=0)
    calibration_target = values["calibration"][3][:, 0].cpu().numpy()
    errors = np.linalg.norm(ensemble_prediction - calibration_target, axis=1)
    context_mean = train_context.mean(axis=0)
    covariance = np.cov(train_context, rowvar=False)
    covariance = (
        0.9 * covariance
        + 0.1
        * np.eye(covariance.shape[0])
        * np.trace(covariance)
        / covariance.shape[0]
    )
    covariance_inverse = np.linalg.pinv(covariance)
    context_residual = calibration_context - context_mean
    ood_scores = np.sum(
        (context_residual @ covariance_inverse) * context_residual, axis=1
    )
    calibration = {
        "conformal_radius": float(np.quantile(errors, 0.90)),
        "context_mean": context_mean.tolist(),
        "context_covariance_inverse": covariance_inverse.tolist(),
        "ood_threshold": float(np.quantile(ood_scores, 0.99)),
    }
    manifest = {
        "schema_version": 2,
        "model_family": "contextual_neural_koopman_v2",
        "architecture": config.as_dict(),
        "feature_names": FEATURE_NAMES,
        "state_names": ["v_y_mps", "yaw_rate_rps"],
        "input_names": ["steering_angle_rad"],
        "sample_time_s": 0.02,
        "nominal_fit_sha256": sha256_path(fit_path),
        "dataset_split_manifest_sha256": hashlib.sha256(
            arguments.split_manifest.expanduser().read_bytes()
        ).hexdigest(),
        "members": members,
        "calibration": calibration,
        "reports": reports,
    }
    manifest_path = output / "ensemble_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"KOOPMAN_V2_ENSEMBLE_TRAINED manifest={manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
