#!/usr/bin/env python3
"""Train the high-speed grip-aware neural Koopman ensemble."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "learning"))
sys.path.insert(0, str(REPO_ROOT / "src/neurogrip_ai"))
sys.path.insert(0, str(REPO_ROOT / "experiments_v2"))

from neurogrip.config import NeuroGripConfig  # noqa: E402
from neurogrip.koopman_grip import GripAwareKoopmanModel  # noqa: E402
from neurogrip_ai.physics_residual import FEATURE_NAMES  # noqa: E402
from train_physics_residual import (  # noqa: E402
    FIELD_MAP,
    HORIZONS,
    SEEDS,
    nominal_matrices,
    sha256_path,
)


@dataclass
class GripWindows:
    """Causal dynamics windows with simulation-only training labels."""

    history: np.ndarray
    state: np.ndarray
    commands: np.ndarray
    targets: np.ndarray
    nominal_a: np.ndarray
    nominal_b: np.ndarray
    grip: np.ndarray
    scenario_ids: list[str]


def parse_arguments() -> argparse.Namespace:
    """Parse immutable high-speed data and bounded training settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root", type=Path, action="append", required=True,
        help="repeat for disjoint excitation and closed-loop run roots",
    )
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--nominal-fit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=90)
    parser.add_argument("--patience", type=int, default=14)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def model_config() -> NeuroGripConfig:
    """Return the schema-4 deployment architecture."""
    return NeuroGripConfig(
        state_dim=2,
        input_dim=1,
        history_steps=25,
        context_dim=10,
        latent_extra=8,
        rank=5,
        hidden_dim=64,
        rollout_steps=20,
        history_feature_dim=len(FEATURE_NAMES),
    )


def read_episode(path: Path) -> dict[str, np.ndarray]:
    """Read finite 50 Hz features and front/rear grip labels."""
    names = ["time_s", *FIELD_MAP.values(), "grip_front", "grip_rear"]
    values = {name: [] for name in names}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                for name in names:
                    values[name].append(float(row[name]))
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"invalid high-speed telemetry in {path}") from error
    result = {
        name: np.asarray(entries, dtype=np.float32)
        for name, entries in values.items()
    }
    if not all(np.all(np.isfinite(value)) for value in result.values()):
        raise ValueError(f"non-finite high-speed telemetry in {path}")
    dt = np.diff(result["time_s"])
    if len(dt) < 100 or np.any(dt <= 0.0):
        raise ValueError(f"insufficient continuous telemetry in {path}")
    rate = 1.0 / float(np.median(dt))
    if not 49.0 <= rate <= 51.0:
        raise ValueError(f"{path} is not a 50 Hz episode")
    return result


def make_windows(episode: dict[str, np.ndarray], scenario_id: str, fit: dict) -> GripWindows:
    """Create causal windows after the complete history sees the current grip."""
    features = np.column_stack(
        [episode[FIELD_MAP[name]] for name in FEATURE_NAMES]
    )
    grip = np.column_stack((episode["grip_front"], episode["grip_rear"]))
    states = features[:, 1:3]
    steering = features[:, 3]
    rows: list[tuple] = []
    for index in range(24, len(features) - max(HORIZONS)):
        history_grip = grip[index - 24:index + 1]
        if np.max(np.ptp(history_grip, axis=0)) > 1e-6:
            continue
        matrix_a, matrix_b = nominal_matrices(float(features[index, 0]), fit)
        rows.append(
            (
                features[index - 24:index + 1],
                states[index],
                steering[index:index + max(HORIZONS)],
                np.stack([states[index + horizon] for horizon in HORIZONS]),
                matrix_a,
                matrix_b,
                grip[index],
            )
        )
    if not rows:
        raise ValueError(f"{scenario_id} has no stable-grip causal windows")
    columns = list(zip(*rows))
    return GripWindows(
        *(np.asarray(column, dtype=np.float32) for column in columns),
        [scenario_id] * len(rows),
    )


def concatenate(items: list[GripWindows]) -> GripWindows:
    """Concatenate whole scenario windows without losing split identity."""
    names = (
        "history", "state", "commands", "targets", "nominal_a", "nominal_b", "grip"
    )
    return GripWindows(
        *(np.concatenate([getattr(item, name) for item in items]) for name in names),
        sum((item.scenario_ids for item in items), []),
    )


def load_splits(
    run_root: Path | list[Path], manifest_path: Path, fit: dict
) -> dict[str, GripWindows]:
    """Load disjoint train/validation/calibration scenarios by complete ID."""
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    splits = document.get("splits", {})
    if not {"train", "validation", "calibration"} <= splits.keys():
        raise ValueError("split manifest lacks train/validation/calibration")
    members = [item for split in splits.values() for item in split]
    if len(members) != len(set(members)):
        raise ValueError("scenario appears in more than one split")
    candidates: dict[str, list[tuple[Path, dict]]] = {}
    roots = [run_root] if isinstance(run_root, Path) else run_root
    for root in roots:
        for path in root.glob("*/episode.csv"):
            metadata_path = path.with_suffix(".metadata.json")
            if metadata_path.is_file():
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                scenario_id = metadata.get("scenario_id")
                candidates.setdefault(scenario_id, []).append((path, metadata))
    result = {}
    for split, identifiers in splits.items():
        windows = []
        for identifier in identifiers:
            if "::" in identifier:
                scenario_id, controller = identifier.rsplit("::", 1)
                matches = [
                    item
                    for item in candidates.get(scenario_id, [])
                    if item[1].get("controller") == controller
                ]
            else:
                scenario_id = identifier
                matches = candidates.get(identifier, [])
            if not matches:
                raise ValueError(f"split {split}: missing {identifier}")
            if len(matches) != 1:
                raise ValueError(
                    f"split {split}: ambiguous {identifier}; specify scenario::controller"
                )
            path, metadata = matches[0]
            if metadata.get("runner", {}).get("termination_reason") not in {
                "excitation_complete",
                "lap_complete",
            }:
                raise ValueError(f"split {split}: incomplete {identifier}")
            windows.append(make_windows(read_episode(path), scenario_id, fit))
        result[split] = concatenate(windows)
    return result


def standardization(train: GripWindows) -> tuple[np.ndarray, np.ndarray]:
    """Compute feature statistics from training scenarios only."""
    flattened = train.history.reshape(-1, train.history.shape[-1])
    return flattened.mean(0).astype(np.float32), (flattened.std(0) + 1e-6).astype(np.float32)


def tensors(batch: GripWindows, mean: np.ndarray, scale: np.ndarray, device):
    """Convert one complete split to tensors."""
    values = (
        (batch.history - mean) / scale,
        batch.state,
        batch.commands,
        batch.targets,
        batch.nominal_a,
        batch.nominal_b,
        batch.grip,
    )
    return tuple(torch.as_tensor(value, dtype=torch.float32, device=device) for value in values)


def loss_components(model: GripAwareKoopmanModel, values, indices=None):
    """Return joint dynamics, lifted-consistency, stability and grip loss."""
    if indices is not None:
        values = tuple(value[indices] for value in values)
    history, state, commands, targets, nominal_a, nominal_b, grip = values
    output = model(history, state, commands, nominal_a, nominal_b)
    horizon_indices = torch.as_tensor([h - 1 for h in HORIZONS], device=history.device)
    predicted = output.state_rollout.index_select(1, horizon_indices)
    physical = torch.mean((predicted - targets) ** 2)
    batch, horizon_count, _ = targets.shape
    context = output.context[:, None, :].expand(-1, horizon_count, -1)
    target_lift = model.lift_state(
        targets.reshape(-1, 2), context.reshape(-1, model.config.context_dim)
    ).reshape(batch, horizon_count, model.config.latent_dim)
    consistency = torch.mean(
        (output.latent_rollout.index_select(1, horizon_indices) - target_lift) ** 2
    )
    # The induced infinity norm is a conservative spectral-radius bound and,
    # unlike eigenvalue gradients, remains defined when the initially
    # identity-like Koopman matrix has repeated eigenvalues.
    spectral_bound = output.koopman.abs().sum(dim=2).amax(dim=1)
    stability = torch.relu(spectral_bound - 1.05).pow(2).mean()
    residual = (
        (output.koopman[:, :2, :2] - nominal_a).pow(2).mean()
        + (output.input_matrix[:, :2, :] - nominal_b).pow(2).mean()
    )
    grip_loss = torch.mean((output.grip_estimate - grip) ** 2)
    total = physical + 0.05 * consistency + 0.20 * stability + 1e-3 * residual + 0.10 * grip_loss
    return total, physical, grip_loss, output


def evaluate(model: GripAwareKoopmanModel, values) -> tuple[float, float, float]:
    """Evaluate joint validation metrics without parameter updates."""
    model.eval()
    with torch.no_grad():
        total, physical, grip, _ = loss_components(model, values)
    return float(total), float(physical), float(torch.sqrt(grip))


def train_member(seed, train_values, validation_values, arguments, config, device):
    """Train one independently seeded member with validation-only stopping."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = GripAwareKoopmanModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=arguments.learning_rate, weight_decay=1e-5)
    count = train_values[0].shape[0]
    generator = torch.Generator(device=device).manual_seed(seed)
    best, best_state, best_epoch, stale = float("inf"), None, 0, 0
    best_metrics = None
    for epoch in range(1, arguments.epochs + 1):
        model.train()
        permutation = torch.randperm(count, generator=generator, device=device)
        for start in range(0, count, arguments.batch_size):
            indices = permutation[start:start + arguments.batch_size]
            loss, _, _, _ = loss_components(model, train_values, indices)
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"non-finite training loss for seed={seed} epoch={epoch}"
                )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        metrics = evaluate(model, validation_values)
        if metrics[0] < best:
            best, best_epoch, stale = metrics[0], epoch, 0
            best_metrics = metrics
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
        else:
            stale += 1
            if stale >= arguments.patience:
                break
    if best_state is None or best_metrics is None:
        raise RuntimeError("training produced no finite checkpoint")
    model.load_state_dict(best_state)
    return model, {
        "seed": seed,
        "best_epoch": best_epoch,
        "validation_loss": best_metrics[0],
        "validation_physical_loss": best_metrics[1],
        "validation_grip_rmse": best_metrics[2],
    }


def outputs(models, values):
    """Return ensemble contexts, predictions and grip estimates."""
    history, state, commands, _, nominal_a, nominal_b, _ = values
    contexts, predictions, grips = [], [], []
    for model in models:
        model.eval()
        with torch.no_grad():
            output = model(history, state, commands[:, :1], nominal_a, nominal_b)
        contexts.append(output.context.cpu().numpy())
        predictions.append(output.state_rollout[:, 0].cpu().numpy())
        grips.append(output.grip_estimate.cpu().numpy())
    return np.mean(contexts, axis=0), np.stack(predictions), np.stack(grips)


def main() -> int:
    """Train and calibrate three high-speed schema-3 model artifacts."""
    arguments = parse_arguments()
    output_dir = arguments.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    fit_path = arguments.nominal_fit.expanduser().resolve()
    fit = json.loads(fit_path.read_text(encoding="utf-8"))
    splits = load_splits(
        [path.expanduser().resolve() for path in arguments.run_root],
        arguments.split_manifest.expanduser().resolve(),
        fit,
    )
    mean, scale = standardization(splits["train"])
    device = torch.device(arguments.device)
    values = {
        split: tensors(batch, mean, scale, device)
        for split, batch in splits.items()
    }
    config = model_config()
    models, members, reports = [], [], []
    for seed in SEEDS:
        model, report = train_member(
            seed, values["train"], values["validation"], arguments, config, device
        )
        checkpoint = output_dir / f"member_seed{seed}.pt"
        torch.save(
            {
                "model_family": "contextual_neural_koopman_grip_v3",
                "config": config.as_dict(),
                "state_dict": {
                    name: value.detach().cpu()
                    for name, value in model.state_dict().items()
                },
                "feature_mean": mean,
                "feature_scale": scale,
                "nominal_fit_sha256": sha256_path(fit_path),
                "seed": seed,
            },
            checkpoint,
        )
        models.append(model)
        members.append(
            {"seed": seed, "path": checkpoint.name, "sha256": sha256_path(checkpoint)}
        )
        reports.append(report)

    train_context, _, _ = outputs(models, values["train"])
    calibration_context, member_prediction, member_grip = outputs(
        models, values["calibration"]
    )
    prediction = member_prediction.mean(0)
    target = values["calibration"][3][:, 0].cpu().numpy()
    dynamics_error = np.linalg.norm(prediction - target, axis=1)
    grip_target = values["calibration"][6].cpu().numpy()
    grip_prediction = member_grip.mean(0)
    grip_error = np.abs(grip_prediction - grip_target)
    context_mean = train_context.mean(0)
    covariance = np.cov(train_context, rowvar=False)
    covariance = 0.9 * covariance + 0.1 * np.eye(len(context_mean)) * np.trace(covariance) / len(context_mean)
    covariance_inverse = np.linalg.pinv(covariance)
    residual = calibration_context - context_mean
    ood = np.sum((residual @ covariance_inverse) * residual, axis=1)
    calibration = {
        "conformal_radius": float(np.quantile(dynamics_error, 0.90)),
        "context_mean": context_mean.tolist(),
        "context_covariance_inverse": covariance_inverse.tolist(),
        "ood_threshold": float(np.quantile(ood, 0.99)),
        "grip_absolute_error_q90": np.quantile(grip_error, 0.90, axis=0).tolist(),
        "grip_mae": np.mean(grip_error, axis=0).tolist(),
    }
    manifest = {
        "schema_version": 3,
        "model_family": "contextual_neural_koopman_grip_v3",
        "architecture": config.as_dict(),
        "feature_names": FEATURE_NAMES,
        "state_names": ["v_y_mps", "yaw_rate_rps"],
        "input_names": ["steering_angle_rad"],
        "grip_output_names": ["front_grip_scale", "rear_grip_scale"],
        "sample_time_s": 0.02,
        "nominal_fit_sha256": sha256_path(fit_path),
        "dataset_split_manifest_sha256": hashlib.sha256(
            arguments.split_manifest.expanduser().read_bytes()
        ).hexdigest(),
        "members": members,
        "calibration": calibration,
        "reports": reports,
    }
    manifest_path = output_dir / "ensemble_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"KOOPMAN_GRIP_V3_TRAINED manifest={manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
