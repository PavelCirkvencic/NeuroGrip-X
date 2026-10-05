#!/usr/bin/env python3
"""Train and compare the contextual Koopman model against its ablation.

Controlled protocol:

* two variants share every setting — ``B1-K`` (one learned global context) and
  ``N1`` (causal context encoder);
* each variant is trained with at least three seeds, reseeding all RNGs before
  every run;
* the best epoch is selected **only** on the validation split;
* the development test split is used for reporting, the sealed test split is
  never touched here;
* per-scenario macro averages and bootstrap confidence intervals are reported,
  along with operator spectral radius and NaN/Inf health checks.

The physics head is a learned auxiliary predictor of the next-step state
increment, not a calibrated force model; the code and docs say so explicitly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from neurogrip.data import build_split_windows, load_catalog, run_frames_by_split
from neurogrip.model import NeuroGripModel
from neurogrip.config import NeuroGripConfig


def parse_arguments() -> argparse.Namespace:
    """Parse catalog, training and model-size options."""
    parser = argparse.ArgumentParser(
        description="Train the NeuroGrip-X contextual Koopman model."
    )
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("runs/models/scenario_neurogrip_latest")
    )
    parser.add_argument("--history-steps", type=int, default=25)
    parser.add_argument("--rollout-steps", type=int, default=20)
    parser.add_argument("--context-dim", type=int, default=6)
    parser.add_argument("--latent-extra", type=int, default=9)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--aux-weight", type=float, default=0.1)
    parser.add_argument("--reg-weight", type=float, default=1e-4)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    return parser.parse_args()


def git_commit() -> str:
    """Return the current Git commit, or 'unknown'."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
        timeout=2,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def seed_everything(seed: int) -> None:
    """Reseed every RNG used by training (called before each run)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def normalize(array: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Standardise an array with precomputed statistics."""
    return (array - mean) / scale


def compute_statistics(train_batch) -> dict:
    """Compute train-only standardization statistics."""
    state_dim = train_batch.current_state.shape[1]
    state_values = np.concatenate(
        [train_batch.current_state, train_batch.target_states.reshape(-1, state_dim)],
        axis=0,
    )
    command_values = np.concatenate(
        [
            train_batch.history[:, :, state_dim:].reshape(-1, 2),
            train_batch.future_commands.reshape(-1, 2),
        ],
        axis=0,
    )
    return {
        "state_mean": state_values.mean(axis=0),
        "state_scale": state_values.std(axis=0) + 1e-6,
        "cmd_mean": command_values.mean(axis=0),
        "cmd_scale": command_values.std(axis=0) + 1e-6,
    }


def predict_windows(model, batch, statistics, device):
    """Return physical rollout predictions and context embeddings for a batch."""
    state_mean, state_scale = statistics["state_mean"], statistics["state_scale"]
    cmd_mean, cmd_scale = statistics["cmd_mean"], statistics["cmd_scale"]
    history_mean = np.concatenate([state_mean, cmd_mean])
    history_scale = np.concatenate([state_scale, cmd_scale])
    with torch.no_grad():
        history = torch.tensor(
            (batch.history - history_mean) / history_scale,
            dtype=torch.float32,
            device=device,
        )
        current = torch.tensor(
            (batch.current_state - state_mean) / state_scale,
            dtype=torch.float32,
            device=device,
        )
        commands = torch.tensor(
            (batch.future_commands - cmd_mean) / cmd_scale,
            dtype=torch.float32,
            device=device,
        )
        output = model(history, current, commands)
    predicted = output.rollout.detach().cpu().numpy() * state_scale + state_mean
    return (
        predicted,
        output.context.detach().cpu().numpy(),
        output.A.detach().cpu().numpy(),
    )


def per_scenario_rmse(batch, predicted: np.ndarray, horizon: int) -> dict[str, float]:
    """Return final-step yaw-rate RMSE per scenario (macro weighting unit)."""
    target = batch.target_states[:, horizon - 1, :]
    prediction = predicted[:, horizon - 1, :]
    error = prediction - target
    scenarios = np.array(batch.scenario_id)
    result = {}
    for scenario in sorted(set(scenarios.tolist())):
        mask = scenarios == scenario
        if not np.any(mask):
            continue
        result[scenario] = float(np.sqrt(np.mean(error[mask, 1] ** 2)))
    return result


def macro_average(values: dict[str, float]) -> float:
    """Return the equal-weight mean over scenarios."""
    return float(np.mean(list(values.values()))) if values else float("nan")


def bootstrap_ci(values: dict[str, float], samples: int, seed: int) -> tuple[float, float]:
    """Return a 95% bootstrap CI for the macro average over scenarios."""
    scenario_values = np.array(list(values.values()), dtype=float)
    if scenario_values.size < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = [
        float(np.mean(rng.choice(scenario_values, size=scenario_values.size, replace=True)))
        for _ in range(samples)
    ]
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def spectral_health(matrices: np.ndarray) -> dict:
    """Return max/mean spectral radius of the context-conditioned operators."""
    radii = np.abs(np.linalg.eigvals(matrices)).max(axis=1)
    return {
        "max_spectral_radius": float(np.max(radii)),
        "mean_spectral_radius": float(np.mean(radii)),
    }


def train_variant(
    config, context_mode, train_batch, validation_batch, statistics, arguments, device
):
    """Fit one variant for one seed with validation-selected best checkpoint."""
    state_mean, state_scale = statistics["state_mean"], statistics["state_scale"]
    cmd_mean, cmd_scale = statistics["cmd_mean"], statistics["cmd_scale"]
    history_mean = np.concatenate([state_mean, cmd_mean])
    history_scale = np.concatenate([state_scale, cmd_scale])
    history = normalize(train_batch.history, history_mean, history_scale).astype(np.float32)
    current = normalize(train_batch.current_state, state_mean, state_scale).astype(np.float32)
    commands = normalize(train_batch.future_commands, cmd_mean, cmd_scale).astype(np.float32)
    targets = normalize(train_batch.target_states, state_mean, state_scale).astype(np.float32)

    history_t = torch.tensor(history, device=device)
    current_t = torch.tensor(current, device=device)
    commands_t = torch.tensor(commands, device=device)
    targets_t = torch.tensor(targets, device=device)
    validation_targets = validation_batch.target_states[:, -1, 1]

    model = NeuroGripModel(config, context_mode=context_mode).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=arguments.learning_rate, weight_decay=arguments.weight_decay
    )
    loss_fn = torch.nn.MSELoss()
    generator = torch.Generator(device="cpu").manual_seed(arguments.seed_for_this_run)
    dataset_size = history_t.shape[0]

    best_state = None
    best_epoch = 0
    best_validation = float("inf")
    history_log = []
    for epoch in range(arguments.epochs):
        model.train()
        permutation = torch.randperm(dataset_size, generator=generator)
        for start in range(0, dataset_size, arguments.batch_size):
            indices = permutation[start : start + arguments.batch_size]
            output = model(history_t[indices], current_t[indices], commands_t[indices])
            rollout_loss = loss_fn(output.rollout, targets_t[indices])
            delta_target = targets_t[indices][:, 0] - current_t[indices]
            auxiliary_loss = loss_fn(output.aux, delta_target)
            regularization = (
                (output.A - model.operator.A0).pow(2).mean()
                + (output.B - model.operator.B0).pow(2).mean()
            )
            loss = (
                rollout_loss
                + arguments.aux_weight * auxiliary_loss
                + arguments.reg_weight * regularization
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        predicted, _, matrices = predict_windows(model, validation_batch, statistics, device)
        if not np.all(np.isfinite(predicted)):
            raise RuntimeError("Non-finite validation prediction; aborting run.")
        if not np.all(np.isfinite(matrices)):
            raise RuntimeError("Non-finite operator matrix; aborting run.")
        validation_rmse = float(
            np.sqrt(np.mean((predicted[:, -1, 1] - validation_targets) ** 2))
        )
        history_log.append({"epoch": epoch + 1, "validation_yaw_rmse": validation_rmse})
        if validation_rmse < best_validation:
            best_validation = validation_rmse
            best_epoch = epoch + 1
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        elif epoch + 1 - best_epoch >= arguments.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"best_epoch": best_epoch, "best_validation_yaw_rmse": best_validation, "history": history_log}


def evaluate_split(model, batch, statistics, device, horizon, arguments):
    """Evaluate one model on one split with per-scenario macro statistics."""
    predicted, contexts, matrices = predict_windows(model, batch, statistics, device)
    if not np.all(np.isfinite(predicted)):
        raise RuntimeError("Non-finite prediction during evaluation.")
    per_scenario = per_scenario_rmse(batch, predicted, horizon)
    return {
        "windows": len(batch),
        "scenarios": len(per_scenario),
        "macro_yaw_rmse": macro_average(per_scenario),
        "macro_ci95": bootstrap_ci(per_scenario, arguments.bootstrap_samples, arguments.seed_for_this_run),
        "per_scenario": per_scenario,
        "operator_health": spectral_health(matrices),
    }


def save_checkpoint(path, model, config, statistics, context_mode):
    """Persist a model variant and its normalization statistics."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": config.as_dict(),
            "context_mode": context_mode,
            "statistics": {key: value.tolist() for key, value in statistics.items()},
        },
        path,
    )


def main() -> int:
    """Train all variants and seeds, then aggregate and save the manifest."""
    arguments = parse_arguments()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = NeuroGripConfig(
        history_steps=arguments.history_steps,
        rollout_steps=arguments.rollout_steps,
        context_dim=arguments.context_dim,
        latent_extra=arguments.latent_extra,
        rank=arguments.rank,
        hidden_dim=arguments.hidden_dim,
    )
    catalog_path = arguments.dataset_manifest.expanduser().resolve()
    catalog = load_catalog(catalog_path)
    grouped = run_frames_by_split(catalog)
    split_batches = build_split_windows(grouped, config)
    if split_batches.get("train") is None:
        raise ValueError("No training windows were built.")
    if split_batches.get("validation") is None:
        raise ValueError("No validation windows were built; cannot select epochs.")
    statistics = compute_statistics(split_batches["train"])

    evaluation_splits = [
        split
        for split in ("validation", "development_test")
        if split_batches.get(split) is not None
    ]
    variants = {"B1-K": "learned_global", "N1": "encoder"}
    output_dir = arguments.output_dir.expanduser().resolve()
    manifest_variants = {}
    checkpoints = {}

    for name, context_mode in variants.items():
        seed_results = []
        for seed in arguments.seeds:
            arguments.seed_for_this_run = seed
            seed_everything(seed)
            model, training = train_variant(
                config,
                context_mode,
                split_batches["train"],
                split_batches["validation"],
                statistics,
                arguments,
                device,
            )
            evaluations = {
                split: evaluate_split(
                    model,
                    split_batches[split],
                    statistics,
                    device,
                    arguments.rollout_steps,
                    arguments,
                )
                for split in evaluation_splits
            }
            checkpoint = output_dir / f"{name.lower().replace('-', '_')}_seed{seed}.pt"
            save_checkpoint(checkpoint, model, config, statistics, context_mode)
            checkpoints.setdefault(name, []).append((training["best_validation_yaw_rmse"], checkpoint))
            seed_results.append(
                {"seed": seed, "training": training, "evaluation": evaluations}
            )
            print(
                f"{name} seed {seed}: best epoch {training['best_epoch']}, "
                f"validation yaw RMSE {training['best_validation_yaw_rmse']:.5f}"
            )

        # Aggregated metrics across seeds, per scenario macro average.
        aggregate = {}
        for split in evaluation_splits:
            macros = [result["evaluation"][split]["macro_yaw_rmse"] for result in seed_results]
            aggregate[split] = {
                "macro_yaw_rmse_mean": float(np.mean(macros)),
                "macro_yaw_rmse_std": float(np.std(macros)),
            }
        manifest_variants[name] = {
            "context_mode": context_mode,
            "seed_results": seed_results,
            "aggregate": aggregate,
        }
        # Canonical checkpoint = seed with best validation RMSE.
        best = min(checkpoints[name], key=lambda item: item[0])
        canonical = output_dir / f"{name.lower().replace('-', '_')}_model.pt"
        canonical.write_bytes(best[1].read_bytes())

    manifest = {
        "schema_version": 2,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_type": "contextual_koopman",
        "physics_head": "learned_auxiliary_next_state_increment_not_force",
        "catalog_path": str(catalog_path),
        "catalog_sha256": hashlib.sha256(catalog_path.read_bytes()).hexdigest(),
        "split_schema_version": catalog.get("split_schema_version"),
        "git_commit": git_commit(),
        "device": str(device),
        "seeds": arguments.seeds,
        "config": config.as_dict(),
        "hyperparameters": {
            "epochs": arguments.epochs,
            "patience": arguments.patience,
            "batch_size": arguments.batch_size,
            "learning_rate": arguments.learning_rate,
            "weight_decay": arguments.weight_decay,
            "aux_weight": arguments.aux_weight,
            "reg_weight": arguments.reg_weight,
        },
        "normalization": {key: value.tolist() for key, value in statistics.items()},
        "window_counts": {
            split: (len(batch) if batch is not None else 0)
            for split, batch in split_batches.items()
        },
        "evaluation_splits": evaluation_splits,
        "sealed_test_used": False,
        "variants": manifest_variants,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "training_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=float) + "\n", encoding="utf-8")
    print(f"\nManifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError) as error:
        print(f"NeuroGrip training failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error
