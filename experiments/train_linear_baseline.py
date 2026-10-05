"""Train a reproducible linear dynamics baseline for NeuroGrip-X."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# Modeled state is [v_x, yaw_rate].  Lateral body velocity is estimated and
# stored in the dataset but deliberately excluded from the model: at the
# configured low-speed envelope with Gazebo's kinematic Ackermann plugin it is
# numerically indistinguishable from zero (std ~4e-5 m/s), so including it
# would inject a degenerate, untrainable channel.  See docs/limitations.md.
FEATURE_COLUMNS = [
    "v_x_t_mps",
    "yaw_rate_t_rps",
    "cmd_linear_x_t_mps",
    "cmd_angular_z_t_rps",
]

TARGET_COLUMNS = [
    "v_x_t1_mps",
    "yaw_rate_t1_rps",
]


def git_commit() -> str:
    """Return the current Git commit, or 'unknown' outside a repository."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
        timeout=2,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def parse_arguments() -> argparse.Namespace:
    """Parse training inputs and reproducibility parameters."""
    parser = argparse.ArgumentParser(
        description="Train a Ridge one-step vehicle-dynamics baseline."
    )
    parser.add_argument(
        "dataset_path",
        nargs="?",
        type=Path,
        help="Processed Parquet dataset. Defaults to the newest transition dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/models"),
        help="Directory for the fitted model and metrics JSON.",
    )
    parser.add_argument(
        "--dataset-manifest",
        type=Path,
        help=(
            "Scenario-grouped dataset_manifest.json. When provided, trains "
            "only on its train split and evaluates validation and test splits."
        ),
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1e-3,
        help="Ridge regularization coefficient.",
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.2,
        help="Random held-out fraction for this development baseline.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the development split.",
    )
    return parser.parse_args()


def newest_dataset(processed_directory: Path) -> Path:
    """Return the newest processed transition dataset."""
    datasets = list(processed_directory.glob("*_transitions.parquet"))
    if not datasets:
        raise FileNotFoundError(
            f"No transition Parquet datasets found in {processed_directory}"
        )
    return max(datasets, key=lambda path: path.stat().st_mtime)


def metrics_by_target(y_true: np.ndarray, y_predicted: np.ndarray) -> dict[str, dict]:
    """Compute regression metrics independently for every physical target."""
    metrics = {}
    for index, target_name in enumerate(TARGET_COLUMNS):
        target_true = y_true[:, index]
        target_predicted = y_predicted[:, index]
        metrics[target_name] = {
            "mae": float(mean_absolute_error(target_true, target_predicted)),
            "rmse": float(np.sqrt(mean_squared_error(target_true, target_predicted))),
            "r2": float(r2_score(target_true, target_predicted)),
        }
    return metrics


def validate_columns(dataframe: pd.DataFrame, source: str) -> None:
    """Check that one dataset has the stable NeuroGrip-X model schema."""
    missing_columns = set(FEATURE_COLUMNS + TARGET_COLUMNS).difference(dataframe.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"{source} is missing expected columns: {missing}")


CATALOG_SPLITS = (
    "train",
    "validation",
    "calibration",
    "development_test",
    "sealed_test",
)
EVALUATION_SPLITS = ("validation", "development_test")


def load_scenario_splits(manifest_path: Path) -> tuple[dict[str, pd.DataFrame], dict]:
    """Load disjoint complete-scenario splits from a catalog, never row-split.

    Returns concatenated frames for every catalog split (empty frames for splits
    without data) plus provenance that includes the catalog SHA-256 and the
    split/feature/target schema versions.
    """
    manifest_path = manifest_path.expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Dataset manifest does not exist: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON dataset manifest: {manifest_path}") from error

    if manifest.get("split_unit") != "scenario_id":
        raise ValueError("Dataset manifest must use scenario_id as its split unit.")
    runs = manifest.get("runs")
    if not isinstance(runs, list):
        raise ValueError("Dataset manifest has no run list.")

    frames_by_split: dict[str, list[pd.DataFrame]] = {
        split: [] for split in CATALOG_SPLITS
    }
    scenario_ids_by_split: dict[str, set[str]] = {
        split: set() for split in CATALOG_SPLITS
    }
    for run in runs:
        if not isinstance(run, dict):
            raise ValueError("Dataset manifest run entries must be mappings.")
        split = run.get("split")
        scenario_id = run.get("scenario_id")
        parquet_path = Path(run.get("parquet_path", ""))
        if split not in frames_by_split:
            raise ValueError(f"Unknown catalog split: {split}")
        if not isinstance(scenario_id, str) or not parquet_path.is_file():
            raise ValueError("Catalog run has invalid scenario_id or Parquet path.")
        dataframe = pd.read_parquet(parquet_path)
        validate_columns(dataframe, str(parquet_path))
        frames_by_split[split].append(dataframe)
        scenario_ids_by_split[split].add(scenario_id)

    if not frames_by_split["train"]:
        raise ValueError("Dataset manifest has no train data.")
    all_scenario_ids = set().union(*scenario_ids_by_split.values())
    if sum(len(ids) for ids in scenario_ids_by_split.values()) != len(all_scenario_ids):
        raise ValueError("A scenario_id appears in more than one catalog split.")

    split_frames = {
        split: pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        for split, frames in frames_by_split.items()
    }
    provenance = {
        "dataset_manifest_path": str(manifest_path),
        "dataset_manifest_sha256": hashlib.sha256(
            manifest_path.read_bytes()
        ).hexdigest(),
        "split_schema_version": manifest.get("split_schema_version"),
        "split_unit": "scenario_id",
        "feature_columns": FEATURE_COLUMNS,
        "target_columns": TARGET_COLUMNS,
        "scenario_ids_by_split": {
            split: sorted(scenario_ids)
            for split, scenario_ids in scenario_ids_by_split.items()
        },
        "run_count_by_split": {
            split: len(frames) for split, frames in frames_by_split.items()
        },
    }
    return split_frames, provenance


def main() -> int:
    """Fit and save a Ridge baseline together with held-out metrics."""
    arguments = parse_arguments()
    if arguments.dataset_manifest and arguments.dataset_path:
        raise ValueError("Use either dataset_path or --dataset-manifest, not both.")

    scenario_mode = arguments.dataset_manifest is not None
    if scenario_mode:
        split_frames, split_provenance = load_scenario_splits(arguments.dataset_manifest)
        x_train = split_frames["train"][FEATURE_COLUMNS].to_numpy(dtype=np.float64)
        y_train = split_frames["train"][TARGET_COLUMNS].to_numpy(dtype=np.float64)
        evaluation_sets = {
            split: (
                split_frames[split][FEATURE_COLUMNS].to_numpy(dtype=np.float64),
                split_frames[split][TARGET_COLUMNS].to_numpy(dtype=np.float64),
            )
            for split in EVALUATION_SPLITS
            if not split_frames[split].empty
        }
        dataset_label = Path(arguments.dataset_manifest).stem
        dataset_reference = str(Path(arguments.dataset_manifest).expanduser().resolve())
    else:
        dataset_path = arguments.dataset_path or newest_dataset(Path("data/processed"))
        dataset_path = dataset_path.expanduser().resolve()
        if not dataset_path.is_file():
            raise FileNotFoundError(f"Dataset does not exist: {dataset_path}")
        if not 0.0 < arguments.validation_fraction < 1.0:
            raise ValueError("validation_fraction must be strictly between 0 and 1.")
        dataframe = pd.read_parquet(dataset_path)
        validate_columns(dataframe, str(dataset_path))
        features = dataframe[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
        targets = dataframe[TARGET_COLUMNS].to_numpy(dtype=np.float64)
        x_train, x_validation, y_train, y_validation = train_test_split(
            features,
            targets,
            test_size=arguments.validation_fraction,
            random_state=arguments.seed,
            shuffle=True,
        )
        evaluation_sets = {"validation": (x_validation, y_validation)}
        split_provenance = None
        dataset_label = dataset_path.stem
        dataset_reference = str(dataset_path)

    model = Pipeline(
        [
            ("standardize", StandardScaler()),
            ("ridge", Ridge(alpha=arguments.alpha)),
        ]
    )
    model.fit(x_train, y_train)
    evaluation_metrics = {}
    for split, (features, targets) in evaluation_sets.items():
        prediction = model.predict(features)
        evaluation_metrics[split] = {
            "rows": int(len(features)),
            "metrics": metrics_by_target(targets, prediction),
        }

    output_dir = arguments.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_stem = f"{dataset_label}_ridge_alpha_{arguments.alpha:g}"
    model_path = output_dir / f"{artifact_stem}.joblib"
    metrics_path = output_dir / f"{artifact_stem}.metrics.json"

    joblib.dump(model, model_path)
    metrics = {
        "schema_version": 2,
        "model_type": "StandardScaler + Ridge",
        "dataset_reference": dataset_reference,
        "model_path": str(model_path),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "feature_columns": FEATURE_COLUMNS,
        "target_columns": TARGET_COLUMNS,
        "feature_schema": FEATURE_COLUMNS,
        "target_schema": TARGET_COLUMNS,
        "checkpoint_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "alpha": arguments.alpha,
        "training_rows": int(len(x_train)),
        "evaluation": evaluation_metrics,
    }
    if scenario_mode:
        metrics["scenario_split"] = split_provenance
    else:
        metrics["development_split"] = {
            "validation_fraction": arguments.validation_fraction,
            "seed": arguments.seed,
            "validation_rows": int(len(evaluation_sets["validation"][0])),
            "warning": (
                "This is a shuffled development split, not the final held-out "
                "scenario benchmark."
            ),
        }
    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, indent=2)
        metrics_file.write("\n")

    print(f"Dataset: {dataset_label}")
    print(f"Training rows: {len(x_train)}")
    for split, result in evaluation_metrics.items():
        print(f"{split} rows: {result['rows']}")
        for target_name, target_metrics in result["metrics"].items():
            print(
                f"{split} {target_name}: MAE={target_metrics['mae']:.6f}, "
                f"RMSE={target_metrics['rmse']:.6f}, R²={target_metrics['r2']:.4f}"
            )
    print(f"Model: {model_path}")
    print(f"Metrics: {metrics_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError) as error:
        print(f"Training failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error
