"""Convert raw NeuroGrip-X recordings into one-step learning transitions."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from neurogrip_control.body_velocity import (
    CausalBodyVelocityEstimator,
    yaw_from_quaternion,
)


REQUIRED_COLUMNS = {
    "time_s",
    "odom_imu_dt_s",
    "cmd_age_s",
    "x_m",
    "y_m",
    "qx",
    "qy",
    "qz",
    "qw",
    "v_x_mps",
    "v_y_mps",
    "yaw_rate_rps",
    "imu_w_z_rps",
    "cmd_linear_x_mps",
    "cmd_angular_z_rps",
}


def parse_arguments() -> argparse.Namespace:
    """Parse paths and quality thresholds."""
    parser = argparse.ArgumentParser(
        description="Create state-command-next-state transitions from a raw CSV."
    )
    parser.add_argument(
        "csv_path",
        nargs="?",
        type=Path,
        help="Raw CSV to process. Defaults to the newest data/raw/*.csv file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory for generated Parquet and metadata files.",
    )
    parser.add_argument(
        "--max-command-age-s",
        type=float,
        default=0.25,
        help="Discard samples whose command is older than this threshold.",
    )
    parser.add_argument(
        "--max-alignment-s",
        type=float,
        default=0.05,
        help="Discard samples whose odometry/IMU time difference exceeds this value.",
    )
    parser.add_argument(
        "--expected-dt-s",
        type=float,
        default=0.02,
        help="Expected transition duration for a 50 Hz dataset.",
    )
    parser.add_argument(
        "--dt-tolerance-s",
        type=float,
        default=0.005,
        help="Allowed absolute error around expected transition duration.",
    )
    parser.add_argument(
        "--require-scenario-manifest",
        action="store_true",
        help=(
            "Reject recordings without a valid scenario manifest in the "
            "sibling .metadata.json file."
        ),
    )
    return parser.parse_args()


def newest_recording(raw_directory: Path) -> Path:
    """Return the newest raw CSV recording."""
    recordings = list(raw_directory.glob("*.csv"))
    if not recordings:
        raise FileNotFoundError(f"No CSV recordings found in {raw_directory}")
    return max(recordings, key=lambda path: path.stat().st_mtime)


def load_raw_recording(csv_path: Path) -> pd.DataFrame:
    """Load and validate the minimum raw logger schema."""
    dataframe = pd.read_csv(csv_path)
    missing_columns = REQUIRED_COLUMNS.difference(dataframe.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"CSV is missing required columns: {missing}")
    if dataframe.empty:
        raise ValueError("CSV contains no data rows.")
    if not dataframe["time_s"].is_monotonic_increasing:
        raise ValueError("CSV time_s must be monotonically increasing.")

    if "excitation_phase" not in dataframe.columns:
        dataframe["excitation_phase"] = "unlabeled"
    return dataframe


def load_run_provenance(csv_path: Path, require_scenario_manifest: bool) -> dict:
    """Load logger metadata and optionally enforce leakage-safe provenance."""
    metadata_path = csv_path.with_suffix(".metadata.json")
    if not metadata_path.is_file():
        if require_scenario_manifest:
            raise ValueError(f"Run metadata does not exist: {metadata_path}")
        return {"metadata_path": None, "scenario_manifest": None}

    try:
        with metadata_path.open(encoding="utf-8") as metadata_file:
            metadata = json.load(metadata_file)
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON run metadata: {metadata_path}") from error

    scenario_manifest = metadata.get("scenario_manifest")
    if require_scenario_manifest:
        if not isinstance(scenario_manifest, dict):
            raise ValueError(
                "Run has no scenario manifest. Re-record with "
                "state_logger scenario_manifest_path:=<manifest>."
            )
        required_keys = {"path", "sha256", "scenario_id", "scenario"}
        missing = required_keys.difference(scenario_manifest)
        if missing:
            raise ValueError(
                "Scenario manifest provenance is incomplete: "
                + ", ".join(sorted(missing))
            )

    return {
        "metadata_path": str(metadata_path),
        "git_commit": metadata.get("git_commit", "unknown"),
        "excitation_profile": metadata.get("excitation_profile", "unknown"),
        "termination_reason": metadata.get("termination_reason", "unknown"),
        "grip_transition_status": metadata.get("grip_transition_status"),
        "scenario_manifest": scenario_manifest,
    }


def add_body_velocity_estimates(
    dataframe: pd.DataFrame,
    max_gap_s: float = 0.10,
) -> pd.DataFrame:
    """Add causal body-frame velocity columns derived from world poses."""
    estimator = CausalBodyVelocityEstimator(max_gap_s=max_gap_s)
    body_v_x = []
    body_v_y = []
    for row in dataframe.itertuples(index=False):
        yaw_rad = yaw_from_quaternion(row.qx, row.qy, row.qz, row.qw)
        v_x_mps, v_y_mps = estimator.update(
            row.time_s, row.x_m, row.y_m, yaw_rad
        )
        body_v_x.append(v_x_mps)
        body_v_y.append(v_y_mps)
    dataframe = dataframe.copy()
    dataframe["v_x_est_mps"] = body_v_x
    dataframe["v_y_est_mps"] = body_v_y
    return dataframe


def create_transitions(
    dataframe: pd.DataFrame,
    max_command_age_s: float,
    max_alignment_s: float,
    expected_dt_s: float,
    dt_tolerance_s: float,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Create quality-filtered one-step transitions from sequential samples."""
    next_time_s = dataframe["time_s"].shift(-1)
    transition_dt_s = next_time_s - dataframe["time_s"]
    next_v_x_mps = dataframe["v_x_est_mps"].shift(-1)
    next_v_y_mps = dataframe["v_y_est_mps"].shift(-1)
    next_yaw_rate_rps = dataframe["imu_w_z_rps"].shift(-1)

    estimate_is_valid = (
        dataframe["v_x_est_mps"].notna() & dataframe["v_y_est_mps"].notna()
    )
    sample_is_valid = (
        dataframe["cmd_age_s"].between(0.0, max_command_age_s)
        & dataframe["odom_imu_dt_s"].abs().le(max_alignment_s)
        & estimate_is_valid
    )
    transition_is_valid = (
        sample_is_valid
        & sample_is_valid.shift(-1, fill_value=False)
        & transition_dt_s.sub(expected_dt_s).abs().le(dt_tolerance_s)
    )

    transitions = pd.DataFrame(
        {
            "time_s": dataframe.loc[transition_is_valid, "time_s"],
            "dt_s": transition_dt_s.loc[transition_is_valid],
            "cmd_age_s": dataframe.loc[transition_is_valid, "cmd_age_s"],
            "odom_imu_dt_s": dataframe.loc[
                transition_is_valid, "odom_imu_dt_s"
            ],
            "phase": dataframe.loc[transition_is_valid, "excitation_phase"],
            "v_x_t_mps": dataframe.loc[transition_is_valid, "v_x_est_mps"],
            "v_y_t_mps": dataframe.loc[transition_is_valid, "v_y_est_mps"],
            "yaw_rate_t_rps": dataframe.loc[transition_is_valid, "imu_w_z_rps"],
            "yaw_rate_odom_t_rps": dataframe.loc[transition_is_valid, "yaw_rate_rps"],
            "cmd_linear_x_t_mps": dataframe.loc[
                transition_is_valid, "cmd_linear_x_mps"
            ],
            "cmd_angular_z_t_rps": dataframe.loc[
                transition_is_valid, "cmd_angular_z_rps"
            ],
            "v_x_t1_mps": next_v_x_mps.loc[transition_is_valid],
            "v_y_t1_mps": next_v_y_mps.loc[transition_is_valid],
            "yaw_rate_t1_rps": next_yaw_rate_rps.loc[transition_is_valid],
        }
    ).dropna()

    counters = {
        "raw_rows": int(len(dataframe)),
        "valid_samples": int(sample_is_valid.sum()),
        "valid_transitions_before_dropna": int(transition_is_valid.sum()),
        "transitions": int(len(transitions)),
    }
    return transitions.reset_index(drop=True), counters


def main() -> int:
    """Prepare a Parquet transition dataset and its provenance metadata."""
    arguments = parse_arguments()
    csv_path = arguments.csv_path or newest_recording(Path("data/raw"))
    csv_path = csv_path.expanduser().resolve()
    if not csv_path.is_file():
        raise FileNotFoundError(f"Raw CSV does not exist: {csv_path}")

    dataframe = load_raw_recording(csv_path)
    dataframe = add_body_velocity_estimates(
        dataframe,
        max_gap_s=arguments.expected_dt_s + arguments.dt_tolerance_s,
    )
    provenance = load_run_provenance(
        csv_path, arguments.require_scenario_manifest
    )
    transitions, counters = create_transitions(
        dataframe,
        arguments.max_command_age_s,
        arguments.max_alignment_s,
        arguments.expected_dt_s,
        arguments.dt_tolerance_s,
    )
    if transitions.empty:
        raise ValueError("No valid transitions remained after quality filtering.")

    output_dir = arguments.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_stem = f"{csv_path.stem}_transitions"
    parquet_path = output_dir / f"{dataset_stem}.parquet"
    metadata_path = output_dir / f"{dataset_stem}.metadata.json"

    transitions.to_parquet(parquet_path, index=False)
    metadata = {
        "schema_version": 2,
        "source_csv": str(csv_path),
        "source_run_provenance": provenance,
        "output_parquet": str(parquet_path),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "state_definition": [
            "v_x_est_mps",
            "v_y_est_mps",
            "imu_yaw_rate_rps",
        ],
        "state_estimator": {
            "name": "causal_body_velocity_backward_difference",
            "world_frame": "ENU",
            "body_frame": "ISO8855_x_forward_y_left_z_up",
            "max_gap_s": arguments.expected_dt_s + arguments.dt_tolerance_s,
        },
        "input_definition": ["command_linear_x_mps", "command_angular_z_rps"],
        "target_definition": ["v_x_t1_mps", "v_y_t1_mps", "yaw_rate_t1_rps"],
        "quality_thresholds": {
            "max_command_age_s": arguments.max_command_age_s,
            "max_alignment_s": arguments.max_alignment_s,
            "expected_dt_s": arguments.expected_dt_s,
            "dt_tolerance_s": arguments.dt_tolerance_s,
        },
        **counters,
    }
    with metadata_path.open("w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, indent=2)
        metadata_file.write("\n")

    print(f"Raw samples: {counters['raw_rows']}")
    print(f"Valid transitions: {counters['transitions']}")
    print(f"Parquet dataset: {parquet_path}")
    print(f"Metadata: {metadata_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError) as error:
        print(f"Preprocessing failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error
