"""Inspect and validate raw NeuroGrip-X vehicle recordings.

The script keeps raw recordings immutable. It reads one CSV created by
``state_logger``, writes a JSON quality summary, and saves diagnostic plots.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


REQUIRED_COLUMNS = {
    "time_s",
    "odom_imu_dt_s",
    "cmd_age_s",
    "x_m",
    "y_m",
    "v_x_mps",
    "yaw_rate_rps",
    "imu_w_z_rps",
    "cmd_linear_x_mps",
    "cmd_angular_z_rps",
}

STALE_COMMAND_THRESHOLD_S = 0.25


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Validate a raw NeuroGrip-X CSV recording and create plots."
    )
    parser.add_argument(
        "csv_path",
        nargs="?",
        type=Path,
        help="Raw CSV to inspect. Defaults to the newest data/raw/*.csv file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for JSON and PNG outputs. Defaults under runs/inspection/.",
    )
    return parser.parse_args()


def newest_recording(raw_directory: Path) -> Path:
    """Return the most recently modified raw CSV recording."""
    recordings = list(raw_directory.glob("*.csv"))
    if not recordings:
        raise FileNotFoundError(f"No CSV recordings found in {raw_directory}")
    return max(recordings, key=lambda path: path.stat().st_mtime)


def validate_recording(dataframe: pd.DataFrame) -> dict[str, float | int | bool]:
    """Compute dataset-quality metrics and reject structurally invalid CSV files."""
    missing_columns = REQUIRED_COLUMNS.difference(dataframe.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"CSV is missing required columns: {missing}")
    if dataframe.empty:
        raise ValueError("CSV contains no data rows.")

    time_deltas = dataframe["time_s"].diff().dropna()
    if (time_deltas <= 0).any():
        raise ValueError("time_s must be strictly increasing.")

    duration_s = float(dataframe["time_s"].iloc[-1] - dataframe["time_s"].iloc[0])
    median_dt_s = float(time_deltas.median()) if not time_deltas.empty else 0.0
    sample_rate_hz = 1.0 / median_dt_s if median_dt_s > 0 else 0.0
    known_command_age = dataframe.loc[dataframe["cmd_age_s"] >= 0, "cmd_age_s"]
    stale_command_fraction = (
        float((known_command_age > STALE_COMMAND_THRESHOLD_S).mean())
        if not known_command_age.empty
        else 1.0
    )

    return {
        "rows": int(len(dataframe)),
        "duration_s": duration_s,
        "median_dt_s": median_dt_s,
        "estimated_sample_rate_hz": sample_rate_hz,
        "max_abs_odom_imu_dt_ms": float(
            dataframe["odom_imu_dt_s"].abs().max() * 1_000.0
        ),
        "mean_odom_imu_dt_ms": float(
            dataframe["odom_imu_dt_s"].mean() * 1_000.0
        ),
        "max_speed_mps": float(dataframe["v_x_mps"].abs().max()),
        "max_yaw_rate_rps": float(dataframe["yaw_rate_rps"].abs().max()),
        "known_command_fraction": float((dataframe["cmd_age_s"] >= 0).mean()),
        "stale_command_fraction": stale_command_fraction,
        "time_is_strictly_increasing": True,
    }


def save_signal_plot(dataframe: pd.DataFrame, output_path: Path) -> None:
    """Save time-series diagnostics for kinematics, commands, and alignment."""
    time_s = (dataframe["time_s"] - dataframe["time_s"].iloc[0]).to_numpy()
    figure, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    axes[0].plot(time_s, dataframe["v_x_mps"].to_numpy(), label="odometry vx")
    axes[0].plot(
        time_s,
        dataframe["cmd_linear_x_mps"].to_numpy(),
        linestyle="--",
        label="command linear x",
    )
    axes[0].set_ylabel("speed [m/s]")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(
        time_s,
        dataframe["yaw_rate_rps"].to_numpy(),
        label="odometry yaw rate",
    )
    axes[1].plot(
        time_s,
        dataframe["imu_w_z_rps"].to_numpy(),
        label="IMU yaw rate",
    )
    axes[1].plot(
        time_s,
        dataframe["cmd_angular_z_rps"].to_numpy(),
        linestyle="--",
        label="command angular z",
    )
    axes[1].set_ylabel("yaw rate [rad/s]")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].plot(time_s, dataframe["odom_imu_dt_s"].to_numpy() * 1_000.0)
    axes[2].axhline(0.0, color="black", linewidth=0.8)
    axes[2].set_xlabel("simulation time from run start [s]")
    axes[2].set_ylabel("odom − IMU [ms]")
    axes[2].grid(True, alpha=0.3)

    figure.suptitle("NeuroGrip-X recording diagnostics")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def save_trajectory_plot(dataframe: pd.DataFrame, output_path: Path) -> None:
    """Save the vehicle XY trajectory with an equal spatial aspect ratio."""
    figure, axis = plt.subplots(figsize=(7, 7))
    scatter = axis.scatter(
        dataframe["x_m"].to_numpy(),
        dataframe["y_m"].to_numpy(),
        c=dataframe["time_s"].to_numpy(),
        s=5,
        cmap="viridis",
    )
    axis.plot(dataframe["x_m"].to_numpy(), dataframe["y_m"].to_numpy(), alpha=0.25)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("x [m]")
    axis.set_ylabel("y [m]")
    axis.set_title("NeuroGrip-X vehicle trajectory")
    axis.grid(True, alpha=0.3)
    colorbar = figure.colorbar(scatter, ax=axis)
    colorbar.set_label("simulation time [s]")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def main() -> int:
    """Inspect a recording and write repeatable quality artifacts."""
    arguments = parse_arguments()
    csv_path = arguments.csv_path or newest_recording(Path("data/raw"))
    csv_path = csv_path.expanduser().resolve()

    if not csv_path.is_file():
        raise FileNotFoundError(f"Recording does not exist: {csv_path}")

    default_output = Path("runs/inspection") / csv_path.stem
    output_dir = (arguments.output_dir or default_output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataframe = pd.read_csv(csv_path)
    summary = validate_recording(dataframe)
    summary.update(
        {
            "csv_path": str(csv_path),
            "inspected_at_utc": datetime.now(timezone.utc).isoformat(),
        }
    )

    save_signal_plot(dataframe, output_dir / "signals.png")
    save_trajectory_plot(dataframe, output_dir / "trajectory.png")

    with (output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
        file.write("\n")

    print(f"Recording: {csv_path.name}")
    print(f"Rows: {summary['rows']}")
    print(f"Duration: {summary['duration_s']:.3f} s")
    print(f"Estimated sample rate: {summary['estimated_sample_rate_hz']:.2f} Hz")
    print(f"Max |odom - IMU|: {summary['max_abs_odom_imu_dt_ms']:.2f} ms")
    print(f"Known command fraction: {summary['known_command_fraction']:.1%}")
    print(f"Stale command fraction: {summary['stale_command_fraction']:.1%}")
    print(f"Outputs: {output_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError) as error:
        print(f"Inspection failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error
