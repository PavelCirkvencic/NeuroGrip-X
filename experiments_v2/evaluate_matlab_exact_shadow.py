#!/usr/bin/env python3
"""Validate a read-only live MATLAB replay of the release C2 MPC decision."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

REQUIRED_FIELDS = {
    "time_s",
    "matlab_steering_rad",
    "python_steering_rad",
    "abs_error_rad",
    "solve_time_ms",
    "deadline_miss",
}
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    """Return the immutable identity of one evidence file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def report_path(path: Path) -> str:
    """Use a repository-relative identity without leaking a workstation path."""
    try:
        return str(path.resolve().relative_to(REPOSITORY_ROOT))
    except ValueError:
        return path.name


def load_trace(path: Path) -> dict[str, np.ndarray]:
    """Load and strictly validate the exact-shadow CSV contract."""
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_FIELDS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: missing fields {sorted(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path}: contains no samples")
    values: dict[str, np.ndarray] = {}
    for field in REQUIRED_FIELDS:
        try:
            values[field] = np.asarray(
                [float(row[field]) for row in rows], dtype=float
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"{path}: invalid numeric {field}") from error
    if any(not np.all(np.isfinite(column)) for column in values.values()):
        raise ValueError(f"{path}: contains NaN or Inf")
    if np.any(np.diff(values["time_s"]) <= 0.0):
        raise ValueError(f"{path}: timestamps are not strictly increasing")
    if np.any(~np.isin(values["deadline_miss"], [0.0, 1.0])):
        raise ValueError(f"{path}: deadline_miss must be Boolean")
    recomputed_error = np.abs(
        values["matlab_steering_rad"] - values["python_steering_rad"]
    )
    if not np.allclose(recomputed_error, values["abs_error_rad"], atol=1e-12):
        raise ValueError(f"{path}: abs_error_rad is inconsistent")
    return values


def summarize(
    trace_path: Path,
    episode_metadata_path: Path,
    source_paths: list[Path],
) -> dict:
    """Build a compact report tied to the trace, episode and source bytes."""
    trace = load_trace(trace_path)
    metadata = json.loads(episode_metadata_path.read_text(encoding="utf-8"))
    termination = metadata.get("runner", {}).get("termination_reason")
    matlab_move = trace["matlab_steering_rad"]
    python_move = trace["python_steering_rad"]
    nonzero = (np.abs(matlab_move) > 1e-5) & (np.abs(python_move) > 1e-5)
    return {
        "schema_version": 1,
        "mode": "matlab_exact_mpc_shadow",
        "read_only": True,
        "capture_file": trace_path.name,
        "capture_sha256": sha256_file(trace_path),
        "episode_termination": termination,
        "samples": int(len(trace["time_s"])),
        "median_abs_error_rad": float(np.median(trace["abs_error_rad"])),
        "max_abs_error_rad": float(np.max(trace["abs_error_rad"])),
        "sign_mismatches": int(
            np.count_nonzero(
                nonzero & (np.sign(matlab_move) != np.sign(python_move))
            )
        ),
        "median_solve_time_ms": float(np.median(trace["solve_time_ms"])),
        "p95_solve_time_ms": float(np.percentile(trace["solve_time_ms"], 95)),
        "max_solve_time_ms": float(np.max(trace["solve_time_ms"])),
        "deadline_ms": 20.0,
        "deadline_misses": int(np.sum(trace["deadline_miss"])),
        "source_sha256": {
            report_path(path): sha256_file(path) for path in source_paths
        },
    }


def parse_arguments() -> argparse.Namespace:
    """Parse explicit evidence inputs and an output report path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shadow-csv", type=Path, required=True)
    parser.add_argument("--episode-metadata", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--min-samples", type=int, default=100)
    parser.add_argument("--max-error-rad", type=float, default=1e-3)
    parser.add_argument("--max-deadline-misses", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    """Validate acceptance gates and persist the compact report."""
    arguments = parse_arguments()
    trace_path = arguments.shadow_csv.expanduser().resolve()
    metadata_path = arguments.episode_metadata.expanduser().resolve()
    source_paths = [
        REPOSITORY_ROOT / "matlab/+neurogrip/python_equivalent_mpc_move.m",
        REPOSITORY_ROOT / "matlab/monitor_live_python_equivalent_shadow.m",
    ]
    if not trace_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError("shadow CSV and episode metadata must exist")
    if arguments.min_samples < 1 or arguments.max_error_rad <= 0.0:
        raise ValueError("invalid exact-shadow acceptance threshold")
    report = summarize(trace_path, metadata_path, source_paths)
    if report["episode_termination"] != "lap_complete":
        raise ValueError("exact shadow episode did not complete a lap")
    if report["samples"] < arguments.min_samples:
        raise ValueError("exact shadow has too few samples")
    if report["max_abs_error_rad"] >= arguments.max_error_rad:
        raise ValueError("exact shadow exceeds the steering parity tolerance")
    if report["sign_mismatches"] != 0:
        raise ValueError("exact shadow contains steering sign mismatches")
    if report["deadline_misses"] > arguments.max_deadline_misses:
        raise ValueError("exact shadow exceeds the real-time deadline budget")
    output_path = arguments.output_json.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        "MATLAB_EXACT_SHADOW_PASS "
        f"samples={report['samples']} "
        f"max_error_rad={report['max_abs_error_rad']:.3e} "
        f"p95_solve_ms={report['p95_solve_time_ms']:.3f} "
        f"deadline_misses={report['deadline_misses']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
