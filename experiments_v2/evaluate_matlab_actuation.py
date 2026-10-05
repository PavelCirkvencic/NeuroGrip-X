#!/usr/bin/env python3
"""Gate one development lap with MATLAB providing lateral actuation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EPISODE_FIELDS = {
    "time_s",
    "lap_progress",
    "boundary_margin_m",
    "e_y_m",
    "cone_collision_count",
}
CANDIDATE_FIELDS = {
    "time_s",
    "steering_rad",
    "solution_valid",
    "solve_time_ms",
}
MUX_FIELDS = {
    "schema_version",
    "time_s",
    "source_id",
    "fallback_count",
    "reject_count",
}


def sha256_file(path: Path) -> str:
    """Return the immutable byte identity of one evidence input."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def report_path(path: Path) -> str:
    """Prefer stable repository-relative paths in committed reports."""
    try:
        return str(path.resolve().relative_to(REPOSITORY_ROOT))
    except ValueError:
        return path.name


def load_numeric_csv(path: Path, required: set[str]) -> dict[str, np.ndarray]:
    """Load required finite numeric columns from a nonempty CSV."""
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: missing fields {sorted(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path}: contains no rows")
    result = {}
    for field in required:
        try:
            result[field] = np.asarray(
                [float(row[field]) for row in rows], dtype=float
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"{path}: invalid numeric {field}") from error
        if not np.all(np.isfinite(result[field])):
            raise ValueError(f"{path}: {field} contains NaN or Inf")
    if np.any(np.diff(result["time_s"]) < 0.0):
        raise ValueError(f"{path}: timestamps are not monotonic")
    return result


def load_events(path: Path) -> list[dict]:
    """Load a strict JSONL event stream."""
    events = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{line_number}: invalid JSON") from error
    return events


def unique_event(events: list[dict], name: str) -> dict:
    """Return exactly one named event."""
    matches = [event for event in events if event.get("event") == name]
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one {name} event, got {len(matches)}"
        )
    return matches[0]


def summarize(
    episode_csv: Path,
    events_jsonl: Path,
    episode_metadata: Path,
    candidate_csv: Path,
    mux_csv: Path,
    source_paths: list[Path],
    *,
    evaluation_split: str = "development",
    performance_claim: bool = False,
    transport: str = "ros2_direct",
) -> dict:
    """Summarize only the controlled interval ending at lap completion."""
    episode = load_numeric_csv(episode_csv, EPISODE_FIELDS)
    candidate = load_numeric_csv(candidate_csv, CANDIDATE_FIELDS)
    mux = load_numeric_csv(mux_csv, MUX_FIELDS)
    events = load_events(events_jsonl)
    metadata = json.loads(episode_metadata.read_text(encoding="utf-8"))
    start_event = unique_event(events, "experiment_start")
    lap_event = unique_event(events, "lap_complete")
    start_time_s = float(start_event["sim_time_s"])
    lap_time_s = float(lap_event["sim_time_s"])
    if lap_time_s <= start_time_s:
        raise ValueError("lap completion precedes experiment start")

    controlled_episode = (
        (episode["time_s"] >= start_time_s)
        & (episode["time_s"] <= lap_time_s)
    )
    if not np.any(controlled_episode):
        raise ValueError("episode has no samples in the controlled interval")
    matlab_rows = np.flatnonzero(
        (mux["source_id"] == 2.0) & (mux["time_s"] <= lap_time_s)
    )
    if not len(matlab_rows):
        raise ValueError(
            "MATLAB never became the active source before lap completion"
        )
    takeover_index = int(matlab_rows[0])
    takeover_time_s = float(mux["time_s"][takeover_index])
    if float(np.max(mux["time_s"])) < lap_time_s:
        raise ValueError("mux trace ends before lap completion")
    active_mux = (
        (mux["time_s"] >= takeover_time_s) & (mux["time_s"] <= lap_time_s)
    )
    active_candidate = (
        (candidate["time_s"] >= takeover_time_s)
        & (candidate["time_s"] <= lap_time_s)
    )
    if not np.any(active_candidate):
        raise ValueError(
            "candidate trace has no samples after MATLAB takeover"
        )
    lap_mux_index = int(np.flatnonzero(mux["time_s"] <= lap_time_s)[-1])
    nearest_episode_index = int(
        np.argmin(np.abs(episode["time_s"] - takeover_time_s))
    )
    post_lap_mux = mux["time_s"] > lap_time_s
    final_fallback_count = float(mux["fallback_count"][-1])
    lap_fallback_count = float(mux["fallback_count"][lap_mux_index])

    report = {
        "schema_version": 1,
        "mode": f"matlab_lateral_actuator_in_loop_{evaluation_split}",
        "evaluation_split": evaluation_split,
        "performance_claim": performance_claim,
        "transport": transport,
        "lateral_authority": "matlab_mpc_active_set_one_step_delay_compensated",
        "longitudinal_authority": "python_shared_speed_policy",
        "episode_termination": metadata.get("runner", {}).get(
            "termination_reason"
        ),
        "lap_safety_valid": bool(lap_event.get("safety_valid", False)),
        "experiment_start_time_s": start_time_s,
        "lap_complete_time_s": lap_time_s,
        "takeover_time_s": takeover_time_s,
        "takeover_progress": float(
            episode["lap_progress"][nearest_episode_index]
        ),
        "matlab_active_fraction_after_experiment_start": float(
            (lap_time_s - takeover_time_s) / (lap_time_s - start_time_s)
        ),
        "active_mux_samples": int(np.count_nonzero(active_mux)),
        "active_candidate_samples": int(np.count_nonzero(active_candidate)),
        "active_non_matlab_mux_samples": int(
            np.count_nonzero(mux["source_id"][active_mux] != 2.0)
        ),
        "active_deadline_misses": int(
            np.count_nonzero(
                candidate["solve_time_ms"][active_candidate] > 20.0
            )
        ),
        "active_invalid_solutions": int(
            np.count_nonzero(
                candidate["solution_valid"][active_candidate] != 1.0
            )
        ),
        "active_median_solve_time_ms": float(
            np.median(candidate["solve_time_ms"][active_candidate])
        ),
        "active_p95_solve_time_ms": float(
            np.percentile(candidate["solve_time_ms"][active_candidate], 95)
        ),
        "active_max_solve_time_ms": float(
            np.max(candidate["solve_time_ms"][active_candidate])
        ),
        "new_rejections_after_takeover": int(
            mux["reject_count"][lap_mux_index]
            - mux["reject_count"][takeover_index]
        ),
        "new_fallbacks_after_takeover": int(
            lap_fallback_count - mux["fallback_count"][takeover_index]
        ),
        "pre_takeover_rejections": int(mux["reject_count"][takeover_index]),
        "post_lap_fallbacks_during_shutdown": int(
            final_fallback_count - lap_fallback_count
            if np.any(post_lap_mux)
            else 0
        ),
        "minimum_boundary_margin_m": float(
            np.min(episode["boundary_margin_m"][controlled_episode])
        ),
        "maximum_absolute_lateral_error_m": float(
            np.max(np.abs(episode["e_y_m"][controlled_episode]))
        ),
        "maximum_collision_count": int(
            np.max(episode["cone_collision_count"][controlled_episode])
        ),
        "evidence_sha256": {
            report_path(path): sha256_file(path)
            for path in (
                episode_csv,
                events_jsonl,
                episode_metadata,
                candidate_csv,
                mux_csv,
            )
        },
        "source_sha256": {
            report_path(path): sha256_file(path) for path in source_paths
        },
    }
    return report


def validate(report: dict) -> None:
    """Enforce safety and continuous-authority development gates."""
    if report["episode_termination"] != "lap_complete":
        raise ValueError("episode did not terminate with lap_complete")
    if not report["lap_safety_valid"]:
        raise ValueError("lap_complete event is not safety-valid")
    if report["takeover_progress"] > 0.15:
        raise ValueError("MATLAB took over too late in the lap")
    if report["matlab_active_fraction_after_experiment_start"] < 0.85:
        raise ValueError("MATLAB controlled too little of the measured lap")
    if report["active_non_matlab_mux_samples"] != 0:
        raise ValueError("actuation source changed before lap completion")
    if report["new_rejections_after_takeover"] != 0:
        raise ValueError("MATLAB candidate was rejected after takeover")
    if report["new_fallbacks_after_takeover"] != 0:
        raise ValueError("MATLAB actuation fell back before lap completion")
    if report["active_deadline_misses"] != 0:
        raise ValueError("MATLAB missed the 20 ms deadline after takeover")
    if report["active_invalid_solutions"] != 0:
        raise ValueError("MATLAB produced an invalid solution after takeover")
    if report["minimum_boundary_margin_m"] <= 0.0:
        raise ValueError("vehicle crossed the track boundary")
    if report["maximum_collision_count"] != 0:
        raise ValueError("vehicle collided with a cone")


def parse_arguments() -> argparse.Namespace:
    """Parse explicit evidence inputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-csv", type=Path, required=True)
    parser.add_argument("--events-jsonl", type=Path, required=True)
    parser.add_argument("--episode-metadata", type=Path, required=True)
    parser.add_argument("--candidate-csv", type=Path, required=True)
    parser.add_argument("--mux-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    """Validate, persist and print the MATLAB actuation gate."""
    arguments = parse_arguments()
    inputs = [
        arguments.episode_csv,
        arguments.events_jsonl,
        arguments.episode_metadata,
        arguments.candidate_csv,
        arguments.mux_csv,
    ]
    if any(not path.expanduser().is_file() for path in inputs):
        raise FileNotFoundError(
            "all MATLAB actuation evidence inputs must exist"
        )
    source_paths = [
        REPOSITORY_ROOT / "matlab/run_live_matlab_tcp_candidate.m",
        REPOSITORY_ROOT / "matlab/+neurogrip/python_equivalent_mpc_move.m",
        REPOSITORY_ROOT / "matlab/+neurogrip/latency_compensated_mpc_move.m",
        REPOSITORY_ROOT
        / "src/neurogrip_control/neurogrip_control/matlab_tcp_bridge.py",
        REPOSITORY_ROOT
        / "src/neurogrip_control/neurogrip_control/candidate_mux.py",
    ]
    report = summarize(
        *(path.expanduser().resolve() for path in inputs),
        source_paths,
        transport="localhost_tcp_ros_bridge",
    )
    validate(report)
    output = arguments.output_json.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        "MATLAB_ACTUATION_DEVELOPMENT_PASS "
        "active_fraction="
        f"{report['matlab_active_fraction_after_experiment_start']:.3f} "
        f"min_margin_m={report['minimum_boundary_margin_m']:.3f} "
        f"max_abs_ey_m={report['maximum_absolute_lateral_error_m']:.3f} "
        f"p95_solve_ms={report['active_p95_solve_time_ms']:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
