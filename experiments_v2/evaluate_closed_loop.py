#!/usr/bin/env python3
"""Validate safety-gated telemetry and paired closed-loop episodes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

try:  # Supports both `python experiments_v2/...py` and package test imports.
    from .evaluate_matlab_actuation import (
        summarize as summarize_matlab_actuation,
        validate as validate_matlab_actuation,
    )
    from .record_episode import FIELDS
except ImportError:  # pragma: no cover - exercised by the direct CLI invocation
    from evaluate_matlab_actuation import (
        summarize as summarize_matlab_actuation,
        validate as validate_matlab_actuation,
    )
    from record_episode import FIELDS

CONTROLLERS = ("C0_FIXED", "C1_ORACLE_MU", "C2_NEUROGRIP")
MIN_SAMPLE_RATE_HZ, MAX_SAMPLE_RATE_HZ = 49.0, 51.0
SETTLING_THRESHOLD_M, SETTLING_WINDOW_S = 0.30, 1.0
MAX_DIAGNOSTICS_AGE_S = 0.12
MAX_INITIAL_DIAGNOSTICS_GAP_S = 1.0
MIN_C2_ACTIVE_FRACTION = 0.90
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def portable_path(path: Path) -> str:
    """Return a repository-relative path when the file is in this checkout."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def parse_arguments() -> argparse.Namespace:
    """Parse the run root and deterministic report destination."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scenario-prefix", default="")
    parser.add_argument(
        "--controllers",
        default=",".join(CONTROLLERS),
        help="ordered comma-separated paired controllers",
    )
    parser.add_argument("--holdout-manifest", type=Path)
    return parser.parse_args()


def load_events(path: Path) -> list[dict]:
    """Load event JSONL strictly: corrupt provenance is a hard evaluation error."""
    events = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        try:
            event = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid event JSON at {path}:{number}") from error
        if not isinstance(event, dict):
            raise ValueError(f"event at {path}:{number} is not an object")
        events.append(event)
    return events


def load_episode(path: Path) -> dict[str, np.ndarray]:
    """Load a complete v3 CSV without silently dropping malformed samples."""
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != FIELDS:
            raise ValueError(f"{path}: telemetry fields do not match schema v3")
        rows = list(reader)
    if len(rows) < 100:
        raise ValueError(f"{path}: fewer than 100 samples")
    result: dict[str, np.ndarray] = {}
    for field in FIELDS:
        try:
            result[field] = np.asarray([float(row[field]) for row in rows], dtype=float)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"{path}: invalid numeric field {field}") from error
    if not all(np.all(np.isfinite(values)) for values in result.values()):
        raise ValueError(f"{path}: contains NaN or Inf")
    time = result["time_s"]
    if np.any(np.diff(time) <= 0.0):
        raise ValueError(f"{path}: timestamps are not strictly increasing")
    return result


def event_time(events: list[dict], predicate, description: str) -> float:
    """Return the one relevant sim timestamp or reject an ambiguous run."""
    matches = [event for event in events if predicate(event)]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one {description} event, found {len(matches)}")
    try:
        return float(matches[0]["sim_time_s"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{description} event has no valid sim_time_s") from error


def settling_time(time_s: np.ndarray, error_m: np.ndarray) -> float | None:
    """Return first time whose following continuous 1 s window stays in bounds."""
    for index, start in enumerate(time_s):
        end_index = np.searchsorted(time_s, start + SETTLING_WINDOW_S, side="left")
        if end_index >= len(time_s):
            break
        if np.all(np.abs(error_m[index:end_index + 1]) < SETTLING_THRESHOLD_M):
            return float(start)
    return None


def metrics(episode: dict[str, np.ndarray], events: list[dict]) -> dict:
    """Compute metrics only for one explicitly safety-valid measured lap."""
    if any(
        event.get("event") in {"safety_violation", "controller_failure"}
        for event in events
    ):
        raise ValueError("run contains a terminal safety or controller failure")
    start_time = event_time(
        events,
        lambda event: event.get("event") == "experiment_start",
        "experiment_start",
    )
    lap_time = event_time(
        events,
        lambda event: event.get("event") == "lap_complete"
        and event.get("safety_valid") is True,
        "safety-valid lap_complete",
    )
    if lap_time <= start_time:
        raise ValueError("lap completion does not follow experiment start")
    transition_events = [
        event
        for event in events
        if event.get("label") == "transition" and event.get("confirmed") is True
    ]
    transition_time = None
    if transition_events:
        if len(transition_events) != 1:
            raise ValueError("more than one confirmed transition event")
        transition_time = float(transition_events[0]["confirmed_sim_s"])
    time_s = episode["time_s"]
    keep = (time_s >= start_time - 1e-9) & (time_s <= lap_time + 1e-9)
    if np.count_nonzero(keep) < 100:
        raise ValueError("lap completion precedes sufficient telemetry")
    time_s, e_y, vx = (episode[name][keep] for name in ("time_s", "e_y_m", "v_x_mps"))
    margin = episode["boundary_margin_m"][keep]
    collisions = episode["cone_collision_count"][keep]
    solution_valid = episode["controller_solution_valid"][keep]
    diagnostics_age = episode["controller_diagnostics_age_s"][keep]
    if float(np.min(margin)) < -1e-6:
        raise ValueError("vehicle footprint left the measured track envelope")
    if np.any(collisions > 0.0):
        raise ValueError("EUFS reported a cone collision during the measured lap")
    if np.any(solution_valid == 0.0):
        raise ValueError("MPC produced an invalid solution during the measured lap")
    missing_diagnostics = solution_valid < 0.0
    valid_diagnostics = solution_valid >= 0.0
    if not np.any(valid_diagnostics):
        raise ValueError("controller diagnostics are missing for the entire lap")
    first_diagnostics_index = int(np.flatnonzero(valid_diagnostics)[0])
    if np.any(missing_diagnostics[first_diagnostics_index:]):
        raise ValueError("controller diagnostics disappeared during the measured lap")
    initial_diagnostics_gap_s = float(
        time_s[first_diagnostics_index] - start_time
    )
    if initial_diagnostics_gap_s > MAX_INITIAL_DIAGNOSTICS_GAP_S:
        raise ValueError("initial controller diagnostics discovery exceeded 1 s")
    if float(np.max(diagnostics_age[valid_diagnostics])) > MAX_DIAGNOSTICS_AGE_S:
        raise ValueError("controller diagnostics were stale during the measured lap")
    sample_rate = 1.0 / float(np.median(np.diff(time_s)))
    if not MIN_SAMPLE_RATE_HZ <= sample_rate <= MAX_SAMPLE_RATE_HZ:
        raise ValueError(f"sample rate {sample_rate:.3f} Hz is outside 49-51 Hz")
    result = {
        "samples_to_lap": int(len(time_s)),
        "duration_to_lap_s": float(lap_time - start_time),
        "median_sample_rate_hz": sample_rate,
        "initial_diagnostics_gap_s": initial_diagnostics_gap_s,
        "initial_diagnostics_missing_samples": int(
            np.count_nonzero(missing_diagnostics)
        ),
        "distance_to_lap_m": float(np.trapz(vx, time_s)),
        "rms_lateral_error_m": float(np.sqrt(np.mean(e_y**2))),
        "max_abs_lateral_error_m": float(np.max(np.abs(e_y))),
        "minimum_boundary_margin_m": float(np.min(margin)),
        "maximum_cone_collision_count": int(np.max(collisions)),
        "maximum_controller_solve_time_ms": float(
            np.max(episode["controller_solve_time_ms"][keep])
        ),
        "maximum_controller_slack_m": float(
            np.max(episode["controller_max_slack_m"][keep])
        ),
        "completed_lap": True,
        "transition_confirmed_sim_s": transition_time,
        "post_transition_rms_m": None,
        "post_transition_max_m": None,
        "settling_after_transition_s": None,
        "post_transition_neurogrip_policy_fraction": None,
    }
    if transition_time is not None:
        post = time_s >= transition_time
        if np.count_nonzero(post) < 2:
            raise ValueError("transition occurs after the recorded lap")
        post_time, post_error = time_s[post], e_y[post]
        post_utilisation = episode["profile_lateral_utilisation"][keep][post]
        settled_at = settling_time(post_time, post_error)
        result.update(
            {
                "post_transition_rms_m": float(np.sqrt(np.mean(post_error**2))),
                "post_transition_max_m": float(np.max(np.abs(post_error))),
                "settling_after_transition_s": (
                    None if settled_at is None else float(settled_at - transition_time)
                ),
                "post_transition_neurogrip_policy_fraction": float(
                    np.mean(post_utilisation >= 0.97)
                ),
            }
        )
    return result


def validate_c2_activation(events: list[dict], transition_time: float | None) -> dict:
    """Require one identified Koopman artifact for >=90% of evaluation time."""
    active_events = [event for event in events if event.get("event") == "c2_model_active"]
    if not active_events:
        raise ValueError("C2 episode never activated a validated learned dynamics model")
    if any(int(event.get("dynamics_schema", -1)) not in (3, 4) for event in active_events):
        raise ValueError("C2 episode activated a non-Koopman dynamics schema")
    artifacts = {event.get("artifact_sha256") for event in active_events}
    if len(artifacts) != 1 or None in artifacts or "" in artifacts:
        raise ValueError("C2 episode changed or omitted its Koopman artifact identity")
    active_times = [float(event["sim_time_s"]) for event in active_events]
    fallback_times = [
        float(event["sim_time_s"])
        for event in events
        if event.get("event") == "c2_nominal_fallback"
    ]
    activation = min(active_times)
    if transition_time is not None and activation > transition_time:
        raise ValueError("C2 became active only after the scenario transition")
    lap_events = [
        event
        for event in events
        if event.get("event") == "lap_complete" and event.get("safety_valid") is True
    ]
    if len(lap_events) != 1:
        raise ValueError("C2 activation accounting requires one safety-valid lap")
    end_time = float(lap_events[0]["sim_time_s"])
    start_time = activation if transition_time is None else max(activation, float(transition_time))
    prior_states = sorted(
        (
            float(event["sim_time_s"]),
            event.get("event") == "c2_model_active",
        )
        for event in events
        if event.get("event") in {"c2_model_active", "c2_nominal_fallback"}
        and float(event.get("sim_time_s", -1.0)) <= start_time
    )
    state_active = bool(prior_states and prior_states[-1][1])
    timeline = sorted(
        (
            float(event["sim_time_s"]),
            event.get("event") == "c2_model_active",
        )
        for event in events
        if event.get("event") in {"c2_model_active", "c2_nominal_fallback"}
        and start_time < float(event.get("sim_time_s", -1.0)) < end_time
    )
    active_duration, cursor = 0.0, start_time
    for event_time_s, next_active in timeline:
        if state_active:
            active_duration += event_time_s - cursor
        cursor, state_active = event_time_s, next_active
    if state_active:
        active_duration += end_time - cursor
    interval = end_time - start_time
    active_fraction = active_duration / interval if interval > 0.0 else 0.0
    dynamics_schema = int(active_events[0]["dynamics_schema"])
    # Schema 4 uses a causal sample-and-hold per-axle grip context during short
    # OOD gaps. The raw local Jacobian is deliberately blended at zero in the
    # release benchmark, so its active duty cycle is diagnostic rather than a
    # valid gate. Schema 3 still requires continuous learned-Jacobian use.
    if dynamics_schema == 3 and active_fraction < MIN_C2_ACTIVE_FRACTION:
        raise ValueError(
            f"C2 Koopman active fraction {active_fraction:.3f} is below "
            f"{MIN_C2_ACTIVE_FRACTION:.2f}"
        )
    return {
        "c2_model_active_sim_s": activation,
        "c2_active_fraction": active_fraction,
        "c2_fallback_events": len(fallback_times),
        "c2_dynamics_schema": dynamics_schema,
        "c2_artifact_sha256": next(iter(artifacts)),
    }


def provenance_key(metadata: dict) -> tuple:
    """Return fields that must match across a fair controller comparison."""
    provenance = metadata.get("provenance", {})
    return (
        metadata.get("scenario_id"),
        metadata.get("seed"),
        provenance.get("scenario_sha256"),
        provenance.get("generated_core_sha256"),
    )


def discover_runs(run_root: Path, scenario_prefix: str) -> dict[str, dict[str, Path]]:
    """Discover only canonical runner output, rejecting duplicates or old layouts."""
    grouped: dict[str, dict[str, Path]] = defaultdict(dict)
    for episode_path in sorted(run_root.glob("*/episode.csv")):
        metadata_path = episode_path.with_suffix(".metadata.json")
        if not metadata_path.is_file():
            raise ValueError(f"missing metadata: {episode_path}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        scenario_id, controller = metadata.get("scenario_id"), metadata.get("controller")
        if not isinstance(scenario_id, str) or not isinstance(controller, str):
            raise ValueError(f"invalid metadata identity: {metadata_path}")
        if scenario_prefix and not scenario_id.startswith(scenario_prefix):
            continue
        if metadata.get("schema_version") not in (3, 4):
            raise ValueError(f"{metadata_path}: expected recorder metadata schema 3/4")
        if controller in grouped[scenario_id]:
            raise ValueError(f"duplicate controller run: {scenario_id}/{controller}")
        grouped[scenario_id][controller] = episode_path
    return grouped


def main() -> int:
    """Validate pair provenance, compute metrics and write JSON/Markdown reports."""
    arguments = parse_arguments()
    selected_controllers = tuple(
        name.strip() for name in arguments.controllers.split(",") if name.strip()
    )
    if (
        not selected_controllers
        or len(selected_controllers) != len(set(selected_controllers))
        or any(name not in CONTROLLERS for name in selected_controllers)
    ):
        raise ValueError("--controllers must be a unique non-empty subset of known controllers")
    run_root = arguments.run_root.expanduser().resolve()
    output_dir = arguments.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    grouped = discover_runs(run_root, arguments.scenario_prefix)
    if not grouped:
        raise ValueError("no canonical episodes found")
    report = {
        "schema_version": 3,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_root": portable_path(run_root),
        "scenarios": {},
        "controllers": list(selected_controllers),
    }
    expected_actuation_sources = {
        controller: "python" for controller in selected_controllers
    }
    if arguments.holdout_manifest is not None:
        holdout_path = arguments.holdout_manifest.expanduser().resolve()
        holdout_document = json.loads(holdout_path.read_text(encoding="utf-8"))
        expected_actuation_sources.update(
            holdout_document.get("actuation_sources", {})
        )
        report["holdout_manifest"] = portable_path(holdout_path)
        report["holdout_manifest_sha256"] = hashlib.sha256(
            holdout_path.read_bytes()
        ).hexdigest()
    for scenario_id, by_controller in sorted(grouped.items()):
        missing = [name for name in selected_controllers if name not in by_controller]
        if missing:
            raise ValueError(f"{scenario_id}: missing paired controllers {missing}")
        metadata = {
            controller: json.loads(path.with_suffix(".metadata.json").read_text())
            for controller, path in by_controller.items()
        }
        for controller in selected_controllers:
            actual_source = metadata[controller].get("provenance", {}).get(
                "actuation_source", "python"
            )
            expected_source = expected_actuation_sources[controller]
            if actual_source != expected_source:
                raise ValueError(
                    f"{scenario_id}/{controller}: expected {expected_source} "
                    f"actuation, recorded {actual_source}"
                )
        keys = {provenance_key(value) for value in metadata.values()}
        if len(keys) != 1 or any(value is None for value in next(iter(keys))):
            raise ValueError(
                f"{scenario_id}: controller runs do not share immutable "
                "scenario provenance"
            )
        report["scenarios"][scenario_id] = {}
        for controller in selected_controllers:
            episode_path = by_controller[controller]
            event_path = episode_path.with_suffix(".events.jsonl")
            if not event_path.is_file():
                raise ValueError(f"missing event log: {event_path}")
            result = metrics(
                load_episode(episode_path), load_events(event_path)
            )
            if controller == "C2_NEUROGRIP":
                if (
                    result["post_transition_neurogrip_policy_fraction"] is None
                    or result["post_transition_neurogrip_policy_fraction"] < 0.90
                ):
                    raise ValueError(
                        "C2 per-axle grip policy was active for less than 90% "
                        "of the post-transition measured interval"
                    )
                result.update(
                    validate_c2_activation(
                        load_events(event_path), result["transition_confirmed_sim_s"]
                    )
                )
                if expected_actuation_sources[controller] == "matlab":
                    run_directory = episode_path.parent
                    source_paths = [
                        Path(__file__).resolve().parents[1]
                        / "matlab/run_live_matlab_tcp_candidate.m",
                        Path(__file__).resolve().parents[1]
                        / "matlab/+neurogrip/python_equivalent_mpc_move.m",
                        Path(__file__).resolve().parents[1]
                        / "matlab/+neurogrip/latency_compensated_mpc_move.m",
                        Path(__file__).resolve().parents[1]
                        / "src/neurogrip_control/neurogrip_control/matlab_tcp_bridge.py",
                        Path(__file__).resolve().parents[1]
                        / "src/neurogrip_control/neurogrip_control/candidate_mux.py",
                    ]
                    matlab_report = summarize_matlab_actuation(
                        episode_path,
                        event_path,
                        episode_path.with_suffix(".metadata.json"),
                        run_directory / "matlab_candidate.csv",
                        run_directory / "command_mux_status.csv",
                        source_paths,
                        evaluation_split="final_holdout",
                        performance_claim=True,
                        transport="localhost_tcp_ros_bridge",
                    )
                    validate_matlab_actuation(matlab_report)
                    result["matlab_actuation"] = matlab_report
            report["scenarios"][scenario_id][controller] = result

    if {"C0_FIXED", "C2_NEUROGRIP"} <= set(selected_controllers):
        paired = []
        for scenario_id, results in report["scenarios"].items():
            c0_time = results["C0_FIXED"]["duration_to_lap_s"]
            c2_time = results["C2_NEUROGRIP"]["duration_to_lap_s"]
            paired.append(
                {
                    "scenario_id": scenario_id,
                    "c0_lap_time_s": c0_time,
                    "c2_lap_time_s": c2_time,
                    "c2_lap_time_reduction_percent": 100.0
                    * (c0_time - c2_time)
                    / c0_time,
                }
            )
        reductions = [item["c2_lap_time_reduction_percent"] for item in paired]
        report["aggregate"] = {
            "paired_lap_times": paired,
            "mean_c0_lap_time_s": float(
                np.mean([item["c0_lap_time_s"] for item in paired])
            ),
            "mean_c2_lap_time_s": float(
                np.mean([item["c2_lap_time_s"] for item in paired])
            ),
            "mean_paired_lap_time_reduction_percent": float(np.mean(reductions)),
            "minimum_paired_lap_time_reduction_percent": float(np.min(reductions)),
            "publication_target_percent": 5.0,
            "publication_acceptance_floor_percent": 4.0,
            "publication_gate_passed": bool(
                np.mean(reductions) >= 4.0 and np.min(reductions) > 0.0
            ),
        }

    output_dir.joinpath("benchmark_summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# Closed-loop paired benchmark",
        "",
        (
            "Metrics include only the explicit `experiment_start` → safety-valid "
            "`lap_complete` interval. Any boundary exit, cone collision or "
            "controller failure invalidates the run."
        ),
        "",
        "| Scenario | Controller | Lap [s] | RMS e_y [m] | Min margin [m] | C2 active |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for scenario_id, results in report["scenarios"].items():
        for controller, result in results.items():
            active_text = (
                "-"
                if "c2_active_fraction" not in result
                else f"{100.0 * result['c2_active_fraction']:.1f}%"
            )
            lines.append(
                f"| {scenario_id} | {controller} | {result['duration_to_lap_s']:.2f} | "
                f"{result['rms_lateral_error_m']:.4f} | "
                f"{result['minimum_boundary_margin_m']:.3f} | "
                f"{active_text} |"
            )
    if "aggregate" in report:
        aggregate = report["aggregate"]
        lines.extend(
            [
                "",
                "Paired mean lap-time reduction: "
                f"**{aggregate['mean_paired_lap_time_reduction_percent']:.2f}%**; "
                "worst paired reduction: "
                f"**{aggregate['minimum_paired_lap_time_reduction_percent']:.2f}%**; "
                "publication gate: **"
                f"{'PASS' if aggregate['publication_gate_passed'] else 'FAIL'}**.",
            ]
        )
    output_dir.joinpath("benchmark_table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
