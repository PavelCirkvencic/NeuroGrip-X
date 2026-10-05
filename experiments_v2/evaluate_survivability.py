#!/usr/bin/env python3
"""Report paired C0/C1/C2 completion and DNF outcomes without hiding either.

``evaluate_closed_loop.py`` is intentionally strict: every controller must
finish a lap before it calculates a common RMS table.  This companion evaluator
is for a deliberately harsher but still provenance-paired scenario where a
baseline may time out.  It never computes a post-transition RMS for a DNF run;
instead it reports that terminal state, final recorded progress and time.  A
C2 result is accepted only if its usual activation/no-post-fallback contract
also passes.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

try:  # Direct CLI and package-test compatible imports.
    from .evaluate_closed_loop import (
        CONTROLLERS,
        discover_runs,
        load_episode,
        load_events,
        metrics,
        provenance_key,
        validate_c2_activation,
    )
except ImportError:  # pragma: no cover - exercised by direct CLI execution
    from evaluate_closed_loop import (
        CONTROLLERS,
        discover_runs,
        load_episode,
        load_events,
        metrics,
        provenance_key,
        validate_c2_activation,
    )


def parse_arguments() -> argparse.Namespace:
    """Parse a canonical run root and a separate immutable report location."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scenario-prefix", default="")
    return parser.parse_args()


def terminal_summary(episode: dict[str, np.ndarray], metadata: dict) -> dict:
    """Describe an incomplete run without inventing a completion-only metric."""
    time_s = episode["time_s"]
    return {
        "completed_lap": False,
        "termination_reason": metadata.get("runner", {}).get("termination_reason"),
        "final_recorded_time_s": float(time_s[-1] - time_s[0]),
        "final_lap_progress": float(episode["lap_progress"][-1]),
        "final_abs_lateral_error_m": float(abs(episode["e_y_m"][-1])),
    }


def paired_survivability_report(run_root: Path, scenario_prefix: str) -> dict:
    """Validate pairing and report completion metrics or explicit DNF summaries."""
    grouped = discover_runs(run_root, scenario_prefix)
    if not grouped:
        raise ValueError("no canonical episodes found")
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_root": str(run_root),
        "scenarios": {},
    }
    for scenario_id, by_controller in sorted(grouped.items()):
        missing = [controller for controller in CONTROLLERS if controller not in by_controller]
        if missing:
            raise ValueError(f"{scenario_id}: missing paired controllers {missing}")
        metadata_by_controller = {
            controller: json.loads(path.with_suffix(".metadata.json").read_text())
            for controller, path in by_controller.items()
        }
        keys = {provenance_key(metadata) for metadata in metadata_by_controller.values()}
        if len(keys) != 1 or any(value is None for value in next(iter(keys))):
            raise ValueError(f"{scenario_id}: controller runs do not share immutable provenance")
        scenario_report = {}
        for controller in CONTROLLERS:
            episode_path = by_controller[controller]
            event_path = episode_path.with_suffix(".events.jsonl")
            if not event_path.is_file():
                raise ValueError(f"missing event log: {event_path}")
            episode, events = load_episode(episode_path), load_events(event_path)
            metadata = metadata_by_controller[controller]
            completed = metadata.get("runner", {}).get("termination_reason") == "lap_complete"
            if completed:
                result = metrics(episode, events)
            else:
                result = terminal_summary(episode, metadata)
            if controller == "C2_NEUROGRIP":
                if not completed:
                    raise ValueError("C2 did not complete; cannot claim a survivability advantage")
                result.update(
                    validate_c2_activation(events, result["transition_confirmed_sim_s"])
                )
            scenario_report[controller] = result
        report["scenarios"][scenario_id] = scenario_report
    report["aggregate"] = aggregate_report(report["scenarios"])
    return report


def aggregate_report(scenarios: dict[str, dict]) -> dict:
    """Summarise all frozen scenarios without ranking DNF partial telemetry."""
    completion = {
        controller: int(sum(
            result[controller]["completed_lap"] for result in scenarios.values()
        ))
        for controller in CONTROLLERS
    }
    paired = []
    for scenario_id, result in sorted(scenarios.items()):
        c0, c2 = result["C0_FIXED"], result["C2_NEUROGRIP"]
        if c0["completed_lap"] and c2["completed_lap"]:
            c0_rms, c2_rms = c0["post_transition_rms_m"], c2["post_transition_rms_m"]
            if c0_rms is not None and c2_rms is not None:
                paired.append({
                    "scenario_id": scenario_id,
                    "c0_post_transition_rms_m": c0_rms,
                    "c2_post_transition_rms_m": c2_rms,
                    "c2_minus_c0_rms_m": c2_rms - c0_rms,
                    "c2_relative_reduction_percent": 100.0 * (c0_rms - c2_rms) / c0_rms,
                })
    return {
        "scenario_count": len(scenarios),
        "completed_laps": completion,
        "c0_c2_completed_pairs": paired,
        "mean_c2_minus_c0_rms_m": (
            None if not paired else float(np.mean([item["c2_minus_c0_rms_m"] for item in paired]))
        ),
        "mean_c2_relative_reduction_percent": (
            None if not paired else float(np.mean([
                item["c2_relative_reduction_percent"] for item in paired
            ]))
        ),
    }


def markdown_report(report: dict) -> str:
    """Render only completion-comparable RMS values and clear DNF outcomes."""
    lines = [
        "# Paired survivability benchmark", "",
        "A DNF/timeout has no common-lap RMS, so it is reported as a terminal outcome rather than ranked by a misleading partial RMS.", "",
        "| Scenario | Controller | Terminal outcome | Progress | Post-transition RMS [m] | Settling [s] |",
        "|---|---|---|---:|---:|---:|",
    ]
    for scenario_id, controllers in report["scenarios"].items():
        for controller, result in controllers.items():
            if result["completed_lap"]:
                terminal = "lap_complete"
                progress = "100.0%"
                post = result["post_transition_rms_m"]
                settling = result["settling_after_transition_s"]
                lines.append(
                    f"| {scenario_id} | {controller} | {terminal} | {progress} | "
                    f"{'-' if post is None else f'{post:.4f}'} | "
                    f"{'-' if settling is None else f'{settling:.2f}'} |"
                )
            else:
                terminal = str(result["termination_reason"] or "unknown")
                lines.append(
                    f"| {scenario_id} | {controller} | {terminal} | "
                    f"{result['final_lap_progress'] * 100.0:.1f}% | - | - |"
                )
    aggregate = report["aggregate"]
    lines.extend([
        "", "## Frozen-matrix aggregate", "",
        "Only scenarios where both C0 and C2 completed contribute to the paired RMS average. A timeout remains a completion result, never a partial-RMS rank.", "",
        f"Completed laps — C0: {aggregate['completed_laps']['C0_FIXED']}/{aggregate['scenario_count']}; "
        f"C1: {aggregate['completed_laps']['C1_ORACLE_MU']}/{aggregate['scenario_count']}; "
        f"C2: {aggregate['completed_laps']['C2_NEUROGRIP']}/{aggregate['scenario_count']}.",
    ])
    if aggregate["mean_c2_relative_reduction_percent"] is None:
        lines.append("No completed C0/C2 pairs had a common post-transition RMS.")
    else:
        lines.append(
            "C0/C2 completed pairs: "
            f"{len(aggregate['c0_c2_completed_pairs'])}; mean C2−C0 post-transition RMS: "
            f"{aggregate['mean_c2_minus_c0_rms_m']:.4f} m; mean C2 relative reduction: "
            f"{aggregate['mean_c2_relative_reduction_percent']:.2f}% (descriptive, not a significance test)."
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    """Write JSON and Markdown from paired run artifacts without side effects."""
    arguments = parse_arguments()
    run_root = arguments.run_root.expanduser().resolve()
    output_dir = arguments.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report = paired_survivability_report(run_root, arguments.scenario_prefix)
    output_dir.joinpath("survivability_summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    text = markdown_report(report)
    output_dir.joinpath("survivability_table.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
