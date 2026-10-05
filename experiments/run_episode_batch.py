#!/usr/bin/env python3
"""Run one or more traceable NeuroGrip-X excitation episodes end-to-end.

Each episode is a full, reproducible unit of work:

1. optionally (re)generate the seeded scenario SDF + manifest;
2. launch the matching headless Gazebo world and its ROS bridges/guard;
3. verify that exactly one ``/model/vehicle_blue/odometry`` publisher exists;
4. run the managed excitation recording launch;
5. tear the simulator down with targeted signals and wait for DDS to settle;
6. push the new raw CSV through the mandatory data pipeline.

The script never touches other simulations: teardown only signals the process
group it created and processes whose command line references the episode's
unique world file.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

ODOMETRY_TOPIC = "/model/vehicle_blue/odometry"
PUBLISHER_PATTERN = re.compile(r"Publisher count:\s*(\d+)")
REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Episode:
    """One scenario seed plus the excitation profile to record."""

    profile: str
    seed: int
    excitation: str
    records: int = 1

    @property
    def scenario_id(self) -> str:
        return f"{self.profile}_seed{self.seed:04d}"

    @property
    def label(self) -> str:
        suffix = f"x{self.records}" if self.records > 1 else ""
        return f"{self.scenario_id}:{self.excitation}{suffix}"


def parse_arguments() -> argparse.Namespace:
    """Parse the episode plan and pipeline options."""
    parser = argparse.ArgumentParser(
        description="Run traceable NeuroGrip-X excitation episodes end-to-end."
    )
    parser.add_argument(
        "--episode",
        action="append",
        required=True,
        metavar="PROFILE:SEED:EXCITATION[:REPEATS]",
        help="Episode spec, e.g. nominal:11:dynamic_v2 or low_grip:12:baseline_v1:2.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw"),
        help="Raw CSV/metadata output directory.",
    )
    parser.add_argument(
        "--world-dir",
        type=Path,
        default=Path("data/generated_worlds"),
        help="Generated scenario directory.",
    )
    parser.add_argument(
        "--sim-timeout-s",
        type=float,
        default=60.0,
        help="Maximum wait for the odometry publisher to appear.",
    )
    parser.add_argument(
        "--recording-timeout-s",
        type=float,
        default=90.0,
        help="Maximum duration of one managed excitation recording.",
    )
    parser.add_argument(
        "--teardown-timeout-s",
        type=float,
        default=30.0,
        help="Maximum wait for the simulator DDS graph to disappear.",
    )
    parser.add_argument(
        "--skip-pipeline",
        action="store_true",
        help="Only record; do not run preprocessing/catalog/training.",
    )
    parser.add_argument(
        "--no-regenerate",
        action="store_true",
        help="Reuse existing scenario SDF/manifest instead of regenerating.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Request Gazebo server-only mode (no GUI rendering).",
    )
    parser.add_argument(
        "--ai-checkpoint",
        type=Path,
        help=(
            "Optional N1 checkpoint. When set, the read-only contextual "
            "inference node runs alongside the managed recording."
        ),
    )
    return parser.parse_args()


def parse_episode(specification: str) -> Episode:
    """Parse one ``PROFILE:SEED:EXCITATION[:REPEATS]`` specification."""
    parts = specification.split(":")
    if len(parts) not in (3, 4):
        raise ValueError(f"Invalid episode specification: {specification}")
    profile, seed_text, excitation = parts[0], parts[1], parts[2]
    records = int(parts[3]) if len(parts) == 4 else 1
    if records < 1:
        raise ValueError(f"Episode repeat count must be >= 1: {specification}")
    return Episode(profile=profile, seed=int(seed_text), excitation=excitation, records=records)


def run_command(command: list[str], **kwargs) -> subprocess.CompletedProcess:
    """Run a command from the repository root and return the result."""
    return subprocess.run(command, cwd=REPO_ROOT, check=False, **kwargs)


def publisher_count() -> int:
    """Return the current odometry publisher count (0 when unknown/absent)."""
    result = run_command(
        ["ros2", "topic", "info", ODOMETRY_TOPIC],
        capture_output=True,
        text=True,
        timeout=15,
    )
    match = PUBLISHER_PATTERN.search(result.stdout + result.stderr)
    return int(match.group(1)) if match else 0


def wait_for_publisher_count(target: int, timeout_s: float) -> bool:
    """Poll the odometry topic until the publisher count reaches ``target``."""
    deadline = time.monotonic() + timeout_s
    last = None
    while time.monotonic() < deadline:
        last = publisher_count()
        if last == target:
            return True
        time.sleep(1.0)
    print(f"  publisher count is {last}, expected {target}", flush=True)
    return False


def matching_sim_pids(token: str) -> list[int]:
    """Return PIDs whose command line references a unique scenario token."""
    pids = []
    for proc_path in Path("/proc").glob("[0-9]*"):
        try:
            command_line = (proc_path / "cmdline").read_bytes().replace(b"\x00", b" ")
        except (OSError, PermissionError):
            continue
        if token.encode() in command_line:
            pids.append(int(proc_path.name))
    return pids


def stop_process_group(process: subprocess.Popen, token: str, timeout_s: float) -> None:
    """Signal the spawned launch and any leftover children, then wait for DDS."""
    if process.poll() is None:
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGINT)
        except (ProcessLookupError, PermissionError):
            pass

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        leftovers = [pid for pid in matching_sim_pids(token) if pid != os.getpid()]
        if process.poll() is not None and not leftovers:
            break
        for pid in leftovers:
            try:
                os.kill(pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass
        time.sleep(1.0)

    if process.poll() is None:
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        process.wait(timeout=10)

    for pid in matching_sim_pids(token):
        if pid == os.getpid():
            continue
        try:
            os.kill(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        time.sleep(0.2)

    # DDS discovery can briefly retain the removed endpoint; wait it out.
    if not wait_for_publisher_count(0, timeout_s):
        raise RuntimeError("Odometry publisher did not disappear after teardown.")


def newest_run_id(output_dir: Path, since_mtime: float) -> str:
    """Return the newest CSV base name changed after ``since_mtime``."""
    candidates = [
        path
        for path in output_dir.glob("run_*.csv")
        if path.stat().st_mtime >= since_mtime - 1.0
    ]
    if not candidates:
        raise RuntimeError("No new raw CSV was produced by the recording.")
    newest = max(candidates, key=lambda path: path.stat().st_mtime)
    return newest.stem


def pipeline_commands(run_id: str, output_dir: Path, world_dir: Path) -> list[list[str]]:
    """Return the mandatory post-recording pipeline commands."""
    csv_path = output_dir / f"{run_id}.csv"
    return [
        [
            sys.executable,
            "experiments/dataset_inspector.py",
            str(csv_path),
            "--output-dir",
            f"runs/inspection/{run_id}",
        ],
        [
            sys.executable,
            "experiments/prepare_transitions.py",
            str(csv_path),
            "--require-scenario-manifest",
        ],
        [
            sys.executable,
            "experiments/build_dataset_manifest.py",
            "--previous-manifest",
            "data/processed/dataset_manifest.json",
        ],
        [
            sys.executable,
            "experiments/train_linear_baseline.py",
            "--dataset-manifest",
            "data/processed/dataset_manifest.json",
            "--output-dir",
            "runs/models/scenario_ridge_latest",
        ],
    ]


def generate_scenario(episode: Episode, world_dir: Path) -> Path:
    """Generate the seeded scenario and return the manifest path."""
    result = run_command(
        [
            "ros2",
            "run",
            "neurogrip_sim",
            "scenario_builder",
            episode.profile,
            "--seed",
            str(episode.seed),
            "--output-dir",
            str(world_dir),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode != 0:
        raise RuntimeError(f"scenario_builder failed: {result.stderr.strip()}")
    manifest_path = world_dir / f"{episode.scenario_id}.manifest.yaml"
    world_path = world_dir / f"{episode.scenario_id}.sdf"
    if not manifest_path.is_file() or not world_path.is_file():
        raise RuntimeError(f"scenario files missing for {episode.scenario_id}")
    return manifest_path


def record_episode(episode: Episode, arguments: argparse.Namespace) -> str:
    """Run one full simulator recording and return the new run id."""
    output_dir = arguments.output_dir.expanduser().resolve()
    world_dir = arguments.world_dir.expanduser().resolve()

    if arguments.no_regenerate:
        manifest_path = world_dir / f"{episode.scenario_id}.manifest.yaml"
    else:
        manifest_path = generate_scenario(episode, world_dir)
    world_path = world_dir / f"{episode.scenario_id}.sdf"

    episode_started_at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    sim_log = (
        REPO_ROOT
        / "runs"
        / "sim_logs"
        / f"{episode.scenario_id}_{episode.excitation}_{episode_started_at}.log"
    )
    sim_log.parent.mkdir(parents=True, exist_ok=True)
    sim_log_file = sim_log.open("w")
    launch_command = [
        "ros2",
        "launch",
        "neurogrip_bringup",
        "vehicle_sim.launch.py",
        f"world_path:={world_path}",
        f"headless:={'true' if arguments.headless else 'false'}",
        f"scenario_manifest_path:={manifest_path}",
    ]
    sim_process = subprocess.Popen(
        launch_command,
        cwd=REPO_ROOT,
        stdout=sim_log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )

    token = world_path.name
    recording_started = time.time()
    try:
        if not wait_for_publisher_count(1, arguments.sim_timeout_s):
            raise RuntimeError(f"Simulator did not publish odometry (log: {sim_log}).")

        recording_log = (
            REPO_ROOT
            / "runs"
            / "recording_logs"
            / f"{episode.scenario_id}_{episode.excitation}_{episode_started_at}.log"
        )
        recording_log.parent.mkdir(parents=True, exist_ok=True)
        recording_command = [
            "ros2",
            "launch",
            "neurogrip_bringup",
            "excitation_recording.launch.py",
            f"scenario_manifest_path:={manifest_path}",
            f"output_dir:={output_dir}",
            f"excitation_profile:={episode.excitation}",
        ]
        if arguments.ai_checkpoint:
            recording_command.extend(
                [
                    f"ai_checkpoint_path:={arguments.ai_checkpoint.expanduser().resolve()}",
                    f"ai_model_root:={REPO_ROOT / 'learning'}",
                ]
            )
        recording = subprocess.Popen(
            recording_command,
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        timed_out = False
        try:
            output, _ = recording.communicate(
                timeout=arguments.recording_timeout_s + 30.0
            )
        except subprocess.TimeoutExpired:
            timed_out = True
            output = "recording launch timed out"
            try:
                os.killpg(os.getpgid(recording.pid), signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass
            try:
                output, _ = recording.communicate(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(recording.pid), signal.SIGKILL)
                output, _ = recording.communicate(timeout=10)
        recording_log.write_text(output or "", encoding="utf-8")
        if timed_out or recording.returncode != 0:
            print((output or "")[-2000:], file=sys.stderr, flush=True)
            raise RuntimeError(
                "Excitation recording launch "
                + ("timed out." if timed_out else "failed.")
            )
    finally:
        stop_process_group(sim_process, token, arguments.teardown_timeout_s)
        sim_log_file.close()

    return newest_run_id(output_dir, recording_started)


def metadata_termination_reason(run_id: str, output_dir: Path) -> str:
    """Read the termination reason from a run's metadata file."""
    metadata_path = output_dir / f"{run_id}.metadata.json"
    document = json.loads(metadata_path.read_text(encoding="utf-8"))
    return str(document.get("termination_reason", "unknown"))


def main() -> int:
    """Execute the planned episodes and their data-pipeline steps."""
    arguments = parse_arguments()
    episodes = [parse_episode(spec) for spec in arguments.episode]
    output_dir = arguments.output_dir.expanduser().resolve()

    successes = 0
    failures = []
    for episode in episodes:
        for repeat_index in range(episode.records):
            label = episode.label
            if episode.records > 1:
                label += f"[{repeat_index + 1}/{episode.records}]"
            print(f"\n=== Recording {label} ===", flush=True)
            try:
                run_id = record_episode(episode, arguments)
            except Exception as error:  # noqa: BLE001 - report and continue
                print(f"RECORDING FAILED for {label}: {error}", file=sys.stderr, flush=True)
                failures.append((label, str(error)))
                continue

            reason = metadata_termination_reason(run_id, output_dir)
            print(f"  run_id={run_id} termination={reason}", flush=True)
            if reason != "excitation_complete":
                failures.append((label, f"unexpected termination: {reason}"))
                continue

            if arguments.skip_pipeline:
                successes += 1
                continue

            print(f"  running data pipeline for {run_id}", flush=True)
            pipeline_ok = True
            for command in pipeline_commands(run_id, output_dir, arguments.world_dir):
                result = run_command(command, capture_output=True, text=True, timeout=300)
                if result.returncode != 0:
                    print(result.stdout[-1500:], flush=True)
                    print(result.stderr[-1500:], file=sys.stderr, flush=True)
                    failures.append((label, f"pipeline step failed: {' '.join(command[:2])}"))
                    pipeline_ok = False
                    break
            if pipeline_ok:
                successes += 1

    print(f"\n=== Batch summary: {successes} successful, {len(failures)} failed ===")
    for label, reason in failures:
        print(f"FAILED {label}: {reason}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
