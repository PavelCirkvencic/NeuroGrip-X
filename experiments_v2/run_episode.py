#!/usr/bin/env python3
"""The sole process authority for traceable NeuroGrip-X v2 episodes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

from scenario_builder_v2 import (
    REPO_ROOT,
    build_core_config,
    load_and_validate_scenario,
)

# EUFS Sim 2 currently constructs std::string directly from getenv(). Keep
# this runner self-contained; without EUFS_MASTER the C++ backend aborts before
# publishing any ROS topic.
os.environ.setdefault("EUFS_MASTER", str(REPO_ROOT / "external/eufs_ws"))
# Every process in this benchmark runs on one workstation. Local discovery
# prevents unrelated LAN DDS traffic from causing deserialisation/watchdog
# faults. MATLAB is isolated behind a localhost TCP transport, while all ROS
# graph participants remain in the ROS 2 Humble environment.
os.environ["ROS_LOCALHOST_ONLY"] = "0"
os.environ["ROS_AUTOMATIC_DISCOVERY_RANGE"] = "LOCALHOST"

VALID_CONTROLLERS = {"EXCITATION", "C0_FIXED", "C1_ORACLE_MU", "C2_NEUROGRIP"}

TRACKS = {
    "small_track": {
        "csv": REPO_ROOT
        / "external/eufs_ws/src/map_lib/maps/tracks/small_track.csv",
        "reference": REPO_ROOT / "artifacts/tracks/small_track_reference.npz",
    },
    "trackdrive": {
        "csv": REPO_ROOT
        / "external/eufs_ws/src/map_lib/maps/competitions/FSUK/2023/trackdrive.csv",
        "reference": REPO_ROOT / "artifacts/tracks/trackdrive_reference.npz",
    },
}


def resolve_track(track_name: str) -> tuple[Path, Path]:
    """Return the matching EUFS CSV and reference artifact for one scenario."""
    if track_name not in TRACKS:
        raise ValueError(f"unsupported scenario track: {track_name}")
    csv_path = TRACKS[track_name]["csv"].resolve()
    reference_path = TRACKS[track_name]["reference"].resolve()
    if not csv_path.is_file() or not reference_path.is_file():
        raise FileNotFoundError(
            f"track {track_name} requires {csv_path} and {reference_path}"
        )
    return csv_path, reference_path


def parse_arguments() -> argparse.Namespace:
    """Parse a deliberately small, stable runner CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", type=Path, required=True)
    parser.add_argument("--controller", choices=sorted(VALID_CONTROLLERS), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--run-root", type=Path, default=Path("runs/eufs_v1/development"))
    parser.add_argument("--target-speed-mps", type=float, default=3.0)
    parser.add_argument("--target-speed-min-mps", type=float, default=1.2)
    parser.add_argument("--fixed-mu", type=float, default=1.0)
    parser.add_argument("--real-time-timeout-s", type=float, default=180.0)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--c2-ensemble-manifest", type=Path)
    parser.add_argument("--c2-model-blend", type=float, default=0.02)
    parser.add_argument("--c2-adaptive-speed-max-mps", type=float)
    parser.add_argument("--scalar-mu-utilisation", type=float, default=0.78)
    parser.add_argument("--neurogrip-utilisation", type=float, default=0.95)
    parser.add_argument("--grip-error-margin-scale", type=float, default=0.50)
    parser.add_argument("--startup-speed-cap-mps", type=float, default=8.0)
    parser.add_argument("--startup-speed-cap-end-progress", type=float, default=0.12)
    parser.add_argument(
        "--actuation-source", choices=("python", "matlab"), default="python"
    )
    parser.add_argument(
        "--matlab-executable",
        type=Path,
        default=Path("/opt/MATLAB/R2026a/bin/matlab"),
    )
    parser.add_argument("--matlab-tcp-port", type=int, default=55980)
    parser.add_argument(
        "--excitation-profile",
        choices=("system_id_v2", "high_speed_v3"),
        default="system_id_v2",
    )
    return parser.parse_args()


def now_utc() -> str:
    """Return a filename-safe UTC timestamp."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sha256_file(path: Path) -> str:
    """Return the byte identity of one generated or source configuration."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_seeded_plugin_config(seed: int, output_path: Path) -> Path:
    """Create deterministic paired EUFS sensor noise from the scenario seed."""
    source = REPO_ROOT / "src/neurogrip_bringup/config/eufs_plugin_params.yaml"
    document = yaml.safe_load(source.read_text(encoding="utf-8"))
    plugins = document["eufs_sim2"]["ros__parameters"]["plugin"]
    plugins["wheel_speed_plugin"]["noise_seed"] = int(seed)
    plugins["imu_plugin"]["noise_seed"] = int(seed + 100_000)
    output_path.write_text(
        yaml.safe_dump(document, sort_keys=False), encoding="utf-8"
    )
    return output_path


def command_process(
    command: list[str],
    log_path: Path,
    *,
    environment: dict[str, str] | None = None,
) -> subprocess.Popen:
    """Start one owned process group and retain its combined log."""
    handle = log_path.open("x", encoding="utf-8")
    return subprocess.Popen(
        command,
        cwd=REPO_ROOT,
        stdout=handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        text=True,
        env=environment,
    )


def matlab_process_environment() -> dict[str, str]:
    """Keep MATLAB independent from every sourced ROS 2 library and setting."""
    environment = os.environ.copy()
    for variable in (
        "AMENT_PREFIX_PATH",
        "COLCON_PREFIX_PATH",
        "CMAKE_PREFIX_PATH",
        "LD_LIBRARY_PATH",
        "PYTHONPATH",
        "ROS_DISTRO",
        "ROS_VERSION",
        "ROS_PYTHON_VERSION",
        "RMW_IMPLEMENTATION",
        "ROS_LOCALHOST_ONLY",
        "ROS_AUTOMATIC_DISCOVERY_RANGE",
    ):
        environment.pop(variable, None)
    return environment


def stop_process(process: subprocess.Popen | None, timeout_s: float = 8.0) -> None:
    """Interrupt only a process group created by this runner, then reap it."""
    if process is None:
        return
    if process.poll() is None:
        try:
            # Let launch propagate one clean Ctrl-C before group enforcement.
            process.send_signal(signal.SIGINT)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            pass

    # A launch parent can exit while a child (notably set_track) remains in the
    # session.  Always enforce cleanup on the exact process group created by
    # command_process, even when the leader has already been reaped.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.1)
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    if process.poll() is None:
        process.wait(timeout=3.0)


def ros_topic_exists(topic: str) -> bool:
    """Check discovery without depending on the long-lived ROS CLI daemon."""
    result = subprocess.run(
        ["ros2", "topic", "list", "--no-daemon", "--spin-time", "0.1"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return topic in result.stdout.splitlines()


def ros_topics_exist(topics: set[str], spin_time_s: float) -> bool:
    """Check several local ROS graph endpoints with discovery spin time."""
    result = subprocess.run(
        [
            "ros2",
            "topic",
            "list",
            "--no-daemon",
            "--spin-time",
            str(spin_time_s),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return topics <= set(result.stdout.splitlines())


def ros_node_exists(node_name: str) -> bool:
    """Return whether a launched node has joined the ROS graph directly."""
    result = subprocess.run(
        ["ros2", "node", "list", "--no-daemon", "--spin-time", "0.1"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return node_name in result.stdout.splitlines()


def wait_until(predicate, timeout_s: float, description: str) -> None:
    """Wait for a clear readiness condition or fail with the condition name."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.25)
    raise RuntimeError(f"timeout waiting for {description}")


def event_seen(path: Path, predicate) -> bool:
    """Inspect valid JSONL events written so far without keeping a file cursor."""
    if not path.is_file():
        return False
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if predicate(event):
            return True
    return False


def merge_final_metadata(path: Path, additions: dict) -> None:
    """Append runner-owned termination truth after recorder closes its hashes."""
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata["runner"] = additions
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def matlab_string(value: Path) -> str:
    """Quote one absolute path for a MATLAB single-quoted string."""
    return str(value.resolve()).replace("'", "''")


def run(arguments: argparse.Namespace) -> int:
    """Run one valid scenario to lap completion, controlled failure, or timeout."""
    scenario_path = arguments.scenario.expanduser().resolve()
    scenario = load_and_validate_scenario(scenario_path)
    track_csv, track_npz = resolve_track(str(scenario["track"]))
    if arguments.seed != scenario["seed"]:
        raise ValueError("--seed must equal scenario.seed; do not silently reseed a scenario")
    if len(scenario["grip_schedule"]) > 2:
        raise ValueError("current scenario manager supports exactly one optional transition")
    if arguments.controller == "C2_NEUROGRIP":
        if arguments.c2_ensemble_manifest is None:
            raise ValueError("C2_NEUROGRIP requires --c2-ensemble-manifest")
        if not arguments.c2_ensemble_manifest.expanduser().is_file():
            raise FileNotFoundError("C2 ensemble manifest does not exist")
        if not 0.0 <= arguments.c2_model_blend <= 1.0:
            raise ValueError("--c2-model-blend must be in [0, 1]")
        if (
            arguments.c2_adaptive_speed_max_mps is not None
            and arguments.c2_adaptive_speed_max_mps < arguments.target_speed_mps
        ):
            raise ValueError(
                "--c2-adaptive-speed-max-mps cannot be below the fixed baseline speed"
            )
    if (
        arguments.actuation_source == "matlab"
        and arguments.controller != "C2_NEUROGRIP"
    ):
        raise ValueError("MATLAB actuation is supported only for C2_NEUROGRIP")
    if not 1 <= arguments.matlab_tcp_port <= 65535:
        raise ValueError("--matlab-tcp-port must be in [1, 65535]")
    if not (
        0.0 < arguments.scalar_mu_utilisation <= 1.0
        and 0.0 < arguments.neurogrip_utilisation <= 1.0
        and 0.0 <= arguments.grip_error_margin_scale <= 1.0
        and 0.0 < arguments.target_speed_min_mps <= arguments.target_speed_mps
        and arguments.target_speed_min_mps
        <= arguments.startup_speed_cap_mps
        <= arguments.target_speed_mps
        and 0.0 <= arguments.startup_speed_cap_end_progress <= 0.25
        and 0.35 <= arguments.fixed_mu <= 1.30
    ):
        raise ValueError("invalid physical speed-profile arguments")

    run_id = f"{scenario['scenario_id']}__{arguments.controller}__seed{arguments.seed:04d}"
    run_root = arguments.run_root.expanduser().resolve()
    run_dir = run_root / run_id
    if run_dir.exists():
        if not arguments.overwrite:
            raise FileExistsError(f"run exists; use --overwrite to archive it first: {run_dir}")
        archive = run_root / "archive" / f"{run_id}__{now_utc()}"
        archive.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(run_dir), str(archive))

    provenance = build_core_config(scenario, REPO_ROOT / "external/eufs_ws/src/vehicle_models/config/DynamicBicycle/ads-dv-calculated.yaml", run_dir / "generated_core.yaml")
    plugin_config = build_seeded_plugin_config(
        arguments.seed, run_dir / "generated_plugin.yaml"
    )
    provenance.update(
        {
            "run_id": run_id,
            "controller": arguments.controller,
            "target_speed_mps": arguments.target_speed_mps,
            "target_speed_min_mps": arguments.target_speed_min_mps,
            "fixed_mu": arguments.fixed_mu,
            "scalar_mu_utilisation": arguments.scalar_mu_utilisation,
            "neurogrip_utilisation": arguments.neurogrip_utilisation,
            "grip_error_margin_scale": arguments.grip_error_margin_scale,
            "startup_speed_cap_mps": arguments.startup_speed_cap_mps,
            "startup_speed_cap_end_progress": (
                arguments.startup_speed_cap_end_progress
            ),
            "actuation_source": arguments.actuation_source,
            "matlab_transport": (
                {
                    "kind": "localhost_tcp_ros_bridge",
                    "host": "127.0.0.1",
                    "port": arguments.matlab_tcp_port,
                }
                if arguments.actuation_source == "matlab"
                else None
            ),
            "c2_model_blend": (
                arguments.c2_model_blend
                if arguments.controller == "C2_NEUROGRIP"
                else None
            ),
            "c2_adaptive_speed_max_mps": (
                arguments.c2_adaptive_speed_max_mps
                if arguments.controller == "C2_NEUROGRIP"
                else None
            ),
            "excitation_profile": (
                arguments.excitation_profile
                if arguments.controller == "EXCITATION"
                else None
            ),
            "scenario_path": str(scenario_path),
            "scenario": scenario,
            "track_csv": str(track_csv),
            "track_npz": str(track_npz),
            "plugin_config": str(plugin_config),
            "plugin_config_sha256": sha256_file(plugin_config),
            "sensor_noise_seeds": {
                "wheel_speed": arguments.seed,
                "imu": arguments.seed + 100_000,
            },
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
        }
    )
    provenance_path = run_dir / "provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")

    schedule = scenario["grip_schedule"]
    initial, transition = schedule[0], schedule[1] if len(schedule) == 2 else None
    processes: list[subprocess.Popen] = []
    eufs = scenario_manager = controller = recorder = None
    matlab_candidate = mux_recorder = None
    matlab_ready_path = run_dir / "matlab_tcp.ready"
    event_path = run_dir / "episode.events.jsonl"
    termination = "runner_exception"
    try:
        if arguments.actuation_source == "matlab":
            matlab_executable = arguments.matlab_executable.expanduser().resolve()
            if not matlab_executable.is_file():
                raise FileNotFoundError(
                    f"MATLAB executable does not exist: {matlab_executable}"
                )
            maximum_matlab_duration_s = arguments.real_time_timeout_s + 60.0
            matlab_code = (
                f"cd('{matlab_string(REPO_ROOT)}'); "
                "addpath(fullfile(pwd,'matlab')); "
                "run_live_matlab_tcp_candidate("
                f"{arguments.matlab_tcp_port},{maximum_matlab_duration_s:.9g},100,"
                f"'{matlab_string(run_dir / 'matlab_candidate.csv')}',"
                f"'{matlab_string(matlab_ready_path)}',20);"
            )
            matlab_candidate = command_process(
                [str(matlab_executable), "-batch", matlab_code],
                run_dir / "matlab_candidate.log",
                environment=matlab_process_environment(),
            )
            processes.append(matlab_candidate)
            mux_recorder = command_process(
                [
                    sys.executable,
                    "experiments_v2/record_command_mux_status.py",
                    "--output",
                    str(run_dir / "command_mux_status.csv"),
                    "--ros-args",
                    "-p",
                    "use_sim_time:=true",
                ],
                run_dir / "command_mux_recorder.log",
            )
            processes.append(mux_recorder)
            wait_until(
                matlab_ready_path.is_file,
                60.0,
                "MATLAB localhost TCP server",
            )
        eufs = command_process(
            [
                "ros2", "launch", "neurogrip_bringup", "eufs_backend.launch.py",
                "headless:=true", f"core_config:={run_dir / 'generated_core.yaml'}",
                f"plugin_config:={plugin_config}", f"seed:={arguments.seed}",
                f"track:={track_csv}",
            ],
            run_dir / "eufs.log",
        )
        processes.append(eufs)
        wait_until(
            lambda: ros_topic_exists("/odom") and ros_topic_exists("/ros_can/wheel_speeds"),
            60.0,
            "EUFS /odom and /ros_can/wheel_speeds",
        )
        recorder = command_process(
            [
                sys.executable, "experiments_v2/record_episode.py", "--output",
                str(run_dir / "episode.csv"), "--scenario-id", scenario["scenario_id"],
                "--controller", arguments.controller, "--seed", str(arguments.seed),
                "--provenance-file", str(provenance_path),
                "--track-npz", str(track_npz), "--ros-args", "-p",
                "use_sim_time:=true",
            ],
            run_dir / "recorder.log",
        )
        processes.append(recorder)
        wait_until(
            lambda: ros_node_exists("/neurogrip_episode_recorder"),
            15.0,
            "episode recorder node",
        )
        scenario_command = [
            "ros2", "run", "neurogrip_control", "scenario_manager", "--ros-args",
            "-p", "use_sim_time:=true", "-p", f"front_grip_scale:={initial['front_scale']}", "-p",
            f"rear_grip_scale:={initial['rear_scale']}", "-p",
            f"transition_start_s:={transition['start_s'] if transition else -1.0}", "-p",
            f"transition_start_progress:={transition.get('start_progress', -1.0) if transition else -1.0}", "-p",
            f"transition_front_grip_scale:={transition['front_scale'] if transition else 1.0}",
            "-p", f"transition_rear_grip_scale:={transition['rear_scale'] if transition else 1.0}",
        ]
        scenario_manager = command_process(scenario_command, run_dir / "scenario.log")
        processes.append(scenario_manager)
        wait_until(
            lambda: event_seen(
                event_path,
                lambda event: event.get("label") == "initial" and event.get("confirmed") is True,
            ),
            20.0,
            "confirmed initial grip readback",
        )
        shared_launch = [
            "ros2", "launch", "neurogrip_bringup",
            f"track_npz:={track_npz}",
            f"steering_delay_s:={scenario['actuator']['steering_delay_s']}",
            f"steering_gain:={scenario['actuator']['steering_gain']}",
        ]
        if arguments.controller == "EXCITATION":
            launch_command = [*shared_launch[:3], "eufs_excitation.launch.py", *shared_launch[3:]]
            launch_command.append(
                f"excitation_profile:={arguments.excitation_profile}"
            )
        else:
            launch_command = [
                *shared_launch[:3], "closed_loop.launch.py", *shared_launch[3:],
                f"controller:={arguments.controller}",
                f"target_speed:={arguments.target_speed_mps}",
                f"target_speed_min:={arguments.target_speed_min_mps}",
                f"fixed_mu:={arguments.fixed_mu}",
                f"nominal_fit:={REPO_ROOT / 'runs/eufs_v1/models/nominal_fit.json'}",
                f"scalar_mu_utilisation:={arguments.scalar_mu_utilisation}",
                f"neurogrip_utilisation:={arguments.neurogrip_utilisation}",
                f"grip_error_margin_scale:={arguments.grip_error_margin_scale}",
                f"startup_speed_cap_mps:={arguments.startup_speed_cap_mps}",
                "startup_speed_cap_end_progress:="
                + str(arguments.startup_speed_cap_end_progress),
                f"actuation_source:={arguments.actuation_source}",
                f"matlab_tcp_port:={arguments.matlab_tcp_port}",
            ]
        if arguments.controller == "C2_NEUROGRIP":
            launch_command.append(
                f"c2_ensemble_manifest:={arguments.c2_ensemble_manifest.expanduser().resolve()}"
            )
            launch_command.append(f"c2_model_blend:={arguments.c2_model_blend}")
            launch_command.append(
                "c2_adaptive_speed_max:="
                + str(
                    arguments.target_speed_mps
                    if arguments.c2_adaptive_speed_max_mps is None
                    else arguments.c2_adaptive_speed_max_mps
                )
            )
        controller = command_process(launch_command, run_dir / "controller.log")
        processes.append(controller)
        wait_until(lambda: ros_topic_exists("/cmd"), 20.0, "single-authority /cmd")
        completed = False
        deadline = time.monotonic() + arguments.real_time_timeout_s
        while time.monotonic() < deadline:
            completion_event = (
                "excitation_complete" if arguments.controller == "EXCITATION" else "lap_complete"
            )
            if event_seen(event_path, lambda event: event.get("event") == completion_event):
                completed, termination = True, completion_event
                break
            terminal_failure = next(
                (
                    name
                    for name in ("safety_violation", "controller_failure")
                    if event_seen(
                        event_path, lambda event, target=name: event.get("event") == target
                    )
                ),
                None,
            )
            if terminal_failure is not None:
                termination = str(terminal_failure)
                break
            monitored = [eufs, scenario_manager, controller, recorder]
            if matlab_candidate is not None:
                monitored.extend([matlab_candidate, mux_recorder])
            if any(
                process is not None and process.poll() not in (None, 0)
                for process in monitored
            ):
                termination = "process_failure"
                break
            time.sleep(0.25)
        if not completed and termination == "runner_exception":
            termination = "timeout"
        if completed:
            time.sleep(0.3)
    finally:
        stop_process(controller)
        if matlab_candidate is not None and matlab_candidate.poll() is None:
            try:
                matlab_candidate.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                pass
        for process in (
            matlab_candidate,
            mux_recorder,
            scenario_manager,
            recorder,
            eufs,
        ):
            stop_process(process)
        matlab_ready_path.unlink(missing_ok=True)
    metadata_path = run_dir / "episode.metadata.json"
    if not metadata_path.is_file():
        raise RuntimeError("recorder did not produce final metadata")
    merge_final_metadata(
        metadata_path,
        {
            "termination_reason": termination,
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "real_time_timeout_s": arguments.real_time_timeout_s,
        },
    )
    if termination == "process_failure":
        raise RuntimeError("one owned ROS process failed; inspect run logs")
    print(f"RUN_COMPLETE run_id={run_id} termination={termination}")
    return 0


def main() -> int:
    """Entrypoint with a nonzero exit for invalid technical execution."""
    try:
        return run(parse_arguments())
    except Exception as error:  # noqa: BLE001 - CLI reports a single useful failure
        print(f"RUN_FAILED: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
