#!/usr/bin/env python3
"""
Preflight dependency verification for NeuroGrip-X.

Checks the Python, ROS 2, Gazebo and MATLAB dependencies used by the project.
Exits non-zero when a required dependency is missing.

Usage:
    python scripts/preflight_check.py [--allow-missing-matlab] [--json PATH]
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MATLAB_ROOT = Path("/opt/MATLAB/R2026a")
NEUROGRIP_PACKAGES = (
    "neurogrip_sim",
    "neurogrip_control",
    "neurogrip_bringup",
    "neurogrip_ai",
    "neurogrip_interfaces",
)


@dataclass
class Check:
    """One named verification result."""

    name: str
    ok: bool
    required: bool
    detail: str


def run(command: list[str], timeout: float = 60.0) -> tuple[int, str]:
    """Run a command and return (returncode, stdout+stderr)."""
    try:
        completed = subprocess.run(
            command, capture_output=True, text=True, timeout=timeout, check=False
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return 1, str(error)
    return completed.returncode, (completed.stdout + completed.stderr).strip()


def check_python() -> Check:
    """Verify a CPython 3.10 interpreter is running this script."""
    version = sys.version.split()[0]
    ok = sys.version_info[:2] == (3, 10)
    return Check("python_interpreter", ok, True, f"{sys.executable} (Python {version})",)


def check_ros_distro() -> Check:
    """Verify ROS 2 Humble is sourced."""
    distro = os.environ.get("ROS_DISTRO", "")
    return Check("ros_distro", distro == "humble", True, f"ROS_DISTRO={distro or 'unset'}")


def check_ros_python() -> Check:
    """Verify rclpy is importable."""
    try:
        import rclpy  # noqa: F401

        return Check("ros_python_import", True, True, "rclpy importable")
    except ImportError as error:  # pragma: no cover - environment dependent
        return Check("ros_python_import", False, True, str(error))


def check_gazebo() -> Check:
    """Verify Gazebo Fortress (ign gazebo) is installed."""
    executable = shutil.which("ign")
    if executable is None:
        return Check("gazebo_fortress", False, True, "`ign` not found on PATH")
    code, output = run(["ign", "gazebo", "--version"])
    match = re.search(r"version\s+([0-9.]+)", output)
    detail = output.splitlines()[0] if output else "no version output"
    ok = code == 0 and "6." in (match.group(1) if match else "")
    return Check("gazebo_fortress", ok, True, detail)


def check_module(module: str, required: bool = True) -> Check:
    """Verify a Python module imports and report its version when available."""
    try:
        imported = importlib.import_module(module)
        version = getattr(imported, "__version__", "unknown")
        return Check(f"python_module_{module}", True, required, f"{module} {version}")
    except ImportError as error:  # pragma: no cover - environment dependent
        return Check(f"python_module_{module}", False, required, str(error))


def check_torch_cuda() -> Check:
    """Verify PyTorch imports and report CUDA availability."""
    try:
        import torch
    except ImportError as error:  # pragma: no cover
        return Check("torch_cuda", False, True, str(error))
    cuda = torch.cuda.is_available()
    detail = f"torch {torch.__version__}, cuda={cuda}"
    if cuda:
        detail += f", device={torch.cuda.get_device_name(0)}"
    return Check("torch_cuda", True, True, detail)


def check_ros_packages() -> list[Check]:
    """Verify every NeuroGrip ROS package is discoverable."""
    code, output = run(["ros2", "pkg", "list"])
    available = set(output.split()) if code == 0 else set()
    checks = []
    for package in NEUROGRIP_PACKAGES:
        checks.append(
            Check(
                f"ros_package_{package}",
                package in available,
                True,
                "found" if package in available else "missing",
            )
        )
    return checks


def matlab_root() -> Path:
    """Return the configured MATLAB installation root."""
    return Path(os.environ.get("MATLAB_ROOT", DEFAULT_MATLAB_ROOT))


def check_matlab(required: bool) -> list[Check]:
    """Verify the MATLAB executable and the toolboxes used by phase 5."""
    root = matlab_root()
    executable = root / "bin" / "matlab"
    checks = [
        Check(
            "matlab_executable",
            executable.is_file() and os.access(executable, os.X_OK),
            required,
            str(executable),
        )
    ]
    toolbox_paths = {
        "simulink": root / "toolbox" / "simulink",
        "ros_toolbox": root / "toolbox" / "ros",
        "mpc_toolbox": root / "toolbox" / "mpc",
    }
    for name, path in toolbox_paths.items():
        checks.append(Check(name, path.is_dir(), required, str(path)))

    adaptive = list((root / "toolbox" / "mpc").glob("mpc/@mpc/mpcmoveAdaptive.m"))
    checks.append(
        Check(
            "mpcmoveAdaptive",
            bool(adaptive),
            required,
            str(adaptive[0]) if adaptive else "mpcmoveAdaptive.m not found",
        )
    )
    return checks


def parse_arguments() -> argparse.Namespace:
    """Parse preflight options."""
    parser = argparse.ArgumentParser(description="Verify NeuroGrip-X dependencies.")
    parser.add_argument(
        "--allow-missing-matlab",
        action="store_true",
        help="Treat MATLAB and toolbox checks as advisory (for CI without MATLAB).",
    )
    parser.add_argument("--json", type=Path, help="Optional JSON report path.")
    return parser.parse_args()


def main() -> int:
    """Run every check and print a report."""
    arguments = parse_arguments()
    checks = [
        check_python(),
        check_ros_distro(),
        check_ros_python(),
        check_gazebo(),
        check_module("numpy"),
        check_module("pandas"),
        check_module("sklearn"),
        check_module("joblib"),
        check_module("pyarrow"),
        check_torch_cuda(),
    ]
    checks.extend(check_ros_packages())
    checks.extend(check_matlab(required=not arguments.allow_missing_matlab))

    failures = [check for check in checks if check.required and not check.ok]
    for check in checks:
        status = "PASS" if check.ok else ("FAIL" if check.required else "WARN")
        print(f"[{status}] {check.name}: {check.detail}")

    report = {
        "passed": not failures,
        "checks": [check.__dict__ for check in checks],
    }
    if arguments.json:
        arguments.json.parent.mkdir(parents=True, exist_ok=True)
        arguments.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print()
    if failures:
        print(f"Preflight FAILED: {len(failures)} required dependency check(s) missing.")
        return 1
    print(f"Preflight passed: {len(checks)} checks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
