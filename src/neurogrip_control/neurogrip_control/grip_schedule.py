"""
Apply a scheduled, unlabelled mid-episode grip transition in Gazebo.

Fortress exposes a runtime ``/world/<world>/wheel_slip`` service
(``ignition.msgs.WheelSlipParametersCmd``).  This node reads the sampled
``grip_transition`` block from a scenario manifest and, once simulation time
reaches ``start_s`` **measured from the start of the excitation protocol**,
scales the per-wheel slip compliance through that service.

It validates the Boolean response, retries failures, and publishes only
pass/fail and timing provenance on ``/neurogrip/grip_transition_status``.  The
transition is never published on a ROS topic; a learned model can only observe
it through the resulting change in vehicle dynamics.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import rclpy
import yaml
from rclpy.node import Node
from std_msgs.msg import String

from neurogrip_control.grip_transition import (
    GripTransitionController,
    is_transition_due,
    parse_boolean_response,
    transition_due_time_s,
)


class GripScheduler(Node):
    """Apply one scenario-defined global grip transition at a simulation time."""

    def __init__(self):
        super().__init__("grip_scheduler")

        self.declare_parameter("world_name", "ackermann_steering")
        self.declare_parameter("scenario_manifest_path", "")
        self.declare_parameter("service_timeout_ms", 3000)
        self.declare_parameter("poll_rate_hz", 10.0)
        self.declare_parameter("max_attempts", 3)
        self.declare_parameter("provenance_dir", "runs/grip_transitions")

        self.world_name = self.get_parameter("world_name").value
        self.service_timeout_ms = int(self.get_parameter("service_timeout_ms").value)
        poll_rate_hz = float(self.get_parameter("poll_rate_hz").value)
        max_attempts = int(self.get_parameter("max_attempts").value)
        self.provenance_dir = Path(
            self.get_parameter("provenance_dir").value
        ).expanduser().resolve()
        manifest_path = self.get_parameter("scenario_manifest_path").value

        self.scenario, self.scenario_id = self.load_scenario(manifest_path)
        self.controller = GripTransitionController(
            self.scenario,
            apply_wheel=self.apply_wheel,
            clock=self.now_s,
            max_attempts=max_attempts,
        )
        self.excitation_start_time_s: float | None = None
        self.reported = False

        if self.controller.expected:
            self.transition = self.scenario["grip_transition"]
            self.get_logger().info(
                "Scheduled grip transition: "
                f"start_offset={self.transition['start_s']:.2f}s "
                f"compliance_scale={self.transition['compliance_scale']:.3f} "
                f"(max_attempts={max_attempts})"
            )
        else:
            self.transition = None
            self.get_logger().info(
                "No grip_transition in scenario; grip scheduler is a no-op."
            )

        self.status_publisher = self.create_publisher(
            String, "/neurogrip/grip_transition_status", 10
        )
        self.create_subscription(
            String, "/neurogrip/excitation_phase", self.phase_callback, 10
        )
        self.create_timer(1.0 / poll_rate_hz, self.tick)

    @staticmethod
    def load_scenario(manifest_path: str) -> tuple[dict, str]:
        """Load the ``scenario`` mapping and id from a YAML manifest."""
        if not manifest_path:
            return {}, "unknown"
        path = Path(manifest_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Scenario manifest does not exist: {path}")
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(document, dict) or not isinstance(
            document.get("scenario"), dict
        ):
            raise ValueError(f"Invalid scenario manifest: {path}")
        return document["scenario"], str(document.get("scenario_id", path.stem))

    def now_s(self) -> float:
        """Return the current simulation time in seconds."""
        return self.get_clock().now().nanoseconds / 1_000_000_000.0

    def phase_callback(self, message: String) -> None:
        """Record the excitation start when the first phase label arrives."""
        if self.excitation_start_time_s is None and message.data not in (
            "complete",
            "unlabeled",
        ):
            self.excitation_start_time_s = self.now_s()
            self.get_logger().info(
                f"Excitation start detected at {self.excitation_start_time_s:.3f}s "
                f"(phase '{message.data}')."
            )

    def tick(self) -> None:
        """Apply the transition once simulation time reaches the schedule."""
        if self.reported or not self.controller.expected:
            return
        if self.excitation_start_time_s is None:
            return
        due_s = transition_due_time_s(self.excitation_start_time_s, self.transition)
        if not is_transition_due(self.now_s(), self.excitation_start_time_s, self.transition):
            return

        provenance = self.controller.apply()
        provenance["scenario_id"] = self.scenario_id
        provenance["excitation_start_time_s"] = self.excitation_start_time_s
        provenance["due_time_s"] = due_s
        provenance["reported_at_s"] = self.now_s()
        self.reported = True

        message = String()
        message.data = json.dumps(provenance)
        self.status_publisher.publish(message)
        self.write_sidecar(provenance)

        if provenance["applied"]:
            self.get_logger().info(
                "Grip transition applied to all wheels; "
                f"max skew {provenance['max_skew_s']:.3f}s."
            )
        else:
            self.get_logger().error(
                "Grip transition FAILED after retries; run must not enter the "
                "benchmark catalog."
            )

    def write_sidecar(self, provenance: dict) -> None:
        """Persist transition provenance under a durable, unique path."""
        self.provenance_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        path = self.provenance_dir / f"{self.scenario_id}_{timestamp}.json"
        path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")

    def apply_wheel(self, wheel_name: str, lateral: float, longitudinal: float) -> bool:
        """Call the wheel-slip service for one wheel and validate the response."""
        request = (
            f'entity: {{name: "{wheel_name}"}} '
            f"slip_compliance_lateral: {lateral:.8f} "
            f"slip_compliance_longitudinal: {longitudinal:.8f}"
        )
        command = [
            "ign",
            "service",
            "-s",
            f"/world/{self.world_name}/wheel_slip",
            "--reqtype",
            "ignition.msgs.WheelSlipParametersCmd",
            "--reptype",
            "ignition.msgs.Boolean",
            "--timeout",
            str(self.service_timeout_ms),
            "--req",
            request,
        ]
        try:
            result = subprocess.run(
                command, capture_output=True, text=True, timeout=15, check=False
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            self.get_logger().warning(f"wheel_slip service call error: {error}")
            return False
        success = parse_boolean_response(
            result.returncode, result.stdout + result.stderr
        )
        if not success:
            self.get_logger().warning(
                f"wheel_slip service returned no confirmation for {wheel_name}."
            )
        return success


def main(args=None):
    """Run the grip scheduler."""
    rclpy.init(args=args)
    node = GripScheduler()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
