"""
ROS-free grip-transition controller with validation, retries and provenance.

The controller turns a scenario's hidden ``grip_transition`` block into a set of
per-wheel wheel-slip service requests.  It only reports success when Gazebo
confirms a Boolean ``true`` for every wheel, retries bounded by
``max_attempts``, and records the application time per wheel plus the maximum
inter-wheel skew.  The grip values themselves never leave this module as ROS
state; only pass/fail and timing provenance are published.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

WHEEL_AXLES = {
    "front_left_wheel": "front",
    "front_right_wheel": "front",
    "rear_left_wheel": "rear",
    "rear_right_wheel": "rear",
}

BOOLEAN_RESPONSE_PATTERN = re.compile(r"data:\s*(true|false)", re.IGNORECASE)


def parse_boolean_response(returncode: int, output: str) -> bool:
    """Interpret a Gazebo Boolean service response safely."""
    if returncode != 0:
        return False
    match = BOOLEAN_RESPONSE_PATTERN.search(output or "")
    return bool(match) and match.group(1).lower() == "true"


def transition_due_time_s(excitation_start_time_s: float, transition: dict) -> float:
    """
    Return the absolute simulation time at which a transition is due.

    ``start_s`` is defined relative to the start of the excitation protocol,
    not to Gazebo startup, so the same scenario schedules identically no matter
    how long the simulator took to come up.
    """
    return float(excitation_start_time_s) + float(transition["start_s"])


def is_transition_due(
    now_s: float, excitation_start_time_s: float | None, transition: dict
) -> bool:
    """Return whether the transition should fire at ``now_s``."""
    if excitation_start_time_s is None:
        return False
    return now_s >= transition_due_time_s(excitation_start_time_s, transition)


def resolve_compliance(
    scenario: dict, transition: dict
) -> dict[str, tuple[float, float]]:
    """Return target (lateral, longitudinal) compliance per wheel link."""
    scale = float(transition["compliance_scale"])
    longitudinal = float(scenario.get("longitudinal_compliance", 0.0)) * scale
    targets = {}
    for wheel_name, axle in WHEEL_AXLES.items():
        lateral = float(scenario.get(f"{axle}_lateral_compliance", 0.0)) * scale
        targets[wheel_name] = (lateral, longitudinal)
    return targets


@dataclass
class WheelOutcome:
    """Per-wheel application outcome."""

    attempts: int = 0
    success: bool = False
    applied_at_s: float | None = None


@dataclass
class GripTransitionController:
    """Apply one scheduled grip transition with bounded retries."""

    scenario: dict
    apply_wheel: object  # callable(wheel, lateral, longitudinal) -> bool
    clock: object  # callable() -> float (simulation seconds)
    max_attempts: int = 3
    applied: bool = False
    provenance: dict | None = field(default=None, repr=False)

    @property
    def expected(self) -> bool:
        """Return whether the scenario defines a transition at all."""
        return isinstance(self.scenario.get("grip_transition"), dict)

    def target_compliance(self) -> dict[str, tuple[float, float]]:
        """Return the per-wheel compliance targets for this scenario."""
        return resolve_compliance(self.scenario, self.scenario["grip_transition"])

    def apply(self) -> dict:
        """Apply the transition, retrying failures, and return provenance."""
        if not self.expected:
            self.applied = True
            self.provenance = {"expected": False, "applied": False, "detail": "no transition"}
            return self.provenance
        if self.applied:
            return self.provenance

        targets = self.target_compliance()
        outcomes = {wheel: WheelOutcome() for wheel in targets}
        remaining = list(targets)

        for attempt in range(1, self.max_attempts + 1):
            for wheel in list(remaining):
                outcome = outcomes[wheel]
                outcome.attempts = attempt
                if self.apply_wheel(wheel, *targets[wheel]):
                    outcome.success = True
                    outcome.applied_at_s = float(self.clock())
                    remaining.remove(wheel)
            if not remaining:
                break

        self.applied = not remaining
        applied_times = [
            outcome.applied_at_s
            for outcome in outcomes.values()
            if outcome.applied_at_s is not None
        ]
        self.provenance = {
            "expected": True,
            "applied": self.applied,
            "max_attempts": self.max_attempts,
            "wheels": {
                wheel: {
                    "attempts": outcome.attempts,
                    "success": outcome.success,
                    "applied_at_s": outcome.applied_at_s,
                }
                for wheel, outcome in outcomes.items()
            },
            "max_skew_s": (
                max(applied_times) - min(applied_times) if applied_times else None
            ),
            "detail": "applied" if self.applied else "failed_after_retries",
        }
        return self.provenance
