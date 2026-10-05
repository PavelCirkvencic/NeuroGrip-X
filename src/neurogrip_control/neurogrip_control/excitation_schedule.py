"""
Deterministic excitation profiles and schedule evaluation (ROS-free).

The schedule is a pure function of elapsed simulation time, so it can be
unit-tested at phase boundaries and late-timer jumps without launching ROS.
Every phase starts at ``phase_elapsed_s == 0``; if the timer is late, the
schedule skips directly to the phase that contains the current elapsed time.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import pi, sin


@dataclass(frozen=True)
class ExcitationPhase:
    """One constant-command or chirp segment in an excitation trajectory."""

    name: str
    duration_s: float
    linear_x_mps: float
    angular_z_rps: float
    waveform: str = "constant"
    chirp_start_hz: float = 0.0
    chirp_end_hz: float = 0.0

    def command_at(self, phase_elapsed_s: float) -> tuple[float, float]:
        """Return the bounded command at elapsed time within this phase."""
        if self.waveform == "constant":
            return self.linear_x_mps, self.angular_z_rps
        if self.waveform == "chirp":
            frequency_slope = (
                self.chirp_end_hz - self.chirp_start_hz
            ) / self.duration_s
            phase_rad = 2.0 * pi * (
                self.chirp_start_hz * phase_elapsed_s
                + 0.5 * frequency_slope * phase_elapsed_s**2
            )
            return self.linear_x_mps, self.angular_z_rps * sin(phase_rad)
        raise ValueError(f"Unknown excitation waveform: {self.waveform}")


EXCITATION_PROFILES = {
    "baseline_v1": (
        ExcitationPhase("settle", 3.0, 0.0, 0.0),
        ExcitationPhase("straight_slow", 5.0, 0.30, 0.0),
        ExcitationPhase("left_constant", 7.0, 0.45, 0.25),
        ExcitationPhase("straight_fast", 5.0, 0.60, 0.0),
        ExcitationPhase("right_constant", 7.0, 0.45, -0.30),
        ExcitationPhase("alternating_left", 2.0, 0.35, 0.35),
        ExcitationPhase("alternating_right", 2.0, 0.35, -0.35),
        ExcitationPhase("alternating_left", 2.0, 0.35, 0.35),
        ExcitationPhase("alternating_right", 2.0, 0.35, -0.35),
        ExcitationPhase("stop", 3.0, 0.0, 0.0),
    ),
    "dynamic_v2": (
        ExcitationPhase("settle", 3.0, 0.0, 0.0),
        ExcitationPhase("straight_mid", 3.0, 0.45, 0.0),
        ExcitationPhase("left_preload", 3.0, 0.60, 0.32),
        ExcitationPhase("right_preload", 3.0, 0.60, -0.32),
        ExcitationPhase(
            "steering_chirp",
            18.0,
            0.65,
            0.42,
            waveform="chirp",
            chirp_start_hz=0.10,
            chirp_end_hz=0.55,
        ),
        ExcitationPhase("coast", 3.0, 0.30, 0.0),
        ExcitationPhase("stop", 3.0, 0.0, 0.0),
    ),
}


@dataclass(frozen=True)
class ScheduleState:
    """Schedule evaluation result for one elapsed-time sample."""

    phase_name: str
    phase_index: int
    phase_elapsed_s: float
    linear_x_mps: float
    angular_z_rps: float
    finished: bool
    entered_new_phase: bool
    completed_now: bool


class ExcitationSchedule:
    """Stateful scheduler that advances deterministically on elapsed time."""

    def __init__(self, phases: tuple[ExcitationPhase, ...]):
        if not phases:
            raise ValueError("Excitation schedule needs at least one phase.")
        self.phases = phases
        offsets = []
        running = 0.0
        for phase in phases:
            offsets.append(running)
            running += phase.duration_s
        self.phase_offsets = offsets
        self.total_duration_s = running
        self.phase_index = 0
        self.finished = False

    def advance(self, elapsed_s: float) -> ScheduleState:
        """Return the schedule state at ``elapsed_s`` and update phase index."""
        if self.finished:
            return ScheduleState(
                phase_name="complete",
                phase_index=len(self.phases) - 1,
                phase_elapsed_s=self.phases[-1].duration_s,
                linear_x_mps=0.0,
                angular_z_rps=0.0,
                finished=True,
                entered_new_phase=False,
                completed_now=False,
            )
        if elapsed_s >= self.total_duration_s:
            self.finished = True
            return ScheduleState(
                phase_name="complete",
                phase_index=len(self.phases) - 1,
                phase_elapsed_s=self.phases[-1].duration_s,
                linear_x_mps=0.0,
                angular_z_rps=0.0,
                finished=True,
                entered_new_phase=False,
                completed_now=True,
            )

        entered_new_phase = False
        while (
            self.phase_index + 1 < len(self.phases)
            and elapsed_s >= self.phase_offsets[self.phase_index + 1]
        ):
            self.phase_index += 1
            entered_new_phase = True

        phase = self.phases[self.phase_index]
        phase_elapsed_s = elapsed_s - self.phase_offsets[self.phase_index]
        linear_x_mps, angular_z_rps = phase.command_at(max(phase_elapsed_s, 0.0))
        return ScheduleState(
            phase_name=phase.name,
            phase_index=self.phase_index,
            phase_elapsed_s=phase_elapsed_s,
            linear_x_mps=linear_x_mps,
            angular_z_rps=angular_z_rps,
            finished=False,
            entered_new_phase=entered_new_phase,
            completed_now=False,
        )
