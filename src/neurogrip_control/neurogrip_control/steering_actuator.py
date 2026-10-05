"""
Deterministic steering actuator delay and gain model.

The NeuroGrip-X scenario system can hide a slow or under/over-acting steering
actuator in the command path.  This model makes that behaviour explicit,
testable at a fixed time step, and reproducible from a seed.  It contains no
ROS dependency so it can be validated without launching Gazebo.

Safety semantics:

* Until enough history exists for the configured pure time delay, the output is
  the safe value 0.0 -- never a command that has not actually been delayed.
* Gain is applied *after* the delay.
* A backward clock jump clears the history and restarts the startup window.
* The command guard hard-clamps the actuator output, and its watchdog bypasses
  the delay entirely (immediate zero) so a stale command cannot keep steering.
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass, field


@dataclass
class SteeringActuator:
    """Apply a pure time delay and a static gain to a steering command."""

    delay_s: float = 0.0
    gain: float = 1.0
    history_s: float = 2.0
    _times: list[float] = field(default_factory=list)
    _values: list[float] = field(default_factory=list)
    _start_time_s: float | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.delay_s < 0.0:
            raise ValueError("steering delay must be non-negative.")
        if self.gain < 0.0:
            raise ValueError("steering gain must be non-negative.")
        if self.history_s < self.delay_s + 1e-9:
            raise ValueError("history_s must be at least delay_s.")

    @property
    def history_ready(self) -> bool:
        """Return whether at least one sample has been recorded."""
        return bool(self._times)

    def reset(self) -> None:
        """Clear the delay history, e.g. on a scenario reset or watchdog stop."""
        self._times.clear()
        self._values.clear()
        self._start_time_s = None

    def _value_at(self, target_time_s: float) -> float:
        """Return the linearly interpolated historical command at ``target``."""
        if not self._times:
            return 0.0
        if target_time_s <= self._times[0]:
            return self._values[0]
        if target_time_s >= self._times[-1]:
            return self._values[-1]
        index = bisect_left(self._times, target_time_s)
        lower_time = self._times[index - 1]
        upper_time = self._times[index]
        span = upper_time - lower_time
        if span <= 0.0:
            return self._values[index]
        fraction = (target_time_s - lower_time) / span
        return self._values[index - 1] + fraction * (
            self._values[index] - self._values[index - 1]
        )

    def _trim(self, cutoff_s: float) -> None:
        """Drop history older than the retention window."""
        drop_index = 0
        while drop_index + 1 < len(self._times) and self._times[drop_index + 1] < cutoff_s:
            drop_index += 1
        if drop_index > 0:
            del self._times[:drop_index]
            del self._values[:drop_index]

    def update(self, time_s: float, command: float) -> float:
        """Record ``command`` at ``time_s`` and return the delayed, gained value."""
        if self._times and time_s < self._times[-1]:
            # A non-monotonic clock means the scenario restarted; start clean.
            self.reset()
        if self._start_time_s is None:
            self._start_time_s = time_s
        self._times.append(time_s)
        self._values.append(command)
        self._trim(time_s - self.history_s)

        if self.delay_s <= 0.0:
            return self.gain * command

        target_time_s = time_s - self.delay_s
        if self._start_time_s is None or target_time_s < self._start_time_s:
            # Not enough history yet: stay at the safe initial value.
            return 0.0
        return self.gain * self._value_at(target_time_s)
