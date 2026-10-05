"""Fixed-step unit tests for the deterministic steering actuator model."""

from __future__ import annotations

import pytest

from neurogrip_control.steering_actuator import SteeringActuator


def test_first_nonzero_command_with_delay_stays_safe():
    """Before history exists the delayed output must be the safe value 0.0."""
    actuator = SteeringActuator(delay_s=0.2)
    assert actuator.update(10.0, 0.5) == 0.0


def test_startup_shorter_than_delay_stays_safe():
    """Every sample inside the startup window returns zero."""
    actuator = SteeringActuator(delay_s=0.2)
    for index in range(5):
        time_s = 0.0 + index * 0.02
        assert actuator.update(time_s, 0.5) == 0.0


def test_zero_delay_is_passthrough():
    """Without delay the command is returned unchanged (up to gain)."""
    actuator = SteeringActuator(delay_s=0.0, gain=1.0)
    assert actuator.update(0.0, 0.3) == pytest.approx(0.3)
    assert actuator.update(0.1, -0.2) == pytest.approx(-0.2)


def test_static_gain_is_applied():
    """Gain scales the delivered command."""
    actuator = SteeringActuator(delay_s=0.0, gain=1.5)
    assert actuator.update(0.0, 0.4) == pytest.approx(0.6)


def test_integer_sample_delay_uses_fixed_step():
    """A two-sample delay returns the value from two steps earlier."""
    actuator = SteeringActuator(delay_s=0.2, gain=2.0)
    commands = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    outputs = [
        actuator.update(index * 0.1, value) for index, value in enumerate(commands)
    ]
    # t=0.2 target 0.0 -> value 0; t=0.3 target 0.1 -> value 1 (x2); t=0.4 -> value 2.
    assert outputs[2] == pytest.approx(0.0)
    assert outputs[3] == pytest.approx(2.0)
    assert outputs[4] == pytest.approx(4.0)


def test_fractional_delay_interpolates():
    """A delay between samples interpolates the command history."""
    actuator = SteeringActuator(delay_s=0.15, gain=1.0)
    commands = [0.0, 1.0, 2.0, 3.0, 4.0]
    actuator.update(0.0, commands[0])
    actuator.update(0.1, commands[1])
    actuator.update(0.2, commands[2])
    actuator.update(0.3, commands[3])
    # At t=0.4 the target is t=0.25, halfway between 2.0 (t=0.2) and 3.0 (t=0.3).
    assert actuator.update(0.4, commands[4]) == pytest.approx(2.5)


def test_gain_applied_after_delay_yields_expected_value():
    """The delivered value equals gain times the delayed historical command."""
    actuator = SteeringActuator(delay_s=0.1, gain=2.0)
    assert actuator.update(0.0, 0.25) == 0.0
    assert actuator.update(0.1, 0.25) == pytest.approx(0.5)
    assert actuator.update(0.2, -0.1) == pytest.approx(0.5)


def test_clock_reset_restarts_startup_window():
    """A backward clock jump clears history and re-enters the safe startup."""
    actuator = SteeringActuator(delay_s=0.2, gain=1.0)
    actuator.update(0.1, 1.0)
    actuator.update(0.3, 1.0)
    assert actuator.update(0.4, 1.0) == pytest.approx(1.0)
    # Clock jumps backwards: history is cleared, so the output is safe again.
    assert actuator.update(0.1, -1.0) == 0.0
    assert actuator.history_ready


def test_invalid_parameters_are_rejected():
    """Negative delay/gain and too-short history are configuration errors."""
    with pytest.raises(ValueError):
        SteeringActuator(delay_s=-0.1)
    with pytest.raises(ValueError):
        SteeringActuator(gain=-1.0)
    with pytest.raises(ValueError):
        SteeringActuator(delay_s=0.5, history_s=0.1)
