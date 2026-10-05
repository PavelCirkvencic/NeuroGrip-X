"""
Static contract test: only ``command_guard`` may publish actuator commands.

The NeuroGrip-X safety model requires that every future controller, AI node or
MATLAB bridge publishes to ``/neurogrip/command_raw`` and never directly to the
Gazebo actuator topic.  This test parses the Python sources with ``ast`` and
fails the moment a second publisher appears, which is exactly the kind of
regression a runtime-only test would miss before launch.
"""

from __future__ import annotations

import ast
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parents[2]
ALLOWED_PUBLISHER_FILES = {"command_guard.py"}
ACTUATOR_TOKEN = "cmd_vel"


def _contains_actuator_token(node: ast.AST) -> bool:
    """Return whether a string constant/joined string mentions the actuator."""
    for sub_node in ast.walk(node):
        if isinstance(sub_node, ast.Constant) and isinstance(sub_node.value, str):
            if ACTUATOR_TOKEN in sub_node.value:
                return True
    return False


def _actuator_variables(tree: ast.AST) -> set[str]:
    """Collect attribute names assigned a string containing the actuator token."""
    names = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not _contains_actuator_token(node.value):
            continue
        for target in node.targets:
            if isinstance(target, ast.Attribute):
                names.add(target.attr)
            elif isinstance(target, ast.Name):
                names.add(target.id)
    return names


def _publisher_topic_is_actuator(call: ast.Call, actuator_variables: set[str]) -> bool:
    """Resolve the topic argument of one ``create_publisher`` call."""
    if len(call.args) < 2:
        return _contains_actuator_token(call)
    topic = call.args[1]
    if _contains_actuator_token(topic):
        return True
    if isinstance(topic, ast.Attribute) and topic.attr in actuator_variables:
        return True
    if isinstance(topic, ast.Name) and topic.id in actuator_variables:
        return True
    return False


def test_only_command_guard_publishes_the_actuator_topic():
    """No Python node other than command_guard may publish the actuator topic."""
    offending_files = []
    for path in sorted(SOURCE_ROOT.rglob("*.py")):
        if "test" in path.parts:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        actuator_variables = _actuator_variables(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr != "create_publisher":
                continue
            if _publisher_topic_is_actuator(node, actuator_variables):
                offending_files.append(path.name)

    unexpected = set(offending_files) - ALLOWED_PUBLISHER_FILES
    assert not unexpected, (
        "Actuator topic may only be published by command_guard, found: "
        f"{sorted(unexpected)}"
    )


def test_excitation_driver_publishes_raw_command_topic():
    """The excitation driver must target the guarded raw command topic."""
    driver_path = SOURCE_ROOT / "neurogrip_control" / "neurogrip_control" / "excitation_driver.py"
    source = driver_path.read_text(encoding="utf-8")
    assert "/neurogrip/command_raw" in source
    assert "cmd_vel" not in source
