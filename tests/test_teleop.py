# Copyright (c) 2026, RoK4 Lab Contributors.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for RoK4-local teleoperation helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


_TELEOP_PATH = Path(__file__).resolve().parents[1] / "scripts/rsl_rl/_teleop.py"
_SPEC = importlib.util.spec_from_file_location("rok4_teleop_test_module", _TELEOP_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_TELEOP = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_TELEOP)


class _KeyboardInterfaceStub:
    def __init__(self) -> None:
        self.callbacks = {}

    def add_callback(self, key, func) -> None:
        self.callbacks[key] = func


def test_incremental_keyboard_command_accumulates_and_resets() -> None:
    """Each key press must change one physical command component by the configured step."""
    command = _TELEOP.IncrementalKeyboardCommand(
        step=0.05,
        command_ranges=((-0.3, 0.85), (-0.3, 0.3), (-0.6, 0.6)),
    )
    interface = _KeyboardInterfaceStub()
    command.bind(interface)

    interface.callbacks["UP"]()
    interface.callbacks["UP"]()
    interface.callbacks["LEFT"]()
    interface.callbacks["X"]()
    assert command.command == (0.1, 0.05, -0.05)

    interface.callbacks["DOWN"]()
    interface.callbacks["RIGHT"]()
    interface.callbacks["Z"]()
    assert command.command == (0.05, 0.0, 0.0)

    interface.callbacks["L"]()
    assert command.command == (0.0, 0.0, 0.0)


def test_incremental_keyboard_command_clamps_to_training_ranges() -> None:
    """Accumulated commands must remain inside the policy training ranges."""
    command = _TELEOP.IncrementalKeyboardCommand(
        step=0.05,
        command_ranges=((-0.3, 0.85), (-0.3, 0.3), (-0.6, 0.6)),
    )
    interface = _KeyboardInterfaceStub()
    command.bind(interface)

    for _ in range(30):
        interface.callbacks["UP"]()
        interface.callbacks["RIGHT"]()
        interface.callbacks["Z"]()
    assert command.command == (0.85, -0.3, 0.6)


@pytest.mark.parametrize("step", [0.0, -0.05])
def test_incremental_keyboard_command_rejects_non_positive_steps(step: float) -> None:
    """The keyboard increment must be strictly positive."""
    with pytest.raises(ValueError, match="must be positive"):
        _TELEOP.IncrementalKeyboardCommand(
            step=step,
            command_ranges=((-0.3, 0.85), (-0.3, 0.3), (-0.6, 0.6)),
        )
