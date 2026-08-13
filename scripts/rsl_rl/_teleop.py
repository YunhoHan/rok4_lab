# Copyright (c) 2026, RoK4 Lab Contributors.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RoK4-local teleoperation command helpers."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import Protocol


class _KeyboardCallbackInterface(Protocol):
    """Interface required to register keyboard callbacks."""

    def add_callback(self, key: str, func: Callable[[], None]) -> None:
        """Register a no-argument callback for one key."""


class IncrementalKeyboardCommand:
    """Accumulate keyboard SE(2) commands in fixed physical-unit increments."""

    _KEY_DELTAS = {
        "NUMPAD_8": (0, 1.0),
        "UP": (0, 1.0),
        "NUMPAD_2": (0, -1.0),
        "DOWN": (0, -1.0),
        "NUMPAD_4": (1, 1.0),
        "LEFT": (1, 1.0),
        "NUMPAD_6": (1, -1.0),
        "RIGHT": (1, -1.0),
        "NUMPAD_7": (2, 1.0),
        "Z": (2, 1.0),
        "NUMPAD_9": (2, -1.0),
        "X": (2, -1.0),
    }

    def __init__(
        self,
        step: float,
        command_ranges: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    ) -> None:
        """Initialize the command accumulator.

        Args:
            step: Increment applied per key press. It is interpreted as [m/s]
                for linear axes and [rad/s] for yaw.
            command_ranges: Lower and upper limits for ``(vx, vy, wz)`` in
                ``[(m/s, m/s), (m/s, m/s), (rad/s, rad/s)]``.
        """
        if step <= 0.0:
            raise ValueError(f"Keyboard command step must be positive, received {step}.")
        for lower, upper in command_ranges:
            if lower > 0.0 or upper < 0.0 or lower >= upper:
                raise ValueError(
                    "Each keyboard command range must be ordered and contain zero, received "
                    f"({lower}, {upper})."
                )

        self._step = step
        self._command_ranges = command_ranges
        self._command = [0.0, 0.0, 0.0]

    @property
    def command(self) -> tuple[float, float, float]:
        """Current ``(vx, vy, wz)`` command [m/s, m/s, rad/s]."""
        return tuple(self._command)

    def bind(self, interface: _KeyboardCallbackInterface) -> None:
        """Bind incremental movement and command-reset callbacks."""
        for key, (axis, direction) in self._KEY_DELTAS.items():
            interface.add_callback(key, partial(self._increment, axis, direction))
        interface.add_callback("L", self.reset)

    def reset(self) -> None:
        """Reset all command components to zero."""
        self._command[:] = (0.0, 0.0, 0.0)

    def _increment(self, axis: int, direction: float) -> None:
        lower, upper = self._command_ranges[axis]
        value = self._command[axis] + direction * self._step
        self._command[axis] = round(min(max(value, lower), upper), 10)
        vx, vy, wz = self._command
        print(f"[INFO] Keyboard command: vx={vx:+.2f} m/s, vy={vy:+.2f} m/s, wz={wz:+.2f} rad/s")
