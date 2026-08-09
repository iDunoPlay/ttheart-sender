"""Global stop key.

Automation that takes over the mouse must always be interruptible. This polls
``GetAsyncKeyState`` for a panic key (default F12) that works even while the
emulator has focus, and provides a sleep that honours it.
"""

from __future__ import annotations

import logging
import time
from typing import Callable, Optional

from ..exceptions import StopRequested

log = logging.getLogger(__name__)

#: Names accepted in ``runner.stop_key`` -> Win32 virtual key codes.
VK_CODES = {
    "esc": 0x1B,
    "escape": 0x1B,
    "space": 0x20,
    "enter": 0x0D,
    "pause": 0x13,
    "scrolllock": 0x91,
    "insert": 0x2D,
    "delete": 0x2E,
    "home": 0x24,
    "end": 0x23,
    **{f"f{n}": 0x6F + n for n in range(1, 13)},  # F1..F12 -> 0x70..0x7B
}


def resolve_vk(name: str) -> Optional[int]:
    """Map a friendly key name (or ``0x7B`` style literal) to a VK code."""
    if not name:
        return None
    key = str(name).strip().casefold()
    if key in VK_CODES:
        return VK_CODES[key]
    try:
        return int(key, 0)
    except ValueError:
        log.warning("Unknown stop_key %r; the stop hotkey is disabled", name)
        return None


class StopKeyWatcher:
    """Reports whether the user has asked the run to stop.

    Also acts as the integration point for other stop sources (Ctrl+C, a future
    GUI button) via :meth:`request_stop`.
    """

    def __init__(self, key: Optional[str] = "f12") -> None:
        self.key_name = key
        self._vk = resolve_vk(key) if key else None
        self._requested = False
        self._get_async_key_state: Optional[Callable[[int], int]] = None
        if self._vk is not None:
            try:
                import ctypes

                self._get_async_key_state = ctypes.windll.user32.GetAsyncKeyState
            except Exception:  # noqa: BLE001 - non-Windows / restricted env
                log.debug("GetAsyncKeyState unavailable; stop key disabled")
                self._get_async_key_state = None

    @property
    def enabled(self) -> bool:
        return self._get_async_key_state is not None

    def request_stop(self) -> None:
        """Ask the current run to stop at the next checkpoint."""
        self._requested = True

    def reset(self) -> None:
        self._requested = False

    def triggered(self) -> bool:
        if self._requested:
            return True
        if self._get_async_key_state is None or self._vk is None:
            return False
        # High-order bit set == key is down right now.
        if self._get_async_key_state(self._vk) & 0x8000:
            self._requested = True
            return True
        return False

    def check(self) -> None:
        """Raise :class:`StopRequested` if the user asked to stop."""
        if self.triggered():
            raise StopRequested(f"Stop requested (key: {self.key_name})")

    def describe(self) -> str:
        if not self.enabled:
            return "stop key: disabled"
        return f"stop key: {str(self.key_name).upper()}"


def interruptible_sleep(
    seconds: float,
    watcher: Optional[StopKeyWatcher] = None,
    *,
    poll_interval: float = 0.05,
) -> None:
    """Sleep, checking the stop key roughly every ``poll_interval`` seconds."""
    if seconds <= 0:
        if watcher is not None:
            watcher.check()
        return
    deadline = time.monotonic() + seconds
    while True:
        if watcher is not None:
            watcher.check()
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        time.sleep(min(poll_interval, remaining))
