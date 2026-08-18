"""Global stop key.

Automation that takes over the mouse must always be interruptible. This polls
``GetAsyncKeyState`` for a panic key (default F12) that works even while the
emulator has focus, and provides a sleep that honours it.
"""

from __future__ import annotations

import logging
import time
from typing import Callable, List, Optional, Tuple

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
        self._reason: Optional[str] = None
        #: Extra stop sources polled alongside the key -- see
        #: :class:`~.cursor.CursorGuard`. Each returns a reason, or None.
        self._guards: List[Tuple[str, Callable[[], Optional[str]]]] = []
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

    def add_guard(self, name: str, guard: Callable[[], Optional[str]]) -> None:
        """Register another stop source, polled at every checkpoint.

        ``guard()`` returns a reason to stop, or None to allow the run to
        continue. Anything that can decide "not now, but ask me again in 50ms"
        belongs here rather than in the runner, because this is the one place
        every wait and every step already calls into.
        """
        self._guards.append((name, guard))

    def request_stop(self, reason: Optional[str] = None) -> None:
        """Ask the current run to stop at the next checkpoint."""
        self._requested = True
        if reason and self._reason is None:
            self._reason = reason

    def reset(self) -> None:
        self._requested = False
        self._reason = None
        for _, guard in self._guards:
            reset = getattr(guard, "reset", None)
            if callable(reset):
                reset()

    def triggered(self) -> bool:
        if self._requested:
            return True
        if self._get_async_key_state is not None and self._vk is not None:
            # High-order bit set == key is down right now.
            if self._get_async_key_state(self._vk) & 0x8000:
                self.request_stop(f"key: {self.key_name}")
                return True
        for name, guard in self._guards:
            reason = guard()
            if reason:
                self.request_stop(f"{name}: {reason}")
                return True
        return False

    def check(self) -> None:
        """Raise :class:`StopRequested` if the user asked to stop."""
        if self.triggered():
            raise StopRequested(f"Stop requested ({self._reason or self.key_name})")

    def describe(self) -> str:
        parts = [f"stop key: {str(self.key_name).upper()}" if self.enabled else "stop key: disabled"]
        parts += [f"guard: {name}" for name, _ in self._guards]
        return ", ".join(parts)


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
