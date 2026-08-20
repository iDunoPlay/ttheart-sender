"""Cursor stop zone.

The flow drives the mouse, so the cursor belongs inside the emulator for the
whole run. If it turns up outside, either the user has grabbed the mouse back
or a click has landed somewhere it should not have -- both are reasons to stop
before the next click goes out.

This reads the *physical* cursor through ``GetCursorPos`` rather than the
:class:`~..control.mouse.MouseController`, so it reports where the pointer
actually is even under ``--dry-run``, where the mouse backend is a no-op.
"""

from __future__ import annotations

import ctypes
import logging
import time
from ctypes import wintypes
from typing import Callable, Optional

from ..geometry import Point, Rect

log = logging.getLogger(__name__)

RectProvider = Callable[[], Optional[Rect]]


def cursor_position() -> Optional[Point]:
    """The physical cursor position, or None where Win32 is unavailable."""
    try:
        point = wintypes.POINT()
        if not ctypes.windll.user32.GetCursorPos(ctypes.byref(point)):
            return None
        return Point(int(point.x), int(point.y))
    except Exception:  # noqa: BLE001 - non-Windows / restricted env
        return None


class CursorGuard:
    """Stop guard that trips once the cursor leaves ``rect_provider()``.

    Registered with :meth:`~..control.hotkey.StopKeyWatcher.add_guard`, so it
    is polled everywhere the panic key is: between steps, inside every wait,
    and around each template search.

    It arms itself the first time it sees the cursor *inside* the window. A run
    that starts with the pointer parked on another monitor is the normal case
    -- the first click of the flow brings the cursor home and arms the guard --
    and without that, every run would stop on its first checkpoint.
    """

    def __init__(
        self,
        rect_provider: RectProvider,
        *,
        margin: int = 0,
        position: Callable[[], Optional[Point]] = cursor_position,
        rect_ttl: float = 0.25,
        enabled: Optional[Callable[[], bool]] = None,
    ) -> None:
        self.rect_provider = rect_provider
        self.margin = max(0, int(margin))
        self.position = position
        self.rect_ttl = rect_ttl
        #: Asked on every poll rather than read once, so the panel can turn
        #: the guard off part way through a run. Registering the guard
        #: conditionally instead would freeze the answer at startup, and this
        #: is the one stop the user reaches for by accident.
        self.enabled = enabled or (lambda: True)
        self._armed = False
        self._rect: Optional[Rect] = None
        self._rect_read_at = 0.0

    def reset(self) -> None:
        self._armed = False
        self._rect = None
        self._rect_read_at = 0.0

    @property
    def armed(self) -> bool:
        return self._armed

    def zone(self) -> Optional[Rect]:
        """The allowed rectangle, re-read at most every ``rect_ttl`` seconds.

        Cached because this runs on every stop checkpoint -- roughly 20 times a
        second inside a wait -- while the emulator window moves approximately
        never.
        """
        now = time.monotonic()
        if self._rect is None or now - self._rect_read_at >= self.rect_ttl:
            rect = self.rect_provider()
            if rect is not None:
                self._rect = rect
                self._rect_read_at = now
        return self._rect

    def __call__(self) -> Optional[str]:
        """Return a stop reason, or None to let the run continue."""
        if not self.enabled():
            # Disarm as well as stand down: coming back on should re-arm on
            # the next cursor sighting inside the window rather than trip
            # immediately on a pointer that wandered off while it was off.
            self._armed = False
            return None
        rect = self.zone()
        if rect is None:
            return None
        point = self.position()
        if point is None:
            return None

        zone = Rect(
            rect.left - self.margin,
            rect.top - self.margin,
            rect.width + 2 * self.margin,
            rect.height + 2 * self.margin,
        )
        if zone.contains(point):
            if not self._armed:
                log.debug("Cursor stop zone armed: %s inside %s", point, zone)
                self._armed = True
            return None

        if not self._armed:
            return None
        return f"cursor left the emulator: {point} is outside {zone}"
