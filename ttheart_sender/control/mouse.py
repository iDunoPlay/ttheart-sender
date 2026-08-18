"""Mouse control.

:class:`MouseController` is the seam: everything upstream depends on the
protocol, never on pyautogui. Swapping in a raw ``SendInput`` backend, an ADB
backend, or a no-op backend for tests means adding a class here.
"""

from __future__ import annotations

import ctypes
import logging
import random
import sys
import time
from ctypes import wintypes
from typing import Optional, Protocol, runtime_checkable

from ..config import InputConfig
from ..geometry import Point

log = logging.getLogger(__name__)

Button = str  # "left" | "right" | "middle"

WHEEL_DELTA = 120  # one detent of a physical wheel
_INPUT_MOUSE = 0
_MOUSEEVENTF_WHEEL = 0x0800


class _MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.POINTER(wintypes.ULONG)),
    ]


class _INPUT(ctypes.Structure):
    class _VALUE(ctypes.Union):
        _fields_ = [("mi", _MOUSEINPUT)]

    _anonymous_ = ("value",)
    _fields_ = [("type", wintypes.DWORD), ("value", _VALUE)]


def send_wheel(notches: int, *, interval: float = 0.05) -> None:
    """Spin the wheel ``notches`` detents at the current cursor position.

    Deliberately not ``pyautogui.scroll``. pyautogui injects the wheel through
    the legacy ``mouse_event`` API -- its source carries a TODO reading "ARG!
    For some reason, SendInput isn't working for mouse events. I'm switching to
    using the older mouse_event win32 function" -- and LDPlayer's render window
    does not act on wheel events delivered that way. Measured against the game's
    ranking list: pyautogui moved 0 pixels at every notch count in both
    directions, while identical notches sent via ``SendInput`` scrolled it
    normally.

    One event per detent, matching a physical wheel, rather than a single
    multiplied delta -- both work here, but per-detent is what apps expect and
    is the safer default for anything with its own scroll acceleration.
    """
    if not notches:
        return
    step = WHEEL_DELTA if notches > 0 else -WHEEL_DELTA
    for index in range(abs(int(notches))):
        if index and interval > 0:
            time.sleep(interval)
        event = _INPUT(
            type=_INPUT_MOUSE,
            mi=_MOUSEINPUT(0, 0, step, _MOUSEEVENTF_WHEEL, 0, None),
        )
        sent = ctypes.windll.user32.SendInput(1, ctypes.byref(event), ctypes.sizeof(_INPUT))
        if sent != 1:
            raise OSError(
                f"SendInput refused the wheel event (error {ctypes.get_last_error()})"
            )


@runtime_checkable
class MouseController(Protocol):
    """The mouse operations the automation engine relies on."""

    def position(self) -> Point: ...

    def move(self, point: Point, *, duration: Optional[float] = None) -> None: ...

    def click(
        self,
        point: Optional[Point] = None,
        *,
        button: Button = "left",
        clicks: int = 1,
        interval: float = 0.08,
        duration: Optional[float] = None,
    ) -> None: ...

    def drag(
        self,
        start: Point,
        end: Point,
        *,
        button: Button = "left",
        duration: float = 0.4,
    ) -> None: ...

    def scroll(self, amount: int, point: Optional[Point] = None) -> None: ...


class PyAutoGuiMouse:
    """pyautogui-backed mouse with configurable jitter and timing.

    Note: pyautogui addresses the *primary* monitor's coordinate space, which is
    why the emulator gets parked at the top-left of the primary display.
    """

    def __init__(self, config: Optional[InputConfig] = None) -> None:
        self.config = config or InputConfig()
        self._pyautogui = _import_pyautogui()
        self._pyautogui.FAILSAFE = self.config.failsafe
        # We insert our own pauses; pyautogui's implicit 0.1s just adds lag.
        self._pyautogui.PAUSE = 0.0

    # -- queries ---------------------------------------------------------
    def position(self) -> Point:
        x, y = self._pyautogui.position()
        return Point(int(x), int(y))

    # -- movement --------------------------------------------------------
    def _jitter(self, point: Point) -> Point:
        radius = self.config.click_jitter
        if radius <= 0:
            return point
        return Point(
            point.x + random.randint(-radius, radius),
            point.y + random.randint(-radius, radius),
        )

    def move(self, point: Point, *, duration: Optional[float] = None) -> None:
        move_duration = self.config.move_duration if duration is None else duration
        self._pyautogui.moveTo(point.x, point.y, duration=max(0.0, move_duration))

    # -- clicking --------------------------------------------------------
    def click(
        self,
        point: Optional[Point] = None,
        *,
        button: Button = "left",
        clicks: int = 1,
        interval: float = 0.08,
        duration: Optional[float] = None,
    ) -> None:
        target = self._jitter(point) if point is not None else self.position()
        if point is not None:
            self.move(target, duration=duration)

        for index in range(max(1, clicks)):
            if index:
                time.sleep(interval)
            self._pyautogui.mouseDown(button=button)
            time.sleep(max(0.0, self.config.click_hold))
            self._pyautogui.mouseUp(button=button)

        log.debug("click %s button=%s clicks=%d", target, button, clicks)
        if self.config.post_click_delay > 0:
            time.sleep(self.config.post_click_delay)

    def drag(
        self,
        start: Point,
        end: Point,
        *,
        button: Button = "left",
        duration: float = 0.4,
    ) -> None:
        self.move(start)
        self._pyautogui.mouseDown(button=button)
        try:
            self._pyautogui.moveTo(end.x, end.y, duration=max(0.0, duration))
        finally:
            self._pyautogui.mouseUp(button=button)
        log.debug("drag %s -> %s", start, end)
        if self.config.post_click_delay > 0:
            time.sleep(self.config.post_click_delay)

    def scroll(self, amount: int, point: Optional[Point] = None) -> None:
        if point is not None:
            self.move(point)
        if sys.platform == "win32":
            send_wheel(int(amount))
        else:
            self._pyautogui.scroll(int(amount))


class NullMouse:
    """Records intent without touching the real cursor -- used by ``--dry-run``."""

    def __init__(self) -> None:
        self.events: list = []

    def position(self) -> Point:
        return Point(0, 0)

    def move(self, point: Point, *, duration: Optional[float] = None) -> None:
        self.events.append(("move", point))

    def click(
        self,
        point: Optional[Point] = None,
        *,
        button: Button = "left",
        clicks: int = 1,
        interval: float = 0.08,
        duration: Optional[float] = None,
    ) -> None:
        self.events.append(("click", point, button, clicks))

    def drag(self, start: Point, end: Point, *, button: Button = "left", duration: float = 0.4) -> None:
        self.events.append(("drag", start, end))

    def scroll(self, amount: int, point: Optional[Point] = None) -> None:
        self.events.append(("scroll", amount, point))


def _import_pyautogui():
    try:
        import pyautogui  # noqa: PLC0415 - imported lazily so tests can run headless
    except Exception as exc:  # noqa: BLE001
        raise ImportError(
            "pyautogui is required for mouse/keyboard control. "
            "Install dependencies with: pip install -r requirements.txt"
        ) from exc
    return pyautogui
