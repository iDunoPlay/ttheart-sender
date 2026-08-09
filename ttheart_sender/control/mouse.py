"""Mouse control.

:class:`MouseController` is the seam: everything upstream depends on the
protocol, never on pyautogui. Swapping in a raw ``SendInput`` backend, an ADB
backend, or a no-op backend for tests means adding a class here.
"""

from __future__ import annotations

import logging
import random
import time
from typing import Optional, Protocol, runtime_checkable

from ..config import InputConfig
from ..geometry import Point

log = logging.getLogger(__name__)

Button = str  # "left" | "right" | "middle"


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
