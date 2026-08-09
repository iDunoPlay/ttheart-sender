"""Find, inspect, move and focus top-level Windows windows.

The emulator is treated as "just a window matching a spec", so retargeting this
app at BlueStacks, MEmu or any other program is a config change (class names /
title substrings) rather than a code change.
"""

from __future__ import annotations

import ctypes
import logging
import time
from ctypes import wintypes
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

import win32api
import win32con
import win32gui
import win32process

from ..config import WindowConfig, WindowTargetConfig
from ..exceptions import WindowNotFoundError
from ..geometry import Insets, Point, Rect

log = logging.getLogger(__name__)

# DwmGetWindowAttribute: the *visual* bounds, excluding the invisible resize
# border Windows 10/11 pads GetWindowRect with.
_DWMWA_EXTENDED_FRAME_BOUNDS = 9


@dataclass(frozen=True)
class WindowInfo:
    """Snapshot of a window at a moment in time."""

    hwnd: int
    title: str
    class_name: str
    pid: int
    window_rect: Rect
    client_rect: Rect  # in screen coordinates
    is_visible: bool
    is_minimized: bool

    def describe(self) -> str:
        state = "minimized" if self.is_minimized else ("visible" if self.is_visible else "hidden")
        return (
            f"hwnd=0x{self.hwnd:08X} pid={self.pid} class={self.class_name!r} "
            f"title={self.title!r} rect={self.window_rect} client={self.client_rect} [{state}]"
        )


class WindowManager:
    """Enumerates top-level windows and matches them against a target spec."""

    def list_windows(self, *, visible_only: bool = True) -> List[WindowInfo]:
        results: List[WindowInfo] = []

        def _callback(hwnd: int, _param) -> bool:
            if visible_only and not win32gui.IsWindowVisible(hwnd):
                return True
            try:
                info = describe_window(hwnd)
            except Exception:  # noqa: BLE001 - window died mid-enumeration
                return True
            if visible_only and not info.title:
                return True
            results.append(info)
            return True

        win32gui.EnumWindows(_callback, None)
        return results

    def find(
        self,
        target: WindowTargetConfig,
        *,
        predicate: Optional[Callable[[WindowInfo], bool]] = None,
    ) -> List[WindowInfo]:
        """Return every window matching ``target``, ordered deterministically.

        Ordering is by on-screen position then hwnd, so "instance 0" keeps
        meaning the same window across runs as long as the layout is stable.
        """
        matches = [w for w in self.list_windows() if matches_target(w, target)]
        if predicate is not None:
            matches = [w for w in matches if predicate(w)]
        matches.sort(key=lambda w: (w.window_rect.left, w.window_rect.top, w.hwnd))
        return matches

    def find_one(self, config: WindowConfig) -> "WindowController":
        """Resolve the single window described by ``config`` or raise."""
        matches = self.find(config.target)
        if not matches:
            raise WindowNotFoundError(
                "No emulator window found. Is LDPlayer running? "
                "Run `python main.py windows` to list open windows, then adjust "
                "window.target.class_names / title_contains in config.yaml."
            )

        if config.instance_title:
            needle = config.instance_title.casefold()
            filtered = [w for w in matches if needle in w.title.casefold()]
            if not filtered:
                available = ", ".join(repr(w.title) for w in matches)
                raise WindowNotFoundError(
                    f"No window title contains {config.instance_title!r}. Found: {available}"
                )
            matches = filtered

        if config.instance >= len(matches):
            raise WindowNotFoundError(
                f"window.instance={config.instance} but only {len(matches)} matching "
                f"window(s) are open (valid range 0..{len(matches) - 1})."
            )

        chosen = matches[config.instance]
        if len(matches) > 1:
            log.info(
                "Found %d matching windows; using instance %d (%r)",
                len(matches),
                config.instance,
                chosen.title,
            )
        return WindowController(chosen.hwnd)

    def find_all_controllers(self, config: WindowConfig) -> List["WindowController"]:
        """Every matching window as a controller -- the multi-instance entry point."""
        return [WindowController(w.hwnd) for w in self.find(config.target)]


class WindowController:
    """Operations on one specific window handle."""

    def __init__(self, hwnd: int) -> None:
        self.hwnd = int(hwnd)

    # -- state -----------------------------------------------------------
    @property
    def exists(self) -> bool:
        return bool(win32gui.IsWindow(self.hwnd))

    def info(self) -> WindowInfo:
        self._ensure_alive()
        return describe_window(self.hwnd)

    def _ensure_alive(self) -> None:
        if not self.exists:
            raise WindowNotFoundError(
                f"Window 0x{self.hwnd:08X} no longer exists (was the emulator closed?)"
            )

    # -- geometry --------------------------------------------------------
    def content_rect(self, *, use_client_area: bool = True, insets: Optional[Insets] = None) -> Rect:
        """The screen-space rectangle we search inside and click within."""
        if self.looks_minimized():
            raise WindowNotFoundError(
                f"Window 0x{self.hwnd:08X} is minimized, so there are no pixels to read. "
                f"Restore it (or run `prepare`) before capturing."
            )
        info = self.info()
        rect = info.client_rect if use_client_area else info.window_rect
        if insets is not None and not insets.is_empty:
            rect = rect.shrink(insets)
        return rect

    def to_screen(self, point: Point, *, content: Optional[Rect] = None, **kwargs) -> Point:
        """Convert a content-relative point to absolute screen coordinates."""
        base = content if content is not None else self.content_rect(**kwargs)
        return Point(base.left + point.x, base.top + point.y)

    # -- manipulation ----------------------------------------------------
    def looks_minimized(self) -> bool:
        """True while the window has no on-screen pixels.

        ``IsIconic`` alone is not enough: during the restore animation the flag
        clears before ``GetWindowRect`` stops reporting the (-32000, -32000)
        minimized position, and capturing in that window yields garbage.
        """
        if win32gui.IsIconic(self.hwnd):
            return True
        rect = get_window_rect(self.hwnd)
        return rect.left <= -30000 or rect.top <= -30000

    def restore(self, *, timeout: float = 5.0) -> bool:
        """Un-minimize and wait until the window really has a screen position."""
        self._ensure_alive()
        if not self.looks_minimized():
            return True

        win32gui.ShowWindow(self.hwnd, win32con.SW_RESTORE)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not self.looks_minimized():
                time.sleep(0.15)  # let the first frame paint
                return True
            time.sleep(0.05)

        log.warning(
            "Window 0x%08X did not finish restoring within %.1fs", self.hwnd, timeout
        )
        return False

    def move_to(
        self,
        position: Point,
        size: Optional[Sequence[int]] = None,
        *,
        activate: bool = False,
    ) -> Rect:
        """Move (and optionally resize) so the *visible* frame lands on ``position``.

        Windows pads ``GetWindowRect`` with an invisible resize border, so a
        naive move to (0, 0) leaves a ~7px gap. We compensate using the DWM
        extended frame bounds.
        """
        self._ensure_alive()
        self.restore()

        window_rect = get_window_rect(self.hwnd)
        visual_rect = get_extended_frame_rect(self.hwnd) or window_rect
        shadow_dx = window_rect.left - visual_rect.left
        shadow_dy = window_rect.top - visual_rect.top

        flags = win32con.SWP_NOZORDER
        if not activate:
            flags |= win32con.SWP_NOACTIVATE

        if size is None:
            width, height = 0, 0
            flags |= win32con.SWP_NOSIZE
        else:
            width, height = int(size[0]), int(size[1])
            # Requested size is visual; grow it back by the invisible border.
            width += window_rect.width - visual_rect.width
            height += window_rect.height - visual_rect.height

        win32gui.SetWindowPos(
            self.hwnd,
            0,
            position.x + shadow_dx,
            position.y + shadow_dy,
            width,
            height,
            flags,
        )
        time.sleep(0.05)
        return get_window_rect(self.hwnd)

    def focus(self) -> bool:
        """Bring the window to the foreground.

        Windows refuses ``SetForegroundWindow`` from a process that does not own
        the current foreground window, so we fall back to the standard
        AttachThreadInput dance before giving up.
        """
        self._ensure_alive()
        self.restore()

        if win32gui.GetForegroundWindow() == self.hwnd:
            return True

        try:
            win32gui.SetForegroundWindow(self.hwnd)
            return True
        except Exception:  # noqa: BLE001 - expected when we lack foreground rights
            pass

        try:
            foreground = win32gui.GetForegroundWindow()
            current_thread = win32api.GetCurrentThreadId()
            target_thread, _ = win32process.GetWindowThreadProcessId(self.hwnd)
            foreground_thread, _ = (
                win32process.GetWindowThreadProcessId(foreground) if foreground else (0, 0)
            )
            attached = []
            for thread in {foreground_thread, target_thread}:
                if thread and thread != current_thread:
                    ctypes.windll.user32.AttachThreadInput(current_thread, thread, True)
                    attached.append(thread)
            try:
                win32gui.BringWindowToTop(self.hwnd)
                win32gui.SetForegroundWindow(self.hwnd)
            finally:
                for thread in attached:
                    ctypes.windll.user32.AttachThreadInput(current_thread, thread, False)
            time.sleep(0.1)
            return win32gui.GetForegroundWindow() == self.hwnd
        except Exception as exc:  # noqa: BLE001
            log.warning("Could not focus window 0x%08X: %s", self.hwnd, exc)
            return False

    def is_foreground(self) -> bool:
        return win32gui.GetForegroundWindow() == self.hwnd

    def prepare(self, config: WindowConfig) -> Rect:
        """Restore, position and focus per config; returns the content rect.

        This is the single choke point every entry point (CLI startup, flows,
        the ``prepare`` command) goes through to get the emulator into a known
        state, so "where is LDPlayer and is it safe to capture/click" is
        answered the same way everywhere.
        """
        self.restore()
        if config.move_on_prepare:
            size = (config.size.width, config.size.height) if config.size else None
            self.move_to(Point(config.position.x, config.position.y), size)
        if config.focus_on_prepare:
            if not self.focus():
                log.warning(
                    "Could not bring the emulator to the front; anything covering "
                    "it will show up in screenshots until it gets focus."
                )
            time.sleep(0.2)  # let the window finish repainting after raising it
        rect = self.content_rect(
            use_client_area=config.use_client_area, insets=config.insets
        )
        log.info("Window ready: %s content=%s", self.info().title, rect)
        return rect

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"WindowController(hwnd=0x{self.hwnd:08X})"


# --------------------------------------------------------------------------
# Win32 helpers
# --------------------------------------------------------------------------
def describe_window(hwnd: int) -> WindowInfo:
    title = win32gui.GetWindowText(hwnd)
    class_name = win32gui.GetClassName(hwnd)
    try:
        _, pid = win32process.GetWindowThreadProcessId(hwnd)
    except Exception:  # noqa: BLE001
        pid = 0
    return WindowInfo(
        hwnd=hwnd,
        title=title,
        class_name=class_name,
        pid=pid,
        window_rect=get_window_rect(hwnd),
        client_rect=get_client_rect_on_screen(hwnd),
        is_visible=bool(win32gui.IsWindowVisible(hwnd)),
        is_minimized=bool(win32gui.IsIconic(hwnd)),
    )


def get_window_rect(hwnd: int) -> Rect:
    left, top, right, bottom = win32gui.GetWindowRect(hwnd)
    return Rect.from_ltrb(left, top, right, bottom)


def get_client_rect_on_screen(hwnd: int) -> Rect:
    """Client area (no title bar / borders) translated into screen coordinates."""
    left, top, right, bottom = win32gui.GetClientRect(hwnd)
    origin_x, origin_y = win32gui.ClientToScreen(hwnd, (left, top))
    return Rect(origin_x, origin_y, right - left, bottom - top)


def get_extended_frame_rect(hwnd: int) -> Optional[Rect]:
    """DWM visual bounds, or None when unavailable (e.g. composition off)."""
    try:
        rect = wintypes.RECT()
        result = ctypes.windll.dwmapi.DwmGetWindowAttribute(
            wintypes.HWND(hwnd),
            ctypes.c_uint(_DWMWA_EXTENDED_FRAME_BOUNDS),
            ctypes.byref(rect),
            ctypes.sizeof(rect),
        )
        if result != 0:
            return None
        return Rect.from_ltrb(rect.left, rect.top, rect.right, rect.bottom)
    except Exception:  # noqa: BLE001
        return None


def matches_target(info: WindowInfo, target: WindowTargetConfig) -> bool:
    """Apply the include/exclude rules of a target spec to one window."""
    class_lower = info.class_name.casefold()
    title_lower = info.title.casefold()

    for excluded in target.exclude_class_names:
        if class_lower == excluded.casefold():
            return False
    for excluded in target.exclude_title_contains:
        if excluded.casefold() in title_lower:
            return False

    class_hit = any(class_lower == name.casefold() for name in target.class_names)
    title_hit = any(needle.casefold() in title_lower for needle in target.title_contains)

    # An empty rule list means "no opinion" rather than "matches nothing".
    if not target.class_names:
        class_hit = not target.match_any
    if not target.title_contains:
        title_hit = not target.match_any

    return (class_hit or title_hit) if target.match_any else (class_hit and title_hit)


def get_virtual_screen_rect() -> Rect:
    """Bounding box of all monitors combined."""
    left = win32api.GetSystemMetrics(win32con.SM_XVIRTUALSCREEN)
    top = win32api.GetSystemMetrics(win32con.SM_YVIRTUALSCREEN)
    width = win32api.GetSystemMetrics(win32con.SM_CXVIRTUALSCREEN)
    height = win32api.GetSystemMetrics(win32con.SM_CYVIRTUALSCREEN)
    return Rect(left, top, width, height)
