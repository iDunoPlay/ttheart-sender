"""Capture template images by pointing at two corners.

Deliberately GUI-free: you position the mouse and tap a key, so it works over
the top of LDPlayer without an overlay window stealing focus or repainting the
region we are about to grab.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Callable, Optional

from ..control.hotkey import resolve_vk
from ..geometry import Point, Rect
from ..screen.capture import ScreenCapture, save_image

log = logging.getLogger(__name__)


def _key_state_fn() -> Optional[Callable[[int], int]]:
    try:
        import ctypes

        return ctypes.windll.user32.GetAsyncKeyState
    except Exception:  # noqa: BLE001
        return None


def _cursor_position() -> Point:
    import ctypes
    from ctypes import wintypes

    point = wintypes.POINT()
    ctypes.windll.user32.GetCursorPos(ctypes.byref(point))
    return Point(int(point.x), int(point.y))


def wait_for_key(
    vk: int,
    *,
    cancel_vk: Optional[int] = None,
    timeout: float = 120.0,
    poll: float = 0.02,
) -> Optional[Point]:
    """Block until ``vk`` is pressed; returns the cursor position at that moment.

    Returns None if the cancel key was pressed or the timeout elapsed.
    """
    get_state = _key_state_fn()
    if get_state is None:
        raise RuntimeError("Keyboard polling is only available on Windows")

    # Drain any state left over from the keypress that started us.
    get_state(vk)
    if cancel_vk:
        get_state(cancel_vk)

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if get_state(vk) & 0x8000:
            position = _cursor_position()
            # Wait for release so one long press is not read as two.
            while get_state(vk) & 0x8000:
                time.sleep(poll)
            return position
        if cancel_vk and get_state(cancel_vk) & 0x8000:
            return None
        time.sleep(poll)
    return None


def snip_region(
    capture: ScreenCapture,
    output: Path,
    *,
    mark_key: str = "f8",
    cancel_key: str = "esc",
    padding: int = 0,
    printer: Callable[[str], None] = print,
) -> Optional[Path]:
    """Interactively grab a rectangle and save it as a template image."""
    mark_vk = resolve_vk(mark_key)
    cancel_vk = resolve_vk(cancel_key)
    if mark_vk is None:
        raise ValueError(f"Unknown mark key {mark_key!r}")

    key_label = mark_key.upper()
    printer(f"Point at the TOP-LEFT corner of the button, then press {key_label} "
            f"(or {cancel_key.upper()} to cancel).")
    first = wait_for_key(mark_vk, cancel_vk=cancel_vk)
    if first is None:
        printer("Cancelled.")
        return None
    printer(f"  top-left = {first}")

    printer(f"Now point at the BOTTOM-RIGHT corner and press {key_label}.")
    second = wait_for_key(mark_vk, cancel_vk=cancel_vk)
    if second is None:
        printer("Cancelled.")
        return None
    printer(f"  bottom-right = {second}")

    left, right = sorted((first.x, second.x))
    top, bottom = sorted((first.y, second.y))
    rect = Rect.from_ltrb(left - padding, top - padding, right + padding, bottom + padding)
    if rect.width < 4 or rect.height < 4:
        printer(f"Region {rect} is too small to be a useful template; aborting.")
        return None

    # Give the user a moment to move the cursor out of the shot.
    time.sleep(0.25)
    image = capture.grab(rect)
    path = save_image(output, image)
    printer(f"Saved {path} ({rect.width}x{rect.height})")
    return path
