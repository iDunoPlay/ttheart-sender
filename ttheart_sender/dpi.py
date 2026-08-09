"""Per-monitor DPI awareness.

Without this, Windows lies to the process about window rectangles and cursor
positions whenever display scaling is not 100%, and every click lands in the
wrong place. Must run before mss / pyautogui / win32 are used for coordinates.
"""

from __future__ import annotations

import logging
import sys

log = logging.getLogger(__name__)

_applied = False

# DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2
_PER_MONITOR_AWARE_V2 = -4


def enable_dpi_awareness() -> bool:
    """Make this process per-monitor DPI aware. Safe to call more than once."""
    global _applied
    if _applied:
        return True
    if not sys.platform.startswith("win"):
        log.debug("DPI awareness skipped: not running on Windows")
        return False

    import ctypes

    # Windows 10 1703+ - the only variant that reports true physical pixels.
    try:
        ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(_PER_MONITOR_AWARE_V2))
        _applied = True
        return True
    except Exception:  # noqa: BLE001 - older Windows, fall through
        pass

    # Windows 8.1+ : 2 == PROCESS_PER_MONITOR_DPI_AWARE
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
        _applied = True
        return True
    except Exception:  # noqa: BLE001
        pass

    # Vista+ : system-DPI aware only, better than nothing.
    try:
        ctypes.windll.user32.SetProcessDPIAware()
        _applied = True
        return True
    except Exception:  # noqa: BLE001
        log.warning("Could not enable DPI awareness; coordinates may be wrong under display scaling")
        return False
