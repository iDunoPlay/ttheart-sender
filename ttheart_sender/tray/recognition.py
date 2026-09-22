"""A window of its own for the live recognition list.

WHY IT IS NOT IN THE CONTROL PANEL. It started there, under the Experiments
tick boxes, and two things went wrong at once. The panel is already 848 logical
px against about 1040px of work area on a 1080p desktop, so the list had to be
capped at five characters and the rest counted -- and five is the whole board
only when nothing is misread, which is the case nobody needs the readout for.
And a board holds forty tsums: the interesting reading is the long tail, which
is exactly what a cap removes.

So the panel keeps only the tick box that opens this, and the list lives here
in full. Removing the five rows also removed what they needed in order to
collapse when empty: recorded control positions, a reflow pass, and a window
resize on every refresh.

A MULTILINE EDIT RATHER THAN A ROW OF LABELS, and that is the whole reason this
module is short. The control panel manages one static per row because each has
to be clicked, greyed or recoloured independently; none of that applies here.
An edit box takes the whole list as one string, scrolls itself when the list is
longer than the window, and needs no layout pass when the number of lines
changes -- which it does on every frame of a round.

Everything here must run on the thread that created the window. The tray posts
to its own message loop to get here; see `TrayApp._refresh`.
"""

from __future__ import annotations

import ctypes
import logging
from typing import Callable, Optional, Sequence

import win32api
import win32con
import win32gui

log = logging.getLogger(__name__)

WINDOW_CLASS = "TTHeartSenderRecognition"

#: Wide enough for "unknown_lightball   12" and a scrollbar, tall enough for a
#: board's worth of characters without scrolling in the normal case.
WIDTH = 260
HEIGHT = 320
MARGIN = 8
#: Sits to the LEFT of the control panel rather than on top of it. Both are
#: parked bottom-right, and a reading you have to move a window to see is one
#: nobody reads.
GAP_FROM_PANEL = 8

ES_MULTILINE = 0x0004
ES_READONLY = 0x0800
ES_AUTOVSCROLL = 0x0040
SW_HIDE = 0
SW_SHOWNOACTIVATE = 4
MONITOR_DEFAULTTONEAREST = 2

EMPTY = ("no round has published yet.",
         "",
         "Start a round with a character",
         "model armed, and this fills in.")


def _dpi_for_window(hwnd: int) -> int:
    try:
        return int(ctypes.windll.user32.GetDpiForWindow(hwnd)) or 96
    except Exception:  # noqa: BLE001 - pre-1607 Windows has no per-window DPI
        return 96


#: FIXED_PITCH | FF_MODERN. The list is name-then-count, so a proportional
#: face would leave the numbers in a ragged column.
_FIXED_MODERN = 1 | 48
_CLEARTYPE = 5
_DEFAULT_CHARSET = 1


def _font(height: int):
    """A fixed-pitch face, so the counts line up down the right of the names.

    `ctypes.windll.gdi32.CreateFontW`, not `win32gui.CreateFont` -- pywin32
    does not export that one, and asking for it raises AttributeError inside
    window creation, where the only symptom is a tick box that appears to do
    nothing. `panel.py` has always built its fonts this way; this did not, and
    that was the whole bug.
    """
    return ctypes.windll.gdi32.CreateFontW(
        -abs(height), 0, 0, 0,
        400,                  # FW_NORMAL
        0, 0, 0,
        _DEFAULT_CHARSET,
        0, 0,                 # default output and clip precision
        _CLEARTYPE,
        _FIXED_MODERN,
        "Consolas",
    )


class RecognitionWindow:
    """A read-only list of what the model is naming, live."""

    def __init__(self, title: str = "Live recognition",
                 on_close: Optional[Callable[[], None]] = None) -> None:
        self._title = title
        #: Called when the user closes the window with its X, so the tick box
        #: that opened it can be put out. Without this the box stays ticked
        #: over a window that is gone, and the next tick does nothing.
        self._on_close = on_close or (lambda: None)
        self._hwnd: Optional[int] = None
        self._edit: Optional[int] = None
        self._font = None
        self._class_atom: Optional[int] = None
        self._scale = 1.0
        self._lines: tuple = ()

    # -- lifecycle -------------------------------------------------------
    def _px(self, value: int) -> int:
        return int(round(value * self._scale))

    def ensure_created(self) -> int:
        if self._hwnd is not None and win32gui.IsWindow(self._hwnd):
            return self._hwnd
        self._create()
        assert self._hwnd is not None
        return self._hwnd

    def _create(self) -> None:
        wc = win32gui.WNDCLASS()
        wc.lpszClassName = WINDOW_CLASS
        wc.hInstance = win32api.GetModuleHandle(None)
        wc.hbrBackground = win32gui.GetSysColorBrush(win32con.COLOR_WINDOW)
        wc.hCursor = win32gui.LoadCursor(0, win32con.IDC_ARROW)
        wc.lpfnWndProc = self._wnd_proc
        try:
            self._class_atom = win32gui.RegisterClass(wc)
        except win32gui.error as exc:
            # ERROR_CLASS_ALREADY_EXISTS: a previous window left it behind.
            if exc.winerror != 1410:
                raise

        style = (win32con.WS_POPUP | win32con.WS_CAPTION
                 | win32con.WS_SYSMENU | win32con.WS_SIZEBOX)
        # TOPMOST is declared HERE, not only asked for later. `_park` also
        # passes HWND_TOPMOST, but SetWindowPos can be refused -- under the
        # full test suite it silently was, leaving ex-style 0x180 with no
        # topmost bit -- and a readout the game covers is the exact bug this
        # window was reported for. Declared at creation it cannot be lost.
        self._hwnd = win32gui.CreateWindowEx(
            win32con.WS_EX_TOOLWINDOW | win32con.WS_EX_TOPMOST,
            WINDOW_CLASS, self._title, style,
            0, 0, 100, 100, 0, 0, wc.hInstance, None)
        self._scale = _dpi_for_window(self._hwnd) / 96.0
        self._font = _font(self._px(13))
        self._edit = win32gui.CreateWindowEx(
            0, "EDIT", "",
            win32con.WS_CHILD | win32con.WS_VISIBLE | win32con.WS_VSCROLL
            | ES_MULTILINE | ES_READONLY | ES_AUTOVSCROLL,
            0, 0, 10, 10, self._hwnd, 0, wc.hInstance, None)
        win32gui.SendMessage(self._edit, win32con.WM_SETFONT, self._font, 1)
        win32gui.SetWindowPos(
            self._hwnd, 0, 0, 0, self._px(WIDTH), self._px(HEIGHT),
            win32con.SWP_NOMOVE | win32con.SWP_NOZORDER
            | win32con.SWP_NOACTIVATE)
        self._layout()
        self.set_lines(self._lines)

    def destroy(self) -> None:
        hwnd, self._hwnd = self._hwnd, None
        self._edit = None
        if hwnd is not None and win32gui.IsWindow(hwnd):
            win32gui.DestroyWindow(hwnd)
        if self._font:
            ctypes.windll.gdi32.DeleteObject(self._font)
            self._font = None
        if self._class_atom is not None:
            try:
                win32gui.UnregisterClass(WINDOW_CLASS, None)
            except win32gui.error:
                pass
            self._class_atom = None

    # -- visibility ------------------------------------------------------
    @property
    def visible(self) -> bool:
        hwnd = self._hwnd
        return (hwnd is not None and win32gui.IsWindow(hwnd)
                and bool(win32gui.IsWindowVisible(hwnd)))

    def show(self, beside=None) -> None:
        hwnd = self.ensure_created()
        self._park(beside)
        # NOACTIVATE: this is a readout watched while the emulator has focus,
        # and stealing focus from the game is how a round gets lost.
        win32gui.ShowWindow(hwnd, SW_SHOWNOACTIVATE)

    def hide(self) -> None:
        if self._hwnd is not None and win32gui.IsWindow(self._hwnd):
            win32gui.ShowWindow(self._hwnd, SW_HIDE)

    def _park(self, beside) -> None:
        """Left of the control panel, or bottom-right if there is not one."""
        try:
            monitor = win32api.MonitorFromWindow(self._hwnd,
                                                 MONITOR_DEFAULTTONEAREST)
            left, top, right, bottom = win32api.GetMonitorInfo(monitor)["Work"]
        except Exception:  # noqa: BLE001 - fall back to the primary screen
            left, top = 0, 0
            right = win32api.GetSystemMetrics(win32con.SM_CXSCREEN)
            bottom = win32api.GetSystemMetrics(win32con.SM_CYSCREEN)
        _, _, w, h = _size(self._hwnd)
        x, y = right - w - self._px(8), bottom - h - self._px(8)
        if beside is not None and win32gui.IsWindow(beside):
            pl, _pt, _pr, pb = win32gui.GetWindowRect(beside)
            x, y = pl - w - self._px(GAP_FROM_PANEL), pb - h
        # TOPMOST, and not optional. The whole use of this window is reading it
        # while the emulator has focus and a round is being played -- and a
        # window the game covers is not a readout. It was shown NOACTIVATE
        # without this, which put it behind the game the moment it opened: the
        # tick box looked like it had done nothing.
        win32gui.SetWindowPos(
            self._hwnd, win32con.HWND_TOPMOST, max(left, x), max(top, y), 0, 0,
            win32con.SWP_NOSIZE | win32con.SWP_NOACTIVATE)

    # -- content ---------------------------------------------------------
    def set_lines(self, lines: Sequence[str]) -> None:
        """Replace the whole list. Cheap enough to call every frame."""
        self._lines = tuple(str(line) for line in lines)
        if self._edit and win32gui.IsWindow(self._edit):
            # CRLF, not LF: a Win32 edit control renders a lone newline as a
            # box rather than a line break.
            win32gui.SetWindowText(
                self._edit, "\r\n".join(self._lines or EMPTY))

    # -- messages --------------------------------------------------------
    def _layout(self) -> None:
        if not (self._hwnd and self._edit and win32gui.IsWindow(self._hwnd)):
            return
        _l, _t, w, h = _size(self._hwnd, client=True)
        m = self._px(MARGIN)
        win32gui.MoveWindow(self._edit, m, m, max(10, w - 2 * m),
                            max(10, h - 2 * m), True)

    def _wnd_proc(self, hwnd: int, msg: int, wparam: int, lparam: int) -> int:
        if msg == win32con.WM_SIZE:
            self._layout()
            return 0
        if msg == win32con.WM_CLOSE:
            # Hidden, never destroyed: the tick box owns its lifetime, and a
            # destroyed window would have to be rebuilt on the next tick.
            self.hide()
            try:
                self._on_close()
            except Exception:  # noqa: BLE001 - a callback must not wedge the X
                log.debug("recognition on_close failed", exc_info=True)
            return 0
        if msg == win32con.WM_DESTROY:
            self._hwnd = None
            self._edit = None
            return 0
        return win32gui.DefWindowProc(hwnd, msg, wparam, lparam)


def _size(hwnd: int, client: bool = False):
    left, top, right, bottom = (win32gui.GetClientRect(hwnd) if client
                                else win32gui.GetWindowRect(hwnd))
    return left, top, right - left, bottom - top
