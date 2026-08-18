"""The control panel window.

A native Win32 window rather than tkinter or Qt, for the same reason
:mod:`.icon` is: the tray already owns a message loop on this thread, and every
window created here is pumped by it. A second toolkit would mean a second event
loop to reconcile, plus its runtime inside the .exe -- pywin32 is already there.

Everything in this module must run on the thread that created the panel;
:class:`~.icon.TrayIcon.post` is how other threads get here.
"""

from __future__ import annotations

import ctypes
import logging
from ctypes import wintypes
from typing import Callable, Dict, List, Optional

import win32api
import win32con
import win32gui

from .settings import PURCHASE_BOXES

log = logging.getLogger(__name__)

WINDOW_CLASS = "TTHeartSenderPanel"

# -- control ids ----------------------------------------------------------
ID_ALWAYS_ON_TOP = 2001
ID_AUTO_PLAY = 2002
ID_BUY = 2003
ID_RUN = 2004
ID_EXIT = 2005
ID_LOGS = 2006
ID_MODE_BASE = 2100
ID_PURCHASE_BASE = 2200

# -- layout, in logical pixels (scaled by the monitor's DPI) --------------
PANEL_WIDTH = 300
MARGIN = 12
ROW = 22
GAP = 6
SECTION_GAP = 10
BUTTON_HEIGHT = 30
LINE_HEIGHT = 2

BS_AUTOCHECKBOX = 0x00000003
BS_AUTORADIOBUTTON = 0x00000009
BS_PUSHBUTTON = 0x00000000
SS_ETCHEDHORZ = 0x00000010
BM_GETCHECK = 0x00F0
BM_SETCHECK = 0x00F1
SW_HIDE = 0
SW_SHOW = 5
MONITOR_DEFAULTTONEAREST = 2


def _scaled_font(height: int, *, bold: bool = False) -> int:
    """A Segoe UI font handle at ``height`` device pixels."""
    return ctypes.windll.gdi32.CreateFontW(
        -abs(height), 0, 0, 0,
        700 if bold else 400,
        0, 0, 0,
        1,  # DEFAULT_CHARSET
        0, 0,
        5,  # CLEARTYPE_QUALITY
        0,
        "Segoe UI",
    )


def _dpi_for_window(hwnd: int) -> int:
    """Monitor DPI, falling back to the desktop's on pre-1607 Windows."""
    try:
        dpi = ctypes.windll.user32.GetDpiForWindow(hwnd)
        if dpi:
            return int(dpi)
    except Exception:  # noqa: BLE001 - older Windows
        pass
    screen = win32gui.GetDC(0)
    try:
        return int(win32gui.GetDeviceCaps(screen, win32con.LOGPIXELSY))
    finally:
        win32gui.ReleaseDC(0, screen)


class ControlPanel:
    """Small always-available window: pick a mode, tick boxes, run.

    The panel owns no state of its own. Every control reads from the callbacks
    handed in at construction and writes back through them, so the panel and
    the tray menu can never disagree about what is selected.
    """

    def __init__(
        self,
        *,
        title: str,
        version_text: str,
        modes,
        get_state: Callable[[], Dict],
        on_mode: Callable[[str], None],
        on_toggle: Callable[[str, bool], None],
        on_purchase: Callable[[str, bool], None],
        on_run: Callable[[], None],
        on_buy: Callable[[], None],
        on_logs: Callable[[], None],
        on_exit: Callable[[], None],
    ) -> None:
        self._title = title
        self._version_text = version_text
        self._modes = list(modes)
        self._get_state = get_state
        self._on_mode = on_mode
        self._on_toggle = on_toggle
        self._on_purchase = on_purchase
        self._on_run = on_run
        self._on_buy = on_buy
        self._on_logs = on_logs
        self._on_exit = on_exit

        self._hwnd: Optional[int] = None
        self._class_atom: Optional[int] = None
        self._controls: Dict[int, int] = {}
        self._fonts: List[int] = []
        self._background = win32gui.GetSysColorBrush(win32con.COLOR_BTNFACE)
        self._scale = 1.0

    # -- lifecycle -------------------------------------------------------
    def _px(self, value: int) -> int:
        return int(round(value * self._scale))

    def ensure_created(self) -> int:
        if self._hwnd is not None and win32gui.IsWindow(self._hwnd):
            return self._hwnd
        self._create()
        assert self._hwnd is not None
        return self._hwnd

    def destroy(self) -> None:
        hwnd, self._hwnd = self._hwnd, None
        if hwnd is not None and win32gui.IsWindow(hwnd):
            win32gui.DestroyWindow(hwnd)
        for font in self._fonts:
            ctypes.windll.gdi32.DeleteObject(font)
        self._fonts.clear()
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
        return hwnd is not None and win32gui.IsWindow(hwnd) and win32gui.IsWindowVisible(hwnd)

    def show(self) -> None:
        hwnd = self.ensure_created()
        self.refresh()
        self._position_bottom_right()
        win32gui.ShowWindow(hwnd, SW_SHOW)
        # A tray click leaves the foreground with whatever the user was using,
        # so ask for it explicitly or the panel opens behind that window.
        try:
            win32gui.SetForegroundWindow(hwnd)
        except win32gui.error:
            log.debug("SetForegroundWindow refused", exc_info=True)

    def hide(self) -> None:
        if self._hwnd is not None and win32gui.IsWindow(self._hwnd):
            win32gui.ShowWindow(self._hwnd, SW_HIDE)

    def toggle(self) -> None:
        if self.visible:
            self.hide()
        else:
            self.show()

    # -- construction ----------------------------------------------------
    def _create(self) -> None:
        wc = win32gui.WNDCLASS()
        wc.lpszClassName = WINDOW_CLASS
        wc.hInstance = win32api.GetModuleHandle(None)
        wc.hbrBackground = self._background
        wc.hCursor = win32gui.LoadCursor(0, win32con.IDC_ARROW)
        wc.lpfnWndProc = self._wnd_proc
        try:
            self._class_atom = win32gui.RegisterClass(wc)
        except win32gui.error as exc:
            # ERROR_CLASS_ALREADY_EXISTS: a previous panel left it behind.
            if exc.winerror != 1410:
                raise

        style = win32con.WS_POPUP | win32con.WS_CAPTION | win32con.WS_SYSMENU
        ex_style = win32con.WS_EX_TOOLWINDOW  # no taskbar button; it lives in the tray
        self._hwnd = win32gui.CreateWindowEx(
            ex_style, WINDOW_CLASS, self._title, style,
            0, 0, 100, 100, 0, 0, wc.hInstance, None,
        )

        dpi = _dpi_for_window(self._hwnd)
        self._scale = dpi / 96.0
        self._fonts = [_scaled_font(self._px(12)), _scaled_font(self._px(12), bold=True)]
        self._build_controls()

    def _build_controls(self) -> None:
        y = MARGIN
        y = self._add_static(self._version_text, y, bold=True)
        y += GAP
        y = self._add_check(ID_ALWAYS_ON_TOP, "Always on top", y)
        y += GAP

        y = self._add_static("Mode", y)
        # One group: BS_AUTORADIOBUTTON only unchecks siblings up to the next
        # WS_GROUP, so the first entry carries it and the rest must not.
        width = (PANEL_WIDTH - 2 * MARGIN) // 3
        for index, mode in enumerate(self._modes):
            extra = win32con.WS_GROUP if index == 0 else 0
            self._add_control(
                ID_MODE_BASE + index, "BUTTON", mode.label,
                BS_AUTORADIOBUTTON | extra,
                MARGIN + index * width, y, width, ROW,
            )
        y += ROW + GAP

        y = self._add_check(ID_AUTO_PLAY, "Auto Play", y)
        y += SECTION_GAP
        y = self._add_line(y)
        y += SECTION_GAP

        y = self._add_static("Purchase box", y, bold=True)
        y += GAP
        for index, (key, label, _default) in enumerate(PURCHASE_BOXES):
            y = self._add_check(ID_PURCHASE_BASE + index, label, y)

        y += SECTION_GAP
        y = self._add_line(y)
        y += SECTION_GAP

        self._add_control(
            ID_BUY, "BUTTON", "Buy tsum", BS_PUSHBUTTON,
            MARGIN, y, PANEL_WIDTH - 2 * MARGIN, BUTTON_HEIGHT,
        )
        y += BUTTON_HEIGHT + GAP
        self._add_control(
            ID_RUN, "BUTTON", "Run", BS_PUSHBUTTON,
            MARGIN, y, PANEL_WIDTH - 2 * MARGIN, BUTTON_HEIGHT,
        )
        y += BUTTON_HEIGHT + GAP

        # Right-clicking the tray icon no longer opens a menu, so the panel is
        # the only way out of the app -- these two are not optional extras.
        half = (PANEL_WIDTH - 2 * MARGIN - GAP) // 2
        self._add_control(ID_LOGS, "BUTTON", "Open logs", BS_PUSHBUTTON,
                          MARGIN, y, half, ROW + 4)
        self._add_control(ID_EXIT, "BUTTON", "Exit", BS_PUSHBUTTON,
                          MARGIN + half + GAP, y, half, ROW + 4)
        y += ROW + 4 + MARGIN

        self._resize(y)

    def _add_control(self, ident: int, cls: str, text: str, style: int,
                     x: int, y: int, width: int, height: int) -> int:
        hwnd = win32gui.CreateWindowEx(
            0, cls, text,
            win32con.WS_CHILD | win32con.WS_VISIBLE | style,
            self._px(x), self._px(y), self._px(width), self._px(height),
            self._hwnd, ident, win32api.GetModuleHandle(None), None,
        )
        win32gui.SendMessage(hwnd, win32con.WM_SETFONT, self._fonts[0], 1)
        self._controls[ident] = hwnd
        return hwnd

    def _add_static(self, text: str, y: int, *, bold: bool = False) -> int:
        hwnd = win32gui.CreateWindowEx(
            0, "STATIC", text, win32con.WS_CHILD | win32con.WS_VISIBLE,
            self._px(MARGIN), self._px(y), self._px(PANEL_WIDTH - 2 * MARGIN), self._px(ROW),
            self._hwnd, 0, win32api.GetModuleHandle(None), None,
        )
        win32gui.SendMessage(hwnd, win32con.WM_SETFONT, self._fonts[1 if bold else 0], 1)
        return y + ROW

    def _add_check(self, ident: int, label: str, y: int) -> int:
        self._add_control(ident, "BUTTON", label, BS_AUTOCHECKBOX,
                          MARGIN, y, PANEL_WIDTH - 2 * MARGIN, ROW)
        return y + ROW

    def _add_line(self, y: int) -> int:
        win32gui.CreateWindowEx(
            0, "STATIC", "", win32con.WS_CHILD | win32con.WS_VISIBLE | SS_ETCHEDHORZ,
            self._px(MARGIN), self._px(y), self._px(PANEL_WIDTH - 2 * MARGIN), self._px(LINE_HEIGHT),
            self._hwnd, 0, win32api.GetModuleHandle(None), None,
        )
        return y + LINE_HEIGHT

    def _resize(self, content_height: int) -> None:
        """Grow the window so ``content_height`` logical px fit in the client area."""
        rect = wintypes.RECT(0, 0, self._px(PANEL_WIDTH), self._px(content_height))
        style = win32gui.GetWindowLong(self._hwnd, win32con.GWL_STYLE)
        ex_style = win32gui.GetWindowLong(self._hwnd, win32con.GWL_EXSTYLE)
        ctypes.windll.user32.AdjustWindowRectEx(
            ctypes.byref(rect), wintypes.DWORD(style), False, wintypes.DWORD(ex_style)
        )
        win32gui.SetWindowPos(
            self._hwnd, 0, 0, 0,
            rect.right - rect.left, rect.bottom - rect.top,
            win32con.SWP_NOMOVE | win32con.SWP_NOZORDER | win32con.SWP_NOACTIVATE,
        )

    def _work_area(self):
        """Desktop minus the taskbar, on the monitor the panel is on.

        The monitor's work area rather than SPI_GETWORKAREA: pywin32 does not
        implement that call, and the per-monitor answer is the correct one on a
        multi-screen desktop anyway.
        """
        try:
            monitor = win32api.MonitorFromWindow(self._hwnd, MONITOR_DEFAULTTONEAREST)
            return win32api.GetMonitorInfo(monitor)["Work"]
        except Exception:  # noqa: BLE001 - fall back to the primary screen
            width = win32api.GetSystemMetrics(win32con.SM_CXSCREEN)
            height = win32api.GetSystemMetrics(win32con.SM_CYSCREEN)
            return (0, 0, width, height)

    def _position_bottom_right(self) -> None:
        """Park above the taskbar, in the working area's bottom-right corner."""
        left, top, right, bottom = self._work_area()
        _, _, width, height = _window_size(self._hwnd)
        margin = self._px(8)
        win32gui.SetWindowPos(
            self._hwnd,
            win32con.HWND_TOPMOST if self._always_on_top() else win32con.HWND_NOTOPMOST,
            max(left, right - width - margin),
            max(top, bottom - height - margin),
            0, 0,
            win32con.SWP_NOSIZE | win32con.SWP_NOACTIVATE,
        )

    def _always_on_top(self) -> bool:
        try:
            return bool(self._get_state().get("always_on_top", True))
        except Exception:  # noqa: BLE001 - never let a callback break layout
            log.debug("state callback failed", exc_info=True)
            return True

    # -- state -----------------------------------------------------------
    def refresh(self) -> None:
        """Push the current state into every control."""
        if self._hwnd is None or not win32gui.IsWindow(self._hwnd):
            return
        state = self._get_state()

        self._set_check(ID_ALWAYS_ON_TOP, state.get("always_on_top", True))
        self._set_check(ID_AUTO_PLAY, state.get("auto_play", False))
        for index, mode in enumerate(self._modes):
            self._set_check(ID_MODE_BASE + index, mode.key == state.get("mode"))
        purchase = state.get("purchase", {})
        for index, (key, _label, default) in enumerate(PURCHASE_BOXES):
            self._set_check(ID_PURCHASE_BASE + index, purchase.get(key, default))

        running = bool(state.get("running"))
        stopping = bool(state.get("stopping"))
        status = state.get("status", "")
        self._set_text(ID_RUN, ("Stop" if running else "Run") + (f"  -  {status}" if status else ""))
        # Stopping already asked the run to unwind; a second click would do
        # nothing, and Run must not restart it before the worker has finished.
        self._enable(ID_RUN, not stopping)
        self._enable(ID_BUY, not (running or stopping))
        self._apply_topmost(state.get("always_on_top", True))

    def _apply_topmost(self, enabled: bool) -> None:
        win32gui.SetWindowPos(
            self._hwnd,
            win32con.HWND_TOPMOST if enabled else win32con.HWND_NOTOPMOST,
            0, 0, 0, 0,
            win32con.SWP_NOMOVE | win32con.SWP_NOSIZE | win32con.SWP_NOACTIVATE,
        )

    def _set_check(self, ident: int, checked: bool) -> None:
        hwnd = self._controls.get(ident)
        if hwnd:
            win32gui.SendMessage(hwnd, BM_SETCHECK, 1 if checked else 0, 0)

    def _get_check(self, ident: int) -> bool:
        hwnd = self._controls.get(ident)
        return bool(hwnd and win32gui.SendMessage(hwnd, BM_GETCHECK, 0, 0))

    def _set_text(self, ident: int, text: str) -> None:
        hwnd = self._controls.get(ident)
        if hwnd:
            win32gui.SetWindowText(hwnd, text)

    def _enable(self, ident: int, enabled: bool) -> None:
        hwnd = self._controls.get(ident)
        if hwnd:
            win32gui.EnableWindow(hwnd, bool(enabled))

    # -- messages --------------------------------------------------------
    def _wnd_proc(self, hwnd: int, msg: int, wparam: int, lparam: int) -> int:
        if msg == win32con.WM_COMMAND:
            self._on_command(win32api.LOWORD(wparam))
            return 0
        if msg in (win32con.WM_CTLCOLORSTATIC, win32con.WM_CTLCOLORBTN):
            # Static/checkbox text paints on white otherwise, which looks like
            # a rendering bug against the dialog-grey background.
            ctypes.windll.gdi32.SetBkMode(wparam, 1)  # TRANSPARENT
            return self._background
        if msg == win32con.WM_CLOSE:
            # The tray icon is the app's life-cycle owner: closing the panel
            # only puts it away.
            self.hide()
            return 0
        if msg == win32con.WM_DESTROY:
            self._hwnd = None
            return 0
        return win32gui.DefWindowProc(hwnd, msg, wparam, lparam)

    def _on_command(self, ident: int) -> None:
        try:
            if ident == ID_ALWAYS_ON_TOP:
                self._on_toggle("always_on_top", self._get_check(ident))
            elif ident == ID_AUTO_PLAY:
                self._on_toggle("auto_play", self._get_check(ident))
            elif ident == ID_BUY:
                self._on_buy()
            elif ident == ID_RUN:
                self._on_run()
            elif ident == ID_LOGS:
                self._on_logs()
            elif ident == ID_EXIT:
                self._on_exit()
            elif ID_MODE_BASE <= ident < ID_MODE_BASE + len(self._modes):
                self._on_mode(self._modes[ident - ID_MODE_BASE].key)
            elif ID_PURCHASE_BASE <= ident < ID_PURCHASE_BASE + len(PURCHASE_BOXES):
                key = PURCHASE_BOXES[ident - ID_PURCHASE_BASE][0]
                self._on_purchase(key, self._get_check(ident))
            else:
                return
        except Exception:  # noqa: BLE001 - a handler must not kill the loop
            log.exception("Panel command %s failed", ident)
        self.refresh()


def _window_size(hwnd: int):
    left, top, right, bottom = win32gui.GetWindowRect(hwnd)
    return left, top, right - left, bottom - top
