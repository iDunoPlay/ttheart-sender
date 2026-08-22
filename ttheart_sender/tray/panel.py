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

from ..version import __version__
from .settings import (
    CLAIM_PATTERN_DEFAULT,
    CLAIM_PATTERNS,
    MINUTE_MAX,
    MINUTE_MIN,
    PURCHASE_BOXES,
    RETURN_HEART_MARKS,
    RETURN_HEART_MINUTES_DEFAULT,
    clamp_minute,
)

log = logging.getLogger(__name__)

WINDOW_CLASS = "TTHeartSenderPanel"

# -- control ids ----------------------------------------------------------
ID_ALWAYS_ON_TOP = 2001
ID_AUTO_PLAY = 2002
ID_BUY = 2003
ID_RUN = 2004
ID_EXIT = 2005
ID_LOGS = 2006
ID_COLLECT_DATA = 2009
ID_RETURN_HEART = 2010
ID_AUTO_UPDATE = 2011
ID_UPDATE = 2012
ID_RESTART_STUCK = 2016
#: A label rather than a control, but it is rewritten on every refresh, so it
#: needs an id to be found again.
ID_UPDATE_STATUS = 2013
#: The big RUNNING/PAUSE word. A label, but recoloured and rewritten on every
#: refresh, so it needs an id.
ID_RUN_STATE = 2015
ID_MODE_BASE = 2100
ID_PURCHASE_BASE = 2200
#: One radio per claim pattern, in :data:`CLAIM_PATTERNS` order.
ID_CLAIM_BASE = 2500
#: One edit box per Return Heart mark, each with an up-down glued to it. The
#: two ranges run in parallel, so the spinner belonging to an edit is always at
#: the same offset -- see :func:`_spin_for`.
ID_MINUTE_BASE = 2300
ID_MINUTE_SPIN_BASE = 2400


def _spin_for(ident: int) -> int:
    """The up-down that belongs to a minute edit box."""
    return ID_MINUTE_SPIN_BASE + (ident - ID_MINUTE_BASE)


# -- layout, in logical pixels (scaled by the monitor's DPI) --------------
PANEL_WIDTH = 300
MARGIN = 12
ROW = 22
GAP = 6
SECTION_GAP = 10
BUTTON_HEIGHT = 30
LINE_HEIGHT = 2
#: The "15" box plus the arrows glued to its right edge, and the "min" after it.
SPINNER_WIDTH = 54
UNIT_LABEL_WIDTH = 30
#: How far the Return Heart spinners sit in from its tick box, so they read as
#: belonging to it rather than as two more rows of the panel.
INDENT = 16
#: "Buy tsum" sits on the "Purchase box" header line, right-aligned.
BUY_WIDTH = 86
HEADER_HEIGHT = 26
#: The update button shares its line with the Auto Update tick box, the same
#: way "Buy tsum" shares one with the Purchase box heading.
UPDATE_WIDTH = 86
#: The line under it that says what the updater is doing.
STATUS_HEIGHT = 18
#: The RUNNING / PAUSE word above the Run button. Tall enough that the state
#: is readable from across the desk, which is the point of it.
STATE_HEIGHT = 30
STATE_FONT = 20
#: Colours for that word. Green while a run is going, red while it is not,
#: amber while it is unwinding -- a state that is neither and lasts long
#: enough to be worth naming.
COLOUR_RUNNING = 0x0000A000   # COLORREF is 0x00BBGGRR, not RGB
COLOUR_PAUSED = 0x000000C0
COLOUR_STOPPING = 0x000080C0

BS_AUTOCHECKBOX = 0x00000003
BS_AUTORADIOBUTTON = 0x00000009
BS_PUSHBUTTON = 0x00000000
SS_ETCHEDHORZ = 0x00000010
SS_CENTERIMAGE = 0x00000200  # vertically centres a single line of text
SS_CENTER = 0x00000001       # ...and this centres it horizontally
BM_GETCHECK = 0x00F0
BM_SETCHECK = 0x00F1
SW_HIDE = 0
SW_SHOW = 5
MONITOR_DEFAULTTONEAREST = 2

# -- edit box + up-down (spinner) ----------------------------------------
ES_RIGHT = 0x00000002
ES_AUTOHSCROLL = 0x00000080
ES_NUMBER = 0x00002000  # digits only, so the text is always parseable
EN_CHANGE = 0x0300
EN_KILLFOCUS = 0x0200

UPDOWN_CLASS = "msctls_updown32"
UDS_SETBUDDYINT = 0x0002   # write the value into the buddy edit as a number
UDS_ALIGNRIGHT = 0x0004    # glue the arrows to the buddy's right edge
UDS_AUTOBUDDY = 0x0010     # adopt the control created just before this one
UDS_ARROWKEYS = 0x0020     # up/down keys work while the edit has focus
UDS_NOTHOUSANDS = 0x0080
UDM_SETRANGE32 = 0x046F
UDM_SETPOS32 = 0x0471


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
        modes,
        get_state: Callable[[], Dict],
        on_mode: Callable[[str], None],
        on_toggle: Callable[[str, bool], None],
        on_return_minute: Callable[[int, int], None],
        on_claim: Callable[[str], None],
        on_purchase: Callable[[str, bool], None],
        on_run: Callable[[], None],
        on_buy: Callable[[], None],
        on_update: Callable[[], None],
        on_logs: Callable[[], None],
        on_exit: Callable[[], None],
    ) -> None:
        #: The version lives in the caption bar, so a screenshot of the panel
        #: is enough to tell which build produced it.
        self._title = title
        self._modes = list(modes)
        self._get_state = get_state
        self._on_mode = on_mode
        self._on_toggle = on_toggle
        self._on_return_minute = on_return_minute
        self._on_claim = on_claim
        self._on_purchase = on_purchase
        self._on_run = on_run
        self._on_buy = on_buy
        self._on_update = on_update
        self._on_logs = on_logs
        self._on_exit = on_exit

        self._hwnd: Optional[int] = None
        self._class_atom: Optional[int] = None
        self._controls: Dict[int, int] = {}
        self._fonts: List[int] = []
        self._background = win32gui.GetSysColorBrush(win32con.COLOR_BTNFACE)
        self._scale = 1.0
        #: True while :meth:`refresh` is writing into a spinner. Its EN_CHANGE
        #: would otherwise be read back as if the user had typed it.
        self._writing_number = False
        #: Colour of the RUNNING/PAUSE word. Set by :meth:`refresh` and read
        #: back in WM_CTLCOLORSTATIC, which is the only place Win32 lets a
        #: static label's text colour be chosen.
        self._state_colour = COLOUR_PAUSED

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
        # The up-down (spinner) class lives in comctl32 and is only registered
        # once the DLL has been initialised -- CreateWindowEx would otherwise
        # fail with "cannot find window class".
        try:
            ctypes.windll.comctl32.InitCommonControls()
        except Exception:  # noqa: BLE001 - the spinner degrades to a plain box
            log.debug("InitCommonControls failed", exc_info=True)

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
        self._fonts = [_scaled_font(self._px(12)), _scaled_font(self._px(12), bold=True),
                       _scaled_font(self._px(STATE_FONT), bold=True)]
        # Creating a number box and giving its spinner a range both fire
        # EN_CHANGE. Read back as user input they would report the layout's
        # placeholder as a choice and overwrite the saved value with it.
        self._writing_number = True
        try:
            self._build_controls()
        finally:
            self._writing_number = False

    def _build_controls(self) -> None:
        y = MARGIN
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
        # Sits with Auto Play rather than in its own section: both decide what
        # a running cycle is allowed to do on its own.
        y = self._add_check(ID_RESTART_STUCK, "Restart when stucked", y)
        y += SECTION_GAP
        y = self._add_line(y)
        y += SECTION_GAP

        y = self._add_return_heart(y)
        y += SECTION_GAP
        y = self._add_line(y)
        y += SECTION_GAP

        y = self._add_claim_pattern(y)
        y += SECTION_GAP
        y = self._add_line(y)
        y += SECTION_GAP

        y = self._add_purchase_header(y)
        y += GAP
        for index, (key, label, _default) in enumerate(PURCHASE_BOXES):
            y = self._add_check(ID_PURCHASE_BASE + index, label, y)

        y += SECTION_GAP
        y = self._add_line(y)
        y += SECTION_GAP

        # One row, its own section: this writes files to disk, which is worth
        # separating from the toggles that only change how a round is played.
        y = self._add_check(ID_COLLECT_DATA, "Data collection", y, bold=True)

        y += SECTION_GAP
        y = self._add_line(y)
        y += SECTION_GAP

        # The state word sits above the button rather than on it: the button
        # is disabled while a run is going, and a disabled control greys its
        # own text, which is exactly the moment the state most needs to be
        # readable.
        state = self._add_static("PAUSE", y, ident=ID_RUN_STATE,
                                 height=STATE_HEIGHT, centred=True,
                                 align_centre=True)
        win32gui.SendMessage(self._controls[ID_RUN_STATE], win32con.WM_SETFONT,
                             self._fonts[2], 1)
        y = state + GAP

        self._add_control(
            ID_RUN, "BUTTON", "Run", BS_PUSHBUTTON,
            MARGIN, y, PANEL_WIDTH - 2 * MARGIN, BUTTON_HEIGHT,
        )
        y += BUTTON_HEIGHT + GAP

        # The tray menu only offers Exit, so Open logs lives here -- and Exit
        # is repeated here because this is where the eye already is.
        half = (PANEL_WIDTH - 2 * MARGIN - GAP) // 2
        self._add_control(ID_LOGS, "BUTTON", "Open logs", BS_PUSHBUTTON,
                          MARGIN, y, half, ROW + 4)
        self._add_control(ID_EXIT, "BUTTON", "Exit", BS_PUSHBUTTON,
                          MARGIN + half + GAP, y, half, ROW + 4)
        y += ROW + 4 + SECTION_GAP

        # Housekeeping, so it sits under the buttons rather than between the
        # switches that decide what a run does.
        y = self._add_line(y)
        y += SECTION_GAP
        y = self._add_update(y)
        y += MARGIN

        self._resize(y)

    def _add_update(self, y: int) -> int:
        """The "Auto Update" tick box, its button, and the status line."""
        button_x = PANEL_WIDTH - MARGIN - UPDATE_WIDTH
        check = self._add_control(
            ID_AUTO_UPDATE, "BUTTON", "Auto Update", BS_AUTOCHECKBOX,
            MARGIN, y, button_x - MARGIN - GAP, HEADER_HEIGHT,
        )
        # Bold for the same reason "Return Heart" is: it heads a section
        # rather than continuing the list of run options above it.
        win32gui.SendMessage(check, win32con.WM_SETFONT, self._fonts[1], 1)
        self._add_control(ID_UPDATE, "BUTTON", "Check", BS_PUSHBUTTON,
                          button_x, y, UPDATE_WIDTH, HEADER_HEIGHT)
        y += HEADER_HEIGHT
        # Indented under the tick box, like the Return Heart marks are: it
        # reports on that switch rather than standing on its own.
        return self._add_static(
            f"v{__version__}", y, ident=ID_UPDATE_STATUS,
            x=MARGIN + INDENT, width=PANEL_WIDTH - MARGIN - (MARGIN + INDENT),
            height=STATUS_HEIGHT, centred=True,
        )

    #: What each Return Heart row is called. Read together they spell the
    #: schedule out: "every hour at 15 min, and at 50 min".
    MINUTE_LABELS = ("Every hour at", "and at")

    def _add_return_heart(self, y: int) -> int:
        """The "Return Heart" tick box and the two marks it sends on."""
        y = self._add_check(ID_RETURN_HEART, "Return Heart", y, bold=True)
        for index in range(RETURN_HEART_MARKS):
            y = self._add_minute_row(
                ID_MINUTE_BASE + index,
                self.MINUTE_LABELS[index],
                RETURN_HEART_MINUTES_DEFAULT[index],
                y,
            )
        return y

    def _add_claim_pattern(self, y: int) -> int:
        """The "Claim pattern" heading and the two ways to empty a mailbox."""
        y = self._add_static("Claim pattern", y, bold=True)
        # WS_GROUP on the first one for the same reason the Mode radios carry
        # it: an auto-radio clears its siblings up to the next WS_GROUP, so
        # without it picking a pattern would put out the selected mode.
        # Indented under the heading, like the Return Heart marks are.
        left = MARGIN + INDENT
        width = (PANEL_WIDTH - MARGIN - left) // len(CLAIM_PATTERNS)
        for index, (_key, label, _flag) in enumerate(CLAIM_PATTERNS):
            extra = win32con.WS_GROUP if index == 0 else 0
            self._add_control(
                ID_CLAIM_BASE + index, "BUTTON", label,
                BS_AUTORADIOBUTTON | extra,
                left + index * width, y, width, ROW,
            )
        return y + ROW

    def _add_minute_row(self, ident: int, label: str, value: int, y: int) -> int:
        """One indented "<label> [15] min" row, aligned with its fellows."""
        unit_x = PANEL_WIDTH - MARGIN - UNIT_LABEL_WIDTH
        spinner_x = unit_x - SPINNER_WIDTH
        label_x = MARGIN + INDENT
        self._add_static(label, y, x=label_x, width=spinner_x - label_x - GAP, centred=True)
        # The edit is created first on purpose: UDS_AUTOBUDDY adopts whichever
        # control was created immediately before the up-down.
        self._add_control(
            ident, "EDIT", str(value),
            win32con.WS_BORDER | win32con.WS_TABSTOP | ES_NUMBER | ES_RIGHT | ES_AUTOHSCROLL,
            spinner_x, y, SPINNER_WIDTH, ROW,
        )
        self._add_spinner(_spin_for(ident), value)
        self._add_static("min", y, x=unit_x + 4, width=UNIT_LABEL_WIDTH, centred=True)
        return y + ROW

    def _add_spinner(self, ident: int, value: int) -> None:
        """Attach up-down arrows to the box just created, if comctl32 allows."""
        try:
            hwnd = win32gui.CreateWindowEx(
                0, UPDOWN_CLASS, "",
                win32con.WS_CHILD | win32con.WS_VISIBLE
                | UDS_AUTOBUDDY | UDS_ALIGNRIGHT | UDS_SETBUDDYINT
                | UDS_ARROWKEYS | UDS_NOTHOUSANDS,
                0, 0, 0, 0,  # UDS_ALIGNRIGHT sizes and places it off the buddy
                self._hwnd, ident, win32api.GetModuleHandle(None), None,
            )
        except win32gui.error:
            # Without the arrows the box is still a perfectly usable number
            # field, so this is a downgrade rather than a failure.
            log.warning("Could not create the spinner for control %s", ident, exc_info=True)
            return
        self._controls[ident] = hwnd
        win32gui.SendMessage(hwnd, UDM_SETRANGE32, MINUTE_MIN, MINUTE_MAX)
        win32gui.SendMessage(hwnd, UDM_SETPOS32, 0, value)

    def _add_purchase_header(self, y: int) -> int:
        """The "Purchase box" heading, with "Buy tsum" on the same line."""
        buy_x = PANEL_WIDTH - MARGIN - BUY_WIDTH
        self._add_static("Purchase box", y, bold=True,
                         width=buy_x - MARGIN - GAP, height=HEADER_HEIGHT, centred=True)
        self._add_control(ID_BUY, "BUTTON", "Buy tsum", BS_PUSHBUTTON,
                          buy_x, y, BUY_WIDTH, HEADER_HEIGHT)
        return y + HEADER_HEIGHT

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

    def _add_static(self, text: str, y: int, *, bold: bool = False,
                    x: Optional[int] = None, width: Optional[int] = None,
                    height: Optional[int] = None, centred: bool = False,
                    align_centre: bool = False, ident: int = 0) -> int:
        """A text label. Pass ``ident`` for one whose text is rewritten later."""
        x = MARGIN if x is None else x
        width = (PANEL_WIDTH - 2 * MARGIN) if width is None else width
        height = ROW if height is None else height
        style = win32con.WS_CHILD | win32con.WS_VISIBLE
        if centred:
            # Labels sharing a line with a taller control have to be centred on
            # it, or they sit against the top of their own box.
            style |= SS_CENTERIMAGE
        if align_centre:
            style |= SS_CENTER
        hwnd = win32gui.CreateWindowEx(
            0, "STATIC", text, style,
            self._px(x), self._px(y), self._px(width), self._px(height),
            self._hwnd, ident, win32api.GetModuleHandle(None), None,
        )
        win32gui.SendMessage(hwnd, win32con.WM_SETFONT, self._fonts[1 if bold else 0], 1)
        if ident:
            self._controls[ident] = hwnd
        return y + height

    def _add_check(self, ident: int, label: str, y: int, *, bold: bool = False) -> int:
        hwnd = self._add_control(ident, "BUTTON", label, BS_AUTOCHECKBOX,
                                 MARGIN, y, PANEL_WIDTH - 2 * MARGIN, ROW)
        if bold:
            # A checkbox that is also a section heading, so it reads as one
            # rather than as another item of the list above it.
            win32gui.SendMessage(hwnd, win32con.WM_SETFONT, self._fonts[1], 1)
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
        self._set_check(ID_AUTO_PLAY, bool(state.get("auto_play", False)))
        self._set_check(ID_RESTART_STUCK, bool(state.get("restart_when_stuck", False)))

        timed = bool(state.get("return_heart", False))
        self._set_check(ID_RETURN_HEART, timed)
        minutes = state.get("return_heart_minutes") or RETURN_HEART_MINUTES_DEFAULT
        for index in range(RETURN_HEART_MARKS):
            ident = ID_MINUTE_BASE + index
            if index < len(minutes):
                self._show_number(ident, minutes[index])
            # The marks mean nothing while hearts go out every cycle anyway;
            # greying them out says so without hiding the values the next tick
            # will use.
            self._enable(ident, timed)
            self._enable(_spin_for(ident), timed)

        for index, mode in enumerate(self._modes):
            self._set_check(ID_MODE_BASE + index, mode.key == state.get("mode"))
        pattern = state.get("claim_pattern", CLAIM_PATTERN_DEFAULT)
        for index, (key, _label, _flag) in enumerate(CLAIM_PATTERNS):
            self._set_check(ID_CLAIM_BASE + index, key == pattern)
        self._set_check(ID_COLLECT_DATA, state.get("collect_data", False))
        purchase = state.get("purchase", {})
        for index, (key, _label, default) in enumerate(PURCHASE_BOXES):
            self._set_check(ID_PURCHASE_BASE + index, purchase.get(key, default))

        running = bool(state.get("running"))
        stopping = bool(state.get("stopping"))
        status = state.get("status", "")

        # The word, not the button, is what says which state the bot is in --
        # big enough to read at a glance and coloured for it.
        if stopping:
            self._state_colour = COLOUR_STOPPING
            word = "STOPPING"
        elif running:
            self._state_colour = COLOUR_RUNNING
            word = "RUNNING"
        else:
            self._state_colour = COLOUR_PAUSED
            # Just the word. A stopped bot only ever reports "Idle (<mode>)",
            # and the mode is already lit in the radio group further up -- so
            # the suffix said nothing the panel was not showing twice over.
            # RUNNING and STOPPING keep theirs: that one names the live job.
            word, status = "PAUSE", ""
        self._set_text(ID_RUN_STATE, word + (f"  -  {status}" if status else ""))
        # Repaint it: the colour is decided in WM_CTLCOLORSTATIC, which only
        # runs when the label is asked to draw.
        hwnd = self._controls.get(ID_RUN_STATE)
        if hwnd:
            win32gui.InvalidateRect(hwnd, None, True)

        self._set_text(ID_RUN, "Run")
        # Only clickable when there is nothing to clash with: a second Run
        # during a run would do nothing, and neither would one while the last
        # one is still unwinding. Use the stop key (F12 by default) to end a
        # run -- the panel deliberately has no button for it.
        self._enable(ID_RUN, not (running or stopping))
        self._enable(ID_BUY, not (running or stopping))

        self._set_check(ID_AUTO_UPDATE, bool(state.get("auto_update", False)))
        self._set_text(ID_UPDATE_STATUS, str(state.get("update_status", "")))
        self._set_text(ID_UPDATE, str(state.get("update_button", "Check")))
        # The button is dead while a check or download is in flight; the
        # status line under it is what says so.
        self._enable(ID_UPDATE, bool(state.get("update_ready", True)))

        self._apply_topmost(state.get("always_on_top", True))

    def _apply_topmost(self, enabled: bool) -> None:
        win32gui.SetWindowPos(
            self._hwnd,
            win32con.HWND_TOPMOST if enabled else win32con.HWND_NOTOPMOST,
            0, 0, 0, 0,
            win32con.SWP_NOMOVE | win32con.SWP_NOSIZE | win32con.SWP_NOACTIVATE,
        )

    def _show_number(self, ident: int, value: int, *, force: bool = False) -> None:
        """Put ``value`` in a spinner without fighting whoever is typing.

        A refresh lands on every state change, including ones the edit box
        itself caused, so rewriting the text unconditionally would drag the
        caret back to the start mid-keystroke.
        """
        hwnd = self._controls.get(ident)
        if not hwnd:
            return
        if not force and win32gui.GetFocus() == hwnd:
            return
        text = str(clamp_minute(value))
        if text == win32gui.GetWindowText(hwnd):
            return
        self._writing_number = True
        try:
            win32gui.SetWindowText(hwnd, text)
        finally:
            self._writing_number = False
        spin = self._controls.get(_spin_for(ident))
        if spin:
            win32gui.SendMessage(spin, UDM_SETPOS32, 0, int(text))

    def _read_number(self, ident: int) -> Optional[int]:
        """What an edit box currently holds, or ``None`` while it is empty.

        Clearing the box to retype it must not be reported as 0 -- that would
        silently move a mark to the top of the hour mid-keystroke.
        """
        hwnd = self._controls.get(ident)
        if not hwnd:
            return None
        text = win32gui.GetWindowText(hwnd).strip()
        if not text:
            return None
        return clamp_minute(text)

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
            self._on_command(win32api.LOWORD(wparam), win32api.HIWORD(wparam))
            return 0
        if msg in (win32con.WM_CTLCOLORSTATIC, win32con.WM_CTLCOLORBTN):
            # Static/checkbox text paints on white otherwise, which looks like
            # a rendering bug against the dialog-grey background.
            ctypes.windll.gdi32.SetBkMode(wparam, 1)  # TRANSPARENT
            if lparam and lparam == self._controls.get(ID_RUN_STATE):
                ctypes.windll.gdi32.SetTextColor(wparam, self._state_colour)
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

    def _on_command(self, ident: int, code: int = 0) -> None:
        if ID_MINUTE_BASE <= ident < ID_MINUTE_BASE + RETURN_HEART_MARKS:
            self._on_minute_notification(ident, code)
            return
        try:
            if ident == ID_ALWAYS_ON_TOP:
                self._on_toggle("always_on_top", self._get_check(ident))
            elif ident == ID_AUTO_PLAY:
                self._on_toggle("auto_play", self._get_check(ident))
            elif ident == ID_RESTART_STUCK:
                self._on_toggle("restart_when_stuck", self._get_check(ident))
            elif ident == ID_COLLECT_DATA:
                self._on_toggle("collect_data", self._get_check(ident))
            elif ident == ID_RETURN_HEART:
                self._on_toggle("return_heart", self._get_check(ident))
            elif ident == ID_AUTO_UPDATE:
                self._on_toggle("auto_update", self._get_check(ident))
            elif ident == ID_UPDATE:
                self._on_update()
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
            elif ID_CLAIM_BASE <= ident < ID_CLAIM_BASE + len(CLAIM_PATTERNS):
                self._on_claim(CLAIM_PATTERNS[ident - ID_CLAIM_BASE][0])
            elif ID_PURCHASE_BASE <= ident < ID_PURCHASE_BASE + len(PURCHASE_BOXES):
                key = PURCHASE_BOXES[ident - ID_PURCHASE_BASE][0]
                self._on_purchase(key, self._get_check(ident))
            else:
                return
        except Exception:  # noqa: BLE001 - a handler must not kill the loop
            log.exception("Panel command %s failed", ident)
        self.refresh()

    def _on_minute_notification(self, ident: int, code: int) -> None:
        """EN_CHANGE while the user (or the arrows) edits a Return Heart mark.

        Handled apart from the buttons because a full :meth:`refresh` here
        would rewrite the very text being typed; only losing focus gets the
        box snapped back to the stored value.
        """
        index = ident - ID_MINUTE_BASE
        if code == EN_KILLFOCUS:
            self._show_number(ident, self._current_minute(index), force=True)
            return
        if code != EN_CHANGE or self._writing_number:
            return
        value = self._read_number(ident)
        if value is None:
            return
        try:
            self._on_return_minute(index, value)
        except Exception:  # noqa: BLE001 - a handler must not kill the loop
            log.exception("Panel Return Heart update failed")

    def _current_minute(self, index: int) -> int:
        """The stored value for one mark, for snapping a box back to."""
        fallback = RETURN_HEART_MINUTES_DEFAULT[index]
        try:
            minutes = self._get_state().get("return_heart_minutes") or ()
        except Exception:  # noqa: BLE001 - never let a callback break the box
            log.debug("state callback failed", exc_info=True)
            return fallback
        return clamp_minute(minutes[index], fallback) if index < len(minutes) else fallback


def _window_size(hwnd: int):
    left, top, right, bottom = win32gui.GetWindowRect(hwnd)
    return left, top, right - left, bottom - top
