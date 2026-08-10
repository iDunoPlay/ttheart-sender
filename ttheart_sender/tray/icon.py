"""A Win32 shell notification icon (the thing in the system tray).

Built directly on ``pywin32`` rather than a tray library because the project
already depends on ``pywin32`` for window handling, and a tray icon is only a
hidden window plus ``Shell_NotifyIcon``. Doing it here also keeps the menu
fully under our control -- radio-checked modes, greyed-out entries and a bold
default item are all awkward to express through a wrapper.

Everything in this module must run on the thread that called :meth:`run`;
:meth:`post` is the only member other threads may touch.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Deque, Dict, List, Optional, Sequence

import win32api
import win32con
import win32gui

log = logging.getLogger(__name__)

#: Icon -> window notifications (mouse activity on the tray icon).
WM_TRAYICON = win32con.WM_APP + 1
#: "Drain the cross-thread callback queue", posted by :meth:`post`.
WM_RUN_CALLBACKS = win32con.WM_APP + 2

#: Command IDs start here; 0 means "the user dismissed the menu".
_FIRST_MENU_ID = 1024
#: Shell truncates the tooltip itself, but silently -- keep it readable.
_TOOLTIP_LIMIT = 127
_BALLOON_TEXT_LIMIT = 255
_BALLOON_TITLE_LIMIT = 63


# --------------------------------------------------------------------------
# Menu description
# --------------------------------------------------------------------------
@dataclass
class MenuItem:
    """One row of the popup menu.

    A row is a separator (no label), a submenu (``items`` set) or a command
    (``action`` set). ``radio`` renders contiguous items as a bullet group.
    """

    label: str = ""
    action: Optional[Callable[[], None]] = None
    items: Optional[Sequence["MenuItem"]] = None
    enabled: bool = True
    checked: bool = False
    radio: bool = False
    #: Rendered bold -- the action a double-click would take.
    default: bool = False

    @property
    def is_separator(self) -> bool:
        return not self.label and self.items is None and self.action is None


def separator() -> MenuItem:
    return MenuItem()


@dataclass
class _MenuBuild:
    """Scratch state for one popup: the command IDs handed out, and what they do."""

    actions: Dict[int, Callable[[], None]] = field(default_factory=dict)
    next_id: int = _FIRST_MENU_ID


# --------------------------------------------------------------------------
# Tray icon
# --------------------------------------------------------------------------
class TrayIcon:
    """Hidden window + shell icon + right-click menu."""

    def __init__(
        self,
        *,
        title: str,
        menu: Callable[[], Sequence[MenuItem]],
        tooltip: Callable[[], str],
        icon_path: Optional[Callable[[], Optional[Path]]] = None,
        on_double_click: Optional[Callable[[], None]] = None,
        window_class: str = "TTHeartSenderTray",
    ) -> None:
        self._title = title
        self._menu_factory = menu
        self._tooltip_factory = tooltip
        self._icon_path_factory = icon_path or (lambda: None)
        self._on_double_click = on_double_click
        self._window_class = window_class

        self._hwnd: Optional[int] = None
        self._class_atom: Optional[int] = None
        self._hicon: Optional[int] = None
        self._loaded_icon: Optional[Path] = None
        self._icon_cache: Dict[Path, int] = {}
        self._added = False
        self._callbacks: Deque[Callable[[], None]] = deque()
        # Explorer broadcasts this after a crash/restart; without re-adding the
        # icon the app keeps running but becomes invisible and unstoppable.
        self._taskbar_created = win32gui.RegisterWindowMessage("TaskbarCreated")

    # -- lifecycle -------------------------------------------------------
    def run(self) -> int:
        """Create the icon and pump messages until :meth:`quit` is called."""
        self._create_window()
        self._add_icon()
        try:
            win32gui.PumpMessages()
        finally:
            self._destroy()
        return 0

    def quit(self) -> None:
        """Ask the message loop to exit (safe from the GUI thread)."""
        if self._hwnd is not None:
            win32gui.PostMessage(self._hwnd, win32con.WM_CLOSE, 0, 0)

    def post(self, callback: Callable[[], None]) -> None:
        """Queue ``callback`` to run on the GUI thread. Thread-safe."""
        self._callbacks.append(callback)  # deque.append is atomic
        if self._hwnd is not None:
            win32gui.PostMessage(self._hwnd, WM_RUN_CALLBACKS, 0, 0)

    # -- appearance ------------------------------------------------------
    def refresh(self) -> None:
        """Re-read the tooltip and icon from their factories."""
        if not self._added or not self._alive():
            return
        flags = win32gui.NIF_MESSAGE | win32gui.NIF_TIP
        hicon = self._current_icon()
        if hicon:
            flags |= win32gui.NIF_ICON
        try:
            win32gui.Shell_NotifyIcon(
                win32gui.NIM_MODIFY,
                (self._hwnd, 0, flags, WM_TRAYICON, hicon, self._tooltip()),
            )
        except win32gui.error:
            log.debug("Shell_NotifyIcon(NIM_MODIFY) failed", exc_info=True)

    def notify(self, title: str, message: str, *, error: bool = False) -> None:
        """Show a balloon / toast from the tray icon."""
        if not self._added or not self._alive():
            return
        icon_flag = win32gui.NIIF_ERROR if error else win32gui.NIIF_INFO
        payload = (
            self._hwnd,
            0,
            win32gui.NIF_INFO,
            WM_TRAYICON,
            self._current_icon(),
            self._tooltip(),
            _clip(message, _BALLOON_TEXT_LIMIT),
            10_000,
            _clip(title, _BALLOON_TITLE_LIMIT),
            icon_flag,
        )
        try:
            win32gui.Shell_NotifyIcon(win32gui.NIM_MODIFY, payload)
        except (win32gui.error, TypeError):
            log.debug("Balloon notification failed", exc_info=True)

    # -- window ----------------------------------------------------------
    def _create_window(self) -> None:
        hinst = win32api.GetModuleHandle(None)
        wc = win32gui.WNDCLASS()
        wc.hInstance = hinst
        wc.lpszClassName = self._window_class
        wc.lpfnWndProc = self._wnd_proc
        self._class_atom = win32gui.RegisterClass(wc)
        self._hwnd = win32gui.CreateWindow(
            self._class_atom,
            self._title,
            win32con.WS_OVERLAPPED | win32con.WS_SYSMENU,
            0, 0, 0, 0,
            0, 0, hinst, None,
        )
        win32gui.UpdateWindow(self._hwnd)
        # Anything queued before the window existed could not be posted.
        if self._callbacks:
            win32gui.PostMessage(self._hwnd, WM_RUN_CALLBACKS, 0, 0)

    def _wnd_proc(self, hwnd: int, msg: int, wparam: int, lparam: int) -> int:
        if msg == WM_TRAYICON:
            self._on_tray_message(lparam)
            return 0
        if msg == WM_RUN_CALLBACKS:
            self._drain_callbacks()
            return 0
        if msg == self._taskbar_created:
            log.info("Explorer restarted; re-adding the tray icon")
            self._added = False
            self._add_icon()
            return 0
        if msg == win32con.WM_CLOSE:
            win32gui.DestroyWindow(hwnd)
            return 0
        if msg == win32con.WM_DESTROY:
            self._remove_icon()
            win32gui.PostQuitMessage(0)
            return 0
        return win32gui.DefWindowProc(hwnd, msg, wparam, lparam)

    def _on_tray_message(self, event: int) -> None:
        if event in (win32con.WM_RBUTTONUP, win32con.WM_CONTEXTMENU):
            self._safely(self._show_menu)
        elif event == win32con.WM_LBUTTONDBLCLK and self._on_double_click:
            self._safely(self._on_double_click)
        else:
            # Balloon show/hide/timeout and hover events all arrive here.
            log.debug("Unhandled tray event: %s", event)

    def _drain_callbacks(self) -> None:
        while True:
            try:
                callback = self._callbacks.popleft()
            except IndexError:
                return
            self._safely(callback)

    def _alive(self) -> bool:
        """True while our window still exists and can be messaged."""
        return self._hwnd is not None and bool(win32gui.IsWindow(self._hwnd))

    @staticmethod
    def _safely(callback: Callable[[], None]) -> None:
        # An exception raised inside a window procedure would otherwise unwind
        # through pywin32 and take the message loop (and the icon) with it.
        try:
            callback()
        except Exception:  # noqa: BLE001
            log.exception("Tray callback failed")

    # -- icon ------------------------------------------------------------
    def _tooltip(self) -> str:
        try:
            return _clip(self._tooltip_factory(), _TOOLTIP_LIMIT)
        except Exception:  # noqa: BLE001
            log.debug("Tooltip factory failed", exc_info=True)
            return self._title

    def _current_icon(self) -> int:
        try:
            path = self._icon_path_factory()
        except Exception:  # noqa: BLE001
            log.debug("Icon factory failed", exc_info=True)
            path = None

        if path is not None:
            cached = self._icon_cache.get(path)
            if cached is None and path.exists():
                cached = self._load_icon_file(path)
                if cached:
                    self._icon_cache[path] = cached
            if cached:
                self._hicon = cached
                return cached

        if self._hicon is None:
            self._hicon = win32gui.LoadIcon(0, win32con.IDI_APPLICATION)
        return self._hicon

    @staticmethod
    def _load_icon_file(path: Path) -> Optional[int]:
        try:
            return win32gui.LoadImage(
                0,
                str(path),
                win32con.IMAGE_ICON,
                0,
                0,
                win32con.LR_LOADFROMFILE | win32con.LR_DEFAULTSIZE,
            )
        except win32gui.error:
            log.warning("Could not load tray icon %s", path)
            return None

    def _add_icon(self) -> None:
        if self._added or self._hwnd is None:
            return
        flags = win32gui.NIF_ICON | win32gui.NIF_MESSAGE | win32gui.NIF_TIP
        win32gui.Shell_NotifyIcon(
            win32gui.NIM_ADD,
            (self._hwnd, 0, flags, WM_TRAYICON, self._current_icon(), self._tooltip()),
        )
        self._added = True

    def _remove_icon(self) -> None:
        if not self._added or self._hwnd is None:
            return
        try:
            win32gui.Shell_NotifyIcon(win32gui.NIM_DELETE, (self._hwnd, 0))
        except win32gui.error:
            log.debug("Shell_NotifyIcon(NIM_DELETE) failed", exc_info=True)
        self._added = False

    def _destroy(self) -> None:
        self._remove_icon()
        for hicon in self._icon_cache.values():
            try:
                win32gui.DestroyIcon(hicon)
            except win32gui.error:
                pass
        self._icon_cache.clear()
        if self._class_atom is not None:
            try:
                win32gui.UnregisterClass(self._class_atom, win32api.GetModuleHandle(None))
            except win32gui.error:
                pass
            self._class_atom = None
        self._hwnd = None

    # -- menu ------------------------------------------------------------
    def _show_menu(self) -> None:
        # TrackPopupMenu runs a nested message loop, so a WM_CLOSE can be
        # dispatched (and the window destroyed) while a menu is still up. Check
        # the handle on the way in, and again before touching it on the way out.
        if not self._alive():
            return
        try:
            items = self._menu_factory()
        except Exception:  # noqa: BLE001
            log.exception("Could not build the tray menu")
            return

        build = _MenuBuild()
        hmenu = self._build_menu(items, build)
        position = win32gui.GetCursorPos()
        try:
            # Required so the menu closes when the user clicks elsewhere; the
            # WM_NULL afterwards is the documented companion to that hack.
            # It fails harmlessly if another app is holding foreground rights.
            try:
                win32gui.SetForegroundWindow(self._hwnd)
            except win32gui.error:
                log.debug("SetForegroundWindow failed", exc_info=True)

            command = win32gui.TrackPopupMenu(
                hmenu,
                win32con.TPM_LEFTALIGN
                | win32con.TPM_RIGHTBUTTON
                | win32con.TPM_RETURNCMD
                | win32con.TPM_NONOTIFY,
                position[0],
                position[1],
                0,
                self._hwnd,
                None,
            )
        finally:
            if self._alive():
                win32gui.PostMessage(self._hwnd, win32con.WM_NULL, 0, 0)
            # DestroyMenu frees attached submenus too, so only the root.
            try:
                win32gui.DestroyMenu(hmenu)
            except win32gui.error:
                log.debug("DestroyMenu failed", exc_info=True)

        action = build.actions.get(command)
        if action is not None and self._alive():
            self._safely(action)

    def _build_menu(self, items: Sequence[MenuItem], build: _MenuBuild) -> int:
        hmenu = win32gui.CreatePopupMenu()
        radio_run: List[tuple] = []

        def flush_radio() -> None:
            if radio_run:
                ids = [item_id for item_id, _ in radio_run]
                checked = next((i for i, on in radio_run if on), None)
                if checked is not None:
                    win32gui.CheckMenuRadioItem(
                        hmenu, ids[0], ids[-1], checked, win32con.MF_BYCOMMAND
                    )
            radio_run.clear()

        for item in items:
            if item.is_separator:
                flush_radio()
                win32gui.AppendMenu(hmenu, win32con.MF_SEPARATOR, 0, "")
                continue

            if item.items is not None:
                flush_radio()
                submenu = self._build_menu(item.items, build)
                flags = win32con.MF_STRING | win32con.MF_POPUP
                if not item.enabled:
                    flags |= win32con.MF_GRAYED
                win32gui.AppendMenu(hmenu, flags, submenu, item.label)
                continue

            item_id = build.next_id
            build.next_id += 1
            flags = win32con.MF_STRING
            if not item.enabled:
                flags |= win32con.MF_GRAYED
            if item.checked and not item.radio:
                flags |= win32con.MF_CHECKED
            if item.default:
                flags |= win32con.MF_DEFAULT
            win32gui.AppendMenu(hmenu, flags, item_id, item.label)
            if item.action is not None and item.enabled:
                build.actions[item_id] = item.action

            if item.radio:
                # IDs are handed out in append order, so a contiguous run of
                # radio items always occupies a contiguous ID range -- which is
                # exactly what CheckMenuRadioItem expects.
                radio_run.append((item_id, item.checked))
            else:
                flush_radio()

        flush_radio()
        return hmenu


def _clip(text: str, limit: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"
