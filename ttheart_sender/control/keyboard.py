"""Keyboard control (same protocol-first shape as :mod:`.mouse`)."""

from __future__ import annotations

import logging
from typing import Optional, Protocol, Sequence, runtime_checkable

from ..config import InputConfig
from .mouse import _import_pyautogui

log = logging.getLogger(__name__)


@runtime_checkable
class KeyboardController(Protocol):
    def press(self, key: str, *, presses: int = 1, interval: float = 0.05) -> None: ...

    def hotkey(self, *keys: str) -> None: ...

    def type_text(self, text: str, *, interval: float = 0.02) -> None: ...


class PyAutoGuiKeyboard:
    def __init__(self, config: Optional[InputConfig] = None) -> None:
        self.config = config or InputConfig()
        self._pyautogui = _import_pyautogui()

    def press(self, key: str, *, presses: int = 1, interval: float = 0.05) -> None:
        self._pyautogui.press(key, presses=max(1, presses), interval=interval)
        log.debug("press %s x%d", key, presses)

    def hotkey(self, *keys: str) -> None:
        if not keys:
            return
        self._pyautogui.hotkey(*keys)
        log.debug("hotkey %s", "+".join(keys))

    def type_text(self, text: str, *, interval: float = 0.02) -> None:
        self._pyautogui.write(text, interval=interval)
        log.debug("type %r", text)


class NullKeyboard:
    """No-op keyboard for ``--dry-run``."""

    def __init__(self) -> None:
        self.events: list = []

    def press(self, key: str, *, presses: int = 1, interval: float = 0.05) -> None:
        self.events.append(("press", key, presses))

    def hotkey(self, *keys: Sequence[str]) -> None:
        self.events.append(("hotkey", tuple(keys)))

    def type_text(self, text: str, *, interval: float = 0.02) -> None:
        self.events.append(("type", text))
