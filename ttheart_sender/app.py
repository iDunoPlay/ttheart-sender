"""Application wiring.

One place assembles every component, so the CLI (and any future GUI or long
running service) only has to say *what* it wants, never *how* the pieces fit
together.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .automation.context import RunContext
from .automation.flow import Flow, list_flows, load_flow_by_name
from .automation.runner import FlowRunner, RunReport
from .config import Config, load_config
from .control.hotkey import StopKeyWatcher
from .control.keyboard import NullKeyboard, PyAutoGuiKeyboard
from .control.mouse import NullMouse, PyAutoGuiMouse
from .dpi import enable_dpi_awareness
from .exceptions import TTHeartError
from .geometry import Rect
from .logging_setup import setup_logging
from .screen.capture import ScreenCapture
from .screen.matcher import TemplateMatcher
from .screen.templates import TemplateLibrary
from .window.manager import WindowController, WindowInfo, WindowManager

log = logging.getLogger(__name__)


class Application:
    """Owns the configured components for one session."""

    def __init__(self, config: Config, *, dry_run: bool = False) -> None:
        self.config = config
        self.dry_run = dry_run

        self.windows = WindowManager()
        self.capture = ScreenCapture()
        self.templates = TemplateLibrary(
            config.templates_dir, ignore_alpha=config.matching.ignore_alpha
        )
        self.matcher = TemplateMatcher(
            confidence=config.matching.confidence,
            grayscale=config.matching.grayscale,
            scales=config.matching.scales,
        )
        self.stop = StopKeyWatcher(config.runner.stop_key)

        if dry_run:
            self.mouse: Any = NullMouse()
            self.keyboard: Any = NullKeyboard()
        else:
            self.mouse = PyAutoGuiMouse(config.input)
            self.keyboard = PyAutoGuiKeyboard(config.input)

        self._window: Optional[WindowController] = None

    # -- construction ----------------------------------------------------
    @classmethod
    def create(
        cls,
        config_path: Optional[Union[str, Path]] = None,
        *,
        log_level: Optional[str] = None,
        dry_run: bool = False,
        quiet_logging: bool = False,
    ) -> "Application":
        # Must happen before anything reads window rects or cursor positions.
        enable_dpi_awareness()
        config = load_config(config_path)
        if log_level:
            config.app.log_level = log_level
        setup_logging(
            config.app.log_level,
            None if quiet_logging else config.log_dir,
        )
        log.debug("Config root: %s", config.root)
        return cls(config, dry_run=dry_run)

    # -- window ----------------------------------------------------------
    def detect_windows(self) -> List[WindowInfo]:
        """Every window matching the configured emulator target."""
        return self.windows.find(self.config.window.target)

    def attach_window(self, *, prepare: bool = True) -> WindowController:
        """Locate the emulator and (optionally) park + focus it."""
        controller = self.windows.find_one(self.config.window)
        if prepare:
            controller.prepare(self.config.window)
        self._window = controller
        return controller

    def startup(
        self,
        *,
        require_window: bool = True,
        prepare: Optional[bool] = None,
    ) -> Optional[WindowController]:
        """Session start: detect the emulator and park it at the top-left.

        This is deliberately the first thing that happens, so every later
        command works against a window that is already restored, positioned and
        focused -- and so a missing emulator is reported once, up front, rather
        than surfacing three steps into a flow.

        ``require_window=False`` downgrades "not found" to a warning, for
        commands that only need it if it happens to be there.
        """
        should_prepare = self.config.window.prepare_on_launch if prepare is None else prepare
        try:
            controller = self.windows.find_one(self.config.window)
        except TTHeartError as exc:
            if require_window:
                raise
            log.warning("%s", exc)
            return None

        info = controller.info()
        log.info("Detected %s (%s)", info.title or "emulator", info.class_name)
        self._window = controller

        if not should_prepare:
            controller.restore()
            log.info("Leaving the window where it is (prepare disabled)")
            return controller

        controller.prepare(self.config.window)
        return controller

    @property
    def window(self) -> Optional[WindowController]:
        return self._window

    def content_rect(self) -> Rect:
        if self._window is None:
            raise RuntimeError("No window attached; call attach_window() first")
        return self._window.content_rect(
            use_client_area=self.config.window.use_client_area,
            insets=self.config.window.insets,
        )

    # -- automation ------------------------------------------------------
    def build_context(self, *, window: Optional[WindowController] = None) -> RunContext:
        controller = window if window is not None else self._window
        ctx = RunContext(
            config=self.config,
            capture=self.capture,
            matcher=self.matcher,
            templates=self.templates,
            mouse=self.mouse,
            keyboard=self.keyboard,
            stop=self.stop,
            window=controller,
            dry_run=self.dry_run,
        )
        ctx.refresh_window()
        return ctx

    def load_flow(self, name: str) -> Flow:
        return load_flow_by_name(self.config.flows_dir, name)

    def available_flows(self) -> List[Path]:
        return list_flows(self.config.flows_dir)

    def run_flow(
        self,
        name: str,
        *,
        variables: Optional[Dict[str, Any]] = None,
        loops: int = 1,
        loop_delay: float = 0.0,
        use_window: bool = True,
        prepare_window: bool = True,
    ) -> RunReport:
        """Load and execute a flow by name."""
        flow = self.load_flow(name)
        # startup() has normally attached already; only do it here if not.
        if use_window and self._window is None:
            self.attach_window(prepare=prepare_window)
        ctx = self.build_context()
        runner = FlowRunner(ctx)
        if self.stop.enabled:
            log.info("Press %s at any time to stop.", str(self.config.runner.stop_key).upper())
        return runner.run(flow, variables=variables, loops=loops, loop_delay=loop_delay)

    # -- lifecycle -------------------------------------------------------
    def close(self) -> None:
        self.capture.close()

    def __enter__(self) -> "Application":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()
