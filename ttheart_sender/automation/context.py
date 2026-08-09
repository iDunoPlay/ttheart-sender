"""Runtime context handed to every action.

Actions never touch mss, OpenCV or win32 directly -- they go through this
object, which owns the one piece of state that would otherwise be duplicated
everywhere: the mapping between *content coordinates* (relative to the
emulator's content area) and *screen coordinates* (what the mouse understands).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from ..config import Config
from ..control.hotkey import StopKeyWatcher, interruptible_sleep
from ..control.keyboard import KeyboardController
from ..control.mouse import MouseController
from ..exceptions import ActionError
from ..geometry import Point, Rect
from ..screen.capture import ScreenCapture, save_image
from ..screen.matcher import MatchResult, TemplateMatcher, annotate
from ..screen.templates import TemplateLibrary
from ..window.manager import WindowController

log = logging.getLogger(__name__)

#: Region shorthands usable in flows.
SCREEN_REGIONS = {"screen", "desktop", "full"}
CONTENT_REGIONS = {"window", "content", "app"}

RegionSpec = Union[None, str, Rect, Sequence[int]]


@dataclass
class RunContext:
    """Everything an action needs, assembled once per run."""

    config: Config
    capture: ScreenCapture
    matcher: TemplateMatcher
    templates: TemplateLibrary
    mouse: MouseController
    keyboard: KeyboardController
    stop: StopKeyWatcher
    window: Optional[WindowController] = None
    #: Content area in screen coordinates; origin of the relative space.
    content_rect: Optional[Rect] = None
    variables: Dict[str, Any] = field(default_factory=dict)
    dry_run: bool = False
    #: Set by the runner so control-flow actions can execute nested steps.
    executor: Optional[Callable[[List[Any]], bool]] = None
    depth: int = 0

    # -- window / coordinates -------------------------------------------
    def refresh_window(self) -> Optional[Rect]:
        """Re-read the window rectangle (call after the emulator was moved)."""
        if self.window is None:
            return None
        self.content_rect = self.window.content_rect(
            use_client_area=self.config.window.use_client_area,
            insets=self.config.window.insets,
        )
        return self.content_rect

    def require_content_rect(self) -> Rect:
        if self.content_rect is None:
            refreshed = self.refresh_window()
            if refreshed is None:
                raise ActionError(
                    "This step needs the emulator window, but no window is attached. "
                    "Run without --no-window, or use region: screen."
                )
        assert self.content_rect is not None
        return self.content_rect

    def resolve_region(self, spec: RegionSpec = None) -> Rect:
        """Turn a flow's ``region:`` value into a screen-space rectangle."""
        if spec is None:
            return self.require_content_rect() if self.window else self.capture.virtual_screen_rect()

        if isinstance(spec, str):
            key = spec.strip().casefold()
            if key in SCREEN_REGIONS:
                return self.capture.virtual_screen_rect()
            if key in CONTENT_REGIONS:
                return self.require_content_rect()
            raise ActionError(
                f"Unknown region {spec!r}. Use 'window', 'screen', or [x, y, width, height]."
            )

        rect = spec if isinstance(spec, Rect) else Rect.from_sequence(spec)
        # Numeric regions are relative to the content area when we have one.
        base = self.require_content_rect() if self.window else Rect(0, 0, 0, 0)
        return rect.translate(base.left, base.top)

    def to_screen(self, point: Point, space: str = "content") -> Point:
        """Convert ``point`` from ``content``/``screen`` space to screen space."""
        key = str(space).strip().casefold()
        if key in SCREEN_REGIONS:
            return point
        if key in CONTENT_REGIONS or key == "content":
            base = self.require_content_rect()
            return Point(base.left + point.x, base.top + point.y)
        raise ActionError(f"Unknown coordinate space {space!r}. Use 'window' or 'screen'.")

    def to_content(self, point: Point) -> Point:
        base = self.require_content_rect()
        return Point(point.x - base.left, point.y - base.top)

    # -- capture ---------------------------------------------------------
    def grab(self, region: RegionSpec = None) -> Tuple[np.ndarray, Rect]:
        """Capture a region; returns the image and the rect it came from."""
        if self.config.window.refresh_rect_each_capture:
            self.refresh_window()
        rect = self.resolve_region(region)
        return self.capture.grab(rect), rect

    # -- matching --------------------------------------------------------
    def find(
        self,
        template_ref: str,
        *,
        region: RegionSpec = None,
        confidence: Optional[float] = None,
        scales: Optional[Sequence[float]] = None,
        grayscale: Optional[bool] = None,
        save_failure_as: Optional[str] = None,
    ) -> Optional[MatchResult]:
        """Look for a template once. The result is in *screen* coordinates."""
        template = self.templates.get(template_ref)
        image, rect = self.grab(region)
        match = self.matcher.find(image, template, confidence=confidence, scales=scales, grayscale=grayscale)
        if match is None:
            if self.config.matching.save_failures:
                self.save_debug(image, save_failure_as or f"miss_{Path(template.name).stem}")
            return None
        return match.translated(rect.left, rect.top)

    def find_all(
        self,
        template_ref: str,
        *,
        region: RegionSpec = None,
        confidence: Optional[float] = None,
        scales: Optional[Sequence[float]] = None,
        grayscale: Optional[bool] = None,
        max_results: int = 20,
    ) -> List[MatchResult]:
        template = self.templates.get(template_ref)
        image, rect = self.grab(region)
        matches = self.matcher.find_all(
            image, template, confidence=confidence, scales=scales, grayscale=grayscale, max_results=max_results
        )
        return [m.translated(rect.left, rect.top) for m in matches]

    def wait_for(
        self,
        template_ref: str,
        *,
        timeout: float,
        poll_interval: Optional[float] = None,
        region: RegionSpec = None,
        confidence: Optional[float] = None,
        scales: Optional[Sequence[float]] = None,
        grayscale: Optional[bool] = None,
        gone: bool = False,
    ) -> Optional[MatchResult]:
        """Poll until the template appears (or disappears when ``gone``).

        Returns the match, or None on timeout. With ``gone=True`` the meaning
        inverts: None means "still visible", so callers check ``is None``.
        """
        interval = (
            poll_interval if poll_interval is not None else self.config.runner.default_poll_interval
        )
        deadline = time.monotonic() + max(0.0, timeout)
        attempt = 0
        while True:
            self.stop.check()
            attempt += 1
            match = self.find(
                template_ref, region=region, confidence=confidence, scales=scales, grayscale=grayscale
            )
            if gone:
                if match is None:
                    return None
            elif match is not None:
                return match

            if time.monotonic() >= deadline:
                log.debug(
                    "wait_for(%s, gone=%s) timed out after %d attempt(s)",
                    template_ref,
                    gone,
                    attempt,
                )
                return match if gone else None
            self.sleep(min(interval, max(0.0, deadline - time.monotonic())))

    # -- input -----------------------------------------------------------
    def click_screen(
        self,
        point: Point,
        *,
        button: str = "left",
        clicks: int = 1,
        interval: float = 0.08,
        duration: Optional[float] = None,
    ) -> None:
        if self.dry_run:
            log.info("%s[dry-run] click %s button=%s clicks=%d", self.indent, point, button, clicks)
            return
        self.mouse.click(point, button=button, clicks=clicks, interval=interval, duration=duration)

    # -- utilities -------------------------------------------------------
    def sleep(self, seconds: float) -> None:
        interruptible_sleep(seconds, self.stop)

    def save_debug(self, image: np.ndarray, label: str, matches: Sequence[MatchResult] = ()) -> Path:
        """Write a (optionally annotated) screenshot into the debug directory."""
        canvas = annotate(image, matches) if matches else image
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in label)
        path = self.config.debug_dir / f"{stamp}_{safe}.png"
        save_image(path, canvas)
        log.debug("Saved debug image %s", path)
        return path

    def set_var(self, key: str, value: Any) -> None:
        self.variables[key] = value

    def execute(self, steps: Sequence[Any]) -> bool:
        """Run nested steps -- the hook control-flow actions build on."""
        if self.executor is None:  # pragma: no cover - runner always sets this
            raise ActionError("No step executor is attached to this context")
        return self.executor(list(steps))

    @property
    def indent(self) -> str:
        return "  " * self.depth
