"""Fast screen capture built on ``mss``.

``mss`` instances are not thread-safe and must be created on the thread that
uses them, so we keep one per thread. Everything returns a BGR ``ndarray``
(OpenCV's native layout) -- never BGRA -- so downstream code never has to guess.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Optional

import cv2
import mss
import numpy as np

from ..geometry import Rect

log = logging.getLogger(__name__)


class ScreenCapture:
    """Grabs pixels from the desktop."""

    def __init__(self) -> None:
        self._local = threading.local()

    # -- internals -------------------------------------------------------
    @property
    def _sct(self) -> "mss.base.MSSBase":
        sct = getattr(self._local, "sct", None)
        if sct is None:
            sct = mss.mss()
            self._local.sct = sct
        return sct

    # -- public API ------------------------------------------------------
    def virtual_screen_rect(self) -> Rect:
        """Bounding box covering every monitor (``monitors[0]`` in mss terms)."""
        monitor = self._sct.monitors[0]
        return Rect(monitor["left"], monitor["top"], monitor["width"], monitor["height"])

    def monitor_rect(self, index: int = 1) -> Rect:
        monitors = self._sct.monitors
        if not 0 <= index < len(monitors):
            raise IndexError(f"monitor {index} does not exist (have {len(monitors) - 1})")
        monitor = monitors[index]
        return Rect(monitor["left"], monitor["top"], monitor["width"], monitor["height"])

    def grab(self, region: Optional[Rect] = None) -> np.ndarray:
        """Capture ``region`` (screen coordinates) or the whole virtual desktop.

        Returns a BGR image. The region is clamped to the visible desktop, so a
        window hanging off-screen yields a smaller image rather than an error.
        """
        target = region if region is not None else self.virtual_screen_rect()
        if region is not None:
            target = target.intersect(self.virtual_screen_rect())
            if target.is_empty:
                raise ValueError(
                    f"Capture region {region} lies entirely outside the desktop "
                    f"{self.virtual_screen_rect()}"
                )

        raw = self._sct.grab(target.as_mss_dict())
        frame = np.asarray(raw, dtype=np.uint8)  # BGRA
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    def grab_and_save(self, path: Path, region: Optional[Rect] = None) -> Path:
        image = self.grab(region)
        return save_image(path, image)

    def close(self) -> None:
        sct = getattr(self._local, "sct", None)
        if sct is not None:
            sct.close()
            self._local.sct = None

    def __enter__(self) -> "ScreenCapture":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


def save_image(path: Path, image: np.ndarray) -> Path:
    """Write an image, creating parent dirs and tolerating non-ASCII paths.

    ``cv2.imwrite`` cannot handle Unicode paths on Windows; encode then write
    the bytes ourselves.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix or ".png"
    ok, buffer = cv2.imencode(suffix, image)
    if not ok:
        raise IOError(f"Failed to encode image as {suffix!r}")
    path.write_bytes(buffer.tobytes())
    return path


def load_image(path: Path, *, with_alpha: bool = True) -> np.ndarray:
    """Read an image (Unicode-path safe) as BGR or BGRA."""
    path = Path(path)
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        raise IOError(f"{path} is empty")
    flags = cv2.IMREAD_UNCHANGED if with_alpha else cv2.IMREAD_COLOR
    image = cv2.imdecode(data, flags)
    if image is None:
        raise IOError(f"{path} is not a readable image")
    if image.ndim == 2:  # grayscale on disk -> normalise to 3 channels
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image
