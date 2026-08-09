"""Template image library.

A "template" is a cropped screenshot of the thing you want to click. Templates
live under ``templates/`` and are referenced from flows by file name (with or
without the extension) or by a path relative to that directory.

Transparent PNGs are supported: the alpha channel becomes a match mask, so you
can cut a button out of a busy background and only the opaque pixels count.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence

import cv2
import numpy as np

from ..exceptions import TemplateNotFoundError
from .capture import load_image

log = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


@dataclass(frozen=True)
class Template:
    """An image to search for, plus optional per-template match overrides."""

    name: str
    path: Path
    image: np.ndarray  # BGR
    mask: Optional[np.ndarray] = None  # single channel, 0/255
    confidence: Optional[float] = None
    scales: Optional[Sequence[float]] = None
    grayscale: Optional[bool] = None

    @property
    def width(self) -> int:
        return int(self.image.shape[1])

    @property
    def height(self) -> int:
        return int(self.image.shape[0])

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return f"{self.name} ({self.width}x{self.height})"


class TemplateLibrary:
    """Resolves template references to loaded images, with caching."""

    def __init__(self, root: Path, *, ignore_alpha: bool = False, cache: bool = True) -> None:
        self.root = Path(root)
        self.ignore_alpha = ignore_alpha
        self._cache_enabled = cache
        self._cache: Dict[Path, Template] = {}

    # -- discovery -------------------------------------------------------
    def available(self) -> List[str]:
        """Every template on disk, as reference strings usable in flows."""
        if not self.root.exists():
            return []
        names = [
            str(p.relative_to(self.root)).replace("\\", "/")
            for p in sorted(self._iter_files())
        ]
        return names

    def _iter_files(self) -> Iterator[Path]:
        for path in self.root.rglob("*"):
            if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
                yield path

    # -- resolution ------------------------------------------------------
    def resolve_path(self, reference: str) -> Path:
        """Turn a flow's template reference into a concrete file path."""
        ref = str(reference).strip().replace("\\", "/")
        if not ref:
            raise TemplateNotFoundError("Empty template reference")

        candidate = Path(ref)
        if candidate.is_absolute():
            if candidate.is_file():
                return candidate
            raise TemplateNotFoundError(f"Template not found: {candidate}")

        direct = self.root / candidate
        if direct.is_file():
            return direct

        # No extension given -- try each supported one.
        if not candidate.suffix:
            for suffix in SUPPORTED_SUFFIXES:
                probe = self.root / f"{candidate}{suffix}"
                if probe.is_file():
                    return probe

        # Last resort: bare name match anywhere under templates/.
        stem = candidate.stem.casefold()
        for path in self._iter_files():
            if path.stem.casefold() == stem:
                return path

        available = self.available()
        hint = f" Available: {available}" if available else f" (no images found in {self.root})"
        raise TemplateNotFoundError(f"Template {reference!r} not found.{hint}")

    def get(
        self,
        reference: str,
        *,
        confidence: Optional[float] = None,
        scales: Optional[Sequence[float]] = None,
    ) -> Template:
        """Load (or fetch from cache) the template named by ``reference``."""
        path = self.resolve_path(reference)
        cached = self._cache.get(path) if self._cache_enabled else None
        if cached is None:
            cached = self._load(path)
            if self._cache_enabled:
                self._cache[path] = cached
        if confidence is None and scales is None:
            return cached
        return Template(
            name=cached.name,
            path=cached.path,
            image=cached.image,
            mask=cached.mask,
            confidence=confidence if confidence is not None else cached.confidence,
            scales=tuple(scales) if scales is not None else cached.scales,
        )

    def _load(self, path: Path) -> Template:
        raw = load_image(path, with_alpha=not self.ignore_alpha)
        mask: Optional[np.ndarray] = None
        if raw.ndim == 3 and raw.shape[2] == 4:
            alpha = raw[:, :, 3]
            image = cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)
            # Only bother with a mask when the alpha channel is actually used.
            if np.any(alpha < 255):
                mask = np.where(alpha > 0, 255, 0).astype(np.uint8)
        else:
            image = raw

        if image.size == 0:
            raise TemplateNotFoundError(f"Template {path} contains no pixels")

        name = str(path.relative_to(self.root)).replace("\\", "/") if _is_under(path, self.root) else path.name
        log.debug("Loaded template %s (%dx%d, mask=%s)", name, image.shape[1], image.shape[0], mask is not None)
        return Template(name=name, path=path, image=image, mask=mask)

    def clear_cache(self) -> None:
        self._cache.clear()


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False
