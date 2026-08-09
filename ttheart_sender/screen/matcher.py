"""OpenCV template matching with multi-scale search and non-max suppression."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Sequence

import cv2
import numpy as np

from ..geometry import Point, Rect
from .templates import Template

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class MatchResult:
    """Where a template was found, in the coordinate space of the searched image."""

    rect: Rect
    confidence: float
    scale: float = 1.0
    template_name: str = ""

    @property
    def center(self) -> Point:
        return self.rect.center

    def translated(self, dx: int, dy: int) -> "MatchResult":
        """Same match expressed in another coordinate space (e.g. screen)."""
        return MatchResult(
            rect=self.rect.translate(dx, dy),
            confidence=self.confidence,
            scale=self.scale,
            template_name=self.template_name,
        )

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return f"{self.template_name or 'match'}@{self.rect} conf={self.confidence:.3f} scale={self.scale:g}"


class TemplateMatcher:
    """Finds templates inside a captured frame.

    Defaults come from config; individual lookups can override confidence and
    scales without constructing a new matcher.
    """

    def __init__(
        self,
        *,
        confidence: float = 0.85,
        grayscale: bool = True,
        scales: Sequence[float] = (1.0,),
    ) -> None:
        self.confidence = float(confidence)
        self.grayscale = bool(grayscale)
        self.scales = tuple(float(s) for s in scales) or (1.0,)

    # -- public API ------------------------------------------------------
    def find(
        self,
        haystack: np.ndarray,
        template: Template,
        *,
        confidence: Optional[float] = None,
        scales: Optional[Sequence[float]] = None,
        grayscale: Optional[bool] = None,
    ) -> Optional[MatchResult]:
        """Best match at or above the threshold, or None."""
        results = self._match(haystack, template, confidence, scales, grayscale, max_results=1)
        return results[0] if results else None

    def find_all(
        self,
        haystack: np.ndarray,
        template: Template,
        *,
        confidence: Optional[float] = None,
        scales: Optional[Sequence[float]] = None,
        grayscale: Optional[bool] = None,
        max_results: int = 20,
        overlap_threshold: float = 0.3,
    ) -> List[MatchResult]:
        """Every distinct match, strongest first, overlaps suppressed."""
        return self._match(
            haystack,
            template,
            confidence,
            scales,
            grayscale,
            max_results=max_results,
            overlap_threshold=overlap_threshold,
        )

    def best_score(
        self,
        haystack: np.ndarray,
        template: Template,
        *,
        scales: Optional[Sequence[float]] = None,
        grayscale: Optional[bool] = None,
    ) -> float:
        """Peak similarity ignoring the threshold -- useful for tuning."""
        results = self._match(haystack, template, 0.0, scales, grayscale, max_results=1)
        return results[0].confidence if results else 0.0

    # -- internals -------------------------------------------------------
    def _match(
        self,
        haystack: np.ndarray,
        template: Template,
        confidence: Optional[float],
        scales: Optional[Sequence[float]],
        grayscale: Optional[bool] = None,
        *,
        max_results: int = 1,
        overlap_threshold: float = 0.3,
    ) -> List[MatchResult]:
        threshold = _first_not_none(confidence, template.confidence, self.confidence)
        scale_list = _first_not_none(scales, template.scales, self.scales)
        # Grayscale ignores color, which is exactly the difference between
        # e.g. gold/silver/bronze badges of otherwise identical shape -- let
        # a lookup opt into color matching to tell those apart.
        use_grayscale = _first_not_none(grayscale, template.grayscale, self.grayscale)

        # A mask forbids TM_CCOEFF_NORMED, so pick the method up front and use
        # the same one for every scale to keep scores comparable.
        use_mask = template.mask is not None
        method = cv2.TM_CCORR_NORMED if use_mask else cv2.TM_CCOEFF_NORMED

        haystack_prepared = self._prepare(haystack, use_grayscale)
        template_prepared = self._prepare(template.image, use_grayscale)

        candidates: List[MatchResult] = []
        for scale in scale_list:
            scaled_template, scaled_mask = _resize(template_prepared, template.mask, scale)
            if scaled_template is None:
                continue
            if (
                scaled_template.shape[0] > haystack_prepared.shape[0]
                or scaled_template.shape[1] > haystack_prepared.shape[1]
            ):
                log.debug(
                    "Skipping scale %.2f for %s: template %dx%d exceeds search area %dx%d",
                    scale,
                    template.name,
                    scaled_template.shape[1],
                    scaled_template.shape[0],
                    haystack_prepared.shape[1],
                    haystack_prepared.shape[0],
                )
                continue

            try:
                scores = cv2.matchTemplate(
                    haystack_prepared, scaled_template, method, mask=scaled_mask
                )
            except cv2.error as exc:  # pragma: no cover - depends on OpenCV build
                log.warning("matchTemplate failed for %s at scale %.2f: %s", template.name, scale, exc)
                continue

            # Masked matching can emit inf/nan where the mask sums to zero.
            scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

            height, width = scaled_template.shape[:2]
            if max_results == 1:
                _, max_val, _, max_loc = cv2.minMaxLoc(scores)
                if max_val >= threshold:
                    candidates.append(
                        MatchResult(
                            rect=Rect(int(max_loc[0]), int(max_loc[1]), width, height),
                            confidence=float(max_val),
                            scale=scale,
                            template_name=template.name,
                        )
                    )
            else:
                ys, xs = np.where(scores >= threshold)
                for x, y in zip(xs, ys):
                    candidates.append(
                        MatchResult(
                            rect=Rect(int(x), int(y), width, height),
                            confidence=float(scores[y, x]),
                            scale=scale,
                            template_name=template.name,
                        )
                    )

        if not candidates:
            return []

        candidates.sort(key=lambda m: m.confidence, reverse=True)
        if max_results == 1:
            return candidates[:1]
        return _suppress_overlaps(candidates, overlap_threshold)[:max_results]

    def _prepare(self, image: np.ndarray, grayscale: bool) -> np.ndarray:
        if grayscale and image.ndim == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image


def _resize(image: np.ndarray, mask: Optional[np.ndarray], scale: float):
    if abs(scale - 1.0) < 1e-6:
        return image, mask
    height = max(1, int(round(image.shape[0] * scale)))
    width = max(1, int(round(image.shape[1] * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(image, (width, height), interpolation=interpolation)
    resized_mask = (
        cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        if mask is not None
        else None
    )
    return resized, resized_mask


def _suppress_overlaps(matches: List[MatchResult], threshold: float) -> List[MatchResult]:
    """Greedy non-maximum suppression so one button is not reported five times."""
    kept: List[MatchResult] = []
    for match in matches:
        if all(match.rect.iou(other.rect) <= threshold for other in kept):
            kept.append(match)
    return kept


def _first_not_none(*values):
    for value in values:
        if value is not None:
            return value
    return None


def annotate(
    image: np.ndarray,
    matches: Sequence[MatchResult],
    *,
    color=(0, 220, 0),
    thickness: int = 2,
) -> np.ndarray:
    """Draw match boxes on a copy of ``image`` -- used for debug screenshots."""
    canvas = image.copy()
    if canvas.ndim == 2:
        canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
    for match in matches:
        left, top, right, bottom = match.rect.as_ltrb()
        cv2.rectangle(canvas, (left, top), (right, bottom), color, thickness)
        cv2.putText(
            canvas,
            f"{match.confidence:.2f}",
            (left, max(12, top - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
    return canvas
