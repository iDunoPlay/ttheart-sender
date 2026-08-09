"""Small immutable geometry value objects shared across the app.

Rectangles are always ``(left, top, width, height)``. Whether the coordinates
are screen-absolute or relative to something else is documented at each call
site -- the type itself is deliberately unopinionated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple


@dataclass(frozen=True)
class Point:
    x: int
    y: int

    def offset(self, dx: int = 0, dy: int = 0) -> "Point":
        return Point(self.x + dx, self.y + dy)

    def as_tuple(self) -> Tuple[int, int]:
        return (self.x, self.y)

    def __iter__(self) -> Iterable[int]:
        yield self.x
        yield self.y

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return f"({self.x}, {self.y})"


@dataclass(frozen=True)
class Insets:
    """Pixels to trim from each edge of a rectangle."""

    left: int = 0
    top: int = 0
    right: int = 0
    bottom: int = 0

    @property
    def is_empty(self) -> bool:
        return not (self.left or self.top or self.right or self.bottom)


@dataclass(frozen=True)
class Rect:
    left: int
    top: int
    width: int
    height: int

    # -- constructors ---------------------------------------------------
    @classmethod
    def from_ltrb(cls, left: int, top: int, right: int, bottom: int) -> "Rect":
        return cls(int(left), int(top), int(right - left), int(bottom - top))

    @classmethod
    def from_sequence(cls, values: Sequence[int]) -> "Rect":
        if len(values) != 4:
            raise ValueError(f"expected 4 values [x, y, width, height], got {len(values)}")
        x, y, w, h = (int(v) for v in values)
        return cls(x, y, w, h)

    # -- derived properties ---------------------------------------------
    @property
    def right(self) -> int:
        return self.left + self.width

    @property
    def bottom(self) -> int:
        return self.top + self.height

    @property
    def area(self) -> int:
        return max(0, self.width) * max(0, self.height)

    @property
    def is_empty(self) -> bool:
        return self.width <= 0 or self.height <= 0

    @property
    def origin(self) -> Point:
        return Point(self.left, self.top)

    @property
    def center(self) -> Point:
        return Point(self.left + self.width // 2, self.top + self.height // 2)

    # -- operations ------------------------------------------------------
    def shrink(self, insets: Insets) -> "Rect":
        """Trim the rectangle by ``insets``, clamped so it never inverts."""
        left = self.left + insets.left
        top = self.top + insets.top
        right = max(left, self.right - insets.right)
        bottom = max(top, self.bottom - insets.bottom)
        return Rect.from_ltrb(left, top, right, bottom)

    def translate(self, dx: int = 0, dy: int = 0) -> "Rect":
        return Rect(self.left + dx, self.top + dy, self.width, self.height)

    def relative_to(self, other: "Rect") -> "Rect":
        """Express this rect in a coordinate space whose origin is ``other``."""
        return self.translate(-other.left, -other.top)

    def contains(self, point: Point) -> bool:
        return self.left <= point.x < self.right and self.top <= point.y < self.bottom

    def intersect(self, other: "Rect") -> "Rect":
        left = max(self.left, other.left)
        top = max(self.top, other.top)
        right = min(self.right, other.right)
        bottom = min(self.bottom, other.bottom)
        if right <= left or bottom <= top:
            return Rect(left, top, 0, 0)
        return Rect.from_ltrb(left, top, right, bottom)

    def iou(self, other: "Rect") -> float:
        inter = self.intersect(other).area
        if inter == 0:
            return 0.0
        union = self.area + other.area - inter
        return inter / union if union else 0.0

    def as_ltrb(self) -> Tuple[int, int, int, int]:
        return (self.left, self.top, self.right, self.bottom)

    def as_tuple(self) -> Tuple[int, int, int, int]:
        return (self.left, self.top, self.width, self.height)

    def as_mss_dict(self) -> dict:
        return {"left": self.left, "top": self.top, "width": self.width, "height": self.height}

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return f"[x={self.left} y={self.top} w={self.width} h={self.height}]"
