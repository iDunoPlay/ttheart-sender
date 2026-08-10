"""Generate the tray/app icons.

Run this only when the artwork changes -- the ``.ico`` files it writes are
committed, so neither the app nor the build needs Pillow at runtime:

    python scripts/make_icons.py
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Tuple

from PIL import Image, ImageDraw

OUT_DIR = Path(__file__).resolve().parent.parent / "ttheart_sender" / "tray" / "assets"

#: Sizes Windows picks between for the tray, taskbar, alt-tab and Explorer.
SIZES = (16, 20, 24, 32, 48, 64, 128, 256)
#: Draw big and downsample -- the curve stays smooth at 16x16.
CANVAS = 1024

RGBA = Tuple[int, int, int, int]

ICONS = {
    # name: (fill, outline)
    "tray-running": ((228, 42, 74, 255), (140, 18, 40, 255)),
    "tray-idle": ((150, 152, 160, 255), (92, 94, 102, 255)),
}


def heart_points(size: int, *, samples: int = 720, margin: float = 0.06):
    """The classic ``16 sin^3 t`` heart, fitted to a square of ``size``."""
    raw = []
    for index in range(samples):
        t = 2.0 * math.pi * index / samples
        x = 16.0 * math.sin(t) ** 3
        y = (
            13.0 * math.cos(t)
            - 5.0 * math.cos(2.0 * t)
            - 2.0 * math.cos(3.0 * t)
            - math.cos(4.0 * t)
        )
        raw.append((x, -y))  # screen Y grows downward

    min_x = min(p[0] for p in raw)
    max_x = max(p[0] for p in raw)
    min_y = min(p[1] for p in raw)
    max_y = max(p[1] for p in raw)

    inset = size * margin
    span = size - 2 * inset
    scale = min(span / (max_x - min_x), span / (max_y - min_y))
    offset_x = (size - (max_x - min_x) * scale) / 2.0
    offset_y = (size - (max_y - min_y) * scale) / 2.0

    return [
        ((x - min_x) * scale + offset_x, (y - min_y) * scale + offset_y)
        for x, y in raw
    ]


def render(fill: RGBA, outline: RGBA) -> Image.Image:
    image = Image.new("RGBA", (CANVAS, CANVAS), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    points = heart_points(CANVAS)
    draw.polygon(points, fill=fill, outline=outline)
    draw.line(points + [points[0]], fill=outline, width=int(CANVAS * 0.035), joint="curve")

    # A soft highlight so the shape still reads as a heart at 16x16.
    highlight = Image.new("RGBA", (CANVAS, CANVAS), (0, 0, 0, 0))
    ImageDraw.Draw(highlight).ellipse(
        (CANVAS * 0.24, CANVAS * 0.20, CANVAS * 0.44, CANVAS * 0.36),
        fill=(255, 255, 255, 90),
    )
    return Image.alpha_composite(image, highlight)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, (fill, outline) in ICONS.items():
        master = render(fill, outline)
        path = OUT_DIR / f"{name}.ico"
        master.save(path, format="ICO", sizes=[(s, s) for s in SIZES])
        print(f"wrote {path} ({', '.join(f'{s}x{s}' for s in SIZES)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
