from __future__ import annotations

from ttheart_sender.geometry import Insets, Point, Rect


def test_rect_from_ltrb_and_derived_values():
    rect = Rect.from_ltrb(10, 20, 110, 70)
    assert rect.as_tuple() == (10, 20, 100, 50)
    assert rect.right == 110 and rect.bottom == 70
    assert rect.center == Point(60, 45)
    assert rect.area == 5000


def test_shrink_clamps_instead_of_inverting():
    rect = Rect(0, 0, 100, 100)
    assert rect.shrink(Insets(10, 10, 10, 10)).as_tuple() == (10, 10, 80, 80)
    collapsed = rect.shrink(Insets(80, 80, 80, 80))
    assert collapsed.width == 0 and collapsed.height == 0


def test_relative_and_translate_round_trip():
    content = Rect(100, 50, 800, 600)
    absolute = Rect(150, 100, 20, 20)
    relative = absolute.relative_to(content)
    assert relative.origin == Point(50, 50)
    assert relative.translate(content.left, content.top) == absolute


def test_intersect_and_iou():
    a = Rect(0, 0, 100, 100)
    b = Rect(50, 50, 100, 100)
    assert a.intersect(b).as_tuple() == (50, 50, 50, 50)
    assert 0.14 < a.iou(b) < 0.15
    assert a.intersect(Rect(500, 500, 10, 10)).is_empty
    assert a.iou(Rect(500, 500, 10, 10)) == 0.0


def test_contains():
    rect = Rect(0, 0, 10, 10)
    assert rect.contains(Point(0, 0))
    assert not rect.contains(Point(10, 10))
