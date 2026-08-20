"""Cursor stop zone: arming, tripping, and how it reaches the runner."""

from __future__ import annotations

import pytest

from ttheart_sender.control.cursor import CursorGuard
from ttheart_sender.control.hotkey import StopKeyWatcher
from ttheart_sender.exceptions import StopRequested
from ttheart_sender.geometry import Point, Rect

WINDOW = Rect(100, 100, 500, 900)


def make_guard(points, rect=WINDOW, **kwargs):
    """Guard fed a scripted sequence of cursor positions."""
    cursor = iter(points)
    return CursorGuard(
        lambda: rect,
        position=lambda: next(cursor),
        rect_ttl=0.0,
        **kwargs,
    )


def test_guard_stays_quiet_until_the_cursor_has_been_inside():
    # A run started from a terminal on another monitor must not stop on its
    # first checkpoint -- the flow's first click is what brings the cursor in.
    guard = make_guard([Point(5, 5), Point(2000, 5), Point(300, 500)])
    assert guard() is None
    assert guard() is None
    assert not guard.armed
    assert guard() is None
    assert guard.armed


def test_guard_trips_once_the_cursor_leaves():
    guard = make_guard([Point(300, 500), Point(2000, 500)])
    assert guard() is None
    reason = guard()
    assert reason and "outside" in reason


def test_margin_keeps_edge_clicks_inside():
    just_outside = Point(WINDOW.right + 8, 500)
    guard = make_guard([Point(300, 500), just_outside], margin=16)
    assert guard() is None
    assert guard() is None

    guard = make_guard([Point(300, 500), just_outside], margin=0)
    assert guard() is None
    assert guard()


def test_unknown_window_never_trips():
    guard = CursorGuard(lambda: None, position=lambda: Point(9999, 9999), rect_ttl=0.0)
    assert guard() is None


def test_unreadable_cursor_never_trips():
    guard = CursorGuard(lambda: WINDOW, position=lambda: None, rect_ttl=0.0)
    assert guard() is None


def test_zone_is_cached_between_checks():
    reads = []

    def rect_provider():
        reads.append(1)
        return WINDOW

    guard = CursorGuard(rect_provider, position=lambda: Point(300, 500), rect_ttl=60.0)
    for _ in range(5):
        guard()
    assert len(reads) == 1


def test_watcher_reports_the_guard_that_stopped_the_run():
    watcher = StopKeyWatcher(None)
    guard = make_guard([Point(300, 500), Point(2000, 500)])
    watcher.add_guard("cursor", guard)

    watcher.check()  # inside: no complaint
    with pytest.raises(StopRequested) as excinfo:
        watcher.check()
    assert "cursor" in str(excinfo.value)


def test_reset_disarms_the_guard_for_the_next_run():
    watcher = StopKeyWatcher(None)
    guard = make_guard([Point(300, 500), Point(2000, 500), Point(2000, 500)])
    watcher.add_guard("cursor", guard)

    watcher.check()
    with pytest.raises(StopRequested):
        watcher.check()

    watcher.reset()
    # Disarmed: the cursor is still outside, but the next run gets to start.
    assert not guard.armed
    watcher.check()


# -- the panel's switch ---------------------------------------------------
def test_the_guard_stands_down_while_it_is_switched_off():
    """The panel flips this mid-run, so it is asked on every poll.

    Registering the guard conditionally at startup -- what the code used to do
    -- froze the answer for the life of the Application, which is no use to a
    tick box.
    """
    on = {"value": True}
    guard = CursorGuard(lambda: Rect(0, 0, 100, 100),
                        position=lambda: Point(50, 50),
                        enabled=lambda: on["value"])

    assert guard() is None            # inside: arms it
    assert guard.armed

    guard.position = lambda: Point(500, 500)
    assert guard() is not None, "outside and armed should stop the run"

    on["value"] = False
    assert guard() is None, "switched off, it must not stop anything"


def test_switching_it_off_disarms_so_it_does_not_trip_on_the_way_back():
    """Otherwise re-ticking the box stops the run instantly.

    The pointer is wherever the user left it while the guard was off -- most
    likely outside -- and an armed guard would trip on the next poll before
    the cursor ever came home.
    """
    on = {"value": True}
    at = {"point": Point(50, 50)}
    guard = CursorGuard(lambda: Rect(0, 0, 100, 100),
                        position=lambda: at["point"],
                        enabled=lambda: on["value"])
    guard()
    assert guard.armed

    on["value"] = False
    at["point"] = Point(900, 900)
    assert guard() is None
    assert not guard.armed, "should have disarmed while off"

    on["value"] = True
    assert guard() is None, "still outside, but no longer armed -- must not trip"

    at["point"] = Point(50, 50)
    guard()
    assert guard.armed, "coming home re-arms it"
    at["point"] = Point(900, 900)
    assert guard() is not None
