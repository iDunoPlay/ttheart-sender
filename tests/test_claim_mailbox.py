"""The single-claim mailbox loop, against a mailbox that eats taps.

The real one does: a claim dialog is modal, and it goes on swallowing taps for
a beat after its OK button has stopped matching -- while the list behind it,
Check button and all, never stops being visible. A tap sent into that window
lands on nothing, and the log reads as a confident `find_click check_button`
followed by no dialog at all.

Nothing here touches pixels: `find` answers from the fake mailbox's state and
`sleep` moves its clock instead of the wall's, so a full mailbox runs in
milliseconds.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ttheart_sender.automation.flow import load_flow
from ttheart_sender.automation.runner import FlowRunner
from ttheart_sender.geometry import Point, Rect
from ttheart_sender.screen.matcher import MatchResult

from test_runner import make_context

FLOW = Path(__file__).resolve().parents[1] / "flows" / "claim_mailbox.yaml"

#: Where the fake draws each thing, in the same places the game does: Check
#: high up in the list, OK far below it in the dialog. They do not overlap --
#: the tap is lost to the dialog's overlay, not to anything covering Check.
CHECK_RECT = Rect(389, 317, 71, 28)
OK_RECT = Rect(337, 570, 102, 41)


class FakeMailbox:
    """A mailbox of ``items`` gifts, claimed one dialog at a time."""

    def __init__(self, items: int, *, blocked_for: float = 2.0) -> None:
        self.items = items
        #: How long taps keep going nowhere after a dialog is dismissed.
        self.blocked_for = blocked_for
        self.now = 0.0
        self.blocked_until = 0.0
        self.dialog_open = False
        self.claimed = 0
        self.taps = 0
        self.lost_taps = 0

    # -- the screen ------------------------------------------------------
    def find(self, template: str, **_kwargs):
        if template == "get_coin":
            return self._match(template, Rect(0, 0, 10, 10))
        if template == "check_button":
            # Visible whenever the mailbox has anything left, dialog or not --
            # this is the part that makes a swallowed tap look like a hit.
            return self._match(template, CHECK_RECT) if self.items else None
        if template == "ok_button":
            return self._match(template, OK_RECT) if self.dialog_open else None
        return None

    def _match(self, template: str, rect: Rect) -> MatchResult:
        return MatchResult(rect=rect, confidence=0.98, template_name=template)

    # -- input -----------------------------------------------------------
    def click(self, point: Point, **_kwargs) -> None:
        self.taps += 1
        if self.now < self.blocked_until:
            self.lost_taps += 1
            return
        if self.dialog_open:
            if OK_RECT.contains(point):
                self.dialog_open = False
                self.items -= 1
                self.claimed += 1
                self.blocked_until = self.now + self.blocked_for
            else:
                self.lost_taps += 1  # modal: everything else is dead
            return
        if CHECK_RECT.contains(point) and self.items:
            self.dialog_open = True

    def sleep(self, seconds: float) -> None:
        self.now += seconds


def claim(items: int, **kwargs):
    mailbox = FakeMailbox(items, **kwargs)
    ctx = make_context()
    ctx.find = mailbox.find
    ctx.click_screen = mailbox.click
    ctx.sleep = mailbox.sleep
    report = FlowRunner(ctx).run(load_flow(FLOW))
    return mailbox, ctx, report


@pytest.mark.parametrize("items", [1, 3, 8])
def test_every_gift_is_claimed(items):
    mailbox, _ctx, report = claim(items)

    assert mailbox.claimed == items
    assert mailbox.items == 0
    assert report.success


def test_a_swallowed_tap_is_repeated_rather_than_costing_the_item():
    """The regression: one item per pass around the loop, tap or no tap.

    ``index`` is the outer loop's counter, so the last value it held says how
    many passes the five gifts took. Before, a tap lost to the closing dialog
    left the pass with nothing to claim and the next one picked the item up --
    five gifts, ten passes, and the log alternating between a dialog and none.
    """
    mailbox, ctx, _report = claim(5)

    assert ctx.variables["item"] == 4, "a pass claimed nothing"
    assert mailbox.lost_taps > 0, "the mailbox never ate a tap -- test proves nothing"


def test_an_empty_mailbox_is_left_alone():
    mailbox, _ctx, _report = claim(0)

    assert mailbox.taps == 0
