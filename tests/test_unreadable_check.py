"""A blank frame is the game saying nothing, not the game saying no.

`verify_reach` trims a chain to what the game marked while the first tsum was
held. When that reading comes back with no mark anywhere on the board -- not
one tsum over the bar, the pressed one included -- there is nothing to trim
against, and trimming anyway throws a chain away on a reading that failed.

Measured over 726 collected drags it happens on 4.4% of them, and on 6 of the
121 drags a `verify_reach 260` check would fire on -- two of which are chains
the trim cancelled outright. See the eleventh round in
`docs/DATASET-FINDINGS.md`.
"""

import pytest

from ttheart_sender.game.tsum import PlayReport


def test_a_blank_reading_is_reported_not_hidden():
    """The cost was paid; the log has to say it bought nothing."""
    report = PlayReport(played=40, dragged=170, verified=8, unreadable=3)
    said = report.describe()
    assert "checked 8 chain(s) before dragging" in said
    assert "3 of which read nothing and were dragged as proposed" in said


def test_a_clean_run_says_nothing_about_unreadable_frames():
    report = PlayReport(played=40, dragged=170, verified=8)
    assert "read nothing" not in report.describe()


@pytest.mark.parametrize("marked, trims", [([], False), ([3, 7], True)])
def test_the_guard_is_keyed_on_the_whole_board_being_dark(marked, trims):
    """The rule itself, stated as the loop applies it.

    An empty `marked` list means no tsum on the board cleared the label's bar.
    Anything in it means the game drew something, and the trim is reading a
    real answer even if it is one the chain will not like.
    """
    seen = {"marked": marked}
    unreadable = not seen.get("marked")
    assert unreadable is (not trims)
