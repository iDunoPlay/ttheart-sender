"""`--verify-reach`: buy the game's opinion only on the chains likely to be wrong.

`--verify-hold` asks the game to check every drag, and that is why it does not
pay -- the check costs the same each time but being wrong does not. Measured
over 303 collected drags, the share where the game accepted every proposed
member falls away with how far the chain reaches from the press: 100% under
90px, 81% at 90-150, 65% at 150-220, 33% at 220-300, 11% beyond. So the check
is bought where the risk is.

The pair of properties that matter, as for every other opt-in rule here: it
does its job when asked, and a run that does not ask for it behaves exactly as
it did before the option existed.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from ttheart_sender.game import tsum
from ttheart_sender.game.dataset import RENDER_FLOOR


class Templates:
    """No max_fever template -- FeverWatch degrades to 'never FEVER'."""

    def get(self, name):
        raise KeyError(name)


def _driver(frame):
    return SimpleNamespace(
        capture=None, matcher=None, templates=Templates(), rect=None,
        grab=lambda: frame,
        to_screen=lambda x, y: (int(x), int(y)),
        check_stop=lambda: None,
        say=lambda msg: None,
    )


def _options(**over):
    opts = tsum.play_defaults()
    opts.duration = 30.0
    opts.countdown = 0.0
    opts.dry_run = False
    opts.settle = 0.0
    opts.move_time = 0.0
    opts.hold = 0.0
    opts.skill = ""
    opts.bubble = ""
    opts.use_base = False
    opts.verify = False          # no clear-check: it grabs a second frame
    for key, value in over.items():
        setattr(opts, key, value)
    return opts


@pytest.fixture
def board(monkeypatch):
    """One playable chain per frame, with the far member's distance settable.

    ``board.reach`` moves the last member of the chain, which is the only
    geometry this rule reads.
    """
    frame = np.zeros((994, 578, 3), np.uint8)
    state = SimpleNamespace(reach=20.0)

    def detect(crop, **kw):
        # Enough detections to clear `--min-tsums`, which exists to recognise
        # that a frame is a board at all. The first three are the chain; the
        # rest are filler the loop never touches.
        pts = [(0.0, 0.0), (10.0, 0.0), (state.reach, 0.0)]
        pts += [(20.0 + 9.0 * i, 300.0) for i in range(27)]
        tsums = [tsum.Tsum(x=x, y=y, r=25.0, kind=1, colour=(0, 0, 0)) for x, y in pts]
        return tsums, 25.0, np.zeros((12, 3), np.float32)

    monkeypatch.setattr(tsum, "_settle", lambda drv, max_wait=0.0: frame)
    monkeypatch.setattr(tsum, "detect", detect)
    monkeypatch.setattr(tsum, "find_chains",
                        lambda *a, **kw: [tsum.Chain(1, (0, 0, 0), [0, 1, 2])])
    monkeypatch.setattr(tsum, "purity_filter", lambda bgr, tsums, nodes, r, tol: list(nodes))
    state.frame = frame
    return state


@pytest.fixture
def spy(monkeypatch):
    """Record whether each drag paid for a check, and at what delay."""
    seen = SimpleNamespace(checks=[], drags=0)

    def drag_chain(points, *, step_px=8.0, per_step=0.006, hold=0.0,
                   after_press=None, **kw):
        seen.drags += 1
        if after_press is not None:
            after_press()

    def marked_by_game(drv, before, board_rect, tsums, nodes, *, delay, **kw):
        seen.checks.append(delay)
        out = kw.get("out")
        if out is not None:
            out.update(marked_frame=None, values=np.zeros(len(tsums)),
                       baseline=0.0, bar=8.0, marked=[])
        return list(nodes)          # the game accepts the whole chain

    monkeypatch.setattr(tsum, "drag_chain", drag_chain)
    monkeypatch.setattr(tsum, "marked_by_game", marked_by_game)
    return seen


def _run(board, opts, drags=3):
    """Play until `drags` chains have gone, then stop."""
    n = {"i": 0}

    def stop_when(_frame):
        n["i"] += 1
        return "stopping" if n["i"] > drags else ""

    return tsum.play_loop(_driver(board.frame), opts, stop_when=stop_when)


# -- inert when not asked for ----------------------------------------------
def test_it_is_off_by_default():
    opts = tsum.play_defaults()
    assert opts.verify_reach == 0.0, "a new rule ships off"


def test_a_far_chain_is_dragged_blind_when_the_rule_is_off(board, spy):
    # The whole revert is deleting the line, so "off" has to mean the old
    # behaviour on exactly the boards the rule would have fired on.
    board.reach = 400.0
    report = _run(board, _options(verify_reach=0.0))

    assert spy.drags > 0, "the fixture has to drag something to prove anything"
    assert spy.checks == [], "nothing may be asked of the game"
    assert report.verified == 0


# -- does its job when asked -----------------------------------------------
def test_a_near_chain_is_not_worth_checking(board, spy):
    # Under 90px the game accepted every member of every collected chain, so
    # paying there is pure loss.
    board.reach = 20.0
    report = _run(board, _options(verify_reach=260.0))

    assert spy.drags > 0
    assert spy.checks == []
    assert report.verified == 0


def test_a_far_chain_is_checked_before_it_is_dragged(board, spy):
    board.reach = 400.0
    report = _run(board, _options(verify_reach=260.0))

    assert len(spy.checks) == spy.drags, "every drag past the threshold is checked"
    assert report.verified == spy.drags


def test_the_threshold_is_the_reach_not_the_chain_length(board, spy):
    # Straddling it in both directions with the same three-member chain: what
    # decides is how far the chain gets from the press, which is the thing the
    # collected drags actually separate on.
    board.reach = 259.0
    assert _run(board, _options(verify_reach=260.0)).verified == 0
    board.reach = 261.0
    assert _run(board, _options(verify_reach=260.0)).verified > 0


# -- the delay, which is why --verify-hold reads noise ----------------------
def test_the_check_reads_at_a_delay_the_highlight_has_rendered_at(board, spy):
    """`--hold-delay` defaults to 0.10, below the 0.15 floor.

    That is the whole reason `--verify-hold` is documented as measured-and-not-
    recommended: it was scored against a mark the game had not drawn. Re-using
    its number here would reproduce the finding rather than the feature.
    """
    board.reach = 400.0
    _run(board, _options(verify_reach=260.0))

    assert spy.checks, "nothing was checked, so nothing is being asserted"
    assert all(d >= RENDER_FLOOR for d in spy.checks)
    assert all(d == 0.25 for d in spy.checks), "the default is the collector's"


def test_a_delay_under_the_render_floor_is_raised_rather_than_obeyed(board, spy):
    # Clamped the way DatasetWriter clamps its own: obeying it silently is how
    # a feature gets measured in the one condition where it cannot work.
    board.reach = 400.0
    _run(board, _options(verify_reach=260.0, verify_delay=0.01))

    assert spy.checks and all(d == RENDER_FLOOR for d in spy.checks)


def test_verify_hold_keeps_its_own_delay_and_still_checks_everything(board, spy):
    # Its A/B was measured at --hold-delay, so this must not quietly re-tune
    # it -- and "every drag" has to keep meaning every drag, near or far.
    board.reach = 20.0
    report = _run(board, _options(verify_hold=True, hold_delay=0.10))

    assert len(spy.checks) == spy.drags, "verify_hold checks near chains too"
    assert all(d == 0.10 for d in spy.checks)
    assert report.verified == spy.drags
