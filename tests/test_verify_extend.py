"""`--verify-extend`: spend the check's answer on both halves of it.

`--verify-reach` presses, reads which tsums the game lights up, and uses that
answer only to take chain members AWAY. The same reading also names partners
the proposal never contained -- 4.0 of them per press, measured over 4,306
collected drags, against a proposal of 4.8 and a dragged chain of 3.4. The
press has already been paid for, so those are free.

The finding that makes it a rule rather than a wish is in
:func:`~ttheart_sender.game.tsum.chain_from_marks`: `adjacency()` refuses to
link two tsums whose `kind` differs, and `kind` is a per-frame k-means id, so
rebuilding with the bot's own ids makes chains SHORTER. The marks have to be
believed about identity, not just about refusal.

Same pair of properties every opt-in rule here is held to: it does its job
when asked, and a run that does not ask for it behaves exactly as it did
before the option existed.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from ttheart_sender.game import tsum


class Templates:
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
    opts.verify = False
    for key, value in over.items():
        setattr(opts, key, value)
    return opts


#: The chain the bot proposes, and two tsums beside it that the game will mark
#: but the proposal never held. Everything sits on one line with every leg
#: inside `link_px`, so the only thing that can stop the rebuild walking to
#: the marked pair is `kind`. Four members because the chain also has to reach
#: past 260px for the check to fire at all, and a leg may not exceed 105px.
CHAIN = [0, 1, 2, 3]
EXTRA = [4, 5]

#: How far apart the tsums on that line sit, once `board.reach` has stretched
#: it. Under `link_px` 105, and clear of the `block` radius either side.
LEG = 100.0


@pytest.fixture
def board(monkeypatch):
    frame = np.zeros((994, 578, 3), np.uint8)
    # 300px: past the 260 threshold, and it divides into three 100px legs --
    # both inside `link_px` 105, which the rebuild has to walk.
    state = SimpleNamespace(reach=300.0, extra_kind=1)

    def detect(crop, **kw):
        # The chain spans `reach` in even steps and the two marked tsums
        # continue the same line, so what decides is the reach, not the shape.
        step = state.reach / (len(CHAIN) - 1)
        pts = [(step * i, 0.0) for i in range(len(CHAIN))]
        pts += [(state.reach + LEG * (i + 1), 0.0) for i in range(len(EXTRA))]
        kinds = [1] * len(CHAIN) + [state.extra_kind] * len(EXTRA)
        pts += [(20.0 + 9.0 * i, 300.0) for i in range(27)]
        kinds += [2] * 27
        tsums = [tsum.Tsum(x=x, y=y, r=25.0, kind=k, colour=(0, 0, 0))
                 for (x, y), k in zip(pts, kinds)]
        return tsums, 25.0, np.zeros((12, 3), np.float32)

    monkeypatch.setattr(tsum, "_settle", lambda drv, max_wait=0.0, tol=2.5, region=None, out=None: frame)
    monkeypatch.setattr(tsum, "detect", detect)
    monkeypatch.setattr(tsum, "find_chains",
                        lambda *a, **kw: [tsum.Chain(1, (0, 0, 0), list(CHAIN))])
    monkeypatch.setattr(tsum, "purity_filter",
                        lambda bgr, tsums, nodes, r, tol: list(nodes))
    state.frame = frame
    return state


@pytest.fixture
def spy(monkeypatch):
    """The game keeps the whole proposal and marks two tsums beside it."""
    seen = SimpleNamespace(dragged=[], floor_mults=[])

    def drag_chain(points, *, step_px=8.0, per_step=0.006, hold=0.0,
                   after_press=None, **kw):
        if after_press is not None:
            points = after_press() or points
        seen.dragged.append(len(points))

    def marked_by_game(drv, before, board_rect, tsums, nodes, *,
                       delay, floor_mult=0.0, **kw):
        seen.floor_mults.append(floor_mult)
        out = kw.get("out")
        if out is not None:
            out.update(marked_frame=None, values=np.zeros(len(tsums)),
                       baseline=0.0, bar=8.0, marked=list(EXTRA))
        return list(nodes)

    monkeypatch.setattr(tsum, "drag_chain", drag_chain)
    monkeypatch.setattr(tsum, "marked_by_game", marked_by_game)
    return seen


def _run(board, opts, drags=3):
    n = {"i": 0}

    def stop_when(_frame):
        n["i"] += 1
        return "stopping" if n["i"] > drags else ""

    return tsum.play_loop(_driver(board.frame), opts, stop_when=stop_when)


# -- inert when not asked for ----------------------------------------------
def test_it_is_off_by_default():
    opts = tsum.play_defaults()
    assert opts.verify_extend is False, "a new play rule ships off"


def test_a_checked_chain_is_only_trimmed_when_the_rule_is_off(board, spy):
    # The whole revert is one line, so "off" has to mean the old behaviour on
    # exactly the boards the rule would have fired on.
    report = _run(board, _options(verify_reach=260.0))

    assert spy.dragged and all(n == len(CHAIN) for n in spy.dragged)
    assert report.extended == 0 and report.extended_by == 0


def test_the_strict_bar_is_not_paid_for_when_the_rule_is_off(board, spy):
    """`floor_mult` moves the bar the marks are READ at, and nothing else.

    Off, the live check must keep reading at the flat threshold its own A/B
    was measured under -- raising it here would re-score `--verify-reach`
    silently, and only in the combination where both happened to be on.
    """
    _run(board, _options(verify_reach=260.0))
    assert spy.floor_mults and all(m == 0.0 for m in spy.floor_mults)


# -- does its job when asked -----------------------------------------------
def test_the_marks_lengthen_a_checked_chain(board, spy):
    report = _run(board, _options(verify_reach=260.0, verify_extend=True))

    assert spy.dragged, "nothing was dragged, so nothing is being asserted"
    assert all(n == len(CHAIN) + len(EXTRA) for n in spy.dragged)
    assert report.extended == len(spy.dragged)
    assert report.extended_by == len(EXTRA) * len(spy.dragged)


def test_it_reads_the_marks_at_the_strict_bar(board, spy):
    # A mark read at the flat 8.0 admits 23-35 tsums per press on a live
    # board. Building a chain through that is building one through the noise.
    _run(board, _options(verify_reach=260.0, verify_extend=True))
    assert spy.floor_mults and all(m == 8.0 for m in spy.floor_mults)


def test_a_near_chain_is_never_rebuilt_because_it_is_never_checked(board, spy):
    # The rule rides on a reading that has already been bought. It must not
    # start buying readings of its own.
    board.reach = 20.0
    report = _run(board, _options(verify_reach=260.0, verify_extend=True))

    assert spy.dragged and spy.floor_mults == []
    assert report.extended == 0


# -- the finding the rule rests on -----------------------------------------
def test_the_game_is_believed_about_identity_not_just_about_refusal():
    """Marked tsums the detector called a different colour still get chained.

    This is the whole finding. With the bot's own `kind` ids the rebuild
    cannot link them and the chain gets shorter, not longer -- measured at
    -29% cleared over the drags a 260px check fires on.
    """
    tsums = [tsum.Tsum(x=LEG * i, y=0.0, r=25.0,
                       kind=(1 if i < len(CHAIN) else 7), colour=(0, 0, 0))
             for i in range(len(CHAIN) + len(EXTRA))]
    grown = tsum.chain_from_marks(tsums, CHAIN, CHAIN, EXTRA, 25.0,
                                  link_px=105.0, block=1.25, max_chain=12)
    assert len(grown) == len(CHAIN) + len(EXTRA),         "the marked tsums are partners, whatever k-means said"
    assert grown[0] == CHAIN[0], "the chain still starts at the tsum pressed"


def test_it_never_returns_less_than_the_trim():
    # A rebuild that cannot beat the trim has to hand the trim back unchanged:
    # the rule may add clears, it may never cost any.
    tsums = [tsum.Tsum(x=1000.0 * i, y=0.0, r=25.0, kind=1, colour=(0, 0, 0))
             for i in range(len(CHAIN) + len(EXTRA))]
    grown = tsum.chain_from_marks(tsums, CHAIN, CHAIN, EXTRA, 25.0,
                                  link_px=105.0, block=1.25, max_chain=12)
    assert grown == CHAIN, "unreachable marks must not shorten the drag"


def test_the_cap_still_holds():
    tsums = [tsum.Tsum(x=LEG * i, y=0.0, r=25.0, kind=1, colour=(0, 0, 0))
             for i in range(len(CHAIN) + len(EXTRA))]
    grown = tsum.chain_from_marks(tsums, CHAIN, CHAIN, EXTRA, 25.0,
                                  link_px=105.0, block=1.25, max_chain=5)
    assert len(grown) == 5, "max_chain still caps a chain the marks grew"


# -- the interaction with the clear check ----------------------------------
@pytest.fixture
def clear_spy(monkeypatch):
    """Record which nodes the clear check was asked about."""
    seen = SimpleNamespace(asked=[])

    def cleared_by_drag(before_crop, after_crop, tsums, nodes, *, tol):
        seen.asked.append(list(nodes))
        return list(nodes), [99.0] * len(nodes), 1.0

    monkeypatch.setattr(tsum, "cleared_by_drag", cleared_by_drag)
    return seen


def _moving_driver(frame):
    """A board that visibly changes after the drag, so `--verify` is happy.

    `before` is the crop the chain was scanned from and `after` is the next
    grab. Two identical frames read as a drag that never registered, and the
    loop takes the stall path instead of ever reaching the clear check -- so
    every grab here comes back different from the last.
    """
    shade = {"n": 0}

    def grab():
        shade["n"] = (shade["n"] + 1) % 4
        return np.full_like(frame, 60 * shade["n"])

    return SimpleNamespace(
        capture=None, matcher=None, templates=Templates(), rect=None,
        grab=grab,
        to_screen=lambda x, y: (int(x), int(y)),
        check_stop=lambda: None,
        say=lambda msg: None,
    )


def _run_verifying(board, opts, drags=1):
    n = {"i": 0}

    def stop_when(_frame):
        n["i"] += 1
        return "stopping" if n["i"] > drags else ""

    return tsum.play_loop(_moving_driver(board.frame), opts, stop_when=stop_when)


def test_the_clear_check_measures_what_was_dragged_not_what_was_proposed(
        board, spy, clear_spy):
    """The two rules have to compose, or the A/B scores the new one at zero.

    `cleared_by_drag` asks which of the DRAGGED tsums are gone. With the marks
    rebuilding the chain, the members they added are not in the proposal at
    all -- so measuring the proposal would count every tsum the rule won as
    having cleared nothing, and "Measure tsums cleared" is exactly the round
    this rule has to be judged by.
    """
    report = _run_verifying(board, _options(verify_reach=260.0, verify_extend=True,
                                            verify=True, verify_clears=True))

    assert clear_spy.asked, "the clear check never ran, so nothing is asserted"
    for nodes in clear_spy.asked:
        assert nodes == CHAIN + EXTRA, "the rebuilt chain is what was dragged"
    assert report.cleared == len(CHAIN + EXTRA) * len(clear_spy.asked)


def test_a_chain_the_marks_grew_is_never_reported_as_trimmed(board, spy):
    # `dropped` used to be a plain subtraction, which goes negative the moment
    # the rebuild returns more than the proposal -- a negative that is truthy,
    # adds itself to `trimmed`, and prints as "dropped -2 it would not accept".
    report = _run_verifying(board, _options(verify_reach=260.0, verify_extend=True,
                                            verify=True))

    assert report.extended > 0, "the rule has to have fired for this to mean anything"
    assert report.trimmed == 0
