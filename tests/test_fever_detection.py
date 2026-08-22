"""FEVER: the radius has to survive it, and the count floor has to give way.

The failure this pins down has no symptom in the log. FEVER repaints the board,
`play_loop` threw the measured radius away at the transition, and the estimator
re-ran on the dimmest and most animated frames of the round -- where it
collapses. A collapsed radius reads the board at half scale, which roughly
doubles the detection count, and a doubled count sits comfortably inside
`--min-tsums`..`--max-tsums`. So the frame is accepted, phantom chains are
found and dragged, nothing clears, and because chains *were* found the
"nothing playable -> tap the shuffle" recovery never runs either. Ten seconds
of FEVER pass with the bot busy and the board untouched.

Measured over 151 captured frames of one round: the per-frame radius estimate
ranges 8-38px on ~26px faces, 55% of frames read under 18px, and 50 of those
pass the count gate anyway.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from ttheart_sender.game import tsum


class Templates:
    """Serves `max_fever` only, which is what makes this file about the radius.

    Handing over a `fever_bonus` banner as well changes what `FeverWatch` is:
    with a banner it reads FEVER off the screen every frame, without one it
    infers it from the meter and a ten-second timer. The `Matcher` below
    cannot tell the two templates apart, so serving both would leave these
    tests arming a path they do not describe.
    """

    def get(self, name):
        if name != "max_fever":
            raise KeyError(name)
        return f"{name}-template"


class Matcher:
    """Matches `max_fever` only while `armed` is set."""

    def __init__(self) -> None:
        self.armed = False

    def find(self, frame, template, confidence=None):
        return object() if self.armed else None


def _driver(frame, matcher, said):
    return SimpleNamespace(
        capture=None,
        matcher=matcher,
        templates=Templates(),
        rect=None,
        grab=lambda: frame,
        to_screen=lambda x, y: (int(x), int(y)),
        check_stop=lambda: None,
        say=said.append,
    )


def _options(**over):
    opts = tsum.play_defaults()
    opts.duration = 30.0
    opts.countdown = 0.0
    opts.dry_run = False
    opts.settle = 0.0
    opts.move_time = 0.0
    opts.hold = 0.0
    opts.shuffle_delay = 0.0
    opts.skill = ""
    opts.bubble = ""
    opts.use_base = False
    for key, value in over.items():
        setattr(opts, key, value)
    return opts


#: A capture size the LAYOUTS table has never seen, so the radius has to be
#: estimated and `--radius-lock` has something to do. The real one (994x578)
#: is measured, and `play_loop` takes the measured number outright -- which is
#: its own test at the bottom of this section.
UNMEASURED = (800, 480, 3)

#: The live LDPlayer capture size, which LAYOUTS measures at 25px.
MEASURED = (994, 578, 3)


@pytest.fixture
def rig(monkeypatch):
    """Drive the loop and record the radius `detect` is handed on each frame."""
    matcher = Matcher()
    palette = np.zeros((12, 3), np.float32)
    state = {"asked": [], "returns": [], "counts": []}
    said: list = []

    def _next(script, fallback):
        """Pop the next scripted value, but let the last one stick.

        A frame can call `detect` twice (the refit branch), so a script that
        popped on every call would run off the end mid-test and silently swap
        in the fallback.
        """
        if not script:
            return fallback
        return script.pop(0) if len(script) > 1 else script[0]

    def fake_detect(crop, **kw):
        state["asked"].append(kw.get("radius"))
        r = _next(state["returns"], 25.0)
        n = _next(state["counts"], 40)
        tsums = [tsum.Tsum(x=10.0 * i, y=10.0, r=r, kind=1, colour=(0, 0, 0))
                 for i in range(n)]
        return tsums, r, palette

    monkeypatch.setattr(tsum, "detect", fake_detect)
    monkeypatch.setattr(tsum, "find_chains", lambda *a, **kw: [])
    monkeypatch.setattr(tsum, "_click_shuffle",
                        lambda *a, **kw: state.setdefault("taps", []).append(1))

    def run(opts, frames, arm_at=None, shape=UNMEASURED):
        """Play `frames` frames, turning FEVER on before frame `arm_at`."""
        seen = {"n": 0}
        frame = np.zeros(shape, np.uint8)
        monkeypatch.setattr(tsum, "_settle", lambda drv, max_wait=0.0: frame)

        def stop_when(_frame):
            seen["n"] += 1
            if arm_at is not None and seen["n"] == arm_at:
                matcher.armed = True
            return "done" if seen["n"] > frames else ""

        tsum.play_loop(_driver(frame, matcher, said), opts, stop_when=stop_when)
        return state

    return SimpleNamespace(run=run, state=state, matcher=matcher, said=said)


# --------------------------------------------------------------------------
# the lock
# --------------------------------------------------------------------------
def test_a_collapsed_read_never_gets_a_vote_on_the_lock(rig):
    """The coverage gate screens the warm-up, so the summary can be the median.

    Over the captured round, 44% of three-frame medians land under 20px on a
    board with ~26px faces, against 9% of maxima -- which is why the lock took
    the max before `_face_coverage` existed. The max is a workaround for a
    dirty pool: it survives collapsed samples by ignoring them, but it also
    inherits any over-read. Screening the pool instead is the fix, and once
    the pool is clean the median is the honest summary.

    Here 10px and 12px cover 6-9% of the board rect where 25-26px cover 35-40%,
    so the two collapsed reads never reach the deque and the lock is the
    median of the three that did.
    """
    rig.state["returns"] = [10.0, 26.0, 12.0, 25.0, 25.0, 25.0]
    rig.run(_options(radius_lock=3), frames=6)

    # The first three frames measure freely, each handed what the last returned.
    assert rig.state["asked"][:3] == [None, 10.0, 26.0]

    locked = [line for line in rig.said if "radius locked" in line]
    assert len(locked) == 1, rig.said
    assert "25.0px" in locked[0], "median of the qualifying 26, 25, 25"
    assert "median of 3: 26, 25, 25" in locked[0], "the collapsed reads never voted"
    assert rig.state["asked"][-1] == 25.0, "and every later frame is handed it"


def test_the_radius_survives_the_fever_transition(rig):
    """The bug: FEVER used to reset the radius and re-measure on a fever frame."""
    rig.state["returns"] = [24.0, 24.0, 9.0, 9.0, 9.0, 9.0]
    rig.run(_options(radius_lock=2), frames=6, arm_at=3)

    after_fever = rig.state["asked"][3:]
    assert None not in after_fever, "FEVER must not re-open the radius estimate"
    assert set(after_fever) == {24.0}


def test_with_the_lock_off_the_old_behaviour_is_exactly_preserved(rig):
    """`radius_lock: 0` is the revert, so it has to re-measure as it always did."""
    rig.state["returns"] = [24.0, 24.0, 9.0, 9.0]
    rig.run(_options(radius_lock=0), frames=4, arm_at=3)

    # Frame 1 measures free, frames 2+ carry the previous estimate forward, and
    # the FEVER flip re-opens it -- which is the behaviour being replaced.
    assert rig.state["asked"][0] is None
    assert None in rig.state["asked"][1:], "the flip should still re-measure"


def test_a_measured_layout_skips_the_estimator_altogether(rig):
    """On the real capture size there is nothing to warm up.

    Every tsum is the same size, so for a capture geometry the LAYOUTS table
    has measured, the radius is a known constant rather than something to
    estimate three times and vote on. That short-circuit is what runs in
    practice -- the estimator path above only applies to a window size nobody
    has measured -- and it protects the FEVER bug in the strongest possible
    way: there is no free estimate for the flip to re-open.
    """
    rig.state["returns"] = [9.0] * 6           # a collapsing estimator...
    rig.run(_options(radius_lock=3), frames=6, arm_at=3, shape=MEASURED)

    assert rig.state["asked"] == [25.0] * 6, "...that is never asked, FEVER or not"
    assert any("measured for this layout" in line for line in rig.said)
    assert not any("radius locked" in line for line in rig.said), \
        "nothing to lock -- it was never estimated"


def test_a_refit_cannot_smuggle_a_collapsed_radius_past_the_lock(rig):
    """The recovery path re-fits COLOURS; it must not re-roll the radius."""
    # Frame 3 comes back implausible, which triggers the refit branch.
    rig.state["returns"] = [25.0, 25.0, 25.0, 25.0, 25.0]
    rig.state["counts"] = [40, 40, 2, 2, 40]
    rig.run(_options(radius_lock=2), frames=4)

    assert None not in rig.state["asked"][2:], \
        "the refit must be handed the locked radius, not a free estimate"


# --------------------------------------------------------------------------
# the count floor
# --------------------------------------------------------------------------
@pytest.mark.parametrize("fever_floor, armed, expect_played", [
    (0, True, False),      # floor not lowered: a faded board reads as a menu
    (15, True, True),      # floor lowered and FEVER running: playable
    (15, False, False),    # floor lowered but no FEVER: the strict floor stands
])
def test_the_fever_floor_applies_only_during_fever(rig, monkeypatch,
                                                   fever_floor, armed, expect_played):
    """A 22-tsum frame is a menu in normal play and a faded board in FEVER.

    The floor's real job -- telling a board from a menu -- is already done by
    the FEVER template, which the game does not draw over a menu. So during
    FEVER the frame is worth reading, and outside it the strict floor stands.
    """
    chain = tsum.Chain(kind=1, colour=(0, 0, 0), nodes=[0, 1, 2])
    monkeypatch.setattr(tsum, "find_chains", lambda *a, **kw: [chain])

    rig.state["counts"] = [22] * 6
    opts = _options(radius_lock=0, min_tsums=30, fever_min_tsums=fever_floor,
                    max_misses=99, dry_run=True)
    rig.run(opts, frames=4, arm_at=1 if armed else None)

    # dry_run stops the loop the moment a chain is playable, so "did the board
    # count as playable" is exactly "did it reach the drag".
    played = any("chain of" in line for line in rig.said)
    assert played is expect_played


def test_the_fever_floor_defaults_to_off(rig):
    """Nothing changes for anyone who does not set it."""
    defaults = vars(tsum.play_defaults())
    assert defaults["fever_min_tsums"] == 0
    assert defaults["radius_lock"] == 0
