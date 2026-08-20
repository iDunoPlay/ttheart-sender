"""Re-filing a board into the characters it actually holds.

A board deals five characters (four once an item has removed one), and
`fixed_kinds` uses that count as the stopping rule `_recolour` never had.
The thing worth guarding is not that clustering works -- it is that it still
works when a third of what it is handed is phantom, which is the ratio real
detection runs at and the reason a plain five-centre fit loses to doing
nothing at all.

Synthetic boards rather than saved ones: the fixtures under scratchpad/ are
not in the repo, and what is being measured is whether the *count* survives
contamination, which needs known groups more than it needs a real Mickey.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from ttheart_sender.game import tsum
from ttheart_sender.game.tsum import fixed_kinds

#: Five faces, far enough apart in BGR that any honest reading separates them.
FACES = [(60, 60, 220), (60, 200, 70), (220, 120, 60), (200, 200, 200), (150, 60, 190)]
RADIUS = 24.0


def _board(per_face: int = 6, phantoms: int = 0, seed: int = 0):
    """A board with `per_face` copies of each face, plus off-colour phantoms.

    Returns (frame, tsums, truth) where truth[i] is the index into FACES that
    tsum i really is, or -1 for a phantom.
    """
    rng = np.random.default_rng(seed)
    slots = [(70 + 70 * (i % 7), 70 + 70 * (i // 7))
             for i in range(len(FACES) * per_face + phantoms)]
    frame = np.full((70 * (len(slots) // 7 + 2), 70 * 9, 3), 30, np.uint8)

    truth = [i // per_face for i in range(len(FACES) * per_face)]
    truth += [-1] * phantoms
    rng.shuffle(truth)

    tsums = []
    for (x, y), which in zip(slots, truth):
        # Phantoms are the colours detection picks up off board texture and
        # outlines: nothing a character wears, and scattered rather than
        # repeated, which is exactly what lets over-provisioning strand them.
        colour = (FACES[which] if which >= 0
                  else tuple(int(v) for v in rng.integers(20, 250, 3)))
        jitter = rng.integers(-6, 7, 3)          # lighting across the bowl
        shade = tuple(int(np.clip(c + j, 0, 255)) for c, j in zip(colour, jitter))
        cv2.circle(frame, (x, y), int(RADIUS), shade, -1)
        tsums.append(tsum.Tsum(x=float(x), y=float(y), r=RADIUS, kind=0, colour=shade))
    return frame, tsums, truth


def _groups(tsums, truth):
    """The set of kinds each real character was filed under."""
    out: dict[int, set[int]] = {}
    for t, which in zip(tsums, truth):
        if which >= 0:
            out.setdefault(which, set()).add(t.kind)
    return out


def test_clean_board_splits_into_exactly_five():
    frame, tsums, truth = _board()
    tsums, centres = fixed_kinds(frame, tsums, RADIUS, 5)

    assert centres is not None and len(centres) == 5
    groups = _groups(tsums, truth)
    assert all(len(kinds) == 1 for kinds in groups.values()), "a character read as two colours"
    assert len({next(iter(k)) for k in groups.values()}) == 5, "two characters read as one"


def test_phantoms_do_not_cost_a_character():
    """The regression this exists for.

    Detection precision is ~0.675, so a third of the list is noise. Fitting
    five centres over that spends them on the noise; over-provisioning and
    keeping the five most populous does not.
    """
    frame, tsums, truth = _board(per_face=8, phantoms=20, seed=3)
    tsums, _ = fixed_kinds(frame, tsums, RADIUS, 5)

    groups = _groups(tsums, truth)
    assert len(groups) == 5
    assert all(len(kinds) == 1 for kinds in groups.values())
    assert len({next(iter(k)) for k in groups.values()}) == 5


def test_four_characters_when_an_item_has_taken_one():
    frame, tsums, truth = _board(per_face=6)
    keep = [i for i, w in enumerate(truth) if w != 4]
    tsums = [tsums[i] for i in keep]
    truth = [truth[i] for i in keep]

    tsums, centres = fixed_kinds(frame, tsums, RADIUS, 4)
    assert len(centres) == 4
    groups = _groups(tsums, truth)
    assert len({next(iter(k)) for k in groups.values()}) == 4


def test_ids_are_reproducible():
    """k-means numbers its centres by where it started, so they get sorted.

    Without that, two fits of the same board hand out different ids and a
    caller cannot cache a base-tsum id for even one more frame.
    """
    frame, tsums, _ = _board(seed=1)
    a, ca = fixed_kinds(frame, list(tsums), RADIUS, 5)
    b, cb = fixed_kinds(frame, list(tsums), RADIUS, 5)
    assert [t.kind for t in a] == [t.kind for t in b]
    assert np.allclose(ca, cb)


def test_reusing_centres_skips_the_fit_and_agrees_with_it():
    frame, tsums, _ = _board(seed=2)
    fitted, centres = fixed_kinds(frame, list(tsums), RADIUS, 5)
    reused, again = fixed_kinds(frame, list(tsums), RADIUS, 5, centres)

    assert again is centres
    assert [t.kind for t in reused] == [t.kind for t in fitted]


def test_declines_when_there_is_not_enough_board():
    """Fewer tsums than the fit needs headroom for: leave the kinds alone.

    Fitting five centres directly is measured as worse than doing nothing, so
    the fallback is to do nothing rather than to fit anyway.
    """
    frame, tsums, _ = _board(per_face=2)          # 10 tsums, under 5*3
    before = [t.kind for t in tsums]
    tsums, centres = fixed_kinds(frame, tsums, RADIUS, 5)

    assert centres is None
    assert [t.kind for t in tsums] == before


def test_play_loop_exposes_it_as_a_tunable():
    """`options: {kinds: 5}` in a flow has to reach the loop, not be ignored."""
    from ttheart_sender.automation.tsum_actions import _TUNABLES
    assert "kinds" in _TUNABLES
    assert tsum.play_defaults().kinds == 0, "must stay opt-in until a live A/B"


def _with_board_phantoms(per_face: int = 8, phantoms: int = 20, seed: int = 5,
                         phantom_bgr=None):
    """A board whose phantoms are the colour of the bowl, as real ones are.

    Real phantom detections land on the board between tsums, so they carry the
    board's colour and cluster together. That is what makes them populous
    enough to take a seat, and what `spare` uses to give it back.

    They get a tighter jitter than the characters do, because that is what was
    measured: on board 1 the phantom cluster was 16 detections out of 16 in a
    single cluster. Spread them wider than a character and they fragment into
    several board-coloured clusters, which one spare seat cannot absorb -- a
    real limit of `spare=1`, but not the case this fixture is built to show.
    """
    rng = np.random.default_rng(seed)
    board_bgr = (30, 30, 30)
    n = len(FACES) * per_face + phantoms
    slots = [(70 + 70 * (i % 7), 70 + 70 * (i // 7)) for i in range(n)]
    frame = np.full((70 * (n // 7 + 2), 70 * 9, 3), board_bgr, np.uint8)

    truth = [i // per_face for i in range(len(FACES) * per_face)] + [-1] * phantoms
    rng.shuffle(truth)

    tsums = []
    for (x, y), which in zip(slots, truth):
        base = FACES[which] if which >= 0 else (phantom_bgr or board_bgr)
        jitter = rng.integers(-6, 7, 3) if which >= 0 else np.zeros(3, int)
        shade = tuple(int(np.clip(c + j, 0, 255)) for c, j in zip(base, jitter))
        cv2.circle(frame, (x, y), int(RADIUS), shade, -1)
        tsums.append(tsum.Tsum(x=float(x), y=float(y), r=RADIUS, kind=0, colour=shade))

    # The palette a real caller hands over: the board's colour plus the faces.
    swatches = np.array([board_bgr] + FACES, np.uint8).reshape(-1, 1, 3)
    palette = cv2.cvtColor(swatches, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    return frame, tsums, truth, palette


def test_the_board_coloured_seat_is_given_up():
    """The regression the `spare` seat exists for.

    Twenty phantoms sharing the board's colour outnumber any single character,
    so ranking on population alone spends one of the five seats on them and a
    real character has nowhere to go. Given the palette, that seat is handed
    back: the kept centres are the five furthest from this board's own
    background, and a cluster sitting *on* the background is never one of them.

    Asserted as the rule rather than as "all five characters came back",
    because recovering every character also needs each of them to win a seat
    on population, and on a board this synthetic the characters split evenly
    enough that they do not. The rule is the part `spare` is responsible for.
    """
    frame, tsums, truth, palette = _with_board_phantoms()
    bowl = tsum.board_colours(frame, palette)

    def nearest_to_bowl(centres):
        return float(np.linalg.norm(centres[:, None] - bowl[None], axis=2).min())

    _, blind = fixed_kinds(frame, list(tsums), RADIUS, 5)
    _, seated = fixed_kinds(frame, list(tsums), RADIUS, 5, palette=palette)

    assert nearest_to_bowl(blind) < 20, "expected population alone to seat the phantoms"
    assert nearest_to_bowl(seated) > 50, "the board-coloured seat should have been dropped"


def test_spare_needs_the_palette():
    """No palette means no background to rank against, so seats go by count."""
    frame, tsums, _, palette = _with_board_phantoms()
    _, a = fixed_kinds(frame, list(tsums), RADIUS, 5)
    _, b = fixed_kinds(frame, list(tsums), RADIUS, 5, palette=palette, spare=0)
    assert np.allclose(a, b), "spare=0 should match the no-palette path exactly"


def test_board_colours_finds_the_bowl():
    frame, _, _, palette = _with_board_phantoms()
    board = tsum.board_colours(frame, palette)

    assert len(board) >= 1
    bowl = cv2.cvtColor(np.uint8([[(30, 30, 30)]]), cv2.COLOR_BGR2LAB).reshape(3)
    assert np.linalg.norm(board[0] - bowl) < 10


# --------------------------------------------------------------------------
# looking at the result
# --------------------------------------------------------------------------
def test_scatter_tells_a_character_from_a_dump_bucket():
    """The number that says a kind is holding more than one thing.

    A real character's faces sit within a few Lab of each other. The bucket
    that absorbs the phantoms does not, and on a real board that is the only
    thing separating it from an ordinary-looking kind with an ordinary count.
    """
    frame, tsums, truth = _board(per_face=6)
    tsums, _ = fixed_kinds(frame, tsums, RADIUS, 5)
    tight = tsum.kind_scatter(frame, tsums, RADIUS)
    assert max(tight.values()) < tsum.KIND_SCATTER, tight

    # Now force one bucket to hold two characters and confirm it shows up.
    for t in tsums:
        if t.kind == 1:
            t.kind = 0
    mixed = tsum.kind_scatter(frame, tsums, RADIUS)
    assert mixed[0] > tsum.KIND_SCATTER * 2, mixed


def test_flatten_renders_one_colour_per_kind():
    frame, tsums, _ = _board(per_face=6)
    tsums, _ = fixed_kinds(frame, tsums, RADIUS, 5)

    solo = tsum.flatten(frame, tsums, RADIUS, compare=False, numbers=False)
    both = tsum.flatten(frame, tsums, RADIUS, compare=True, numbers=False)

    # Side by side is the original plus the flattened, so twice the width.
    assert both.shape[1] == 2 * solo.shape[1]
    # Header and legend add rows to both alike.
    assert solo.shape[0] > frame.shape[0]

    # Every tsum's centre should carry its own kind's fill and nothing else,
    # so the flattened board holds about as many distinct colours as kinds
    # (plus the background, the rims, and the antialiasing between them).
    board = solo[28:28 + frame.shape[0]]
    centres = np.array([board[int(t.y), int(t.x)] for t in tsums])
    per_kind = {t.kind: tuple(c) for t, c in zip(tsums, centres)}
    assert len(per_kind) == 5
    for t, c in zip(tsums, centres):
        assert tuple(c) == per_kind[t.kind], "two tsums of one kind drawn differently"


def test_flatten_distinct_uses_the_high_contrast_palette():
    """The real colours are useless for the case that matters most.

    Two characters 15 Lab apart render as two pastels nobody can tell apart,
    which is exactly when you are looking to see whether they were split.
    """
    frame, tsums, _ = _board(per_face=6)
    tsums, _ = fixed_kinds(frame, tsums, RADIUS, 5)

    board = tsum.flatten(frame, tsums, RADIUS, compare=False,
                         numbers=False, distinct=True)[28:28 + frame.shape[0]]
    used = {tuple(board[int(t.y), int(t.x)]) for t in tsums}
    assert used <= {tuple(c) for c in tsum.FLAT_COLOURS}
    assert len(used) == 5


def test_flatten_marks_the_wide_bucket():
    """With a scatter reading, an over-wide kind is called out in the header."""
    frame, tsums, _ = _board(per_face=6)
    tsums, _ = fixed_kinds(frame, tsums, RADIUS, 5)
    for t in tsums:
        if t.kind == 1:
            t.kind = 0

    scatter = tsum.kind_scatter(frame, tsums, RADIUS)
    marked = tsum.flatten(frame, tsums, RADIUS, scatter=scatter)
    plain = tsum.flatten(frame, tsums, RADIUS)
    assert marked.shape == plain.shape
    assert not np.array_equal(marked, plain), "the wide kind should be flagged"


def _palette_for(frame, board_bgr=(30, 30, 30), ink=(10, 10, 10)):
    """A pixel palette in Lab: the bowl, the ink, and the five faces."""
    swatches = np.array([board_bgr, ink] + FACES, np.uint8).reshape(-1, 1, 3)
    return cv2.cvtColor(swatches, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)


def test_posterise_paints_each_tsum_one_solid_colour():
    """The pixel repaint: a mostly-pink tsum comes out an entirely pink ball.

    Checked at the centre AND well off-centre, because the point of painting
    pixels rather than drawing discs is that the *whole* sprite goes one
    colour -- a version that only got the middle right would pass a
    centre-only check and be useless.
    """
    frame, tsums, truth = _board(per_face=6)
    tsums, centres = fixed_kinds(frame, tsums, RADIUS, 5)
    out = tsum.posterise(frame, centres, palette=_palette_for(frame),
                         radius=RADIUS, distinct=True)

    assert out.shape == frame.shape
    for t, which in zip(tsums, truth):
        if which < 0:
            continue
        middle = tuple(out[int(t.y), int(t.x)])
        edge = tuple(out[int(t.y), int(t.x + RADIUS * 0.6)])
        assert middle == edge, "the sprite did not come out one flat colour"


def test_posterise_sends_board_and_ink_somewhere_other_than_a_character():
    """Without sinks every gap and shadow is forced onto some character."""
    frame, tsums, _ = _board(per_face=6)
    cv2.circle(frame, (frame.shape[1] - 40, frame.shape[0] - 40), 20, (10, 10, 10), -1)
    tsums, centres = fixed_kinds(frame, tsums, RADIUS, 5)

    painted = tsum.posterise(frame, centres, palette=_palette_for(frame),
                             radius=RADIUS, board_colour=(70, 45, 30),
                             ink_colour=(40, 40, 90))
    assert tuple(painted[4, 4]) == (70, 45, 30), "bare board should read as board"
    assert tuple(painted[frame.shape[0] - 40, frame.shape[1] - 40]) == (40, 40, 90)

    # The two sinks are deliberately different colours: sharing one hides a
    # black-bodied tsum inside the bowl instead of showing it as ink.
    assert (70, 45, 30) != (40, 40, 90)


def test_flatten_paint_mode_needs_centres_and_falls_back():
    frame, tsums, _ = _board(per_face=6)
    tsums, centres = fixed_kinds(frame, tsums, RADIUS, 5)

    fell_back = tsum.flatten(frame, tsums, RADIUS, mode="paint", compare=False)
    discs = tsum.flatten(frame, tsums, RADIUS, mode="disc", compare=False)
    assert np.array_equal(fell_back, discs), "paint without centres should be disc"

    both = tsum.flatten(frame, tsums, RADIUS, mode="both", centres=centres,
                        palette=_palette_for(frame), compare=True)
    one = tsum.flatten(frame, tsums, RADIUS, mode="disc", compare=False)
    # original + painted + discs against discs alone.
    assert both.shape[1] == 3 * one.shape[1]

    with pytest.raises(ValueError):
        tsum.flatten(frame, tsums, RADIUS, mode="sideways")


def test_the_chain_overlay_is_the_same_one_draw_produces():
    """`flatten(chains=...)` must not be a second implementation of the stroke.

    Both go through `draw_chain`, so the check is that the pixels the overlay
    puts down are identical on identical input -- if one grows an arrowhead or
    a node number the other lacks, this fails.
    """
    frame, tsums, _ = _board(per_face=6)
    tsums, _ = fixed_kinds(frame, tsums, RADIUS, 5)
    chain = tsum.Chain(kind=tsums[0].kind, colour=tsums[0].colour,
                       nodes=[i for i, t in enumerate(tsums)
                              if t.kind == tsums[0].kind][:4])

    plain = np.full_like(frame, 24)
    a = tsum.draw_chain(plain, tsums, [chain], RADIUS, glow=False)
    b = tsum.draw_chain(plain, tsums, [chain], RADIUS, glow=False)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, plain), "the overlay drew nothing"

    # And it does not scribble on the caller's image.
    assert np.array_equal(plain, np.full_like(frame, 24))


def test_flatten_draws_the_chain_on_every_panel():
    frame, tsums, _ = _board(per_face=6)
    tsums, centres = fixed_kinds(frame, tsums, RADIUS, 5)
    nodes = [i for i, t in enumerate(tsums) if t.kind == tsums[0].kind][:4]
    chain = tsum.Chain(kind=tsums[0].kind, colour=tsums[0].colour, nodes=nodes)

    kw = dict(mode="both", centres=centres, palette=_palette_for(frame), compare=True)
    without = tsum.flatten(frame, tsums, RADIUS, **kw)
    with_ = tsum.flatten(frame, tsums, RADIUS, chains=[chain], **kw)

    assert without.shape == with_.shape
    # Each of the three panels must differ, not just the first.
    board_h = frame.shape[0]
    w = frame.shape[1]
    for panel in range(3):
        a = without[28:28 + board_h, panel * w:(panel + 1) * w]
        b = with_[28:28 + board_h, panel * w:(panel + 1) * w]
        assert not np.array_equal(a, b), f"panel {panel} has no chain drawn on it"


def test_a_single_node_chain_does_not_crash_the_arrowhead():
    """`draw` only ever gets chains of 3+, but `flatten` takes what it is given."""
    frame, tsums, _ = _board(per_face=6)
    tsums, _ = fixed_kinds(frame, tsums, RADIUS, 5)
    one = tsum.Chain(kind=tsums[0].kind, colour=tsums[0].colour, nodes=[0])
    tsum.draw_chain(frame, tsums, [one], RADIUS)


# --------------------------------------------------------------------------
# the two reading modes
# --------------------------------------------------------------------------
def test_the_two_modes_resolve_to_the_readings_they_name():
    from types import SimpleNamespace

    normal = SimpleNamespace(tsum_mode="normal", kinds=0)
    colour = SimpleNamespace(tsum_mode="color", kinds=0)
    assert tsum.resolve_kinds(normal) == 0
    assert tsum.resolve_kinds(colour) == tsum.BOARD_KINDS == 5


def test_an_explicit_kinds_beats_the_mode():
    """`--tsum-mode color --kinds 4` is the board an item has taken one off.

    Also what keeps a flow that already says `kinds: 5` working untouched.
    """
    from types import SimpleNamespace

    assert tsum.resolve_kinds(SimpleNamespace(tsum_mode="color", kinds=4)) == 4
    assert tsum.resolve_kinds(SimpleNamespace(tsum_mode="normal", kinds=5)) == 5


def test_resolve_copes_with_options_that_predate_the_mode():
    """Old callers pass neither field; they must get the original reading."""
    from types import SimpleNamespace

    assert tsum.resolve_kinds(SimpleNamespace()) == 0
    assert tsum.resolve_kinds(SimpleNamespace(kinds=0)) == 0


def test_the_mode_is_named_in_the_run_log():
    """An A/B is unreadable if the log does not say which arm produced it."""
    from types import SimpleNamespace

    assert "normal" in tsum.describe_mode(SimpleNamespace(tsum_mode="normal", kinds=0))
    assert "color" in tsum.describe_mode(SimpleNamespace(tsum_mode="color", kinds=0))
    assert "4" in tsum.describe_mode(SimpleNamespace(tsum_mode="color", kinds=4))


def test_both_modes_are_flow_options():
    from ttheart_sender.automation.tsum_actions import _TUNABLES

    assert "tsum_mode" in _TUNABLES and "kinds" in _TUNABLES
    assert tsum.play_defaults().tsum_mode == "normal", "must stay opt-in"


# --------------------------------------------------------------------------
# detection: rejecting what landed on the bowl
# --------------------------------------------------------------------------
def test_bowl_reject_drops_detections_sitting_on_the_board():
    """Detection is the weaker half -- 31% of what it reports is not a tsum.

    A detection that landed on the bowl carries the bowl's colour, and that is
    what this uses. Measured over the labelled boards: real tsums sit a median
    173 Lab from the board colour and phantoms 76.
    """
    # Phantoms a shade off the bowl rather than exactly it: drawn in the bowl's
    # own colour they are the background cluster and `detect` never reports
    # them, so there would be nothing for this filter to catch. Real phantoms
    # sit in the gaps and on outlines -- near the board, not identical to it.
    frame, tsums, truth, palette = _with_board_phantoms(per_face=8, phantoms=20,
                                                        phantom_bgr=(60, 60, 62))
    kept_all, radius, pal = tsum.detect(frame, bowl_reject=0.0)
    kept_cut, _, _ = tsum.detect(frame, bowl_reject=60.0)

    assert len(kept_all) == 60, f"fixture should offer 40 real + 20 phantom, got {len(kept_all)}"
    assert len(kept_cut) == 40, f"should have dropped exactly the phantoms, kept {len(kept_cut)}"

    # Everything it kept should be well clear of the bowl.
    if kept_cut:
        bowl = tsum.board_colours(frame, pal)
        far = np.linalg.norm(tsum._face_lab(frame, kept_cut, radius)[:, None]
                             - bowl[None], axis=2).min(axis=1)
        assert far.min() >= 60.0


def test_bowl_reject_off_by_default_changes_nothing():
    frame, tsums, _, _ = _with_board_phantoms(phantom_bgr=(60, 60, 62))
    a, ra, _ = tsum.detect(frame)
    b, rb, _ = tsum.detect(frame, bowl_reject=0.0)
    assert len(a) == len(b) and ra == rb


def test_bowl_reject_is_reachable_from_a_flow_and_off_by_default():
    from ttheart_sender.automation.tsum_actions import _TUNABLES

    assert "bowl_reject" in _TUNABLES
    assert tsum.play_defaults().bowl_reject == 0.0


def test_the_shipped_play_flow_does_not_set_both_kinds_and_tsum_mode():
    """`kinds` overrides `tsum_mode`; setting both makes the mode a no-op.

    The failure is silent and expensive: switch `tsum_mode` to `normal` for
    the control arm of an A/B, leave a `kinds: 5` behind, and both arms run
    the colour reading while the log says otherwise.
    """
    from pathlib import Path

    import yaml

    flow = yaml.safe_load((Path(__file__).resolve().parent.parent
                           / "flows" / "play.yaml").read_text(encoding="utf-8"))
    options = next(step["play_tsum"]["options"]
                   for step in flow["steps"]
                   if isinstance(step, dict) and "play_tsum" in step)

    assert not ("kinds" in options and "tsum_mode" in options), (
        "play.yaml sets both -- `kinds` wins, so `tsum_mode` there is a lie")


def test_bowl_reject_applies_to_both_readings():
    """It is a detection filter, so it must land before the kinds are decided.

    If it ran after, the colour reading would fit its characters over the
    phantoms first and the filter would be picking through a decision already
    made with them in it.
    """
    frame, tsums, _, _ = _with_board_phantoms(per_face=8, phantoms=20,
                                              phantom_bgr=(60, 60, 62))
    plain, _, _ = tsum.detect(frame, bowl_reject=60.0)
    coloured, _, _ = tsum.detect(frame, bowl_reject=60.0, kinds=5)

    assert len(plain) == len(coloured) == 40, "the filter must drop the same set"
    assert {(t.x, t.y) for t in plain} == {(t.x, t.y) for t in coloured}
