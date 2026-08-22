"""Fitting a palette from collected samples, and the score that justifies it.

The collector's tests guard the moment a label is captured. These guard the
other end: that a palette is only ever recommended on evidence, that the
evidence is measured on rounds the fit never saw, and that turning it on
cannot silently fall back to the behaviour it was meant to replace.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from ttheart_sender.game import learn, tsum


#: Six well-separated BGR faces on a flat bowl. Deliberately easy: these tests
#: are about the machinery around the fit, and a corpus whose answer is in
#: doubt could not tell a broken split from a hard board.
CHARS = [(60, 60, 220), (70, 190, 240), (200, 150, 70),
         (80, 200, 90), (190, 90, 200), (240, 230, 230)]
BOWL = (120, 90, 60)


def _session(folder: Path, n: int = 6, *, kind_noise: float = 1.0, seed: int = 0):
    """Write one session folder of fabricated but well-formed samples.

    `kind_noise` is what the recorded `kind` looks like: 1.0 reshuffles it
    every frame, which is per-frame clustering at its worst, and 0.0 records
    the true character, which is per-frame clustering being already right.
    """
    import random

    rng = random.Random(seed)
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "samples.jsonl").open("w", encoding="utf-8") as fh:
        for idx in range(1, n + 1):
            img = np.full((450, 450, 3), BOWL, np.uint8)
            tsums, truth = [], []
            for gy in range(7):
                for gx in range(7):
                    c = rng.randrange(len(CHARS))
                    x, y = 30 + gx * 60, 30 + gy * 60
                    cv2.circle(img, (x, y), 25, CHARS[c], -1)
                    tsums.append({"x": float(x), "y": float(y), "r": 25.0})
                    truth.append(c)
            # The game marks same-character AND reachable, so a same-character
            # tsum on the far side stays dark. Reproduced here because it is
            # the reason an unmarked tsum is only a weak negative, and a
            # fixture that marked every match would quietly delete that.
            #
            # The reach has to clear `learn.AURA` or the fixture marks nothing
            # that counts: measured on a real board the game marks partners at
            # 109px, 131px and 167px, well outside the 90px glow, and it is
            # exactly those that are evidence. A 60px grid puts the second ring
            # out at 120px.
            head = 24            # the middle of a 7x7 grid, at (210, 210)
            hx, hy = tsums[head]["x"], tsums[head]["y"]
            marked = [i for i, t in enumerate(tsums)
                      if i != head and truth[i] == truth[head]
                      and abs(t["x"] - hx) <= 150 and abs(t["y"] - hy) <= 150]
            for i, t in enumerate(tsums):
                t["kind"] = rng.randrange(12) if rng.random() < kind_noise else truth[i]
            cv2.imwrite(str(folder / f"{idx:04d}_before.jpg"), img)
            fh.write(json.dumps({
                "schema": 2, "index": idx, "board": [0, 0, 450, 450],
                "radius": 25.0, "fever": False, "tsums": tsums, "head": head,
                "proposed": [head], "kept": [head], "marked": marked,
                "baseline": 1.0, "bar": 8.0, "marks": [], "options": {},
            }) + "\n")


def _corpus(root: Path, sessions: int = 3, **kw) -> Path:
    for s in range(sessions):
        _session(root / f"2026010{s}_000000_{s}", seed=s, **kw)
    return root


# -- the artifact ----------------------------------------------------------
def test_palette_survives_a_round_trip(tmp_path):
    # The centres are the whole file. Two decimal places is a deliberate
    # rounding -- 1.0 is already near the limit of what a Lab difference
    # means -- so this pins that it is a rounding and not a truncation.
    centres = np.array([[10.5, 20.25, 30.75], [200.0, 128.0, 128.0]], np.float32)
    p = learn.Palette(centres=centres, face_counts={1: 7}, metrics={"agreement": 0.9})
    p.save(tmp_path / "p.json")
    back = learn.Palette.load(tmp_path / "p.json")

    assert np.abs(back.centres - centres).max() < 0.01
    assert back.face_counts == {1: 7}
    assert back.k == 2


def test_a_palette_from_another_schema_is_refused(tmp_path):
    # An id is only worth anything because it means one fixed thing. Centres
    # fitted under different rules would still load and still index -- and
    # would quietly mean something else, which is the one failure this whole
    # feature cannot tolerate.
    (tmp_path / "p.json").write_text(json.dumps(
        {"schema": learn.SCHEMA + 1, "centres": [[1, 2, 3]]}), encoding="utf-8")
    with pytest.raises(ValueError, match="schema"):
        learn.Palette.load(tmp_path / "p.json")


# -- the corpus ------------------------------------------------------------
def test_rows_that_are_not_boards_are_left_out():
    # A menu scores a handful of phantom detections. Its colours are menu
    # furniture and must not end up in a palette used to read a board.
    assert not learn.usable({"tsums": [{"x": 1, "y": 1, "r": 9, "kind": 0}], "head": 0})
    assert learn.usable({"tsums": [{"x": 1, "y": 1, "r": 9, "kind": 0}] * 20, "head": 3})
    # A head index pointing past the end is a corrupt row, not a hard one.
    assert not learn.usable({"tsums": [{"x": 1, "y": 1, "r": 9, "kind": 0}] * 20, "head": 99})


def test_a_truncated_last_line_does_not_lose_the_session(tmp_path):
    # What a killed round leaves behind. Losing the other 24 samples over it
    # would make stopping a round expensive.
    _session(tmp_path / "s1", n=3)
    with (tmp_path / "s1" / "samples.jsonl").open("a", encoding="utf-8") as fh:
        fh.write('{"schema": 2, "index": 4, "tsu')
    assert len(list(learn.iter_rows(tmp_path))) == 3


def test_the_holdout_splits_whole_sessions(tmp_path):
    # Samples inside one session are the same board minutes apart under the
    # same equipped tsum. Splitting per-sample would put near-duplicates on
    # both sides and score the fit against frames it effectively saw.
    rows = list(learn.iter_rows(_corpus(tmp_path, sessions=4)))
    train, test = learn.split_rows(rows, 0.25, seed=0)

    assert test, "a four-session corpus must hold something out"
    assert not ({f for f, _ in train} & {f for f, _ in test})
    assert len(train) + len(test) == len(rows)


def test_one_session_cannot_be_held_out_from_itself(tmp_path):
    # Reported rather than faked: `build` marks the metrics `held_out: False`
    # and the CLI refuses to recommend on them.
    rows = list(learn.iter_rows(_corpus(tmp_path, sessions=1)))
    train, test = learn.split_rows(rows, 0.25, seed=0)
    assert test == []
    assert len(train) == len(rows)


# -- the score -------------------------------------------------------------
def test_the_baseline_is_what_ran_live_not_a_re_derivation(tmp_path):
    # The recorded `kind` is per-frame k-means' own answer on that frame. It
    # is the number to beat, and reading it from the row rather than recomputing
    # it is what makes the comparison a like-for-like on identical frames.
    root = _corpus(tmp_path, sessions=2, kind_noise=0.0)
    rows = list(learn.iter_rows(root))
    m = learn.agreement(rows, np.zeros((1, 3), np.float32))

    # Every centre collapsed to one, so the palette agrees with everything...
    assert m["agreement"] == 1.0
    # ...and the baseline is untouched by that, because it came off the rows.
    assert m["baseline"] == 1.0
    # The collapse is visible in the one place it can be: nothing is split.
    assert m["split"] == 0.0


def test_a_real_fit_beats_a_reshuffled_per_frame_kind(tmp_path):
    p = learn.build(_corpus(tmp_path, sessions=4, kind_noise=1.0), k=8, seed=0)

    assert p.metrics["held_out"] is True
    assert p.metrics["agreement"] > 0.9
    assert p.metrics["agreement"] > p.metrics["baseline"] + 0.2
    # Six characters were drawn and the marks should have landed on centres
    # that are actually distinct.
    assert len(p.faces()) >= 4


def test_agreement_alone_pays_a_palette_to_merge_characters(tmp_path):
    """Why the verdict is `balanced` and not `agreement`.

    Fewer clusters means more tsums share an id, which lifts agreement on
    confirmed same-character pairs *and* costs separation on the negatives.
    Reading only the first number therefore rewards exactly the failure the
    second one exists to catch -- and it arrives gradually, so no threshold on
    `split` sees it coming. Measured on the first real corpus: k=6 scored
    37.6%/72.4% against k=24's 26.6%/86.6%, and the k=6 fit was 1.5 points
    WORSE than per-frame clustering once both halves were counted.
    """
    rows = list(learn.iter_rows(_corpus(tmp_path, sessions=3, n=12)))
    merged = learn.agreement(rows, learn.fit(rows, k=2, seed=0))
    split_up = learn.agreement(rows, learn.fit(rows, k=8, seed=0))

    # `>=` not `>`: on a corpus this separable k=8 already agrees on every
    # pair, so agreement has no room to rise. The cost still shows up.
    assert merged["agreement"] >= split_up["agreement"]
    assert merged["split"] < split_up["split"], "and it must cost separation"
    # The verdict prices both, so it cannot be gamed by turning `-k` down.
    assert merged["balanced"] == pytest.approx(
        (merged["agreement"] + merged["split"]) / 2, abs=1e-3)


def test_the_baseline_is_scored_on_both_halves_too(tmp_path):
    # The comparison is only like-for-like if the per-frame `kind` is put
    # through the identical negatives. Scoring the palette on two numbers and
    # the baseline on one is how a collapse gets waved through.
    rows = list(learn.iter_rows(_corpus(tmp_path, sessions=2, kind_noise=0.0)))
    m = learn.agreement(rows, learn.fit(rows, k=8, seed=0))

    assert m["baseline"] == 1.0, "a truthful recorded kind agrees on every pair"
    assert m["baseline_split"] > 0.0
    assert m["baseline_balanced"] == pytest.approx(
        (m["baseline"] + m["baseline_split"]) / 2, abs=1e-3)


def test_marks_inside_the_glow_are_not_counted_as_evidence(tmp_path):
    # The glow washes over whatever is under it, so a tsum in there clears the
    # bar for being near the press rather than for being the same character.
    # `tsum dataset` has always excluded them; this pins that the palette is
    # scored on the same population.
    _session(tmp_path / "s0", n=4, seed=0)
    rows = list(learn.iter_rows(tmp_path))
    row = rows[0][1]
    ts, head = row["tsums"], int(row["head"])
    far = learn._outside_glow(ts, head)

    assert not far[head], "the pressed tsum is inside its own glow"
    assert far.any() and not far.all(), "the fixture must straddle the aura"
    marked = [i for i in row["marked"] if far[i]]
    assert marked and len(marked) < len(row["marked"]), (
        "the fixture must mark partners both inside and outside the glow, or "
        "this test cannot tell the filter is applied")

    m = learn.agreement(rows, learn.fit(rows, k=8, seed=0))
    assert m["pairs"] <= sum(len(r["marked"]) for _, r in rows)


def test_unmarked_tsums_are_only_weak_negatives(tmp_path):
    # The game marks same-character AND reachable, so a same-character tsum
    # out of reach is dark. `split` must therefore never be read as accuracy --
    # this pins that a correct palette does NOT score 1.0 on it.
    # Scored over the whole corpus rather than the held-out quarter: `split`
    # is a rate over a handful of negatives per sample, so a six-sample
    # holdout can miss the confound by luck and this test would then be
    # asserting on a coin toss.
    root = _corpus(tmp_path, sessions=4, n=25)
    rows = list(learn.iter_rows(root))
    centres = learn.fit(rows, k=8, seed=0)
    m = learn.agreement(rows, centres)

    assert m["agreement"] > 0.9, "the palette should be right about identity"
    assert 0.4 < m["split"] < 1.0, (
        "some unmarked tsums are the same character out of reach, so a "
        "correct palette must not score 1.0 here")


# -- the consumption seam --------------------------------------------------
def test_no_palette_means_exactly_the_old_behaviour():
    assert tsum._load_palette("", lambda *_: None) is None
    assert tsum.play_defaults().palette == ""


def test_a_palette_that_will_not_load_stops_the_round(tmp_path):
    # Falling back to the per-frame fit would produce a round that looks
    # exactly like one where the palette did not help -- which is the reading
    # that gets a working palette thrown away.
    with pytest.raises((OSError, ValueError)):
        tsum._load_palette(str(tmp_path / "nope.json"), lambda *_: None)


def test_learned_centres_are_what_detect_reads_the_board_through(tmp_path):
    # The seam already existed: `detect(palette=...)` has always accepted
    # centres from an earlier frame. Passing learned ones changes no detection
    # code -- this pins that they are used verbatim rather than refitted.
    _session(tmp_path / "s0", n=4, seed=0)
    p = learn.build(tmp_path, k=8, seed=0, holdout=0)
    p.save(tmp_path / "p.json")

    said = []
    centres = tsum._load_palette(str(tmp_path / "p.json"), said.append)
    assert np.abs(centres - p.centres).max() < 0.01
    assert said and "palette:" in said[0]

    crop = learn.crop_of(tmp_path / "s0", {"index": 1})
    _, _, used = tsum.detect(crop, k=p.k, radius=25.0, palette=centres)
    assert np.array_equal(used, centres)


def test_the_flow_can_turn_it_on_in_one_line():
    # `options:` in play.yaml is validated against the play loop's own option
    # set, so this is what makes `palette: models/palette.json` a legal step
    # rather than a rejected typo.
    from ttheart_sender.automation.tsum_actions import _TUNABLES
    assert "palette" in _TUNABLES
