"""Sample collection: the labels are free, so the only job is not losing them.

The game's highlight is visible for a moment in the middle of a drag, which is
why this is captured live rather than reconstructed later. Everything here
guards that one moment: it is recorded, it is capped, and a disk that will not
take it costs a warning rather than a round.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from ttheart_sender.game import tsum
from ttheart_sender.game.dataset import DatasetWriter


#: A reading that passes both quality gates: a still board (low baseline) and
#: one tsum lit rather than the whole screen.
CLEAN = {"values": [0.0, 40.0], "baseline": 1.0, "marked": [1]}


def _sample(writer, **over):
    tsums = [tsum.Tsum(x=10.0, y=10.0, r=24.0, kind=1, colour=(1, 2, 3)),
             tsum.Tsum(x=40.0, y=10.0, r=24.0, kind=1, colour=(1, 2, 3))]
    kwargs = dict(board=(8, 265, 32, 32), radius=24.0, tsums=tsums,
                  head=0, proposed=[0, 1], kept=[0], fever=False,
                  reading=dict(CLEAN))
    kwargs.update(over)
    writer.record(np.zeros((32, 32, 3), np.uint8), np.ones((32, 32, 3), np.uint8), **kwargs)


def test_nothing_is_written_until_there_is_a_sample(tmp_path):
    # `enabled: true` in config.yaml should cost nothing on a round that never
    # gets a drag away -- no empty folders to explain.
    writer = DatasetWriter(tmp_path / "dataset")
    assert not (tmp_path / "dataset").exists()
    assert writer.summary() == ""


def test_a_sample_is_two_frames_and_a_row(tmp_path):
    writer = DatasetWriter(tmp_path / "dataset")
    _sample(writer)
    writer.close()

    session = next((tmp_path / "dataset").glob("*_*"))
    assert (session / "0001_before.jpg").exists()
    assert (session / "0001_marked.jpg").exists(), "the labelled frame is the whole point"

    rows = [json.loads(line) for line in (session / "samples.jsonl").read_text().splitlines()]
    assert len(rows) == 1
    row = rows[0]
    assert row["board"] == [8, 265, 32, 32]
    assert row["proposed"] == [0, 1] and row["kept"] == [0]
    assert [t["kind"] for t in row["tsums"]] == [1, 1]
    assert row["schema"] and row["version"], "a session has to say what wrote it"
    # The reading travels with the sample. Schema 1 did not record it, so the
    # only way to find out that it had photographed nothing was to re-decode
    # 803MB of JPEG -- `tsum dataset` reads these three fields instead.
    assert row["baseline"] == 1.0
    assert row["marks"] == [0.0, 40.0] and row["marked"] == [1]
    assert row["capture"] == {"delay": writer.delay, "frames": writer.frames,
                              "gap": writer.gap, "floor_mult": writer.floor_mult}


def test_a_round_stops_collecting_at_the_limit(tmp_path):
    # Boards inside one round are highly alike; the cap is what keeps a long
    # session from being mostly duplicates.
    writer = DatasetWriter(tmp_path / "dataset", per_round=3)
    for _ in range(10):
        _sample(writer)
    writer.close()

    session = next((tmp_path / "dataset").glob("*_*"))
    assert len(list(session.glob("*_before.jpg"))) == 3


def test_the_stride_spreads_samples_across_the_round(tmp_path):
    writer = DatasetWriter(tmp_path / "dataset", per_round=2, every=3)
    wanted = []
    for _ in range(9):
        wanted.append(writer.wants())
        writer.seen_drag()
        if wanted[-1]:
            _sample(writer)
    # Every third drag, and nothing once the cap is reached.
    assert wanted == [True, False, False, True, False, False, False, False, False]


def test_a_write_failure_disables_collection_instead_of_raising(tmp_path, monkeypatch):
    writer = DatasetWriter(tmp_path / "dataset")
    monkeypatch.setattr("cv2.imwrite", lambda *a, **kw: (_ for _ in ()).throw(OSError("full")))

    _sample(writer)          # must not raise -- a round is worth more than a sample

    assert not writer.enabled
    assert writer.wants() is False


# -- the live path ---------------------------------------------------------
class Templates:
    def get(self, name):
        raise KeyError(name)


def test_collection_reads_the_marks_without_changing_the_drag(tmp_path, monkeypatch):
    """--dataset alone must not trim the stroke; only --verify-hold does that.

    Both features need the same press-and-look, so the loop runs it when
    either is on. The one that is off has to stay off: a round played for
    data has to be the round that would have been played anyway, or the
    samples describe a bot nobody runs.
    """
    frame = np.zeros((994, 578, 3), np.uint8)
    # Enough of them to pass the loop's own plausibility gate -- a frame with
    # four blobs is a menu, not a board, and is discarded before any drag.
    tsums = [tsum.Tsum(x=20.0 * (i % 20) + 20, y=30.0 * (i // 20) + 30,
                       r=24.0, kind=1, colour=(0, 0, 0)) for i in range(30)]
    chain = tsum.Chain(kind=1, colour=(0, 0, 0), nodes=[0, 1, 2])

    monkeypatch.setattr(tsum, "_settle", lambda drv, max_wait=0.0: frame)
    monkeypatch.setattr(tsum, "detect",
                        lambda crop, **kw: (tsums, 24.0, np.zeros((12, 3), np.float32)))
    monkeypatch.setattr(tsum, "find_chains", lambda *a, **kw: [chain])
    monkeypatch.setattr(tsum, "purity_filter", lambda bgr, ts, nodes, r, tol: list(nodes))

    dragged = []

    def fake_drag(screen, *, step_px, per_step, hold, after_press=None):
        trimmed = after_press() if after_press else None
        dragged.append((list(screen), trimmed))

    monkeypatch.setattr(tsum, "drag_chain", fake_drag)

    opts = tsum.play_defaults()
    opts.duration, opts.settle, opts.hold_delay = 30.0, 0.0, 0.0
    opts.skill, opts.bubble, opts.use_base = "", "", False
    opts.verify_hold = False                      # collection only
    opts.dataset = str(tmp_path / "dataset")
    opts.dataset_every, opts.dataset_limit = 1, 2

    drv = SimpleNamespace(capture=None, matcher=None, templates=Templates(), rect=None,
                          grab=lambda: frame, to_screen=lambda x, y: (int(x), int(y)),
                          check_stop=lambda: None, say=lambda msg: None)
    tsum.play_loop(drv, opts, stop_when=lambda _f: "done" if dragged else "")

    proposed, trimmed = dragged[0]
    assert trimmed == proposed, "the stroke the game was asked about is the one dragged"

    session = next((tmp_path / "dataset").glob("*_*"))
    row = json.loads((session / "samples.jsonl").read_text().splitlines()[0])
    assert row["proposed"] == [0, 1, 2]
    assert row["options"]["hold_aura"] == opts.hold_aura, "settings travel with the sample"
    assert (session / "0001_marked.jpg").exists()


def test_config_switches_collection_on_for_every_flow_that_plays(monkeypatch, tmp_path):
    """`enabled: true` in config.yaml has to be the whole switch.

    Rounds are played from flows and from the tray, not from a command line,
    so a knob that only exists as a CLI flag would never be reachable by the
    person collecting the data.
    """
    from ttheart_sender.automation import tsum_actions
    from ttheart_sender.automation.flow import parse_step
    from ttheart_sender.automation.params import Params
    from ttheart_sender.config import Config
    from ttheart_sender.game.tsum import PlayReport

    config = Config()
    config.output_root = tmp_path
    config.dataset.enabled = True
    config.dataset.per_round = 7

    seen = {}
    monkeypatch.setattr(tsum_actions.Driver, "from_context",
                        classmethod(lambda cls, ctx, say=None: object()))
    monkeypatch.setattr(tsum_actions, "play_loop",
                        lambda drv, opts, stop_when=None: seen.update(opts=opts) or PlayReport())

    ctx = SimpleNamespace(config=config, dry_run=False, indent="", templates=None,
                          matcher=None, set_var=lambda k, v: None)
    step = parse_step({"play_tsum": {"duration": 5, "until_found": ["timeup"]}},
                      source="test", path="steps[0]")
    monkeypatch.setattr(tsum_actions, "_stop_checker", lambda *a, **kw: None)

    tsum_actions.act_play_tsum(ctx, Params(step))

    assert seen["opts"].dataset == str(tmp_path / "dataset"), "beside the .exe, not in a temp dir"
    assert seen["opts"].dataset_limit == 7


# -- what schema 1 got wrong -----------------------------------------------
#
# 5,729 samples were collected before anybody could tell they carried no
# label. Each test below is one of the two reasons, pinned so the next
# collection cannot repeat it.

def test_the_delay_cannot_be_set_below_the_render_floor(tmp_path):
    """Under ~0.15s the game has not drawn the highlight yet.

    The number is not a preference. `--verify-hold` runs at 0.10 on purpose --
    it pays the delay on every drag -- and the collector inherited it, so
    every "marked" frame was photographed before there was anything to
    photograph. A config asking for that back is asking for another empty
    night, so it is clamped rather than obeyed.
    """
    from ttheart_sender.game.dataset import RENDER_FLOOR

    assert DatasetWriter(tmp_path, delay=0.10).delay == RENDER_FLOOR
    assert DatasetWriter(tmp_path, delay=0.40).delay == 0.40, "generous is fine"


def test_a_sample_taken_while_the_board_moved_is_refused(tmp_path):
    """A moving board clears any mark threshold, so the reading is motion.

    Measured over schema 1: the median untouched tsum had already changed
    more than the 8.0 threshold on 56% of samples. Those are not noisy
    labels -- on disk they are indistinguishable from correct ones, so the
    refusal has to happen while the baseline is still known.
    """
    writer = DatasetWriter(tmp_path / "dataset", max_motion=12.0)
    _sample(writer, reading={"values": [90.0, 95.0], "baseline": 88.0, "marked": [1]})

    assert writer.written == 0
    assert writer.refused == {"board still moving": 1}
    assert not (tmp_path / "dataset").exists(), "and nothing on disk to mistake for data"


def test_a_reading_that_lights_the_whole_board_is_refused(tmp_path):
    """The game marks one character, never most of the screen.

    A wash that big is a screen transition or FEVER, not an answer -- the
    same judgement `assist --max-marked` makes on the live path.
    """
    tsums = [tsum.Tsum(x=10.0 * i, y=10.0, r=24.0, kind=1, colour=(1, 2, 3))
             for i in range(10)]
    writer = DatasetWriter(tmp_path / "dataset")
    _sample(writer, tsums=tsums,
            reading={"values": [40.0] * 10, "baseline": 1.0, "marked": list(range(9))})

    assert writer.written == 0
    assert writer.refused == {"most of the board lit up": 1}


def test_a_round_that_collects_nothing_says_why(tmp_path):
    """Otherwise a broken collection is indistinguishable from a quiet one.

    `enabled: true` and an empty folder next morning has two explanations,
    and only the report separates them.
    """
    writer = DatasetWriter(tmp_path / "dataset")
    _sample(writer, reading={"values": [90.0], "baseline": 88.0, "marked": []})

    assert "board still moving" in writer.summary()


def test_collection_stops_at_the_dataset_budget(tmp_path):
    """There is no per-day cap anywhere else in the app.

    `per_round` bounds one round; nothing bounded the sum of them, and
    unattended that came to 803MB and 5,729 samples in one night.
    """
    root = tmp_path / "dataset"
    first = DatasetWriter(root, per_round=2)
    _sample(first)
    first.close()

    capped = DatasetWriter(root, max_total=1)
    _sample(capped)

    assert capped.written == 0 and not capped.enabled
    assert len(list(root.glob("*_*"))) == 1, "no second session folder was opened"


def test_no_budget_means_no_budget(tmp_path):
    """0 keeps the old behaviour for anyone who wants the disk filled."""
    root = tmp_path / "dataset"
    DatasetWriter(root, max_mb=0, max_total=0)
    writer = DatasetWriter(root, max_mb=0, max_total=0)
    _sample(writer)
    assert writer.written == 1


def test_the_collector_reads_the_mark_on_its_own_terms(monkeypatch, tmp_path):
    """A sampled drag waits longer and looks more than once; `play` does not.

    Both features read the same highlight through `marked_by_game`, but they
    cannot share settings. `--verify-hold` pays on every drag, which is why it
    is tuned down to one frame at 0.10s; collection pays on one drag in four,
    and 0.10s is precisely what made schema 1 worthless. So the read has to
    take the collector's settings on the drags it samples and only those.
    """
    import numpy as np

    board, calls, grabs = (0, 0, 8, 8), [], []
    frames = [np.full((8, 8, 3), v, np.uint8) for v in (10, 11, 12)]

    def grab():
        calls.append("grab")
        grabs.append(len(grabs))
        return frames[len(grabs) - 1]

    drv = SimpleNamespace(grab=grab)
    monkeypatch.setattr(tsum.time, "sleep", lambda s: calls.append(f"sleep{s}"))

    tsums = [tsum.Tsum(x=2.0, y=2.0, r=3.0, kind=1, colour=(0, 0, 0)),
             tsum.Tsum(x=6.0, y=6.0, r=3.0, kind=1, colour=(0, 0, 0))]
    seen: dict = {}
    tsum.marked_by_game(drv, np.zeros((8, 8, 3), np.uint8), board, tsums, [0, 1],
                        delay=0.25, threshold=8.0, aura=0.0, frames=3, gap=0.05,
                        out=seen)

    assert calls.count("grab") == 3, "three frames, so a settling tsum cannot pass"
    assert calls[0] == "sleep0.25", "and the wait is the collector's, not 0.10"
    # min across the three frames, so the reading is the change that persisted.
    assert seen["values"] == [10.0, 10.0]
    assert "baseline" in seen and "marked_frame" in seen


# -- reading the onset off `hold` ------------------------------------------

def _onset(values, threshold=8.0):
    """Drive `_mark_onset` over a synthetic hold and capture what it prints."""
    import io, contextlib
    import numpy as np

    size = 60
    before = np.zeros((size, size, 3), np.uint8)
    tsums = [tsum.Tsum(x=15.0, y=15.0, r=8.0, kind=1, colour=(0, 0, 0)),
             tsum.Tsum(x=45.0, y=15.0, r=8.0, kind=1, colour=(0, 0, 0))]
    shots, stamps = [], []
    for i, v in enumerate(values):
        frame = np.zeros((size, size, 3), np.uint8)
        import cv2
        cv2.circle(frame, (45, 15), 5, (v, v, v), -1)
        shots.append(frame)
        stamps.append(i * 0.1)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        tsum._mark_onset(shots, stamps, before, (0, 0, size, size), tsums, [1],
                         8.0, threshold)
    return buf.getvalue()


def test_onset_reads_the_delay_off_a_mark_that_fades_in(tmp_path):
    """The one number collection needs, and the one `hold` never reported.

    It sampled a three-second hold and printed only the strongest reaction, so
    "the mark exists" and "the mark exists by 0.25s" were the same output.
    """
    out = _onset([0, 0, 0, 60, 200, 200, 200])

    assert "first over 8 at +0.30s" in out
    assert "80% of peak at +0.40s" in out
    assert "set dataset.delay to about 0.50" in out, "with headroom over the onset"


def test_an_onset_in_the_very_first_frame_is_refused(tmp_path):
    """A first-frame peak cannot be timed, so no delay may be recommended.

    Two readings fit it -- the mark renders faster than a press can be
    photographed, or the baseline already differed (a chain still glowing, a
    score popup in flight) -- and a single hold cannot separate them. What
    matters is that neither justifies a number: reporting the onset as +0.00s
    recommends 0.15s, which is exactly the timing that made the first 5,729
    samples worthless.
    """
    out = _onset([200, 200, 200, 200])

    assert "UNRESOLVED" in out
    assert "set dataset.delay" not in out, "no recommendation from a broken read"


def test_a_hold_with_no_mark_reports_no_onset(tmp_path):
    """Nothing to time, so nothing is said -- the verdict above already spoke."""
    assert _onset([0, 0, 0, 0]) == ""


# -- the label bar and the trim bar are not the same number -----------------

def test_the_floor_relative_bar_labels_but_never_trims(monkeypatch):
    """`floor_mult` raises the label's bar and leaves `--verify-hold`'s alone.

    Only one caller trims: `--verify-hold`, whose A/B was measured against the
    fixed `--hold-threshold` and lost decisively under it. `floor_mult` exists
    because that fixed number sits inside a live board's own noise and made
    every collected label read like half the board -- but applying it to the
    trim as well would quietly re-score `--verify-hold` in exactly the
    combination where collection happens to be on too, which is the one place
    nobody would look for it.

    So: a chain member that clears the fixed threshold but not the
    floor-relative bar is dragged, and is not in the label.
    """
    import numpy as np

    size = 60
    before = np.zeros((size, size, 3), np.uint8)
    # A board with a lively floor: 12.0 of change everywhere, so a 5x bar
    # lands at 60 and the fixed 8.0 admits anything at all.
    frame = np.full((size, size, 3), 12, np.uint8)
    import cv2
    cv2.circle(frame, (45, 15), 5, (200, 200, 200), -1)   # a real mark
    monkeypatch.setattr(tsum.time, "sleep", lambda s: None)
    drv = SimpleNamespace(grab=lambda: frame)

    tsums = [tsum.Tsum(x=15.0, y=15.0, r=8.0, kind=1, colour=(0, 0, 0)),
             tsum.Tsum(x=45.0, y=15.0, r=8.0, kind=1, colour=(0, 0, 0)),
             tsum.Tsum(x=15.0, y=45.0, r=8.0, kind=1, colour=(0, 0, 0)),
             tsum.Tsum(x=45.0, y=45.0, r=8.0, kind=1, colour=(0, 0, 0))]
    seen: dict = {}
    kept = tsum.marked_by_game(drv, before, (0, 0, size, size), tsums, [0, 1, 2],
                               delay=0.0, threshold=8.0, aura=0.0,
                               floor_mult=5.0, out=seen)

    assert seen["bar"] > 8.0, "the label is scored against the board's own floor"
    assert kept == [0, 1, 2], "the trim still runs at the fixed threshold"
    assert 2 not in seen["marked"], "but the floor-level change is not a label"


def test_the_collector_scores_marks_at_eight_times_the_floor():
    """The default that the 11,537-sample collection settled.

    Swept over that corpus, tsums reacting between 2x and 8x the board's own
    floor are no more the pressed character than the board average is -- lift
    1.00 against a 25% same-kind base rate -- and everything above 8x carries
    the signal. The first tuned value, 5, came from 19 samples and spent about
    a third of every label on noise.
    """
    from pathlib import Path

    from ttheart_sender.config import DatasetConfig

    assert DatasetWriter(Path("x")).floor_mult == 8.0
    assert DatasetConfig().floor_mult == 8.0
    assert tsum.play_defaults().dataset_floor_mult == 8.0
