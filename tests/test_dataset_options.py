"""A collection has to state its own conditions, without being asked.

Three rounds running, a corpus arrived that could not say what it was played
at. The eleventh round could not tell whether `verify_reach` had been armed
while it collected; the twelfth bolted `fit_effort` onto a hand-written dict
of a dozen keys; the thirteenth had to be *told* by the player which tsum was
equipped, and `verify_clears` -- the switch the whole round was about -- was
never recorded at all.

The common cause is not any of those omissions. It is that the list was
curated, and the key that matters is always the one nobody remembered to add.
So the tests here are about the property rather than the keys: everything the
parser can produce is recorded, and a flag invented tomorrow is recorded by
the same code with no one editing it.
"""

from __future__ import annotations

import json

import numpy as np

from ttheart_sender.game import tsum
from ttheart_sender.game.dataset import SCHEMA, DatasetWriter

CLEAN = {"values": [0.0, 40.0], "baseline": 1.0, "marked": [1]}


def _sample(writer, **over):
    tsums = [tsum.Tsum(x=10.0, y=10.0, r=24.0, kind=1, colour=(1, 2, 3)),
             tsum.Tsum(x=40.0, y=10.0, r=24.0, kind=1, colour=(1, 2, 3))]
    kwargs = dict(board=(8, 265, 32, 32), radius=24.0, tsums=tsums,
                  head=0, proposed=[0, 1], kept=[0], fever=False,
                  reading=dict(CLEAN))
    kwargs.update(over)
    writer.record(np.zeros((32, 32, 3), np.uint8),
                  np.ones((32, 32, 3), np.uint8), **kwargs)


def _row(tmp_path, **over):
    writer = DatasetWriter(tmp_path / "dataset")
    _sample(writer, **over)
    # The row is staged until the drag it describes is over -- see
    # `DatasetWriter.note_outcome`. `close` is what `play_loop` does in its
    # `finally`, so this is the same flush a real round performs.
    writer.close()
    line = next((tmp_path / "dataset").glob("*/samples.jsonl")).read_text().splitlines()[0]
    return json.loads(line)


# -- the snapshot is not curated -------------------------------------------
def test_every_play_setting_is_recorded():
    """No subset. The subset is the bug this file exists about."""
    opts = tsum.play_defaults()
    snapshot = tsum.play_settings(opts)
    assert set(snapshot) == set(vars(opts)), (
        "a setting the parser can produce is missing from the snapshot")


def test_a_setting_added_tomorrow_is_recorded_without_editing_anything():
    # The property that keeps this from going stale again: the snapshot reads
    # the namespace, so it cannot be out of date with the parser.
    opts = tsum.play_defaults()
    opts.a_rule_invented_next_month = True
    assert tsum.play_settings(opts)["a_rule_invented_next_month"] is True


def test_the_snapshot_is_json_and_stays_json():
    """Recording the whole namespace is only safe while it is all scalars.

    If a future option arrives as a Path, a list or an object, this fails here
    rather than by writing a row that cannot be read back.
    """
    snapshot = tsum.play_settings(tsum.play_defaults())
    assert json.loads(json.dumps(snapshot)) == snapshot
    for key, value in snapshot.items():
        assert isinstance(value, (bool, int, float, str, type(None))), key


def test_the_effective_value_is_recorded_not_the_requested_one():
    # `play_loop` clamps `verify_delay` up to the render floor. A corpus that
    # recorded the asked-for value would describe a reading nobody took.
    opts = tsum.play_defaults()
    opts.verify_delay = 0.01
    assert tsum.play_settings(opts)["verify_delay"] == 0.01
    opts.verify_delay = 0.15
    assert tsum.play_settings(opts)["verify_delay"] == 0.15


def test_the_experiment_switches_are_all_in_there(tmp_path):
    # Named explicitly because each of these was, at some point, the switch a
    # round turned on and the corpus could not report.
    row = _row(tmp_path, options=tsum.play_settings(tsum.play_defaults()))
    for key in ("fit_effort", "verify_clears", "verify_extend", "verify_reach",
                "verify_hold", "max_chain", "block", "link_px", "mode"):
        assert key in row["options"], key


# -- which tsum was equipped -----------------------------------------------
def test_read_base_kind_hands_back_the_icon_colour():
    """The only cross-session handle on the equipped character.

    The returned cluster index cannot serve -- it is a per-frame k-means id.
    """
    frame = np.zeros((994, 578, 3), np.uint8)
    frame[:] = (40, 90, 160)
    centres = np.array([[100.0, 128.0, 128.0], [60.0, 150.0, 200.0]], np.float32)

    seen: dict = {}
    kind, distance = tsum.read_base_kind(frame, centres, spec="76,854,20", out=seen)

    assert kind in (0, 1) and distance >= 0.0
    assert len(seen["icon_lab"]) == 3
    assert all(isinstance(v, float) for v in seen["icon_lab"])


def test_the_equipped_tsum_is_written_into_the_row(tmp_path):
    base = {"kind": 3, "lab": [55.0, 140.0, 190.0], "distance": 4.2}
    row = _row(tmp_path, base=base)
    assert row["base"] == base


def test_a_row_says_so_when_the_equipped_tsum_was_never_read(tmp_path):
    # `--use-base` off means it was not read. That has to read as "unknown"
    # rather than as a character, or a corpus silently claims to be one.
    row = _row(tmp_path)
    assert row["base"] is None


def test_the_schema_says_the_row_is_self_describing(tmp_path):
    row = _row(tmp_path)
    assert row["schema"] == SCHEMA >= 3


# -- end to end, through the loop that actually writes them ----------------
class _Templates:
    def get(self, name):
        raise KeyError(name)


def test_a_collected_round_writes_its_own_conditions(monkeypatch, tmp_path):
    """The whole point, exercised through `play_loop` rather than the writer.

    A corpus handed over on its own has to answer both questions nobody could
    answer for three rounds: which switches were armed, and which tsum was
    equipped.
    """
    from types import SimpleNamespace

    frame = np.zeros((994, 578, 3), np.uint8)
    frame[:] = (30, 80, 150)
    tsums = [tsum.Tsum(x=20.0 * (i % 20) + 20, y=30.0 * (i // 20) + 30,
                       r=24.0, kind=1, colour=(0, 0, 0)) for i in range(30)]
    chain = tsum.Chain(kind=1, colour=(0, 0, 0), nodes=[0, 1, 2])
    centres = np.tile(np.array([[50.0, 130.0, 190.0]], np.float32), (12, 1))

    monkeypatch.setattr(tsum, "_settle", lambda drv, max_wait=0.0, tol=2.5, region=None, out=None: frame)
    monkeypatch.setattr(tsum, "detect", lambda crop, **kw: (tsums, 24.0, centres))
    monkeypatch.setattr(tsum, "find_chains", lambda *a, **kw: [chain])
    monkeypatch.setattr(tsum, "purity_filter",
                        lambda bgr, ts, nodes, r, tol: list(nodes))

    dragged = []

    def fake_drag(screen, *, step_px, per_step, hold, after_press=None):
        if after_press:
            after_press()
        dragged.append(list(screen))

    monkeypatch.setattr(tsum, "drag_chain", fake_drag)

    opts = tsum.play_defaults()
    opts.duration, opts.settle, opts.hold_delay = 30.0, 0.0, 0.0
    opts.skill, opts.bubble = "", ""
    opts.use_base = True                     # so the equipped tsum is read
    opts.verify_clears = True                # an armed switch to find again
    opts.dataset = str(tmp_path / "dataset")
    opts.dataset_every, opts.dataset_limit = 1, 2

    drv = SimpleNamespace(capture=None, matcher=None, templates=_Templates(),
                          rect=None, grab=lambda: frame,
                          to_screen=lambda x, y: (int(x), int(y)),
                          check_stop=lambda: None, say=lambda msg: None)
    tsum.play_loop(drv, opts, stop_when=lambda _f: "done" if dragged else "")

    session = next((tmp_path / "dataset").glob("*_*"))
    row = json.loads((session / "samples.jsonl").read_text().splitlines()[0])

    assert row["schema"] >= 3
    # The switch that was on is findable without knowing to look for it.
    assert row["options"]["verify_clears"] is True
    assert set(row["options"]) == set(vars(opts)), "the whole namespace travels"
    # And which character played it.
    assert isinstance(row["base"], dict)
    assert len(row["base"]["lab"]) == 3


# -- what the drag actually did --------------------------------------------
def test_the_clear_result_is_written_into_the_sample_that_caused_it(tmp_path):
    """`verify_clears` measured what left the board and told only the log.

    So a round played to measure clears could not be handed over as a
    dataset: the answer stayed on the machine that played it. The sample is
    staged at the press and the outcome is attached when the drag is done.
    """
    writer = DatasetWriter(tmp_path / "dataset")
    _sample(writer)
    writer.note_outcome(cleared=[0, 1], clear_values=[44.0, 51.0], clear_idle=2.0)
    writer.close()

    line = next((tmp_path / "dataset").glob("*/samples.jsonl")).read_text().splitlines()[0]
    row = json.loads(line)
    assert row["cleared"] == [0, 1]
    assert row["clear_idle"] == 2.0


def test_a_staged_row_survives_a_round_that_never_measured_anything(tmp_path):
    # The clear check is optional. A sample taken without one still has to
    # reach the disk -- just with no outcome on it.
    writer = DatasetWriter(tmp_path / "dataset")
    _sample(writer)
    writer.close()

    row = json.loads(next((tmp_path / "dataset").glob("*/samples.jsonl"))
                     .read_text().splitlines()[0])
    assert "cleared" not in row, "absent means never measured, not measured as zero"


def test_the_next_press_flushes_the_drag_before_it(tmp_path):
    writer = DatasetWriter(tmp_path / "dataset")
    _sample(writer)
    writer.note_outcome(cleared=[0])
    _sample(writer)
    writer.note_outcome(cleared=[1])
    writer.close()

    lines = next((tmp_path / "dataset").glob("*/samples.jsonl")).read_text().splitlines()
    assert [json.loads(v)["cleared"] for v in lines] == [[0], [1]], (
        "each outcome belongs to the drag that produced it, in order")


def test_an_outcome_with_no_sample_in_flight_is_ignored(tmp_path):
    # Only one drag in `every` is sampled, but the clear check runs on all of
    # them -- so most calls arrive with nothing staged and must not explode.
    writer = DatasetWriter(tmp_path / "dataset")
    writer.note_outcome(cleared=[9])
    _sample(writer)
    writer.close()

    row = json.loads(next((tmp_path / "dataset").glob("*/samples.jsonl"))
                     .read_text().splitlines()[0])
    assert "cleared" not in row, "an unsampled drag's result must not land on a sample"


def test_a_measuring_round_hands_over_its_own_answer(monkeypatch, tmp_path):
    """End to end: "Measure tsums cleared" now lands in the corpus.

    Before this, the only record of what a drag cleared was a log line on the
    machine that played the round -- which, in this project, is a different
    machine from the one the dataset is read on.
    """
    from types import SimpleNamespace

    frame = np.zeros((994, 578, 3), np.uint8)
    tsums = [tsum.Tsum(x=20.0 * (i % 20) + 20, y=30.0 * (i // 20) + 30,
                       r=24.0, kind=1, colour=(0, 0, 0)) for i in range(30)]
    chain = tsum.Chain(kind=1, colour=(0, 0, 0), nodes=[0, 1, 2])

    monkeypatch.setattr(tsum, "_settle", lambda drv, max_wait=0.0, tol=2.5, region=None, out=None: frame)
    monkeypatch.setattr(tsum, "detect",
                        lambda crop, **kw: (tsums, 24.0, np.zeros((12, 3), np.float32)))
    monkeypatch.setattr(tsum, "find_chains", lambda *a, **kw: [chain])
    monkeypatch.setattr(tsum, "purity_filter",
                        lambda bgr, ts, nodes, r, tol: list(nodes))
    monkeypatch.setattr(tsum, "cleared_by_drag",
                        lambda b, a, ts, nodes, *, tol: ([nodes[0]],
                                                         [90.0] * len(nodes), 3.0))

    def marked_by_game(drv, before, board_rect, ts, nodes, *, delay, **kw):
        # The grabs below deliberately differ frame to frame so `--verify`
        # sees the drag register; a real reading off them would be all motion
        # and the collector would refuse the sample as unusable.
        out = kw.get("out")
        if out is not None:
            values = [0.0] * len(ts)
            values[nodes[1]] = 40.0
            out.update(marked_frame=np.zeros_like(frame), values=values,
                       baseline=1.0, bar=8.0, marked=[nodes[1]])
        return list(nodes)

    monkeypatch.setattr(tsum, "marked_by_game", marked_by_game)

    dragged = []

    def fake_drag(screen, *, step_px, per_step, hold, after_press=None):
        if after_press:
            after_press()
        dragged.append(list(screen))

    monkeypatch.setattr(tsum, "drag_chain", fake_drag)

    shade = {"n": 0}

    def grab():
        shade["n"] = (shade["n"] + 1) % 4
        return np.full_like(frame, 60 * shade["n"])

    opts = tsum.play_defaults()
    opts.duration, opts.settle, opts.hold_delay = 30.0, 0.0, 0.0
    opts.skill, opts.bubble, opts.use_base = "", "", False
    opts.verify, opts.verify_clears = True, True
    opts.dataset = str(tmp_path / "dataset")
    opts.dataset_every, opts.dataset_limit = 1, 2

    drv = SimpleNamespace(capture=None, matcher=None, templates=_Templates(),
                          rect=None, grab=grab,
                          to_screen=lambda x, y: (int(x), int(y)),
                          check_stop=lambda: None, say=lambda msg: None)
    tsum.play_loop(drv, opts, stop_when=lambda _f: "done" if dragged else "")

    session = next((tmp_path / "dataset").glob("*_*"))
    row = json.loads((session / "samples.jsonl").read_text().splitlines()[0])

    assert row["options"]["verify_clears"] is True, "the switch is on the record"
    assert row["cleared"] == [0], "and so is what it measured"
    assert row["dragged"] == [0, 1, 2], "beside what was attempted"
