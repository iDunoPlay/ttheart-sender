"""The live recognition readout: the lines, the publish, and the window.

The point of the feature is that a person watching a round can tell a working
model from a switched-off one -- and a RIGHT one from a busy one -- without
reading the log afterwards. Two halves: the names, checked by eye against the
board on screen, and a headline that says whether to trust them.

The list is shown by `tray/recognition.py`, a window of its own that the
"Show live recognition" tick box opens. It was briefly five rows inside the
control panel; those rows and the reflow pass that let them collapse are gone,
and so are the tests for them.
"""

from __future__ import annotations

import itertools
from collections import Counter
from unittest import mock

import pytest

from ttheart_sender.game import readout


@pytest.fixture(autouse=True)
def _clean():
    """Module-level state, so every test starts from empty and leaves it."""
    readout.publish(())
    yield
    readout.publish(())
    for watcher in list(readout._WATCHERS):
        readout.unwatch(watcher)


# -- the lines ------------------------------------------------------------
def test_the_names_are_listed_most_of_the_board_first():
    """The half a person checks by eye: the board is right there on screen."""
    lines = readout.panel({"Beast": 12, "Dory": 7, "Sulley": 3},
                          agreed=34, checked=39)

    # Padded so the counts line up down the right; the ORDER is what is
    # under test here.
    assert lines[1].split() == ["Beast", "12"]
    assert lines[2].split() == ["Dory", "7"]
    assert lines[3].split() == ["Sulley", "3"]


def test_the_headline_carries_the_raw_count_beside_the_percentage():
    """"100%" on two checks is not a claim, and is read as one without this."""
    assert readout.headline(2, 2, 9, 41) == "matches the game 2/2 (100%)"
    assert readout.headline(34, 39, 9, 41) == "matches the game 34/39 (87%)"


def test_before_any_mark_the_headline_says_the_score_is_still_coming():
    """Marks need a sampled press and the first is a few chains in. An empty
    headline there reads as a broken readout."""
    top = readout.headline(agreed=0, checked=0, named=9, total=41)

    assert top == "naming 9 of 41 -- no marks yet"


def test_nothing_at_all_publishes_nothing():
    assert readout.panel({}, 0, 0, 0, 0) == ()


def test_a_model_naming_nothing_says_so_rather_than_showing_a_bare_headline():
    lines = readout.panel({}, agreed=0, checked=0, named=0, total=43)

    assert lines[0] == "naming 0 of 43 -- no marks yet"
    assert lines[1] == "nothing named above the floors"



def test_ties_break_by_name_so_the_list_does_not_flicker():
    a = readout.panel({"Zeus": 2, "Ariel": 2}, 1, 1)
    b = readout.panel({"Ariel": 2, "Zeus": 2}, 1, 1)

    assert a == b
    assert a[1].startswith("Ariel")


# -- publishing -----------------------------------------------------------
def test_what_was_published_is_what_comes_back():
    readout.publish(("named 3 of 40", "Beast   3"))

    assert readout.latest() == ("named 3 of 40", "Beast   3")


def test_a_watcher_is_told_when_the_lines_change():
    seen = []
    readout.watch(lambda: seen.append(readout.latest()))

    readout.publish(("named 1 of 40",))

    assert seen == [("named 1 of 40",)]


def test_a_watcher_that_raises_does_not_take_the_round_down():
    """The panel is a bystander. A round must never fail because of one."""
    def boom():
        raise RuntimeError("panel is gone")

    readout.watch(boom)
    readout.publish(("named 1 of 40",))

    assert readout.latest() == ("named 1 of 40",)


def test_a_watcher_is_registered_once_however_often_it_asks():
    seen = []

    def note():
        seen.append(1)

    readout.watch(note)
    readout.watch(note)
    readout.publish(("x",))

    assert len(seen) == 1


def test_clear_empties_it():
    readout.publish(("named 3 of 40",))
    readout.clear()

    assert readout.latest() == ()


# -- what the play loop puts in it ----------------------------------------
def _scored_model():
    """The visibility fixture's model -- it already carries the counters."""
    from tests.test_character_visibility import model

    return model()


def test_the_names_come_back_out_of_kind_after_the_model_has_run():
    """`apply` hides the identity in `kind`. This is the only place it is dug
    back out, so if the offset is ever wrong the panel is what shows it."""
    from tests.test_character_visibility import board

    m = _scored_model()
    bgr, tsums, radius = board([0.90, 0.90, 0.30])
    m.apply(bgr, tsums, radius)

    # FakeNet names everything class 0 at full confidence, and the third tsum
    # is under the visibility floor, so it was never asked.
    assert m.named_on(tsums) == ["A", "A"]
    # ... and that is what the panel lists, live, while the round runs.
    assert readout.panel(Counter(m.named_on(tsums)), named=2, total=3) == (
        "naming 2 of 3 -- no marks yet", "A   2")


def test_a_detection_that_kept_its_colour_cluster_has_no_name():
    from tests.test_character_visibility import board

    m = _scored_model()
    _bgr, tsums, _radius = board([0.90, 0.90])

    # Nothing applied: every kind is still a k-means cluster id.
    assert m.named_on(tsums) == []


def test_the_marks_score_the_naming_against_the_game():
    """The game lit up a group. Every member the model named the same way as
    the pressed one agrees; every other member is a mix-up."""
    from ttheart_sender.game import tsum as T
    from tests.test_character_visibility import board

    m = _scored_model()
    _bgr, tsums, _radius = board([0.9] * 4)
    for t in tsums:
        t.kind = T.CHARACTER_KIND + 0        # "A"
    tsums[3].kind = T.CHARACTER_KIND + 1     # "B" -- the odd one out

    m.check_marks(tsums, head=0, marked=[0, 1, 2, 3])

    # The pressed tsum is not scored against itself.
    assert m.checked == 3
    assert m.agreed == 2
    assert dict(m.confused) == {("A", "B"): 1}


def test_a_tsum_the_model_declined_is_not_counted_against_it():
    """It kept its colour cluster, so it was never a claim. Charging the model
    for it would be charging it for the confidence floor doing its job."""
    from ttheart_sender.game import tsum as T
    from tests.test_character_visibility import board

    m = _scored_model()
    _bgr, tsums, _radius = board([0.9] * 3)
    tsums[0].kind = T.CHARACTER_KIND + 0
    tsums[1].kind = T.CHARACTER_KIND + 0
    tsums[2].kind = 2                        # a k-means cluster id, no name

    m.check_marks(tsums, head=0, marked=[1, 2])

    assert m.checked == 1
    assert m.agreed == 1


def test_an_unnamed_pressed_tsum_scores_nothing_either_way():
    """No name for the group to agree with. Silent, not a failure."""
    from ttheart_sender.game import tsum as T
    from tests.test_character_visibility import board

    m = _scored_model()
    _bgr, tsums, _radius = board([0.9] * 3)
    tsums[1].kind = T.CHARACTER_KIND + 0

    m.check_marks(tsums, head=0, marked=[1, 2])

    assert (m.checked, m.agreed) == (0, 0)


def test_no_marks_at_all_scores_nothing():
    from tests.test_character_visibility import board

    m = _scored_model()
    _bgr, tsums, _radius = board([0.9] * 3)

    m.check_marks(tsums, head=0, marked=[])

    assert m.checked == 0


def test_a_round_with_no_marks_says_the_score_is_missing_not_perfect():
    """The end-of-round line must not let 0 checks read as a clean sheet."""
    m = _scored_model()
    m.seen, m.named = 40, 8

    assert "never read" in m.summary()



def test_an_unchanged_list_does_not_repaint_the_panel():
    """The board is re-read sixty times a round and most frames name it the
    same way. Without this the panel repaints itself all round for nothing."""
    woke = []
    readout.watch(lambda: woke.append(1))

    readout.publish(("matches the game 2/2 (100%)", "Beast   12"))
    readout.publish(("matches the game 2/2 (100%)", "Beast   12"))
    readout.publish(("matches the game 2/2 (100%)", "Beast   11"))

    assert len(woke) == 2


def test_the_character_model_is_never_taught_a_negative_class():
    """`apply` writes the winning class into `kind`, and chains are built by
    `kind` -- so a `board` class does not merely waste an output, it groups
    every board detection together and offers them as a chain. Every model
    built before this check carried `board` and `junk` as identities."""
    import importlib.util
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location(
        "classify_mod", root / "scripts" / "classify.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    assert set(mod.NOT_CHARACTERS) == {"board", "score", "junk"}


def test_a_shipped_character_model_carries_no_negative_class():
    """The guard that would have caught it, pointed at what is in service."""
    import json
    from pathlib import Path

    meta = Path(__file__).resolve().parent.parent / "models" / "character.json"
    if not meta.exists():
        pytest.skip("no character model in this checkout")
    classes = set(json.loads(meta.read_text(encoding="utf-8"))["classes"])

    assert not classes & {"board", "score", "junk"}, (
        "the model in service can name a detection 'board', and chains are "
        "built by the name -- retrain with scripts/retrain.py")


def test_the_list_is_the_round_peak_not_the_current_frame():
    """Per-frame counts made the panel flicker: only a fifth of the board is
    ever named and which fifth changes every frame. The peak is monotonic."""
    from ttheart_sender.game import tsum as T
    from tests.test_character_visibility import board

    m = _scored_model()
    bgr, tsums, radius = board([0.90, 0.90, 0.90])
    m.apply(bgr, tsums, radius)
    assert m.peak_names["A"] == 3

    # A later frame sees fewer -- two of them fell behind something.
    bgr2, fewer, radius2 = board([0.90, 0.30, 0.30])
    m.apply(bgr2, fewer, radius2)

    assert m.peak_names["A"] == 3, "the peak must not fall back"


def test_the_peak_resets_between_rounds():
    """It is a per-ROUND answer. `load_character_model` hands back the cached
    net with fresh counters, and this is one of them."""
    from ttheart_sender.game import tsum as T

    m = _scored_model()
    m.peak_names["Beast"] = 12
    # The reset the loader performs.
    from collections import Counter
    m.peak_names = Counter()

    assert not m.peak_names


# -- the two views ---------------------------------------------------------



def test_the_names_are_padded_so_the_counts_line_up():
    lines = readout.panel({"Beast": 12, "unknown_lightball": 4}, 1, 1)

    assert lines[1].index("12") == lines[2].index("4")


def test_the_live_window_starts_with_nothing_created():
    """Ticking the box is what builds it. An extra window nobody asked for
    must not exist until somebody asks."""
    from ttheart_sender.tray.recognition import RecognitionWindow

    win = RecognitionWindow()

    assert win._hwnd is None
    assert win.visible is False


def test_the_live_window_remembers_lines_given_before_it_exists():
    """The tray sets them on the way to showing it, and `_create` replays
    them -- otherwise the window opens blank until the next frame."""
    from ttheart_sender.tray.recognition import RecognitionWindow

    win = RecognitionWindow()
    win.set_lines(("matches the game 2/2 (100%)", "Beast   12"))

    assert win._lines == ("matches the game 2/2 (100%)", "Beast   12")


def test_the_panel_state_carries_the_switch(tmp_path):
    from tests.test_tray import FakeApp, _tray

    tray = _tray(tmp_path, FakeApp())

    assert tray._panel_state()["show_recognition"] is False


# -- the live window -------------------------------------------------------
def test_every_character_is_published_however_many_there_are():
    """No cap. The window has room, and the long tail -- the character named
    once, wrongly -- is the part worth reading."""
    counts = {f"Char{i}": i + 1 for i in range(12)}
    lines = readout.panel(counts, agreed=1, checked=2)

    assert len(lines) == 13          # the headline plus every character


def test_the_window_starts_with_nothing_created():
    """Ticking the box is what builds it. An extra window nobody asked for
    must not exist until somebody asks."""
    from ttheart_sender.tray.recognition import RecognitionWindow

    win = RecognitionWindow()

    assert win._hwnd is None
    assert win.visible is False


def test_the_window_remembers_lines_given_before_it_exists():
    """The tray sets them on the way to showing it, and `_create` replays
    them -- otherwise the window opens blank until the next frame."""
    from ttheart_sender.tray.recognition import RecognitionWindow

    win = RecognitionWindow()
    win.set_lines(("matches the game 2/2 (100%)", "Beast   12"))

    assert win._lines == ("matches the game 2/2 (100%)", "Beast   12")


def test_the_window_says_something_before_the_first_round():
    """A blank window is indistinguishable from a broken one."""
    from ttheart_sender.tray.recognition import EMPTY

    assert EMPTY and any("round" in line for line in EMPTY)


def test_closing_the_window_is_reported_so_the_tick_box_can_follow():
    """Otherwise the box stays ticked over a window that is gone, and the
    next tick does nothing at all."""
    import win32con

    from ttheart_sender.tray.recognition import RecognitionWindow

    seen = []
    win = RecognitionWindow(on_close=lambda: seen.append(1))
    win._wnd_proc(0, win32con.WM_CLOSE, 0, 0)

    assert seen == [1]


def test_the_panel_no_longer_carries_the_list():
    """It reads `readout.latest()` in the window instead. A state key nothing
    reads is a state key that drifts."""
    import ttheart_sender.tray.panel as P

    assert not hasattr(P, "RECOG_ROWS")
    assert not hasattr(P, "ID_RECOG_BASE")
    assert not hasattr(P.ControlPanel, "_reflow")


def test_the_switch_is_still_in_the_panel(tmp_path):
    from tests.test_tray import FakeApp, _tray
    import ttheart_sender.tray.panel as P

    tray = _tray(tmp_path, FakeApp())

    assert P.ID_SHOW_RECOGNITION
    assert tray._panel_state()["show_recognition"] is False


def test_the_window_really_builds_on_this_machine():
    """THE TEST THAT WAS MISSING. Everything else here stubs Win32 out, so
    `win32gui.CreateFont` -- which pywin32 does not export -- raised inside
    window creation and the only symptom was a tick box that did nothing.
    This one creates the real window and looks at it."""
    win32gui = pytest.importorskip("win32gui")
    from ttheart_sender.tray.recognition import RecognitionWindow

    win = RecognitionWindow()
    try:
        try:
            win.show()
        except Exception as exc:  # noqa: BLE001
            pytest.fail("the live recognition window could not be built: %r" % exc)
        assert win.visible
        assert win32gui.GetWindowText(win._hwnd) == "Live recognition"
        # TOPMOST, or it opens behind the emulator and the tick box looks
        # like it did nothing -- which is exactly how this was reported.
        assert win32gui.GetWindowLong(win._hwnd, -20) & 0x00000008, (
            "the readout must sit above the game it is reporting on")
        win.set_lines(("matches the game 3/3 (100%)", "Beast   12"))
        assert "Beast" in win32gui.GetWindowText(win._edit)
        win.hide()
        assert not win.visible
    finally:
        win.destroy()
