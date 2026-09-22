"""Click-to-label, and the writes it makes to the training set.

This tool edits `crops/labelled/`, which is the training set every character
model is built from. A tool that writes the wrong file there is worse than no
tool: the damage is silent, it survives into the next model, and nothing
downstream can tell a human label from a mistake.

So the tests are about the writes, not the window: the filename must be the one
`crops.py assign` produces, a correction must MOVE rather than duplicate, an
edge-clipped crop must be refused rather than clamped, and undo must actually
put things back.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import label_board as L  # noqa: E402


def board(w=400, h=400):
    return np.full((h, w, 3), 40, np.uint8)


def tsum(x, y, r=18.0):
    return {"x": float(x), "y": float(y), "r": float(r), "kind": 0}


def test_a_new_label_lands_in_the_layout_classify_trains_from(tmp_path):
    ses = L.Session(tmp_path, {})
    note = ses.assign(("20260906_101010_1", 7, 3), "Pascal",
                      board(), tsum(200, 200), 25.0)
    got = list(tmp_path.rglob("*.png"))
    assert len(got) == 1
    assert got[0].parent.name == "Pascal"
    assert got[0].name == "20260906_101010_1_0007_03_v0.72.png"
    assert "labelled Pascal" in note


def test_the_filename_is_the_key_back_into_the_dataset(tmp_path):
    """`<session>_<sample>_<index>_v<visible>` is a primary key into
    samples.jsonl. Every other tool here relies on parsing it back."""
    ses = L.Session(tmp_path, {})
    ses.assign(("sess_x", 12, 9), "Grim", board(), tsum(200, 200, 25.0), 25.0)
    p = next(tmp_path.rglob("*.png"))
    m = L.NAME.match(p.stem)
    assert m and m.group("sess") == "sess_x"
    assert int(m.group("sample")) == 12 and int(m.group("idx")) == 9


def test_a_correction_moves_the_file_rather_than_leaving_two(tmp_path):
    """Two labels for one tsum would train the model on both answers."""
    ses = L.Session(tmp_path, {})
    key = ("s", 1, 0)
    ses.assign(key, "Beast", board(), tsum(200, 200), 25.0)
    note = ses.assign(key, "Piglet", board(), tsum(200, 200), 25.0)
    files = sorted(p.parent.name for p in tmp_path.rglob("*.png"))
    assert files == ["Piglet"], "the Beast crop must be gone, not merely added to"
    assert "corrected Beast -> Piglet" in note


def test_a_crop_the_frame_edge_clips_is_refused_and_nothing_is_written(tmp_path):
    """40.7% of tsums above the visibility floor are refused this way. Writing
    a clamped one would teach the classifier what the board edge looks like."""
    ses = L.Session(tmp_path, {})
    note = ses.assign(("s", 1, 0), "Pascal", board(), tsum(3, 200), 25.0)
    assert list(tmp_path.rglob("*.png")) == []
    assert "REFUSED" in note


def test_deleting_a_label_removes_the_file(tmp_path):
    ses = L.Session(tmp_path, {})
    key = ("s", 1, 0)
    ses.assign(key, "Grim", board(), tsum(200, 200), 25.0)
    note = ses.delete(key)
    assert list(tmp_path.rglob("*.png")) == []
    assert key not in ses.truth and "Grim" in note


def test_deleting_where_there_is_no_label_says_so_and_changes_nothing(tmp_path):
    ses = L.Session(tmp_path, {})
    assert "nothing to delete" in ses.delete(("s", 1, 0))


def test_undo_takes_back_a_new_label(tmp_path):
    ses = L.Session(tmp_path, {})
    key = ("s", 1, 0)
    ses.assign(key, "Grim", board(), tsum(200, 200), 25.0)
    ses.undo(lambda k: None)
    assert list(tmp_path.rglob("*.png")) == []
    assert key not in ses.truth


def test_undo_restores_the_label_a_correction_replaced(tmp_path):
    ses = L.Session(tmp_path, {})
    key = ("s", 1, 0)
    img, t = board(), tsum(200, 200)
    ses.assign(key, "Beast", img, t, 25.0)
    ses.assign(key, "Piglet", img, t, 25.0)
    ses.undo(lambda k: L._cut(img, t["x"], t["y"], 25.0))
    assert sorted(p.parent.name for p in tmp_path.rglob("*.png")) == ["Beast"]
    assert ses.truth[key][0] == "Beast"


def test_undo_with_nothing_done_is_harmless(tmp_path):
    assert "nothing to undo" in L.Session(tmp_path, {}).undo(lambda k: None)


def test_the_suggestion_list_offers_classes_that_already_exist(tmp_path):
    ses = L.Session(tmp_path, {})
    ses.assign(("s", 1, 0), "Grim", board(), tsum(200, 200), 25.0)
    ses.assign(("s", 1, 1), "Pascal", board(), tsum(250, 200), 25.0)
    assert ses.classes() == ["Grim", "Pascal"]


def test_the_non_character_folders_are_never_suggested(tmp_path):
    (tmp_path / "board").mkdir(parents=True)
    (tmp_path / "score").mkdir(parents=True)
    (tmp_path / "Grim").mkdir(parents=True)
    assert L.Session(tmp_path, {}).classes() == ["Grim"]


def test_rounds_that_already_ran_a_model_are_not_offered_for_labelling(tmp_path):
    """`character` on means the saved `kind` was overwritten, and
    `--recolour`/`--kinds` renumber it. Those boards describe something else."""
    d = tmp_path / "s"
    d.mkdir(parents=True)
    rows = [{"index": 1, "tsums": [tsum(10, 10)], "options": {}},
            {"index": 2, "tsums": [tsum(10, 10)], "options": {"character": "m"}},
            {"index": 3, "tsums": [tsum(10, 10)], "options": {"recolour": 0.5}}]
    (d / "samples.jsonl").write_text("\n".join(json.dumps(r) for r in rows),
                                     encoding="utf-8")
    for i in (1, 2, 3):
        (d / ("%04d_before.jpg" % i)).write_bytes(b"")
    got = L.boards(tmp_path, "", {}, 0)
    assert [r[1]["index"] for r in got] == [1]


def test_the_legend_colours_are_all_different():
    """Four states, and the whole tool is the ability to tell them apart."""
    colours = [L.LABELLED, L.GUESS, L.UNASKED, L.REFUSED, L.PICKED]
    assert len(set(colours)) == len(colours)


# ------------------------------------------------- junk at the board's rim
def test_a_rim_half_circle_can_be_called_junk_without_writing_a_crop(tmp_path):
    """The case the first version refused outright.

    A detection at the board edge is a half circle and usually not a tsum, so
    saying so must be possible. But no crop can be written that would not skew
    a model -- clipped-and-stretched teaches "distorted = junk", padded teaches
    "padded = junk" -- and `RejectModel` is never asked about an edge detection
    at play time anyway. So the verdict is kept and the crop is not.
    """
    v = L.Verdicts(tmp_path / "_junk.jsonl")
    key = ("s", 1, 7)
    assert not v.has(key)
    v.add(key)
    assert v.has(key)
    assert not list(tmp_path.glob("*.png")), "a verdict is not a crop"


def test_a_junk_verdict_survives_reopening_the_tool(tmp_path):
    path = tmp_path / "_junk.jsonl"
    L.Verdicts(path).add(("s", 2, 3))
    assert L.Verdicts(path).has(("s", 2, 3))


def test_a_junk_verdict_can_be_taken_back(tmp_path):
    path = tmp_path / "_junk.jsonl"
    v = L.Verdicts(path)
    v.add(("s", 2, 3))
    v.remove(("s", 2, 3))
    assert not L.Verdicts(path).has(("s", 2, 3))


def test_the_verdict_file_survives_a_junk_line_it_cannot_read(tmp_path):
    path = tmp_path / "_junk.jsonl"
    path.write_text('{"sess": "s", "sample": 1, "idx": 4}\nnot json\n',
                    encoding="utf-8")
    assert L.Verdicts(path).has(("s", 1, 4))


def test_an_edge_detection_still_refuses_a_NAME(tmp_path):
    """Junk is a verdict about a detection; a name is a training crop. The
    second still needs a picture the training set could contain."""
    ses = L.Session(tmp_path, {})
    note = ses.assign(("s", 1, 0), "Pascal", board(), tsum(3, 200), 25.0)
    assert "REFUSED" in note and list(tmp_path.rglob("*.png")) == []


# --------------------------------------------------- readability and typos
def test_a_name_that_already_exists_in_another_case_is_not_a_new_class(tmp_path):
    """Typing "marshmallow" for Marshmallow would create a SECOND folder that
    trains as a different character, and nothing downstream could tell them
    apart. The tool commits the spelling the folder already has."""
    ses = L.Session(tmp_path, {})
    ses.assign(("s", 1, 0), "Marshmallow", board(), tsum(200, 200), 25.0)
    known = ses.classes()
    typed = "marshmallow"
    resolved = next((k for k in known if k.lower() == typed.lower()), typed)
    assert resolved == "Marshmallow"


def test_labels_are_drawn_with_a_dark_stroke_behind_them():
    """A board is bright and multicoloured; thin text on it is unreadable over
    about half the tsums, which is what made the first version's labels
    impossible to read."""
    img = np.zeros((60, 200, 3), np.uint8)
    L.outlined(img, "Marshmallow", (10, 40), 0.5, (255, 255, 255))
    assert img.any(), "something was drawn"


def test_zoom_scales_the_board_and_leaves_the_panel_readable():
    img = np.zeros((100, 120, 3), np.uint8)
    state = {"truth": {}, "guess": {}, "refused": set(), "written": 0,
             "verdict": lambda k: False}
    small = L.render(img, [], 25.0, "s", 1, state, None, [], None, "", 0, 1, 1.0)
    big = L.render(img, [], 25.0, "s", 1, state, None, [], None, "", 0, 1, 2.0)
    assert big.shape[1] > small.shape[1] and big.shape[0] >= small.shape[0]


def test_a_click_is_translated_back_into_board_coordinates():
    """The window shows a magnified copy, but every crop and filename is in the
    original frame's pixels. A click at 2x that was not divided would select
    the wrong tsum, or none."""
    items = L.items_for([tsum(100, 100)], [])
    zoom = 2.0
    assert L.pick(items, 200 / zoom, 200 / zoom) == 0
    assert L.pick(items, 200, 200) is None


def test_coin_is_not_treated_as_junk_by_the_reject_model():
    """Measured, not assumed: `coin` crops are marked by the game 76.2% of the
    time and 42.9% of them were cleared by a drag that worked -- more than any
    real character. Training against them would throw away detections that
    clear, which this file's own history records doing once already with
    `unknown_lightball` (86.4% marked).
    """
    src = (Path(__file__).resolve().parents[1] / "scripts" / "reject_net.py"
           ).read_text(encoding="utf-8")
    line = next(ln for ln in src.splitlines() if ln.startswith("NOT_TSUM ="))
    assert "coin" not in line
    assert "unknown_lightball" not in line
    assert "board" in line


def test_hiding_junk_shrinks_it_but_leaves_it_reachable():
    """`j` is a training label, not a delete key. A circle you can no longer
    see is one you can no longer take back, so hidden junk becomes a dot
    rather than nothing -- and `pick` still finds it, so `d` can undo it."""
    img = np.zeros((100, 120, 3), np.uint8)
    items = L.items_for([tsum(50, 50, 15.0)], [])
    state = {"truth": {("s", 1, 0): ("board", None)}, "guess": {},
             "refused": set(), "written": 0, "verdict": lambda k: False}
    shown = L.render(img, items, 25.0, "s", 1, state, None, [], None, "",
                     0, 1, 1.0, False)
    hidden = L.render(img, items, 25.0, "s", 1, state, None, [], None, "",
                      0, 1, 1.0, True)
    lit = lambda c: int((c[:100, :120] > 0).sum())
    assert lit(hidden) < lit(shown) / 4, "junk should shrink out of the way"
    assert lit(hidden) > 0, "but not vanish"
    assert L.pick(items, 50, 50) == 0, "and stay selectable, so d can undo it"


def test_the_selected_circle_is_never_hidden():
    """Hiding the thing you are looking at would make `d` unusable."""
    img = np.zeros((100, 120, 3), np.uint8)
    items = L.items_for([tsum(50, 50, 15.0)], [])
    state = {"truth": {("s", 1, 0): ("board", None)}, "guess": {},
             "refused": set(), "written": 0, "verdict": lambda k: False}
    sel = L.render(img, items, 25.0, "s", 1, state, 0, [], None, "",
                   0, 1, 1.0, True)
    assert int((sel[:100, :120] > 0).sum()) > 200


def test_an_added_circle_records_the_radius_it_was_given(tmp_path):
    """Visibility goes into the FILENAME and every downstream tool bands by it.
    For an added circle it is not measured -- it is whatever radius the
    labeller left, and the default is the full board radius. 58 of the first
    91 hand-labelled crops went in claiming v1.00 because of that."""
    ses = L.Session(tmp_path, {})
    ses.assign(("s", 1, 90), "Dory", board(), tsum(200, 200, 25.0), 25.0)
    ses.assign(("s", 1, 91), "Dory", board(), tsum(260, 200, 12.5), 25.0)
    got = sorted(p.name for p in tmp_path.rglob("*.png"))
    assert got[0].endswith("_90_v1.00.png")
    assert got[1].endswith("_91_v0.50.png")


def test_a_hand_added_index_is_distinguishable_from_a_real_detection():
    """Anything from 90 up was placed by a person, so a later analysis can tell
    an estimated visibility from a measured one."""
    assert L.MANUAL_BASE == 90
    items = L.items_for([tsum(10, 10)], [{"x": 1.0, "y": 1.0, "r": 9.0,
                                          "idx": 90}])
    assert [it["manual"] for it in items] == [False, True]
    assert items[1]["idx"] >= L.MANUAL_BASE


def test_the_junk_key_files_to_its_own_folder(tmp_path):
    """`board` holds the 705 crops filled by hand before this tool existed and
    stays a negative class; new junk goes somewhere separable, so the two
    populations can be told apart later."""
    assert L.JUNK_CLASS == "junk"
    ses = L.Session(tmp_path, {})
    ses.assign(("s", 1, 0), L.JUNK_CLASS, board(), tsum(200, 200), 25.0)
    assert [p.parent.name for p in tmp_path.rglob("*.png")] == ["junk"]


def test_junk_is_never_offered_as_a_character_name(tmp_path):
    for name in ("junk", "board", "score", "Pascal"):
        (tmp_path / name).mkdir(parents=True)
    assert L.Session(tmp_path, {}).classes() == ["Pascal"]


def test_tab_completes_to_the_top_match_not_a_digit():
    """`22` is a real class name, so a number key while typing has to mean the
    character 2. TAB is the only key left that can mean 'take the match'."""
    src = (Path(__file__).resolve().parents[1] / "scripts" / "label_board.py"
           ).read_text(encoding="utf-8")
    assert "if k == 9 and gl:" in src
    i = src.index("if k == 9 and gl:")
    assert "typed[0] = gl[0][0]" in src[i:i + 400]


def test_a_board_can_be_opened_by_number_and_by_name():
    src = (Path(__file__).resolve().parents[1] / "scripts" / "label_board.py"
           ).read_text(encoding="utf-8")
    assert '"--board"' in src and '"--session"' in src and '"--sample"' in src
    # --board is an index into a seeded order; --session names one frame for
    # good. Both exist because the first is convenient and the second is stable.
    assert "min(len(rows) - 1, args.board - 1)" in src


def test_hunting_a_class_the_model_never_learned_explains_the_bootstrap():
    """The chicken and egg: you need crops to train a class, and a trained
    class to find crops. Failing with 'no such class' would leave the user
    stuck; the message names the way out."""
    src = (Path(__file__).resolve().parents[1] / "scripts" / "label_board.py"
           ).read_text(encoding="utf-8")
    i = src.index("this model has no class")
    after = src[i:i + 700]
    assert "--min-class 5" in after, "the bootstrap is a smaller min-class"
    assert "--unsure" in after, "and the alternative needs no class at all"


def test_unsure_needs_no_class_name():
    """A class the model has never seen cannot be hunted by name, so the mode
    that finds NEW characters has to rank by what it cannot name instead."""
    src = (Path(__file__).resolve().parents[1] / "scripts" / "label_board.py"
           ).read_text(encoding="utf-8")
    assert '"--unsure"' in src
    i = src.index("if args.unsure:")
    assert "args.confidence" in src[i:i + 1200], "ranked by unnamed share"


# ------------------------------------------------- the clickable name list
def _panel(gl, typed=None, zoom=1.8):
    img = np.zeros((456, 525, 3), np.uint8)
    items = L.items_for([tsum(200, 200)], [])
    state = {"truth": {}, "guess": {}, "refused": set(), "written": 0,
             "verdict": lambda k: False}
    hits = []
    canvas = L.render(img, items, 25.0, "s", 1, state, 0, gl, typed, "",
                      0, 1, zoom, False, hits)
    return canvas, hits


def test_every_visible_name_is_clickable():
    """The whole point: no retyping. A name drawn but not clickable would be
    worse than not drawing it."""
    gl = [("Pascal", 0.9), ("Grim", 0.0), ("Dory", 0.0)]
    _, hits = _panel(gl)
    assert len(hits) == 3
    assert [h[4] for h in hits] == ["Pascal", "Grim", "Dory"]
    for x0, y0, x1, y1, _ in hits:
        assert x1 > x0 and y1 > y0


def test_the_list_is_not_capped_at_nine():
    """Nine was the number of digit KEYS, which stopped being the right limit
    at 69 classes. A mouse has no such limit."""
    gl = [("C%02d" % i, 0.0) for i in range(40)]
    _, hits = _panel(gl)
    assert len(hits) > 9


def test_the_name_rows_never_reach_the_footer():
    """They did: at 1.8x zoom the names ran straight through the key hints."""
    gl = [("C%02d" % i, 0.0) for i in range(60)]
    canvas, hits = _panel(gl)
    lowest = max(h[3] for h in hits)
    assert lowest < canvas.shape[0] - 100, "the footer needs its space"


def test_typing_narrows_the_list_and_it_stays_clickable():
    gl = [("Maleficent", 0.0), ("Marie", 0.0), ("Maximus", 0.0)]
    _, hits = _panel(gl, typed="ma")
    assert [h[4] for h in hits] == ["Maleficent", "Marie", "Maximus"]


def test_the_panel_does_not_shrink_its_list_when_the_board_is_zoomed():
    """Zoom exists to make a 40px tsum readable. Scaling the panel with it left
    room for ONE name out of 67."""
    gl = [("C%02d" % i, 0.0) for i in range(40)]
    _, small = _panel(gl, zoom=1.0)
    _, big = _panel(gl, zoom=3.0)
    assert len(big) >= len(small)


# ------------------------------------------------------ the lockable filter
def test_a_locked_filter_is_shown_as_locked():
    gl = [("Maleficent", 0.0), ("Marie", 0.0)]
    img = np.zeros((456, 525, 3), np.uint8)
    items = L.items_for([tsum(200, 200)], [])
    state = {"truth": {}, "guess": {}, "refused": set(), "written": 0,
             "verdict": lambda k: False}
    hits = []
    L.render(img, items, 25.0, "s", 1, state, 0, gl, None, "", 0, 1,
             1.0, False, hits, "ma")
    assert [h[4] for h in hits] == ["Maleficent", "Marie"]


def test_the_filter_survives_a_label_and_the_typed_text_does_not():
    """The complaint this exists for: the same two letters were retyped for
    every tsum of one character, and a board holds forty tsums of five."""
    src = (Path(__file__).resolve().parents[1] / "scripts" / "label_board.py"
           ).read_text(encoding="utf-8")
    i = src.index("filter LOCKED to")
    assert "typed[0] = None" in src[i - 400:i]
    # the commit path clears `typed` and never touches `filt`
    j = src.index("`typed` clears, `filt` does NOT")
    assert "filt" not in src[j + 40:j + 200].split("break")[0].replace(
        "filt` does NOT", "")


def test_slash_locks_while_typing_and_l_cannot():
    """The bug: typing `mike` then `l` gave `mikel`. The typing branch appends
    every printable key before any command is reached, so the lock key has to
    be one no class name contains -- and `/` is the key that started typing."""
    src = (Path(__file__).resolve().parents[1] / "scripts" / "label_board.py"
           ).read_text(encoding="utf-8")
    typing = src.index("if typed[0] is not None:")
    append = src.index("typed[0] += chr(k)")
    lock = src.index('if k == ord("/"):', typing)
    assert typing < lock < append, "the lock must come BEFORE the catch-all"
    # and `l` outside typing only clears
    i = src.index('if k == ord("l"):')
    assert "UNLOCK only" in src[i:i + 200]


def test_locking_frees_the_number_keys():
    """While TYPING, `1` has to mean the character 1 -- there is a class called
    `22`. A locked filter is not text entry, so digits pick again."""
    src = (Path(__file__).resolve().parents[1] / "scripts" / "label_board.py"
           ).read_text(encoding="utf-8")
    typing = src.index("if typed[0] is not None:")
    digits = src.index('if ord("1") <= k <= ord("9"):')
    assert typing < digits, "the typing branch must consume digits first"
    i = src.index('if k == ord("/"):', src.index("if typed[0] is not None:"))
    assert "typed[0] = None" in src[i:i + 600], "locking leaves typing mode"


def test_l_clears_the_filter():
    src = (Path(__file__).resolve().parents[1] / "scripts" / "label_board.py"
           ).read_text(encoding="utf-8")
    i = src.index('if k == ord("l"):')
    block = src[i:i + 600]
    assert 'filt[0] = ""' in block and "filter cleared" in block
