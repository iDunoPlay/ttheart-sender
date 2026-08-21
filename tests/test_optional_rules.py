"""The two opt-in rules, and the guarantee that leaving them off changes nothing.

`--bowl-reject` (detection) and `--mode blob` (adjacency) were both measured
well offline and neither has been A/B'd over a live round. They ship switched
off for that reason, so the property that actually matters here is not that
they work -- it is that a run which does not ask for them behaves exactly as it
did before they existed. Each rule therefore gets a pair: one test that it does
its job when asked, one that it is inert when not.

The revert path is the switch, so these also pin the switch itself: the option
names have to survive in `play_defaults()`, or `options:` in a flow silently
stops being able to turn either rule off again.
"""

from __future__ import annotations

import numpy as np
import pytest

from ttheart_sender.game import tsum
from ttheart_sender.game.tsum import Tsum, blob_adjacency, board_colours, detect, find_chains


# --------------------------------------------------------------------------
# a board to detect on
# --------------------------------------------------------------------------
def _board(bg=(120, 60, 40)) -> np.ndarray:
    """Three well-separated tsums of one colour on a flat bowl."""
    img = np.full((300, 300, 3), bg, np.uint8)
    for cx in (70, 150, 230):
        tsum._draw_tsum(img, cx, 150, 28, (40, 220, 240))
    return img


# --------------------------------------------------------------------------
# --bowl-reject
# --------------------------------------------------------------------------
def test_bowl_reject_is_off_by_default_and_detection_is_untouched():
    board = _board()
    plain = detect(board)[0]
    explicit = detect(board, bowl_reject=0.0)[0]

    assert len(plain) == len(explicit)
    assert [(t.x, t.y) for t in plain] == [(t.x, t.y) for t in explicit]


def test_bowl_reject_drops_what_sits_on_the_bowl():
    """A cut far above any real separation must empty the board, not thin it.

    Blunt on purpose: it proves the rule is wired to the *colour distance* and
    not to something that happens to correlate with it, which a gentler
    threshold on a synthetic board would not distinguish.
    """
    board = _board()
    assert detect(board)[0], "the fixture has to detect something to begin with"
    assert detect(board, bowl_reject=1e6)[0] == []


def test_board_colours_finds_the_bowl_without_the_label_image():
    """It reads the border ring, which is board on any real crop."""
    board = _board(bg=(200, 30, 30))
    _tsums, _radius, centres = detect(board)

    bowl = board_colours(board, centres)
    assert len(bowl), "the bowl is most of the frame; something must match it"

    # Whatever it picked has to be nearer the actual background colour than to
    # the tsums painted on top of it.
    import cv2
    as_lab = lambda bgr: cv2.cvtColor(  # noqa: E731 - one-liner, used twice
        np.uint8([[list(bgr)]]), cv2.COLOR_BGR2LAB)[0][0].astype(float)
    to_bg = np.linalg.norm(bowl - as_lab((200, 30, 30)), axis=1).min()
    to_face = np.linalg.norm(bowl - as_lab((40, 220, 240)), axis=1).min()
    assert to_bg < to_face


def test_bowl_reject_is_a_play_option_so_a_flow_can_turn_it_off():
    defaults = vars(tsum.play_defaults())
    assert "bowl_reject" in defaults
    assert defaults["bowl_reject"] == 0.0, "the default has to be the old behaviour"


# --------------------------------------------------------------------------
# --mode blob
# --------------------------------------------------------------------------
def _labels(shape, kind_at) -> np.ndarray:
    """A cluster map: `kind_at` paints kind 1, everything else is 0 (board)."""
    labels = np.zeros(shape, np.int32)
    kind_at(labels)
    return labels


def test_blob_links_two_tsums_joined_by_their_own_colour():
    tsums = [Tsum(x=60.0, y=100.0, r=20.0, kind=1, colour=(0, 0, 0)),
             Tsum(x=140.0, y=100.0, r=20.0, kind=1, colour=(0, 0, 0))]
    labels = _labels((200, 200), lambda m: m.__setitem__((slice(80, 121), slice(40, 161)), 1))

    adj = blob_adjacency(labels, tsums, 20.0)
    assert adj[0] == {1} and adj[1] == {0}


def test_blob_refuses_two_tsums_with_bare_board_between_them():
    """Same distance as the test above; only the mask differs."""
    tsums = [Tsum(x=60.0, y=100.0, r=20.0, kind=1, colour=(0, 0, 0)),
             Tsum(x=140.0, y=100.0, r=20.0, kind=1, colour=(0, 0, 0))]
    labels = _labels((200, 200), lambda m: (m.__setitem__((slice(80, 121), slice(40, 76)), 1),
                                            m.__setitem__((slice(80, 121), slice(125, 161)), 1)))

    # The gap is 49px against a 0.9r = 18px dilation from each side, so the two
    # grown blobs still fall short of each other.
    adj = blob_adjacency(labels, tsums, 20.0)
    assert adj[0] == set() and adj[1] == set()


def test_blob_never_links_across_kinds():
    tsums = [Tsum(x=60.0, y=100.0, r=20.0, kind=1, colour=(0, 0, 0)),
             Tsum(x=140.0, y=100.0, r=20.0, kind=2, colour=(0, 0, 0))]
    labels = _labels((200, 200), lambda m: m.__setitem__((slice(80, 121), slice(40, 161)), 1))

    adj = blob_adjacency(labels, tsums, 20.0)
    assert adj[0] == set() and adj[1] == set()


def test_find_chains_ignores_labels_unless_the_mode_asks_for_them():
    """Passing a mask is not enough -- "touch" stays "touch"."""
    tsums = [Tsum(x=40.0 + 45.0 * i, y=100.0, r=22.0, kind=1, colour=(0, 0, 0))
             for i in range(3)]
    labels = _labels((200, 300), lambda m: m.__setitem__((slice(70, 131), slice(20, 151)), 1))

    without = find_chains(tsums, 22.0, link_px=105, block=1.25)
    with_mask = find_chains(tsums, 22.0, link_px=105, block=1.25, labels=labels)
    assert [c.nodes for c in without] == [c.nodes for c in with_mask]


def test_blob_mode_falls_back_to_touch_when_no_mask_is_available():
    """`play_loop` only builds the mask when it has a palette; a frame before
    the first fit has none, and that must not silently produce an empty graph."""
    tsums = [Tsum(x=40.0 + 45.0 * i, y=100.0, r=22.0, kind=1, colour=(0, 0, 0))
             for i in range(3)]

    touch = find_chains(tsums, 22.0, link_px=105, block=1.25, mode="touch")
    blob_no_mask = find_chains(tsums, 22.0, link_px=105, block=1.25, mode="blob")
    assert [c.nodes for c in touch] == [c.nodes for c in blob_no_mask]
    assert blob_no_mask and len(blob_no_mask[0]) == 3


def test_blob_is_a_mode_a_flow_can_select_and_unselect():
    defaults = vars(tsum.play_defaults())
    assert defaults["mode"] == "touch", "the default has to be the old behaviour"

    import argparse
    ap = argparse.ArgumentParser(add_help=False)
    tsum.add_play_args(ap, merge_default=False)
    assert ap.parse_args(["--mode", "blob"]).mode == "blob"
    with pytest.raises(SystemExit):
        ap.parse_args(["--mode", "nonsense"])


# --------------------------------------------------------------------------
# the revert path has to be reachable
# --------------------------------------------------------------------------
def test_every_subcommand_can_print_its_help():
    """`--help` is how anyone finds the switch to turn these off again.

    It was broken before these rules landed: argparse %-formats help strings,
    so the literal "43% of them" in `--verify-hold`'s help parsed as a `% o`
    octal conversion and `play --help` died with a TypeError. Nothing caught it
    because nothing had ever rendered the help.
    """
    import contextlib
    import io

    parser = _root_parser()
    for sub in _subparsers(parser).values():
        with contextlib.redirect_stdout(io.StringIO()):
            sub.format_help()          # raises if a help string is malformed


def test_help_names_both_switches():
    """Whatever else changes, --help has to keep saying how to get back."""
    parser = _root_parser()
    subs = _subparsers(parser)

    play_help = subs["play"].format_help()
    assert "--bowl-reject" in play_help
    assert "blob" in play_help, "the mode has to be discoverable from --help"


def _root_parser():
    """The CLI's own parser, built the way `main()` builds it."""
    import argparse
    import unittest.mock as mock

    captured = {}

    def grab(self, *args, **kwargs):
        # main() builds its parser inline and immediately parses, so the only
        # way to reach the finished object is to intercept that call.
        captured.setdefault("parser", self)
        raise SystemExit(0)

    with mock.patch.object(argparse.ArgumentParser, "parse_args", grab):
        with pytest.raises(SystemExit):
            tsum.main()
    return captured["parser"]


def _subparsers(parser):
    import argparse

    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return action.choices
    raise AssertionError("the CLI has no subcommands any more")
