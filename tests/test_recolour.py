"""`--recolour`: re-decide identity off the faces instead of off the pixels.

Pixel-level k-means segments the board well and classifies it badly -- one
character lit differently across the board lands in two clusters, and
`adjacency` will not link across a kind difference, so those partners are
unreachable at any distance. `_recolour` samples each face once and merges
agglomeratively, which is the fix already written for that.

It ships off, and the reason is that the offline evidence disagrees with
itself -- see the flag's help and `docs/DATASET-FINDINGS.md`. What is tested
here is not whether it helps, which only a round decides, but the two
properties every opt-in rule in this project owes: it is inert unless asked
for, and when asked for it does not quietly break something else.

The something else is the base tsum, and it is the whole reason this file
exists.
"""

from __future__ import annotations

import numpy as np

from ttheart_sender.game import tsum


def _board(colours, radius=20):
    """One disc per colour, left to right, on a dark ground."""
    img = np.full((80, 80 * len(colours), 3), 20, np.uint8)
    tsums = []
    for i, bgr in enumerate(colours):
        cx, cy = 40 + 80 * i, 40
        img[:] = img
        import cv2
        cv2.circle(img, (cx, cy), radius, bgr, -1)
        tsums.append(tsum.Tsum(x=float(cx), y=float(cy), r=float(radius),
                               kind=i, colour=bgr))
    return img, tsums


def test_it_is_off_by_default():
    assert tsum.play_defaults().recolour == 0.0, "a new rule ships off"


def test_the_flow_action_accepts_it():
    from ttheart_sender.automation import tsum_actions
    assert "recolour" in tsum_actions._TUNABLES


def test_recolour_merges_two_readings_of_one_colour():
    """Two faces the same colour become one kind; a third stays its own."""
    img, tsums = _board([(200, 60, 60), (198, 62, 58), (60, 200, 60)])
    out = tsum._recolour(img, list(tsums), 20.0, 35.0)

    assert out[0].kind == out[1].kind, "near-identical faces must merge"
    assert out[2].kind != out[0].kind, "a clearly different face must not"


def test_the_base_tsum_survives_a_renumbering():
    """The bug this rule would otherwise ship with.

    `read_base_kind` returns an index into the PALETTE centres. `_recolour`
    renumbers `kind` to its own group ids, which have nothing to do with
    k-means cluster numbers -- so the cached `base` would point at an
    arbitrary group, the bot would stop preferring the equipped character,
    and the skill would stop charging with nothing in the log to say why.

    `_base_from_faces` re-derives it from the icon's own Lab colour, which is
    what survives a renumbering.
    """
    import cv2
    red, green = (60, 60, 200), (60, 200, 60)
    img, tsums = _board([red, red, green, green])
    out = tsum._recolour(img, list(tsums), 20.0, 35.0)

    # The icon shows the GREEN character, so the green group is the base --
    # whatever number the regrouping happened to give it.
    icon_lab = cv2.cvtColor(np.uint8([[green]]), cv2.COLOR_BGR2LAB)[0][0]
    base = tsum._base_from_faces(img, out, 20.0, [float(v) for v in icon_lab])

    assert base is not None
    assert base == out[2].kind == out[3].kind
    assert base != out[0].kind, "it must not pick the red group"


def test_no_icon_colour_means_no_guess():
    """A corpus collected with `--no-base` has no icon Lab to match against,
    and inventing one would silently prefer an arbitrary character."""
    img, tsums = _board([(200, 60, 60), (60, 200, 60)])
    assert tsum._base_from_faces(img, tsums, 20.0, None) is None
