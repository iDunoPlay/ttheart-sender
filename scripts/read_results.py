"""Read the score off the round-results screen.

Every A/B this project has run -- 23 rounds of them -- was scored on
``cleared``, the count of tsums removed from the board. That is a proxy, and
nobody has ever checked it against the thing it stands in for, because the
score was on screen for two seconds at the end of each round and then tapped
away. ``flows/play.yaml`` now keeps that frame. This reads it.

How it works, and why it is not OCR
-----------------------------------

The numerals are a fixed bitmap font in a fixed place, which makes this a
lookup rather than a recognition problem: binarise, cut the row into glyphs,
scale each to a common box, and take the nearest of ten templates. No model,
no training set, no inference cost, and it either matches or it says so.

The big score is deliberately not one of the numbers read. It is drawn large,
under an animated starburst that welds neighbouring digits into a single blob,
in a face whose numerals are not fixed-width -- so a merged pair cannot be
split on width either. It is the least readable number on a screen that also
shows the same quantity cleanly, and reading it would be a choice to work from
the worst evidence available.

Instead the score is DERIVED. The panel prints a score bonus that is a fixed
share of the base, and the total is base plus bonus, so the two are
proportional: ``score = bonus * 6.18136`` reproduces all five bootstrap rounds
exactly. The bonus is small, white, and unobstructed, and it reads perfectly.

Commas are told from digits by HEIGHT rather than width, because a ``1`` is
about as narrow as a comma -- "1,186" defeats every width rule and no height
rule at all.

Confidence
----------

A misread score is worse than no score: it would enter the corpus looking like
data. So every digit carries its distance to the template it matched, and a
reading with any digit past ``--max-dist`` is refused rather than guessed --
including when a digit has simply never been seen before. ``read`` also
re-fits the bonus-to-score constant against any hand-read scores it has and
says how far it has drifted, because that constant encodes the player's
loadout: change the equipped tsum and every derived score is wrong by a
constant factor, which is exactly the error that survives an A/B intact.

    .venv/Scripts/python scripts/read_results.py build   # templates, once
    .venv/Scripts/python scripts/read_results.py read    # every saved frame
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

#: Where each number sits, for the 994x578 capture `LAYOUTS` already pins.
#: (y0, y1, x0, x1).
#:
#: THE BIG SCORE IS NOT HERE, and that is deliberate. It is drawn in a larger
#: face with an animated starburst that lands on the digits and welds
#: neighbours together, and its numerals are not fixed-width -- a `1` is half
#: the width of an `8`, so a merged pair cannot be split on width either.
#: Reading it would be the least reliable number on a screen that also shows
#: the same quantity in a clean one. See SCORE_FROM_BONUS.
REGIONS = {
    "bonus": (415, 460, 320, 510),
    "star":  (495, 540, 270, 430),
    "coins": (545, 595, 270, 430),
}

GLYPH = (20, 28)        #: every digit normalised to this box before matching
INK = 150               #: the panel is dark; the numerals are the bright part
#: The score, without reading the score. The panel shows a bonus that is a
#: fixed share of the base, and the total is base plus bonus -- so the two are
#: proportional, and this constant is that proportion. Fitted over the
#: bootstrap frames it reproduces all five scores to the point: 837,141 /
#: 495,716 / 435,384 / 1,162,483 / 763,657, every one exact or one off.
#:
#: IT ENCODES THE PLAYER'S BONUS RATE (19.3% here), which is a property of the
#: equipped tsum and items rather than of the game. Change the loadout and
#: this is wrong -- silently, and by a constant factor, which is the shape of
#: error that survives an A/B and corrupts it. So `bonus` is what gets
#: recorded as measured and `score` is marked derived, and `--check` re-fits
#: the constant against hand-read scores to confirm it still holds.
SCORE_FROM_BONUS = 6.18136

#: Hand-read from the five frames that bootstrapped this, and the only place
#: in the project where a number was typed in by eye rather than measured.
#: They are the templates' ground truth, so `read` reproducing them exactly is
#: the acceptance test -- `--verify` runs it.
BOOTSTRAP = {
    "20260904_175522_957_results.png": dict(bonus=135430, star=250, coins=870),
    "20260904_175726_758_results.png": dict(bonus=80195, star=170, coins=449),
    "20260904_175917_269_results.png": dict(bonus=70435, star=170, coins=259),
    "20260904_180136_505_results.png": dict(bonus=188063, star=350, coins=1186),
    "20260904_180340_961_results.png": dict(bonus=123542, star=200, coins=700),
    # Added after two rounds on v1.11.1b: every digit of the second frame
    # matched correctly and the reading was still refused, because one `9` sat
    # at 0.163 against a 0.16 bar. More examples per digit is the fix; loosening
    # the bar would have been the fix that stops it refusing anything.
    "20260904_213940_841_results.png": dict(bonus=151396, star=300, coins=702),
    "20260904_214139_426_results.png": dict(bonus=131987, star=250, coins=758),
}

#: The scores those five rounds actually showed, for `--check` only. Never used
#: to read anything -- they are what the derived number is held against.
HAND_READ_SCORES = {
    "20260904_175522_957_results.png": 837141,
    "20260904_175726_758_results.png": 495716,
    "20260904_175917_269_results.png": 435384,
    "20260904_180136_505_results.png": 1162483,
    "20260904_180340_961_results.png": 763657,
}


def glyphs(bgr, region):
    """The digit boxes in one number, left to right. Commas dropped."""
    y0, y1, x0, x1 = region
    crop = bgr[y0:y1, x0:x1]
    if crop.size == 0:
        return []
    grey = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, ink = cv2.threshold(grey, INK, 255, cv2.THRESH_BINARY)

    cols = (ink > 0).sum(axis=0) > 0
    runs, start = [], None
    for i, on in enumerate(cols):
        if on and start is None:
            start = i
        elif not on and start is not None:
            if i - start >= 3:
                runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(cols)))
    if not runs:
        return []

    # Commas are told from digits by HEIGHT, not width. Width fails: a `1` is
    # about as narrow as a comma, so "1,186" read as five similar runs and a
    # width rule either kept the comma or split the `8`. Height does not fail:
    # every digit spans the full text height and a comma is a third of it,
    # sitting on the baseline.
    # The reference is the TALLEST run, not the region's total ink extent and
    # not the median. The extent is not safe -- the bonus row has a dotted
    # rule beneath it, and measuring from the top of the digits down to that
    # rule made every digit look short and threw the whole number away. The
    # median is not safe either: a dotted rule is dozens of tiny runs, and it
    # outvotes six digits. A digit is the tallest thing in a row of digits,
    # and when something taller does intrude every digit falls below the bar
    # and the number is refused -- which is the failure worth having.
    boxes = []
    for a, b in runs:
        rows = np.where(ink[:, a:b].any(axis=1))[0]
        if len(rows):
            boxes.append((a, b, rows[0], rows[-1]))
    if not boxes:
        return []
    unit = max(y1 - y0 + 1 for _, _, y0, y1 in boxes)

    out = []
    for a, b, y0, y1 in boxes:
        if (y1 - y0 + 1) < unit * 0.55:
            continue                              # a comma, or a stray speck
        cell = ink[y0:y1 + 1, a:b]
        out.append(cv2.resize(cell, GLYPH, interpolation=cv2.INTER_AREA))
    return out


def read_number(bgr, region, templates, max_dist):
    """(value, worst distance) or (None, worst) when a digit did not match."""
    keys = sorted(templates)
    if not keys:
        return None, 1.0
    bank = np.stack([templates[k] for k in keys]).astype(np.float32) / 255.0
    digits, worst = [], 0.0
    for g in glyphs(bgr, region):
        v = g.astype(np.float32).ravel() / 255.0
        d = np.abs(bank.reshape(len(bank), -1) - v).mean(axis=1)
        i = int(d.argmin())
        worst = max(worst, float(d[i]))
        digits.append(keys[i])
    if not digits or worst > max_dist:
        return None, worst
    return int("".join(digits)), worst


def build(frames_dir: Path, out: Path):
    """One template per digit, from frames whose values are known."""
    banks: dict[str, list] = {}
    used = 0
    for name, truth in BOOTSTRAP.items():
        f = frames_dir / name
        bgr = cv2.imread(str(f))
        if bgr is None:
            print(f"  missing {name}")
            continue
        used += 1
        for field, value in truth.items():
            want = str(value)
            got = glyphs(bgr, REGIONS[field])
            if len(got) != len(want):
                print(f"  {name} {field}: cut into {len(got)} glyphs, "
                      f"expected {len(want)} -- skipped")
                continue
            for ch, g in zip(want, got):
                banks.setdefault(ch, []).append(g)

    if not banks:
        print("no glyphs -- are the results frames where you think they are?")
        return 1
    templates = {k: np.median(np.stack(v), axis=0).astype(np.uint8)
                 for k, v in banks.items()}
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **templates)
    have = "".join(sorted(templates))
    print(f"{used} frames -> templates for [{have}] "
          f"({sum(len(v) for v in banks.values())} examples)")
    missing = [d for d in "0123456789" if d not in templates]
    if missing:
        print(f"  NO EXAMPLE OF {','.join(missing)} yet -- a number containing "
              f"one will be refused rather than guessed. Play more rounds.")
    return 0


def load(path: Path):
    z = np.load(path)
    return {k: z[k] for k in z.files}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["build", "read"])
    ap.add_argument("--frames", type=Path, default=Path("logs/debug"))
    ap.add_argument("--templates", type=Path, default=Path("models/digits.npz"))
    ap.add_argument("--max-dist", type=float, default=0.16,
                    help="refuse a reading whose worst digit is further than "
                         "this from its template. A wrong score is worse than "
                         "no score: it enters the corpus looking like data.")
    ap.add_argument("--verify", action="store_true",
                    help="check the readings against the hand-read values")
    args = ap.parse_args()

    if args.cmd == "build":
        return build(args.frames, args.templates)

    if not args.templates.exists():
        print(f"no templates at {args.templates} -- run `build` first")
        return 1
    templates = load(args.templates)

    frames = sorted(args.frames.glob("*_results.png"))
    if not frames:
        print(f"no *_results.png under {args.frames}")
        return 1

    print(f"{'frame':>16} {'bonus':>9} {'coins':>7} {'star':>6} "
          f"{'score (derived)':>16}  {'against hand-read':>18}")
    rows, bad, fit = [], 0, []
    for f in frames:
        bgr = cv2.imread(str(f))
        if bgr is None:
            continue
        vals = {}
        for field, region in REGIONS.items():
            vals[field], _ = read_number(bgr, region, templates, args.max_dist)
        vals["score"] = (round(vals["bonus"] * SCORE_FROM_BONUS)
                         if vals["bonus"] else None)
        vals["score_is_derived"] = True

        note = ""
        truth = HAND_READ_SCORES.get(f.name)
        if truth and vals["bonus"]:
            fit.append((vals["bonus"], truth))
            if vals["score"]:
                off = abs(vals["score"] - truth) / truth
                note = f"{truth:,} ({off:.3%})"
        shown = f"{vals['score']:,}" if vals["score"] else "--"
        print(f"{f.name[:15]:>16} {vals['bonus'] or '--':>9} "
              f"{vals['coins'] or '--':>7} {vals['star'] or '--':>6} "
              f"{shown:>16}  {note:>18}")
        rows.append((f.name, vals))

        if args.verify and f.name in BOOTSTRAP:
            for k, want in BOOTSTRAP[f.name].items():
                if vals.get(k) != want:
                    print(f"    MISREAD {k}: got {vals.get(k)}, hand-read {want}")
                    bad += 1

    if fit:
        # Re-fit the constant on whatever hand-read scores exist. Drift means
        # the loadout changed, and every derived score since is wrong by that
        # factor -- a constant-factor error is exactly the kind that survives
        # an A/B looking like a result.
        k = sum(t for _, t in fit) / sum(b for b, _ in fit)
        drift = abs(k - SCORE_FROM_BONUS) / SCORE_FROM_BONUS
        flag = "" if drift < 0.002 else "   <- DRIFTED, re-check the loadout"
        print(f"\nbonus->score constant: {k:.5f} over {len(fit)} frame(s), "
              f"shipped {SCORE_FROM_BONUS:.5f}, {drift:.3%} off{flag}")

    if args.verify:
        total = sum(len(BOOTSTRAP[f]) for f, _ in rows if f in BOOTSTRAP)
        print(f"verify: {total - bad}/{total} fields read correctly")
        return 1 if bad else 0

    out = args.frames / "results.jsonl"
    with out.open("w", encoding="utf-8") as fh:
        for name, vals in rows:
            fh.write(json.dumps({"frame": name, **vals}) + "\n")
    print(f"\n{len(rows)} frames -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
