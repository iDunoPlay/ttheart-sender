"""One command: label counts, train, score on the frozen set, promote or refuse.

The four steps of a retrain were four commands, each with flags you had to
remember, and getting one wrong was silent. `--min-class` is the example that
prompted this: its default drops every class under 20 crops, so a labelling
session that added Dory (9), Maleficent (6) and Eeyore (5) trained none of
them, and the only sign was one line in the middle of the output.

    python scripts/retrain.py                  # the whole loop
    python scripts/retrain.py --dry-run        # everything except the promotion
    python scripts/retrain.py --min-class 5    # include small, new classes
    python scripts/retrain.py --reject         # the "is this a tsum" model

It runs the same scripts by hand documented in `docs/RETRAINING.md` -- it does
not reimplement them, so anything it can do you can still do a step at a time.

What it will not do for you: promote a model that fails a check. That decision
stays with `promote.py`, which refuses an unknown crop profile, a candidate
trained on the golden set, or one worse than the model in service.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

#: `<session>_<sample>_<index>_v<visible>.png`. The SESSION is what the
#: train/test split is made on, so it is what a class needs a spread of.
NAME = re.compile(
    r"^(?P<sess>.+?)_(?P<sample>\d{4})_(?P<idx>\d{2,})_v(?P<vis>[\d.]+)$")

HERE = Path(__file__).resolve().parent
NOT_CHARACTERS = {"board", "score", "junk"}


def counts(root: Path):
    out = {}
    for d in sorted(root.iterdir()) if root.is_dir() else []:
        if d.is_dir():
            out[d.name] = len(list(d.glob("*.png")))
    return out


def sessions_per_class(root: Path):
    """How many distinct SESSIONS each class has crops from.

    The number that matters more than the crop count, and the one nothing was
    reporting. The split is by session, so a class whose crops all come from
    one round lands entirely on one side of it: trained and never scored, or
    scored having never been trained. The second is how Cleo came to read 0%
    recall on 41 crops it was never shown one example of.

    Twenty crops of the same three tsums also is not twenty examples. Crops
    from one board are near-duplicates -- that is why the split is by session
    in the first place.
    """
    out = defaultdict(set)
    for d in sorted(root.iterdir()) if root.is_dir() else []:
        if not d.is_dir():
            continue
        for p in d.glob("*.png"):
            m = NAME.match(p.stem)
            if m:
                out[d.name].add(m.group("sess"))
    return out


def duplicates(root: Path):
    """Crops that exist under more than one class.

    Moving a crop between folders by hand is the supported way to fix a label:
    `crops.py assign` has always worked that way, and nothing outside the
    folder records a crop's class -- the sidecars key on
    `(session, sample, index)`.

    COPYING one instead is the mistake, and it is silent. The same picture then
    trains as two different characters at once and nothing downstream can tell
    which was meant.
    """
    seen, dupes = {}, {}
    for d in sorted(root.iterdir()) if root.is_dir() else []:
        if not d.is_dir():
            continue
        for p in d.glob("*.png"):
            if p.name in seen:
                dupes.setdefault(p.name, [seen[p.name]]).append(d.name)
            else:
                seen[p.name] = d.name
    return dupes


def survey(root: Path, min_class: int) -> None:
    """What is about to be trained, and what is about to be dropped.

    Printed FIRST and loudly, because the drop is the thing that silently
    undoes an evening of labelling.
    """
    got = counts(root)
    if not got:
        print("no labelled crops under %s" % root)
        return
    keep = {k: v for k, v in got.items() if v >= min_class}
    drop = {k: v for k, v in got.items() if v < min_class}
    chars = [k for k in keep if k not in NOT_CHARACTERS]
    print("labels in %s" % root)
    print("  %d class(es), %d crop(s)" % (len(got), sum(got.values())))
    print("  training %d class(es) at --min-class %d" % (len(chars), min_class))
    dupes = duplicates(root)
    if dupes:
        print("\n  THE SAME CROP IS IN TWO CLASSES -- %d of them." % len(dupes))
        for name, where in list(dupes.items())[:8]:
            print("    %s  in  %s" % (name, " and ".join(where)))
        print("  Moving a crop between folders is fine; COPYING one is not."
              "\n  It will train as both characters at once. Delete the wrong"
              "\n  copy before training.")
    # Negatives are not characters and have no target to reach: `board`,
    # `score` and `junk` train the reject model, where "thin" means
    # nothing. Listing them beside the characters made `score` read as 8
    # crops of homework it is not.
    empty = sorted(k for k, v in got.items() if v == 0)
    thin = {k: v for k, v in drop.items()
            if k not in NOT_CHARACTERS and v > 0}
    neg_thin = {k: v for k, v in drop.items() if k in NOT_CHARACTERS}
    if thin:
        print()
        print('  THIN -- under --min-class %d, so nothing you labelled'
              % min_class)
        print('  for these will be learned. How many more crops each needs:')
        for k, v in sorted(thin.items(), key=lambda kv: -kv[1]):
            print('    %-18s %3d crops   (+%d)' % (k, v, min_class - v))
        print()
        print('    %d MORE CROPS makes every one of them trainable.'
              % sum(min_class - v for v in thin.values()))
        print('    Either label more, or lower --min-class -- a class under')
        print('    about 20 is hard to learn AND hard to score, so the cut')
        print('    is a real trade rather than a formality.')
    if empty:
        print()
        print('  EMPTY folder(s) -- they train nothing, and are usually a')
        print('  name typed before its crops arrived: %s' % ', '.join(empty))
    # A class can clear the crop cut and still be unusable, because 20 crops
    # from one round are 20 near-duplicates. Reported here because nothing
    # reported it before and Cleo cost a retrain to find.
    spread = sessions_per_class(root)
    fragile = sorted((len(spread[k]), got[k], k) for k in got
                     if k not in NOT_CHARACTERS and got[k] >= min_class
                     and len(spread[k]) < 3)
    if fragile:
        print()
        print('  PASSES THE CROP CUT BUT COMES FROM TOO FEW ROUNDS. The split')
        print('  is by session, so a class from one round is trained and never')
        print('  scored, or scored having never been trained:')
        for n_s, n_c, k in fragile:
            print('    %-18s %3d crops from %d session(s)%s'
                  % (k, n_c, n_s,
                     '  <- one round only' if n_s < 2 else ''))
        print('    Label these on a DIFFERENT board, not more of the same one.')

    if neg_thin:
        print()
        print('  Not characters, so no target to reach -- these train the')
        print('  REJECT model instead: %s'
              % ', '.join('%s %d' % kv for kv in sorted(neg_thin.items())))
    print()


def run(cmd, what: str) -> bool:
    print("=" * 72)
    print("%s\n  %s" % (what, " ".join(str(c) for c in cmd[1:])))
    print("=" * 72, flush=True)
    return subprocess.run(cmd).returncode == 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--crops", type=Path, default=Path("crops/labelled"))
    ap.add_argument("--min-class", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--crop-profile", default="plain")
    ap.add_argument("--golden", type=Path, default=Path("models/golden.json"))
    ap.add_argument("--reject", action="store_true",
                    help="retrain the 'is this a tsum at all' model instead of "
                         "the character model")
    ap.add_argument("--dry-run", action="store_true",
                    help="train and score, but stop before promoting")
    ap.add_argument("--survey-only", action="store_true",
                    help="just say what would be trained and dropped")
    args = ap.parse_args()

    live = Path("models/reject.onnx" if args.reject else "models/character.onnx")
    cand = live.with_name(live.stem + "_candidate.onnx")
    py = sys.executable

    survey(args.crops, args.min_class)
    if args.survey_only:
        return 0

    if args.reject:
        train = [py, str(HERE / "reject_net.py"), "--epochs", str(args.epochs),
                 "--onnx", str(cand)]
    else:
        train = [py, str(HERE / "classify.py"), "--epochs", str(args.epochs),
                 "--split", "random", "--min-class", str(args.min_class),
                 "--crop-profile", args.crop_profile,
                 "--golden", str(args.golden), "--onnx", str(cand)]
    if not run(train, "1/3  train a candidate"):
        print("\ntraining failed -- nothing was promoted.")
        return 1

    if not args.reject:
        run([py, str(HERE / "recog_eval.py"), "--model", str(cand),
             "--golden", str(args.golden)],
            "2/3  score it on the FROZEN golden set")
    else:
        print("\n2/3  (the reject model is scored by its own held-out AUC above)")

    promote = [py, str(HERE / "promote.py"), str(cand), "--to", str(live)]
    if args.dry_run:
        promote.append("--dry-run")
    ok = run(promote, "3/3  promotion checks"
             + (" (dry run)" if args.dry_run else ""))
    print()
    if args.dry_run:
        print("Dry run: %s is trained and scored, nothing was promoted."
              % cand.name)
        print("Run again without --dry-run to put it into service.")
    elif ok:
        print("%s is now in service. Undo with:" % live.name)
        print("  copy %s %s" % (live.with_suffix(".prev.onnx").name, live.name))
        print("  copy %s %s" % (live.with_suffix(".prev.json").name,
                                live.with_suffix(".json").name))
    else:
        print("The candidate was REFUSED and nothing changed. It is still at")
        print("  %s" % cand)
    return 0 if ok or args.dry_run else 1


if __name__ == "__main__":
    raise SystemExit(main())
