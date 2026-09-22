"""Put a candidate model into service, or refuse it and say why.

Copying a `.onnx` over the shipped one is two commands and no thought, which is
exactly the problem: the checks that make a new model trustworthy are the ones
easiest to skip when you are pleased with a number. So the copy lives behind
them.

    python scripts/promote.py models/character_candidate.onnx
    python scripts/promote.py models/reject_candidate.onnx --to models/reject.onnx

What it refuses, and why each one has cost this project something:

* **A crop profile this build does not have.** A model trained on padded crops
  and served unpadded ones does not error, it just reads as a worse model.
* **Training on the golden set.** Then its score on that set is memory, and
  every comparison against every other candidate is meaningless.
* **A drop against the model in service**, on the frozen set, on identical
  boards. Not the candidate's own held-out split -- two models split
  differently are scored on different boards, and this project once read 97.9%
  against 94.9% for two models that were exactly as good.

It always writes `<name>.prev.onnx` and `<name>.prev.json` first, so going back
is one command.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from recog_eval import load, per_class, probabilities  # noqa: E402
from ttheart_sender.game import crop as crop_rules  # noqa: E402

GOLDEN = Path("models/golden.json")
#: How much worse a candidate may be and still be promoted. Zero: a candidate
#: that is not at least as good has no argument for replacing what works.
TOLERANCE = 0.0
#: How much held-out AUC a reject candidate may give up before it is refused.
#: Not zero, because `reject_net.py` re-splits by session every run, so two
#: trainings of the same data land a little apart. 0.01 is well inside that and
#: far below the 0.056 drop this was written after -- a retrain on 1,347 new
#: `junk` crops fell from 0.9664 to 0.9105 and went into service without a
#: word, because this branch printed the number and compared nothing.
#:
#: WHAT THIS CHECK CANNOT SEE, and it matters: AUC is measured on whatever
#: negative set the candidate was trained against. Change which crops count as
#: negatives -- `--negative-min-visible` does exactly that -- and the test set
#: changes with it, so two AUCs are not strictly comparable. A candidate that
#: dropped every hard negative would score beautifully here. The number to read
#: beside it is `real tsums lost`, printed by `reject_net.py`: the positives do
#: not move, so that one IS comparable, and it is the number a board filter has
#: lost rounds on before.
AUC_TOLERANCE = 0.01


def score(model: Path, sessions: set, crops: Path):
    """(top-1, macro F1, n) on the given sessions, or None if it cannot be."""
    import cv2
    meta = json.loads(model.with_suffix(".json").read_text(encoding="utf-8"))
    names = list(meta["classes"])
    trained = set(meta.get("train_sessions", []))
    if sessions & trained:
        return None, len(sessions & trained), meta
    size = int(meta.get("size", 96))
    mean = np.asarray(meta.get("mean", [0.485, 0.456, 0.406]), np.float32)
    std = np.asarray(meta.get("std", [0.229, 0.224, 0.225]), np.float32)
    paths, labels, _, _ = load(crops, sessions)
    index = {n: i for i, n in enumerate(names)}
    keep = [i for i, n in enumerate(labels) if n in index]
    if not keep:
        return None, 0, meta
    net = cv2.dnn.readNetFromONNX(str(model))
    from ttheart_sender.game.tsum import _warm_up
    _warm_up(net, size)
    prob, _ = probabilities(net, [paths[i] for i in keep], size, mean, std)
    y = np.asarray([index[labels[i]] for i in keep])
    pred = prob.argmax(1)
    _, macro = per_class(y, pred, names)
    return (float((pred == y).mean()), macro, len(y)), 0, meta


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("candidate", type=Path)
    ap.add_argument("--to", type=Path,
                    help="the model in service. Defaults to the candidate's "
                         "name with _candidate stripped")
    ap.add_argument("--golden", type=Path, default=GOLDEN)
    ap.add_argument("--crops", type=Path, default=Path("crops/labelled"))
    ap.add_argument("--dry-run", action="store_true",
                    help="run every check and change nothing. A promotion tool "
                         "without this invites exactly the accident it exists "
                         "to prevent -- the first test run of this script "
                         "swapped a shipped model while merely being tried")
    ap.add_argument("--force", action="store_true",
                    help="promote despite a failed check. Prints which one, "
                         "so the reason is on the record")
    args = ap.parse_args()

    cand = args.candidate
    live = args.to or cand.with_name(
        cand.name.replace("_candidate", "").replace("_new", ""))
    if not cand.exists():
        print("no such candidate: %s" % cand)
        return 2
    print("candidate : %s" % cand)
    print("in service: %s%s" % (live, "" if live.exists() else "  (none yet)"))

    fails = []
    meta = json.loads(cand.with_suffix(".json").read_text(encoding="utf-8"))

    # 1 -- the crop rule it was trained under must be one this build has.
    try:
        prof = crop_rules.profile(meta.get("crop_profile"))
        print("\n  crop profile : %s  OK" % prof.name)
    except ValueError as exc:
        fails.append(str(exc))
        print("\n  crop profile : FAIL -- %s" % exc)

    # 2 and 3 -- the golden set, and the comparison on it.
    if not args.golden.exists():
        fails.append("no golden set at %s -- freeze one first" % args.golden)
        print("  golden set   : FAIL -- none frozen")
    else:
        sessions = set(json.loads(args.golden.read_text(encoding="utf-8"))["sessions"])
        got, leaked, _ = score(cand, sessions, args.crops)
        # A reject model answers "is this a tsum at all", so scoring it on the
        # golden CHARACTER crops measures nothing -- its classes are not
        # character names. The LEAK check still applies: nothing may train on
        # those sessions, whatever question it answers. Its own held-out AUC is
        # printed by `reject_net.py` and is the number to read for it.
        is_reject = "not_tsum_classes" in meta or "held_out_auc" in meta
        if leaked:
            fails.append("the candidate trained on %d golden session(s)" % leaked)
            print("  golden leak  : FAIL -- trained on %d of them" % leaked)
        elif is_reject:
            print("  golden score : n/a -- this answers 'is it a tsum at all',"
                  "\n                 so the character crops cannot score it.")
            # ITS OWN NUMBER, COMPARED. The golden set cannot score a reject
            # model, and for a while that meant NOTHING scored it: this branch
            # printed the AUC and promoted whatever it was handed. A retrain on
            # 1,347 new `junk` crops fell 0.9664 -> 0.9105, took the fake-catch
            # rate at the shipped floor from 82.8% to 48.3% and the real-tsum
            # loss from 2.9% to 4.5%, and went into service silently.
            auc = meta.get("held_out_auc")
            print("  held-out AUC : %s (from reject_net.py)"
                  % ("%.4f" % auc if isinstance(auc, (int, float)) else "?"))
            base_auc = None
            if live.exists() and live.with_suffix(".json").exists():
                base_meta = json.loads(
                    live.with_suffix(".json").read_text(encoding="utf-8"))
                base_auc = base_meta.get("held_out_auc")
            if not isinstance(auc, (int, float)):
                fails.append("the candidate reports no held-out AUC")
                print("  comparison   : FAIL -- nothing to compare")
            elif not isinstance(base_auc, (int, float)):
                print("  in service   : reports no AUC"
                      "\n                 -- nothing to compare against, so "
                      "this promotion is unmeasured")
            else:
                print("  in service   : %.4f" % base_auc)
                print("  change       : %+.4f" % (auc - base_auc))
                if auc < base_auc - AUC_TOLERANCE:
                    fails.append("the candidate has a worse held-out AUC "
                                 "(%.4f against %.4f)" % (auc, base_auc))
                    print("  comparison   : FAIL -- not better")
                else:
                    print("  comparison   : OK")
        elif got is None:
            fails.append("the candidate scores nothing on the golden set")
            print("  golden score : FAIL -- no scoreable crops")
        else:
            print("  golden score : top-1 %.2f%%   macro F1 %.2f%%   on %d crops"
                  % (100 * got[0], 100 * got[1], got[2]))
            base = None
            if live.exists():
                base, base_leak, _ = score(live, sessions, args.crops)
                if base is None:
                    print("  in service   : cannot be scored on the golden set"
                          + (" (it trained on %d of them)" % base_leak
                             if base_leak else "")
                          + "\n                 -- nothing to compare against, "
                            "so this promotion is unmeasured")
                else:
                    print("  in service   : top-1 %.2f%%   macro F1 %.2f%%"
                          % (100 * base[0], 100 * base[1]))
                    d1, df = got[0] - base[0], got[1] - base[1]
                    print("  change       : top-1 %+.2fpp   macro F1 %+.2fpp"
                          % (100 * d1, 100 * df))
                    if d1 < -TOLERANCE or df < -TOLERANCE:
                        fails.append("the candidate is worse on the golden set")
                        print("  comparison   : FAIL -- not better")
                    else:
                        print("  comparison   : OK")

    if fails and not args.force:
        print("\nREFUSED:")
        for f in fails:
            print("  - %s" % f)
        print("\nNothing was changed. Pass --force to promote anyway.")
        return 1
    if fails:
        print("\nFORCED past %d failed check(s):" % len(fails))
        for f in fails:
            print("  - %s" % f)

    if args.dry_run:
        print("\nDRY RUN -- every check above ran, nothing was changed.")
        return 0

    if live.exists():
        for suffix in (".onnx", ".json"):
            src = live.with_suffix(suffix)
            if src.exists():
                shutil.copy2(src, live.with_suffix(".prev" + suffix))
        print("\nbacked up the model in service to %s"
              % live.with_suffix(".prev.onnx").name)
    for suffix in (".onnx", ".json"):
        src = cand.with_suffix(suffix)
        if src.exists():
            shutil.copy2(src, live.with_suffix(suffix))
    print("promoted %s -> %s" % (cand.name, live.name))
    print("to undo: copy %s back over %s"
          % (live.with_suffix(".prev.onnx").name, live.name))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
