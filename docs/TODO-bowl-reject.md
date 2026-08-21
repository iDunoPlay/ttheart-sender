# TODO: decide whether `--bowl-reject` becomes the default

Status: **implemented, opt-in, enabled in `flows/play.yaml` at 40, live A/B
done.** A played round preferred 40 clearly over 60 — see
[The round settled it](#the-round-settled-it).

`board_colours()` plus a filter at the end of `detect()` in
`ttheart_sender/game/tsum.py`. The code default is `0` — off — so removing
`bowl_reject:` from the flow is the entire revert.

This restores work that was measured under the `--kinds` investigation and then
removed with it in v1.7.4 (`c69a314`, "Reverted Play Style"). The revert was
correct about the identity half and collateral about this half: the two are
independent, and only the identity half was the unsettled one.

## The problem it addresses

Detection's two error rates are not equal, and the worse one is precision:

| | reading |
|---|---|
| precision | **0.675** — 31% of what detection reports is not a tsum |
| recall | 0.874 — 13% of real tsums are never seen |

A phantom is not a harmless extra row in a list. It is a chain member the
cursor walks to and the game does not clear, so a phantom in the middle of a
proposed chain is a drag that accomplishes nothing.

## The signal

A detection that landed on the bowl instead of on a tsum carries the bowl's
colour, and `board_colours()` already knows what that is — the crop's border
ring is board on every real frame, so assigning just those pixels to the
already-fitted palette reproduces `_background_clusters` exactly (10 of 10
labelled boards) for ~2.5ms, without a second quantisation.

Lab distance from that colour, per detection:

| | p10 | median | p90 |
|---|---:|---:|---:|
| real tsums (422) | 74 | **173** | 212 |
| phantoms (190) | 38 | **76** | 143 |

Overlapping, so the cutoff buys phantoms with real tsums rather than for free.
The exchange rate is what makes it worth taking. Through `eval`:

| `--bowl-reject` | precision | recall | f1 |
|---|---:|---:|---:|
| 0 (off) | 0.675 | 0.874 | 0.762 |
| 40 | 0.734 | 0.844 | 0.785 |
| 50 | 0.746 | 0.830 | 0.786 |
| **60** | **0.769** | 0.814 | **0.791** |
| 70 | 0.785 | 0.774 | 0.779 |
| 80 | 0.805 | 0.728 | 0.765 |

40–60 is a plateau rather than a tuned edge. Re-runnable:

```
python -m ttheart_sender.game.tsum eval --dir scratchpad --sweep bowl_reject=0,40,50,60,70,80
```

## One feature, not a classifier — measured

The obvious next move is more per-detection features, and it does not work.
Four were scored over the same 612 detections (bowl distance, the area of the
colour component the peak sits in, centre-vs-annulus colour contrast, and peak
depth), combined with logistic regression and validated leave-one-board-out:

| | AUC |
|---|---:|
| bowl distance alone | **0.869** |
| all four, in-sample | 0.912 |
| all four, leave-one-board-out | **0.865** |

In-sample it looks like a gain and held out by board it is not there at all.
Peak depth and colour purity are useless as phantom evidence in their own right
(AUC 0.42 and 0.49 — phantoms are, if anything, *deeper* than real tsums).
Bowl distance already captures what per-detection colour evidence has to offer;
the remaining precision has to come from somewhere else.

## What it does not touch

Recall. 13% of real tsums are never seen, and that is the ceiling on chain
length — `play_loop`'s own comment: *"chain length is set by detection recall,
not by the search: a tsum missed in the middle of a run of six splits it into a
three and a two."* `bowl_reject` makes that slightly worse, not better, which
is exactly the trade the live A/B has to price.

The knobs that were supposed to reach recall have been swept since and do not:

| sweep | result |
|---|---|
| `floor_frac` × `heal_frac`, 12 combinations | the defaults 0.42/0.9 win; nothing within 0.02 |
| `merge=true` | 0.762 → **0.613** (recall 0.874 → 0.600) |
| `include_dark=true` | 0.759 → 0.762, i.e. noise |
| `scale` 0.75 / 1.5 / 2.0 | 0.622 / 0.659 / 0.625 — all worse |

## The round settled it

Played live, the rule is a clear improvement and **40 is the setting, not 60**.

That is the opposite of what the offline table above prefers, and the gap it
overturns is small: 0.791 f1 against 0.785, which `eval` itself flags as noise
over ten boards. So this is not a measurement being contradicted, it is a
measurement being asked a question it cannot answer. `eval` scores whether a
detection is a tsum. It cannot score whether the tsum that got dropped was
holding a chain together, and at 60 enough of them are that the chains get
shorter without the stalls falling to pay for it.

Worth keeping as a general caution about this repo's offline scores: a
difference `eval` calls noise is not a difference the round has to agree with,
and where they disagree the round wins.

## Still open

**Nothing here prices FEVER.** Every number above comes from normal-play
boards. FEVER repaints the board in neon, so the bowl's own colour moves and
this rule's threshold moves with it; `play_loop` refits the palette on entering
FEVER, but the exchange rate has not been measured there. See
`docs/TODO-fever-detection.md`, which fixes a much larger FEVER problem — a
collapsing radius estimate — and leaves this one open.

## Related

`docs/TODO-fever-detection.md` and `docs/TODO-blob-adjacency.md` are the other
opt-in rules, and they touch different parts of the pipeline. Turn them off one
at a time.
