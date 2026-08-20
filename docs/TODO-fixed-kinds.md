# TODO: decide whether `--kinds 5` becomes the default

Status: **implemented, opt-in, awaiting a live A/B.** Added in
`ttheart_sender/game/tsum.py` as `fixed_kinds()`.

Two named readings, because this is a choice between two ways of deciding what
a character is rather than a dial to tune:

| | how identity is decided |
|---|---|
| `--tsum-mode normal` (default) | the pixel colour clusters as they fall — the original |
| `--tsum-mode color` | every detection re-filed into the five characters a board holds |

On `play` and `analyze`, and as `options: {tsum_mode: color}` in a flow.
`--kinds N` is still there underneath and wins when set, which is how the
four-character board an item leaves behind is reached (`--tsum-mode color
--kinds 4`) and what keeps a flow that already says `kinds: 5` working. The
play log opens with the reading in force, so an A/B's two logs say which arm
each one is. Nothing changes unless you pass it.

## The fact it is built on

A board deals **five characters**, and four once an item has removed one. That
number is not a tuning knob and was not inferred from the labels — it comes
from the game.

It is the calibration `_recolour()` never had. That function samples exactly
the right evidence (one median Lab per face) and merges up to a *distance
threshold*, and its own docstring records why no threshold works: the labels
hold only pairs that DO match, so a positive-only score keeps improving as you
merge more, right up to "everything is one character". Tuned that way it read
80% → 89% of drawn links and was a clear end-to-end regression — chains of 27
and 31 tsums.

Given the count, the threshold is unnecessary: the merge stops at five groups
whatever the distances are, and that runaway cannot happen.

## Face colour does separate the hand

Measured first, before building anything. Taking the hand-marked groups as
given and classifying each tsum to its nearest group centroid:

| measure | reading |
|---|---:|
| classified correctly, groups given | **95.1%** (349/367 over ten boards) |
| scatter within one character | ~7 Lab |
| gap to the closest *other* character | 20–30 Lab |

So colour is sufficient evidence. The only question was ever how to find the
centres without the answer key.

## Fitting five centres directly is worse than doing nothing

| centres | splits | merges | total |
|---|---:|---:|---:|
| pixel clusters, `k=12` (the default today) | 14 | 18 | **32** |
| `kinds=4` | 12 | 40 | 52 |
| **five, fitted directly** | 10 | 38 | **48** |

Detection precision is ~0.675, so a third of what the fit is handed is
phantom. k-means minimises total variance; the phantom mass is real mass, it
buys centres with it, and two genuine characters are folded together to pay.

## Over-provisioning fixes it

Fit `3n` centres, keep the `n` most populous. A character is about a fifth of
a crowded board; a phantom sits wherever its stray pixels are, and thinly.

| centres fitted | splits | merges | total |
|---|---:|---:|---:|
| 5 | 12 | 36 | 48 |
| 10 | 12 | 22 | 34 |
| **15 (3n)** | **9** | **13** | **22** |
| 20 | 12 | 14 | 26 |
| 25 | 14 | 17 | 31 |

14 through 18 all land at 22–25 and all five seeds agree, so `over=3` sits on
a plateau rather than a tuned edge. Confirmed through `eval` itself:

Detection f1 stays identical to three decimals throughout, which is the
control: `kinds` only relabels, so anything that moved it would have been a
bug. (The merge count improves again once the spare seat below is added --
these are the numbers without it.) Filtering on
detection depth was tried as an alternative and is a straight loss (`deep .80`
→ 55 total); phantoms are not shallow.

**This is not the positive-only trap that shelved `_recolour`.** Splits and
merges pull in opposite directions and both improved at once, and `n` is fixed
by the game rather than fitted, so "merge more" is not a move the method can
make.

## The seat a phantom used to take

Population alone is not enough, and this was the first version's one clear
defect. Only 2 of 10 boards came out clean; the usual failure was a single
merge with a single cause. Board 1, by kind:

| kind | detections | real tsums | phantoms | labelled groups inside |
|---:|---:|---:|---:|---|
| **0** | **16** | **0** | **16** | — none — |
| 1 | 14 | 10 | 4 | g2:1, g3:9 |
| 2 | 10 | 8 | 2 | g2:9 |
| 3 | 11 | 9 | 2 | g1:9 |
| 4 | 13 | 13 | 0 | **g0:7, g4:6** |

Kind 0 was 16 detections and every one a phantom. Populous enough to hold a
seat, which left four seats for five characters — so two genuine characters
shared kind 4. **The merge was never a clustering failure; it was a seat
shortage.** Confirmed by measuring the merged pairs themselves: they sat a
median **55.7 Lab apart by face colour**. Characters that far apart are not
being confused with each other. They merged because there was no bucket left.

### The fix: one spare seat, given up to the bowl

Take `n + spare` seats and surrender the most board-coloured ones. Phantoms
are detections that landed on the bowl, so they carry its colour:

| | median Lab to nearest board colour |
|---|---:|
| mostly-phantom clusters | **56.0** |
| real characters | **175.5** |

A fixed threshold cannot use that — the distributions overlap, 10 real
characters sit closer to their board than the worst phantom does, and board 3
inverts entirely (its kind 2 is 100% real at 42.3 Lab). **It does not need
one.** Exactly `spare` seats are dropped every time; only the *ranking*
against this board's own background decides which. Nothing is tuned.

| rule | splits | merges | total |
|---|---:|---:|---:|
| pixel clusters, `k=12` | 14 | 18 | 32 |
| top `n` by population | 8.8 | 13.4 | 22.2 |
| **top `n+1`, drop the most board-like** | **8.4** | **9.4** | **17.8** |
| top `n+2`, drop 2 | 10.0 | 10.2 | 20.2 |
| top `n+3`, drop 3 | 9.0 | 9.0 | 18.0 |

Five seeds averaged; through `eval` it reads **9 splits and 8 merges**.
Every labelled character now gets its own bucket on **8 boards of 10**, up
from 6.

`board_colours()` supplies the background without a second quantisation:
the crop's border ring is always board, so assigning just those pixels to an
already-fitted palette reproduces `_background_clusters` exactly (10 of 10
boards) for ~2.5ms. It needs the palette, which is why `fixed_kinds` takes
one; without it the seats go by population alone and this is all skipped.

### What it trades

With no seat of their own, phantoms are assigned to the nearest real
character rather than quarantined in a bucket of their own. On board 1 the
five characters now each have a kind, but one of them carries 20 phantoms.

`--purity` is the guard already in place for exactly this, and it is on by
default at 35 Lab. Measured against it: it drops **69%** of the redistributed
phantoms, for **13%** of real tsums wrongly dropped. Not free, and the live
figures to watch are the stall count and the clear percentage.

## Whole-tsum colour: what it can and cannot do

Prompted by the observation that a tsum "is really only one colour". Four of
the five characters on `scratchpad/idle_03.png` genuinely are — Piglet pink,
Pooh yellow, Donald white, Eeyore grey, each one solid region in
`--debug-dir`'s `clusters.png`.

**Mickey is the exception, structurally.** His body is black, and so is every
other tsum's outline and every shadow between them: in `clusters.png` they are
one connected red mesh over the whole board. That is what `detect`'s
`include_dark` note already says — treating them as one cluster fuses the
board, discarding them loses Mickey — and it is why "the whole tsum is one
colour" cannot be turned into a segmentation rule.

Which leaves Mickey identified by his *peach face*, sitting close to Donald's
*white face*. On `idle_03` those two merged into one kind of 11 (5 Mickeys, 6
Donalds), **15.4 Lab apart by face** against a within-character scatter of ~7.
By the brightness of their surround they are **138.5** apart.

So the surround does carry what the face cannot — but it is not worth adding
as a descriptor, for two measured reasons:

1. **It does not generalise.** Over the pairs that actually merge, the
   surround is the bigger gap in only 4 of 13. The rest merge from the seat
   shortage above, at face gaps of 55–87 Lab where no extra evidence is
   needed.
2. **It costs accuracy overall.** Nearest-centroid with groups given:
   face-only **94.6%**, face+surround **94.3%**, surround-only 85.0%. Three
   more dimensions dilute the four characters that were already fine.

And with the spare seat in place, `idle_03` separates Mickey without any of
it: kind #1 comes out **5 detections, all 5 of them Mickey**, surround L 76
against Donald's 214. The seat shortage was the whole problem.

Sampling a bigger disc does not implement "whole tsum" either — `t.r` is the
*face* blob's inscribed radius, so even 1.1r never reaches Mickey's body.
Measured at 0.45/0.60/0.75/0.90/1.10r the accuracy is 95.1/95.1/96.5/95.9/94.8,
i.e. flat, and the scatter rises at the top end as neighbours bleed in.

## Looking at the result

```
python -m ttheart_sender.game.tsum analyze BOARD.png --kinds 5        --flat flat.png --flat-distinct
```

The stroke the loop would drag is drawn on every panel, through the same
`draw_chain()` that `draw()` uses — so it is the same overlay, not a lookalike
that drifts. `--flat-no-chain` turns it off. Seeing the path over the flattened
board is what shows whether a chain stays on one character or crosses a
boundary the split got wrong; a chain that leaves its colour is visible
immediately, and a member sitting on bare board is a phantom.

Two views, answering different questions. `--flat-mode paint` (the default)
recolours the **pixels** via `posterise()`: every pixel goes to the nearest
character colour, so each tsum comes out a solid ball of one colour keeping its
real outline. No detections are involved, which makes it the honest picture of
what colour alone can do. `--flat-mode disc` draws a disc per **detection** —
what the play loop actually works from, phantoms and all. `--flat-mode both`
shows them together, and where they disagree is the interesting part.

`--flat-distinct` swaps the characters' own colours for high-contrast ones.
Necessary for the case that matters most: two characters 15 Lab apart repaint
as two pastels nobody can tell apart, and telling them apart is the whole point.

Read the *original* panel too. Neither view can show a tsum that was never
detected, and misses are the bigger error (recall 0.874 against precision
0.675). Look for board that stayed empty.

The painted view is where "the whole tsum is one colour" gets its clearest
answer. Piglet, Pooh, Donald and Eeyore each come out as solid balls. The bowl
and the ink are painted separately, and in the ink you can watch Mickey's body
join up with the mesh of outlines running across the entire board — the same
finding as the section above, but as a picture rather than a paragraph. Give
both sinks one colour and he simply disappears into the bowl, which reads as a
detection failure when it is nothing of the kind; that is why they are two.

`kind_scatter()` is the number behind the picture: mean Lab distance from each
kind's own median. A real character reads near `KIND_SCATTER` (7.0); anything
past double that is holding more than one thing, and both the legend and the
`per kind` line say so. On `idle_03` at `--kinds 5`:

```
per kind: #0:36~44 <- BASE TOO WIDE, #1:5~0, #2:6~0, #3:8~1, #4:7~0
```

Four tight kinds and one bucket at 44. The counts alone (36, 5, 6, 8, 7) only
hint at it; the spread names it. That bucket is where the phantoms go now that
`spare` has taken their seat away — see [What it trades](#what-it-trades).

## Scored on the chain, not the grouping

The splits/merges figures above score how detections are *filed*. What gets
dragged is a chain, so it was worth scoring that directly — at `play.yaml`'s
real settings (`block 1.25`, `link_px 105`, `max_chain 12`) over the ten
labelled boards, with the hand-marked groups as the outside answer:

| | `kinds=0` | `kinds=5` |
|---|---:|---:|
| chains found | 49 | **63** |
| chains that are genuinely one character | 61% | **95%** |
| …after `--purity 35` | 61% | **97%** |
| chain members that are phantoms, as proposed | 23% | 44% |
| phantom share of the members actually **dragged** | 23% | **36%** |

More chains, and far more of them real — but the phantom cost is not paid off.

**Correcting an earlier reading of this table.** The last row was first
computed as "phantoms surviving `--purity`, over the *pre-purity* member
count", which gave 24% against 23% and read as though `--purity` erased the
problem. That denominator is wrong: `--purity` removes members, so dividing
survivors by the original total understates the share of what the cursor
actually walks through. On the honest denominator it is **36% against 23%**,
and `--purity` narrows the gap rather than closing it. `--bowl-reject 60`
takes it to 27% and is the better lever — see [Detection is the weaker
half](#detection-is-the-weaker-half).

### A metric that looked alarming and was circular

The first pass at this scored chains on *colour scatter* — how far a chain's
members sit from each other — and read median **5.0 Lab for `kinds=0` against
18.8 for `kinds=5`**, which looks like a serious regression.

It is not a measurement. With `kinds=0` a chain lives inside a single
pixel-colour cluster, so its members are close in colour *by construction*;
the number is a restatement of how the grouping was built, not evidence about
whether the chain is one character. Against the labels — an answer from
outside the clustering — the ordering reverses completely, 61% to 95%.

Worth recording because the pull was strong: the scatter number is cheap, it
moved a long way, and it pointed at a real mechanism (phantoms joining
character buckets). It was still measuring the wrong thing. `spare=0` versus
`spare=1` scores 18.5 against 18.8 on it, so it does not even separate the
change that actually redistributes the phantoms.

## Detection is the weaker half

Worth stating plainly, because the two halves of the pipeline are easy to
conflate and only one of them is in trouble.

**Detection** finds *where* the tsums are — the `(x, y, r)` list. **Identity**
decides *which* of those are the same character — the `kind`. Over the ten
labelled boards:

| | reading |
|---|---|
| detection precision | **0.690** — 31% of what it reports is not a tsum |
| detection recall | **0.844** — 16% of real tsums are never seen |
| identity, groups given | 95.1% correct |
| chains genuinely one character (`--tsum-mode color`) | 95% |

Identity has a 3x margin to work with (within a character ~7 Lab, between the
closest two 20–30) and uses it. Detection does not, and everything downstream
inherits both failures: a phantom is a chain member the cursor walks to and the
game does not clear, and — `play_loop`'s own comment — *"chain length is set by
detection recall, not by the search: a tsum missed in the middle of a run of
six splits it into a three and a two."*

### `--bowl-reject`: the cheap half of the precision problem

A detection that landed on the bowl instead of on a tsum carries the bowl's
colour, and `board_colours()` already knows what that is. Distance from it,
per detection:

| | p10 | median | p90 |
|---|---:|---:|---:|
| real tsums (422) | 74 | **173** | 212 |
| phantoms (190) | 38 | **76** | 143 |

Overlapping, so the cutoff buys phantoms with real tsums rather than for free —
but the exchange rate is good. Through `eval`:

| `--bowl-reject` | precision | recall | f1 |
|---|---:|---:|---:|
| 0 (off) | 0.675 | 0.874 | 0.762 |
| 40 | 0.733 | 0.850 | 0.787 |
| 50 | 0.746 | 0.830 | 0.786 |
| **60** | **0.769** | 0.814 | **0.791** |
| 70 | 0.785 | 0.774 | 0.779 |
| 100 | 0.838 | 0.706 | 0.767 |

40–60 is a plateau rather than a tuned edge. In the chains, at 60:

| | normal | color |
|---|---:|---:|
| phantom share of dragged members, off | 23% | 36% |
| …with `--bowl-reject 60` | **15%** | **27%** |

It costs a little: chains found 63 → 58 and mean length 4.9 → 4.6 under
`color`, which is the recall side of the trade showing up. Off by default,
and it needs the same live A/B as everything else here.

### What it does not touch

Recall. 16% of real tsums are never seen, and that is the ceiling on chain
length. The known causes are already written down elsewhere in this module:
two-tone faces (`--merge`, and `play2` exists to A/B it), tsums buried past
the peak floor (`floor_frac`), and black-faced characters dropped wholesale by
`include_dark`. `eval --sweep` reaches all of them; none has been swept since
`bowl_reject` changed the precision/recall balance, so the old best values may
no longer be best.

## Centres are per frame, not per round

The hand of five is fixed for a round, so caching the centres ought to be free.
It is not. `fixed_kinds` takes a `centres` argument and `play_loop` deliberately
does not use it:

| | median centre movement | p90 | share over 20 Lab |
|---|---:|---:|---:|
| same frame, different seed | 0.0 | 0.0 | 4% |
| consecutive in-round frames | 5.5 | 50.4 | **25%** |

The fit is deterministic; the *board* is not. A quarter of centres move more
than 20 Lab between one frame and the next — at a within-character scatter of
~7, that is a different character. The population ranking is the suspect: a
character a drag has just cleared is briefly rarer than a phantom cluster and
loses its seat.

Consequence for the caller: a fresh fit renumbers everything, so `base` — the
equipped tsum's id, which `play_loop` caches across frames — is stale the
moment the kinds are refitted. They are refit and re-read together. Getting
this wrong would point `--base-only` at the wrong character silently.

Cost is 6–7ms per frame against detection's ~50ms.

## Before this is the default

1. **A full-round A/B on the live emulator.** The blocking item, the same one
   `--mode blob` and `--mode reach` carry. What to watch: `cleared` and the
   count the game would not accept, not chain length on its own — longer chains
   are what a merge produces too. `--verify-clears` is required for either.
2. **Whether redistributing the phantoms costs more than the seat is worth.**
   `--purity` catches 69% of them; the other 31% are chain members sitting on
   open board. Only a played round prices that.
3. **Whether it should refit every frame or track the hand.** Refitting is what
   ships because the reuse measurement failed, not because refitting was shown
   to be better. Tracking centres with an update rule was not tried.
4. **`spare` is 1 on a measurement where 3 scored nearly the same** (17.8 vs
   18.0) and 2 scored worse (20.2). That non-monotonicity across ten boards is
   noise, not structure. Re-measure before reading anything into the value.

## Related

`docs/DATASET-FINDINGS.md` closes on over-detection as "a separate problem…
not addressed here". This is the same problem seen from the other end: the
phantoms are numerous enough and consistent enough in colour to form their own
cluster, which is what steals the seat. A collection labelled by the game's own
highlight would settle the per-board rule directly — every tsum the game
declines to mark is a negative example.
