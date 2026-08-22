# Getting a label out of the game

Status: **settled -- the third collection is labelled and worth training on.**
11,537 samples score 1.95x on the appearance test against a 1.3 bar, and the
delay was never the blocker: the threshold was. The remaining tuning question
(`floor_mult` 5, 6, or something else) is answered below, at 8. Rounds one to
seven, which is how that was arrived at, follow.

`dataset.enabled: true` ran overnight on 2026-08-19 and produced **5,729
samples across 296 sessions, 803MB**, between 19:46 and 08:59. None of it is
usable as training data. Nothing was wrong with the writer, the schema, or the
disk — the "marked" frames simply do not contain the game's answer.

This is the record of how that was established and what now stops it
recurring. The check itself is a command:

```
python -m ttheart_sender.game.tsum dataset --dir H:/Tsum/dataset --reread
```

which replays the analysis below and exits non-zero when a collection is not
worth training on.

## What a sample was supposed to be

Holding a tsum makes the game light up every tsum you could link it to — same
character *and* reachable, the two judgements this module makes worst. That
highlight is the label, and `marked_by_game()` reads it as the per-tsum change
between the board before the press and the board during it. `dataset.py`
saves both frames.

## What the samples actually contain

Replayed over 2,000 of the samples, scoring only tsums outside the 90px glow
so the splash cannot flatter the reading:

| measure | reading | what it should be |
|---|---:|---|
| board motion at press time (median change over untouched tsums) | p50 **7.8**, p90 **77.3** | well under the 8.0 mark threshold |
| samples where motion alone already clears the threshold | **48.8%** | near zero |
| share of the off-glow board read as "marked" | **48.7%** | ~one character's worth |
| marked tsums' colour distance to the pressed tsum, vs. the rest of the board | **1.15x closer** | far above 1 |
| marked tsums sharing the pressed tsum's k-means cluster | **22.9%** | above the base rate |
| — the same figure for marking at *random* | **21.9%** | — |

Stable across sample size: a 400-sample pass reads 1.08x, 49.5% and
23.1%/22.7% on the same measures.

The last two lines are the verdict. A reading that agrees with the clustering
no more often than chance does, and picks tsums no more similar to the pressed
one than the board average, is not a reading of anything. Splitting by board
motion does not rescue it: the appearance lift stays between 1.08x and 1.16x
in every motion band, including the stillest 4.8% of samples.

For contrast, on the same frames, k-means separates cleanly — mean Lab
distance to the pressed tsum is **16.2** within its cluster and **96.1**
outside it. The detector being corrected was working; the correction was noise.

## Why

Two causes, both in how the label was photographed.

**1. The delay was below the render floor.** Collection reused
`--hold-delay`, whose default is **0.10s**. That number belongs to
`--verify-hold`, which pays it on *every* drag and is therefore tuned for
throughput. The floor at which the game has finished drawing the highlight is
~0.15s — already written down in this repo, in `_assist`'s docstring, as the
reason `assist` waits 0.25s. So the "marked" frame was photographed before
there was a mark to photograph.

**2. One frame, against a board that was still moving.** `marks_on_board()`
already documents that one frame reports ~10.7 false marks where three report
none, and takes the per-pixel minimum across frames for exactly this reason.
`marked_by_game()` took a single frame and compared it to an absolute
threshold, on a board where — measured above — the median untouched tsum had
already moved enough to clear that threshold half the time. The `before` crop
is captured at detection time and the `marked` frame lands after path
computation, cursor travel and the press, so the gap is much longer than the
delay suggests.

Neither cause is visible in a sample folder. Both are invisible in the JSONL
too, because schema 1 recorded neither the reading nor the settings it was
taken at — which is why finding this needed a full re-decode of 803MB.

## What changed

Schema 2. The collector no longer borrows `--verify-hold`'s settings.

* **It owns its delay**, default `0.25` (matching `assist`), and
  `DatasetWriter` clamps anything below `RENDER_FLOOR = 0.15` rather than
  obeying it. The throughput argument that set 0.10 does not apply to one drag
  in four, capped at 20 per round.
* **It reads three frames** 0.05s apart and keeps only what changed in all of
  them — the same defence `marks_on_board()` takes.
* **It refuses what it cannot read.** A sample whose baseline motion exceeds
  `max_motion` (12.0), or whose reading lights more than half the board, is
  dropped at the moment the information to judge it still exists. A round that
  collects nothing now says why, instead of leaving an empty folder with two
  possible explanations.
* **Every row carries its own reading** — `baseline`, `marks`, `marked`, and
  the `capture` settings used. `tsum dataset` scores a schema 2 collection
  without decoding a single JPEG; `--reread` is only needed for schema 1.

`--verify-hold` itself is unchanged. It is opt-in, already recorded as having
lost its A/B decisively (196 dragged tsums against 527), and changing its
delay would invalidate that measurement. Its help text now carries the
finding: replayed over these 5,729 drags, it drops a mean **3 of 4** chain
members on **43%** of them, on a signal measured here to be noise.

## Second collection: motion fixed, marks still absent

19 samples on the schema 2 defaults (0.25s, 3 frames), 2026-08-20.

| measure | schema 1 | schema 2 | verdict |
|---|---:|---:|---|
| board motion p50 | 7.8 | **3.0** | fixed |
| samples past the 8.0 threshold | 48.8% | **0.0%** | fixed |
| share of board read as marked | 48.7% | 23.3% | better |
| appearance lift | 1.15x | **1.22x** | still under 1.3 |
| k-means agreement vs. base rate | 22.9 / 21.9 | 23.0 / 18.8 | still ~chance |

**Cause 2 is fixed and cause 1 is not.** The board is genuinely still now — the
diff images are near-black away from the press — but the tsums that do react
look no more like the pressed character than the board average does. Whatever
changed, it is not the game naming a character.

Looking at the frames says why: the only thing that lights up is the pressed
tsum's own glow and its chain counter. There is no board-wide highlight in the
frame at 0.25s. Two samples also show the *previous* press's glow still burning
in the `before` crop, which the diff then reads as a change when it vanishes.

So the remaining question is the one this cannot answer from a dataset:
**does the board-wide highlight render later than 0.25s, or not at all in this
build?** `hold --hold 3.0` presses one tsum and photographs for three seconds,
which is exactly that measurement. It has to run before any more collecting.

A note on measuring it: a count of "strongly reacting tsums" was tried here as
a cheap way to separate "no highlight" from "highlight present", at four
thresholds (4x and 10x the noise floor, 50% and 80% of the pressed tsum's own
glow). None separated the two, because the pressed tsum's glow reads **55**
against a **168** score popup drifting across the board. The appearance test is
the only discriminator that holds up, which is why `tsum dataset` now always
decodes images for it and reports INCONCLUSIVE rather than passing when it
cannot.

## Third round: `hold` pressed bare water

`hold --hold 3.0` was run to answer whether the highlight renders at all. It
reported:

```
134 tsums, r~12.2px, board still (drift 0.00)
pressed tsum changed by 0.0
every other tsum, most-changed first: 124.1, 124.0, 120.7, ... 4.5, 3.4, 3.1
9 other tsum(s) changed by more than 8.0
predicted partners (7): 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
-> the pressed tsum reacted and its predicted partners did not. ... this route is closed.
```

**The verdict was wrong in every clause, and the run proves nothing either
way.** Taken in order:

* *"the pressed tsum reacted"* — it read **0.0**, printed three lines above.
* *"this route is closed"* — nine tsums reacted at **112-124** against a board
  median of **0.0**. `hold_diff.png` shows four or five green tsums lit and
  the rest of the frame pure black. Something was drawn; the code never
  checked.
* The press never landed. The pixels at the press point are HSV
  `[109, 235, 99]` — the board's dark blue water, in a gap between tsums,
  confirmed by eye on a zoom. A press that hits nothing draws no chain counter,
  which is exactly the 0.0 that was misread as a reaction.

Why it pressed water: **detection over-split**. 134 detections at r~12.2px on
a board whose tsums are ~28px. Re-run on the saved frame:

| `-k` | detections | radius |
|---:|---:|---:|
| 8 | 64 | 18.9px |
| **12 (default)** | **134** | **12.2px** |
| 16 | 149 | 11.3px |

At the default this board is read as roughly double the tsums at half the
size, so the "tsum" the probe picked was a phantom on background — and the
seven "predicted partners" were phantoms too, which is why they all read 0.0.

Fixed in `_hold`: it now checks whether the press landed before concluding
anything, reports when the board reacted somewhere other than the predicted
partners (a real mark plus wrong clustering — the case that was filed under
"route closed"), and names over-splitting when the counts look implausible.

The original question is still open. Re-run at a `-k` that reads this board
sanely:

```
python -m ttheart_sender.game.tsum hold --hold 3.0 -k 8
```

Unresolved, and not to be guessed at: the green tsums that lit are **not** the
pressed character — the press point is background, so there is no pressed
character to compare against. They may be the game's idle hint (which `idle`
exists to film) rather than a response to any press. Only a run whose press
actually lands can tell.

## Fourth round: the marks are real

`hold --hold 3.0 -k 8`, on a frame detection read properly (50 tsums,
r~27.3px):

```
pressed tsum changed by 86.4
predicted partners (3): 64.9, 64.0, 61.1
everything else: median 7.5, max 65.1
-> 3/3 predicted partners reacted and the rest of the board did not.
```

`hold_diff.png` shows four same-character tsums lit in the bottom-left corner
and the rest of the board at faint outline noise. **The game marks matches,
the mark is clean, and the route is open.** The premise the whole collection
rests on is confirmed.

Two details worth keeping:

* **One unpredicted tsum reacted as hard as the partners** (65.1, against
  partners at 61-65). Clustering found three of four. That one missing tsum is
  precisely the correction a dataset exists to supply — the label is not just
  real, it is already disagreeing usefully.
* **The 8.0 threshold sits in the noise.** 23 tsums cleared it while the board
  median was 7.5 and the real marks were at 61-65. The signal has a factor of
  eight of headroom, so the threshold should be derived from the board's own
  floor rather than fixed. `marked_by_game` now records that floor as
  `baseline`; nothing yet uses it to set the threshold.

### `k` was not the fault

The obvious inference from round three — that `k=12` is wrong — is wrong.
Swept over the ten labelled boards:

| `-k` | f1 | precision | recall |
|---:|---:|---:|---:|
| 6 | 0.603 | 0.623 | 0.584 |
| 8 | 0.624 | 0.621 | 0.628 |
| 10 | 0.632 | 0.616 | 0.650 |
| **12** | **0.762** | 0.675 | 0.874 |
| 16 | 0.578 | 0.475 | 0.738 |

`k=12` wins by a wide margin and stays the default. What round three actually
exposed is that it fails *catastrophically* on the occasional frame — 134
detections at r~12px — and that `hold` had no guard against being handed one.
`play_loop` has always had that guard (`--min-tsums 20`, `--max-tsums 110`,
and 134 is past it), so `play` would have discarded that frame untouched.

Fixed: `hold` now applies the same gate before it presses anything. Note the
open question this leaves — even at its best, `k=12` scores precision 0.675,
so about a third of detections are phantoms. Over-detection is real; it is
just not what broke round three.

### `hold` now reports when the mark arrives

The one number the collector needs and nothing measured: `hold` sampled across
the whole three seconds and reported only the strongest reaction, so "the mark
exists" and "the mark exists by 0.25s" were indistinguishable. It now reports
the first frame the partners clear the threshold, the first they reach 80% of
peak, and a suggested `dataset.delay` with headroom. The clock starts once the
touch has been delivered, which is the same instant `marked_by_game` starts
counting from, so the number transfers directly.

## Fifth round: the onset is still unmeasured

The first run with onset reporting produced a number that cannot be true:

```
101 tsums, r~15.1px
predicted partners (5): 77.5, 40.8, 40.8, 40.6, 7.0
everything else: median 4.6, max 170.6
   mark onset: first over 8 at +0.00s, 80% of peak at +0.00s (peak 40)
   -> set dataset.delay to about 0.15
```

Nothing the game draws is at full strength in the frame grabbed milliseconds
after the touch. `hold_diff.png` says what really happened: a chain counter
reading **2** and a link line between two tsums. The press plus the cursor
wiggle that delivers it had drawn a short chain, and its glow — 170 against
partners at 40 — was already in flight when the first frame was taken.

Left alone, that reading recommends `0.15` — the render floor, i.e. exactly
the timing that made schema 1 worthless. Three fixes:

* **`_mark_onset` refuses a first-frame peak.** An onset of +0.00s means the
  change did not start with the press, so no delay can be derived from it. It
  now says so and recommends nothing.
* **The verdict flags competing activity.** Its median test is right and was
  passed here legitimately, but it read "the rest of the board did not react"
  while 35 non-partners cleared the threshold and the strongest hit 170. It
  now adds a warning when the non-partners outnumber or outweigh the partners.
* **`hold` warns when the radius disagrees with the count.** 101 detections at
  r~15px passed the 20-110 count gate, but tsums fill the board, so that many
  imply ~23px and these were fragments. Measured on four frames: a well-read
  board sits at 0.84 of the implied radius, badly-read ones at 0.61-0.66. The
  gate warns below 0.75 rather than refusing — four frames is enough to flag
  one, not enough to discard one.

## Sixth round: two more of my own guards were wrong

```
50 tsums, r~22.6px, board still (drift 0.28)
  WARNING: 50 tsums that size should measure ~33px, not 22.6px.
pressed tsum changed by 91.0
predicted partners (5): 94.7, 1.8, 2.1, 2.0, 4.9
everything else: median 3.8, max 20.9
-> 1 partners reacted, but so did the rest of the board -- that is motion, not marking.
```

A still board, a landed press, one tsum lit at **94.7** against a floor of
**3.8**, and the verdict called it motion. Both guards misfired:

**The radius warning was too tight.** It fired on a perfectly good 50-tsum
board reading 22.6px where another frame of the same board read 27.3px —
frame-to-frame radius wobbles that much with nothing wrong. Over-splitting
needs *both* signals: a real board holds ~50-70 tsums, so it is the **count**
that says "split" and the radius only confirms it. Now requires count > 75 as
well, which leaves the two genuinely over-split frames flagged and stops the
false alarm on good ones.

**The verdict tangled two questions.** It tested `median(predicted partners)`
against the board floor — but with five predictions of which one was real, the
median partner reads 2.1 against a floor of 3.8, so a clean mark scored as
motion. *The clustering being wrong is the finding, not a reason to disbelieve
the mark.*

Rewritten to read the game's answer off the board first and score the
clustering against it second: the floor is the board's own median, the bar is
five times it, and whatever clears the bar is what the game drew — a reading
that does not care whether `expect` was any good. Replayed over all three
landed presses:

| run | floor | marked | clustering right | old verdict | new |
|---|---:|---:|---:|---|---|
| four | 7.5 | 4 | 3 of 4 | marks confirmed | marks confirmed |
| five | 4.6 | 15 | 4 of 15 | marks confirmed | marks confirmed, onset refused |
| **six** | **3.8** | **2** | **1 of 2** | **"motion, not marking"** | **marks confirmed** |

Round six is the sharpest data point so far: the game marked one or two tsums
where clustering predicted six. That over-prediction is exactly what a
collected sample is for.

## Seventh round: the threshold was the other half

```
75 tsums, r~20.0px, board still
pressed tsum changed by 80.8       predicted partners: 74.0, 75.4, 75.2, 72.0
board floor 6.8, so a mark has to clear 33.8
-> the game marked 15 tsum(s). Clustering predicted 4 and got 4 of the 15 right.
   mark onset: UNRESOLVED -- already at full strength in the first frame
```

Marks confirmed a third time, and clustering was right about all four of its
predictions while missing eleven more. But the onset came back unmeasurable
again, which sent this back to the collected samples with a specific question:
**if `hold` reads marks so cleanly, why does the collector not?**

The answer is the threshold. Across every `hold` run, real marks sit **8x to
25x** above the board's own floor — 61-65 against 7.5, 95 against 3.8, 75-167
against 6.8. The fixed `8.0` sits *inside* that floor's noise on a live board,
which is why every collected reading looked like half the board lighting up.
Re-scoring the 19 collected samples against a floor-relative bar:

| bar | share of board "marked" | appearance lift |
|---|---:|---:|
| fixed 8.0 | 23.3% | 1.22x |
| 4x floor | 8.2 per board | 1.26x |
| 5x floor | 7.6 per board | 1.29x |
| **6x floor** | **13.8%** | **1.35x** |

So the collector's samples do contain more signal than the fixed threshold
could see. `marked_by_game` now takes a `floor_mult` (collector default 5,
recorded per sample as `bar`), `tsum dataset` scores against the bar the
sample was judged on, and `--floor-mult` re-scores either collection on equal
terms. `--verify-hold` keeps the fixed threshold, since its A/B was measured
under it.

**This is an improvement, not a resolution.** 1.35x is barely over the 1.3
bar, from 19 samples, against `hold` readings that are unambiguous. The
verdict now has a MARGINAL band saying exactly that rather than passing —
filtering to samples whose press clearly landed does not lift it either
(1.21x-1.36x), so something about a mid-round board still differs from an idle
one, and 19 samples cannot say what.

### Two clock fixes

`_mark_onset` started its clock *after* the wiggle that delivers the touch,
losing ~0.1s of the press and guaranteeing every onset read "+0.00s, already
at peak". It now starts at `mouseDown`. And UNMEASURABLE became **UNRESOLVED**,
because a first-frame peak has two readings a single hold cannot separate: the
mark renders faster than a press can be photographed, or the baseline already
differed. The message names both and still recommends no number.

## Eighth round: the collection is labelled

`dataset.enabled: true` ran from 2026-08-20 to 2026-08-22 on the schema 2
defaults and produced **11,537 samples across 743 sessions, 1,479MB** -- every
row schema 2, every row captured at the same settings (0.25s, 3 frames, 0.05s
gap, 5x floor). 2,132 of them are fever boards.

```
python -m ttheart_sender.game.tsum dataset --dir H:/Tsum/dataset --appearance 1500
```

| measure | schema 1 | 19 samples | **11,537 samples** | bar |
|---|---:|---:|---:|---|
| board motion p50 | 7.8 | 3.0 | **3.1** | under 8 |
| samples past the 8.0 threshold | 48.8% | 0.0% | **9.5%** | near zero |
| share of board read as marked | 48.7% | 23.3% | **14.9%** | one character's worth |
| appearance lift | 1.15x | 1.22x | **1.95x** | over 1.3, ideally 1.6 |
| k-means agreement vs. base rate | 22.9 / 21.9 | 23.0 / 18.8 | **37.1 / 26.1** | above base |

**The marks are in the frames.** The lift clears both the 1.3 bar and the 1.6
"cleanly" bar, on 75x the samples the MARGINAL verdict was issued on, and
k-means now agrees with the label half again as often as random marking does
-- which is the shape a useful label has: agreeing more often than chance,
disagreeing often enough to be worth collecting.

Open question 1 (whether `dataset.delay` was ever the blocker) resolves as
**no**. Nothing about the delay changed between the 19-sample run at 1.22x and
this one at 1.95x. Open question 2 (why a mid-round board reads worse than an
idle one) resolves as **it does not** -- it read worse because 19 samples
cannot measure a 2x effect, which is what the MARGINAL band exists to say.

### The threshold was the whole story, and 5x was still too low

Open question 3 -- whether `floor_mult` should be 5, 6 or something else --
was a 19-sample guess. Re-swept on this corpus, scoring every tsum outside the
90px glow by its Lab distance to the pressed tsum against the board average:

| band, as a multiple of the board's own floor | tsums per board | appearance lift | same kind as pressed |
|---|---:|---:|---:|
| 0-1 | 22.4 | 0.97x | 22.6% |
| 1-2 | 9.5 | 1.00x | 23.8% |
| 2-3 | 3.9 | 0.99x | 24.1% |
| 3-4 | 1.8 | 0.98x | 24.7% |
| 4-6 | 2.1 | 0.99x | 25.0% |
| 6-8 | 1.0 | 0.99x | 26.0% |
| **8-12** | **1.4** | **1.14x** | **34.3%** |
| **12-20** | **1.5** | **1.14x** | **36.1%** |
| **20+** | **2.8** | **1.22x** | **40.0%** |

Base rate over all far tsums: 25.1% same kind, 1.00x by construction. 800
samples, 37,066 tsums.

Nothing below **8x the floor carries any signal at all** -- lift 1.00 and
same-kind indistinguishable from marking at random, in every band. The signal
switches on between 6-8x and 8-12x and does not fade above it. So
`floor_mult: 5` was admitting about 3 noise tsums per board on top of ~5.7
real ones, and roughly a third of every collected label was noise.

The break holds when the corpus is split by fever (1.14x above 8x either way),
by detection count (over-split boards included), and by board stillness in two
of three terciles. It fails on the stillest third, where the floor is under
2.2 and 8x lands at ~17 absolute -- inside the render and JPEG noise. So the
fixed 8.0 stays underneath the multiple as a floor, as it already did.

**The multiple, not the absolute change, is what discriminates.** Tested the
other way round, with an absolute bar and no floor relative to it:

| bar | marks per board | lift | same kind | boards labelled with nothing |
|---|---:|---:|---:|---:|
| abs 8 (schema 1) | 12.3 | 1.06x | 29.5% | 30 / 800 |
| abs 8, **5x floor** (old default) | 6.8 | 1.13x | 35.0% | 40 / 800 |
| abs 8, **8x floor** (new default) | 5.4 | 1.18x | 37.8% | 69 / 800 |
| abs 40, no floor | 5.3 | 1.19x | 37.7% | 119 / 800 |
| abs 40, 8x floor | 4.6 | 1.22x | 39.1% | 128 / 800 |
| abs 80, 8x floor | 3.2 | 1.26x | 40.5% | 201 / 800 |

8x floor dominates abs 40: the same label quality for 50 more boards kept. And
a large absolute change that does *not* clear the floor multiple is noise --
tsums over 40 absolute but between 4x and 8x their own board's floor read
0.99x lift and 27.2% same-kind, which is a busy board moving, not a mark.
Above 80 the gains are bought purely with recall, which a corpus this size has
no reason to spend.

**`floor_mult` default is now 8.0** (`config.yaml`, `DatasetConfig`,
`--dataset-floor-mult`). Re-scored at it, this collection reads **2.23x** with
39.8% k-means agreement against a 26.0% base rate, at 11.6% of the board
marked.

### A bug this found: the label's bar was silently re-scoring `--verify-hold`

`marked_by_game` computed one bar and used it for two jobs -- the label it
hands the collector, and the trim `--verify-hold` applies to the stroke. The
comment said `floor_mult` of 0 "keeps the old fixed behaviour for
`--verify-hold`, whose A/B was measured under it", and the call site passes 0
when not collecting, so the intent was right. But with both features on, the
collector's `floor_mult` re-scored the trim as well -- a stricter bar dropping
more chain members, in the one combination nobody would think to look at, and
it would have got stricter again with this change.

Split: the trim always runs at the fixed `threshold`, and `floor_mult` only
ever raises the bar the *label* is scored at. Pinned by a test.

## What has not been settled

1. **Whether `floor_mult` should go higher still.** 8 is where the signal
   switches on, not where it peaks -- quality keeps creeping up to 80
   absolute, bought with boards that end up labelled with nothing. Worth
   re-sweeping once something is actually trained on this, since that is the
   only test that prices recall against purity properly.
2. **Whether the `before` crop needs to be clean.** The previous press's glow
   survives into it, and the reading charges that disappearance to the wrong
   tsum. Unchanged from before, and now the cheapest remaining source of
   noise in an otherwise usable label.
3. **Over-detection.** Boards in this collection run to 110 detections at a
   median radius of 19.8px against a real board of ~50-60 tsums; p90 is 85.
   The label is now good enough to measure this against rather than guess at
   -- a tsum the game marks that detection split in two is visible in the
   data.
4. **Whether the board can be still enough in fast play.** 9.5% of samples
   sit past the 8.0 motion threshold, against 0% in the 19-sample run, which
   is the first honest measurement of this. `max_motion` at 12.0 is what
   keeps the worst of them out; nothing yet says whether 12 is the right
   number.

## Related, from the same pass

`docs/TODO-blob-adjacency.md` item 2 names the missing ingredient for settling
`--mode blob`: *"negative link examples (pairs a human marks as not chainable),
which the `label` flow does not currently collect."* A working collection is
exactly that ingredient -- every tsum the game declines to mark is a negative
example, free and in bulk. **That item is now unblocked**: 11,537 labelled
boards at ~5.4 marks per board leave roughly 40 negatives each.

Also confirmed while measuring: schema 1 recorded boards with up to **110**
detections at a median radius of **19.8px**, against a real board of ~50-60
tsums, and one sample in the first session is a pre-round "READY" screen with
23 phantom detections on empty water. The play loop's plausibility gate let it
through. Over-detection is a separate problem from this one and is not
addressed here, but the collection is a good corpus for it -- the images are
fine even where the labels are not.
