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

## Ninth round: the collection was trained on, and the answer was no

303 samples over 21 sessions, the first corpus put through `tsum learn`. The
thing being tested was the obvious use of a label that says *these are the same
character*: fit one colour palette over the whole corpus, hand it to `detect`
through the `palette=` argument it already accepts, and let a cluster id mean
the same character on every frame instead of being re-derived per frame and
thrown away seven times a round.

It does not work, and the way it fails is more useful than the fact that it
does.

### No k beat the per-frame fit

Fitted on 16 sessions, scored on 5 held out:

| k | agree | split | balanced |
|---|---|---|---|
| per-frame k-means | 33.7% | 79.3% | **56.5%** |
| 6 | 37.6% | 72.4% | 55.0% |
| 8 | 33.7% | 77.3% | 55.5% |
| 12 | 30.7% | 81.5% | 56.1% |
| 24 | 26.6% | 86.6% | 56.6% |
| 32 | 24.1% | 85.9% | 55.0% |

`agree` is the share of game-confirmed same-character pairs given one id;
`split` the share of weak negatives given different ids. The best balanced
score is +0.1 over doing nothing, which is noise.

### The trap in the middle of that table

Agreement falls monotonically as `k` rises, and split rises to meet it. They
trade against each other across the entire range, so **neither has a maximum
that means anything**, and a verdict read off either one alone is measuring the
trade rather than the quality.

Read off agreement, k=6 is a 3.9-point win over per-frame clustering. It is
actually 1.5 points worse. All it did was merge characters -- which lifts
agreement, because merged characters agree, and costs only split. This was very
nearly shipped as a positive result: the first version of `learn` compared
agreement against a baseline and computed `split` for the palette *but not for
the baseline*, so the two sides were scored on different questions. The verdict
is now `balanced`, the mean of both halves, computed identically for both.

Pinned by `tests/test_learn.py::test_agreement_alone_pays_a_palette_to_merge_characters`.

### Why a global palette loses to a per-frame one

Per-frame k-means is *relative*. It partitions whatever is on this board into k
groups, so one character lands together even when its absolute colour has
drifted. A global codebook needs absolute Lab to be stable frame to frame.

It is not, and not by a little. Over 275 labelled groups, the same character's
face colours spread **45.6 Lab within a single frame** -- wider than the 43.9
spread of group *means between* frames. The variation that a global palette
would have to model is not between-frame drift at all; it is inside one frame,
inside one character, where no palette of any size can reach it.

Two controls, same corpus:

* the game's confirmed same-character set is only **1.18x** more
  colour-coherent than a random group of the same size off the same board;
* excluding the 18% of marks that sit inside the 90px glow -- where a reaction
  means proximity, not identity -- moves that to 1.11x rather than fixing it.

This is `## The ceiling: colour cannot tell characters apart` in
`docs/COMMANDS.md`, measured from the other side. That section priced colour as
a way of separating two characters. This prices it as a way of recognising
*one*, against the game's own answer instead of against a hand label, and it
comes out barely above chance.

### The appearance test's headline is a mean of ratios

Chasing the above turned up a measurement fault in `tsum dataset` itself. Its
verdict number is the mean of a per-sample ratio, and a ratio has no upper
bound: a sample whose marked tsums are nearly the pressed tsum's own colour
divides by something near zero.

On this corpus:

```text
mean of per-sample ratios   :  3.94   <- what it reported
median of per-sample ratios :  1.30
ratio of pooled means       :  1.28
p90 / p99 / max             :  3.38 / 70.87 / 122.29
```

**13 samples of 280 (4.6%) supply 62% of the headline**, on denominators as
small as 0.46. Those are not bad samples -- a tiny denominator is a very strong
label -- but a mean over them describes the tail, not the collection, and the
1.6x bar the verdict is read against was never meant to be cleared that way.
The 2.23x reported for the 11,537-sample collection in the eighth round is the
same statistic and carries the same caveat.

`dataset` now prints the median beside the mean, says what share of the mean
the outliers carry, and warns when the two land on opposite sides of the bar.
The thresholds are unchanged: they were calibrated against the mean, and moving
the statistic under them would invalidate every earlier round in this document.

### What this does and does not settle

Settled: a global colour palette is not the way to use this collection, and no
amount of more of the same data changes that. `learn` is kept -- it fits,
scores, refuses, and is the harness the next attempt gets measured in.

Not settled, and now the obvious next thing: **reachability, which is
colour-independent**. The marked set is the game stating which tsums are
chainable from the pressed one, and that is the question `adjacency` guesses
at. The `TODO-blob-adjacency.md` item unblocked in the eighth round is still
the best-priced piece of work in this corpus, and nothing measured here
weakens it -- everything above is about identity, and none of it is about
reach.

## Tenth round: what the collection was actually good for

The ninth round used the collection to try to fix *identity* and failed. This
round used it to price *decisions*, and that is what it turned out to be for.

Every collected row carries `proposed` (the chain the bot chose) beside `kept`
(the subset the game marked while the first tsum was held). That pair is a
recording of the bot being right and wrong, drag by drag, with the game as the
judge -- and nothing had ever read it.

### The waste, measured

Over 303 drags at the live settings (`link_px 105`, `block 1.25`, `min_chain 3`):

```text
chain the bot proposed : mean 4.69  median 4
what the game accepted : mean 3.41  median 3
-> 27.3% of proposed members the game would not take
-> 22.4% of drags end up under min_chain: the stroke runs and clears nothing
```

### `link_px` cannot fix it, and that is the surprising part

Taking the 788 proposed members that sit outside the 90px glow -- where being
marked is evidence rather than proximity -- 49.2% were refused. Standardised
mean difference between kept and refused members:

| feature | kept | refused | separation |
|---|---|---|---|
| position in chain | 3.1 | 4.7 | **+0.70** |
| distance to the pressed tsum | 169px | 215px | **+0.62** |
| Lab distance to the pressed tsum | 10.4 | 18.5 | +0.57 |
| **leg length to the previous member** | **81.2px** | **82.5px** | **+0.06** |

`link_px` is the most carefully tuned number in `flows/play.yaml` and it
**separates a good chain member from a refused one by 0.06 standard
deviations**, which is to say not at all. Leg length decides which links the
graph offers; it says nothing about which of them the game will honour. Two
things follow: the p90-of-labelled-gaps method that set 105 was answering a
different question than the one that matters, and no amount of re-tuning it
addresses the 27%.

What does predict refusal is how far the chain gets from the press. Share of
drags where the game accepted *every* member:

| reach of chain | drags | clean |
|---|---|---|
| under 90px | 16 | 100% |
| 90-150px | 99 | 81% |
| 150-220px | 96 | 65% |
| 220-300px | 57 | 33% |
| over 300px | 35 | 11% |

### Two rules tried and rejected before the one that worked

**Head-anchored colour truncation** -- cut the chain at the first member whose
face colour is too far from the pressed tsum's. Colour carries real signal here
(+0.57, better than it manages anywhere in the ninth round), but truncation
loses good members to catch bad ones and every threshold from 60 down to 15
cleared *less* than doing nothing. Rejected.

**A lower `max_chain`** -- errors concentrate in the tail, so capping the chain
mechanically avoids them. It does: clears per second of stroke run 18.8 at
`max_chain 3` against 10.6 at 12. Not shipped, because `PlayReport` already
notes that fifteen 3-chains and eight 6-chains "are not close, and the second
scores far better" -- the metric this optimises is not the metric that wins,
and nothing here prices the scoring curve. Left alone deliberately.

### `--verify-hold` was condemned for the wrong reason

It is documented as measured-and-not-recommended because at `--hold-delay 0.10`
-- below the 0.15 render floor -- the highlight has not been drawn and the
reading is noise. True, but not the whole story: replaying it at the collector's
working delay, where the marks are real,

```text
                    cleared   stroke+hold
today (blind)           688         64.6s
verify every drag       906        115.0s
```

it clears 32% more and takes 78% longer. **The check is not too noisy; it is
too evenly spread.** It costs the same on every drag while the risk it buys
against is concentrated in a fifth of them.

### `--verify-reach`, which is the same check bought where it pays

Replayed over the same 303 drags, counting a drag as clearing only when every
member was accepted:

| verify when reach > | holds paid | cleared | time | clears/s |
|---|---|---|---|---|
| never (today) | 0 | 688 | 64.6s | 10.6 |
| 220px | 92 (30%) | 853 | 69.0s | 12.4 |
| **260px** | **56 (18%)** | **819** | **64.3s** | **12.7** |
| 300px | 35 (12%) | 787 | 62.5s | 12.6 |
| every drag | 303 | 906 | 115.0s | 7.9 |

260 is the recommendation because it is best or near-best under both a
one-frame and a three-frame reading cost; 220 wins only under the cheaper one.
The ordering survives any fixed per-drag overhead -- adding a constant to every
row compresses the differences without reversing them, and at a generous 1.0s
of settle and detection per drag the 260 row is still 22% ahead.

Shipped **off**, as `verify_reach` in `flows/play.yaml`, with its own
`--verify-delay` defaulting to 0.25 and clamped up to the render floor so this
version cannot repeat the mistake that sank the last one. `PlayReport` gained
`verified`, printed at the end of a round as `checked N chain(s) before
dragging (M% of drags)`, so the cost sits in the log beside the trimmed and
rejected counts it bought.

### What would falsify it

The replay assumes a drag clears nothing when any member is refused -- the
behaviour named elsewhere in the code as *"a 3-chain it only marks two of pops
nothing at all"*. If the game is in fact lenient and pops the accepted members
anyway, today already collects 906 and this rule buys nothing but delay. The
collection cannot distinguish those two, because it never recorded what
actually cleared. A round with `--verify-clears` on can, and that is the test.

## Eleventh round: the tenth round replicated, on a corpus it never saw

726 drags over 51 sessions, collected 2026-09-02 on v1.8.4 at the schema 2
defaults with `floor_mult` at its new 8.0 — the first corpus collected after
the eighth round changed that default. 101MB. It passes the label check:

```
python -m ttheart_sender.game.tsum dataset --dir dataset --appearance 700
```

| measure | 11,537 samples (8th) | **726 samples (11th)** | bar |
|---|---:|---:|---|
| board motion p50 | 3.1 | **3.8** | under 8 |
| samples past the 8.0 motion threshold | 9.5% | **12.1%** | near zero |
| share of the board read as marked | 14.9% (11.6% at 8x) | **13.3%** | one character's worth |
| appearance lift, mean | 1.95x (2.23x at 8x) | **3.12x** | over 1.3 |
| appearance lift, median | — | **1.17x** | — |
| k-means agreement vs. base rate | 37.1 / 26.1 | **37.7 / 25.4** | above base |

The mean/median split is the ninth round's caveat firing as designed: 31
samples of 663 carry 53% of the mean. The verdict passes on the mean, says so,
and says to trust the median — 1.17x is a thin per-sample signal on a corpus
this size, and it is the k-means line (37.7% against a 25.4% base rate) that
carries the verdict, not the headline ratio.

### The tenth round's numbers came back, from different rounds on a different day

The point of this round was replication. The tenth round priced the bot's
decisions on 303 drags; nothing about that analysis had ever been re-run on
data it had not been derived from. Same measurements, new corpus, using
`scripts/replay_decisions.py`:

| measure | tenth (303 drags) | **eleventh (726 drags)** |
|---|---:|---:|
| chain the bot proposed | 4.69 | **4.55** |
| what the game accepted | 3.41 | **3.23** |
| proposed members refused | 27.3% | **29.0%** |
| drags left under `min_chain` | 22.4% | **29.3%** |
| leg length, kept vs refused (SMD) | 0.06 | **0.07** |
| position in chain (SMD) | 0.70 | **0.61** |
| distance to the pressed tsum (SMD) | 0.62 | **0.47** |

Share of drags where the game accepted *every* member, by how far the chain
reaches from the press:

| reach | tenth | **eleventh** |
|---|---:|---:|
| under 90px | 100% (16 drags) | **100%** (43) |
| 90-150px | 81% | **65%** (259) |
| 150-220px | 65% | **54%** (244) |
| 220-300px | 33% | **39%** (104) |
| over 300px | 11% | **7%** (76) |

The ladder is the same shape and the same conclusion: **reach predicts refusal
and nothing else does.** `link_px` is now measured at 0.06 and 0.07 standard
deviations of separation on two independent corpora, which is as close to
settled as anything in this document. The middle bands read a few points worse
here, and this corpus is 41.2% fever against 18% in the eighth-round
collection, which is the likeliest reason — fever boards refuse less (24.6%
against 31.7%) but the normal boards in between are busier.

### `verify_reach 260` re-priced, and it holds

Replayed over the 726 drags, counting a drag as clearing only when every member
was accepted, at three costs for one check:

| verify past | holds | cleared | clears/s @0.17 | @0.28 | @0.41 |
|---|---:|---:|---:|---:|---:|
| never | 0 | 1436 | 8.59 | 8.59 | 8.59 |
| 300px | 76 | 1640 | 10.08 | 9.59 | **9.07** |
| **260px** | **121** | **1692** | 10.39 | **9.60** | 8.82 |
| 220px | 180 | 1759 | **10.47** | 9.37 | 8.33 |
| 180px | 288 | 1844 | 10.43 | 8.84 | 7.50 |
| every drag | 726 | 1931 | 8.46 | 6.27 | 4.80 |

260 is best or within a point of best in every column — +21% at a cheap read,
+12% at a realistic one, +3% at the most pessimistic — and `every drag` loses
at every cost, which is `--verify-hold` again. **No change: 260 stays.** The
one thing the sweep says that is new is that 300 wins the most expensive
column, so if the check ever gets slower the threshold should move up, not
down.

### Truncation is not the free version of the check

The obvious cheap alternative — cut the chain at the first member too far from
the press, pay no hold at all:

| rule | drags | mean length | cleared | time | chains of 6+ |
|---|---:|---:|---:|---:|---:|
| today | 726 | 4.55 | 1436 | 167.1s | 20.8% |
| truncate past 300px | 725 | 4.26 | 1449 | 158.8s | 18.6% |
| truncate past 260px | 723 | 4.08 | 1452 | 153.0s | 13.0% |
| truncate past 220px | 720 | 3.84 | 1451 | 145.6s | 8.1% |
| truncate past 180px | 698 | 3.46 | 1400 | 130.5s | 2.1% |

It looks free — the same tsums cleared for 5-13% less stroke time — and it is
the tenth round's `max_chain` trap wearing a different hat. What it actually
does is trade long chains for short ones: at 260px the share of 6+ chains falls
by a third, and `PlayReport` already records that fifteen 3-chains and eight
6-chains are not close. Nothing offline prices the scoring curve, so a rule
whose entire effect is to shorten chains cannot be justified from this corpus.
**Not shipped, deliberately**, and for the same reason as last time. The 300px
row is the only one that barely moves the length distribution, and it buys 5%
of stroke time — not enough to be worth a switch.

### Two things this corpus says that the last one could not

**Over-detection has gone away.** The eighth round measured boards running to
110 detections at a median radius of 19.8px, p90 of 85, against a real board of
50-70 tsums — and a `hold` run that pressed bare water because of it. Here:

| | eighth round | **eleventh** |
|---|---:|---:|
| detections per board, p90 | 85 | **52** |
| maximum | 110 | **60** |
| boards past the over-split flag (>75) | many | **0** |

Not one board in 726 is over-split. Whatever changed in detection between
v1.8.2 and v1.8.4 fixed the failure mode round three was written about, and the
`hold` guards added for it have had nothing to catch since. Item 3 of "what has
not been settled" in the eighth round is closed by measurement rather than by
work.

**A fifth of readings are not as legible as the trim assumes.** On 20.1% of
drags the pressed tsum itself does not clear its own bar, and on 4.4% the whole
board shows no mark at all. The first number is not evidence that the press
missed — 89.7% of those drags still show marks elsewhere, so the game did
respond and it is the reading of the pressed tsum's own disk that is
unreliable. The second is: a frame with nothing on it has not said "refused",
it has said nothing, and the trim reads the two identically.

The size of the problem is small and the fix is cheap: of the 121 drags a
`verify_reach 260` check fires on, 6 read nothing, and 2 of those are chains
the trim cancels outright on the strength of a frame that said nothing. Worth a
guard — *reading unusable, drag as proposed* — but not worth claiming a gain
for: 6 drags in 726 is the honest size of it, and the same guard keyed on the
silent-head signal instead would fire on 24 drags, 23 of which show marks
elsewhere on the board and are read perfectly well — the shape of a guard that
misfires.

### The recall gap is still the biggest thing in the corpus

Per press, the game marks a mean of **5.95** tsums. The chain the bot proposes
contains **1.42** of the ones outside the glow, and there are **3.98** marked
tsums per drag that it never proposes at all — tsums the game has just stated
are the same character *and* reachable from the one being held, sitting unused
while the chain reaches 200px in the other direction for a member that gets
refused.

This is the same finding as the tenth round's, from the other side: the bot
both over-reaches (29% refused) and under-collects (4 free partners ignored),
and neither is a threshold problem. It is the adjacency question in
`docs/TODO-blob-adjacency.md`, and this corpus supplies both halves of what it
needs — every marked tsum is a positive reachability example and every unmarked
one is a negative. **It remains the best-priced piece of work available.**

### A provenance gap, closed for next time

This collection arrived without a log beside it, and the row's `options` record
the chain-building settings but never recorded `verify_reach` or
`verify_delay`. So whether the reach check was live while these 726 drags were
played cannot be established from the corpus itself. It does not affect
anything above — `proposed` is written before any trim, so the replay is
answering the same question either way — but it is exactly the kind of
"the settings it was taken at were not recorded" gap that cost a full re-decode
of 803MB in the first round. `dataset.py` now records both, so the next corpus
can say.

### What is still open

Unchanged and untouched by this round: **whether the game is lenient.** Every
number above assumes a drag with one refused member clears nothing. A round
with `--verify-clears` records what actually left the board and settles it, and
until it is run, `verify_reach`'s +12% is a well-replicated estimate of
something that might be zero.

## Twelfth round: the colour fit is a coin flip on a quarter of the board

The eleventh round priced decisions. This one went after the three things the
project actually wants to improve -- colour, detection, gameplay -- and found
that the first two are one problem, measured for the first time.

### The measurement

Take one collected frame. Fit its colours three times with nothing different
between the runs but the k-means seed, detect at the same `k` and the same
locked radius each time, and ask how much the three reads agree about which
tsums exist. Over 60 boards:

| fit | two reads agree | count spread | cost per fit |
|---|---:|---:|---:|
| **4 restarts, 20 iterations, eps 1.0 (today)** | **73.1%** | **9.1 tsums** | **51ms** |
| 8, 40, 0.5 | 83.3% | 6.2 | 138ms |
| 16, 60, 0.25 | 91.4% | 3.0 | 349ms |
| 8, 40, 0.5, on 160k pixels instead of 40k | 82.3% | 6.6 | 541ms |

**A quarter of the board is decided by the seed.** On the worst frame two
reads of the same image shared 36% of their detections; the count of tsums on
one board swings by nine between two reads of the same pixels. Everything
downstream inherits that roll: which tsums exist, what colour each is, which
chain is best, and every number the previous eleven rounds measured.

The last row is the one that says what the problem is not. Sampling four times
as much of the board buys nothing -- 82.3% against 83.3% at four times the
cost. It is the number of restarts the fit gets, not how much of the board it
looks at.

### Why the cost objection does not apply

The obvious reason nobody had turned the fit up is that it runs live. It does
not, much: `play_loop` caches its centres and only fits when there is nothing
to reuse -- the first frame, a fever transition, a shuffle, a recalibration.
Counted in this repo's own log, **about five fits in a round**. At the
strictest setting that is ~1.5s added to a round that runs for minutes, and it
is spent precisely on the frames that are hardest to read: `learn`'s docstring
already names the fever transition as "the dimmest and most animated frame of
the round", and that is a frame currently read with four restarts.

Shipped as `fit_effort` (1, 2, 3 -> the first three rows above), default **1**,
opt-in through the tray panel's "Steady colour fit" box or `flows/play.yaml`'s
`vars:` block. Default 1 because every
measurement in this document was taken under it, and moving the default would
re-price the lot.

**What it does not claim.** Stability is not correctness. A steadier fit reads
the same board the same way twice; nothing here says the reading it settles on
is the better one. The collection cannot say either -- every sample in it was
taken at level 1. A round can, and that is the test: `recalibrated (N -> M
tsums)` lines that sit closer together, fewer of them, and `dragged` over a
whole round against a level 1 round.

### What this does to the ninth round's verdict

It does not overturn it, and it is worth being precise about why. The ninth
round asked whether a learned global palette gets *identity* right more often
than the per-frame fit, and the answer was no. Re-fitted on this corpus, the
answer is still no:

```
                     agree   split  balanced
  learned palette    37.8%   79.5%     58.6%
  per-frame k-means  38.9%   80.2%     59.5%   <- what runs today
-> NO BETTER (-0.9 points balanced)
```

But identity was never the palette's only possible use. A fixed palette does
not fit anything, so it cannot be unstable -- it reads the same board the same
way every time, by construction. That is a different claim from the one the
ninth round refused, and it is the claim `fit_effort` now buys most of without
the palette's drawbacks. Whether the remaining gap is worth a palette is an
open question, and one this corpus cannot settle: a palette-based detection
was measured here at 53% of confirmed tsums kept against the per-frame fit's
63%, and both numbers are biased by the fact that the confirmed positions
come from a per-frame fit that was played.

### The bias, stated plainly

`scripts/sweep_detect.py` scores a candidate setting on the tsums the game
confirmed are real. It is the biggest ground truth this project has -- 3,470
confirmed tsums in this collection against ten hand-labelled boards -- and it
has one flaw that cannot be designed out: **the confirmed positions come from
the run that was played**, so the live setting scores 100% for free and every
candidate is measured on tsums the live setting already found. It can catch a
regression. It cannot prove an improvement. The script says so where it prints
the number, and ranks nothing on it.

### Gameplay: the guard the eleventh round asked for

`verify_reach` trims a chain to what the game marked. When that reading comes
back blank -- not one tsum on the whole board over the bar, the pressed one
included -- the loop now drags the chain as proposed instead of trimming on a
reading that failed. Measured over the 726 collected drags: 4.4% of readings
are blank, 6 of the 121 a `verify_reach 260` check fires on, and 2 of those are
chains the trim cancelled outright.

Small, and deliberately shipped without a switch: a blank frame is the game
saying nothing, not the game saying no, and acting on it was never a tuning
choice. `PlayReport` reports the count so the log says how much of the check's
cost bought nothing, and a run where most checks read blank is a capture
problem rather than a chain problem.

### And the measurement the next round owes itself

`verify_clears` is now the panel's "Measure tsums cleared" box, wired through
`flows/play.yaml`, rather than a CLI-only flag. It is not a play rule; it is the one measurement that settles
whether the game clears a partly-refused chain, which is the assumption every
`verify_reach` number rests on.

### Keeping score

`scripts/scorecard.py` prints one row per collection -- colour lift, detection
plausibility, refusal rate -- and appends it to `docs/SCORECARD.md`. The point
is not the numbers, which are all in this document already; it is that they
are computed the same way every time, so two collections can be put beside
each other. The baseline row:

| collected | samples | settings | colour lift | plausible | found | refused | dead drags |
|---|---:|---|---:|---:|---:|---:|---:|
| 2026-09-02 | 726 | k12 link105 fit1 floor8.0 | 1.47x | 98% | 42 | 29% | 29% |

## Thirteenth round: the steady fit held, and the recall gap turned out to be colour

Collected with **steady colour fit ON (`fit_effort` 3) and measure tsums
cleared OFF** -- one switch changed from the twelfth round, as step 1 asks.
321 sessions, 4,306 drags, six times the baseline corpus. The last 28 samples
(`20260903_092108`) were taken after the box was unticked and read `fit1`;
0.6% of the corpus, and nothing below turns on them.

**Both corpora were played with Beast equipped**, stated by the player rather
than read from the data -- nothing in `samples.jsonl` records which tsum was
equipped. That is good for the A/B, because the comparison is not confounded
by a change of character, and it is a limit on everything below: the equipped
tsum decides how the board is filled and which skill fires, so these numbers
are Beast's numbers until a second character says otherwise. **`options`
should carry the equipped tsum**; this is the third provenance gap in three
rounds, after the collection window's play log and the eleventh round's
`verify_reach`.

Three things came out of it: `fit_effort` 3 is now the default, a fever-aware
verify check is dead, and the recall gap -- the biggest open item for three
rounds -- is not the graph problem everyone assumed.

### `fit_effort` 3: the played round the twelfth asked for

The twelfth round measured stability over 60 boards and shipped the switch
off, because "every sample in the collection was taken at level 1" and only a
round could say whether a steadier read is a better one. This is that round.

Re-measured on the new corpus, at k 12, 150 frames, three seeds each:

```
             setting   kept  found  stable  pairs  balanced  live here
    live (as played)   100%   41.1       -      -         -          -
           k 12 fit1    64%   41.7     74%    353     57.2%      57.9%
           k 12 fit3    63%   41.4     93%    348     57.4%      57.9%
```

**The stability figure replicated almost exactly** -- 74%/93% here against
73.1%/91.4% over the 60 boards, on an independent corpus six times the size.
And the columns beside it did not move: detections per board 41.7 vs 41.4,
`balanced` 57.2% vs 57.4% against live's 57.9% either way, `kept` within a
point.

Nothing moved on the scorecard either, except in the right direction:

| | baseline (726 drags, fit1) | this round (4,306 drags, fit3) |
|---|---:|---:|
| colour lift | 1.47x | **1.51x** |
| boards read at a plausible size | 98% | 97% |
| detections found (median) | 42 | 42 |
| over-split boards | 0 | 0 |
| members refused | 29% | 30% |
| dead drags | 29% | **26%** |

A read that is twice as steady, costs ~1.5s a round, and gives up nothing.
**Shipped as the default** -- `fit_effort: 3` in all three flows' `vars:`, and
setting it back to 1 is the whole revert. The panel box went with it: a
settled setting does not want a switch whose off position nobody should
choose, and leaving it there would have been worse than untidy. The tray's
overrides are applied over a flow's `vars:`, so a panel still sending
`fit_effort` would have overwritten the revert on every run started from the
panel -- silently, and only from the panel.

What would falsify it: `found` drifting off 42, an over-split board appearing
(none in 726 + 4,306 drags), or a level 1 round out-dragging a level 3 one.

**What this does not settle.** The `recalibrated (N -> M tsums)` lines the
twelfth round named as the live evidence were not read. The log on the machine
these numbers were computed on ends 2026-09-02 20:45 and resumes at the next
tray start on 09-03 12:39, while the samples run 21:20 to 09:27 -- because
**rounds are played on a separate machine and only the dataset is carried
back**. The log is not lost; it is on the play machine, and nobody fetched it.
The offline replication is strong enough to move the default without it, but
the live half of that test is still owed.

That split is worth stating once, because it decides what a round can measure:
anything that reaches only the log stays on the machine that played it. It is
why `verify_clears` -- whose entire output was a log line -- could not be
handed over as a dataset until schema 3 wrote the count into the sample.

### `k 12` re-confirmed under the steadier fit

Since `fit_effort` changes the fit, the ninth round's `k` had to be re-asked
under it. It survives -- `k 12` keeps the most confirmed tsums (63%, against
56% at k 6, 61% at k 8, 46% at k 16) and sits within half a point of live on
`balanced`. k 16 scores best on `balanced` (58.9%) and loses a quarter of the
tsums to get there, which is the trade the ninth round already refused.

### `verify_reach 260` replicated a third time, and re-priced

Third corpus, third time it holds. Clean rate still falls away with reach --
100% under 90px, 69% at 90-150, 55% at 150-220, 37% at 220-260, 28% at
260-300, 8% past 300 -- and leg length still separates nothing.

Re-priced over 4,306 drags, 260px is best or near-best in all three cost
columns (+28.9% / +18.0% / +7.3% against no check). 220px wins the cheap
column (+31.8%) and loses the dear one, which is exactly why 260 was the
number chosen rather than 220. **No change.**

### A fever-aware check: measured, and it is not a finding

FEVER is 35% of drags in this corpus and the check pays very differently
inside it -- refusals run 34.4% in normal play against 21.4% in FEVER, and
clean drags 48.8% against 60.5%. That looks like a rule: stop paying for
checks while FEVER is running.

It is not. Priced over ten normal/fever threshold pairs at all three costs,
the best fever-aware pair beats a flat 260/260 by +2.3% in the cheap column
and *loses* by 4.6% in the dear one, and exempting FEVER entirely
(`260/off`) is never better at any cost. **A rule whose ranking flips across
the three cost columns is not a finding** -- this document's own test, and
this is the first time it has killed something. `verify_reach` stays flat.

### The recall gap is a colour problem wearing a graph's clothes

This is the round's real result. The gap has been the top open item since the
tenth round, phrased as reachability: the game marks a mean 6.1 partners per
press, the bot's chain holds 4.8, and **4.0 marked tsums per press are never
proposed at all**. `docs/TODO-blob-adjacency.md` treats that as an adjacency
problem to be fixed with a better contact test.

The corpus can test it directly, because a `verify_reach` check has already
pressed and already read the marks. Rebuild the chain from what the game lit
instead of only trimming the proposal to it, over the 806 drags a 260px check
fires on:

| rebuilt from the marks, using... | cleared | mean length | longer on |
|---|---:|---:|---:|
| today: trim the proposal to `kept` | 3,018 | 3.74 | -- |
| the bot's own `kind` ids | 2,133 | 2.65 (**-29%**) | 4% of drags |
| the game's word on identity | 3,353 | 4.16 (**+11%**) | 39% of drags |

(Over the 806 of those 853 drags whose reading returned a mark at all.)

**The same rebuild, on the same tsums, with the same graph, wins or loses
entirely on who is believed about colour.** `adjacency()` will not link two
tsums whose `kind` differs, and `kind` is a per-frame k-means id -- so the
partners the game just named are unreachable to the bot's graph not because
they are too far apart but because the detector called them a different
character. No contact test fixes that.

That reframes the top open item. It also explains why `link_px` could never
predict refusal (tenth round, replicated in the eleventh): it was being asked
a question about distance when the answer was about colour.

### `verify_extend`, and the honest price of it

Shipped as an opt-in play rule: on a check that has already been paid for,
rebuild the chain from the marks rather than only trimming to them.
`chain_from_marks()`, `--verify-extend`, the panel's "Rebuild chains from
marks", off by default.

Against the trim **at an identical reading cost**, over all 4,306 drags --
`scripts/replay_decisions.py` prints this table, calling the shipped
`chain_from_marks` rather than a copy of it:

```
   reading      rule   holds  cleared     time  clears/s  vs trim    >=6
     0.17s      trim     853    10768   1018.0     10.58    +0.0%   6.7%
     0.17s   rebuild     853    11801   1055.3     11.18    +5.7%   9.4%
     0.28s      trim     853    10768   1111.9      9.68    +0.0%   6.7%
     0.28s   rebuild     853    11801   1149.2     10.27    +6.0%   9.4%
     0.41s      trim     853    10768   1222.7      8.81    +0.0%   6.7%
     0.41s   rebuild     853    11801   1260.1      9.37    +6.3%   9.4%
```

Same sign and the same size in all three columns, on the same holds: 1,033
more tsums cleared for 37s more stroke. The marks grew 323 of the 853 checked
chains (38%), and the **6+ column rises with the total** -- 6.7% to 9.4% -- so
this is not the truncation trade wearing a new hat, which the fifth round
closed. A rebuild that cannot beat the trim hands the trim back unchanged, so
the rule can add clears and never subtracts any; what it can cost is a member
that is not really a partner, and only a round prices that.

**The cost that nearly killed it, stated because it is the interesting part.**
The corpus's `marked` was read at 8x the board's noise floor over three
frames: the *collector's* read. A live `verify_reach` check reads one frame at
the flat 8.0 threshold, and at that bar a press admits 23-35 tsums -- a chain
built through those is a chain built through the board's noise. So the rule
has to raise the live bar, and if it also needed the collector's three frames
it would be dead on arrival: priced at the 0.41s column it returns +11.7%
against no check, where today's trim at the 0.17s it actually pays returns
+28.9%. **A candidate that needs a dearer reading has to beat the incumbent's
reading, not its own.**

It survives only because the strict bar is free: `floor_mult` needs the
board's noise floor, which the same single frame already yields. So the rule
raises the bar and keeps the one frame, and `floor_mult` is threaded to move
the *label's* bar only -- the trim keeps its fixed threshold either way, so
arming this cannot silently re-score `verify_reach`'s own A/B. A test pins
that split.

**What would falsify it**, and why it is off:

* The strict bar over one frame is not the reading this was measured under.
  If a single frame at 8x is dirtier than the collector's three, the rebuilt
  chains will contain tsums that are not partners -- a score popup drifting
  across the board reads as a mark, and 13.7% of the board outside the glow
  reads as marked at all.
* It assumes the game accepts a member it **marked**. That is the same
  assumption `verify_reach` rests on, now load-bearing in two places, and
  `verify_clears` is still the only thing that can settle it.

WHAT TO WATCH LIVE: `the marks rebuilt N chain(s) (M% of checks), adding K
member(s)`, printed beside the `checked N chain(s)` cost. If M is near zero
the marks add nothing; if `dragged` climbs and `cleared` does not, the added
members are not partners.

### The provenance gap, closed at the cause rather than the symptom

Three rounds, three versions of the same failure: the eleventh could not say
whether `verify_reach` was armed while it collected, the twelfth bolted
`fit_effort` onto the same hand-written dict, and this one had to be *told*
which tsum was equipped -- while `verify_clears`, the switch the round was
defined by, was never recorded at all.

The cause is not any of those omissions. It is that `options` was a curated
list of a dozen keys, and **the key that matters is always the one nobody
remembered to add**. So the list is gone. Schema 3 records the whole play
namespace (`play_settings()`, 73 keys, ~1.3KB of JSONL against ~90KB of JPEG
per sample), taken at write time so a value the loop clamped mid-round is
recorded as what was used. A flag invented next month is recorded by the same
function with nobody editing it, and a test holds that property rather than
holding a list of keys.

`base` joins it: the equipped tsum, as the median Lab colour of its skill
icon. There is no name to be had -- the cluster id is a per-frame k-means id
and means nothing between sessions -- but the icon is a fixed sprite in a
fixed place, so the same character lands within a few Lab units across
sessions and a different one does not. `scorecard.py` and
`replay_decisions.py` now print the armed switches and the equipped tsum back
from the corpus, and warn when a corpus mixes two settings combinations --
which it immediately did, catching this round's own 28-sample `fit1` tail
without being told to look.

The first thing it caught about the rounds already banked: every scorecard row
so far reads `k12 link105 fit… floor8.0`, and the rounds were in fact played
at `block 1.25` and `max_chain 12` -- neither the CLI default, and neither ever
in a row.

### A tool that could not measure the switch it was asked about

`scripts/sweep_detect.py` fitted at `fit_effort` 1 unconditionally -- the
level is not a parameter of `detect()`'s default path -- so its `stable`
column scored a setting nobody was running, and could not have scored this
round's corpus at all. It now takes `--fit-effort`, and the numbers above are
from it. Worth recording as its own line: the sweep is the instrument, and an
instrument that silently reads one fixed setting will keep confirming it.

### What this round did not settle

* **Whether the game clears a partly-refused chain.** Still the debt, and now
  owed by two rules instead of one. It was the switch left OFF this round.
* **The live `recalibrated` evidence for `fit_effort`**, lost with the play
  log for the collection window.
* **Whether a palette is worth having after `fit_effort` 3.** At 93% stable
  the gap a fixed palette would close is now seven points, and it still loses
  on identity. Close to closed, not closed.
* **The `before` crop**, still not clean: the previous press's glow survives
  into it.
