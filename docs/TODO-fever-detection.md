# FEVER: why the bot went quiet for ten seconds

Status: **diagnosed, fixed, opt-in, enabled in `flows/play.yaml`
(`radius_lock: 5`, `fever_min_tsums: 12`), awaiting a live round.** Both code
defaults are `0` — off — so deleting either line is the whole revert.

Reported from play: *"whenever in fever, the whole board is black and I can see
it clearly, the detection goes worse until fever ends without a single fan
spinned."* Both halves of that turned out to be one bug, and the giveaway was
the second half: the fan is the recovery, and the recovery not firing means the
loop never believed anything was wrong.

## The mechanism

1. FEVER starts. `play_loop` sees the transition and drops its cached fit:
   `palette, radius, base = None, opts.radius, None`. Dropping the palette and
   the base is right — both are colour-bound, and FEVER repaints the board.
   **Dropping the radius is not.** FEVER does not resize the tsums.
2. So the radius is re-measured, on the dimmest and most heavily animated
   frames of the round — precisely where the estimator fails.
3. It collapses. Over 151 captured frames of one round the per-frame estimate
   ranges **8px to 38px on a board whose faces measure ~26px**, and 55% of
   frames read under 18px.
4. A radius at half the truth reads the board at half scale, which roughly
   **doubles** the detection count — 52 detections at r=11 where 19 sit at
   r=24.
5. **A doubled count is still inside `--min-tsums`..`--max-tsums`.** 50 of the
   83 collapsed frames pass the 20..110 gate. The frame is accepted.
6. Phantom chains are found on a phantom board and dragged. Nothing clears.
7. And because chains *were* found, `misses` never increments, so the
   `misses >= --max-misses` recovery — the one that taps the fan — never runs.

The result has no symptom. No error, no `skip` in the log, no shuffle: just ten
seconds of the bot working hard on a board that isn't there.

## What was measured

Frames come from the 151 in-play captures in `scratchpad/fever*.png`
(18 of them dim enough to be FEVER proper, by board-crop brightness).

| | reading |
|---|---|
| per-frame radius estimate, all frames | p10 **8.0**, p50 16.7, p90 25.3 |
| a face measured off the pixels with a ruler grid | **~26px** |
| frames reading under 18px | 83/151 (**55%**) |
| ...of those, accepted by the count gate anyway | **50** |
| frames the count gate rejects outright | 37 |

Two fixes fall out, and one idea that looked obvious and does not work.

### 1. `radius_lock` — measure once, then stop

Take the radius over the first N frames that look like a board and hold it for
the round. The lock is the **maximum** of the samples, not the median: the
estimator fails in one direction only — it collapses when a mask comes out
fragmented, and `open_ratio` already guards against inflation — so the median
inherits the collapse.

| warm-up | summary | lock p10 | p50 | p90 | locks under 20px |
|---|---|---:|---:|---:|---:|
| 3 frames | median | 14.6 | 20.2 | 24.4 | 44% |
| 3 frames | **max** | 20.0 | 24.0 | 27.5 | 9% |
| 5 frames | **max** | 22.2 | 24.6 | 28.0 | **2%** |
| 8 frames | max | 23.3 | 25.5 | 31.3 | 0% |

`5` ships. Also plugged: the refit branch and the shuffle recovery both used to
re-open the estimate, so a collapsed radius had two more ways back in.

### 2. `fever_min_tsums` — the count floor has to give way

With the radius correct, FEVER frames genuinely hold fewer readable tsums — the
board is faded and half-covered by the FEVER BONUS and COMBO overlays. At a
locked 24px the dim frames read a median of 20 detections, i.e. sitting exactly
on the default floor:

| floor | dim frames playable |
|---|---|
| 20 (the default) | 5/18 |
| 15 | 7/18 |
| **12** | **10/18** |
| 10 | 13/18 |

`--min-tsums` is not a quality bar; it is a "this frame is not a board" test,
and during FEVER that job is already done — and done better — by the template
`FeverWatch` matched, because the game does not draw a FEVER meter over a menu.
So the floor can be lowered there without giving up what it protects. 12 keeps
clear air above the true fade-to-black frames, which read 0, 1 and 4.

### 3. Restoring the contrast does not help — measured

The obvious reading of "the board is black" is that FEVER crushes the contrast
and the quantiser cannot separate colours any more. Both standard repairs were
tried on the crop before `detect` saw it:

| treatment | dim frames: radius p50 | within 20% of truth | gate accepts |
|---|---:|---:|---:|
| none | 17.9 | 8/18 | 10/18 |
| linear L stretch | 15.8 | 5/18 | 12/18 |
| CLAHE on L | 17.1 | 2/18 | 10/18 |

Both are losses on the measure that matters. The board is dim but its colours
are still separable — k-means was never the thing failing. Recorded so this
does not get proposed again.

## Not FEVER-specific

Worth stating plainly: 55% of *all* frames read a collapsed radius, not 55% of
FEVER frames. Normal play carries the same defect and mostly gets away with it,
because the radius is measured once on frame 1 and held — a round that happens
to start on a good frame plays fine, and one that starts on a bad frame is
quietly worse for its whole duration. FEVER only made it visible by forcing a
re-measurement at the worst possible moment. `radius_lock` fixes both.

## Still open

1. **A live round.** The blocking item. What to watch: the log now prints
   `radius locked at NNpx` once per round — under ~20 means the warm-up caught
   a bad patch — and chains actually being dragged between the `FEVER` and
   `NORMAL` lines rather than a run of `skip`s.
2. **Whether the lock should re-open at all.** It currently never does. If the
   emulator is resized mid-round every frame afterwards is wrong, where today
   it would recover on the next refit. Nothing in `play_loop` survives a resize
   anyway (`LAYOUTS` is keyed by capture size), so this is theoretical until it
   is not.
3. **The estimator itself is untouched.** `_estimate` takes the 99th percentile
   of the distance transform over `0.9`, and on a fragmented mask that
   percentile is measuring fragments. The lock routes around it rather than
   fixing it. A direct fix would help every frame, not just the ones after the
   warm-up — see `docs/TODO-bowl-reject.md` for the other half of the
   detection-quality picture.
4. **`fever_min_tsums` at 12 has headroom.** 10 would make 13 of 18 frames
   playable. Whether the extra three are worth the risk of reading a fade as a
   board is a question for the round, not the corpus.
