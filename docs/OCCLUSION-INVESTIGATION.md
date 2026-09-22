# Occlusion and multi-frame tracking — 2026-09-06

*Requested by `NEXT_TASK_OCCLUSION_MULTI_FRAME.md`. This is the investigation,
and it stops where that document says to stop. No gameplay code was changed.*

**Result in one line: the pile does not move, so there is no second look to
take. The preferred first experiment is refuted on data already on disk.**

---

## A. Current detection system

| | |
|---|---|
| **How** | `detect()` at `tsum.py:1347`. k-means quantises the board into `k=12` colour clusters, background clusters are dropped, dark pixels are split off and handled separately (`include_dark=True` in play), then each colour mask is holes-filled and a distance transform finds tsum centres. |
| **Centre** | The peak of the distance transform of that colour's mask — the centre of the largest inscribed disc, not a bounding-box centre. |
| **Bounding box** | There isn't one. A tsum is `(x, y, r)`. |
| **Radius** | **`t.r` is the VISIBLE extent, not the true size.** The board-wide `radius` is estimated as `percentile(dt, 99) / 0.9` — the tail is dominated by fully visible tsums. So `visibility = t.r / radius` is the fraction of a tsum that is showing, and it is the only occlusion measure the project has. |
| **Duplicate detections** | Guarded: touching tsums sit ~2.44r apart and de-duplication uses that. |
| **Disappear / reappear** | **Yes, and more than expected — see B.** |
| **Frame capture** | On demand, at decision time. There is no continuous capture and no recording mode. |
| **Frames before a decision** | **One.** Measured over 738 rounds: median 111 frames in 75.6s = **1.46 frames/s**, against ~65 presses. The bot grabs a frame, decides, drags. There is no spare frame lying around. |
| **Timestamps** | Yes, `time` per sample in `samples.jsonl`. |

The crop the classifier sees comes from `_character_crop`, a fixed square of
half-width 1.0r about the detected centre — so for a buried tsum it is mostly a
picture of the neighbours lying on top of it.

---

## B. Is there frame data, and can a tsum be tracked?

### There are no frame sequences in this corpus

744 session folders, 738 with `samples.jsonl`, ~10,200 samples, 13.8 per round.

    gap between consecutive SAMPLES:  median 3.87s   min 1.98s
    share of gaps under 1.0s:         0.00%

Between two stored samples the bot has played several chains and the pile has
collapsed. **They are not a sequence.** No video, no recording mode, and the
3-frame burst that collection takes at 0.05s spacing to read the marks is used
and discarded, never written.

**The one genuine pair is `NNNN_before.jpg` → `NNNN_marked.jpg`**, the same
board ~0.25s apart across a hold. That is what everything below uses.

### Tracking works, because there is almost nothing to track

`scripts/track_probe.py`, 200 samples over 179 sessions, `fit_effort=3`, both
frames detected by the same offline pipeline using the round's own options and
a shared palette:

    matched within 1.0 radius     control 100.0%     +0.25s  70.6%
    median displacement (px)              0.00               1.41
    p90 displacement (px)                 0.00              15.13
    ambiguous (runner-up < 1.5x)           0.0%               0.5%

The control is the same frame read twice: **100% at 0.00px**, so the detector is
deterministic given a frame and a palette, and every pixel of movement in the
test column is the frame's content changing.

**Median displacement over 0.25s is 1.41px, on a board where tsums sit ~61px
apart.** Ambiguity is 0.5%. Nearest-neighbour matching is not merely adequate,
it is overkill — there is no tracking problem to solve.

> A first version of this compared re-detections against the detections the bot
> stored at play time and reported a 72% "control". That is a comparison of two
> pipelines, not of two frames: the play loop caches its palette across frames,
> and `include_dark=True` at play against the function default `False` alone
> dropped every black tsum. Both are fixed; the 100% control is what the
> corrected design gives.

### The 29.4% that do not match is mostly the game's own highlight

    tsums WITHIN 2 radii of a marked tsum :  44.5% matched  (n=1,300)
    tsums AWAY from every mark            :  75.6% matched  (n=6,851)

The game paints its link highlight over the marked tsums in this very frame,
and detection is built on colour, so the segmentation near a mark is wrecked.
Away from the marks, **24.4% of detections still fail to reappear 0.25s later on
a board that moved 1.41px.** That is real churn — a quarter of the board is not
a stable object between frames — and it is a finding about *detection*, not
about tracking. It is an upper bound: the second frame is a separate JPEG
encode and the game animates its tsums.

---

## C. Visibility recovery — the number that decides the idea

Over 5,461 unmarked tsums, change in `r / radius` across 0.25s:

    p10  -0.024      median  +0.000      p90  +0.024
    more visible 32.3%    less visible 31.7%    unchanged 36.0%

**The median tsum's visibility does not change at all.** The distribution is
symmetric noise about zero, ±0.024 at the deciles. Nothing is being revealed.

And the question the task names as decisive:

> **Buried tsums (below 0.55 visible) that rose above it 0.25s later:
> 145 of 4,186 = 3.5%**

Take that at face value and be generous — assume every 0.25s step is an
independent 3.5% shot, which it is not, because a tsum at the bottom of the pile
stays at the bottom:

    to make half the buried tsums readable:  ~19 steps  =  ~4.8 s per decision

A round is 75s and contains ~65 decisions. This is not a tuning cost, it is two
orders of magnitude. And the true figure is worse than the bound, because the
pile only genuinely rearranges when a drag clears the tsums lying on top — at
which point the board is a different board and the old identity is not wanted.

---

## D. Single-frame vs best-visible vs aggregation

`scripts/multiframe_eval.py`. Ground truth is the folder a person put the crop
in; the tsums the game marked are excluded, since their second observation is a
picture of the highlight.

    390 crops from 156 samples
    second observation more visible for 28.2% of them (median gain -0.024)

    visibility band          n     A single   B best-vis     C mean
    high  (>=0.70)          79        96.2%        94.9%      97.5%
    medium(0.60-0.70)      234        99.6%        99.6%      99.6%
    low   (0.55-0.60)       77       100.0%        98.7%      98.7%
    ALL                    390        99.0%        98.5%      99.0%

    B changed the answer on 2 crops of 390; C on 6.

**No gain. B is slightly worse; C is identical.** The mechanism is in the
`-0.024` median: the second observation is usually no better, so "best visible"
picks the same frame 72% of the time and, when it does switch, switches on noise.

**Honest limit of this table:** every human-labelled crop is ≥0.55 visible, so
single-frame is already at 99% and there is no headroom to win. It cannot show
that multi-frame helps *buried* tsums. But §C answers that directly and without
labels: buried tsums do not become visible, so there is no better observation
for any method to find.

---

## E. Group integrity

This is the part that matters more than face accuracy, and it was already
measured. `scripts/group_eval.py`, 378 paired drags, against the game's marks:

    method    agreement    base    lift   ids/board   cleared   vs kmeans
    kmeans        42.6%   27.0%   1.58x      7.29       3.21    --
    named         35.4%   22.3%   1.59x      8.37       2.90    -9.6%  REAL
    merge         42.7%   27.0%   1.58x      7.29       3.21    +0.1%  noise
    colour4       48.6%   32.0%   1.52x      4.00       3.34    +3.9%  REAL
    colour5       44.1%   28.0%   1.57x      5.00       3.30    +2.7%  REAL
    model5        73.2%   73.1%   1.00x      3.90       2.46   -23.3%  REAL
    both5         73.3%   73.2%   1.00x      3.90       2.48   -23.3%  REAL

Two things kill the whole identity route, and neither is about accuracy:

1. **`named` puts 8.37 ids on a board where the game allows 5.** Naming only the
   readable fifth splits one character into a character id for its visible
   members and a colour id for its buried ones. **−9.6% cleared.**
2. **`model5` — the model's own output forced into the game's 5 groups —
   has lift 1.00x.** It knows *nothing* beyond the sizes of the groups it made.
   The fingerprint is undefined for the ~78% of the board the model never reads,
   so forcing 5 groups just spreads that noise over the whole board: **−23.3%.**

So the fragmentation remedy that already exists in the code does not work, and
better recognition of buried tsums would have to reach nearly the *whole* board
before it stopped making grouping worse. Partial progress is negative progress.

*(The `cleared` column is a simulation, and `docs/IDENTITY.md` §10 records that
this metric was once disqualified by an ORACLE scoring below the shipped rule.
`lift` and `ids/board` are counts and are the load-bearing numbers here.)*

---

## F. Performance

    detect(), fit_effort=3, cached palette   104 ms per frame   (600 fits timed)
    character model, 46-tsum board            12-14 ms
    live capture rate                         1.46 frames/s
    marked share of a board                   14.3% median

An extra observation is not free and is not one frame's worth of latency: the
bot would have to **wait** for the board to change, then pay another ~104ms
detect and ~13ms recognition. §C says the wait is seconds, per decision.

---

## G. The four approaches, ranked

| | approach | expected benefit | difficulty | CPU | new data? | verdict |
|---|---|---|---|---|---|---|
| **C** | multi-frame tracking | **measured: none** — visibility median change +0.000, 3.5% recovery per 0.25s | low | +104ms/frame **plus seconds of waiting** | frames are not recorded | **refuted, do not build** |
| **D** | true-radius detection | would make `visibility` a mask rather than a scalar, and centre the crop on the tsum instead of the visible crescent | high — the radius *is* the measurement | medium | yes, a labelled true-extent set | plausible, expensive, unproven |
| **A** | crescent-following crop | the only one that changes what the model *sees* without new pixels; today's crop of a buried tsum is mostly its neighbours | **low** — a crop-window change plus a retrain | ~0 | no, existing crops re-cut | **cheapest technically promising** |
| **B** | mask the neighbours | same idea, stronger; the detector already has per-tsum masks to do it with | medium | small | no | promising, after A |

C is refuted by measurement. A and B are the same bet at different prices, and
A is much the cheaper.

---

## H. Recommendation — and it is not an occlusion experiment

**Do not build any of the four yet.** §E is the blocker, and it applies to all
of them equally: every one is a way to name more of the board, and naming *more*
of the board is only worth anything if it reaches nearly *all* of it. At 22%
named the bot loses 9.6%. There is no evidence anywhere in this project that the
curve turns back up, and `model5` at lift 1.00x is evidence that it may not.

So the single cheapest experiment with the highest chance of improving real
gameplay is the one that tests that assumption directly, and it needs no new
model, no new data and no rounds:

> **Measure the fragmentation curve.** Using the marks as ground truth, simulate
> naming a controlled share of each board *correctly* — 20%, 40%, 60%, 80%,
> 100% — and plot `ids/board` and cleared-per-drag against it. This is a small
> extension to `group_eval.py`, which already has the boards, the marks and the
> scoring.

It answers, in one offline run, the question every occlusion approach depends
on: **how much of the board must be named before naming helps at all?**

* If the curve only turns positive above ~90%, then A, B and D are all dead on
  arrival and the identity route should be closed for good.
* If it turns positive around 50-60%, that is a concrete target, and **A
  (crescent crop)** becomes the right next build with a number to beat.

The second candidate, if you would rather spend the run on something that could
ship: **`kinds: 4` is already implemented, already in `flows/play.yaml`, and
already off.** `colour4` is the only intervention in the whole grouping table
with a real positive effect (+3.9%), it needs zero engineering, and the only
reason it has never shipped is that the metric scoring it was disqualified —
which a played A/B is exactly the thing to settle.

---

## I. Files

    inspected : ttheart_sender/game/tsum.py (detect, CharacterModel, _character_crop,
                _quantise, FIT_EFFORT), ttheart_sender/game/dataset.py,
                scripts/crops.py, scripts/classify.py, scripts/group_eval.py,
                flows/play.yaml, dataset/*/samples.jsonl, dataset/*/round.json
    changed   : none
    added     : scripts/track_probe.py       tracking + visibility-recovery probe
                scripts/multiframe_eval.py   A / B / C recognition comparison
                tests/test_track_probe.py    6 tests
    tests     : 598 pass (592 + 6)

Gameplay is untouched: chain generation, chain ranking, the acceptance model,
drag execution and timing are all unchanged, and both the character model and
the chain ranker still ship off.

---

*Companion documents: [IDENTITY.md](IDENTITY.md) closes the identity line;
[RECOGNITION-AUDIT.md](RECOGNITION-AUDIT.md) measures the classifier;
[DATASET-FINDINGS.md](DATASET-FINDINGS.md) is the round-by-round record.*
