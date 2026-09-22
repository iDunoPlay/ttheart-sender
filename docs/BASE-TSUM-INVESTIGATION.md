# Why the base tsum changes how well the bot plays — 2026-09-06

*Requested by `NEXT_TASK_BASE_TSUM_SELF_CALIBRATION.md`. Investigation only; no
gameplay code was changed. Reproduce with `scripts/base_probe.py`.*

**The observation is real and has a mechanism, and the mechanism has nothing to
do with Beast being Beast.**

---

## A. Why Beast works better

The base tsum is **not a character** anywhere in this code. `read_base_kind`
takes the median Lab colour of the skill icon in the corner of the screen and
returns `argmin` over the board's 12 k-means palette centres — **the nearest
colour cluster**. `find_chains` then flags chains of that cluster `is_base`, and
chains sort by `(is_base, len)`, so a base chain is played ahead of every longer
chain.

That is the whole mechanism: **one colour match, used to reorder candidates.**

So the failure mode is a colour match. And `read_base_kind` returns `argmin`
with **no threshold**. Its own docstring says:

> *"Returns (cluster index, Lab distance). A large distance means no cluster
> really matched and the caller should not trust it."*

The play loop never checks the distance. The `tsum board` CLI does — it prints
`(WEAK -- check --base)` above a distance of 30 — but the code that actually
plays rounds takes `argmin` unconditionally, every frame, forever.

**Beast works better because Beast's colour reliably wins that `argmin`.** Any
equipped tsum whose icon colour is not distinctly closest to one board cluster
gets silently mapped to the wrong one, and the bot then prioritises the wrong
colour group on every frame of the round — paying the cost of the preference and
getting none of its benefit.

---

## B. The exact code path

    skill icon patch  (BASE_ICON, or the layout's "base" spec)
        -> median Lab of a centre disc            read_base_kind, tsum.py:1599
        -> argmin over the 12 palette centres     tsum.py:1654   <-- NO THRESHOLD
        -> base_kind                              play loop, tsum.py:3689
        -> Chain(is_base = kind == base_kind)     find_chains, tsum.py:2037
        -> chains.sort((is_base, len), reverse)   tsum.py:2038, 2061
        -> best = chains[0]

    detection      NOT touched by the base
    features       NOT touched by the base
    grouping       NOT touched by the base
    chain ranking  THE ONLY STAGE THE BASE TOUCHES

Two side paths exist and lead to the same place: `_base_from_faces` re-derives
the base from the icon's Lab colour when `--recolour`, `--kinds` or the
character model have renumbered `kind`. Same colour comparison, different
numbering.

---

## C. Beast vs other bases — the comparison cannot be made

    737 sessions, 10,108 samples with an icon colour
    spread of the SESSION medians:  1.0 Lab units

**Every recorded round in this corpus was played with the same equipped tsum.**
There is no data for Mickey, Donald, Rex or Randall, or for any base but this
one, so the player's comparison cannot be reproduced offline at all. That is the
single most important practical fact in this report: the observation is not
dismissed, it is **untested, because nothing has ever recorded another base.**

> A first pass reported "more than one equipped tsum" from a per-sample spread
> of 116.8 Lab units. That was the wrong statistic for the question: it mixes
> across-session differences with within-session noise. The session medians
> agree to 1.0 unit and the within-session spread is also 1.0 unit. Both are
> now computed and printed separately, and `tests/test_base_probe.py` pins the
> distinction.

---

## D. First point of divergence — chain ranking, and nothing earlier

The same 120 stored boards, re-run under **every possible base cluster** and
under no base at all:

    candidate chains IDENTICAL under every base :  100.0% of boards
    chain CHOSEN changes with the base          :   62.5% of boards
    chosen length with no base                  :   4.77 tsums
    lost by pointing the base at the WORST cluster: 1.02 tsums

The candidate set never moves — not the detections, not the groups, not the
links. **The base changes only which candidate is played.** A base pointed at
the wrong cluster costs about **one tsum per drag**, on 62.5% of boards.

### What the preference costs when it is working

Within a single build, comparing drags the bot played on a base chain against
drags it played on any other chain:

    build       n     base%   clr base   clr other      diff
    1.11.7b   4209    17.6%      2.95        3.22    -0.27 +/- 0.10  REAL
    1.11.1e   3452    16.3%      2.91        3.22    -0.31 +/- 0.11  REAL
    1.11.6c   1975    19.1%      2.99        3.21    -0.22 +/- 0.13  REAL

Consistent across three independent builds: **a base chain clears ~0.27 fewer
tsums than the chain the bot would otherwise have played**, on the ~18% of
drags where the preference fires. That is the price of the rule, and it is paid
in exchange for charging the skill gauge.

### Why the round-level numbers must not be read as the benefit

Rounds with a higher share of base drags clear more (+21.7 cleared) and reach
FEVER far more (+8.0pp, outside 2 s.e.). That looks like proof the preference
works. **It is not**, and the reason is visible in the same table: base share
also correlates positively with chains played (+0.243) *and* with cleared per
chain (+0.121) — including the very quantity the drag-level test shows it
*lowers*. A round that reaches FEVER early has a denser, faster-refilling board
for longer, which raises the base share as a consequence. **The causation
plausibly runs backwards**, and this corpus cannot separate the two.

---

## E. The failure is in the palette, not the icon

    icon Lab distance to the matched cluster:  median 14.7  p90 45.6  max 65.7
    above the CLI's own WEAK line of 30:       24.0% of samples

Yet the icon itself is read **steadily**: the spread within one session is 1.0
Lab unit, on a fixed sprite in a fixed place.

So the icon is not moving — **the board's palette is**. The 12 clusters are
re-derived per board, and on nearly a quarter of frames none of them lands near
the icon's colour. `argmin` still returns one.

**With Beast equipped, the base priority is already pointing at an arbitrary
cluster on ~24% of frames.** For a base whose colour is rarer on the board, or
closer to the background, that share can only be higher. This is the concrete,
measured form of the player's observation.

---

## F. The five-character constraint

It can help, but not as a grouping constraint — `docs/OCCLUSION-INVESTIGATION.md`
measured forcing 5 groups at **−23.3%**, with a knowledge score of 1.00x.

Where it *can* help is exactly here: the board has at most 5 characters, so the
icon should match **one of at most five well-separated face colours**, not one
of twelve k-means clusters fitted to every pixel on the board. A match against
the five face-colour groups is a far better-posed question than a match against
a palette that includes outlines, shadows and background.

---

## G. Self-calibration and online self-training

* **Self-calibration is already in the code and is the right shape.**
  `_base_from_faces` matches the icon colour against the *faces on the board*
  rather than the raw palette. It runs only when `--recolour` or `--kinds` is
  on. Making it the primary path is a small change and directly targets §E.
* **Online self-training should not be built.** The project already has the
  decisive precedent: `docs/IDENTITY.md` §5 harvested 614 buried crops using the
  game's own marks — a far stronger label than a model's own guess — and it had
  **no effect**. A loop that trains on its own predictions has strictly less
  information than that and a way to reinforce its own errors.

---

## H. Detection churn

Measured in `docs/OCCLUSION-INVESTIGATION.md`: away from the game's highlight, on
a board that moved 1.41px, **24.4% of detections fail to reappear 0.25s later**.
Cause not yet isolated. It is a plausible contributor to the palette instability
in §E, since both are downstream of the same per-frame k-means fit — but that
is a hypothesis, not a measurement.

---

## I. Single best next experiment

**Play about 20 rounds with a different tsum equipped. No code change is
needed.**

The collector already records `base.lab` and `base.distance` on every sample, so
those rounds will answer directly:

* does the icon match a cluster as well for another tsum as it does for this one?
* what share of frames exceed the WEAK line of 30?
* does the base preference still fire at ~18% of drags, or does it point at a
  group that is not the equipped character at all?

It is the cheapest possible experiment — **20 minutes of play, zero
engineering** — and nothing else can be decided without it, because §C shows
there is currently no data on any base but one.

**The fix it would license**, once the failure is confirmed: refuse the base
preference when the match is weak. `read_base_kind` already returns the
distance and already documents that a large one should not be trusted; the play
loop simply has to honour it, and fall back to plain longest-chain when it
cannot tell. That generalises to any equipped tsum, which is the stated goal,
and it ships off with a one-line revert like everything else here.

---

## J. Files

    inspected : ttheart_sender/game/tsum.py (read_base_kind, _base_from_faces,
                find_chains, BASE_ICON/BASE_INDEX, the play loop's base block),
                flows/play.yaml, dataset/*/samples.jsonl, dataset/*/round.json
    changed   : none
    added     : scripts/base_probe.py       the three measurements above
                tests/test_base_probe.py    6 tests
    tests     : 604 pass (598 + 6)

Chain generation, chain ranking, the acceptance model, drag execution and timing
are unchanged. The character model and the chain ranker both still ship off.

---

*Companions: [OCCLUSION-INVESTIGATION.md](OCCLUSION-INVESTIGATION.md),
[RECOGNITION-AUDIT.md](RECOGNITION-AUDIT.md), [IDENTITY.md](IDENTITY.md).*
