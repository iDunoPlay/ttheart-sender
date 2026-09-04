# Improving the gameplay from collected data

The loop, start to finish, in the order it has to happen. Each step refuses to
be useful without the one before it, and the last step is what makes the next
round cheaper than this one.

Why this file exists: the loop is run every few weeks and forgotten in
between. `docs/DATASET-FINDINGS.md` is the *record* of thirteen rounds of it --
what was measured and what it settled -- not the recipe. This is the recipe.

The one rule underneath all of it: **a played round overrules an offline
number.** Everything below produces evidence; only a round decides.

## The three things being improved

Every measurement here belongs to one of three, and mixing them is how a round
goes wrong:

| target | the question | measured by |
|---|---|---|
| **colour** | does `kind` mean the same character the game means? | `tsum dataset`, `tsum learn`, `sweep_detect.py` |
| **detection** | are the tsums found, at a believable size, twice running? | `sweep_detect.py --stability`, `tsum eval` |
| **gameplay** | of the chains proposed, which does the game accept? | `replay_decisions.py` |

`scripts/scorecard.py` prints one line for each and appends the row to
`docs/SCORECARD.md`, which is the only place their history lives.

---

## 0. Know what is already settled before spending a night collecting

The rounds in `DATASET-FINDINGS.md` closed these. They stay closed:

* **Colour cannot carry identity across frames.** A global palette
  (`tsum learn`) loses to per-frame k-means on identity, twice measured, and
  more data does not change it -- the same character's face colours spread
  45.6 Lab *within a single frame*.
* **`link_px` does not predict refusal.** Leg length separates a chain member
  the game keeps from one it refuses by 0.06-0.07 standard deviations, on two
  independent corpora. Re-tuning it answers a different question.
* **The label's bar is a multiple of the board's own noise floor**, not a
  fixed number: `floor_mult` 8.0. Under 8x, a tsum that reacts is no more the
  pressed character than the board average is.
* **`--verify-hold` is the wrong shape**, not the wrong idea: the check costs
  the same on every drag and the risk is concentrated in a fifth of them.
  `verify_reach` is the same check bought where it pays.
* **Shortening chains is not an improvement.** Truncation and a lower
  `max_chain` both win on clears-per-second by trading long chains for short
  ones, and nothing offline prices the scoring curve.
* **The recall gap is colour, not adjacency.** The ~4 marked tsums a press
  names and the bot never proposes are unreachable because `adjacency()` will
  not link across a `kind` difference -- rebuild a chain from the marks with
  the bot's own ids and it gets 29% *shorter*; with the game's word on
  identity, 11% longer. A better contact test does not answer it.
* **FEVER does not want its own `verify_reach`.** The check pays very
  differently inside FEVER (21% refusals against 34%), but every fever-aware
  threshold pair flips sign across the three cost columns.
* **A candidate that needs a dearer reading must beat the incumbent's
  reading, not its own.** `verify_extend` scored +11.7% at the three-frame
  cost it was measured under, against the trim's +28.9% at the one frame it
  actually pays for. Same rule, opposite verdict.

## 1. Collect

```yaml
# config.yaml
dataset:
  enabled: true
```

or tick it in the tray. Then **play more than one session** -- sessions are the
unit of holdout, so a corpus of one proves nothing about the next.

| setting | value | why |
|---|---:|---|
| `delay` | 0.25 | under the 0.15s render floor the highlight has not been drawn |
| `frames` / `gap` | 3 / 0.05 | only what changed in *all* frames counts |
| `floor_mult` | 8.0 | the bar the label is judged at |
| `max_motion` | 12.0 | drop samples taken while the board was falling |
| `per_round` / `every` | 20 / 4 | boards inside one round are alike |
| `max_mb` | 2048 | a night writes ~800MB |

~700 drags over 50 sessions prices a play rule. Settling a *threshold* took
11,537. **Turn it off when the round is done** -- it pays a 0.25s hold on one
drag in four.

If you are testing a switch from step 5, change **one** and note it in the
scorecard row's label. Two at once and neither is measured. The exception is
`verify_clears`, which decides nothing about how a round is played -- it is
the *ruler*, and a play rule is best judged with it on. What it does change is
the clock, so never compare a measuring round's `clears/s` to a round that was
not measuring.

**You no longer have to write down what you turned on.** From schema 3 a row
carries *every* play setting in force and the equipped tsum's skill-icon
colour, so steps 3 and 4 print the conditions back rather than trusting a
label. `scorecard.py` names every setting that differs from the defaults, and
`replay_decisions.py` opens with an `armed:` line, an `equipped tsum:` line,
and a warning if the corpus mixes more than one settings combination. The
`--label` is now for *why* a round was played, not for what it was played at.

Schema 2 corpora recorded a curated subset, so both tools mark those rows
`[partial: pre-schema-3]`: an absent setting there is unknown, not default.

## 2. Check the labels before believing anything

```
python -m ttheart_sender.game.tsum dataset --dir dataset --appearance 700
```

* **`-> the marks are in the frames`** -- go on.
* **MARGINAL** -- thin signal, usually too few samples. Collect more.
* **INCONCLUSIVE** -- it could not run the appearance test. Stop.
* **Mean far above the median** (it says so) -- a few samples carry the
  headline. Trust the median.

Schema 1 folders need `--reread` and are only worth their images.

## 3. Take the scorecard

```
python scripts/scorecard.py --dir dataset --append --label "fit_effort 3"
```

Seconds, no images decoded. Three headlines, one row in `docs/SCORECARD.md`,
and the settings baked into the row because a number without them compares to
nothing.

## 4. Replay the decisions -- where the gameplay findings come from

```
python scripts/replay_decisions.py --dir dataset
```

Every row carries `proposed` (the chain the bot chose) beside `kept` (what the
game marked while the first tsum was held): the bot being right and wrong,
drag by drag, with the game as judge. It prints the waste, the clean rate by
reach, per-feature separation, the `verify_reach` sweep, truncation rules,
unreadable readings, the recall gap, and board health.

## 5. Sweep colour and detection -- where those findings come from

```
python scripts/sweep_detect.py --dir dataset --samples 150 --stability        -k 12 --fit-effort 1,3
python scripts/sweep_detect.py --dir dataset -k 6,8,12,16 --fit-effort 3
```

**Pass `--fit-effort` the level the round was played at.** Without it the
sweep fits at level 1 whatever the corpus was collected under, and `stable`
then measures a setting nobody is running -- which is what it silently did
until the thirteenth round.

This scores candidate settings against the tsums **the game confirmed are
real** -- thousands of them, against the ten boards `tsum eval` has. Three
columns, three different questions: `kept` (detection), `stable` (does one
frame read the same twice), `balanced` (colour).

Its one unfixable flaw, stated where it prints: the confirmed positions come
from the run that was played, so live scores 100% on `kept` for free. **It
catches regressions; it cannot prove an improvement.** `stable` is the column
with no such bias, because it compares a setting only against itself.

## 6. How to read any of it without fooling yourself

Every one of these was a mistake made once already:

* **`cleared` decides, `clears/s` prices.** Fifteen 3-chains and eight
  6-chains are not close, and the second scores far better.
* **A ratio has no upper bound.** Means over per-sample ratios describe their
  tail; read the median beside them.
* **Two numbers that trade against each other have no maximum that means
  anything.** Agreement rises as split falls across the whole range of `k`.
* **Compute both sides of a comparison the same way**, on the same population.
  A candidate that finds fewer tsums must not then be scored on the easier
  subset it did find.
* **A guard that fires is not a guard that is right.** Two guards misfired on
  good boards before anyone checked.
* **Do not silently re-score an existing measurement.** `--verify-hold`'s A/B
  was taken under the fixed threshold; `floor_mult` raises the *label's* bar
  only, and a test pins that split.
* **Stability is not correctness.** Reading the same board the same way twice
  says nothing about the reading being right.

## 7. Ship a rule off by default

* a flag on `play` with a default that means "off" (`_TUNABLES` picks it up
  from `play_defaults()` automatically, so a flow can set it);
* a line in `flows/play.yaml` reading a flow variable that defaults to off,
  so the `vars:` block at the top is the revert -- and, when the switch is one
  a round is meant to A/B, a tick box in the tray panel wired to the same
  variable (`PanelSettings` -> `AutomationService._variables()` ->
  `run_flow: vars` all the way down; a gap anywhere and the called flow's own
  default silently wins);
* the numbers that justify it above the line, **including what would falsify
  it**;
* a **WHAT TO WATCH LIVE** line naming the log line that shows cost and
  benefit side by side;
* a test. `tests/test_fit_effort.py` and `tests/test_verify_reach.py` are the
  pattern.

## 8. Prove it on a round

Turn one line on, play, read the end-of-round line:

```
played 176 chains, dragged 741 tsums (mean 4.2/chain), 0 drag(s) did not
register; the game trimmed 12 member(s) and rejected 0 chain(s) outright;
checked 16 chain(s) before dragging (9% of drags)
```

Collect while doing it, take the scorecard again, and put the two rows side by
side. If a rule looks good offline and worse over a played round, the round
wins -- that has already happened once, to `--verify-hold`.

## 9. Write the round down

Append a section to `docs/DATASET-FINDINGS.md`: what was asked, what was
measured, what it settled, and -- the part that pays off later -- **what it did
not**. If a finding contradicts a comment in `flows/play.yaml` or a docstring,
fix the comment in the same pass. Those numbers are load-bearing.

---

## Where this stands, and what to try next

**Leave this one on, permanently:**

* **Measure tsums cleared** (`verify_clears: true`). It was rationed for three
  rounds on a comment that was simply wrong -- it does not cost a capture,
  `--verify` already grabs the frame it reads, and the price is a few disk
  means. From schema 3 it writes `cleared` and `dragged` into the sample, so
  every collection carries the one headline that is not a proxy. There is no
  reason to save it for a special round.

**Confirmed by a played round, leave on:**

1. **Rebuild chains from marks** (`verify_extend: true`) -- **settled by the
   fifteenth round.** Ten rounds, 863 drags, measured with `verify_clears`:
   of the members the rebuild adds, **22 of 22 cleared**, against 87.5% for
   the members already in the chain. The worry it shipped with -- that it
   believes the game about identity as well as about refusal -- was the right
   worry and the answer is that the marks are right. It costs no reading: the
   check was already bought by `verify_reach`.

**Settled, and shipped:** the check is OFF (`verify_reach: 0`). 19 played
rounds, both arms: +22.1% tsums cleared per round from the log, +11.9%
cleared per second from the corpus, chains per round 80.0 -> 97.1, 6+ chains
15.2% -> 24.1%. Cleared per drag was unchanged, so the check was never
improving the drags it fired on -- the 0.25s was dead time and the trim was
cutting good chains short. `verify_extend` follows it off: the rebuild was
never wrong, it just has no reading to ride on now.

**Colour identity is closed, four ways.** A learned palette, `purity`, every
descriptor computable from the face crops, and now `_recolour` re-scored
symmetrically against the game's marks (flat: 55.1% balanced to 55.6%). The
cause is measured: the median tsum shows 0.42 of its own radius and the signal
recovers to 0.636 AUC on the least-buried ones. Stop looking for a better rule
over these pixels; the answer is more pixels. `--recolour` ships off with the
base-kind landmine removed, for the one round that could still surprise us.

**The settings are exhausted.** `verify_reach` (off, +22%), `recolour`
(rejected, -6.5%), `bowl_reject` (kept at 40, 0 costs -9.6%), `block` (flat
0.9 to 2.5), `max_chain` (12, the player's own preference beat 16),
`include_dark` (on, worth 2.2 points in FEVER and nothing outside it). What
is left is not a knob.

**Read a result with `python scripts/rounds.py`** -- one row per round, and
`--settings` names what each was played at. The end-of-round line scrolls past
during play; two rounds here were read late because of it.

**Superseded next step (kept for the reasoning):** every chain setting was tuned while the
check was on, i.e. while a bad chain was trimmed before it was dragged.
Nothing trims now. Re-sweep `link_px`, `block`, `max_chain` and `min_chain`
one at a time against the new baseline.

**Superseded -- do not quote its number:** `verify_reach 260`. The rule is
sound and cheap, but its +27%/+17%/+7% was computed against a baseline
modelled as clearing *zero* on any chain with a refused member. The fifteenth
round measured that baseline directly and it clears 70-93% of its members: a
refusal costs its own clear, not the drag. The benefit is overstated by an
amount nobody has derived, and the sweep that chose 260 ran under the same
model. `replay_decisions.py` has been corrected -- it prints
`all-or-nothing` and `per-member` side by side rather than replacing one with
the other -- so what is owed now is a corpus to run it over, not a code
change. Take the sweep on the next collection before quoting any number for
this rule.

**Confirmed, no action:** `fit_effort 3` (the default -- stability 74% -> 93%
replicated on a second corpus; note that three corpora now show it moves no
outcome, so the claim is "steadier, and free", not "better"), `verify_reach:
260` (+27%/+17%/+7% across the three cost columns, replicated on four
corpora), `floor_mult 8.0`, `k 12`, and over-splitting, which has not occurred
once in 2,541 boards.

**Do not re-derive from a label:** which switches were on, which build played
it, and which tsum was equipped are read off the corpus by steps 3 and 4. If
they say `[partial: pre-schema-3]`, the corpus predates that and an unnamed
setting is unknown rather than default -- which is how a round was collected
with the colour fit silently back at level 1.

**Open, in the order they are worth doing:**

1. **Reachability, re-aimed at colour -- and now at the capture.** The cheap
   way out is closed: `scripts/identity_probe.py` scores every descriptor
   computable from the face crops against the game's own marks, and none
   beats plain median Lab on held-out sessions (0.561 AUC; shape 0.510; a
   learned linear metric no better). The reason is occlusion, not absence --
   restricted to the least-buried tsums the same descriptors reach 0.636,
   and the median tsum shows only 0.42 of its own radius. So this is a
   question about capture resolution before it is a question about a model.

2. **Reachability, the original framing.** Still the biggest lever, but the
   thirteenth round moved where it points: the ~4 marked tsums a press names
   and the bot never proposes are missed because `adjacency()` will not link
   across a `kind` difference, not because they are too far apart. Rebuilt
   with the bot's own ids a chain gets *shorter*. So the work is making
   `kind` agree with the game about identity -- or letting a chain cross a
   `kind` boundary on evidence -- and `docs/TODO-blob-adjacency.md`, which
   attacks the contact test, does not answer it. Every unmarked tsum in the
   collection is still a free negative example.
2. **Whether stability is worth a palette after `fit_effort` 3.** The gap a
   fixed palette would close is now seven points rather than twenty-seven,
   and it still loses on identity. Nearly closed for good.
3. **`floor_mult` above 8** -- only worth re-sweeping against something
   actually trained on the corpus.
4. **The `before` crop is not clean.** The previous press's glow survives into
   it and the reading charges its disappearance to the wrong tsum.
5. **Fetch the play machine's log when a finding needs one.** Rounds are
   played on a separate machine and only `dataset/` is carried back, so
   anything that reaches only the log -- `recalibrated (N -> M tsums)`, the
   end-of-round line -- stays there. From schema 3 the numbers that decide
   live in the samples instead, which is the durable fix; until then, copy
   `logs/ttheart.log` off the play machine before it rotates.
