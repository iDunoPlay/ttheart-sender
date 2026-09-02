# Improving the gameplay from collected data

The loop, start to finish, in the order it has to happen. Each step refuses to
be useful without the one before it, and the last step is what makes the next
round cheaper than this one.

Why this file exists: the loop is run every few weeks and forgotten in
between. `docs/DATASET-FINDINGS.md` is the *record* of twelve rounds of it --
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
scorecard row's label. Two at once and neither is measured.

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
python scripts/sweep_detect.py --dir dataset --samples 150 --stability
python scripts/sweep_detect.py --dir dataset -k 6,8,12,16
```

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

**Ready to try -- both are tick boxes in the tray panel, under "Data
collection":**

1. **Steady colour fit** (`fit_effort: 3`) -- **colour and detection.** The per-frame k-means is a coin
   flip on a quarter of the board: two reads of one frame at the same `k` and
   radius agree on 73.1% of tsums and their counts differ by nine. At level 3
   that is 91.4% and three, for ~1.5s a round, because the loop only fits about
   five times. Watch the `recalibrated (N -> M tsums)` lines close up.
2. **Measure tsums cleared** (`verify_clears: true`) -- **gameplay, as a
   measuring round.** Everything
   `verify_reach` claims assumes a drag with one refused member clears
   *nothing*. If the game is lenient, the check buys delay and no clears. No
   collection can settle it; one round with this on can. Do this before tuning
   `verify_reach` any further.

**Confirmed, no action:** `verify_reach: 260` (+12% clears/s at a realistic
check cost, replicated on two corpora), `floor_mult 8.0`, `k 12`, and
over-splitting, which has not occurred once in 726 boards.

**Open, in the order they are worth doing:**

1. **Reachability.** The game marks ~6 partners per press; the bot's chain
   contains ~1.4 of the far ones and never proposes ~4 tsums the game just
   said were legal. `docs/TODO-blob-adjacency.md`, and every unmarked tsum in
   the collection is a free negative example for it. Biggest lever available.
2. **Whether stability is worth a palette after `fit_effort`.** A fixed
   palette is 100% stable by construction but loses on identity. If level 3
   buys most of the stability, the palette question is closed for good.
3. **`floor_mult` above 8** -- only worth re-sweeping against something
   actually trained on the corpus.
4. **The `before` crop is not clean.** The previous press's glow survives into
   it and the reading charges its disappearance to the wrong tsum.
