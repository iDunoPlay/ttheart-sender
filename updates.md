# Updates

Running log of what each working session found and what it changed. Newest
section last, each stamped with the date and time it was written. The rules it
works under are in [docs/IMPROVEMENT-LOOP.md](docs/IMPROVEMENT-LOOP.md); the
round-by-round record is [docs/DATASET-FINDINGS.md](docs/DATASET-FINDINGS.md).

---

## 2026-09-05 20:15 — The first clean A/B this project has ever had

143 rounds arrived, **72 ON / 71 OFF, all on one build (1.11.6c)**, with the
arm stamped into every `round.json`. That is the thing four documents in a row
could not get: rounds 34, 35 and 36 all end with "`reject_model` is still
unmeasured", and the only previous reading was 13 rounds of 1.11.5 against 244
of 1.11.1e, confounded by the build.

The metric, the threshold and the decision rule were fixed **before** these
rounds were played (`docs/DATASET-FINDINGS.md`, the pre-registration written at
15:10 today), so nothing below was chosen after seeing the numbers.

### The verdict: REVERT

| | ON (n=72) | OFF (n=71) | diff | p |
|---|---:|---:|---:|---:|
| **cleared** (pre-registered primary) | 256.9 | 275.4 | **−18.5** | **0.036** |
| FEVER share | 41.2% | 47.1% | **−5.9pp** | **0.030** |
| score | 514,851 | 556,170 | −41,319 | 0.457 |
| coins | 527 | 577 | −49 | 0.370 |
| frames per round | 106.6 | 111.2 | −4.6 | 0.050 |
| **dead drags** | **0.750** | **1.437** | **−0.687** | **0.001** |

The primary was at power — 71 rounds an arm against the 63 required for a 10%
read — and it moved the wrong way. Score and coins agree in direction and are
nowhere near power, exactly as the pre-registration said they would not be.

**The filter does its job and loses anyway.** Dead drags fell **48%**, which is
the largest effect in the table and precisely what it was built for. It is not
enough.

### It causes collapsed rounds, and that is the whole of it

| | ON | OFF | |
|---|---:|---:|---|
| rounds that never reached FEVER at all | **6 (8.3%)** | 1 (1.4%) | |
| rounds under 15% FEVER | **9 (12.5%)** | 1 (1.4%) | **p=0.009** |

A **nine-fold** increase in the failure mode that this morning's analysis
identified as the single biggest pot of recoverable score — 17 collapsed rounds
were costing 5.5% of everything the corpus had ever scored. The filter takes a
1.4% collapse rate to 12.5%.

That is what the mean differences above are made of. It is not a small
uniform tax on every round; it is a handful of rounds destroyed.

### Where the clears actually went

Per press, split by arm:

| | ON | OFF | diff | p |
|---|---:|---:|---:|---:|
| detections per board | 40.19 | 40.40 | −0.21 | 0.775 |
| median visibility | 0.420 | 0.429 | −0.009 | 0.170 |
| **chain members proposed** | **4.757** | **4.700** | +0.057 | 0.805 |
| **members the game accepted** | **3.261** | **3.435** | **−0.174** | **0.008** |

**The bot proposed exactly the same chains and the game accepted fewer of
them.** Acceptance fell from 73.1% to 68.6%. The filter did not shorten what
gets proposed — it made the same-length proposals worse.

### One mechanism found, confirmed, and too small to be the answer

`adjacency()` rejects a pair when **any other tsum's centre** lies within
`block` radii of the segment joining them
([tsum.py:1459](ttheart_sender/game/tsum.py#L1459)). A detection is therefore
doing two jobs: it is a chain candidate, and it is an obstacle. The filter
deletes it from both. So a pair that was correctly blocked by an intervening
tsum becomes linked once the blocker is thrown away, and the bot strokes
through a gap the game will not join.

Tested on 144 frames from 40 OFF-arm rounds, which are unfiltered and can
therefore be scored both ways:

    detections 5671, filter drops 437 (7.7%)
    adjacency edges  3049 -> 2938  (-3.6%)
    links that exist ONLY after the drop: 27
      of those, passing within the block radius of a DROPPED detection: 27 (100%)

**Every single invented link goes through a deleted blocker.** The mechanism is
real and unambiguous. It is also 0.19 links per frame against 3,049 edges, and
the net edge count *falls*, so it is not what cost 18 clears.

The dominant cost is the blunt one: the filter deletes 7.7% of the board, and
round 34's mark test already said its rejections are game-confirmed at 15.4%
against a 17.2% base rate — *"no longer enriched for real tsums, but only just
below chance."* That write-up said the test could rule a filter out and could
not certify one, and that a played round would have to decide. It has.

### What this settles

* **`reject_model` stays off.** Four documents of "still unmeasured" close here.
  Nothing needs editing: `""` is already the flow default. Untick the radio.
* **The offline mark test was right to refuse to certify it**, and the house
  rule that a played round overrules an offline number is now paid for rather
  than asserted.
* **Do not retrain the reject model at 64px.** That was STEP 2 of the plan and
  it is a pure cost reduction on a component that is being removed. It would
  make a losing filter cheaper.
* **`cleared` was the right primary.** It resolved at 71 rounds an arm; score
  would have needed 593 and returned p=0.457 on the same data, pointing the
  same way. A run scored on the objective would have concluded nothing.

---

## 2026-09-05 20:15 — A bug in the decision rule, caught by the result it reported

`scripts/ab_eval.py` printed:

    DISTRUST. cleared improved, but FEVER share fell -0.1 (p=0.030)

`cleared` had fallen 18.4. The verdict branch tested the guardrail vetoes
**before** the primary's direction, so a run where both went down was reported
as a win with a caveat. The right decision was one branch further down, under a
sentence that said the opposite of the number printed beside it.

Fixed: a primary that fell is a REVERT whatever the guardrails did, and the
guardrails are quoted as agreeing rather than as an objection. Re-run:

    REVERT. cleared fell -18.4 [-35.7, -1.2], p=0.036.
            Guardrails agree: FEVER share -0.059 (p=0.030)

Worth recording because it is the second time in two days that a correct
measurement was nearly filed under the wrong conclusion by the layer that
words it.

---

## 2026-09-05 20:15 — What to do next, in order

1. **Untick the board filter.** Nothing else to revert.
2. **Drop STEP 2 (the 64px reject model) from the plan.** It optimises a
   component that just lost. If the reject idea is ever revived it needs a
   different shape, not a cheaper net — see 4.
3. **The collapse is now the whole target, and the corpus is much better
   placed to attack it.** 71 baseline rounds on one build with clean telemetry,
   and a collapse rate of 1.4% to work against rather than the mixed-build
   6.1% this morning's number was drawn from.
4. **The one idea worth keeping from the filter.** A detection does two jobs in
   `adjacency` — candidate and obstacle — and the filter removed it from both.
   A rejected detection could be made *unchainable but still blocking*, which
   is a two-line change and keeps the dead-drag win without loosening the
   block test. It is worth writing down and **not worth playing yet**: the
   blocker effect measured 0.19 links a frame, so it cannot recover an 18-clear
   loss, and it would be an experiment run to rescue a component that has
   already been beaten on its own terms.
5. **Still the real question, unchanged since this morning:** the bot proposes
   4.7 members and the game takes 3.4, and in a collapsed round it takes 3.0
   of 5.6. Fixing the quality of the proposals is the work. The board filter
   was an attempt at it from the detection side, and detection was measurably
   not the problem — the boards in the two arms are the same board (40.19
   detections against 40.40, p=0.775).

---

## 2026-09-05 21:05 — STEP 2 done: the refusal, one member at a time

Built `scripts/proposal_dataset.py`: **7,352 proposed chain members** from
1,975 presses over the 143 clean 1.11.6c rounds, each with the game's own
verdict on it. The label was already in the corpus and nobody had used it --
`kept` is `marked_by_game(...)`, so "did the game accept this member" is free
at every chain position.

Three things came out, and they point the same way.

**1. Acceptance collapses with chain position.**

    position    accepted        position    accepted
           1       97.9%               5       35.0%
           2       68.8%               7       19.1%
           3       54.4%               8       18.6%

`max_chain` is 12 and the chain sort is "longest wins", so the bot
systematically prefers the proposals with the most members it is about to be
refused on.

**2. A refusal is very nearly the end of the chain.**

    after an ACCEPTED member   accepted 80.9%   (n=5,403)
    after a REFUSED  member    accepted 13.8%   (n=1,949)

**86% of everything proposed after the first refusal is waste.** This sharpens
the twenty-sixth round rather than contradicting it: the game does skip a
refused member and keep linking, but what follows is almost all refused too.

**3. The bot drags 100% of what it proposes** — including the 37% the game
already refused while the chain was held. `verify_reach` is 0, so nothing
trims. That is the shipped baseline, now measured rather than assumed.

### The shipped rule scores exactly 0.5, and that is the finding

    model             ROC-AUC  PR-AUC   Brier
    adjacency          0.5000  0.5987  0.2328
    link_geom          0.8153  0.8921  0.1894
    link_colour        0.8419  0.9087  0.1741

Every member `adjacency` proposed passed its own test by construction, so it
gives them all the same score and **has nothing to rank them by**. It cannot
prefer the member the game will take because it does not distinguish them.

`link_colour` reaches **0.842 ROC / 0.909 PR** on data no link model has ever
seen — every 1.11.6c session postdates all of their splits. It is not
calibrated (over-confident low, under-confident high, but monotone), so it is
already a good ranker and not yet a usable probability.

**Nothing shipped to play.** STEP 2 was the dataset and its validation.

---

## 2026-09-05 21:05 — UI: experiments are tick boxes again

As asked. One thing worth saying plainly: the radio group was not decoration.
A round played with two rules armed cannot say which was responsible, and the
thirty-fifth round lost 18 rounds to exactly that.

So the constraint is kept and **moved from the control to the record**. Ticking
a second row is allowed and is said out loud three ways — a log warning, a line
under the boxes that reads *"2 armed -- a round cannot say which one did it"*,
and a notification. Nothing downstream guesses: `experiment` (singular) answers
only when exactly one is armed, so the A/B pair goes empty rather than
alternating one rule while another stays on for every round.

**The board filter's row is gone.** It was played and it lost; a row for it
would invite a round spent re-finding that, which is the same reason the
character model has never had one. The code, the model and the corpus all
stay — only the invitation goes. A saved settings file still naming it loads
without it.

Three bugs surfaced while doing this, all found by the suite:

* A pre-existing `experiments` property returning a **dict** shadowed the new
  set-valued one — later definition wins — so `key in armed` tested the dict's
  *keys*, every row read as armed, and a baseline run sent every experiment's
  ON value. Renamed to `experiment_states`.
* `PanelSettings.from_dict` only assigns bool and str fields, so the now
  list-valued `experiment` was never read from the file at all.
* The migration took its list branch unconditionally, and the field's own
  default is an empty list — which silently dropped what every tick-box-era
  file had armed.

**571 tests pass.**

---

## 2026-09-05 21:40 — Asked what to collect next, and the answer is nothing

Three questions, three measurements.

### More rounds would not help the acceptance model

The learning curve on the 7,352-member proposal dataset, held out by round:

    train rounds   members   held-out AUC
              20     1,125         0.8585
              60     3,184         0.8773   +0.0188
             100     5,253         0.8795   +0.0022

**Flat from 60 rounds.** Forty more rounds bought +0.0022. The binding
constraint is not sample size, so a collection run aimed at this would be
hours spent for nothing.

> A leak in my own dataset, found while doing this. `prev_accepted` is the
> PREVIOUS member's label, and at decision time the chain is proposed whole --
> the bot knows nothing about any of it. Including it was worth +0.038 AUC
> (0.9171 against 0.8795) and would have been a model scoring a question the
> bot never gets to ask. I flagged `marked` as a leak when building the file
> and missed this one. The honest number is **0.8795**, still well clear of
> `link_colour`'s 0.842 and `adjacency`'s 0.500.

Dropping chain position, length and path as well costs only another 0.007
(0.8723), so nearly all the signal is plain geometry -- which is what
`adjacency` already has and does nothing with.

### And the case for the remaining experiment rows does not survive 143 rounds

`settle_board` and `step_px` are both throughput levers, and their shared
premise is written into `flows/play.yaml`: *"score tracks CHAINS PLAYED
(r=+0.91) and not mean chain length (r=+0.06) -- so more chains per round is
the objective."* That was measured over **five** rounds. Over the 140 scored
rounds of the clean single-build baseline:

    feature        r with score
    settle                0.794
    cleared               0.765
    fever                 0.725
    frames                0.492
    chain length          0.243
    dragged              -0.027
    presses (chains)     -0.166

**It flips sign.** Chains played correlates −0.17 with score, not +0.91. The
five-round number was noise, and two experiment rows rest on it.

`settle` at +0.794 is not a lever either -- it is the consequence running
backwards. A round that clears a lot causes big cascades, and big cascades
take longer to settle, so waiting *more* goes with scoring more. Cutting the
wait does not cause the clears.

So none of the three remaining rows has a live rationale: two rest on a
refuted premise, and `four_groups` was scored on the simulated-cleared metric
the twenty-sixth round disqualified.

### What to tick: nothing

The baseline is the answer this time. Rounds played with everything unticked
still add to the corpus and cost nothing, but no experiment is worth arming
until the offline acceptance work says which one.

### Storage

    dataset total                    1,475 MB   (budget 2,048)
      jpg, 1.11.6c sessions            271 MB   KEEP -- the clean baseline
      jpg, every older session         543 MB   safe to delete
      debug results frames (.png)      248 MB   mostly spent
      samples.jsonl + round.json        29 MB   NEVER delete

The 29 MB of JSONL is the entire analysis substrate -- every number in this
file was computed from it, and the images are only needed to re-derive face
colours and crops. Deleting the older builds' images frees **543 MB** and
loses nothing that is not already superseded: those rounds are the confounded
mixed-build corpus, and their telemetry stays behind.

Keep the results frames for the 31 rounds whose score read UNKNOWN -- those
are the only rounds whose score could still be recovered.

---

## 2026-09-05 22:10 — In plain English: where this is, and what to do next

Written for someone picking this up cold, including a second assistant. No
jargon, and every number below is measured, not estimated.

### What the bot does, in one paragraph

It looks at the Tsum Tsum board, finds about 40 tsums, groups them by face
colour, picks a run of same-coloured tsums that are close enough to join, and
drags a line through them. Do that ~110 times in a 75-second round. The score
comes mostly from reaching FEVER mode, and reaching FEVER comes from clearing
a lot of tsums.

### The one thing wrong with it

**When the bot proposes a chain, the game refuses about a third of it.**

The game tells you this for free. While you hold the first tsum, it lights up
every tsum it will actually accept. The bot records that. Nobody had looked at
it one tsum at a time until today.

Out of 7,352 proposed tsums across 143 rounds:

* The 1st tsum after the head: the game takes it **97.9%** of the time.
* The 3rd: **54.4%**.
* The 7th: **19.1%**.
* And the bot is allowed to propose up to 12.

Worse: once the game refuses one tsum, it refuses nearly everything after it.

* Next tsum after an ACCEPTED one: taken **80.9%** of the time.
* Next tsum after a REFUSED one: taken **13.8%**.

**So 86% of everything the bot proposes after its first mistake is wasted.**
And the bot drags through all of it anyway -- it never looks at what the game
lit up, it just strokes the whole line it planned.

### Why the bot cannot currently do better

The rule that builds chains (`adjacency`) asks one yes/no question: are these
two tsums the same colour and close enough with nothing in the way? Every
tsum in a proposed chain passed that test, so the rule scores them all
identically. **It has no way to prefer the tsum the game will actually take,
because it cannot tell them apart at all.** Measured: 0.500 AUC, which is
exactly the score of a coin flip.

A model that already exists in this repo (`models/link_colour.pt`, trained
weeks ago for a different purpose) scores **0.842** on the same question, on
rounds it has never seen. A small model trained on the new per-tsum dataset
scores **0.879**.

That gap -- 0.50 against 0.88 -- is the opportunity.

### The next steps, in order

**Step 1 (me, offline, needs no rounds from you).** Take the existing
`models/link_colour.pt`, fix its calibration, and replay every press already
recorded. For each one ask: if the bot had cut the chain where the model
predicts the first refusal, would the game have accepted a bigger share? This
is arithmetic on data already on disk.

**Step 2 (decision point).** If Step 1 says yes, there is one change worth
playing: **stop the chain at the first predicted refusal instead of running to
12.** If Step 1 says no, the acceptance signal is real but not actionable, and
the next thing to look at is which chain gets picked rather than how long it
is.

**Step 3 (you, ~143 rounds).** Whatever Step 2 chooses gets a new tick box and
the A/B treatment: alternate on/off round by round, then `scripts/ab_eval.py`.
143 rounds is the number because `cleared` needs 63 per arm to resolve a 10%
change, and score itself would need 593 per arm -- which is why score is a
guardrail here and not the decider.

**Do not collect anything for Step 1 or 2.** They use data you already have.

### Why the tick boxes are still there if none should be ticked

A fair question, and the panel was inconsistent until now. There are three
states a rule can be in:

1. **Played and lost** -- the row is DELETED. The character model and the
   board filter both went this way. Keeping a row would invite someone to
   spend a night re-finding a settled answer.
2. **Unproven** -- the row is a live option. That is what this panel is for.
3. **Never played, and the reason for proposing it has since been withdrawn.**

All three remaining rows are in state 3, which had no name until today.
Deleting them would claim they had lost, and no round has said that. Leaving
them looking live would invite five hours of play for a question nobody
currently has. So each label now ends with **"(no evidence)"** and each row
carries a note saying exactly which claim was withdrawn:

* *Wait for the board* and *Faster stroke* were both proposed because "score
  tracks chains played, r=+0.91". That was five rounds. Over 140 rounds it is
  **-0.17** -- it flips sign. More chains per round is not the objective.
* *Force 4 colour groups* was scored on simulated tsums-cleared-per-drag, a
  metric the twenty-sixth round disqualified when a perfect-knowledge ORACLE
  scored below the shipped rule on it.

So: tick nothing. If one of them ever gets a real case, the mark comes off.

### For a second assistant picking this up

Everything needed is in the repo. Start here:

| what | where |
|---|---|
| this log, newest last | `updates.md` |
| the full round-by-round record | `docs/DATASET-FINDINGS.md` |
| why identity work is closed | `docs/IDENTITY.md` |
| the per-tsum acceptance data | `dataset/proposals.npz`, built by `scripts/proposal_dataset.py` |
| how an A/B is read | `scripts/ab_eval.py` |
| the raw rounds | `dataset/<session>/samples.jsonl` + `round.json` |

Rebuild the dataset and its metrics with:

    .venv/Scripts/python scripts/proposal_dataset.py --eval

**House rules that are not negotiable**, each paid for in lost rounds:

1. **A played round overrules any offline number.** The board filter had 0.998
   AUC offline and lost 143 rounds of play.
2. **One build per comparison.** Every reading before 2026-09-05 was
   confounded by comparing across builds.
3. **Pick the metric before seeing the data.** `cleared` is the decider;
   score, coins and FEVER are guardrails that can only veto.
4. **Never use `clear rate` as the decider.** It is a ratio whose denominator
   the change itself moves.
5. **Anything that ships off, ships off**, with a one-line revert.
6. **Do not label more character crops.** Closed twice, from both directions.
7. **`marked` and `prev_accepted` are leaks** in the acceptance dataset --
   both are the answer, not an input the bot has at decision time.

**Open question a second pair of eyes would help with:** the bot picks the
LONGEST chain (`sort by (is_base, len)`). Given that acceptance falls to 19%
by the 7th tsum, is expected-accepted-length a better sort key than raw
length -- and can that be shown on recorded presses without playing a round?

---

## 2026-09-05 23:30 — The ranker works offline. Here is exactly what to do.

### What I did

Trained a model that predicts, for each tsum in a proposed chain, whether the
game will accept it. Then replayed 597 presses you already played and asked:
if the bot had picked the chain with the most **expected accepted members**
instead of the **longest** chain, would it have done better?

### What it found

**Good news.** Ranking by expected accepted members is worth **+0.109 accepted
tsums per press** (+4.7%). That number survived the hard test: choosing with
one model and scoring with a second, independently trained one gave +0.457
against the first model's +0.467. So it is not the model flattering its own
choice.

The model is also honest about totals — over 597 presses it predicted 2.274
accepted and the game actually gave 2.310. And it is cheap: **2.74 ms a
frame**, against 27 ms for the board filter that lost.

**Bad news, and it kills an idea from the plan.** Cutting the chain short at
the first likely refusal — which sounds obviously right — **loses**. Measured
with the game's own answers, not predictions:

    accepted as played:     2.310 per press
    accepted if truncated:  1.883 per press   <- 18% worse

It looks like a win on "acceptance ratio" (65.7% → 85.3%), but that is the
same trap as clear rate: the ratio improves because you removed the
denominator. The reason is a finding from months ago doing its job — the game
**skips** a refused tsum and keeps going, so trying a long shot costs one slot,
not the rest of the chain. A 15%-likely tsum is nearly free to attempt.

So the rule that shipped **re-picks the chain but never shortens it.**

### What you need to do — the steps

**Step 1. Free up disk first.** The dataset is at 1,475 MB of a 2,048 MB cap,
and collection stops silently when it fills. Delete the `.jpg` files in the
283 pre-1.11.6c session folders (543 MB) — keep every `samples.jsonl` and
`round.json`, that 29 MB is the whole analysis. Keep the results frames for
the 9 rounds whose score read UNKNOWN.

**Step 2. Tick two boxes:**

* **"Pick the chain the game will accept"** — the new row. It is the only one
  without "(no evidence)" after it, because it is the only one with a
  measurement behind it.
* **"Alternate it round by round (A/B)"**

Leave the other three unticked.

**Step 3. Play about 272 rounds** (~6 hours). That is not padding: +4.7% is a
small effect, and this corpus's own spread needs ~136 rounds an arm to
resolve it on accepted-per-press. At 143 rounds the honest answer would be
"cannot tell", which is what happened to the board filter's first reading.

**Step 4. Run `scripts/ab_eval.py`.** It reports the arms, the power, and the
verdict.

### What I expect, and what would change my mind

Expect a small win: more accepted tsums per press, slightly more cleared,
maybe nothing visible in score. If accepted-per-press does not move at all,
the model is right about single members and wrong about whole chains, and the
next question is why. If `cleared` falls while accepted rises, the ranker is
picking chains that are accepted but clear less — which would be new and worth
knowing.

**One caution.** Everything about this is better than the board filter looked
at the same stage: calibrated, debiased, ten times cheaper, agreeing with two
earlier independent findings. The board filter had 0.998 AUC offline and lost
143 rounds of play. Offline numbers do not decide this. The round does.

### Two bugs worth recording

* I named a helper `_face_lab` when a `_face_lab` already existed with a
  different signature, and it silently shadowed it — six unrelated tests
  failed. **Second shadowing bug in two days**, after a dict-valued
  `experiments` property hid a set-valued one and made every experiment read
  as armed on a baseline run.
* A test I wrote asserted a shorter chain should beat a longer one under a
  position-based score. It failed, and it was right to: expected-accepted is a
  **sum of probabilities**, so adding a member can never lower it. That is why
  the ranker cannot truncate — and, given truncation measurably loses, why it
  should not.

**584 tests pass.**
