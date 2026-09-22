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

---

## 2026-09-06 — The A/B was right to reject, and I have to own why

### The short version

Your 312 rounds gave a clean REJECT. The verdict is correct. But **the rounds
tested a broken build, and the bug was mine.**

### What the numbers said

    cleared        266.2 ON vs 266.0 OFF    p=0.970
    accepted/press  2.298 vs 2.322          p=0.601   CI [-0.113, +0.065]
    score         537,755 vs 532,807        p=0.896

The confidence interval on accepted-per-press **excludes my offline prediction
of +0.109**. That is not "unproven" — it is ruled out.

### The ranker did fire, and behaved sensibly

Per round it re-picked 21%, 25%, 14%, 19%, 30%, 23%, 25%, 21% of presses,
against the 23.8% predicted offline. And what it did made sense:

    acceptance rate per member   63.2% vs 61.1%   (+2.1pp — better)
    members proposed per press    3.63 vs 3.80    (−0.17  — fewer)
    accepted members per press    2.298 vs 2.323  (−0.03  — cancels)

**It traded quantity for quality at break-even.** Which is why the bug was hard
to see: nothing looked broken.

### The bug

The model reads each tsum's face colour. At play time I fed it the *k-means
cluster* colour instead, to save a pixel read — and a chain is all one kind by
construction, so every member had the same cluster colour. Both colour
features came out **exactly zero**, on every member of every chain.

    trained on:  mean 0.42, never zero
    served:      0. 0. 0. 0. 0. 0. 0. 0. 0.

Two of eighteen inputs were dead while it chose. Those two are worth 0.027
AUC, so it was not a useless model — it was a good model being handed
constants it had been trained to read.

**That cost you about seven hours of play, and it was my mistake.** I wrote a
comment justifying the shortcut and never checked that the two code paths
produced the same numbers.

### Fixed, and pinned

There were *two* implementations of "face colour" plus the shortcut. Now there
is one function: the trainer imports the runtime's, `rank()` takes the frame,
and a test asserts both produce identical rows for the same board. The
shortcut is deleted, not deprecated.

Retrained: **0.867 AUC**, honest offline gain now **+0.074** accepted members
per press. Cost 7.24 ms a frame (was 2.74), still ~1% of a frame.
**586 tests pass.** Version bumped to 1.11.8.

### And my recommendation is to stop here, not to re-run

+0.074 on 2.31 is **+3.2%**. Proving that live needs **~938 rounds — about 20
hours.** I am not asking you for that on a 3% effect.

I checked the obvious escapes first:

* **The length knob** — swept at 0, 0.1, 0.2, 0.35, 0.5. Zero is already best.
* **Letting the model build chains** instead of ranking them — cannot be
  evaluated at all. Every one of the 7,352 training rows came from a chain
  `adjacency` built, and `adjacency` only builds same-chain-colour chains. The
  feature `same_kind_head` is **1.0000 in every row, zero variance.** Ask the
  model about a chain `adjacency` would never build and it invents an answer:
  it claimed 0.79 acceptance per member against a true base rate of 63%.

That last point is the real lesson, and it is structural: **a model trained on
what the bot already does can only price small changes to what the bot already
does.** Ranking is a small change. Rebuilding how chains are made is not.

### What I would do next, if you want to continue

One collection, and it is a different KIND of collection — for coverage, not
for a verdict. Play a share of presses on a deliberately different chain (a
random candidate, or the one `adjacency` ranks last) and record the game's
answers. That is the only way to learn about chains the bot never plays, and
everything about improving chain *generation* needs it underneath.

It would be the first time this project spends rounds to see something new
rather than to settle a yes/no. Your call whether that is worth it — the
honest expected payoff is unknown, which is exactly what makes it worth
knowing.

### The state of play, plainly

The bot's chain choice is close to its ceiling given how it builds candidates.
The per-member acceptance model is genuinely good (0.867 against `adjacency`'s
0.500 coin-flip) and converting that into better *choices* is worth ~3%, which
this project cannot measure. Detection is not the problem, identity is closed,
throughput is not the objective, and now ranking is measured out too.

---

## 2026-09-06 17:40 -- Recognition audit: the premise was wrong, and the answer was already on disk

The next-task document asked me to convert a "Beast-only" training module into
multi-class recognition. **There is no Beast-only training module.**
`scripts/classify.py` has been multi-class since round 31: 38 shipped classes,
5,642 hand-labelled crops in 49 folders, 371 sessions. Beast is the tsum you
have EQUIPPED, recorded in the scorecard so rounds compare -- it is not a
training assumption anywhere in the code. It is in fact the *least* labelled
character in the set: 25 crops of 5,642.

So instead of rewriting what already existed, I audited it and built the
evaluation the project genuinely did not have.

### The number nobody had: the shipped model, on a fixed test set

`scripts/classify.py` prints a confusion matrix for the model it just trained,
in torch, in memory. Nothing scored **the artifact the bot actually loads**.
`scripts/recog_eval.py` does: `models/character.onnx` through `cv2.dnn`, on the
92 sessions recorded as held out inside `models/character.json`. It trains
nothing, so it is repeatable and the test set cannot drift.

    1,540 crops, 92 held-out sessions

    over the 37 classes it was TAUGHT      including the untaught
      top-1        97.3%                     top-1        94.7%
      top-3        98.5%                     top-3        95.9%
      macro F1     94.8%                     macro F1     90.6%
      worst class  66.7%  CaveOfWonders      Cleo scores 0% -- never taught

    latency  0.25-0.31 ms/crop  =  12-14 ms for a 46-tsum board
    at the shipped 0.85 floor:  96.0% coverage, 97.6% accuracy

### Two real dataset faults, found by the tool on its first run

**Cleo has 41 held-out crops and zero training crops** -- one session, entirely
on the wrong side of the split. It scores 0% at something it was never shown
once, and that is arithmetic, not skill.

Worse, its crops have to go somewhere, and they went to Beast: **25 of the 31
crops the model called Beast are Cleo.** That pushed Beast's precision to 16.1%
and printed **Beast as the worst class the model was taught** -- which is false,
its recall is 83.3%. I nearly reported that. Recomputing the taught headline on
the taught subset moves the worst class from a fictitious Beast 27.0% to a real
CaveOfWonders 66.7% at n=2. `tests/test_recog_eval.py` pins it so the next
person cannot report it either.

### The finding that matters for gameplay

Recognition is **reliable on the picture it was trained for and blind to the
rest of the board**, and that is not fixable with labels:

* Every one of the 5,642 labelled crops is at least **0.55 visible** -- measured,
  0.0% below it. The median tsum on a real board shows **0.41**.
* So the model is excellent on roughly the **least occluded fifth** of a board
  and, measured against the game's own marks, sits at the chance rate below the
  floor while still reporting 0.75-0.88 confidence.
* Labelling buried crops was already tested -- 614 of them -- and had **no
  effect**. The pixels are not there.

And the step the task assumes comes next has already been played, in the wrong
direction: naming characters scored **-10.9% cleared** over 407 held-out boards,
because naming only the visible members splits one character into two groups and
puts 8.59 ids on a board where the game allows 5. **A chain needs its group whole
more than it needs the group correctly named.**

### What I did and did not touch

Two new files, `scripts/recog_eval.py` and `tests/test_recog_eval.py`. **No model
retrained, no gameplay file touched**, `character: ""` unchanged, the character
model still ships off. **592 tests pass** (586 + 6).

Full audit, including all fifteen questions answered against the code, in
`docs/RECOGNITION-AUDIT.md`.

### Cheapest things worth doing next

1. Retrain with Cleo on the training side -- 41 crops already labelled, costs
   2.6 points of top-1 today and poisons Beast. Free.
2. Give the 9 single-session classes a second session so they can be trained
   AND scored.
3. Look at `WhiteRabbit -> LittleOyster` (10) -- the only confusion between two
   well-taught classes, and the only genuine model error on the list.
4. Do **not** label more buried crops.

None of these needs rounds played, because accuracy is not what is blocking the
gameplay path. Occlusion is, and that is a detection problem.

---

## 2026-09-06 19:15 -- Occlusion and multi-frame tracking: the pile does not move

The task asked whether the same physical tsum can be tracked across frames and
identified from its best-visible observation. Investigated on data already on
disk. **Refuted.**

### There is no frame sequence in this corpus

Stored samples are **3.87s apart at the median, never under 1.98s**, with a
round's worth of clearing in between. No video, no recording mode. The 3-frame
burst collection takes at 0.05s to read the marks is used and discarded. The
only real pair is `NNNN_before.jpg` -> `NNNN_marked.jpg`, ~0.25s apart.

Live, the bot captures **1.46 frames/s** against ~65 presses: one frame per
decision, none spare.

### Tracking is trivial, and that is the bad news

200 samples, both frames through the same offline pipeline with a shared
palette and the round's own options:

    matched within 1 radius   control 100.0%   +0.25s  70.6%
    median displacement          0.00 px            1.41 px
    ambiguous                     0.0%               0.5%

The control is one frame read twice: 100% at 0.00px, so the detector is
deterministic and all movement in the test column is real. **1.41px on a board
where tsums sit 61px apart.** Nearest-neighbour is the whole tracker.

> The first version of this compared re-detections against the detections
> stored at play time and reported a 72% control. That compares two pipelines,
> not two frames -- the loop caches its palette, and `include_dark=True` at
> play against the function default `False` dropped every black tsum on its
> own. Corrected, the control is 100%.

### The number that decides it

Visibility change over 0.25s, 5,461 unmarked tsums:

    p10 -0.024    median +0.000    p90 +0.024
    more visible 32.3%   less 31.7%   unchanged 36.0%

    buried tsums that rose above 0.55 visible:  145 of 4,186  =  3.5%

Treat each 0.25s as an independent 3.5% shot -- far too generous, a tsum at the
bottom of the pile stays there -- and making half the buried tsums readable
costs **~4.8s per decision** in a 75s round with ~65 decisions.

### The three methods, measured anyway

390 human-labelled crops, marked tsums excluded:

    band                  n    A single   B best-vis   C mean
    high (>=0.70)        79      96.2%        94.9%     97.5%
    medium(0.60-0.70)   234      99.6%        99.6%     99.6%
    low (0.55-0.60)      77     100.0%        98.7%     98.7%
    ALL                 390      99.0%        98.5%     99.0%

B changed 2 answers of 390 and lost both. The honest limit: labelled crops are
all >=0.55 visible so A is already at ceiling. But the visibility measurement
above answers it without labels.

### The real wall is group integrity, and it is worse than remembered

`group_eval.py`, 378 paired drags:

    method    agreement   lift   ids/board  cleared   vs kmeans
    kmeans        42.6%  1.58x      7.29      3.21    --
    named         35.4%  1.59x      8.37      2.90    -9.6%   REAL
    colour4       48.6%  1.52x      4.00      3.34    +3.9%   REAL
    model5        73.2%  1.00x      3.90      2.46   -23.3%   REAL

`model5` is the classifier's own output forced into the game's 5 groups -- the
obvious fix for `named`'s fragmentation -- and its **lift is 1.00x: it knows
nothing beyond the sizes of the groups it made**, because it has no opinion
about the 78% of the board it never reads. So the remedy already in the code
does not work, and naming more of the board is negative until it reaches nearly
all of it. That applies to all four occlusion approaches equally.

### A side finding worth more than the one I was asked for

Away from the game's highlight, on a board that moved 1.41px, **24.4% of
detections fail to reappear 0.25s later.** A quarter of the board is not a
stable object between frames. Upper bound -- separate JPEG encode, and the game
animates -- but nobody has ever asked this question, and it bears directly on
chain building.

### Recommendation

Not an occlusion build. One offline run first: **measure the fragmentation
curve** -- simulate naming 20/40/60/80/100% of each board correctly and plot
ids/board and cleared against it. `group_eval.py` already has the boards, the
marks and the scoring. It settles the question every occlusion approach depends
on: how much of the board must be named before naming helps at all.

If the answer is "above ~90%", the identity route closes for good. If it is
"50-60%", the crescent-following crop (approach A) is the cheapest way to aim
at it.

Runner-up, if the effort should go somewhere shippable: **`kinds: 4`** is the
only positive intervention in the table (+3.9%), already written, already in
`flows/play.yaml`, already off, needing rounds rather than engineering.

### Files

Added `scripts/track_probe.py`, `scripts/multiframe_eval.py`,
`tests/test_track_probe.py`, `docs/OCCLUSION-INVESTIGATION.md`. **No gameplay
file touched.** 598 tests pass.

---

## 2026-09-06 21:30 -- The base tsum: the observation is real, the mechanism is a colour match with no guard

The player reports gameplay is better with Beast equipped. Investigated. **The
observation has a mechanism and it has nothing to do with Beast being Beast.**

### The base tsum is not a character anywhere in this code

`read_base_kind` takes the median Lab of the skill icon and returns `argmin`
over the board's 12 k-means palette centres -- the nearest COLOUR CLUSTER.
`find_chains` flags that cluster `is_base` and sorts `(is_base, len)`, so a base
chain is played ahead of every longer chain. One colour match, used to reorder
candidates. That is all of it.

### The guard the code documents and never applies

    "A large distance means no cluster really matched and the caller should
     not trust it."   -- read_base_kind's own docstring

The play loop never checks the distance. `tsum board` does, printing
`(WEAK -- check --base)` above 30. The code that plays rounds takes `argmin`
unconditionally, every frame.

    icon distance to the matched cluster: median 14.7  p90 45.6  max 65.7
    above the CLI's own WEAK line of 30:  24.0% of samples

**With Beast equipped, the base priority already points at an arbitrary cluster
on a quarter of frames.**

### And the instability is in the palette, not the icon

The icon reads steadily: spread WITHIN a session is 1.0 Lab unit, on a fixed
sprite in a fixed place. The 12 clusters are refitted per board, and on ~24% of
frames none lands near it. So a base whose colour is common and distinctive wins
the `argmin` reliably; a rarer one loses it more often, and nothing reports it.

### Where the base diverges: chain ranking, and nothing earlier

120 stored boards re-run under every possible base cluster and under none:

    candidate chains IDENTICAL under every base :  100.0%
    chain CHOSEN changes with the base          :   62.5%
    lost by pointing the base at the worst cluster: 1.02 tsums

Detection, features and grouping are untouched.

### What the preference costs when it works

    build       n     base%   clr base   clr other      diff
    1.11.7b   4209    17.6%      2.95        3.22    -0.27 +/- 0.10  REAL
    1.11.1e   3452    16.3%      2.91        3.22    -0.31 +/- 0.11  REAL
    1.11.6c   1975    19.1%      2.99        3.21    -0.22 +/- 0.13  REAL

Three independent builds agree: a base chain clears ~0.27 fewer tsums than the
chain otherwise played, on ~18% of drags. That is the price, paid to charge the
skill gauge.

**The tempting number, and why it must not be quoted.** Rounds with more base
drags clear more (+21.7) and reach FEVER more (+8.0pp). But base share also
correlates +0.121 with cleared-per-chain -- the very quantity the drag-level
test shows it LOWERS. Reaching FEVER early makes the board denser for longer,
which raises base share as a consequence. The causation plausibly runs
backwards and this corpus cannot separate it.

### The comparison asked for cannot be made

    737 sessions, 10,108 samples
    spread of the SESSION medians of the icon colour: 1.0 Lab unit

**Every round ever recorded used the same equipped tsum.** No data exists for
any other base. The observation is not dismissed -- it is untested.

> A first pass read "more than one equipped tsum" off a per-sample spread of
> 116.8 Lab units. Wrong statistic: it mixes across-session differences with
> within-session noise. Both are now computed and printed separately and
> `tests/test_base_probe.py` pins the distinction.

### Next experiment: 20 rounds, zero engineering

Play ~20 rounds with a different tsum equipped. The collector already records
`base.lab` and `base.distance` per sample, so those rounds answer whether the
match degrades, by how much, and whether the preference still fires at ~18%.
Nothing else can be decided first.

The fix it would license: **refuse the base preference when the match is weak**
and fall back to longest-chain. The distance is already measured and already
documented as untrustworthy when large. That generalises to any equipped tsum,
which is the actual goal.

### Also worth recording

`_base_from_faces` -- matching the icon against the BOARD'S FACES rather than
the raw palette -- already exists and is the better-posed question (at most 5
face colours, not 12 clusters including outlines and background). It runs only
under `--recolour`/`--kinds`. Making it the primary path is small and targets
the failure directly.

And one thing nobody has ever measured: **the skill firing.** The base
preference costs 0.27 cleared a drag and is supposed to pay for itself by
charging the gauge. If it is not charging, the preference is a pure loss.

### Files

Added `scripts/base_probe.py`, `tests/test_base_probe.py`,
`docs/BASE-TSUM-INVESTIGATION.md`. **No gameplay file touched.** 604 tests pass.

---

## 2026-09-06 23:10 -- The 43 rounds on a second base: my prediction was wrong, and the real mechanism is the opposite one

43 rounds arrived on **1.11.8** with a different tsum equipped -- icon Lab
**223,137,138** against the corpus's **199,134,165**, 36.9 Lab units apart. The
first data this project has ever had on a second base.

### The comparison is legitimate, and here is why

Build and base are perfectly confounded: every 1.11.8 round is the new base,
every older round the old one. But the 1.11.8 option set is **byte-identical**
to the 1.11.7b `chain_model: ""` arm -- link, block, link_px, k, include_dark,
bowl_reject, fit_effort, mode, max_chain, use_base all equal -- and the only
code difference between those builds is a chain ranker that is **off in both**.
So 43 rounds (new base) against the 155-round 1.11.7b OFF arm (old base) is one
setting changed, not two.

### The prediction, and its refutation

I predicted a different base would match its colour cluster **worse**. It
matches **better**:

    icon Lab        rounds  dist med   >WEAK 30   board share   base drags
    199,134,165        737      14.7      24.0%        14.2%        17.4%
    223,137,138         43      15.9       9.7%        20.2%        50.9%

**9.7% weak against 24.0%.** The new base is the more reliable match by a factor
of two and a half. The failure I went looking for is not there.

### What is actually happening

The new base's colour covers **20.2% of the board against 14.4%** (+5.8pp
+/- 1.3, REAL) and is the **biggest group on 44.8% of frames against 13.6%**.

So the preference fires on **50.9% of drags instead of 17.4%** -- three times as
often. And a base chain has always cleared less than the chain otherwise
played: -0.27, -0.31, -0.22 across three earlier builds, -0.18 +/- 0.27 here.

**The bot is not matching the wrong colour. It is matching the right colour and
then obeying the preference three times as often, paying its cost three times
as often.**

### The outcome

    metric           NEW base    OLD base       diff
    FEVER share         0.33        0.45     -0.12 +/- 0.07   REAL
    cleared           256.49      266.92    -10.43 +/- 24.11  noise
    chains played      81.56       79.45     +2.11 +/- 5.18   noise
    dead drags          3.4%        1.1%     of collected samples

**FEVER share falls 12 points**, which is the metric the thirtieth round found
score turns on (r = +0.808). Cleared barely moves. So the player's observation
is **confirmed in the direction they reported**, and confirmed on the measure
that matters most -- but not for the reason I proposed.

### What this data cannot settle

Two explanations remain, and one round of collection cannot separate them:

* **(a) over-firing.** The preference fires on half of all drags and each firing
  costs ~0.2 cleared, so the rule itself is the harm.
* **(b) the skill.** A different equipped tsum has a **different skill**, and
  the base preference exists to charge that gauge. A weaker or slower skill
  would depress FEVER regardless of how the chains were picked.

The clear-size distributions are close (mean 2.94 vs 3.19, p99 9 vs 8) and the
collector only samples ~14 drags a round, so no skill signature is visible
either way. **This is the honest limit of the 43 rounds.**

### The experiment that separates them

**A/B `use_base` with the new tsum still equipped.** One setting, already in the
options, alternating round by round on one build:

* if FEVER recovers with the preference off -> **(a)**, and the fix is the
  guard: refuse the base preference when it would fire too often or match too
  weakly;
* if FEVER stays down -> **(b)**, the skill, and the base preference is
  exonerated -- the answer is then simply which tsum to equip.

Either way it is decisive, and it needs no new code beyond an A/B arm.

### And one thing nobody has ever measured

The base preference costs ~0.27 cleared a drag and is supposed to repay that by
charging the skill gauge. **Nothing in this project has ever recorded the skill
firing.** Until it does, the preference's benefit is assumed rather than
measured, on both bases.

### Files

`scripts/base_probe.py` extended with a per-equipped-tsum table (section 1b).
**No gameplay file touched.** 604 tests pass.

---

## 2026-09-07 00:40 -- Per-tsum profiles, the Beast backup, and the skill counter nobody had

### First, a correction I have to make

I said last round that **"nothing in this project has ever recorded the skill
firing."** That was wrong. The play loop measures the gold ring
(`skill_gold`, tsum.py:3021), fires the skill when it is charged, and logs
`skill charged (ring 0.70) -- firing`. What is missing is only that the firing
is **not counted in `round.json`**, so it cannot be compared across rounds
without parsing logs.

So I parsed the logs. Over **262 Beast rounds** on 1.11.7b:

    skill fires per round:  mean 2.25   median 2   max 7   rounds with 0: 10%

    corr(skill fires, FEVER share) = +0.741
    corr(skill fires, cleared)     = +0.770

**That is the strongest relationship in this project**, alongside score-to-FEVER
at +0.808. The skill is not a side benefit of the base preference -- it is close
to being the engine of the round. Which makes the base preference's whole
premise (charge the gauge, pay ~0.27 cleared a drag for it) look far better
motivated than the drag-level cost alone suggested.

The logs in `dataset/logs/` stop at 2026-09-06 09:34 and the second base's
rounds start at 11:58, so **the comparison cannot be completed yet**: the
emulator machine's newer `ttheart.log` has to be copied over. That single file
would settle whether the second base's 12-point FEVER drop is over-firing or
simply a weaker skill.

### And a premise that had to be checked before building on it

The player said labelled/cropped tsums seem to play better. **`character` is
`""` in all 781 recorded rounds** -- every build, every session. The character
model has never played a round, so the 5,642 labelled crops cannot have affected
gameplay yet. The performance difference between the two bases is fully
accounted for by the base-colour mechanism already measured. Labelling is
valuable for the day the model ships; it is not what moved these rounds.

### What was built: profiles

`ttheart_sender/game/profiles.py` plus `profiles/*.json`. A profile is a named
bundle of the tunables `flows/play.yaml` already declares, keyed to an equipped
tsum, selected with `profile:` (flow) or `--profile` (CLI). Empty is the default
and changes nothing.

    profiles/beast.json         737 rounds, icon Lab 199,134,165
    profiles/unknown-223.json    43 rounds, icon Lab 223,137,138

**`beast.json` is a backup, not a retyped guess.** Its 39 settings were read out
of `dataset/*/samples.jsonl` -- the options those 737 rounds actually ran under.
Machine-specific values (dataset path, duration, emulator geometry) are
deliberately excluded: a profile is about the tsum, not the machine.

**`unknown-223.json` is deliberately identical to it.** The 43 rounds scored 12
points less FEVER on exactly these settings, so the file is the place to diverge
them -- but diverging is a decision to be A/B'd, not something to pre-empt in a
data file. A test asserts the two remain identical until someone changes one on
purpose.

Three properties, each because a silent failure here would poison every future
measurement:

* **An unknown option name raises.** `use_bases: false` does not get ignored;
  it refuses to load and names the offending key. A profile that says it turns
  something off and does not is the worst outcome available.
* **A profile can only reach options that already exist** -- `ALLOWED` is a
  whitelist of the flow's own tunables, so a profile is different *values*,
  never a new lever.
* **The icon is checked.** A profile carries the icon Lab it was measured on,
  and the moment `read_base_kind` reads the icon the loop compares them. The
  reading is stable to ~1.0 Lab unit within a session and the two tsums are 36.9
  apart, so this is reliable. Mismatch warns loudly and plays on -- refusing to
  play would be a worse failure than a loud line.

Wired at every hop: declared and forwarded in `play.yaml`, `resume.yaml` and
`launch.yaml` (a var declared in one flow and not the next is how `ab` came to
be silently overridden), and `profiles/` added to `build.py`'s `DATA_DIRS`.

**24 tests. 624 pass. Version 1.11.9.** Nothing changes for anyone who does not
name a profile.

### Next

1. Copy the emulator machine's `logs/ttheart.log` covering 2026-09-06 11:58 to
   13:10. It settles over-firing vs weaker skill, and costs nothing.
2. Add `skill_fires` to `round.json`. A counter, no behaviour change, and it
   ends the log-parsing forever.
3. Then A/B `use_base` on the second tsum, which is now a one-line profile edit
   rather than a code change.

---

## 2026-09-07 02:15 -- The panel is down to one row, and recognition scored per equipped tsum

### The panel

Three of the four experiment rows were dead and are gone:

    board_filter   PLAYED and LOST     143 rounds: cleared -18.5, FEVER -5.9pp
    chain_ranker   PLAYED and REJECTED 312 rounds: cleared 266.2 vs 266.0
    settle_board   PREMISE WITHDRAWN   never played
    fast_stroke    PREMISE WITHDRAWN   never played

`chain_ranker`'s row still described itself as "THE ONLY ROW WITH EVIDENCE
BEHIND IT... Never played." Both halves were stale: it has been played, and it
lost. That text sitting on the panel was an invitation to run it again.

**`four_groups` is the only row left**, and it earned it. Re-measured on 378
paired drags against the game's own marks it is the only grouping rule ever
scored ABOVE the baseline -- **+3.9%**, where naming characters scores -9.6% and
forcing the model into 5 groups scores -23.3%. Its caveat is unchanged and now
says so on the row: the metric is the simulated cleared-per-drag that the
twenty-sixth round disqualified, and playing it is the only way that resolves.

Retiring a row takes it off the panel; it does not take the lever out of the
flow. All four remain ordinary `flows/play.yaml` options, and
`normalize_experiment` already drops retired keys, so a settings file naming any
of them opens clean. A new `RETIRED_EXPERIMENTS` tuple records what went and
why, so re-proposing one costs somebody a read.

Twelve tray tests used the retired keys as stand-ins for "some experiment". The
machinery they describe -- refusing to A/B two armed rules, sending each to its
own variable, saying out loud when more than one is on -- still exists above one
row, so they now bring their own two-row catalogue via a `two_rows` fixture
rather than depending on how many rows happen to ship. Several also had to stop
conflating a row's KEY with the flow VARIABLE it sends: `four_groups` sends
`kinds`, where `settle_board` sent `settle_board` and hid the distinction.

Profiles were made rename-proof at the same time: `unknown-223` became
`beans_camo_vil`, and the tests now discover the shipped profiles instead of
naming them.

### Recognition, per equipped tsum

The obvious measurement is impossible, and the reason is the finding:

    labelled crops from beast sessions            805
    labelled crops from beans_camo_vil sessions     0

**Every labelled crop whose session still exists came from Beast rounds.** So
`scripts/recog_by_tsum.py` uses the marks instead -- name the head where the
model is visible and confident, then ask what it calls each tsum the game
confirmed is the same character. No labels needed, works on any session.

    equipped tsum     samples  usable    0.55-0.65      0.65-0.75      0.75-1.01
    beast                9691     504   43 +/- 6%      85 +/- 4%      63 +/- 6%
                                          n=277          n=354          n=276
    beans_camo_vil        613      44   35 +/- 20%     55 +/- 18%     70 +/- 19%
                                          n=23           n=29           n=23

**One band separates: 85 +/- 4% against 55 +/- 18% in 0.65-0.75.** The other two
overlap. 44 usable heads against 504 is thin, and the error bars are printed
beside every cell for that reason -- a bare "55.2%" from 29 observations reads
exactly like a bare "84.7%" from 354 otherwise.

And the reading to resist: this scores the BOARDS played with each tsum, not
each tsum's own face. The equipped tsum changes which characters are on the
board, so lower agreement may mean those boards hold characters the model knows
less well -- not that recognition is worse for that tsum. Separating those needs
the per-character breakdown on those boards, which is the obvious next step and
was not taken here.

Figures are a FLOOR either way: the head's name comes from the model, so a
mistaken head marks a whole group wrong.

### Files

Added `scripts/recog_by_tsum.py`, `tests/test_recog_by_tsum.py`. Changed
`ttheart_sender/tray/service.py` (EXPERIMENTS trimmed, RETIRED_EXPERIMENTS
added), `tests/test_tray.py`, `tests/test_profiles.py`. **624 tests pass.
Version 1.11.10.**

---

## 2026-09-07 03:30 -- A whole-board validator, and the 40.7% nobody could see

### Why a fifth scorer was worth building

Four offline scorers already existed and all of them start partway down the
pipeline. `recog_eval.py` reads crops `crops.py` cut and saved months ago;
`mark_probe.py` and `recog_by_tsum.py` only look at tsums the game marked;
`group_eval.py` scores grouping rules. **None runs the app's own path on a whole
screenshot, and none shows you a board.**

`scripts/board_check.py` does: stored detections -> the runtime's own
`CharacterModel.probabilities` -> the shipped visibility and confidence floors,
over saved `_before.jpg`, with `--overlay` writing the board back out with the
names drawn on it.

Reading crops that were successfully cut is **survivorship**, and it hid what
this found on its first run.

### What it says

    300 boards, 13,693 detections

    visibility     tsums   asked   named    human-labelled     game's marks
    0.00-0.55       9341      0%      0%              --               --
    0.55-0.65       2434     70%     66%     99 +/- 1%  n=304   25 +/- 43%  n=4
    0.65-0.75        950     63%     58%    100 +/- 0%  n=145   50 +/- 41%  n=6
    0.75-1.01        968     51%     44%    100 +/- 1%  n=208   60 +/- 44%  n=5

    against human labels : 99.4% of 657
    recognition          : 5.4 ms a board

**99.4% correct on 657 human-labelled tsums, through the full app path.** The
recognition layer is not broken; the recognition audit's 97.3% holds end to end.

### The finding: ASKED falls as visibility RISES

70% -> 63% -> **51%**. The tsums showing MOST of themselves are offered to the
model LEAST, which is backwards. Counting `asked` apart from `named` is the only
reason it was visible at all -- a single "named %" cannot separate "the crop was
refused" from "the model declined".

Diagnosed, over 1,845 tsums above the floor:

    refused because the crop window ran off the frame edge:  751  (40.7%)

    kept    : median visibility 0.62, median margin from the frame edge  86 px
    refused : median visibility 0.64, median margin from the frame edge   1 px

    0.55-0.65   refused 37% of 1085
    0.65-0.75   refused 40% of  381
    0.75-1.01   refused 52% of  379

**Two in five readable tsums are never offered to the model, and it is worst for
the clearest ones.** The mechanism is plain: `_character_crop` takes a square of
half-width one radius about the centre and REFUSES when the frame edge would
clip it. The board crop is tight around the play area, so a tsum at the rim sits
within one radius of the edge -- median margin 1 px. And the tsums at the rim
are exactly the ones with nothing on top of them, which is why the refusal rate
climbs with visibility.

### This is not a bug, and that is what makes it interesting

The refusal is deliberate and correct given the training set: `crops.py`
excluded clipped crops, so a clipped crop at play time would be a picture the
model has never seen one example of. The comment says so.

But it means the readable board is **19% of detections, not the 22% assumed**,
and the missing 40% are not random -- they are the rim, systematically. That
matters for the fragmentation result in `docs/OCCLUSION-INVESTIGATION.md`: the
`named` rule was already losing 9.6% by splitting one character between a named
group and a colour group, and it is splitting off an even smaller, more biased
slice than anybody thought.

The fix is not a play-loop change -- it is a training-set change: pad the frame
(replicate border) in BOTH `crops.py` and `_character_crop`, so a rim tsum gets
a consistent picture and the model is trained on it. That is a retrain, and it
belongs behind the fragmentation curve rather than in front of it.

### Files

Added `scripts/board_check.py`, `tests/test_board_check.py` (8 tests).
**No gameplay file touched. 632 tests pass.**

One counting bug was caught and fixed before reporting: `named` came out ABOVE
`asked`, which is impossible -- the accuracy counter was still writing into the
`named` slot after the column was added.

---

## 2026-09-07 05:00 -- Click-to-label, and what all those circles actually are

### The gap it fills

Labelling here has only ever worked one way: `crops.py cluster` groups crops by
an embedding, emits a contact sheet per cluster, and `crops.py assign 03 Mickey`
names the WHOLE cluster. That is what made 3,958 labels affordable -- twenty
decisions instead of two thousand -- but **it has no way to fix ONE crop.** A
wrong tsum in a sheet stayed wrong until somebody moved the file by hand.

`scripts/label_board.py` is the other half. One board at a time, every detection
circled, click a tsum and name it. The model's own guess sits beside each one,
so the work is mostly confirming and occasionally correcting.

    click        select a tsum
    1-9          accept a suggestion (model top-3, then existing classes)
    a-z + Enter  type a new name
    d / u        delete / undo
    n / p / q    next / previous / quit

It writes `crops/labelled/<Name>/<session>_<sample>_<index>_v<visible>.png` --
the exact layout and filename `crops.py assign` produces, cut by **crops.py's
own `_cut`**, imported rather than reimplemented. `classify.py` picks these up
with no change anywhere.

Two refusals are deliberate and both are load-bearing:

* **A crop the frame edge would clip is refused**, and the tool says so rather
  than writing a clamped one. A sliver stretched to a square teaches a
  classifier what the board edge looks like.
* **The model's guess is never written as a label.** It is displayed; a person
  presses the key. `docs/IDENTITY.md` section 9 has been the rule since the
  first mark-harvest.

A correction MOVES the file. Leaving the old one behind would train the model on
both answers at once, and `tests/test_label_board.py` pins that.

### The player's observation, quantified

> "almost all the tsum other than main tsum has the circle"

Correct, and it is the shape of the problem rather than a display bug. Over 60
boards, 2,475 detections:

    grey  - buried below the 0.55 visibility floor : 1915   77%
    grey  - crop clipped by the frame edge         :  224    9%
    named - model confident enough to name         :  285   12%
    asked but not confident                        :   51    2%

**86% of the board gets a grey circle**, and only one tsum in eight is ever
named. The colours now say which kind of grey it is, which the earlier overlay
could not: `board_check.py` drew "never asked" as one colour, and the 9% that
are refused at the rim look identical to the 77% that are simply buried. They
are different problems -- one needs a different picture, the other needs only a
padded crop and a retrain.

### Two rendering bugs found by looking at the output

Both were invisible in the numbers and obvious in the picture, which is the
argument for the tool:

* A rim tsum's circle was drawn **across the legend panel** -- its centre is
  within one radius of the edge, which is the whole 40.7% finding. Drawing now
  goes into a clipped view of the board area.
* The status line and the write counter were drawn at a fixed row 28, y=468, on
  a 456px canvas: **never rendered at all.** The one line that tells you whether
  your keystroke did anything. Now anchored to the bottom.

### Files

Added `scripts/label_board.py`, `tests/test_label_board.py` (13 tests).
**No gameplay file touched. 645 tests pass. Version 1.11.12.**

---

## 2026-09-07 06:40 -- Junk marking, added circles, and a retrain that honestly did nothing

### The labeller, second pass

Three edits now, not one:

    NAME     click a tsum, press a number or type a name
    JUNK     press `j` -- empty bowl, board graphics, an effect flash
    ADD      click bare board where a tsum plainly is, then name it

Single letters are commands and typing a name starts with `/`. The first
version started typing on any a-z, which meant `j` could never mean junk.

**Junk goes to `crops/labelled/board/`**, and that is not a dumping ground: it
is the class `scripts/reject_net.py` trains its "is this a tsum at all" model
against (`NOT_TSUM`). So marking junk improves the reject model directly. Its
docstring is the warning worth repeating -- 763 crops were once treated as junk
merely for being unlabelled and 21% turned out to be tsums the game had drawn
its highlight over. **An absence of a label is not a negative label; this key is
a label.**

**Added circles** get an index from 90 up, so they can never collide with a real
detection, and a line in `<out>/_manual.jsonl` so the circle is still there when
you come back to the board. `-`/`=` resize, `x` removes one, and `x` refuses to
touch a real detection.

### A defect found by looking at the junk folder

`reject_net.py` had `NOT_TSUM = ("board",)`. `crops/labelled/score/` holds **10
crops of the score readout** -- UI, not a tsum -- and they were being trained as
**chainable tsums**. Now `NOT_TSUM = ("board", "score")`. Small, but the wrong
sign, and it is the same "an absence of a label is not a label" rule pointed in
the other direction.

### No class may have zero training sessions

`scripts/classify.py` grew `--cover-classes` (on by default). A tsum is equipped
for a run of consecutive rounds, so a character appearing in ONE session lands
entirely on one side of any session split -- and landing on the held-out side is
not a hard test, it is no test at all. It moves the cheapest sessions back until
every class has training data, and prints each move:

    moved 1 session(s) into training so no class is left with none:
      Cleo   <- 20260904_115656_30352  (73 crops leave the held-out set)

### And the retrain did not improve anything, which is the finding

Trained at `--min-class 20`, 5,587 crops, 39 classes, best epoch 20, 97.9% on
its own held-out set against the shipped model's 94.9%.

**That comparison is worthless** -- different splits, different held-out sets.
Scored properly, both models on the **36 sessions NEITHER trained on**:

    OLD (shipped)   top-1 97.46%   macro F1 92.52%   986 crops, 27 classes
    NEW             top-1 97.46%   macro F1 92.67%   986 crops, 27 classes

**Identical.** The Cleo fix was a MEASUREMENT artifact, not a model deficiency:
it repairs the reported number and changes nothing about what the model can
read. I said earlier this was worth "+2.6 points of top-1" -- that was the
reported metric, and on a fair comparison the real gain is zero.

`models/character_new.*` is kept as a candidate and **the shipped model is
unchanged**, because there is no reason to churn an artifact for +0.15pp of
macro F1. `models/character.onnx.bak` is a backup either way.

### So where the recognition gain actually is

Not in more training on the same crops -- 97.5% on visible crops is at this
data's ceiling. Two places, and both are COVERAGE rather than accuracy:

1. **The 40.7% refused at the frame edge.** Well-visible tsums the model is
   never shown, because `_cut` refuses a crop the edge would clip. Padding the
   frame in BOTH the extractor and the runtime would take the readable share of
   a board from ~19% to ~28%. That is a re-extract and a retrain, entirely
   offline.
2. **`WhiteRabbit -> LittleOyster`, 10 errors** -- the largest confusion between
   two well-taught classes, and the only one on the list that is the model's
   fault rather than the dataset's. `--only-wrong` puts exactly those boards in
   front of you.

### Files

Rewrote `scripts/label_board.py` (junk, added circles, `/` to type).
Changed `scripts/reject_net.py` (`NOT_TSUM`), `scripts/classify.py`
(`--cover-classes`). **No gameplay file touched. 645 tests pass.
Version 1.11.13.**

---

## 2026-09-07 08:10 -- Junk on rim half-circles, and the wiring plan for recognition

### The refusal that was wrong, and the one that was right

`j` on a half circle at the board's rim was **refused outright**. That was my
error: a rim detection is usually not a tsum, and saying so must be possible.

But writing a crop for it is not the fix, and the reason is worth pinning:

* **clip and stretch it** -> `reject_net.py` learns *distorted = junk*;
* **pad the frame and cut a square** -> it learns *padded = junk*, because every
  other crop in the set is unpadded.

Either is train/serve skew -- the bug that cost 312 rounds. And it is moot:
`RejectModel` reads its crops through `_character_crop` too (tsum.py:851), so at
play time **it is never asked about an edge detection at all.** A crop written
there would train it for a question nobody poses.

So `j` on a rim detection now records the verdict in `<out>/_junk.jsonl` and
writes **no crop**. The circle turns junk-coloured, `d` takes it back, and the
count is the evidence for whether padding the frame is worth doing -- those are
precisely the detections padding would make trainable.

### Documented

`docs/LABELLING-TOOLS.md`: what each of the eight tools answers, the full key
map and legend for `label_board.py`, where every edit lands on disk, and the
three-step plan for wiring recognition into the app.

The plan's load-bearing paragraph is about **how** edge padding must be wired,
not whether: it has to reach `crops.py::_cut` and `tsum.py::_character_crop`
**together**, through ONE shared function, with a test asserting the trainer and
the runtime cut identically -- the pattern
`test_the_trainer_and_the_runtime_cut_the_same_colour` already set after the
last skew bug. Extractor and runtime disagreeing about a crop is the same class
of failure as extractor and runtime disagreeing about a colour.

### Files

`scripts/label_board.py` (`Verdicts`), `tests/test_label_board.py` (+5, now 18),
`docs/LABELLING-TOOLS.md` new. **No gameplay file touched. 650 tests pass.
Version 1.11.14.**

---

## 2026-09-07 10:20 -- The offline lab: golden set, shared crop rule, and edge padding measured

Executed the first engineering milestone from `TSUM_RECOGNITION_NEXT_PLAN.md`:
baseline + golden test set + shared crop function + edge-padding experiment.

### One crop rule, named by the model

`ttheart_sender/game/crop.py` is now the only place a crop is cut.
`scripts/crops.py::_cut` and `tsum.py::_character_crop` are both thin wrappers
over it, and `test_the_trainer_and_the_runtime_cut_the_same_crop` asserts they
agree byte for byte.

A `CropProfile` carries window, stored size, padding and border mode. **The
model records its profile in its own `.json` and the runtime reads it back**;
a profile this build does not have RAISES rather than falling back to the
default. Serving a padded-trained model unpadded crops would not error -- it
would just read as a worse model, which is precisely the shape of the bug that
cost 312 rounds.

### A frozen golden set

`scripts/golden.py freeze` -> **78 sessions, 1,306 crops, 42 classes**, in
`models/golden.json`. Session names rather than copies of the crops, so a
relabelling reaches the golden set too.

Two guards, because both failures are silent: `classify.py` subtracts the
golden sessions before splitting, and `recog_eval.py --golden` **refuses** a
model that trained on any of them.

**The shipped model is refused by that guard**, correctly -- it predates the
freeze and trained on 45 of the 78 sessions. It is not comparable to anything,
which is the entire reason the set had to exist: the last retrain read 97.9%
against 94.9% and was, on a common set, exactly as good. The comparable
baseline is now `models/character_candidate.onnx`: **top-1 96.8%, top-3 98.9%,
macro F1 95.0%** on the golden set.

Classes appearing in only one session are deliberately NOT frozen -- taking
their only session would make them impossible to learn, which is worse than
leaving them unscored. Seven are in that position.

### Confidence is a probability here, and that had to be checked

`recog_eval.py` now reports accuracy WITHIN each confidence band, not only the
coverage curve:

    0.00-0.50   n=13      0.0%
    0.50-0.70   n=17     47.1%
    0.70-0.80   n=9      66.7%
    0.80-0.90   n=11     63.6%
    0.90-0.99   n=45     84.4%
    0.99-1.01   n=1193   99.6%

Monotone, so a threshold is worth having. That is NOT true everywhere in this
project -- on buried crops the model stays 0.75-0.88 confident all the way down
to chance -- so it had to be measured rather than assumed.

### Edge padding, measured

Same model, same 200 boards, only the crop profile changed:

    profile      asked (0.55-65 / 65-75 / 75-100)   named   accuracy
    plain            70%     67%     51%             19%     99.5%
    padded_v1       100%    100%    100%             25%     99.5%

* **Coverage 19% -> 25% of detections named**, a 32% relative increase.
* **The refusal disappears entirely** -- every tsum above the visibility floor
  is now asked, in every band.
* **Accuracy unchanged** on the 426 checkable crops, exactly as
  `test_padding_changes_nothing_away_from_the_edge` predicts: padding only
  alters crops the edge was clipping, and every existing labelled crop is
  non-edge by construction.

Two limits stated rather than buried. I estimated ~28% coverage; it is **25%**.
And the newly reachable crops have **no labels**, so what is measured is that
they now exist, not that they are read correctly -- the marks column hints they
are (0.65-0.75 agreement 33% -> 78%) but n goes 3 -> 9.

`label_board.py --crop-profile padded_v1` is what labels that new population.

### Files

New: `ttheart_sender/game/crop.py`, `scripts/golden.py`, `models/golden.json`,
`tests/test_crop_profile.py` (13), `tests/test_golden.py` (10).
Changed: `tsum.py` and `scripts/crops.py` (both delegate), `classify.py`
(golden exclusion, `--crop-profile`, metadata), `recog_eval.py` (`--golden`,
confidence bands), `board_check.py` and `label_board.py` (`--crop-profile`).
**No gameplay behaviour changed. 673 tests pass. Version 1.12.0.**

---

## 2026-09-07 12:05 -- Readable labels, in-place renaming, and the coin that is not junk

### The board is legible now

The labels were drawn at font scale 0.32, unoutlined, on a bright multicoloured
board -- unreadable over roughly half the tsums. Three fixes:

* **`--zoom`, default 1.8x**, adjustable live with `[` and `]`. The BOARD is
  magnified and the text drawn on top afterwards, so labels stay crisp rather
  than being blown up with the pixels. A click is divided back into board
  coordinates before anything uses it.
* **Every label has a dark stroke behind it**, so it reads over any tsum.
* Panel fonts up, and the selected tsum's current name is now shown **large and
  first** under `NOW:`.

### Renaming was already possible and was not discoverable

The panel now says **CHANGE IT TO:** when a tsum already has a name, and marks
the current one `<- now` in the list. Two typo guards came with it:

* while typing, the list narrows to classes that MATCH the prefix;
* a name matching an existing class in any case commits with the folder's own
  spelling, so `marshmallow` cannot create a second folder beside
  `Marshmallow`.

That last one matters more than it looks: two folders differing by case train as
two different characters and nothing downstream can tell them apart.

### The coin: asked for, measured, refused

The player asked for `coin` to be treated as junk -- "it shouldn't be detected".
Reasonable, and **the data says the opposite.** Checking every labelled crop
against whether the game ever marked or cleared that detection:

    class                marked   cleared   verdict
    board                  0.0%      8.9%   clean negative
    score                  0.0%      0.0%   thin (n=6), same signature
    coin                  76.2%     42.9%   THE GAME SAYS TSUM
    unknown_lightball     86.4%     22.9%   the link highlight
    a real character     15-25%    10-20%   for comparison

**A `coin` crop is marked more often than any real character, and 42.9% of them
were cleared by a drag that worked.** Whatever is in that folder, the game
chains it -- most likely a tsum with a coin drawn over it. Adding it to the
negatives would train the reject model to discard detections that clear, which
`reject_net.py`'s own docstring records having done once with
`unknown_lightball` (86.4% marked): it scored 0.999 AUC while throwing away
game-confirmed tsums.

I had already made the change before checking. It is reverted, the numbers are
in the source beside the list, and `test_coin_is_not_treated_as_junk_by_the_reject_model`
pins it so nobody re-adds it from the folder name.

`score` stays: 0.0% marked and 0.0% cleared over the 6 of its 10 crops that can
be checked -- thin, but pointing the same way as `board` and never the other.

### Files

`scripts/label_board.py` (zoom, outlined text, rename panel, typo guards),
`scripts/reject_net.py` (evidence table beside `NOT_TSUM`),
`tests/test_label_board.py` (+5, now 23), `docs/LABELLING-TOOLS.md`.
**No gameplay file touched. 677 tests pass. Version 1.12.1.**

---

## 2026-09-07 13:30 -- Selection was the bug, and the negative class was never checked

### Why half-circles could not be marked junk

The `j` handler was correct. **Selecting them was impossible.** `pick()`
required a click inside the detection's own radius -- and `r` is the VISIBLE
radius:

    detected radius: p5 6.2px  p25 8.2px  median 10.8px  p75 14.0px
    under 8px:  20.0% of detections
    under 12px: 59.4%

A rim half-circle has the smallest `r` on the board, so **the detections most
worth marking junk were the least clickable.** Fixed two ways: selection takes
the nearest detection within **14px** (median centre spacing is ~61px, so still
unambiguous), and **adding moved to right-click**. One button could not serve
both -- a near-miss wants selection to be forgiving and adding to be certain.

### The negative class was never cross-checked

`reject_net.py` has always held that a crop the GAME marked or cleared is a tsum
whatever a person labelled it. That rescue only ever ran on the leftover pile in
`crops/all`. It never reached the negative FOLDER:

    crops/labelled/board: 705 crops, 418 checkable, 37 game-CONFIRMED (8.9%)

Thirty-seven real tsums were training as "not a tsum". `load()` now applies the
same test to `NOT_TSUM` folders and prints what it hands back:

    37 crop(s) in the NEGATIVE folders were marked or cleared by the game
       and are counted as tsums, not junk

This is also the standing guard against the coin mistake: press `j` on coins and
they land in `board/` at 76.2% confirmed -- and come straight back out, with a
count saying so.

### Files

`scripts/label_board.py` (SLACK, right-click add), `scripts/reject_net.py`
(rescue in `load`), `tests/test_reject_net.py` (new, 6),
`docs/LABELLING-TOOLS.md` (retraining section). **No gameplay file touched.
684 tests pass. Version 1.12.2.**

---

## 2026-09-07 14:15 -- Removing a circle, and why there is no delete key

Asked how to remove an incorrect circle. There is no delete key and there
should not be: a detection comes from a saved frame, so it cannot be taken out
of the past. **Marking it junk is how it is removed from the future.**

    left-click it  ->  j   turns red, reads "junk"
                       h   shrinks every junk-marked circle to a dot
                       d   takes the mark back

What `j` does depends on the circle: a normal detection's crop MOVES into
`crops/labelled/board/` (the class `reject_net.py` trains against); one already
named a character is moved out of that character; a rim half-circle records a
verdict with no crop; and a circle you added yourself is removed outright with
`x`.

Added `h` because "remove" plainly meant "get it out of my way". It shrinks
junk to a 2px dot rather than hiding it: **a circle you can no longer see is one
you can no longer take back**, and `j` is a training label rather than a delete
key, so it has to stay selectable for `d`. The selected circle is never shrunk,
or `d` would be unusable on it.

Stated in the docs where it was missing: **none of this changes a round.**
`flows/play.yaml` carries `reject_model: ""` -- the reject model ships off,
because a 143-round A/B lost. Marking junk builds the evidence for a better one;
it does not change play today.

### Files

`scripts/label_board.py` (`h`), `tests/test_label_board.py` (+2, now 25),
`docs/LABELLING-TOOLS.md`. **No gameplay file touched. 686 tests pass.
Version 1.12.3.**

---

## 2026-09-07 15:00 -- The first labelling session, checked on disk

91 crops from 3-4 boards. Everything landed, and two things are worth recording.

### New names make new classes, as intended

`Dory`, `Eeyore` and `Maleficent` were created automatically. `classify.py`
reads the folders, so they are classes the next time it trains -- nothing else
has to be told.

    Beast 18 (16 added by hand)   board 17      Tramp 10 (10)
    Dory 9 (1)                    LittleOyster 8 (8)
    Maleficent 6 (6)              Rex 6 (6)     Eeyore 5 (4)
    Goofy 4 (4)  BeastIdle 4  Maximus 2 (2)  Sulley 1 (1)  coin 1

Plus 19 junk verdicts on rim half-circles and 69 hand-placed circles recorded.

### 58 of 91 crops went in claiming v1.00, and that number is not measured

**64% were added by hand, and every one recorded `v=1.00`** -- because an added
circle defaults to the full board radius and the filename's visibility is
`r / radius`. The CROP pixels are right (cut at the board radius like every
other crop); the recorded visibility is an estimate the labeller never made.

It matters because `_v` is what `--min-visible` filters on and what every
report bands by. A half-buried tsum recorded as fully visible teaches the model
what "v1.00" looks like using a picture that is not.

The panel now shows `visible 1.00 (estimate, -/=)` in the added colour whenever
a hand-placed circle is selected, and the docs say so. Index >= 90 already marks
these, so any later analysis can separate estimated from measured.

### And the training distribution has changed

**31 of the 91 new crops are below 0.55 visible. The historic set had zero.**
Every one of the 5,642 crops before today was at or above that floor -- which is
why `CHARACTER_MIN_VISIBLE = 0.55` exists at all. Labelling buried tsums is not
wrong, but it is a different training set, and `docs/IDENTITY.md` section 5
measured 614 mark-labelled buried crops as having **no effect**. Worth measuring
against the golden set before assuming it helps.

### Files

`scripts/label_board.py` (visibility surfaced), `tests/test_label_board.py`
(+2, now 27), `docs/LABELLING-TOOLS.md`. **No gameplay file touched. 688 tests
pass. Version 1.12.4.**

---

## 2026-09-07 16:10 -- The retraining runbook, and a promotion gate that means it

`docs/RETRAINING.md`: label -> train -> score -> promote -> (separately) enable,
with what each step touches and whether it can change a round. The short answer
threaded through it is that **only the last step can**, and it is off for a
measured reason.

### The gate

A runbook is only worth writing if its checks cannot be skipped, so
`scripts/promote.py` holds them:

* **an unknown crop profile** -- a model trained on padded crops and served
  unpadded ones does not error, it reads as a worse model;
* **training on the golden set** -- then its score there is memory;
* **worse than the model in service on the golden set** -- the only comparison
  made on identical boards.

It backs up to `<name>.prev.onnx` first, and `--force` prints which check it
walked past.

Expect the first promotion to print `unmeasured`: the shipped
`models/character.onnx` predates the freeze and trained on 45 of the 78 golden
sessions, so it cannot be scored there. That is not a bug, it is why the set was
frozen -- the first promotion establishes the comparable baseline.

### And the accident the tool now prevents

**The first test run of `promote.py` promoted a model while merely being tried.**
No `--dry-run` existed. It swapped `models/character.onnx`, said so, and I put
the backup back -- but a promotion tool without a dry run invites exactly the
accident it exists to prevent. `--dry-run` runs every check and returns before
anything is copied, and `test_a_dry_run_returns_before_anything_is_copied`
asserts the guard stays ahead of the first `shutil.copy2`.

The shipped model is unchanged: 38 classes, held-out 0.9485, no crop profile
recorded (it predates them).

### Files

New: `docs/RETRAINING.md`, `scripts/promote.py`, `tests/test_promote.py` (7).
**No gameplay file touched. 695 tests pass. Version 1.12.5.**

---

## 2026-09-07 17:30 -- Why the newly labelled classes never trained, and one command instead of four

### The bug the player hit

Dory, Maleficent and Eeyore were labelled and then **did not appear in
training**. Cause: `--min-class` defaults to **20** and drops everything under
it.

    Tramp        19 crops   (+1)      <- one crop short
    Dory          9 crops   (+11)
    Maleficent    6 crops   (+14)
    Eeyore        5 crops   (+15)

It was printed -- one line in the middle of several hundred -- which is not the
same as being told.

### And `--min-class` was a lie when lowered

Setting `--min-class 5` let them past the filter and then `classify.py` aborted
anyway: `if min(counts.values()) < 20: return 1`, hard-coded, whatever the flag
said. The flag exists precisely so a newly labelled class can be trained before
it reaches 20 crops.

Now: under 20 with the cut at 20 or above is a filter bug and still stops; under
20 because you lowered the cut on purpose warns loudly and continues, saying
those classes can be learned but not scored.

### One command

`scripts/retrain.py` -- survey, train, score on the golden set, promotion checks.

    scripts/retrain.py --survey-only      what will train, what will not, and by how much
    scripts/retrain.py --dry-run          everything except the promotion
    scripts/retrain.py --min-class 5      include the small new classes
    scripts/retrain.py --reject           the "is this a tsum" model

It shells out to the same four scripts rather than reimplementing them, so
every step is still runnable alone. The survey runs FIRST, because the drop is
the thing that silently undoes an evening.

### What including the thin classes costs, measured

    --min-class 20 :  top-1 96.4%   macro F1 94.2%   (Dory etc. absent)
    --min-class  5 :  top-1 95.4%   macro F1 88.3%   (Dory etc. trained)

**The model did not get worse.** The report started including classes that were
previously invisible, and macro F1 weights a 9-crop class the same as a
500-crop one. Worth stating in the runbook so the fall is not read as a
regression.

### Files

New: `scripts/retrain.py`, `tests/test_retrain.py` (6). Changed:
`scripts/classify.py` (the thin-class guard), `docs/RETRAINING.md`.
**No gameplay file touched. 701 tests pass. Version 1.12.6.**

---

## 2026-09-07 18:20 -- 26 crops were invisible to the split, and I wrote them that way

Asked whether files can be dropped into `crops/labelled/` by hand. Checking the
contract found a bug in my own tool.

### The filename is a primary key, and mine stopped parsing

`<session>_<sample>_<index>_v<visible>.png` is how every tool gets from a crop
back to the detection it came from. The pattern accepted **exactly two digits**
of index. `label_board.py` numbers hand-added circles from `MANUAL_BASE = 90`,
so adding more than ten circles on one board produces index 100+:

    20260905_061026_8872_0005_108_v1.00.png   ->   NO MATCH

**26 crops from the first labelling session did not parse.** Nothing errors: a
crop that does not parse is filed under session `"unknown"`, and then it lands
on one side of the split with every other such crop, cannot be excluded by the
golden set, and cannot be filtered by `--min-visible`.

The pattern was duplicated in **seven scripts**. All widened to `\d{2,}`; all
5,730 crops now parse. `tests/test_crop_filenames.py` reads the pattern out of
every script and asserts each one handles a three-digit index, plus a corpus
check that no shipped crop is orphaned -- so the next copy cannot drift either.

### And the answer to the question

Yes, dropping files in works: a folder is a class, a `.png` in it is a crop.
Three things to know, now in `docs/RETRAINING.md`:

* `.jpg` is skipped in silence -- `load` globs `*.png`;
* the picture must be a square cut at one radius about a tsum's centre, or it
  is a different distribution;
* a filename that does not parse still TRAINS, but behaves as one shared
  session called `unknown`, which quietly weakens the held-out number.

### Files

`scripts/*.py` (seven filename patterns), `tests/test_crop_filenames.py` (new,
5), `docs/RETRAINING.md`. **No gameplay file touched. 705 tests pass.
Version 1.12.7.**

---

## 2026-09-07 19:05 -- 34 crops are labelled as two characters at once

Asked whether moving a crop from `BeastIdle/` to `Beast/` works. It does, and
nothing else needs touching: every tool derives a crop's class from the folder
it walks, and the sidecars key on `(session, sample, index)`.

The only way to get it wrong is to COPY instead of MOVE, so `retrain.py`'s
survey now checks. It found the mistake already in the corpus:

    THE SAME CROP IS IN TWO CLASSES -- 34 of them.

    22 + Sulley                     14
    CheshireCat + Sebastian         11
    LightningMcQueen + Sebastian     3
    BeastIdle + coin                 1
    CheshireCat + Piglet             1
    Grim + Rex                       1
    coin + score                     1
    22 + Sulley + Tramp              1
    Sulley + Tramp                   1

16 of the 34 are byte-identical; the rest share a filename -- and therefore a
session, sample and detection -- while differing in pixels, so they were cut at
different times.

**The two largest pairs are the model's two worst confusions.** `recog_eval`
reports `22 -> Sulley` and `CheshireCat -> Sebastian` at the top of its
confusion list, and those are exactly the classes with 14 and 11 contradictory
labels. A classifier cannot separate two characters when the same picture is
filed as both, so some of what has been read as visual similarity is a label
defect instead.

Not auto-fixed: which of the two labels is right is a person's call, and
deleting the wrong copy is a decision about the data rather than a cleanup.

### On BeastIdle -> Beast specifically

The model's own numbers, on the golden set: **Beast recall 37.8%, precision
100%**, and its single largest confusion is `Beast -> BeastIdle` at 23 crops.
It never calls something Beast wrongly; it misses two Beasts in three, and most
of the misses go to BeastIdle.

The game's marks cannot settle whether they are one character -- **0 samples**
have a labelled Beast head with a labelled BeastIdle partner or the reverse, so
there is no co-occurrence to read. That is the player's call to make from
seeing the game; if they are the same tsum, merging removes the largest
confusion on the board and roughly doubles Beast's recall.

### Files

`scripts/retrain.py` (`duplicates`), `tests/test_retrain.py` (+2, now 8),
`docs/RETRAINING.md`. **No gameplay file touched. 707 tests pass.
Version 1.12.8.**

---

## 2026-09-07 19:45 -- `junk` is a negative class now

Asked for a `crops/labelled/junk/` folder whose contents are all rejected. Wired
it, and created the folder -- it did not exist yet (only the `_junk.jsonl`
sidecar did, and a new `Gaston` class had appeared).

    reject_net.py     NOT_TSUM = ("board", "score", "junk")
    label_board.py    JUNK_CLASS = "junk"   -- `j` files there from now on
    six tools         NOT_CHARACTERS = {"board", "score", "junk"}

`board` keeps the 705 crops a person filled before the click tool existed and
stays a negative, so the two populations remain separable: `board` is verified
empty bowl and board graphics at 0.0% marked, `junk` is whatever gets pointed at.

**No evidence check was needed before adding it, and that is the point.** The
rescue added earlier covers every `NOT_TSUM` folder, so anything dropped into
`junk/` that the game MARKED or CLEARED is handed back as a tsum at train time
with a count saying how many. Verified end to end:

    NOT_TSUM = ('board', 'score', 'junk')
      Pascal -> y=1     board -> y=0     junk -> y=0

    with the game confirming one of them:
    2 crop(s) in the NEGATIVE folders were marked or cleared by the game
       and are counted as tsums, not junk

That is the guarantee `coin` did not have when it was added on appearance and
removed on evidence. A mistaken junk call now costs a line of output rather than
a worse model.

`junk` still becomes a class in the CHARACTER model, as `board` and `score`
already do -- the two models answer different questions, and a classifier that
can output "junk" is a soft reject. It is excluded from every identity metric.

### Files

`scripts/reject_net.py`, `scripts/label_board.py`, and six tools'
`NOT_CHARACTERS`; `tests/test_reject_net.py` (+2), `tests/test_label_board.py`
(+2). **No gameplay file touched. 711 tests pass. Version 1.12.9.**

---

## 2026-09-07 20:30 -- 23 contradictory labels resolved, and what `22` cost

Asked to auto-delete duplicates. `scripts/dedupe.py` does it, and **nothing is
deleted** -- losing copies move to `crops/duplicates/`, so a wrong call costs a
move back rather than a re-labelling session.

### Timestamps were tried first and rejected

The obvious rule is "keep the newest". It does not work here: most pairs carry
the **same mtime to the second**, because a move on one volume preserves it, and
where they differ they point both ways. Checked before being discarded.

### The rule that does hold, and it is asymmetric

**A negative beats a character.** If a crop is in `junk`/`board`/`score` on one
side and a character on the other, the negative wins -- because
`reject_net.py` re-checks its negatives against the game and hands back anything
the game MARKED or CLEARED. **A wrong junk call self-corrects; a wrong character
label does not.** That is a real difference in the consequences of being wrong,
not a preference.

**Two character labels give up both copies.** `22` against `Sulley`,
`CheshireCat` against `Sebastian`: the filename says which detection it is and
nothing says which name is right. Deleting one at random leaves a label that is
wrong half the time; keeping both guarantees a taught contradiction. Removing
both costs one crop and leaves the detection free to be labelled again.

    23 crops, 39 files moved
      kept in (none)   16     both were character labels
      kept in junk      6
      kept in score     1

### And the bill, which is the interesting part

    22        51 -> 16 crops   NOW UNDER THE TRAINING CUT
    Tramp     19 -> 17 crops   under it too
    Sulley   158 -> 142

**Over a quarter of class `22` was also filed as Sulley.** That is most of why
`22 -> Sulley` sat at the top of the confusion list -- read as visual
similarity, it was a label defect. The class is now too thin to train and needs
re-labelling rather than a retrain, which `dedupe.py` says on its way out.

### Files

New: `scripts/dedupe.py`, `tests/test_dedupe.py` (7). **No gameplay file
touched. 718 tests pass. Version 1.12.10.**

---

## 2026-09-07 21:15 -- Re-labelling when the board is gone

Asked for the re-labelling command. The honest answer needed a measurement
first:

    39 quarantined crops, across 10 boards
    38 of them from sessions NO LONGER in dataset/
    boards still re-openable: 1

    labelled crops whose board still exists:  Tramp 13, Sulley 1, 22 ZERO

**`label_board.py` cannot re-label any of `22`.** Every frame it came from has
been deleted from the corpus. That is the class the dedupe took from 51 crops
to 16, so it is precisely the one that needs work, and the board tool cannot
reach it.

### The tool that can was already here

`crops.py label` -- click the crops that are one character, press `s`, name them
once -- has existed since the labelling pipeline was built. It read its crops
from `clusters.json`, so it could only open an embedding cluster.

Now it takes `--dir`:

    python scripts/crops.py label --dir crops/duplicates/22

Nine lines: the paths come from a folder or from a cluster, everything after is
the UI that already worked. For a crop whose board is gone, the picture is the
only thing left to judge, and that is exactly what this UI shows.

### So the answer is two commands, not one

    board still exists   ->  scripts/label_board.py
    board is gone        ->  scripts/crops.py label --dir <folder>

Both write into `crops/labelled/<name>/` in the same format; `classify.py`
cannot tell which produced a crop.

### Files

`scripts/crops.py` (`label --dir`), `docs/RETRAINING.md`. **No gameplay file
touched. 718 tests pass. Version 1.12.11.**

---

## 2026-09-07 22:00 -- Jumping to a board, TAB completion, and what actually stops the junk circles

### Two gaps closed

**`--board 19`** opens the board the panel numbers 19, and
**`--session X --sample N`** opens one exact frame. The first is convenient and
only holds while `--seed` does; the second is stable, which is the one to write
down.

**TAB completes to the top match** while typing a name. The match list existed
and was display-only: digits could not pick from it, because `22` is a real
class name and a number key while typing has to mean the character 2. TAB was
the only key left.

### And the question underneath all of it

"How do I stop the tool drawing circles on junk after I have labelled it?"

The labeller does not decide what is detected -- `detect()` does, and it runs
before any model. Circles on empty bowl and board graphics are **false
detections**, and the thing that removes them is the REJECT model, which is
what the `junk`/`board` folders train. The chain is:

    label junk  ->  retrain.py --reject  ->  reject_model: "models/reject.onnx"

Only the last step changes a round, and it is off today because a 143-round A/B
lost: cleared -18.5, FEVER -5.9pp, and the collapse rate went 1.4% -> 12.5%.
That model deleted 7.7% of the board.

**But that verdict was about a different model.** It was trained on 763 crops
left UNLABELLED -- inference from silence, which the file's own docstring says
is not a label -- against 40 crops a person pointed at on purpose plus the 705
verified `board` crops now cross-checked against the game. Whether the new one
behaves differently is a question for a played A/B, not for an offline number,
and it is the player's rounds to spend.

### Files

`scripts/label_board.py` (`--board`, `--session`, `--sample`, TAB),
`tests/test_label_board.py` (+2, now 31), `docs/RETRAINING.md`.
**No gameplay file touched. 720 tests pass. Version 1.12.12.**

---

## 2026-09-07 22:40 -- "Do I have to label all 10,747 boards?"

No, and the question exposed a confusion worth writing down.

### What is actually labelled

    boards in the corpus              10,747
    boards with at least one label     1,399
    character crops                     5,137
    negatives (board 721, score 12, junk 148)   881

Training reads CROPS, not boards. A board with one crop labelled contributes
that crop. There is no threshold and nothing is wasted by stopping.

The 148 junk crops are **17% of the whole negative set** -- not a rounding
error. Retrained with them: **0.9878 held-out AUC**, and the cross-check handed
back **71 crops** the game had marked or cleared.

### The confusion: labelling does not remove circles

`detect()` draws the circles and **contains no model**. Colour clustering plus a
distance transform, run before anything is trained. Retraining cannot alter it,
and the same board opened tomorrow has every circle in the same place.

    a character name  ->  character.onnx  ->  WHICH tsum this is
    `j` junk          ->  reject.onnx     ->  NOT a tsum, drop it at play time

So junk labels never clean up the labelling tool. They teach the reject model to
drop that kind of detection **during a round**, and only after
`reject_model:` is set in `flows/play.yaml`. In the tool, a marked crop comes
back red reading `junk`, and `h` shrinks them out of the way.

### And a number not to over-read

An earlier smoke run printed 0.9927 AUC at one epoch, against 0.9878 now over
twenty. **Those are not comparable** -- the negative set gained 148 crops, the
dedupe removed 39, and the session split moved. Whether the new negatives help a
ROUND is a played A/B, not an offline number, and the last reject model to be
played lost 143 rounds.

### Files

`docs/RETRAINING.md`. **No code and no gameplay file touched. 720 tests pass.**

---

## 2026-09-07 23:30 -- The board filter is back on the panel, for a different model

The player would rather test what is labelled than spend a week labelling
first. That is this project's own rule -- a played round beats any offline
number -- so the job was to make those rounds READABLE rather than to talk them
out of it.

### The reject model, trained honestly

`reject_net.py` now honours the golden set: 78 sessions withheld entirely, so
`promote.py` stops having to be forced. That costs 1,941 crops of training data
and the AUC with it -- **0.9880 -> 0.9664** -- and the lower number is the true
one. The higher one was scored on sessions it had trained on.

    floor   fakes rejected   real tsums lost   kept per 40-tsum board
     0.10            82.8%              2.9%                   38.8
     0.25            85.5%              3.9%                   38.4
     0.50            87.1%              5.9%                   37.6

**At the shipped floor of 0.10 it removes about 3% of the board.** The model
that lost 143 rounds removed **7.7%**, and that was the whole mechanism: losing
real tsums cost more than the false ones did. Less than half as aggressive is a
real difference, and it is still not a promise.

### `promote.py` learned that a reject model is a different question

It was refusing every reject candidate, because scoring one on the golden
CHARACTER crops finds nothing to score -- its classes are not character names.
The LEAK check still applies, since nothing may train on those sessions
whatever question it answers; the SCORE check is now skipped for it and its
held-out AUC printed instead.

### The row is back, carrying its own obituary

`board_filter` returns to the panel after being retired for losing. The row's
note leads with **PLAYED AND LOST, 143 rounds, cleared -18.5, 7.7% of the board
deleted**, then says what is different: the old one trained on 763 crops left
UNLABELLED -- inference from silence -- and this one on crops a person pointed
at, cross-checked against the game.

`test_the_board_filter_is_offered_again_and_says_why` asserts the note still
carries the old result. A row that forgot it would be the same invitation that
got it retired.

### Files

`scripts/reject_net.py` (`--golden`), `scripts/promote.py` (reject models),
`ttheart_sender/tray/service.py` (row restored), `tests/test_tray.py`.
`models/reject.onnx` promoted, backup at `reject.prev.onnx`.
**722 tests pass. Version 1.13.0.**

---

## 2026-09-08 00:20 -- "10k boards, how can I finish" -- you do not, and 159 is the real number

The player looked at 10,747 boards and reasonably concluded it was hopeless.
Counted instead:

    character classes labelled          56
    crops already labelled           5,407
    crops needed to make EVERY existing class trainable   159

**159 crops, not 10,747 boards.** The big classes are saturated -- CheshireCat
553, Pascal 421, Monstro 330 -- and another Pascal crop teaches nothing. The
whole remaining job is a handful each of Gaston, Hades, MURandy, Dory, Rafiki,
Sisu, Olaf and about five more.

Also worth correcting: **the corpus is not Beast-only.** The equipped tsum is
Beast, but a board holds five characters drawn from the collection, and 56
distinct ones are already labelled.

### The reason it FELT like 10,747

There was no way to find the boards that hold a class you need. Two now:

    label_board.py --needs Dory      ranks boards by the model's belief it is there
    label_board.py --unsure          ranks by how much of a board it cannot name

### And the bootstrap, stated rather than hidden

`--needs Dory` cannot work on the shipped model: Dory has 9 crops, under
`--min-class`, so it was never trained -- **you need crops to train a class and
a trained class to find crops.** The tool says so and names both ways out:

    scripts/retrain.py --min-class 5 --dry-run
    scripts/label_board.py --needs Dory --model models/character_candidate.onnx

A class with 9 crops trains badly and still points at the right boards, which is
all the hunt needs. `--unsure` needs no class at all, which is how a genuinely
NEW character gets found.

### Files

`scripts/label_board.py` (`--needs`, `--unsure`, `--scan`),
`tests/test_label_board.py` (+2, now 33), `docs/RETRAINING.md`.
**No gameplay file touched. 724 tests pass. Version 1.13.1.**

---

## 2026-09-08 01:00 -- The thin-class survey, and two entries that were not work

Asked for a command that shows which classes are still thin. It exists --
`retrain.py --survey-only` -- and running it on the current labels showed two
things in that list that had no business being there.

### The progress since the last count

    5,407 -> 6,540 crops     56 -> 59 classes     37 -> 43 trainable

Sisu, 22 and Tramp all cleared the cut. Olaf went 4 -> 18.

### `score` was being listed as homework

`score`, `board` and `junk` are NEGATIVES -- they train the reject model, where
"thin" means nothing at all. Listing `score 12` beside the characters read as
8 crops of work that does not exist. Now split out:

    Not characters, so no target to reach -- these train the
    REJECT model instead: score 12

### And an empty folder was being counted as 20 crops of work

`Scrump` has zero crops -- a name typed before its crops arrived. It was
sitting in the list asking for 20. Now called out separately, because an empty
folder is a typo to fix rather than a labelling job.

### The number that matters

With those two removed, the real remaining job is stated once, at the bottom:

    134 MORE CROPS makes every one of them trainable.

Down from 159 at the last count, despite three new classes appearing.

### Files

`scripts/retrain.py` (survey split), `tests/test_retrain.py` (+3, now 11).
**No gameplay file touched. 727 tests pass. Version 1.13.2.**

---

## 2026-09-08 02:10 -- Names are clickable, and searchable

Asked for a way to pick a name with the mouse instead of retyping it on every
circle. The panel's list is now a hit-tested control:

* **every visible name is clickable** -- select a tsum, click a name, done;
* `/` and a letter or two narrows the list, which stays clickable;
* `1`-`9` and TAB still work, for the first nine and the top match.

`render` hands back the rectangle each row was actually drawn in, rather than
letting the click handler re-derive the geometry. Two copies of that arithmetic
is how a list stops matching what it selects.

### Three things that had to change with it

**The list was capped at nine** -- because nine is how many digit keys there
are. A mouse has no such limit, so it now shows every class that fits.

**The panel scaled with `--zoom` and should not.** Zoom exists to make a 40px
tsum readable; scaling the panel with it left room for **one name out of 67** at
1.8x. Panel metrics are now fixed and only the board labels scale.

**The rows ran through the footer.** A flat 60px reservation was not the height
of an eight-line footer at any zoom; it is computed from the row height now.

### And one I broke on the way

Adding the panel branch removed the line that scales a click back into board
coordinates, so at any zoom a board click would have selected the wrong tsum --
or none. Restored, with the ordering that matters written down: the panel's hit
rectangles are in WINDOW pixels, so the scaling has to happen after the panel
test, not before it.

### Files

`scripts/label_board.py`, `tests/test_label_board.py` (+5, now 38),
`docs/LABELLING-TOOLS.md`. **No gameplay file touched. 732 tests pass.
Version 1.13.3.**

---

## 2026-09-08 02:50 -- A search you can lock

The search cleared after every label, so the same two letters were retyped for
every tsum of one character -- and a board holds about forty tsums of five
characters, so that is the common case, not the corner one.

**`l` locks the current search.** It survives labelling; typed text does not.

    /  m a      narrows to Maleficent, Marie, Marshmallow, Maximus
    l           LOCKED "ma"
    click a tsum, press 3     click the next, press 3     ...
    l           clears it

### And locking gives the number keys back

While TYPING, `1` has to mean the character "1" -- there is a class called
`22`, so digits cannot double as shortcuts. A locked filter is **not text
entry**, so `1`-`9` pick from the list again. That is why this is a lock rather
than a "keep the last search" flag: the mode change is the feature.

### The canvas was sized by the wrong thing

At zoom 1.0 a 456px board gave a 520px canvas and the name list had room for
**one** entry. The list was being sized by the board, which has nothing to do
with how many classes exist. The canvas is now at least as tall as the panel
needs:

    zoom 1.0 -> 11 clickable names      (was 1)
    zoom 1.8 -> 14
    zoom 3.0 -> 40

### Files

`scripts/label_board.py`, `tests/test_label_board.py` (+4, now 42),
`docs/LABELLING-TOOLS.md`. **No gameplay file touched. 736 tests pass.
Version 1.13.4.**

---

## 2026-09-08 03:10 -- `l` could never have locked, and the reason is the design

Reported: typing `/mike` then `l` gave **`mikel`**, not a lock.

Correct, and unfixable as specified. The typing branch appends **every**
printable key before any command is reached -- it has to, or a class named
`22` could not be typed. So `l` mid-typing is the letter l, and no ordering
change rescues it without breaking names.

The lock key has to be one **no class name contains**. `/` is that key, and it
is the one that STARTED typing, so it reads as a toggle:

    /  m i k e     type
    /              LOCK
    l              clear the lock (safe -- you are not typing then)

`l` is now unlock-only, and says so where it is defined. The typing line shows
`> mike_    (/ locks)` so the key is visible at the moment it is wanted.

### Files

`scripts/label_board.py`, `tests/test_label_board.py` (+1, now 43, one
rewritten), `docs/LABELLING-TOOLS.md`. **No gameplay file touched. 737 tests
pass. Version 1.13.5.**

---

## 2026-09-08 03:40 -- "Does cropping the same tsum twice help?" No, but the real answer is about ROUNDS

Asked whether multiple crops of one tsum on one board helps. Measured the
corpus rather than answering from theory.

    hand-added circles: 1598 across 123 boards
    pairs closer than one radius (the SAME tsum twice): 4

So it is barely happening -- though one pair sits at **0.0px apart**, which is
the same picture labelled twice.

### The distinction that matters

Two circles on the SAME tsum are near-identical pictures: no new information,
and it lets a class pass the 20-crop cut without having 20 examples. Several
circles on DIFFERENT tsums of one character on one board is normal and fine --
a board genuinely holds 8-12 of a character.

### But crops were the wrong thing to be counting

**The split is by SESSION.** A class whose crops all come from one round lands
entirely on one side of it: trained and never scored, or scored having never
been trained. The second is exactly how Cleo read 0% recall on 41 crops it was
never shown one example of -- and nothing in any report was counting rounds.

Now it does:

    PASSES THE CROP CUT BUT COMES FROM TOO FEW ROUNDS.
      Flounder            37 crops from 1 session(s)  <- one round only
      Cleo                49 crops from 2 session(s)
      Label these on a DIFFERENT board, not more of the same one.

The player's new classes are healthy by this measure -- Dory 70 crops over 7
rounds, Rafiki 75 over 7 -- so the labelling has been spread well. Only the two
inherited classes are fragile.

### Files

`scripts/retrain.py` (`sessions_per_class`), `tests/test_retrain.py` (+3, now
14, one tightened). **No gameplay file touched. 739 tests pass. Version
1.13.6.**

## v1.13.9 -- the panel names the board, live, with a headline that qualifies it

The character model reports itself twice and both lines go to the log: one when
it loads, one when the round is over. Neither is visible while a round is being
watched, which is when the question is asked. Two evenings went to doubting a
model that was working, because 19% named looks exactly like 0% named from the
outside.

This took two attempts to get right, and both were half of it:

* a census published at ROUND START -- names, but stale before they could be
  read, and silent on whether they were correct;
* an agreement score with no names -- correct, but nothing a person could
  check by eye, and the first thing it produced was `recognition 100% (2 of 2)`,
  which is not a verdict and reads as one.

**Both halves, together.** Under the Experiments tick boxes, updated every
frame:

    matches the game 34/39 (87%)
      Beast    12
      Dory      7
      Sulley    3

The list is what a person checks against the screen -- the board is right
there, and a panel saying Beast while the screen is full of Dory is wrong in a
way no percentage communicates. The headline says whether to trust it.

* **The raw count always travels with the percentage.** `100%` on two checks is
  not a claim about the model and is read as one without `(2/2)` beside it.
* **Why that count grows slowly, now written down where it is computed.** Marks
  are read on a sampled press, one in four. The game marks a mean 6.1 tsums per
  press and only the ones the model named can be scored -- about a fifth. That
  is roughly one check every four chains: single figures in the first minute,
  near 25 by the end of a ~95-chain round.
* **Unchanged lines wake nobody.** The board is re-read sixty times a round and
  most frames name it exactly as the last one did, so `publish` compares before
  it notifies. Live updates cost a tuple comparison, not a window repaint.
* Four states that used to be indistinguishable are four lines:
  `character model off`, `naming 0 of 43 -- no marks yet / nothing named above
  the floors`, `no marks read -- tick Data collection`, and a real score. The
  end-of-round swap replaces only the HEADLINE, so the last names stay up.

**What the score is not.** The group's name comes from the model's own reading
of the pressed tsum, so this is agreement with ITSELF across a set the game
says is one character. A model that calls the whole group Dory when it is
really Beast reads 100%. That limit is in the docstring and
`scripts/board_check.py` stays the place the absolute number lives -- but the
failure that costs a round is one character read as SEVERAL, which is what
stops a chain being found, and this catches exactly that. The end-of-round log
line names the worst mix-ups.

The panel block still costs no height until a round publishes into it:
`ControlPanel._reflow` hides the unused rows and moves the eleven controls
below them up. Measured: 848px collapsed (unchanged), 960px full, against about
1040px of work area on a 1080p desktop. `tests/test_readout.py` is also the
only test that runs the panel's layout arithmetic at all -- the tray tests mock
`ControlPanel` out entirely -- and it asserts the collapsed height has not
moved.

**764 tests pass. One fails on purpose:**
`test_the_character_model_ships_off` guards the house rule that a new detection
rule ships off in `flows/play.yaml`. It is red because the character model is
armed for a live trial. Setting `character: ""` is the revert and turns it
green again.

## v1.14.0 -- the character model makes the grouping WORSE. Measured.

The player asked the question nothing here had answered: *"if it doesn't
recognise in the first place but the app plays smoothly, then the character
model isn't needed -- what is the point?"*

Every scorer in this repo asks whether the model reads a CROP correctly.
`recog_eval` reads 94.5% on the golden set, `board_check` reads 100% against
human labels and 86.7% against the marks. None of them asks whether being
correct changes the grouping the round actually plays on, which is the only
reason the model exists at all.

`scripts/naming_gain.py` asks it. `adjacency()` and `find_chains()` group by
`kind`; the marks say which tsums are genuinely the same character and
reachable. So: score the k-means grouping against the marks, then score the
grouping the model produces, on the SAME drags, played with the model off.

**800 sampled drags:**

                         recall   precision   partners offered
      colour clusters     43.3%       28.6%    11.04
      + character model   38.2%       30.0%     9.08
      change              -5.2pp      +1.4pp    -1.96

      paired on the same drag: better on 4, worse on 150, unchanged on 646
      sign test p = 0.0000

**The mechanism, and it is not subtle.** The chain head is named on 14% of
drags. On the other 86% the head keeps its colour cluster while some of its
partners get renamed -- and a renamed partner leaves the head's group. The
shipped rule REPLACES `kind` with the model's identity, so naming one member of
a pair splits the pair. The model is not wrong; the rule that consumes it is.

**The fix that follows from the mechanism, also measured.** `--rule merge`
starts from the clusters and only ever JOINS: two tsums the model names the
same way end in one group, and two tsums k-means grouped stay grouped whatever
the model thinks of one of them. It removes the harm and adds nothing --
**+0.4pp recall, precision unchanged, better on 3 of 300 and worse on 0,
p = 0.25.** k-means already gets colour grouping right; the model rarely
disagrees in a way that helps.

**So the character model should stay OFF**, and the labelling is not what is
holding the score back. What the labelling did buy is this measurement -- there
was no model to test the idea with before it -- and an offline lab that can
refuse a bad one.

**Where the recall actually goes.** 57% of the tsums the game marks are in a
DIFFERENT colour cluster from the head, so identity really is the gap. But
70% of a board is under 0.55 visible and the model is at the chance rate down
there at unchanged confidence, so it can address at most the fifth of the board
it can see -- and in practice it breaks more than it fixes. More labels of
VISIBLE crops cannot reach the buried four fifths. `mark_probe.py harvest` is
the documented path: train on mark-labelled buried crops and re-run. If the low
bands stay flat, the pictures are the limit and no amount of labelling changes
it.

A bug in the first version of this script is worth recording: it cropped
`_before.jpg` by the board rect, but that jpg IS the board crop already. The
double crop fed the model a sliver and refused 82% of heads for falling off the
edge. `board_check.py` had it right; the numbers above are from the fixed
version, and both versions reached the same conclusion.

**771 tests pass. One fails on purpose:** `test_the_character_model_ships_off`
is red because `character:` is armed in `flows/play.yaml`. Setting it back to
`""` is the revert and the recommendation.

## v1.14.1 -- `junk` was teaching the reject model "hard to see", not "not a tsum"

The player asked how to regenerate the reject model, saying board and junk were
still being detected. The first answer was that nothing was filtering them --
`reject_model: ""` in `flows/play.yaml`. The second was that a retrain was
overdue: the model in service learned junk from 148 crops and the folder now
held 1,495.

**The retrain made it worse, and the promotion gate let it through.**

    held-out AUC          0.9664  ->  0.9114
    fakes rejected @0.10   82.8%  ->   48.3%
    real tsums lost @0.10   2.9%  ->    4.5%

Worse on BOTH sides at once, which is the signature of a shortcut rather than a
weaker model. Rolled back to `reject.prev.onnx`, verified by hash.

**Why the gate missed it.** `promote.py` compares character candidates on the
golden set. A reject model cannot be scored there -- its classes are not
character names -- so that branch printed its AUC and compared NOTHING. It
promoted whatever it was handed. It now reads the live model's `held_out_auc`
and refuses a candidate more than `AUC_TOLERANCE` (0.01) below it. Confirmed
against the bad candidate: `REFUSED -- worse held-out AUC (0.9114 against
0.9664)`.

**Why the retrain went wrong.** Measured over the corpus:

    board  721 crops   median 0.76 visible   99% at or above 0.55
    junk  1495 crops   median 0.38 visible    8% at or above 0.55
    real  8005 crops   median 0.72 visible

A `board` crop is a printed empty slot and looks like one at full size. A
`junk` crop, as the folder is actually used, is mostly a detection nobody could
identify -- and **a detection nobody can identify is what a buried real tsum
looks like.** 70% of a live board is under 0.55 visible. 359 of the negatives
were game-confirmed tsums outright.

**The fix: negatives get their own floor.** `reject_net.py
--negative-min-visible` drops negative crops below a visibility a person could
actually have judged. At 0.55:

    held-out AUC          0.9664  ->  0.9921
    fakes rejected @0.10   82.8%  ->   90.7%
    real tsums lost @0.10   2.9%  ->    1.6%

Promoted through the gate.

**One honesty note, written into `AUC_TOLERANCE`.** AUC is measured against
whatever negative set the candidate trained on, so changing which crops count
as negatives changes the test set too -- a candidate that dropped every hard
negative would score beautifully. `real tsums lost` is the comparable number,
because the positives do not move, and it is what a board filter has lost 143
rounds on before.

**768 tests pass. Four fail on purpose**, all the same guard: `play.yaml` now
arms both `character:` and `reject_model:` for live trials, and the house rule
is that a new detection rule ships off. Setting either back to `""` turns its
guards green.

## v1.15.0 -- the character model could name a detection `board`, and chains are built by the name

The player, watching the new panel readout: *"mostly detected is
board/junk/unknown_lightball, although the reject model is armed and the
checkbox is ticked, so I don't know why it's still being detected."*

The panel was telling the truth. `classify.py` trained on **every folder** under
`crops/labelled` and excluded nothing, so every character model ever built
carried `board` and `junk` as identities a detection could be given:

    models/character.json.bak    38 classes   coin, unknown_lightball
    models/character.prev.json   39 classes   board, coin, unknown_lightball
    models/character.json        64 classes   board, junk, coin, unknown_lightball

**Why it is worse than a wasted output.** `apply` writes the winning class into
`kind`, and `adjacency`/`find_chains` group by `kind`. So every detection named
`board` was grouped WITH THE OTHER BOARD DETECTIONS and offered as a chain. The
negatives were not merely present in the output; they were chainable.

**And it is a different failure from the reject model.** The two run in order --
`rejector.keep` drops what is not a tsum, then `characters.apply` names what
survived. A survivor named `board` is the classifier's fault, not the filter's.
Both were reported as "board is still being detected" and only one of them was.

`classify.py` now has `NOT_CHARACTERS = ("board", "score", "junk")`. Two tests
guard it: one on the constant, one on the model actually in service.

**`coin` and `unknown_lightball` stay**, on evidence. Against the game's own
marks and clears:

    unknown_lightball  178 crops   72.5% confirmed
    coin                92 crops   41.3% confirmed
    junk              1495 crops   21.5% confirmed
    board              721 crops    5.1% confirmed
    Beast              327 crops    4.9% confirmed   <- a real character

They are real objects the game links, and grouping coins with coins is what the
game itself does. `junk` being confirmed more often than `Beast` is the same
label-noise finding that broke the reject model in v1.14.1.

**Removing the negatives made the model better, not just cleaner:**

    golden top-1   94.76%  ->  96.88%   (+2.12pp)
    golden macro F1 88.67%  ->  93.32%   (+4.65pp)

## The panel readout shows the round's PEAK, not the current frame

*"the tsum name appear and remove appear and remove from each action."*

Of course it did. The board is re-read sixty times a round, tsums fall through
most of it, and only the fifth above the visibility floor is ever named -- so
which fifth changes every frame. A running total is no better; it grows without
bound. `CharacterModel.peak_names` records the most of each character seen on
any one frame this round: stable, monotonic, and an answer to the question
actually being asked -- *what is on this board*.

## The proposed playstyle is the shipped one

> "understand whole board pattern, including what tsum is included, look for the
> main tsum object with at least 3 chain, if no exist then look for other 4 kind
> tsum object, clear any longest chain, then repeat."

That is `find_chains` as written:

    chains.sort(key=lambda c: (c.is_base, len(c)), reverse=True)

with `MIN_CHAIN = 3` and `min_chain: 3` in the flow. Chains are built for every
kind on the board, the equipped tsum sorts first, then the longest, and the loop
repeats each frame. The step that was NOT working is the first one -- knowing
what is on the board -- and that is what the two fixes above address.

**772 tests pass. Four fail on purpose:** the ships-off guards, because
`character:` and `reject_model:` are both armed in `flows/play.yaml` for the
live trial.

## v1.16.0 -- "Show live recognition": a window of its own for the whole list

Asked for directly: a tick box in the Experiments section that opens a second
panel showing every recognised tsum, live.

It also fixes a compromise the readout had been living with. The control panel
is 870 logical px against about 1040px of work area on a 1080p desktop, so the
in-panel list was capped at five characters and the rest counted -- and five is
the whole board only when nothing is misread, which is the case nobody needs a
readout for. A board holds forty tsums; the interesting reading is the long
tail, and a cap is exactly what removes it.

**One publisher, two views.** `readout.panel()` now returns the FULL list --
headline plus every character, names padded so the counts line up -- and
`readout.fit(lines, rows)` is the control panel's share of it. Capping at
publish time would have meant publishing twice, or the window showing a list
the panel had already truncated.

    WINDOW                        PANEL (fitted to 5 rows)
    matches the game 34/39 (87%)  matches the game 34/39 (87%)
    Beast               12        Beast               12
    Dory                 7        Dory                 7
    unknown_lightball    4        unknown_lightball    4
    Sulley               3        +3 more -- tick Show live recognition
    coin                 2
    Mike                 1
    Rafiki               1

`fit` budgets for its own overflow line. The first version did not, and cut the
"+N more" back off -- the exact failure the function exists to prevent, caught
by a test rather than by a round.

**`ttheart_sender/tray/recognition.py`** is a separate Win32 window, and short
because it uses a multiline read-only EDIT rather than a row of labels. The
control panel manages one static per row because each has to be clicked, greyed
or recoloured; none of that applies to a list. An edit takes the whole thing as
one string, scrolls itself, and needs no layout pass when the line count changes
-- which it does on every frame of a round.

* Shown with `SW_SHOWNOACTIVATE`. This is watched while the emulator has focus,
  and stealing focus from the game is how a round gets lost.
* Parked against the control panel's left edge. A reading you have to move a
  window to see is one nobody reads.
* Closing it with its own X puts the tick box out, via `on_close`. Without that
  the box stays ticked over a window that is gone and the next tick does
  nothing.
* Hidden, never destroyed, so re-ticking does not rebuild it.
* Remembered in `PanelSettings.show_recognition`, and reopened at launch if it
  was open at exit. Off by default -- it is a debugging view.

**The panel's own block shrank from 7 rows to 5.** It was `readout.LIMIT + 2`
while it was the only place to see the names; now the window holds all of them
and the panel's job is the headline and a glance. Net effect on height, with
the new tick box included: 848 -> 870 collapsed, 950 full, against ~1040px of
work area. It still costs nothing until a round publishes -- `_reflow` hides
the unused rows and moves the eleven controls below them up.

**779 tests pass. Four fail on purpose:** the ships-off guards, because
`character:` and `reject_model:` are both armed in `flows/play.yaml` for the
live trial.

## v1.16.1 -- the live list is only in its own window now

Asked for straight after v1.16.0: take the live text out from under the
Experiments tick boxes and leave it in the new window.

Right call, and it removed more than five rows of labels. Those rows had to
collapse when empty or the panel opened taller than it ever had -- so
`_build_controls` recorded the x and y of every control built after them,
`_reflow` hid the unused rows and moved eleven controls up, and `refresh`
resized and re-parked the window. All of that existed to fit a list into a
window of fixed-height clickable controls, which is not what a list is.

Gone with it: `RECOG_ROW`, `RECOG_ROWS`, `ID_RECOG_BASE`, `_reflow`, `_below`,
`_recording`, `_content_height`, `_recog_shown`, the recording hooks in the
three control creators, `readout.fit`, `readout.LIMIT`, and the `recognition`
key in the panel state dict that nothing read any more. `_build_controls` ends
at `self._resize(y)` again, and the panel is a flat 870 logical px whatever a
round is doing.

`readout.panel()` was already publishing the full list for the window, so
nothing about what is shown changed -- only where. The "Show live recognition"
tick box stays exactly where it was, in the Experiments section.

**777 tests pass. Four fail on purpose:** the ships-off guards, because
`character:` and `reject_model:` are both armed in `flows/play.yaml` for the
live trial.

## v1.16.2 -- the live window never opened, twice over

Reported: *"I don't see the new panel show when I checked the box, and in game
also not appearing."* Two separate bugs, and the second would have hidden the
first being fixed.

**1. It threw on every tick.** `win32gui.CreateFont` does not exist -- pywin32
does not export it. The AttributeError came out of `_create`, through
`_show_recognition`, into the panel's WM_COMMAND handler, and the only symptom
a person saw was a tick box that did nothing. `panel.py` has always built its
fonts with `ctypes.windll.gdi32.CreateFontW`; this module did not.

**2. It opened behind the game.** Shown with `SW_SHOWNOACTIVATE` and no z-order
-- correct about not stealing focus, wrong about everything else. The entire
use of this window is reading it while the emulator has focus and a round is
running, and a window the game covers is not a readout. It is `HWND_TOPMOST`
now, and that is not optional.

**Why no test caught it.** Every other test in `test_readout.py` stubs Win32
out, which is right for layout arithmetic and useless for "does this window
exist". `test_the_window_really_builds_on_this_machine` creates the real thing
and looks at it: the caption, the TOPMOST bit, the text reaching the edit
control, and hide/show. It skips where there is no `win32gui`.

**And a failure is no longer silent.** `_show_recognition` now catches, logs,
puts its own tick box out and raises a tray notification. A view must not take
the tray down, but it must not fail quietly either -- quiet is what turned a
one-line bug into a bug report.

**778 tests pass. Four fail on purpose:** the ships-off guards, because
`character:` and `reject_model:` are both armed in `flows/play.yaml`.

## v1.17.0 -- the five-character rule, and why Beast is never named

Two observations from the player, both correct, both measurable:

> "It is supposed to have only 4 tsum kinds plus the base tsum (Beast) each
> round. For now it detects more than 6, treating a real tsum as some other
> fake tsum. And Beast doesn't appear in the list until the Beast skill
> activates."

### The board is over-segmented, and that is the root of both

Over 5,528 stored boards, k-means produces **7 or 8 colour clusters** on a
board that holds **five characters**:

    distinct clusters on one board: 7 x1366, 8 x1342, 6 x927, 9 x831, 5 x484

So the average character is split across about one and a half groups before
any model is involved. `find_chains` groups by cluster, so half of every
character is already unreachable from the other half.

### Why Beast is never named

Not the model. On the frozen golden set **Beast reads 99.2% F1** -- 100%
precision, 98.4% recall over 62 held-out crops.

It is the base cluster. Over 3,000 Beast-equipped boards, the cluster
`read_base_kind` picks:

    is EMPTY on                     15% of boards
    holds a median of                6 tsums (13% of the board)
    is the BIGGEST group on only    10% of boards

Beast is guaranteed present and usually plentiful, so a group holding 13% of
the board and absent on one board in seven is not the whole of it -- it is one
of the two-or-more clusters Beast has been split into. The player's own detail
fits: the skill firing changes how those tsums are drawn, they clear the 0.55
visibility floor, and only then does the name appear.

### The five-character rule, implemented and measured

`scripts/vocab_probe.py` accumulates the model's votes over a round, keeps the
top N names, and hands everything else back to its colour cluster. Scored
against the game's own marks, the way `naming_gain.py` scores:

    vocabulary   names refused   recall    change   better / worse
    none (ships)          0%      37.8%         -        -
    top 5 (the rule)      1%      37.9%    +0.1pp    1 / 0
    top 3                 5%      38.6%    +0.8pp   10 / 0
    top 2                14%      39.3%    +1.5pp   21 / 0
    top 1                44%      41.1%    +3.2pp   39 / 0
    NO NAMING AT ALL       --     43.3%    +5.5pp   (from naming_gain.py)

**It never makes a single drag worse at any setting** -- worse on 0 of 307
every time -- and it is monotone. Every name the model is allowed to use costs
recall, and the best rule is the one that names least. The limit of the curve
is naming nothing, which is exactly the v1.14.0 finding reached from the other
direction.

The player's rule is right about the game. Applied at 5 it barely bites,
because a round only spends a median of **4** names anyway: the tail of ones
and twos that looks so wrong in the panel is a small share of the naming, and
the damage is done by the confident majority, not by the tail.

**So the recommendation does not change: `character: ""`.** What would change
it is fixing the split -- one character in one group -- and that is a
clustering problem, not a recognition one.

**782 tests pass. Four fail on purpose:** the ships-off guards, because
`character:` and `reject_model:` are armed in `flows/play.yaml`.

## v1.18.0 -- five groups is neutral, four is a trade, and the metric decides which

The player asked for 5 forced colour groups instead of 4, and supplied the rule
behind it: a round holds the equipped tsum plus four random others, and an
in-game item cuts that to four INCLUDING the base. `_regroup`'s docstring
already carried the same rule and a table from `group_eval.py` -- groups=4 best
at +4.6%, groups=5 at +2.4% -- on the metric the twenty-sixth round
disqualified when an ORACLE grouping scored below the shipped rule on it.

`scripts/group_probe.py` re-asks it against the game's own marks. `_regroup`
only rewrites `kind`, so the same detections are re-sorted and every index
still lines up with the marks it is scored against.

**Three metrics, and they do not agree. That is the finding.**

Scoring the whole same-kind group:

    groups   made   recall   precision
    k-means   7.3    42.4%       29.6%
    3         3.0    52.8%       23.4%   +10.4pp
    4         4.0    46.6%       26.5%   +4.2pp
    5         5.0    42.4%       28.0%   -0.0pp

But that charges a grouping rule for same-character tsums buried across the
board, which `find_chains` would never have offered either -- and the bias
grows with group size, landing hardest on the rule under test. Scoring only
what the round could REACH (adjacency from the head, capped at 12):

    groups   made   recall   precision   offered
    k-means   7.3    28.2%       52.2%     4.15
    3         3.0    33.8%       43.3%     6.20   +5.5pp
    4         4.0    29.8%       50.0%     4.79   +1.6pp
    5         5.0    27.3%       54.6%     3.97   -0.9pp

Recall up with precision down is still not a verdict: a bigger group offers
more partners and more strangers at once. **The game breaks the tie.** It
accepts 97.9% of first members and 19.1% of sevenths, and once it refuses one
it refuses 86% of what follows -- so a chain is worth the RUN of correct
members before the first wrong one, not the total it contains:

    groups   ACCEPTED RUN            paired vs k-means
    k-means      1.43                 --
    3            1.16   -19%          better 26, worse 67
    4            1.28   -10%          better 19, worse 42
    5            1.34    -6%          better 16, worse 27
    6            1.41    -2%          better 10, worse 13
    7            1.36    -5%          better  3, worse 19

**On the metric grounded in measured game behaviour, per-frame k-means beats
every forced count, and forcing fewer is monotonically worse.**

**What that means, and it is not what either of us expected.** The five-
character rule is TRUE about the game. Forcing k-means to five piles of face
colour does not recover it -- it merges the wrong tsums. The count was never
the problem; which tsum lands in which group is. That is also why k-means at an
adaptive 7.3 groups beats a forced 7 (1.43 against 1.36): it is not the number
that helps, it is fitting the number to the board.

**Caveat, stated because it could change the answer.** The accepted run walks
the reachable set NEAREST-FIRST, and the round walks `longest_path` then
`orient_chain`. The ordering is a proxy for the one the game is actually shown.

**787 tests pass. Four fail on purpose:** the ships-off guards, because
`character:` and `reject_model:` are armed in `flows/play.yaml`.

## v1.19.0 -- recognition IS a better identity signal, and still loses

The player's argument, and it follows from every finding so far:

> "It merges the wrong tsums. This is why recognition must be in place... I
> want the app to know only 5 kinds are in the game and recognise all of them
> correctly, not guessing, so it won't link the wrong tsum. Each tsum I
> labelled has its face pattern with colour -- just train the app to remember
> those."

`scripts/embed_probe.py` builds it. Not classification -- the classifier
reaches a fifth of the board and refuses the rest -- but grouping in the space
the classifier LEARNED. MobileNetV3-Small's penultimate layer is a 1024-d
description of a tsum face trained to separate characters, which is exactly
"the face pattern with colour". Four arms, all scored on the chain the round
could drag, against the game's own marks:

    feature                       recall   precision   offered    RUN
    median face colour (ships)     28.7%      49.1%      4.45     1.37
    median face colour, 5 piles    26.8%      53.5%      4.13     1.24  -10%
    learned face, 5 piles          20.9%      69.0%      2.30     1.17  -15%
    nearest remembered face        20.0%      71.9%      2.09     1.08  -21%
    nearest of the round's 5       23.5%      68.7%      2.49     1.24  -10%

**The diagnosis is confirmed.** Median face colour groups two tsums correctly
49% of the time. The learned face gets **69-72%**. Colour really does merge the
wrong tsums, by almost exactly the margin the player described.

**The five-character rule helps too, and measurably.** Restricting prototypes
to the round's own cast beats offering all sixty-five: RUN 1.24 against 1.08,
**+15%**. Both halves of the proposal do what they were expected to.

**And it still loses to plain colour**, because recall falls from 28.7% to
20-23%. High precision, low recall, and the shape says what is happening:
recognition does not merge the wrong tsums, it SPLITS the right ones. A buried
Beast and a well-seen Beast do not land in the same pile, and 70% of a board is
under the visibility floor. Colour is wrong more often but offers nearly twice
as many partners, and the game's refusal rule rewards the longer correct run.

**So the one lever left is the pictures, not the algorithm.** Five ways of
using identity have now been measured -- naming, a per-round vocabulary, forced
group counts, embedding clustering, remembered prototypes -- and each is
limited by the same thing at the same place. `mark_probe.py harvest` is the
only untried idea that attacks it: the game's marks are free labels for BURIED
crops, which is the exact population every method above fails on.

### Two bugs found by building it

* **Eleven groups instead of five.** A tsum whose crop cannot be cut at the
  frame edge -- about 30% of a board -- kept the id it already had, so five new
  piles plus the old leftovers came to eleven. More fragmented than the rule
  being replaced. They are now placed by face colour into the nearest pile the
  embedding built, and `test_every_tsum_lands_in_one_of_the_groups` holds it.
* **The live window's TOPMOST bit was not sticking.** `SetWindowPos` with
  `HWND_TOPMOST` can be refused, and under the full suite it silently was --
  ex-style 0x180, no topmost bit -- which is exactly the "in game also not
  appearing" report. It is declared in `CreateWindowEx` now, where it cannot
  be lost, and the real-window test catches it.

**792 tests pass. Four fail on purpose:** the ships-off guards, because
`character:` and `reject_model:` are armed in `flows/play.yaml`.

## v1.20.0 -- the harvest answers it: the buried crops cannot be read

The last untried lever. `mark_probe.py harvest` uses the game's own marks as
labels for BURIED crops -- the population every identity method has failed on,
and the one no amount of hand labelling reaches, because a person labels what
they can see.

**The crops are exactly what was missing:**

    crops/marked    2765 crops   median 0.44 visible   71% BELOW the floor
    crops/labelled  8005 crops   median 0.72 visible    5% below

Harvested from `--half first` only, so the other half scores it; golden
sessions are withheld by `classify.py` after the merge, so nothing leaks.

**Scored on one fixed set of groups (`--head-model` pinned), on the half that
was never harvested:**

    partner visible      n   BASELINE   +MARKED    mean conf, base -> marked
    0.00-0.30          221      4.1%      6.8%          0.88 -> 0.53
    0.30-0.35          149      0.0%      5.4%          0.88 -> 0.49
    0.35-0.40           82      2.4%      4.9%          0.84 -> 0.56
    0.40-0.45          132     11.4%      7.6%          0.85 -> 0.66
    0.45-0.50          132     19.7%     23.5%          0.92 -> 0.68
    0.50-0.55           79     27.8%     35.4%          0.90 -> 0.74
    0.55-0.65          210     56.7%     55.7%          0.95 -> 0.84
    0.65-1.01          231     68.0%     64.9%          0.97 -> 0.90
                ALL   1236     28.3%     29.4%

**Chance is 11.2%.** Below 0.40 visible BOTH models sit at or under it -- the
marked one moved 2.4% to 4.9% and is still worse than guessing the commonest
class. Training on 2,765 mark-labelled buried crops did not make a buried tsum
readable, and the reason is not fixable by data: a crop centred on a tsum that
is 30% showing is mostly whatever is lying on top of it. **The pictures are the
limit, and that is now measured rather than suspected.**

### What the harvest DID fix, and it is worth keeping

**Calibration.** The old model is *confidently wrong* down there -- 0.88 mean
softmax at 4% accuracy -- and `character_min_visible` exists only because of
it: "it stays CONFIDENT down there, at unchanged confidence the whole way
down". The marked model does not:

    reject floor   BASELINE agrees / covers    +MARKED agrees / covers
            0.00      28.3% / 100.0%              29.4% / 100.0%
            0.80      32.7% /  83.6%              48.6% /  47.2%
            0.95      35.3% /  76.1%              57.7% /  34.6%

At a 0.95 floor it is **57.7% right against 35.3%**. It abstains instead of
guessing, which is the behaviour the visibility floor was invented to fake.

### It still does not help the round

    naming, baseline model   recall -5.2pp   better 4, worse 150
    naming, marked model     recall -5.1pp   better 2, worse 136
    embedding grouping       RUN 1.17 -> 1.05
    nearest of the round's 5 RUN 1.24 -> 1.06

The head is named on 11% of drags rather than 14% -- the calibrated model
abstains more -- so `replace` splits just as many partners away. Better honesty
about an unreadable picture is not a readable picture.

### Where the line closes

Six ways of using identity have now been measured: naming, a per-round
vocabulary, forced group counts, embedding clustering, remembered prototypes,
and training on the buried population itself. Every one is limited in the same
place by the same thing, and the last one proves the cause rather than
inferring it. `character: ""` stands, and no further labelling changes it.

Nothing was promoted. `models/character.onnx` is untouched; the candidate sits
at `models/character_marked.onnx` with its `.pt` beside it.

**792 tests pass. Four fail on purpose:** the ships-off guards, because
`character:` and `reject_model:` are armed in `flows/play.yaml`.
