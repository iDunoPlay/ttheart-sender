# Update — 2026-09-07 (recognition: six methods, one wall)

Standalone briefing. Fuller engineering log in `updates.md`.

---

## What this project is

1. A bot that plays Tsum Tsum in an Android emulator on Windows.
2. A round is ~75 seconds, ~95 chains dragged. Score is driven mostly by
   reaching **FEVER**.
3. **Detection contains no model.** The board is found by k-means colour
   quantisation plus a distance transform, and tsums are grouped by `kind`.
   `adjacency()` and `find_chains()` chain tsums that share one.
4. The **character model** is a separate, optional layer that renames `kind`
   using a trained classifier. It is what this update is about.

---

## The question

The player, after weeks of hand-labelling crops:

> "It merges the wrong tsums. This is why recognition must be in place. I want
> the app to know only 5 kinds of tsum are in the game and recognise all of
> them correctly, not guessing, so it won't link the wrong tsum. Each tsum I
> labelled has its face pattern with colour — just train the app to remember
> those."

Every part of that diagnosis turned out to be **correct**. The conclusion still
went the other way, and the reason is worth having in one place.

---

## The root problem, measured

Over **5,528 stored boards**, per-frame k-means produces:

```
distinct colour clusters on ONE board:
  7 ×1366    8 ×1342    6 ×927    9 ×831    5 ×484
```

**7–8 groups on a board that holds 5 characters.** The average character is
split across about one and a half groups *before any model runs* — and
`find_chains` groups by `kind`, so half of every character is already
unreachable from the other half.

The same thing seen from the other side: on **3,000 Beast-equipped boards**,
where Beast is guaranteed present and plentiful, the cluster `read_base_kind`
picks is **empty on 15%**, holds a **median 6 tsums (13% of the board)**, and is
the **biggest group on only 10%**.

That is the player's "merges the wrong tsums", in the detector's own terms.

---

## Why Beast is never named — it is not the model

Reported: *"Beast doesn't appear in the list until the Beast skill activates."*

On the frozen golden set **Beast reads 99.2% F1** (100% precision, 98.4%
recall over 62 held-out crops). The model knows Beast.

It is the base cluster above. Beast's tsums are split across two or more
clusters, and the model only names tsums above the visibility floor. When the
skill fires the board is redrawn, those tsums clear the floor, and the name
appears — exactly as observed.

---

## Six ways of using identity. All measured. All fail in the same place.

Everything below is scored against **the game's own marks** — holding a tsum
makes the game light up every tsum that is the same character *and* reachable,
which is free ground truth recorded on every sampled press.

### 1. Naming the board (`naming_gain.py`) — 800 drags

```
                     recall   precision   partners offered
  colour clusters     43.3%       28.6%    11.04
  + character model   38.2%       30.0%     9.08
  change              -5.2pp      +1.4pp    -1.96
  paired: better on 4, worse on 150         p = 0.0000
```

**The mechanism.** The chain head is named on **14%** of drags. On the other
86% the head keeps its colour cluster while some partners get renamed — and the
shipped rule *replaces* `kind`, so a renamed partner **leaves the head's group**.
Naming one member of a pair splits the pair.

`--rule merge` (only ever join, never split) removes the harm and adds nothing:
+0.4pp, better on 3 of 300, worse on 0, p = 0.25.

### 2. The five-character rule (`vocab_probe.py`)

The player's rule, implemented: accumulate votes over a round, keep the top N
names, hand the rest back to their colour cluster.

```
vocabulary        refused   recall    change   better / worse
none (ships)           0%    37.8%         -      -
top 5 (the rule)       1%    37.9%    +0.1pp   1 / 0
top 3                  5%    38.6%    +0.8pp  10 / 0
top 2                 14%    39.3%    +1.5pp  21 / 0
top 1                 44%    41.1%    +3.2pp  39 / 0
NO NAMING AT ALL        —    43.3%    +5.5pp
```

**Never worse on a single drag at any setting**, and monotone. Every name the
model is allowed to use costs recall; the limit of the curve is naming nothing.
At 5 it barely bites, because a round only spends a median of **4** names
anyway — the tail of ones and twos that looks wrong in the panel is a small
share of the naming, and the damage comes from the confident majority.

### 3. Forcing the group count (`group_probe.py`)

Scored on the chain the round could actually drag, and reported as the **run of
correct members before the first wrong one** — because the game accepts 97.9%
of first members, 19.1% of sevenths, and once it refuses one it refuses **86% of
what follows**.

```
groups   made   recall   precision   ACCEPTED RUN
k-means   7.3    28.2%     52.2%        1.43
3         3.0    33.8%     43.3%        1.16   -19%
4         4.0    29.8%     50.0%        1.28   -10%
5         5.0    27.3%     54.6%        1.34    -6%
6         6.0    25.8%     60.3%        1.41    -2%
7         7.0    23.7%     61.4%        1.36    -5%
```

Adaptive k-means at 7.3 groups beats a **forced** 7 (1.43 vs 1.36). It is not
the number that helps — it is fitting the number to the board. Forcing five
piles of face colour does not recover the five characters; it merges the wrong
tsums a different way.

### 4 & 5. The learned face, and remembered prototypes (`embed_probe.py`)

Not classification — grouping in the space the classifier **learned**.
MobileNetV3's penultimate layer is a 1024-d description of a tsum face, which
is literally "the face pattern with colour". Plus prototypes: one mean face per
labelled character, every tsum assigned to the nearest, no confidence floor.

```
feature                       recall   precision   offered    RUN
median face colour (ships)     28.7%     49.1%      4.45     1.37
median face colour, 5 piles    26.8%     53.5%      4.13     1.24  -10%
learned face, 5 piles          20.9%     69.0%      2.30     1.17  -15%
nearest remembered face        20.0%     71.9%      2.09     1.08  -21%
nearest of the round's 5       23.5%     68.7%      2.49     1.24  -10%
```

**The diagnosis is confirmed.** Colour groups two tsums correctly **49%** of the
time; the learned face gets **69–72%**. And the five-character rule helps here
too — restricting to the round's cast beats offering all 65, **+15%**.

**Both halves of the proposal do exactly what was expected. It still loses**,
because recall falls from 28.7% to 20–23%:

> Recognition does not merge the wrong tsums. It **splits the right ones**.

A buried Beast and a well-seen Beast do not land in the same pile.

### 6. Training on the buried population itself (`mark_probe.py harvest`)

The last lever, and the only one that needed no more labelling. The game's
marks are free labels for **buried** crops — the population every method above
fails on, and the one hand-labelling can never reach, because a person labels
what they can see.

```
crops/marked    2765 crops   median 0.44 visible   71% BELOW the floor
crops/labelled  8005 crops   median 0.72 visible    5% below
```

Harvested from half the sessions so the other half scores it; golden sessions
withheld; head groups pinned so the comparison is skill and nothing else.

```
partner visible      n   BASELINE   +MARKED    mean confidence
0.00-0.30          221      4.1%      6.8%     0.88 -> 0.53
0.30-0.35          149      0.0%      5.4%     0.88 -> 0.49
0.35-0.40           82      2.4%      4.9%     0.84 -> 0.56
0.40-0.45          132     11.4%      7.6%     0.85 -> 0.66
0.45-0.50          132     19.7%     23.5%     0.92 -> 0.68
0.50-0.55           79     27.8%     35.4%     0.90 -> 0.74
0.55-0.65          210     56.7%     55.7%     0.95 -> 0.84
0.65-1.01          231     68.0%     64.9%     0.97 -> 0.90
            ALL   1236     28.3%     29.4%
```

**Chance is 11.2%. Below 0.40 visible, both models sit at or under it.**

Training on 2,765 mark-labelled buried crops moved the worst band from 2.4% to
4.9% — still worse than guessing the commonest class. The cause is not fixable
by data: **a crop centred on a tsum that is 30% showing is mostly whatever is
lying on top of it.**

---

## What the harvest did fix, and it is real

**Calibration.** The old model is *confidently wrong* down there — 0.88 mean
softmax at 4% accuracy — and `character_min_visible` exists only because of it.
The marked model does not:

```
reject floor    BASELINE agrees / covers    +MARKED agrees / covers
        0.80       32.7% /  83.6%             48.6% /  47.2%
        0.95       35.3% /  76.1%             57.7% /  34.6%
```

At a 0.95 floor: **57.7% right against 35.3%**. It abstains instead of guessing
— the behaviour the visibility floor was invented to fake.

It still does not help a round. Naming with the calibrated model costs −5.1pp
against the old −5.2pp, and the head is named on 11% of drags rather than 14%
because it abstains more. **Better honesty about an unreadable picture is not a
readable picture.**

---

## The recommendation

```yaml
# flows/play.yaml line 104
character: ""
```

Six methods, one wall, and the last one proves the cause rather than inferring
it. **No further labelling changes this.**

---

## Bugs found on the way, all fixed

**The character model could name a detection `board`.** `classify.py` trained on
every folder under `crops/labelled` and excluded nothing, so every model ever
built carried `board` and `junk` as identities. Worse than a wasted output:
`apply` writes the winning class into `kind` and chains are built by `kind`, so
every detection named `board` was **grouped with the other board detections and
offered as a chain**. Removing them made the model better, not just cleaner —
golden top-1 **94.76% → 96.88%**, macro F1 **88.67% → 93.32%**.

**`junk` was teaching the reject model "hard to see", not "not a tsum".**

```
board  721 crops   median 0.76 visible   99% at/above 0.55
junk  1495 crops   median 0.38 visible    8% at/above 0.55
```

A junk crop, as the folder is actually used, is a detection nobody could
identify — which is what a *buried real tsum* looks like. Retraining on 1,347
new junk crops took AUC 0.9664 → 0.9114 and was **promoted silently**, because
`promote.py` printed a reject model's AUC and compared nothing. The gate now
refuses a worse one; `reject_net.py --negative-min-visible` holds negatives to
a visibility a person could have judged. Result: **AUC 0.9921, 90.7% of fakes
rejected for 1.6% of real tsums lost**, against 82.8% / 2.9% before.

**Three independent confirmations that `junk` holds real tsums:** the game
confirms 21.5% of junk crops against 4.9% for `Beast`; 359 negatives were
game-confirmed outright; and the harvest picked up `junk` and `coin` as
characters the game had marked.

**The live recognition window never opened.** `win32gui.CreateFont` does not
exist in pywin32 — the AttributeError came out of window creation and the only
symptom was a tick box that did nothing. And once fixed it opened *behind* the
emulator, because `SetWindowPos(HWND_TOPMOST)` can be silently refused. TOPMOST
is now declared at `CreateWindowEx`, and a test builds the real window.

**A bug of mine, recorded because it overstated a result.** The first
`naming_gain.py` cropped `_before.jpg` by the board rect — but that jpg *is* the
board crop. The double crop refused 82% of chain heads for falling off the edge
and reported −6.1pp instead of −5.2pp. Same conclusion, wrong reason.

---

## Also shipped

**A live recognition readout**, in a window of its own opened by
**"Show live recognition"** in the Experiments section:

```
matches the game 34/39 (87%)
Beast               12
Dory                 7
unknown_lightball    4
```

The list is checked by eye against the board; the headline is agreement with
the game's marks, and the raw count always travels with the percentage —
`100%` on two checks is not a claim. Four states that used to be
indistinguishable are now four lines: off, naming nothing, never scored, and a
real score. It shows the round's **peak** per character rather than the current
frame, because per-frame counts flickered: only a fifth of the board is ever
named and which fifth changes every frame.

---

## What is actually left

The score turns on **FEVER share**, and two switches are armed in `play.yaml`
and have **never been played**:

| switch | what is known | what to watch |
|---|---|---|
| `reject_model` | AUC 0.9921; 90.7% of fakes for 1.6% of real tsums. The *previous* one lost 143 rounds by deleting 7.7% of the board | `cleared` — `dragged` falling while `cleared` holds is the whole claim |
| `four_groups` | the only grouping rule ever measured above baseline (+3.9% simulated) | `cleared`, then dead drags |

Each is ~40 rounds per arm with **"Alternate it round by round (A/B)"** ticked,
read with `scripts/ab_eval.py`. That is where a real score improvement is still
available, and it costs play time rather than labelling time.

---

## State

- **792 tests pass.**
- **Four fail on purpose:** the ships-off guards, because `character:` and
  `reject_model:` are both armed in `flows/play.yaml`. Reverting either to `""`
  turns its guards green.
- Nothing was promoted from this work. `models/character.onnx` is untouched;
  the buried-trained candidate sits at `models/character_marked.onnx`.
- Version **1.20.0**.

## New tools, all offline, none touching gameplay

| script | asks |
|---|---|
| `naming_gain.py` | does naming change the grouping the round plays on? |
| `vocab_probe.py` | does capping a round to N names help? |
| `group_probe.py` | how many colour groups should a board be sorted into? |
| `embed_probe.py` | does the learned face group better than median colour? |
| `mark_probe.py harvest` | can the buried majority be learned at all? |
