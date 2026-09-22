# Labelling and recognition tools — 2026-09-07

*How to use them, what each one can and cannot tell you, and the plan for
wiring recognition improvements into the app.*

---

## 1. The tools, and which question each answers

| tool | question | ground truth | needs labels? |
|---|---|---|---|
| `label_board.py` | *what is this tsum?* — click and name it | you | it makes them |
| `board_check.py` | *what does the app believe about a whole board?* | labels + marks | some |
| `recog_eval.py` | *how good is the shipped model on a fixed test set?* | labels | yes |
| `recog_by_tsum.py` | *is recognition worse for one equipped tsum?* | the game's marks | no |
| `mark_probe.py` | *how far down the visibility range does it hold?* | the game's marks | no |
| `group_eval.py` | *does naming tsums group the board better?* | the game's marks | no |
| `classify.py` | trains the classifier | labels | yes |
| `reject_net.py` | trains "is this a tsum at all" | `board`/`score` folders | yes |

The loop is: **label → train → measure → find the errors → label those.**

    .venv/Scripts/python scripts/label_board.py --only-wrong
    .venv/Scripts/python scripts/classify.py --epochs 30 --split random --min-class 20 --onnx models/character_new.onnx
    .venv/Scripts/python scripts/recog_eval.py --model models/character_new.onnx
    .venv/Scripts/python scripts/board_check.py --samples 300 --with-truth

---

## 2. `label_board.py` — the click tool

    .venv/Scripts/python scripts/label_board.py
    .venv/Scripts/python scripts/label_board.py --only-wrong
    .venv/Scripts/python scripts/label_board.py --profile beans_camo_vil

### Keys

| key | does |
|---|---|
| **LEFT click** | select the nearest detection (within 14px — see below) |
| **click a NAME in the panel** | apply it to the selected tsum — no typing |
| **RIGHT click** | ADD a circle where the detector missed a tsum |
| **1**–**9** | accept a suggestion (model's top-3, then your existing classes) |
| **/** | start typing a name; TAB completes, Enter commits, Esc cancels |
| **l** | LOCK the search so it survives labelling, and free the number keys |
| **j** | mark junk — not a tsum |
| **d** | remove the label, or take back a junk verdict |
| **h** | shrink junk-marked circles to dots, to get them out of the way |
| **x** | remove an added circle (refuses to touch a real detection) |
| **-** / **=** | shrink / grow an added circle |
| **[** / **]** | zoom out / in (default 1.8x) |
| **n** / **p** | next / previous board |
| **u** | undo the last write |
| **q** | quit |

### Labelling many tsums of the same character

A board holds about forty tsums and only five characters, so the same name is
wanted over and over. Press **`l`** to LOCK the current search:

    /  m a      narrows the list to Maleficent, Marie, Marshmallow, Maximus
    /           locks it -- the panel reads  LOCKED "ma"
    click a tsum, press 3        (or click the name)
    click the next tsum, press 3
    ...
    l           clears the filter

**Why `/` and not `l`:** while you are typing, every printable key is part of
the name — typing `mike` then `l` gave you `mikel`. The lock key has to be one
no class name contains, and `/` is the key that started typing, so it reads as a
toggle. `l` clears a lock, which is safe because you are not typing then.

**Locking also gives the number keys back.** While you are TYPING, `1` has to
mean the character "1" — there is a class called `22`. A locked filter is not
text entry, so `1`–`9` pick from the list again.

The lock survives labelling; typed text does not. That is the whole difference.

### Picking a name with the mouse

The panel lists your classes and **every one of them is clickable**. Select a
tsum, click a name, done — no typing and no digit to remember.

With 69 classes the list will not all fit, so press `/` and type a letter or
two: the list narrows to what matches and stays clickable. `1`–`9` still work
for the first nine, and TAB still completes.

### To change a name that is wrong

Click the circle. The panel shows **NOW: <name>** in large text, then
**CHANGE IT TO:** with numbered options — the model's top three, then your
existing classes, with `<- now` beside the one it currently has. Press the
number, or `/` and type. The file is **moved**, not duplicated.

While you type, the list narrows to classes that match what you have typed so
far, and a name that matches an existing class in any case commits with the
folder's own spelling — typing `marshmallow` cannot create a second
`marshmallow/` beside `Marshmallow/`.

Single letters are commands, so a name starts with `/`. The first version began
typing on any a–z, which meant `j` could never mean junk.

### The legend

| colour | meaning |
|---|---|
| green | you named it |
| red | you marked it junk |
| light blue | you added this circle |
| amber | the model's guess, unconfirmed |
| grey | never asked — below the 0.55 visibility floor |
| purple | the frame edge clips its crop |

**86% of a board is grey or purple.** That is not a display fault: 77% of
detections are buried below the visibility floor and a further 9% are clipped
at the rim. Only about one tsum in eight is ever named.

### Where the edits go

* **Names** → `crops/labelled/<Name>/<session>_<sample>_<index>_v<visible>.png`,
  the exact layout and filename `crops.py assign` produces, cut by **crops.py's
  own `_cut`**, imported rather than copied. `classify.py` picks them up with no
  change anywhere.
* **Junk** → `crops/labelled/junk/`, one of the negative classes
  `reject_net.py` trains its "is this a tsum at all" model against
  (`board`, `score`, `junk`). Marking junk therefore improves the reject
  model directly. `board` keeps the 705 crops filled by hand before this
  tool existed, so the two populations stay separable.
* **Added circles** → the same folders, with an index from **90** up so it can
  never collide with a real detection, plus a line in `_manual.jsonl` so the
  circle is still there when you return to the board.
* **Junk on a rim half-circle** → a line in `_junk.jsonl`, and **no crop**.

### Why a rim half-circle records a verdict but no crop

You asked for this and it is the one place the tool deliberately does less than
you asked. `_cut` refuses a crop the frame edge would clip, and both ways of
forcing one make a model worse:

* **clip and stretch it** → `reject_net.py` learns *distorted = junk*;
* **pad the frame and cut a square** → it learns *padded = junk*, because every
  other crop in the set is unpadded.

Either is train/serve skew, which is the bug that cost this project 312 rounds
of play. And it is moot in any case: `RejectModel` reads its crops through
`_character_crop` too, so **at play time it is never asked about an edge
detection at all**. A crop written here would train it for a question nobody
poses.

So the circle turns junk-coloured, your verdict is kept, and nothing is
poisoned. The count of those verdicts is also the evidence for §4 — they are
exactly the detections that padding would make trainable.

### A new name makes a new class, immediately

Type a name the tool has never seen and it creates `crops/labelled/<Name>/` on
the spot. The first labelling session produced `Dory`, `Eeyore` and
`Maleficent` that way. Nothing else has to be told: `classify.py` reads the
folders, so a new character is a class the next time you train.

There is no undo for the folder itself, only for the crops in it — an empty
folder left behind is harmless and dropped by `--min-class`.

### Visibility on a circle you added is an ESTIMATE, not a measurement

A crop's filename ends `_v<visible>`, and every tool here bands by it. For a
real detection that number is measured. **For a circle you added it is whatever
radius you left**, and the default is the full board radius — so it records
`v1.00`.

The first session went in at **58 of 91 crops hand-added, every one claiming
v1.00**. If any of those were over a partly-buried tsum, the training set now
says "fully visible" about a picture that is not.

The panel now shows `visible 1.00 (estimate, -/=)` in the added colour while one
is selected. Use `-` and `=` to shrink the circle onto what you can actually
see. Anything with an index of **90 or more** was placed by hand, so a later
analysis can always tell an estimate from a measurement.

### Removing a circle that should not be there

There is no delete key, and that is deliberate: a detection comes from the
saved frame, so it cannot be taken out of the past. **Marking it junk is how it
gets removed from the future.**

1. **Left-click** it.
2. Press **`j`**. It turns red and reads `junk`.
3. Press **`h`** if you want the marked ones shrunk to dots and out of your way.
4. Press **`d`** on it to take the mark back.

What that actually does depends on the circle:

| the circle | pressing `j` |
|---|---|
| a normal detection | its crop moves to `crops/labelled/board/`, the class `reject_net.py` trains against |
| already named a character | the crop is **moved** out of that character and into `board/` |
| a rim half-circle | the verdict goes to `_junk.jsonl`; no crop, for the reasons above |
| one you added yourself | press **`x`** instead — it removes the circle entirely |

**Nothing changes in play until you retrain**, and `reject_model: ""` in
`flows/play.yaml` — the reject model ships off, because a 143-round A/B lost.
So marking junk builds the evidence; it does not change a round today.

### Why selection is forgiving, and adding moved to right-click

`r` in a detection is the **visible** radius, and measured over the corpus it is
under **8px for 20%** of detections and under **12px for 59%**. A rim
half-circle — exactly the thing most worth marking junk — has the smallest `r`
of all, so the tsums you most needed to reach were the ones you could not click.

Selection now takes the nearest detection within **14px** of the click (against
a median centre-to-centre spacing of ~61px, so it stays unambiguous). Adding
moved to **right-click**, because a near-miss wants opposite things from the two
actions: selecting should be forgiving, adding should be sure you did not mean
an existing circle.

### Junk is a label, and it is checked against the game

`crops/labelled/board/` is the negative class `reject_net.py` trains on, so `j`
is a training label rather than a delete key. Whether a class belongs there is
**measured**, not decided by what it looks like — every crop is checked against
whether the game ever marked or cleared that detection:

| class | marked by the game | cleared by a drag | verdict |
|---|---|---|---|
| `board` | 0.0% | 8.9% | clean negative — trained against |
| `junk` | — | — | yours; checked the same way at train time |
| `score` | 0.0% (n=6) | 0.0% | thin, but the same signature |
| **`coin`** | **76.2%** | **42.9%** | **a tsum. NOT junk** |
| `unknown_lightball` | 86.4% | 22.9% | the game's link highlight |
| a real character | 15–25% | 10–20% | for comparison |

**`coin` looks like it should be ignored and is not.** Those crops are marked by
the game more often than any real character, and nearly half of them were
cleared by a drag that worked — most likely a tsum with a coin drawn over it.
Adding it to the negatives would train the reject model to throw away
detections that clear. `unknown_lightball` was added once for exactly that
reason and had to be taken out again.

So: press `j` for bowl, board graphics and effect flashes. Do not press it for
something the game plays with.

### Retraining after you have marked junk

    .venv/Scripts/python scripts/reject_net.py --epochs 20 --onnx models/reject_candidate.onnx

Read the first lines it prints. It cross-checks your negatives against the
game's own answers and hands back anything the game marked or cleared:

    82472 detections the game itself confirmed
    37 crop(s) in the NEGATIVE folders were marked or cleared by the game
       and are counted as tsums, not junk

**That line is the safety net.** A crop the game dragged and cleared is a tsum
whatever folder it is in, so it is rescued rather than trained against. If you
mark a whole class junk that the game actually plays with — coins, say — those
crops arrive in `board/` and come straight back out, and the count tells you it
happened.

The check reaches the negative folder only because the dataset is present; with
`--dataset` pointing nowhere the folders are taken at face value. Keep the
dataset around when you retrain.

Export to `reject_candidate.onnx` rather than over `reject.onnx`, then compare
before promoting. The reject model is the one with the clearer gameplay payout:
a false detection gets chained, the stroke crosses empty board, and the chain
spends a slot on nothing.

### Two rules the tool will not break

1. **The model's guess is never written as a label.** It is shown; you press the
   key. `docs/IDENTITY.md` §9.
2. **A correction moves the file.** Leaving the old one behind would train the
   model on both answers at once.

---

## 3. What labelling can and cannot buy right now

Measured, not assumed — both models scored on the 36 sessions **neither**
trained on:

    shipped model   top-1 97.46%   macro F1 92.52%   986 crops
    retrained       top-1 97.46%   macro F1 92.67%   986 crops

**Retraining on the same crops changes nothing.** 97.5% is this data's ceiling
for the crops the model can already see. So more labels of the same kind will
not move accuracy.

What *will* move is **coverage** — how much of a board the model is allowed to
look at. Today that is 19% of detections.

---

## 3b. The offline lab, as it now stands

Three mechanisms were added so that "did this get better?" has an answer.

### The golden set — `scripts/golden.py`

    .venv/Scripts/python scripts/golden.py status
    .venv/Scripts/python scripts/golden.py freeze --share 0.25   # once

**78 sessions, 1,306 crops, 42 classes, frozen 2026-09-06** into
`models/golden.json`. Never trained on; every candidate is scored on it.

It stores session NAMES, not copies of the crops, so a relabelling reaches the
golden set too — what must not change is *which boards* are asked about.

Two guards, because both failures are silent:

* `classify.py` subtracts the golden sessions before it splits anything;
* `recog_eval.py --golden` **refuses** a model that trained on any of them,
  rather than warning.

The shipped `models/character.onnx` is refused by that second guard, and
correctly: it predates the freeze and trained on 45 of the 78. **It is not
comparable to anything, and that is the whole reason the set exists** — the
last retrain read 97.9% against 94.9% and was, on a common set, exactly as good.
The comparable baseline is now `models/character_candidate.onnx`.

Classes appearing in only one session are deliberately left out of the golden
set: freezing their only session would make them impossible to *learn*, which is
worse than leaving them unscored. Seven are in that position today.

### Crop profiles — `ttheart_sender/game/crop.py`

One implementation, imported by `scripts/crops.py` **and** the runtime.
`tests/test_crop_profile.py::test_the_trainer_and_the_runtime_cut_the_same_crop`
asserts they agree.

    plain       refuses a crop the frame edge would clip  (every model so far)
    padded_v1   extends the frame outward, replicating the border

**A model records its profile in its own `.json`, and the runtime reads it
back.** A profile this build does not have is *refused*, not defaulted — a model
trained on padded crops and served unpadded ones would simply read as a worse
model, which is the shape of the bug that cost 312 rounds.

### The metrics

`recog_eval.py` now reports accuracy **within** each confidence band, not only
the coverage curve. On the candidate, over the golden set:

    confidence        n   accuracy
     0.00-0.50       13      0.0%
     0.50-0.70       17     47.1%
     0.70-0.80        9     66.7%
     0.80-0.90       11     63.6%
     0.90-0.99       45     84.4%
     0.99-1.01     1193     99.6%

That is monotone, which means **confidence here really is a probability of being
right** and a threshold is worth having. It was not true of buried crops, where
the model stayed 0.75–0.88 confident all the way down to chance — so this had to
be measured rather than assumed.

Candidate on the golden set: **top-1 96.8%, top-3 98.9%, macro F1 95.0%**,
0.24 ms a crop.

---

## 3c. Edge padding, measured

The experiment, run offline on 200 boards with the same model, changing only the
crop profile:

    profile      asked (0.55-0.65 / 0.65-0.75 / 0.75-1.00)   named   accuracy
    plain              70%      67%      51%                  19%     99.5%
    padded_v1         100%     100%     100%                  25%     99.5%

* **Coverage 19% → 25% of detections named** — a 32% relative increase.
* **The refusal disappears**: every tsum above the visibility floor is now
  asked, in every band.
* **Accuracy is unchanged** on the 426 crops that can be checked, which is what
  `test_padding_changes_nothing_away_from_the_edge` predicts: padding only
  alters crops the edge was clipping, and every existing labelled crop is
  non-edge by construction.

Two honest limits:

1. I previously estimated ~28% coverage. It is **25%**.
2. The newly reachable crops have **no labels**, so their accuracy is not
   measured here — only that they now exist. The marks column hints they are
   fine (0.65–0.75 agreement went 33% → 78%) but n is 3 → 9.

**So padding is a coverage win, pending labels on the population it opens up.**
That is what `label_board.py --crop-profile padded_v1` is for: it lets you label
the rim tsums that could not previously be cropped at all.

---

## 4. Wiring the improvement into the app

Three changes, in the order their evidence supports, each shipping off and
reverted by one line.

### Step 1 — edge padding (the largest recognition win available)

**What it is.** `_cut` and `_character_crop` take a square of one radius about
a tsum's centre and refuse it when the frame edge would clip it. A tsum at the
board's rim sits a median of **1 pixel** from that edge. Padding means
replicating the border outward by one radius *before* cutting, so a rim tsum
yields a full square instead of nothing.

**What it buys.** 40.7% of tsums above the visibility floor are refused this
way — 52% of the clearest ones. Padding takes a board's readable share from
**~19% to ~28%**.

**How it must be wired, and this is the whole risk.** The padding has to reach
`scripts/crops.py::_cut` and `ttheart_sender/game/tsum.py::_character_crop`
**together**. If the extractor pads and the runtime does not, the model is
trained on padded rim crops and served none; if the runtime pads and the
extractor did not, it is served a picture it never trained on. That is exactly
the shape of the bug that wasted 312 rounds, so:

1. add the padding to **one** shared function, not two;
2. add a test that asserts the trainer and the runtime produce identical crops
   for the same board — the pattern `test_the_trainer_and_the_runtime_cut_the_same_colour` already sets;
3. re-extract, retrain, and compare on sessions **neither** model trained on;
4. ship it behind a flag that defaults to today's behaviour.

**Cost:** a re-extract and a retrain, both offline. No rounds.

### Step 2 — the confusions that are the model's own fault

`WhiteRabbit → LittleOyster`, 10 errors, between two well-taught classes. It is
the only confusion on the list that more data can plausibly fix rather than
being an artifact of the split. `--only-wrong` puts those boards first.

### Step 3 — the reject model, from your junk marks

Every `j` you press adds a crop to the class `reject_net.py` trains against, and
that model has a clearer payout than the character model: a false detection gets
chained, the stroke passes over empty board, and the chain spends a slot on
nothing. Retrain it once there are a few hundred new junk crops:

    .venv/Scripts/python scripts/reject_net.py --epochs 20 --onnx models/reject.onnx

### What is deliberately NOT in this plan

* **Replacing `kind` with the character name at play time.** Measured at
  **−9.6% cleared**, because naming only the readable fifth splits one character
  into a named group and a colour group. Until coverage is much higher this
  stays off.
* **Online self-training.** The bot training on its own guesses. `docs/IDENTITY.md`
  §5 tried something strictly stronger — 614 crops labelled by the game's own
  marks — and it had no effect.

---

## 5. Fixes made while building these

* `reject_net.py` had `NOT_TSUM = ("board",)`, so the 10 crops in
  `crops/labelled/score/` — the score readout, plainly not a tsum — were being
  trained as **chainable tsums**. Now `("board", "score")`.
* `classify.py` grew `--cover-classes` (on by default): no class may have zero
  training sessions. A character equipped for one run of rounds lands entirely
  on one side of a session split, and landing on the held-out side is not a hard
  test but no test at all. It moves the cheapest sessions back and prints each
  move. Cleo was the live case — 41 held-out crops, 0 training crops, 0% recall,
  and its crops fell on Beast and dragged Beast's precision to 16%.
  **Note it fixes the measurement, not the model:** on a fair comparison the
  retrained model is no more accurate.

---

*Companions: [RECOGNITION-AUDIT.md](RECOGNITION-AUDIT.md),
[IDENTITY.md](IDENTITY.md), [OCCLUSION-INVESTIGATION.md](OCCLUSION-INVESTIGATION.md).*
