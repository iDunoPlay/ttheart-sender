# How to retrain, and how that changes recognition

*A runbook. Every command here is offline — nothing below changes how a round
plays until the last section, which is a separate decision.*

---

## The short version — one command

```
.venv/Scripts/python scripts/label_board.py            # label
.venv/Scripts/python scripts/retrain.py --survey-only  # what will train, what will not
.venv/Scripts/python scripts/retrain.py                # train, score, promote or refuse
```

`retrain.py` runs the four steps below in order and stops at the first failure.
`--dry-run` does everything except the promotion. `--reject` retrains the
"is this a tsum at all" model instead.

**None of that changes gameplay.** Only §6 does, and it is off today for a
measured reason.

### Always run `--survey-only` first

    labels in crops\labelled
      54 class(es), 5732 crop(s)
      training 38 class(es) at --min-class 20

      DROPPED -- under --min-class 20, so nothing you labelled for
      these will be learned. How many more crops each needs:
        Tramp               19 crops   (+1)
        Dory                 9 crops   (+11)
        Maleficent           6 crops   (+14)
        Eeyore               5 crops   (+15)

**This is the thing that quietly wastes a labelling session.** `--min-class`
defaults to 20 and drops everything under it, so a evening spent adding Dory,
Maleficent and Eeyore trains none of them.

Two ways forward, and they are a real trade:

* **label more of them** — 20 crops is roughly where a class becomes learnable
  *and* scoreable;
* **lower the cut**: `--min-class 5`. They will be learned. They cannot be
  scored reliably, so their rows in every report afterwards are provisional —
  and the headline drops, because a thin class scores badly and macro F1 weights
  it the same as a 500-crop one. Measured on the current labels: **top-1 96.4%
  and macro F1 94.2% at `--min-class 20`, against 95.4% / 88.3% at 5.**

That fall is not the model getting worse. It is the report including classes
that were previously invisible.

---

## The four steps, if you want to run them yourself

`retrain.py` runs exactly these and nothing more, so any of them can still be
run alone.

## 1. Label

    .venv/Scripts/python scripts/label_board.py
    .venv/Scripts/python scripts/label_board.py --only-wrong
    .venv/Scripts/python scripts/label_board.py --crop-profile padded_v1

Everything you press writes straight into `crops/labelled/<Name>/`. A name the
tool has never seen creates the folder there and then — `Dory`, `Eeyore` and
`Maleficent` all arrived that way. Nothing else needs telling.

**Check what you actually wrote** before training on it:

    ls crops/labelled                       # new classes appear as folders
    find crops/labelled -name '*.png' -newermt '-1 day' | wc -l

Two things worth knowing about your own crops:

* An index of **90 or more** means you placed that circle by hand, and its
  `_v<visible>` is the radius you left, not a measurement. The default is the
  full board radius, so it records `v1.00`.
* Crops below **0.55** visible are new territory. Every one of the 5,642 crops
  before this tool existed was at or above that floor.

### 10,747 boards is not the job. 159 crops is.

The corpus has 10,747 boards and nobody is going to look at them. That is fine,
because the boards are not what is short. Counted:

    character classes labelled          56
    crops already labelled           5,407
    crops needed to make EVERY existing class trainable   159

The big classes are saturated -- CheshireCat 553, Pascal 421, Monstro 330 --
and another Pascal crop teaches the model nothing. What is missing is a handful
each of the thin ones:

    Gaston 1 (+19)   TheDarkMirror 2 (+18)   Olaf 4 (+16)   Hades 5 (+15)
    MURandy 5 (+15)  Eeyore 8 (+12)          Dory 9 (+11)   Oliver 9 (+11)
    Rafiki 9 (+11)   Sisu 9 (+11)            Dante 11 (+9)  Maleficent 12 (+8)

### Finding those crops without scrolling

    scripts/label_board.py --needs Dory        # boards likely to hold one
    scripts/label_board.py --unsure            # boards it cannot name at all

`--needs` ranks boards by how strongly the model thinks that class is present.
`--unsure` ranks by how much of a board it cannot name, which is where the
classes it does NOT know live.

**A class the model has never trained on cannot be hunted by name** -- you need
crops to train a class and a trained class to find crops. Break it with a
smaller cut:

    scripts/retrain.py --min-class 5 --dry-run
    scripts/label_board.py --needs Dory --model models/character_candidate.onnx

A class with 9 crops trains badly and still points at the right boards, which is
all this needs.

### Do I have to label every board first?

**No.** Retraining works at any time, with whatever is labelled. There is no
threshold to reach and nothing is wasted by stopping.

    boards in the corpus              10,747
    boards with at least one label     1,399
    character crops                     5,137
    negatives (board + score + junk)      881

You are training on the crops, not on the boards. A board with one crop labelled
contributes that crop.

### The circles come back, and no amount of labelling changes that

This is the thing worth being clear about. **`detect()` draws the circles, and
it contains no model.** It is colour clustering plus a distance transform, it
runs before anything is trained, and retraining cannot alter it. Open the same
board tomorrow and every circle will be in the same place.

What labelling buys is downstream of that:

| what you label | which model | what it does |
|---|---|---|
| a character name | `character.onnx` | says WHICH tsum a detection is |
| `j` junk | `reject.onnx` | says a detection is NOT a tsum, so the play loop can drop it |

So junk labels never remove a circle from the labelling tool. They teach the
reject model to drop that kind of detection **during a round**, and only once
`reject_model:` is set in `flows/play.yaml`.

In the tool itself, a crop you have marked comes back **red and reading `junk`**
the next time you open that board, so you can see it is dealt with. Press `h` to
shrink every marked circle to a dot and get them out of the way.

### Opening one particular board

    scripts/label_board.py --board 19                  # the panel's "board 19/10747"
    scripts/label_board.py --session 20260905_020856_8872 --sample 6

`--board` is an index into the shuffled order, which only holds while `--seed`
does. `--session`/`--sample` names one frame for good, and is what to write down
if you want to come back to it.

### Typing a name quickly

Press `/`, type a letter or two, and the panel's list narrows to the classes
that match. Then:

* **TAB** fills in the top match;
* **Enter** commits.

Digits cannot pick from that list, and the reason is `22` — it is a real class
name, so a number key while typing has to mean the character 2. TAB is the only
key left that can mean "take the match".

Enter also fixes case on its own: typing `marshmallow` commits as
`Marshmallow` rather than creating a second folder.

### Re-labelling

Two tools, and which one you need depends on whether the board still exists.

**The board is still in `dataset/`** — label it in place, in context:

    .venv/Scripts/python scripts/label_board.py
    .venv/Scripts/python scripts/label_board.py --only-wrong

**The board is gone** — only the crop survives, so judge the picture:

    .venv/Scripts/python scripts/crops.py label --dir crops/duplicates/22

Click the crops that are one character, press `s`, type the name. They move
into `crops/labelled/<name>/`. Crops nobody claims stay where they are, which
is a perfectly good answer for a fragment.

This matters more than it sounds: **38 of the 39 crops `dedupe.py` quarantined
came from sessions no longer in `dataset/`**, and `22` has no crops at all whose
board still exists. For those, `label_board.py` cannot help — there is no frame
left to open.

### Fixing a label: just move the file

Moving a crop from `crops/labelled/BeastIdle/` to `crops/labelled/Beast/`
**works and is the supported way to fix a label.** Nothing else records a
crop's class: every tool derives it from the folder it walks, and the sidecars
(`_manual.jsonl`, `_junk.jsonl`) key on `(session, sample, index)`. The label
tool does the same thing internally.

**Move it. Do not copy it.** A crop in two folders trains as two characters at
once and nothing downstream can tell which was meant. `retrain.py --survey-only`
now checks for this and names the offenders.

### Can I just drag files into `crops/labelled/` myself?

**Yes, mostly.** A folder is a class and a `.png` in it is a crop — that is the
whole contract, and `crops.py assign` has always worked exactly that way.

What has to be true:

| | |
|---|---|
| the folder name | becomes the class name, verbatim |
| the file | must be **`.png`** — a `.jpg` is skipped in silence |
| the picture | must be a square crop of one tsum, cut at **one radius** about its centre. Anything else is a different distribution |
| the filename | should be `<session>_<sample>_<index>_v<visible>.png` |

**The filename is the part that bites.** It is a primary key back into
`samples.jsonl`, and every tool parses it to find out which SESSION a crop
belongs to. A name that does not parse does not fail — the crop is filed under
session `"unknown"`, and then:

* every unparsed crop lands on **one side of the train/test split together**;
* the **golden set cannot exclude it**, because that is a list of sessions;
* `--min-visible` cannot filter it, because there is no `v` to read.

So the crop still trains, and the held-out number quietly means less. If you are
adding pictures from outside the corpus, put them in a folder and know that they
behave as one session.

To check nothing was orphaned:

    .venv/Scripts/python -m pytest tests/test_crop_filenames.py -q

---

## 2. Train a candidate

    .venv/Scripts/python scripts/classify.py ^
        --epochs 30 --split random --min-class 20 ^
        --crop-profile plain ^
        --onnx models/character_candidate.onnx

**Always export to `_candidate`, never over `models/character.onnx`.** The
promotion step needs both to exist so it can compare them.

What the flags are for:

| flag | why |
|---|---|
| `--min-class 20` | the cut. **Anything under it is dropped**, which is what left Dory, Maleficent and Eeyore untrained. `retrain.py --survey-only` says who is about to be dropped and by how much |
| `--split random` | a chronological split strands whole characters on one side |
| `--crop-profile` | recorded in the model's `.json`; the runtime reads it back and cuts crops the same way |
| `--golden` | defaults to `models/golden.json`; those sessions are withheld automatically |

Read these two lines in the output:

    golden set: 78 session(s) withheld from training ...
    moved 1 session(s) into training so no class is left with none: Cleo <- ...

The first says the frozen test set was respected. The second says a class that
appeared in only one session was rescued into training — without it, that class
scores 0% at something it was never shown, and drags down whatever its crops
land on.

Training takes about 50 seconds on the RTX 3070.

---

## 3. Score it on the frozen set

    .venv/Scripts/python scripts/recog_eval.py ^
        --model models/character_candidate.onnx --golden models/golden.json

**Never quote the number `classify.py` prints at the end.** That is the model's
own held-out split, and two models split differently are scored on different
boards. This project once read **97.9% against 94.9%** for two models that were,
on identical crops, exactly as good.

The golden set is 78 sessions frozen once and never trained on, so every
candidate is asked the same questions. What to read:

* **top-1 and macro F1** — macro F1 weights a 20-crop class the same as a
  566-crop one, so it is the one that notices a rare character breaking.
* **the per-class table** — sorted worst first.
* **the confidence bands** — if accuracy rises with confidence, a threshold is
  worth having. It currently does: 47% at 0.5–0.7 against 99.6% above 0.99.
* **most confused pairs** — the shortest route to your next labelling session.

To see what changed on real boards rather than on crops:

    .venv/Scripts/python scripts/board_check.py --samples 300 --with-truth

---

## 4. Promote

    .venv/Scripts/python scripts/promote.py models/character_candidate.onnx --dry-run
    .venv/Scripts/python scripts/promote.py models/character_candidate.onnx

The dry run does every check and changes nothing. Without `--dry-run` it copies
the candidate into service, after backing the current one up to
`models/character.prev.onnx`.

It **refuses** three things:

| refusal | why it matters |
|---|---|
| a crop profile this build does not have | a model trained on padded crops and served unpadded ones does not error, it just reads as a worse model |
| the candidate trained on golden sessions | then its score there is memory, and every comparison is meaningless |
| worse than the model in service, on golden | the only comparison made on identical boards |

`--force` promotes anyway and prints which check it walked past, so the reason
is on the record.

**To undo:**

    copy models\character.prev.onnx models\character.onnx
    copy models\character.prev.json models\character.json

### One thing to expect the first time

`models/character.onnx` as shipped **predates the golden set and trained on 45
of its 78 sessions**, so it cannot be scored there and the comparison prints
`unmeasured`. That is not a bug — it is why the set was frozen. The first
promotion is the one that establishes a comparable baseline; every one after it
gets a real before-and-after.

---

## 5. The reject model — same shape, different question

`reject_net.py` answers *is this a tsum at all*, which is a different model from
*which character is this*. Your `j` presses feed it.

    .venv/Scripts/python scripts/reject_net.py --epochs 20 --onnx models/reject_candidate.onnx
    .venv/Scripts/python scripts/promote.py models/reject_candidate.onnx --to models/reject.onnx

Read this line in its output:

    37 crop(s) in the NEGATIVE folders were marked or cleared by the game
       and are counted as tsums, not junk

That is the safety net. A crop the game marked or cleared is a tsum whatever
folder it is in, so it is handed back rather than trained against. If you mark a
class junk that the game actually plays with — coins are marked 76.2% of the
time — those crops come straight back out and the count says so.

---

## 6. Turning recognition on in gameplay — a separate decision

Everything above changes an artifact on disk. **None of it changes a round.**
`flows/play.yaml` carries:

    character: ""

To use the model, set it to `models/character.onnx`. Before you do, the
measured reason it is off:

Naming only the tsums the model can read **splits one character into two
groups** — a character id for its visible members, a colour id for its buried
ones — and puts 8.37 ids on a board where the game only ever uses 5. Scored over
378 paired drags against the game's own answers, that is **−9.6% cleared**.
Forcing the model's output into 5 groups is worse: **−23.3%**, with a knowledge
score of 1.00x, meaning it knows nothing beyond the sizes of the groups it made.

So a better classifier does not become better gameplay until it can name most of
a board, and today it names about 19% of detections (25% under `padded_v1`).
That is the gap to close, and it is a coverage problem rather than an accuracy
one — recognition is already 96–97% on what it can see.

If you do turn it on: change one line, play rounds with it alternating on and
off, and read the result with `scripts/ab_eval.py`. One line back to `""` is the
revert.

---

## What each command actually touches

| command | writes | changes a round? |
|---|---|---|
| `label_board.py` | `crops/labelled/**`, `_manual.jsonl`, `_junk.jsonl` | no |
| `classify.py` | `models/<name>_candidate.onnx` + `.json` + `.pt` | no |
| `recog_eval.py` | nothing | no |
| `board_check.py` | nothing, unless `--overlay` | no |
| `golden.py freeze` | `models/golden.json` | no |
| `promote.py` | `models/character.onnx`, `.prev.*` backup | **only if `character:` is set** |
| editing `flows/play.yaml` | that file | **yes** |

---

*Companions: [LABELLING-TOOLS.md](LABELLING-TOOLS.md) for the tools themselves,
[RECOGNITION-AUDIT.md](RECOGNITION-AUDIT.md) for where the numbers came from,
[IDENTITY.md](IDENTITY.md) for why naming characters has not paid off yet.*
