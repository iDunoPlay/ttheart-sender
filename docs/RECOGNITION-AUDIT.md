# Recognition audit — 2026-09-06

*Requested by `NEXT_TASK_TSUM_FACE_RECOGNITION.md`, which asked for an audit
before any change. This is the audit. No gameplay code was touched.*

---

## 0. The headline: the task's premise does not match the repository

The instruction opens with

> The current training module is built around **one manually selected base
> Tsum, currently Beast**.

**That is not what this repository does, and it has not been true since round
31.** `scripts/classify.py` is a multi-class classifier. `models/character.json`
ships **38 classes**. `crops/labelled/` holds **5,642 hand-labelled crops** in
**49 class folders**.

Beast appears in this project for a different reason: it is the **equipped tsum
the player plays with**, recorded in `docs/SCORECARD.md` so that rounds are
comparable. It is not a training assumption anywhere. The three code/doc hits
for "Beast" are a scorecard note and two comments using it as an example.

Beast is, if anything, the *least* represented character: **25 crops of 5,642
(0.4%)**, noted as a gap in `docs/IDENTITY.md` §8 a day before this task was
written.

So the requested work — "convert Beast-only training into multi-class
recognition" — is already done. What follows is the audit of the thing that
actually exists, and it reaches a different and more useful conclusion.

---

## 1. The fifteen questions, answered from the code

| # | question | answer |
|---|---|---|
| 1 | Where is the Beast assumption? | **Nowhere in code.** `docs/SCORECARD.md:23` records it as the equipped tsum; `scripts/group_eval.py:34-36` uses it in a worked example. |
| 2 | How are crops generated? | `scripts/crops.py extract`, from the recorded corpus or live. Half-width **1.0r** (not 1.5r — a wider window is mostly the neighbour), saved at 64px. |
| 3 | Where are crops stored? | `crops/all/` unlabelled, `crops/labelled/<Class>/*.png` labelled, plus `crops/marked/`, `crops/rejected/`, `crops/sheets/`. |
| 4 | How are labels represented? | **The folder name is the label.** Filename `<session>_<sample>_<index>_v<visible>.png` carries the session and the visibility, and is a primary key back into `samples.jsonl`. |
| 5 | Binary, one-vs-background, or multi-class? | **Multi-class, 38 classes.** Plus an explicit UNKNOWN via a softmax floor. |
| 6 | Architecture? | `torchvision` **MobileNetV3-Small**, ImageNet weights, last layer replaced (`--backbone resnet18` also available). |
| 7 | Preprocessing? | Resize 64→96 `INTER_LINEAR`, `/255`, ImageNet mean/std, NCHW, contiguous. Train-time augmentation: flip, ±0.35 rad rotation, ±10% scale, ±20% brightness. |
| 8 | How are samples selected? | Class-weighted loss for imbalance; `--min-class 20` drops rare classes; **`crops.py` only ever wrote crops at `v >= 0.55`** (see §3). |
| 9 | Data for identities other than Beast? | **Yes — 48 other folders.** CheshireCat 566, Pascal 419, Monstro 321, Duchess 290… |
| 10 | How is it evaluated? | Held out **by session, never by crop** — boards inside a round are near-duplicates. Confusion matrix + per-class recall/precision, both accuracies printed. Split is chronological by default. |
| 11 | Live inference? | `CharacterModel` in `ttheart_sender/game/tsum.py:1088`, ONNX through `cv2.dnn`, capped batch of 64, warmed up before the first board. |
| 12 | Confidence? | Softmax max. `--character-confidence` default **0.85**. A crop below it keeps its k-means `kind`. |
| 13 | Where does identity reach the board? | `CharacterModel.apply` rewrites `Tsum.kind` to `CHARACTER_KIND + class_id`, called at `tsum.py:3677`. |
| 14 | Where are they grouped? | `adjacency()` links tsums sharing a `kind`; `find_chains()` walks it. Identity is therefore a **relabelling of the existing grouping key**, not a new path. |
| 15 | Tests? | `tests/test_character.py` (10) and `tests/test_character_visibility.py` (10). |

---

## 2. The dataset, measured

**5,642 crops, 49 folders, 371 sessions.** 38 of those folders are classes the
shipped model has an output for; the rest are below `--min-class 20`.

    CheshireCat 566   Marie       161   Genie        84   Stitch      31
    Pascal      419   Sulley      157   Daisy        72   Lucifer     30
    Monstro     321   Piglet      157   Donald       67   Beast       25
    Duchess     290   Marshmallow 148   Alien        64   Streetwear  23
    Randall     236   WhiteRabbit 142   Wade         51   Cheesethe   23
    Dumbo       233   Baymax      142   CaveOfWond   51   Elliot      21
    LittleOyst  229   Sebastian   141   22           51   Goofy       20
    Rex         188   Grim        118   Maximus      50   ...then <20:
    unknown_lb  167   Patch       112   Cleo         41   Tramp 9, Oliver 9,
    (board 705, coin 92, score 10 are not characters)      MissBianca 6,
                                                           LightningMcQueen 6,
    Flounder 37, Flik 36                                   Olaf 4, Mike 4,
                                                           Sisu 3, Dante 1,
                                                           Sadness 1, Lotso 1,
                                                           TheDarkMirror 1

Train/validation/test as the task asks for it does exist, but as **two** splits
rather than three: sessions are split train/held-out and the held-out session
list is **baked into `models/character.json`** (212 train / 92 held out). That
list is what makes the test set fixed, and `scripts/recog_eval.py` now reads it
rather than re-splitting. There is no separate validation set; the best epoch is
chosen on the held-out set, which mildly optimistically biases the number below.

---

## 3. The limitation that governs everything

**Every labelled crop is at least 0.55 visible. Measured, not assumed:**

    5,642 of 5,642 crops.  min 0.55,  p5 0.55,  median 0.64,  max 1.00.
    share below 0.55:  0.0%

The median tsum on a real board shows **0.41** of itself (95,302 detections,
`docs/IDENTITY.md` §1). So the training set is the **least occluded fifth of a
board**, and every accuracy figure in this document describes that fifth.

`docs/IDENTITY.md` §4 already measured the rest, against the game's own marks,
on sessions no labelled crop came from: above 0.55 visible the model agrees with
the character the game confirms **74%** of the time; below it, **0–28%**, against
a 12.5% chance rate — **at an unchanged confidence of 0.75–0.88 all the way
down.** Confidence does not fall where competence does, which is why
`CHARACTER_MIN_VISIBLE = 0.55` had to be a separate rule and why no threshold
can substitute for it.

§5 of that document then tested the obvious fix — labelling buried crops using
the game's marks, 614 of them — and found **no effect**. The picture is not
there to read.

---

## 4. Evaluation, on the fixed held-out sessions

New tool, `scripts/recog_eval.py`. It scores the **shipped artifact**
(`models/character.onnx` through `cv2.dnn` — what the bot loads) on the **92
sessions recorded as held out inside `models/character.json`**. It trains
nothing, so it is repeatable and the test set cannot drift.

    1,540 crops, 92 held-out sessions, visibility 0.55-1.00 median 0.64

    -- over the 37 classes the model was TAUGHT --
      top-1 accuracy      97.3%    on 1,499 crops
      top-3 accuracy      98.5%
      macro F1            94.8%    over 35 scored classes
      worst class F1      66.7%    (CaveOfWonders, n=2)

    -- including classes with NO training crop --
      top-1 accuracy      94.7%    on all 1,540 crops
      top-3 accuracy      95.9%
      macro F1            90.6%
      scored at 0% by arithmetic, never shown one example:  Cleo
      no held-out crop at all, not averaged in:  Flounder, Wade

    latency  0.25-0.31 ms/crop   =  12-14 ms for a 46-tsum board

    most confused (true -> predicted):
      Cleo -> Beast          25
      Cleo -> Flounder       16
      WhiteRabbit -> LittleOyster  10
      Alien -> BeastIdle      3
      22 -> Randall           2

    confidence   median 1.000   correct 1.000   wrong 0.796

      floor   coverage   accuracy
       0.00     100.0%      94.7%
       0.70      97.4%      96.7%
       0.85      96.0%      97.6%   <- shipped
       0.95      94.0%      98.5%

**Both headlines are printed and neither may be quoted alone.** The gap between
them is a hole in the label set, not a defect in the model, and the first run of
this tool proved why that distinction has to be built in rather than remembered:

* **Cleo has 41 held-out crops and 0 training crops**, all from one session. It
  scores 0% at something it was never shown once.
* Its crops then land on Beast — **25 of the 31 crops called Beast are Cleo** —
  which dropped Beast's precision to 16.1% and made **Beast** print as the worst
  class the model was taught. It is not: Beast's recall is 83.3%. Recomputing
  the taught headline on the taught subset moves the worst class from a
  fictitious Beast 27.0% to a real CaveOfWonders 66.7% at n=2.

`tests/test_recog_eval.py` pins both halves. **592 tests pass** (586 + 6).

---

## 5. Answering the task's decision tree

The task asks: reliable, or unreliable?

**Both, and the split is not about the model.**

* On the picture it was trained for — a tsum showing more than 55% of itself —
  recognition is **reliable**: 97.3% top-1, 94.8% macro F1, 12–14 ms a board.
  There is no case for rebuilding it.
* On a **real board**, only ~22% of detections clear that bar. The other 78% are
  never asked, keep their k-means `kind`, and are grouped by colour exactly as
  they were before.

So the honest answer to *"can the bot correctly identify the tsum faces that
matter to gameplay?"* is: **it can identify the ones it can see, and it cannot
see most of them.** That is a detection problem, and §9 of `docs/IDENTITY.md`
lists the four things that would move it — an adaptive crop window, occlusion
masking, cross-frame tracking, a true-radius detector. None is a labelling job.

### The part the task's chain assumes, which was already measured

The task's premise is `correct recognition → correct grouping → better chains`.
That link **has been played, and it went the wrong way.** `scripts/group_eval.py`
over 407 held-out boards:

    method   agreement    lift   ids/board   cleared
    kmeans       39.9%   1.47x      7.29       3.32
    named        31.5%   1.56x      8.59       2.96   -10.9%

The classifier **knows more per group** (lift 1.56 vs 1.47) and still plays
worse, because naming only the visible members splits one character into two
groups — a character id for the crops it could read, a colour id for the rest —
and **`ids/board` rises to 8.59 against the game's own limit of 5 characters.**

**A chain needs its group whole more than it needs the group correctly named.**
Better recognition on a fifth of the board makes that fragmentation worse, not
better. This is why `flows/play.yaml` carries `character: ""`.

---

## 6. What is actually worth doing, cheapest first

1. **Train Cleo** — 41 crops already labelled, sitting in one session, currently
   costing 2.6 points of top-1 and poisoning Beast's precision. One retraining
   run. Free.
2. **Give Cleo, and the 8 crops in `LightningMcQueen`, `Mike`, `MissBianca`,
   `Sadness`, `Sisu`, a second session** so they can be both trained and scored.
   9 classes still come from a single round.
3. **Look at `WhiteRabbit → LittleOyster` (10)** — the only confusion between two
   well-taught classes, and the only one on this list that is a model error.
4. **Do not label more buried crops.** §5 of `docs/IDENTITY.md` tested it; no
   effect.

None of these changes gameplay, and none of them is worth playing rounds to
test, because §5 above shows the gameplay path is blocked by fragmentation
rather than by accuracy.

---

## 7. What changed in this audit

| file | change |
|---|---|
| `scripts/recog_eval.py` | **new** — scores the shipped ONNX on the fixed held-out sessions; macro F1, top-3, per-class, confusion, confidence, reject curve, latency |
| `tests/test_recog_eval.py` | **new** — 6 tests, pinning the untaught-class arithmetic |

No model was retrained. No gameplay file was touched. `character: ""` is
unchanged and the character model still ships **off**.

---

*Round-by-round record in [DATASET-FINDINGS.md](DATASET-FINDINGS.md); the
identity line of work is closed out in [IDENTITY.md](IDENTITY.md).*
