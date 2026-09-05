# Identity: what the bot can know about which character a tsum is

*Rounds 31-32, 2026-09-05. Closes a line of work that ran through five
attempts. Read this before starting a sixth.*

The bot chains tsums that are the same character. Everything about how well it
plays flows from how well it can tell one character from another, so this has
been attacked repeatedly: colour clustering, a pairwise embedding, forced
regrouping, mark-harvested labels, and finally a supervised classifier trained
on 3,958 hand-labelled crops.

The classifier works. It is **97.5% accurate on held-out sessions** and it
**cannot be used**, and this document exists because those two facts are not in
tension and the reason took three separate experiments to isolate.

**Identity was never the bottleneck. Occlusion is.**

---

## 1. The question, and why the obvious answer is wrong

A Tsum Tsum board holds about 40 tsums in a pile. At most **5 characters** are
on a board at once (4 if an item is used), and the pile is dense: tsums sit on
top of each other, so most of what you can see of any given tsum is a crescent
of it poking out from under its neighbours.

The pipeline detects a tsum as a centre and a radius, and records how much of
it is showing as `r / radius`. Call that its **visibility**. Over 95,302
detections in 2,372 frames:

| percentile | visibility |
|---|---|
| p5 | 0.22 |
| p25 | 0.31 |
| **p50** | **0.41** |
| p75 | 0.53 |
| p95 | 0.74 |

**The median tsum on a board shows 41% of itself.** Only 23% of detections
show more than half.

That number is the whole document. Every method below was measured against a
population that is mostly occluded, and every one of them succeeded on the
visible quarter and failed on the rest.

---

## 2. The five attempts

| attempt | what it scored | verdict |
|---|---|---|
| per-frame k-means on face colour (`kind`) | 37% agreement with the game's marks | shipping, and it is the baseline everything else has to beat |
| forced regrouping to N colour groups (`kinds`) | +3.1% simulated cleared at N=4 | in `play.yaml`, off; the metric that scored it was later disqualified |
| pairwise embedding (`embed_net.py`) | 0.647 AUC held out | not enough to decide identity; used only to group crops for labelling |
| mark-harvested buried labels | no effect (§5) | retired |
| supervised classifier (`classify.py`) | 97.5% held out, **-10.9% cleared** (§6) | retired |

They fail in different ways and for the same reason.

---

## 3. The classifier is genuinely good

`scripts/classify.py`, MobileNetV3-Small, transfer-learned on
`crops/labelled/`. Held out **by session** -- never by crop, because boards
inside one round are near-duplicates and the same physical tsum appears in
consecutive frames, so a random split reports memory.

    3,958 crops, 44 classes, 31 above --min-class 20
    73 train sessions / 32 test sessions

    accuracy over the 29 classes that had training crops   97.5%
    accuracy including the 2 that had none                 89.8%
    coverage at reject 0.50                                99.0%

    worst confusions: CheshireCat -> Sebastian (6), 22 -> Sulley (5)

Both accuracies are printed, and that is deliberate. **Donald** appears in 2
rounds and **Flounder** in 1; a random session split can put every session of
such a class on the held-out side, and it then scores 0% recall because it was
never shown a single example. That is arithmetic, not a finding, and quoting
only the number it drags down reports the model failing at something it was
never asked to learn.

Trained in **29 seconds** on an RTX 3070.

### Against the alternatives

| method | agreement with the game's confirmed partner |
|---|---|
| per-frame k-means `kind` | 37% |
| pairwise embedding | 0.647 AUC |
| **classifier, visible tsums** | **74%** |

Naming a character from labels is not a tuning improvement over clustering its
colour. It is a different quality of answer, and the player's instinct that
"colour is not the best solution after all" was correct.

---

## 4. Where it stops working, and how we know without labels

### The free ground truth

When the bot holds a tsum, **the game lights up every tsum that is the same
character and reachable**. `marked` in `samples.jsonl` is therefore ground
truth about identity that nobody wrote down -- on sessions nobody labelled, at
every visibility, including the 78% of a board no labelled crop has ever come
from.

`scripts/mark_probe.py` uses it: name the head where the model is on solid
ground (visible, and confident), then ask what it calls each confirmed partner.
**They must agree.** Every disagreement is a real error, at a visibility read
straight off the detection.

### The result, twice

Round 31, over **94 sessions no labelled crop came from**:

    partner visible      n    agrees  mean conf
      0.00-0.30         48      10.4%      0.75
      0.30-0.35         34       0.0%      0.80
      0.35-0.40         23      13.0%      0.75
      0.40-0.45         45      20.0%      0.77
      0.45-0.50         31      12.9%      0.84
      0.50-0.55         29      27.6%      0.88
      0.55-0.65         78      74.4%      0.96
      0.65-1.00         55      65.5%      0.94

    chance rate (always the commonest head class): 12.5%

Round 32, after six new classes and 167 new labels, over 62 sessions:

    partner visible      n    agrees  mean conf
      0.00-0.30         30      10.0%      0.65
      0.30-0.35         21       4.8%      0.84
      0.40-0.45         26      30.8%      0.76
      0.55-0.65         51      74.5%      0.94
      0.65-1.00         35      62.9%      0.97

    chance rate: 19.5%

**Identical shape.** More labels and more classes did not move the low bands at
all. The cliff is a property of the picture, not of the label set.

### The trap: confidence does not fall where competence does

Look at the `mean conf` column. It sits at **0.75-0.88 all the way down**. The
model is not hedging on crops it cannot read -- it is confidently wrong.

    reject floor   agrees   on this share of partners
            0.60    41.8%                      83.1%
            0.80    48.3%                      70.6%
            0.90    52.6%                      61.5%
            0.95    54.9%                      56.3%

At a 0.90 floor it still names 61.5% of partners and is right 52.6% of the
time. **`--character-confidence` cannot catch this and no setting of it ever
could**, which is why the visibility floor had to be a separate rule.

### Why every labelled crop is above the line

`scripts/crops.py` filtered at `v >= 0.55` when it extracted the training set,
and nothing downstream knew. Every one of the 3,958 labelled crops is above it;
the median board detection is at 0.41. **The model was trained on the least
occluded fifth of the board and asked at play time about all of it.**

---

## 5. Labelling the buried crops does not fix it (tested)

The marks give a way to label buried crops for free. The game says a partner is
the same character as the head, so a partner at 0.28 visible inherits a name
that could never have been read off it. Where the head is a crop a person
already labelled -- and a crop's file name is a primary key back into
`samples.jsonl` -- the name is human and no model is involved at any point.

614 such crops were harvested (68% of them below 0.55 visible) from the first
half of the corpus, added to training, and the model re-probed on the second
half.

Scored on **one fixed set of groups with one fixed set of head names**
(`--head-model`, so both models are judged on the same rounds):

|  | baseline | + 614 buried crops |
|---|---|---|
| all partners | 36.0% | 33.3% |
| at reject 0.80 | 49.0% | 50.6% |
| at reject 0.90 | 53.0% | 53.7% |

**No effect.**

> A first run appeared to show **+4.9pp** overall and a much better reject
> curve. That run let each model choose the groups it felt sure enough about,
> so the two tables were computed over *different rounds*. Pinning the groups
> erased the whole effect. The gain was population selection, not skill. It is
> recorded here because it was nearly reported as a result.

The low bands stayed flat because **the picture is not there to read**. A crop
centred on a 30%-visible tsum is mostly a picture of whatever is lying on top
of it. This is not a labelling gap, and more labels will not close it.

---

## 6. The decisive measurement: accurate, and still worse to play with

`scripts/group_eval.py` scores a grouping rule in **tsums cleared per drag**,
over 407 held-out boards the classifier never trained on:

    method   agreement    lift   ids/board   cleared
    kmeans       39.9%   1.47x      7.29       3.32
    named        31.5%   1.56x      8.59       2.96    -10.9%  REAL
    merge        39.9%   1.47x      7.28       3.32     +0.0%  noise

Read `lift` and `ids/board` together:

* **`lift` 1.56x vs 1.47x** -- the model *knows more per group* than k-means.
* **`ids/board` 8.59 vs 7.29** -- and it puts *more distinct groups* on a board
  than the game's own limit of 5 allows.

### The mechanism

`named` is the hybrid production rule: rename the crops the model is sure of,
leave the rest with their colour cluster. So a character's **visible** members
get a character id while its **buried** members keep a colour id, and one
character becomes two groups.

**A chain needs its group whole far more than it needs the group correctly
named.** The hybrid buys accuracy on a sixth of the board and pays for it in
fragmentation across all of it.

### `merge`, and why it changed nothing

`merge` was written to do the opposite: let each k-means cluster vote for a
character among its confident members, and join clusters that vote alike. The
id count can then only fall. It moved nothing (7.28 vs 7.29), for the reason
printed above the table:

    13976 of 16547 crops (84.5%) never reached the model: buried below 0.55
    visible, or clipped by the frame edge.
    So it grouped 2571 (15.5%) of the board.

At roughly **five readable tsums per board spread over seven clusters**, there
is almost never a cluster with two confident votes to merge on.

---

## 7. What shipped

**`CHARACTER_MIN_VISIBLE = 0.55`** in `ttheart_sender/game/tsum.py` -- the
character model is never asked about a tsum showing less than the training
set's own cut-off. Below the floor a tsum keeps its k-means `kind`, which
agrees with the game 37% of the time: poor, but more than twice what the model
manages down there, and it does not arrive dressed as certainty.

* Exposed as `--character-min-visible`; `0` restores the old behaviour.
* The end-of-round line now reports how much of the board was **never asked**,
  so "named 20% of detections" cannot be mistaken for a model that looked and
  declined.
* `flows/play.yaml` carries `character: ""` -- **off**, with the value and the
  revert on one line, per the house rule.

The tooling stays, because it is how the question was answered and how it would
be re-asked:

| script | what it is for |
|---|---|
| `scripts/classify.py` | train the classifier; `--device`, best-epoch checkpointing, session split stamped into the artifact, and both accuracies printed |
| `scripts/label_gaps.py` | run the model over unlabelled sessions, keep only the crops it refuses, write one contact sheet per group -- turns "label a corpus" into "name a dozen sheets" |
| `scripts/mark_probe.py` | score any model against the game's own marks, banded by visibility; `harvest` turns the marks into buried-crop labels |
| `scripts/group_eval.py` | score a grouping rule in tsums cleared per drag, `--held-out` |

---

## 8. What is still worth labelling, and what is not

**Not buried crops.** §5 settles that.

What the label set is genuinely short of is **rounds per class**:

* 9 of 44 classes come from a **single round** -- CheesetheMouse, Cleo,
  Flounder, Flik, Lotso, Mike, Monstro, Oliver, Sisu. A class that exists in
  one session cannot be both trained and scored under any session split.
* **Beast**, the equipped base tsum, has **9 crops**. The character every round
  is played for is among the least labelled things in the set.
* 62 of 172 sessions have never had a crop labelled; 153 crops in 3 sheets are
  currently unnamed.

But note what that buys: a better classifier, which §6 shows cannot be used.
**Label more only if the detection side changes first.**

---

## 9. What would revive this

One thing, and it is not a model: **a different picture.**

Everything above holds because a crop centred on a buried tsum contains its
neighbours rather than itself. Anything that recovers a buried tsum's own
pixels would move every number in this document at once:

* a crop window that adapts to the visible crescent rather than taking a fixed
  square at the detected radius;
* segmentation that masks the occluding neighbours out of the crop;
* tracking across frames -- the same physical tsum is well visible in *some*
  frame as the pile shifts, and its identity there is its identity earlier;
* a detector whose radius estimate is the tsum's true radius rather than its
  visible extent, so `visibility` becomes a mask rather than a scalar.

These are detection problems. **No amount of labelling reaches them**, and
that is the single most useful thing this document has to say.

---

## 10. Two things this does not claim

**About 15% of marked partners are highlight-rendered, not
character-rendered.** The thirty-third round found that the game draws its link
highlight over a tsum it has marked, and `_before.jpg` is captured with the
marks up -- so some of the partner crops scored in §4 are pictures of the mark.
Too small to explain a fall from 74% to the chance rate, so the cliff stands,
but the low bands are a floor with a known bias rather than a clean
measurement. A re-run should exclude marked partners.

**The probe's head names come from the model.** A mistaken head mislabels a
whole group, so every agreement figure above is a **floor** on accuracy rather
than an estimate of it. A systematically wrong head would drag every band down
together, and the bands do not move together -- but that is an argument, not a
control. Where human-named heads were available they were used, and they agree.

**Nothing here has been played.** Every number is offline. The simulated
`cleared` column is the same metric the twenty-sixth round disqualified with an
ORACLE that scored *below* the shipped rule, so a few percent either way in
that column is not evidence. The **-10.9%** is quoted because its mechanism is
visible directly in `ids/board`, which is a count and not a simulation.

And after the thirtieth round's finding that score turns on **FEVER share**
(r = +0.808) rather than on clear rate, with two rounds in fifteen scoring ~1%
of normal because they never reached FEVER, a 3% grouping effect is not where
the score is. That is the next thing worth measuring, and it is a different
question entirely.

---

*Full round-by-round record in [DATASET-FINDINGS.md](DATASET-FINDINGS.md),
rounds 31 and 32.*
