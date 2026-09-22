# Is the character model reading the tsum, or its colour? — 2026-09-08

*Requested by `Tsum Tsum Object Recognition Improvement Plan.md`, which asked
for an audit and a baseline before any change, and for each experiment to be
recorded whether or not it worked. This is that record.*

*Continues [RECOGNITION-AUDIT.md](RECOGNITION-AUDIT.md) (2026-09-06), which
audited the same model and did not test this.*

---

## 0. The headline

The shipped model scores **97.9%** on held-out sessions and **44.5%** on the
same crops in greyscale. Two dozen classes go from **100% to 0%**.

The plan asked whether colour is the primary feature. It is.

---

## 1. Most of the plan was already done

Before measuring anything, the pipeline was audited against §1 and §5–§16. The
plan's structural recommendations describe this repository as it already is:

| plan asks for | status | where |
|---|---|---|
| §7 crop centred on the tsum's bounding box | **already** | `game/crop.py`, centred at the detection |
| §6 tsum at 60–85% of the crop, consistent framing | **already** | `window=1.0` radii → an inscribed disc, **78.5%** |
| §5 one tsum, neighbours excluded | **already** | 1.0r, not 1.5r; two touching tsums sit 2.44r apart |
| §13 detect → crop → normalise → classify | **already** | `crops.py` → `crop.py` → `classify.py` |
| §14 transfer learning, lightweight backbone | **already** | MobileNetV3-Small, ImageNet weights, 96px |
| §15/§16 no leakage between splits | **already** | split by **session**, never by crop |
| §17 dataset quality checker | **already** | `crops.py status`, `recog_eval.py`, `golden.py` |

The by-session split deserves emphasis, because it is what makes the 97.9%
trustworthy enough to be worth attacking: the number is not near-duplicate
leakage. It is a real number, earned by the wrong evidence.

**One gap, and it is the whole finding.** `classify.py:augment` perturbs
geometry — flip, ±20°, ±10% scale — and one global brightness gain. It never
touches hue or saturation. So across the entire training set each character's
colour is a *perfectly stable* cue: the cheapest separating feature available,
and therefore the one the network takes.

---

## 2. The measurement

New tool, `scripts/colour_probe.py`. It serves a model **its own held-out
crops** (the sessions its `.json` names) under perturbations that leave the
object intact and move the colour. No retraining is needed to run it.

1,805 held-out crops, 62 classes, 114 sessions.

| variant | kind | shipped `character.onnx` |
|---|---|---|
| normal | baseline | **97.9%** |
| grey | colour | **44.5%** |
| desat 50% | colour | 92.2% |
| hue +30 | colour | 86.0% |
| hue +60 | colour | 58.8% |
| hue +120 | colour | **44.2%** |
| hue +180 | colour | 57.2% |
| bright ×1.3 | light | 94.0% |
| dark ×0.7 | light | 92.5% |
| contrast ×1.4 / ×0.6 | light | 96.7% / 96.5% |
| rot 8° | geometry | 98.3% |
| shift 8% | geometry | 97.5% |
| scale 1.15 | geometry | 97.6% |

**The geometry rows are the control, and they are what make this readable.**
They do not move. The model is near-perfectly invariant to exactly what `augment`
perturbs, and collapses under exactly what `augment` leaves alone. That is not a
coincidence to be interpreted; it is the augmentation policy, measured.

### Which classes (§12)

Twenty-four classes score **100% normally and 0% in greyscale** — WhiteRabbit,
Wade, Sulley, Sisu, Rajah, Pooh, Mushu, Lucifer, Hades, Elliot, Eeyore,
CruzRamirez, CheshireCat, Alien, AirplaneDonald, 22, and more. Their identity is
carried entirely by hue. Nothing about their shape is being used.

---

## 3. The fix, and what it costs

`classify.py --colour-aug` (new, **off by default**): hue rotated about the grey
axis, saturation scaled, a share of each batch greyed outright, plus value and
contrast jitter. Luminance-preserving and pixel-preserving — it moves colour
without moving structure, so it cannot be confused with the geometric
augmentation that buys a different invariance.

Both models below are trained from the same seed, the same split, the same
classes; the *only* difference is the flag.

| variant | baseline | `--colour-aug` |
|---|---|---|
| **normal** | **98.3%** | **96.6%** |
| grey | 38.0% | **96.5%** |
| desat 50% | 89.8% | 96.0% |
| hue +30 | 88.0% | 95.5% |
| hue +60 | 64.7% | 93.5% |
| hue +120 | 47.8% | 96.6% |
| hue +180 | 48.5% | 91.2% |
| bright ×1.3 | 96.1% | 93.1% |
| dark ×0.7 | 93.1% | 95.9% |
| rot 8° | 98.2% | 96.6% |

**Worst case across every perturbation: 38.0% → 91.2%. The cost is 1.7 points
on unperturbed crops.**

On the frozen golden set at the shipped reject floor of 0.85, the trade is:

| | accuracy | coverage |
|---|---|---|
| baseline | 99.3% | 97.3% |
| `--colour-aug` | 98.5% | 95.6% |

### An intermediate result worth keeping

The first colour-augmented model preserved luminance *exactly* and was **worse
than baseline on brightness** (dark ×0.7: 86.3% vs 93.1%). A model that has
stopped reading chroma reads luma instead, and nothing had ever perturbed luma
honestly — the existing ±20% gain multiplies the *normalised* tensor, which is
an affine move about each channel's ImageNet mean rather than a brightness
change. Adding value and contrast jitter in 0..1 space fixed it (86.3% → 95.9%).
Removing one shortcut promotes the next one; it has to be measured, not assumed.

### The new confusions are the honest cost

Baseline's worst pairs are `LittleOyster→Piglet`, `22→Sulley`. The
colour-augmented model's include **`Daisy→Donald`** — two characters with nearly
the same silhouette, separated in the training set almost entirely by hue. This
is what §19 is actually asking for: colour demoted to secondary, and a small
number of genuinely shape-ambiguous pairs left over as the residue.

---

## 4. Background leakage: measured, and not a problem (§10, §18 Q9)

The probe masks the crop at the inscribed disc, filled with the crop's own edge
mean rather than black:

| | baseline | `--colour-aug` |
|---|---|---|
| surround only (tsum blanked) | **5.9%** | **4.3%** |

Against a majority-class rate of roughly 6%, the surround alone carries
**no identity**. There is no background shortcut to remove, and §10's
`RandomErasing`/`RandomCrop` remedies are not needed.

*The complementary "tsum only" row (78.9% / 52.3%) is **not** evidence that the
model needs context: the hard circular mask is itself out of distribution and
clips into the tsum. Only the surround-only row supports a conclusion.*

---

## 5. What governs all of it

Every number here is measured on crops **≥0.55 visible**, because that is all
`crops/labelled/` contains. The median tsum on a real board shows **0.41**
([RECOGNITION-AUDIT.md](RECOGNITION-AUDIT.md) §3). So this describes the least
occluded fifth of a board, and the robustness bought above is robustness on that
fifth. That ceiling was already tested and is not liftable by labelling more
buried crops — `IDENTITY.md` §5 tried it, and found no effect.

---

## 6. What ships

**Nothing, yet.** `--colour-aug` is off by default and no model was promoted.
Every model in `models/` remains reproducible by omitting the flag, and the
exported `.json` now records `colour_aug` so a jittered model and a clean one
can never be compared on clean accuracy alone and the better one quietly retired.

The decision this leaves open is a real one, and the probe cannot close it: the
colour-augmented model is **1.7 points worse on the crops the bot actually
sees** and dramatically better on crops it currently never sees. Whether the
game's lighting, fever flashes and skill effects move a tsum's hue enough to
matter is a question about gameplay, and by this project's convention **a played
round decides it** — the same bar `reject_model` was held to.

---

## 7. What changed

| file | change |
|---|---|
| `scripts/colour_probe.py` | **new** — colour/light/geometry/context robustness table for any exported model, on its own held-out sessions |
| `scripts/classify.py` | **new** `colour_jitter()`; **new** opt-in `--colour-aug`, `--hue`, `--grey-p`; manifest records `colour_aug` |
| `tests/test_colour_aug.py` | **new** — 5 tests: inert when off, greys in BGR order, moves colour without moving pixels, hue really rotates, flag is off by default and recorded |

No gameplay file was touched. No model was promoted. `character: ""` is
unchanged and the character model still ships **off**.

---

*Round-by-round record in [DATASET-FINDINGS.md](DATASET-FINDINGS.md); the prior
audit is [RECOGNITION-AUDIT.md](RECOGNITION-AUDIT.md).*
