# TODO: decide whether `--mode blob` becomes the default

Status: **implemented, opt-in, not yet settled.** Added in `ttheart_sender/game/tsum.py`
as `blob_adjacency()` plus a third `--mode` alongside `touch` and `reach`.
Nothing changes unless you pass `--mode blob`, so this is safe to leave sitting.

## What it is

`adjacency()` decides two tsums are linked when their centres are close enough
(`--link-px`) and no third tsum sits on the segment between them. `blob_adjacency()`
asks the question that test approximates: is there a continuous run of this
colour joining these two specifically? It masks the board to the pair's cluster,
keeps only what falls inside their two disks, and tests connectivity.

The idea came from an external HSV-based script (`tsum_link_finder.py`, written
against two saved screenshots). Most of that script does not transfer — see
[Why only this part](#why-only-this-part) — but this one piece measured well.

## Why the mask has to be grown first

The clusters here are *face* colour, and two touching sprites still have an
outline and a body between their faces. Measured over the labelled boards:

- median gap between the face blobs of two tsums a human chained: **23.4px**,
  against a detected radius of ~24
- share of those pairs whose blobs are one connected component: **2.2%**

So a literal contact test scores ~0%. Dilating the mask outward by `0.9r` to
approximate the sprite's full extent is what makes it work. Below ~0.6r almost
nothing links; the score saturates at 0.9r and is flat to 1.2r, so 0.9 sits in
the middle of a plateau rather than on a tuned edge.

This is the same problem `tsum_link_finder` solves by hand with its find-colour
vs link-colour split (Mickey located by his peach face, linked by his black
body). Growing the mask gets the same effect without authoring a second HSV
window per character.

## The numbers

Replayed through the 97 hand-drawn links `score` uses, on identical detections:

| rule                                | links accepted | edges made | cost (62-tsum board) |
|-------------------------------------|---------------:|-----------:|---------------------:|
| `adjacency()`, `--link-px 105`      |          76.3% |        246 |               1.3 ms |
| `adjacency()`, `--link-px 150`      |          83.5% |        387 |                    — |
| `adjacency()` ceiling, any distance |          84.5% |          — |                    — |
| **`blob_adjacency()`, grown 0.9r**  |      **86.6%** |    **291** |          **60 ms**   |

It clears what `adjacency()` can reach at *any* distance setting, and does it
with a sparser graph than the loose setting it beats — more of the human's
links accepted, fewer invented.

Cost note: the first version ran 344ms. Restricting the connectivity test to a
bounding box around each pair, and dilating each kind's mask once instead of
per pair, brought it to 60ms. Further speedups are available if wanted — most
obviously, run the cheap rule first and only fall back to the blob test for
pairs it rejects within reach.

## What still has to happen before this is the default

1. **A full-round A/B against `touch` on the live emulator.** This is the
   blocking item. `reach` mode carries the same unresolved note and for the
   same reason — when it is run, both comments should say which won.

2. **Confirm the score is not being gamed.** The labels record only links a
   human *did* draw, so acceptance is a positive-only measure and rises for
   free as a rule accepts more — the exact trap documented on `_recolour()`,
   which looked like an 80%→89% win and was a clear end-to-end regression.
   Edge count is the control here and blob wins on both axes at once, which is
   real evidence but not proof. Calibrating this properly needs negative link
   examples (pairs a human marks as *not* chainable), which the `label` flow
   does not currently collect.

3. **Decide whether 60ms is affordable.** It roughly doubles per-frame think
   time. `live -n 20` reports the real per-frame cost; run it in both modes.
   Fine if reaction speed is not the binding constraint, not fine if it is.

## Why only this part

The source script was rejected wholesale for reasons worth recording, so nobody
re-proposes it:

- **`FIXES`** — hand-entered `add`/`drop` coordinate lists keyed by screenshot
  *filename*, 12 corrections on one demo board alone. The detector's unaided
  output was wrong by ~12 tsums and the corrections are what made it look
  right. A live bot has no filename to look up, so none of the demonstrated
  accuracy transfers.
- **Hand-authored HSV windows per named character** (9 characters, 2 palettes).
  The game has hundreds and rotates event boards. The k-means approach exists
  precisely so a new character costs nothing.
- **`PLAYFIELD` as a fixed rect.** Already solved here by `LAYOUTS`, keyed by
  capture size, because live LDPlayer grabs carry emulator chrome that saved
  screenshots do not.
- **No base-tsum concept** — it takes the longest chain, ignoring that clearing
  the equipped tsum is what charges the skill.
- **Unbudgeted exhaustive DFS** in its `longest_path`, on the stated assumption
  that components are under 15 nodes. Boards here run ~62 tsums with components
  well past that; `longest_path()` here takes a time budget for that reason.

## Related, from the same investigation

Separately confirmed while looking at this: detection never uses grayscale, and
should not. Scored over the 10 labelled boards — Lab **f1 0.759**, raw BGR
**0.645**, grayscale **0.500** (22.4% of character pairs become inseparable in
grayscale, and merges triple).

~~Worth fixing regardless of what happens above:~~ **done** — the darkness test
now goes through `_lightness()`, which names the assumption that a cluster
centre's channel 0 is lightness. That is true of Lab and of nothing else; hand
the pipeline BGR centres and channel 0 is blue, so every warm tsum — Pooh,
Tigger, Pluto — would silently read as "dark" and get discarded (measured: f1
0.759 → 0.504, with nothing in the failure pointing at the colourspace).

## Second look: `tsum_board.py`

The external script came back as a five-board successor with its own FINDINGS
section. Reviewed against the ten labelled boards; **one behavioural change**
came out of it, and it is not one of the findings. Recorded here so this does
not get a third review from scratch.

### What it confirms (no change needed)

- **Blob connectivity.** The successor independently arrived at the same rule
  as `blob_adjacency()` — mask to the character's colour, keep what falls
  inside the two sprites' disks, dilate, test connectivity. It reaches it from
  the opposite direction: where this module dilates the *face* mask by `0.9r`
  to approximate the sprite, the script authors a second HSV window per
  character for the part that actually touches (Mickey located by his peach
  face, linked by his black body). Two readers built independently and landing
  on connectivity-over-distance is a point in the rule's favour, but it is not
  the A/B item 1 asks for and it does nothing about the positive-only scoring
  problem in item 2 — that script has no negative link examples either.
- **Fixed `k`.** Its FINDING 1: silhouette-based cluster-count selection
  recovered the right K on hand-labelled centres and picked the *same wrong*
  K on all five boards once fed real detector output, because false positives
  form clusters of their own. `k` stays a parameter here, and `marked_by_game()`
  stays the way ground truth is obtained.
- **Refitting colour per board.** Its FINDING 2: HSV windows do not transfer
  between boards — the same character read S~33-41 on one board and S~100 on
  another, on scene lighting alone. That is the argument for quantising per
  frame rather than shipping constants, and for the plausibility check in
  `play_loop()` that throws a stale palette away and refits.
- **Peak recall vs. splitting.** Its FINDING 4 is the trade the `1.6r`
  de-duplication already handles, reached the same way.

### What was ported, measured, and rejected

Its FINDING 3/5 — arbitrate a contested sprite by *normalised* coverage, each
candidate scored against what its own character reaches, because raw scores are
not comparable across characters. Sound for that script: its windows are
hand-authored and **nest**, so Donald's box (any hue, S<22, V>215) is a superset
of Pluto's white muzzle and wins every Pluto outright — measured there as Pluto
falling from 7 detections to 1.

That failure mode cannot occur here. k-means centres *partition* colour space,
so no cluster is a superset of another and peak depth is already comparable
across kinds. Ported anyway and scored:

| arbitration                                  |   f1 |
|----------------------------------------------|-----:|
| unclipped depth (current)                    | **0.762** |
| normalised depth, global sort key            | 0.748 |
| normalised depth, colliding different-kind pairs only | 0.749 |

Both variants lose, and for a reason worth keeping: normalising lifts every
peak of a shallow cluster, and the shallow clusters here are mostly the
spurious ones, so phantoms start taking slots off real tsums (boards 4 and 5
lost 4 real detections each). The narrower variant is the one the script itself
recommends — "a serviceable tie-break and a bad classifier" — and it is still a
loss. The finding is real; it is a fix for a defect this pipeline does not have.

### The one change

`detect()` now sorts de-duplication candidates on unclipped peak depth instead
of on `t.r`. `t.r` is clipped at the radius, so every fully visible tsum ties
there and the winner of a collision fell through to insertion order — i.e. to
whichever cluster id k-means happened to number first, which is not stable
between fits. Scores identically (f1 0.762 either way); it is a determinism
fix, not an accuracy one.

### Still rejected, same as before

`FIXES` (hand-entered add/drop coordinates keyed by screenshot filename),
hand-authored HSV windows per named character, `PLAYFIELD` as a fixed rect, no
base-tsum concept, and an unbudgeted exhaustive `longest_path`. The successor
kept all five. Its own FINDINGS 1 and 2 are, read plainly, the case against its
own approach: per-board calibration is described there as "mandatory, not an
optimisation", and a live bot has no board name to look one up by.
