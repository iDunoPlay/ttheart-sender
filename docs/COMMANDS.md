# Commands

Quick reference for the two things this repo does: sending hearts (the flow
runner, `main.py`) and playing Tsum Tsum (the board reader, now
`ttheart_sender/game/tsum.py`).

The board reader moved out of `scratchpad/` so flows could use it — see [Play
from a flow](#play-from-a-flow). Every command and flag is unchanged; reach it
as `python -m ttheart_sender.game.tsum <anything>`.

`scratchpad/` is git-ignored and holds the labelled boards `score` and `eval`
read (`boardN.png` + their `.label.json`) — see [Calibration](#calibration-teaching-it-what-a-board-really-looks-like).
Everything else in there is regenerable debug output — safe to clear whenever
it gets big, but the boards and their labels are hand-made and are not.

Run everything from `C:\GitHub\ttheart-sender`. Prefix with
`.\.venv\Scripts\python` if `python` isn't the venv's.

**Press F12 to stop anything.** Slamming the cursor into a screen corner also
aborts (pyautogui's fail-safe).

---

## Setup / sanity checks

```powershell
python main.py detect      # is LDPlayer found?
python main.py prepare     # park it top-left and focus it
python main.py shot        # screenshot what the app sees
```

If `detect` finds nothing, run `python main.py windows`, find LDPlayer, and copy
its `class=` into `window.target.class_names` in `config.yaml`.

---

## Send hearts

Two entry points. `flows/launch.yaml` launches the game and clears the startup
prompts, then hands off to `flows/resume.yaml` — the forever loop that claims
the mailbox (item by item, or in one pass through the game's Claim All dialog
when `claim_all` is set), sends hearts (every cycle, or on the clock when
`return_heart_timed` is set), and (if `play_chance_percent` says so) plays a
round instead of pausing between cycles. `resume` on its own skips the launch
and assumes the game is already up.

```powershell
python main.py run launch --dry-run   # log the clicks, send nothing
python main.py run launch             # launch the game, then loop
python main.py run resume             # game already open
python main.py tray                   # the same two, from the system tray
```

Useful while building or fixing a flow:

```powershell
python main.py flows                  # list flows
python main.py validate               # parse them without running
python main.py snip send_heart        # re-crop a button (F8 at each corner)
python main.py find send_heart        # test-match it right now
python main.py point                  # hover + F8 to read a coordinate
python main.py region                 # F8 at two corners -> the --board string
```

---

## Play Tsum Tsum

Reads the board, finds a chain of same-character tsums, and drags it. Start a
game first — it refuses to act on menus.

```powershell
python -m ttheart_sender.game.tsum play --duration 50
```

That's the whole command. Board rect, skill-icon position and cluster count are
picked automatically from the frame size, so they only need passing if you're on
a layout that hasn't been measured.

A round is 60s; `--duration 50` stops before the results screen appears.

### Looking at a single board

```powershell
python main.py shot -o scratchpad/board.png
python -m ttheart_sender.game.tsum analyze scratchpad/board.png -o out.png
```

Writes an overlay PNG (every detected tsum, best chain drawn on top) plus a JSON
sidecar with the drag waypoints. Add `--debug-dir dbg` to also dump the colour
clustering and the patch it sampled for the base tsum — the first place to look
when detection misbehaves.

### Other subcommands

```powershell
python -m ttheart_sender.game.tsum learn          # fit a palette from collected samples
python -m ttheart_sender.game.tsum live -n 20      # time capture+detect+path per frame
python -m ttheart_sender.game.tsum skillcheck      # watch the skill button's gold reading
```

---

## Calibration: teaching it what a board really looks like

Four commands, one loop. `grab` collects boards, `label` marks them up, and the
two readers turn the marks into settings: `score` for how far apart chained
tsums are, `eval` for whether detection finds the right tsums at all.

```powershell
python -m ttheart_sender.game.tsum grab -n 10        # start a round; it grabs on a timer
python -m ttheart_sender.game.tsum label scratchpad/board5.png
python -m ttheart_sender.game.tsum score             # -> --link-px
python -m ttheart_sender.game.tsum eval              # -> detection precision/recall
```

`grab` drops any frame whose detection count is outside `--min-tsums`
/`--max-tsums`, so the countdown, the results screen and the Home screen don't
end up in the dataset. Existing `boardN.png` files are never overwritten.

### Marking a board

`label` opens the board with every detection circled. Four modes, selected with
the number keys:

| Key | Mode | Click does |
|---|---|---|
| `1` | `path` | Add the tsum to the chain you'd drag — the original behaviour |
| `2` | `missed` | Mark a real tsum that detection did **not** find (no snapping) |
| `3` | `false` | Mark a circled "tsum" that isn't one — click again to un-mark |
| `4` | `group` | Add a tsum to a same-character group |

`n` starts the next path/group, `u` undoes, `c` clears the current one, `s`
saves, `q` quits. The banner across the top shows the mode and the running
counts.

**`r` is the one that matters.** It flags the board as fully reviewed, and
`eval` ignores boards without it. That's deliberate: a half-marked board looks
like a board with no mistakes on it, so scoring one would reward whichever
setting misses the most. Only press `r` once you've swept the whole board for
misses and false positives.

Saving writes every detection present at review time into the label file, which
is what lets `eval` re-run detection with *different* parameters and still know
which of the new detections were approved.

### The opt-in rules, and how to switch them off

`flows/play.yaml` turns on rules that are off in the code's own defaults. Each
is one line under `options:`, and **deleting the line is the whole revert** —
no rebuild, nothing else to change:

| line | what it does | costs | state |
|---|---|---|---|
| `bowl_reject: 40` | drops detections whose colour says they landed on the bowl, not on a tsum (precision 0.675 → 0.734) | recall 0.874 → 0.844 | on; a live round preferred 40 to 60 |
| `radius_lock: 5` | measures the tsum radius over five frames and holds it for the round, instead of re-measuring on every refit and at the FEVER transition | a bad warm-up gets held too | on; the FEVER fix |
| `fever_min_tsums: 12` | lowers the "is this even a board" floor while FEVER runs, where the FEVER template already answers that | a fade to black could read as a board | on; the FEVER fix |
| `mode: blob` | decides links by whether the two tsums' colour blobs join, rather than by centre distance (76.3% → 86.6% of hand-drawn links accepted) | ~60ms a frame against ~1.3ms | **off** — no live improvement yet |
| `palette: models/…` | reads the board through colours learned offline by `tsum learn` instead of re-fitting k-means per frame | none at runtime | **off** — fitted and refused: no `k` beat the per-frame fit |
| `verify_reach: 260` | asks the game to check a chain before dragging it, but only when it reaches further than 260px from the press — where 2 drags in 3 are wrong | ~0.25s on the ~18% of drags that qualify | **off** — replay says +19% cleared; needs a live round |

They are independent and touch different parts of the pipeline, so **turn them
off one at a time**: switching several back at once says nothing about which
one was at fault. `docs/TODO-bowl-reject.md`, `docs/TODO-fever-detection.md`
and `docs/TODO-blob-adjacency.md` carry the measurements behind each.

### What each reader tells you

`score` reads the gaps between tsums you chained in `path` mode and reports
them in pixels — the unit `--link-px` uses — plus what each candidate threshold
would capture. It then replays every hand-drawn link through `adjacency()`
itself, which is the part that actually decides, because distance is only one
of its three gates.

From 80 labelled links, that replay is the reason `flows/play.yaml` now sets
`link_px: 150`:

| `--link-px` | accepted | wrong kind | too far | blocked |
|---|---|---|---|---|
| 90 | 62.5% | 4 | 17 | 9 |
| 100 *(old default)* | 71.2% | 4 | 10 | 9 |
| 150 | **81.2%** | 4 | 1 | 10 |
| ∞ | 82.5% | 4 | 0 | 10 |

At the shipped 100, `touch` mode refused nearly three in ten of the drags a
human considered valid. Raising it to 150 recovers ten points; past that
nothing is left to win, because the remaining 18% dies on the third-tsum block
test and on kind errors, neither of which any distance can fix.

**These labels are all positive examples**, which bounds what they can tell
you. Every link is one a human drew, so the replay measures how many real links
the rule *rejects* and can say nothing about how many impossible ones it would
*accept*. Loosening the rule always looks free here.

That is not a hypothetical caveat — it already cost a round. `link_px` was set
to 150 off the table above, and in play it was clearly worse: longer chains,
but repeatedly dragging across tsums in the way and stalling. `mode: reach`,
which drops the distance and block gates entirely, was worse still. **A number
read off positive-only labels needs a played round before it is believed.**

### Why longer reach loses: the leading-character rule

The game adds a tsum to your stroke only if it matches the one you started on;
a non-matching tsum you drag over is skipped, not fatal. So a chain that mixes
characters still clears its leading-character subset, and only stalls outright
when that subset falls below three. Chain *length* therefore is not the thing
to maximise — the leading-character subset is, and a longer reach grows the
chain and the impostors together.

Measured over the labelled boards, per drag:

| `link_px` | `block` | dragged | actually clears | wasted | stalls |
|---|---|---|---|---|---|
| 90 | 1.25 | 6.3 | 2.5 | 3.8 | 2/6 |
| 105 | 1.25 | 6.6 | 3.6 | 3.0 | 1/9 |
| 120 | 1.25 | 6.7 | 4.0 | 2.7 | 1/9 |
| 150 | 0.75 | 7.4 | 3.6 | 3.8 | 1/8 |

`flows/play.yaml` runs `link_px: 105, block: 1.25` on this. `block` went up
from 0.75 because 0.75 radii is ~19px while a tsum is ~25px — one lying bodily
across the line did not count as blocking it, which is exactly the crossing
behaviour seen in play.

### The opening hop decides the whole drag

The first tsum pressed sets the character for the stroke, so an opening hop the
game does not register wastes the entire drag — where a bad hop later on costs
one skipped tsum and the stroke carries on. Nothing in the path search knew
that: `longest_path` maximised node count and returned whichever maximal path
it reached first, so a chain could open with a long jump across a clump it
could have walked in order.

Two fixes, neither of which shortens a chain:

- **`orient_chain`** — a path `a-b-c-d` is the same tsums dragged either way, so
  the direction is free. Take whichever end opens shorter.
- **nearest-first exploration** in `longest_path`, so the first maximal path the
  search reaches is one that took short hops. Ties are still pruned; resolving
  them properly is exponential and this runs ten times a second.

Measured over the labelled boards, at `link_px 105 / block 1.25`, that moved
effective clears per drag from **3.6 to 4.3**, cut waste from 3.0 to 2.3, and
removed the stall entirely (1/9 → 0/9), at no cost in decision time (1.8ms).

`--first-leg-px` caps the opening hop explicitly, and defaults to **0, off**,
because measurement said it was unnecessary: orientation alone brings the mean
opening hop to 64px, well inside any cap worth setting, and 60px starts
trimming chains and costs clears (4.3 → 3.1, and two stalls reappear).

### The ceiling: colour cannot tell characters apart

Roughly 40% of every drag is wasted on tsums that will not clear, and no
distance setting moves that much. Nearly every top chain spans two labelled
characters, because "kind" is a colour cluster and two characters sharing a
dominant colour become one.

`purity_filter` is meant to be the defence and **does nothing at its default**.
Sampling each chain member's colour and measuring its Lab distance from the
chain median:

| | p50 | p90 | max |
|---|---|---|---|
| right character | 2.0 | 23.4 | 26.9 |
| wrong character | 6.4 | 19.8 | 33.7 |

The maximum is 33.7, so `--purity 35` never drops anything — verified by
`eval`-style replay giving byte-identical results at `0` and `35`. Nor does any
lower value help: at 15 it drops 19% of wrong-character members and 24% of
right ones, worse than a coin flip. The distributions overlap because merged
clusters are colour-similar *by construction*.

So colour is exhausted. Separating characters needs a signal colour does not
carry — shape, or a small learned classifier over the face crop. Until then,
~40% drag waste is the floor, and tuning distance only redistributes it.

### Asking the game instead — and it answers

```powershell
python -m ttheart_sender.game.tsum hold   # press one tsum, film what lights up
python -m ttheart_sender.game.tsum idle   # touch nothing, film the hint
```

**Holding a tsum makes the game mark every tsum you could link to it.**
Measured: press one Piglet, and six tsums light up — five more Piglets plus one
warthog sitting inside the glow's aura. Three of the five are 109px, 131px and
167px away, far outside that aura, so they are marked deliberately and not
splashed by it.

Two things make this the most valuable result in this document.

First, **it is right where the colour clustering is wrong.** Those six Piglets
were filed under clusters 1, 8, 8, 5, 1 and 1 — one character split three ways,
the exact failure `eval` scores. The game is not confused by any of it.

Second, **it answers reachability too, not just identity.** A dozen further
Piglets on the same board were left unmarked, so the highlight is not "every
tsum of this character" — it is "every tsum you can actually chain from the one
you are touching". That is precisely the question `adjacency` exists to guess
at, and this is the true answer rather than a threshold fitted to labels.

The cost is close to nothing, because **the press is not an extra action** — it
is the first step of the drag either way. Press, read the marks, then complete
the drag through them without ever releasing.

Watch out for two things before trusting a reading:

- **The glow has an aura ~90px across**, and anything inside it changes whether
  the game meant it or not. That is where the warthog false positive came from.
  Any implementation needs to discount tsums whose reaction is explained by
  sitting under the aura.
- **The board must be still.** A press on a settling pile reports the settling.
  `hold` settles twice and compares before it presses, and aborts if the board
  is still drifting.

**The idle hint fires too, and looks identical**, which is what first suggested
all of this — but it takes seconds to appear, so it is useless during a 60s
round. `idle` filmed 45s at 5fps mid-round and caught nothing.

#### …and reading it every drag still loses

`--verify-hold` does the obvious thing with all of the above: press the chain's
first tsum, read which members the game marks, drop the rest, drag what is
left. It is off by default because it was measured over full rounds and lost,
twice, decisively.

| | chains | dragged | mean | trimmed | rejected |
|---|---|---|---|---|---|
| plain | 104 | **527** | 5.1 | — | — |
| `--verify-hold` | 53 | 196 | 3.7 | 43 | 27 |

Two things sink it. Reading costs ~0.1s of a 50s round on *every* drag, which
halves the number of chains played; and the trim shortens the ones that
survive. Even valuing the plain run's output at the ~60% that actually clears,
~316 beats 196 comfortably.

**The lesson generalises: throughput dominates accuracy here.** Dragging fast
and wasting 40% of each stroke beats dragging carefully at half the rate. Any
future accuracy work has to be free at run time to be worth anything.

Two results worth keeping from the attempt. The game **rejected 36% of the
chains detection proposed**, which independently confirms the ~40% waste
measured from the labelled boards by an entirely different route. And lowering
`--hold-delay` from 0.15 to 0.10 *raised* trimming from 32 to 43 — those extra
11 were marks that had not rendered yet, so anyone retrying this should treat
0.15 as a floor, not a default.

If this is picked up again, the promising shape is not per-drag verification
but **per-round calibration**: press once, learn which colour clusters are
really one character, and play the rest of the round fast with the clusters
corrected. That pays the cost once instead of a hundred times.

Four earlier runs of `hold` found nothing, and every one was a bug in the probe
rather than an answer: no focus click, so an unfocused window ate the press; a
baseline taken while the pile was still falling; a picker that weighted
centrality so weakly it chose a tsum off the bottom edge of the play area; and
a pick with no partners, which cannot distinguish "marks nothing" from "had
nothing to mark". If a probe reports a negative, suspect the probe first.

#### …but it wins when you are the one pressing

```powershell
python -m ttheart_sender.game.tsum assist            # hold a tsum, keep holding
python -m ttheart_sender.game.tsum assist --dry-run  # report the path, move nothing
python -m ttheart_sender.game.tsum assist --debug    # + write what it saw per press
```

`assist` is a separate mode, not a change to `play` — switch between them by
running one or the other. You press and hold a tsum yourself; it reads which
tsums the game marked, orders them into a path starting at the one under your
finger, and warps the cursor through them. **It never presses** — your press
is what starts the stroke — and by default it *does* release, the moment the
path is walked. Pass `--no-auto-release` to keep the release yourself.

That default is measured, not tidiness. In the first live run, releasing early
was the commonest way a press was wasted: three of six good readings were cut
short, one after 25 of 41 legs. A long chain is seconds of walking and the game
gives no sign that it is still going, so asking for a steadier hand was never
the fix.

This is the same reading `--verify-hold` does, and none of what sank it
applies. That run lost because it paid ~0.1s on every drag and halved the
chains played; here you choose when to press, so there is no throughput to
lose. Better, it gets a *cleaner* read: `--hold-delay` was being squeezed to
0.10 to save time, below the 0.15 floor where marks have not rendered, whereas
a human holds far longer than either. `--delay` defaults to 0.25 for that
reason.

Three differences from `play`'s chain building, all because the game is
answering rather than being guessed at:

- **No length cap** (`--max-chain 0`). `play` caps at 8 to keep a *guessed*
  tour tight; there is no guess here, and longer scores better.
- **The path is forced to start under your finger.** `orient_chain` may reverse
  a path to shorten its opening hop and `_nearest_neighbor_tour` picks its own
  start — free choices only while nothing is being touched yet.
- **Hits under the glow are kept but counted separately.** `marked_by_game`
  keeps everything within the aura *regardless*, because its risky move is
  dropping a member from a chain. Here the risky move is the opposite one, so
  a reaction inside the aura has to clear the same threshold as any other, and
  the report says how many hits were aura-close so a bad reading is visible.

It refuses rather than guesses, and prints why: you pressed somewhere that is
not on a detected tsum, the game marked fewer than `--min-chain` tsums (below
three it clears nothing, so a short path is worse than none), or **more than
`--max-marked` of the board lit up**.

That last one replaces the drift check `hold` uses, and is worth explaining.
Refusing up front on a moving board was tried first and was wrong twice over.
It measured drift across the *whole frame*, which never reads still during a
round and never could — the score counter, the timer and the FEVER meter all
animate continuously, so it was a HUD activity meter that said nothing about
the pile. And even measured correctly, the pile is rarely perfectly still
mid-round, and you are the one who picked the moment.

So the reading is judged instead of the precondition. A board that shifted
between the baseline and the hold differs from itself *everywhere*, so nearly
every tsum clears the threshold — indistinguishable from marks one tsum at a
time, obvious in the aggregate. Real readings came in at 15% and 32% of the
board; a press caught mid-settle reported 86%; a synthetic 4px shift measures
59%. Hence a 50% default. `--drift` still exists and still refuses up front,
now measured over the board alone, but defaults to 0 (off).

The button is watched globally, because a press that lands on the emulator
sends this process no event to subscribe to. Clicks anywhere else on the
desktop are ignored silently, so leaving `assist` running while you use another
window is harmless.

#### Narrowing what it looks at

Every board reader — `assist`, `play`, `analyze`, `hold` — takes `--board
x,y,w,h`, in pixels relative to the emulator's content area. Without it the
rect comes from the frame size, via the layouts measured in `LAYOUTS`.

**The live layout (994x578) carries two rects, and FEVER decides which.**

```python
"board":       (10, 314, 525, 456),   # normal play
"fever_board": (22, 291, 502, 451),   # FEVER's ~10s
```

**Do not tighten them.** On 2026-08-20 both were re-measured by eye to
`(22,395,507,370)` and `(24,324,497,439)` — rects that frame the board more
neatly on screen — and play got markedly worse. Reverted the same day.

The tight rect fits wholly inside the one above, so the comparison is one
image cropped two ways rather than two captures:

| | the rects above | the tight pair |
|---|---|---|
| 15 in-play frames from that morning | 72 tsums, r 17.7px, 1/15 collapsed | 99 tsums, r 12.3px, **9/15 collapsed** |
| 10 hand-labelled boards, detection | **f1 0.697**, 342 real found, 176 phantoms | f1 0.416, 193 real found, 590 phantoms |

("collapsed" = radius read under 12px.) The mechanism: a rect that slices tsums
at its edge makes them read *smaller* than they are — the distance transform
measures the visible inscribed radius — so the radius estimate drops and one
tsum detects as two or three. The tight rect therefore finds *more* things and
fewer of them are tsums: 44% fewer real ones at triple the phantoms. It also
cuts 114 labelled tsums out of frame entirely.

**A rect wants to be generous.** Overshooting into the bowl costs phantoms.
Undershooting costs the radius estimate, and nothing downstream can recover
from that.

Untested and possibly better: `(8,265,522,535)`, the rect the labelled boards
were marked with, scores f1 **0.775** — better than either. Not adopted,
because those labels were made with that rect so the comparison flatters it,
and because it is larger than any frame collected since, leaving nothing
current to check it against. A/B it with `--verify-clears` before believing it.

FEVER gets its own rect and hands it back afterwards; the pile rides up during
it, which is why the two differ. The 40%-vs-none FEVER radius-collapse figure
quoted in older notes belonged to `8,265,522,535`, a different rect again — the
one flagged as untested above.

An explicit `--board` is never swapped out — someone who passed a rect asked
for that rect.

Narrowing is safe for the skill icon: `read_base_kind` is handed the full frame
rather than the crop, because the button sits below the play area.

Measure it by marking two corners:

```powershell
python main.py region -o scratchpad/region.png
#   Point at the TOP-LEFT corner of the region and press F8
#   Now point at the BOTTOM-RIGHT corner and press F8
#
#   Region: 502x451 at content (22, 291)
#     --board 22,291,502,451
```

Then paste it in:

```powershell
python -m ttheart_sender.game.tsum assist --board 22,291,502,451
```

Corners can be marked in either order or on either diagonal. `-o` saves a crop
of what you selected, which is worth doing: a board rect that is slightly wrong
is not *obviously* wrong — detection still returns tsums, just the wrong set,
and half a tsum sliced by the edge of the rect reads as a smaller tsum and
drags the radius estimate down with it. Look at the crop rather than trusting
the numbers.

`point` gives the same coordinate space one point at a time (`content=(x, y)`)
if you would rather do the subtraction yourself, and `--board full` uses the
whole captured frame.

Two things a narrower region changes, both worth knowing before you shrink it:

- **Marks outside it are invisible.** The game lights up every linkable tsum on
  the board, so cropping does not stop it marking them — it stops `assist`
  *seeing* them, and the chain it draws is that much shorter.
- **Detection gets faster**, since k-means and the distance transform both
  scale with area. That is the real reason to crop: `live -n 20` will tell you
  what you actually bought.

| flag | what it changes |
|---|---|
| `--board x,y,w,h` | restrict every reader to that rect; `region` measures it |
| `--delay 0.25` | wait before reading the marks; 0.15 is the floor |
| `--threshold 8.0` | mean pixel change that counts as a mark |
| `--aura 90` | radius the glow splashes over |
| `--min-chain 3` | below this the cursor does not move |
| `--max-chain 0` | 0 = drag everything marked |
| `--mark-frames 3` | read the marks from N frames; only persistent change counts |
| `--mark-gap 0.05` | seconds between those frames |
| `--max-marked 0.5` | reject a reading where over half the board lit up |
| `--drift 0` | 0 = off; refuse up front on a board moving this much |
| `--no-auto-release` | keep the release yourself instead of it letting go |
| `--dry-run` / `--debug` | report only / save the marked board and raw diff |

#### FEVER

FEVER turns the board's background black and animates it for ~10 seconds, and
detection is visibly worse throughout. Two separate things are going on, and
only one of them is currently fixed.

**The mark reading is fixed.** A two-frame diff cannot tell a mark from
anything else that moved, and during FEVER everything moves — so a single
frame reads the animation as marks. `assist` now samples `--mark-frames`
frames (default 3) spaced `--mark-gap` apart and keeps only what changed in
*all* of them: a mark holds still for the whole hold, a sparkle is somewhere
else by the next frame. Measured over six boards with 45 moving sparkles
painted on:

| frames | real marks kept | false marks |
|---|---|---|
| 1 | 2.67 / 3 | 10.7 |
| 2 | 2.67 / 3 | 0.7 |
| **3** | **2.67 / 3** | **0.0** |

It costs ~110ms of the hold, which is free here for the same reason the
generous `--delay` is. The same trick suppresses a still-settling pile, so it
helps outside FEVER too.

**FEVER is detected, not inferred.** `templates/max_fever.png` is the meter at
full — gold, the instant before FEVER starts. `FeverWatch` matches it on every
frame and opens a ten-second window; everything else asks the window, not the
template, because the gold bar is a *trigger* visible for a moment rather than
a state that lasts.

Two details it needs, both measured on the 151 captured frames:

- **Its own confidence, 0.75.** The three frames that really are the trigger
  score 0.79-0.82 and the next best scores 0.66, so the shipped 0.85 default
  matches *none* of them and anything in 0.70-0.78 separates them cleanly.
- **A ten-second window.** After the cleanest trigger the FEVER glow persists
  for 19 frames at 0.5s spacing — 9.5s, which is the ten seconds FEVER runs.

The window re-arms on every match rather than only the first: the bar sits full
for several frames before it starts draining, and FEVER has not begun counting
down until it does.

A missing `max_fever` template disables the switch and says so, rather than
failing — `templates/` predating it still runs.

**Do not detect FEVER by board darkness.** It was tried and it is wrong: during
FEVER the board is a dark navy *ringed with neon*, but it stays full of
brightly lit tsums, so median lightness barely moves. Frames that do read dark
are mostly skill activations, and real FEVER frames were missed entirely.

To capture more FEVER frames, note that `grab` **drops** frames whose tsum
count looks implausible — during FEVER, precisely the frames worth keeping.
Widen the filter:

```powershell
python -m ttheart_sender.game.tsum grab --prefix fever -n 120 --interval 0.5 `
    --min-tsums 0 --max-tsums 100000
```

One limit worth knowing: the mark is read as a difference, so a glow that
barely changes a tsum's pixels reads as no mark. Keep your hand still while it
draws — a physical mouse movement fights the warped cursor — and note that a
touchscreen cannot work this way at all, since the cursor is what gets moved.

`eval` re-detects every reviewed board, matches detections to ground truth by
position, and reports precision, recall and f1, plus two kind errors the
groups expose: **splits** (one character read as several colour clusters, so
its chain gets broken up) and **merges** (two characters read as one cluster,
so it plays a chain the game rejects).

`--sweep` tries a grid and ranks by f1:

```powershell
python -m ttheart_sender.game.tsum eval --sweep k=8,12,16 --sweep floor_frac=0.35,0.42
```

Repeatable, cartesian, and any `detect` parameter is fair game — `k`, `radius`,
`include_dark`, `dark_l`, `merge`, `heal_frac`, `open_ratio`, `recolour`,
`floor_frac`, `hole_frac`, `scale` — plus four that shrink the board rect
instead: `trim_top`, `trim_bottom`, `trim_left`, `trim_right`, in pixels, for
tuning `LAYOUTS`. A misspelled name is an error, not a silently ignored knob.
Treat a winner from three or four boards with suspicion; `eval` says so itself
when the top two are within noise of each other.

### What it measured (10 boards, Aug 2026)

| | |
|---|---|
| precision | **0.679** — one detection in three is not a tsum |
| recall | **0.878** |
| f1 | **0.765** |
| kind errors | 14 splits, 18 merges across 46 groups |

Three sweeps found no better setting, which is the useful part:

- **`k` and the peak-finding knobs are already optimal.** `k=12`,
  `floor_frac=0.42`, `heal_frac=0.9` won their sweeps. Raising `floor_frac`
  trades recall for precision monotonically — it slides along one curve rather
  than finding a better one.
- **`include_dark` is not the phantom source it looks like.** Turning it off
  removes 20 of 208 false positives and costs 15 real detections. Roughly a
  wash, despite the warning in `detect`'s docstring.
- **Trimming the board rect backfires.** False positives concentrate at the
  bottom of the rect, on the bowl rim, so cropping it out looks obvious — and
  `trim_bottom=27` drops precision to 0.509 and *raises* false positives to
  344. The rect is the region k-means fits its palette on, not a mask: the rim
  in frame gets its own cluster, which the background test then discards. Take
  it out and rim-coloured pixels are reassigned to tsum clusters instead.

So detection is at the ceiling of what this pipeline reaches, and the remaining
errors are structural — phantoms in the gaps between tsums, and "kind = colour
cluster" being unable to separate two characters that share a dominant colour
or hold one two-tone character together.

Two caveats on those numbers. They measure `detect` alone, while the play loop
additionally runs `purity_filter` over chain candidates, so a phantom only
costs a drag if it survives that too — the end-to-end figure is better than
0.679 and is not what this measures. And the ground truth was marked up from
`k=12` detections, so it is biased toward what that configuration finds: a
setting that locates a tsum in a place `k=12` never proposed can only be
credited if a human separately marked the miss.

---

## Did the drag actually clear anything?

`played N chains, cleared M tsums` used to be a claim nobody had checked: `M`
was the length of the chains *proposed*, so a 3-chain the game marked only two
of counted as three cleared while nothing popped at all. The report now says
`dragged`, which is what that number always was, and only prints `cleared`
when something measured it.

`--verify` cannot measure it. That check is the mean difference across the
whole crop against `--change-tol 2.0`, and it answers a different question --
did the emulator see the stroke. A live board passes it on animation alone:
the score counter, the timer, the FEVER meter and a settling pile all move.

`--verify-clears` looks only where the chain was:

```powershell
python -m ttheart_sender.game.tsum play --duration 90 --verify-clears
```

```
  BASE chain of 5  (52 tsums, 96ms)
    popped 5/5 (78/128/35/145/142 vs idle 4)
  #3 chain of 3  (49 tsums, 91ms)
    popped 0/3 (6/5/7 vs idle 5)
```

A tsum that cleared is replaced by whatever falls into its place -- a large
change inside its own disk. One that jiggled is the same face a pixel or two
over. Measured on a real board: a 1-3px whole-board shift moves a disk by
**5-8**, while a tsum actually replaced moves by **35-145**. `--clear-tol 20`
sits in that gap, and every reading prints the board's own idle noise beside
it (the median over the tsums that were *not* dragged) so a round tells you
whether the threshold needs moving.

It costs no extra capture: `--verify` already grabs the frame this reads.

Two failures it separates that used to look identical:

| | what it means | the cure |
|---|---|---|
| board did not change | the emulator missed the stroke | walk slower, `--per-step` |
| board changed, nothing popped | the chain was not one character | do not offer that kind again |

Only the second is reported as `the game would not accept`; it blacklists the
kind and leaves the drag speed alone, because slowing down cannot fix a chain
that was misread rather than mis-delivered.

Still off by default: it is a measurement, and the point of measuring is to
compare a round with it against a round without.

---

## Collecting training data while you play

The ceiling above is the reason this exists: colour cannot separate two
characters that share one, and the fix is a signal colour does not carry. That
needs data, and the game already gives it away — holding a tsum lights up
every tsum it counts as linkable, which is identity *and* reachability, judged
by the only authority that matters. Off by default; switch it on in
`config.yaml`:

```yaml
dataset:
  enabled: true
  dir: dataset            # relative paths land next to the .exe
  per_round: 20           # cap per round
  every: 4                # sample every Nth drag
  quality: 85

  delay: 0.25             # wait for the mark to render before photographing it
  frames: 3               # only what changed in ALL of them counts as a mark
  gap: 0.05
  floor_mult: 8.0         # a mark must beat the board's own noise floor by this much
  max_motion: 12.0        # drop samples taken while the board was still falling

  max_mb: 2048            # stop once the whole dataset folder reaches this
  max_total: 0            # or once it holds this many samples. 0 = no cap
```

or per run, without touching the config:

```powershell
python -m ttheart_sender.game.tsum play --dataset dataset --dataset-limit 20
```

Flow runs (`python main.py play`, `python main.py run resume`) read the
config, not the flags -- a flow can still override per step with
`options: {dataset: some/other/dir}`.

From the tray, tick **Data collection** in the panel. It writes to the same
place and takes effect on the next round, no restart. The box and
`dataset.enabled` name the same switch: a first launch shows whatever
config.yaml says, and after that the box is the one that decides -- unticking
it has to survive a config file that still says `true`.

Each sampled drag writes three things into `dataset/<timestamp>_<pid>/`:

| | |
|---|---|
| `NNNN_before.jpg` | the board crop the chain was chosen from |
| `NNNN_marked.jpg` | the same crop while the game shows what it marked |
| `samples.jsonl` | detections, the chain, what survived, settings in force |

**The label is the difference between the two images**, so nothing needs
annotating by hand — and a character you equip for the first time labels
itself the first time it is pressed, which is exactly the case a fixed
classifier would have to be retrained for.

### Check a collection before you trust it

```powershell
python -m ttheart_sender.game.tsum dataset --dir dataset
```

It reports whether the marks are actually in the frames — how still the board
was at press time, how much of it read as marked, and whether the marked tsums
look like one character — and exits non-zero when the answer is no.

**Run it on the first session, not after the tenth hour.** The first
collection ran all night and produced 5,729 samples with no usable label in
any of them, because the highlight was photographed 0.10s after the press,
before the game had drawn it. The defaults above fix that; this command is how
you confirm it.

They do work: the third collection, 11,537 samples, reads 1.95x on the
appearance test against a 1.3 bar and is worth training on. It also moved
`floor_mult` from 5 to 8 — a tsum reacting at less than 8x its own board's
noise floor turns out to be indistinguishable from the board average, so the
old default spent about a third of every label on noise. Full write-up in
[DATASET-FINDINGS.md](DATASET-FINDINGS.md).

### What it costs

The press is the start of the drag either way, so the only cost is the
`delay` pause and `frames` captures, on sampled drags only. At the defaults
that is ~20 sampled drags a round rather than the ~100 `--verify-hold` pays
for — the throughput trap that sank per-drag verification does not apply at
1-in-4, which is what buys collection a delay long enough for the mark to
actually be there. The stroke itself is untouched unless `--verify-hold` is
*also* on: with only `--dataset`, the marks are recorded and the chain is
dragged exactly as proposed, so the samples describe the bot you actually run.

Disk is the real cost: ~150 KB a sample, so ~3 MB a round at the defaults, or
roughly 300 MB per hundred rounds.

**There is no per-day cap** — nothing rations collection by date, and
`per_round` bounds one round rather than the sum of them. Left running
unattended, the first collection wrote 803 MB and 5,729 samples in thirteen
hours. `max_mb` (2 GB by default) and `max_total` are the budgets that stop
it; they are checked against the whole dataset directory when each round opens
its session folder, so an already-spent budget means that round collects
nothing rather than that it fails. Clear the folder out, or raise the cap, to
resume. Drop `per_round` or `quality` to slow the fill rate instead.

### Sending it on

Zip a session folder. The crops are the board rect only — no account name, no
score, no window chrome — and `samples.jsonl` carries the app version and a
schema number so an old session stays readable after the format moves on.

### Turning a collection into played rounds

```powershell
python -m ttheart_sender.game.tsum dataset   # are the labels real?
python -m ttheart_sender.game.tsum learn     # fit a palette, and score it
```

Everything above this line writes. `learn` is the only thing that reads a
collection back into the bot, and what it produces is one file of colour
centres — `models/palette.json` by default.

**What it fixes.** `Tsum.kind` is a k-means cluster id, and the code says what
that is worth: *"stable within one frame only"*. The fit is re-derived from one
crop and thrown away every time the board is repainted — entering FEVER,
leaving it, a shuffle, a recalibration — so cluster #3 is one character on this
frame and something else two frames later, and nothing can be remembered about
#3 in between. `learn` fits one palette over every frame you have collected and
hands it to `detect`, which has always accepted centres from an earlier frame.
No detection code changes. The id simply stops moving.

**What it does not fix.** Two characters sharing a dominant colour still merge —
that is [the ceiling](#the-ceiling-colour-cannot-tell-characters-apart), and a
stable palette does not add a signal colour never carried. The ~40% drag waste
is unchanged. What goes away is the *drift*: an id that meant Piglet a moment
ago and means Pooh now, and a palette re-fitted on the dimmest frame of the
round.

**The score.** Holding a tsum makes the game mark same-character *and*
reachable, so `head` and everything in `marked` are one character on the game's
own word. `learn` scores a palette on two halves of that, and scores the
per-frame `kind` recorded live on the identical frames the same way:

* **agree** — of the pairs the game confirmed, how many share an id.
* **split** — of the weak negatives, how many are given a *different* id.
* **balanced** — the mean of the two. This is the verdict.

Marks inside the 90px glow are excluded from both. The glow washes over
whatever is under it, so a reaction there means proximity, not identity —
`marked_by_game` says so, and 18% of recorded marks sit inside it.

It refuses to recommend on four grounds:

* **Scoring on frames it was fitted on.** The holdout is by *session*, not by
  sample — samples inside one session are the same board minutes apart under
  the same equipped tsum, so a per-sample split would score the fit against
  frames it effectively saw. One session means no holdout and the verdict is
  `UNPROVEN`.
* **Collapse.** A palette below 50% on `split` is rejected however well it
  agreed.
* **Agreement bought with separation.** This is the subtle one, and the reason
  the verdict is `balanced`. Merging two characters into one id *raises* agree
  and only costs split, so a verdict read off agree alone pays a palette to
  collapse — and it arrives gradually, so no threshold on `split` catches it.
* **No lift.** Under two points of balanced improvement prints `NO BETTER`.

Only on `BETTER` does it print the line to add — one entry under `options:` in
`flows/play.yaml`, deleted to revert:

```yaml
palette: models/palette.json
```

A file that will not load stops the round rather than quietly falling back: a
round that silently ran without the palette it was told to use looks exactly
like a round where the palette did not help, and that is the reading that would
get a working one thrown away.

`-k` does **not** have to match what `play` runs with. `_quantise` ignores its
`k` entirely when a palette is supplied, so the corpus fit is free to use as
many centres as it needs.

### What it said the first time — a global palette did not work

The first real corpus was 303 samples over 21 sessions. Every k from 6 to 96
was fitted and scored against five held-out sessions:

| k | agree | split | balanced |
|---|---|---|---|
| per-frame k-means | 33.7% | 79.3% | **56.5%** |
| 6 | 37.6% | 72.4% | 55.0% |
| 12 | 30.7% | 81.5% | 56.1% |
| 24 | 26.6% | 86.6% | 56.6% |
| 32 | 24.1% | 85.9% | 55.0% |

**Nothing beat the per-frame fit.** The best is +0.1 at k=24, which is noise.

This is worth reading carefully, because the shape of the table is the finding.
Agreement falls monotonically as `k` rises and split rises to meet it: the two
halves trade against each other across the whole range, and *neither one alone
has a maximum that means anything*. Read off agreement, k=6 looks like a
3.9-point win — and it is 1.5 points worse than doing nothing.

The reason a global palette loses is that per-frame k-means is *relative*. It
partitions whatever is on this board into k groups, so the same character lands
together even when its absolute colour has drifted. A global codebook needs
absolute Lab to be stable across frames, and it is not. Measured over 275
labelled groups, the same character's face colours spread 45.6 Lab **within a
single frame** — wider than the 43.9 spread of group means *between* frames.

So this is the [colour ceiling](#the-ceiling-colour-cannot-tell-characters-apart)
again, measured from the other side and against the game's own answer rather
than against a threshold. Colour is not merely exhausted as a way of telling
two characters apart; it is barely coherent within one character. A stable
palette cannot fix that, and neither can more sessions of the same data.

The machinery is kept anyway. It fits, scores, refuses, and is the harness the
next attempt is measured in — and the next attempt should not be another colour
model. `docs/TODO-blob-adjacency.md` names what is: the marked sets are ground
truth for **reachability**, which is colour-independent, and every tsum the game
declines to mark is a negative link example.

---

## Play from a flow

`play_tsum` is a flow action, so a round is a step like any other:

```yaml
- play_tsum:
    duration: 90              # backstop, not the plan
    until_found: scoreboard   # how the round really ends
    require_played: 1         # fail the step if it never got a chain away
    options:                  # any `play` flag, underscored
      min_chain: 4
```

`flows/play.yaml` is a whole round built from it — clear leftover popups, Play,
Start, wait for the in-play UI, play, clear the results:

```powershell
python main.py play              # one round, start to results
python main.py play --loops 0    # keep playing rounds until F12
python main.py run play          # the same thing, the long way
```

Include it in another flow with `- run_flow: play`.

Note the difference between the two `play`s: `python main.py play` runs that
whole flow and starts the round itself, while `python -m
ttheart_sender.game.tsum play` is the bare loop for tuning — it drags chains on
a round you already started and knows nothing about menus.

There is deliberately **no countdown**: the flow waits for `gameplay_footer`
(the FEVER bar and skill button) instead of guessing how long the 3-2-1 takes.

| Parameter | Meaning |
| --- | --- |
| `duration` | Seconds before it gives up waiting for the end template |
| `until_found` | Template that means the round is over (checked before each move); a list stops on any of them |
| `until_gone` | Same, inverted — a template that vanishes when the round ends; a list must *all* be gone |
| `confidence` | Match threshold for those two only |
| `require_played` | Minimum chains for the step to count as successful |
| `options` | Any `play` flag by its long name with `_` for `-` |

Both stop templates are matched against the frame the loop already grabbed, so
watching for the end costs no extra screenshot. It's checked *before* anything
on that frame is clicked — once the round is over the board is gone, and a
"chain" found on the results screen would drag across live buttons.

That check only runs *between* drags, though, so a banner that shows for a
second or two can appear and disappear inside a single drag and never be seen.
Name several endings to get more than one chance at it:

```yaml
- play_tsum:
    duration: 300
    until_found: [timeup, gameplay_clock_zero]   # whichever lands first
```

`until_found` stops on **any** template in the list; `until_gone` needs **all**
of its templates to be gone, so one flaky match on a still-live board can't end
the round early. Each name is one more match per iteration — list the two or
three endings that really occur, not everything that might.

`--dry-run` skips this step entirely: the drag code drives pyautogui directly
for speed, so unlike the other actions it has no no-op backend.

---

## Play options worth knowing

Everything has a default; these are the ones you'd actually reach for.

| Flag | Default | What it's for |
| --- | --- | --- |
| `--duration N` | `0` (one chain) | Seconds to keep playing |
| `--dry-run` | off | Print the drag path, touch nothing |
| `--min-chain N` | `3` | Skip chains shorter than this and re-look |
| `--max-chain N` | `8` | Cap chain length; picks the tightest cluster of N |
| `--link-px N` | `100` | Link distance **in pixels** (see note below) |
| `--per-step S` | `0.004` | Pause per drag step — the main speed knob |
| `--step-px N` | `8` | Cursor step along each leg; larger drops touch events |
| `--base-only` | off | Only play chains of your equipped tsum |
| `--skill ""` | on | Disable auto-firing the skill |
| `--bubble ""` | on | Disable bubble tapping |

### Bubbles need a one-time capture

`--bubble` is a comma-separated list of template names, and defaults to
`bubble,time_bubble,coin_bubble,score_bubble` — the magic bubble plus the time,
coin and score bubbles. Each frame every listed kind is matched and the
strongest hit is tapped, up to four taps per frame, re-grabbing between taps
because bubbles drift.

```powershell
python main.py snip time_bubble     # capture a new kind
python main.py find time_bubble     # score it against the screen right now
```

A name with no `templates/<name>.png` behind it prints a warning at startup and
is skipped; the rest still work. If a bubble is on screen and never gets
tapped, check `find` — it reports the match score, and anything under
`--bubble-confidence` (default `0.80`) is ignored. Templates are matched at
scale 1.0 only, so a crop taken at a different emulator resolution will not
match.

### Why `--link-px` is in pixels

A threshold in "tsum diameters" divides by the detected radius, and that
estimate is not stable — the same board measured 30.6 and 24.0 on different
runs, which moves the cutoff 27% and flips real links between accepted and
rejected. In raw pixels the same data is tight (median 74px, p90 95px). The
emulator runs at a fixed resolution, so pixels are the stable unit.

`--link-px 0` falls back to the old diameter-relative `--link`.

---

## When it gets stuck

`play` taps the fan/shuffle button and recalibrates on four conditions, so a
stuck board recovers on its own rather than looping:

| Trigger | Default |
| --- | --- |
| No playable chain N times running | `--max-misses 6` |
| Drags that don't change the board | `--max-stalls 4` |
| The same exact chain repeating | `--max-repeats 3` |
| The same chain *length* repeating | `--repeat-len 5` |

Recalibrating throws away the cached colour palette, radius and base tsum — the
same state a fresh process starts with, which is why restarting the command by
hand used to un-stick it.

If it's still stuck, in rough order of likelihood:

- **Something is covering LDPlayer.** Captures are screen-region reads, not
  window reads, so an overlapping window gets detected as the board. Keep the
  emulator in front.
- **Drags aren't registering.** Lower `--step-px` to 5, then raise `--per-step`.
- **Detection looks wrong.** Run `analyze --debug-dir dbg` on a fresh shot and
  check `dbg/clusters.png`; each character should be one flat colour.
