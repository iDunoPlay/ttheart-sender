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
the mailbox, sends hearts, and (if `play_chance_percent` says so) breaks off to
play a round. `resume` on its own skips the launch and assumes the game is
already up.

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
| `until_found` | Template that means the round is over (checked before each move) |
| `until_gone` | Same, inverted — a template that vanishes when the round ends |
| `confidence` | Match threshold for those two only |
| `require_played` | Minimum chains for the step to count as successful |
| `options` | Any `play` flag by its long name with `_` for `-` |

Both stop templates are matched against the frame the loop already grabbed, so
watching for the end costs no extra screenshot. It's checked *before* anything
on that frame is clicked — once the round is over the board is gone, and a
"chain" found on the results screen would drag across live buttons.

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
