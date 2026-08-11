# Commands

Quick reference for the two things this repo does: sending hearts (the flow
runner, `main.py`) and playing Tsum Tsum (the board reader, now
`ttheart_sender/game/tsum.py`).

The board reader moved out of `scratchpad/` so flows could use it — see [Play
from a flow](#play-from-a-flow). Every command and flag is unchanged; reach it
as `python -m ttheart_sender.game.tsum <anything>`.

`scratchpad/` is git-ignored and holds only the labelled boards `score` reads
(`board2/3/4.png` + their `.label.json`). Everything else in there is
regenerable debug output — safe to clear whenever it gets big.

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
python -m ttheart_sender.game.tsum label board.png # click tsums to record a chain you'd drag
python -m ttheart_sender.game.tsum score           # what your labels imply about the link rule
```

`label` + `score` are the calibration loop: every pair you click is one example
of a link the game accepts, and `score` reads the threshold off them rather than
someone guessing it.

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
