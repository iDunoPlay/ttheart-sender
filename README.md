# ttheart-sender

Screenshot-driven click automation for Windows, built around LDPlayer.

You take a cropped screenshot of a button once. From then on the app finds that
button on screen and clicks it — no fixed coordinates that break when a menu
shifts. Routines are written as small YAML files, so adding a new one never
means touching Python.

Windows only (it uses Win32 window handles and real mouse events).

---

## Install

```powershell
cd C:\GitHub\ttheart-sender
py -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
```

Run everything through `python main.py <command>` (or `.\.venv\Scripts\python main.py ...`).

## First run

```powershell
python main.py detect      # is LDPlayer found?
python main.py prepare     # park it at the top-left of the screen and focus it
python main.py shot        # screenshot the emulator, so you can see what the app sees
```

`detect` should print something like:

```
[0] hwnd=0x00031658 pid=38356 class='LDPlayerMainFrame' title='LDPlayer' rect=[x=0 y=0 w=580 h=995] ... <- window.instance
```

If it finds nothing, run `python main.py windows`, locate LDPlayer in the list,
and copy its `class=` value into `window.target.class_names` in `config.yaml`.

## Building a routine

**1. Capture the buttons you want to click.**

```powershell
python main.py snip heart_button
```

Point the mouse at the top-left corner of the button, press **F8**; point at the
bottom-right corner, press **F8** again. The crop is saved to
`templates/heart_button.png`.

**2. Check it matches.**

```powershell
python main.py find heart_button
#   conf=0.998 screen=(210, 200) content=(209, 200) scale=1
```

If it misses, the command still prints the best similarity it saw, so you know
whether to lower `matching.confidence` or re-crop.

**3. Write the flow** — `flows/send_heart.yaml`:

```yaml
name: send_heart
vars:
  taps: 10
steps:
  - prepare_window
  - repeat:
      times: ${taps}
      steps:
        - find_click: {template: heart_button, timeout: 5}
        - wait: {min: 0.4, max: 0.9}
        - find_click: {template: confirm_button, timeout: 1}
          optional: true
```

**4. Run it.**

```powershell
python main.py run send_heart --dry-run    # log the clicks without sending them
python main.py run send_heart              # for real
python main.py run send_heart --var taps=50 --loops 3
```

**Press F12 at any time to stop.** Slamming the cursor into a screen corner
also aborts (pyautogui's fail-safe).

## Commands

| Command | What it does |
| --- | --- |
| `windows [--filter X]` | List every open window with its class name |
| `detect` | Show the emulator windows matching your config |
| `prepare` | Restore, move to the top-left, focus; print the coordinate origin |
| `snip NAME` | Capture a template image between two F8 presses |
| `point` | Hover the mouse and press F8 to print its screen/content coordinates (ESC to stop) |
| `shot [-o FILE] [--screen] [--region x,y,w,h]` | Save a screenshot |
| `find TEMPLATE [--all] [--click] [--save]` | Test-match a template right now |
| `templates` | List template images and their sizes |
| `flows` / `validate [FLOW]` | List flows / parse them without running |
| `actions` | List every action the flow language supports |
| `run FLOW [--var K=V] [--loops N] [--dry-run]` | Run a flow |
| `tray [--mode M] [--start] [--play]` | Sit in the system tray; left-click for the panel |

Global flags: `-c/--config`, `-v/--verbose`, `-q/--quiet`.

## Tray mode

For day-to-day use there is no need for a console window. `python main.py tray`
(or the built `.exe`) puts a heart icon next to the clock — grey when idle, red
while a flow is running.

**Left-click it** and the control panel opens in the bottom-right corner of the
screen, above the taskbar:

```
ttheart-sender v0.1.0   <- which build this is
[x] Always on top
Mode
 (o) Resume  ( ) Launch  ( ) Play (beta)
[ ] Auto Play
────────────────────────
Purchase box
[ ] Premium box plus
[x] Premium box
[x] Pick-up capsule
[x] Happiness capsule
────────────────────────
[      Buy tsum       ]  <- runs purchase_box with the ticks above
[ Stop - Running: ... ]  <- Run when idle, Stop while a run is going
[ Open logs ] [ Exit  ]
```

Right-clicking the icon does nothing: the panel replaced the old context menu,
which is why **Exit** and **Open logs** are on it — they are the only way out
of the app.

Every choice is written to `ttheart-settings.json` next to the logs and reloaded
at the next launch, so the panel opens the way you left it. Delete that file to
go back to the defaults shown above. `--mode` and `--play` still work on the
command line and win over the saved file for that session.

**Auto Play** is the on/off switch for the dice roll in the Resume/Launch
cycle, and it starts **unticked**: the tray passes `play_chance_percent: 0`, so
the cycle only claims gifts and sends hearts. Tick it and the flow's own
`play_chance_percent` applies instead — 8% per cycle as shipped. Change those
odds in `vars:` at the top of [`flows/resume.yaml`](flows/resume.yaml) and
[`flows/launch.yaml`](flows/launch.yaml), or per run with
`run resume --var play_chance_percent=25`. Start the tray with `--play` to have
it ticked from the outset.
(The dedicated **Play** mode is unaffected — it plays rounds outright.)

A console `run resume` has no panel to ask, so it uses the flow's 8% as written.

Picking a mode or toggling a tick only *selects* it; **Run** is what launches
it, and a run keeps whatever it started with. **Buy tsum** greys out while
anything is running, since one emulator cannot serve two flows. **Stop** does
exactly what pressing **F12** does — both set the same flag, which every flow
checks between steps and during waits, so a run stops within a second or two
even in the middle of a long pause. A balloon tells you how each run ended.

### Stopping a run

Three things stop a run, all through the same switch:

* **F12** anywhere, even while the emulator has focus.
* The panel's **Stop** button.
* **Moving the mouse out of the emulator window.** The flow drives the cursor,
  so a cursor outside LDPlayer means either you have taken the mouse back or a
  click went somewhere it should not have. The guard arms the first time it
  sees the cursor inside the window — a run started from another monitor is
  not stopped before its first click — and `runner.cursor_exit_margin` (16px)
  keeps an edge click from counting as an escape. Turn it off with
  `runner.stop_on_cursor_exit: false` in `config.yaml`.

Only one tray instance runs at a time; launching a second one exits immediately
rather than having two bots fight over the same emulator.

Modes live in [`ttheart_sender/tray/modes.py`](ttheart_sender/tray/modes.py) —
adding one is a single `Mode(...)` entry naming a flow in `flows/`.

A fuller command-by-command walkthrough — including the Tsum Tsum board reader
and its calibration loop — lives in [`docs/COMMANDS.md`](docs/COMMANDS.md).

## Version

The number lives in one file,
[`ttheart_sender/version.py`](ttheart_sender/version.py). Everything reads it
from there — `main.py --version`, the top row of the tray menu, the icon
tooltip, the first line of every tray log, and `pyproject.toml` (via
`[tool.setuptools.dynamic]`). Releasing is a one-line edit; nothing else needs
touching.

## Building the .exe

```powershell
.\.venv\Scripts\pip install pyinstaller
.\.venv\Scripts\python build.py
```

Double-clicking the result goes straight to the tray — no console window.

| Flag | Output | Notes |
| --- | --- | --- |
| *(none)* | `dist\ttheart-sender\` | Ship the **whole folder**; the `.exe` needs `_internal\`. Starts in well under a second |
| `--onefile` | `dist\ttheart-sender-portable\ttheart-sender.exe` | **One file, nothing else needed.** ~72 MB, ~4–5 s to start (it unpacks itself each launch) |
| `--console` | either | Keep a console window so log output is visible while debugging |
| `--with-data` | `--onefile` only | Also emit editable `config.yaml` / `flows\` / `templates\` beside the `.exe` |

`config.yaml`, `flows\` and `templates\` are **always baked into the `.exe`**, so
the one-file build runs from an empty folder with no data files at all.

The folder build additionally drops editable copies next to the `.exe`, and
those copies take priority when present — so you can retune a flow or re-snip a
template without rebuilding. Delete them and the built-in copies take over
again. That override works for the one-file build too: drop a `config.yaml`
beside it (add `flows\` and `templates\` if you want to change those as well).

Logs always land in `logs\` next to the `.exe`, never inside the temporary
unpack directory.

To start it with Windows, put a shortcut to the `.exe` in
`shell:startup` — add `--start` to its Target if you want it running on boot.

The icons are generated, not hand-drawn: `python scripts/make_icons.py` rewrites
`ttheart_sender/tray/assets/*.ico` if you change the artwork.

## Coordinates

`prepare` parks LDPlayer at screen (0, 0) so window and screen coordinates
almost coincide, then everything in a flow is expressed **relative to the
emulator's content area**:

```yaml
- click: [270, 880]              # content-relative (default)
- click: {at: [1500, 40], space: screen}   # absolute desktop pixels
```

`find`/`prepare` print both, so you can read a coordinate off a screenshot and
use it directly.

LDPlayer's own title bar and right-hand toolbar are inside the client area. To
exclude them from searches (faster, fewer false matches), set
`window.insets` in `config.yaml` — measured values for LDPlayer 9.5 at 578×994
are `top: 35, right: 42`.

## Flow language

Steps take either form, and can be mixed:

```yaml
- find_click: {template: heart}          # shorthand
- action: find_click                     # explicit
  with: {template: heart}
```

Any step accepts these keys:

| Key | Meaning |
| --- | --- |
| `name` | Label used in logs |
| `optional` | A failure logs a warning instead of failing the run |
| `retries` / `retry_delay` | Re-run the step N more times before giving up |
| `enabled: false` | Skip it without deleting it |

Any step that searches accepts `template`, `timeout`, `poll_interval`,
`confidence`, `scales`, and `region` (`[x, y, w, h]` relative to the content
area, or `screen`).

`repeat` also takes `stop_when_still: true`, which ends the loop as soon as an
iteration leaves `region` unchanged. It is what a scrolling loop wants: a list
clamped at its top or bottom returns the identical picture every pass, so
without it the loop spends its whole `max_iterations` budget on scrolls that
cannot move anything. Give the `delay` enough room for the app's own settle
animation, or every frame lands mid-bounce and nothing ever looks still.

Actions: `find`, `find_click` (`tap`), `click_all`, `wait_for`,
`wait_until_gone`, `click`, `move`, `drag`, `scroll`, `key`, `hotkey`, `type`,
`wait`, `log`, `set`, `add` (`incr`), `screenshot`, `if` (`when`), `if_found`,
`chance`, `repeat`,
`while_found`,
`run_flow`, `prepare_window`, `stop`. Run `python main.py actions` for the
one-line summary of each, and read [flows/example.yaml](flows/example.yaml) for
an annotated tour of all of them.

Variables interpolate as `${name}` anywhere in a flow, and come from `vars:`,
`--var k=v`, `set:`, `save_as:`, and loop counters.

> Inside `{braces}`, quote them — `max: "${gap}"` — because YAML reads a bare
> `{` as the start of a nested mapping.

## Configuration

Everything lives in [config.yaml](config.yaml), fully commented. For settings
specific to one machine, create `config.local.yaml` (git-ignored); it is deep
merged on top. Unknown keys are rejected at startup rather than ignored, so a
typo fails loudly.

The knobs you will actually touch:

- `matching.confidence` — raise for fewer false hits, lower if matches are missed
- `matching.scales` — add `[0.9, 1.0, 1.1]` if the emulator gets resized
- `window.insets` — crop LDPlayer's chrome out of the search area
- `input.*` — click timing and jitter
- `runner.stop_key` — the panic key (default F12)

## Troubleshooting

| Symptom | Cause |
| --- | --- |
| "No emulator window found" | Wrong `class_names`; check `python main.py windows` |
| Template never matches | Emulator resolution changed since you snipped it — re-snip, or add `scales` |
| Screenshot shows the wrong app | Something is covering LDPlayer; capture reads real screen pixels |
| Clicks land slightly off | Display scaling — the app sets per-monitor DPI awareness, but re-run `prepare` after changing scaling |
| Clicks land on the wrong monitor | Keep LDPlayer on the **primary** monitor; pyautogui addresses that coordinate space |
| "is minimized, so there are no pixels to read" | Restore LDPlayer, or run `prepare` first |

## Extending it

The design keeps the four moving parts independent, so most changes are local:

| Want to… | Touch |
| --- | --- |
| Add a step type to the YAML language | `ttheart_sender/automation/actions.py` — one `@action`-decorated function |
| Support another emulator | `config.yaml` `window.target` (no code) |
| Change how clicks are sent (raw SendInput, ADB taps) | Add a class implementing `MouseController` in `control/mouse.py` |
| Change matching (feature matching, OCR) | `screen/matcher.py`; `TemplateMatcher` is the only consumer |
| Drive several instances at once | `WindowManager.find_all_controllers()` already returns them; build one `RunContext` per window |
| Add a GUI | Drive `Application` from `app.py`; the CLI is a thin layer over it |

Adding an action is the common case:

```python
@action("swipe_up", primary="distance", summary="Swipe upward from the centre")
def act_swipe_up(ctx: RunContext, params: Params) -> ActionResult:
    distance = params.integer("distance", 400)
    centre = ctx.require_content_rect().center
    ctx.mouse.drag(centre, centre.offset(dy=-distance), duration=0.4)
    return ActionResult.ok(distance)
```

It is immediately usable as `- swipe_up: 500`, appears in `main.py actions`,
and gets parameter validation for free.

### Layout

```
ttheart_sender/
  app.py             assembles everything; the entry point for CLI or GUI
  cli.py             argparse front end
  config.py          typed config, YAML loading, validation
  geometry.py        Point / Rect / Insets
  window/manager.py  window discovery, positioning, focus (Win32)
  screen/            capture (mss), template library, matcher (OpenCV)
  control/           mouse, keyboard, stop key
  automation/        registry, flow parser, params, context, actions, runner
  tray/              system tray icon, menu, mode definitions, run service
build.py             PyInstaller build (see "Building the .exe")
ttheart_tray.py      windowed entry point the .exe is built from
```

## Tests

```powershell
.\.venv\Scripts\python -m pytest -q
```

83 tests covering config, geometry, flow parsing, matching, the runner and the
tray. They run headless — no emulator or screen access required.
