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
 ttheart-sender v1.4.1   <- the caption says which build this is
[x] Always on top
Mode
 (o) Resume  ( ) Launch  ( ) Play (beta)
[ ] Auto Play                             <- play a round between cycles
────────────────────────
[x] Return Heart                          <- send hearts on the clock
     Every hour at        [ 15 |v| ] min
     and at               [ 50 |v| ] min
────────────────────────
Claim pattern                             <- how the mailbox is emptied
 (o) Single claim   ( ) Claim all
────────────────────────
Purchase box          [   Buy tsum   ]  <- runs purchase_box with the ticks
[ ] Premium Box+
[x] Premium Box
[x] Pick-up Capsule
[x] Happiness Box
────────────────────────
[ Stop - Running: ... ]  <- Run when idle, Stop while a run is going
[ Open logs ] [ Exit  ]
────────────────────────
[x] Auto Update       [    Update    ]  <- what the check found, and the
     v1.7.0 available                      button that acts on it
```

The panel opens by itself at launch, so there is nothing to discover on a first
run; after that, left-clicking the icon puts it away and brings it back.
Right-clicking the icon offers **Exit** and nothing else — a way out that does
not depend on finding the panel first, which is also why **Exit** and **Open
logs** are repeated on the panel itself.

Every choice is written to `ttheart-settings.json` next to the logs and reloaded
at the next launch, so the panel opens the way you left it. Delete that file to
go back to the defaults shown above. `--mode` and `--play` still work on the
command line and win over the saved file for that session.

**Auto Play** decides what a Resume/Launch cycle does between passes, and it
starts **unticked**: the cycle claims gifts, sends hearts and pauses 30–90s
between them. Tick it and every cycle plays a round *instead of* that pause. It
is all or nothing — the tray passes `play_chance_percent: 100` when it is
ticked and `0` when it is not.

The flows keep the finer grain for console runs: `run resume --var
play_chance_percent=25` plays a quarter of the time, and the `vars:` in
[`flows/resume.yaml`](flows/resume.yaml) and
[`flows/launch.yaml`](flows/launch.yaml) are what a console run with no `--var`
uses. Start the tray with `--play` to have the box ticked from the outset. (The
dedicated **Play** mode is unaffected — it plays rounds outright.)

**Return Heart** puts the heart-sending pass on the clock instead of running it
every cycle. Unticked — the default — nothing changes: every cycle works
through the ranking list. Tick it and the two spinners become minutes of the
hour, so `15` and `50` send at 9:15, 9:50, 10:15, 10:50 and so on. In between,
a cycle still claims the mailbox and plays or waits; it just goes home instead
of opening the rankings.

A mark that went by *before* the run started counts as already handled: start
the bot at 9:20 and the first send is at 9:50, not straight away. Stopping and
restarting therefore cannot squeeze in extra passes. Once running the gate does
catch up — a round long enough to step over 9:50 still sends on the far side of
it, once. Set both spinners to the same minute for one send an hour. From a
console it is `run resume --var return_heart_timed=true --var
return_heart_minutes=15,50`.

**Claim pattern** decides how a cycle empties the mailbox, and the two ways are
mutually exclusive — whichever is selected, the other never runs.

* **Single claim** (the default) taps **Check** on one item at a time and keeps
  going until the mailbox has none left. Slower, and the pattern that returns a
  heart per request.
* **Claim all** makes one pass through the game's own **Claim All** dialog with
  hearts and medals toggled on, then dismisses whatever it reports. Faster, and
  it takes the mailbox in one go, so nothing is checked item by item.

Both live in [`flows/claim_mailbox.yaml`](flows/claim_mailbox.yaml) as the two
branches of one switch, so `launch` and `resume` claim the same way. From a
console it is `run resume --var claim_all=true`.

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

## Updates

New builds are published as a single `.exe` on the repository's
[Releases](https://github.com/iDunoPlay/ttheart-sender/releases) page, and the
panel watches for them: 20 seconds after launch and every six hours after that,
it asks GitHub for the newest release and puts the answer on the line under
**Auto Update** — `v1.6.0 - up to date`, `v1.7.0 available`, or why it could not
ask. The button beside the box acts on whatever it found: **Check** now,
**Update** to install, **Retry** after a failure.

**Auto Update** — ticked on a fresh install — decides whether finding a build is
enough to install it. The check itself runs either way, so unticking the box
does not blind the panel; it only stops it acting by itself. Either way an
update **never interrupts a run**: while a flow is going the install waits, and
the panel keeps saying the version is available until you press **Stop**.

Installing swaps the running program out from under itself, which on Windows
means renaming rather than overwriting:

1. the new `.exe` is downloaded beside the old one as `ttheart-sender.exe.new`,
   and thrown away unless it arrives at the exact size the release advertises;
2. the running build is renamed to `ttheart-sender.exe.old` — allowed even
   while it is executing — and the download takes its name;
3. a small `ttheart-sender-update.cmd` is launched, the tray exits, and the
   script deletes the old build (which only succeeds once the process is
   really gone), starts the new one and deletes itself.

That script is started with PyInstaller's `_PYI_*` handoff variables stripped
from the environment, and clears them again itself. A one-file build runs as a
child of its own bootloader, which uses those variables to say "the archive is
already unpacked, it is in this temp folder". Inherited by the replacement
`.exe` they make it skip unpacking and load `python311.dll` out of the *dying*
process's `_MEIxxxxxx` directory, moments before it is deleted — the update
lands correctly and then the restart dies with *Failed to load Python DLL*.

`config.yaml`, `flows/` and `templates/` beside the `.exe` are untouched, as is
`ttheart-settings.json` — the panel comes back the way you left it. Anything
left behind by an interrupted swap is cleaned up at the next launch.

Two installs cannot update themselves this way and say so on the status line: a
**source checkout** (`git pull` instead) and the **folder build**, whose `.exe`
needs the matching `_internal\` beside it. For those the button becomes
**Open** and takes you to the release page.

Turn the whole thing off — or point it at a fork — in `config.yaml`:

```yaml
update:
  enabled: false             # never ask GitHub anything
  repo: someone/their-fork   # empty = this project's own repo
  check_interval_hours: 6
  include_prereleases: false
```

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
`chance`, `time_gate`, `repeat`,
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
