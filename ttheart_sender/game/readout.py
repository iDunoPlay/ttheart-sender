"""What the character model is reading, live, for the tray to show.

WHERE IT IS SHOWN. The "Show live recognition" tick box in the control panel's
Experiments section opens a window of its own -- see `tray/recognition.py`.
This list was briefly squeezed into the control panel itself and that was
wrong twice over: the panel is a fixed-height window of clickable controls, and
the space it could spare capped the list at five names, which drops the long
tail. The tail is the part worth reading.

WHY THIS EXISTS. The model reports itself twice, both times into the log: one
line when it loads and one when the round is over. Neither is visible while a
round is being watched, and the question a person actually has -- *is it naming
the right things* -- is asked WHILE looking at the board, by comparing what the
panel says against what is plainly on screen. Two evenings were spent doubting
a model that was working, because 19% named looks identical to 0% named from
the outside.

SO THE PANEL SHOWS TWO THINGS, and they answer different halves of it:

* **The names, live.** Whatever the model is calling the board, most of it
  first. This is the half a person checks by eye -- the board is right there,
  and a list that says Beast when the screen is full of Dory is wrong in a way
  no percentage communicates. Every character, uncapped: the window has room.
* **The headline: agreement with the game's own marks.** Holding a tsum makes
  the game light up everything that is both the same character and reachable,
  so every sampled press hands back a free label for one whole group. It is the
  only ground truth a live round produces.

WHY THE HEADLINE'S COUNT GROWS SLOWLY. Marks are read on a SAMPLED press, one
in four. The game marks a mean 6.1 tsums per press, of which only the ones the
model actually named can be scored -- about a fifth. That is roughly one check
every four chains, so a round of ~95 chains ends near 25 and the first minute
shows single figures. "100% (2 of 2)" is two checks, not a verdict; the count
is printed beside the percentage for exactly that reason.

WHAT THE SCORE IS NOT. The group's name comes from the model's own reading of
the pressed tsum, so this measures whether the model agrees with ITSELF across
a set the game says is one character. A model that calls the whole group Dory
when it is really Beast scores 100%. It still catches the failure that costs a
round -- one character read as several, which is what splits a chain -- and
`scripts/board_check.py` is where the absolute number lives.

THREAD NOTE. `publish` is called on the automation worker thread and `latest`
on the GUI thread, so both take the lock and the watcher callback runs on the
publisher's thread. The tray's callback does nothing but post to its own
message loop, which is the only safe thing a foreign thread may do to a window.
"""

from __future__ import annotations

import logging
import threading
from typing import Callable, List, Mapping, Sequence, Tuple

log = logging.getLogger(__name__)

#: Said when the model is not loaded at all. A distinct line on purpose --
#: "off" and "on but naming nothing" are the two states that look the same
#: from outside the log, and telling them apart is the point of the panel.
OFF = "character model off"

_LOCK = threading.RLock()
_LINES: Tuple[str, ...] = ()
_WATCHERS: List[Callable[[], None]] = []


def headline(agreed: int, checked: int, named: int, total: int) -> str:
    """The one line that says whether to trust the list under it.

    The raw count travels WITH the percentage and never without it. "100%" on
    two checks is not a claim about the model, and a percentage on its own is
    read as one.
    """
    if checked > 0:
        agreed = max(0, min(int(agreed), int(checked)))
        return (f"matches the game {agreed}/{checked} "
                f"({agreed / checked:.0%})")
    if total > 0:
        # Marks need a sampled press and the first is a few chains in. Saying
        # so beats an empty headline, which reads as a broken readout.
        return f"naming {named} of {total} -- no marks yet"
    return ""


def panel(counts: Mapping[str, int] | None, agreed: int = 0, checked: int = 0,
          named: int = 0, total: int = 0) -> Tuple[str, ...]:
    """The whole readout, UNCAPPED: the headline, then every character.

    Published in full because two readers want different amounts of it -- the
    control panel has room for a handful and the "Show live recognition"
    window has room for all of it. Capping here would mean publishing twice,
    or the window showing a list the panel had already truncated.
    """
    top = headline(agreed, checked, named, total)
    if not top:
        return ()
    # Count first, then name, so a tie does not reorder itself between
    # refreshes. A list that reshuffles while it is being read is noise.
    rows = sorted((counts or {}).items(), key=lambda kv: (-kv[1], kv[0]))
    width = max((len(n) for n, _ in rows), default=0)
    lines = [top] + [f"{name:<{width}}   {n}" for name, n in rows]
    if not rows:
        lines.append("nothing named above the floors")
    return tuple(lines)


def publish(lines: Sequence[str]) -> None:
    """Replace what the panel shows, and wake whoever is watching.

    Unchanged lines wake nobody. The board is re-read sixty times a round and
    most frames name it exactly as the last one did, so without this the panel
    would repaint itself for no reason through the whole round.
    """
    global _LINES
    fresh = tuple(str(line) for line in lines)
    with _LOCK:
        if fresh == _LINES:
            return
        _LINES = fresh
        watchers = list(_WATCHERS)
    for callback in watchers:
        try:
            callback()
        except Exception:  # noqa: BLE001 - a panel must never stop a round
            log.debug("recognition watcher failed", exc_info=True)


def clear() -> None:
    """Forget the last round. Called when a round STARTS, not when it ends.

    Clearing at the end would blank the panel the moment the round finishes,
    which is exactly when a person looks at it. The reading stays up until the
    next round has something of its own to say.
    """
    publish(())


def latest() -> Tuple[str, ...]:
    with _LOCK:
        return _LINES


def watch(callback: Callable[[], None]) -> None:
    """Ask to be told when the lines change. Registered once, never removed."""
    with _LOCK:
        if callback not in _WATCHERS:
            _WATCHERS.append(callback)


def unwatch(callback: Callable[[], None]) -> None:
    with _LOCK:
        if callback in _WATCHERS:
            _WATCHERS.remove(callback)
