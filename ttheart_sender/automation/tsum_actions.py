"""Flow actions that play the game, rather than drive its menus.

Kept apart from :mod:`.actions` because everything here is Tsum Tsum specific:
the generic actions know about templates and coordinates, these know what a
tsum is. The play loop itself lives in :mod:`ttheart_sender.game.tsum` -- this
module is the adapter between a YAML step and that loop.
"""

from __future__ import annotations

import logging
from typing import Optional

from ..exceptions import ActionError
from ..game.tsum import Driver, play_defaults, play_loop
from .context import RunContext
from .params import Params
from .registry import ActionResult, action

log = logging.getLogger(__name__)

#: Read straight off the play loop's option set, so `options:` accepts exactly
#: the knobs the CLI exposes and typos are rejected instead of ignored.
_TUNABLES = frozenset(vars(play_defaults()))

#: Set by this action rather than by the flow: `duration` and the stop
#: templates have dedicated parameters, and the rest make no sense here.
_RESERVED = frozenset({"duration", "countdown", "dry_run", "no_prepare"})


def _stop_checker(ctx: RunContext, until_found: Optional[str], until_gone: Optional[str],
                  confidence: Optional[float]):
    """Build the "is the round over?" test the play loop calls each frame.

    Matching happens on the frame the loop already grabbed rather than through
    ``ctx.find``, which would take a second screenshot of the same pixels a few
    milliseconds later -- between chains that is pure latency, and the loop
    runs this every iteration.
    """
    if not until_found and not until_gone:
        return None

    found = ctx.templates.get(until_found) if until_found else None
    gone = ctx.templates.get(until_gone) if until_gone else None

    def check(frame) -> str:
        if found is not None and ctx.matcher.find(frame, found, confidence=confidence):
            return f"{until_found} appeared -- round over"
        if gone is not None and ctx.matcher.find(frame, gone, confidence=confidence) is None:
            return f"{until_gone} is gone -- round over"
        return ""

    return check


@action("play_tsum", primary="duration", summary="Play the board: read tsums, drag chains")
def act_play_tsum(ctx: RunContext, params: Params) -> ActionResult:
    """Play the board until the round ends.

    The round normally ends on a template (``until_found: scoreboard``), with
    ``duration`` as the backstop for when that template never shows up. Unlike
    the CLI there is no countdown: a flow only reaches this step once it has
    already waited for the match to start.
    """
    duration = params.number("duration", 0.0)
    until_found = params.optional_string("until_found", None)
    until_gone = params.optional_string("until_gone", None)
    confidence = params.optional_number("confidence", None)
    require_played = params.integer("require_played", 0)
    overrides = params.mapping("options", {})

    if duration <= 0 and not until_found and not until_gone:
        raise ActionError(
            f"{params.step.location}: play_tsum needs a stopping condition -- "
            f"set 'duration', 'until_found' or 'until_gone'"
        )

    unknown = sorted(set(overrides) - _TUNABLES)
    if unknown:
        raise ActionError(
            f"{params.step.location}: unknown play option(s) {unknown}. "
            f"Valid options: {sorted(_TUNABLES - _RESERVED)}"
        )
    reserved = sorted(set(overrides) & _RESERVED)
    if reserved:
        raise ActionError(
            f"{params.step.location}: {reserved} cannot be set under 'options' -- "
            f"use the step's own parameters"
        )

    if ctx.dry_run:
        # The loop drives pyautogui directly for drag speed, so it has no
        # no-op backend to fall back on the way ctx.mouse does.
        log.info("%s[dry-run] play_tsum (skipped)", ctx.indent)
        return ActionResult.ok(0)

    opts = play_defaults()
    for key, value in overrides.items():
        setattr(opts, key, value)
    opts.duration = duration
    opts.countdown = 0.0
    opts.dry_run = False

    drv = Driver.from_context(ctx, say=lambda msg: log.info("%s%s", ctx.indent, msg))
    report = play_loop(drv, opts, stop_when=_stop_checker(ctx, until_found, until_gone, confidence))

    ctx.set_var("tsum_chains_played", report.played)
    ctx.set_var("tsum_tsums_cleared", report.cleared)
    ctx.set_var("tsum_drags_stalled", report.stalled)
    ctx.set_var("tsum_stop_reason", report.reason)
    log.info("%splay_tsum -> %s, %s", ctx.indent, report.describe(),
             report.reason or "no reason given")

    # A round that never got a single chain away means the loop was looking at
    # something that was not a board -- worth failing on, so the flow's retry
    # or optional handling can deal with it instead of silently moving on.
    if report.played < require_played:
        return ActionResult.fail(
            f"played {report.played} chain(s), expected at least {require_played}",
            report.played,
        )
    return ActionResult.ok(report.played)
