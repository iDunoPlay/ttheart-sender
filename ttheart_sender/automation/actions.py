"""Built-in flow actions.

Each action is a function taking ``(ctx, params)`` and returning an
:class:`~.registry.ActionResult`. Register new ones with the ``@action``
decorator -- see :mod:`.registry`.

Conventions followed by every action here:

* Coordinates default to the emulator content area; pass ``space: screen`` for
  absolute desktop coordinates.
* Anything that searches accepts ``template``, ``region``, ``confidence``,
  ``scales``, ``timeout`` and ``poll_interval``.
* A "not found" outcome is a soft failure (``ActionResult.fail``), not an
  exception, so ``optional: true`` and ``retries:`` can handle it.
"""

from __future__ import annotations

import logging
import random
import re
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ..exceptions import ActionError, FlowAborted, WindowNotFoundError
from ..geometry import Point
from ..window.manager import WindowManager
from .context import RunContext
from .params import Params
from .registry import ActionResult, action

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Timing / bookkeeping
# --------------------------------------------------------------------------
@action("wait", aliases=("sleep",), primary="seconds", summary="Pause for N seconds")
def act_wait(ctx: RunContext, params: Params) -> ActionResult:
    # `wait: 0.5`, `wait: {seconds: 0.5}` and `wait: {min: 0.3, max: 0.9}`.
    if params.has("min") or params.has("max"):
        low = params.number("min", 0.0)
        high = params.number("max", low)
        seconds = random.uniform(min(low, high), max(low, high))
    else:
        seconds = params.duration("seconds", 1.0)
    log.info("%swait %.2fs", ctx.indent, seconds)
    ctx.sleep(seconds)
    return ActionResult.ok(seconds)


@action("log", primary="message", summary="Write a message to the log")
def act_log(ctx: RunContext, params: Params) -> ActionResult:
    message = params.string("message", "")
    level = params.string("level", "info").lower()
    getattr(log, level if level in {"debug", "info", "warning", "error"} else "info")(
        "%s%s", ctx.indent, message
    )
    return ActionResult.ok(message)


@action("set", summary="Assign flow variables")
def act_set(ctx: RunContext, params: Params) -> ActionResult:
    # Every parameter is a variable name, so `- set: {taps: 5, mode: fast}`
    # reads naturally. `values:` is accepted for names that clash with a
    # common step key (name, optional, retries...).
    values = dict(params.mapping("values"))
    for key in list(params.step.params):
        if key != "values":
            values[key] = params.raw(key)
    for key, value in values.items():
        ctx.set_var(key, value)
    log.debug("%sset %s", ctx.indent, values)
    return ActionResult.ok(values)


def _as_number(value: Any, where: str, key: str) -> float:
    # --var only ever yields strings, and a flag counted as 1/0 is the whole
    # point of `add`, so 'true'/'false' have to read as numbers too.
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, str):
        lowered = value.strip().casefold()
        if lowered in {"true", "yes", "on"}:
            return 1.0
        if lowered in {"false", "no", "off"}:
            return 0.0
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ActionError(f"{where}: {key!r} is not a number: {value!r}") from exc


@action("add", aliases=("incr",), summary="Add to numeric flow variables")
def act_add(ctx: RunContext, params: Params) -> ActionResult:
    # Same shape as `set:` -- `- add: {bought: 1}` -- but the amount is added
    # to what is already there. An unset variable counts as 0, so a counter
    # needs no initialiser, and a negative amount subtracts. Booleans count as
    # 1/0, which is what makes `- add: {enabled_boxes: ${happiness_box}}` a
    # tally of the flags that are switched on.
    amounts = dict(params.mapping("values"))
    for key in list(params.step.params):
        if key != "values":
            amounts[key] = params.raw(key)

    where = params.step.location
    totals: Dict[str, Any] = {}
    for key, amount in amounts.items():
        total = _as_number(ctx.variables.get(key, 0), where, key) + _as_number(amount, where, key)
        # Keep whole numbers whole: `times: ${n}` and logs both read better
        # as 3 than as 3.0.
        totals[key] = int(total) if float(total).is_integer() else total
        ctx.set_var(key, totals[key])

    log.debug("%sadd %s -> %s", ctx.indent, amounts, totals)
    return ActionResult.ok(totals)


#: Where :func:`act_time_gate` remembers the mark it last fired, when the step
#: does not name a variable of its own. One gate per flag, so two gates can run
#: side by side without treading on each other's history.
TIME_GATE_STATE_SUFFIX = "_last_mark"


def _minute_marks(value: Any, where: str) -> List[int]:
    """The ``minutes:`` parameter as sorted, de-duplicated minutes of the hour.

    A list is the natural way to write it in YAML, but ``--var
    return_heart_minutes=15,50`` can only ever hand over a string, so a
    separated string reads the same way.
    """
    if value is None:
        return []
    if isinstance(value, str):
        items: Sequence[Any] = [part for part in re.split(r"[,;\s]+", value.strip()) if part]
    elif isinstance(value, (list, tuple)):
        items = value
    else:
        items = [value]

    marks = set()
    for item in items:
        try:
            minute = int(round(float(item)))
        except (TypeError, ValueError) as exc:
            raise ActionError(f"{where}: {item!r} is not a minute of the hour") from exc
        if not 0 <= minute <= 59:
            raise ActionError(f"{where}: minute {minute} is outside 0-59")
        marks.add(minute)
    return sorted(marks)


def _mark_at_or_before(now: datetime, marks: Sequence[int]) -> datetime:
    """The most recent time one of ``marks`` came round, at or before ``now``."""
    hour = now.replace(minute=0, second=0, microsecond=0)
    passed = [minute for minute in marks if minute <= now.minute]
    if passed:
        return hour.replace(minute=passed[-1])
    # None have come round yet this hour, so the last one was the final mark of
    # the hour before.
    return hour - timedelta(hours=1) + timedelta(minutes=marks[-1])


def _mark_after(now: datetime, marks: Sequence[int]) -> datetime:
    """The next time one of ``marks`` comes round. Only used for the log line."""
    hour = now.replace(minute=0, second=0, microsecond=0)
    upcoming = [minute for minute in marks if minute > now.minute]
    if upcoming:
        return hour.replace(minute=upcoming[0])
    return hour + timedelta(hours=1, minutes=marks[0])


@action("time_gate", primary="minutes", summary="Raise a flag when a minute of the hour comes round")
def act_time_gate(ctx: RunContext, params: Params) -> ActionResult:
    """Set ``save_as`` true on the first check after a minute-of-hour mark.

    Built for a bot loop that has to do something on the clock rather than
    every cycle -- `- time_gate: {minutes: [15, 50], save_as: send_heart_check}`
    puts the flag up once at :15 and once at :50, whichever cycle happens to be
    running when the minute goes by.

    Marks that went by *before* the first check are treated as already handled:
    a restart at 9:20pm waits for 9:50 instead of firing immediately, so
    stopping and starting the bot cannot squeeze in extra rounds. Once armed,
    the gate does catch up -- a cycle long enough to step over a mark still
    raises the flag on the far side of it, just once.

    Lowering the flag afterwards is the flow's job; this step only ever raises
    it, so the branch that consumes it decides what "done" means.
    """
    marks = _minute_marks(params.raw("minutes"), params.step.location)
    flag = params.string("save_as")
    # Kept in a flow variable rather than on the action so that a re-entered
    # flow (resume.yaml is called from launch.yaml) keeps its history.
    state_key = params.string("state", f"{flag}{TIME_GATE_STATE_SUFFIX}")

    if not marks:
        # An empty schedule is "never", not an error: it is what an unset
        # panel field or `--var minutes=` amounts to.
        log.debug("%stime_gate has no marks; %s stays down", ctx.indent, flag)
        ctx.set_var(flag, False)
        return ActionResult.ok(False)

    now = datetime.now()
    current = _mark_at_or_before(now, marks).isoformat(timespec="minutes")
    previous = ctx.variables.get(state_key)

    if previous is None:
        ctx.set_var(state_key, current)
        ctx.set_var(flag, False)
        log.info(
            "%stime_gate armed at :%02d -- %s next at %s",
            ctx.indent, now.minute, flag, _mark_after(now, marks).strftime("%H:%M"),
        )
        return ActionResult.ok(False)

    hit = current != str(previous)
    if hit:
        ctx.set_var(state_key, current)
        log.info("%stime_gate reached %s -- %s up", ctx.indent, current[-5:], flag)
    else:
        log.debug(
            "%stime_gate waiting for %s", ctx.indent, _mark_after(now, marks).strftime("%H:%M")
        )
    ctx.set_var(flag, hit)
    return ActionResult.ok(hit)


@action("stop", primary="message", summary="End the flow early")
def act_stop(ctx: RunContext, params: Params) -> ActionResult:
    message = params.string("message", "stop requested by flow")
    success = params.boolean("success", True)
    raise FlowAborted(message, success=success)


# --------------------------------------------------------------------------
# Pointer
# --------------------------------------------------------------------------
@action("click", primary="at", summary="Click at fixed coordinates")
def act_click(ctx: RunContext, params: Params) -> ActionResult:
    at = params.point("at")
    space = params.string("space", "content")
    button = params.string("button", "left")
    clicks = params.integer("clicks", 1)
    interval = params.number("interval", 0.08)
    if at is None:
        raise ActionError(f"{params.step.location}: 'at' is required")

    target = ctx.to_screen(at, space)
    log.info("%sclick %s (%s space) button=%s", ctx.indent, at, space, button)
    ctx.click_screen(target, button=button, clicks=clicks, interval=interval)
    return ActionResult.ok(target)


@action("move", primary="at", summary="Move the cursor without clicking")
def act_move(ctx: RunContext, params: Params) -> ActionResult:
    at = params.point("at")
    space = params.string("space", "content")
    if at is None:
        raise ActionError(f"{params.step.location}: 'at' is required")
    target = ctx.to_screen(at, space)
    if not ctx.dry_run:
        ctx.mouse.move(target)
    return ActionResult.ok(target)


@action("drag", summary="Drag from one point to another")
def act_drag(ctx: RunContext, params: Params) -> ActionResult:
    start = params.point("from")
    end = params.point("to")
    space = params.string("space", "content")
    button = params.string("button", "left")
    duration = params.duration("duration", 0.4)
    if start is None or end is None:
        raise ActionError(f"{params.step.location}: 'from' and 'to' are both required")

    screen_start = ctx.to_screen(start, space)
    screen_end = ctx.to_screen(end, space)
    log.info("%sdrag %s -> %s", ctx.indent, start, end)
    if not ctx.dry_run:
        ctx.mouse.drag(screen_start, screen_end, button=button, duration=duration)
    return ActionResult.ok((screen_start, screen_end))


@action("scroll", primary="amount", summary="Scroll the wheel")
def act_scroll(ctx: RunContext, params: Params) -> ActionResult:
    amount = params.integer("amount")
    at = params.point("at", None)
    space = params.string("space", "content")
    target = ctx.to_screen(at, space) if at is not None else None
    log.info("%sscroll %d", ctx.indent, amount)
    if not ctx.dry_run:
        ctx.mouse.scroll(amount, target)
    return ActionResult.ok(amount)


# --------------------------------------------------------------------------
# Keyboard
# --------------------------------------------------------------------------
@action("key", aliases=("press",), primary="key", summary="Press a key")
def act_key(ctx: RunContext, params: Params) -> ActionResult:
    key = params.string("key")
    presses = params.integer("presses", 1)
    interval = params.number("interval", 0.05)
    log.info("%skey %s x%d", ctx.indent, key, presses)
    if not ctx.dry_run:
        ctx.keyboard.press(key, presses=presses, interval=interval)
    return ActionResult.ok(key)


@action("hotkey", primary="keys", summary="Press a key combination")
def act_hotkey(ctx: RunContext, params: Params) -> ActionResult:
    keys = params.string_list("keys")
    if not keys:
        raise ActionError(f"{params.step.location}: 'keys' must list at least one key")
    log.info("%shotkey %s", ctx.indent, "+".join(keys))
    if not ctx.dry_run:
        ctx.keyboard.hotkey(*keys)
    return ActionResult.ok(keys)


@action("type", aliases=("type_text",), primary="text", summary="Type literal text")
def act_type(ctx: RunContext, params: Params) -> ActionResult:
    text = params.string("text")
    interval = params.number("interval", 0.02)
    log.info("%stype %r", ctx.indent, text)
    if not ctx.dry_run:
        ctx.keyboard.type_text(text, interval=interval)
    return ActionResult.ok(text)


# --------------------------------------------------------------------------
# Vision
# --------------------------------------------------------------------------
def _search_kwargs(params: Params) -> dict:
    """The parameter block shared by every template-searching action."""
    return {
        "region": params.raw("region", None),
        "confidence": params.optional_number("confidence", None),
        "scales": params.number_list("scales", None),
        # Grayscale (the default) ignores color, so it can't tell same-shaped
        # icons apart when only their color differs (e.g. gold/silver/bronze
        # medals). Set `grayscale: false` on a lookup to use color matching.
        "grayscale": params.raw("grayscale", None),
    }


@action("find", primary="template", summary="Look for a template (no click)")
def act_find(ctx: RunContext, params: Params) -> ActionResult:
    template = params.string("template")
    timeout = params.number("timeout", 0.0)
    poll_interval = params.optional_number("poll_interval", None)
    save_as = params.optional_string("save_as", None)
    search = _search_kwargs(params)

    if timeout > 0:
        match = ctx.wait_for(template, timeout=timeout, poll_interval=poll_interval, **search)
    else:
        match = ctx.find(template, **search)

    if save_as:
        ctx.set_var(save_as, None if match is None else match.center.as_tuple())
        ctx.set_var(f"{save_as}_confidence", 0.0 if match is None else round(match.confidence, 4))

    if match is None:
        log.info("%sfind %s -> not found", ctx.indent, template)
        return ActionResult.fail(f"template {template!r} not found")
    log.info("%sfind %s -> %s conf=%.3f", ctx.indent, template, match.center, match.confidence)
    return ActionResult.ok(match)


@action("find_click", aliases=("click_image", "tap"), primary="template",
        summary="Find a template and click it")
def act_find_click(ctx: RunContext, params: Params) -> ActionResult:
    template = params.string("template")
    timeout = params.number("timeout", ctx.config.runner.default_timeout)
    poll_interval = params.optional_number("poll_interval", None)
    offset = params.offset("offset", Point(0, 0))
    button = params.string("button", "left")
    clicks = params.integer("clicks", 1)
    interval = params.number("interval", 0.08)
    save_as = params.optional_string("save_as", None)
    search = _search_kwargs(params)

    if timeout > 0:
        match = ctx.wait_for(template, timeout=timeout, poll_interval=poll_interval, **search)
    else:
        match = ctx.find(template, **search)

    if match is None:
        log.info("%sfind_click %s -> not found", ctx.indent, template)
        return ActionResult.fail(f"template {template!r} not found within {timeout:g}s")

    target = match.center.offset(offset.x, offset.y)
    if save_as:
        ctx.set_var(save_as, target.as_tuple())
    log.info(
        "%sfind_click %s -> %s conf=%.3f", ctx.indent, template, target, match.confidence
    )
    ctx.click_screen(target, button=button, clicks=clicks, interval=interval)
    return ActionResult.ok(match)


@action("click_all", primary="template", summary="Click every match of a template")
def act_click_all(ctx: RunContext, params: Params) -> ActionResult:
    template = params.string("template")
    limit = params.integer("limit", 20)
    offset = params.offset("offset", Point(0, 0))
    button = params.string("button", "left")
    search = _search_kwargs(params)
    # Touch 'delay' now so it's marked consumed even if no matches are found
    # below -- the actual re-rolled value is read fresh in the loop.
    params.raw("delay", None)

    matches = ctx.find_all(template, max_results=limit, **search)
    if not matches:
        return ActionResult.fail(f"template {template!r} not found")

    log.info("%sclick_all %s -> %d match(es)", ctx.indent, template, len(matches))
    for match in matches:
        ctx.stop.check()
        ctx.click_screen(match.center.offset(offset.x, offset.y), button=button)
        # Re-rolled every iteration so a {min, max} delay varies each time,
        # not just once for the whole click_all call.
        ctx.sleep(params.duration("delay", 0.15))
    return ActionResult.ok(matches)


@action("wait_for", primary="template", summary="Block until a template appears")
def act_wait_for(ctx: RunContext, params: Params) -> ActionResult:
    template = params.string("template")
    timeout = params.number("timeout", ctx.config.runner.default_timeout)
    poll_interval = params.optional_number("poll_interval", None)
    save_as = params.optional_string("save_as", None)
    search = _search_kwargs(params)

    match = ctx.wait_for(template, timeout=timeout, poll_interval=poll_interval, **search)
    if save_as and match is not None:
        ctx.set_var(save_as, match.center.as_tuple())
    if match is None:
        return ActionResult.fail(f"{template!r} did not appear within {timeout:g}s")
    log.info("%swait_for %s -> appeared", ctx.indent, template)
    return ActionResult.ok(match)


@action("wait_until_gone", primary="template", summary="Block until a template disappears")
def act_wait_until_gone(ctx: RunContext, params: Params) -> ActionResult:
    template = params.string("template")
    timeout = params.number("timeout", ctx.config.runner.default_timeout)
    poll_interval = params.optional_number("poll_interval", None)
    search = _search_kwargs(params)

    still_visible = ctx.wait_for(
        template, timeout=timeout, poll_interval=poll_interval, gone=True, **search
    )
    if still_visible is not None:
        return ActionResult.fail(f"{template!r} was still visible after {timeout:g}s")
    log.info("%swait_until_gone %s -> gone", ctx.indent, template)
    return ActionResult.ok()


@action("screen_stuck", primary="seconds",
        summary="Report whether the screen is frozen and shows nothing we recognise")
def act_screen_stuck(ctx: RunContext, params: Params) -> ActionResult:
    """Is the game wedged? Sets ``save_as`` true when it looks like it is.

    Stillness on its own is not evidence of a crash. A healthy menu sits
    perfectly still -- LDPlayer's own home screen measures pixel-identical over
    ten seconds, mean difference 0.0000 -- so "unchanged" would condemn a
    screen that is simply not animating. What separates the two is whether
    anything we recognise is on it: a landmark template proves the game is
    alive however still it is, and a screen with no landmark *and* no movement
    is one nothing can be done with.

    Never fails, whatever it finds -- the verdict is a variable for the flow to
    branch on, not a step outcome, so a stuck screen doesn't also abort the
    flow that is trying to recover it.
    """
    seconds = params.duration("seconds", 30.0)
    # Zero, and deliberately so: this is the only tolerance whose error runs in
    # the safe direction. Anything above it lets a *small* animation -- one
    # bobbing tsum, a blinking cursor -- average away to nothing across the
    # whole content area and read as frozen, and the cost of that mistake is
    # restarting a healthy emulator. At zero, the faintest movement anywhere
    # proves life, and only a picture that is pixel-for-pixel identical counts
    # as stuck, which is what a real freeze actually looks like: the idle
    # measurement that motivated this action came back at exactly 0.0000.
    # Raise it only if capture noise ever produces a false "alive".
    tolerance = params.number("tolerance", 0.0)
    alive = params.string_list("alive", [])
    poll_interval = params.number("poll_interval", 0.5)
    save_as = params.optional_string("save_as", None)
    search = _search_kwargs(params)
    region = search["region"]

    def verdict(stuck: bool, why: str) -> ActionResult:
        if save_as:
            ctx.set_var(save_as, stuck)
        log.info("%sscreen_stuck -> %s (%s)", ctx.indent, stuck, why)
        return ActionResult.ok(stuck)

    # Landmarks first, and not only to be quick about it: this is the test that
    # keeps a still-but-healthy screen out of the restart path. It also answers
    # in milliseconds where the watch below costs `seconds`.
    for template in alive:
        if ctx.find(template, **search) is not None:
            return verdict(False, f"{template} on screen")

    # Nothing recognised, so now stillness means something. Return the moment
    # the picture moves -- only a genuinely frozen screen pays the full wait.
    baseline, _ = ctx.grab(region)
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        ctx.stop.check()
        ctx.sleep(poll_interval)
        frame, _ = ctx.grab(region)
        if not _frames_match(frame, baseline, tolerance):
            return verdict(False, "screen is still moving")

    return verdict(True, f"nothing recognised and no change in {seconds:.0f}s")


@action("screenshot", primary="path", summary="Save a screenshot")
def act_screenshot(ctx: RunContext, params: Params) -> ActionResult:
    label = params.optional_string("path", None)
    region = params.raw("region", None)
    image, _ = ctx.grab(region)
    path = ctx.save_debug(image, label or "screenshot")
    log.info("%sscreenshot -> %s", ctx.indent, path)
    return ActionResult.ok(path)


# --------------------------------------------------------------------------
# Control flow
# --------------------------------------------------------------------------
@action("if_found", primary="template", child_keys=("then", "else"),
        summary="Branch on whether a template is visible")
def act_if_found(ctx: RunContext, params: Params) -> ActionResult:
    template = params.string("template")
    timeout = params.number("timeout", 0.0)
    poll_interval = params.optional_number("poll_interval", None)
    search = _search_kwargs(params)
    then_steps = params.steps("then")
    else_steps = params.steps("else")

    if timeout > 0:
        match = ctx.wait_for(template, timeout=timeout, poll_interval=poll_interval, **search)
    else:
        match = ctx.find(template, **search)

    branch = "then" if match is not None else "else"
    log.info("%sif_found %s -> %s", ctx.indent, template, branch)
    steps = then_steps if match is not None else else_steps
    if not steps:
        return ActionResult.ok(match)
    return ActionResult(bool(ctx.execute(steps)), match)


@action("if", aliases=("when",), primary="value", child_keys=("then", "else"),
        summary="Branch on a flow variable's value")
def act_if(ctx: RunContext, params: Params) -> ActionResult:
    # `if: ${flag}`, or `if: {value: "${flag}", then: [...], else: [...]}`.
    # Nothing is read from the screen -- this is the plain boolean branch for
    # flags coming from `vars:`, `set:`, `save_as:` or `--var flag=true`.
    # An undefined variable interpolates to itself and fails the parameter
    # check, rather than quietly taking the else branch.
    if not params.has("value"):
        raise ActionError(f"{params.step.location}: 'value' is required")
    hit = params.boolean("value")
    then_steps = params.steps("then")
    else_steps = params.steps("else")

    log.info("%sif %s -> %s", ctx.indent, hit, "then" if hit else "else")
    steps = then_steps if hit else else_steps
    if not steps:
        return ActionResult.ok(hit)
    return ActionResult(bool(ctx.execute(steps)), hit)


@action("chance", aliases=("maybe",), primary="p", child_keys=("then", "else"),
        summary="Roll the dice and branch on the outcome")
def act_chance(ctx: RunContext, params: Params) -> ActionResult:
    # `chance: 0.3`, `chance: {percent: 30, then: [...]}`. The roll itself is
    # never a failure -- only the branch's own steps can fail the step.
    if params.has("percent"):
        probability = params.number("percent") / 100.0
    else:
        probability = params.number("p", 0.5)
    probability = min(1.0, max(0.0, probability))
    save_as = params.optional_string("save_as", None)
    then_steps = params.steps("then")
    else_steps = params.steps("else")

    roll = random.random()
    hit = roll < probability
    if save_as:
        ctx.set_var(save_as, hit)

    log.info(
        "%schance %.0f%% -> %s (roll %.3f)",
        ctx.indent, probability * 100, "then" if hit else "else", roll,
    )
    steps = then_steps if hit else else_steps
    if not steps:
        return ActionResult.ok(hit)
    return ActionResult(bool(ctx.execute(steps)), hit)


def _frames_match(first, second, tolerance: float) -> bool:
    """Whether two captures of the same region are the same picture.

    ``tolerance`` is the mean absolute difference per channel allowed before
    the frames count as changed -- 0 means pixel-identical, which is what a
    screen capture of a static list actually gives.
    """
    if first.shape != second.shape:
        return False
    if tolerance <= 0:
        return bool(np.array_equal(first, second))
    difference = np.abs(first.astype(np.int16) - second.astype(np.int16))
    return float(difference.mean()) <= tolerance


@action("repeat", child_keys=("steps",), summary="Repeat nested steps")
def act_repeat(ctx: RunContext, params: Params) -> ActionResult:
    steps = params.steps("steps")
    times = params.optional_number("times", None)
    forever = params.boolean("forever", False)
    while_found = params.optional_string("while_found", None)
    until_found = params.optional_string("until_found", None)
    duration = params.optional_number("duration", None)
    counter_var = params.string("counter", "index")
    max_iterations = params.integer("max_iterations", ctx.config.runner.max_iterations)
    stop_on_failure = params.boolean("stop_on_failure", False)
    # while_found/until_found naturally run out of iterations when something
    # unexpected is covering the screen (a popup, a stuck animation...). By
    # default that's silently treated as "done" like any other exhausted
    # loop; require_found makes it a reportable failure instead, since
    # exhausting the budget without the condition ever triggering usually
    # means the flow is now looking at a screen it doesn't recognize.
    require_found = params.boolean("require_found", False)
    # Scrolling loops run out of screen long before they run out of
    # iterations: a list clamped at its top or bottom returns the identical
    # picture for every remaining pass, so the condition can never start
    # matching and the only thing left to spend is time. Comparing the region
    # between iterations turns "wait for max_iterations" into "stop the moment
    # the screen stops moving".
    stop_when_still = params.boolean("stop_when_still", False)
    still_tolerance = params.number("still_tolerance", 0.0)
    search = _search_kwargs(params)
    # Touch 'delay' now so it's marked consumed even if the loop below never
    # runs (e.g. until_found/while_found's condition resolves on the very
    # first check) -- the actual re-rolled value is read fresh in the loop.
    params.raw("delay", None)

    if not steps:
        return ActionResult.ok(0)

    modes = [
        times is not None,
        forever,
        while_found is not None,
        until_found is not None,
        duration is not None,
    ]
    if sum(1 for m in modes if m) > 1:
        raise ActionError(
            f"{params.step.location}: choose exactly one of times / forever / "
            f"while_found / until_found / duration"
        )
    if not any(modes):
        # stop_when_still is a stopping condition in its own right: "scroll
        # until the screen stops moving" needs no other mode, and defaulting
        # to a single pass would make it meaningless. max_iterations is the
        # budget in that case.
        if not stop_when_still:
            times = 1.0

    limit = int(times) if times is not None else max_iterations
    limit = min(limit, max_iterations)
    deadline = time.monotonic() + duration if duration is not None else None

    completed = 0
    condition_met = False
    previous_frame: Optional[np.ndarray] = None
    for index in range(limit):
        ctx.stop.check()

        if deadline is not None and time.monotonic() >= deadline:
            log.info("%srepeat: %.1fs elapsed, stopping", ctx.indent, duration)
            break
        if while_found is not None and ctx.find(while_found, **search) is None:
            log.info("%srepeat: %r no longer visible, stopping", ctx.indent, while_found)
            condition_met = True
            break
        if until_found is not None and ctx.find(until_found, **search) is not None:
            log.info("%srepeat: %r appeared, stopping", ctx.indent, until_found)
            condition_met = True
            break

        # Checked after the conditions above so a loop that has both a match
        # and a frozen screen still reports the match. Leaving early here is
        # not "condition met": require_found still reports the failure, just
        # seconds in rather than a full budget later.
        if stop_when_still:
            frame, _ = ctx.grab(search.get("region"))
            if previous_frame is not None and _frames_match(frame, previous_frame, still_tolerance):
                log.info(
                    "%srepeat: screen stopped changing after %d iteration(s), stopping",
                    ctx.indent, completed,
                )
                break
            previous_frame = frame

        ctx.set_var(counter_var, index)
        ctx.set_var(f"{counter_var}_1based", index + 1)
        log.info("%srepeat iteration %d/%s", ctx.indent, index + 1, limit if times is not None else "?")

        succeeded = ctx.execute(steps)
        completed += 1
        if not succeeded and stop_on_failure:
            return ActionResult.fail(f"iteration {index + 1} failed", completed)
        # Re-rolled every iteration so a {min, max} delay varies each time.
        ctx.sleep(params.duration("delay", 0.0))

    if require_found and (while_found is not None or until_found is not None) and not condition_met:
        target = while_found if while_found is not None else until_found
        return ActionResult.fail(
            f"{target!r} condition was never met after {completed} attempt(s)", completed
        )

    return ActionResult.ok(completed)


@action("while_found", primary="template", child_keys=("steps",),
        summary="Repeat nested steps while a template stays visible")
def act_while_found(ctx: RunContext, params: Params) -> ActionResult:
    template = params.string("template")
    steps = params.steps("steps")
    max_iterations = params.integer("max_iterations", ctx.config.runner.max_iterations)
    search = _search_kwargs(params)
    # Touch 'delay' now so it's marked consumed even if the loop below never
    # runs (e.g. the template isn't visible on the very first check) -- the
    # actual re-rolled value is read fresh inside the loop.
    params.raw("delay", None)

    completed = 0
    while completed < max_iterations:
        ctx.stop.check()
        if ctx.find(template, **search) is None:
            break
        ctx.set_var("index", completed)
        ctx.execute(steps)
        completed += 1
        # Re-rolled every iteration so a {min, max} delay varies each time.
        ctx.sleep(params.duration("delay", 0.0))
    return ActionResult.ok(completed)


@action("run_flow", aliases=("call",), primary="flow", summary="Run another flow file")
def act_run_flow(ctx: RunContext, params: Params) -> ActionResult:
    from .flow import load_flow_by_name

    name = params.string("flow")
    variables = params.mapping("vars", {})
    isolate = params.boolean("isolate", True)

    flow = load_flow_by_name(ctx.config.flows_dir, name)
    log.info("%srun_flow %s", ctx.indent, flow.name)

    saved = dict(ctx.variables)
    ctx.variables.update(flow.vars)
    ctx.variables.update(variables)
    try:
        success = ctx.execute(flow.steps)
    finally:
        if isolate:
            ctx.variables.clear()
            ctx.variables.update(saved)
    return ActionResult(success, flow.name)


def _reattach_window(ctx: RunContext, timeout: float) -> bool:
    """Point ``ctx.window`` at whatever now matches ``window.target``.

    The handle is swapped *inside* the existing controller rather than by
    handing back a new one: the Application keeps that same object for the
    cursor stop zone, so replacing it would leave the guard reading a dead
    handle and quietly never tripping again.
    """
    manager = WindowManager()
    deadline = time.monotonic() + max(0.0, timeout)
    while True:
        ctx.stop.check()
        try:
            found = manager.find_one(ctx.config.window)
        except WindowNotFoundError:
            found = None
        if found is not None:
            if found.hwnd != ctx.window.hwnd:
                log.info(
                    "%sprepare_window re-attached 0x%08X -> 0x%08X",
                    ctx.indent,
                    ctx.window.hwnd,
                    found.hwnd,
                )
            ctx.window.hwnd = found.hwnd
            return True
        if time.monotonic() >= deadline:
            return False
        ctx.sleep(ctx.config.runner.default_poll_interval)


@action("prepare_window", summary="Re-detect, reposition and focus the emulator window")
def act_prepare_window(ctx: RunContext, params: Params) -> ActionResult:
    focus = params.boolean("focus", True)
    if ctx.window is None:
        return ActionResult.fail("no window attached")

    # Restarting the emulator hands it a brand new window handle, so the one
    # attached at startup is dead by the time a restart flow gets back here.
    # `redetect: true` waits for a window matching window.target to reappear;
    # a handle that has *already* gone away is re-detected whether it was asked
    # for or not, because failing on a window we know how to find again helps
    # nobody.
    if params.boolean("redetect", False) or not ctx.window.exists:
        timeout = params.duration("timeout", 60.0)
        if not _reattach_window(ctx, timeout):
            return ActionResult.fail(
                f"no window matching window.target appeared within {timeout:.0f}s"
            )

    ctx.window.restore()
    if params.boolean("move", ctx.config.window.move_on_prepare):
        size = (ctx.config.window.size.width, ctx.config.window.size.height) if ctx.config.window.size else None
        ctx.window.move_to(
            Point(ctx.config.window.position.x, ctx.config.window.position.y), size
        )
    if focus:
        ctx.window.focus()
    rect = ctx.refresh_window()
    log.info("%sprepare_window -> %s", ctx.indent, rect)
    return ActionResult.ok(rect)
