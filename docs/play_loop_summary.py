"""
ttheart-sender: how the Tsum Tsum play loop works
=================================================

THIS FILE IS A SUMMARY, NOT THE RUNNING CODE. It is a readable distillation of
`ttheart_sender/game/tsum.py` (~3200 lines), written to be pasted into a chat
as context. Bodies are simplified and some error handling is elided; the
control flow, the guards, and the reasons are faithful. The real module is the
authority -- if the two disagree, the module wins.

Real code:    ttheart_sender/game/tsum.py
Entry points: `python -m ttheart_sender.game.tsum play`  (CLI)
              the `play_tsum` flow action  (ttheart_sender/automation/tsum_actions.py)


THE PROBLEM
-----------
Disney Tsum Tsum is played on a bowl of ~50-70 overlapping plush characters
("tsums"). You clear them by dragging ONE continuous stroke through 3+ tsums of
the SAME character. This bot runs against an Android emulator (LDPlayer) on
Windows, driving a real mouse via pyautogui.

Three things make it hard, and every decision below traces to one of them:

  1. Tsums OVERLAP, TILT, and get half-buried. Template matching is at its
     worst here, so detection is colour segmentation + distance transform
     instead -- the "count the overlapping coins" trick. It is blind to
     orientation, and a new character costs nothing.
  2. The bot must decide what counts as the SAME character without knowing the
     roster. It clusters colour per frame; cluster ids are meaningful only
     within one frame's fit.
  3. Nearly every failure is SILENT. A drag the emulator half-samples registers
     as a 2-link, clears nothing, and the next scan finds the identical chain
     and retries forever. Most of play_loop is guards against that class of bug.


PIPELINE (one iteration)
------------------------
    grab frame
      -> settle            wait for the pile to stop falling
      -> stop_when?        round over -> exit before touching anything
      -> fire skill        if the ring has gone gold
      -> pop bubbles       worth more than a chain, and they expire
      -> detect()          colour cluster + distance transform -> [Tsum]
      -> find_chains()     adjacency graph -> longest simple path per component
      -> purity_filter()   last sanity check on chain colour
      -> anti-stuck guards miss streak / repetition / stall
      -> drag_chain()      one continuous stroke, walked in ~8px steps
      -> verify            did the board actually change?
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional


# ==========================================================================
# DATA
# ==========================================================================

@dataclass
class Tsum:
    x: float
    y: float
    r: float
    kind: int      # colour-cluster id -- STABLE ONLY WITHIN ONE FRAME'S FIT
    colour: tuple  # BGR, for drawing


@dataclass
class Chain:
    kind: int
    colour: tuple
    nodes: list = field(default_factory=list)   # indices into the tsum list
    is_base: bool = False                       # matches the equipped tsum


@dataclass
class PlayReport:
    """What one run did.

    `played` alone cannot compare two settings: fifteen 3-chains and eight
    6-chains are not close, and the second is far better. `cleared` and
    `stalled` are what an A/B actually turns on.
    """
    played: int = 0      # chains dragged
    cleared: int = 0     # total tsums in chains that verifiably cleared
    stalled: int = 0     # drags that ran and changed nothing
    trimmed: int = 0     # --verify-hold: members the game declined to mark
    abandoned: int = 0   # --verify-hold: presses released without dragging
    reason: str = ""     # why the loop ended
    stopped: bool = False


@dataclass
class Driver:
    """Everything the loop needs from the outside world.

    The loop is written against this rather than against the CLI's Application
    or a flow's RunContext, so ONE implementation serves both entry points.
    Both already own the same capture / matcher / template objects, so each
    constructor is a handful of lines and no play logic is duplicated.

    `rect` is the emulator's content area in screen coordinates -- the only
    thing that knows where the game sits on the desktop. Everything else the
    loop computes is content-relative.
    """
    capture: Any
    matcher: Any
    templates: Any
    rect: Any
    check_stop: Callable[[], None] = lambda: None    # raises StopRequested
    say: Callable[[Any], None] = print

    def grab(self):
        """-> BGR frame of `rect`."""

    def to_screen(self, x, y) -> tuple[int, int]:
        """Content-relative point -> absolute screen coordinates."""


# ==========================================================================
# SUPPORTING PIECES  (signature + the reason each exists)
# ==========================================================================

def detect(bgr, *, k=12, radius=None, palette=None, scale=1.0,
           include_dark=False, merge=False):
    """Locate every tsum. Returns (tsums, radius, palette/centres).

    k-means the board in Lab colour, then split each colour blob into
    individual tsums with a distance transform. `palette` reuses centres from
    an earlier frame -- the fit is most of the cost here and the colours do not
    change mid-run, so refitting every frame buys nothing.

    Deliberately NOT template matching: tsums overlap, tilt and get buried,
    which is exactly the case template matching handles worst.
    """


def find_chains(tsums, radius, link=1.75, *, block=0.75, link_px=100.0,
                base_kind=None, base_only=False, mode="touch", max_chain=8,
                first_leg_px=0.0, labels=None) -> list[Chain]:
    """Every playable chain, best first.

    Builds an adjacency graph over same-kind tsums, takes connected components,
    and finds the LONGEST SIMPLE PATH in each -- not the component size. A hub
    touching three others is a 4-tsum component but only a 3-tsum link, because
    a link is one continuous drag that cannot revisit a tsum.

    With `base_kind` set, chains of that type sort ahead of all others however
    long the others are: clearing your equipped tsum is what charges the skill,
    so a 3-chain of the base beats a 7-chain of anything else.

    `mode`:
      "touch" (default) -- linked when within `link_px` AND no third tsum sits
        on the segment between them. Distance alone invents chains you cannot
        actually play, because a different tsum is usually sitting in the gap.
      "reach" -- any two same-kind tsums link regardless of distance; the drag
        passes harmlessly over off-type tsums. Arguably closer to how the game
        is really played; not the default because it has not been A/B'd.
      "blob" -- reads contact off the colour mask instead of inferring it from
        distance. Accepts more real links, ~45x slower. Not the default; see
        docs/TODO-blob-adjacency.md.
    """


def purity_filter(bgr, tsums, nodes, radius, tol=35.0) -> list[int]:
    """Drop chain members that do not match the rest of the chain's colour.

    One misclassified member turns a 3-chain into a 2-link the game ignores --
    the drag runs, nothing clears, the board is unchanged. Cheap to check
    directly: sample each member's colour off the image and discard anything
    far from the chain's median.
    """


def _settle(drv, *, max_wait=0.9, tol=2.5):
    """Wait for the board to stop moving; return the frame it stopped on.

    Cleared tsums drop and the pile collapses; detecting mid-fall gives
    coordinates already stale by the time the drag runs. Frame diffing costs
    ~5ms, so this beats any fixed delay big enough to be safe -- it returns the
    instant the board is still. CAPPED, because FEVER animates continuously and
    would otherwise wait forever.
    """


def drag_chain(points, *, step_px=8.0, hold=0.05, per_step=0.004,
               after_press=None):
    """Drag through every point in order as ONE continuous stroke.

    The emulator turns mouse movement into touch movement, and only sees the
    positions it is actually given. Jumping corner to corner teleports past the
    tsums in between, so each leg is walked in ~8px steps -- the chain is built
    from what the finger passes OVER, not from where it stops.

    `after_press` runs once the first point is held and before anything moves,
    and returns the points to actually walk. That window is the only moment the
    game will tell you which tsums it accepts from here (see `marked_by_game`),
    and it costs nothing to use, because pressing the first tsum starts the drag
    either way. Returning a single point walks nowhere and releases -- which is
    how a chain the game rejects gets abandoned before it wastes a stroke.
    """


def marked_by_game(drv, before_crop, board, tsums, nodes, *,
                   delay=0.10, threshold=8.0, aura=90.0,
                   frames=1, gap=0.0) -> list[int]:
    """Of a chain's members, which ones did the game light up?

    Holding a tsum makes the game mark everything linkable from it -- both the
    same character AND actually reachable, which are the two judgements this
    module makes worst. Measured on a real board: pressing one Piglet marked
    five more that colour clustering had filed under three different kinds. So
    ask the game rather than trust the clusters.

    Members within `aura` of the pressed tsum are kept regardless -- the glow is
    ~90px across and washes over whatever is under it, so a reaction there means
    nothing either way. Only DISTANT members are ever dropped, which makes this
    a conservative trim: it removes the long reaches the guesswork gets wrong,
    and never second-guesses the near ones.

    THE DEFAULTS ABOVE DO NOT WORK. They are `--verify-hold`'s, which pays this
    read on every drag and is tuned for speed, and at 0.10s the game has not
    finished drawing the highlight (the floor is ~0.15s, see `assist`). One
    frame against a still-settling board reads motion instead: measured over
    5,729 collected drags, the median untouched tsum already cleared the 8.0
    threshold on half of them, and the tsums this returned were no more alike
    than the board average. `--verify-hold` is opt-in and stays as it is; the
    sample collector passes its own delay=0.25 and frames=3, which is what the
    multi-frame minimum is for. See docs/DATASET-FINDINGS.md.
    """


def skill_gold(frame, spec, r_in=30.0, r_out=52.0) -> float:
    """Fraction of the skill button's ring that has gone gold.

    The button resists template matching because the character portrait in the
    middle never changes -- only the plate behind it lights up when the skill is
    charged. So ignore the portrait entirely and sample the RING around it with
    an HSV gold mask. Across 14 uncharged captures the ring reads 0.000, so any
    real signal is unambiguous.
    """


def _pop_bubbles(drv, frame, templates, confidence=0.80, **kw) -> dict:
    """Tap any bubble on screen; returns {template name: how many popped}.

    Uses the project's own template matcher rather than a colour rule invented
    here -- capture the art once with `python main.py snip <name>` and this
    finds it, so a new special item costs a snip and a name in `--bubble`
    rather than a code change. The frame is re-grabbed after every tap: bubbles
    drift and pop with an animation, so stale coordinates go wrong fast.
    """


def _click_shuffle(drv, spec, times=3, delay=0.3, hold=0.05, move_time=0.05):
    """Tap a fixed content-relative point N times -- the shuffle button."""


# ==========================================================================
# THE MAIN LOOP
# ==========================================================================

def play_loop(drv: Driver, opts, *, stop_when: Optional[Callable] = None) -> PlayReport:
    """Grab the board, pick the best chain, drag it -- once, or on a loop.

    `stop_when` is called with each settled frame BEFORE anything is clicked on
    it. Returning a string ends the run and becomes the report's reason -- that
    is how a flow ends the round on "the scoreboard appeared" rather than on a
    stopwatch.

    NOTE ON STRUCTURE: this reads long because it is mostly guards. The happy
    path is ~15 lines (detect -> find_chains -> drag). Everything else is here
    because a specific silent failure was observed in a real run.
    """
    say = drv.say
    bubbles = _load_bubbles(drv, opts.bubble) if opts.bubble and not opts.dry_run else []
    deadline = time.perf_counter() + opts.duration if opts.duration > 0 else None
    report = PlayReport()

    # Fitting the palette is the expensive half of detection, and the tsums in
    # play do not change mid-game, so it is fit once and reused. Same for the
    # radius and the base-tsum lookup.
    palette, radius, base = None, opts.radius, None
    played = misses = stalls = 0
    per_step = opts.per_step
    skip_kinds: set[int] = set()                 # kinds that just failed
    recent = deque(maxlen=opts.repeat_window)    # chain signatures
    lengths = deque(maxlen=opts.repeat_len)      # chain lengths
    no_settle = False   # last pass never touched the board -> nothing to wait for

    try:
        while True:
            drv.check_stop()
            frame = (drv.grab() if opts.dry_run or palette is None or no_settle
                     else _settle(drv, max_wait=opts.settle))
            no_settle = False

            # --- exit hook, asked before anything on this frame is touched ---
            # Once the round is over the board is gone, and a chain "found" on
            # the results screen would drag across whatever buttons sit there.
            if stop_when is not None:
                reason = stop_when(frame)
                if reason:
                    report.reason = reason
                    break

            # --- skill first: it is charged BY the chains already played, and
            #     firing it is worth more than any single chain ---
            if opts.skill and not opts.dry_run:
                gold = skill_gold(frame, opts.skill, opts.skill_inner, opts.skill_outer)
                if gold >= opts.skill_gold:
                    _click_shuffle(drv, opts.skill, 1, 0.0, opts.hold, opts.move_time)
                    frame = _settle(drv, max_wait=max(opts.settle, 1.2))

            # --- bubbles: worth more than a chain, and they vanish on their own ---
            if bubbles:
                if _pop_bubbles(drv, frame, bubbles, opts.bubble_confidence):
                    frame = _settle(drv, max_wait=opts.settle)

            bx, by, bw, bh = _board_rect(frame.shape, opts.board)
            crop = frame[by:by + bh, bx:bx + bw]

            # ------------------------- DETECT -------------------------
            tsums, radius, palette = detect(crop, k=opts.k, radius=radius,
                                            palette=palette, scale=opts.scale,
                                            include_dark=opts.include_dark,
                                            merge=opts.merge)

            # FEVER repaints the whole board in neon, so a palette fit during
            # normal play stops matching anything and the count collapses.
            # Rather than trust a stale fit, throw it away and refit whenever
            # the count looks wrong -- that covers entering fever, leaving it,
            # and calibrating off a menu frame by accident.
            #
            # The CEILING matters as much as the floor: a cached radius that has
            # drifted small blows the count UP rather than down, and that reads
            # as a healthy board unless both ends are checked.
            plausible = opts.min_tsums <= len(tsums) <= opts.max_tsums
            if not plausible and palette is not None:
                fresh, fresh_r, fresh_pal = detect(crop, k=opts.k, scale=opts.scale)
                if abs(len(fresh) - opts.min_tsums) < abs(len(tsums) - opts.min_tsums):
                    tsums, radius, palette, base = fresh, fresh_r, fresh_pal, None

            if base is None and opts.use_base:
                base, _ = read_base_kind(frame, palette, spec=opts.base)

            # ------------------------- CHOOSE -------------------------
            chains = find_chains(tsums, radius, opts.link, block=opts.block,
                                 link_px=opts.link_px, base_kind=base,
                                 base_only=opts.base_only, mode=opts.mode,
                                 max_chain=opts.max_chain,
                                 first_leg_px=opts.first_leg_px)

            # A real board is crowded but bounded. Menus and the results screen
            # still yield blobs and can still produce a "chain", and dragging
            # that would swipe across live UI buttons. The Home screen scores
            # 200+ "tsums" off portraits and panel texture, sails past any
            # minimum, and produces a confident chain EVERY single frame.
            if not (opts.min_tsums <= len(tsums) <= opts.max_tsums):
                chains = []

            chains = [c for c in chains
                      if len(c) >= opts.min_chain and c.kind not in skip_kinds]

            # ----------------- GUARD 1: nothing playable -----------------
            if not chains:
                misses += 1
                if deadline is None or time.perf_counter() >= deadline:
                    report.reason = "no chain" if deadline is None else "elapsed"
                    break
                # A miss streak usually is not an empty board. Restarting the
                # command by hand routinely un-sticks exactly this, and the only
                # thing a restart does differently is start with NO cached
                # palette, radius, base or skip-list -- so do that here instead
                # of giving up. `skip_kinds` especially: it only ever clears on
                # a successful drag, so a long miss streak with no success in
                # between could otherwise blacklist every kind on the board
                # forever.
                if misses >= opts.max_misses:
                    _click_shuffle(drv, opts.shuffle, opts.shuffle_clicks,
                                   opts.shuffle_delay, opts.hold, opts.move_time)
                    _settle(drv, max_wait=max(opts.settle, 1.5))
                    palette, radius, base = None, opts.radius, None
                    skip_kinds.clear()
                    misses = 0
                continue
            misses = 0

            # ----------------- GUARD 2: colour purity -----------------
            # Last gate before committing to a drag. A chain failing this is a
            # reason to try the NEXT chain, not to throw the frame away: the
            # board usually offers several, and discarding all of them because
            # the best one had an odd member stalls the run outright.
            best = None
            for cand in chains:
                clean = purity_filter(crop, tsums, cand.nodes, radius, opts.purity)
                if len(clean) >= max(3, opts.min_chain):
                    best = Chain(cand.kind, cand.colour, clean, cand.is_base)
                    break
            if best is None:
                misses += 1
                # ... same recovery ladder as GUARD 1 ...
                continue

            # --------------- GUARD 3: board not advancing ---------------
            # The least obvious way to get stuck: chains keep being found and
            # drags keep LOOKING like they landed, but the board never advances.
            # Neither other guard catches it -- it is not a miss streak (chains
            # exist), and not a stall streak either, because idle tsums jiggle
            # enough to push the frame diff past --change-tol even when nothing
            # cleared. The giveaway is repetition, detected two ways:
            #
            # (a) coarse -- the same chain LENGTH N times running. This is the
            #     tell when positions drift a few px each frame, so the exact
            #     signature differs every time and (b) never fires.
            lengths.append(len(best))
            if len(lengths) >= opts.repeat_len and len(set(lengths)) == 1:
                _click_shuffle(drv, opts.shuffle, opts.shuffle_clicks,
                               opts.shuffle_delay, opts.hold, opts.move_time)
                palette, radius, base = None, opts.radius, None
                skip_kinds.clear(); lengths.clear(); recent.clear()
                continue

            # (b) exact -- the same chain signature repeatedly. Endpoints are
            #     quantised (// 12) so a few px of detection jitter does not
            #     read as a different chain each time.
            sig = (best.kind, len(best),
                   int(tsums[best.nodes[0]].x) // 12, int(tsums[best.nodes[0]].y) // 12,
                   int(tsums[best.nodes[-1]].x) // 12, int(tsums[best.nodes[-1]].y) // 12)
            recent.append(sig)
            if recent.count(sig) >= opts.max_repeats:
                _click_shuffle(drv, opts.shuffle, opts.shuffle_clicks,
                               opts.shuffle_delay, opts.hold, opts.move_time)
                palette, radius, base = None, opts.radius, None
                skip_kinds.clear(); recent.clear()
                continue

            # --------------------------- ACT ---------------------------
            # crop -> content area -> screen. pyautogui works in screen
            # coordinates, which is why the window is parked at the top-left.
            screen = [drv.to_screen(int(tsums[i].x) + bx, int(tsums[i].y) + by)
                      for i in best.nodes]
            if opts.dry_run:
                report.reason = "dry run"
                break

            before, probe = crop, {}

            def _ask_the_game():
                """Trim the chain to what the game marks, while it is held."""
                kept = marked_by_game(drv, before, (bx, by, bw, bh), tsums,
                                      best.nodes, delay=opts.hold_delay,
                                      threshold=opts.hold_threshold,
                                      aura=opts.hold_aura)
                probe.update(kept=len(kept), dropped=len(best.nodes) - len(kept))
                if len(kept) < opts.min_chain:
                    # Abandon before moving. Releasing on one tsum clears
                    # nothing, which is cheaper than dragging a chain the game
                    # has already said it will not accept.
                    probe["abandoned"] = True
                    return [screen[0]]
                return [drv.to_screen(bx + tsums[i].x, by + tsums[i].y) for i in kept]

            drag_chain(screen, step_px=opts.step_px, per_step=per_step,
                       hold=opts.hold,
                       after_press=_ask_the_game if opts.verify_hold else None)

            if probe.get("abandoned"):
                misses += 1
                report.abandoned += 1
                # Without this the next scan finds the identical chain, presses
                # it, and is rejected again -- the board has not changed, so
                # nothing about the answer would either.
                skip_kinds.add(best.kind)
                no_settle = True      # do not wait for a board that never moved
                continue
            if probe.get("dropped"):
                report.trimmed += probe["dropped"]

            # ------------- GUARD 4: did anything actually clear? -------------
            # A drag the emulator only half-sampled registers as a 2-link, which
            # is below the game's minimum -- nothing pops, the board is
            # unchanged, and the next scan finds the identical chain and tries
            # it again forever. THAT is the freeze this block prevents.
            if opts.verify:
                after = drv.grab()[by:by + bh, bx:bx + bw]
                changed = float(np.mean(cv2.absdiff(after, before)))
                if changed < opts.change_tol:
                    stalls += 1
                    report.stalled += 1
                    # Two cures applied together: walk the path more slowly so
                    # every tsum gets sampled, and stop re-offering the chain
                    # that just failed.
                    per_step = min(per_step * 2, 0.05)
                    skip_kinds.add(best.kind)
                    if stalls >= opts.max_stalls:
                        # If slow-and-skip has not found a working chain in N
                        # tries, that combination is not it -- drop back to
                        # defaults and let the next recalibration try fresh.
                        _click_shuffle(drv, opts.shuffle, opts.shuffle_clicks,
                                       opts.shuffle_delay, opts.hold, opts.move_time)
                        palette, radius, base = None, opts.radius, None
                        per_step, stalls = opts.per_step, 0
                        skip_kinds.clear()
                    continue
                stalls = 0
                skip_kinds.clear()

            played += 1
            # What was DRAGGED, not what was proposed: with --verify-hold the
            # chain may have been trimmed after the press.
            report.cleared += probe.get("kept", len(best))

            if deadline is None or time.perf_counter() >= deadline:
                report.reason = "one chain played" if deadline is None else "elapsed"
                break

    except StopRequested:
        # The stop key is the user aborting everything, not this loop failing:
        # report what was played, then let it travel on to whoever is driving
        # (the CLI exits, the flow runner stops the whole flow).
        report.played, report.stopped = played, True
        raise

    report.played = played
    return report


# ==========================================================================
# THE RECOVERY LADDER  (the part worth understanding)
# ==========================================================================
#
# Four distinct ways the bot gets stuck, each needing its own detector, because
# none of them is visible to the others:
#
#   1. NOTHING PLAYABLE     no chain passes min_chain for `max_misses` frames
#      -> shuffle + drop every cached fit (palette, radius, base, skip_kinds)
#
#   2. ALL CHAINS IMPURE    every chain on the board had odd-coloured members
#      -> same ladder as (1)
#
#   3. BOARD NOT ADVANCING  chains found, drags look fine, nothing changes.
#      Invisible to (1) because chains exist, and invisible to (4) because idle
#      tsums jiggle past --change-tol. Detected by REPETITION, two ways:
#        (a) same chain length `repeat_len` times running
#        (b) same quantised signature `max_repeats` times in `repeat_window`
#      -> shuffle + drop caches
#
#   4. DRAG NOT REGISTERING before/after frame diff < change_tol
#      -> slow the stroke (per_step *= 2, capped) AND blacklist that kind;
#         after `max_stalls`, shuffle + reset speed + drop caches
#
# The common cure is "throw away every cached fit and start clean", because
# that is empirically what restarting the command by hand does -- and
# restarting by hand was observed to un-stick all four.
#
#
# KEY TUNABLES (defaults)
# -----------------------
#   detection  -k 12                    colour clusters
#              --scale 1.0              detect on a downscaled copy (0.5 ~= 4x faster)
#              --min-tsums 20 / --max-tsums 110    plausible-board window
#   linking    --mode touch             touch | reach | blob
#              --link-px 100            link distance in PIXELS (stable across frames)
#              --block 0.75             third-tsum blocking radius, in radii
#              --max-chain 8            cap; picks the tightest cluster of N
#              --purity 35.0            colour tolerance for chain members
#   timing     --settle 0.9             max wait for the pile to stop falling
#              --step-px 8.0            drag resolution
#              --per-step 0.004         seconds per drag step (doubles on a stall)
#   safety     --verify                 confirm the board changed (on by default)
#              --verify-hold            ask the game which members it accepts
#              --max-misses 6 / --max-repeats 3 / --max-stalls 4
#
#
# WHY THE COORDINATE DANCE
# ------------------------
#   detection coords    -> relative to the BOARD CROP
#   + (bx, by)          -> relative to the emulator CONTENT AREA
#   + drv.to_screen()   -> absolute SCREEN, which is all pyautogui understands
#
# The window is parked at the top-left of the desktop so this stays stable for
# a whole run; `rect` is the single place that knows where the content sits.
