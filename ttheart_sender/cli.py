"""Command line interface.

    python main.py                      no idea where to start? this detects
                                         LDPlayer and parks it at the top-left
    python main.py windows              list open windows (find LDPlayer's class)
    python main.py detect               show detected emulator instances
    python main.py prepare              park LDPlayer at the top-left and focus it
    python main.py snip heart_button    capture a template image interactively
    python main.py point                hover + press a key to read its coordinates
    python main.py shot                 save a screenshot of the emulator content area
    python main.py find heart_button    test-match a template and report the score
    python main.py flows                list available flows
    python main.py actions              list every action the flow language supports
    python main.py run send_heart       run a flow

Any command that touches the screen (prepare/shot/snip/point/find/run) detects
LDPlayer and parks it at the top-left before doing anything else -- that's
what makes coordinates in flows and templates stable. Skip it with the
global --no-window (don't attach at all) or --no-prepare (attach but leave
it where it is) flags.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from . import __version__
from .app import Application
from .automation.flow import list_flows
from .automation.registry import registered_actions
from .exceptions import TTHeartError
from .geometry import Rect

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Parser
# --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ttheart",
        description="Screenshot-driven click automation for Windows / LDPlayer.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--version", action="version", version=f"ttheart-sender {__version__}")
    parser.add_argument("-c", "--config", help="Path to config.yaml (or its directory)")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Verbose (DEBUG) logging"
    )
    parser.add_argument("-q", "--quiet", action="store_true", help="Only log warnings and errors")
    parser.add_argument(
        "--no-window",
        action="store_true",
        help="Skip emulator detection entirely (screen-only operation)",
    )
    parser.add_argument(
        "--no-prepare",
        action="store_true",
        help="Detect the emulator but leave it where it is instead of moving/focusing it",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    p_windows = sub.add_parser("windows", help="List open top-level windows")
    p_windows.add_argument("--filter", help="Only show windows whose title/class contains this")
    p_windows.set_defaults(func=cmd_windows, needs_window=False)

    p_detect = sub.add_parser("detect", help="Show emulator windows matching the config")
    p_detect.set_defaults(func=cmd_detect, needs_window=False)

    p_prepare = sub.add_parser("prepare", help="Restore, move to top-left and focus the emulator")
    p_prepare.add_argument("--no-focus", action="store_true", help="Do not steal focus")
    p_prepare.set_defaults(func=cmd_prepare, needs_window=True)

    p_shot = sub.add_parser("shot", help="Save a screenshot")
    p_shot.add_argument("-o", "--out", help="Output path (default: logs/debug/<timestamp>.png)")
    p_shot.add_argument(
        "--screen", action="store_true", help="Capture the whole desktop instead of the emulator"
    )
    p_shot.add_argument("--region", help="x,y,width,height relative to the emulator content area")
    p_shot.add_argument(
        "--no-focus", action="store_true", help="Capture without bringing the emulator forward"
    )
    p_shot.set_defaults(func=cmd_shot, needs_window=True)

    p_snip = sub.add_parser("snip", help="Interactively capture a template image")
    p_snip.add_argument("name", help="Template name, e.g. heart_button")
    p_snip.add_argument("--key", default="f8", help="Key that marks a corner (default: f8)")
    p_snip.add_argument("--padding", type=int, default=0, help="Extra pixels around the region")
    p_snip.set_defaults(func=cmd_snip, needs_window=True)

    p_point = sub.add_parser(
        "point", help="Hover the mouse and press a key to read its coordinates"
    )
    p_point.add_argument("--key", default="f8", help="Key that reads a point (default: f8)")
    p_point.set_defaults(func=cmd_point, needs_window=True)

    p_find = sub.add_parser("find", help="Test-match a template against the screen right now")
    p_find.add_argument("template", help="Template name or path")
    p_find.add_argument("--confidence", type=float, help="Override the match threshold")
    p_find.add_argument("--all", action="store_true", help="Report every match, not just the best")
    p_find.add_argument("--screen", action="store_true", help="Search the whole desktop")
    p_find.add_argument("--click", action="store_true", help="Click the best match")
    p_find.add_argument("--save", action="store_true", help="Save an annotated debug screenshot")
    p_find.add_argument(
        "--no-focus", action="store_true", help="Search without bringing the emulator forward"
    )
    p_find.set_defaults(func=cmd_find, needs_window=True)

    p_templates = sub.add_parser("templates", help="List template images")
    p_templates.set_defaults(func=cmd_templates, needs_window=False)

    p_flows = sub.add_parser("flows", help="List available flows")
    p_flows.set_defaults(func=cmd_flows, needs_window=False)

    p_actions = sub.add_parser("actions", help="List every supported flow action")
    p_actions.set_defaults(func=cmd_actions, needs_window=False)

    p_validate = sub.add_parser("validate", help="Parse a flow (or all flows) without running it")
    p_validate.add_argument("flow", nargs="?", help="Flow name; omit to validate every flow")
    p_validate.set_defaults(func=cmd_validate, needs_window=False)

    p_run = sub.add_parser("run", help="Run a flow")
    p_run.add_argument("flow", help="Flow name, e.g. send_heart")
    p_run.add_argument(
        "--var",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a flow variable (repeatable)",
    )
    p_run.add_argument("--loops", type=int, default=1, help="Repeat the whole flow N times (0 = forever)")
    p_run.add_argument("--loop-delay", type=float, default=0.0, help="Seconds between loops")
    p_run.add_argument("--dry-run", action="store_true", help="Log clicks instead of performing them")
    p_run.set_defaults(func=cmd_run, needs_window=True)

    return parser


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------
def cmd_windows(app: Application, args: argparse.Namespace) -> int:
    needle = (args.filter or "").casefold()
    rows = app.windows.list_windows()
    shown = 0
    for info in sorted(rows, key=lambda w: (w.class_name.casefold(), w.title.casefold())):
        if needle and needle not in info.title.casefold() and needle not in info.class_name.casefold():
            continue
        print(info.describe())
        shown += 1
    print(f"\n{shown} of {len(rows)} visible window(s).")
    if not shown:
        print("Nothing matched. Is the program running and un-minimized?")
    return 0


def cmd_detect(app: Application, args: argparse.Namespace) -> int:
    matches = app.detect_windows()
    if not matches:
        print("No emulator window matched the config.")
        print(f"  class_names     : {app.config.window.target.class_names}")
        print(f"  title_contains  : {app.config.window.target.title_contains}")
        print("\nRun `python main.py windows` and copy the real class name into config.yaml.")
        return 1
    for index, info in enumerate(matches):
        marker = " <- window.instance" if index == app.config.window.instance else ""
        print(f"[{index}] {info.describe()}{marker}")
    return 0


def cmd_prepare(app: Application, args: argparse.Namespace) -> int:
    if app.window is None:
        raise TTHeartError("No emulator window attached (was --no-window passed?)")
    if args.no_focus:
        app.config.window.focus_on_prepare = False
    # main() already ran startup(), but re-run explicitly so this command's own
    # flags (e.g. --no-focus) always take effect, even under --no-prepare.
    app.window.prepare(app.config.window)
    info = app.window.info()
    content = app.content_rect()
    print(f"Window : {info.title!r} ({info.class_name})")
    print(f"Frame  : {info.window_rect}")
    print(f"Client : {info.client_rect}")
    print(f"Content: {content}   <- coordinates in flows are relative to this")
    return 0


def cmd_shot(app: Application, args: argparse.Namespace) -> int:
    region: Optional[Rect] = None
    if args.screen:
        region = app.capture.virtual_screen_rect()
    else:
        region = app.content_rect()
        if args.region:
            offset = _parse_rect(args.region)
            region = offset.translate(region.left, region.top)

    image = app.capture.grab(region)
    if args.out:
        from .screen.capture import save_image

        path = save_image(Path(args.out), image)
    else:
        ctx = app.build_context()
        path = ctx.save_debug(image, "shot")
    print(f"Saved {path} ({region.width}x{region.height} from {region})")
    return 0


def cmd_snip(app: Application, args: argparse.Namespace) -> int:
    from .tools.snipper import snip_region

    name = args.name if Path(args.name).suffix else f"{args.name}.png"
    output = app.config.templates_dir / name
    if output.exists():
        print(f"Note: {output} already exists and will be overwritten.")
    path = snip_region(
        app.capture,
        output,
        mark_key=args.key,
        padding=args.padding,
    )
    if path is None:
        return 1
    print(f"\nUse it in a flow as:\n  - find_click: {{template: {Path(name).stem}}}")
    return 0


def cmd_point(app: Application, args: argparse.Namespace) -> int:
    from .control.hotkey import resolve_vk
    from .tools.snipper import wait_for_key

    mark_vk = resolve_vk(args.key)
    cancel_vk = resolve_vk("esc")
    if mark_vk is None:
        raise TTHeartError(f"Unknown key {args.key!r}")

    ctx = app.build_context()
    content = app.content_rect()
    key_label = args.key.upper()
    print(f"Content area: {content}")
    print(f"Hover the mouse anywhere and press {key_label} to read its coordinates.")
    print("Press ESC to stop.\n")

    while True:
        point = wait_for_key(mark_vk, cancel_vk=cancel_vk, timeout=300.0)
        if point is None:
            print("Stopped.")
            return 0
        if content.contains(point):
            rel = ctx.to_content(point)
            print(f"screen={point}   content=({rel.x}, {rel.y})   <- use this in flows, e.g. click: [{rel.x}, {rel.y}]")
        else:
            print(f"screen={point}   (outside the emulator content area {content})")


def cmd_find(app: Application, args: argparse.Namespace) -> int:
    ctx = app.build_context()
    region = "screen" if args.screen else None

    template = app.templates.get(args.template)
    image, rect = ctx.grab(region)
    print(f"Searching {rect} for {template} (threshold {args.confidence or app.config.matching.confidence:.2f})")

    if args.all:
        matches = app.matcher.find_all(image, template, confidence=args.confidence)
    else:
        best = app.matcher.find(image, template, confidence=args.confidence)
        matches = [best] if best else []

    if not matches:
        score = app.matcher.best_score(image, template)
        print(f"No match. Best similarity was {score:.3f} -- lower the threshold or re-snip the template.")
        if args.save:
            print(f"Debug image: {ctx.save_debug(image, 'find_miss')}")
        return 1

    for match in matches:
        screen = match.translated(rect.left, rect.top)
        content = ctx.to_content(screen.center) if ctx.window else screen.center
        print(f"  conf={match.confidence:.3f} screen={screen.center} content={content} scale={match.scale:g}")

    if args.save:
        print(f"Debug image: {ctx.save_debug(image, 'find_hit', matches)}")

    if args.click:
        best = matches[0].translated(rect.left, rect.top)
        print(f"Clicking {best.center}")
        ctx.click_screen(best.center)
    return 0


def cmd_templates(app: Application, args: argparse.Namespace) -> int:
    names = app.templates.available()
    if not names:
        print(f"No templates in {app.config.templates_dir}. Create one with: python main.py snip <name>")
        return 1
    for name in names:
        template = app.templates.get(name)
        mask = " (masked)" if template.mask is not None else ""
        print(f"  {name:<40} {template.width}x{template.height}{mask}")
    print(f"\n{len(names)} template(s) in {app.config.templates_dir}")
    return 0


def cmd_flows(app: Application, args: argparse.Namespace) -> int:
    paths = list_flows(app.config.flows_dir)
    if not paths:
        print(f"No flows in {app.config.flows_dir}")
        return 1
    for path in paths:
        try:
            flow = app.load_flow(str(path))
            description = f" -- {flow.description}" if flow.description else ""
            print(f"  {path.stem:<24} {len(flow.steps)} step(s){description}")
        except TTHeartError as exc:
            print(f"  {path.stem:<24} [INVALID] {exc}")
    return 0


def cmd_actions(app: Application, args: argparse.Namespace) -> int:
    for spec in registered_actions():
        aliases = f" (aka {', '.join(spec.aliases)})" if spec.aliases else ""
        print(f"  {spec.name:<18}{spec.summary}{aliases}")
    return 0


def cmd_validate(app: Application, args: argparse.Namespace) -> int:
    targets = [args.flow] if args.flow else [str(p) for p in list_flows(app.config.flows_dir)]
    if not targets:
        print(f"No flows to validate in {app.config.flows_dir}")
        return 1
    failures = 0
    for target in targets:
        try:
            flow = app.load_flow(target)
            print(f"OK   {flow.name} ({_count_steps(flow.steps)} step(s))")
        except TTHeartError as exc:
            failures += 1
            print(f"FAIL {target}: {exc}")
    return 1 if failures else 0


def cmd_run(app: Application, args: argparse.Namespace) -> int:
    variables = _parse_vars(args.var)
    report = app.run_flow(
        args.flow,
        variables=variables,
        loops=args.loops,
        loop_delay=args.loop_delay,
        use_window=not args.no_window,
        prepare_window=not args.no_prepare,
    )
    return 0 if report.success else 1


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _parse_vars(pairs: Sequence[str]) -> Dict[str, Any]:
    import yaml

    variables: Dict[str, Any] = {}
    for pair in pairs:
        if "=" not in pair:
            raise TTHeartError(f"--var expects KEY=VALUE, got {pair!r}")
        key, _, raw = pair.partition("=")
        # YAML-parse the value so numbers and booleans keep their type.
        try:
            variables[key.strip()] = yaml.safe_load(raw)
        except yaml.YAMLError:
            variables[key.strip()] = raw
    return variables


def _parse_rect(text: str) -> Rect:
    parts = [p for p in text.replace(" ", "").split(",") if p]
    if len(parts) != 4:
        raise TTHeartError(f"--region expects x,y,width,height -- got {text!r}")
    try:
        return Rect.from_sequence([int(p) for p in parts])
    except ValueError as exc:
        raise TTHeartError(f"--region values must be whole numbers: {text!r}") from exc


def _count_steps(steps) -> int:
    total = 0
    for step in steps:
        total += 1
        for children in step.children.values():
            total += _count_steps(children)
    return total


#: Every subcommand name, used to detect "the user didn't pass one".
_COMMANDS = {
    "windows", "detect", "prepare", "shot", "snip", "point", "find",
    "templates", "flows", "actions", "validate", "run",
}


def main(argv: Optional[List[str]] = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)

    # `python main.py` with nothing else is almost always "just find LDPlayer
    # and park it" -- do that instead of dumping a usage error.
    if not any(token in _COMMANDS for token in raw_argv) and not any(
        token in ("-h", "--help", "--version") for token in raw_argv
    ):
        print(
            "No command given -- detecting and parking LDPlayer (same as `prepare`).\n"
            "Run `python main.py -h` to see everything else you can do.\n",
            flush=True,
        )
        raw_argv = raw_argv + ["prepare"]

    parser = build_parser()
    args = parser.parse_args(raw_argv)

    level = "DEBUG" if args.verbose else ("WARNING" if args.quiet else None)
    dry_run = bool(getattr(args, "dry_run", False))

    try:
        with Application.create(args.config, log_level=level, dry_run=dry_run) as app:
            if dry_run:
                log.warning("DRY RUN: clicks and keystrokes are logged, not sent")

            if getattr(args, "no_focus", False):
                app.config.window.focus_on_prepare = False

            # The first thing any window-touching command does: find LDPlayer
            # and park it at the top-left so every later coordinate is stable.
            # `--screen` searches skip it since they never touch the window.
            needs_window = bool(getattr(args, "needs_window", False)) and not getattr(
                args, "screen", False
            )
            if needs_window and not args.no_window:
                app.startup(require_window=True, prepare=not args.no_prepare)
            elif args.no_window:
                log.debug("Skipping emulator detection (--no-window)")

            return args.func(app, args)
    except TTHeartError as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
