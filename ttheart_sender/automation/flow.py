"""Flow definitions and the YAML parser.

A flow file looks like::

    name: send_heart
    description: Tap the heart button ten times
    vars:
      taps: 10
    defaults:
      confidence: 0.87
    steps:
      - repeat:
          times: ${taps}
          steps:
            - find_click: {template: heart_button, timeout: 5}
            - wait: {min: 0.4, max: 0.9}

Two step syntaxes are accepted and can be mixed:

* **shorthand** -- ``- find_click: {template: x}``, optionally with common keys
  alongside: ``- find_click: {...}\\n  optional: true``
* **explicit** -- ``- action: find_click\\n  with: {template: x}``

Everything is validated at load time (action names, nested step lists, common
keys) so a typo fails before the mouse starts moving.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import yaml

from ..exceptions import FlowParseError
from .registry import ActionSpec, get_action, is_action

log = logging.getLogger(__name__)

FLOW_SUFFIXES = (".yaml", ".yml")

#: Keys allowed on every step regardless of action.
COMMON_STEP_KEYS = {"action", "with", "name", "optional", "retries", "retry_delay", "enabled"}


@dataclass
class Step:
    """One instruction in a flow."""

    action: str
    params: Dict[str, Any] = field(default_factory=dict)
    #: Nested step lists keyed by param name (``steps``, ``then``, ``else``...).
    children: Dict[str, List["Step"]] = field(default_factory=dict)
    name: str = ""
    #: A failing optional step logs a warning instead of aborting the flow.
    optional: bool = False
    retries: int = 0
    retry_delay: float = 0.5
    enabled: bool = True
    #: Human-readable position, used in every error message.
    path: str = "step"
    source: str = ""

    @property
    def label(self) -> str:
        return self.name or self.action

    @property
    def location(self) -> str:
        prefix = f"{self.source} " if self.source else ""
        return f"{prefix}{self.path} ({self.action})"

    @property
    def spec(self) -> ActionSpec:
        spec = get_action(self.action)
        if spec is None:  # pragma: no cover - guarded at parse time
            raise FlowParseError(f"{self.location}: unknown action {self.action!r}")
        return spec


@dataclass
class Flow:
    name: str
    steps: List[Step] = field(default_factory=list)
    description: str = ""
    #: Initial variables, overridable from the CLI with ``--var k=v``.
    vars: Dict[str, Any] = field(default_factory=dict)
    #: Fallback parameter values applied to every step in this flow.
    defaults: Dict[str, Any] = field(default_factory=dict)
    path: Optional[Path] = None

    @property
    def source(self) -> str:
        return self.path.name if self.path else self.name


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def load_flow(path: Union[str, Path]) -> Flow:
    """Parse a flow YAML file."""
    path = Path(path)
    if not path.is_file():
        raise FlowParseError(f"Flow file not found: {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise FlowParseError(f"{path.name}: invalid YAML -- {exc}") from exc
    return parse_flow(raw, source=path.name, path=path)


def load_flow_by_name(flows_dir: Union[str, Path], name: str) -> Flow:
    """Resolve a flow by file name, stem, or path relative to ``flows_dir``."""
    flows_dir = Path(flows_dir)
    candidate = Path(name)

    if candidate.is_absolute() and candidate.is_file():
        return load_flow(candidate)

    direct = flows_dir / candidate
    if direct.is_file():
        return load_flow(direct)
    for suffix in FLOW_SUFFIXES:
        probe = flows_dir / f"{candidate}{suffix}"
        if probe.is_file():
            return load_flow(probe)

    if candidate.is_file():
        return load_flow(candidate)

    available = [f.stem for f in list_flows(flows_dir)]
    hint = f" Available flows: {available}" if available else f" (no flows found in {flows_dir})"
    raise FlowParseError(f"Flow {name!r} not found.{hint}")


def list_flows(flows_dir: Union[str, Path]) -> List[Path]:
    flows_dir = Path(flows_dir)
    if not flows_dir.exists():
        return []
    return sorted(p for p in flows_dir.rglob("*") if p.suffix.lower() in FLOW_SUFFIXES)


def parse_flow(raw: Any, *, source: str = "<inline>", path: Optional[Path] = None) -> Flow:
    if raw is None:
        raise FlowParseError(f"{source}: flow file is empty")
    if not isinstance(raw, dict):
        raise FlowParseError(f"{source}: expected a mapping at the top level, got {type(raw).__name__}")

    known = {"name", "description", "vars", "defaults", "steps"}
    unknown = sorted(set(raw) - known)
    if unknown:
        raise FlowParseError(f"{source}: unknown top-level key(s) {unknown}. Valid: {sorted(known)}")

    if "steps" not in raw:
        raise FlowParseError(f"{source}: missing required key 'steps'")

    name = str(raw.get("name") or (path.stem if path else source))
    variables = raw.get("vars") or {}
    if not isinstance(variables, dict):
        raise FlowParseError(f"{source}: 'vars' must be a mapping")
    defaults = raw.get("defaults") or {}
    if not isinstance(defaults, dict):
        raise FlowParseError(f"{source}: 'defaults' must be a mapping")

    steps = parse_steps(raw["steps"], source=source, path="steps")
    return Flow(
        name=name,
        steps=steps,
        description=str(raw.get("description") or ""),
        vars=dict(variables),
        defaults=dict(defaults),
        path=path,
    )


def parse_steps(raw: Any, *, source: str, path: str) -> List[Step]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise FlowParseError(f"{source} {path}: expected a list of steps, got {type(raw).__name__}")
    return [parse_step(item, source=source, path=f"{path}[{index}]") for index, item in enumerate(raw)]


def parse_step(raw: Any, *, source: str, path: str) -> Step:
    # `- wait` with no parameters at all.
    if isinstance(raw, str):
        if not is_action(raw):
            raise FlowParseError(_unknown_action_message(source, path, raw))
        raw = {raw: {}}

    if not isinstance(raw, dict):
        raise FlowParseError(
            f"{source} {path}: expected a step mapping, got {type(raw).__name__}"
        )

    action_name, params = _split_action(raw, source=source, path=path)
    spec = get_action(action_name)
    if spec is None:
        raise FlowParseError(_unknown_action_message(source, path, action_name))

    # A scalar (or list) shorthand maps onto the action's primary parameter.
    if not isinstance(params, dict):
        if spec.primary is None:
            raise FlowParseError(
                f"{source} {path}: action {action_name!r} needs a mapping of parameters, "
                f"got {type(params).__name__}"
            )
        params = {spec.primary: params}

    children: Dict[str, List[Step]] = {}
    remaining: Dict[str, Any] = {}
    for key, value in params.items():
        if key in spec.child_keys:
            children[key] = parse_steps(value, source=source, path=f"{path}.{key}")
        else:
            remaining[key] = value

    step = Step(
        action=spec.name,
        params=remaining,
        children=children,
        name=str(raw.get("name") or ""),
        optional=_as_bool(raw.get("optional", False), source, path, "optional"),
        retries=_as_int(raw.get("retries", 0), source, path, "retries"),
        retry_delay=_as_float(raw.get("retry_delay", 0.5), source, path, "retry_delay"),
        enabled=_as_bool(raw.get("enabled", True), source, path, "enabled"),
        path=path,
        source=source,
    )
    return step


def _split_action(raw: Dict[str, Any], *, source: str, path: str):
    """Work out which key names the action and which holds its parameters."""
    if "action" in raw:
        action_name = raw["action"]
        if not isinstance(action_name, str):
            raise FlowParseError(f"{source} {path}: 'action' must be a string, got {action_name!r}")
        params = raw.get("with", {})
        # Tolerate parameters written inline next to `action:` instead of under `with:`.
        inline = {k: v for k, v in raw.items() if k not in COMMON_STEP_KEYS}
        if inline:
            if not isinstance(params, dict):
                raise FlowParseError(
                    f"{source} {path}: cannot mix a scalar 'with:' value with inline parameters"
                )
            params = {**inline, **params}
        return action_name, params

    action_keys = [key for key in raw if key not in COMMON_STEP_KEYS and is_action(key)]
    if len(action_keys) == 1:
        return action_keys[0], raw[action_keys[0]]

    if not action_keys:
        candidates = sorted(set(raw) - COMMON_STEP_KEYS)
        raise FlowParseError(_unknown_action_message(source, path, candidates[0] if candidates else "?"))

    raise FlowParseError(
        f"{source} {path}: found several action keys {sorted(action_keys)} in one step. "
        f"Split them into separate list items."
    )


def _unknown_action_message(source: str, path: str, name: Any) -> str:
    from .registry import action_names

    known = action_names()
    suggestion = _closest(str(name), known)
    hint = f" Did you mean {suggestion!r}?" if suggestion else ""
    return f"{source} {path}: unknown action {name!r}.{hint} Known actions: {known}"


def _closest(name: str, options: Sequence[str]) -> Optional[str]:
    import difflib

    matches = difflib.get_close_matches(name, options, n=1, cutoff=0.7)
    return matches[0] if matches else None


def _as_bool(value: Any, source: str, path: str, key: str) -> bool:
    if isinstance(value, bool):
        return value
    raise FlowParseError(f"{source} {path}: '{key}' must be true or false, got {value!r}")


def _as_int(value: Any, source: str, path: str, key: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise FlowParseError(f"{source} {path}: '{key}' must be a whole number, got {value!r}") from exc


def _as_float(value: Any, source: str, path: str, key: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise FlowParseError(f"{source} {path}: '{key}' must be a number, got {value!r}") from exc
