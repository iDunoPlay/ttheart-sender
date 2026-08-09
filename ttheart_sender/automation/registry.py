"""Action registry.

Adding a new step type to the YAML language is one decorated function::

    @action("swipe_left", summary="Swipe from the right edge to the left")
    def swipe_left(ctx: RunContext, params: Params) -> ActionResult:
        ...

Nothing else needs to change: the parser, the CLI's ``actions`` listing and the
validation errors all read from this registry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

ActionFn = Callable[..., "ActionResult"]


@dataclass(frozen=True)
class ActionResult:
    """Outcome of one step.

    ``success=False`` is a *soft* failure (template not found, condition not
    met) which the runner may retry or, for optional steps, ignore. Raise
    :class:`~ttheart_sender.exceptions.ActionError` for hard failures.
    """

    success: bool = True
    value: Any = None
    message: str = ""

    def __bool__(self) -> bool:
        return self.success

    @classmethod
    def ok(cls, value: Any = None, message: str = "") -> "ActionResult":
        return cls(True, value, message)

    @classmethod
    def fail(cls, message: str = "", value: Any = None) -> "ActionResult":
        return cls(False, value, message)


@dataclass(frozen=True)
class ActionSpec:
    name: str
    fn: ActionFn
    aliases: Tuple[str, ...] = ()
    #: Param name a scalar shorthand maps to, e.g. ``- wait: 0.5``.
    primary: Optional[str] = None
    #: Param keys whose values are nested step lists (parsed recursively).
    child_keys: Tuple[str, ...] = ()
    summary: str = ""

    @property
    def has_children(self) -> bool:
        return bool(self.child_keys)


_REGISTRY: Dict[str, ActionSpec] = {}
_ALIASES: Dict[str, str] = {}


def action(
    name: str,
    *,
    aliases: Sequence[str] = (),
    primary: Optional[str] = None,
    child_keys: Sequence[str] = (),
    summary: str = "",
) -> Callable[[ActionFn], ActionFn]:
    """Register a function as a flow action."""

    def decorator(fn: ActionFn) -> ActionFn:
        spec = ActionSpec(
            name=name,
            fn=fn,
            aliases=tuple(aliases),
            primary=primary,
            child_keys=tuple(child_keys),
            summary=summary or _first_docstring_line(fn),
        )
        register(spec)
        return fn

    return decorator


def _first_docstring_line(fn: ActionFn) -> str:
    lines = (fn.__doc__ or "").strip().splitlines()
    return lines[0].strip() if lines else ""


def register(spec: ActionSpec, *, replace: bool = False) -> None:
    if not replace and spec.name in _REGISTRY:
        raise ValueError(f"Action {spec.name!r} is already registered")
    _REGISTRY[spec.name] = spec
    for alias in spec.aliases:
        if not replace and alias in _ALIASES:
            raise ValueError(f"Action alias {alias!r} is already registered")
        _ALIASES[alias] = spec.name


def get_action(name: str) -> Optional[ActionSpec]:
    key = str(name)
    if key in _REGISTRY:
        return _REGISTRY[key]
    resolved = _ALIASES.get(key)
    return _REGISTRY.get(resolved) if resolved else None


def is_action(name: Any) -> bool:
    return isinstance(name, str) and get_action(name) is not None


def registered_actions() -> List[ActionSpec]:
    return sorted(_REGISTRY.values(), key=lambda spec: spec.name)


def action_names() -> List[str]:
    return sorted(set(_REGISTRY) | set(_ALIASES))
