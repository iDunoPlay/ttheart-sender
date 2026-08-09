"""Typed, validated access to a step's parameters.

Two things make this worth a dedicated class:

* **Variable interpolation.** ``${name}`` in any string is substituted from the
  run's variables at read time (not parse time), so loop counters work.
* **Unknown-key detection.** Every getter records the key it consumed;
  :meth:`ensure_consumed` then reports typos like ``confidance:`` instead of
  silently using the default.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Set

from ..exceptions import ActionError
from ..geometry import Point, Rect

if TYPE_CHECKING:  # pragma: no cover
    from .flow import Step

_VAR_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_.]*)\}")
_MISSING = object()


def interpolate(value: Any, variables: Dict[str, Any]) -> Any:
    """Replace ``${var}`` references anywhere inside ``value``."""
    if isinstance(value, str):
        # A string that is exactly one reference keeps the variable's type.
        whole = _VAR_PATTERN.fullmatch(value.strip())
        if whole:
            return variables.get(whole.group(1), value)
        return _VAR_PATTERN.sub(lambda m: str(variables.get(m.group(1), m.group(0))), value)
    if isinstance(value, list):
        return [interpolate(item, variables) for item in value]
    if isinstance(value, dict):
        return {key: interpolate(item, variables) for key, item in value.items()}
    return value


class Params:
    """Reader over one step's parameter mapping."""

    def __init__(
        self,
        step: "Step",
        variables: Optional[Dict[str, Any]] = None,
        defaults: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.step = step
        self.variables = variables if variables is not None else {}
        self.defaults = defaults or {}
        self._raw = step.params
        self._consumed: Set[str] = set(step.children)

    # -- low level -------------------------------------------------------
    def has(self, key: str) -> bool:
        return key in self._raw

    def raw(self, key: str, default: Any = None) -> Any:
        self._consumed.add(key)
        if key not in self._raw:
            return self.defaults.get(key, default)
        return interpolate(self._raw[key], self.variables)

    def _get(self, key: str, default: Any) -> Any:
        self._consumed.add(key)
        if key in self._raw:
            return interpolate(self._raw[key], self.variables)
        if key in self.defaults:
            return self.defaults[key]
        return default

    def _error(self, key: str, expected: str, value: Any) -> ActionError:
        return ActionError(
            f"{self.step.location}: parameter {key!r} expected {expected}, got {value!r}"
        )

    # -- typed getters ---------------------------------------------------
    def string(self, key: str, default: Any = _MISSING) -> str:
        value = self._get(key, default)
        if value is _MISSING:
            raise ActionError(f"{self.step.location}: missing required parameter {key!r}")
        if value is None:
            return ""
        return str(value)

    def integer(self, key: str, default: Any = _MISSING) -> int:
        value = self._get(key, default)
        if value is _MISSING:
            raise ActionError(f"{self.step.location}: missing required parameter {key!r}")
        try:
            return int(value)
        except (TypeError, ValueError) as exc:
            raise self._error(key, "a whole number", value) from exc

    def number(self, key: str, default: Any = _MISSING) -> float:
        value = self._get(key, default)
        if value is _MISSING:
            raise ActionError(f"{self.step.location}: missing required parameter {key!r}")
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise self._error(key, "a number", value) from exc

    def boolean(self, key: str, default: bool = False) -> bool:
        value = self._get(key, default)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().casefold()
            if lowered in {"true", "yes", "on", "1"}:
                return True
            if lowered in {"false", "no", "off", "0"}:
                return False
        if isinstance(value, (int, float)):
            return bool(value)
        raise self._error(key, "true or false", value)

    def optional_string(self, key: str, default: Optional[str] = None) -> Optional[str]:
        value = self._get(key, default)
        return None if value is None else str(value)

    def optional_number(self, key: str, default: Optional[float] = None) -> Optional[float]:
        value = self._get(key, default)
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise self._error(key, "a number or null", value) from exc

    def number_list(self, key: str, default: Optional[Sequence[float]] = None) -> Optional[List[float]]:
        value = self._get(key, default)
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return [float(value)]
        if not isinstance(value, (list, tuple)):
            raise self._error(key, "a list of numbers", value)
        try:
            return [float(item) for item in value]
        except (TypeError, ValueError) as exc:
            raise self._error(key, "a list of numbers", value) from exc

    def string_list(self, key: str, default: Optional[Sequence[str]] = None) -> List[str]:
        value = self._get(key, default if default is not None else [])
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        if not isinstance(value, (list, tuple)):
            raise self._error(key, "a string or list of strings", value)
        return [str(item) for item in value]

    def mapping(self, key: str, default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        value = self._get(key, default if default is not None else {})
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise self._error(key, "a mapping", value)
        return value

    def duration(self, key: str, default: Any = _MISSING) -> float:
        """Seconds, either a number or ``{min: a, max: b}`` for a random wait."""
        value = self._get(key, default)
        if value is _MISSING:
            raise ActionError(f"{self.step.location}: missing required parameter {key!r}")
        if isinstance(value, dict):
            import random

            low = float(value.get("min", 0.0))
            high = float(value.get("max", low))
            return random.uniform(min(low, high), max(low, high))
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise self._error(key, "seconds, or {min, max}", value) from exc

    def point(self, key: str, default: Any = _MISSING) -> Optional[Point]:
        """``[x, y]`` or ``{x: .., y: ..}``."""
        value = self._get(key, default)
        if value is _MISSING:
            raise ActionError(f"{self.step.location}: missing required parameter {key!r}")
        if value is None:
            return None
        if isinstance(value, Point):
            return value
        if isinstance(value, dict) and {"x", "y"} <= set(value):
            return Point(int(value["x"]), int(value["y"]))
        if isinstance(value, (list, tuple)) and len(value) == 2:
            return Point(int(value[0]), int(value[1]))
        raise self._error(key, "[x, y] or {x: .., y: ..}", value)

    def offset(self, key: str, default: Point = Point(0, 0)) -> Point:
        value = self.point(key, None)
        return value if value is not None else default

    def rect(self, key: str, default: Any = None) -> Optional[Rect]:
        """``[x, y, w, h]`` or ``{x, y, width, height}``."""
        value = self._get(key, default)
        if value is None:
            return None
        if isinstance(value, Rect):
            return value
        if isinstance(value, dict):
            try:
                return Rect(
                    int(value["x"]), int(value["y"]), int(value["width"]), int(value["height"])
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise self._error(key, "{x, y, width, height}", value) from exc
        if isinstance(value, (list, tuple)) and len(value) == 4:
            return Rect.from_sequence(value)
        raise self._error(key, "[x, y, width, height]", value)

    def steps(self, key: str) -> List["Step"]:
        self._consumed.add(key)
        return self.step.children.get(key, [])

    # -- validation ------------------------------------------------------
    def ensure_consumed(self) -> None:
        """Raise if the step declared parameters the action never read."""
        unknown = sorted(set(self._raw) - self._consumed)
        if unknown:
            raise ActionError(
                f"{self.step.location}: unknown parameter(s) {unknown} for action "
                f"{self.step.action!r}"
            )
