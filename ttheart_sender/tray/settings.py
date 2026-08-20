"""Panel settings that survive a restart.

Deliberately a separate file from ``config.yaml``: that one is hand-written and
full of comments a round-trip through a YAML writer would strip, while these
are toggles the user flips from the UI several times a session. Kept as JSON so
a corrupt or hand-mangled file is easy to spot and safe to delete -- anything
unreadable falls back to the defaults rather than blocking startup.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List

log = logging.getLogger(__name__)

SETTINGS_FILENAME = "ttheart-settings.json"

#: Panel checkbox -> the flow variable it overrides in purchase_box.yaml.
#: The label is the user's wording; the key is what the flow actually reads.
PURCHASE_BOXES = (
    ("premium_box_plus", "Premium Box+", False),
    ("premium_box", "Premium Box", True),
    ("pick_up_capsule", "Pick-up Capsule", True),
    ("happiness_box", "Happiness Box", True),
)

#: Panel radio -> what the mailbox pass does, as a `claim_all` flow flag.
#: "single" taps Check on one item at a time until the mailbox is empty;
#: "all" makes one pass through the game's own Claim All dialog instead.
#: Exactly one of the two runs -- see flows/claim_mailbox.yaml.
CLAIM_PATTERNS = (
    ("single", "Single claim", False),
    ("all", "Claim all", True),
)
#: The pattern a fresh install uses: item by item, because that is the one
#: that returns a heart per request.
CLAIM_PATTERN_DEFAULT = CLAIM_PATTERNS[0][0]


def claim_all_flag(pattern: Any) -> bool:
    """The `claim_all` flow variable behind a pattern key."""
    for key, _label, flag in CLAIM_PATTERNS:
        if key == pattern:
            return flag
    return False


def normalize_claim_pattern(value: Any, default: str = CLAIM_PATTERN_DEFAULT) -> str:
    """A stored or user-supplied pattern forced back onto a real radio.

    Runs on hand-edited JSON, so anything unrecognised falls back rather than
    leaving the panel with no button lit.
    """
    text = str(value).strip().casefold() if value is not None else ""
    for key, _label, _flag in CLAIM_PATTERNS:
        if key == text:
            return key
    return default


#: Bounds for the Return Heart spinners. They are minutes of the hour, so the
#: clock itself sets the range.
MINUTE_MIN = 0
MINUTE_MAX = 59
#: The two marks a fresh install sends on: quarter past and ten to.
RETURN_HEART_MINUTES_DEFAULT = (15, 50)
#: How many marks the panel offers. Two spinners, so two per hour.
RETURN_HEART_MARKS = len(RETURN_HEART_MINUTES_DEFAULT)


def clamp_minute(value: Any, default: int = MINUTE_MIN) -> int:
    """A spinner/settings value forced into a real minute of the hour.

    Anything unparseable falls back to ``default`` rather than raising: this
    runs on hand-edited JSON and on half-typed text straight out of the edit
    box, neither of which should be able to break the panel.
    """
    try:
        number = int(round(float(value)))
    except (TypeError, ValueError):
        return default
    return max(MINUTE_MIN, min(MINUTE_MAX, number))


def clamp_minutes(values: Any) -> List[int]:
    """A whole set of marks, padded and trimmed to :data:`RETURN_HEART_MARKS`.

    The panel has a fixed number of boxes, so the list handed around has to
    have a fixed length however mangled the stored value is.
    """
    items = list(values) if isinstance(values, (list, tuple)) else []
    marks = []
    for index in range(RETURN_HEART_MARKS):
        fallback = RETURN_HEART_MINUTES_DEFAULT[index]
        marks.append(clamp_minute(items[index], fallback) if index < len(items) else fallback)
    return marks


def _default_purchase() -> Dict[str, bool]:
    return {key: default for key, _, default in PURCHASE_BOXES}


def _default_minutes() -> List[int]:
    return list(RETURN_HEART_MINUTES_DEFAULT)


@dataclass
class PanelSettings:
    """Everything the panel remembers between launches."""

    mode: str = "resume"
    always_on_top: bool = True
    #: Whether a Resume/Launch cycle plays a round instead of waiting. All or
    #: nothing: the panel used to offer odds, and now offers a switch.
    auto_play: bool = False
    #: Send hearts on the clock instead of every cycle. Off by default so an
    #: existing install keeps behaving the way it did before this section
    #: existed.
    return_heart: bool = False
    #: Minutes of the hour to send on while ``return_heart`` is set --
    #: :data:`RETURN_HEART_MARKS` of them, one per spinner.
    return_heart_minutes: List[int] = field(default_factory=_default_minutes)
    #: Which of :data:`CLAIM_PATTERNS` the mailbox pass uses. "single" by
    #: default, which is what the flow did before the radio existed.
    claim_pattern: str = CLAIM_PATTERN_DEFAULT
    #: Install a newer release by itself once one is found. The *check* runs
    #: either way -- this only decides whether the panel acts on the answer.
    #: On by default: a bot left running unattended is exactly the thing that
    #: should not be several versions behind, and an install is held back
    #: until the run it would interrupt has finished.
    auto_update: bool = True
    #: Save training samples while playing. Seeded from config.yaml's
    #: `dataset.enabled` the first time the panel runs, and the panel's own
    #: switch after that -- see :meth:`..tray.app.TrayApp._seed_collection`.
    collect_data: bool = False
    purchase: Dict[str, bool] = field(default_factory=_default_purchase)

    # -- conversion ------------------------------------------------------
    @classmethod
    def from_dict(cls, raw: Any) -> "PanelSettings":
        settings = cls()
        if not isinstance(raw, dict):
            return settings

        # The two collection-valued fields are read separately below, because
        # the loop decides what a value means from the default's type.
        nested = {"purchase", "return_heart_minutes"}
        known = {f.name for f in fields(cls)}
        for key, value in raw.items():
            if key not in known or key in nested:
                continue
            current = getattr(settings, key)
            # Keep the default whenever the stored value is the wrong shape --
            # a hand-edited file should not be able to crash the panel.
            if isinstance(current, bool):
                setattr(settings, key, bool(value))
            elif isinstance(current, str):
                setattr(settings, key, str(value))

        stored = raw.get("purchase")
        if isinstance(stored, dict):
            for key in settings.purchase:
                if key in stored:
                    settings.purchase[key] = bool(stored[key])
        # clamp_minutes copes with a missing, short or nonsense list on its
        # own, so there is nothing to check first.
        settings.return_heart_minutes = clamp_minutes(raw.get("return_heart_minutes"))
        # The loop above already copied it across as a plain string; this is
        # what keeps a typo in the file from lighting neither radio.
        settings.claim_pattern = normalize_claim_pattern(settings.claim_pattern)
        return settings

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    # -- persistence -----------------------------------------------------
    @classmethod
    def load(cls, path: Path) -> "PanelSettings":
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return cls()
        except (OSError, ValueError) as exc:
            log.warning("Ignoring unreadable %s (%s); using defaults", path.name, exc)
            return cls()
        return cls.from_dict(raw)

    def save(self, path: Path) -> bool:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
            return True
        except OSError as exc:
            # A read-only install directory is not worth killing the app over.
            log.warning("Could not save %s: %s", path.name, exc)
            return False


def settings_path(config) -> Path:
    """Where the settings file lives for this install.

    ``output_root`` rather than ``root``: in a one-file .exe the root is a temp
    directory that is wiped on exit, which would quietly lose every setting.
    """
    return config.output_root / SETTINGS_FILENAME
