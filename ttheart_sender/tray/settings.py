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
from typing import Any, Dict

log = logging.getLogger(__name__)

SETTINGS_FILENAME = "ttheart-settings.json"

#: Panel checkbox -> the flow variable it overrides in purchase_box.yaml.
#: The label is the user's wording; the key is what the flow actually reads.
PURCHASE_BOXES = (
    ("premium_box_plus", "Premium box plus", False),
    ("premium_box", "Premium box", True),
    ("pick_up_capsule", "Pick-up capsule", True),
    ("happiness_box", "Happiness capsule", True),
)


def _default_purchase() -> Dict[str, bool]:
    return {key: default for key, _, default in PURCHASE_BOXES}


@dataclass
class PanelSettings:
    """Everything the panel remembers between launches."""

    mode: str = "resume"
    always_on_top: bool = True
    auto_play: bool = False
    purchase: Dict[str, bool] = field(default_factory=_default_purchase)

    # -- conversion ------------------------------------------------------
    @classmethod
    def from_dict(cls, raw: Any) -> "PanelSettings":
        settings = cls()
        if not isinstance(raw, dict):
            return settings

        known = {f.name for f in fields(cls)}
        for key, value in raw.items():
            if key not in known or key == "purchase":
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
