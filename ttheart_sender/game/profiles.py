"""Per-equipped-tsum settings bundles, and the backup of what Beast played.

**Why this exists.** 737 rounds were played with one tsum equipped and 43 with
another, and the second scored 12 points less FEVER share on identical settings
(`docs/BASE-TSUM-INVESTIGATION.md`). The settings that suit one equipped tsum are
not the settings that suit another, and until now there was nowhere to write
that down: a change good for Beast was a change for everybody.

A profile is a named bundle of the same tunables `flows/play.yaml` declares, and
nothing more. It cannot introduce a setting the play loop does not already have
-- :func:`apply` refuses an unknown key rather than ignoring it, because a
silently-dropped setting is how a profile comes to mean something different from
what it says.

**Identity is the icon colour, not a name.** `read_base_kind` reads the median
Lab of the skill icon, and that measurement is stable to about 1.0 Lab unit
within a session while two different tsums sit 36.9 apart -- measured, over 780
sessions. So a profile carries the icon colour it was built for, and the play
loop can check that the profile you selected is the tsum you actually equipped.
Getting that wrong silently is worse than not having profiles at all.

Ships inert: no profile selected means the settings come from the flow exactly
as they always have.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

#: Where profiles live. Shipped as a data directory, like `flows` and `models`.
DIR = Path("profiles")

#: Every setting a profile is allowed to carry. Deliberately a whitelist of the
#: tunables `flows/play.yaml` already declares: a profile is a different set of
#: values for the knobs that exist, never a way to reach a new one. Anything
#: outside this raises rather than being applied or dropped.
ALLOWED = frozenset({
    # detection
    "k", "include_dark", "bowl_reject", "fit_effort", "scale", "recolour",
    "kinds", "merge", "radius_cover", "radius_lock", "purity",
    # linking and chains
    "link", "link_px", "block", "min_chain", "max_chain", "mode",
    "first_leg_px", "max_tsums", "min_tsums",
    # the equipped tsum's own rules
    "use_base", "base_only",
    # models, all of which ship off
    "character", "character_confidence", "character_min_visible",
    "reject_model", "reject_floor", "chain_model", "chain_bonus", "palette",
    # play rules
    "settle_board", "settle", "step_px", "verify_clears", "verify_extend",
    "verify_reach", "skill_gold", "skill_inner", "skill_outer",
})

#: How close an icon reading must sit to a profile's `icon_lab` to count as the
#: same equipped tsum. The within-session spread of the reading is ~1.0 Lab unit
#: and the two tsums in the corpus are 36.9 apart, so this is loose enough never
#: to reject the right profile and tight enough to catch the wrong one.
MATCH = 8.0


class ProfileError(Exception):
    """A profile that cannot be honoured. Never swallowed: a round played with
    settings other than the ones named is a round that measures nothing."""


def available(root: Optional[Path] = None) -> list[str]:
    root = Path(root or DIR)
    if not root.is_dir():
        return []
    return sorted(p.stem for p in root.glob("*.json"))


def load(name: str, root: Optional[Path] = None) -> dict:
    """One profile by file stem. Unknown names raise, and say what exists."""
    root = Path(root or DIR)
    path = root / f"{name}.json"
    if not path.is_file():
        have = ", ".join(available(root)) or "none"
        raise ProfileError(f"no profile {name!r} in {root} -- have: {have}")
    try:
        prof = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ProfileError(f"profile {name!r} is unreadable: {exc}") from exc
    if not isinstance(prof.get("settings"), dict):
        raise ProfileError(f"profile {name!r} has no settings block")
    bad = sorted(set(prof["settings"]) - ALLOWED)
    if bad:
        raise ProfileError(
            f"profile {name!r} sets {', '.join(bad)}, which the play loop has "
            f"no such option for. Fix the name or add it to profiles.ALLOWED "
            f"-- it will NOT be applied silently.")
    prof.setdefault("name", name)
    return prof


def identify(icon_lab, root: Optional[Path] = None) -> Optional[str]:
    """Which profile's tsum this icon colour is, or None.

    Used to check the selection rather than to make it. Choosing a profile
    automatically from the icon would mean a misread icon silently changes
    every setting mid-session, which is a worse failure than a wrong name.
    """
    if not icon_lab:
        return None
    best, best_d = None, MATCH
    for name in available(root):
        try:
            ref = load(name, root).get("icon_lab")
        except ProfileError:
            continue
        if not ref or len(ref) != len(icon_lab):
            continue
        d = sum((float(a) - float(b)) ** 2
                for a, b in zip(ref, icon_lab)) ** 0.5
        if d < best_d:
            best, best_d = name, d
    return best


def apply(opts: Any, prof: dict, say=None) -> list[str]:
    """Put a profile's settings onto `opts`, loudly. Returns what changed.

    Every change is announced. A profile that quietly agrees with the flow is
    indistinguishable from one that was never loaded, and this project has
    already lost rounds to a setting that looked applied and was not.
    """
    changed = []
    for key, value in sorted(prof.get("settings", {}).items()):
        if key not in ALLOWED:                     # load() checks; belt and braces
            raise ProfileError(f"profile sets unknown option {key!r}")
        old = getattr(opts, key, None)
        if old == value:
            continue
        setattr(opts, key, value)
        changed.append(f"{key}: {old!r} -> {value!r}")
    if say:
        who = prof.get("name", "?")
        if changed:
            say(f"    profile {who}: {len(changed)} setting(s) changed")
            for line in changed:
                say(f"      {line}")
        else:
            say(f"    profile {who}: already matches the flow, nothing changed")
    return changed


def check_icon(prof: dict, icon_lab, say=None) -> bool:
    """Warn when the selected profile is not the tsum actually equipped.

    Not an error, and deliberately so: the icon can be misread, the player may
    be testing on purpose, and refusing to play is a worse outcome than a loud
    line in the log. But it is said every time, because a profile applied to
    the wrong tsum is exactly the silent mismatch this module exists to stop.
    """
    ref = prof.get("icon_lab")
    if not ref or not icon_lab or len(ref) != len(icon_lab):
        return True
    d = sum((float(a) - float(b)) ** 2 for a, b in zip(ref, icon_lab)) ** 0.5
    ok = d <= MATCH
    if say and not ok:
        say(f"    WARNING: profile {prof.get('name', '?')} was built for icon "
            f"Lab {[round(float(v)) for v in ref]}, but the equipped tsum "
            f"reads {[round(float(v)) for v in icon_lab]} ({d:.1f} away). "
            f"These settings were measured on a different tsum.")
    return ok
