"""Per-tsum settings bundles: they must be loud, or they are worse than nothing.

A profile changes how a round plays. The failures that matter are all silent
ones -- a typo'd option that is ignored, a profile that does nothing, a bundle
measured on Beast applied to a different tsum. Each of those produces a round
whose settings are not the settings anyone believes it ran under, which is the
one thing this project cannot afford: every conclusion here rests on knowing
what was on.

So these tests are mostly about noise, not about values.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ttheart_sender.game import profiles  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


class Opts:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


def write(root: Path, name: str, settings: dict, icon=None, **extra):
    root.mkdir(parents=True, exist_ok=True)
    body = {"name": name, "settings": settings}
    if icon is not None:
        body["icon_lab"] = icon
    body.update(extra)
    (root / f"{name}.json").write_text(json.dumps(body), encoding="utf-8")


# ---------------------------------------------------------------- the shipped
def test_the_beast_backup_ships_and_is_loadable():
    prof = profiles.load("beast", ROOT / "profiles")
    assert prof["icon_lab"] == [199, 134, 165]
    assert prof["settings"]["use_base"] is True
    assert prof["rounds_played"] == 737


def test_the_beast_backup_is_what_the_rounds_actually_played():
    """The point of a backup is that it is not a retyped guess."""
    s = profiles.load("beast", ROOT / "profiles")["settings"]
    assert s["link"] == 1.75 and s["link_px"] == 105
    assert s["block"] == 1.25 and s["mode"] == "touch"
    assert s["fit_effort"] == 3 and s["bowl_reject"] == 40
    # Everything that ships off must still be off in the backup.
    assert s["character"] == "" and s["chain_model"] == ""
    assert s["reject_model"] == "" and s["kinds"] == 0


def _others():
    """Every shipped profile that is not Beast, found rather than named.

    These files get renamed the moment somebody works out which tsum they are
    -- `unknown-223` became `beans_camo_vil` -- and a test that hard-codes the
    stem fails on the rename rather than on anything real.
    """
    return [n for n in profiles.available(ROOT / "profiles") if n != "beast"]


def test_every_shipped_profile_carries_a_distinct_icon():
    seen = [(n, profiles.load(n, ROOT / "profiles")["icon_lab"])
            for n in profiles.available(ROOT / "profiles")]
    for i, (na, a) in enumerate(seen):
        for nb, b in seen[i + 1:]:
            d = sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5
            assert d > profiles.MATCH, f"{na} and {nb} resolve to one profile"


def test_the_other_profiles_still_match_beast_so_the_difference_is_the_tsum():
    """43 rounds scored worse on IDENTICAL settings. Divergence is a decision
    to be made and A/B'd, not something to pre-empt in a data file."""
    a = profiles.load("beast", ROOT / "profiles")["settings"]
    for name in _others():
        assert profiles.load(name, ROOT / "profiles")["settings"] == a, name


def test_every_shipped_profile_names_itself_after_its_own_file():
    """`name` is what the loader falls back to and what the log prints. A file
    renamed without its `name` following would report the old tsum."""
    for name in profiles.available(ROOT / "profiles"):
        assert profiles.load(name, ROOT / "profiles")["name"] == name


# ------------------------------------------------------------------- failures
def test_an_unknown_profile_raises_and_says_what_exists(tmp_path):
    write(tmp_path, "beast", {"use_base": True})
    with pytest.raises(profiles.ProfileError) as e:
        profiles.load("mickey", tmp_path)
    assert "beast" in str(e.value)


def test_a_typo_in_an_option_name_raises_rather_than_being_ignored(tmp_path):
    """The failure this whole module exists to prevent: a profile that says it
    turns something off, does not, and plays a round anyway."""
    write(tmp_path, "p", {"use_bases": False})
    with pytest.raises(profiles.ProfileError) as e:
        profiles.load("p", tmp_path)
    assert "use_bases" in str(e.value)


def test_a_profile_cannot_reach_an_option_the_play_loop_does_not_have(tmp_path):
    write(tmp_path, "p", {"duration": 9999})
    with pytest.raises(profiles.ProfileError):
        profiles.load("p", tmp_path)


def test_a_profile_without_settings_raises(tmp_path):
    (tmp_path).mkdir(parents=True, exist_ok=True)
    (tmp_path / "p.json").write_text(json.dumps({"name": "p"}), encoding="utf-8")
    with pytest.raises(profiles.ProfileError):
        profiles.load("p", tmp_path)


# -------------------------------------------------------------------- applying
def test_apply_changes_only_what_differs_and_reports_it(tmp_path):
    write(tmp_path, "p", {"use_base": False, "kinds": 0})
    opts = Opts(use_base=True, kinds=0)
    changed = profiles.apply(opts, profiles.load("p", tmp_path))
    assert opts.use_base is False
    assert len(changed) == 1 and "use_base" in changed[0]


def test_a_profile_that_changes_nothing_says_so(tmp_path):
    write(tmp_path, "p", {"use_base": True})
    said = []
    profiles.apply(Opts(use_base=True), profiles.load("p", tmp_path), said.append)
    assert any("nothing changed" in s for s in said)


def test_every_change_is_announced(tmp_path):
    write(tmp_path, "p", {"use_base": False, "kinds": 4})
    said = []
    profiles.apply(Opts(use_base=True, kinds=0), profiles.load("p", tmp_path),
                   said.append)
    assert any("use_base" in s for s in said) and any("kinds" in s for s in said)


# -------------------------------------------------------------------- identity
def test_identify_finds_the_tsum_the_icon_belongs_to(tmp_path):
    write(tmp_path, "beast", {"use_base": True}, icon=[199, 134, 165])
    write(tmp_path, "other", {"use_base": True}, icon=[223, 137, 138])
    assert profiles.identify([199, 134, 166], tmp_path) == "beast"
    assert profiles.identify([223, 136, 138], tmp_path) == "other"


def test_identify_refuses_a_tsum_it_has_no_profile_for(tmp_path):
    write(tmp_path, "beast", {"use_base": True}, icon=[199, 134, 165])
    assert profiles.identify([20, 200, 30], tmp_path) is None


def test_the_wrong_profile_for_the_equipped_tsum_is_warned_not_swallowed():
    prof = {"name": "beast", "icon_lab": [199, 134, 165], "settings": {}}
    said = []
    assert profiles.check_icon(prof, [223, 137, 138], said.append) is False
    assert any("WARNING" in s for s in said)


def test_the_right_profile_passes_quietly():
    prof = {"name": "beast", "icon_lab": [199, 134, 165], "settings": {}}
    said = []
    assert profiles.check_icon(prof, [199, 135, 165], said.append) is True
    assert said == []


def test_a_missing_icon_reading_is_not_treated_as_a_mismatch():
    """The icon can fail to read. That is not evidence of the wrong tsum, and
    crying wolf on it would train the warning to be ignored."""
    prof = {"name": "beast", "icon_lab": [199, 134, 165], "settings": {}}
    assert profiles.check_icon(prof, None) is True


# ------------------------------------------------------------------- shipping
def test_the_flow_ships_the_profile_off():
    text = (ROOT / "flows" / "play.yaml").read_text(encoding="utf-8")
    assert 'profile: ""' in text


def test_every_flow_that_forwards_settings_declares_it():
    """A var declared in one flow and not the next is how a setting comes to be
    silently overridden at a hop."""
    for name in ("play", "resume", "launch"):
        text = (ROOT / "flows" / f"{name}.yaml").read_text(encoding="utf-8")
        assert 'profile: ""' in text, name
        assert 'profile: "${profile}"' in text, name


def test_the_build_ships_the_profiles_directory():
    text = (ROOT / "build.py").read_text(encoding="utf-8")
    assert '"profiles"' in text


def test_the_play_options_actually_carry_a_profile_field():
    """The flow sets `profile` by setattr onto `play_defaults()`. If the CLI
    parser does not define it, the flow would set an attribute nothing reads
    and the profile would silently never apply."""
    from ttheart_sender.game.tsum import play_defaults
    assert play_defaults().profile == ""
