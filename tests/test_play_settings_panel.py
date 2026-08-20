"""The panel's "Play Settings" section, from the radio to the board reader.

The chain is long and every link is silent if it breaks: panel radio ->
`AutomationService._variables()` -> flow `vars:` -> `${...}` in the
`play_tsum` step's `options:` -> `resolve_kinds`. A break anywhere leaves the
panel showing "Normal" while the bot reads the board the other way, with
nothing in the log to say so -- so the last test here walks the whole chain
rather than trusting the pieces.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from ttheart_sender.automation.params import interpolate
from ttheart_sender.game import tsum
from ttheart_sender.tray.service import BOWL_REJECT_VAR, TSUM_MODE_VAR
from ttheart_sender.tray.settings import (
    BOWL_REJECT_OFF,
    BOWL_REJECT_ON,
    TSUM_MODE_DEFAULT,
    TSUM_MODES,
    PanelSettings,
    bowl_reject_value,
    normalize_tsum_mode,
)

FLOW = Path(__file__).resolve().parent.parent / "flows" / "play.yaml"


def _play_options():
    flow = yaml.safe_load(FLOW.read_text(encoding="utf-8"))
    step = next(s["play_tsum"] for s in flow["steps"]
                if isinstance(s, dict) and "play_tsum" in s)
    return flow.get("vars", {}), step["options"]


# -- the two controls -----------------------------------------------------
def test_the_radio_offers_exactly_normal_and_color_with_normal_default():
    assert [key for key, _label in TSUM_MODES] == ["normal", "color"]
    assert [label for _key, label in TSUM_MODES] == ["Normal", "Color"]
    assert TSUM_MODE_DEFAULT == "normal"
    assert PanelSettings().tsum_mode == "normal"


def test_the_tick_box_is_the_two_settings_worth_having():
    assert bowl_reject_value(True) == BOWL_REJECT_ON == 60
    assert bowl_reject_value(False) == BOWL_REJECT_OFF == 0


def test_a_mangled_stored_mode_still_lights_a_button():
    """Runs on hand-edited JSON, so it must not leave the panel blank."""
    for junk in ("", "  COLOR  ", "colour", None, 7, "normal"):
        assert normalize_tsum_mode(junk) in {key for key, _ in TSUM_MODES}
    assert normalize_tsum_mode("  COLOR  ") == "color"
    assert normalize_tsum_mode("colour") == TSUM_MODE_DEFAULT


def test_settings_survive_a_round_trip(tmp_path):
    path = tmp_path / "s.json"
    before = PanelSettings(tsum_mode="color", bowl_reject=False)
    assert before.save(path)
    after = PanelSettings.load(path)
    assert after.tsum_mode == "color"
    assert after.bowl_reject is False


# -- the flow forwards them ----------------------------------------------
def test_play_yaml_declares_both_as_vars_the_tray_can_override():
    variables, options = _play_options()
    for name in (TSUM_MODE_VAR, BOWL_REJECT_VAR):
        assert name in variables, f"{name} must be a flow var for --var/the tray to reach it"
        assert options.get(name) == "${%s}" % name, \
            f"{name} must be forwarded into options, not hard-coded there"


def test_a_whole_reference_keeps_the_variable_type():
    """`bowl_reject: "${bowl_reject}"` has to arrive as a number, not "60".

    `setattr` on the options namespace does no coercion, so a string would
    reach numpy and fail somewhere unrecognisable.
    """
    _variables, options = _play_options()
    out = interpolate(options, {TSUM_MODE_VAR: "color", BOWL_REJECT_VAR: 60})
    assert out[BOWL_REJECT_VAR] == 60 and isinstance(out[BOWL_REJECT_VAR], int)
    assert out[TSUM_MODE_VAR] == "color"


# -- the whole chain ------------------------------------------------------
@pytest.mark.parametrize(
    "mode, ticked, expect_kinds, expect_bowl",
    [("normal", True, 0, 60), ("normal", False, 0, 0),
     ("color", True, 5, 60), ("color", False, 5, 0)],
)
def test_the_panel_choice_reaches_the_board_reader(mode, ticked, expect_kinds, expect_bowl):
    """Radio and tick box -> what `play_loop` actually runs with."""
    _variables, options = _play_options()

    # What the service hands the run for this pair of controls.
    handed = {TSUM_MODE_VAR: mode, BOWL_REJECT_VAR: bowl_reject_value(ticked)}

    # What the step then applies to the play options.
    opts = tsum.play_defaults()
    for key, value in interpolate(options, handed).items():
        setattr(opts, key, value)

    assert tsum.resolve_kinds(opts) == expect_kinds
    assert opts.bowl_reject == expect_bowl
    assert mode in tsum.describe_mode(opts)
