"""The run modes offered by the tray menu.

Each mode is just a saved set of ``run`` arguments -- picking "Resume" in the
tray does exactly what ``python main.py run resume`` does from a console.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

#: Mode selected when the app starts with no saved preference.
DEFAULT_MODE = "resume"


@dataclass(frozen=True)
class Mode:
    """One entry in the tray's Mode submenu."""

    key: str
    label: str
    flow: str
    #: How many times to run the flow; 0 means "until stopped".
    loops: int = 1
    loop_delay: float = 0.0
    description: str = ""

    @property
    def command(self) -> str:
        """The equivalent console command, shown in logs and tooltips."""
        parts = ["run", self.flow]
        if self.loops != 1:
            parts += ["--loops", str(self.loops)]
        if self.loop_delay:
            parts += ["--loop-delay", str(self.loop_delay)]
        return "python main.py " + " ".join(parts)


MODES: Tuple[Mode, ...] = (
    Mode(
        key="resume",
        label="Resume",
        flow="resume",
        # resume.yaml is a `repeat: {forever: true}` bot loop already, so one
        # pass through the flow never returns until the stop key is pressed.
        loops=1,
        description="Send hearts without restarting LDPlayer",
    ),
    Mode(
        key="start",
        label="Start",
        flow="start",
        loops=1,
        description="Launch Tsum Tsum and clear the startup prompts",
    ),
    Mode(
        key="play",
        label="Play",
        flow="play",
        # play.yaml is a single round, so the repetition has to come from here.
        loops=0,
        loop_delay=1.0,
        description="Play rounds back to back until stopped",
    ),
)


def get_mode(key: str) -> Optional[Mode]:
    for mode in MODES:
        if mode.key == key:
            return mode
    return None
