"""Game-specific logic: reading the Tsum Tsum board and playing it.

Everything here is about *this game*, as opposed to the generic
screen/window/automation machinery in the rest of the package. Flow actions
that play the game live in :mod:`ttheart_sender.automation.tsum_actions` and
build on :func:`ttheart_sender.game.tsum.play_loop`.
"""

from __future__ import annotations
