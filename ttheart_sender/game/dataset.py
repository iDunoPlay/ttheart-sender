"""Training samples captured while playing, for improving tsum detection.

Why this exists in the app rather than as an offline script: the label cannot
be recovered afterwards. Holding a tsum makes the game light up every tsum you
could link it to -- same character *and* reachable -- and that highlight lives
for a fraction of a second in the middle of a drag. It is the one moment the
game tells us what its own answer is, so a sample is the pair of frames either
side of it: the board as detection saw it, and the board with the game's
answer painted on top. Everything a model needs to learn identity is the
difference between those two images.

That makes the labels free and open-ended. A character the collection has
never held before labels itself the first time it is pressed, which matters
here because equipping a different tsum changes the colours on the board and
the current palette-clustering approach has to re-derive identity from scratch
every frame.

What is written, per sample:

* ``NNNN_before.jpg`` -- the board crop the chain was chosen from
* ``NNNN_marked.jpg`` -- the same crop while the game shows what it marked
* one line in ``samples.jsonl`` -- detections, the chain, what survived, and
  the settings in force

Failures here must never cost a round: every write is best-effort, and the
first error switches the writer off for the rest of the session rather than
raising into the play loop.

What the first 803MB proved
---------------------------

Schema 1 collected 5,729 samples and not one usable label, for two reasons
that are now guarded here rather than discovered again offline. Both are
measured in ``docs/DATASET-FINDINGS.md``.

* It inherited ``--hold-delay 0.10`` from ``--verify-hold``, which exists to
  keep that feature's per-drag cost down. 0.10s is below the 0.15s floor the
  ``assist`` docs already name as the point where marks have not finished
  rendering, so the "marked" frame was photographed before the game had drawn
  anything. Collection now owns its delay and defaults to 0.25s, as `assist`
  does, because the throughput argument that set 0.10 does not apply to one
  drag in four.
* It photographed the mark once, against a board that was still moving. At
  press time the median tsum had already shifted enough to clear the 8.0
  threshold on 56% of samples, so the diff read motion. Collection now takes
  the same defence `marks_on_board` takes -- several frames, keep only what
  changed in all of them -- and refuses a board too unsettled to read.

Every sample now carries the reading it was judged on (``baseline``,
``marks``), so the next dataset can be scored by `tsum dataset` without
re-decoding a byte of JPEG.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Sequence

import cv2
import numpy as np

from ..version import __version__

log = logging.getLogger(__name__)

#: Bump when the meaning of a field changes, so old sessions stay readable.
#:
#: 2 -- the collector owns its capture settings instead of borrowing
#: ``--verify-hold``'s, and every row carries the mark reading (``baseline``,
#: ``marks``, ``capture``) it was judged on. Schema 1 rows have no usable
#: label; see the module docstring.
SCHEMA = 2

README = """\
Tsum detection training data
============================

Each sample is two images and one line of JSON:

  NNNN_before.jpg   the board as the bot saw it, just before pressing a tsum
  NNNN_marked.jpg   the same board while the game highlighted what it marked
  samples.jsonl     detections, the proposed chain, and what the game accepted

The highlight in `marked` is the game's own answer to "which tsums are the
same character as the one being held, and reachable from it". That is the
label -- it does not need to be annotated by hand.

Check a collection before trusting it:

  python -m ttheart_sender.game.tsum dataset --dir <this folder>

It reports whether the marks are actually in the frames. A `schema` of 1 in
samples.jsonl means they are not -- those sessions were photographed before
the game had drawn the highlight, and only the images are worth keeping.

Nothing here identifies you: the crops are the board area only, no account
name, no score, no window chrome.

To send it on, zip the whole session folder.
"""


#: Below this the game has not finished drawing the highlight, so the frame
#: photographs a board with no answer on it. Named in the ``assist`` help and
#: confirmed the hard way by schema 1, which sampled at 0.10s and produced
#: 5,729 labels indistinguishable from noise.
RENDER_FLOOR = 0.15


class DatasetWriter:
    """Writes samples for one play session into its own folder.

    The capture arguments are the collector's own, deliberately not shared
    with ``--verify-hold``. That feature reads the same highlight but pays for
    the read on every single drag, so it is tuned for speed and settles for a
    delay below :data:`RENDER_FLOOR` and a single frame. Collection pays on
    one drag in four, capped at ``per_round`` -- it can afford to wait for the
    mark to render and photograph it more than once, and schema 1 is the
    record of what happens when it does not.

    ``max_mb`` and ``max_total`` are budgets across the whole dataset
    directory, not this session: without them a night of play writes until the
    disk says no (measured: 803MB and 5,729 samples between 19:46 and 08:59).
    """

    def __init__(self, root: Path, *, per_round: int = 20, every: int = 4,
                 quality: int = 85, delay: float = 0.25, frames: int = 3,
                 gap: float = 0.05, floor_mult: float = 5.0,
                 max_motion: float = 12.0,
                 max_marked: float = 0.5, max_mb: float = 2048.0,
                 max_total: int = 0) -> None:
        self.root = Path(root)
        self.per_round = max(0, int(per_round))
        self.every = max(1, int(every))
        self.quality = int(min(100, max(40, quality)))
        # Clamped rather than rejected: a config that asks for a delay under
        # the render floor is asking for schema 1 back, and silently obeying
        # it costs another night of unusable frames.
        self.delay = max(RENDER_FLOOR, float(delay))
        self.frames = max(1, int(frames))
        self.gap = max(0.0, float(gap))
        # A mark has to beat the board's own floor by this much. Measured with
        # `hold` on four boards the game's marks sit 8x-25x above it, while
        # the fixed 8.0 threshold sits inside the floor's noise on a live
        # board. 0 falls back to the fixed threshold.
        self.floor_mult = max(0.0, float(floor_mult))
        self.max_motion = float(max_motion)
        self.max_marked = float(max_marked)
        self.max_mb = max(0.0, float(max_mb))
        self.max_total = max(0, int(max_total))
        self.written = 0
        self.seen = 0
        #: Samples refused by a quality gate, by reason -- reported at the end
        #: of a round so a collection that gathers nothing says why.
        self.refused: dict[str, int] = {}
        self.enabled = self.per_round > 0
        self.dir: Optional[Path] = None
        self._file = None

    # -- lifecycle -------------------------------------------------------
    def _open(self) -> bool:
        """Create the session folder on the first sample, not before.

        A round that never reaches a drag should leave no empty directory
        behind, and `enabled` in config.yaml should cost nothing until it
        actually collects something.
        """
        if self._file is not None:
            return True
        if not self._budget_left():
            self.enabled = False
            return False
        try:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.dir = self.root / f"{stamp}_{os.getpid()}"
            self.dir.mkdir(parents=True, exist_ok=True)
            readme = self.root / "README.txt"
            # Rewritten when it has drifted, not only when it is missing: a
            # folder that already holds schema 1 sessions carries the old note,
            # which does not mention how to check a collection -- and that
            # advice is the whole reason the note exists.
            if not readme.exists() or readme.read_text(encoding="utf-8") != README:
                readme.write_text(README, encoding="utf-8")
            self._file = (self.dir / "samples.jsonl").open("a", encoding="utf-8")
            log.info("collecting detection samples into %s", self.dir)
            return True
        except OSError as exc:
            self._disable(exc)
            return False

    def _budget_left(self) -> bool:
        """Is there room in the dataset directory for another session?

        Checked once per session rather than per sample: walking the tree
        costs milliseconds and the per-round cap already bounds how far one
        session can overshoot. A budget that is already spent turns collection
        off for this round instead of raising, exactly like a full disk.
        """
        if not self.max_mb and not self.max_total:
            return True
        try:
            total_bytes = 0
            total_rows = 0
            for path in self.root.rglob("*"):
                if not path.is_file():
                    continue
                total_bytes += path.stat().st_size
                if path.name == "samples.jsonl":
                    with path.open("rb") as fh:
                        total_rows += sum(1 for _ in fh)
        except OSError:
            # An unreadable dataset directory is not a reason to skip a round;
            # the write itself will fail loudly enough if it is really broken.
            return True
        if self.max_mb and total_bytes >= self.max_mb * 1024 * 1024:
            log.warning("dataset budget reached: %.0fMB of %.0fMB in %s -- "
                        "collecting nothing more until it is cleared out",
                        total_bytes / 1024 / 1024, self.max_mb, self.root)
            return False
        if self.max_total and total_rows >= self.max_total:
            log.warning("dataset budget reached: %d of %d samples in %s -- "
                        "collecting nothing more until it is cleared out",
                        total_rows, self.max_total, self.root)
            return False
        return True

    def _disable(self, exc: BaseException) -> None:
        log.warning("sample collection off for this run: %s", exc)
        self.enabled = False
        self.close()

    def close(self) -> None:
        if self._file is not None:
            try:
                self._file.close()
            finally:
                self._file = None

    # -- capture ---------------------------------------------------------
    def wants(self) -> bool:
        """Should the next drag be sampled?

        Called before the press, because sampling decides whether the loop
        pays for the extra capture at all. Every Nth drag rather than the
        first N: a round's boards are most alike back to back, so a stride
        buys more variety per megabyte than a burst at the start.
        """
        if not self.enabled or self.written >= self.per_round:
            return False
        return self.seen % self.every == 0

    def seen_drag(self) -> None:
        self.seen += 1

    def _refuse(self, why: str) -> None:
        self.refused[why] = self.refused.get(why, 0) + 1

    def usable(self, reading: Optional[dict], count: int) -> bool:
        """Is this reading a label, or is it a photograph of the board moving?

        Both gates exist because schema 1 had neither. A sample that fails one
        is not merely noisy -- it is indistinguishable from a correct reading
        once it is on disk, so it has to be refused at the moment the
        information to judge it still exists.

        `baseline` is the median change over the tsums the chain did not
        touch: the board's own idle jiggle, which is the yardstick any mark has
        to beat. Above `max_motion` the pile is still falling and every tsum
        clears any threshold. `max_marked` catches the other end -- a reading
        where most of the board lit up is a screen transition or a FEVER wash,
        not the game naming one character.
        """
        if reading is None:
            self._refuse("no reading")
            return False
        baseline = float(reading.get("baseline", 0.0))
        if self.max_motion and baseline > self.max_motion:
            self._refuse("board still moving")
            return False
        marked = reading.get("marked") or []
        if self.max_marked and count and len(marked) / count > self.max_marked:
            self._refuse("most of the board lit up")
            return False
        return True

    def record(self, before: np.ndarray, marked: Optional[np.ndarray], *,
               board: Sequence[int], radius: float, tsums: Sequence[Any],
               head: int, proposed: Sequence[int], kept: Sequence[int],
               fever: bool, options: Optional[dict] = None,
               reading: Optional[dict] = None) -> None:
        """Write one sample. Never raises."""
        # The cap is enforced here as well as in `wants`, so a caller that
        # samples on its own schedule still cannot fill a disk.
        if not self.enabled or marked is None or self.written >= self.per_round:
            return
        if not self.usable(reading, len(tsums)):
            return
        if not self._open():
            return
        index = self.written + 1
        try:
            params = [int(cv2.IMWRITE_JPEG_QUALITY), self.quality]
            assert self.dir is not None
            cv2.imwrite(str(self.dir / f"{index:04d}_before.jpg"), before, params)
            cv2.imwrite(str(self.dir / f"{index:04d}_marked.jpg"), marked, params)
            row = {
                "schema": SCHEMA,
                "version": __version__,
                "index": index,
                "time": time.time(),
                # Where the crop sits in the emulator's content area, so a
                # sample can be lined up with a full screenshot later.
                "board": [int(v) for v in board],
                "radius": round(float(radius), 2),
                "fever": bool(fever),
                # Detection's answer: position, size, and the per-frame colour
                # cluster it assigned. The cluster id is what a learned
                # identity is meant to replace, so it is recorded as a
                # baseline to score against, not as truth.
                "tsums": [{"x": round(float(t.x), 1), "y": round(float(t.y), 1),
                           "r": round(float(t.r), 1), "kind": int(t.kind)} for t in tsums],
                "head": int(head),
                "proposed": [int(i) for i in proposed],
                "kept": [int(i) for i in kept],
                # How the label was photographed. Schema 1 rows are unusable
                # because of these three numbers and did not record them, so
                # the fault could only be found by re-reading every JPEG.
                "capture": {"delay": round(self.delay, 3),
                            "frames": self.frames,
                            "gap": round(self.gap, 3),
                            "floor_mult": round(self.floor_mult, 2)},
                # The reading itself, so scoring never has to re-derive it:
                # per-tsum change, the idle-motion yardstick it was judged
                # against, and which tsums cleared it.
                "baseline": round(float(reading.get("baseline", 0.0)), 2),
                # What a mark actually had to clear, which is the baseline
                # times `floor_mult` on anything schema 2 collected.
                "bar": round(float(reading.get("bar", 0.0)), 2),
                "marks": [round(float(v), 1) for v in reading.get("values", [])],
                "marked": [int(i) for i in reading.get("marked", [])],
                "options": options or {},
            }
            self._file.write(json.dumps(row, separators=(",", ":")) + "\n")
            self._file.flush()
            self.written = index
        except (OSError, cv2.error, ValueError) as exc:
            self._disable(exc)

    def summary(self) -> str:
        refused = ", ".join(f"{n} {why}" for why, n in sorted(self.refused.items()))
        if not self.written:
            # A round that gathered nothing is worth a line only when it tried
            # and was refused -- that is the difference between a quiet
            # setting and a board this collector cannot read.
            return f"collected no samples ({refused})" if refused else ""
        note = f"; refused {refused}" if refused else ""
        return f"collected {self.written} sample(s) in {self.dir}{note}"
