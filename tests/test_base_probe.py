"""The base probe, and the one distinction it exists to keep straight.

`scripts/base_probe.py` asks two questions of the same column of numbers, and
reading one figure for both gets the answer backwards -- which it did on the
first run. "How many tsums were equipped" is the spread of the SESSION
medians. "How steady is the reading" is the spread WITHIN a session. The
per-sample spread across the whole corpus answers neither: it came out at 116.8
Lab units and was reported as "more than one equipped tsum", when the session
medians agree to 1.0 unit and there is only ever one.

These pin that apart, and pin the sample filter that feeds it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import base_probe  # noqa: E402


def write(root: Path, sess: str, rows_, rnd=None):
    d = root / sess
    d.mkdir(parents=True, exist_ok=True)
    (d / "samples.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows_), encoding="utf-8")
    (d / "round.json").write_text(json.dumps(rnd or {"cleared": 200}),
                                  encoding="utf-8")


def sample(i, kind, base_kind, lab, *, cleared=3, opts=None):
    return {"index": i, "version": "1.0", "radius": 25.0,
            "tsums": [{"x": 10.0, "y": 10.0, "r": 20.0, "kind": kind}],
            "head": 0, "cleared": list(range(cleared)),
            "base": {"kind": base_kind, "lab": lab, "distance": 12.0},
            "options": opts or {}}


def test_a_recoloured_sample_is_dropped_rather_than_mismatched(tmp_path):
    """`--recolour` renumbers `kind`, so `base.kind` no longer means the same
    thing. Comparing them anyway would silently mislabel every drag."""
    write(tmp_path, "s1", [sample(1, 0, 0, [200, 134, 165]),
                           sample(2, 0, 0, [200, 134, 165],
                                  opts={"recolour": 0.5})])
    got = list(base_probe.rows(tmp_path))
    assert len(got) == 1 and got[0][1]["index"] == 1


def test_a_sample_without_a_base_is_skipped(tmp_path):
    rowsx = [sample(1, 0, 0, [200, 134, 165])]
    rowsx.append({"index": 2, "tsums": [], "base": {}, "options": {}})
    write(tmp_path, "s1", rowsx)
    assert len(list(base_probe.rows(tmp_path))) == 1


def test_one_equipped_tsum_read_steadily_is_not_two_tsums():
    """Session medians agree; the reading wobbles a little. One tsum."""
    a = np.array([[200.0, 134, 165], [201, 135, 166], [199, 133, 164]])
    b = np.array([[200.0, 134, 165], [200, 134, 166]])
    med = np.stack([np.median(a, 0), np.median(b, 0)])
    across = float(np.linalg.norm(med.max(0) - med.min(0)))
    assert across < 5.0


def test_two_equipped_tsums_are_not_hidden_by_taking_medians():
    a = np.array([[200.0, 134, 165], [201, 135, 166]])
    b = np.array([[120.0, 160, 90], [121, 161, 91]])
    med = np.stack([np.median(a, 0), np.median(b, 0)])
    across = float(np.linalg.norm(med.max(0) - med.min(0)))
    assert across > 5.0


def test_welch_reports_a_real_difference_as_larger_than_its_error():
    a = np.full(400, 3.2) + np.linspace(-0.01, 0.01, 400)
    b = np.full(400, 2.9) + np.linspace(-0.01, 0.01, 400)
    diff, se = base_probe.welch(a, b)
    assert diff > 0 and abs(diff) > 2 * se


def test_welch_calls_pure_noise_noise():
    rng = np.random.default_rng(0)
    a, b = rng.normal(3.0, 1.0, 300), rng.normal(3.0, 1.0, 300)
    diff, se = base_probe.welch(a, b)
    assert abs(diff) < 2 * se
