# Scorecard

One row per collection, written by `python scripts/scorecard.py --append`.
Every number is measured against the game's own marks -- see
`docs/IMPROVEMENT-LOOP.md` for how a row is produced and
`docs/DATASET-FINDINGS.md` for what the rounds behind them concluded.

* **colour lift** -- how much likelier a game-confirmed partner is to share the
  pressed tsum's `kind` than a random far tsum is. 1.00x knows nothing.
* **plausible** -- share of boards read at a believable size (20-110
  detections); **found** is the median count.
* **refused** -- share of proposed chain members the game would not take;
  **dead drags** run and clear nothing.

| collected | samples | settings | colour lift | plausible | found | refused | dead drags |
|---|---:|---|---:|---:|---:|---:|---:|
| 2026-09-02 | 726 | k12 link105 fit1 floor8.0 — baseline before fit_effort | 1.47x | 98% | 42 | 29% | 29% |
