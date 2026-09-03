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

The **settings** column is computed, not typed: it names every play setting
that differs from the defaults, so a flag added later appears without anyone
editing the script. Rows marked `[partial: pre-schema-3 row]` come from a
corpus that recorded only a curated subset -- there, a setting the column does
not name is *unknown*, not default.

The equipped tsum decides how the board is filled and which skill fires, so a
row from another character is not comparable without saying so. **The two rows
below were played with Beast**, stated by the player because nothing recorded
it at the time; from schema 3 the column carries `base Lab(...)`, read off the
skill icon, and a `!` on it means the corpus mixes more than one character.

| collected | samples | settings | colour lift | plausible | found | refused | dead drags |
|---|---:|---|---:|---:|---:|---:|---:|
| 2026-09-02 | 726 | k12 link105 fit1 floor8.0 — baseline before fit_effort | 1.47x | 98% | 42 | 29% | 29% | -- |
| 2026-09-02 | 4306 | k12 link105 fit3 reach260 floor8.0 — fit_effort 3 (steady colour fit ON, verify_clears OFF) | 1.51x | 97% | 42 | 30% | 26% | -- |
| 2026-09-03 | 1494 | k12 link105 block 1.25 max_chain 12 verify_reach 260 floor8.0 [partial: pre-schema-3 row] — intended as the verify_clears round; fit_effort silently back to 1, and v1.9.0 could not record either | 1.54x | 96% | 41 | 29% | 25% | -- |
