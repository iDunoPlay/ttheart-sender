# Roadmap status: what exists, what does not, what to build next

Answers Phase 25 of *Tsum Tsum AI Gameplay — Continuous Improvement Roadmap*.
Everything marked DONE has been read in the code; everything with a number
beside it was measured over the 1,230-sample corpus, not estimated.

## Status table

| Phase | Component | Status | Where / evidence |
|---|---|---|---|
| 0 | Baseline mode | **PARTIAL** | Every rule ships off by default and `flows/play.yaml` is the one-line revert, so a baseline exists in practice. It is not a *named* mode, and it does not record what Phase 0 asks for — see the gap below. |
| 1 | Tsum dataset collection | **DONE** | `game/dataset.py` schema 3. 1,230 samples, 88 sessions. Labels are free: the game's own marks. |
| 1 | PyTorch classifier | **DONE, AND REJECTED** | `scripts/classify.py` → `models/character.onnx`. 95.1% on held-out sessions; **−24.2% ± 8.7% in play**. Off. |
| 2 | Board detection | **DONE** | `LAYOUTS` + pinned `window.size`. |
| 2 | Tsum localization | **DONE** | `detect()`, radius lock, bowl reject. |
| 2 | Board reconstruction | **PARTIAL** | `Tsum` carries `x, y, r, kind, colour`. **No confidence fields, no `tracking_id`.** Phase 24's confidence chain cannot be built on it as it stands. |
| 3 | Temporal tracking | **MISSING** | No frame-to-frame linkage anywhere. Every frame is re-clustered from scratch, which is exactly why `kind` is documented as "stable within one frame only". |
| 4 | Chain generation | **DONE** | `adjacency()` + `find_chains()`; `link_px`, `block`, `max_chain`, reach rules. |
| 5 | Chain scoring | **MINIMAL** | The entire strategy is `chains.sort(key=(c.is_base, len(c)), reverse=True)` — two features, hand-ordered, no weights, no telemetry of alternatives. |
| 6 | Candidate simulator | **MISSING** | Nothing estimates a chain's result before executing it. |
| 7 | Telemetry | **PARTIAL** | Board, chain, marks, cleared, per-member clear values, full options block. **No score, coins, or combo.** |
| 8 | Experience dataset | **MISSING** | Blocked on Phase 7's gap. |
| 9 | Value model | **MISSING** | Blocked on Phase 8. |
| 10 | Shadow mode | **MISSING** | |
| 11 | A/B testing | **PARTIAL** | Done by hand: flip a flag in `play.yaml`, play rounds, read `scripts/rounds.py`. Real and used — 23 rounds of it — but manual and scored on the wrong number (below). |
| 15 | Chain execution | **DONE** | `per_step`, `hold`, `move_time`, stall detection, speed recalibration, shuffle recovery. |
| 19 | Replay system | **PARTIAL** | `scripts/replay_decisions.py` replays saved decisions offline. Its clear constants are now known to be wrong — see below. |
| 20 | Model versioning | **MINIMAL** | `models/palette.json`, `palette_v11.json`. No metadata, no recorded gameplay performance. |

## Three places the roadmap and the measurements disagree

These are worth settling before following the priority order in Phase 23,
because each one would send effort somewhere the data says is not the problem.

**1. "Make Tsum recognition extremely reliable" is the wrong target.**
Phase 23 puts recognition first. We built it: 95.1% held-out accuracy, and it
cost 24% of the clear rate. Two reasons. A board holds **at most 5 characters,
4 with an item**, so the board never needs a character *named* — it needs to
know which of five piles a tsum is in. And the classifier's confidence
fallback ("if unsure, keep the colour cluster") guarantees that a character
named on half its crops carries a name *and* two cluster ids, which
`adjacency` links across neither. It doubled the splitting it was built to fix
(11.20 ids per board against 7.17).

The reliable-recognition goal should be restated as **consistency within one
board**, measured in tsums cleared per drag, not accuracy against a class list.

**2. "A longer chain may not always be the best action" — but the cost model
underneath that is wrong in our favour.** This project assumed for six rounds
that the game refuses a chain at its first bad member and loses the rest.
Measured: **25% of drags clear members 3 and 5 but not 4**, the skipped member
moves 0.61× the idle board and the one past it 11.4×, and 79.3% of past-hole
members had nothing cleared below them to fall into. **The game skips a
refused member and keeps linking.** A wrong guess costs one slot, not the
tail. `replay_decisions.py`'s prefix constants describe a mechanism the game
does not have.

**3. Phase 24 says never optimise score while vision is unreliable. Agreed —
but we have never optimised score at all.**

## The highest-priority missing component

**Phase 0. The bot cannot measure the thing it is supposed to maximise.**

Every A/B in this project — 23 rounds of them — has been scored on `cleared`,
the count of tsums removed from the board. That is not the objective. Tsum
Tsum scores **superlinearly in chain length**, and the round is dominated by
fever, bombs, combo and skill. Four 3-chains and one 12-chain both clear 12
tsums and score very differently.

So the single most important unknown in this codebase is a question nobody has
asked: **does clearing more tsums actually score more?** If the correlation is
weak, then some fraction of 23 rounds of tuning was spent on a proxy that
diverges from the goal, and every future value model would inherit the same
mistake.

It is cheap to answer. The score does not need per-frame OCR — one number at
the end of each round is enough to re-score every flag against the real
objective, and the flow already recognises that screen (`scoreboard`,
`tsum_score`, `last_bonus`, `timeup` templates at `flows/play.yaml:290`).

This is also literally what Phase 0 asks to record and what the roadmap builds
everything else on: Phases 8–18 all need a reward signal, and there is none.

### What to build, in order

1. **Capture the results screen** at the end of every round, and save it
   beside the round's telemetry. Needs no reading yet — it just stops the
   evidence being thrown away, and it is the input to step 2.
2. **Read score and coins** off it. A fixed font in a fixed place: template
   digits first, and a small PyTorch digit head only if that is not enough.
3. **Re-score the existing corpus.** `cleared` vs score, per round. This
   either validates 23 rounds of tuning or redirects it.
4. **Then** Phase 5 telemetry: log the features of every *candidate* chain,
   not just the chosen one. That is the training set for Phase 9, and it costs
   nothing to collect once the reward is measurable.

Temporal tracking (Phase 3) is the strongest *second* candidate and is worth
doing after this: it attacks identity, which is the measured bottleneck, and
it needs no labels. But it should be judged against a real reward, not against
`cleared`, which is why it comes second.
