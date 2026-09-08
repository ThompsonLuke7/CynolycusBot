# Meta-labelling premise test — the answer is worse than "no"

2026-09-04. `scripts/thesis_test/recall_test.py`

**This test is completely independent of the Meta retrain.** It reads
`intraday_structure`'s own decision ledgers and touches no model, no matrix and
no training run. It could have been run at any point.

## Why this specific test

Meta-labelling only helps when the primary rule has good **recall** — it fires on
most of the moves worth taking, even if it also fires on junk, and ML then
supplies **precision** and sizing. It cannot help when the rule *misses* the
moves, because ML cannot rescue a trade that was never proposed.

The engine has written **100,422 abstentions**, and **98.8% of them are one rule**:
`invalidation_wider_than_max_atr` — declined because the stop would have been too
wide. So the premise reduces to a sharp question: *do the setups rejected for a
wide stop make the move anyway?*

## Method

De-duplicated to one row per (setup_id, ticker, day) — the engine re-evaluates a
setup every bar, so raw abstention counts are re-evaluations, not decisions. Then
measured forward excursion in the setup's own direction from the decision minute
on SIP 1-minute bars, ATR-normalised. Against a **same-ticker random-minute
control**, so "these names move a lot" cannot masquerade as a result.

## Result

Median forward excursion, ATR units:

| group | horizon | n | MFE | MAE | MFE−MAE | share MFE>1 |
|---|---|---|---|---|---|---|
| DECLINED | 60m | 156 | **0.509** | 0.730 | −0.211 | 25% |
| CONFIRMED | 60m | 83 | **0.501** | 0.659 | −0.383 | 22% |
| **CONTROL (random)** | 60m | 832 | **0.758** | 0.775 | −0.069 | **38%** |

Same shape at 15m and 30m.

**Two findings, and the second is the serious one.**

**1. The gate carries no information.** Confirmed 0.501 vs declined 0.509 — the
setups the engine declines perform *marginally better* than the ones it takes.
Gaps of −0.008 to −0.043 across horizons. Whatever
`invalidation_wider_than_max_atr` is selecting on, it is not forward move.

**2. Both are worse than random.** A random minute on the same ticker produces
**0.758** ATR of favourable excursion against the engine's confirmed **0.501** —
and 38% of random minutes reach 1 ATR against 22% of confirmed setups. The
engine's MFE−MAE is also the worst of the three (−0.383 vs −0.069 for control).

That is not a neutral filter. It is **negative selection**: the engine is
entering at moments with worse forward risk/reward than picking a minute at
random on the same names.

## What it means for meta-labelling

**Do not build meta-labelling on this engine.** The design needs a primary model
with recall — one that catches the moves and over-fires. This one does not
over-fire on good setups; it selects moments that are worse than random. Layering
ML on top to pick *which* of those to take would be filtering a negatively-selected
pool.

The problem is upstream of both recall and precision: **the entry criteria are
selecting the wrong moments.** That has to be fixed before any ML layer is worth
building on it, and the natural suspect is the same one the earlier work found —
setups are confirmed after the move has already been made, which is consistent
with entering at exhaustion and with the previously measured −1.76R post-exit
drift.

## Limits — this is suggestive, not final

* **n = 83 confirmed setups** with usable bars. Small.
* Only 133 of 289 abstention tickers have cached 1-minute bars, so the sample is
  restricted to the cached set rather than being a random draw.
* **Time-of-day confound, unresolved.** The engine acts at specific moments
  (breakouts, pullback completions, opening range); the control samples any RTH
  minute. If the engine systematically acts *after* a move, lower forward MFE is
  partly mechanical rather than a selection failure. Distinguishing those needs a
  control matched on time-of-day and on recent realised move, which this does not
  do.
* All modelled setups — the engine has **zero real fills** to date.

The direction is consistent enough across three horizons, and the gap to control
large enough, that I would not build on this engine before resolving it. But
"negative selection" should be re-tested with a matched control before it is
treated as established.
