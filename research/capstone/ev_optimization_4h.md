# EV / win-rate optimization experiments — 4H modules (2026-07-14)

Driver: `backtests/ev_experiments_4h.py` (raw outputs in `backtests/ev_experiments_4h/`,
gitignored; headline tables reproduced here). Companion to
`research/capstone/leakage_audit.md` §0.6 — starts from the honest
val-select/test-freeze baselines and asks where EV per trade and win rate can
be improved for `momentum_expansion` and `multi_ticker_swing_htf`, without
re-tuning on the frozen test window.

## Method (leak-safe selection chain)

1. **E1 (label space)** — walk-forward OOF scores restricted to timestamps
   strictly before each strategy's test cutoff, low-price gate applied:
   EV by cross-sectional score rank, and big-mover capture (precision/recall
   of forward moves ≥10/15/20% inside the 25-bar label window).
2. **E2 (policy space)** — vectorized path-based sweep over
   `top_k × conviction-z gate × side × tp_atr × sl_atr × max_hold`
   (same execution semantics as `family_backtest`: next-bar-open entry,
   ATR-at-signal TP/SL, same-bar SL priority, $1k fixed notional, no costs).
   Run on pre-test OOF scores split into **3 sequential ~10-month folds**;
   configs ranked by **worst-fold** expectancy / win rate, with a 20bp
   round-trip cost haircut column. Rewarding the worst fold selects
   robustness, not one lucky regime.
3. **E3 (transfer check)** — shortlisted configs re-simulated with the exact
   `fb.simulate` engine using the **deployed** model's scores
   (momentum: xgb_classifier s45; HTF: lgbm_classifier s46) on the
   **validation** window. Final config per module chosen here.
4. **E4 (one-shot)** — the single chosen config per module run **once** on the
   frozen test window. No iteration on test.

## E1 findings (pre-test OOF, label space)

- **Momentum (long)**: top-20 rank buckets carry ~2.0–2.35% 25-bar close-EV
  vs 0.72% for the rest of the universe. Average max-up excursion inside the
  window is 15–17% vs 11–12% max-down — the edge lives in the *excursion*,
  not the close, so exit design dominates. Top-5 precision on ≥20% movers is
  25.1% vs 6.4% base rate (3.9x lift), and precision stays ≈25–27% out to
  top-50 — widening the book catches more movers at similar hit quality,
  at the cost of lower close-EV (rank 21–50: 1.38%).
- **HTF (long)**: same shape — 2.9% close-EV in top-5 decaying to 1.0% in the
  tail; ≥20% mover precision 28.5% in top-5 vs 10.4% base (2.75x).
- **HTF (short) is toxic**: bottom-ranked names have *negative* short EV in
  every rank bucket (−0.4% to −1.0% per window) and mover-capture lift
  **below 1** (0.16–0.54): the lowest-scored names are *less* likely to crash
  than the average name. The long-model's bottom ranks select quiet names,
  not short candidates. Half of the clean HTF baseline's 23k test trades are
  these shorts.

## E2 findings (3-fold pre-test policy sweep, OOF scores)

Common structure that is top-ranked in **every** fold, both modules:

| Lever | Direction | Why it works |
|---|---|---|
| Selectivity | top-3/5 + cross-sectional conviction gate (z ≥ 1–2) beats top-20 | EV concentrates in extreme, high-conviction ranks |
| Side (HTF) | long-only | shorts are structurally −EV (E1) |
| Stop | wide (5 ATR), never 2 ATR | tight stops whipsaw out of +EV excursions |
| Target | tp2 → max WR (~70%); tp4–6 → max EV | WR/EV trade-off is set by the TP multiple |

Baselines on the same folds (worst-fold expectancy, net-of-20bp worst fold):
momentum k5/tp2/sl4/h75 → 0.61% (0.41%); HTF k20/both/tp5/sl2/h25 → 0.34%
(0.14%), and the identical HTF policy long-only → 0.44% with WR +3pp —
dropping shorts is a pure improvement even before re-tuning.

## E3 → E4: chosen configs and frozen-test confirmation

All rows are the exact engine with deployed model scores. Baselines from
`family_compare_clean` (leakage_audit §0.6). One-shot test rows were run once.

**momentum_expansion** (long-only, xgb_classifier s45)

| Config | Window | Trades | WR | EV/trade | net 20bp | PF | maxDD | tp/sl/hold/k/z |
|---|---|---|---|---|---|---|---|---|
| baseline clean | test | 3,876 | 74.7% | +2.39% | +2.19% | 1.53 | −15.3% | 2/4/75/k5/— |
| **final_balanced_k3_z2** | val | 1,601 | 74.3% | +3.65% | +3.45% | 1.54 | −16.8% | 2/5/75/k3/z2 |
| **final_balanced_k3_z2** | **test (one-shot)** | **1,496** | **77.4%** | **+3.76%** | **+3.56%** | **1.63** | **−9.7%** | 2/5/75/k3/z2 |

**multi_ticker_swing_htf** (lgbm_classifier s46)

| Config | Window | Trades | WR | EV/trade | net 20bp | PF | maxDD | tp/sl/hold/k/z/side |
|---|---|---|---|---|---|---|---|---|
| baseline clean | test | 23,173 | 39.0% | +1.45% | +1.25% | 1.49 | −18.8% | 5/2/25/k20/—/both |
| baseline long-only | val | 12,650 | 40.8% | +1.63% | +1.43% | 1.35 | −43.7% | 5/2/25/k20/—/long |
| **final_ev_k5_z1_long** | val | 2,911 | 51.6% | +2.63% | +2.43% | 1.42 | −27.5% | 6/5/25/k5/z1/long |
| **final_ev_k5_z1_long** | **test (one-shot)** | **2,723** | **56.9%** | **+5.01%** | **+4.81%** | **1.89** | **−15.2%** | 6/5/25/k5/z1/long |

Improvement vs the honest baselines, on the frozen test window:

- momentum: WR +2.7pp, EV/trade +57%, PF 1.53→1.63, maxDD −15.3%→−9.7%.
- HTF: WR +17.9pp (39.0→56.9%), EV/trade 3.5x (1.45→5.01%), PF 1.49→1.89,
  maxDD −18.8%→−15.2%.

## Caveats

- Fixed $1k notional per signal, no concurrency/capital constraints, no
  costs in headline EV (net-of-20bp column supplied); `max_dd` and
  `ret_over_dd` are on cumulative fixed-notional PnL, not compounded equity.
  `ret_over_dd` is not comparable across configs with different trade counts
  (the clean HTF baseline's 17.9 vs final's 8.95 reflects 23k vs 2.7k trades,
  not better risk-adjusted quality).
- HTF final config exits 63% of trades at the 25-bar time stop (tp6 rarely
  hit) — EV comes from wide stops letting winners run to the horizon; it
  behaves like "hold-25-bars unless stopped", which is also simpler to run
  live.
- The test window is trend-favorable (val→test EV improved for both).
  Fold-worst pre-test numbers (momentum +1.09%, HTF +0.86% per trade net of
  costs ≥ 0) are the conservative planning numbers, not the test row.
- This is the second config family ever evaluated on the frozen test window
  (after the clean baselines). Selection multiplicity is minimal —
  configs were fully fixed by pre-test folds + validation before the single
  test run — but live/paper confirmation should precede any deployment.
- OOF fold boundaries imply the E2 sweep mixes model refits; the conviction
  gate is cross-sectional (per-timestamp z), so it is robust to score-scale
  drift between refits.

## Next steps (roadmap-aligned)

1. Paper-trade the two final configs alongside the current policies
   (momentum: k3/z2/tp2/sl5/h75; HTF: long-only k5/z1/tp6/sl5/h25) and
   compare audit-log EV/WR after ~4–6 weeks.
2. If adopted, wire the conviction-z gate and long-only HTF flag into the
   live order policy configs (config change, not code — both runners already
   rank cross-sectionally per timestamp).
3. Portfolio-level backtest with position sizing/concurrency limits remains
   the missing fidelity step before citing compounded returns anywhere.
