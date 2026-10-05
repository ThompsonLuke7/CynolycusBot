# 4H feature study → Momentum/HTF retrain → Meta retrain

Status (2026-09-30): Stage 0 arms + Stage 1 screen DONE for momentum -> `research/feature_screen_2026-09-29/README.md`.
HTF Stage 1 BLOCKED: its OOF keeps only future-defined swing-zone rows (48.8%); rebuild HTF labels/OOF on all rows first.
Originally: PLAN (2026-09-25). Training runs on Colab; local work is limited to features, labels, matrices and training-free screens.

## Why now
- **Stale data.** Both 4H models were trained 2026-06-15 on data ending 2026-05-14 (momentum) and 2026-06-02 (HTF).
- **Wrong universe.** They were trained on ~1,080 tickers but score a ~2,850-name live universe.
- **Volatility dominates.** The top features by cross-family stability are volatility and price level (`daily_atr_pct`, `xsec_atr_pct_rank`, `atr_pct_14`, `low_price_flag`), then earnings timing and calendar-of-year terms (`week_sin/cos`). Both labels reward forward *max* return, which rises mechanically with volatility. The models largely learned "pick volatile names", which is the vol/beta tilt seen in the permuted-label control study. Chart-structure features barely register.
- **Missing inputs.** There are no candle-shape features, no daily 100 EMA (the 4H `ema_dist_100` is roughly a daily EMA50), and no return window longer than 4 weeks.

## Division of labor (agreed principle)
The model answers **which names, relative to peers at this bar**. The harness answers **how much gross exposure, sizing, and event/regime risk**. Consequences:
- Rare, unprecedented market-wide shocks (war, policy shocks) are the harness's job. There is too little history for a model to learn them.
- Labels should be cross-sectional (relative to same-bar peers), so the model is not asked to time the market.
- Regime features stay only as *context* for how setups behave, e.g. pullbacks in risk-off. Calendar-of-year features go, because with ~5 years of history `week_of_year` identifies specific market episodes (memorization, not signal).

## Stage 0 — label decision (gates everything)
Features can only be judged against a label, so run arms side by side:
- **L0** current labels (baseline).
- **L1** forward return ranked within same-bar volatility buckets (vol-neutral).
- **L2** forward return excess vs vol-matched peers.

Every arm gets a permuted-label control model (see `project_topk_null_control`). Note that ATR-unit labels were previously shown to tilt toward low-vol names; L1/L2 neutralize volatility rather than divide by it.

## Stage 1 — local, training-free screen
Reuse `strategies/momentum_expansion/ablation/` (walk-forward folds, rank-IC bootstrap, BH-FDR). For every existing and candidate feature, measure against each label:
- (a) univariate rank IC;
- (b) incremental IC after residualizing on the deployed OOF score **and** on volatility, so features that only restate volatility are exposed.

Outputs: a keep / drop / candidate table, with the dead-weight list for the current 106/111.

Candidate blocks (all point-in-time from the local bar cache):
1. **Daily MA structure.** Distance to daily EMA20/50/100/200 in ATR units, EMA50/100 slope, days since the last close below EMA50/100, EMA100 touches in the last 20d, and a daily stack including the 100.
2. **Candle shape (daily and 4H).** Body/range, upper/lower wick ratios, close-location value, hammer / engulfing / inside-bar / NR7 flags, consecutive-bar counts.
3. **Leader/base context.** Daily ret_63/126/252, their cross-sectional ranks, depth of pullback from the 126d high in ATR, base length, and a prior-run × pullback interaction.
4. **Volume structure.** Up/down-volume ratio, volume dry-up in the base (vol_10/vol_50), OBV/AD slope. Volume uses per-date ranks only (IEX→SIP switch in 2026Q3).
5. **Bulk pandas-ta pool (SPY-intraday approach).** The full indicator set on daily bars as a screening pool only. Nothing enters the model without surviving (b).
6. **Later, needs data.** PIT fundamentals / EPS surprise (SEC XBRL), and the IV surface (nightly capture started 2026-09, too short to train on yet).

## Stage 2 — Colab block ablation
Arms: baseline; −calendar; +each surviving block; +all survivors. Use walk-forward folds with a 21d embargo and multiple seeds.

Judge by:
- top-k excess vs a vol-matched control;
- NDCG@k;
- the gap to the permuted-label control.

Use permutation importance on validation, not gain/cover, which favor continuous high-cardinality columns like ATR. Do not touch the final test window.

## Stage 3 — retrain
Rebuild features/labels on the live PIT universe (not the old 1,080) through the latest complete label window. Train on the full history, not only the last 3-6 months: 3-6 months is one regime and too few independent bars. Recency weighting can be one Stage-2 arm. Then rebuild the Meta matrix and retrain Meta. Per the 2026-09 result that theme features hurt Meta, drop theme features unless the repaired labels change that.

## Open decisions for the user
1. Adopt L1/L2 label arms (changes what the models optimize)?
2. Train on the live ~2,850 universe (more small/illiquid names) or a liquidity-floored subset?
3. Approve Stage 1 locally (minutes per run, no training)?
