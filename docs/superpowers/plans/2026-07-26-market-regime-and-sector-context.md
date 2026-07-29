# Market-Regime, Sector-Context, Options-Delta and Portfolio-Risk Integration

Date: 2026-07-26
Planner: Claude (Fable)
Source: external review of a competitor product ("Prism") dashboards, reduced to the
subset that is (a) reproducible from data we already hold, (b) not already implemented
here, and (c) testable under this repo's validation standards.

---

## 1. What the source analysis actually recommends, and what survives contact with this repo

The review proposed ~22 features. After inspecting the codebase, most collapse into four
buildable workstreams. Everything else is either already present, unbuildable, or judged
not worth the engineering time.

### Build (this plan)

| Item | Why it survives |
|---|---|
| Daily risk-appetite composite | Four ratio z-scores; all inputs are cached daily bars. Not present anywhere in repo. |
| Daily liquidity-stress composite | Amihud + dollar volume + credit + realized vol; all cached. Not present. |
| Sector leadership / breadth / dispersion | We have all 11 sector ETFs at 1D and 4H. Only per-ticker `rs_sector_5/20` exists today; no sector-level state. |
| Sector ETF assignment repair | Real defect confirmed (see §2). |
| Options snapshot-change features | We already persist strike ladders nightly; only *static* levels are consumed. Deltas are unbuilt. |
| Portfolio covariance / sizing layer | No portfolio-level risk layer exists; sizing is per-signal `$5,000` target notional. |

### Already present (do not rebuild)

`rs_spy_*`, `rs_qqq_*`, `rs_sector_*`, `beta_spy_60`, `corr_spy_60`, `regime_spy_trend`,
`regime_vix_z` — all in `strategies/momentum_expansion/features/feature_matrix_4h.py`.
GEX ladders, walls, magnets, gamma flip, DTE buckets, nightly snapshots — all in
`strategies/dealer_positioning` + `Data/dealer_positioning`. VWAP, structural levels,
V-reversal, dealer plates — `strategies/intraday_structure`. News/catalysts —
`signals/catalysts`, `signals/news`.

### Rejected

HMM regime states (smoothed probabilities are non-causal; a causal refit is strictly worse
than the realized-vol/trend state we already have), RRG visualization, narrative baskets
(CapEx / sovereign / debasement / K-shape), the political "administration" calendar leg,
"signed money flow" as an order-flow claim, the 5-day reversal factor, social/leaderboard
features, the global-events map, and any purchase of live OPRA or Level-2 data.

---

## 2. Defects and constraints found during investigation

These change the plan relative to the source recommendation and must be respected.

**D1 — Sector metadata is empty.** The source recommends replacing the `XLK` fallback in
`feature_matrix_4h.py:84` with "a metadata-sector-to-ETF mapping". That mapping does not
exist: `load_candidate_metadata()` returns 1125 rows with `sector == "Unknown"` for
**100%** of tickers, and the underlying `swing_trader_universe_v3.csv` is likewise all
`Unknown`. The fallback must therefore be replaced by an *empirical* assignment
(rolling correlation to the 11 sector ETFs), not a metadata join.

**D2 — Changing the fallback is a live-model-affecting change.** `rs_sector_5` and
`rs_sector_20` are in the deployed feature manifest. Silently reassigning ~1000 tickers off
`XLK` changes those columns for the currently deployed boosters, which were trained under
`XLK`-fallback semantics. The new resolver ships **behind a config flag, default off**, and
only flips after ablation evidence.

**D3 — New feature columns must not enter `FEATURE_COLUMNS_4H` by default.**
`build_training_matrix` does `dropna(how="any")` across the feature list; a column that is
NaN before the regime table's start date would silently delete the entire pre-2025 training
set. New features live in a separate exported block, opted into by the ablation trainer.

**D4 — RSP, UUP and VIXY daily history is short.** RSP starts 2025-04-30 (310 rows), UUP
2025-05-01, VIXY 2025-05-01, while SPY/HYG/LQD/sector ETFs all start 2020-07-27. A 252-day
z-score of `RSP/SPY` therefore has ~58 usable days over a 6-year training window. Backfill
is required before the risk-appetite composite is meaningful.

**D5 — Options snapshot history is 16 trading days** (`Data/dealer_positioning/historical_snapshots`,
2026-07-02 → 2026-07-24). Snapshot-*change* features are buildable but **cannot be validated**
at this sample size. Ship the code and accumulate; make no performance claim.

**D6 — No local training.** Per AGENTS.md and repo convention, XGBoost/LightGBM training runs
on Colab, not in the terminal. The ablation workstream delivers a harness plus a Colab export,
plus a training-free local incremental-value screen. It does not deliver trained ablation arms.

**D7 — Bar schema.** `Data/shared/bars/1d/*.parquet` is a flat frame with a `timestamp`
column (UTC, 04:00 stamps) and a `RangeIndex` — not a DatetimeIndex. Use
`strategies.momentum_expansion.data.load_bars.load_1d`, do not read the parquet naively.

---

## 3. Time-correctness contract (applies to every workstream)

* The daily regime table carries **both** `date` (session date, ET) and `available_at`
  (UTC timestamp at which the row became knowable = that session's close + a settle margin).
* Any consumer joins with `available_at <= decision_timestamp`, as-of backward. A 4H bar on
  session D therefore sees regime state through D-1 for the morning bar; it may see D only
  after the daily bar for D is confirmed complete.
* All rolling statistics end at `t`. No centered windows, no `.rolling(center=True)`, no
  smoothed/backward-looking filters, no forward-fill across a missing session without an
  explicit staleness flag.
* Z-scores use a trailing 252-session window with `min_periods` set explicitly; rows before
  `min_periods` are NaN, never zero-filled.
* Every composite emits a companion `*_n_components` count and a `*_stale_days` age so a
  partially-available composite is visible to the model rather than silently rescaled.

---

## 4. Workstreams

### WS-A — `signals/market_regime/` (new package)

Deliverables:
* `config.py` — required symbols, windows, output paths, freshness tolerances.
* `sector_map.py` — `sector_etf_for(ticker, asof)` resolving in order: curated map →
  empirical rolling-correlation assignment (120 trading days of daily log returns vs the 11
  sector ETFs, recomputed monthly, point-in-time, cached to parquet with the asof date) →
  `None` (explicit unknown). Behind flag `SECTOR_RESOLVER_ENABLED`, default off (D2).
* `daily_regime.py` — builds the canonical table:
  `date, available_at, risk_appetite_z, credit_risk_z, liquidity_stress_z, breadth_z,
   sector_dispersion_z, spy_rv20_z, spy_trend_state` plus each composite's components and
  `*_n_components` / `*_stale_days`.
  - `risk_appetite_z` = mean of trailing-252 z of `log(XLY/XLP)`, `log(IWM/SPY)`,
    `log(HYG/LQD)`, `log(RSP/SPY)`.
  - `liquidity_stress_z` = mean of `z(Amihud_20) − z(DollarVolume_20) − z(log(HYG/LQD)) + z(RV20_SPY)`.
    Note the sign convention explicitly in code: rising value = *more* stress.
  - `credit_risk_z` = trailing z of `log(HYG/LQD)`.
  - `breadth_z` = fraction of the 11 sector ETFs above their own 20d and 50d SMA, z-scored.
  - `sector_dispersion_z` = cross-sectional std of sector 21d excess return vs SPY, z-scored.
* `sector_state.py` — per sector ETF per date: `excess_21d`, `excess_63d`, `rank_21d`,
  `rank_63d`, `rs_accel = excess_21d − excess_63d/3`, `above_20d`, `above_50d`.
* CLI `python -m signals.market_regime.build --start ... --end ...` writing
  `Data/shared/market_regime/daily_regime.parquet` and `sector_state.parquet`
  (atomic temp-then-rename, matching `core/state_store.py`).
* Data backfill for D4: extend RSP, UUP, VIXY daily bars back to 2020 via
  `strategies.momentum_expansion.data.bars.fetch_one`.

Tests: synthetic-fixture correctness for each composite; a leakage test proving a row's
values are byte-identical when future sessions are appended; `available_at` monotonicity;
short-history behavior (NaN, not zero); staleness flags.

### WS-B — 4H feature integration (depends on WS-A)

Add an optional `daily_regime` / `sector_state` injection to `build_ticker_features_4h`,
producing: `sector_excess_21d`, `sector_excess_63d`, `sector_rank_21d`, `sector_rank_63d`,
`sector_rs_accel`, `sector_breadth_20`, `sector_breadth_50`, `sector_dispersion_21d`,
`risk_appetite_z`, `liquidity_stress_z`, `credit_risk_z`, `regime_available`,
`regime_stale_days`, and three interactions (`rs_spy_20 × sector_rank_63d`,
`ret_20 × risk_appetite_z`, `breakout_20 × liquidity_stress_z`).

Exported as `REGIME_FEATURE_COLUMNS_4H`, **not** appended to `FEATURE_COLUMNS_4H` (D3).
Because the live panel (`live_feature_panel_4h.py`) calls the same builder, research/live
parity is automatic — but the live path must fail loudly on a stale regime table rather
than scoring on stale context.

### WS-C — Options snapshot-delta features

From `Data/dealer_positioning/historical_snapshots/*/dealer_strike_ladder.parquet` and
`dealer_level_summary.parquet`: `wall_change_1d/3d`, `gex_concentration_change`,
`gamma_flip_velocity`, `distance_to_call_wall_atr`, `distance_to_put_wall_atr`,
`level_stability_days`, `oi_change_by_strike`, `oi_change_by_dte`, `volume_to_prior_oi`,
`iv_skew_change`, `near_level_option_volume_share`, plus snapshot freshness/availability
flags. Naming must not imply observed dealer trades. No validation claim (D5).

### WS-D — Portfolio risk / sizing layer (research only)

Ledoit–Wolf shrunk covariance, per-position / per-sector exposure caps, volatility
targeting, correlation-aware top-N sizing, turnover and liquidity constraints, whole-share
rebalance plan. Compared against equal-weight top-10 and the current `$5,000` fixed target
notional. Not wired into live execution in this plan.

### WS-E — Ablation harness (depends on WS-B; bounded by D6)

Feature-block definitions (baseline / +risk / +liquidity / +sector / +all), fixed folds and
seeds, metric computation (rank IC, NDCG@10, top-10 forward return, win rate, turnover,
Sharpe, MaxDD), week-block bootstrap CIs, and BH-FDR at q ≤ 0.10 reusing
`scripts/confluence_discovery/search.py::bh_fdr`. Plus a **training-free** local screen:
per-feature rank IC against the existing `meta_good` / expansion labels by walk-forward
period. Colab export for the actual trained arms.

---

## 5. Acceptance criteria

A regime feature block is adopted into the live manifest only if it shows positive
incremental rank IC in at least three walk-forward test periods, improves top-10 expectancy
or MaxDD without materially raising turnover, has a week-block bootstrap CI excluding zero,
and survives BH-FDR at q ≤ 0.10. Until then the block ships as opt-in research code.

---

## 6. Out of scope

Live-order-path changes, retraining or redeploying any booster, purchasing market data,
and every item listed as Rejected in §1.
