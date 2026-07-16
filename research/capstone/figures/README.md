# Capstone figure set

Every figure in this directory is generated **only from the locked artifacts**
registered in `scripts/capstone/reproduce_results.py` (the same registry behind
`research/capstone/results_lock.json`). No model is retrained and nothing is
fetched from the network. Regenerate with:

```bash
PYTHONPATH=. .venv/bin/python scripts/capstone/make_figures.py            # all figures
PYTHONPATH=. .venv/bin/python scripts/capstone/make_figures.py --only fig01,fig08
```

figs 11 and 13 read `research/capstone/exit_policy_grid.csv` (committed). To
regenerate that grid from scratch — needed only if the meta OOF models change:

```bash
PYTHONPATH=. .venv/bin/python scripts/capstone/build_meta_scored_from_oof.py  # -> /tmp/meta_scored.parquet
PYTHONPATH=. .venv/bin/python scripts/capstone/exit_policy_grid.py            # -> exit_policy_grid.csv
```

Every figure carries a provenance footer: date window, universe, **split tag**
(`test(frozen)` / `test(comp)` / `wf-oof` / `paper` / `artifact` — same meaning
as in `results_lock.json` and `leakage_audit.md`), benchmark, source artifacts,
and the git commit + generation date. Figures use the shared
`core/shared_plotting` package with the paper-light theme (`LIGHT_THEME`); the
categorical palette was validated with the data-viz palette checks (lightness
band, chroma floor, CVD ΔE ≥ 12 on adjacent pairs, ≥ 3:1 contrast on white).

**Shared conventions (figs 01–03, 05–07):** event-driven backtest trades book
**$1,000 notional per trade on a $100k base, not compounded**, concurrency
unconstrained — identical to the locked `family_compare_clean` /
`sweep_v2_clean` artifacts. P&L is booked at trade exit. Swing trade
timestamps are recovered by positional index into the same raw 30m caches the
backtest simulated on; the loader verifies every trade's entry price against
that bar and aborts below 99% consistency (current run: **100.00%**).

## Figure → claim map

| Figure | Claim it supports | Split | Sources |
|---|---|---|---|
| `fig01_equity_curves.png` | The three val-selected, test-frozen order policies were profitable out-of-sample over 2025-05→2026-07 and (HTF, momentum) beat SPY buy-and-hold under the stated per-trade convention. | test(frozen) | `family_compare_clean/*_frozen_test_trades.parquet`, `sweep_v2_clean/best_v2_clean_trades.parquet`, `Data/shared/bars/1d/SPY.parquet` |
| `fig02_drawdown.png` | Frozen-test drawdowns are visible and reported, not hidden: momentum −15.3%, HTF −18.7% of peak equity; drawdown clusters in the Nov-2025 and spring-2026 corrections. | test(frozen) | same as fig01 |
| `fig03_rolling_sharpe.png` | Risk-adjusted performance is time-varying — every strategy has weak stretches (momentum ≈ 0 around 2025-12→2026-01); no claim of uniform outperformance. | test(frozen) | same as fig01 |
| `fig04_selection_bias_correction.png` | The audit's headline: selecting the policy on the reporting split inflated ret/DD ~7.4× (momentum) and ~2.3× (HTF), and the swing win rate by ~14 pp (61.3% → 48.7% combined). Honest numbers are the frozen ones. | test(comp)/artifact → test(frozen) | `results_lock.json`; biased ret/DD from `leakage_audit.md` §0.6 |
| `fig05_regime_performance.png` | Frozen-test performance split by a fixed, stated regime rule (SPY close vs 200-day SMA at entry). The risk-off sample is small (n=186–1,154) and comes almost entirely from the spring-2026 correction — read as a robustness check, not proof of regime edge. | test(frozen) | same trades as fig01 + SPY 1d bars |
| `fig06_trade_return_distributions.png` | Per-trade return shape: momentum is tp-dominated (74.7% win), HTF is a positive-skew / sub-50%-win-rate system (median trade is negative), swing is near-symmetric around a small positive mean. | test(frozen) | same trades as fig01 |
| `fig07_hold_times.png` | Hold-time profile per strategy (median 4.0 / 5.5 / 1.1 trading days; end-spikes = max-hold exits) and the paper options ledger's ~125-min median hold. | test(frozen) + paper | same trades as fig01; `multiticker_20260528_20260529_closed_performance_rebuilt.csv` |
| `fig08_oof_decile_lift.png` | The ranking signal survives clean walk-forward OOF scoring (21-day embargo): top decile beats the pool mean in all five models (momentum +4.6% vs +1.5%; HTF +6.8% vs +2.2%; meta upside +6.3% vs +1.9%). | wf-oof | `oof_preds.parquet` for momentum / HTF / meta quality / meta upside (+ live `s_combo` reconstruction) |
| `fig09_meta_calibration.png` | Calibration, reported honestly: meta **upside** is directionally calibrated but over-predicts at high scores; meta **quality** is poorly calibrated OOS (near-flat reliability) — consistent with its ~0 OOF Spearman in the lock. Rank usage (top-K), not probability usage, is what the live system relies on. | wf-oof | meta `{quality,upside}/oof_preds.parquet` |
| `fig10_feature_importance.png` | What the five deployed winners actually key on, **by gain**: swing → short-horizon returns; momentum **and HTF** → ATR/volatility scale + earnings distance; meta → liquidity percentile + momentum ranks. Descriptive, train-time only. | artifact | `feature_importance*.csv` beside each locked model; **HTF gain recomputed from `model_lgbm_classifier_seed46.joblib`** (see split-vs-gain note below) |
| `fig11_meta_exit_policy.png` | Meta selection has edge but the pre-2026-07-12 rank-drop-out exit destroyed it: +0.57% mean/trade vs +5.87% (target+20), +4.96% (scale-out), +4.40% (deployed live) on the same entries. Motivates the July 2026 exit-policy change. | wf-oof | `research/capstone/exit_policy_grid.csv` (OOF `s_combo` rescore, audit §4.3) |
| `fig12_paper_sessions.png` | Paper trading reported unfiltered: the 2-session options ledger lost $6,246 (win 34%, n=123) even though fresh-entry calls averaged +36% — execution/staleness, not selection, drove losses. Small n; anecdotal. | paper | `multiticker_20260528_20260529_closed_performance_rebuilt.csv` |
| `fig13_scaleout_grid.png` | The scale-out trim is a **win-rate ⇄ expectancy dial, not a free lunch**: trimming less and later maximizes mean (25%@+50% → +6.23%); trimming more and earlier maximizes win rate and median (75%@+10% → 65.1% win, median +5.58%). The live 50%@+20% sits mid-curve. | wf-oof | `research/capstone/exit_policy_grid.csv` |

## Reconciliation & caveats (read before citing)

* **Swing max drawdown.** The locked `bt_v2_clean_max_dd_pct = −74.48%` is the
  sweep's own convention: cumulative **sum of per-trade returns in trade order**
  measured against its own peak (documented artifact, `leakage_audit.md`). On
  the $100k-base booked-P&L equity used in figs 01–02 the same trades draw down
  only −0.9%, because each trade risks $1k of a $100k base. Neither number is a
  portfolio drawdown under real position sizing — a sized portfolio backtest
  remains future work. Cite one convention explicitly, never both interchangeably.
* **Momentum/HTF drawdown cross-check.** Fig02's equity convention reproduces
  the locked summaries: momentum −15.26% vs locked −15.298%, HTF −18.73% vs
  −18.781% (residual = ordering of same-bar exits).
* **fig08 "universe" line** is the mean over **all** pool rows in the OOF
  window. The lock's `oof_universe_mean_fwd_close_ret` rows exclude the top-K
  rows, so they differ in the third decimal — both are stated definitions.
* **fwd_close_return** in figs 08–09 is the fixed label-horizon forward return
  with overlapping windows — signal quality, not a compoundable equity curve.
* **Feature importance is GAIN, and the HTF panel had to be recomputed
  (2026-07-15).** The trainers' `feature_importance()` preferred
  `model.feature_importances_`, which for **LightGBM's sklearn wrapper defaults
  to `importance_type="split"`** (raw count of times a feature was used) — not
  gain. XGBoost's `feature_importances_` *is* gain, so only the LGBM winner
  (HTF) was affected, and the tell was on disk: LGBM values are integers summing
  to 4,154 while every XGB file is floats summing to 1.0. Split count inflates
  high-cardinality features: `week_of_year` (52 distinct values) took **8.9% of
  splits but only 3.9% of gain**, while `daily_atr_pct` had **3.9% of splits and
  34.0% of gain**. The earlier version of fig10 therefore showed HTF as a
  seasonality model; by gain it is an ATR/volatility model, consistent with
  momentum. fig10 now recomputes gain from the saved booster and leaves the
  (stale) training CSV untouched; the trainer bug itself is fixed in
  `strategies/{multi_ticker_swing_htf,momentum_expansion}/data/training_export/colab_competition.py`
  and `signals/meta_context/meta_ranker/colab_competition.py`, so future exports
  write gain directly. **Existing `feature_importance_lgbm_*.csv` artifacts on
  disk are still split counts** — do not cite them as gain.
* **The meta ranker does not behave like a confluence ensemble.** Gain is
  dominated by **liquidity**: `dollar_vol_pctile_252` alone is 45.8% (quality) /
  35.2% (upside), stable across all 7 seeds, with momentum (`mom_xs_rank` +
  `mom_score`) adding ~21–24%. The HTF and theme legs are wired in but nearly
  inert — `htf_score` 0.35% (rank 36/67), `signal_agreement` (the literal
  confluence feature) 0.11% (rank 45/67), `theme_heat_score` 0.35% — and 21/67
  features carry exactly zero gain. This independently corroborates the 2026-07
  confluence-discovery null result and explains meta_quality's ≈0 OOF Spearman
  and flat calibration in fig09. Do not describe the meta ranker as a
  cross-signal confluence model on the strength of these artifacts.
* **No standalone catalyst/news model.** Catalyst & news enter as meta-ranker
  features; among the five locked paper models there is no separate
  catalyst model artifact, so lift/calibration is reported for the meta ranker
  (and momentum/HTF) only.
* **figs 11 & 13 are SHARES ONLY, and stops are near-inert there.** The meta
  exit harness walks 4H **stock** OHLC (`Data/shared/bars/4h`); there is no
  option-premium path in it. The deployed 50% stop binds on **1 of 1,430** stock
  trades and the 35% trail on 11 — they look like no-ops here precisely because
  they exist for the *option premium* case (the live 2026-07-09 COHR −85% / BE
  −79% losses), which this harness cannot price. Never cite fig11 as evidence
  that the live stop/trail is unnecessary.
* **fig11's policies differ in holding period, not just exit rule.** Rank
  drop-out averages 1.7 bars held, scale-out 25.0, and `target +20% full exit`
  42.7 (it has no horizon, so it rides to `MAX_HOLD=60` waiting for +20%). Its
  high median (+15.7%) reflects that most surviving trades book exactly +20%.
  Compare means across these with that in mind.
* **fig13 partials are not comparable on return-per-bar.** Every partial
  (25/50/75%) holds n=1,430 with `avg_hold` pinned at exactly 25 bars, because
  trimming does not end the trade — only the 100% full-exit row exits early
  (n 1,764→1,447 as the trigger rises). So partial ret/bar is just mean/25.
* **Concurrency.** The $1k-per-trade convention leaves concurrent position
  count unconstrained (HTF top-20 holds many names at once); capital usage
  therefore varies over time and Sharpe/return figures are per-convention, not
  account-level.
* **fig05 regime split** uses only two regimes because the frozen test window
  contains a single sustained risk-off episode; finer regime grids on one year
  of test data would be fitting noise.
