# Capstone figure set

Every figure in this directory is generated **only from the locked artifacts**
registered in `scripts/capstone/reproduce_results.py` (the same registry behind
`research/capstone/results_lock.json`). No model is retrained and nothing is
fetched from the network. Regenerate with:

```bash
PYTHONPATH=. .venv/bin/python scripts/capstone/make_figures.py            # all figures
PYTHONPATH=. .venv/bin/python scripts/capstone/make_figures.py --only fig01,fig08
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
| `fig10_feature_importance.png` | What the five deployed winners actually key on (gain share): swing → short-horizon returns; momentum → ATR/volatility scale + earnings distance; HTF → seasonality + regime; meta → liquidity percentile + momentum ranks. Descriptive, train-time only. | artifact | `feature_importance*.csv` beside each locked model (families/seeds match `winner_family_seed` in the lock) |
| `fig11_meta_exit_policy.png` | Meta selection has edge but the pre-2026-07-12 rank-drop-out exit destroyed it: +0.57% mean/trade vs +5.87% (target+20) and +4.96% (scale-out) on the same entries. Motivates the July 2026 exit-policy change. | wf-oof | `results_lock.json` `meta_exit_policy` rows (OOF `s_combo` rescore, audit §4.3) |
| `fig12_paper_sessions.png` | Paper trading reported unfiltered: the 2-session options ledger lost $6,246 (win 34%, n=123) even though fresh-entry calls averaged +36% — execution/staleness, not selection, drove losses. Small n; anecdotal. | paper | `multiticker_20260528_20260529_closed_performance_rebuilt.csv` |

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
* **No standalone catalyst/news model.** Catalyst & news enter as meta-ranker
  features; among the five locked paper models there is no separate
  catalyst model artifact, so lift/calibration is reported for the meta ranker
  (and momentum/HTF) only.
* **Concurrency.** The $1k-per-trade convention leaves concurrent position
  count unconstrained (HTF top-20 holds many names at once); capital usage
  therefore varies over time and Sharpe/return figures are per-convention, not
  account-level.
* **fig05 regime split** uses only two regimes because the frozen test window
  contains a single sustained risk-off episode; finer regime grids on one year
  of test data would be fitting noise.
