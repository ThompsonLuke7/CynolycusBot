# Model bundle evaluation — 2026-06-06

Evaluation of the three Colab-trained bundles imported on Jun 5–6, using the
out-of-fold (OOF) predictions shipped inside each bundle. OOF selection is
leakage-free, so reading realized forward returns off the OOF `score` is a
fair estimate of live ranking performance — no re-inference or retraining.

Bundles:
- Theme: `theme_expansion/models/bundle/`
- HTF swing: `multi_ticker_swing_htf/data/bundle/`
- Momentum: `momentum_expansion/data/training_import/bundle/`

Plots + per-model `summary.json` written in-place:
- `theme_expansion/outputs/plots/oof_eval/`
- `multi_ticker_swing_htf/plots/oof_eval/`
- `momentum_expansion/plots/output/oof_eval/`

Combined metrics: `meta_context/oof_eval_summary.json`.
Generator: `meta_context/oof_model_eval.py`.

---

## Verdicts at a glance

| Model | OOF rank IC (mean, t-stat) | Decile monotonicity | Selection edge (top vs universe) | Exported trees | Verdict |
|---|---|---|---|---|---|
| **Momentum** | 0.086 (t=18.3) | 0.99 (clean) | +3.2% fwd close, +4.1% vs bottom | **808** | **Healthy — ship-worthy** |
| **HTF swing** | 0.115 (t=23.9) | 0.99 (clean) | +1.35% fwd close | **1 (14 leaves)** | **OOF good, BUT exported model is degenerate** |
| **Theme** | 0.021 (t=3.1) | 0.19 (noisy) | +2.2% but bottom>universe | 9 | **Marginal — not robust** |

---

## 1. Momentum expansion — healthy, the strongest of the three

- Target `expansion_survival_score`, 808 trees (lr 0.02, depth 3), RMSE stable
  ~0.244 across all 7 folds — no fold blow-ups.
- OOF rank IC mean **0.086** (t≈18), positive in **68%** of 4H bars.
- Decile lift is **cleanly monotonic** (rank-corr 0.99). Top decile realized
  16.5% avg forward max-return vs 5.4% bottom; top-1% hits 25% avg / 13%
  median, 34% exceed +20%.
- Selection backtest (top-5 by score each 4H bar): cumulative realized forward
  close-return clearly above universe mean and well above bottom-5 — a real,
  persistent, time-stable edge (`selection_backtest.png`).
- **No blocking issues.** This is the model to move forward on.

## 2. HTF swing — strong OOF signal, but the SHIPPED model is broken

- OOF metrics are the best of the three: IC mean **0.115** (t≈24), positive in
  **80%** of bars, decile monotonicity 0.99. Top decile separates forward close
  return 4.6% vs 1.1% and cuts downside (−5.4% vs −8.0%).
- **Critical problem:** those OOF predictions came from the per-fold models
  (best_iteration 5–87 trees). The **exported `htf_swing_xgb.json` contains a
  single tree with 14 leaves** (`best_n_estimators=1`). It can emit **at most 14
  distinct scores** across the entire universe — a coarse step function, not a
  usable ranker. It will **not** reproduce the OOF performance live.
- **Root cause (was the code wrong, or just re-export?): both, but mostly a code
  brittleness — the signal is fine.** The walk-forward CV is correct and shows a
  strong signal. The bug is in the *final export* block of
  `multi_ticker_swing_htf/models/colab/htf_swing_train_colab.py`: it picks the
  final tree count from a **single held-out recent tail** via early stopping
  (`best_n = selector.best_iteration + 1`). For HTF that tail (≈2025-08→2026-05,
  a regime-shifted window) early-stopped at tree 0 on RMSE, so `best_n = 1` and
  the refit produced a single 14-leaf tree. The momentum model ran the *same*
  logic and landed on 808 trees; the 30m swing model never hit it — it is
  data/regime-specific fragility, not a logic error in the math.
- **Fix applied (this change):** `best_n` is now floored at the median of the
  walk-forward fold best_iterations and a hard minimum (`MIN_FINAL_ESTIMATORS`),
  so the exported model can never be shallower than what CV supported. Lines
  ~226–236 of the colab script.
- **You still need to re-export**: the fix only takes effect on the next Colab
  run (the final refit is an xgboost `fit`, which I do not run locally). After
  re-export the shipped model should carry ~20–40 trees and reproduce the OOF
  performance. The example-trade panel below already reflects the OOF (per-fold)
  model, i.e. the performance you'll get once re-exported correctly.

## 3. Theme expansion — marginal, not robust

- Target `fwd_theme_excess_benchmark_20d`, only 9 trees (early-stopped almost
  immediately), per-fold Spearman ranges −0.03 to +0.06.
- OOF IC mean **0.021** — statistically non-zero (t≈3.1 over 578 days) but
  economically tiny; decile lift is noisy (monotonicity 0.19).
- The top-1% score bucket has positive *mean* excess (8.2%) but **negative
  median (−0.28%)** — driven by a few outliers, not a reliable signal.
- In the selection backtest the bottom-decile actually beats universe mean,
  i.e. low scores don't identify losers. Treat as weak/early; not ready to
  drive allocation on its own.

---

## Example-trade plots (how a pick plays out live)

`example_trades.png` in each `oof_eval/` folder shows top-1 "live picks" on
evenly-spaced dates across 2022→2026, drawn on **real 4H price bars** (themes:
cumulative theme index). Each panel marks the entry, the forward window, and the
realized outcome envelope (max-favorable / max-adverse / close) taken straight
from the OOF row. Generator: `meta_context/oof_example_trades.py`.

- Momentum / HTF: real per-ticker 4H OHLCV from `Data/shared/bars/4h` (100%
  ticker coverage).
- HTF panels use OOF (per-fold) scores, so they preview the *correctly
  re-exported* model, not the broken 1-tree artifact.

## What was NOT run, and why (native event-driven harnesses)

The native option-P&L simulators were **not** run. Honest blockers:

1. **Feature data is Git-LFS pointers** locally (e.g. `features_4h.parquet` is a
   135-byte pointer to a 1.5 GB object). The native harnesses score per-bar off
   these full matrices + per-ticker 1h/4h price bars — none materialized here.
   Running them needs `git lfs pull` of multi-GB objects.
2. **Momentum harness is not CLI-wired** (`main.py` has no `--backtest` flag) and
   the inference dir `models/expansion_v1/` is empty — it needs the model
   installed and a custom driver.
3. **HTF has no inference or backtest code at all** — only labels/features.
   "Adapting the swing harness" is a real build: the swing harness is a 30m
   *classifier* with 10m execution; HTF is a 4H *regression* score with
   different features.
4. **Theme native scripts (07/12/29) evaluate a rule-based theme rotation on
   precomputed theme scores — not the new booster** — and script 07 pulls from
   yfinance over the network.

The OOF realized-return backtest above already provides a rigorous,
leakage-free performance read for all three without any of that. The native
sims add synthetic-option P&L on top, but are a heavier, separate effort with
consequential side-effects (installing models, multi-GB pulls, long runs).
