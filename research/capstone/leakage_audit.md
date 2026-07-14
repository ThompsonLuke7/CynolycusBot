# Capstone Leakage & Reproducibility Audit

Date: 2026-07-12 (Claude, capstone results-lock task)
Scope: the five models the capstone paper discusses —
`multi_ticker_swing` (primary), `momentum_expansion`, `multi_ticker_swing_htf`,
`meta_ranker`, and `spy_intraday` (baseline / limitation study).

Every claim below is labeled **VERIFIED** (read the code / artifact on this date) or
**ASSUMED** (inferred, not directly executed). File references are to the current
working tree on branch `capstone-repro-audit`.

---

## 0. Cross-cutting findings (apply to multiple models)

### 0.1 No purge/embargo at time-split boundaries — LOW severity, quantified
**VERIFIED.** Both split implementations are plain row-fraction splits of the
time-sorted pooled matrix, with no gap between splits:

- `strategies/multi_ticker_swing/models/train.py::_time_split` (70/15/15)
- `strategies/model_training/colab_competition.py::time_split` (60/20/20 — used by
  momentum, HTF, and meta competitions)

All labels are forward-looking, so rows at the end of train have labels computed
from bars at the start of val (and val→test likewise). Overlap horizons:

| Model | Label look-forward | Overlap window at each boundary |
|---|---|---|
| multi_ticker_swing | pivot confirm 3 bars + T-1 shift + follow-through 12 bars ≈ **16×30m bars (~1.3 trading days)** | ~16 bars × ~200-900 tickers |
| momentum_expansion | forward window **25×4H bars (~12.5 trading days)** | ~25 bars × ~3,000 tickers (~5% of a split) |
| multi_ticker_swing_htf | pivot right 3 + forward stats up to **38×4H bars (~19 trading days)** | ~38 bars × ~1,100 tickers |
| meta_ranker | labels inherited from momentum OOF forward windows (25×4H bars) | same as momentum |

Impact: inflates val/test metrics only for rows in the first days of each split;
with multi-year splits and millions of rows this is a small bias, but the paper
should state that splits are contiguous without an embargo. The **walk-forward OOF**
paths (see 0.3) DO use a 21-day embargo and are the clean series.

### 0.2 Test-set model selection ("winner's curse") — HIGH severity for headline metrics
**VERIFIED.** The competition bundles pick the deployed winner **by a test-set
metric** across 4 families × 5 seeds (20 candidates):

- momentum: `primary_metric: test_ndcg_at_10`, winner xgb_classifier seed 45
  (`strategies/momentum_expansion/models/expansion_v1/eval_metrics.json`)
- HTF: `primary_metric: test_ndcg_at_10`, winner lgbm_classifier seed 46
  (`strategies/multi_ticker_swing_htf/models/eval_metrics.json`)
- meta quality/upside: `primary_metric: test_ndcg_at_20`, winners xgb_classifier
  seeds 46/48 (`signals/meta_context/meta_ranker/models/*/eval_metrics.json`)

Consequence: the winner's test metrics are optimistically biased (max over 20
correlated draws). For the paper, either (a) report the winner's **walk-forward OOF**
performance (generated after selection, embargoed — clean), or (b) report the
test metric together with the family/seed distribution (`seed_results.csv`,
`model_family_summary.csv` are saved in each bundle) so selection effects are visible.
`reproduce_results.py` reports both the winner metric and the across-candidate spread.

### 0.3 Deployed "final fit" boosters see the test window — MEDIUM severity, live-only
**VERIFIED.** `momentum_train_colab.py::_final_fit` (same pattern in HTF/meta
trainers) refits the shipped booster on the first 80% of days and uses the last 20%
(= the test window) as the early-stopping eval set. The reported test metrics come
from the 60%-trained competition model, NOT the shipped booster — so the *reported
numbers* are not contaminated by this, but:

- any **backtest that scores the test window with the deployed booster is partially
  in-sample** (see 4.3 meta `backtest_exits.py`), and
- research (OOF-scored) vs live (deployed-booster-scored) signal distributions differ.

### 0.4 Universe survivorship / as-of-today metadata — MEDIUM severity, affects all backtests
**VERIFIED (mechanism), ASSUMED (magnitude).**
- Universe CSVs (`swing_trader_universe_v3.csv`, `shared_universe.csv`) are curated
  as of 2026 and applied to history from 2020/2021. Tickers that delisted or faded
  before 2026 are under-represented → survivorship bias in all backtests.
- `sector_id` / `market_cap_bucket` (swing Cat 11) are a **present-day snapshot**
  applied to all historical bars — a stock's 2026 cap bucket encodes its growth
  since 2021 (mild lookahead through a static feature).
The paper should state the universe definition date explicitly and note this bias.

### 0.5 Documented-and-retracted items (report honestly, do not cite as wins)
**VERIFIED** via LIVING_SUMMARY + memory:
- The meta "breadth/entry-day gate" used `trend_persistence` — a **forward label** —
  as a feature; it was retracted before wiring. Leak-free version had no edge.
- The 2026-07-07 confluence-discovery study consumed its one-shot test read:
  zero certified interactions. The meta-matrix test window is burned for
  confluence-style claims until ~6 more labeled months accrue.

---

## 0.6 Patches applied (2026-07-13) — the same-split selection bias is now fixed and quantified

Three of the four selection-bias findings above (§0.2 for momentum/HTF/meta, §1.4
for swing) have a concrete "how big was the bias" answer, obtained by adding a
**val-select / test-freeze** variant next to each original test-selected backtest
and running both on the current tree. Nothing about the underlying models changed —
only the order-policy grid search was moved off the reporting window.

New code (this session):
- `scripts/capstone/family_backtest_clean.py` — momentum & HTF order-policy sweep,
  selects on the validation window, freezes on test. Outputs:
  `strategies/{momentum_expansion,multi_ticker_swing_htf}/backtest/results/family_compare_clean/`.
- `strategies/multi_ticker_swing/backtest/sweep_v2_clean.py` — same patch for the
  swing 5m-confirmation sweep. Output: `strategies/multi_ticker_swing/backtest/results/sweep_v2_clean/`.
- `scripts/capstone/build_meta_scored_from_oof.py` — rebuilds the meta `s_combo`
  signal from the clean walk-forward OOF quality+upside scores instead of the
  deployed (partially in-sample) boosters, for use with the existing
  `signals/meta_context/meta_ranker/backtest_exits.py`.
- **Bug fix in the process**: `strategies/multi_ticker_swing/backtest/sweep_v2.py`'s
  local `load_raw_30m`/`load_raw_5m` assumed `timestamp` was always a DataFrame
  column. Most raw 30m/5m caches now store it as a `DatetimeIndex` (an on-disk
  format change since sweep_v2.py was last touched), so the loader was silently
  `except`-skipping most tickers on every sweep_v2 run — the reason the current
  `sweep_v2_summary.csv` best combo (n=6,904 trades) undercounts the stale
  `best_v2_grouped.json` (n=9,274). Fixed to match the canonical loader's fallback
  (`strategies/multi_ticker_swing/data/load_data.py::_ensure_utc_index`).

**Measured selection-bias magnitude** (same models, same grid, only the selection
window moved):

| Model | Test-selected (original, same-split tuned) | Val-selected / test-frozen (clean) | Bias factor |
|---|---|---|---|
| momentum_expansion | ret/DD **44.6x**, return 944%, PF 2.22, WR 66.5% (lgbm_classifier, tp=5/sl=5/top20/hold75 — rails to the loosest grid edge) | ret/DD **6.05x**, return 92.5%, PF 1.53, WR 74.7%, 3,876 trades (xgb_classifier — the actual deployed winner; tp=2/sl=4/top5/hold75) | **~7.4x inflated** |
| multi_ticker_swing_htf | ret/DD **41.4x**, return 887% (xgb_ranker, tp=5/sl=5/top20/hold75) | ret/DD **17.9x**, return 336%, PF 1.49, WR 39.0%, 23,173 trades (lgbm_classifier — the deployed winner; tp=5/sl=2/top20/hold25) | **~2.3x inflated** |
| meta exit-policy (target+20% full exit) | mean +6.15%, median +17.4%, win 68% (deployed-booster scores, partially in-sample) | mean +5.87%, median +15.7%, win 66.6%, n=1,259 (clean walk-forward OOF scores) | **~1.05x — negligible**, confirms the deployed-booster version was not meaningfully biased *for this particular comparison* |
| meta exit-policy (current live, rank-dropout g=0) | mean +0.98%, median +0.1% | mean +0.57%, median +0.07%, win 51.0%, n=3,327 | same qualitative conclusion either way: dropout is the worst tested exit |
| multi_ticker_swing sweep_v2 (§1.4) | WR 62.6%/60.0% long/short (stale `best_v2_grouped.json`, predates the loader fix, selected on its own reporting split) | WR **50.1%/47.2%** long/short, combined **48.7%**, PF 1.36, Sharpe 1.91, 4,096 trades (`sweep_v2_clean`, combo `e0.7_c6_trail_arm2.5_gb25_sl3.0` selected on VAL, frozen on TEST) | **~14 percentage points inflated** — the single largest correction in this audit |

**Interpretation for the paper**: the momentum, HTF, and swing order-policy
backtests were the most exposed — both test-selection and the grid's tendency to
rail to its loosest edge (e.g. momentum/HTF: tp=5×ATR, sl=5×ATR, hold=75 bars, the
maximum allowed in every case) combined to overstate ret/DD by 2–7x, and the
swing win rate by ~14 percentage points (62.6%/60.0% → 48.7% combined — the single
largest correction in this audit, and a materially different conclusion: swing's
5m-confirmed entries are closer to a coin flip than the original figures implied).
The meta exit-policy comparison, by contrast, held up well under the clean
re-score: its conclusion (rank-based `dropped_out` is the worst tested exit; a
fixed take-profit or scale-out is far better) is **not an artifact of the
leakage** — it reproduces almost exactly on walk-forward OOF scores. **Cite the
clean (`*_clean`) numbers as the paper's headline equity-tier results**; the
original test-selected numbers remain on disk as the audit's worked example of
the selection-bias magnitude.

**Caveat on `max_dd_pct`** in the momentum/HTF/swing tables above: these backtest
engines compute drawdown as the min of a *cumulative sum of per-trade `pnl_pct`*
(fixed notional per trade, not a compounded equity curve), so values like swing's
-74.48% are cumulative percentage-points given up across many trades' worth of
independent risk, not "the strategy's equity fell 74% peak-to-trough." Treat these
as a relative ranking signal across policies, not a literal account drawdown; a
real equity-curve drawdown requires a portfolio-level position-sizing/concurrency
model that these order-policy sweeps do not implement.

---

## 1. multi_ticker_swing (30m, primary system)

### 1.1 Labels — clean by construction
**VERIFIED** (`labels/build_labels.py`, `plots/generate_soft_swing_30m_plots.py`).
Soft swing-zone labels from fractal pivots (3 bars each side), shifted T-1
within-session, ±1-bar neighbor weights, session-aware windows, first-in-run filter,
and a 12-bar/1-ATR follow-through filter. Labels are forward-looking **by design**
(they are labels); no label information is written into feature columns.

### 1.2 Features — causal, with two structural caveats
**VERIFIED** (`features/build_features.py`, all 11 categories read):
- All rolling windows are backward-looking; no `shift(-)`, no centered windows,
  no bfill on features.
- Daily (Cat 12) context is `shift(1)` before mapping to intraday bars — a bar on
  day D sees only day D-1 daily values.
- Cross-asset context (Cat 8) uses `reindex(..., method="ffill")` — as-of joins.
- Gap follow-through features use only same-session completed bars.
- Fractal pivots are explicitly **excluded** from features (used only in labels);
  Cat 4 uses rolling 20-bar extremes + the causal `atr_swing_state` instead.
- Caveat 1: Cat 11 sector/cap identity is a present-day snapshot (see 0.4).
- Caveat 2: `_finalize_open_window_vol` forward-fills an open-window statistic
  through the session — value is stale-but-causal, fine.

### 1.3 Split & training — clean, small boundary overlap
**VERIFIED** (`models/train.py`): global time-sorted 70/15/15 row split, OOF
(5 sequential folds) on train, early stopping on val, test used only for reporting.
No purge (see 0.1; ~16-bar overlap). Neutral downweight and soft weights are
sample-weight-only (no label leakage). `random_state: 42` fixed.

### 1.4 Backtest sweeps — the paper's headline numbers are test-tuned and the trades file is gone — HIGH severity
**VERIFIED** (`backtest/sweep_v2.py`, saved artifacts):
- `sweep_v2` runs **180 parameter combos on one split (default `test`)**, picks the
  best by Sharpe **on that same split**, and writes the winning combo's grouped
  stats to `best_v2_grouped.json`. The advisor doc's *"62.6% long WR (4,690 trades),
  60.0% short WR (4,584)"* aggregates exactly from that JSON — i.e. these are
  **test-selected policy numbers**, not untuned out-of-sample performance.
- `long_wr`/`short_wr` in that file are **PnL-based win rates of the executed policy**
  (with 5m confirmation, trailing exits, stops), NOT raw directional accuracy.
  The advisor doc calls them "directional win rate" — the paper should rename or
  regenerate.
- The doc's "62.3% overall" does not reproduce: the trade-weighted combination of
  the saved groups is **61.3%** ((4,690×0.6262 + 4,584×0.5999)/9,274).
- `best_v2_trades.parquet` (the per-trade record behind the JSON) **no longer
  exists**, and the saved `sweep_v2_summary.csv` top rows (n=6,904–7,580, WR≈33%)
  are inconsistent with the grouped JSON (n=9,274) — the JSON is a **stale artifact
  from an earlier run/config that cannot currently be regenerated bit-for-bit**.

**Patched (2026-07-13, §0.6):** `strategies/multi_ticker_swing/backtest/sweep_v2_clean.py`
selects the combo on the validation split and freezes it on test, using the SAME
grid, after fixing the raw-loader bug that had been silently dropping most tickers
(see §0.6). Cite `sweep_v2_clean/best_v2_clean_summary.json` as the paper's swing
backtest number instead of the artifacts above; those remain on disk as the
worked example of both the selection bias and the loader bug's impact on trade
counts. The newer OOF ranker policy work (`results_oof/`, `BACKTEST_CONFIG_V2`)
already moved toward per-side grids — same caveat applies to whichever split the
grid ran on; not repatched in this pass.

### 1.5 Research→live feature parity — strong by construction
**VERIFIED** (`live/feature_builder.py`): live calls the **same**
`build_ticker_features()` used offline, same universe CSV encodings, same config.
Deviations:
- Live computes on a rolling 500-bar window; EWM-based features (EMA-10/20/50,
  daily EMAs) have warmup differences vs full-history values. With 500 bars ≥ 10×
  the largest span this is small but nonzero. **ASSUMED small; not numerically
  re-validated in this audit.**
- 252-bar rolling percentile features have identical values once ≥252 bars are
  present (min_periods=50 met by MIN_BARS=200 gate). **VERIFIED logic.**
- Ranker path (`compute_latest_full`) adds bar-location + as-of prior-day daily
  context from the training-export parquet (`DailyContextLookup` — as-of prior day,
  VERIFIED); the docstring notes it was validated identical to the offline parquet
  (**ASSUMED**, validation not re-run here).

### 1.6 Paper/live evidence (option returns)
The +40.6% / +28.1% / +36.2% option-return figures come from saved analysis CSVs
under `Data/analysis/multi_ticker_swing_live/experiments/` (present, locked by
`reproduce_results.py`). These are **paper-trading** results over 2 trading days
(May 28–29, 2026) — tiny sample, must be labeled as anecdotal paper evidence.

---

## 2. momentum_expansion (4H ranker)

### 2.1 Labels — clean
**VERIFIED** (`labels/expansion_labels.py`): forward stats over [T+1, T+25] 4H bars
(high/low/close paths, SPY-alpha, ATR-adjusted), per-ticker **causal** rolling
quantile ranks (`rolling(...).rank(pct=True)`), cross-sectional
`expansion_survival_score` ranks within-timestamp forward stats (label-side only).
Incomplete forward windows → NaN target (dropped). No feature contamination found.

### 2.2 Features — causal
**VERIFIED** (grep + structure of `features/feature_matrix_4h.py`): no `shift(-)`,
no `center=True`, no bfill. 106 features incl. daily/weekly context (shift-based),
regime and RS features. (Full line-by-line read not performed on all 744 lines;
**pattern-scan VERIFIED, exhaustive read ASSUMED**.)

### 2.3 Training matrix — one minor full-sample operation
**VERIFIED**: `_drop_correlated_features` computes pairwise correlations on the
**full matrix (train+val+test)** before splitting. At threshold 0.995 this only
removes near-duplicates, so practical leakage is negligible — but it is technically
test-informed feature selection. State it; do not fix for the paper.

### 2.4 Splits & selection
- Competition: 60/20/20 row split, no embargo (0.1); winner by test NDCG@10 (0.2);
  shipped booster early-stops on the test window (0.3). n = 1,390,783 / 463,594 /
  463,595 rows.
- Walk-forward OOF (`oof_preds.parquet`, 21-day embargo, 18-month train windows)
  is the clean OOS series and feeds the meta matrix. **VERIFIED** in
  `strategies/model_training/colab_competition.py::walk_forward_oof / date_folds`.

### 2.5 Live parity
**VERIFIED**: live runner (`live/runner.py`) builds features via the same
`build_ticker_features_4h` and now wires `load_1d` at both call sites — the
2026-06 "live dark" bug (df_1d=None → 15 all-NaN daily features → dropna killed
every row) is fixed in code. The candidate filter is OFF for training and ON for
execution by design (documented in config; the 2026-05-31 sweep validated broad
training + filtered execution).
**Residual risk:** rank-based selection with `min_score: 0.0` means live behavior
depends on universe breadth at scoring time; research/backtest used the full matrix
universe. **ASSUMED equivalent; not re-validated.**

---

## 3. multi_ticker_swing_htf (4H pivot swing)

### 3.1 Labels — clean
**VERIFIED** (`labels.py`): 4H fractal pivots (3/3), core shift 1 bar, ±1 zone,
forward quality stats over bars 13–38. Pivot detection requires 3 future bars —
label-side only. First/last `left`/`right` rows correctly zeroed.

### 3.2 Features / split / selection
Features are **the same** `FEATURE_COLUMNS_4H` builder as momentum (2.2 applies).
Competition: 60/20/20 (680,266 / 226,755 / 226,756 rows), winner lgbm_classifier
seed 46 **by test NDCG@10** (0.2 applies), no embargo (0.1 applies, 38-bar horizon).
Walk-forward OOF exists (`models/oof_preds.parquet`).

### 3.3 Live parity — indirect
**VERIFIED**: the HTF live runner does NOT re-score; it reads `htf_score` off the
shared meta matrix, which `update_meta_matrix.py` populates using the deployed
`HTFSwingScorer` on features built by the same `build_ticker_features_4h`.
Parity therefore reduces to the meta matrix updater's parity (4.4).

---

## 4. meta_ranker (confluence ensemble)

### 4.1 Training matrix — best-designed pipeline in the repo
**VERIFIED** (`signals/meta_context/build_meta_ranker_matrix.py`):
- Base momentum/HTF scores come from **walk-forward OOF with 21-day embargo**
  (leak-free stacking — the meta model never sees an in-sample base score).
- All context feeds (theme, news catalyst, forward guidance, FINRA) join
  **as-of prior day** (`_asof_prior_day*`).
- Labels (`trade_quality`, `meta_good`, `meta_upside`) are built from the forward
  outcome columns carried in the momentum OOF.
- The manifest records explicit `leakage_controls` and walk-forward config.

### 4.2 Competition split/selection
Same shared engine: 60/20/20 (948,456 / 316,152 / 316,152 rows), no embargo (0.1),
winners by test NDCG@20 (0.2), final-fit boosters see test for early stopping (0.3).

### 4.3 Exit-policy backtest ("holdout 2025-07-01+") — partially in-sample scores — MEDIUM-HIGH severity
**VERIFIED** (`backtest_exits.py`): it reads `/tmp/meta_scored.parquet`, which is
produced by `score.py` using the **deployed** boosters. Those boosters' final fit
trained on the first ~80% of matrix days — a window that extends **past 2025-07-01**.
So a meaningful prefix of the "holdout 2025-07-01+" exit backtest is scored by a
model that trained on those rows. The *relative ranking of exit rules* (the purpose
of that backtest — stop/TP-scale-out/trail vs rank-dropout) is plausibly robust to
this, but the absolute per-trade numbers (+6–7% mean) should not be cited as clean
OOS.

**Patched and confirmed (2026-07-13, §0.6):** `scripts/capstone/build_meta_scored_from_oof.py`
rebuilds `s_combo` from `models/{quality,upside}/oof_preds.parquet` (walk-forward,
21-day embargo) instead of the deployed boosters, then `backtest_exits.py` runs
unmodified against it (762,932 scored rows, 407,698 in the 2025-07-01+ holdout).
Result: **the original conclusion was NOT an artifact of the leakage.** Clean
numbers are close to the original — target+20% full exit: mean +5.87% / median
+15.7% / win 66.6% (n=1,259) vs the original +6.15%/+17.4%/68%; current-live
rank-dropout(g=0): mean +0.57%/median +0.07%/win 51.0% (n=3,327) vs the original
+0.98%/+0.1%. Same ranking, same magnitude order — dropout remains clearly the
worst tested exit, target/scale-out remains clearly the best. Cite either the
locked clean numbers (`reproduce_results.py`'s `meta_exit_policy` rows) or the
original — both support the same conclusion; the clean ones are the defensible
citation. Options results there remain a leverage MODEL (stock paths only).

### 4.4 Live parity
**VERIFIED**: `update_meta_matrix.py` re-uses the training feature builder
("skew-free"), scores with deployed models, and computes the same cross-sectional
context columns. Structural difference vs research: research matrix uses OOF scores,
live matrix uses deployed-model scores → the meta model consumes a slightly
different score distribution live than it was trained on. This is the standard
stacking compromise; state it in the paper.
Known live defect (2026-07-09, fixed): rank-based `dropped_out` exits + uncapped
option notional destroyed realized PnL (-$38k) despite selection edge — execution,
not signal. Exit policy has since been switched to stop/scale-out/trail (backtested).

---

## 5. spy_intraday (baseline / limitation study)

### 5.1 Status
The paper uses this as the honest negative result: good-looking research metrics,
weak live performance after realistic option execution. That framing is supported.

### 5.2 What was verified
- **Fixed on-disk splits**: `Data/processed/spy/splits/...` with
  `train_idx.npy / val_idx.npy / test_idx.npy` — fixed boundaries, reproducible.
- **GA feature selection** (`Models/ga_xgboost/ga_xgboost.py`) fits on the TRAIN
  arrays only, with an internal sequential train/val split for fitness — the GA
  does not see test. 1,541 → 200 features per side.
- A leakage checker exists (`Features/test_leakage.py`: label-correlation /
  identical-column scan). **Not re-run in this audit.**
- Artifacts locked: `Data/models/ga_xgboost/10min/{long,short}/swing/`
  (xgb_model.json, best_mask.npy, p_*_test.npy, meta.json). Live path loads
  xgb_model.json + best_mask (matches memory + meta.json inspection).
- Label mode `swing` (pivot labels), matching the corrected target choice.

### 5.3 Assumed / not verified
- The 1,541-feature engineering stack (feature_sets/) was not line-audited here;
  it long predates the other modules and its live weakness is already the paper's
  point. **ASSUMED adequate for a baseline narrative.**

---

## 6. What the paper can safely claim (summary table)

| Claim | Basis | Status |
|---|---|---|
| Swing model test-set classification metrics (acc, per-class precision/WR) | `models/eval_metrics.json` + recompute from `p_swing_probs.parquet` | CLEAN (16-bar boundary overlap, negligible) |
| Swing "62.6%/60.0% backtest win rates" | `sweep_v2/best_v2_grouped.json` | SUPERSEDED — cite `sweep_v2_clean/best_v2_clean_summary.json` instead (§0.6, §1.4) |
| Swing paper-trading option returns (May 28–29) | analysis CSVs | Paper-trading, 2 days, anecdotal — label as such |
| Momentum/HTF winner test NDCG@10 (model selection) | competition bundles | Selection-biased (picked on test) — pair with the seed spread or cite OOF instead |
| Momentum/HTF order-policy equity backtest (CAGR/ret-DD tier) | `family_compare/comparison_summary.json` | SUPERSEDED — cite `family_compare_clean/comparison_summary_clean.json` instead (§0.6; was inflated 2.3–7.4x by same-split selection) |
| Momentum/HTF walk-forward OOF Spearman / top-K lift | `oof_preds.parquet` | CLEAN (21d embargo) |
| Meta OOF-scored ranking quality | `models/*/oof_preds.parquet` | CLEAN |
| Meta exit-policy backtest absolute returns | `backtest_exits.py` | CLEAN once run against OOF-derived `s_combo` (§0.6, §4.3) — confirms the original deployed-booster version's conclusion, cite either |
| SPY baseline research-vs-live gap | fixed splits + live audit logs | CLEAN (the point of the section) |

`scripts/capstone/reproduce_results.py` regenerates every number in the CLEAN and
SUPERSEDED-fixed rows from fixed artifacts and prints the metrics table with these
tags; `research/capstone/results_lock.json` pins the values + artifact hashes, and
`scripts/capstone/tests/test_results_lock.py` asserts them.
