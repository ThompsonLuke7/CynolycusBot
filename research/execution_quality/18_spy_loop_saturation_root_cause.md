# SPY live-loop saturation: root cause, measurements, and why the obvious fix is blocked

2026-09-04, following the 2026-09-03 daily review.

## Symptom

The SPY live loop ran `busy=100%` from roughly 09:50 to 16:12 ET on 09-02 and
09-03. Individual 10-minute SPY bars took 331s, 409s, 477s, 551s, 624s, 715s,
930s and 1,168s against a 10-minute (600s) decision budget. Bar-queue depth
reached 248. The daytrader has not confirmed a self-initiated entry in three
sessions.

## Root cause

The bar loop is single-threaded (`UI/live_dashboard.py`, `_run`). Every
10-minute decision reaches
`LiveIndependentMetaXGBAgent._build_setup_feature_frame`, which calls
`build_tree_feature_frame_from_1m(df_1m)` on the **entire** 1-minute history —
553,094 rows, ~5.6 years — producing a 56,706 x 2,299 frame, then predicts the
competition ensemble over all of it. Only the newest row is consumed.

A second, smaller instance of the same pattern is in
`LiveMetaXGBAgent._build_base_frame`: it has an incremental branch guarded on
`_precomputed_base_frame`, but that attribute is only ever populated from the
warmup parquet handed in at construction. Live on 09-03 that file was missing —
`[live] Cached setup feature frame unavailable: Missing setup feature frame
file: Data/inference/spy/10min/debug_matrices_warmup/spy/
live_meta_matrix_on_trace_ts_live_2026_03_27.parquet` — so the full-rebuild
branch ran every decision and never cached its result.

## Measurements

Measured on the live prefill cache
(`Data/raw/spy/spy_intraday_1min_runtime_rth_cache.parquet`, 553,094 rows),
Python 3.12, this machine, no GPU.

`build_tree_feature_frame_from_1m` — the dominant term:

| 1m input | out rows | out cols | elapsed |
|---|---:|---:|---:|
| 30d tail (8,589) | 872 | 2,167 | 9.2s |
| 120d tail (32,396) | 3,269 | 2,274 | 38.5s |
| 900d tail (234,829) | 24,146 | 2,299 | 278.0s |
| full (553,094) | 56,706 | 2,299 | **392.0s** |

`build_meta_feature_frame_from_1m` (without a GA predictor) is not the problem:
0.6s at 30d/120d, 2.7s at 900d, 5.3s full.

So the ~392s tree rebuild, plus ensemble inference over the same frame,
accounts for the observed 331-1,168s per decision.

## Why the obvious fix is blocked

Caching the frame and recomputing only a trailing overlap makes the decision
cheap, and a prototype did exactly that. It must be rejected: it changes model
inputs.

Comparing the newest 20 rows of an incrementally-built tree frame against a
full rebuild, over the 430 features selected by the live swing models
(`Data/models/ga_xgboost/10min/single/swing_{support_,}single/
selected_features.txt`), **53 selected features differ**:

* **29 become NaN.** They live on the `__1d` and `__4h` higher timeframes
  (`DEFAULT_FEATURE_TIMEFRAMES` = 30m/1h/4h/1d) and need tens of *daily* bars.
  A 15-day overlap supplies ~15. Lengthening the overlap fixes these, at the
  cost of the speedup — a daily indicator with a 26-period EWM needs on the
  order of a year of history for the initial condition to decay out.
* **24 drift numerically**, and six of those cannot be fixed by *any* finite
  overlap because they are unbounded accumulators or recursive filters counted
  from the first row of whatever frame produced them: `OBV__30m`, `AD__4h`,
  `OBVe_12__1d`, `atr_swing_bars_since_flip` and its `__4h`/`__1d` variants.
  Others in this group (`VIDYA_14__1d`, `RSX_14__1d`, `HWM/HWW/HWPCT_1__1d`,
  `PSARr`, `FISHERT_9_1__1d`) are recursive with very long memory.

This is why the live dashboard's `meta_base_frame_append_lookback_days` default
is **900** while `live_runner.py`, `replay_runner.py` and `live_inference.py`
all default to **120**: 900 days is sized for the daily-timeframe features.
That inconsistency is itself worth resolving — 120 days is too short for the
`__1d` features, so research and live are not computing the same values today.

The same reasoning applies to the meta base frame, where two selected features
(`day_id`, `trend_persistence`) are counted from the frame origin and one
(`range_regime_8_32`) is EWM-based.

## What would make this safe

In rough order of value:

1. **Compute only the indicators the models use.** 2,299 columns are built and
   430 are consumed. Indicators are independent, so restricting the set is
   value-preserving by construction. Needs a reliable feature-name ->
   pandas_ta-indicator mapping, because one indicator emits several columns.
   This is the only option that is both fast and provably value-identical.
2. **Move the decision off the ingest thread** so a slow decision stops
   starving everything else sharing the loop. Does not make the decision
   faster and introduces concurrency on policy state.
3. **Accept an incremental frame and revalidate the models against it.** Only
   defensible with a full backtest on incrementally-built features; the six
   unbounded accumulators would have to be dropped from the feature set first,
   since they are not reproducible in a streaming context at all — arguably
   they should not be model inputs regardless.

Restoring the missing warmup parquet is *not* sufficient: it would move live
onto the 900-day overlap path (278s, still over budget once inference is added)
and that path has the same accumulator discontinuity at the join.

## Reproducing

Benchmarks and the equivalence comparison were run ad hoc against the live
prefill cache and the model feature lists; both are described precisely enough
above to rebuild. Nothing in `Data/` was modified.
