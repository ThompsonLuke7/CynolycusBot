# Promoting the OOF long+short rankers to live (30m multi_ticker_swing)

`RankerSwingScanner` ([ranker_scanner.py](ranker_scanner.py)) replaces the classifier-based
`SwingScanner` with the walk-forward OOF long + short rankers. It is a **near drop-in**: it
reconstructs `p_long_dir` / `p_short_dir` so `entry_threshold`, `ev_score`, the SPY `spy_min` macro
veto, the catalyst tilt, options execution, and the dashboard all work **unchanged**.

## How it works
1. Run the **long** ranker and the **short** ranker on every candidate.
2. Map each raw score → `P(swing winner)` via a per-side **isotonic calibrator** (`calib_{long,short}.joblib`, fit on OOF preds).
3. Normalise to directional probs: `p_long_dir = P_long/(P_long+P_short)` — same 0..1 scale the classifier produced.
4. Direction = stronger side clearing `entry_threshold`; `ev_score = p_dir·avg_win + (1−p_dir)·avg_loss` as before.

`get_directional_p("SPY")` returns the same `(p_long_dir, p_short_dir)`, so the macro filter is untouched.

## The one change to go live (in `runner.py`)
```python
# from strategies.multi_ticker_swing.live.scanner import SwingScanner
from strategies.multi_ticker_swing.live.ranker_scanner import RankerSwingScanner
...
self._scanner = RankerSwingScanner(            # was SwingScanner(self._fb, model_path=MODEL_PATH, ...)
    self._fb,
    max_entries_per_bar=max_entries_per_bar,
    catalyst_signal=catalyst_signal,
)
```
(`model_path` is accepted-and-ignored, so even leaving it in place is harmless.)

## Feature gap — CLOSED
`feature_builder.compute_latest_full()` now produces the full **192-feature** set:
- **bar-location** (~28 feats) computed live via `context_features.compute_bar_location()`, which reuses
  the exact offline functions (`signals.location_features`). **Validated 28/28 identical** to the
  precomputed `bar_location_context_features.parquet` at a common timestamp.
- **daily context** (~45 feats: earnings/macro/treasury/news/theme) via `DailyContextLookup` — as-of
  prior-day from `daily_context_features.parquet`, matching the training join policy.

`RankerSwingScanner` calls `compute_latest_full()`. Two operational notes:
- **Refresh dependency:** the nightly pipeline must keep `daily_context_features.parquet` current
  (it currently ends at the training cutoff). Stale = the daily context lags, but base + bar-location
  (the high-importance features) are always live-fresh.
- **Perf:** bar-location is recomputed per ticker each 30m scan; if the full-universe scan gets tight
  inside the 30-minute window, cache/incrementalize the location frame.

## Per-side order policy (from the grids — apply in the execution/risk config, not the scanner)
- **LONG:** TP **3.5×ATR**, SL **2.0–2.5×ATR** (let winners run).
- **SHORT:** TP **2.5×ATR**, SL **2.0×ATR** (bank the drop fast; shorts are ~2× the drawdown — size tighter).
These differ by direction, so the live exit logic should branch on `signal.direction`.

## Macro filter
With a genuinely predictive short ranker (long/short hedge, combined DD −2.9%), the old "puts only when
market's bad" gate is too restrictive. Recommend relaxing the short-side `spy_min` (let the combined
book self-hedge) rather than a hard short veto.

## Artifacts
- `models/oof_ranker_20260618/` — `model_ranker_{long,short}.{joblib,native}`, `calib_{long,short}.joblib`, `meta.json`
- `data/bundle/oof_preds_{long,short}.parquet` — leak-free OOF scores (validation/backtest)
- Tests: `tests/test_ranker_scanner.py`

## Before flipping live
The scanner is unit-tested but has **not** been run against the live feature builder end-to-end (needs a
running market session / paper feed). Recommend a **paper-trading shakeout** first via `combined_server`
(swing-paper queue) before the live queue.
