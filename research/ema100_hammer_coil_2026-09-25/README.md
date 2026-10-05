# AMD 2026-09-03: did we catch it, and does the "100 EMA + hammer + coil" setup have edge?

Status: in-sample event study on the full 2020-07..2026-09 bar cache (no held-out set). Research only; nothing wired live.

## 1. What the live stack decided (2026-08-26 .. 2026-09-11)

AMD 1d: 09-03 low 440.5 vs EMA100 445.0, close 456.2 (hammer), then 629.3 on 09-24 (+38%).

| Module | AMD on/around 09-03 | Source |
|---|---|---|
| Momentum Expansion | rank ~780-1,070 of ~2,845 (score 0.04-0.06); only top-10 are traded | `meta_ranker_matrix.parquet` (live, deployed models) |
| HTF Swing | rank ~550-1,480 of ~2,845 (score 0.13-0.23) | same |
| Meta Ranker | never in top-10 targets | `Data/inference/meta_ranker/live_signal_audit.jsonl` |
| Dealer Ranker | swing rank 607/721 on 09-03, change direction **bearish**; flipped to bullish-change rank 10 on 09-04 (after the move started) | `Data/dealer_positioning/rankings/` |

AMD appears in no live audit or trade record. Its live theme was `midstream_energy_infrastructure` then `analog_mixed_signal_semiconductors`, which is a theme-assignment bug.

Why the models missed it by design: both base models rank the whole ~2,850-name universe on ~5-15 day forward max-return/alpha composites (`momentum_config.LABEL_CONFIG`, `multi_ticker_swing_htf/config.PIVOT_LABEL_CONFIG`). A megacap at a pullback low scores in the middle of that distribution because small caps routinely produce larger forward moves. No feature encodes "distance to EMA100", "hammer", or "leader pulling back".

## 2. Event study (`event_study.py`)

Universe: top 300 by 60d median $-volume, ranked per date. Signal at the day-t close, entry at the t+1 open. Returns are excess vs SPY. 95% CI from a month-block bootstrap. Events with a |1d return| >35% inside the window are dropped (split guard).

| Arm | n | 10d mean xs | 95% CI | hit |
|---|---|---|---|---|
| A all liquid days | 444k | -0.19% | [-0.39, +0.02] | 47% |
| C uptrend + touch EMA100 | 15.6k | -0.23% | [-0.49, +0.04] | 47% |
| D C + hammer | 2,154 | -0.01% | [-0.38, +0.40] | 48% |
| F C + hammer + coil (user's setup) | 420 | +0.22% | [-0.55, +1.06] | 48% |
| H F + prior leader (126d ret >40%) | 43 | +3.47% | [-0.53, +7.18] | 65% |
| K F, top-50 megacaps only | 59 | +0.41% | [-1.06, +1.98] | 51% |

Findings:
- The user's setup (F) is **not measurable here as an edge**. The CI half-width is ~0.8pp, so an edge under ~1pp over 10d cannot be detected. It is not evidence of no edge. It is unstable by year (2025: -1.7%, 2023: +3.0%).
- AMD has touched EMA100 in an uptrend 36 times since 2021, and most touches did nothing or failed (e.g. Aug-Sep 2023, Apr/Jun/Oct 2024, Feb 2026). 2026-09-03 is AMD's only F event.
- Megacap restriction does not help.
- The only interesting arm is H (leader base + full setup): +3.5% at 10d, 65% hit. But n=43, the CI crosses zero, and it was picked post hoc from 11 arms x 3 horizons. It includes the motivating AMD event (+22.5%); without it the mean is ~+3.0%. The bar cache is survivor-shaped, which flatters "leader" arms.

## 3. Next testable variant
Pre-register arm H (leader pullback to EMA100 + hammer + coil) as a **shadow watch**. Evaluate it prospectively and on a survivorship-clean universe. Also add EMA-distance, candle-shape, and leader-base features to an ablation of the HTF ranker. Do not wire into live sizing on this evidence.

## 4. Feature-matrix review (2026-09-25 follow-up)

Correction to §1: the Momentum (106 features) and HTF (111 features) matrices **do** include MA distance and coil features. MA distance: `ema_dist_10/20/50/100` in ATR units, `ema_stack_4`, `daily_ema_stack`, `daily_dist_200dma_atr`. Coil: `compression_5_20`, `is_compressed_5_20`, `compression_count_20`, `range_contraction_20_60`, `close_tightness_10`, `base_range_60_atr`, `base_position_60`. Both share `FEATURE_COLUMNS_4H` (`strategies/momentum_expansion/features/feature_matrix_4h.py`), which uses pandas-ta only for ADX/RSI/MACD. The SPY-intraday "all pandas-ta indicators" path (`strategies/spy_intraday/Features/feature_sets/pandas_ta_indicators.py`) was never ported to these modules.

Actually missing:
- The **daily** 100 EMA. The 4H `ema_dist_100` is roughly a daily EMA50, and daily bars only get a 200DMA distance.
- Any candle-shape feature (wick/body ratios, hammer/engulfing).
- A 3-6 month "leader" return. The longest window is `weekly_ret_4`.

## 5. Stop-based execution (`stop_exit_study.py`)

Entry at open t+1. Stop 0.2% under the signal-day low. Exit: time15 (close t+15), or trail20 (close below EMA20, armed only after the first close above EMA20; max 40d). Raw returns, no costs.

| Arm | exit | n | mean ret | 95% CI | win | mean R |
|---|---|---|---|---|---|---|
| C touch100 (control) | time15 | 14,541 | +0.11% | [-0.16, +0.37] | 24% | +0.25 |
| F hammer+coil | time15 | 396 | -0.04% | [-0.63, +0.59] | 22% | -0.20 |
| F hammer+coil | trail20 | 397 | -0.33% | [-0.74, +0.12] | 25% | -0.20 |
| H F+leader | time15 | 39 | +0.54% | [-2.09, +3.72] | 31% | -0.13 |
| H F+leader | trail20 | 39 | -0.23% | [-1.79, +1.23] | 38% | -0.27 |

The tight stop under the hammer low (~2%) is hit by 70-78% of trades, and the median outcome is -1R. No arm beats the control. Stop-and-trail execution does not create the edge that fixed-horizon returns lacked. H's +3.5% fixed-horizon mean does not survive a stop, so its winners typically dipped first.

## 6. Theme labels scrambled by a carry-forward off-by-one (FIXED 2026-09-25)
Correction: dense membership (every ticker scored against every theme) is by design. For non-noise tickers the argmax agrees with the hard HDBSCAN cluster 97.6% of the time, and the clusters are coherent. The real bug was in the labels. Step05 names this week's clusters by matching them to last week's centroids (`LABEL_STABILITY_THRESHOLD` 0.90). `_load_prior_centroids` built those centroids from last week's cluster ids, but named them from the registry snapshot *before the newest*. Step05 runs before this week's registry is written, so that snapshot was two runs old, and the ids pointed at different groups. Names therefore random-walked week to week.
- Proof: the 9/21 "liberty_media_formula1" centroid was 9/14 cluster 69, which was `memory_storage`. So the semis cluster (AMD, ARM, INTC, MU, SNDK, STX, WDC, SMCI, DELL, MCHP) inherited the F1 name.
- Scale on 9/21: 55 of 138 clusters (40%) whose description names tickers contain none of them. Examples: petroleum_refining = hotel REITs; precious_metals_mining = medical devices; latam_airport_operators = bitcoin miners/AI datacenters.
- Fix: `step06_discovery._load_prior_centroids` names prior clusters from the registry snapshot with the same date as the `.prior` clusters file, and returns {} if there is none. Also added `--relabel-all` (`weekly_run(relabel_all=True)`) to skip carry-forward once. Current names are wrong and would otherwise be carried forward indefinitely. Three regression tests were added, two of which fail on the old code; dynamic_theme suite 104 pass.
- NOT yet run: one weekly run with `--relabel-all` (~188 Claude labeling calls) is needed to repair the names.
