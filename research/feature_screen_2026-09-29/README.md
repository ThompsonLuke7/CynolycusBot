# 4H feature study, Stage 1: label arms, local screen, and "is the vol-tilted model usable anywhere?"

Date: 2026-09-29/30. Plan: `docs/superpowers/plans/2026-09-25-4h-feature-study-plan.md`.
Everything here is training-free and in-sample-free: it uses the deployed models' walk-forward OOF scores
(momentum 2022-11-14..2026-05-14) and reads production parquets read-only.

Code (all in `strategies/momentum_expansion/ablation/`):
- `candidate_features.py`: daily blocks (MA structure, candle, leader/base, volume). Lagged one session and NaN'd while a non-organic corporate action sits inside each column's lookback.
- `label_arms.py`: L0 = current MFE label; L1 = MFE ranked within same-bar ATR% quintiles; L2 = executable 10-session return minus same-bar ATR%-decile peers; R = raw executable return.
- `run_feature_screen.py`: builds the panel and runs per-bar univariate and incremental (partial) IC, with a week-block bootstrap, per-fold sign, BH-FDR, and a redundancy test against the deployed keeps.
- `run_subset_eval.py`: deployed-model top-3 inside price/beta/ATR/ADV/regime subsets vs a "most volatile 3" control.
- Tests: `tests/test_feature_screen.py`. These cover no same-bar or future reads, the split mask, vol-neutral arms, the forward split guard, and the filtered-OOF guard.

Panel: the last 4H bar of each session (it contains the close). Entry is the next session's open, and exit is the close 10 sessions later, SPY-excess. The corporate-action guard drops windows that contain a non-organic gap. Momentum panel: 788k rows, 875 sessions, 1,080 tickers. Sanity check: corr(own 10-session return, matrix 25-bar forward close return) = +0.81.

Final test reserved: rows after 2026-05-14 are not in any matrix and were not touched. Stage 2/3 must be judged there.

## 1. The label is the problem (momentum)

Per-bar Spearman IC, mean over 875 bars (folds with the same sign out of 7):

| signal | L0 current MFE | L1 MFE within vol bucket | L2 vol-matched realized excess | R raw realized |
|---|---|---|---|---|
| deployed score | +0.372 (7/7) | +0.066 (7/7) | **-0.054 (7/7)** | +0.005 (2/7) |
| daily_atr_pct alone | +0.364 (7/7) | +0.022 | -0.059 (7/7) | +0.007 |
| dist_to_52w_low_atr | -0.099 | +0.019 | **+0.046 (7/7)** | +0.022 (6/7) |
| daily_dist_200dma_atr | -0.082 | +0.010 | **+0.039 (7/7)** | +0.020 (5/7) |

The deployed score is a volatility ranking. Its L0 IC is almost exactly that of daily ATR% alone. On a vol-matched realized basis it is reliably negative. Single trend-level features that the model barely uses beat it on L2 and R.

## 2. Screen verdicts (momentum, label L2; incremental IC controls for the score, ATR%, 4H ATR%, beta, dollar-vol percentile, log price)

- **Deployed features that carry L2 signal**: all trend/level features. These are dist_to_52w_low (+0.029), daily_ema_stack, daily_dist_200dma, near-52w-high, bars_since_52w_high (-), corr_spy_60, days_to_earnings (-), weekly_trend_state, ema_dist_100, atr_expand_14_60, and range_contraction_20_60.
- **Dead weight on L2 (58 of 106)**: every 1-20 bar return and RS column, breakout/compression/base-count columns, realized-vol columns (expected once vol is controlled), and the earnings-window flags. This is a linear screen. Stage 2 must confirm it before anything is dropped, since a tree can use interactions.
- **Undefined (17)**: calendar and market-regime columns. They are constant within a bar, so cross-sectional IC is undefined. week_sin/cos etc. are dropped by the plan anyway.
- **Leak-risk, excluded from verdicts**: `market_cap_bucket` is one static 2026 snapshot stamped on all history, with alphabetical category codes. `low_price_flag` passes 7/7, but the universe is survivor-shaped: it grows 965→1,073 names from 2022 to 2026 and never shrinks, so dead sub-$5 names are absent.
- **New candidate blocks, beyond the 30 deployed keeps**:
  - On L2 they are almost fully redundant. Only `d_upper_wick` adds (-0.006, q<0.01). The daily-MA block mostly restates `daily_dist_200dma_atr`/`daily_ema_stack` (e.g. `d_ema200_dist_atr` has the same IC to 4 decimals).
  - On L1, 9 add, mainly leader/base: `d_pullback_126h_atr` +0.020, `d_base_len_126` +0.013, `d_run_x_pullback` +0.013 (7/7), `d_ret_126` +0.010, `d_days_above_ema100`, `d_ret_63`, `d_range_atr`, `d_days_above_ema50`, `d_upper_wick` (-).
- Block 5 (the bulk pandas-ta pool) is **not run yet**. It is available locally but slow (~1 s per ticker per indicator set).

## 3. Can the vol-tilted model be used anywhere? (the user's question)

### 3a. As a directional selector inside a subset: no

`run_subset_eval.py`: the model's top 3 within 30 subsets (price bands, beta/ATR/ADV quintiles, low-price × high-beta combos, SPY trend, VIX). Each is compared with the same subset's equal weight and with "top 3 by ATR%". 10-session SPY-excess returns, 2-week-block bootstrap, 7 folds.

- In no subset does the model beat the most-volatile-3 control after FDR (all q ≥ 0.49 on the winsorised edge).
- Low price × high beta, specifically:
  - <$10 × beta Q4-5: -0.52%/trade [-1.83, +0.83].
  - <$20 × beta Q5: -0.69% [-1.92, +0.64].
  - Within-subset IC is about 0, and the top-3 hit rate vs the subset median is about 50%.
- The raw "top 3 beats the subset by +4-6% per 10 days" in low-price/high-vol cells has three problems:
  - it is matched by the vol control;
  - it halves when returns are winsorised at the pooled 1st/99th percentile (all names: +5.9% → +2.7%), so it is carried by a few +100% names;
  - it sits in the part of the universe most inflated by survivorship.
- Best cell: the most-liquid ADV quintile, winsorised edge +0.92% [+0.21, +1.72], 5/7 folds, IC +0.027. That is one of 30 tests, q=0.49. **Not measurable**, not a rejection.
- MDE is about 1-2 pp per 10-session trade.
- Residualising the score on ATR/beta/price and taking its top 3 underperforms equal weight (-0.8%). Beyond vol, the score has no direction.

### 3b. As a volatility-expansion forecaster: yes

Partial IC of the score vs the forward 12-session range (up + down excursion):

| controls | partial IC | bars positive |
|---|---|---|
| trailing ATR% / 4H ATR% / RV20 | +0.266 | 99% |
| + RV5, vol regime, ATR expansion, compression, vol-of-vol, price, beta, earnings timing | +0.152 | 97% |

It is two-sided: upside excursion +0.062, downside +0.065, 10-session return -0.001. Walk-forward linear forecast of log range (fit before each year, 21d embargo), OOS R²:

| test year | trailing vol only | + score |
|---|---|---|
| 2024 | 0.454 | 0.507 |
| 2025 | 0.472 | 0.490 |
| 2026 YTD | 0.405 | 0.442 |

So the model is usable as a **forward-range forecast for the harness**: vol-target sizing and stop width. Combined with a directional selector, it can also serve as a gate so directional trades go to names whose expected range clears costs. It is **not** usable as a long-vol options selector yet. That needs an implied-move comparison, and the IV-surface capture only started 2026-09 (and options carry the ~35% round-trip hurdle).

## 4. HTF: the OOF is selected on the future (blocking finding)

`colab_competition.walk_forward_oof` drops rows whose regression target is NaN. For HTF that target, `htf_swing_score`, exists only inside long/short swing zones built from pivots with `pivot_right_bars=3`, i.e. 3 future bars. So the HTF OOF, and its training set, holds only the 48.8% of rows later confirmed near a swing low or high.

Evidence:
- On those rows, `ret_3` has IC -0.104 vs the 10-session excess return, uniform across every ADV quintile including the most liquid (loser-minus-winner +2.5%/10d at a $161 median price).
- On all rows in the momentum panel, the same feature has IC -0.002.
- On the shared (ticker, date) rows, momentum also shows -0.097.

This is selection, not a feature leak: HTF features align with the raw 4H bars exactly.

Consequences:
- The HTF Stage-1 screen and subset results are invalid. They are quarantined at `ablation/results/feature_screen/htf_INVALID_future_selected_oof/`.
- `run_feature_screen.assert_oof_unfiltered` now fails fast when OOF covers <90% of matrix rows. HTF: 48.8%, rejected. Momentum: 100.1%, passes.
- Every past conclusion drawn from the HTF OOF is suspect. That includes rank-depth "HTF OOF monotone k1>k2>k3", "HTF has real signal at the top", and HTF p@1. This is a plausible explanation for the unresolved OOF-vs-live contradiction (live HTF top-1 was significantly negative).
- Meta's matrix left-joins `htf_score` from that OOF (81% present). The missingness itself barely moves Meta's label (within-bar rank 0.505 vs 0.497), and `htf_score`'s IC with meta_label is -0.03. Meta is not obviously inflated, but its HTF inputs were trained on a population that does not exist live.

Fix: HTF must be trained and OOF-scored on all rows. The L1/L2 arms are defined for every row, so adopting them for HTF removes the pivot selection by construction. This needs a Colab run.

## Next steps (roadmap: 4H study)

1. Stage 2 (Colab), momentum: arms L0 vs L1 vs L2, each with a permuted-label control.
   - Feature sets: deployed; deployed minus calendar; deployed minus the 58 dead columns; plus leader/base (L1 adds).
   - Judge on the untouched 2026-05-15+ window: top-k vs a vol-matched control, and NDCG.
2. HTF: rebuild the labels as L1/L2 on all rows, and regenerate the OOF over all rows before any HTF result is trusted.
3. Harness (can run locally): sizing/stop-width backtest using the score-augmented range forecast vs trailing ATR.
4. Optional: the block-5 pandas-ta pool through the same screen; a PIT market cap to replace `market_cap_bucket`; a survivor-free universe to test the low-price vol premium at all.
