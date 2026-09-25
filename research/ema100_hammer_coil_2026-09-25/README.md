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
