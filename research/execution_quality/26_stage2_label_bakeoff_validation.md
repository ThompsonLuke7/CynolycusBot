# Stage 2 label bake-off: validation read. No arm qualifies; the final test stays sealed.

2026-11-07. `strategies/momentum_expansion/ablation/eval_stage2.py`, rule `v1-2026-11-07`
(pre-registered before any result), results hash `9b2c0b913edc`, git `87625f4086`,
15/15 arms, 1,000 week-block bootstrap draws.
Outputs: `strategies/momentum_expansion/ablation/results/stage2_eval/`
(`validation_report.md`, `selection.json`, `validation_{ic,paired,topk}.csv`).

**Scope.** The momentum and HTF retrain targets only: 15 XGBoost arms trained on the
Stage 2 matrix (walk-forward with 24-month train and 6-month test folds; OOF from
2022-04).
- Validation = OOF bars before 2026-05-15, minus each yardstick's embargo.
- Not in scope: the deployed momentum OOF (it predates the current universe; L0 stands
  in for it), Meta, and the live execution path.

## Decision

| slot | yardstick | pick | best candidate | why it failed |
|---|---|---|---|---|
| momentum | L2_10 | **none** | L2_20: IC +0.012 [−0.002, +0.025] | IC CI includes 0; noise-twin gap q = 0.083 |
| htf | L2_20 | **none** | L2_20__no_cal_noearn: IC +0.018 [−0.001, +0.037] | IC CI includes 0; noise-twin gap q = 0.12 |

- No live change follows from Stage 2.
- The 2026-05-15+ final test was **not opened**. Under the rule there is nothing to
  confirm. Opening it now would also waste it: it holds about 4 months of matured
  labels, against about 4 years in validation. Its MDE would be roughly 3.5× the
  validation MDE (about 0.07–0.10 IC, assuming the same per-week variance), far above
  any effect seen here.

## Findings

**1. The deployed label (L0, MFE ≥ 20%) is a volatility ranker with no other
information.**
- Its IC on the vol-neutral yardsticks is −0.051 / −0.064, against −0.055 / −0.069 for
  the VOL row, which ranks by `daily_atr_pct` alone.
- It forecasts forward range as well as ATR does: fmfe_10 IC is +0.384 for L0 and
  +0.370 for VOL.
- Its top-3 picks sit at the 98th ATR percentile.
- Once vol, beta, ADV and price are regressed out of the score, its top-3 excess is
  −0.3% (10 sessions) and −0.7% (20 sessions), positive in only 3 of 9 folds.
- This agrees with `project_4h_label_vol_forecaster` and the 2026-11-03 account
  post-mortem.

**2. The new labels remove the vol tilt but add no rank information we can measure.**
- Every new arm beats L0 by +0.055 to +0.081 IC, and every one of those CIs is far
  above 0. That gain is the vol penalty going away.
- On top of that, the best ICs are +0.012 to +0.018, while the MDE is 0.016 to 0.030.
- **This null is underpowered.** It means "not measurable here", not "no effect". An IC
  of +0.015 would be worth having, and this design cannot detect it.

**3. The effects shift over time.**
- Every L2 arm is negative in 2022–2023 and positive in 2024–2025. For example,
  L2_20 on L2_20 scores −0.005, −0.021, +0.051, +0.035 and +0.022 for 2022 through 2026.
- This could be a regime effect or a learning curve: fold 0 trains on about 0.77M rows,
  and several early folds stop at iteration 0. These results cannot tell the two apart.

**4. PIV is consistent but too small to trade.**
- IC is positive in 8 of 9 folds.
- It beats its noise twin with q = 0.017, so it passes FDR.
- But its IC is only +0.006, and the IC CI just touches 0 (−0.0003).
- Top-3 net is +0.1% per 10 sessions.

**5. Exploratory, not pre-registered: dropping the calendar features helps the top of
the ranking.** Top-3 excess after regressing vol, beta, ADV and price out of the score:

| arm | 10 sessions | folds+ | 20 sessions | folds+ |
|---|---|---|---|---|
| L2_20 (with calendar) | +0.8% [−0.6, +2.4] | 3/9 | +1.3% [−0.6, +3.4] | 4/9 |
| L2_20__no_cal | +1.9% [+0.6, +3.2] | 6/9 | +3.3% [+1.1, +5.7] | 7/9 |
| L2_20__no_cal_leader | +1.6% [+0.5, +2.8] | 8/9 | +2.4% [+0.8, +4.4] | 8/9 |
| L2_20__no_cal_noearn | +1.2% [+0.2, +2.2] | 7/9 | +2.7% [+0.9, +4.5] | 8/9 |

- Calendar features are constant within a bar, so they cannot rank names. Trees use
  them to memorise period-specific regimes instead. They carry 13–17% of the
  importance (gain) in the L2 arms (L2_20: 15.8%).
- The three no-calendar arms share most of their features, so this is roughly one
  finding, not three.
- It sits among about 70 secondary cells. Treat it as a hypothesis for the next
  pre-registered variant, not as a result.

**6. Gross top-k levels are skew artefacts. Do not read them as edge.**
- L0's top-3 shows +5.5% "net" per 10 sessions, and VOL shows +4.1%.
- Among the three highest-ATR names per bar (median price $8.76), the mean 10-session
  return of +3.5% comes entirely from the 1.3% of picks that more than doubled. Without
  them the mean is −0.2% and the median is −0.8%.
- Some of those doublings may be corporate-action jumps in the partly unadjusted bar
  cache (`project_bar_cache_corporate_actions`).
- Hit rate against the L2 base rate carries the same vol inflation: VOL scores +6.8pp.

**7. One noise control learned a tilt.**
- L2_40__perm scores IC +0.008 to +0.010 on the L2 yardsticks, with CIs above 0.
- It leans toward low vol (fmfe_10 IC −0.031), and low vol reads as positive on L2.
- So that arm's gap to its twin is conservative. Read both the twin and the VOL row,
  never one alone.

**8. The earnings features do not leak much, but they reward variance.**
- They come from realised yfinance report dates, which are not point-in-time.
- A possible acquisition leak (rows with a report 45–90 days ago and no next report
  within 90 days) covers 0.22% of rows and hits the L2_20 top quintile 21.9% of the
  time, against a 20% base. That is too small to matter.
- Rows with `earnings_in_fwd_window = 1` hit the L2_10 top quintile 27% of the time,
  against 19%. But their mean 20-session excess return is +0.55%, against +0.35%: their
  returns are more spread out, not better.
- Method: a one-off check that grouped the Stage 2 matrix by which earnings fields
  are missing, using only bars before 2026-05-15, and compared y_L2 rates
  with the 20-session SPY-excess return.

## Next testable variants (the hypothesis class is not exhausted, only underpowered)

1. **Pre-register v2 now, before any more looking.** Write it into the evaluator
   docstring and keep the test window sealed:
   - features: L2_20 without calendar features;
   - score: residualised on percentile ranks of vol, beta, ADV and price;
   - primary metric: residualised IC and residualised top-3 against a noise twin.

   Open the test once, after it holds at least 12 months of matured labels (about
   2027-06). Validation results cannot confirm v2, because v2 was chosen by looking at
   them.
2. **Power.** Extend the matrix back before 2020. Twice the validation span cuts the
   MDE by about 1.4×, close to the effects in finding 2. This needs a check that bars and
   features for 2016–2019 are point-in-time.
3. **Structural.** Drop calendar features from the next momentum and HTF retrains,
   whatever the label.
