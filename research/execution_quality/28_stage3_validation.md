# Stage 3 validation: rule v2 on the point-in-time matrix

Evaluated 2026-11-19 against the rule fixed on 2026-11-08 in
`27_stage3_preregistration.md`. This is **validation only**: the final test window
(2026-05-15 on) was not read.

- **Run details:**
  - results `stage3_results.tar` (hash 99b0450a1cc8), 6/6 arms;
  - git 87625f4086, 1,000 bootstrap draws;
  - runtime 38 min, peak 5.3 GB.
- **Outputs:**
  - report: `strategies/momentum_expansion/ablation/results/stage3_eval/validation_report.md`;
  - frozen selection: `selection.json`, hash 621d0acaa1cd.
- **Training:**
  - 15 walk-forward folds, 2019-03 to 2026-09, about 11M out-of-fold rows per arm.
  - Real arms kept a median of 66–119 trees; their noise twins kept 1–3.

## Decision under the pre-registered rule

| Slot (yardstick) | Pick | Neutral IC [95% CI] | MDE | vs noise twin | BH q | Folds + | S_new |
|---|---|---|---|---|---|---|---|
| momentum (L2_10) | **L2_20__pit** | +0.0093 [+0.0023, +0.0166] | 0.0103 | +0.0117 [+0.0033, +0.0203] | 0.028 | 11/15 | +0.0042 |
| htf (L2_20) | none | L2_20__pit: +0.0095 [−0.0023, +0.0197] | 0.0157 | +0.0107 [−0.0020, +0.0223] | 0.112 | 11/15 | +0.0027 |

**Momentum slot.** L2_20__pit (the 20-session vol-neutral label, without calendar features)
meets all four conditions on the 10-session yardstick, so it is the pick. The selection is
frozen.

**HTF slot.** No pick. The point estimate equals the momentum slot's, but the 20-session
yardstick has about 1.5× the MDE. The CI crosses 0. That is **not measurable here**; it is not
"no effect".

**Final test.** It stays sealed. The pre-registration requires at least 12 months of matured
20-session test labels, and the current matrix ends 2026-09-24 (about 100 days of test). The
earliest opening is about 2027-06, on a rebuilt matrix, opened once and only with the user's
agreement. Until then, "picked" does not mean "confirmed", and nothing is deployed.

## What the pick does and does not show

1. **The ranking is weakly better than noise across the whole cross-section.**
   - IC is a whole-universe rank measure over about 3,000 names a bar.
   - Precision beats the 20% base by about 4 points (+0.039 to +0.047).
2. **It does not make money at the top yet.**

   | Top-3, 10 sessions | Gross | Net of the assumed cost |
   |---|---|---|
   | Neutral score | −0.04% | −0.72% |
   | Raw L2_20 score | +0.23% [−0.36, +0.89] | −0.59% |

   - At k = 1 to 10 the gross is never above +0.6%, and the net is negative throughout.
3. **It is regime-dependent.** Neutral IC on L2_10 by calendar year:

   | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 (to May) |
   |---|---|---|---|---|---|---|---|
   | −0.002 | +0.026 | −0.007 | −0.014 | −0.002 | +0.028 | +0.033 | +0.011 |

   - Only 4 of 8 years are positive; the mean comes from 2020 and 2024–25.
   - Stage 2 showed the same split.
   - The 11/15 count uses 6-month folds, which hides the sign changes.
4. **The fresh evidence is thin.**
   - S_new (bars before 2022-04, never read by Stage 2) is +0.0042, but it is positive only
     because of 2020; 2019 and 2021 are negative.
   - The 2022–26 part of validation overlaps the period that suggested rule v2.
5. **The effect sits at the noise scale.**
   - The pick's IC is below its own MDE.
   - On L2_10, the six noise-trained cells span −0.0024 to +0.0064, and two of them have CIs
     above 0.
   - Each arm has a single noise twin (one seed), and the bar bootstrap does not include the
     variance between twin draws.
   - The pick sits about 2.4 SD above the mean of the noise cells: real, but small.
6. **L0 is still a volatility ranker.**
   - IC is −0.046, against −0.046 for the pure volatility row.
   - `daily_atr_pct` carries 32% of its gain.
   - It is positive in 0/15 folds.
   - Neutralised, it is still −0.014.

## Live-parity defects found the same night (they block any forward use)

The decision-state archive's first parity reports (2026-11-18 and 11-19) failed on momentum
and dealer. Investigated 2026-11-19:

- **Momentum's daily features are one session staler live than in training.**
  - Cause: `feature_matrix_4h._map_to_4h` lags every daily series with `.shift(1)`.
    Historically, a 4H bar on date D reads D's row, which carries D−1's data. Live,
    intraday, there is no row for D yet, so the backward as-of reads D−1's row, which
    carries D−2's data.
  - Evidence on AAPL: live `daily_ret_1` was 11-13's return on 11-17 and 11-16's return on
    11-18; research had 11-16's and 11-17's.
  - Effect: across all names, `daily_ret_1` has Spearman +0.32 live vs research, and
    `daily_gap_pct` +0.14.
  - It affects every daily_*/weekly_* feature, on every bar decided before the day's daily
    bar exists. Regime features (for example `regime_vix_z`) mismatch the same way.
  - Training features are unaffected; only live inputs are wrong.
- **Momentum cross-sectional ranks and volume.**
  - The `xsec_*` ranks inherit the daily and volume differences (Spearman 0.90–1.00).
  - One afternoon bar's volume differs (the bar was revised after the decision).
- **Dealer.** The dated rankings file is rewritten after the 15:45 decision: 11-18's now holds
  a 17:45 capture, and 11-17's was rebuilt at 23:31. The archive is the only record of what
  dealer actually ranked on. This is a reference-mutability problem, not a builder gap.
- **SPY is not archived.** The hook went into the CLI runner (`live_runner._make_15m_handler`),
  but the server runs SPY through `UI/live_dashboard.py`'s own 15m handlers.

A Stage 3 model scored on live momentum inputs today would see stale daily features. Forward
(shadow) evidence for the pick counts only after the daily-lag fix.

## Next testable steps (roadmap order)

1. **Fix the live daily lag.** At the live edge, use the last completed daily row's unshifted
   value. Then confirm momentum parity goes green. This changes live momentum inputs to match
   training, so it needs a go-ahead.
2. **Move the SPY archive hook** into the dashboard's 15m handlers.
3. **Net-of-cost selectivity (checklist 1.1)** on the pick's validation OOF: depth, holding
   period and weighting. The question is whether the IC can be turned into net return before
   the test opens. Choosing the construction on validation keeps the test clean.
4. **Prospective shadow.** Train a final L2_20__pit booster on pre-test data (Colab) and score
   the archived live momentum vectors nightly, after step 1. That builds forward
   out-of-sample evidence alongside the sealed test.
5. **HTF slot.** It is underpowered, not refuted. The test window, once matured, roughly
   doubles the 20-session evidence.

## Parity fixes applied 2026-11-19 (same night)

1. **Daily features.** `feature_matrix_4h._with_live_edge_daily_rows` adds a placeholder daily
   row for today, so the shift(1) lag picks up the last completed session.
   - Real data: history is unchanged on about 3,000 bars per ticker for AAPL, NVDA, PLTR and
     SOFI, and the live edge matches exactly for all 12 lagged features.
   - The two gap features need the official open. The momentum runner now fetches today's opens
     from Alpaca SIP daily bars (`data/session_open.py`, `SessionOpenDailyLoader`) and adds them
     as a partial row. They matched the daily cache exactly on 20/20 names. Without an open, the
     gap keeps its old meaning.
2. **Cross-sectional ranks.** The live panel is built over the research universe (shared
   eligible, 2,888 names) instead of the snapshot (2,679). The snapshot still gates picks.
3. **Metadata features** (sector, cap bucket, asset type, ETF flag), which the live panel had
   left at 0. Live now uses the research build's encodings (`research_ticker_meta`): 2,806/2,806
   identical.
4. **Context bars.** The intraday refresher now runs
   `refresh_shared_context_bars.py --context-only` before the Meta matrix step. This needs a
   combined-server restart.

**Rehearsal of the 11-18 14:31 ET decision with fixes 1–2** (context cut at the bar, standing in
for fix 4): 103 of 106 model features matched research on more than 99.9% of rows. The other
three were the metadata features that fix 3 then resolved.

**Parity check.**
- Momentum rows decided before the fix are excluded (`FIXED_FROM`).
- Dealer is NOT_CHECKED with the reason (its reference is rewritten after the decision).
- **SPY:** the archive hook now sits in `UI/live_dashboard.py`'s live 15m handler, which needs
  the same restart.
- The first real momentum check is tonight, on today's 14:31 and 16:31 runs.

**Monetisation:** see `29_stage3_selectivity.md`. Not monetisable long-only at the assumed costs.
