# Feature inventory, and what beta-adjusting the label would actually buy

2026-08-30.

---

## 1. Beta-adjusted alpha vs ATR-adjusted return — measured

First, an algebraic fact worth stating plainly: **`fwd_atr_adj_return` IS forward
MFE in ATR units.** `fwd_max_return / atr_pct` and `(high − ref) / atr` are the
same quantity (`atr_pct = atr / close`). Measured within-bar rank correlation
between them: **ρ = 1.000**. So the 0.25-weight component already *is* the
tradeable target.

Three candidate versions of the 0.40-weight component, within-bar rank correlations:

| version | vs ATR-adj (=MFE) | vs raw return | vs ticker ATR% | vs beta |
|---|---|---|---|---|
| **current** `ret − bench_max` | (≡ raw return) | **+1.000** | **+0.23 … +0.26** | — |
| quick fix `ret − β·bench_max` | 0.84 – 0.87 | 0.90 – 0.95 | −0.09 … −0.16 | — |
| **residual-path** (build the β-residual path, then take its max) | **0.86 – 0.87** | 0.77 | **−0.02 … +0.01** | −0.05 … −0.07 |

**Answers to your two questions:**

*Would a beta-adjusted alpha tell us anything different from ATR-adjusted return?*
**Yes, but only about 25% different** — ρ ≈ 0.86. They are not redundant, but they
are close. They control different things: ATR normalisation removes *total* risk
(systematic + idiosyncratic); beta adjustment removes *market co-movement*
specifically. A high-beta name and a high-idiosyncratic-vol name get treated
differently by the two, which is the entire distinct 14%.

*Should we have both?* **Yes — but not at 0.40/0.25.** At ρ = 0.86 the composite
would be spending 0.65 of its weight on two near-copies of the same measurement
while presenting them as two dimensions. Give the tradeable one the larger weight
and the market-relative one the smaller, rather than the reverse.

Use the **residual-path** version, not the quick fix. `max(stock) − β·max(SPY)` is
not a decomposition — a max operator does not distribute over a linear model, so
that form is only approximately beta-neutral (it still leaves −0.09…−0.16 of
volatility tilt). Building the residual return path first and taking its max is
the version that means what it says, and it lands at −0.02.

## 2. Yes, this requires retraining — and you are right about what a label is for

A label is a *definition of what good looks like*, computed with hindsight. It
does not have to be predictable to be correct, and "there is always money to be
made some way" is the right framing for designing one.

But there are two independent quality bars, and they have been conflated:

1. **Does the label define the right thing?** Currently 40% of its weight defines
   "most volatile name", which is not what you meant by "best candidate". This is
   a defect and it is cheap to fix.
2. **Can the features predict it?** Currently ρ ≈ 0.07–0.20. This is the binding
   constraint and it is not a label problem.

Fixing (1) is necessary for the label to mean what you intend. It will not fix (2).

**Expect the measured ρ to FALL after the fix, and do not read that as failure.**
The model score's own within-bar correlation with ticker ATR% is **+0.17**
(momentum) — almost exactly its correlation with its current target (+0.19). A
large share of what the model has learned is "rank the volatile names", and
volatility is strongly autocorrelated, so it is the *easy* part of the current
target to predict. Removing it removes predictable-but-useless signal. A drop
from ρ 0.19 to, say, 0.10 against a target that now measures the right thing is
progress, not regression — the correct comparison is against tradeable outcome,
not against the old target.

## 3. Feature inventory — what is in the matrix, and what is not

`training_matrix_4h.parquet` — built **2026-06-14**, data ends **2026-05-14**,
2,317,972 rows x 115 columns, 1,081 tickers, 2020-09-08 → 2026-05-14.
106 features + 9 label columns.

| group | n | examples |
|---|---|---|
| price / trend / structure | ~40 | `ema_dist_*`, `ema_slope_*`, `adx_14`, `rsi_14`, `macd_hist_norm`, `ret_*`, `compression_*`, `breakout_*`, `base_*`, `dist_to_52w_*`, `range_pos_20` |
| volatility | 6 | `atr_pct_14`, `atr_expand_14_60`, `realized_vol_5/20`, `vol_regime_5_60`, `vol_of_vol_20` |
| volume / liquidity | 7 | `rvol_20`, `dollar_vol_pctile_252`, `dollar_vol_surge_20`, `volume_z_20`, `volume_spike_20` |
| relative strength | 8 | `rs_spy_*`, `rs_qqq_*`, `rs_sector_*`, `beta_spy_60`, `corr_spy_60` |
| **regime** | **4** | `regime_spy_trend`, `regime_spy_ret_20`, `regime_vix_z`, `regime_vix_high` |
| multi-timeframe | 13 | `daily_*`, `weekly_*` |
| calendar / seasonality | 16 | `dow`, `month`, sin/cos encodings, `is_month_start/end` |
| earnings | 5 | `days_to_earnings`, `is_pre_earnings_3d`, `earnings_in_fwd_window` |
| static categorical | 5 | `sector_id`, `market_cap_bucket`, `asset_type`, `is_etf` |
| cross-sectional ranks | 6 | `xsec_ret_5_rank`, `xsec_atr_pct_rank`, `xsec_near_high_rank` |

**Absent entirely: options/dealer, themes, news/catalysts, social attention,
forward guidance, rates/credit, macro events.** It is a pure price / volume /
calendar matrix. Regime is 4 columns, all SPY/VIX.

### The alt data exists — it is just in the OTHER matrix

`meta_ranker_matrix.parquet` (rebuilt **2026-08-28**, 704,638 rows x 79 cols,
2025-07-24 → 2026-08-28) already carries almost everything the momentum matrix
lacks:

* **themes (12)** — `theme_heat_score`, `theme_breadth`, `theme_acceleration`, `theme_strength`, `membership_score`, `related_theme_heat`, `theme_age_days`, `theme_newness_score`, `within_theme_mom_rank`, `theme_crowding_frac`
* **news / catalysts (16)** — `news_catalyst_score` + mean/std/count, `news_unique_sources`, `news_high_alpha_count`, `news_breaking_count`, `news_bull/bear_alignment`, and five state probabilities (`news_p_bull_steady`, `news_p_v_bounce`, `news_p_crash_stayed`, …)
* **rates (8)** — `treasury_3m/2y/10y/30y`, `treasury_spread_2s10s`, `treasury_spread_3m10y`, `treasury_inverted`, `treasury_10y_change_5d`
* **macro events (8)** — `days_to_macro_event`, high-impact variants, forward counts
* **forward guidance (3)** — `fg_guidance_score`, `fg_guidance_count_90d`, `fg_days_since_guidance`
* **cross-signal (3)** — `mom_xs_rank`, `htf_xs_rank`, `signal_agreement`

**The two matrices are almost exactly complementary, and nothing has both.**
Momentum has 106 technical features and zero alt data; Meta has ~47 alt features
and consumes `mom_score`/`htf_score` as pre-computed inputs instead of raw
technicals.

### History depth decides what can actually be added

The training window is 2020-09 → 2026-05. A feature is only usable across it if
it *has* that history:

| source | coverage | usable in a 5.7-year matrix? |
|---|---|---|
| **market regime panel** (`daily_regime.parquet`, **36 cols**) | 2020-07 → 2026-08 | **YES — full window** |
| **sector state** (11 cols) | 2020-07 → 2026-08 | **YES — full window** |
| news library (544,423 rows, 31 cols) | 2023-01 → 2026-08 | partial (~60% of window) |
| meta matrix alt features | 2025-07 → 2026-08 | ~13 months only |
| dealer positioning | 2026-07 → 2026-08 | **NO — 2 months** |

The single biggest free win is the **regime panel**: 36 point-in-time columns
with `available_at` and per-component staleness — `risk_appetite_z` (four
sub-components), `liquidity_stress_z` (four), `credit_risk_z`, `breadth_z`,
`sector_dispersion_z`, `spy_rv20_z`, `spy_trend_state` — covering the entire
training window, against the 4 crude SPY/VIX columns the matrix uses today.
(Note `risk_appetite_hyg_iei_z` is the corrected credit ratio; `credit_risk_hyg_lqd_z`
is the duration-contaminated one and should not be used.)

## 4. Recommended order

1. **Fix `fwd_max_alpha`** to the residual-path form, and re-weight so the
   tradeable component leads. Requires a retrain — by design.
2. **Add the regime panel** (36 cols, full history) to the momentum matrix.
   Zero blockers, no coverage compromise.
3. **Rebuild the matrix** — it is 3.5 months stale (data ends 2026-05-14).
4. **News features** for the 2023+ portion, with an explicit missing-ness
   indicator rather than an imputed zero for 2020-2022.
5. Theme/dealer features only into a *short-horizon* model, never the 5-year one.

Horizon alignment (Stage 6) stays after these: a relabelled target on unchanged
features inherits the same ceiling.
