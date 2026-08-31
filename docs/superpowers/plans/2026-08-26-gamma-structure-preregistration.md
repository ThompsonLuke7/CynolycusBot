# Pre-Registration: Does Gamma Structure Add Anything Over Volatility Controls?

**Created:** 2026-08-26
**Status:** REGISTERED — no analysis has been run against these targets
**Implements:** Phase 2 of `2026-08-25-option-gamma-structure-upgrade.md`
**Readout:** Study A is runnable **now**; Study B not before **2027-03-01** (see §6)

> **Amendment 2026-08-26 (same day, before any analysis was run).** The original
> version gated *everything* on 150 nightly snapshot dates. That was wrong, and
> the objection that caught it was correct: it confused the unit of observation.
> A cross-sectional forecast of next-session realized volatility is observed once
> per symbol per **date**, so it really is date-limited. But whether price rejects
> or penetrates a gamma level is observed once per **touch**, and the intraday
> archive already holds 222 call-wall and 251 put-wall touch events on SPY alone
> across 41 sessions. That question is testable today. The design is split into
> Study A (level interaction, runnable now) and Study B (cross-sectional RV panel,
> still date-limited).

This document is written **before** the study, and its point is to be binding.
The critique that prompted this work made exactly one methodological claim that
is not arguable: searching 24 thresholds turns a 2.3% result into a 45.4% chance
of finding "significance." Everything below exists so that cannot happen here.

Any deviation from this document must be recorded as an amendment **with its
date**, above the results, not silently folded into the method.

---

## 0. Two studies, two different units of observation

| | **Study A — level interaction** | **Study B — dispersion panel** |
|---|---|---|
| Question | At a gamma level, does price reject or penetrate more often than chance? | Does gamma structure forecast next-session realized volatility beyond vol controls? |
| Unit | one **touch event** | one symbol-**date** |
| Data | intraday ETF archive, 5 symbols, 41 sessions | nightly equity cross-section |
| Sample today | ~473 SPY touch events; est. 2,000-3,000 across 5 symbols | 34 dates |
| Independence limit | 41 session clusters | 34 date clusters |
| Status | **runnable now** | needs 150 dates (~2027-03) |

Study A is the one that corresponds to how these levels are actually traded --
as places where price does something -- and it is also the one the day-trading
callouts are implicitly claiming. It should be run first.

Study B remains gated. Nothing about Study A's sample helps it: 41 sessions of
touches say nothing about a 707-symbol cross-section.

---

## 1. Hypothesis

**H1 (primary).** Option gamma structure carries information about *future
realized volatility* beyond what trailing realized volatility, implied
volatility, and ATR already carry.

**H0.** It does not. Conditional on the controls, gamma features add nothing
out of sample.

**Explicitly not tested:** whether gamma structure predicts direction. The
literature, the critique, and the internal evidence agree it does not, and no
directional target appears anywhere in this design.

---

### Study A hypothesis

**H1-A.** Conditional on price touching an estimated gamma level, the
probability of rejection differs from the unconditional base rate, and the
difference is larger where `structure_confidence` and `call_wall_stability` are
high.

**H0-A.** Touch outcomes at gamma levels are indistinguishable from touch
outcomes at matched control levels.

**The control matters more than the treatment here.** A wall is near a round
number, near recent highs, and near where volume already traded. So the
comparison is not "rejection rate at walls vs 50%" -- it is **rejection rate at
walls vs rejection rate at matched non-gamma levels** at the same distance from
spot, same time of day, same session. Without that control, this study would
rediscover support and resistance and credit gamma for it.

### Study A design, fixed now

* **Event.** Spot comes within 15bps of the estimated call wall or put wall.
  Consecutive in-band snapshots collapse into one event; a new event requires
  leaving the band for at least 15 minutes.
* **Outcome.** Within 30 minutes of the touch: *rejection* if price moves at
  least 25bps back from the level without closing 15bps beyond it; *penetration*
  if it closes 15bps beyond; *neither* otherwise (reported, not dropped).
* **Controls.** For each touch, a matched pseudo-level at the same distance from
  spot on the opposite side, and a fixed-offset level at the same absolute
  distance. Same clock time, same session.
* **Primary statistic.** Rejection rate at gamma levels minus rejection rate at
  matched controls, with **session-clustered** standard errors (n = 41 clusters,
  not 473 events).
* **Pre-declared conditioning variables — these three and no others:**
  `structure_confidence`, `call_wall_stability`, `zero_dte_gamma_share`.
* **Decision.** A gap of >= 8 percentage points with a session-clustered
  interval excluding zero graduates the level-touch feature into the intraday
  structure engine as a level-strength weight. Below that, it is recorded and
  not wired.

**Power, stated in advance.** With 41 session clusters and roughly 470 SPY
events, an 8pp gap is near the detection floor. Pooling the five ETFs raises the
event count but not the cluster count much, since the sessions are shared. If
the honest answer comes back "underpowered", that is the finding, and the
response is more sessions rather than a smaller threshold.

---

## 2. Study B primary target — one, fixed

```
y = realized volatility of the underlying over the next full session,
    close-to-close, annualized at 252 trading days
```

One target. Chosen now. It is not swapped for a secondary if it disappoints.

### Secondary targets (reported, never selected on)

Reported in the readout for completeness. **A secondary result cannot be
promoted to the headline** — if the primary is null, the study is null.

1. absolute close-to-close return over the next session
2. high-low range over the next session, in ATR units
3. P(|move| > 1 ATR) over the next session
4. penetration vs rejection at the nearest gamma node, conditional on touch

---

## 3. Models

```
Model A (control):
    y ~ trailing_RV_20d + trailing_RV_5d + atm_iv + atr_14d + market_regime

Model B (treatment):
    Model A + gamma feature groups
```

The controls exist because of the confound the critique names: recent
volatility drives current option positioning *and* predicts future volatility,
because volatility is autocorrelated. A gamma feature that only reproduces
trailing RV must show up as no gain over Model A.

**Estimator.** Ridge regression first. A gradient-boosted model is permitted as
a secondary specification, reported alongside, never instead. An interpretable
baseline that shows nothing is a more trustworthy null than a flexible model
that finds something.

**Metric.** Out-of-sample R² for the continuous targets, AUC for the binary
ones. The reported result is **Δ (B − A)**, never B's level.

---

## 4. Feature groups — declared now, ablated as blocks

Ablation is by group. No feature-level search, no "best subset."

| Group | Features |
|---|---|
| **G1 unsigned topology** | `gamma_density_1pct/2_5pct/5pct`, `gamma_concentration`, `gamma_entropy`, `nearest_node_distance_pct`, `void_above_width_pct`, `void_below_width_pct` |
| **G2 estimated signed** | `estimated_net_gex`, `estimated_dealer_imbalance`, `pct_to_call_wall`, `pct_to_put_wall`, `pct_to_gamma_flip` |
| **G3 stability** | `call_wall_stability`, `put_wall_stability`, `gamma_flip_stability`, `node_rank_stability`, `estimated_net_gex_sensitivity` |
| **G4 velocity** | `wall_change_1d`, `wall_change_3d`, `gamma_flip_velocity`, `level_stability_days`, `distance_to_call_wall_atr` |
| **G5 term structure** | `zero_dte_gamma_share`, `short_gamma_share`, `weekly_gamma_share`, `gamma_term_slope` |

**Prediction, recorded in advance:** if anything survives, G1 and G3 are the
most likely, because they do not depend on the sign assumption. G2 is the group
the critique attacks most directly and the one most likely to be null.

### The one permitted interaction

The 2026-07 confluence study certified zero cross-signal interactions and
consumed the test set to roughly 2027. Interaction *mining* is therefore closed.

Exactly one interaction is registered here, chosen on mechanism rather than on
search: **`estimated_dealer_imbalance × structure_confidence`**. If the sign
assumption carries information at all, it should carry more where the structure
is well measured. No other interaction may be tested under this registration.

---

## 5. Sample, splits, and the untouched test set

**Universe.** Two panels, analyzed separately, never pooled:

* **Panel N (nightly, cross-sectional):** the captured equity universe, one
  observation per symbol per snapshot date, scope `through_month`.
* **Panel I (intraday, time series):** SPY, QQQ, IWM, GLD, SLV at 1-minute
  cadence, predicting realized volatility over the next 30 minutes.

**Splits.** Walk-forward by calendar, expanding window. Selection happens on
validation folds only. **The final 20% of the calendar is the test set and is
read exactly once**, at readout.

**Clustering.** Panel N observations on one date are not independent — one
market moves them all. Standard errors cluster by date. Effective n is the
number of *dates*, not the number of rows, and the readout must state both.

---

## 6. Minimum sample and the readout date

As of 2026-08-25:

| Panel | Coverage | Effective n |
|---|---|---|
| N (nightly) | 34 dates, 2026-07-02 → 2026-08-25 | ~34 |
| I (intraday) | 41 sessions, 2026-06-12 → 2026-08-25 | ~41 sessions |

Both are far short. Registered minimums, to be met before the test set is
touched:

* **Panel N: 150 distinct snapshot dates.**
* **Panel I: 120 distinct sessions per symbol.**

At the current accumulation rate (~1 nightly capture and ~1 session per trading
day) Panel N reaches 150 dates around **2027-03**. That is the earliest readout.

**No peeking rule.** Validation-fold analysis may run earlier to shake out
pipeline bugs, on the explicit condition that no threshold, feature group, or
model is selected on it and no result from it is reported as evidence. The test
set stays sealed until both minimums are met.

---

## 7. Decision rule, fixed in advance

| Outcome | Decision |
|---|---|
| Δ R² ≥ +0.01 on the primary, out of sample, stable across folds | Gamma features graduate to a sizing/dispersion input. Still not a direction signal. |
| 0 < Δ R² < +0.01 | Recorded as a real but immaterial effect. No wiring change. Do not re-specify to chase it. |
| Δ R² ≤ 0 | **H0 accepted and published as a null.** The features stay in the artifacts as description; nothing consumes them for sizing. |

The middle row is the one that matters. Given the Cboe-sponsored SPX study found
roughly a 0.2 percentage-point annualized volatility effect using *reconstructed
actual* market-maker positions, a small effect is the expected outcome here and
a large one should trigger a bug hunt before a celebration.

**A null is a publishable result.** It closes a question that has been open in
this repository since the dealer module was written, and it is written to
`research/` with the same weight as a positive.

---

## 8. Known ways this could produce a false positive

Listed now so that finding one later is a check, not a rescue.

1. **Volatility autocorrelation leaking through an imperfect control.** Mitigated
   by two trailing-RV horizons plus ATM IV, not by one.
2. **Cross-sectional correlation inflating significance.** Mitigated by
   date-clustered errors and by reporting effective n as date count.
3. **Survivorship in the captured universe.** The nightly capture screens on
   liquidity, so the panel is conditioned on names that stayed liquid. Reported
   as a limitation; not corrected.
4. **Overlapping windows** in Panel I. 30-minute forward windows sampled each
   minute overlap heavily; block bootstrap by session for inference.
5. **Regime confounding.** The sample begins 2026-06 and covers one volatility
   regime. Any positive result is a single-regime result and must say so.

---

## 9. What is already built

Phase 0 and Phase 1 of the parent plan are implemented, so the study will run
against features that exist rather than features that need inventing:

- `strategies/dealer_positioning/topology.py` — G1, G5
- `strategies/dealer_positioning/confidence.py` — the interaction term's second half
- `strategies/dealer_positioning/stability.py` — G3
- `strategies/dealer_positioning/level_dynamics_feed.py` — G4
- `strategies/dealer_positioning/scripts/build_intraday_level_dynamics.py` — Panel I

What remains for Phase 2 is the target construction, the walk-forward harness,
and the readout — none of which should be built until the sample approaches the
registered minimum, because building it earlier creates the temptation this
document exists to remove.
