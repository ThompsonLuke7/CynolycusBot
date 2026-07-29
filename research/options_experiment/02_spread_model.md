# Spread Model — Results and Scope Verdict

**Author:** Claude, 2026-07-26. Computed from `data/spread_estimator_validation.csv`
(401 real contracts) via `scripts/validate_spread_estimators.py`.

**Why this exists:** Gate G1 showed mid-pricing is ~22% optimistic round-trip, and the original
spread model (a 4–5 variable regression, R² = 0.10) could not tell a wide-market contract from a
tight one. That specifically threatened the multi-leg half of the experiment, since leg count
multiplies exactly the cost we could not measure per contract.

**Ground truth:** the realized half-spread implied by 470 real executed fills.
Median 13.0%, IQR (7.1%, 23.8%). *Caveat, load-bearing:* these residuals contain our own pricing
error as well as spread, so they are an **upper bound** on true spread, not a clean measurement.
All correlations below inherit that noise.

---

## Results

| Estimator | Coverage | Spearman | R² | median est. | median actual |
|---|---:|---:|---:|---:|---:|
| **Roll (1984)** | 56% | **+0.523** | 0.118 | 5.1% | 10.8% |
| Corwin-Schultz (2012) | 99% | **−0.326** | 0.025 | 3.0% | 13.0% |
| Price clustering | 70% | +0.201 | 0.021 | 9.2% | 10.6% |
| Regression (baseline) | 100% | +0.340 | 0.122 | 15.3% | 13.0% |
| Combined **as originally shipped** | 100% | +0.044 | 0.008 | 3.5% | 13.0% |
| **Combined, after fix** | **100%** | **+0.516** | 0.111 | — | 13.0% |

### Finding 1 — Corwin-Schultz is wrong-signed on option data

Spearman **−0.326, p = 2.8e-11**. Not noise: it ranks contracts *backwards*.

The mechanism is a broken assumption, not a coding error. Corwin-Schultz infers spread from
consecutive high/low ranges, assuming the intraday range is driven mainly by bid/ask bounce. For
options that collapses — the range is dominated by real movement in the underlying, and the
widest-ranging contracts tend to be the actively traded ones with the *tightest* spreads. Hence the
inversion. This is a genuine, reportable result about applying an equity-microstructure estimator to
options.

### Finding 2 — the shipped combined estimator was worse than every component

The fallback ladder ran Roll → **Corwin-Schultz** → clustering → regression, so Corwin-Schultz served
155 of 401 contracts and dragged the blend to Spearman 0.044 / R² 0.008 — worse than the regression
it was built to improve on, and worse than doing nothing.

**Fixed** by removing Corwin-Schultz from the ladder entirely (`research/options_lab/spread_estimators.py`;
regression test added asserting it is never selected, even as sole candidate). Result: **Spearman 0.044 → 0.516
at 100% coverage.** Method mix now: Roll 225, regression 121, clustering 55.

### Finding 3 — R² is the wrong yardstick here; ranking is what we need

On R² nothing beat the 0.10 bar (regression 0.122, Roll 0.118, fixed blend 0.111 — indistinguishable).
R² is dominated by level error and outliers in a heavy-tailed, noisy target.

But we do not need to predict the *level* of a contract's spread. We need to know **which contracts are
wider than which**, so that a 2-leg structure on a thin name is penalized more than a 1-leg on a liquid
one. On that metric the fixed estimator is decisively better: Spearman **0.516 vs 0.340**, and it separates
terciles cleanly and monotonically:

| Tercile by estimated spread | n | median *realized* half-spread |
|---|---:|---:|
| tight | 134 | **7.7%** |
| mid | 133 | 12.5% |
| wide | 134 | **22.0%** |

A 2.9× separation between the tight and wide terciles, on out-of-model realized data. That is real,
usable discrimination — the regression baseline could not do this.

---

## Verdict on scope

**Multi-leg comparisons are conditionally unblocked**, with three binding conditions:

1. **Relative only.** Rank structures against each other; never quote an absolute P&L. G1 already
   forbids absolute claims and nothing here changes that.
2. **Pessimistic bound required**, exactly as pre-registered. Roll *underestimates the level*
   (median 5.1% vs 10.8% realized), so a Roll-derived cost is optimistic and must be scaled to the
   realized-level anchor before use, with the pessimistic case reported alongside.
3. **Estimator provenance must be reported.** Every Phase 3 result must state the method mix behind
   its cost estimates. Conclusions resting mainly on regression-served contracts (the 30% where Roll
   and clustering both fail) are weaker than those resting on Roll-served ones, and must be labeled
   as such rather than pooled silently.

**Still forbidden:** any conclusion whose edge is smaller than the optimistic/pessimistic bracket;
any mid-priced number; any claim that a specific structure earns a specific dollar amount.

**Honest summary.** The strict pre-registered bar (beat R² = 0.10) was *not* cleared on its own terms.
What was achieved is better contract *ranking* — which is what the multi-leg question actually turns on —
plus the removal of a wrong-signed estimator that was silently corrupting every cost estimate. I am
treating that as sufficient to proceed to multi-leg with the conditions above, and flagging that this
is a judgment call that relaxes the letter of the pre-registration in favor of its intent. A reviewer
who disagrees should restrict Phase 3 to single-leg structures, where the case is unambiguous.
