# Parabolic filter on SHARES — it works, once the label is right

**Author:** Claude, 2026-07-28. Scripts: `scripts/apply_parabolic_filter_to_shares.py`,
`scripts/build_forward_excursion_labels.py`. Shares pay no option spread, so the filter's value
shows up undiluted. Filter trains on the first 60% of each module's history and only *scores* the
remaining 40% — every number here is out-of-sample, with week-block bootstrap CIs.

---

## 1. The first attempt failed because the LABEL was wrong

The original label was `MFE >= 4 ATR`. ATR-normalizing destroys the thing "parabolic" means:

| volatility bucket | ATR as % of price | what "4 ATR" actually is | hits 4 ATR | hits **+25%** |
|---|---:|---:|---:|---:|
| low vol | 0.9% | **a 3.4% move** | 33.5% | **0.0%** |
| q3 | 3.7% | 14.6% | 39.9% | 18.9% |
| HIGH vol | 6.3% | **25.3%** | 28.5% | 30.1% |

corr(ATR%, hits 4 ATR) = **−0.05**; corr(ATR%, hits +25%) = **+0.35**.

A 4-ATR move on a quiet stock is a 3.4% drift. The filter therefore learned to find **low-volatility**
names — ATR fell monotonically 8.09% → 2.78% across its own score buckets — which is the exact
opposite of a gamma-squeeze candidate. Result: higher win rate (65% → 83%) but *smaller* wins
(avg win 15.8% → 6.5%), and **no significant improvement in mean return at any selectivity**.

**Fix: label on percentage move.** `MFE >= +25% within 20 bars`.

---

## 2. With the %-based label, the filter works

OOS AUC improves: momentum **0.626 → 0.671**, HTF **0.629 → 0.819**.
And ATR% now *rises* with the score (momentum 1.74% → 6.03%; HTF 1.04% → 5.65%) — it is finally
selecting volatile, squeeze-capable names.

### momentum_expansion (OOS n=1,534, all trades mean +2.95%)

| filter bucket | n | big-move rate | ATR% | **mean return** | win rate | Sharpe |
|---|---:|---:|---:|---:|---:|---:|
| Q1 | 307 | 5% | 1.74% | +0.93% | 76.9% | 0.18 |
| Q3 | 306 | 32% | 6.95% | +1.98% | 71.2% | 0.11 |
| **Q5 HIGH** | 307 | **42%** | 6.03% | **+6.58%** | **82.4%** | **0.47** |

| selectivity | mean return | lift vs taking everything | 95% CI | verdict |
|---|---:|---:|---|---|
| top 30% | +5.53% | **+2.58pp** | [+0.81, +4.57] | **significant** |
| top 20% | +6.57% | **+3.61pp** | [+0.79, +5.84] | **significant** |
| top 10% | +6.49% | +3.53pp | [−0.05, +6.75] | n.s. (n=153) |

### multi_ticker_swing_htf (OOS n=9,925, all trades mean +1.45%)

| filter bucket | n | big-move rate | ATR% | **mean return** | win rate | Sharpe |
|---|---:|---:|---:|---:|---:|---:|
| Q1 | 1,985 | 0% | 1.04% | +0.28% | 39.4% | 0.09 |
| Q4 | 1,985 | 23% | 4.67% | +2.46% | 42.2% | 0.15 |
| **Q5 HIGH** | 1,985 | **28%** | 5.65% | **+4.64%** | 48.4% | **0.27** |

| selectivity | mean return | lift | 95% CI | verdict |
|---|---:|---:|---|---|
| top 30% | +4.37% | **+2.92pp** | [+1.18, +4.79] | **significant** |

Mean return per trade roughly **doubles to triples** in both modules, monotonically in the score.

---

## 3. But check the interpretable baseline — and it changes the recommendation per module

Per the pre-registration, a model only earns its place if it beats a simple rule. Testing
"just take the highest-ATR% names, no model," on the same OOS window:

| module | simple ATR rule | the model | verdict |
|---|---|---|---|
| momentum_expansion (top 30%) | +0.76pp, CI [−2.52, +3.47] **n.s.** | +2.58pp **sig** | **model wins** |
| momentum_expansion (top 20%) | −0.26pp, CI [−4.80, +4.22] **n.s.** | +3.61pp **sig** | **model wins** |
| multi_ticker_swing_htf (top 30%) | +2.43pp, CI [+0.76, +4.34] **sig** | +2.92pp **sig** | **tie — use the rule** |
| multi_ticker_swing_htf (top 20%) | +2.75pp, CI [+0.63, +5.37] **sig** | — | **tie — use the rule** |

**Recommendation, per module:**
- **momentum_expansion → use the model.** A volatility sort alone is worthless here (n.s. at both
  cut-offs); the model finds something beyond volatility worth +2.6 to +3.6pp per trade.
- **multi_ticker_swing_htf → use a plain ATR% filter.** The model's advantage over a one-line rule is
  within noise. Do not ship an XGBoost dependency for +0.5pp that the CIs cannot separate.

That HTF's high AUC (0.819) is mostly volatility being predictable is exactly why the baseline check
matters: a strong-looking AUC translated to no incremental economic value over a trivial rule.

---

## 4. Caveats

- Entries and exits are unchanged — this is a *selection* result. Taking the top 20–30% means
  taking fewer trades, so total P&L depends on whether the freed capital is redeployed.
- The +25% / 20-bar threshold was chosen to match the stated intent ("parabolic"), not tuned. It
  should be swept before deployment, and the sweep costs test-set budget.
- momentum's top-10% cell is not significant (n=153). Top 20–30% is the usable operating range.
- HTF's underlying expectancy remains modest; the filter improves it but does not transform it.
- This says nothing about options. The corrected option cost model
  (see `08_...md`, cents-based not percentage-based) still needs to be applied to Phase 3 before any
  option conclusion is re-quoted.
