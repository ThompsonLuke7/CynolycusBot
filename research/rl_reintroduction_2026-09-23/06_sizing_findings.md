# Stage 6 — sizing: the reduction, a null, and one usable positive

2026-09-23. Script: `06_sizing.py`. Output: `data/stage6_sizing.json`.
**Validation only; the test block is never read.**

## 0. The reduction — why the obvious sizing experiment does not exist

Under full counterfactual feedback (plan C4) and a reward **linear in size**, expected value
is `Σ m(s_i)·R_i`, maximised pointwise by `m = max` wherever `E[R|s] > 0` and `m = 0`
elsewhere. That is **Stage 2's selection problem restated**, and Stage 2 answered it: on this
state, nothing fitted beats its own permuted-label null. So a learned linear-objective sizer
inherits that null by construction and there is nothing further to test.

What *is* new is the **non-linear** objective. Log growth / Kelly / any drawdown-constrained
objective depends on the predictive **distribution**, not just its mean (`f* ≈ mean/variance`).
That makes dispersion prediction the live question — and dispersion is the one quantity this
repo has already shown to be highly predictable (forward 20d RV, OOS R² 0.742) while direction
is not (0.003). So this stage tests dispersion, not direction.

## A. R-unit dispersion is NOT predictable — and that is a compliment to the ATR risk unit

Predicting `|r_10|` from the same 20 features, train → validation:

| arm | OOS R² | Spearman |
|---|---|---|
| real | **−0.0220** | 0.093 |
| permuted-label | +0.0036 | 0.073 |

The real model is **worse than its null** on R². This looks like it contradicts
`forward_rv_predictability.py`'s R² = 0.742, and it does not — the two predict different
things:

* that study predicted forward **realised volatility in absolute terms** from HAR-RV lags;
* this predicts `|forward return| ÷ ATR`, i.e. dispersion **after** the known volatility level
  has already been divided out.

So the finding is: **the ATR-based risk unit already absorbs essentially all the predictable
dispersion.** There is no residual vol signal left for a learned sizer to exploit on top of it.
That is a positive result about the existing risk framework, not a failure of the experiment.

## B. The incumbent score DOES carry size information — the one usable positive

Realised Kelly fraction by `mom_score` decile (within-bar deciles, validation, ~23.4k rows
each; `f*` maximises mean log growth over a fixed-fraction grid, with wipe-out bounded):

| decile | mean R | sd R | **Kelly f\*** | log growth at f\* | share ≥ +3R |
|---|---|---|---|---|---|
| 1 | 0.009 | 1.187 | **0.02** | −0.0001 | 0.35% |
| 2 | 0.018 | 1.243 | 0.02 | +0.0001 | 0.69% |
| 3 | 0.040 | 1.252 | 0.02 | +0.0005 | 0.90% |
| 4 | 0.043 | 1.218 | 0.02 | +0.0006 | 0.70% |
| 5 | 0.078 | 1.279 | 0.04 | +0.0018 | 1.33% |
| 6 | 0.063 | 1.337 | 0.04 | +0.0011 | 1.79% |
| 7 | 0.082 | 1.297 | 0.04 | +0.0019 | 1.83% |
| 8 | 0.137 | 1.511 | 0.08 | +0.0053 | 1.90% |
| 9 | 0.195 | 1.965 | 0.10 | +0.0086 | 2.52% |
| 10 | 0.209 | 1.573 | **0.12** | **+0.0119** | 3.57% |

**Monotone in every column that matters.** The optimal fraction rises **6x** from the bottom
decile to the top (0.02 → 0.12), mean R rises 24x, and the share of trades reaching +3R rises
10x. Decile 1 has a *negative* growth rate at its own optimum — it should not be traded at all.

So **score-proportional sizing is supported by the data**, with a measured 6x spread. This is
the incumbent's certified ordering showing up in a third independent form (after Stage 2's
top-k excess and its 8.8x tail concentration).

## C. A learned dispersion-scaled sizer fails its null

Top-3 picks, n=732, all policies normalised to the **same average size** so this is a shape
comparison, not a leverage comparison. Flat Kelly `f*` = 0.24, log growth 0.05601/trade.

| policy | f\* | log growth | mean sized R | sd sized R | worst sized R |
|---|---|---|---|---|---|
| flat | 0.24 | **0.05601** | 0.501 | 1.778 | −2.618 |
| inverse predicted dispersion | 0.22 | 0.03543 | 0.346 | 1.463 | −2.518 |
| inverse realised ATR | 0.16 | 0.04272 | 0.537 | 2.207 | −4.516 |
| permuted dispersion (null) | 0.16 | 0.04140 | 0.491 | 2.045 | −6.006 |

* inverse-predicted-dispersion vs flat: **−0.02057** log growth
* permuted-dispersion vs flat: −0.01461
* **net of null: −0.00597**

No gain, and the null accounts for most of the (negative) effect. Note `inverse realised ATR`
also loses — as it must, since `R` is *already* ATR-normalised, so dividing by ATR again
double-counts the same adjustment. That row is a useful sanity check that the harness behaves
the way the algebra says it should.

## Verdict

* **Learned sizing is rejected**, for two reasons that are independent: the linear objective
  reduces to Stage 2's null, and the non-linear objective needs residual dispersion, which
  §A shows is not there once ATR has been divided out.
* **Score-proportional sizing is supported** — 6x spread in Kelly fraction across incumbent
  deciles, monotone, and the bottom decile is unprofitable at any fraction. That is a
  configuration finding, not a model.
* **Next testable variant:** the objective, not the estimator. Everything above optimises mean
  log growth on independent trades. The live book is a *portfolio* — correlated positions,
  shared risk budget — and portfolio-level sizing (correlation-aware, drawdown-constrained) is
  a different problem this stage does not touch. `research/portfolio_lab/` is where that lives.
