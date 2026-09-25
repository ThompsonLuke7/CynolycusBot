# Stage 4 — fitted-Q on HOLD/EXIT: the state is load-bearing, the ceiling is "hold 30"

2026-09-23. Script: `04_fitted_q_exit.py`. Output: `data/stage4_fitted_q.json`.
**Validation only; the test block is never read.**

Run as a **confirmatory test with a pre-declared expectation of null** — Stage 3's kill
criterion had already rejected a learned exit on these features, and the plan's named next
variant was features, not an optimiser. Its one new contribution is the **offset-only control
arm**: an identically-trained FQI whose only feature is position age, which asks directly
whether any state-dependent exit signal exists beyond a holding horizon.

## Method

Because feedback is full and the market exogenous, `Q(s, EXIT)` needs no model — exiting pays
`r_close[t] − cost`, which is observed. Only `Q(s, HOLD)` is learned, by bootstrapping

```
target(s) = γ · max( r_exit(s'), Q_{k−1}(s', HOLD) )
```

over the next position-day `s'` of the same entry, 6 iterations, XGBoost regressor. This is a
genuine RL estimator (it bootstraps through the Bellman operator) of the same quantity Stage 3
estimated by hindsight regression. 280,800 train / 65,880 validation position-days;
7,533 / 1,772 entries. γ ∈ {1.0, 0.99}.

## Results, γ = 1.0

| arm | n | best fixed (`fixed_30`) | oracle | **fqi_full** | gain vs best fixed | 95% CI | mean exit day |
|---|---|---|---|---|---|---|---|
| mom_score | 434 | 2.076 | 3.286 | **1.969** | **−0.107** | [−0.244, +0.030] | 29.2 |
| htf_score | 606 | 1.727 | 2.887 | **1.794** | **+0.067** | [−0.015, +0.168] | 29.5 |
| random_k | 732 | 0.417 | 1.585 | 0.411 | −0.006 | [−0.031, +0.020] | 29.2 |

Controls, same runs:

| arm | fqi_offset_only | gain | fqi_perm_state | gain | **net: full − offset-only** |
|---|---|---|---|---|---|
| mom_score | 0.841 | −1.235 [−1.60, −0.88] | 0.856 | −1.220 | **+1.128** |
| htf_score | 0.920 | −0.807 [−1.04, −0.60] | 0.926 | −0.801 | **+0.874** |
| random_k | 0.331 | −0.086 [−0.16, −0.02] | 0.331 | −0.086 | +0.080 |

γ = 0.99 is uniformly slightly worse (mom −0.145, htf +0.037), which is what discounting
should do to a problem whose payoff is at the end.

## What this says

**1. The null holds against the benchmark that matters.** FQI does not beat the best fixed
horizon on either ranked arm — momentum's point estimate is negative and both CIs straddle
zero. The pre-declared expectation is confirmed.

**2. But the state is genuinely load-bearing, which Stage 3 did not show.** An offset-only FQI
collapses to 0.84R (momentum) against fqi_full's 1.97R — a **+1.13R** gap — and a
permuted-state FQI is equally bad (0.86R). So the features *are* being used, and used
correctly: offset-only exits at day ~22 and leaves 1.1R on the table, while the full agent
learns to keep holding to day ~29.

This is the sharpest available statement of the whole exit result:

> The state contains enough information to tell the agent **not to exit early**, and that is
> exactly what it learns. It contains nothing beyond that — no timing signal that beats simply
> holding to the horizon.

**3. The contrast with Stage 3 is informative, not contradictory.** Stage 3's hindsight
imitation was matched by its permuted-label twin; Stage 4's bootstrapped agent beats its
permuted-state twin by +1.11R. The difference is structural: FQI's target is built from the
path itself, so it cannot collapse to "predict the offset-conditional mean". Both arrive at the
same policy — hold to the horizon — but only FQI demonstrates that it got there from the state.

**4. Even on random entries** FQI nets +0.080R over offset-only, so this is a generic property
of position-state information, not something the ranking creates. Consistent with Stage 3's
finding that 83% of the headroom is generic.

## Verdict

**Exit RL is rejected on these features**, on two independent estimators (hindsight imitation,
bootstrapped FQI) with three different controls (permuted label, permuted state, offset-only).
The pre-registered next variant stands unchanged: **features**, and specifically
forward-realised-volatility (`forward_rv_predictability.py`, OOS R² 0.742) and decision-time
IV (shipped 2026-09-22, still accumulating). Neither existed when this state was assembled.

**The actionable finding remains Stage 3 §A.1**, and it needs no model: the live exit rule
returns +0.091R where holding 30 sessions returns +0.662R. Two independent RL estimators have
now converged on "hold to the horizon" as the optimal policy in this action space, which is
corroboration of that configuration change from a different direction.
