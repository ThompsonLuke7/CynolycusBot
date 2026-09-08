# Label bake-off — results

2026-08-31. Scripts: `scripts/label_bakeoff/{build_eval_targets,run_bakeoff,stability_check}.py`

## Method

Every candidate is trained with the **same features, rows, split and
hyperparameters** — only the training target changes. Scored on two targets built
from 4H bars that **no candidate equals**:

* `eval_mfe_10b` — forward MFE in ATR over 10 x 4H bars (~5 trading days, the
  horizon actually traded per Stage 6)
* `eval_clean_10b` — the same minus adverse excursion (prefers clean moves)

Time-ordered split with a **60-bar embargo at both boundaries** (wider than the
longest label look-forward; the capstone audit flagged production splits have
none). Scored by **within-bar** rank correlation, so a market-wide up day cannot
read as ranking skill, plus top-decile lift. 500-ticker subsample, full history
each. Momentum/HTF: 560k train / 187k val / ~305k test. Meta: 45k / 15k / 42k.

**The decision metric is tradeable alignment, not learnability** — learnability
falls by construction when the volatility component is removed.

---

## Momentum — a real trade-off, and my proposed fix was the worst candidate

| candidate | learnability | rho vs MFE_5d | lift MFE | rho vs clean | lift clean |
|---|---|---|---|---|---|
| **M0_current** | 0.127 | 0.066 | 0.299 | **0.048** | **0.388** |
| M1_beta_alpha | 0.090 | **0.034** | 0.129 | 0.006 | 0.114 |
| M2_drop_alpha | 0.090 | 0.075 | 0.258 | −0.017 | −0.133 |
| **M3_pure_atr_adj** | 0.129 | **0.104** | **0.345** | 0.000 | −0.007 |
| M4_atr_dd | 0.119 | 0.044 | 0.122 | −0.031 | −0.223 |
| M5_reweighted | 0.073 | 0.075 | 0.232 | 0.003 | 0.041 |

Paired bootstrap over 740 test bars:

| comparison | paired diff | 95% CI | verdict |
|---|---|---|---|
| M3 − M0 on **MFE** | **+0.0383** | [+0.0242, +0.0523] | real; M3 wins 58.9% of bars |
| M3 − M0 on **clean** | **−0.0483** | [−0.0632, −0.0337] | real; M3 wins only 39.7% |

**`M1_beta_alpha` — the fix I recommended — is the worst candidate on the primary
target** (0.034 vs 0.066 current). Beta-adjusting the alpha component actively
destroys ordering ability.

The genuine finding is a trade-off, not a winner: **M3 captures more raw upside,
M0 captures more *clean* upside**, and both differences are outside noise. Given
Stage 4/5 — the exit hands back essentially all of the MFE, and MAE exceeds MFE at
every horizon — the clean target is the more relevant one under the current exit,
and **the current label already wins it.**

## HTF — the scale "bug" is load-bearing; fixing it makes things worse

| candidate | learnability | rho vs MFE_5d | lift MFE | rho vs clean | lift clean |
|---|---|---|---|---|---|
| **H0_current** | 0.100 | 0.085 | **0.320** | 0.009 | **0.110** |
| H1_zscored | **0.028** | **0.037** | 0.191 | −0.002 | 0.108 |
| H2_ranked | **0.011** | **0.033** | 0.038 | −0.002 | 0.050 |
| H3_pure_atr_adj | 0.110 | 0.094 | 0.195 | 0.010 | 0.066 |
| H4_rank_dd_heavy | 0.067 | 0.029 | 0.041 | −0.030 | −0.172 |

Making the stated 35/25/25/15 weights *real* — by z-scoring (H1) or ranking (H2)
the components — **halves the ordering ability and collapses learnability**
(0.100 → 0.028 → 0.011).

The reason is now obvious in hindsight: because `atr_adjusted` swamps the sum at
91.6% effective weight, HTF's label is already ~"vol-normalised forward MFE",
which is a good target. Honouring the nominal weights would move 35% onto the
alpha term (a volatility ranking) and 25% onto an inert drawdown term. **The
misspecification is what makes it work.**

## Meta — this one is genuinely broken

| candidate | objective | rho vs MFE_5d | rho vs clean |
|---|---|---|---|
| **T0_meta_good** (deployed) | binary:logistic | **+0.0035** | +0.0002 |
| T1_good_beta_neutral | binary:logistic | −0.0061 | −0.0047 |
| T2_trade_quality | reg | −0.015 | −0.031 |
| **T3_rank_composite** | reg | **+0.0434** | **+0.0290** |
| T4_pure_atr_adj | reg | +0.0405 | +0.0051 |

T0 was first scored with a regression objective, which is unfair to a 13.8%-positive
binary target; re-run with `binary:logistic` it improves from −0.015 to **+0.0035**.
That is the fair number, and it is still **indistinguishable from zero**: the
deployed Meta label produces predictions with essentially no ability to order
forward tradeable move.

Paired bootstrap, 105 test bars, T0 with its proper objective:

| comparison | paired diff | 95% CI | verdict |
|---|---|---|---|
| T3 − T0 on MFE | **+0.0398** | [+0.0076, +0.0717] | real; T3 wins 61.9% of bars |

Note `T1_good_beta_neutral` — replacing the beta-biased `fwd_max_alpha > 0` gate
with a beta-neutral one — is **worse than the current gate**. The defect is not
the beta bias; it is the sparse binary structure. A dense rank target fixes it.

---

## What this changes

**Only Meta's label should change.** Momentum's and HTF's current labels win or
tie on the metric that matters, and every "fix" I proposed for them made things
worse — three times, in three different ways.

That collapses the retraining scope dramatically. Because the base labels do not
change, the base models do not retrain, `MOM_OOF`/`HTF_OOF` do not change, and
there is no cascade:

```
  momentum  ->  UNCHANGED   (no retrain, OOF stands)
  HTF       ->  UNCHANGED   (no retrain, OOF stands)
  meta      ->  new rank-composite label + regime panel + date refresh -> retrain
```

**One model, not three.**

## Recommended plan

1. Change Meta's target from `meta_good` to the T3 rank composite
   (`0.45*rank(atr_adj) + 0.25*rank(beta_alpha) + 0.15*rank(drawdown, desc) + 0.15*rank(persistence)`),
   ranked within decision bar. Keep `meta_good` alongside for continuity.
2. Add the 36-column regime panel (full 2020-2026 history) to the Meta matrix.
3. Rebuild the Meta matrix (also refreshes it past 2026-08-28) and retrain Meta only.
4. Leave momentum and HTF alone. Their matrices are stale (data ends 2026-05-14)
   and a refresh is worthwhile on its own, but it is a *data* refresh, not a
   label change, and it does not require the label work above.

## Limits

* One test split per module, one 500-ticker subsample. The paired bootstraps
  address bar-to-bar noise, not split or subsample choice.
* Meta has only **105 test bars** (13 months of matrix history). Its CI is wide
  [+0.008, +0.072] and it is the weakest evidence of the three, even though it is
  the clearest direction.
* All candidates used one shared hyperparameter set; the deployed models are
  competition winners with tuned parameters, so absolute rho here is lower than
  production. The *comparison* holds the learner fixed, which is what matters.
* Absolute effect sizes remain small (rho 0.04-0.10). This changes which label is
  least-bad; it does not make any of them strong.
