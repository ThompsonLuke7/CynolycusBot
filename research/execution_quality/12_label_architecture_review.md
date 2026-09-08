# Before retraining: what actually depends on what, and why one label fix is three different fixes

2026-08-31. Read-only review. No code changed.

---

## 1. The dependency graph — your mental model is correct

```
  momentum/training_matrix_4h.parquet   (2.32M x 115, 2020-09 -> 2026-05, built 06-14)
        |  label: expansion_survival_score
        v
  momentum model  ---- walk-forward OOF (21d embargo) ---> mom_score ---.
                                                                         |
  htf/training_matrix_4h.parquet        (2.34M x 123, built 06-14)       |
        |  label: htf_swing_score / htf_top_swing_target                 |
        v                                                                |
  HTF model       ---- walk-forward OOF (21d embargo) ---> htf_score ----+
                                                                         |
                                                                         v
                              meta_ranker_matrix.parquet  (704k x 79, 2025-07 -> 2026-08, built 08-28)
                                = mom/htf OOF  +  theme(12) + news(16) + treasury(8)
                                  + macro(8) + guidance(3) + regime(4) + ticker meta
                                |  labels: trade_quality (regression) / meta_good (binary, deployed)
                                v
                              meta model
```

This is **stacked generalization**, and it is built correctly: the base signals
entering Meta are *out-of-fold* predictions with a 21-day embargo, so Meta never
sees an in-sample base prediction. Meta is not duplicating Momentum/HTF — it is a
level-1 model over their leak-free predictions plus alt data none of them see.

Two things worth knowing that are easy to miss:

* **HTF's LIVE runner reads `meta_ranker_matrix.parquet`** for inference features
  (`multi_ticker_swing_htf/live/runner.py:76`), not its own training matrix.
* **The retraining cascade is not optional.** Change Momentum's label →
  retrain Momentum → regenerate `MOM_OOF` → rebuild the Meta matrix → retrain
  Meta. Same for HTF. There is no way to change a base label and leave Meta
  alone, because `mom_score`/`htf_score` are Meta's features. **If we change any
  base label, all three models are retrained.** That is the argument for getting
  all of it right in one pass, which is what you are asking for.

## 2. My previous recommendation was too narrow — three labels, three DIFFERENT defects

I told you to "fix `fwd_max_alpha` and re-weight". That is correct for Momentum
and **wrong or insufficient for the other two.**

| | **momentum** | **HTF** | **meta** |
|---|---|---|---|
| composite built from | **ranks** of each component, within timestamp | **raw** components | **raw** components |
| `fwd_max_alpha` status | **pure no-op** — rho(alpha, raw return) = **1.000** | present, but ~4% of effect | **not** a no-op (raw-value target) |
| nominal vs effective weights | nominal = effective (ranks are all [0,1]) | **fiction** (below) | n/a |
| measured defect | 0.40 weight ranks **volatility** (rho +0.23..+0.26 vs ATR%) | scale mismatch swamps the stated design | `meta_good` is **2x beta-biased** |

### HTF's real defect is not alpha — it is that nothing is normalised

HTF sums raw components with no scaling. `atr_adjusted` has sd 6.085; `alpha` has
sd 0.184 — a 33x difference. Effective weight share (nominal weight x component sd,
normalised):

| component | nominal | **effective** |
|---|---|---|
| alpha | 35% | **3.9%** |
| atr_adjusted | 25% | **91.6%** |
| drawdown | 25% | **1.3%** |
| persistence | 15% | **3.2%** |

HTF's label is, in effect, **92% "forward MFE in ATR units" and nothing else.**
The drawdown penalty that is supposed to make it prefer *clean* moves contributes
1.3% — it is functionally absent, which is consistent with HTF holding through
large adverse excursions.

There is an irony worth sitting with: HTF has accidentally the **best-aligned**
effective target of the three (vol-normalised MFE is the tradeable quantity) and
the **worst** model (score vs its own composite rho ~ 0.00). That is more evidence
that the binding constraint is features, not labels.

### Meta's defect is in the deployed binary target

`trade_quality` (regression) = `1.0*alpha + 0.25*persist - 1.0*drawdown - 0.25*giveback`.
The alpha and drawdown beta-tilts largely cancel: corr(trade_quality, beta) = **+0.035**.
Nearly beta-neutral, by accident rather than design.

But the **deployed** target is `meta_good` (manifest `target_column`), a binary gate:
`fwd_max_return >= 12% AND drawdown <= 8% AND fwd_max_alpha > 0 AND liquidity >= 0.40`.
Measured on 704,638 rows:

* `meta_good` pass rate **14.6%**
* the `fwd_max_alpha > 0` gate passes **80%** of rows — it removes one row in five
* pass rate by beta half: **high-beta 19.3% vs low-beta 9.9%** — the deployed
  positive class is nearly **2x more likely** for a high-beta name

So Meta's classifier is being taught that high-beta names are "good setups".

## 3. What is already verified clean (so we do not re-litigate it)

* **No label leakage into features.** The matrix carries 9 forward/label columns,
  but the declared feature list (`FEATURE_COLUMNS_4H`, 108 names) contains **none**
  of them. The 2026-07 parabolic-filter fake-AUC incident was a *consumer* reading
  the matrix without excluding them, not a defect in the matrix.
* **OOF discipline.** Base scores entering Meta are walk-forward OOF with a 21-day
  embargo.
* **Prior audit scope.** `research/capstone/leakage_audit.md` (2026-07-12) covers
  splits, embargo and model selection. It does **not** examine label *semantics* —
  whether the composite measures what it claims. That question was not previously
  asked, which is why "we triple-checked this" and "the alpha term is a no-op" are
  both true at once.

Two minor items found in passing: `FEATURE_COLUMNS_4H` declares
`xsec_dollar_vol_surge_20_rank` and `xsec_rs_spy_20_rank`, which are **not in the
matrix** (108 declared vs 106 present). Harmless today but it means the declared
schema and the built matrix have already drifted.

## 4. The risk that should gate the retrain

The model's own score correlates **+0.17** with ticker ATR% (momentum), against
**+0.19** with its current target. **A large share of what the model has learned is
volatility persistence** — which is real, autocorrelated, and easy to predict.

Removing the volatility component removes the easy part of the target. The new
label may be *more correct and less learnable*. Trading a mediocre-but-learnable
target for a correct-but-unlearnable one is a real way to spend a full retrain and
end up worse.

**This is testable offline, without retraining anything.** All forward-outcome
columns already exist in the matrix. The gate:

1. Construct the candidate labels on the existing matrix (no rebuild needed).
2. Fit a fast model on a subsample with a proper time split + embargo.
3. Compare the current and candidate labels on **two** axes:
   * **learnability** — validation IC / NDCG against the label itself
   * **tradeable alignment** — does the *predicted* rank order realised forward
     MFE in ATR, out of sample

**The decision must be made on (2b), not (2a).** Learnability will fall by
construction; that is the point. The candidate wins only if predicted rank orders
*tradeable outcome* better than today's does.

Cost: hours on the existing matrices, versus a multi-day full-stack retrain.

## 5. Recommended sequence

1. **Run the label bake-off** above. Candidates: (a) current, (b) momentum with
   residual-path alpha + re-weight, (c) momentum with alpha dropped and weight
   moved to `fwd_atr_adj_return`, (d) HTF with components z-scored before
   weighting so the stated weights become real, (e) Meta with a beta-neutral
   `meta_good` gate.
2. **Only then** decide the label, because the retrain is one shot across three
   models.
3. Add the regime panel (36 cols, full 2020-2026 history) in the *same* rebuild —
   it is independent of the label question and there is no reason to pay the
   rebuild cost twice.
4. Rebuild all three matrices, retrain in dependency order:
   momentum + HTF → OOF → meta matrix → meta.
5. Fix the two declared-but-missing feature columns while the schema is open.

**Do not start at step 3.** The rebuild is cheap relative to the retrain, but the
retrain is only worth doing once and the label question is not yet settled by
evidence — only by diagnosis.
