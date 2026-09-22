# Two controls, two different answers

2026-09-20. Scripts: `scripts/intraday_audit/matched_control.py`,
`scripts/intraday_audit/gate_and_timing.py`, `scripts/horizon_thesis/topk_depth_null.py`.

Both experiments here exist because a measurement was missing a control. One
finding dies; the other survives and is now on firmer ground than before.

---

# A. intraday_structure — "negative selection" is RETRACTED

## A.1 What `20_recall_test.md` claimed, and its own caveat

It reported CONFIRMED setups reaching 0.501 ATR of favourable excursion against
0.758 for a random minute on the same ticker, and called that negative
selection. Its stated limit: the control was NOT matched. The engine fires at
breakouts, pullback completions and opening range; the control sampled any RTH
minute. If the engine acts AFTER a move, lower forward excursion is mechanical.

## A.2 The matched control

Each event is paired with a minute on the **same ticker, a different day**,
within +/-15 minutes of the same time of day and +/-0.25 ATR of the same
trailing 30-minute run-up **signed in the setup's own direction**. Balance is
printed before the result, because an unbalanced "matched" control proves
nothing:

| group | n | median time | median run-up (ATR) |
|---|---|---|---|
| CONFIRMED (matched subset) | 71 | 10:47 | 0.151 |
| its matched control | 71 | 10:54 | 0.135 |
| DECLINED (matched subset) | 146 | 09:48 | 0.373 |
| its matched control | 146 | 09:56 | 0.302 |

**Paired result — each event against its own match, median MFE difference:**

| group | 15m | 30m | 60m |
|---|---|---|---|
| CONFIRMED | −0.004 (p=0.92) | +0.015 (p=0.73) | −0.029 (p=0.72) |
| DECLINED | −0.025 (p=0.54) | **−0.085 (p=0.008)** | **−0.159 (p=0.014)** |

**Confirmed setups are indistinguishable from comparable moments at every
horizon.** The negative-selection finding was the time-of-day / run-up confound
its own caveat named.

Two further corrections to the original:
* A **stale-bar bug** in the first version of this script: the 1m cache spans
  2026-07-08..08-28 while the ledgers run to 09-18, and events past the cache
  silently took the last bar in the file as their "decision minute" (median
  16:28 = the cache's final bar). With that guarded, even the UNPAIRED
  confirmed-vs-random gap is significant only at 60m (p=0.021), where the
  original had all three horizons at p<=0.003.
* **The gate carries information after all.** The original compared declined to
  confirmed directly (0.509 vs 0.501) and concluded neither was informative.
  Against matched controls, declined setups are significantly WORSE than
  comparable moments while confirmed ones are not — the gate is declining
  genuinely worse setups.

## A.3 Step 2 — the stop-width question cannot be answered yet

**Instrumentation gap:** on `setup_abstention` events `proposed_invalidation`,
`proposed_target` and `reward_risk` are **0% populated** — the engine abstains
inside `build_target_plan` before computing them. The width that 98.8% of its
decisions turn on is never written down.

`room_to_support_atr` was used as a proxy and it does not hold up: its median is
**1.00 ATR** across events declined for exceeding a **2.0 ATR** cap, and only 25%
exceed 2.0. It is measuring a different distance than the invalidation. The
`rho(width, MFE/width) = -0.60` that falls out is mechanical (width is the
denominator), not evidence.

**So "widen the cap and size down" stays untested.** The prerequisite is one
logging change: record the proposed invalidation and implied risk on the
abstention event.

## A.4 Step 3 — no exhaustion effect

Correlation between the run-up already in hand at the decision and forward MFE:

| group | 30m | 60m |
|---|---|---|
| CONFIRMED (n=83) | +0.038 | +0.026 |
| DECLINED (n=132) | −0.020 | −0.013 |

Flat. **The engine is not confirming into exhaustion** — that hypothesis, which
both the original doc and I favoured, is not supported.

## A.5 Where this leaves ML on top

The answer is still no, but **for a different and weaker reason**. Not because
the engine selects worse-than-random moments — it does not — but because its
confirmed setups show no measurable edge over comparable moments, so there is
no demonstrated edge for an ML layer to refine. With 83 confirmed setups inside
the bar window and **zero real fills**, the honest position is that the engine's
edge is unmeasured, not that it is negative.

---

# B. Selectivity — the depth gradient survives, the LEVELS do not

## B.1 The contamination is real and large

A model trained on within-bar **permuted labels**, identical protocol
(`topk_depth_null.py`), on raw forward returns:

| k | REAL excess (10d) | NULL excess (10d) |
|---|---|---|
| 1 | +0.95% (p=0.35) | **+2.12% (p=0.03)** |
| 3 | +0.51% (p=0.53) | **+2.05% (p=0.006)** |
| 10 | +0.24% (p=0.61) | **+1.62% (p=0.001)** |

At 30d the null reaches **+7.64%**, significant, while the real model is
negative. The mechanism is visible in what each holds:

| model | k | ATR | beta | realised vol |
|---|---|---|---|---|
| REAL | 3 | 0.85x | 1.04x | 0.91x |
| NULL | 3 | **1.41x** | **1.44x** | **1.56x** |

**Never read a model-based top-k excess on raw returns without this arm.** The
deployed `mom_score`'s top-10 carries **~2.0x universe beta**, so its LEVELS
(the +3.3% to +18.8% in `24_...md` §5) are substantially a beta premium.

## B.2 But the DEPTH gradient is not the tilt

The tilt is flat across depth — `mom_score` beta ratio 1.87 (k=1), 1.93 (k=3),
1.98 (k=10) — so it cannot make shallow beat deep; if anything it works against
it. Residualising returns on beta and liquidity **within each bar**, with the
corporate-action guard applied:

| hold | k=3 | k=10 | k3−k10 raw | k3−k10 residualised |
|---|---|---|---|---|
| 5d | +3.30% | +2.21% | +1.10pp | **+1.07pp** (p<0.001) |
| 10d | +5.96% | +4.30% | +1.66pp | **+1.54pp** (p<0.001) |
| 15d | +9.29% | +6.47% | +2.82pp | **+2.65pp** (p<0.001) |

**`RANKING_CONFIG["top_n"] = 3` stands**, and on better evidence than it was set
on: the specific objection (a volatility/beta tilt concentrated at the top) has
been tested and rejected.

Caveat recorded: a freshly-trained model on the momentum training matrix showed
**no** depth gradient (+0.27/+0.01/−0.28pp, all p>0.49). That is a different
estimator on a different split from the deployed walk-forward OOF score. The
parameter follows the deployed score, which is what runs.

## B.3 A guard that matters more than its size

Without the corporate-action guard the same gradient reads +2.36/+4.31/+7.20pp —
roughly double. The guard drops ~0.006% of rows, but flagged rows are 186x
over-represented at rank 1, so it moves the answer by more than half. Any depth
number computed on this cache without it is inflated.
