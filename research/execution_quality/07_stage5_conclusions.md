# Stage 5 — conclusions and recommendations

Rules fixed in `05_preregistration.md`; evidence in `06_stage4_findings.md`.
Counterfactuals: `stage5_stop_counterfactual.py`, `stage5_profit_protect.py`.

---

## The context these results sit in

Realized P&L over the study window (2026-07-13 → 08-28, paper):

| module | closed rows | realized |
|---|---|---|
| dealer_ranker | 37 | −$53,204 |
| meta_ranker | 23 | −$50,746 |
| momentum_expansion | 34 | −$36,981 |
| multi_ticker_swing_htf | 61 | −$8,496 |
| **4H subtotal** | **155** | **−$149,427** |
| multi_ticker_swing (30m, gross) | 117 | −$57,149 |
| spy_daytrader (gross) | 101 | −$368 |

The question was "how far are we from perfect execution". The answer is that
execution is not where the loss is coming from.

---

## What the study found, in order of how actionable it is

### 1. Do not retrain. The ranking is not the constraint. (rule A1)

Within the same decision bar, a module's top-3 names beat its own lower-ranked
names in **1 of 12 module×horizon cells**, and that one (Meta at 10d) reverses in
the holdout. Score-to-forward-MFE correlation is +0.035, +0.039, +0.011, −0.152.
Ranked names drifted **down** over the sample (median −0.293 ATR at 10d).

New data will not fix a map that does not predict at these horizons. The test
could have detected a ~30% relative improvement in forward MFE; it found nothing
at that scale, in a 2-month down-drifting small/mid-cap tape.

**Recommendation: no retraining. Revisit only with a different feature set or a
different label horizon — not with more rows of the same thing.**

### 2. Do not change the entry. It is already on the cheap side. (rule B2)

We enter late 62–90% of the time. Being early costs a median 0.50 ATR of
drawdown; being late costs 0.20 ATR of missed move — for Meta and HTF the ratio
is 7:1. The 20–31 minute decision lag costs ~nothing (`entry_slip` median
−0.01 ATR: we fill marginally *better* than the signal-time price). The delay
counterfactual reverses sign between explore and holdout, so there is no evidence
for adding or removing a confirmation delay.

**Recommendation: no change. Your instinct that late beats early is correct and
is now measured. Close "improve entry timing" as a work item.**

### 3. Do NOT tighten the stop — the obvious reading of the giveback is wrong

Stage 4C4 showed a stopped position reaches +0.32 ATR, travels to −0.55 ATR, and
recovers only 0.30 ATR afterwards. That reads as "the stop is too loose".

**Replaying every closed lifecycle against tighter stops refutes it.** Applied to
all positions rather than only the ones that stopped (the Stage-4 cut conditioned
on the outcome), mean return and win rate fall monotonically as the stop tightens:

| stop | median | mean | win rate | stopped |
|---|---|---|---|---|
| 0.50 ATR | −0.071 | 0.056 | 35.1% | 42% |
| 1.00 ATR | −0.035 | 0.104 | 40.2% | 21% |
| 2.00 ATR | −0.013 | 0.216 | 44.8% | 8% |
| **actual** | **−0.007** | **0.235** | **47.6%** | 0% |

The same stop that cuts a loser cuts the winners. **Recommendation: leave the
stop where it is.**

### 4. Profit protection buys consistency by destroying the mean — the real finding

The giveback is real: median MFE 0.35 ATR, median realized −0.01 ATR. An armed
give-back rule (once MFE ≥ 1.0 ATR, exit on a 40% retrace of the peak) does
exactly what it promises on the median, in **both** splits — and costs the mean in
both:

| split | | median | mean | win rate | p90 |
|---|---|---|---|---|---|
| explore | actual | −0.003 | **0.299** | 49.5% | 2.355 |
| explore | arm 0.5 / give 40% | **+0.261** | 0.104 | **62.8%** | 0.661 |
| holdout | actual | −0.044 | **−0.035** | 39.7% | 1.382 |
| holdout | arm 0.5 / give 40% | **−0.022** | −0.144 | **44.1%** | 0.461 |

Why: **the returns are almost entirely a right tail.** The top 10% of lifecycles
account for **185%** of the total underlying gain — everything else is net
negative. In broker P&L, the top 3 winners are +$44,570 against a total of
−$113,130; remove them and it is −$157,700. Every give-back rule cuts precisely
the trades that pay for the system.

This matches the repo's earlier never-profitable-exit result (2026-07-28: better
tail losses, worse mean → no live change). It is the same wall from a different
direction.

**Recommendation: do not deploy profit protection as a global rule.** It is
defensible *only* per module, and only where the mean is already negative —
Meta (mean −0.641 → −0.312) and dealer (−0.120 → −0.017) improve, while momentum
(1.232 → 0.391) and HTF (1.121 → 0.548) are gutted. Even then, n = 21 and 31.

### 5. The intraday engine's problem is the direction call, not the stop width

Its invalidation is **1.13 single-minute bars wide**, and 44% of setups have a
stop narrower than one minute's range — which argues for widening. The control
says no: in the 3 hours after a setup closes, price moves a median **−1.76R
against the setup**, 35% positive, versus **−0.19R and 49%** for matched random
windows on the same tickers and directions. These setups are followed by adverse
continuation, not by the move they predicted. Widening would buy larger losses.

**Recommendation: work on the entry criteria, not the invalidation width.
Thresholds there are already flagged uncalibrated, and execution promotion should
stay gated.** (Residual confound: the control does not match on "just moved
adversely", so the true gap is smaller than −1.76 vs −0.19.)

### 6. The feed is not the problem — Stage 0's hypothesis, retracted

Stage 0 measured the live IEX tape missing a median 22% of RTH minutes and flagged
it as a possible cause. Measured on the intraday engine's own setups, IEX was
missing **4%** and SIP saw only **5%** more range. The 22% came from microcaps the
*4H* modules trade, and those modules decide on 4-hour bars where missing minutes
are irrelevant. Also, real-time SIP is not in this subscription at all
(15-minute delay), so the original framing was doubly wrong.

**Recommendation: no data-plan upgrade on this evidence.**

---

## What is left open, honestly

* **The afternoon 4H bar** (18:00Z) defers its entries ~17.7 hours to the next
  open, crossing an overnight gap. 21 entries, 7 closed — genuinely undecidable
  today. **Revisit at n ≥ 20 closed.** The mechanism is real even though the
  effect is not yet measurable.
* **Only 54% of planned entries become positions**, and three of four modules
  buy *below* the top of their own ranking. Since A found the ranking carries no
  information, this currently costs nothing — but it would matter immediately if
  a future model did rank.
* **Everything here is one ~2-month regime** in which the ranked universe fell.
  A ranking that fails in a down tape is not proven to fail in an up one.
* The holdout is 68 closed lifecycles. It was used only to check the sign of
  pooled effects, never to certify a per-module parameter, and it did its job:
  it killed the delay rule and it flagged the profit-protect mean cost.

## The one-line answer

Execution is not the problem. The entry is well-placed and cheap, the stop is
already at the right width, and the exit's giveback cannot be recovered without
destroying the right tail that produces all the returns. **The constraint is that
the signals do not rank forward moves at the horizons being traded** — and that
is a research problem, not an execution or a retraining one.
