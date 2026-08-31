# Stage 4 — findings

Run 2026-08-28 against the rules fixed in `05_preregistration.md`.
Scripts: `stage4a_signal_quality.py`, `stage4b_entry.py`, `stage4c_exit.py`, `stage4c5_intraday.py`

---

## A. The ranking does not order forward moves. **Do not retrain.** (rule A1)

Within each decision bar, the module's top-3 names against its own lower-ranked
names on the *same* bar — same day, same tape, same universe, so only rank varies.
Median forward MFE difference, ATR units, bootstrap 95% CI:

| module | 1d | 3d | 10d |
|---|---|---|---|
| meta_ranker | −0.033 [−0.080,+0.045] | +0.040 [−0.110,+0.175] | **+0.329 [+0.056,+0.603]** |
| momentum_expansion | +0.038 [−0.037,+0.118] | +0.127 [−0.069,+0.249] | +0.165 [−0.097,+0.464] |
| multi_ticker_swing_htf | −0.047 [−0.114,+0.031] | −0.025 [−0.163,+0.074] | −0.059 [−0.228,+0.140] |
| dealer_ranker | −0.088 [−0.379,+0.197] | −0.233 [−0.549,+0.125] | −0.052 [−0.419,+0.436] |

**One cell of twelve is significant, and it does not replicate.** Meta's 10d edge
becomes −0.037 [−0.485,+0.428] in the holdout. Per the pre-registered stopping
rule, an effect that reverses sign across the split is *not established*.

Score is no better than rank. Pearson r between score and forward 3d MFE:
meta **+0.035**, momentum **+0.039**, HTF **+0.011**, dealer **−0.152**. The
score quintiles are non-monotonic in every module.

Drift control (all ranked names pooled, explore split): median forward return
**+0.012 ATR at 1d, −0.032 at 3d, −0.293 at 10d**; share positive 51% → 48% → 40%.
The names these modules rank drifted *down* over the sample. So "our picks went
up" was never available as a claim, and the top of the ranking is not
distinguishable from the rest of the same day's list.

**Verdict: A1 fires. The ranking is not the constraint, and retraining on the same
features is not the fix.** The effort should go to the exit (C) — the ranking is
already doing all it can, which is close to nothing at these horizons.

*Power.* With n≈430 per module and CI half-widths of ~0.10–0.15 ATR at 3d against
a median MFE of ~0.45 ATR, this test could detect roughly a **30% relative
improvement** in forward MFE. A real but smaller edge would be invisible here.
The claim is "no edge large enough to matter for execution", not "provably zero".
Sample is ~2 months and one regime — a down-drifting small/mid-cap tape.

---

## B. The entry is fine. **Recommend no change.** (rule B2)

| module | n | % late | phase median | missed_leg (late) | pre_adverse (early) | vs oracle |
|---|---|---|---|---|---|---|
| spy_daytrader | 101 | 90% | +42m | 0.072 | 0.050 | 0.074 |
| dealer_ranker | 40 | 88% | +350m | 0.487 | 0.618 | 0.222 |
| meta_ranker | 61 | 82% | +41m | 0.133 | **0.950** | 0.161 |
| momentum_expansion | 84 | 82% | +291m | 0.546 | **1.054** | 0.317 |
| multi_ticker_swing | 126 | 78% | +48m | 0.266 | 0.370 | 0.138 |
| multi_ticker_swing_htf | 100 | 62% | +29m | 0.129 | **0.931** | 0.142 |

In five of six modules, being early costs more than being late — for Meta and HTF
by **7x** (0.95 vs 0.13 ATR). Only the SPY daytrader reverses it, and there both
numbers are trivially small. We are late 62–90% of the time, which is the
cheap side of the trade.

**B3, the delay counterfactual, is a clean null.** Re-pricing every joined entry
at availability + {0,5,15,30,60,120,240} trading minutes on one common subset:
explore says waiting is *worse* (+0.017 to +0.093 ATR vs oracle), holdout says it
is *better* (−0.021 to −0.216). Sign reverses across the split → **not
established**. There is no evidence for adding a confirmation delay, and none for
removing one.

**B4 is underpowered and stays that way.** The 21 overnight-deferred entries pay
more against the oracle (+0.435 vs +0.176 ATR) but only 7 have closed, so the
realized comparison (+0.899 vs −0.064) rests on 7 trades. Pre-registered rule:
no recommendation below n=20. **Revisit when 20+ have closed** — the mechanism
(afternoon bar completes at the close, so the entry crosses an overnight gap) is
real and worth resolving, but it is not resolvable today.

---

## C. The exit is where the money is — and the modules split in two

### C1/C2 — per module

| module | n | MFE | realized | giveback | prem 3d | giveback ÷ prem3d | verdict |
|---|---|---|---|---|---|---|---|
| meta_ranker | 21 | 1.497 | −0.246 | 1.171 | 0.315 | **3.7x** | too loose |
| momentum_expansion | 32 | 1.215 | +0.075 | 0.987 | 0.359 | **2.7x** | too loose |
| multi_ticker_swing_htf | 57 | 1.507 | +0.511 | 0.790 | 0.357 | **2.2x** | too loose |
| dealer_ranker | 31 | 0.264 | −0.266 | 0.722 | 0.346 | **2.1x** | too loose |
| multi_ticker_swing | 117 | 0.439 | −0.043 | 0.481 | 0.565 | 0.9x | balanced |
| spy_daytrader | 95 | 0.030 | −0.005 | 0.039 | 0.468 | 0.1x | too tight |

**All four 4H modules hand back 2–4x more than they leave on the table.** They
reach a real profit — Meta and HTF median MFE ~1.5 ATR — and exit at or below
break-even. That is rule C2: too loose, hold too long past the peak.

The pooled number says the opposite (explore ratio 1.10, holdout 1.91, i.e.
"slightly too tight") because it is dominated by the 117 swing and 95 SPY rows.
This is exactly the reason C4 was pre-registered: **no exit recommendation should
ever be made on the pooled figure.**

### C4 — by exit reason, the decisive cut

| exit reason | n | MFE | realized | giveback | prem 3d | reading |
|---|---|---|---|---|---|---|
| **stop** | 67 | 0.315 | **−0.552** | **0.999** | 0.299 | fires far too late |
| **take_profit** | 49 | 3.027 | **+2.620** | 0.336 | 0.412 | working — keeps 87% of peak |
| (no ledger reason: swing + SPY) | 220 | 0.139 | −0.008 | 0.190 | 0.507 | too tight |

The two mechanisms are behaving in opposite directions and the fix is different
for each:

* **The take-profit is the healthiest thing in the system.** 3.03 ATR reached,
  2.62 ATR kept. Leave it alone.
* **The stop is the problem.** A stopped position reaches only +0.32 ATR, then
  travels to **−0.55 ATR** before the stop fires, and recovers just 0.30 ATR in
  the following three days. The loss is realised almost in full, and there is no
  meaningful bounce being avoided by waiting. Median time-to-peak sits at
  **73–79% of the hold** for every 4H module, so the position spends its last
  quarter giving back.

**This is the single highest-value change the study found, and it is a stop
change, not a model change.**

### C5 — the intraday structure engine: the stop is inside the noise, but widening it is not the fix

Against SIP 1-minute bars, 71 setups with a usable pre-entry window:

* median invalidation distance **0.171** price units
* median 1-minute high–low range in the hour before entry **0.117**
* **the stop is 1.13 single-minute bars wide**, and **44% of setups have a stop
  narrower than one minute's range**

That is a stop inside the noise band, and on its own it argues for widening.
**The control says otherwise.** In the 3 hours after a setup closes, price moves
a median **−1.76R against the setup's direction**, with only 35% positive. A
matched control — same ticker, same direction, random 3-hour windows — gives
**−0.19R and 49% positive**. So these setups are not being shaken out of moves
that then work; they are followed by adverse continuation well beyond the
tickers' baseline drift. Widening the invalidation would mostly buy larger losses.

**The defect is the direction call, not the stop width.** Combined with the
ledger's own MFE 0.33R vs MAE 0.17R and 61% of setups closing at or below zero
gross, the engine's entry criteria are what need work — and its thresholds are
already flagged uncalibrated in `docs/PROJECT_STATUS.md`.

*Residual confound, stated rather than buried:* the control matches ticker and
direction but not "conditional on having just moved adversely". Some of the
−1.76R is momentum continuation that any post-decline sample would show. The gap
to the control is large (−1.76 vs −0.19) but the true effect is smaller than the
raw difference.

### The feed question, answered and downgraded

Stage 0 found the live IEX tape missing a median 22% of RTH minutes and flagged it
as a plausible cause of the intraday engine's problems. Measured directly on the
engine's own setups (±45 min around entry, 23 sampled): **IEX was missing a median
4% of minutes, and SIP saw only 5% more price range.**

The 22% figure came from names the *4H modules* trade — microcaps like FBRX — and
the 4H modules decide on 4-hour bars, where a few missing minutes are irrelevant.
The intraday engine watches liquid names, where IEX is nearly complete.

**The feed is not a leading explanation for anything here, and the data-plan
upgrade floated in Stage 0 is not justified by this evidence.** Recorded as a
corrected hypothesis.
