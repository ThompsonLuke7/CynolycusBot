# Liquidity-bucketed grid — the filter does not rescue options

2026-09-08. `scripts/thesis_test/liquidity_bucketed_grid.py`

## Result: no

129 usable contracts (up from 39), drawn from the signal spine and restricted to
names under a 20% option spread. **Mean return per contract, 8-day hold, no stop:**

| bucket | n | median spread | **GROSS** | with cost | shares |
|---|---|---|---|---|---|
| **top 10% tightest** | 16 | **4.9%** | **−23.6%** | −27.5% | −0.3% |
| top 25% | 37 | 6.0% | −42.6% | −46.1% | −6.1% |
| under 10% | 40 | 6.8% | −44.5% | −47.9% | −6.5% |
| under 15% | 72 | 7.9% | −42.9% | −47.9% | −7.2% |
| all | 129 | 13.7% | −34.0% | −42.5% | −6.4% |

**The gross option return is already deeply negative before any cost is charged.**
In the tightest bucket, cost adds only ~4 percentage points (−23.6% → −27.5%). So
on this sample the spread was never the binding problem — the selection was.

By hold, tightest bucket only (n=16, median spread 4.9%):

| hold | gross option | with cost | shares |
|---|---|---|---|
| 5d | −22.7% | −26.7% | −2.3% |
| 8d | −23.6% | −27.5% | −0.3% |
| 10d | −26.5% | −30.3% | **+0.9%** |
| 15d | −30.6% | −34.4% | **+5.4%** |
| 20d | −41.3% | −44.3% | **+6.7%** |

Shares improve monotonically with hold and turn positive from 10 days; options get
worse. Same shape as every other cut of this data.

## Retraction: the +$5,110 headline does not replicate

The earlier thesis test reported +$5,110 per $1,000 at 8 days with no stop, on 39
contracts — a mean of **+13.1%** per contract. On this larger sample the same
policy returns **−34.0%** gross.

I checked whether the gap is traded-vs-declined entries. It is not:

| | n | gross option, 8d |
|---|---|---|
| traded (policy acted) | 18 | −21.3% |
| not traded (policy declined) | 111 | −36.1% |
| **original sample (traded, all spreads)** | **39** | **+13.1%** |

The policy's own selection does help (−21.3% vs −36.1%), but both are negative.
The +13.1% belongs to one 39-contract sample and does not survive expansion.

**I flagged n=39 as unreliable each time I reported it; this is that caveat
cashing out.** The optimistic number should be treated as withdrawn, and the
conclusion from `18_cost_validated_verdict.md` — do not remove the option stop —
stands and is now better supported.

---

# What `invalidation_wider_than_max_atr` actually does

`strategies/intraday_structure/target_manager.py:52-58`:

```python
minimum_risk = 0.25 * atr
if risk < minimum_risk:  risk = minimum_risk          # floor
if risk > config.target.max_invalidation_atr * atr:   # 2.0
    return INVALIDATION_TOO_WIDE
```

It is a **per-trade risk cap**. The detector proposes a structural stop (below
EMA20/VWAP minus a tolerance, for a long). If that structural stop sits more than
**2.0 × ATR** from price, the setup is declined; if it sits closer than
**0.25 × ATR**, the stop is pushed out to the floor.

So the gate admits only setups whose structurally-correct stop happens to be
between 0.25 and 2.0 ATR away — and 98.8% of all abstentions are this one rule
firing.

**What that selects for is proximity to the level, not quality of setup.** It
admits bars where price is sitting right on its structural level, which is why the
admitted setups end up with stops measured at **1.13 one-minute bars wide** and
44% of them narrower than a single minute's range.

## Would widening it help? No — but it would not ruin it either

The recall test answers this directly. Setups declined by this rule reach a median
**0.509 ATR** of forward favourable excursion at 60 minutes; setups it confirms
reach **0.501**. Statistically the same, at all three horizons.

So raising `max_invalidation_atr` would admit setups that are **no better than the
ones already taken**, at larger risk per trade. It would increase volume and
per-trade risk without improving selection.

The reason widening cannot fix this is the second recall finding: **both groups
underperform a random minute on the same ticker** (control 0.758 ATR vs confirmed
0.501). The problem is not where the gate is set. It is that the setup-detection
process as a whole is choosing worse-than-random moments — so admitting more of
its output does not help.

---

# Does the recall test say anything about retraining? No.

**`intraday_structure` contains no machine-learning model.** There is no xgboost,
lightgbm, sklearn or joblib import anywhere in the package, and no model artifact.
It is entirely rules-based.

So the recall test says **nothing** about whether the ML models should be
retrained. It is a statement about a rules engine, and the two questions are
independent:

* **Retraining Meta** is decided by the bake-off, which found the deployed
  `meta_good` label orders forward move no better than chance (+0.0035) against
  +0.0434 for the rank composite. That still stands, the new label and regime
  panel are already in the matrix, and the Colab bundle is built. **Retrain Meta.**
* **Momentum and HTF** were tested in the same bake-off and their labels won.
  They need a data refresh (stale since 2026-05-14), not a retrain.
* **`intraday_structure`** cannot be "retrained" because it was never trained. It
  needs its entry criteria reconsidered, and the recall test says the current ones
  select worse-than-random moments.
