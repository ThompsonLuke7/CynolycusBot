# Cost-validated verdict — do NOT remove the option stop

2026-09-02. `scripts/thesis_test/cost_adjusted_grid.py`

## The short answer

**The gross result does not survive measured transaction costs.** Every option
cell goes negative. The change you proposed — remove the option stop, shorten the
horizon — would not be enough on its own.

| policy | gross | **with measured cost** |
|---|---|---|
| OPTION 8d, no stop *(best gross)* | +5,110 | **−1,761** |
| OPTION 10d, no stop | +3,092 | −3,465 |
| OPTION 8d, −60% | +3,590 | −3,045 |
| OPTION 20d, −39% *(live today)* | −4,341 | −9,740 |
| **SHARES 20d** | +2,340 | **+2,340** |

Per $1,000 deployed, 39 usable contracts. Shares carry 5bp; options carry the
measured spread.

## What was measured, and what could not be

**Could not be measured — stated rather than guessed.** Historical option
**quotes** do not exist on this subscription: `/v1beta1/options/quotes` returns
404 and `/quotes/latest` is empty for an expired symbol. Only trades and daily
bars are served. So a true quote-by-quote replay is impossible; this is a
trade-price replay with a measured cost haircut, which is a weaker construction.

**Entry cost — measured, same modules as this study.** n=94 option entries having
both a realized broker fill and the mid quoted at order time:
**median +4.0% above mid** (meta +1.5%, HTF +2.3%, momentum +3.3%, dealer +7.9%),
84% paid above mid.

**Exit cost — measured, from the 30m module's 575 real fills.** n=566 filled sells
carrying a quote at fill time — the only such records in the repo:
**median +12.2% worse than mid**, mean +18.7%, **97% filled worse than mid**.
Quoted spread at those moments: median **25.1% of mid** (p75 55%, p90 120%).

**Round trip: 16.2% of premium.**

### Three conservatisms, named

1. The exit cost comes from a **different module** whose contracts were
   short-dated and often far OTM by exit — where spreads are widest. It is likely
   **pessimistic** for the 35–45 DTE monthlies replayed here.
2. A bar close is itself a **trade print**, already struck inside the spread, so
   charging a half-spread on top double-counts part of the cost.
3. Daily bars approximate an intraday stop touch by the daily low, which makes
   stop policies look slightly worse than reality.

All three push the option numbers **down**. The verdict should be read as a lower
bound, which is why the break-even below matters more than the point estimate.

## The number that actually decides this

Sweeping round-trip cost at the measured 1:3 entry:exit shape:

| policy | 0% | 4% | 8% | 12% | 16% |
|---|---|---|---|---|---|
| 8d, no stop | 5,110 | 3,363 | 1,650 | −29 | −1,676 |
| 8d, −60% | 3,590 | 1,903 | 249 | −1,372 | −2,963 |

* **Break-even round-trip cost, 8d no stop: 11.9% of premium**
* **Break-even, 8d with a −60% stop: 8.6%**
* **Measured: 16.2%**

The strategy is **~4.3 percentage points of round-trip cost short of viable.**
That is not hopeless, and it is not close to free either.

## What this changes about the diagnosis

The binding constraint is **not** the stop and **not** the horizon. Both matter —
the live −39%/21-day combination is the worst cell in every scenario — but fixing
them leaves the strategy negative. **The binding constraint is exit execution
cost**, and it is concentrated there: entry is already only 4.0%, while exit is
12.2% and 97% of exits fill worse than mid.

That reframes the work. The question is no longer "which stop and which horizon"
but "can an option exit be executed for under ~8% of premium instead of 12%".
Candidates, none tested:

* exit with a limit ladder rather than a market order (the exit path currently
  uses market orders for anything without a resting limit)
* exit before the contract goes deep OTM, where the spread blows out
* trade contracts with tighter quoted spreads — the median quoted spread on the
  30m module's exits was 25.1% of mid, which is not a market anyone should be
  crossing twice
* larger premium per contract; the median entry premium here is **$2.44**, and
  percentage spreads are worst on cheap options

## Recommendation

**Do not remove the option stop, and do not shorten the horizon on this evidence.**
Neither change makes the option expression viable at measured costs, and removing
the stop without also fixing the exit cost would remove a guard while leaving the
loss intact.

**Shares at 15–25 days is the only expression that is positive after costs** in
this sample (+$1,118 at 15d, +$2,340 at 20d, +$2,495 at 25d). That is also the
horizon your own thesis points at — the instrument is what differs.

If the option path is to be rescued, the experiment is an **exit-execution**
study, not another stop/horizon sweep. And the honest prerequisite is forward
data: the mark-capture infrastructure exists and has never been switched on, so
the next 50 option exits could measure this directly on our own fills rather than
borrowing the 30m module's.

## Limits

n=39 contracts, entries chosen by the current weak ranker, one ~2-month regime,
and the exit-cost estimate is borrowed across modules. The break-even framing is
more robust than any single cell, because it does not depend on which cost
estimate is exactly right — only on where 11.9% sits relative to what we can
actually achieve.
