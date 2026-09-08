# The rankings do work — at the top only. Selectivity is the lever.

2026-09-08. `scripts/thesis_test/ml_recall_test.py` + rank-depth analysis

## The finding

Precision at picking the day's **top 5% of forward 10-day MFE**, by how deep into
the ranking you go. Base rate 5.1%.

| module | base | **top 1** | top 3 | top 5 | top 10 | all ranked |
|---|---|---|---|---|---|---|
| **multi_ticker_swing_htf** | 5.1% | **17.6%** | 11.6% | 8.0% | 6.7% | 4.9% |
| **meta_ranker** | 5.1% | **13.6%** | 13.3% | 11.4% | 7.9% | 6.5% |
| dealer_ranker | 5.1% | 0.0% | 0.0% | 0.8% | 2.3% | 2.3% |
| momentum_expansion | 5.1% | 0.0% | 0.0% | 0.0% | 0.0% | 1.0% |

**HTF's single top-ranked name hits a big mover 17.6% of the time against a 5.1%
base — a 3.5x lift. Meta's top name hits 13.6%, 2.7x, and holds 13.3% at top-3.**

And for both, precision **decays monotonically with depth** — 17.6 → 11.6 → 8.0
→ 6.7 → 4.9. That monotonic decay across five levels is much harder to get by
chance than any single cell, and it is the signature of a ranking that genuinely
orders.

**The signal is real and the system dilutes it.** Taking the top 10 spends most of
the capital on names ranked 4-10, where precision is at or near the base rate.

This is the first clearly positive, actionable result in the whole investigation,
and it reframes the answer to "how do other people do this".

## Why our tests kept saying "impossible" while people visibly do it

Every test so far measured the **average decision on a fixed universe at a fixed
time**. That is not what a discretionary trader does.

* **Selectivity.** The system ranks ~2,900 names and takes the top 10, twice a
  day. A discretionary trader takes 1-3 trades a day from the same opportunity
  set. They are operating at the top-1 end of that table, where precision is
  3.5x base — we are operating at the top-10 end, where it is 1.3x.
* **What we cannot see.** Real-time catalyst reaction, level 2 and order flow,
  float rotation, halt dynamics, and discretionary sizing are all outside our
  feature set. A trader watching a halt resume is using information the 4H bar
  does not contain.
* **Survivorship.** Taiwan, every trade on the national exchange 1992-2006: under
  1% of day traders reliably profitable net of fees. Brazil index futures: of
  those persisting past 300 sessions, 97% lost money. The people who film
  themselves are drawn from the surviving tail. That does not make it impossible
  — you have done it yourself — but it does mean "I see it working" cannot
  calibrate how hard it is.

The honest synthesis: **the system's problem was never that no edge exists. It is
that a 2.7-3.5x edge at the very top of the ranking was being spread across ten
positions until it averaged down to noise, and then expressed in an instrument
that charges 16% round trip.**

## Do the ML modules have recall?

Recall against a top-10-of-2,900 ceiling of 0.34% is the wrong frame; precision
vs base rate is the informative one, and it is in the table above.

Overall lift (all ranked names, liquidity-matched base rate):

| module | precision | lift |
|---|---|---|
| meta_ranker | 6.9% | **1.32x** |
| multi_ticker_swing_htf | 4.1% | 0.79x |
| dealer_ranker | 3.4% | 0.67x |
| momentum_expansion | 0.9% | 0.17x |

**Caveat that limits these numbers:** only 54-84% of each module's ranked names
fall inside the scored pool (they need `is_eligible`, daily bars, and a
dollar-volume reading). Momentum's 0.17x rests on ~54% of its picks and should
not be quoted as a point estimate. The liquidity confound was checked and is not
the explanation — the liquid-only base rate (5.2%) is essentially identical to
the full-universe one (5.1%).

What survives the caveat: **meta and HTF have real signal concentrated at the top
of their rankings; momentum and dealer do not show it at any depth.**

## What we learned from the option-liquidity work

1. **The spread filter does not rescue options.** In the tightest-spread decile
   (median 4.9%), gross option return is already −23.6% per contract before any
   cost. Cost adds ~4 points. Selection, not spread, was the binding problem on
   that sample.
2. **The viable option universe is small**: 68 names under a 10% spread, 17 under
   5%, zero under 2%. Not 100-200.
3. **Spread is not just share price.** corr(log price, spread) = −0.61, but within
   every price bucket the best name is 4-10x tighter than the median. A spread
   filter adds information a price filter would not.
4. **The +$5,110 headline is withdrawn** — it does not replicate on 129 contracts.

## Actionable steps, in order

1. **Cut position count at the top of the ranking.** Take top 1-3 instead of top
   10 for Meta and HTF. This is a parameter (`--top-k`), not a research project,
   and the measured precision gain is 1.3x → 2.7-3.5x. **Highest value, lowest
   cost item found in this entire investigation.**
2. **Concentrate the freed capital** rather than shrinking the book — the point is
   the same dollars in fewer, better names.
3. **Investigate momentum and dealer separately.** Neither shows lift at any
   depth. Momentum's label won the bake-off, so this is a live-path or universe
   problem, not a label one.
4. **Express in shares, not options**, until an edge is demonstrated. Shares are
   the only post-cost positive expression measured, and they improve with hold
   (+6.7% per position at 20 days in the tight-spread bucket).
5. **Retrain Meta** on the rank-composite label — independent of all of the above,
   bundle already built.

## Limits

35 decision bars per module; HTF's top-1 cell is ~6 hits out of 35. The monotonic
decay is the robust part, not the individual cells. One ~2-month window. Forward
MFE at 10 days is not the same as realised P&L after exits and costs — this says
the *ranking* orders, not that a strategy built on it makes money.
