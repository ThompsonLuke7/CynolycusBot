# Multi-leg options experiment — execution cost overwhelms the gross vertical benefit

**Run:** 2026-09-04  
**Protocol:** `docs/superpowers/plans/2026-09-04-multileg-options-preregistration.md`  
**Code:** `scripts/multileg_options_experiment/`  
**Mode:** historical research only; no live or paper order behavior changed.

## Verdict

**The hypothesis is mechanically plausible but not validated for this book. Do not route signals to
multi-leg options from this result.**

The cleanest directional test, the bull-call debit spread, improved mean return on capital from
**-19.4% to +8.2% gross** on its 14 paired trades. That is the intended mechanism: the short wing
offsets premium and theta. But the advantage disappeared at only a **$0.04 half-spread per leg**
(-22.1% spread versus -22.2% call), and reversed sharply at the repo's measured **$0.08** reference
(-52.4% versus -25.0%). Its approximate break-even execution cost is four cents per leg per side,
half the observed reference.

No candidate met the registered minimum of 100 trades / 20 tickers / eight weeks. The usable
multi-leg cells contained only **2–23 paired trades over one or two weeks**, so none is decision
eligible. This is a data limitation, not a null result.

## What was tested

The input was the 58 completed August-expiry paths from the latest 4H long-call thesis study. For
each signal, the run fetched every required contract from Alpaca and required each leg to pass the
standing derivative sanity gate: at least six daily prints, >=60% trading-session coverage, <=40%
unchanged closes, and call/put return correlation with the underlying having the correct sign and
at least 0.50 magnitude. Entry and exit required prints for every leg on the same day; nothing was
forward-filled.

The experiment used an eight-session hold and no premium stop, matching the preceding study's best
gross long-call cell. Results were normalized by maximum loss (same-expiry defined-risk positions),
calendar net debit (explicit approximation), or stock capital. Execution scenarios charged every
option leg at entry and exit plus $0.65 per contract per side:

- gross: $0.00 half-spread;
- tight: $0.04 per option share;
- measured reference: $0.08 per option share;
- stress: $0.12 per option share.

Dollar cost, rather than percent of premium, is intentional: the live-fill study found the spread
was roughly fixed in cents, and a ratio fitted on cheap contracts is not scale-invariant.

## Primary readout

All comparisons below use the candidate's exact complete-case trades and the same long-call trades.
CI values are calendar-week block-bootstrap intervals where at least two weeks exist; with only two
blocks they are descriptive, not reliable inference.

| structure | paired n | mean at $0.08 | paired long call | difference | assessment |
|---|---:|---:|---:|---:|---|
| long shares | 34 | **-3.2%** | -21.1% | +17.9pp | best broad comparator; still negative, 2 weeks |
| covered call | 23 | **-3.1%** | -42.2% | +39.1pp | strongest option overlay, but primarily a stock position |
| call backspread | 11 | -31.3% | -48.7% | +17.4pp | less bad than calls; 0/11 winners |
| bull put credit spread | 5 | -44.9% | -58.3% | +13.5pp | far too sparse |
| bull call debit spread | 14 | **-52.4%** | -25.0% | **-27.4pp** | gross benefit fails realistic leg cost |
| broken-wing call butterfly | 6 | -51.3% | -31.9% | -19.4pp | too sparse; cost-sensitive |
| protective collar | 9 | -33.0% | -21.9% | -11.0pp | no benefit on these paired rows |
| call butterfly | 5 | -193.1% | -48.1% | -145.0pp | capital-small structures are overwhelmed by per-leg toll |
| call calendar | 8 | -504.6% | -72.7% | -432.0pp | unusable with trade-bar pricing; do not interpret |
| call diagonal | 8 | -480.6% | -56.4% | -424.2pp | unusable with trade-bar pricing; do not interpret |

At the measured cost, covered calls also beat shares on their 33 direct complete cases (-2.6% versus
negative 4.9%), but the difference is one two-week sample in which the bullish signals lost money; selling
upside naturally cushions that particular outcome. It does not establish a durable covered-call
edge and could be harmful precisely when the right tail returns.

## Volatility and neutral diagnostics

Long straddles (n=12, -26.9%) and strangles (n=12, -27.0%) both lost money at the measured cost.
The signals did not display a usable long-volatility edge here. Iron condors had only four paired
trades and lost 106%; iron butterflies had two and gained 37.9%. The latter is noise, not evidence.
These structures do not express the signal's bullish direction and cannot support a routing change.

## Coverage and the decisive limitation

Only **34 of 58** incumbent calls survived all validation plus synchronized entry/exit. Adding legs
collapsed coverage further:

- bull call spread: 14 paired trades;
- backspread: 11;
- covered call: 23;
- straddle/strangle: 12 each;
- butterflies/condors: 2–6;
- calendars/diagonals: 8.

The run rejected 61 leg paths for low coverage, 35 for too few bars, and 38 for wrong/weak
underlying correlation. It also rejected eight butterflies whose asynchronous trade closes implied
a free or near-free payoff bound, plus two structures whose option print was below intrinsic value.
Those rejections are important: a same-day trade close is still not a synchronized executable combo
quote.

This is why the report does **not** treat even individually valid leg bars as equivalent to a
historical NBBO spread ticket. The data can show that gross theta-offset mechanisms exist and that
leg-count cost is likely binding. It cannot prove an executable multi-leg edge.

The registered -39% structure-stop diagnostic was not run. Combining each long leg's daily low and
each short leg's daily high would join prices from different moments and fabricate a path; closes
alone would miss intraday stops. Forward synchronized combo marks are required.

## Strategy taxonomy researched

The names in retail material are numerous, but most reduce to a smaller set of distinct payoff and
expiry families. The Options Industry Council's strategy directory and OCC quick guide were used as
the authoritative taxonomy; Cboe's guide independently confirms the common vertical and iron-condor
constructions.

| family | canonical structures and common aliases | relevance here |
|---|---|---|
| vertical spreads | bull call/debit call, bull put/credit put, bear put/debit put, bear call/credit call; double bull/bear | primary directional theta-offset family |
| time spreads | long/short call or put calendars (horizontals), diagonals, double calendars/diagonals | sell near theta, retain farther-dated exposure; needs synchronized term quotes |
| butterflies | long/short call butterfly, long/short put butterfly, iron butterfly/iron fly, broken-wing butterfly | defined range or target-price payoff; three/four-leg toll |
| condors | long/short call condor, long/short put condor, iron condor, reverse iron condor | wider-range butterfly variants |
| volatility combinations | long/short straddle, long/short strangle | direction-neutral long/short volatility |
| ratio structures | call/put ratio spreads, call/put backspreads, covered ratio spread, stock repair | backspreads retain convex tail; some ratios have unbounded loss |
| stock overlays | covered call/buy-write, covered put, protective/married put, collar/fence, cash-secured put, cash-backed call, covered strangle/combination | alter stock income/tails rather than replace stock outright |
| synthetics and parity trades | synthetic long/short stock, synthetic long put, risk reversal, conversion/reversal, box, jelly roll | mostly duplicate stock/financing exposure, not alpha from theta mitigation |
| named composites | seagulls, jade lizards, Christmas trees, double diagonals | combinations of the primitives above, not independent evidence families |

Uncovered short calls/straddles/strangles, short ratios with an uncovered tail, and financing
arbitrages were deliberately excluded. The nervous system's policy requires bounded risk, and those
structures do not test whether the bullish signal can be expressed more efficiently.

## Decision and next experiment

1. **No routing change.** No option or share execution policy was modified.
2. **Do not deploy bull-call spreads on historical gross performance.** Their gross improvement was
   real in this tiny sample, but it required <=$0.04 half-spread per leg and failed at the measured
   $0.08 reference.
3. **Covered calls are the only sensible forward challenger from this readout**, because they stayed
   near flat and paid only one option-leg toll. They must be evaluated against shares, not calls, and
   only on a signal variant willing to cap upside.
4. **Forward capture is the binding next step:** save simultaneous bid/ask, quote timestamps, combo
   net debit/credit, and eventual executable exit quotes for the incumbent call, one bull-call
   spread, and shares on every eligible paper signal. Pre-register a readout at >=100 paired trades
   across >=8 weeks. Calendars and four-leg structures should wait until two-leg execution passes.

## Sources

- Options Industry Council, [All Strategies](https://prd-web.optionseducation.org/strategies/all-strategies-en)
- OCC/OIC, [Options Strategies Quick Guide](https://www.theocc.com/getmedia/f34f8a0d-806f-4f1a-adf7-d49d8d94b16e/option-strategies-quick-guide.pdf)
- Cboe, [Common Options Trading Strategies](https://cdn.cboe.com/resources/options/Trading_Strategies.pdf)
