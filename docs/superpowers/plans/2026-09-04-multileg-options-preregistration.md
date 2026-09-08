# Pre-registration — multi-leg option structure experiment

**Registered:** 2026-09-04, before fetching any additional contract paths or computing any
multi-leg result.
**Status:** research only; no order-routing change is authorized by this experiment.

## Question

Can a bounded-risk multi-leg expression improve the repo's recent 35–45 DTE long-call result by
selling some theta/volatility and reducing premium at risk, after charging execution cost once per
leg? The incumbent is the same approximately 3%-OTM monthly call used by
`scripts/thesis_test/fetch_contract_paths.py`.

## Data-validity gate

Alpaca historical option bars are trade prints, not quotes. A structure is usable only when every
option leg:

1. has at least six daily bars;
2. covers at least 60% of underlying trading sessions in its observed window;
3. has no more than 40% unchanged closes; and
4. has daily-return correlation of at least +0.50 for calls or at most -0.50 for puts versus the
   underlying.

Entry and exit require an actual trade bar for every leg on the same date. No leg is forward-filled.
The entry bar must be no more than one underlying trading session after the signal. Any structure
failing these rules is unavailable, not zero-return.

## Fixed structures

All strikes are nearest listed strikes to the stated spot-relative targets. Same-expiry structures
use the incumbent monthly expiry.

| group | structure | legs |
|---|---|---|
| primary directional | long call | buy incumbent approximately +3% OTM call |
| primary directional | bull call debit spread | buy incumbent call; sell approximately +10% call |
| primary directional | bull put credit spread | sell approximately -5% put; buy approximately -10% put |
| primary directional | call backspread | sell ATM call; buy 2 approximately +10% calls |
| primary directional | call butterfly | buy ATM; sell 2 approximately +5%; buy approximately +10% calls |
| primary directional | broken-wing call butterfly | buy ATM; sell 2 approximately +5%; buy approximately +15% calls |
| primary directional | call calendar | sell nearest listed 7–28 DTE call; buy monthly call at same strike |
| primary directional | call diagonal | sell nearest listed 7–28 DTE approximately +5% call; buy monthly ATM call |
| stock overlay | covered call | buy 100 shares; sell approximately +10% call |
| stock overlay | protective collar | buy 100 shares; buy approximately -10% put; sell approximately +10% call |
| volatility diagnostic | long straddle | buy ATM call and put |
| volatility diagnostic | long strangle | buy approximately +5% call and -5% put |
| neutral diagnostic | short iron condor | buy -10% put, sell -5% put, sell +5% call, buy +10% call |
| neutral diagnostic | short iron butterfly | buy -10% put, sell ATM put and call, buy +10% call |

The diagnostics cannot confirm a directional routing rule; they answer whether these signals also
carry magnitude or range information. Uncovered short options, ratio spreads with an unbounded tail,
boxes/conversions, and synthetic-stock duplicates are catalogued in the report but excluded from
performance testing because they either violate defined-risk policy or do not test the stated theta
hypothesis.

## Outcomes and costs

- Fixed hold: eight underlying trading sessions, the best gross cell in the preceding thesis study.
- No premium stop is primary. A -39% stop is reported only for positive-debit structures as a
  diagnostic, using daily lows conservatively; it cannot be implemented faithfully for a structure
  without synchronized intraday quotes.
- Capital denominator: maximum loss for same-expiry defined-risk option structures; net debit as an
  explicitly approximate bound for calendars/diagonals; stock purchase cost for stock overlays.
- Sizing: fractional structures normalized to $1,000 capital for research comparability. This is not
  an executable order-size rule.
- Gross fills: observed trade-bar closes.
- Cost sensitivity: $0.04, $0.08, and $0.12 half-spread per option share per transaction, charged on
  every leg at entry and exit, plus $0.65/contract/leg/side. $0.08 is the live-fill median documented
  in `research/options_experiment/11_option_cost_reference.md`; dollar costs are used because a
  percentage calibrated on cheap contracts is not scale-invariant.

## Inference and decision rules

For each candidate, compare its return on allocated capital with the incumbent long call on the
candidate's exact complete-case rows. Report n, distinct tickers/weeks, mean, median, win rate,
worst decile, paired mean difference, and a calendar-week block-bootstrap 95% interval (seed
20260904, 10,000 draws).

A candidate may be called better only if all are true:

1. at least 100 trades, 20 tickers, and eight calendar weeks;
2. the paired mean improvement CI excludes zero;
3. mean improvement is positive under the $0.12 per-leg half-spread case;
4. worst-decile return is no worse than the long-call baseline; and
5. the result is directionally consistent across the three 4H source modules.

The existing sample is already used/in-sample. Even if these rules pass, the outcome is a candidate
for forward shadow validation, not a production rule. Failure to meet sample size is
**insufficient evidence**, not evidence that the structure has no value.

