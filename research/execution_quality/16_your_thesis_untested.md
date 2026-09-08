# Your actual thesis has never been tested — and the underlying data supports it

2026-09-02.

## The measurement

Your stated edge: *catch a momentum wave, hold 1–3 weeks on monthly expirations,
get paid by the gamma squeeze.* The system holds a median of **4–5 trading days**.
So the strategy you describe has never actually run.

Underlying distribution for the **actual traded entries**, by holding horizon:

| hold | n | median MFE | p90 | p95 | MFE>10% | MFE>20% | **MFE>30%** | median MAE |
|---|---|---|---|---|---|---|---|---|
| **5d** (what the system does) | 207 | 6.9% | 23.9% | 34.2% | 34% | 14% | **6%** | −9.2% |
| 10d | 201 | 8.9% | 34.3% | 51.6% | 44% | 24% | 12% | −13.0% |
| **15d** (your thesis) | 190 | 10.2% | 41.4% | 61.3% | 51% | 30% | **18%** | −16.0% |
| 20d | 183 | 13.4% | 46.3% | 62.0% | 59% | 33% | **20%** | −16.8% |
| 25d | 177 | 14.5% | 49.0% | 63.5% | 63% | 35% | 21% | −17.6% |

**The fat right tail you have been describing is real, and it forms after the
system has already exited.** Trades reaching +30% underlying triple from 6% to
18% between day 5 and day 15. p95 goes from +34% to +61%.

## Why this partially reverses my earlier conclusion

I concluded the option wrapper was ~5x too expensive for the signal: edge +4.5%
against a −22.8% cost. **That measurement is correct for how the system currently
trades — 4–5 day holds on 21-DTE contracts — and it is not a valid test of your
thesis**, because at a 5-day hold the tail that pays for the premium has not
formed yet.

The asymmetry matters specifically for options. Between 5d and 15d:

* MFE > 30% goes 6% → 18% (**3x**)
* median MAE goes −9.2% → −16.0% (**1.7x**)

A long call's downside is capped at the premium, so a worsening MAE costs a call
much less than it costs shares, while a tripling right tail pays a call far more.
**That is the structural argument for options on a right-tail thesis, and it only
appears at your horizon.**

## What is still not established

Whether the tail is fat *enough* to pay the theta. That needs honest option
marks, not arithmetic on my part — the 2026-07 retraction happened precisely
because option P&L was inferred rather than measured. The well-posed version:

* hold to 15–20 trading days, on **monthly** expiries (35–45 DTE at entry, so the
  contract is not in its terminal theta decay during the hold)
* price with real forward bid/ask marks — the capture infrastructure already
  exists (`strategies/spy_intraday/Policy/option_mark_capture.py`, and the 30m
  module's 575 real fills are the only trustworthy option P&L in the project)
* compare against the same entries expressed in shares

That is a specific, bounded experiment, and it is the one that decides whether
the higher-timeframe modules are worth continuing.

## What this does NOT overturn

* Stage 4A: the ranking is weak. A fatter tail at a longer horizon does not make
  the *selection* better; it means the instrument and the hold are mismatched to
  the selection you already have.
* Stage 5: naively holding longer harvests more adverse excursion too — in
  **shares**. The option asymmetry is what changes that calculus, and only if the
  premium is paid for a contract with enough life left.
