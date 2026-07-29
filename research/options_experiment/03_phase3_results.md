# Phase 3 — Counterfactual Sweep Results

**Author:** Claude, 2026-07-27. Computed from `data/phase3_counterfactual.parquet`
(110,969 rows, 5,998 of 6,009 sampled trades, 698 tickers, ~10h runtime) via
`scripts/run_options_counterfactual_sweep.py`.

**Scope:** the three pre-registered powered modules only. Week coverage 100% (52/52, 43/43, 54/54).
Entries and exits are frozen from the existing backtests — **only the instrument changes.**

**Reading rules, from `01_gate_g1_verdict.md` and `02_spread_model.md`:** all claims are
*relative*; no absolute P&L is a forecast; conclusions must hold at the **pessimistic** cost bound.
Optimistic/mid numbers are excluded entirely.

---

## Headline: shares beat every options structure, in every module, under every framing

Paired comparison — same trade, same entry, same exit, only the instrument differs — restricted to
trades where **both** instruments were executable, so availability is not doing the work.

### Shares minus naked-long incumbent, matched risk, week-block bootstrap (5,000 reps)

| Module | cost basis | lift | 95% CI | n | weeks | verdict |
|---|---|---:|---|---:|---:|---|
| momentum_expansion | calibrated | **+0.097** | [+0.049, +0.146] | 597 | 52 | significant |
| momentum_expansion | pessimistic | **+0.246** | [+0.195, +0.295] | 597 | 52 | significant |
| multi_ticker_swing | calibrated | **+0.193** | [+0.153, +0.238] | 686 | 43 | significant |
| multi_ticker_swing | pessimistic | **+0.342** | [+0.292, +0.400] | 686 | 43 | significant |
| multi_ticker_swing_htf | calibrated | **+0.104** | [+0.020, +0.193] | 499 | 54 | significant |
| multi_ticker_swing_htf | pessimistic | **+0.253** | [+0.160, +0.350] | 499 | 54 | significant |

Every confidence interval excludes zero, in the same direction, under both cost assumptions.
Units are return per dollar at risk.

### Robustness: total net dollars on identical paired trades, both sizing modes

| Module | sizing | shares (cal) | options (cal) | shares (pess) | options (pess) |
|---|---|---:|---:|---:|---:|
| momentum_expansion | matched risk | +507,330 | +217,661 | +507,330 | −225,561 |
| multi_ticker_swing | matched risk | +86,153 | −574,724 | +86,153 | −1,085,751 |
| multi_ticker_swing_htf | matched risk | −478,186 | −736,435 | −478,186 | −1,109,570 |
| momentum_expansion | matched notional | +131,958 | +16,006 | +131,958 | −68,000 |
| multi_ticker_swing | matched notional | +7,004 | −44,442 | +7,004 | −86,595 |
| multi_ticker_swing_htf | matched notional | −20,226 | −73,264 | −20,226 | −121,394 |

The ordering never changes. Note momentum options are *positive* under calibrated costs and *negative*
under pessimistic — the single clearest illustration of why the pessimistic bound is mandatory.

### Full strategy ranking (matched risk, pessimistic, lift vs incumbent)

| Strategy | momentum | 30m swing | HTF |
|---|---:|---:|---:|
| **long_shares** | **+0.246** | **+0.342** | **+0.253** |
| long_call/put ATM (incumbent) | 0.000 | 0.000 | 0.000 |
| long OTM-30 | −0.029 | −0.055 | −0.109 |
| deep_itm | −0.053 | −0.069 | −0.036 |
| vertical_credit | −0.607 | −0.209 | −0.250 |
| vertical_debit (wide) | −0.811 | −0.540 | −0.759 |
| vertical_debit (narrow) | −2.161 | −1.262 | −1.554 |

**Multi-leg structures are catastrophically worse, not marginally worse.** A narrow debit spread loses
1.3–2.2 additional units of return per unit risked versus simply buying the option outright. The
mechanism is the one flagged in `02_spread_model.md`: every leg pays the spread, and these are not
liquid contracts.

---

## The finding that may matter more than the ranking

**The incumbent strategy was only executable on 33% of signals.**

True executable fraction — share of sampled trades where the structure could be built *and* passed the
live liquidity gate (OI ≥ 500, volume ≥ 100, the same floors live trading uses):

| Strategy | momentum | 30m swing | HTF |
|---|---:|---:|---:|
| long_shares | **100%** | **100%** | **100%** |
| long_call ATM | 30% | 40% | 36% |
| long_put ATM | n/a | 26% | 14% |
| deep_itm | 13% | 7% | 7% |
| vertical_debit (narrow) | 15% | 18% | 9% |
| vertical_credit | 13% | 27% | 11% |

Dominant blocking reason: `leg_illiquid` (43,369 occurrences), then `no_expiry_in_bucket` (13,517)
and `no_strike_at_width` (8,469).

For roughly two-thirds of the signals these modules generate, the options question is **moot** — the
contract you would need either does not trade or fails the liquidity floor. This is consistent with
the live system's own history (294 equity / 0 option routing; the Dealer Ranker's 10/10 chain shutout)
and it means instrument selection is not the lever it appears to be.

Note deep ITM at 7–13%: stock replacement is essentially unavailable on this book.

---

## Is there any regime where options win? No.

Option minus shares, pessimistic, matched risk, 1,782 paired trades. Options win on **34%** overall,
mean difference **−0.285**. No slice reverses it:

| Realized move | n | option win rate | mean diff |
|---|---:|---:|---:|
| 0.5–2 ATR | 505 | 44% | −0.183 |
| 2–4 ATR | 452 | 41% | −0.170 |
| **> 4 ATR** | 128 | **23%** | **−0.714** |

| Holding period | n | option win rate | mean diff |
|---|---:|---:|---:|
| 0–2 days | 703 | 34% | −0.247 |
| 6–10 days | 317 | 43% | −0.195 |
| > 20 days | 78 | 14% | −0.596 |

| Side | n | option win rate | mean diff |
|---|---:|---:|---:|
| long | 1,376 | 31% | −0.301 |
| short | 406 | 42% | −0.229 |

The most counterintuitive cell is the most informative: on the **biggest moves (>4 ATR), options do
worst**, exactly where convexity is supposed to pay. Under matched risk, a share position with a tight
stop has a small risk denominator, so a large move produces a very high return per dollar risked; the
option's denominator is the whole premium. Cheap convexity loses to a tight stop.

Registered hypotheses H1–H5, H9, H10 find no supporting regime here. H6 (side asymmetry) survives in
weakened form: shorts are less bad for options than longs (42% vs 31% win), but options still lose.

---

## Defects and caveats — read before using any number above

1. **`extended_dte_long` is a duplicate.** In the near-DTE bucket it selected the *identical contract*
   as `long_call_atm` in 100% of 1,376 overlapping cases. It is not a real variant and has been
   excluded from the ranking. **Registered hypothesis H3 (longer DTE helps) is therefore UNTESTED** in
   this bucket and must be re-run against the `far` bucket before H3 can be resolved either way.
2. **The matched-risk framing rewards tight stops, and assumes stops execute.** Gaps are captured — 8
   of 6,009 share trades lost more than their stop distance, worst −3.32× risk — but at only 0.1%,
   real gap frequency is likely understated because exits are evaluated at bar closes. A long option's
   max loss is contractual; a stop is not. This asymmetry genuinely favors options in the tail and is
   **not** fully reflected above.
3. **A survivorship trap I hit and corrected.** Unbuilt structures carry `sizing_mode = None`, so
   filtering on a sizing mode silently conditions on success and reports ~100% executability. The
   correct denominator is per (trade, strategy) *attempt*. Any future analysis of this parquet must
   aggregate attempts, not built rows.
4. **HTF is a losing book in this sample** (shares −$478k at matched risk) — its median trade is
   −1.71%. "Shares beat options" there means shares lose *less*. It is not an endorsement of the
   signal.
5. Calendars, diagonals, butterflies and the vol-structure family were not swept in this first pass.
   Given verticals lose by 0.2–2.2 units on liquidity cost alone, structures with more legs are very
   unlikely to reverse the finding, but they are formally untested.
6. This is Phase 3 — **descriptive only**. No routing rule has been fitted or validated. Frozen-test
   confirmation belongs to Phase 4.

---

## Provisional conclusion

For these three modules, on this book, the instrument question has a boring answer: **buy the stock.**
Naked long options — the current default — lose to shares at matched risk in every module under both
cost assumptions, and structured alternatives lose by far more, mostly to spread cost on illiquid
contracts. And for two-thirds of signals the options are not tradable at all.

This is the pre-registered null outcome (§6 of the pre-registration): *instrument structure is not a
profitable lever for this signal set at this liquidity level.* The remaining live question is not
"which structure" but whether the **underlying signals** — particularly HTF's negative expectancy and
the live book's put performance — are worth trading in any instrument.
