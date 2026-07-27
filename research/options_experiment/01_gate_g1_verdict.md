# Gate G1 — Verdict

**Author:** Claude, 2026-07-26. Numbers computed directly from `data/g1a_iv_check.csv`,
`data/g1b_reprice.csv`, `data/g1c_calibration.json` (produced by `scripts/validate_options_gate_g1.py`).
This is an independent verdict written from the raw outputs.

---

## Verdict: **FAIL as a point estimate. CONDITIONAL PASS as a bracket.**

We **cannot** claim to reproduce the dollar level of an options trade's P&L.
We **can** rank strategies relatively, and we can bound the truth between an optimistic and a
pessimistic assumption. Every downstream conclusion must therefore be a *relative* claim that
survives the pessimistic bound. No absolute P&L number from this system may be quoted as a forecast.

---

## G1b — the decisive test: repricing 575 real executed trades

470 of 575 (82%) were repriceable from real cached bars. Timestamp alignment was excellent
(median gap 1 minute at both entry and exit), so what follows is not a timing artifact.

**Mid-price modeling is systematically optimistic, in both directions:**

| | signed median error | interpretation |
|---|---:|---|
| Entry | **−8.51%** | our model buys *cheaper* than you actually paid |
| Exit | **+13.92%** | our model sells *higher* than you actually got |
| Round trip | **≈22.4%** | pure spread-crossing optimism |

This is the textbook signature of pricing at mid while real fills cross the bid/ask. It is not a
modeling error in the greeks or the IV solver — it is the absence of the bid/ask data Alpaca does not serve.

**Consequence, stated plainly:**

| Basis | Total P&L over the 470 trades | Error vs actual |
|---|---:|---:|
| **Actual (real fills)** | **−$13,906** | — |
| Mid-priced (optimistic) | +$10,774 | **+$24,680** |
| Flat-spread adjusted (pessimistic, 25.6% round trip) | −$24,248 | −$10,342 |

**Mid-pricing turns a real $13.9k loss into a fake $10.8k profit.** Had we run Phase 3 on mid prices
and skipped this gate, every result would have been fiction, and the fiction would have been
flattering — the exact failure mode this gate exists to catch.

**What survives:** correlation between simulated and actual P&L is **0.953** at mid and **0.938**
spread-adjusted. The *shape* is right; the *level* is biased. Relative ranking of trades is trustworthy.

**The bias is NOT side-dependent** — entry bias is −7.3% for calls and −9.9% for puts; exit is +12.7%
and +14.7%. This matters: the calls/puts asymmetry in the live book (+$4.6k vs −$18.5k here) is a **real
trading result, not a pricing artifact.** That hypothesis (H6) remains live and uncontaminated.

**The truth falls inside our bracket** (−$24,248 < −$13,906 < +$10,774), which validates the
three-assumption approach registered in the plan. But note the bracket is **$35k wide against $14k of
actual P&L**. Any strategy edge smaller than the bracket width is unresolvable by this method.

---

## G1a — cross-source IV check vs Schwab

| Contract set | n | median abs error (vol pts) | IQR |
|---|---:|---:|---|
| All | 3,697 | 14.87 | (3.98, 162.68) |
| **High-vega (≥$0.05/vol-pt) — the usable set** | 208 | **4.21** | (1.85, 11.30) |
| Low-vega wings (known ill-conditioned) | 3,489 | 17.10 | (4.28, 183.90) |
| Near-ATM, 0–7 DTE | 94 | 3.67 | — |

This **confirms rather than contradicts** the Phase 1b finding: IV inversion is only meaningful where
vega is material. On the high-vega contracts we agree with Schwab to ~4 vol points; on the wings the
comparison is meaningless in both directions.

Two caveats that stop me reading more into this: only 208 of 3,697 comparison points are in the usable
high-vega set, so the validation sample is thin; and Schwab's IV comes from their own model and
timing, so disagreement is not purely our error. **Operational rule (already in the plan): compute IV
rank, term slope and skew from near-ATM contracts only.** A 4-vol-point error is tolerable for a
*regime feature* like IV rank; it would not be tolerable for pricing a structure, which is why pricing
comes from observed market prices rather than from a fitted vol.

---

## G1c — the spread model, and the most serious limitation found

The calibrated median round-trip cost is **25.6%**, close to the 22.4% optimism actually observed —
so the model gets the *average* roughly right.

But its cross-sectional fit is **R² = 0.043**. It has essentially no power to say *which* contracts
have wide spreads and which have narrow ones. It predicts approximately the median, always.

**Why this specifically threatens the central question.** The thing you asked — *should I use a spread
instead of a naked call?* — is a comparison between structures with **different leg counts and
different liquidity profiles**, where the difference in spread cost is often the whole margin. A
2-leg vertical on a thin small-cap pays spread twice on wide markets; a 1-leg call on a liquid name
pays it once on a tight market. Our model currently cannot distinguish those two cases, and it is
precisely that distinction the answer turns on.

A flat spread also demonstrably overcorrects: applied uniformly it produced −$24,248 against −$13,906
actual, overshooting the loss by 74%.

**Required before Phase 3 conclusions on multi-leg structures:** replace the 4-variable regression with
a per-contract empirical spread proxy — the dispersion of real trade prints within a bar (via
`/v1beta1/options/trades`, which does work) and intrabar high/low relative to VWAP. This estimates each
contract's own spread from its own trading, instead of assigning every contract the median. Until that
lands, **single-leg comparisons are trustworthy in ranking; multi-leg comparisons are not.**

---

## What this gate permits and forbids

**Permitted:**
- Relative ranking of instruments on the same trade (correlation 0.94–0.95 supports this).
- Conclusions that hold at the pessimistic bound, per pre-registration.
- Single-leg comparisons (shares vs naked long vs deep ITM), where spread is paid once and the
  bias is uniform.
- Treating the calls/puts asymmetry as a real effect.

**Forbidden:**
- Any absolute P&L forecast. The bracket is $35k wide on $14k of P&L.
- Any mid-priced result, in any report, ever.
- Multi-leg conclusions until the per-contract spread proxy replaces the R²=0.04 regression.
- Any claim of edge smaller than the optimistic/pessimistic bracket width.

## Coverage

105 of 575 trades (18%) could not be repriced. Phase 5 must report results on the 82% and state the
excluded fraction rather than silently conditioning on repriceable trades.
