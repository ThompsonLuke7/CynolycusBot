# Re-analysis: the parabolic-tail thesis, and where Phase 3 overstated

**Author:** Claude, 2026-07-27. Prompted by user challenge: Phase 3's "shares beat options"
conclusion did not match the intended strategy, which is to capture parabolic/gamma-squeeze moves on
a *filtered, liquid* subset — not to replace shares across all signals.

**The challenge was largely correct.** This note records what Phase 3 got right, what it framed
badly, and what remains genuinely untested.

---

## 1. Where Phase 3 overstated

Phase 3 answered: *"Should options replace shares as the default instrument across all signals?"*
Answer: no, robustly.

That is **not** the strategy in question. The intended strategy is: liquid optionable names, filtered
by signal quality and dealer positioning, aiming at the right tail. Phase 3's headline averaged over
the whole distribution, which drowns exactly the effect the strategy targets.

Two specific framing errors:

1. **Matched-risk was the headline metric.** It divides by stop distance, so a share position with a
   3% stop gets a tiny denominator and a huge risk-adjusted return. That framing structurally favors
   tight-stopped shares and structurally penalizes options, whose denominator is the whole premium.
   It is a legitimate metric but it is the *wrong one* for a capital-efficiency thesis.
2. **Return on capital deployed was specified in the Phase 3 plan and never reported.** That is the
   metric the thesis actually turns on, and its omission is what made the conclusion look more
   decisive than the evidence supports.

---

## 2. The tail thesis is CORRECT — options dominate the parabolic move

Long trades where a call was executable (n=1,376), matched notional, pessimistic costs:

### Top 5% of realized moves

| | total P&L | capital deployed | return on capital |
|---|---:|---:|---:|
| shares | $78,415 | $340,000 | **23.1%** |
| **calls** | $72,514 | **$58,118** | **124.8%** |

Options produced **93% of the dollars on 17% of the capital.**

### By move size

| realized move | n | share return | **option return** | option beats shares |
|---|---:|---:|---:|---:|
| < 1 ATR | 41 | −0.4% | −72.1% | 0% |
| 1–2 ATR | 420 | +4.7% | −16.6% | 40% |
| 2–3 ATR | 299 | +5.5% | −11.2% | 40% |
| 3–4 ATR | 90 | −18.8% | −94.9% | 10% |
| **4–6 ATR** | 116 | +4.0% | **+33.6%** | **50%** |

Best single option outcome: **+360% on capital**, versus +42% for the best share outcome.

This is exactly the "right tool for the right job" pattern: options lose small and often, then win
enormously in the tail. The convexity is real and it is measurable in this data.

---

## 3. Why it still does not pay — the filter is the missing piece, not the instrument

The tail is real but too rare, and the entry cost too high, for an unfiltered options program.

**Does the module's own score select the tail?** Partly — and only for momentum:

| momentum score quartile | n | parabolic rate (≥4 ATR) |
|---|---:|---:|
| Q1 (worst) | 150 | 4.0% |
| Q2 | 149 | 13.4% |
| Q3 | 149 | 16.1% |
| **Q4 (best)** | 149 | **19.5%** |

Spearman score→move = **+0.133, p = 0.001 — genuinely predictive.**
HTF: Spearman −0.017, p = 0.75 — **not predictive at all.**

**But a 4.9× lift in parabolic rate still does not make options profitable:**

| momentum longs, return on capital | shares | options (calibrated) | options (pessimistic) |
|---|---:|---:|---:|
| Q4 (best scores) | +1.9% | −5.6% | **−20.1%** |

Top-quartile bootstrap, options minus shares: **−22.0pp, 95% CI [−36.7, −4.5] — shares still win.**

The arithmetic: a ~20% parabolic hit rate cannot carry a ~22% round-trip spread cost plus theta on
the 80% that fail. The gap is not small; it needs either a materially higher hit rate or materially
cheaper entry.

---

## 4. The gamma-squeeze thesis has NEVER been tested — the data does not overlap

This is the most important finding in this note.

| dataset | coverage |
|---|---|
| Trade history (all 3 modules) | 2025-05-20 → **2026-06-04** |
| Dealer/GEX snapshots | **2026-07-02** → 2026-07-24 (16 days) |
| Dealer rankings | 2026-07-02 → 2026-07-24 (15 files) |

**Overlap: zero. Trades inside the dealer-data window: 0.**

Every conclusion in Phase 3 was computed with **no dealer positioning, no GEX, no gamma-flip, no call
wall** — the entire mechanism the strategy is built on. Phase 3 does not refute the gamma-squeeze
thesis; it never touched it.

### It is reconstructible, and that is the recommended next step

Verified this session: Alpaca's contracts endpoint returns **`open_interest` populated on 100% of
expired contracts**, with an `open_interest_date` and `close_price` — e.g.
`AAPL250606C00110000, open_interest=117, open_interest_date=2025-06-05`.

Combined with the per-contract daily bars already cached (volume, trade count) and the greeks engine
in `research/options_lab/pricing.py`, that is enough to **reconstruct a historical dealer-positioning
surface** — strike-level gamma exposure, call/put walls, gamma flip — across the full 2025-05 → 2026-06
trade window, and finally test the actual thesis.

**Caveat, load-bearing:** the OI returned is a single value dated near expiry, not a daily time
series. A faithful daily GEX reconstruction is not possible from it. What is possible is (a) a
near-expiry OI snapshot, (b) full daily *volume*-based gamma proxies, and (c) OI interpolation
backward using daily volume. That is a proxy, and it must be validated against the 16 real Schwab
snapshot days before any conclusion rests on it.

---

## 5. Direct answers to the open design questions

**Calls vs puts — evidence says options should be long-only.**

| | executable share | live-book P&L |
|---|---:|---:|
| calls | 36–40% | **+$4,623** |
| puts | 14–26% | **−$18,529** |

Puts are worse on both axes, and Gate G1 established this asymmetry is a **real trading result, not a
pricing artifact** (pricing bias was −7.3% calls vs −9.9% puts — essentially identical). Puts are also
structurally disadvantaged: you pay put skew, and downside moves in these names are rarely parabolic in
the way upside squeezes are. **Recommendation: options long-only; express shorts in shares, or not at
all.**

**Market regime filter — module level or order level?** Not answerable from this data (regime/dealer
data does not overlap). Structurally, though, the evidence says the ordering should be:

1. **Liquidity gate first** (already working, and correctly excluding ~two-thirds of names — this is
   by design, not a defect).
2. **Parabolic-likelihood filter second** — this is the piece that does not exist yet and is the
   binding constraint. Momentum score is weakly predictive (+0.133); dealer/GEX state is the untested
   candidate.
3. **Instrument choice last**, conditional on 1 and 2.

Regime belongs at the *order* layer, not per module, because the same regime state should gate every
module's instrument choice consistently — but that is a design argument, not yet an evidenced one.

**Only long?** For options, yes on current evidence. For shares, the modules already run both sides
and HTF's shorts are not the problem — HTF's overall expectancy is (median trade −1.71%).

---

## 6. Revised conclusion

Phase 3's finding stands *as scoped*: options are not a blanket replacement for shares, and multi-leg
structures are badly hurt by spread cost on illiquid contracts.

Phase 3's finding does **not** establish that the gamma-squeeze strategy is wrong. The convexity the
strategy targets is real and measurable — options captured 93% of tail dollars on 17% of capital. What
is missing is a *selector* that identifies those trades in advance. The module scores are too weak
(momentum) or useless (HTF) on their own, and the dealer-positioning layer intended to do that job has
never been tested against a single trade.

**Next step, in priority order:**
1. Reconstruct historical dealer positioning from expired-contract OI + cached bars; validate the
   proxy against the 16 real Schwab snapshot days.
2. Re-run the tail analysis conditioned on reconstructed GEX state — the actual thesis test.
3. Only then revisit instrument routing.
