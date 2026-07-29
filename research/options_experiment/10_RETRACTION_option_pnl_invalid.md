# RETRACTION — the option P&L results in this experiment are invalid

**Author:** Claude, 2026-07-28. Prompted by the user's instruction to re-question every assumption
before the clean re-run. The deepest assumption failed.

---

## The claim being retracted

Every option profit-and-loss number produced in this experiment — Phase 3's strategy rankings, the
let-it-run exit comparison, the cheapness gate, the premium-selling test, and the headline
"shares beat options" conclusion — was computed by marking option positions from Alpaca daily/30-min
**trade bars**.

**Those bars cannot mark these positions.** The results built on them do not mean what they appear to.

---

## Evidence

**1. Nearly half the trades show literally zero option price change.**

| check | result |
|---|---:|
| module-exit trades with `exit_px == entry_px` **exactly** | **45.9%** of 800 |
| trades with abs(return) < 0.5% | 46.6% |

**2. The contracts barely trade.**

| metric | value |
|---|---:|
| bars recorded per contract over its entire life (median) | **11** |
| contracts with < 10 bars total | **45%** |
| median daily volume on days it does trade | 10 contracts |

Alpaca serves **trade** bars, not quotes. A bar exists only on days the contract traded. A "price" at
an arbitrary decision timestamp is therefore the **last print — potentially days stale** — not a mark.
Historical option quotes are unavailable (the endpoint 404s, established in Phase 1a), so there is no
way to mark these positions correctly.

**3. The decisive test: the options do not track their own underlyings.**

| subset | n | corr(stock return, option return) | stock >+1% → option |
|---|---:|---:|---|
| all | 800 | **+0.093** | +10.88% → **−2.73%** |
| stale prints only | 367 | undefined (no variance) | +9.57% → +0.00% |
| "live" prints only | 433 | **+0.101** | +12.13% → **−5.36%** |

A real ATM call's return correlates with its underlying's direction at roughly **+0.9**. Observing
**+0.10** — and calls *losing* while the stock gains 12% — is not an economic finding. It is proof
that the price series is not tracking the option's value.

Note this survives the obvious fix: removing identical prints does not repair it (+0.093 → +0.101),
because the remaining prints are still sparse and mistimed relative to the decision timestamps.

---

## Why it took this long to catch

Two earlier errors (percentage-vs-cents cost model; ATR-normalised parabolic label) each produced
plausible-looking, wrong answers that were *directionally consistent* with each other — options lose,
for explicable reasons. Each fix changed the magnitude but not the sign, which made the conclusion
look robust to correction. The tell was always visible in the data (median option return exactly
0.00%) and was not checked until the underlying/option correlation was computed directly.

**Process lesson:** when instrument A is a derivative of instrument B, the first sanity check must be
that A's returns actually correlate with B's. That check costs one line and would have invalidated
the entire option branch on day one.

---

## What is retracted

- Phase 3 (`03_phase3_results.md`) — all option strategy rankings and the shares-vs-options verdict.
- `07_let_it_run_and_verdict.md` — all exit-policy comparisons and the "thesis falsified" conclusion.
- `08_cheapness_selling_and_root_cause.md` — the cheapness gate, the premium-selling result, and the
  "spread is the root cause" claim (already partially corrected; now fully retracted).
- `04_tail_and_thesis_review.md` §2 — the "options capture 93% of tail dollars on 17% of capital"
  figure, which came from the same price series.

## What still stands

- **The shares results** (`09_shares_parabolic_filter.md`). These use only underlying bars, which are
  dense and reliable. The parabolic filter's lift — momentum top-20% **+3.61pp**, CI [+0.79, +5.84];
  HTF top-30% **+2.92pp**, CI [+1.18, +4.79] — is unaffected, as is the finding that HTF needs only an
  ATR% rule while momentum needs the model.
- **Gate G1's real-fill analysis** (`01_gate_g1_verdict.md`). Those 575 trades used **actual executed
  fills**, not bar-derived marks. The measured ~8-cent half-spread on a $0.94 median premium is real,
  as is the calls (+$4,623) vs puts (−$18,529) asymmetry.
- **Executability** (`03_...md` §"the finding that may matter more"). Contract *existence* and
  liquidity gating come from contract metadata and volume, not from price marks: the incumbent
  strategy was tradable on only ~33% of signals, deep ITM on 7–13%.
- **The infrastructure** — `research/options_lab/` (397 tests), the spine, forward-excursion labels,
  and 4.3GB of cached data all remain valid and reusable.

---

## What would be needed to answer the options question properly

1. **Forward capture of option marks.** Snapshot bid/ask (or at minimum mid) daily for candidate
   contracts and build the history going forward. This is the only reliable path and it starts from
   zero history today.
2. **Or restrict to genuinely liquid contracts** — ones trading most sessions, so trade bars
   approximate a continuous mark. On this universe that leaves very few names, which is itself the
   answer for this book.
3. **The 575 real fills remain the only trustworthy option P&L we have.** Any near-term claim about
   options should rest on those and be scoped to their sample size.

**Bottom line:** this experiment cannot say whether options beat shares on these signals. It can say
that the data required to answer it does not exist for this universe, and it has produced a validated
improvement to the shares path in the process.
