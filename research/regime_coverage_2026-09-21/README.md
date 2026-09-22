# Regime coverage audit — are all modules trading the same setup?

**Date:** 2026-09-21 · **Agent:** Claude · **Status:** investigation only, nothing wired

Triggered by an NBIS chart: price bounced off a rising 100 EMA on 2026-09-01 and ran +30%
in five sessions, and no module entered. The question asked was whether every module is
really trading one breakout regime on different timeframes.

---

## 1. Answer: no. The modules are spread out — but not by design.

Empirical, not from reading config: for all 334 closed trades that carry an `entry_bar`,
where did the module buy relative to the ticker's own structure on the entry date?

| module | n | median `range_pos_20` | median dist. from EMA100 (ATR) | % above EMA100 | median prior 20d return |
|---|---|---|---|---|---|
| momentum_expansion | 79 | 0.858 | **+3.53** | 98.7% | **+24.0%** |
| dealer_ranker | 49 | 0.543 | +0.54 | 57.1% | +4.5% |
| spy_daytrader | 7 | 0.470 | +2.45 | 100% | −0.8% |
| meta_ranker | 51 | 0.403 | −0.77 | 37.3% | −3.1% |
| multi_ticker_swing | 36 | 0.246 | +0.73 | 61.1% | −2.2% |
| multi_ticker_swing_htf | 112 | **0.138** | **−2.03** | 25.0% | **−18.7%** |

The two biggest option traders sit at **opposite** extremes:

- **momentum_expansion** buys strength — 43% of its entries are more than **4 ATR above**
  the 100 EMA, 72% more than 2 ATR above, after a median +24% 20-day run.
- **multi_ticker_swing_htf** buys weakness — 53% of entries are on a *falling* 100 EMA with
  price *below* it, after a median −18.7% 20-day fall. Falling knives, not breakouts.

`multi_ticker_swing` has an explicit dip formula in
`strategies/multi_ticker_swing/features/build_features.py:285`:
`long_pos_score = (1.0 - range_pos)` — it scores longs *higher* the lower in the range price
sits. `dealer_ranker` is genuinely orthogonal (gamma-driven, spread across all quadrants).
`intraday_structure` is two-sided (43 short / 40 long).

So the system is not monolithic. What it lacks is a *designed* allocation across regimes —
the spread is emergent, and nobody chose it.

### The one hard gate that is real

`filter_momentum_candidates` (`strategies/momentum_expansion/inference/candidate_filter.py:74`)
is used by **exactly one** module — the momentum ranker (`inference/ranker.py:149`). Meta does
not inherit it; Meta consumes `mom_score` as a feature, which is why Meta's entries skew the
other way.

For NBIS: the ticker **was** in the universe snapshot on 2026-09-02 (2,702 rows), so the
universe gate did not drop it. `min_range_pos_20 = 0.45` did. 4H `range_pos_20` was 0.087 /
0.191 / 0.416 on Sep 1/2/3 and first cleared the gate on Sep 3 18:00. NBIS re-entered the
ranked pool on **Sep 8**, the blow-off top, where momentum fired `pullback_continuation` at a
bar close of 249.90 and was then blocked by options buying power ($5,612 needed, $3,958 left).

### Train/serve population mismatch (real defect, unfixed)

`candidate_filter.py:3` claims the gate "keeps that gate identical in both places."
`momentum_config.py:145` sets `apply_momentum_candidate_filter: False` for training
(`labels/expansion_labels.py:373`). The ranker is trained on unfiltered rows **including
pullbacks** and served only near-high rows. The docstring asserts the opposite of what the
config does.

---

## 2. The pullback setup was tested. It has no edge.

Universe: 2,965 tickers, ~460k stock-days, 2026-01-02..2026-08-20, $5M median-dollar-volume
floor. Controls: (A) whole universe, (B) **same-day, same ATR-decile** draws — B is the one
that matters, because `research/execution_quality/25_matched_control_and_depth_null.md`
already showed a shuffled-score null misses the vol/beta tilt.

Four variants, pre-specified before running, excess forward return vs control B:

| variant | n | 10d | 20d | p(20d) | months positive |
|---|---|---|---|---|---|
| 1. generic pullback to rising EMA100 | 35,011 | −0.37pp | **−0.83pp** | 7e-11 | 2/8 |
| 2. + top-25% 120d relative strength | 11,411 | −0.74pp | **−1.96pp** | 8e-11 | 3/8 |
| 3. + also within 15% of 52w high | 2,131 | −0.14pp | −0.62pp | 0.03 | 3/8 |
| 4. momentum's OWN zone (>3 ATR extended) | 89,292 | −0.11pp | −0.42pp | 6e-10 | 4/8 |

**All four are negative.** Adding relative strength — the best-grounded addition in the
momentum literature — made it *worse*, not better.

### The steelman: geometry, not mean return

The chart argument is really about asymmetry (tight stop under the EMA, big upside), not mean
return. So: MFE/MAE over the next 20 sessions, in ATR units, vs the same matched control.

| cohort | n | MFE (ATR) | MAE (ATR) | MFE/MAE |
|---|---|---|---|---|
| pullback to rising EMA100 | 35,752 | 2.07 | 1.71 | **1.21** |
| ↳ its matched control | 177,191 | 2.16 | 1.80 | 1.20 |
| extended >3 ATR above | 93,676 | 2.30 | 2.18 | **1.05** |
| ↳ its matched control | 372,475 | 2.20 | 1.94 | 1.14 |

This is the one directional result worth keeping: **buying extended is measurably worse than a
volatility-matched random entry** (1.05 vs 1.14). **Buying the pullback is merely neutral**
(1.21 vs 1.20). Moving momentum from "extended" to "pullback" is worth avoiding harm, not
finding edge. NBIS is a chart selected *because* it worked — a survivor anecdote.

---

## 3. What actually drives the losses

Exit family across all 484 closed trades:

| exit family | n | total P&L | median | win rate |
|---|---|---|---|---|
| stop | 114 | **−$294,217** | −$2,578 | 0.9% |
| other | 94 | −$71,027 | −$40 | 19.1% |
| target/invalidation | 46 | −$4,744 | −$56 | 19.6% |
| horizon/time | 148 | +$20,376 | $0 | 38.5% |
| take_profit | 82 | **+$49,237** | +$308 | 91.5% |

Stops are the whole loss. **But they are not too tight**: after a stop fires, the underlying
goes on to return −2.94% over the next 10 sessions (n=87, t=−1.27, p=0.21) — i.e. the names
keep falling. We are not being shaken out at local lows; we are exiting positions that
continue to deteriorate. That is an entry-selection problem, not a stop-width problem.

Also: **41 of 79** momentum trades and 55% of Meta's exit on `horizon` — a clock, not a
thesis. Only `take_profit` is reliably positive.

### The user's volatility hypothesis — not supported as stated

Momentum's median vol-expansion ratio at entry is **0.98x** its own 100-day norm; only 16.5%
enter while vol is >1.25x normal. It is not timing vol spikes. What it *is* doing is trading
structurally jumpy names — median daily ATR **6.94% of price**, ~2x a typical stock-day — and
entering after a >+10% five-day move 60.8% of the time (>+20%: 41.8%). Right symptom
("jumpy"), wrong mechanism (name selection, not vol timing).

---

## 4. Regime strategy families worth considering

Separating peer-reviewed evidence from vendor marketing — several sources found for VCP-style
claims (e.g. "90.77% success rate") are unsourced promotional content and should not be used.

**Well-evidenced:**
- **Time-series / trend momentum** — AQR's century-of-evidence work across stocks, bonds, FX,
  commodities since 1880.
- **Cross-sectional momentum** with the known crash risk — Daniel & Moskowitz, *Momentum
  Crashes* (JFE 2016): crashes are partly forecastable, concentrated in panic states after
  market declines and high volatility, contemporaneous with rebounds. Directly relevant: our
  momentum module buys high-beta extension with no regime conditioning.
- **Volatility-scaled momentum** — Barroso & Santa-Clara (2015); scaling by trailing realised
  vol roughly doubles Sharpe vs static momentum. This is the cheapest well-supported change
  available to us and is a *sizing* rule, not a new signal.
- **52-week-high effect** — George & Hwang; nearness to the 52w high predicts drift and
  interacts with PEAD.
- **PEAD** — the most durable underreaction anomaly; an event-conditioned family we do not
  trade at all.
- **Short-term reversal** — the mirror of momentum at 1–5 day horizons; interacts with
  turnover.

**Not supported by our own data** (section 2): moving-average pullback as a standalone
price-structure screen.

---

## 5. Limits

- Bar cache `Data/shared/bars/1d` is **unadjusted and survivor-shaped**. Levels are inflated.
  The setup-vs-control **contrast** is largely immune because both arms draw from the same
  universe on the same day, but absolute forward returns here should not be quoted.
- P&L by quadrant/exit-family is confounded by position sizing, option-vs-equity route, and
  the accounting gaps in `research/profit_investigation_2026-09-19/`. Treat the *structural*
  profiles as solid and the *P&L attributions* as directional only.
- 8 months, one universe, no CIs on the monthly breakdown. Four pre-specified tests: at
  alpha=.05 there is a ~19% chance of one false positive, and all four came back negative
  anyway.
- The `entry_bar` field is absent on 127 of 484 closed trades (all of `intraday_structure`),
  so those are excluded from the structural profile.

## Reproduce

```
.venv/bin/python research/regime_coverage_2026-09-21/regime_profile.py   # structural profile
.venv/bin/python research/regime_coverage_2026-09-21/regime_matrix.py    # quadrant map
.venv/bin/python research/regime_coverage_2026-09-21/pullback_study.py   # setup vs 2 controls
.venv/bin/python research/regime_coverage_2026-09-21/variants.py         # 4 pre-specified variants
.venv/bin/python research/regime_coverage_2026-09-21/geometry.py         # MFE/MAE steelman
.venv/bin/python research/regime_coverage_2026-09-21/vol_regime.py       # vol hypothesis
.venv/bin/python research/regime_coverage_2026-09-21/post_stop.py        # post-stop recovery
```
