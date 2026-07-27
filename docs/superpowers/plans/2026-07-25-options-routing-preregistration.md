# Pre-Registration — Options Instrument Routing Experiment

**Registered:** 2026-07-25, before any counterfactual result was computed and before any frozen-test data was touched.
**Companion to:** `2026-07-25-options-instrument-routing-experiment.md`
**Purpose:** fix the hypothesis set, the decision rules, and the test-set budget in advance, so that Phase 3/4 results cannot be reverse-fit to whatever the data happens to show.

The precedent this exists to avoid: the 2026-07 confluence-discovery study certified zero cross-signal interactions after the fact, having burned test data on an unbounded hypothesis search. That outcome was correct but expensive. Here the search space is bounded up front.

---

## 1. Primary hypothesis

**H0 (null):** Conditional on the signal, instrument choice does not matter — a naked long call/put (the incumbent, B1) is not systematically improvable by any structured alternative, after realistic spread and liquidity costs, at matched risk.

**H1:** There exist observable-at-signal-time states in which a specific structure beats B1 at matched risk, with the effect surviving the pessimistic spread assumption and a block bootstrap over time.

A well-powered failure to reject H0 is a **publishable result and a valid endpoint.** It would mean: keep swinging naked calls, and spend effort on signal quality instead of instrument engineering. That outcome must be reported as prominently as a positive one.

---

## 2. Registered hypotheses

Each is a directional claim with a named mechanism, a routing feature, and a candidate instrument. Anything not on this list is exploratory and may not be reported as a confirmed finding without a fresh test window.

| # | Hypothesis | Mechanism | Trigger feature (at signal_ts) | Instrument |
|---|---|---|---|---|
| H1 | High IV rank degrades naked long premium | You buy inflated vol; IV mean-reverts against you even when direction is right | `iv_rank` high | Vertical debit spread (sells the inflated wing back) |
| H2 | Low IV rank favors naked long premium | Cheap convexity; theta drag is small relative to the move | `iv_rank` low | Naked long option (B1 is already correct here) |
| H3 | Long expected holding period favors defined-risk / longer DTE | Theta compounds against short-DTE naked longs; module median holds run 8–27 bars | module's expected `bars_held` | Debit spread or extended-DTE long |
| H4 | Small target move favors shares or spreads over naked long | If the tp/sl distance is under ~1 ATR, premium decay eats the move | `tp_distance / atr_at_entry` low | Shares |
| H5 | Large expected move favors naked long / OTM convexity | Convexity pays only when the move is big relative to premium | `expected_move` vs `target_move` | Naked long option |
| H6 | Side asymmetry: puts underperform calls as naked longs | Existing evidence: calls +$5.6k vs puts −$25k over the same 575 live trades; put skew makes downside premium expensive | `direction == -1` | Put debit spread or short shares |
| H7 | Earnings-in-window destroys naked long premium via IV crush | Direction can be right and the position still loses to the vol collapse | `earnings_in_window` | Debit spread, calendar, or skip |
| H8 | Thin chains make multi-leg strategies non-viable regardless of edge | Spread paid N times on wide markets exceeds the structural edge | `liquidity_tier` low | Shares (forced) |
| H9 | Deep-ITM stock replacement improves capital efficiency without changing exposure | ~0.8 delta at a fraction of notional frees buying power at similar directional P&L | any, capital-constrained | Deep ITM call/put |
| H10 | Dealer gamma regime conditions the right structure | Negative-GEX regimes trend/expand (favors convexity); positive-GEX regimes pin (favors defined-risk or premium selling) | `net_gex` sign / gamma state | Naked long vs credit spread |

Interaction terms are **not** registered. Any interaction found is exploratory and requires a separate, later test window.

---

## 3. Decision rules, fixed in advance

- **Primary metric:** PnL per dollar of max loss (matched-risk expectancy), on frozen test data.
- **Secondary, reported always:** matched-notional net PnL, trade-series Sharpe, worst-1% tail, executable fraction, and cost drag as % of gross.
- **Significance:** 95% CI from a **block bootstrap over time** (blocks = calendar weeks, to respect the heavy time-clustering of trades). iid bootstrap is disallowed.
- **Spread robustness:** an effect counts only if it survives the **pessimistic** (full-modeled-spread) assumption. Optimistic/mid results may be shown for context but never as the headline.
- **Multiple comparisons:** 10 registered hypotheses → Benjamini-Hochberg FDR control at q=0.10 across the primary metric. Reported alongside raw p-values.
- **Minimum sample:** no recommendation from any cell with n < 100 trades or fewer than 20 distinct tickers or fewer than 8 distinct weeks. Cells below any of these thresholds are reported as **"insufficient power"** with the power floor stated — never as a weak recommendation.

---

## 4. Power analysis (to be completed from Phase 0 counts, before Phase 3 runs)

For each module, using the Phase 0 spine's realized n and per-trade PnL dispersion, compute and record here the **minimum detectable effect** at 80% power, q=0.10, under the block bootstrap:

**Completed 2026-07-25 from the Phase 0 spine, before any Phase 3 result was computed.**

MDE at 80% power, two-sided α=0.10 (matching BH q=0.10). `MDE_iid` uses raw n; `MDE_block` deflates
n to the number of distinct calendar weeks, a deliberately conservative stand-in for the registered
week-block bootstrap. **`MDE_block` is the binding number.**

| Module | n | weeks | tickers | per-trade sd | MDE_iid | MDE_block | median trade | Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| multi_ticker_swing (30m) | 19,539 | 43 | 336 | 3.33% | 0.06pp | **1.27pp** | +1.35% | **well powered** |
| multi_ticker_swing_htf | 23,173 | 54 | 749 | 10.91% | 0.18pp | **3.70pp** | −1.71% | **powered** |
| momentum_expansion | 3,876 | 52 | 284 | 13.76% | 0.55pp | **4.75pp** | +5.14% | **powered** |
| dealer_ranker | 70 | 2 | 37 | 2.37% | 0.71pp | **4.18pp** | −0.03% | **insufficient — 2 weeks** |
| meta_ranker | 3 | 2 | 2 | 16.81% | 24.16pp | **29.59pp** | +0.73% | **insufficient — n=3** |
| intraday_structure | 0 | — | — | — | — | — | — | **no ledger exists** |

**Consequences, binding:**
- The experiment can only answer the routing question for **three modules**: 30m swing, HTF swing, and
  momentum expansion. Together they carry 46,588 of 46,759 spine rows (99.6%), so this is not a
  meaningful loss of scope.
- **dealer_ranker, meta_ranker, and intraday_structure are declared underpowered up front.** They may be
  reported descriptively but **may not receive a routing recommendation**, and no later slicing,
  pooling, or re-cutting may resurrect them. dealer_ranker's n=70 spans only 2 calendar weeks — week-block
  resampling from 2 blocks is not inference. meta_ranker has 3 closed trades.
- Note the week counts are low relative to n for every module: trades are heavily clustered, exactly the
  condition that makes iid bootstrap dishonest. The gap between MDE_iid and MDE_block (often 20×)
  quantifies how much an iid analysis would have overstated significance.

**Caveat on units.** This table is computed on *underlying* signed return, the only outcome available
pre-Phase-3. The registered primary metric is PnL per dollar of max loss, whose dispersion differs —
option positions have far higher variance than the underlying. The MDEs above therefore establish
*relative* power across modules and the shape of the clustering penalty, not the final significance
threshold. **The table is recomputed on the actual matched-risk metric once Phase 3 emits it, and the
recomputed version governs.** The three verdicts above (which modules qualify at all) are driven by
n and week counts, which do not change, so those verdicts are final.

---

## 5. Test-set budget

- Frozen-test data is touched **once**, at the end of Phase 4, for the registered hypotheses only.
- All strategy-parameter grid search, feature engineering, and rule-threshold selection happens on train/validation. Thresholds are frozen before the test run.
- The Phase 5 hindsight replay on the 575 real live trades is **in-sample and hypothesis-generating by construction.** Nothing it finds may be reported as validated. It may propose new hypotheses, which then require an unburned window.
- Before the test run, check and record which modules' frozen test windows remain unburned by prior studies (the confluence study burned some through ~2027). A module with no clean window gets a walk-forward evaluation instead, labeled as such.

---

## 6. What would falsify the whole program

If, on frozen test data, the best registered routing rule fails to beat the always-naked-long incumbent at matched risk under pessimistic spreads, the conclusion is:

> Instrument structure is not a profitable lever for this signal set at this liquidity level. Naked long premium remains the default; effort belongs elsewhere.

That conclusion gets written up with the same care as a positive result, and the routing cards say "shares or naked long, by liquidity" rather than inventing a rule to justify the work.
