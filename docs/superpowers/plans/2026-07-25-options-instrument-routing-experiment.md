# Options Instrument Routing Experiment

**Date:** 2026-07-25
**Status:** Plan — approved for Phase 0/1 execution
**Question:** For each signal our modules generate, which instrument should we trade — shares, a naked long call/put (current default), or a structured options position — and under what conditions?

---

## 1. Framing

Today the system routes almost everything to either shares or a single long ATM-ish call/put. The user's stated problem is that naked long premium "has low risk management": it is a single tool applied to every job regardless of time-to-target, volatility level, expected move size, or conviction.

The correct framing is **not** "which options strategy is best on average." It is a **routing decision made at signal time**:

> Given a signal from module *M* with features observable at the decision timestamp, choose an instrument *I* from a defined menu to maximize risk-adjusted return per unit of capital/buying power.

The deliverable is therefore a **routing policy** — ideally an interpretable rule table, one card per module — not a leaderboard. A leaderboard of average strategy returns is the failure mode to avoid, because it hides the whole point: the right tool depends on state.

### Modules in scope

| Module | Cadence | Ledger / signal source |
|---|---|---|
| `momentum_expansion` | 4H | `backtests/position_sizing_4h/*_trades.parquet` (strategy=`momentum`), live audit |
| `multi_ticker_swing_htf` | 4H | same parquet (strategy=`htf`), `backtests/ev_experiments_4h/htf_final_frozen_test_trades.parquet` |
| `meta_ranker` | 4H | `Data/inference/meta_ranker/closed_trades.jsonl`, `live_signal_audit.jsonl` |
| `dealer_positioning` Ranker | 4H (near-close) | `Data/dealer_positioning/rankings/*.parquet`, `Data/analysis/dealer_ranker_july_exploratory/trade_outcomes.csv` |
| `multi_ticker_swing` | 30m | `strategies/multi_ticker_swing/models/p_swing_probs_600.parquet` + `data/processed/features_30m.parquet`, `backtest/results*/` |
| `dealer_positioning` levels (Amethyst) | daily snapshot | `Data/dealer_positioning/historical_snapshots/` (16 days), per-symbol chains |
| `intraday_structure` | intraday | paper-only; `strategies/intraday_structure/` engine + `dealer_plate.py` |

The options-positioning/structure/ranking modules serve a dual role: they are both **signal sources** (their own trades get routed) and **feature sources** (their GEX/VEX/gamma-regime state is a candidate routing feature for the other five modules). Both roles are tested.

---

## 2. The decisive data finding

The original assumption was that we would have to synthesize option prices from a modeled IV surface, because the repo only holds 16 trading days of real chain snapshots (2026-07-02 → 07-24, ~700 symbols, per-strike IV/greeks/OI, from Schwab).

**That assumption is wrong.** Alpaca's data API serves real historical option data for *expired* contracts, verified live this session:

| Capability | Endpoint | Verified |
|---|---|---|
| Expired contract discovery (full strike grid per expiry) | `/v2/options/contracts?status=inactive` | ✅ 200, works for small caps (AEHR 26, FCEL 216, ASPI 44, KULR 280 contracts for Aug-2025 expiry alone) |
| Daily OHLCV+VWAP+trade count | `/v1beta1/options/bars` `timeframe=1Day` | ✅ |
| 30-minute bars | same, `timeframe=30Min` | ✅ |
| 1-minute bars | same, `timeframe=1Min` | ✅ |
| Executed trade prints | `/v1beta1/options/trades` | ✅ 200 |
| History depth | — | ✅ hard floor **2024-01-18** (confirmed platform-wide in Phase 1a across contracts with differing listing horizons), covering the entire 2025-05 → 2026-06 backtest window |
| Historical bid/ask quotes | `/v1beta1/options/quotes` | ❌ 404 — **not available** |

**Consequences for the design:**

1. **No synthetic pricer is needed for entry/exit valuation.** Every leg of every structure is priced from real market bars at real timestamps. This removes what would have been the dominant source of experiment risk.
2. **IV is *implied from* real prices, not modeled.** We back out IV per contract per bar via a solver against the observed mid/VWAP. This gives a true historical IV surface for our own universe, which is itself a reusable research asset (IV rank, term structure, skew, IV/RV premium — all real).
3. **Real option volume and trade count per bar** give a direct, non-modeled liquidity gate. This matters enormously: the live system already discovered its own universe is chain-poor (294 equity / 0 option routing; Dealer Ranker's 10/10 `no_non_0dte_call_contracts` shutout). A 4-leg structure on a thin name is not tradable, and now we can *prove* that per trade rather than assume it.
4. **The one real gap is bid/ask.** Fill realism must be modeled, not observed. Mitigation in §6.

---

## 3. Phase plan

### Phase 0 — Experiment spine (unified signal ledger)

**Output:** `research/options_experiment/data/signal_spine.parquet` + `00_inventory.md`

Normalize every in-scope module's trades/signals into one schema:

```
module, ticker, signal_ts, entry_ts, exit_ts, direction, entry_px_underlying,
exit_px_underlying, exit_reason, bars_held, atr_at_entry, tp_price, sl_price,
score, cadence, source_file, is_frozen_test, is_live_real
```

Requirements:
- Preserve the distinction between **signal_ts** (decision time), **entry_ts** (fill time), and **exit_ts** — per repo time-correctness rules. Any routing feature must be computed strictly at `signal_ts`.
- Tag each row's provenance: backtest frozen-test / backtest in-sample / live paper real. These are never pooled in reporting.
- Report per module: n, date range, ticker count, holding-period distribution, realized move distribution, win rate, exit-reason mix. This distribution *is* the input to instrument choice (a 2-bar median hold and a 27-bar median hold want different tools).
- Flag and quantify coverage gaps: tickers with no `Data/shared/bars` history, trades outside Alpaca's option-history window (pre-2024-02).

**Gate G0:** ≥80% of trades in each module's ledger must map to a ticker with a discoverable option chain in the relevant expiry window. Modules failing this are reported as "shares-only by data availability," which is itself a finding.

> **G0 RESULT (2026-07-25): PASS for all modules** — 98–100% chain availability, measured by direct
> Alpaca `/v2/options/contracts` discovery. Zero trades predate the 2024-01-18 history floor.
>
> **Measurement caveat, learned the hard way:** G0 must be measured by *contract discovery*, never by
> membership in the 757-symbol Schwab dealer universe. That set is a liquidity screen and using it as an
> optionability proxy understates availability by 40–80pp, initially producing four false FAILs. Chain
> existence is not the binding constraint; **tradable liquidity at a given strike/expiry is**, and that is
> a per-contract question for `liquidity.py` in Phase 3, reported as executable fraction — not a
> universe-membership test.

---

### Phase 1 — Historical options data layer

**New package:** `research/options_lab/`

| File | Responsibility |
|---|---|
| `chain_cache.py` | Alpaca client: expired-contract discovery, bar fetch (1Day/30Min/1Min), trade prints; on-disk parquet cache under `Data/options_history/{ticker}/{expiry}.parquet`; rate-limit + retry (reuse the 429-backoff pattern from `strategies/dealer_positioning/schwab_adapter.py`) |
| `pricing.py` | Black-Scholes + Bjerksund-Stensland (American), analytic greeks, **IV solver** (Brent) inverting observed price → IV |
| `surface.py` | Per-(ticker, date) surface built from real chain bars: ATM IV, term slope, skew fit, IV rank/percentile over trailing window, IV/RV premium vs Yang-Zhang realized vol from `Data/shared/bars` |
| `fills.py` | Spread/slippage model (§6) + commission ($0.65/contract default, configurable) |
| `liquidity.py` | Per-contract tradability gate from real OI/volume/trade-count; multi-leg gate = worst leg |

**Validation (Gate G1) — must pass before Phase 3 results are quotable:**
1. **Cross-source IV check:** for the 16 days where Schwab snapshots exist, compare our Alpaca-implied IV against Schwab's reported `call_iv`/`put_iv` per (symbol, strike, expiry). Report median absolute vol-point error by moneyness × DTE bucket.
2. **Real-fill check:** reprice the 575 real live option trades in `Data/analysis/multi_ticker_swing_live/paired_option_trades.csv` at their real entry and exit timestamps from cached bars. Compare to actual fills. Report error distribution in % of premium, and simulated-vs-actual PnL bias.
3. **Acceptance:** no systematic (signed) bias in the naked-long-call baseline greater than the modeled spread cost. A failure here means the fill model is wrong, not the data — retune §6 and re-run.

Risk-free rate from `signals/meta_context/data/processed/fmp_treasury_rates.parquet`; earnings dates from `signals/events/data/processed/earnings_dates.parquet` (needed for the earnings-proximity feature and for IV-crush attribution).

---

### Phase 2 — Strategy library

**File:** `research/options_lab/strategies.py` — each strategy is a composable leg-set exposing entry cost/credit, max loss, max gain, breakevens, entry greeks, buying-power requirement, and assignment risk.

**Baselines (mandatory in every comparison):**
- **B0 — long shares**, $5,000 target notional (matches current live sizing).
- **B1 — naked long call/put**, using each module's actual live DTE and moneyness convention. *This is the incumbent; every result is reported as lift over B1.*

**Directional, defined-risk:**
- Vertical debit spread (grid over width and long-leg delta)
- Vertical credit spread (opposite side — the "I want theta on my side" tool)
- Broken-wing butterfly

**Stock replacement / capital efficiency:**
- Deep ITM call/put (delta 0.70–0.85)
- Extended-DTE long option (45–90 DTE)

**Hedged / income overlays on shares:**
- Covered call, collar, cash-secured put

**Volatility-structure (aimed at intraday_structure and dealer-positioning theses where magnitude is the claim and direction is weaker):**
- Straddle / strangle
- Calendar and diagonal
- Ratio backspread

**Parameter grid per strategy:** DTE bucket {0–2, 3–7, 8–21, 22–45, 46–90} × entry delta/moneyness × width × exit overlay.

**Exit translation — critical design point.** Each underlying module already owns an exit (tp / sl / time / signal flip). The option position is exited at the **same underlying event**, valued from the option's real bar at that timestamp. Option-native exits (premium stop, profit target %, theta/DTE roll, delta re-center) are layered as **separate overlays** so that improvement is attributable to *instrument* vs *exit policy* independently. Conflating the two is how the existing `option_exit_policy_grid` results become hard to interpret.

---

### Phase 3 — Counterfactual sweep

Replay every trade in the Phase 0 spine through every Phase 2 strategy variant, pricing from real option bars at the module's native granularity (4H modules → 30Min bars aggregated; 30m swing → 30Min bars; intraday → 1Min).

**No signal re-tuning.** The entry/exit decisions are frozen from the existing backtests. Only the instrument changes. This isolates the routing question from signal quality.

**Metrics per (module × strategy × parameter cell):**
net PnL, expectancy, PnL per $ risked, PnL per $ buying power, win rate, trade-series Sharpe, sequential-equity max drawdown, worst-1%/5% tail, commission+spread drag as % of gross, and **executable fraction** (share of trades passing the real-liquidity gate).

**Capital normalization is mandatory.** Report every comparison under two framings: matched notional (same underlying exposure) and matched max-loss (same dollars at risk). Raw PnL comparisons between a $5,000 share position and a $400 spread are meaningless and will not appear in the report.

**Regime slices:** realized-vol bucket, IV rank, IV/RV premium, dealer gamma regime (net GEX sign, where the dealer module covers the name), SPY trend regime, holding period, realized move size in ATR units, earnings-in-window, module score decile, side (long/short — the existing evidence shows calls +$5.6k vs puts −$25k, so side asymmetry is a live hypothesis, not an afterthought).

---

### Phase 4 — The routing policy (primary deliverable)

Build a per-trade dataset where features are **only** what is knowable at `signal_ts`, and the label is the risk-normalized realized PnL of *each* candidate instrument on that trade — a contextual-bandit / multi-label setup, not single-class classification.

**Candidate features:** module, score, ATR%, realized vol (multi-window), current IV, IV rank/percentile, IV/RV premium, term-structure slope, skew, expected move over the module's typical hold, target move implied by tp/sl distance, DTE availability, dealer gamma state, earnings proximity, liquidity tier, time-of-day, market regime.

**Policies compared:**
- (a) always shares
- (b) always naked long option — **the incumbent**
- (c) best single strategy per module
- (d) **interpretable rule set** (2–4 thresholds) — *the preferred deliverable*
- (e) learned router (gradient-boosted, for reference/ceiling only)

We prefer (d). A rule like *"IV rank > 60 and expected hold > 5 bars → debit spread; expected move > 1.5 ATR and IV rank < 40 → naked long; otherwise shares"* is auditable, live-implementable, and degrades gracefully. (e) exists to tell us how much (d) leaves on the table.

**Statistical discipline** — the confluence-discovery null result (2026-07) is the cautionary precedent:
- **Power analysis first.** Given n per module and per regime cell, state the minimum detectable effect. Cells under the floor get "insufficient power," never a recommendation.
- **Pre-register** the hypothesis list and the test-set budget before touching frozen test data. Check which modules' frozen test windows are still unburned.
- Confidence intervals via **block bootstrap over time** — trades are heavily time-clustered; iid bootstrap would overstate significance.
- Fit on train, select on validation, report once on test.

---

### Phase 5 — Hindsight replay on real trades

Kept separate from Phase 3 because it uses **real fills** — highest fidelity, smallest sample, and in-sample by construction.

**Input:** `paired_option_trades.csv` (575 closed real option trades with option marks, underlying marks, MFE/MAE), `alpaca_fills_normalized.csv`, plus each module's live audit and closed-trade ledgers.

**Method:** for each real trade, take the real entry timestamp and real underlying path, then price the counterfactual structures from the cached real option bars for the same expiries. Where the trade falls inside the 16-day Schwab snapshot window, cross-check against the snapshot chain.

**Anchoring advantage:** we know one exact point on the real surface for every trade — the contract we actually filled. Any counterfactual structure on the same underlying/expiry is priced relative to a verified anchor, which tightly bounds the error.

**Output:** "what we made" vs "what each alternative structure would have made," per module and per side, on identical entries and exits.

**Labeling:** this is explicitly **hindsight, in-sample, hypothesis-generating**. It does not establish forward edge. Its job is to propose rules for Phase 4 to test on frozen data.

---

### Phase 6 — Reporting

- `research/options_experiment/REPORT.md` — findings, decision table, caveats.
- One **instrument routing card** per module: the rule, the expected lift, the conditions under which it does *not* apply, and the liquidity coverage.
- Plots via `core/shared_plotting` conventions: routing-policy equity curves, strategy×regime edge heatmaps, IV-surface validation residuals, payoff-at-entry vs realized-outcome scatter. All plots labeled in-sample / validation / frozen-test / live-real.
- Append to `LIVING_SUMMARY.md` per repo convention.

---

## 4. Success criteria

The experiment succeeds if it produces **either**:
1. A validated routing rule that beats the always-naked-long incumbent on frozen test data, with stated conditions and coverage; **or**
2. A well-powered null result — "naked long premium is not systematically improvable by structure within our signal set and liquidity constraints" — which is an equally publishable, decision-useful outcome and would settle the question.

It fails if it produces a leaderboard of average returns with no conditioning, or a rule fit on data that was also used to select it.

---

## 5. Non-goals

- No changes to live trading code or live routing in this work. Research only.
- No signal/model retuning. Entries and exits stay frozen.
- No new broker integrations beyond read-only historical data pulls.

---

## 6. Known risks and limitations

1. **No historical bid/ask.** Alpaca serves bars and trade prints but not historical quotes. Fill realism is modeled: spread estimated per contract from (a) the intrabar high/low and VWAP-vs-close dispersion, (b) real trade-print clustering, (c) a spread-vs-(moneyness, DTE, OI, ADV) curve calibrated on the 16 days of Schwab snapshots where true bid/ask is known. **Multi-leg structures pay this cost N times**, so a 4-leg structure must clear a materially higher bar than a 1-leg position. All results are reported at three spread assumptions (optimistic mid / calibrated / pessimistic full-spread), and any conclusion that does not survive the pessimistic case is not a conclusion.
2. **Bar-granularity path error.** Exits are evaluated on bar closes for the primary result. Intrabar-touch variants (stops/trails) are reported separately and labeled optimistic.
3. **Thin chains.** Much of the 750-ticker universe has poor option liquidity — the live system already proved this. Expect a large share of momentum/HTF names to be shares-only. That is a legitimate finding, and the report must state what fraction of each module's flow is actually optionable.
4. **Short-premium strategies are systematically under-penalized by backtests** — assignment, gap risk, margin expansion, and pin risk are not fully captured. Every short-leg strategy must report tail loss and buying-power usage explicitly, and no short-premium recommendation ships without an explicit tail-risk statement.
5. **Sample size.** Dealer Ranker has only 15 days of rankings; intraday_structure is paper-only and new. These modules will likely land in "insufficient power" and should not be forced to a recommendation.
6. **Options history starts 2024-01-18.** Any backtest trade before that cannot be routed. Quantified in Phase 0.

8. **IV inversion is ill-conditioned for deep ITM/OTM at very short T** (Phase 1a/1b finding). Where vega < ~$0.05/vol-point — roughly 23% of a moneyness×DTE×vol grid — the recovered IV is self-consistent (reprices the input to 1e-4) but is not identifiable as *the* generating vol. Consequence: IV rank, term slope, and skew must be computed from **near-ATM contracts only**, where vega is largest. Any routing feature derived from a low-vega contract's IV is unreliable and must be excluded, not smoothed over.

9. **Treasury curve coverage currently ends 2026-07-06** (`fmp_treasury_rates.parquet`, nightly-job lag). This does not affect Phase 3 (backtest window ends 2026-06) but **does** block Phase 5 hindsight on live July-2026 trades. The rate helper raises rather than extrapolating, so this will fail loudly rather than silently. Needs a refresh of the treasury fetch job before Phase 5.
7. **Data volume.** Caching chains for 750 tickers × 14 months at 30Min granularity is large. Phase 1 fetches lazily — only the (ticker, expiry, window) tuples the spine actually needs — and caches to disk. Cost is API time, not money.

---

## 7. Execution order

Phase 0 and Phase 1 are independent and run in parallel. Phase 2 depends on Phase 1's pricing/liquidity primitives. Phase 3 depends on 0+1+2. Phase 4 depends on 3. Phase 5 depends on 1 only and can run alongside 3.

Gates G0 and G1 are hard: no Phase 3 numbers are quoted until G1 passes.
