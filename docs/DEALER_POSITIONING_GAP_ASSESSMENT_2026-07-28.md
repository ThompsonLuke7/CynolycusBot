# Dealer Positioning and Auction-Playbook Gap Assessment

Date: 2026-07-28

## Executive conclusion

The repository already contains most of the **software skeleton** described in the supplied
“translation layer” response. It does not yet contain most of the **historical evidence needed to
calibrate and validate that skeleton**.

The proposed Auction Playbook Engine substantially corresponds to
`strategies/intraday_structure/`, built on top of `strategies/dealer_positioning/`. The repo already
has candidate routing, causal one-minute detectors, persistent setup state, dealer-derived levels,
targets/invalidation, replay, paper monitoring, current-chain option selection, and nightly dealer
snapshot/ranking automation.

The shortest honest description is:

> Architecture: roughly 70–80% present.  
> Reproducible research data and validation: roughly 25–35% present.  
> Credible option-level backtesting: not present.

Those percentages are an engineering estimate, not a performance claim.

## What already exists

### Dealer-positioning pipeline

- Schwab current-chain ingestion and parsing of OI, volume, IV, delta, gamma, and vega.
- OI/Greeks-derived call/put GEX and VEX strike ladders.
- Call wall, put wall, magnets, vega walls, air-gap scores, and a gamma sign-crossing level.
- Three expiration scopes: `daily_week`, `through_month`, and `two_months`.
- Broad-universe daily capture with liquidity filtering and immutable dated parquet artifacts.
- Snapshot-over-snapshot map changes, cross-sectional ranks, and a dealer swing-potential ranker.
- A deterministic, opt-in structural gate consumed by the 30-minute swing option path.
- A separate dealer-ranked option experiment with ATM contract selection, two-sided quote checks,
  OI/volume gates, paper/live separation, audit logging, and shared position management.
- Dealer dashboards, level-history plots, and combined-server integration.

Observed local inventory on 2026-07-28:

- 16 dated snapshot directories, of which 14 contain data.
- Nonempty history spans 2026-07-02 through 2026-07-28.
- Recent captures cover roughly 700–730 symbols and about 84,000 strike rows per populated day.
- Ranking history contains 9,755 rows across 762 symbols.
- Level-dynamics summary and strike-level parquet artifacts exist.

### Auction/translation layer

`strategies/intraday_structure/` already supplies the layer that the pasted response said was
missing:

- Candidate adapters for Momentum, HTF, Meta, Dealer Ranker, 30-minute swing, a high-liquidity
  universe, and a manual watchlist.
- Persistent per-ticker/per-direction/per-playbook state.
- One-minute causal features and one-bar entry delay.
- V reversal, breakout continuation, VWAP reclaim, structural rejection, trend pullback, and
  exhaustion playbooks.
- Explicit setup state, evidence, invalidation, targets, target extension, runway score, and
  warnings.
- Session VWAP, candidate-anchored VWAP, prior-day high/low/close, premarket high/low, opening
  range, intraday swing levels, liquidity zones, a rolling volume-profile HVN/LVN, round numbers,
  and dealer levels.
- Synchronized SPY/QQQ/VIXY and optional sector context.
- A provider interface for live option-flow prints and faster quote/trade price updates.
- Deterministic chronological replay, causal labels, conservative same-bar collision handling,
  costs, and fixed-input ablations.
- Paper-only dashboard, restart recovery, duplicate suppression, and append-only transitions.

The live transition log is not empty: it contains 7,139 transitions from 2026-07-21 through
2026-07-28, including 330 confirmations, 329 running transitions, 76 target hits, and 31 extensions.
This demonstrates operability, not profitability.

### Existing underlying and option plumbing

- Strong SPY-specific one-minute underlying history.
- Broad 5/10/30-minute and 1h/4h/daily caches used elsewhere in the repo.
- Live one-minute bars through the shared Alpaca stream.
- Live/current option snapshots and two-sided quote retrieval for selected contracts.
- 575 historical real option round trips that remain valid for fill/cost analysis.
- Forward SPY option bid/ask mark capture was added separately on 2026-07-28.

## What remains to build or harden

### 1. Complete the auction-context feature set

The current engine has session and microstructure features, but it does not fully encode the
specific auction language in the supplied response.

Still needed:

- Explicit multi-day balance/trend classification.
- Overnight inventory and overnight high/low behavior distinct from generic premarket extrema.
- Prior-session value-area high/low and POC from a defined volume-profile methodology.
- Equal-high/equal-low clustering as a first-class liquidity feature.
- A dedicated sweep/reclaim detector with configurable reclaim window.
- Acceptance/rejection statistics: time below/above, number of closes, volume beyond the level,
  excursion in ATR, value migration, reclaim speed, and retest behavior.
- A playbook-router output contract with explicit `context_regime`, `active_level`, `setup_type`,
  `trigger_state`, `invalidation_level`, `target_sequence`, `expected_time_to_target`,
  `preferred_instrument`, and `no_trade_reason`.

This is mostly incremental work on the existing engine, not a rewrite.

### 2. Make dealer semantics and calculations research-grade

Current GEX is a proxy:

- Calls are assigned positive exposure and puts negative exposure.
- Total OI is treated as if its dealer-side sign were known.
- The current `gamma_flip` is derived from strike-ladder net-GEX sign changes/cumulative changes.
  It is not a full hypothetical-spot sweep that reprices gamma and finds zero aggregate exposure.
- Cross-sectional rankings mix several heuristic transformations whose predictive value is not yet
  established.

Needed:

- Version and name the current calculation explicitly as an OI-sign-assumption proxy.
- Add alternative sign conventions and a spot-sweep zero-gamma implementation as separate,
  preregistered variants.
- Normalize scale-sensitive quantities before cross-symbol comparison.
- Record vendor field availability, chain query time, underlying quote time, OI date, and calculation
  version in every artifact.
- Test stability across expiration scopes, capture times, symbols, and reasonable formula variants.

No formula variant should be selected on the final test period.

### 3. Persist replayable point-in-time market data

The live engine persists state and transitions, but not a canonical immutable broad one-minute event
store. Its bounded recent histories cannot recreate prior sessions after they roll out.

Needed:

- Append-only candidate-scoped one-minute OHLCV, including extended hours when overnight playbooks
  are evaluated.
- SPY, QQQ, VIXY, sector ETF, and eventually breadth observations aligned to every decision.
- Event time, arrival/availability time, and capture time.
- Data-quality reports for missing bars, duplicates, session gaps, late bars, halts, and feed tier.
- A daily manifest tying candidates, bars, dealer snapshot version, configuration, and code version
  together.

This is the most important connection needed for automatic reproducibility.

### 4. Capture dealer maps at decision-relevant times

Current broad snapshots are usually captured around 15:45 ET and/or after the close. They can seed
the following session but cannot causally explain a same-day morning pivot.

Needed:

- At minimum: previous close, pre-open, open/09:45, midday, and near-close captures for a bounded
  liquid universe.
- Prefer event-triggered refresh for the small active candidate set.
- Preserve every capture rather than keeping only one symbol/scope row per date.
- Track when OI itself was last updated; intraday movement in an OI-derived map often comes from
  spot/Greeks/chain changes, not newly published OI.
- Independently test whether intraday map changes are stable enough to use.

### 5. Produce a real setup/trade ledger

The transition log describes state-machine lifecycle events. It is not a closed-trade ledger.

Needed:

- One immutable record per confirmed setup containing signal inputs, entry intent, modeled or paper
  fill, size, invalidation, targets, exit, MFE/MAE, costs, and outcome.
- Separate underlying-policy results from option-instrument results.
- Reconciliation from candidate to transition to fill to close.
- Duplicate/restart invariants and daily completeness checks.

Until this exists, the module cannot enter the common experiment spine or support automated daily
performance attribution.

### 6. Build and validate the option translation layer

The repo can choose a current ATM option, but it does not yet estimate whether a particular contract
has positive expected value for a particular playbook.

Needed:

- Contract-policy features: DTE, delta/moneyness, IV, implied move hurdle, spread, depth/size if
  available, OI, volume, and time-to-expiry.
- Setup-specific outcomes: target-before-stop-before-expiry, time to target, MFE/MAE, realized
  volatility, and chop/continuation probability.
- A reject-by-default policy when expected underlying movement/timing does not clear implied move
  and execution cost.
- Underlying-structure-based stops with option-price risk caps as a separate safety layer.
- Quote-based paper fills and forward mark capture for every considered/selected contract, not only
  filled contracts.

Historical Alpaca option trade bars are not fit for this purpose. The repo's own retraction found
45.9% identical entry/exit marks and only +0.093 correlation between underlying and supposed option
returns. Credible historical option testing therefore requires purchased NBBO/quote history or
forward collection.

### 7. Add live flow only after the basic pipeline works

The type/interface exists, but no production provider feeds `OptionFlowPrint`.

If pursued, ingest immutable OPRA/vendor records containing contract, exchange/vendor timestamp,
price, size, contemporaneous bid/ask, IV/Greeks or inputs to reproduce them, correction/cancel state,
and multi-leg classification. Validate coverage and clock skew before using “sweep” or
bid/ask-side labels. This is useful, but it is not required for the first price-plus-static-dealer
playbook validation.

## Data: have versus need

| Data | Already present | Still needed |
|---|---|---|
| Underlying 1m | Live shared stream; strong SPY history | Broad candidate-scoped immutable history, extended hours, arrival times |
| Higher-timeframe context | Broad 30m/1h/4h/daily caches and existing rankers | Explicit multi-day balance/auction-regime outputs |
| Session levels | VWAP, opening range, prior H/L/close, premarket H/L, rolling profile | Prior VAH/VAL/POC, overnight inventory, equal highs/lows, acceptance metrics |
| Dealer map | 14 populated daily broad snapshots; OI/Greeks ladders and level dynamics | Multiple decision-time captures, calculation/version metadata, formula variants |
| Dealer inventory | None; only inferred proxy | Cannot be directly obtained from standard chain data; keep assumption-based variants |
| Market context | SPY/QQQ/VIXY and optional sector | Synchronized one-minute breadth and better sector coverage |
| Option contract metadata | Current contracts, OI, volume, IV/Greeks, bid/ask | Point-in-time history for every considered contract |
| Option execution | Current quotes and 575 real fills; SPY forward marks starting now | Broad forward NBBO marks or purchased historical NBBO; realistic fill/rejection data |
| Live options flow | Provider-neutral interface only | Actual OPRA/vendor feed plus correction, quote, and multi-leg normalization |
| Outcomes | Transition states and causal label code | Closed setup ledger, frozen validation/test cohorts, option-level outcomes |

## Recommended implementation sequence and estimate

Assumption: one experienced engineer working mostly full-time, reusing current architecture.
Calendar time is longer than coding time because forward evidence must accumulate.

### Phase A — Reproducibility spine and collectors

Engineering: **1–2 weeks**

- Candidate-scoped immutable one-minute/extended-hours capture.
- Multi-time dealer capture for bounded symbols.
- Dataset/config/code manifests and daily quality checks.
- Closed setup/trade ledger and reconciliation.

Then collect at least **6–8 trading weeks** before threshold selection. Three to six months is better
for regime diversity.

### Phase B — Complete auction features and router

Engineering: **1–2 weeks**

- Balance/overnight/value-area/equal-level features.
- Sweep/reclaim and quantitative acceptance/rejection.
- Unified playbook-router schema and no-trade reasons.

This can be built while Phase A data accumulates.

### Phase C — Frozen replay and dealer ablation

Engineering: **1–2 weeks** after sufficient data exists

- Price-only baseline.
- Price plus session/auction structure.
- Price plus static dealer map.
- Dealer-map-change and market/sector variants.
- Walk-forward or fixed validation/test protocol with setup/regime/time-of-day reporting.

Do not add ML until deterministic cohorts have enough events and stable lift.

### Phase D — Options policy

Engineering: **2–4 weeks**, plus data acquisition

- Candidate-contract quote/mark collector.
- Contract eligibility and implied-move hurdle.
- Setup-specific contract policy and quote-based paper simulation.
- Fill/mark reconciliation and underlying-versus-option sanity checks.

Data calendar:

- **Purchased historical one-minute NBBO:** evaluation can begin as soon as the data is normalized.
- **Forward-only capture:** expect at least **8–12 weeks** for a first narrow SPY/QQQ/liquid-name
  read and **3–6 months** for a more credible result.

### Phase E — Live flow and execution approval

Engineering: **2–4 additional weeks**, excluding vendor onboarding and paper observation

- OPRA/vendor normalization and quality gates.
- Flow ablation.
- Only after positive untouched-test and paper evidence: a separately approved execution seam.

### Practical total

- Reproducible underlying/dealer **paper research system**: about **3–5 engineering weeks**, with
  useful conclusions gated by **6–12 weeks of new data**.
- Credible automatic **option selection and paper evaluation**: about **6–10 engineering weeks**
  total, with either paid historical NBBO or **3–6 months of forward quote collection**.
- Live-capital readiness cannot be estimated honestly until the frozen replay and paper gates pass.

## Recommended immediate roadmap

1. Treat `intraday_structure` as the canonical playbook router; do not create another parallel
   engine.
2. Build the immutable replay dataset and real setup ledger first.
3. Add the missing auction-context features while data accumulates.
4. Freeze price-only versus price-plus-dealer ablations.
5. Keep dealer structure as a context/target modifier unless untouched evidence supports more.
6. Restrict initial option research to SPY, QQQ, and a small liquid-name set with captured NBBO.
7. Keep all execution paper-only.

## Verification performed

- Read the dealer pipeline, capture/ranking/dynamics, gate, option runner, intraday engine,
  detectors, replay, labels, options adapters, documentation, automation, and current artifacts.
- Counted local snapshot/ranking/transition coverage.
- Confirmed the local broad one-minute historical gap and the absence of a closed intraday setup
  ledger.
- Ran `pytest strategies/dealer_positioning/tests strategies/intraday_structure/tests -q`:
  **91 passed**.

