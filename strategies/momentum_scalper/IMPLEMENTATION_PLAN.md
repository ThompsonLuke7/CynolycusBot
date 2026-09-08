# Momentum Scalper Completion Plan

Status: proposed implementation plan; no trading authorization  
Strategy boundary: standalone Ross-style premarket, low-float momentum strategy  
Initial route: long common shares, shadow mode and paper trading only  

## Implementation status — 2026-08-31

Implemented in code: v1 paper-only configuration; causal scanner contracts for
prior close, time-matched RVOL, float, catalyst availability, quotes, daily
resistance and halts; causal setup detectors; one-portfolio quote-aware replay;
fill-aware partial/exit state; source-fitness report; shadow ledger/session;
and a paper-order bridge to the shared idempotent execution contract.

Not complete—and intentionally not claimed complete: licensed/entitled SIP and
historical data validation, point-in-time reference/catalyst backfill, an
out-of-sample study, shadow-session parity evidence, broker-filled paper results,
and the required 6–8-week frozen paper evaluation. The shared gateway remains
paper-only and no live broker write path was added.

## 1. Decision Summary

The momentum scalper remains a separate strategy. It may reuse neutral repository
services—market/news/halt ingestion, point-in-time reference data, broker contracts,
execution journaling, reconciliation, monitoring, and kill switches—but it must own
its scanner, setup state, entries, exits, portfolio, risk limits, research results,
and promotion gates.

The current package is an MVP scaffold, not a system that can be promoted. The
completion sequence is:

1. Freeze the strategy specification and data contracts.
2. Prove that the available feeds can observe the setup and support execution.
3. Rebuild the premarket scanner and catalyst/reference joins point-in-time.
4. Implement explicit, causal pattern state machines.
5. Implement portfolio, risk, exits, and broker-independent order intents.
6. Replace row-by-row replay with a chronological event-driven simulator.
7. Run deterministic research and preregistered ablations.
8. Run live shadow capture and verify research/live parity.
9. Run frozen paper trading through the shared execution gateway.
10. Consider a tightly controlled live pilot only after every promotion gate passes.

Machine learning is not on the critical path. A model may later rank already-valid
setups, but it must not define or repair the core strategy.

## 2. Scope and Non-Goals

### In scope for v1

- US exchange-listed common shares; long-only.
- Premarket discovery beginning at 04:00 America/New_York.
- Low-priced, low-float gap-and-go runners with a time-valid material catalyst.
- True prior-close gap, time-of-day relative volume, liquidity, spread, float,
  daily resistance, former-runner context, and halt/resumption state.
- HOD/premarket-high breakout, first pullback, bull flag, and flat-top breakout.
- Share execution with extended-hours-aware limit orders.
- Structure-defined stops, partial profit-taking, breakeven management, first-red-
  candle/failed-breakout exits, time exits, and emergency liquidation policy.
- One chronological portfolio shared by replay and live policy code.
- Shadow, paper, and later live modes with separate configuration and storage.

### Explicitly out of scope for v1

- Short selling, locates, or borrow logic.
- Options routing.
- Autonomous trades based only on social attention or a news classifier.
- Level 2 order-book prediction unless a licensed historical and live feed is obtained.
- A black-box model that overrides hard eligibility, risk, or stale-data gates.
- Reusing intraday-structure entry, exit, or setup decisions.

## 3. Current-State Blockers

| Area | Current behavior | Required disposition |
|---|---|---|
| Historical scanner | Loads only the selected day, so prior close falls back to the first premarket print | Load the previous eligible session and corporate-action state explicitly |
| RVOL | Divides cumulative premarket volume by the mean of the same partial session | Use cumulative time-of-day volume versus prior comparable sessions |
| Float | Metadata is not loaded by the daily builder; missing float passes | Point-in-time snapshot join; missing required float fails closed |
| News | A day-level flag has no publication/observation cutoff | Join only records available at or before the decision timestamp |
| Halts | Downloaded but not joined; labels hard-code no halt | Ingest halt/resume/status events into scanner, replay, and execution state |
| Spread | One-minute high-low range is called spread | Use bid/ask quotes from the entitled live/historical feed |
| Patterns | Stateless one-row approximations include the trigger bar in resistance | Implement multi-event state machines with prior-only levels |
| Entry price | Feature rows omit the `close` required by entry policy | Carry an immutable quote/trigger snapshot into every decision |
| Exit | Percentage trail is mislabeled ATR and cannot run after the hard target | Implement fill-aware partial and remainder state transitions |
| Replay | Re-enters the same ticker repeatedly and assumes optimistic bar fills | Use one event clock, one account, conservative execution, and order lifecycle |
| Training | One 80/20 row split with no session grouping or embargo | Defer ML; later use date-grouped walk-forward OOS evaluation |
| Live execution | Emits intents only | Connect paper intents to the shared idempotent gateway after safety extensions |

Existing package-local data directories are empty. New immutable artifacts should
follow repository conventions under `Data/raw`, `Data/processed`, `Data/models`, and
`Data/inference/live_runs`; do not overwrite any historical artifact.

## 4. Frozen v1 Strategy Specification

The first task is to turn the public strategy description into a versioned,
machine-testable specification. Defaults below are an initial baseline to
preregister, not claims of optimality.

### 4.1 Eligible security

- Active, tradable US common share on NASDAQ, NYSE, or AMEX.
- Exclude ETFs, warrants, rights, units, preferreds, OTC securities, inactive
  symbols, and symbols the broker marks non-tradable.
- Price baseline: $1–$20 at the decision time.
- Float baseline: at most 10 million shares for the strict study; preregister
  separate 20 million, 50 million, and 100 million ablation buckets.
- Missing float, prior close, corporate-action state, tradability, or quote means
  ineligible; never substitute a current value into historical decisions.

### 4.2 Premarket runner eligibility

- Session begins at 04:00 ET and uses the official trading calendar.
- Gap from the previous eligible session's adjusted close is at least 10%.
- Cumulative premarket volume is at least 250,000 shares.
- Time-of-day cumulative RVOL is at least 5x its historical expectation.
- There is a material catalyst available before the decision, unless a separately
  preregistered no-news ablation is being evaluated.
- The symbol has a valid two-sided quote, acceptable spread, recent trades, and
  enough displayed/realized liquidity for the proposed size.
- The daily chart has no disqualifying nearby resistance under the frozen rule.
- Scanner snapshots record every pass/fail reason, not only selected names.

### 4.3 Candidate ranking

Ranking happens only after hard eligibility. The deterministic baseline should
rank, with frozen transforms, by:

1. Catalyst materiality and recency.
2. Gap and current price acceleration.
3. Time-of-day RVOL and cumulative premarket dollar volume.
4. Lower point-in-time float / float rotation.
5. Distance to valid HOD or consolidation trigger.
6. Spread, quote depth proxy, trade frequency, and adverse liquidity penalty.
7. Daily resistance and former-runner evidence.

Do not normalize a row by values computed later in the day. Persist the component
scores, source observations, final rank, and configuration version.

### 4.4 Supported setup state machines

Each candidate owns a state that advances only on newly received market events:

`DISCOVERED -> IMPULSE -> PULLBACK/CONSOLIDATION -> ARMED -> TRIGGERED -> ORDER_PENDING -> IN_POSITION -> EXIT_PENDING -> CLOSED`

Any state can transition to `INVALIDATED`, `HALTED`, `DATA_STALE`, or `SESSION_END`.

Implement setups independently:

- **Premarket-high/HOD breakout:** resistance is fixed from observations strictly
  before the trigger; require renewed volume and a trade/quote-confirmed break.
- **First pullback:** count completed impulse/pullback cycles; require the first
  orderly pullback to hold its defined support and reclaim its trigger.
- **Bull flag:** define pole, retracement limit, consolidation duration/range,
  declining pullback volume, and breakout confirmation.
- **Flat top:** require repeated prior rejections within a price tolerance, rising
  support, and a new break above the prior-only level.

Every setup must define arming, trigger, invalidation, maximum chase, reset, and
expiry rules. A single bar cannot both establish resistance and break it.

### 4.5 Entry and order behavior

- Long shares only.
- Premarket orders are limit orders with an explicit expiration and extended-hours
  eligibility; no market or bracket assumption before regular hours.
- Limit price is derived from a fresh quote and capped by the maximum allowed
  spread/slippage. It is never derived from a one-minute close.
- If the trigger moves beyond the chase limit before acknowledgement/fill, cancel
  or refuse according to the frozen order policy.
- No second entry while an order or position for that symbol is unresolved.
- A broker acknowledgement is not ownership; position state starts from confirmed
  fills and handles partial fills explicitly.

### 4.6 Risk and exits

- Position size = configured risk budget divided by structure-stop distance, then
  capped by buying power, symbol exposure, quote/liquidity limits, and portfolio
  limits. Risk amounts remain configurable and account-independent.
- Initial stop is based on setup invalidation, not a fixed percentage alone.
- Never average down in v1.
- Support a frozen partial-profit schedule; after a confirmed partial, manage only
  the confirmed remaining quantity and move the stop only when the rule says so.
- Supported remainder exits: breakeven/structure stop, first red candle after the
  specified extension, failed breakout, loss of momentum/volume, trailing support,
  maximum hold, and session cutoff.
- Halted positions remain owned and visible; never manufacture a fill during a halt.
- Data disconnect or stale quote blocks new entries. Existing positions invoke the
  separately tested degraded-risk procedure.
- Portfolio fuses: maximum concurrent positions, trades per day, consecutive losses,
  gross exposure, symbol exposure, realized daily loss, total daily risk, stale feed,
  reconciliation failure, and manual kill switch.

Because extended-hours bracket orders are not supported by Alpaca, premarket stop
protection must be an actively monitored, fill-aware exit using eligible limit
orders. This operational risk must be exercised in paper and disconnect tests.

## 5. Data Contracts and Source-Fitness Gate

No downstream build begins until a standalone source-fitness report passes.

### 5.1 Required immutable observations

- `TradeEvent`: symbol, exchange, price, size, conditions, event time, received time,
  sequence/ID, feed, entitlement.
- `QuoteEvent`: bid/ask price and size, venues, conditions, event time, received time,
  feed, entitlement.
- `BarEvent`: interval, OHLCV, trade count, VWAP, event time, availability time.
- `TradingStatusEvent`: halt/LULD/resume reason, event time, received time.
- `ReferenceSnapshot`: symbol identity, security type, exchange, tradability, float,
  shares, corporate actions, source time, observation time.
- `CatalystRecord`: headline/event, company mapping, materiality, direction where
  explicitly supported, publication time, first-observed time, source, content hash.
- `ScannerSnapshot`: all raw inputs, derived values, pass/fail reasons, rank, config.
- `SetupTransition`: prior/new state, event that caused it, levels, reason, config.
- `DecisionIntent`: snapshot IDs, setup state, trigger/stop/targets, risk decision.
- `Order/FillEvent`: deterministic identity, submission/ack/fill times, quantities,
  prices, rejects, replacements, cancellations, and reconciliation evidence.

### 5.2 Feed validation

For the intended subscription/account, measure and record:

- Whether the live feed is SIP or IEX; a successful response is not proof of SIP.
- Premarket trade and quote coverage on representative low-float runners.
- Coverage against an independent aggregate for volume, high/low, and halts.
- Quote freshness, crossed/locked frequency, missing intervals, duplicates,
  out-of-order events, corrections, queue drops, reconnect gaps, and event-to-receipt
  latency.
- Whether all required symbols/channels can be subscribed concurrently.
- Historical availability for trades, quotes, conditions, statuses, and corrections.
- Broker paper eligibility for each symbol and extended-hours limit-order behavior.

SIP—not single-exchange IEX—is the minimum acceptable live research feed for this
cross-market scanner unless a controlled study proves the reduced feed is adequate.
If historical NBBO/trade data cannot be obtained, scanner recall may be studied with
bars, but execution P&L must remain explicitly unvalidated rather than synthesized.

### 5.3 Point-in-time data rules

- Previous close comes from the preceding eligible session after applying only
  corporate actions known at the decision time.
- RVOL baseline uses only earlier sessions, matched by elapsed session time.
- Float/reference snapshots use the most recent observation available at that time;
  store age and source. No current float may be backfilled into old decisions.
- News/catalysts join on `available_at <= decision_at`, never on calendar date alone.
- Halts and resumes change state only when their observation reaches the system.
- Raw data is immutable; repairs create a new transformation/version and manifest.

## 6. Target Package Layout

Keep the strategy under `strategies/momentum_scalper`, but replace the coupled MVP
flow with narrow components:

```text
strategies/momentum_scalper/
  README.md
  IMPLEMENTATION_PLAN.md
  config/
    schema.py
    momentum_scalper_v1.json
  contracts.py
  data/
    adapters.py
    validation.py
    reference.py
    catalysts.py
    halts.py
  scanner/
    rvol.py
    daily_context.py
    premarket.py
    ranking.py
  patterns/
    base.py
    hod_breakout.py
    first_pullback.py
    bull_flag.py
    flat_top.py
  portfolio/
    sizing.py
    risk.py
    positions.py
  execution/
    entry_policy.py
    exit_policy.py
    order_manager.py
  replay/
    event_engine.py
    fill_models.py
    reports.py
  live/
    runner.py
    shadow.py
    health.py
  tests/
```

The strategy runner emits standard decision/order contracts to the shared nervous
system. The shared system does not import momentum-specific scanner or pattern code.
The existing shared `OrderRequest` and Alpaca adapter need an audited
`extended_hours` field and validation; today the adapter does not transmit it.

## 7. Phased Work Plan and Exit Gates

### Phase 0 — Strategy charter and experiment registration

Deliverables:

- `momentum_scalper_v1.json` with every eligibility, setup, exit, session, and risk
  parameter; no magic constants in code.
- Strategy rulebook with examples and counterexamples for every transition.
- Primary metrics, ablations, train/validation/test dates, execution scenarios, and
  promotion criteria registered before results are inspected.
- Explicit paper-only default and separate storage namespaces.

Exit gate: every intended decision can be explained from the versioned rulebook;
ambiguous strategy details are resolved before implementation.

### Phase 1 — Source-fitness and entitlement proof

Deliverables:

- Read-only diagnostic capturing SIP trades, quotes, bars, statuses/LULD, and news
  from 04:00–11:00 ET on representative sessions.
- Entitlement, completeness, latency, drop/reconnect, and cross-source comparison
  report.
- Historical-data feasibility report for quote-aware replay, point-in-time float,
  catalysts, delistings, corporate actions, and halts.
- Go/no-go decision for the primary market-data vendor.

Exit gate: the source actually observes early runners and quotes closely enough to
evaluate the intended entries. If it fails, change the data source before coding the
strategy around it.

### Phase 2 — Point-in-time universe and scanner

Deliverables:

- Historical daily eligible-universe snapshots including inactive/delisted symbols.
- Previous-session close and corporate-action service.
- Time-of-day cumulative RVOL baseline and tests around missing sessions/holidays.
- Event-driven premarket scanner with complete pass/decline ledger.
- Correct float, catalyst, quote, spread, daily resistance, and halt joins.
- Historical scanner recall report on a preregistered set of runner days and randomly
  sampled quiet days.

Exit gate: no future input is used; manual reconstruction and automated snapshots
agree; missing critical inputs fail closed; scanner recall/false-positive metrics are
reported without tuning against the final evaluation period.

### Phase 3 — Setup state machines

Deliverables:

- Separate HOD breakout, first-pullback, bull-flag, and flat-top implementations.
- Immutable transition ledger and deterministic replay of identical input events.
- Unit fixtures for valid, invalid, reset, chase, stale-data, halt, and resumption
  paths.
- Annotated playback report showing why every trigger did or did not occur.

Exit gate: research and streaming execution produce byte-equivalent transitions from
the same ordered events; a trigger never uses its own future high/close/volume.

### Phase 4 — Portfolio, risk, and exit engine

Deliverables:

- Fill-aware position state and risk-based share sizing.
- Partial-fill, partial-profit, replacement, cancellation, and remaining-quantity
  logic.
- Strategy-specific portfolio fuses and an operator kill switch.
- Active premarket exit procedure compatible with extended-hours limit-only rules.
- Recovery from restart by reconciling broker positions/orders before accepting data.
- Tests for duplicate callbacks, lost acknowledgements, conflicting positions,
  overfills, stale quotes, disconnects, halts, and session boundaries.

Exit gate: entries fail closed under ambiguity; risk-reducing actions remain possible;
the strategy never believes it owns a quantity the broker has not confirmed.

### Phase 5 — Chronological replay and deterministic baseline

Deliverables:

- One event-driven replay clock for scanner, setups, orders, fills, positions, and
  portfolio limits.
- Conservative fill models: next eligible trade, quote-cross/marketable-limit,
  partial fill, no-fill-at-touch, adverse same-interval ordering, and halt handling.
- Costs, spread, slippage, liquidity/participation caps, rejects, cancellations, and
  delayed execution.
- Named deterministic baseline and preregistered ablations for float, catalyst,
  pattern, time window, and exit components.
- Reports by session, setup, price, float, time of day, catalyst family, spread,
  liquidity, market regime, and runner rank.

Required metrics:

- Net expectancy and total return; win rate; average win/loss; profit factor.
- MFE, MAE, right-tail capture, giveback, time in trade, and time to peak/failure.
- Max drawdown, daily loss distribution, exposure, turnover, and trade count.
- Scanner precision/recall proxies, setup conversion, order fill/cancel/reject rates.
- Results under every conservative fill/slippage scenario and reasonable parameter
  sensitivity.

Exit gate: the untouched out-of-sample period has positive net expectancy under the
preselected conservative execution case; results are not dependent on one ticker,
day, catalyst, or parameter edge. Otherwise revise or stop—do not promote.

### Phase 6 — Live shadow mode

Deliverables:

- Supervised 04:00 ET runner that consumes live events but cannot submit orders.
- Append-only raw-event, scanner, transition, intent, latency, and health ledgers.
- Dashboard/alerts for runners, armed setups, stale inputs, halts, decisions, and
  data/worker health.
- Nightly replay of captured events with live-versus-replay parity diff.
- Measurement of event receipt to scanner, setup, decision, and hypothetical submit
  latency; freeze latency/staleness budgets after an initial measurement period.

Exit gate: at least ten clean trading sessions, no unexplained parity differences,
no material data gaps, and no unresolved safety incident. This is an engineering
gate, not a profitability gate.

### Phase 7 — Frozen paper trading

Deliverables:

- Add `extended_hours` to the shared order contract/hash, Alpaca adapter, fixtures,
  and gateway tests; reject invalid order type/TIF combinations.
- Connect momentum decisions to the paper-only idempotent execution gateway.
- Consume broker trade updates and reconcile orders/positions at startup, on timeout,
  periodically, and before shutdown.
- Record decision-to-submit, submit-to-ack, ack-to-fill, slippage versus decision NBBO,
  partial fills, cancels, rejects, disconnects, and halt behavior.
- Daily automated review comparing expected, simulated, paper, and broker-authoritative
  outcomes.

Study gate:

- Minimum 6–8 weeks under one frozen strategy/configuration.
- At least 100 independent paper entry fills and enough observations in every setup
  being considered for promotion; continue the study if confidence is inconclusive.
- Positive net expectancy after replacing optimistic paper fills with measured,
  conservative live-execution assumptions.
- Bootstrap lower confidence bound for expectancy above zero under the registered
  analysis, acceptable drawdown/daily loss, and stability across time/setup buckets.
- Zero duplicate orders, unmanaged positions, unbounded losses, or unexplained
  broker/local divergence.

Paper fills are not proof of executable live fills. Promotion analysis must penalize
paper results using measured quote and latency behavior.

### Phase 8 — Live-readiness review and limited pilot

The current shared gateway deliberately refuses production-live writes. Changing
that is a separate, audited project and requires explicit authorization in the
current session.

Required before any live order:

- Independent review of source entitlement, research integrity, paper evidence,
  strategy/replay parity, order lifecycle, reconciliation, and emergency procedures.
- Broker account mode, symbol, side, quantity, limit, extended-hours flag, buying
  power, and all portfolio limits verified before submission.
- Live-specific configuration stored separately; no paper credential or artifact
  reuse by accident.
- Supervised pilot at the minimum practical risk, one concurrent position, and a
  small trade/day limit; no automatic risk scaling.
- Hard daily-loss lockout, manual kill switch, heartbeat alert, restart recovery,
  broker-side position monitor, and tested flatten/cancel runbook.
- Automatic fallback to no-new-entries on any feed, clock, database, broker,
  reconciliation, or ownership anomaly.

Promotion beyond minimum size requires another frozen evidence window. It is not an
automatic consequence of completing the first live pilot.

## 8. Test and Verification Matrix

### Unit/property tests

- Calendar, session, timezone, DST, previous close, splits, reverse splits, and symbol
  changes.
- RVOL uses prior sessions only and is scale-consistent through time.
- Point-in-time joins never select a future metadata/news/halt observation.
- Every state transition, reset, invalidation, and setup-specific boundary.
- Position sizing caps, rounding, buying power, maximum risk, and partial quantities.
- Order hash/idempotency includes every execution-affecting field.

### Integration tests

- Recorded SIP trade/quote/status stream through scanner to shadow intent.
- News publication/observation delay and duplicate-content handling.
- Halt before trigger, halt after fill, delayed resume, and no resume event.
- Extended-hours order acceptance/rejection, cancel/replace, partial fill, and stale
  order expiry against paper mocks and the paper broker where safe.
- Crash after intent reservation, after broker POST, after acknowledgement, and after
  partial fill; recovery must not duplicate the order.

### Replay/research checks

- Feature and transition parity between batch replay and streaming code.
- Same-bar ambiguity resolved adversely or excluded, never optimistically.
- No overlapping positions or duplicate entries beyond the frozen policy.
- Baseline and variants use identical dates, universe, costs, fills, and risk.
- Results include quiet periods, failed runners, delistings, and missing-data days.

### Operational checks

- Queue saturation/drop test, slow consumer, websocket disconnect/reconnect, clock
  skew, stale quote, database outage, broker timeout, and duplicate update.
- Start with an existing broker position or working order.
- Dashboard/alert outage does not bypass risk, and risk outage blocks new entries.
- End-of-day reconciliation proves local quantities equal broker quantities.

## 9. Machine-Learning Decision Point

Do not train the existing XGBoost path until the deterministic system and labels are
valid. If the rule baseline passes replay and shadow gates:

- Restrict ML to ranking or abstention among hard-valid setup triggers.
- Define labels from the same conservative fill/exit simulator used for evaluation.
- Split by trading date with purging/embargo for overlapping forward windows.
- Fit preprocessing on training data only; use walk-forward OOF predictions.
- Calibrate probabilities and compare net expectancy/coverage against the named rule
  baseline on untouched dates.
- Require stable incremental value in ablation. Keep the deterministic rule if ML
  does not improve conservative out-of-sample execution results.

## 10. Recommended Implementation Order

The critical path is:

1. Phase 0 strategy/config freeze.
2. Phase 1 SIP/quotes/trades/extended-hours/metadata source proof.
3. Point-in-time universe, prior close, RVOL, catalyst, float, and halt plumbing.
4. Scanner and pattern state machines.
5. Portfolio/risk/exit logic.
6. Chronological replay and preregistered study.
7. Shadow mode and parity.
8. Paper execution and 6–8-week frozen study.
9. Live-readiness review.

Catalyst latency and source quality are part of the critical path. Social momentum
and presidential/policy signals remain separate context modules and should not delay
the deterministic v1 unless the preregistered experiment specifically evaluates
their incremental contribution.

## 11. Definition of Done

The module is finished for paper operation only when:

- The exact v1 strategy and configuration are versioned and fully tested.
- Live and replay consume the same contracts and produce identical causal states.
- True prior close, RVOL, float, catalyst, quote/spread, resistance, and halt data are
  point-in-time and source-validated.
- The event-driven replay passes its untouched OOS and robustness gates.
- Shadow capture demonstrates stable data, latency, and deterministic parity.
- Paper orders are idempotent, fill-aware, reconciled, and protected by portfolio
  fuses; no strategy code can select a live account.
- The 6–8-week frozen paper gate is complete and decision-useful.
- Documentation includes operations, failure recovery, daily review, and rollback.

It is finished for live consideration only after a separate audited review approves
the production gateway change and the user explicitly authorizes live trading. No
backtest, shadow result, or paper result alone authorizes that transition.

## References

- Warrior Trading, Momentum Day Trading Strategy:
  https://www.warriortrading.com/momentum-day-trading-strategy/
- Warrior Trading, scanner usage:
  https://support.warriortrading.com/support/solutions/articles/19000117763-scanners-how-to-load-use-them-in-the-wt-chat-room
- Alpaca, real-time stock data and SIP/IEX feeds:
  https://docs.alpaca.markets/us/docs/real-time-stock-pricing-data
- Alpaca, market-data feed differences:
  https://docs.alpaca.markets/us/docs/market-data-faq
- Alpaca, extended-hours order requirements:
  https://docs.alpaca.markets/us/docs/orders-at-alpaca
