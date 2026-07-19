# Intraday Structure Engine: Repository Audit and Design

## Audit result

The repository already contained most upstream data and several reusable calculations, but no unified per-candidate intraday setup registry. The v1 implementation therefore adds a narrow confirmation service and adapters; it does not replace any ranker, execution policy, broker client, stream, or dealer calculation.

### A. Existing reusable components

| Capability | Existing implementation | Reuse in v1 |
|---|---|---|
| Shared live 1-minute bars | `UI/shared_stream.py`, backed by Alpaca IEX `StockDataStream.subscribe_bars` | New opt-in queue; no second WebSocket |
| 1m aggregation/live lifecycle examples | `core/API/Alpaca_API/market_data/bar_aggregator.py`, swing runner confirmation watchers | Shared bar schema and persistent-state patterns |
| Causal session VWAP, prior-day levels, swing distances | `strategies/multi_ticker_swing/features/build_features.py` | Definitions mirrored at 1-minute cadence; no future fractal pivots |
| Causal support/resistance zones | `signals/location_features.add_liquidity_zone_features` | Called directly by unified level provider |
| SPY/QQQ/sector/VIX context | 30m swing and 4h Momentum features; shared caches; live stream includes SPY/QQQ/VIXY | Synchronized causal 1-minute context, with partial-context warnings |
| Static options positioning | `strategies/dealer_positioning`: Schwab chain parser, GEX ladder, walls, magnets, gamma flip, DTE buckets | Read-only JSON adapter with as-of/future-snapshot guard |
| Stored broad options activity | `signals/news/data/processed/cboe_options_summary.parquet` and `cboe_unusual_strikes.parquet` | Typed static snapshot adapter; explicitly not OPRA flow/dealer inventory |
| Existing candidate outputs | JSONL signal audits for Meta, Momentum, HTF, Dealer Ranker; swing session `signal` records | Audit candidate feed; source scores/evidence preserved |
| Alert schemas | `core/live_signal_audit.py` plus module-specific JSONL | New typed signal schema and transition-only JSONL |
| Dashboards/server | `UI/combined_server.py`, hub, common UI chrome | Opt-in paper-only dashboard/API on port 8774 |
| Trade lifecycle examples | swing position manager, SPY order policy, shared 4h execution engine, dealer pending-break logic | Concepts reused; not coupled because their states describe orders/positions rather than intraday setups |

### Available frequencies and data semantics

- Live: 1-minute OHLCV bars for the combined-server universe. Bar payloads may include trade count and bar VWAP. Schwab has an alternative chart-bar stream.
- Cached broad universe: roughly 3,093 symbols at 1-hour, 4-hour, and daily frequencies; the swing module also maintains 5/10/30-minute caches.
- Cached 1-minute: strong SPY-specific history and current runtime cache. There is no single broad, point-in-time 1-minute history for every ranker candidate.
- Trades/quotes: broker clients can request snapshots/quotes, and options execution code records selected-contract bid/ask when queried. The shared equity stream currently subscribes to bars only; it does not fan out every trade or quote.
- Live options flow: unavailable as a true trade-by-trade OPRA feed. Stored CBOE unusual-strike data is snapshot/aggregate data, not a live flow tape.
- Sector context: sector ETF mappings and 1h/4h caches exist. Live sector context is available only when the candidate carries `sector_etf`; otherwise the engine reports partial context.
- Breadth: higher-timeframe breadth/regime research exists, but no synchronized one-minute breadth stream is currently wired. The field remains optional.
- Volume profile: no shared canonical provider existed. v1 adds a small causal rolling profile inside the unified level provider, alongside the reused liquidity zones.

### Existing overlap and duplication avoided

- Dealer GEX/walls/gamma flip are consumed, not recomputed from chains.
- Candidate selection remains with the rankers. The engine does not rescore the universe.
- The existing WebSocket is reused. No parallel market-data connection is opened.
- Existing broker execution is untouched. There is deliberately no `submit_order` path in this module.
- Existing 30-minute and 4-hour feature matrices are not rebuilt at 1-minute cadence.

### Leakage and integrity risks found

1. Confirmed fractal pivots can require future bars. v1 uses rolling extrema and already-causal liquidity zones only.
2. `live_gamma_levels.json` is the latest snapshot and would leak in historical replay. The adapter rejects any snapshot after the decision time. Historical options ablations require timestamped archives.
3. A ranker's 4-hour bar timestamp can precede actual scoring/availability. `Candidate` stores both `timestamp` and `available_at`; replay schedules on `available_at`.
4. Current-bar OHLCV is valid only for decisions stamped at bar close. Faster quote/trade updates must use their own event/availability timestamps when later added.
5. Broad 1-minute replay cannot be inferred from 30-minute/4-hour bars. Replay fails if median intraday spacing exceeds 90 seconds.
6. Same-bar target/stop order is unknowable from OHLC. Labels conservatively assign invalidation first.
7. Dealer positioning is an estimate based on OI/Greeks and sign assumptions. It is never described as observed dealer inventory.
8. CBOE unusual-strike snapshots do not contain a complete bid/ask classified OPRA tape. They remain static context, not Mode B flow.

## B. Missing components addressed

- Unified candidate schema with source time and availability time.
- Persistent active setup registry and transition log.
- Modular V-reversal, breakout, VWAP reclaim, structural rejection, trend pullback, and exhaustion rules.
- Unified structural levels with ATR/percentage clustering and source metadata.
- Transparent runway score and causal target extension policy.
- Provider-neutral live options flow interface.
- Typed trade/quote price-update hook for faster running-setup stop/target management; detection remains bar-close based.
- Chronological replay, conservative labels, friction parameters, and ablation runner.
- Feature-flagged live/dashboard integration and restart recovery.

Still missing from repo data, not fabricated by v1: broad historical 1-minute candidate data, historical full-chain Greeks/OI with exact availability, true live OPRA prints, synchronized one-minute breadth, and validated setup-specific thresholds.

## C. Proposed/implemented architecture

```text
30m / Momentum / HTF / Meta / Dealer audits     manual watchlist
                     \                            /
                      Candidate availability adapter
                                  |
shared 1m bar stream ---> limited candidate registry <--- SPY/QQQ/sector context
                                  |
              causal features + unified structural levels
                                  |
        five entry detectors + exhaustion/target manager
                                  |
        persistent state / transition alerts / signal schema
                         |                    |
                 replay + labels        read-only dashboard
                         |
                paper validation only
```

State is keyed by `(ticker, direction, setup_type)`. Candidate refreshes update an existing active setup rather than duplicating it. Recent bar histories, candidates, and setup state are atomically persisted. Only meaningful state changes enter the append-only transition log.

Optional `PriceUpdate` events can advance an already-running setup to target or invalidation between bar closes. They cannot create, arm, confirm, or otherwise change detector evidence.

## D. Files created or modified

Created:

- `strategies/intraday_structure/`: config, models, features, market/options/level adapters, runway, detectors, target manager, engine, candidate feed, persistence, runner, replay, labels, CLI, tests, and module README.
- `UI/intraday_structure_dashboard.py`: setup cards, target/risk/evidence display, transition timeline, state API, and manual candidate API.
- `docs/intraday_structure/`: audit/design, feature, and replay documentation.
- `examples/intraday_structure/example_signal.json`: schema example.

Modified:

- `UI/combined_server.py`: opt-in queue/dashboard on port 8774; default remains off.
- `UI/hub_dashboard.py`: conditional module card only when enabled.
- `pyproject.toml`: includes the focused test directory.
- `README.md`, `docs/PROJECT_STATUS.md`: module registry/status links.
- `LIVING_SUMMARY.md`: durable implementation/validation handoff.

## E. Implementation order

1. Audit and define event/availability time contracts.
2. Add typed schemas and versioned, paper-only configuration.
3. Build causal features, options adapters, structural levels, and runway scoring.
4. Implement modular detectors, lifecycle, dynamic targets, persistence, and duplicate suppression.
5. Add replay/labels/frictions/ablations.
6. Attach an opt-in subscriber/dashboard to the existing combined server.
7. Validate synthetic paths, restart behavior, deterministic replay, and existing hub/server compatibility.

## F. Risks and open assumptions

- Thresholds are interpretable starting hypotheses, not calibrated production policy.
- Relative volume currently uses a rolling intraday baseline; a point-in-time same-minute-of-day baseline should replace it after broad history exists.
- Candidate audit feeds expose the latest decision, not a transactional message bus. `available_at` prevents early replay use, but a future unified publisher would be cleaner.
- Static dealer snapshots may be stale or absent for most equities. Missing options data is neutral and warned, never imputed.
- Price bars cannot reproduce quote spread, queue priority, halts, or exact intrabar ordering.
- The v1 dashboard is monitoring-only. Execution integration is intentionally deferred until replay and paper evidence exists.

Next roadmap step: collect candidate-scoped 1-minute bars and timestamped dealer snapshots during paper sessions, freeze a validation/test protocol, and run setup-level plus options/context ablations before considering any deterministic execution approval interface.
