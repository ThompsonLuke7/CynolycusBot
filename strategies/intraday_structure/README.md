# Intraday Structure Engine

This module is a deterministic, event-driven confirmation layer between broad candidate rankers and execution. Version 1 is rules-first, paper-only, and has no order-submission interface.

It monitors candidates from the existing 30-minute swing audit, 4-hour Momentum/HTF/Meta/Dealer Ranker audits, and a manual watchlist. One-minute bars drive five entry detectors; a sixth exhaustion detector manages confirmed/running structures. State, recent bar history, transitions, targets, evidence, and warnings persist across restarts.

The public `on_price_update(PriceUpdate(...))` hook can manage stops/targets faster when a future trade/quote fanout is available. It cannot run detectors or confirm entries.

Lifecycle:

`WATCHING → SETUP_DETECTED → ARMED → CONFIRMED → RUNNING → TARGET_REACHED → EXTENDED → EXHAUSTED/INVALIDATED → CLOSED`

Not every detector uses every state. Alerts are emitted only on state changes.

## Safe live monitoring

The combined-server integration is off by default:

```bash
./.venv/bin/python -m UI.combined_server --intraday-structure
```

The dashboard is then available at `http://127.0.0.1:8774`. This enables monitoring only; existing trading modules remain unchanged.

## Replay

Replay requires true chronological 1-minute OHLCV. The loader rejects higher-timeframe inputs instead of fabricating intraday paths:

```bash
./.venv/bin/python -m strategies.intraday_structure.main replay --bars path/to/1m.parquet --candidates path/to/candidates.jsonl --output Data/inference/intraday_structure/replay
```

Each candidate should distinguish the source signal time from availability:

```json
{"ticker":"MU","timestamp":"2026-07-17T14:00:00Z","available_at":"2026-07-17T14:20:00Z","direction":"long","sources":["meta_ranker"],"score":0.91,"pivot":860.0}
```

The replay outputs transitions, causal event labels, modeled trades with configurable spread/slippage/commission, and metrics. It makes no profitability claim; thresholds are hypotheses until tested on collected candidate-level 1-minute history.

See [the audit and architecture](../../docs/intraday_structure/REPO_AUDIT_AND_DESIGN.md), [feature definitions](../../docs/intraday_structure/FEATURES.md), and [replay protocol](../../docs/intraday_structure/BACKTEST.md).
