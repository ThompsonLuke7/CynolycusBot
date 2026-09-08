# Momentum Scalper

Standalone, paper-only implementation of a Ross-style premarket low-float
momentum strategy. It does not share intraday-structure decisions, positions,
entries, exits, or promotion evidence.

The full roadmap, strategy boundary, validation gates, and paper-to-live
promotion criteria are in [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md).

## Current v1 behavior

- Requires point-in-time prior close, float, material catalyst, two-sided quote,
  daily resistance context, halt state, premarket volume, and time-matched RVOL.
- Detects causal HOD breakout, first pullback, bull flag, and flat-top signals.
- Replays a single portfolio with next-quote entries, conservative stop ordering,
  partial exits, and no fabricated fills when quotes are missing.
- Emits paper-only order intents. The shared execution gateway remains responsible
  for idempotency, broker acknowledgement, fills, and reconciliation.

The module is not validated for paper performance yet. No code path can submit a
live order.

## Causal replay

The replay requires vendor-supplied point-in-time parquet inputs; it intentionally
does not infer spreads, news availability, or float from minute bars.

```bash
python -m strategies.momentum_scalper.backtests.replay_engine \
  --bars Data/raw/momentum_scalper/bars.parquet \
  --metadata Data/raw/momentum_scalper/reference.parquet \
  --news Data/raw/momentum_scalper/catalysts.parquet \
  --quotes Data/raw/momentum_scalper/quotes.parquet \
  --halts Data/raw/momentum_scalper/halts.parquet \
  --risk-budget-dollars 100 \
  --output-dir Data/inference/momentum_scalper/replay_YYYYMMDD
```

## Shadow and paper modes

`MomentumShadowRunner` receives the same market/reference inputs and records
intents without submitting them. `execution.paper_bridge.to_paper_order_request`
converts an explicit paper intent to the shared `OrderRequest` contract. A caller
must still create a complete shared decision record and use the paper gateway.

Before any paper-submit session, run the authenticated but read-only preflight:

```bash
python -m strategies.momentum_scalper.live.readiness --json
```

It requires the Alpaca paper endpoint, a readable paper account, current SIP
quotes (not IEX), and the QA-paper durable journal/database configuration. It
does not place an order. The market-event streamer is SIP-only and records
vendor event time separately from local receipt time:

```python
from alpaca.data.enums import DataFeed
from core.API.Alpaca_API.market_data.live_stream import AlpacaQuoteTradeStatusStreamer
```

`MomentumSIPMarketFeed` wires a preselected candidate list into
`LiveMarketBuffer`; it has no broker-order capability. The premarket universe,
point-in-time float, and material-catalyst sources must be present before a
candidate list can be selected.

## Retired MVP paths

The former day-level scanner, one-minute pseudo-spread features, row-wise labels,
and XGBoost flow are not a valid basis for this strategy. They are retained only
as historical code; do not use them to report research or execution performance.
