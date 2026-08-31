# Intraday Structure Engine

This module is a deterministic, event-driven confirmation layer between broad candidate rankers and execution. Version 1 is rules-first, paper-only, and has no order-submission interface.

It monitors candidates from the existing 30-minute swing audit, 4-hour Momentum/HTF/Meta/Dealer Ranker audits, and a manual watchlist. One-minute bars drive five entry detectors; a sixth exhaustion detector manages confirmed/running structures. State, recent bar history, transitions, targets, evidence, and warnings persist across restarts.

## Measurement

Three append-only artifacts under `Data/inference/intraday_structure/`:

| File | What it is |
|---|---|
| `closed_setups.jsonl` | One immutable row per setup that reached a terminal state: entry, exit, invalidation, targets, MFE/MAE, R after costs, and the decision inputs (candidate sources, runway, regime, the level families behind the target). |
| `abstentions.jsonl` | One row per confirmation the engine **declined**, with the specific reason and the numbers behind it. |
| `decision_events.jsonl` | One ordered paper funnel: catalyst filtering, candidate registration/capacity eviction, setup transitions/abstentions/closes, and candidate-level 5/15/30/60-minute MFE/MAE even when no setup confirms. These are event-time model outcomes, not fills. |
| `premarket_plan.json` | The pre-open plan: trigger, invalidation, full target ladder, and an AVOID list with reasons. |
| `Data/archive/intraday_1m/bars_*.jsonl` | Every streamed 1-minute bar that arrived, with arrival time separate from event time. Forward-only; the sole input for the D2 counterfactual. Size scales with the streamed universe; each completed session gets a coverage/latency/drop manifest. Disable with `archive_bars: false`. |

Before these existed the module could run for a month and still not answer "did
any of that work?" — `state.json` is a snapshot that overwrites a closed setup's
entry price, and `transitions.jsonl` records state changes without prices. Both
are still written; the ledger is what makes the module measurable.

Read it:

```bash
./.venv/bin/python -m strategies.intraday_structure.main report
./.venv/bin/python -m strategies.intraday_structure.main funnel-report
```

Every ledger row carries the cost assumptions it was priced under, so a later
config change cannot silently re-price history; the report refuses to blend rows
priced differently, and flags any bucket under 30 setups.

## Opening and catalyst discovery

The combined stream includes the 750 highest-ADV eligible shared-universe names
(only 159 subscriptions beyond the existing live swing universe as of
2026-08-28). From 04:00 through 11:00 ET, a rules-first paper feed promotes:

- gap leaders whose volume pace confirms the displacement; and
- post-open acceleration leaders whose short-window return, session range or
  market-relative strength, and volume pace agree.

The existing setup detectors still decide whether price structure confirms.
Discovery itself cannot place an order. The source baseline is the locally
available `shared_universe.csv`; its file timestamp, the market-data event time,
the local availability time, and their lag are recorded separately.

Fresh catalyst-ledger rows are promoted only when the ticker has a verified
eligible liquidity baseline, the record is directional and material, the
headline passes strict subject relevance, and it is not a backward-looking
price recap. Ambiguous English-word tickers such as NOW require an explicit
company subject. Rejection reasons are written to `decision_events.jsonl`.

Candidate capacity is still hard-bounded. Up to 25 opening and 15 catalyst
candidates receive reserved retention inside the 140-key limit; unused slots
flow back to the existing ranker/dealer/liquidity sources. Existing active
setups and the manual watchlist remain higher-priority than passive seeds.

A price-only baseline over stored 1-minute history — the weakest arm of the
eventual ablation, with no ranker and no dealer context:

```bash
./.venv/bin/python -m scripts.run_intraday_structure_baseline \
    --bars Data/raw/spy/spy_intraday_1min_runtime_rth_cache.parquet \
    --symbol SPY --start 2025-08-25 --end 2026-08-24 \
    --output Data/analysis/intraday_structure_baseline/spy_1y
```

## Pre-open plan

```bash
./.venv/bin/python -m scripts.build_premarket_plan --top 25
```

Runs from the combined server at 09:00 ET (before the 09:35/09:37 flushes), is
read-only, and publishes the whole target ladder up front rather than one rung
at a time — you cannot judge reward:risk against a destination that has not been
named. It builds from stored daily/hourly bars and the prior session's dealer
snapshot; `Data/shared/bars/1h` is RTH-only, so overnight and premarket levels
are **not** available and every plan says so in its `warnings`.

## Context regime and abstention

Each confirmation decision is labelled `TRENDING_UP / TRENDING_DOWN / BALANCED /
COMPRESSED` by rule — no model — and the label is recorded whether the setup is
taken or declined. Letting the regime *veto* a setup is a separate switch
(`regime.veto_enabled`, default **off**), because recording a label changes
nothing while blocking a trade is a trading change that has to be measured.

When available, it also reads the captured broad dealer-level summaries (walls, gamma flip, magnets, floor/ceiling, GEX estimates) strictly as-of the bar being evaluated. The default live watchlist combines the 50 highest-ADV eligible shared-universe names with high structural-potential and high dealer-map-change names, then price must still confirm. Confirmed signals carry an auditable `dealer_plate` field; it is qualified only when a strong, fresh, reachable dealer destination supports the structure. This is a paper-only hypothesis label, not observed dealer inventory, an options order, or a profitability claim.

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

The current July dealer summaries can seed the next session's watchlist, but they cannot recreate a same-day pre-pivot decision if their `captured_at` time was later than the pivot. Broad 1-minute bar collection is therefore still required for a valid baseline-versus-price-only-versus-dealer-plate alpha comparison.

See [the audit and architecture](../../docs/intraday_structure/REPO_AUDIT_AND_DESIGN.md), [feature definitions](../../docs/intraday_structure/FEATURES.md), and [replay protocol](../../docs/intraday_structure/BACKTEST.md).
