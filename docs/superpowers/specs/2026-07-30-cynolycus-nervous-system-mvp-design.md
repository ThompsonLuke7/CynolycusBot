# Cynolycus Shared Context and Policy Nervous System MVP Design

**Status:** Approved 2026-07-30

**Target branch:** `nervous-system`

**Initial runtime scope:** Development, replay, shadow, and QA-paper

**Production-live:** Recognized configuration value but denied by policy in this MVP

## 1. Executive decision

Cynolycus will gain a shared, auditable decision layer implemented as a modular
monolith under `core/nervous_system/`. Existing domain producers continue to
calculate market, sector, theme, ticker, catalyst, dealer, and broker state.
Thin adapters beside those producers translate their outputs into shared
Pydantic contracts and persist versioned operational state in PostgreSQL.

The first complete vertical slice is the existing 4H Meta Ranker:

```text
existing causal producers
  -> versioned state records
  -> immutable ContextSnapshot
  -> unchanged Meta TradeIntent
  -> deterministic PolicyDecision
  -> equity or option InstrumentPlan
  -> one-to-four-leg OrderRequest
  -> ExecutionGateway
  -> Alpaca paper adapter
  -> ExecutionReport
  -> immutable DecisionRecord
```

The MVP includes the full options contract and construction framework, including
defined-risk spreads, rather than designing a stock-only interface that would
need to be replaced. It ends in QA-paper operation. Enabling production-live is
a separate future project requiring explicit approval and evidence.

PostgreSQL stores mutable and transactional operational truth. Parquet remains
the authoritative format for historical bars, feature matrices, options
snapshots, backtests, and other analytical artifacts. Existing JSON, JSONL, and
CSV operational artifacts are imported as historical evidence with lineage;
they do not remain competing authorities after cutover.

## 2. Problem being solved

The repository already contains useful causal features, readiness checks,
broker snapshots, execution helpers, and audit records, but they are joined by
implicit file conventions and strategy-specific code. The resulting risks are:

- duplicated or contradictory operational state;
- no single as-of rule across market, theme, catalyst, dealer, and portfolio
  context;
- overwritten theme membership outputs that cannot be replayed historically;
- inconsistent event, observation, publication, and availability timestamps;
- Meta, HTF, Momentum, Dealer, and other paths constructing or submitting
  orders through different boundaries;
- strategy ownership being inferred from mutable local files instead of
  confirmed broker fills;
- context fields being calculated without a guaranteed, measurable effect on
  downstream risk or instrument selection;
- incomplete reconstruction of why a trade was approved, changed, rejected, or
  filled;
- stale pending option contracts surviving from an after-close decision into a
  later session;
- no durable transactional store for jobs, decisions, orders, and
  reconciliation.

This project adds the shared nervous system without changing the Meta model or
discarding existing causal producers.

## 3. Goals

The MVP must:

1. Define strict, versioned contracts for all shared state and trading
   decisions.
2. Make `available_at` a mandatory causal boundary.
3. Store operational history and complete decision chains in PostgreSQL.
4. Import all available historical operational JSON, JSONL, and CSV evidence
   with source lineage and idempotency.
5. Build immutable and historically reproducible context snapshots.
6. Preserve the Meta Ranker's raw rankings exactly.
7. Apply policy through named, machine-readable vetoes and modifiers.
8. Make enforce-mode modulation alter the actual risk budget, permissions, or
   order request.
9. Support equity and the approved full options suite, including spreads.
10. Route automated orders through one execution and reconciliation boundary.
11. Record every order in PostgreSQL and an independent durable execution
    journal.
12. Use the same context, policy, exposure, and instrument code for replay,
    shadow, and QA-paper.
13. Remain portable from local Docker PostgreSQL to GCP Cloud SQL.

## 4. Non-goals

The MVP does not:

- change, retrain, or tune the Meta Ranker;
- relabel heuristic scores, percentiles, z-scores, or ranks as probabilities;
- claim policy-added profitability before replay and QA-paper evidence exists;
- enable production-live order submission;
- move analytical Parquet datasets into PostgreSQL;
- reconstruct unavailable historical information from future data;
- add Kafka, Kubernetes, microservices, BigQuery, or a second scheduling
  framework;
- automatically enforce unvalidated contextual thresholds;
- refactor unrelated strategy research code.

## 5. Approaches considered

### 5.1 Chosen: `core/nervous_system/` modular monolith

The shared contracts, repositories, snapshot builder, policy, exposure,
execution, replay, and orchestration live under one root package. Domain
adapters remain beside the current producers.

This provides one explicit inward dependency boundary while avoiding a large
move of working strategy code.

### 5.2 Rejected: move all domains into new top-level packages

Creating new top-level `context/`, `portfolio/`, and `execution/` trees would
make the architecture visually pure but require moving mature modules, changing
many imports, and increasing regression risk before behavior is proven.

### 5.3 Rejected: external services and message broker

Separating state, policy, and execution into services would add deployment,
networking, serialization, and failure modes before the single-process
contracts are stable. A typed in-process event bus plus transactional outbox is
sufficient for the current scale and can later support an external transport
without changing domain contracts.

## 6. Package structure and dependency direction

```text
core/nervous_system/
├── contracts/
├── config/
├── persistence/
├── data_registry/
├── context/
├── policy/
├── portfolio/
├── execution/
├── orchestration/
├── replay/
└── tests/
```

Responsibilities:

- `contracts/`: Pydantic models and enums exchanged across boundaries.
- `config/`: validated environment, freshness, policy, exposure, and execution
  configuration.
- `persistence/`: SQLAlchemy mappings, repositories, transactions, Alembic
  integration, import ledgers, and connection factories.
- `data_registry/`: immutable source identities, hashes, versions, and lineage.
- `context/`: as-of queries, freshness profiles, snapshot construction, and
  data-quality aggregation.
- `policy/`: deterministic rules, conflict resolution, sizing modifiers,
  permissions, and reason codes.
- `portfolio/`: canonical exposure aggregation and broker-observation
  reconciliation.
- `execution/`: instrument planning, final validation, broker gateways,
  idempotency, execution journal, reports, and reconciliation.
- `orchestration/`: job records, typed events, transactional outbox, and
  coordinator functions.
- `replay/`: historical providers, deterministic replay, parity checks, and
  outcome attachment.

Dependency rules:

1. `contracts` and validated configuration do not import strategy modules.
2. Shared nervous-system code does not import Meta, Momentum, HTF, Dealer, or
   UI implementations.
3. Domain adapters import contracts and repositories, not the reverse.
4. Broker-specific code implements interfaces owned by `execution`.
5. UI code reads repositories and projections; it does not own state.
6. Strategy code emits an intent and receives a decision or report; it does not
   submit directly after migration.

Representative edge adapters remain with their owners:

```text
signals/market_regime/nervous_system_adapter.py
themes/dynamic_theme/nervous_system_adapter.py
signals/catalysts/nervous_system_adapter.py
strategies/dealer_positioning/nervous_system_adapter.py
signals/meta_context/meta_ranker/nervous_system_adapter.py
```

The exact adapter filename may follow the local package convention, but its
dependency direction may not be reversed.

## 7. Authority and storage model

| Information | Authority | Secondary evidence |
|---|---|---|
| Historical bars and features | Versioned Parquet | Source registry and hashes |
| Historical option chains/marks | Versioned Parquet | Source registry and fitness reports |
| Calculated operational state | PostgreSQL | Original artifacts and producer logs |
| Context snapshots and decisions | PostgreSQL | Content hashes referenced by the execution journal |
| Account, orders, fills, positions | Alpaca | Timestamped PostgreSQL observations |
| Strategy ownership | Fill-confirmed PostgreSQL mapping | Decision and reconciliation records |
| Configuration used for a decision | Immutable config snapshot in PostgreSQL | Version-controlled defaults |
| Research outcomes | Versioned research artifacts | Outcome records linked after decisions |

Existing JSON, JSONL, and CSV files are immutable import evidence. During
transition, compatibility projections may still be written for old dashboards,
but those projections are generated from PostgreSQL and are not read as
co-equal operational truth.

## 8. Environments and execution modes

Two dimensions are explicit:

```text
RuntimeEnvironment:
  DEVELOPMENT
  QA_PAPER
  PRODUCTION_LIVE

PolicyExecutionMode:
  OFF
  SHADOW
  ENFORCE
```

- `DEVELOPMENT` uses simulated or fixture broker adapters.
- `QA_PAPER` may use only Alpaca paper credentials and the configured paper
  account identity.
- `PRODUCTION_LIVE` is parsed and represented but always receives the
  `ENV_PRODUCTION_LIVE_DISABLED_MVP` hard veto.
- `OFF` preserves baseline comparison while the adapter is being tested.
- `SHADOW` records the exact counterfactual policy and order without changing
  broker behavior.
- `ENFORCE` makes the policy budget and permissions authoritative for the
  generated QA-paper order.

Environment, mode, account identity, and credential profile are validated
together. A paper process cannot silently select a live endpoint or live
credential profile.

## 9. Contract model

The shared package is named `contracts` because these classes define versioned
promises between producers, persistence, replay, policy, and execution. The
individual objects are Pydantic v2 models.

Contract defaults:

- frozen models;
- unknown fields forbidden;
- UTC-aware timestamps required;
- naive timestamps rejected;
- non-finite numeric values rejected;
- probabilities constrained to `[0, 1]`;
- explicit `UNKNOWN` enum values;
- canonical JSON serialization and lossless round trips;
- `schema_version` on every persisted contract.

### 9.1 Common state envelope

Every state record carries:

```text
state_id
state_type
entity_id
as_of
available_at
generated_at
valid_until
source_window_start
source_window_end
schema_version
producer
model_version
feature_version
config_version
lineage_ids
data_quality
```

Semantics:

- `as_of`: market or business time described by the record.
- `available_at`: earliest time the system could have consumed this exact
  record.
- `generated_at`: time the producer materialized this representation.
- `valid_until`: exclusive freshness boundary for the configured consumer.
- `decision_time`: time at which a strategy decision is evaluated.
- `submitted_at`: time an order request reaches the broker boundary.
- `filled_at`: broker-confirmed fill time.
- `evaluated_at`: later outcome measurement time.

Snapshot eligibility is always:

```text
available_at <= decision_time < valid_until
```

`as_of` is never substituted for `available_at`.

### 9.2 Data quality

`DataQualitySummary` contains structured:

- missing components;
- stale components;
- invalid fields;
- fallback values and reasons;
- source lineage;
- timestamp confidence;
- quarantined legacy fields;
- warnings;
- a severity suitable for policy requirements.

Unknown information remains unknown. Legacy data without a defensible
availability timestamp may be retained as raw evidence but cannot become an
eligible state record.

### 9.3 Trading contracts

`TradeIntent` preserves the strategy request without policy mutation:

```text
intent_id
strategy_id
ticker
direction
raw_score
raw_probability
expected_return
expected_holding_period
entry_window
preferred_entry
invalidation
target
stop
position_size_requested
instrument_preferences
feature_timestamp
model_version
feature_version
reason_codes
```

For current Meta output, rank and combo values populate score fields.
`raw_probability` remains null unless the producing model is actually
calibrated and versioned as a probability model.

`PolicyDecision` records:

- action and approved direction;
- hard vetoes;
- ordered soft modifiers;
- budget before and after every rule;
- final risk budget;
- allowed instrument families;
- stop, target, and holding-period adjustments;
- hedge or collateral requirements;
- rule and configuration versions;
- machine-readable reason codes.

`OrderRequest` supports equity and one-to-four option legs and records:

- originating decision and policy IDs;
- environment and account alias;
- side, order type, time in force, and limit semantics;
- net debit or credit;
- maximum loss;
- buying-power or collateral requirement;
- parent quantity and leg ratios;
- quote snapshot ID;
- idempotency key and request hash;
- expiration and supersession links.

`ExecutionReport` is composed from append-only execution events rather than
mutating the original request.

`DecisionRecord` links immutable identities and content hashes for:

- source manifest;
- context snapshot;
- raw strategy output;
- trade intent;
- policy decision;
- instrument candidates;
- order request;
- execution events;
- configuration;
- model and feature versions.

`DecisionOutcome` is a separate append-only hindsight record. It cannot be
loaded as a decision-time input.

### 9.4 Identity and hashing

Business records use UUID identities. Canonical content hashes exclude storage
metadata that would make equivalent content hash differently. The hash
canonicalization version is explicit.

Ids and hashes serve different purposes:

- IDs establish relationships and idempotency.
- Content hashes establish equivalence and replay parity.
- Broker IDs establish external authority.
- Source hashes establish immutable lineage.

## 10. PostgreSQL persistence

Use:

- PostgreSQL;
- SQLAlchemy 2.x synchronous repositories, matching the repository's current
  synchronous runtime;
- psycopg 3;
- Alembic migrations;
- PostgreSQL schema `nervous_system`.

### 10.1 Hybrid state representation

State records use relational envelope columns plus a validated, versioned JSONB
payload. Frequently queried identity, type, time, version, quality severity,
and hash fields remain indexed columns. State-specific fields remain in JSONB.

This avoids seven nearly identical state-table implementations while retaining
efficient as-of selection and contract evolution.

### 10.2 Core tables

The schema contains:

- `state_records`
- `context_snapshots`
- `trade_intents`
- `policy_decisions`
- `policy_modifiers`
- `order_requests`
- `order_legs`
- `execution_events`
- `decision_records`
- `decision_outcomes`
- `portfolio_observations`
- `portfolio_ownership`
- `source_artifacts`
- `import_runs`
- `import_items`
- `import_quarantine`
- `lineage_edges`
- `config_snapshots`
- `job_runs`
- `job_events`
- `outbox_events`
- `alerts`

Important constraints include:

- unique record IDs;
- unique canonical content hashes where semantic deduplication is intended;
- unique import identity by artifact hash, source row/line identity, importer
  version, and normalized content hash;
- unique client order IDs per environment/account;
- foreign keys from orders to approved policies;
- foreign keys from decision records to their complete chain;
- one active ownership allocation per broker position component and strategy;
- UTC timestamp checks;
- valid exclusive time windows;
- immutable decision payloads.

Primary state lookup indexes cover:

```text
(state_type, entity_id, available_at DESC)
(state_type, entity_id, as_of DESC)
(valid_until)
(content_hash)
```

JSONB indexes are added only for demonstrated queries, not indiscriminately.

### 10.3 Repository boundaries

Repositories expose contract-oriented operations such as:

```python
save_state(state)
get_latest_valid_state(state_type, entity_id, decision_time)
get_state_as_of(state_type, entity_id, decision_time)
save_context_snapshot(snapshot)
save_trade_intent(intent)
save_policy_decision(decision)
save_order_request(order)
append_execution_event(event)
save_decision_record(record)
append_decision_outcome(outcome)
```

Callers do not construct SQLAlchemy ORM objects or issue SQL directly.

### 10.4 Transactions

A transaction persists the intent, snapshot reference, policy, candidate
selection, and planned order before broker submission. Broker calls do not run
inside a database transaction. Their results append in a subsequent
transaction.

Typed orchestration events are inserted into `outbox_events` in the same
transaction as the state change they describe. Dispatch marks delivery
separately and is idempotent by event ID.

## 11. Historical operational import

All available relevant operational history is imported because the current
testing history is measured in months, not years.

The importer:

1. Inventories relevant JSON, JSONL, and CSV artifacts.
2. Registers path or URI, SHA-256 hash, size, and source type before parsing.
3. Uses a source-specific legacy adapter rather than one permissive parser.
4. Preserves file, row or line number, raw payload, importer version, warnings,
   and normalized record identity.
5. Validates normalized output through the current contract version.
6. Quarantines invalid or causally unusable records without aborting unrelated
   records.
7. Produces counts for discovered, parsed, imported, duplicated, skipped, and
   quarantined items.
8. Is safe to rerun without creating duplicate state or execution events.
9. Never edits or deletes source artifacts.

Missing semantic fields become `UNKNOWN` or explicit quality warnings. Missing
causal timestamps are not guessed from filesystem modification time. Such
records remain source evidence or quarantine entries and are excluded from
historical snapshots.

After import:

- current broker orders, fills, and positions are queried;
- imported ownership is matched only where evidence is sufficient;
- unmatched broker positions are `UNASSIGNED`;
- counts and latest-record selection are compared with legacy views;
- compatibility projections are generated from PostgreSQL;
- read cutover occurs only after parity and reconciliation pass.

Parquet bars, feature matrices, options snapshots, and research outputs remain
outside this importer and are registered by reference and hash.

## 12. Local PostgreSQL and GCP portability

Development and integration tests use a local Docker PostgreSQL service with a
persistent named volume. No manual table creation is required; Alembic is the
only schema installation path.

QA-paper uses Cloud SQL for PostgreSQL. The same SQLAlchemy models, Alembic
migrations, repositories, and importer run in both environments. Connection
configuration is the only environment-specific layer:

- local TCP URL for Docker;
- Cloud SQL Unix socket or approved secure connector path for Cloud Run;
- credentials supplied through local secrets or GCP Secret Manager.

The local development database is disposable. Cloud SQL is built by running
the same migrations and replaying immutable import artifacts, not by making the
local database a permanent upstream authority.

Recommended migration sequence:

1. Complete the active GCS/storage foundation.
2. Develop contracts and persistence locally.
3. Provision Cloud SQL before QA-paper deployment.
4. Apply Alembic migrations to Cloud SQL.
5. Run and verify the idempotent historical importer against Cloud SQL.
6. Deploy QA-paper services with Cloud SQL connectivity.

## 13. State producers and adapters

| State | Current source | Adapter responsibility |
|---|---|---|
| Market | `signals/market_regime/` | Preserve causal regime output, version, availability, and staleness |
| Sector | `signals/market_regime/` and mappings | Use one canonical resolver and preserve declared legacy fallback behavior |
| Theme | `themes/dynamic_theme/` | Persist versioned taxonomy, membership weights, ranks, and state instead of overwriting |
| Ticker | 4H feature matrix and live panel | Select the exact closed decision bar and its price/features |
| Catalyst | `signals/catalysts/`, news, earnings, social | Normalize event, publication, observation, and availability times |
| Dealer | `strategies/dealer_positioning/` | Persist captured states and level dynamics without future reconstruction |
| Portfolio | broker snapshots, orders, fills, local ownership | Treat broker facts as authoritative and local ownership as an attribution |

Existing reusable components include:

- causal market and sector state with `available_at`;
- DST-safe timing and future-append tests;
- causal 4H feature matrices and live panels;
- dealer captures with `captured_at` and level dynamics with `available_at`;
- `core/broker_equity_snapshot.py`;
- `core/live_readiness.py`;
- `core/live_signal_audit.py`;
- planning, exit, and duplicate helpers in `core/live_4h_exec.py`;
- deterministic replay discipline in
  `strategies/intraday_structure/replay.py`;
- scheduler, startup queue, and job guards under `UI/` and `core/`.

Required corrections:

- Theme outputs become historical, not overwrite-only.
- Catalyst adapters reject unsafe exact-time merges and represent uncertain
  availability explicitly.
- Ticker reference price is taken from the selected decision bar, fixing the
  current latest-bar ambiguity.
- Dealer heuristics remain scores unless separately calibrated.
- Sector mappings are consolidated behind one resolver while retaining an
  explicit compatibility mode for frozen-model parity.
- Partial producer refreshes cannot silently publish a complete snapshot.

## 14. Context snapshot construction

The builder interface is:

```python
build_snapshot(
    decision_time,
    strategy_id,
    entity_ids,
    freshness_profile,
) -> ContextSnapshot
```

For each required state, the builder queries only records satisfying the
availability and validity predicate. It never selects a plain "latest" record
without the decision-time predicate.

The immutable snapshot contains:

- full validated state payloads needed by policy;
- referenced state IDs and content hashes;
- decision timestamp and trading session;
- the exact freshness profile;
- missing, stale, invalid, and fallback components;
- source, feature, model, config, and taxonomy lineage;
- a canonical snapshot content hash.

Freshness profiles are strategy-specific rather than global.

For current Meta scheduling:

- 14:20 ET uses the latest fully closed causal 4H bar;
- 16:20 ET may use the newly closed bar;
- daily market regime published after approximately 16:30 remains prior-session
  state for both decisions;
- a 15:45 dealer capture may inform 16:20 but cannot inform 14:20;
- nightly theme and catalyst products remain prior-session unless their
  producers prove an earlier `available_at`.

Required operational state missing or critically stale causes an entry veto.
Optional context becomes `UNKNOWN` and may remove a modifier without
necessarily vetoing. Broker-authoritative risk-reducing exits remain possible
under degraded context and are explicitly audited.

## 15. Meta Ranker intent adapter

The Meta adapter consumes the exact ranked rows and selected decision bar
already produced by the current pipeline. It emits `TradeIntent` without
changing ranking, score thresholds, universe, or model features.

The adapter must preserve:

- ticker ordering;
- `s_upside`, `s_quality`, and percentile `s_combo` semantics;
- selected feature timestamp;
- existing strategy-side entry, invalidation, target, stop, and sizing request;
- model and feature versions;
- existing strategy reason codes.

The acceptance fixture records representative pre-adapter Meta output and
requires exact raw ranking parity after the adapter. Policy effects are
measured only after this boundary.

After-close deferred entries persist the intent and originating decision
identity, not a stale OCC contract, quantity, or limit price.

## 16. Deterministic policy

The policy interface is a pure function:

```python
evaluate_policy(
    intent: TradeIntent,
    snapshot: ContextSnapshot,
    portfolio: PortfolioState,
    config: PolicyConfig,
) -> PolicyDecision
```

It does not read files, clocks, databases, APIs, or mutable global state.

Evaluation order:

1. Environment and strategy permissions.
2. Timestamp, freshness, and data-quality requirements.
3. Duplicate orders, ownership conflicts, and cooldowns.
4. Account, daily-loss, and buying-power limits.
5. Ticker, sector, theme, and correlated exposure.
6. Earnings, event, and liquidity restrictions.
7. Market, sector, theme, ticker, catalyst, dealer, portfolio, and
   data-quality modifiers.
8. Instrument permissions, collateral requirements, and final risk budget.

Hard vetoes dominate all other rules. Instrument permissions use set
intersection, so a permissive rule cannot restore an instrument removed by a
more restrictive rule.

Every modifier records:

```text
rule_id
rule_version
input value
configured threshold or mapping
multiplier
budget before
budget after
reason code
```

Modifiers execute in stable order. Final budget is bounded by account,
strategy, ticker, sector, theme, correlated-exposure, and maximum-loss caps.
Context cannot exceed those caps. Contextual multipliers are capped at `1.0`
initially; any context-driven increase above the strategy request requires a
separate validated policy version.

MVP hard vetoes include:

- critical stale or invalid state;
- disabled strategy or environment;
- production-live submission;
- duplicate or conflicting active order;
- maximum daily loss breach;
- ticker, sector, theme, correlated, or expiration exposure breach;
- insufficient buying power or collateral;
- prohibited earnings or event window;
- inadequate equity or option liquidity;
- excessive option spread;
- missing dealer state for a dealer-dependent strategy;
- naked short option;
- uncovered ratio structure;
- unavailable durable audit path for a new entry.

The policy distinguishes `ENTRY`, `ADJUSTMENT`, and `EXIT`. Context freshness
that blocks entry does not automatically block a risk-reducing exit.

Current context research is not strong enough to justify silently enforcing
new thresholds. Operational safety rules move to enforce first. Context rules
run as exact shadow counterfactuals, then move individually to QA-paper enforce
only after replay and paper acceptance.

## 17. Canonical portfolio exposure

Portfolio state starts from a broker account, positions, open orders, and fills
observation. It adds strategy attribution only after fill reconciliation.

Every position maps where available to:

- ticker;
- sector;
- weighted themes;
- strategy;
- long/short direction;
- beta and delta-equivalent exposure;
- momentum and volatility exposure;
- high-beta growth, rate, commodity, crypto, and AI-capex sensitivities.

Options add:

- delta-equivalent notional;
- gamma;
- vega;
- theta;
- maximum loss;
- collateral requirement;
- expiration concentration;
- underlying and factor correlation.

Unknown Greeks or mappings are explicit quality conditions. Maximum loss and
collateral for a proposed order must be deterministically known before
approval. Portfolio overlap must recognize economically related positions,
including leveraged ETFs and correlated AI/data-center names, as shared risk
rather than independent ticker slots.

## 18. Equity and full options construction

Policy chooses direction, final risk budget, and permitted instrument families.
`InstrumentPlanner` converts that result into deterministic candidates:

```python
construct_order(
    intent,
    policy_decision,
    context_snapshot,
    option_chain_snapshot,
) -> OrderRequest
```

Supported structures:

- equity long and short where permitted;
- long calls and puts;
- covered calls;
- cash-secured puts;
- protective puts;
- collars;
- debit and credit verticals;
- calendars and diagonals when broker-valid;
- long straddles and strangles;
- butterflies and iron butterflies;
- condors and iron condors;
- rolls represented as linked close/open workflows.

Generic option requests contain one to four legs. Each leg records underlying,
OCC symbol, call/put, strike, expiry, buy/sell, integer ratio, open/close intent,
quote timestamp, bid, ask, and limit basis.

Candidate filters include:

- exact broker-tradable contract;
- fresh, non-crossed quotes;
- configured maximum bid/ask spread;
- minimum quote size, open interest, and volume where available;
- configured DTE range;
- holding-period compatibility;
- earnings and expiration constraints;
- account option permission;
- determinable maximum loss and collateral.

Candidate scoring may use:

- maximum loss and net debit/credit;
- breakevens;
- available Greeks;
- expected move;
- implied volatility, skew, and term structure;
- theta and vega exposure;
- liquidity;
- dealer context;
- intent horizon and invalidation.

These remain transparent selection features. Probability of profit is null
unless supplied by a separately calibrated and versioned model.

Sizing uses structure risk:

- long option: premium;
- debit spread: net debit;
- defined-risk credit spread: width minus credit;
- butterfly or condor: exact payoff-derived loss;
- cash-secured put: full assignment cash;
- covered call: verified blocks of 100 shares;
- collar: verified shares plus option payoff;
- unknown or unbounded loss: reject.

Naked short options and uncovered ratio spreads are prohibited in every
environment.

A short straddle or strangle is therefore prohibited unless every short call
is covered by verified shares and every short put is cash-secured under an
explicitly modeled covered structure. Debit calendars and diagonals must model
expiration and assignment risk; if their worst-case loss or collateral cannot
be established deterministically, they are rejected.

Mixed equity-option combinations cannot be one atomic Alpaca order. Covered
calls, protective puts, and collars therefore require verified existing shares
or a staged workflow. A roll of a four-leg spread requires a confirmed close
and a separately approved open because an eight-leg atomic order is not
available.

Fallback is explicit in intent and policy:

- use equity;
- use a simpler permitted structure;
- or reject.

The planner never silently substitutes an instrument.

## 19. Execution gateway

Every migrated automated order flows through:

```text
PolicyDecision
  -> InstrumentPlanner
  -> OrderRequest
  -> ExecutionGateway
  -> broker adapter
  -> ExecutionReport
```

Final preflight validates:

- environment and account;
- unexpired policy approval;
- decision-chain hashes;
- latest broker buying power, positions, and pending orders;
- duplicate and conflicting requests;
- market session and order type;
- contract tradability and account option level;
- quote freshness and price bands;
- maximum loss, collateral, and quantity.

The client order ID is deterministic from environment, account alias, decision
ID, order-request hash, and attempt number. An ambiguous submission timeout
causes a broker lookup by client order ID before any retry.

Execution states are append-only:

```text
PLANNED
  -> SUBMISSION_PENDING
  -> ACCEPTED
  -> PARTIALLY_FILLED
  -> FILLED

terminal/recovery:
  REJECTED
  CANCELED
  EXPIRED
  UNKNOWN
  RECONCILIATION_REQUIRED
```

Each event contains parent and leg status, broker IDs, quantities, prices,
broker timestamps, observation time, and a sanitized raw response. Cancel and
replace creates a linked request; it does not mutate history.

## 20. Durable execution journal

Every execution event is written to an independent immutable journal in
addition to PostgreSQL.

Submission sequence:

1. Validate the complete request.
2. Atomically write and fsync `SUBMISSION_INTENT`.
3. Persist the planned request in PostgreSQL.
4. Call the broker.
5. Immediately write the broker response event.
6. Append the execution event in PostgreSQL.

Local development and local QA use one atomic JSON file per event under:

```text
${CYNOLYCUS_OPERATIONAL_ROOT}/execution_journal/YYYY/MM/DD/{event_id}.json
```

The implementation writes a temporary file, fsyncs it, atomically renames it,
and fsyncs the containing directory. One file per event avoids concurrent JSONL
append corruption.

Cloud Run cannot treat its ephemeral filesystem as durable. GCP deployments
write one immutable GCS object per event using the same journal interface and
event identity.

Each event includes:

- event, order, policy, decision, and client order IDs;
- sanitized request or response payload;
- broker IDs and status;
- event and observation timestamps;
- previous hash within the order chain;
- canonical content hash;
- PostgreSQL persistence status.

Journal replay is idempotent. On startup it finds events absent from
PostgreSQL, imports them, queries Alpaca for unresolved client order IDs, and
records reconciliation results.

New entries require both PostgreSQL and the configured durable journal to be
healthy. Risk-reducing exits may continue during a persistence outage and are
recovered from the journal and broker. If the whole application is down, it
cannot create a local event; broker reconciliation remains authoritative for
orders already accepted. Manual orders created while the application is down
return as `UNASSIGNED`.

## 21. Broker reconciliation and ownership

Alpaca is authoritative for account, order, fill, and position facts.
PostgreSQL records observations and strategy attribution.

Rules:

- Submission does not create position ownership.
- Confirmed fills create or adjust ownership.
- Partial fills allocate only filled quantity.
- Exits reduce confirmed ownership proportionally or by explicit lot mapping.
- Unmatched positions are `UNASSIGNED`.
- Reconciliation ambiguity blocks new entries but not risk-reducing exits.

Reconciliation runs:

- at startup;
- before a new entry;
- after submission;
- after partial or final fills;
- on a schedule;
- after journal replay or database recovery.

After-close pending entries store original intent and lineage. At the next
entry window, the system rebuilds context, reevaluates policy, reloads the
option chain, and creates a new superseding request. Stale OCC symbols,
quantities, and limit prices are not reused.

## 22. Strategy migration boundary

The Meta Ranker is the only strategy whose complete order path is activated in
the initial vertical slice. Momentum, HTF, Dealer, SPY intraday, and other
strategies remain inventoried legacy paths until their own adapters and parity
tests are complete.

Migration sequence:

1. Meta.
2. Momentum and HTF.
3. Dealer and future strategies.
4. Remaining automated strategy paths.
5. Disable direct automated broker submission imports.
6. Add a regression test preventing strategy packages from importing broker
   submission clients outside approved adapters.

During transition, legacy paths are explicitly flagged and cannot be mistaken
for nervous-system-governed orders. Final system acceptance requires no
automated strategy to bypass policy and execution. Broker read-only and market
data clients remain permitted.

## 23. Orchestration and events

The MVP reuses:

- `UI/combined_server.py`;
- `UI/nightly_scheduler.py`;
- shared refresh windows;
- startup queue;
- existing job guards.

A thin coordinator composes existing functions:

```text
refresh required producers
  -> record JobRun
  -> persist states
  -> build snapshot
  -> emit Meta intent
  -> evaluate policy
  -> construct instrument
  -> shadow or execute
  -> finalize DecisionRecord
```

Every stage has dependency IDs, input/output IDs, start/end times, heartbeat,
status, and failure reason. Failure of a required stage prevents downstream
entry evaluation.

The in-process typed event bus handles events such as:

```text
StateUpdated
SnapshotCreated
TradeIntentCreated
PolicyDecisionCreated
OrderRequestCreated
ExecutionEventReceived
ReconciliationCompleted
```

The transactional outbox persists each event with the state change before
dispatch. Consumers are idempotent by event ID. Kafka or another external
transport can later implement the same interface without changing contracts.

Cloud Scheduler and Cloud Run replace triggers and process hosting only. They
call the same coordinator functions rather than duplicating trading logic.

## 24. Replay and outcomes

Replay uses the same snapshot, exposure, policy, and instrument construction
functions as QA-paper:

```python
replay_decision(
    decision_time,
    strategy_id,
    source_manifest,
    policy_version,
    config_version,
) -> ReplayResult
```

Replay providers supply historical state and analytical references without live
API calls.

Required properties:

- `available_at <= decision_time` for every input;
- correct calendar, session, and timezone handling;
- frozen source, config, feature, and model versions;
- deterministic snapshot and policy hashes;
- future-append invariance;
- outcomes stored separately from decisions.

Replay can run a historical policy version or a candidate version and compare
them without rewriting the original DecisionRecord.

Options performance evidence is accepted only when based on broker fills or
validated historical quotes/marks. Before P&L use, the source must pass:

- entitlement/tier verification;
- bars or quote coverage;
- stale identical-price checks;
- derivative-versus-underlying directional correlation;
- spread and mark sanity.

Stale trade prints and synthetic underlying returns are not valid option marks.

## 25. Read-only visibility and alerts

Existing dashboards gain nervous-system read views for:

- producer freshness and data quality;
- current market, sector, theme, ticker, catalyst, dealer, and portfolio state;
- snapshot composition and lineage;
- raw intent versus policy decision;
- ordered modifier waterfall;
- instrument candidates and rejection reasons;
- selected option payoff and collateral;
- order, leg, and fill lifecycle;
- PostgreSQL/journal parity;
- broker reconciliation differences;
- job dependencies and failures.

The UI does not become a write authority. Configuration changes remain
version-controlled or use a separately audited administrative path outside the
initial MVP.

Alerts cover:

- required stale or missing state;
- failed or stuck jobs;
- invalid configuration;
- unresolved broker differences;
- journal replay lag;
- PostgreSQL outage;
- journal outage;
- duplicate or ambiguous order;
- attempted production-live submission;
- replay hash mismatch.

## 26. Failure behavior

| Failure | Entry behavior | Exit/recovery behavior |
|---|---|---|
| Required producer stale | Block | Permit broker-backed risk reduction |
| PostgreSQL unavailable | Block | Journal exit, then reconcile |
| Durable journal unavailable | Block | Persist if possible and reconcile broker |
| Broker timeout after POST | Do not retry blindly | Query by client order ID |
| App crashes before POST | No broker order exists | Replay planned record |
| App crashes after POST | Block duplicate | Query broker and append result |
| Full app/server outage | No new decisions | Broker retains accepted orders; reconcile on restart |
| Partial fill | Block conflicting entry | Record filled quantity and manage confirmed exposure |
| Unknown ownership | Block conflicting increase | Allow explicit risk-reducing action |
| Historical import error | Quarantine item | Continue unrelated items and report counts |
| Production-live requested | Hard veto | Read-only broker reconciliation only |

## 27. Security and configuration

- Secrets are never stored in contracts, PostgreSQL payloads, journals, source
  artifacts, or Git.
- Logs and raw broker responses are sanitized before persistence.
- Local secrets stay in ignored environment files.
- GCP credentials use Secret Manager and service accounts with least privilege.
- The database URL is environment-provided.
- Account aliases are stored; private account identifiers are minimized and
  access-controlled.
- QA-paper and production-live credentials, storage, and account identities are
  separate.
- Config snapshots record effective non-secret values and a content hash.

## 28. Verification strategy

### 28.1 Contract tests

- Reject naive timestamps and invalid windows.
- Validate finite numbers and probability bounds.
- Preserve `UNKNOWN`.
- Forbid unknown fields.
- Round-trip every contract through canonical JSON.
- Stabilize content hashes across equivalent serialization.

### 28.2 Persistence tests

- Apply all Alembic migrations to an empty PostgreSQL database.
- Exercise repository save and as-of queries.
- Prove the exclusive validity boundary.
- Prove append-only decision and execution behavior.
- Verify uniqueness and idempotency constraints.
- Upgrade and downgrade migrations where safe.

### 28.3 Import tests

- Inventory representative JSON, JSONL, and CSV sources.
- Preserve artifact hash and row/line lineage.
- Rerun without duplicates.
- Quarantine malformed and causally unusable records.
- Reconcile discovered/imported/skipped/quarantined counts.
- Preserve source files byte-for-byte.

### 28.4 Causality tests

- Reject records available after decision time.
- Verify Meta 14:20 and 16:20 freshness profiles.
- Append future state and prove an earlier snapshot hash is unchanged.
- Prove theme membership, catalyst, and dealer state are historically correct.
- Prove reference price comes from the selected decision bar.

### 28.5 Strategy parity

- Freeze representative current Meta inputs and rankings.
- Require exact ticker order and raw score parity.
- Confirm the intent adapter introduces no ranking or model change.

### 28.6 Policy tests

- Hard veto dominance.
- Stable modifier ordering.
- Budget before/after trace.
- Permission intersection.
- Account and portfolio caps.
- Entry versus risk-reducing exit behavior.
- Shadow counterfactual versus enforce behavior.
- Production-live denial.
- Every change emits a reason code.

### 28.7 Option tests

- Leg orientation and OCC identity.
- Debit/credit sign.
- Breakeven and payoff geometry.
- Maximum loss and collateral.
- Parent quantity and leg ratios.
- Quote freshness and spread filters.
- Deterministic candidate selection.
- Every supported structure.
- Naked short and uncovered ratio rejection.
- Staged covered/protective/collar and roll workflows.

### 28.8 Gateway and recovery tests

- Duplicate client order prevention.
- Broker rejection and partial fill handling.
- Ambiguous timeout lookup.
- Cancel/replace lineage.
- Crash before submission.
- Crash after broker acceptance but before database append.
- Local and GCS journal idempotent replay.
- Broker/PostgreSQL ownership reconciliation.
- Manual order import as `UNASSIGNED`.

### 28.9 Replay and source-fitness tests

- Historical decision hash reproduction.
- Candidate policy comparison without record mutation.
- Future-append invariance.
- No live API use during replay.
- Options quote coverage, staleness, and derivative correlation checks before
  P&L.

### 28.10 End-to-end acceptance

The MVP requires:

- every evaluated Meta intent linked to a DecisionRecord;
- every submitted order linked to an approved PolicyDecision;
- every submitted order present in PostgreSQL and the durable journal;
- zero duplicate submission in retry and crash tests;
- exact Meta ranking parity before policy;
- exact historical snapshot reconstruction by hash;
- no future-availability leakage;
- complete broker ownership reconciliation or explicit `UNASSIGNED` state;
- a visible reason and budget waterfall for every policy change;
- deterministic payoff tests for the full supported options suite;
- production-live submission denied.

## 29. Rollout and rollback

Rollout:

```text
historical import dry run
  -> historical import and parity
  -> deterministic replay
  -> live-data shadow
  -> QA-paper hard-safety enforcement
  -> QA-paper selected contextual enforcement
  -> Momentum and HTF adapters
  -> Dealer and remaining strategy adapters
  -> direct automated submission paths disabled
```

Context rules move from shadow to enforce individually through versioned
configuration after replay and QA-paper evidence. There is no all-at-once
context switch.

Rollback:

- disable new entries through a kill switch;
- retain broker reconciliation;
- retain risk-reducing exits;
- retain journals and PostgreSQL evidence;
- revert to the previous contract/config version where schema-compatible;
- never reactivate an ungoverned direct entry path.

## 30. Full-system roadmap after the MVP

1. Migrate Momentum and HTF intent adapters.
2. Complete canonical portfolio factor and weighted-theme aggregation.
3. Expand versioned market, sector, theme, catalyst, and dealer engines.
4. Integrate Dealer and intraday strategies.
5. Evaluate policy-added value by regime and strategy.
6. Calibrate genuine probabilities where useful and scientifically justified.
7. Complete Cloud SQL, GCS journal, and Cloud Run operational deployment.
8. Accumulate sustained QA-paper evidence.
9. Consider production-live only as a separate explicitly approved,
   fail-closed project.

## 31. External constraints verified during design

- Alpaca multi-leg options use `order_class="mleg"` and support up to four
  option legs:
  <https://docs.alpaca.markets/us/docs/options-level-3-trading>
- Alpaca order creation reference:
  <https://docs.alpaca.markets/us/v1.4.2/reference/postorder>
- Alpaca options overview:
  <https://docs.alpaca.markets/us/docs/options-trading-overview>
- Cloud Run can connect to Cloud SQL using encrypted Unix sockets or supported
  connectors:
  <https://docs.cloud.google.com/sql/docs/postgres/connect-run>
- Google Database Migration Service exists for later PostgreSQL lift-and-shift,
  but the MVP can recreate its small operational history from immutable
  artifacts:
  <https://docs.cloud.google.com/database-migration/docs/overview>

## 32. Final design decisions

- Keep the package name `contracts`, despite the classes being Pydantic models,
  to avoid confusion with trained ML models.
- Place the modular monolith under `core/nervous_system/`.
- Keep adapters beside existing domain producers.
- Import all available operational history, not only forward data.
- Retain Parquet for analytical history.
- Use local Docker PostgreSQL for development and Cloud SQL for QA-paper.
- Include the full approved options suite and one-to-four-leg order model.
- Reject naked short options and uncovered ratio spreads everywhere.
- End the MVP in QA-paper; production-live is only a denied flag.
- Record every execution event in PostgreSQL and a separate durable journal.
- Preserve broker authority and assign strategy ownership only from confirmed
  fills.
- Preserve the current Meta ranking before policy.
- Enforce operational safety before contextual modulation.
- Require contextual modulation to alter the real downstream order when its
  versioned rule is placed in enforce mode.
