# Cynolycus Nervous System MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build one causal, auditable Meta Ranker decision path from shared state through policy, full equity/options construction, QA-paper execution, durable journaling, broker reconciliation, replay, and Cloud SQL portability.

**Architecture:** Add a modular monolith under `core/nervous_system/`; leave existing market, theme, catalyst, dealer, and strategy producers in place behind inward-facing adapters. PostgreSQL owns operational state and decisions, Parquet remains analytical authority, and Alpaca remains authoritative for account/order/fill/position facts. The first enforced vertical slice is Meta; later strategy migrations are a gated roadmap.

**Tech Stack:** Python 3.12, Pydantic v2, SQLAlchemy 2.x, psycopg 3, Alembic, PostgreSQL 16, pytest, Pandas/PyArrow, Alpaca REST, Docker Compose, Google Cloud SQL, Google Cloud Storage.

## Global Constraints

- Work from the approved design in `docs/superpowers/specs/2026-07-30-cynolycus-nervous-system-mvp-design.md`.
- Use the repository-root package layout; do not introduce a `src/` tree.
- Put shared code under `core/nervous_system/`; keep producer adapters beside their existing domains.
- Preserve current unrelated worktree edits, especially the 2026-07-30 readiness/regime and after-close deferral fixes.
- At execution time, use `superpowers:using-git-worktrees` and start from the latest committed `nervous-system` branch after outstanding local changes have been intentionally resolved.
- Execute with LUNA subagents under `superpowers:subagent-driven-development`;
  give each worker one task/commit boundary and require coordinator review
  before advancing dependent tasks.
- Use TDD: add a failing focused test, observe the expected failure, implement the smallest complete behavior, and rerun focused tests before broader tests.
- Every persisted decision-time input must have timezone-aware UTC `available_at`; never substitute `as_of`.
- Snapshot eligibility is exactly `available_at <= decision_time < valid_until`.
- Never infer unavailable historical timestamps from file modification time.
- Preserve raw artifacts byte-for-byte; imports add lineage and quarantine records without rewriting sources.
- Keep Parquet authoritative for bars, feature matrices, historical option snapshots, and backtests.
- Keep Alpaca authoritative for account, order, fill, and position facts.
- Assign strategy ownership only from broker-confirmed fills; unmatched positions are `UNASSIGNED`.
- Preserve exact Meta raw ranking and selected-bar behavior before policy.
- Do not map `s_combo`, z-scores, ranks, dealer heuristics, or other uncalibrated values to probability fields.
- Support equity and one-to-four-leg option orders, including the approved spread suite.
- Always reject naked short options, uncovered ratio spreads, and any structure with unknown maximum loss or collateral.
- `DEVELOPMENT` is simulated, `QA_PAPER` is Alpaca paper only, and `PRODUCTION_LIVE` is represented but hard-vetoed.
- All new entries require healthy PostgreSQL and durable journal sinks; risk-reducing exits remain fail-operational and reconcile afterward.
- Never blindly retry an ambiguous broker POST; query by deterministic client order ID first.
- No automated strategy may bypass the gateway after its migration gate.
- Do not claim policy or option profitability without replay/QA-paper evidence and valid fill/mark source checks.
- Append a maximum-three-line entry to `LIVING_SUMMARY.md` at each substantive implementation endpoint.

---

## Execution map

```text
Tasks 1-7: contracts, config, PostgreSQL, repositories
        |
        +--> Task 8: historical import
        |
        +--> Tasks 9-12 in parallel: producer adapters
                          |
                          v
Task 13: snapshots --> Task 14: Meta intent parity
                          |
              +-----------+-----------+
              v                       v
        Task 15 policy         Task 16 portfolio
              |                       |
              +-----------+-----------+
                          v
              Tasks 17-18 options
                          |
              Tasks 19-21 broker/journal/gateway
                          |
              Tasks 22-23 orchestration + Meta cutover
                          |
              Tasks 24-26 replay/UI/GCP
                          |
                    Task 27 acceptance
```

Tasks 9-12 are safe LUNA parallel work after Task 7. Tasks 15-16 can run in
parallel after Task 14. Tasks 17, 19, and 20 can begin in parallel after Task
4. Task 18 requires Tasks 16-17. Task 21 requires Tasks 15, 18, 19, and 20 and
is the synchronization point before orchestration.

## Planned file map

```text
core/nervous_system/
├── __init__.py
├── contracts/
│   ├── __init__.py
│   ├── base.py
│   ├── enums.py
│   ├── quality.py
│   ├── states.py
│   ├── context.py
│   ├── intent.py
│   ├── policy.py
│   ├── orders.py
│   ├── execution.py
│   └── decisions.py
├── config/
│   ├── __init__.py
│   ├── runtime.py
│   ├── freshness.py
│   ├── policy.py
│   ├── portfolio.py
│   ├── options.py
│   ├── source_fitness.py
│   └── legacy_sources.toml
├── persistence/
│   ├── __init__.py
│   ├── database.py
│   ├── uow.py
│   ├── alembic.ini
│   ├── migrations/
│   │   ├── env.py
│   │   ├── script.py.mako
│   │   └── versions/
│   │       ├── 0001_state_registry.py
│   │       └── 0002_decision_execution.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── state.py
│   │   ├── registry.py
│   │   ├── decision.py
│   │   ├── execution.py
│   │   └── operations.py
│   └── repositories/
│       ├── __init__.py
│       ├── state.py
│       ├── decision.py
│       ├── execution.py
│       ├── registry.py
│       └── operations.py
├── data_registry/
│   ├── __init__.py
│   ├── artifacts.py
│   ├── lineage.py
│   ├── parsers.py
│   ├── legacy_adapters.py
│   └── import_legacy.py
├── context/
│   ├── __init__.py
│   ├── requirements.py
│   ├── readiness_adapter.py
│   └── snapshot_builder.py
├── policy/
│   ├── __init__.py
│   ├── reason_codes.py
│   ├── permissions.py
│   ├── rules.py
│   └── engine.py
├── portfolio/
│   ├── __init__.py
│   ├── exposure.py
│   ├── ownership.py
│   └── reconciliation.py
├── execution/
│   ├── __init__.py
│   ├── broker.py
│   ├── alpaca_adapter.py
│   ├── journal.py
│   ├── gateway.py
│   ├── reconciliation.py
│   ├── pending.py
│   └── options/
│       ├── __init__.py
│       ├── quotes.py
│       ├── payoff.py
│       ├── structures.py
│       └── selector.py
├── orchestration/
│   ├── __init__.py
│   ├── events.py
│   ├── jobs.py
│   ├── outbox.py
│   ├── read_models.py
│   ├── alerts.py
│   └── coordinator.py
├── replay/
│   ├── __init__.py
│   ├── providers.py
│   ├── source_fitness.py
│   └── runner.py
└── tests/
    ├── conftest.py
    ├── fixtures/
    └── test_*.py
```

Primary existing files modified by the MVP are:

```text
requirements.txt
pyproject.toml
.env.example
compose.nervous-system.yaml
core/API/Alpaca_API/options/options_api.py
core/broker_equity_snapshot.py
core/live_4h_exec.py
signals/market_regime/build.py
themes/dynamic_theme/config.py
themes/dynamic_theme/pipeline.py
themes/dynamic_theme/stages/step08_memberships.py
themes/dynamic_theme/tests/test_seed_and_stability.py
signals/news/schema.py
signals/news/pipeline.py
signals/events/schema.py
signals/events/collectors.py
signals/catalysts/pipeline.py
signals/meta_context/meta_ranker/score.py
signals/meta_context/meta_ranker/live_runner.py
signals/meta_context/meta_ranker/options_exec.py
signals/meta_context/meta_ranker/run_4h_loop.py
strategies/intraday_structure/replay.py
UI/meta_ranker_dashboard.py
UI/combined_server.py
docs/GCP_MIGRATION_TUTORIAL.md
LIVING_SUMMARY.md
```

All created adapter, test, script, operations, and acceptance files are named
in their owning task below.

### Task 1: Package, dependencies, and local PostgreSQL

**Files:**

- Create: `core/nervous_system/__init__.py`
- Create: package `__init__.py` files shown in the planned file map
- Create: `core/nervous_system/tests/test_package_layout.py`
- Create: `compose.nervous-system.yaml`
- Create: `.env.example`
- Modify: `requirements.txt`
- Modify: `pyproject.toml`

**Interfaces:**

- Consumes: Python 3.12 and the current repository-root import layout.
- Produces: importable `core.nervous_system`, PostgreSQL 16 on
  `127.0.0.1:55432`, and pytest discovery under
  `core/nervous_system/tests`.

- [ ] **Step 1: Add the failing package-layout test**

```python
from importlib import import_module
from pathlib import Path


def test_nervous_system_package_and_required_subpackages_exist():
    root = Path("core/nervous_system")
    names = (
        "contracts", "config", "persistence", "data_registry", "context",
        "policy", "portfolio", "execution", "orchestration", "replay",
    )
    assert root.is_dir()
    for name in names:
        import_module(f"core.nervous_system.{name}")
```

- [ ] **Step 2: Run the test and verify the expected failure**

Run:

```bash
./.venv/bin/python -m pytest core/nervous_system/tests/test_package_layout.py -q
```

Expected: collection or assertion failure because `core/nervous_system` does
not exist.

- [ ] **Step 3: Add pinned dependency ranges and pytest discovery**

Append to `requirements.txt`:

```text
pydantic>=2.11,<3
SQLAlchemy>=2.0,<3
psycopg[binary]>=3.2,<4
alembic>=1.16,<2
google-cloud-storage>=3,<4
```

Add `"core/nervous_system/tests"` and
`"core/API/Alpaca_API/tests"` to `tool.pytest.ini_options.testpaths`. Add:

```toml
"postgres: tests requiring the local PostgreSQL integration database",
```

to the marker list.

- [ ] **Step 4: Create the package directories and minimal exports**

Each package `__init__.py` contains only a one-line module docstring. The root
file contains:

```python
"""Shared causal context, policy, portfolio, and execution nervous system."""

__all__: list[str] = []
```

- [ ] **Step 5: Add the local database definition**

Create `compose.nervous-system.yaml`:

```yaml
services:
  postgres:
    image: postgres:16
    environment:
      POSTGRES_DB: cynolycus
      POSTGRES_USER: cynolycus
      POSTGRES_PASSWORD: cynolycus_dev_only
    ports:
      - "127.0.0.1:55432:5432"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U cynolycus -d cynolycus"]
      interval: 2s
      timeout: 3s
      retries: 20
    volumes:
      - cynolycus_nervous_system_pg:/var/lib/postgresql/data

volumes:
  cynolycus_nervous_system_pg:
```

Create `.env.example` without real credentials:

```dotenv
CYNOLYCUS_ENVIRONMENT=development
CYNOLYCUS_NERVOUS_SYSTEM_MODE=off
CYNOLYCUS_DATABASE_URL=postgresql+psycopg://cynolycus:cynolycus_dev_only@127.0.0.1:55432/cynolycus
CYNOLYCUS_DB_POOL_SIZE=5
CYNOLYCUS_DB_MAX_OVERFLOW=5
CYNOLYCUS_OPERATIONAL_ROOT=Data/operational/nervous_system
CYNOLYCUS_EXECUTION_JOURNAL=local
CYNOLYCUS_EXECUTION_JOURNAL_BUCKET=
CYNOLYCUS_ACCOUNT_ALIAS=paper
```

- [ ] **Step 6: Install dependencies and verify package and database health**

Run:

```bash
./.venv/bin/pip install -r requirements.txt
./.venv/bin/python -m pytest core/nervous_system/tests/test_package_layout.py -q
docker compose -f compose.nervous-system.yaml up -d postgres
docker compose -f compose.nervous-system.yaml exec postgres pg_isready -U cynolycus -d cynolycus
```

Expected: test passes and `pg_isready` reports `accepting connections`.

- [ ] **Step 7: Commit**

```bash
git add requirements.txt pyproject.toml .env.example compose.nervous-system.yaml core/nervous_system
git commit -m "build: scaffold nervous system runtime"
```

### Task 2: Common contracts, UTC time, quality, and canonical hashes

**Files:**

- Create: `core/nervous_system/contracts/base.py`
- Create: `core/nervous_system/contracts/enums.py`
- Create: `core/nervous_system/contracts/quality.py`
- Modify: `core/nervous_system/contracts/__init__.py`
- Create: `core/nervous_system/tests/test_contract_base.py`

**Interfaces:**

- Consumes: Pydantic v2.
- Produces:
  `ContractModel`, `UtcDatetime`, `FiniteFloat`, `Probability`,
  `canonical_json(model) -> str`, `content_hash(model) -> str`,
  `LineageRef`, `DataQualityIssue`, and `DataQualitySummary`.

- [ ] **Step 1: Write failing validation and hashing tests**

```python
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from core.nervous_system.contracts.base import ContractModel, UtcDatetime, content_hash
from core.nervous_system.contracts.quality import DataQualitySummary


class Example(ContractModel):
    observed_at: UtcDatetime
    value: float


def test_contract_normalizes_aware_time_to_utc_and_hashes_stably():
    model = Example(
        observed_at=datetime.fromisoformat("2026-07-30T10:00:00-04:00"),
        value=1.25,
    )
    assert model.observed_at == datetime(2026, 7, 30, 14, tzinfo=timezone.utc)
    assert content_hash(model) == content_hash(Example.model_validate_json(model.model_dump_json()))


def test_contract_rejects_naive_time_unknown_fields_and_nonfinite_number():
    with pytest.raises(ValidationError):
        Example(observed_at=datetime(2026, 7, 30, 10), value=1.0)
    with pytest.raises(ValidationError):
        Example(
            observed_at=datetime(2026, 7, 30, 14, tzinfo=timezone.utc),
            value=float("nan"),
            surprise=True,
        )


def test_empty_quality_summary_is_explicitly_healthy():
    quality = DataQualitySummary()
    assert quality.is_usable is True
    assert quality.issues == ()
```

- [ ] **Step 2: Run and verify import failure**

Run:

```bash
./.venv/bin/python -m pytest core/nervous_system/tests/test_contract_base.py -q
```

Expected: FAIL because the contract modules are absent.

- [ ] **Step 3: Implement the strict base types and canonical serialization**

Use these definitions in `base.py`:

```python
from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from typing import Annotated, Any

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, model_validator


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")
    return value.astimezone(timezone.utc)


def _finite(value: float) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("number must be finite")
    return value


UtcDatetime = Annotated[datetime, AfterValidator(_utc)]
FiniteFloat = Annotated[float, AfterValidator(_finite)]
Probability = Annotated[FiniteFloat, Field(ge=0.0, le=1.0)]
PositiveSchemaVersion = Annotated[int, Field(ge=1)]


class ContractModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    @model_validator(mode="after")
    def reject_nonfinite_recursively(self) -> "ContractModel":
        def visit(value: Any) -> None:
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError("contract contains a non-finite number")
            if isinstance(value, dict):
                for item in value.values():
                    visit(item)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    visit(item)

        visit(self.model_dump())
        return self


def canonical_json(model: ContractModel) -> str:
    payload = model.model_dump(mode="json", exclude_none=False)
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def content_hash(model: ContractModel) -> str:
    return hashlib.sha256(canonical_json(model).encode("utf-8")).hexdigest()
```

- [ ] **Step 4: Implement all required enums with `UNKNOWN`**

`enums.py` defines `str, Enum` classes:

```text
Direction: UNKNOWN, LONG, SHORT, NEUTRAL
MarketRegime: UNKNOWN, STRONG_RISK_ON, RISK_ON, NEUTRAL, DETERIORATING, RISK_OFF, CRISIS
ThemeRegime: UNKNOWN, LEADERSHIP, ACCUMULATION, HEALTHY, NEUTRAL, DETERIORATING, DISTRIBUTION, LIQUIDATION
TickerSetup: UNKNOWN, BREAKOUT, PULLBACK_IN_UPTREND, MOMENTUM_CONTINUATION, CONFIRMED_REVERSAL, COUNTERTREND_BOUNCE, FAILED_RECLAIM, BREAKDOWN, EXHAUSTION, RANGE_BOUND
DealerRegime: UNKNOWN, POSITIVE_GAMMA, NEUTRAL_GAMMA, SHORT_GAMMA, PINNING, UPSIDE_ACCELERATION, DOWNSIDE_ACCELERATION
PolicyAction: APPROVE, APPROVE_REDUCED, REJECT, DEFER, EXIT, REDUCE, HEDGE
RuntimeEnvironment: DEVELOPMENT, QA_PAPER, PRODUCTION_LIVE
PolicyMode: OFF, SHADOW, ENFORCE
DecisionKind: ENTRY, ADJUSTMENT, EXIT
StateType: MARKET, SECTOR, THEME, TICKER, CATALYST_EVENT, CATALYST_PRESSURE, DEALER, PORTFOLIO, READINESS
AssetClass: EQUITY, OPTION
OptionType: CALL, PUT
OrderSide: BUY, SELL
PositionIntent: BUY_TO_OPEN, BUY_TO_CLOSE, SELL_TO_OPEN, SELL_TO_CLOSE
DebitCredit: DEBIT, CREDIT
ExecutionStatus: PLANNED, SUBMISSION_PENDING, ACCEPTED, PARTIALLY_FILLED, FILLED, REJECTED, CANCELED, EXPIRED, UNKNOWN, RECONCILIATION_REQUIRED
SubmissionAttemptStatus: RESERVED, JOURNALED, SUBMITTING, ACCEPTED, REJECTED, AMBIGUOUS, RECONCILIATION_REQUIRED
InstrumentFamily: EQUITY, SINGLE_OPTION, VERTICAL, CALENDAR, DIAGONAL, STRADDLE, STRANGLE, BUTTERFLY, IRON_BUTTERFLY, CONDOR, IRON_CONDOR, COVERED_CALL, CASH_SECURED_PUT, PROTECTIVE_PUT, COLLAR, ROLL
MissingStateAction: REJECT, WARN, OMIT
ModifierOperation: MULTIPLY, CAP
DataQualitySeverity: INFO, WARNING, ERROR, CRITICAL
```

- [ ] **Step 5: Implement quality and lineage contracts**

```python
class LineageRef(ContractModel):
    source_id: str
    content_hash: str
    record_locator: str | None = None


class DataQualityIssue(ContractModel):
    code: str
    severity: DataQualitySeverity
    component: str
    message: str
    fallback_used: str | None = None


class DataQualitySummary(ContractModel):
    issues: tuple[DataQualityIssue, ...] = ()

    @property
    def is_usable(self) -> bool:
        return not any(
            issue.severity in {DataQualitySeverity.ERROR, DataQualitySeverity.CRITICAL}
            for issue in self.issues
        )
```

- [ ] **Step 6: Run focused contract tests**

Run:

```bash
./.venv/bin/python -m pytest core/nervous_system/tests/test_contract_base.py -q
```

Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add core/nervous_system/contracts core/nervous_system/tests/test_contract_base.py
git commit -m "feat: add strict nervous system contract foundation"
```

### Task 3: State and context contracts

**Files:**

- Create: `core/nervous_system/contracts/states.py`
- Create: `core/nervous_system/contracts/context.py`
- Modify: `core/nervous_system/contracts/__init__.py`
- Create: `core/nervous_system/tests/test_state_contracts.py`

**Interfaces:**

- Consumes: Task 2 base types, enums, lineage, and quality.
- Produces:
  `StateEnvelope`, `MarketState`, `SectorState`, `ThemeMembership`,
  `ThemeState`, `TickerState`, `CatalystEvent`, `CatalystPressure`,
  `DealerState`, `PortfolioPosition`, `PortfolioState`, `ReadinessState`,
  `StateRequest`, `FreshnessResult`, and `ContextSnapshot`.

- [ ] **Step 1: Write failing state-window and snapshot tests**

```python
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest
from pydantic import ValidationError

from core.nervous_system.contracts.context import ContextSnapshot
from core.nervous_system.contracts.enums import MarketRegime, StateType
from core.nervous_system.contracts.quality import DataQualitySummary
from core.nervous_system.contracts.states import MarketState

UTC = timezone.utc


def market_state() -> MarketState:
    available = datetime(2026, 7, 29, 20, 30, tzinfo=UTC)
    return MarketState(
        state_id=uuid4(),
        state_type=StateType.MARKET,
        entity_id="US",
        as_of=datetime(2026, 7, 29, 20, 0, tzinfo=UTC),
        available_at=available,
        generated_at=available,
        valid_until=available + timedelta(days=2),
        source_window_start=datetime(2025, 7, 29, 20, 0, tzinfo=UTC),
        source_window_end=datetime(2026, 7, 29, 20, 0, tzinfo=UTC),
        schema_version=1,
        producer="signals.market_regime",
        model_version="rules@1",
        feature_version="market-regime@1",
        config_version="market-regime@1",
        lineage_ids=(),
        data_quality=DataQualitySummary(),
        regime=MarketRegime.NEUTRAL,
        risk_on_probability=None,
        risk_off_probability=None,
        metrics={"risk_appetite_z": -0.2},
        reason_codes=("MARKET_REGIME_RULES_ONLY",),
    )


def test_state_rejects_nonexclusive_or_reversed_validity_window():
    state = market_state()
    with pytest.raises(ValidationError):
        state.model_copy(update={"valid_until": state.available_at})


def test_snapshot_embeds_state_and_hash_references():
    state = market_state()
    snapshot = ContextSnapshot.from_states(
        snapshot_id=uuid4(),
        decision_time=datetime(2026, 7, 30, 18, 20, tzinfo=UTC),
        strategy_id="meta_ranker",
        ticker="AMD",
        states=(state,),
        freshness_profile="meta_4h_1420@1",
    )
    assert snapshot.market_state == state
    assert snapshot.state_ids == (state.state_id,)
```

- [ ] **Step 2: Run and verify missing-contract failure**

Run:

```bash
./.venv/bin/python -m pytest core/nervous_system/tests/test_state_contracts.py -q
```

Expected: FAIL because `states.py` and `context.py` do not exist.

- [ ] **Step 3: Implement the common state envelope**

```python
class StateEnvelope(ContractModel):
    state_id: UUID
    state_type: StateType
    entity_id: str
    as_of: UtcDatetime
    available_at: UtcDatetime
    generated_at: UtcDatetime
    valid_until: UtcDatetime
    source_window_start: UtcDatetime
    source_window_end: UtcDatetime
    schema_version: PositiveSchemaVersion
    producer: str
    model_version: str
    feature_version: str
    config_version: str
    lineage_ids: tuple[str, ...]
    data_quality: DataQualitySummary

    @model_validator(mode="after")
    def validate_time_window(self) -> "StateEnvelope":
        if self.source_window_end > self.as_of:
            raise ValueError("source window ends after as_of")
        if self.available_at > self.generated_at:
            raise ValueError("generated_at precedes available_at")
        if self.valid_until <= self.available_at:
            raise ValueError("valid_until must be exclusive and after available_at")
        return self
```

- [ ] **Step 4: Implement explicit state payload fields**

Use these exact fields:

```text
MarketState:
  regime, risk_on_probability?, risk_off_probability?,
  metrics: dict[str, FiniteFloat],
  transition_probabilities: dict[str, Probability],
  reason_codes

SectorState:
  sector_id, sector_regime, relative_strength?, breadth?, momentum?,
  volatility?, rotation_rank?, rank_change?, capital_flow_direction,
  transition_probabilities

ThemeMembership:
  ticker, theme_id, weight, membership_version, effective_from, effective_until?

ThemeState:
  theme_id, theme_regime, relative_strength?, breadth?, momentum?,
  distribution_score?, correlation_score?, volatility_score?,
  catalyst_pressure?, dealer_fragility?, leadership_score?, rotation_rank?,
  transition_probabilities

TickerState:
  ticker, selected_bar, reference_price, ticker_setup, trend_state,
  relative_strength_state, support_state, volume_state, reversal_state,
  breakdown_state, theme_alignment?, market_alignment?, dealer_alignment?,
  metrics, transition_probabilities

CatalystEvent:
  event_id, ticker?, event_type, event_time?, published_at?,
  observed_at, available_at inherited from envelope,
  source, headline?, channel, relation_confidence?, is_direct?

CatalystPressure:
  scope_type, scope_id, channel_scores, aggregate_score?,
  event_ids, transition_probabilities

DealerState:
  ticker, dealer_regime, spot, total_gex?, call_wall?, put_wall?,
  nearest_magnet?, gamma_flip?, air_gap_above_score?,
  air_gap_below_score?, pinning_score?, acceleration_score?, metrics

PortfolioPosition:
  broker_position_id, symbol, underlying, asset_class, quantity,
  average_entry_price?, current_price?, market_value?,
  strategy_id?, ownership_status

PortfolioState:
  account_alias, equity, cash, buying_power, day_pl?,
  positions, open_order_ids, broker_observed_at

ReadinessState:
  job, status, ready, completed_at, checked_at, max_age_hours,
  latest_required_session, reason_codes
```

All optional numeric values use `FiniteFloat`; probability maps use
`Probability`. Use `UNKNOWN` enum/string values instead of inventing
categorical state.

- [ ] **Step 5: Implement snapshot fields and `from_states`**

`StateRequest` contains:

```text
state_type, entity_id, required, bar_bound
```

`FreshnessResult` contains:

```text
state_type, entity_id, required, status, selected_state_id?, age_seconds?,
max_age_seconds, reason_code
```

`ContextSnapshot` contains:

```text
snapshot_id, decision_time, strategy_id, ticker, freshness_profile,
market_state?, sector_states, theme_memberships, theme_states,
ticker_state?, catalyst_events, catalyst_pressures, dealer_state?,
portfolio_state?, readiness_state?, state_ids, state_hashes, stale_inputs, missing_inputs,
data_quality, config_version, model_versions, feature_versions,
schema_version, content_hash
```

`from_states` must dispatch by concrete state type, calculate stable
`state_hashes`, and reject any state with
`available_at > decision_time` or `decision_time >= valid_until`.

- [ ] **Step 6: Run focused tests and serialization round trips**

Run:

```bash
./.venv/bin/python -m pytest core/nervous_system/tests/test_state_contracts.py core/nervous_system/tests/test_contract_base.py -q
```

Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add core/nervous_system/contracts core/nervous_system/tests/test_state_contracts.py
git commit -m "feat: define shared context state contracts"
```

### Task 4: Trading, policy, order, execution, and decision contracts

**Files:**

- Create: `core/nervous_system/contracts/intent.py`
- Create: `core/nervous_system/contracts/policy.py`
- Create: `core/nervous_system/contracts/orders.py`
- Create: `core/nervous_system/contracts/execution.py`
- Create: `core/nervous_system/contracts/decisions.py`
- Modify: `core/nervous_system/contracts/__init__.py`
- Create: `core/nervous_system/tests/test_trading_contracts.py`

**Interfaces:**

- Consumes: Tasks 2-3 types.
- Produces:
  `TradeIntent`, `PolicyModifier`, `PolicyDecision`, `OptionLeg`,
  `OrderRequest`, `ExecutionEvent`, `ExecutionReport`, `DecisionRecord`,
  `HashedDecisionArtifact`, and `DecisionOutcome`.

- [ ] **Step 1: Write failing probability, leg-count, and decision-chain tests**

```python
from datetime import datetime, timezone
from decimal import Decimal
from uuid import uuid4

import pytest
from pydantic import ValidationError

from core.nervous_system.contracts.enums import (
    AssetClass, DebitCredit, Direction, InstrumentFamily, OptionType,
    OrderSide, PolicyAction, PositionIntent, RuntimeEnvironment,
)
from core.nervous_system.contracts.intent import TradeIntent
from core.nervous_system.contracts.orders import OptionLeg, OrderRequest

NOW = datetime(2026, 7, 30, 18, 20, tzinfo=timezone.utc)


def test_uncalibrated_meta_intent_keeps_probability_null():
    intent = TradeIntent(
        intent_id=uuid4(), strategy_id="meta_ranker", ticker="AMD",
        direction=Direction.LONG, decision_kind="ENTRY", raw_score=0.97,
        raw_probability=None, expected_return=None, expected_holding_period="53x4h",
        entry_window="current-or-next-open", preferred_entry=None,
        invalidation=None, target=None, stop=None,
        position_size_requested=Decimal("5000"),
        instrument_preferences=(InstrumentFamily.VERTICAL, InstrumentFamily.EQUITY),
        feature_timestamp=NOW, created_at=NOW,
        model_version="meta-combo@current", feature_version="meta-matrix@current",
        reason_codes=("META_TOP_K",),
    )
    assert intent.raw_probability is None
    with pytest.raises(ValidationError):
        intent.model_copy(update={"raw_probability": 1.01})


def test_order_rejects_more_than_four_option_legs():
    leg = OptionLeg(
        symbol="AMD260821C00200000", underlying="AMD", option_type=OptionType.CALL,
        strike=Decimal("200"), expiration="2026-08-21", side=OrderSide.BUY,
        ratio=1, position_intent=PositionIntent.BUY_TO_OPEN,
        quote_at=NOW, bid=Decimal("4.90"), ask=Decimal("5.10"),
    )
    with pytest.raises(ValidationError):
        OrderRequest(
            order_request_id=uuid4(), decision_id=uuid4(), policy_decision_id=uuid4(),
            environment=RuntimeEnvironment.QA_PAPER, account_alias="paper",
            instrument_family=InstrumentFamily.VERTICAL, legs=(leg, leg, leg, leg, leg),
            parent_quantity=1, debit_credit=DebitCredit.DEBIT,
            net_limit_price=Decimal("5.00"), maximum_loss=Decimal("500"),
            buying_power_required=Decimal("500"), time_in_force="day",
            order_type="limit", idempotency_key="ns-test", request_hash="a" * 64,
            created_at=NOW, expires_at=NOW.replace(hour=19),
        )
```

- [ ] **Step 2: Run and verify missing-contract failure**

Run:

```bash
./.venv/bin/python -m pytest core/nervous_system/tests/test_trading_contracts.py -q
```

Expected: FAIL because trading contract modules are absent.

- [ ] **Step 3: Implement `TradeIntent` and policy contracts**

`TradeIntent` uses the fields shown in the test plus:

```text
snapshot_id?
selected_bar
entry_window
preferred_entry?
invalidation?
target?
stop?
```

Money and price values use `Decimal`; score and expected return values use
`FiniteFloat`; only `raw_probability` uses `Probability`.

Define:

```python
class PolicyModifier(ContractModel):
    rule_id: str
    rule_version: str
    operation: ModifierOperation
    input_value: str
    configured_condition: str
    configured_value: Decimal
    budget_before: Decimal
    budget_after: Decimal
    reason_code: str
    source_state_id: UUID | None
    config_version: str


class PolicyDecision(ContractModel):
    policy_decision_id: UUID
    intent_id: UUID
    snapshot_id: UUID
    environment: RuntimeEnvironment
    mode: PolicyMode
    action: PolicyAction
    approved_direction: Direction
    base_risk_budget: Decimal
    final_risk_budget: Decimal
    allowed_instruments: frozenset[InstrumentFamily]
    hard_vetoes: tuple[str, ...]
    modifiers: tuple[PolicyModifier, ...]
    stop_adjustment: FiniteFloat | None
    target_adjustment: FiniteFloat | None
    holding_period_adjustment: int | None
    hedge_requirement: str | None
    reason_codes: tuple[str, ...]
    policy_version: str
    config_version: str
    created_at: UtcDatetime
    expires_at: UtcDatetime
```

Validate `final_risk_budget >= 0`, reject `APPROVE` with hard vetoes, and
require zero budget for `REJECT`.

- [ ] **Step 4: Implement option legs and generic requests**

`OptionLeg` fields are exactly those in the test. Validate positive strike,
positive ratio, bid/ask nonnegative, ask not below bid, and expiration not
before `quote_at.date()`.

`OrderRequest` supports either:

- equity: `equity_symbol`, `equity_side`, no option legs; or
- options: one-to-four `legs`, no equity symbol.

Validate:

- `expires_at > created_at`;
- positive parent quantity;
- exact 64-character SHA-256 request hash;
- `maximum_loss` and `buying_power_required` are nonnegative;
- credit requests have positive credit limit magnitude;
- no mixed equity and option payload.

- [ ] **Step 5: Implement append-only execution and decision contracts**

```python
class ExecutionEvent(ContractModel):
    execution_event_id: UUID
    order_request_id: UUID
    status: ExecutionStatus
    observed_at: UtcDatetime
    broker_event_at: UtcDatetime | None
    client_order_id: str
    broker_order_id: str | None
    broker_parent_order_id: str | None
    filled_quantity: Decimal
    average_fill_price: Decimal | None
    leg_reports: tuple[dict[str, str], ...]
    sanitized_response: dict[str, object]
    previous_event_hash: str | None
    event_hash: str


class ExecutionReport(ContractModel):
    order_request_id: UUID
    events: tuple[ExecutionEvent, ...]
    current_status: ExecutionStatus


class HashedDecisionArtifact(ContractModel):
    artifact_type: str
    schema_version: PositiveSchemaVersion
    content_hash: str
    payload: dict[str, object]


class DecisionRecord(ContractModel):
    decision_record_id: UUID
    decision_time: UtcDatetime
    snapshot_id: UUID
    intent_id: UUID
    policy_decision_id: UUID
    order_request_ids: tuple[UUID, ...]
    source_manifest_hash: str
    snapshot_hash: str
    intent_hash: str
    policy_hash: str
    raw_strategy_output: HashedDecisionArtifact
    exposure_report: HashedDecisionArtifact
    instrument_candidates: HashedDecisionArtifact
    instrument_selection: HashedDecisionArtifact
    order_hashes: tuple[str, ...]
    config_hash: str
    model_versions: dict[str, str]
    feature_versions: dict[str, str]
    schema_version: PositiveSchemaVersion


class DecisionOutcome(ContractModel):
    outcome_id: UUID
    decision_record_id: UUID
    evaluated_at: UtcDatetime
    horizon: str
    underlying_return: FiniteFloat | None
    instrument_return: FiniteFloat | None
    source_fitness_report_id: UUID | None
    metrics: dict[str, FiniteFloat]
```

`DecisionRecord` links execution transitively through immutable order request
IDs; later fills append without mutating the record. Require event hashes to
chain in tuple order and forbid an outcome evaluation time before the
decision. A veto or upstream failure still supplies hashed
`NOT_RUN` artifacts with the blocking stage/reason, so the record never has an
ambiguous missing decision stage.

- [ ] **Step 6: Run contract tests**

Run:

```bash
./.venv/bin/python -m pytest core/nervous_system/tests/test_trading_contracts.py core/nervous_system/tests/test_contract_base.py -q
```

Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add core/nervous_system/contracts core/nervous_system/tests/test_trading_contracts.py
git commit -m "feat: define policy order and execution contracts"
```

### Task 5: Runtime settings and connection factory

**Files:**

- Create: `core/nervous_system/config/runtime.py`
- Create: `core/nervous_system/persistence/database.py`
- Modify: `core/nervous_system/config/__init__.py`
- Modify: `core/nervous_system/persistence/__init__.py`
- Create: `core/nervous_system/tests/test_runtime_config.py`

**Interfaces:**

- Consumes: `RuntimeEnvironment` and `PolicyMode`.
- Produces:
  `NervousSystemSettings.from_env(environ: Mapping[str, str] | None = None)`,
  `create_database_engine(settings) -> Engine`, and
  `create_session_factory(engine) -> sessionmaker[Session]`.

- [ ] **Step 1: Write failing environment-separation tests**

```python
import pytest
from pydantic import ValidationError

from core.nervous_system.config.runtime import NervousSystemSettings
from core.nervous_system.contracts.enums import PolicyMode, RuntimeEnvironment


def test_qa_paper_requires_paper_alias_and_never_accepts_live_url():
    settings = NervousSystemSettings.from_env({
        "CYNOLYCUS_ENVIRONMENT": "qa-paper",
        "CYNOLYCUS_NERVOUS_SYSTEM_MODE": "shadow",
        "CYNOLYCUS_DATABASE_URL": "postgresql+psycopg://u:p@db/cynolycus",
        "CYNOLYCUS_OPERATIONAL_ROOT": "Data/operational/nervous_system",
        "CYNOLYCUS_EXECUTION_JOURNAL": "gcs",
        "CYNOLYCUS_EXECUTION_JOURNAL_BUCKET": "cynolycus-qa-journal",
        "CYNOLYCUS_ACCOUNT_ALIAS": "paper",
    })
    assert settings.environment is RuntimeEnvironment.QA_PAPER
    assert settings.policy_mode is PolicyMode.SHADOW
    with pytest.raises(ValidationError):
        settings.model_copy(update={"account_alias": "live"})


def test_gcs_journal_requires_bucket():
    with pytest.raises(ValidationError):
        NervousSystemSettings.from_env({
            "CYNOLYCUS_ENVIRONMENT": "qa-paper",
            "CYNOLYCUS_NERVOUS_SYSTEM_MODE": "shadow",
            "CYNOLYCUS_DATABASE_URL": "postgresql+psycopg://u:p@db/cynolycus",
            "CYNOLYCUS_OPERATIONAL_ROOT": "Data/operational/nervous_system",
            "CYNOLYCUS_EXECUTION_JOURNAL": "gcs",
            "CYNOLYCUS_ACCOUNT_ALIAS": "paper",
        })
```

- [ ] **Step 2: Run and verify missing-settings failure**

Run:

```bash
./.venv/bin/python -m pytest core/nervous_system/tests/test_runtime_config.py -q
```

Expected: FAIL because `NervousSystemSettings` does not exist.

- [ ] **Step 3: Implement validated settings**

```python
class NervousSystemSettings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    environment: RuntimeEnvironment
    policy_mode: PolicyMode
    database_url: str
    db_pool_size: int = 5
    db_max_overflow: int = 5
    operational_root: Path
    journal_backend: Literal["local", "gcs"]
    gcs_bucket: str | None = None
    account_alias: str

    @model_validator(mode="after")
    def validate_boundary(self) -> "NervousSystemSettings":
        if not self.database_url.startswith("postgresql+psycopg://"):
            raise ValueError("database_url must use postgresql+psycopg")
        if self.environment is RuntimeEnvironment.QA_PAPER and self.account_alias.lower() != "paper":
            raise ValueError("QA_PAPER requires the paper account alias")
        if self.environment is RuntimeEnvironment.QA_PAPER and self.journal_backend != "gcs":
            raise ValueError("QA_PAPER requires the durable GCS journal")
        if self.journal_backend == "gcs" and not self.gcs_bucket:
            raise ValueError("gcs journal requires CYNOLYCUS_EXECUTION_JOURNAL_BUCKET")
        return self
```

`from_env` maps the nine exact `CYNOLYCUS_*` names from `.env.example`,
normalizes enum hyphens to enum values, and raises a single validation error
listing missing names.

- [ ] **Step 4: Implement the SQLAlchemy connection boundary**

```python
def create_database_engine(settings: NervousSystemSettings) -> Engine:
    return create_engine(
        settings.database_url,
        pool_pre_ping=True,
        pool_size=settings.db_pool_size,
        max_overflow=settings.db_max_overflow,
        future=True,
    )


def create_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
```

Add `database_healthcheck(engine) -> bool` that executes `SELECT 1` and returns
`False` only for `SQLAlchemyError`.

- [ ] **Step 5: Run focused tests**

Run:

```bash
./.venv/bin/python -m pytest core/nervous_system/tests/test_runtime_config.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add core/nervous_system/config core/nervous_system/persistence core/nervous_system/tests/test_runtime_config.py
git commit -m "feat: validate nervous system runtime boundaries"
```

### Task 6: PostgreSQL schema and Alembic migrations

**Files:**

- Create: `core/nervous_system/persistence/models/base.py`
- Create: `core/nervous_system/persistence/models/state.py`
- Create: `core/nervous_system/persistence/models/registry.py`
- Create: `core/nervous_system/persistence/models/decision.py`
- Create: `core/nervous_system/persistence/models/execution.py`
- Create: `core/nervous_system/persistence/models/operations.py`
- Create: `core/nervous_system/persistence/models/__init__.py`
- Create: `core/nervous_system/persistence/alembic.ini`
- Create: `core/nervous_system/persistence/migrations/env.py`
- Create: `core/nervous_system/persistence/migrations/script.py.mako`
- Create: `core/nervous_system/persistence/migrations/versions/0001_state_registry.py`
- Create: `core/nervous_system/persistence/migrations/versions/0002_decision_execution.py`
- Create: `core/nervous_system/tests/conftest.py`
- Create: `core/nervous_system/tests/test_migrations.py`

**Interfaces:**

- Consumes: Task 5 database URL and SQLAlchemy engine.
- Produces: PostgreSQL schema `nervous_system`, SQLAlchemy `Base.metadata`, and
  revisions `0001_state_registry` then `0002_decision_execution`.

- [ ] **Step 1: Add the integration fixture and failing migration test**

```python
@pytest.fixture(scope="session")
def postgres_url() -> str:
    value = os.environ.get("NERVOUS_SYSTEM_TEST_DATABASE_URL")
    if not value:
        pytest.skip("set NERVOUS_SYSTEM_TEST_DATABASE_URL for postgres tests")
    return value


@pytest.mark.postgres
def test_upgrade_head_creates_complete_schema(postgres_url):
    cfg = Config("core/nervous_system/persistence/alembic.ini")
    cfg.set_main_option("sqlalchemy.url", postgres_url)
    command.upgrade(cfg, "head")
    engine = create_engine(postgres_url)
    names = set(inspect(engine).get_table_names(schema="nervous_system"))
    assert {
        "state_records", "context_snapshots", "trade_intents",
        "policy_decisions", "policy_modifiers", "order_requests", "order_legs",
        "submission_attempts",
        "execution_events", "decision_records", "decision_outcomes",
        "portfolio_observations", "portfolio_ownership", "source_artifacts",
        "import_runs", "import_items", "import_quarantine", "lineage_edges",
        "config_snapshots", "job_runs", "job_events", "outbox_events", "alerts",
    } <= names
```

- [ ] **Step 2: Run against the local database and verify failure**

Run:

```bash
docker compose -f compose.nervous-system.yaml up -d postgres
NERVOUS_SYSTEM_TEST_DATABASE_URL=postgresql+psycopg://cynolycus:cynolycus_dev_only@127.0.0.1:55432/cynolycus ./.venv/bin/python -m pytest core/nervous_system/tests/test_migrations.py -q
```

Expected: FAIL because Alembic configuration and models are absent.

- [ ] **Step 3: Define shared ORM conventions**

```python
SCHEMA = "nervous_system"


class Base(DeclarativeBase):
    metadata = MetaData(schema=SCHEMA)


class UUIDPrimaryKey:
    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True)


class CreatedAt:
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
```

Use native UUID, timezone-aware `TIMESTAMPTZ`, `NUMERIC` for money, and JSONB
for validated contract payloads. Store enums as constrained strings so adding
an enum member does not require a PostgreSQL enum rewrite.

- [ ] **Step 4: Implement `0001_state_registry`**

Create these tables with exact key columns:

```text
state_records:
  state_id PK, state_type, entity_id, as_of, available_at, generated_at,
  valid_until, schema_version, producer, model_version, feature_version,
  config_version, quality_severity, content_hash, payload JSONB, created_at

context_snapshots:
  snapshot_id PK, decision_time, strategy_id, ticker, freshness_profile,
  content_hash, payload JSONB, created_at

portfolio_observations:
  observation_id PK, account_alias, broker_observed_at, content_hash,
  payload JSONB, created_at

source_artifacts:
  source_id PK, uri, sha256, byte_size, source_kind, discovered_at,
  metadata JSONB

import_runs:
  import_run_id PK, importer_version, started_at, finished_at?, status,
  counts JSONB

import_items:
  import_item_id PK, import_run_id FK, source_id FK, record_locator,
  normalized_hash, target_type, target_id?, status, warnings JSONB

import_quarantine:
  quarantine_id PK, import_run_id FK, source_id FK, record_locator,
  raw_payload JSONB, error_code, error_message, created_at

lineage_edges:
  lineage_edge_id PK, source_id FK, target_type, target_id, relationship,
  created_at

config_snapshots:
  config_snapshot_id PK, config_version, content_hash, payload JSONB, created_at
```

Add these exact deduplication constraints:

```text
source_artifacts: UNIQUE(uri, sha256)
state_records: UNIQUE(content_hash)
config_snapshots: UNIQUE(content_hash)
import_items: UNIQUE(source_id, record_locator, importer_version, normalized_hash)
order_requests: UNIQUE(environment, account_alias, idempotency_key)
submission_attempts: UNIQUE(environment, account_alias, client_order_id)
execution_events: UNIQUE(event_hash)
```

Add state indexes:

```text
(state_type, entity_id, available_at DESC)
(state_type, entity_id, as_of DESC)
(valid_until)
(content_hash)
```

Add database checks for `valid_until > available_at`, nonnegative money/
quantity fields, and UTC-aware application validation. Do not add broad JSONB
GIN indexes without a measured query.

- [ ] **Step 5: Implement `0002_decision_execution`**

Create:

```text
trade_intents:
  intent_id PK, strategy_id, ticker, decision_time, snapshot_id FK,
  content_hash, payload JSONB, created_at

policy_decisions:
  policy_decision_id PK, intent_id FK, snapshot_id FK, action,
  final_risk_budget NUMERIC, content_hash, payload JSONB, created_at

policy_modifiers:
  modifier_id PK, policy_decision_id FK, sequence_no, rule_id,
  operation, configured_value NUMERIC, budget_before NUMERIC, budget_after NUMERIC,
  reason_code, payload JSONB

order_requests:
  order_request_id PK, decision_record_id?, policy_decision_id FK,
  environment, account_alias, idempotency_key,
  request_hash, status, payload JSONB, created_at, expires_at

order_legs:
  order_leg_id PK, order_request_id FK, sequence_no, symbol, side,
  position_intent, ratio, payload JSONB

submission_attempts:
  submission_attempt_id PK, order_request_id FK, attempt_no,
  environment, account_alias, client_order_id, status, reserved_at, journaled_at?,
  broker_called_at?, resolved_at?, broker_order_id?, error_code?,
  payload JSONB

execution_events:
  execution_event_id PK, order_request_id FK, status, client_order_id,
  broker_order_id?, observed_at, broker_event_at?, event_hash UNIQUE,
  previous_event_hash?, payload JSONB

decision_records:
  decision_record_id PK, decision_time, snapshot_id FK, intent_id FK,
  policy_decision_id FK, content_hash UNIQUE, payload JSONB, created_at

decision_outcomes:
  outcome_id PK, decision_record_id FK, evaluated_at, horizon,
  payload JSONB, created_at

portfolio_ownership:
  ownership_id PK, account_alias, broker_position_key, strategy_id?,
  ownership_status, quantity NUMERIC, source_fill_ids JSONB,
  effective_at, ended_at?

job_runs:
  job_run_id PK, job_name, status, started_at, finished_at?, heartbeat_at?,
  dependency_ids JSONB, input_ids JSONB, output_ids JSONB, error?

job_events:
  job_event_id PK, job_run_id FK, status, observed_at, payload JSONB

outbox_events:
  outbox_event_id PK, event_type, aggregate_type, aggregate_id,
  payload JSONB, created_at, available_at, claimed_by?, claimed_until?,
  delivered_at?, delivery_attempts, last_error?

alerts:
  alert_id PK, dedup_key UNIQUE, code, severity, component, entity_id?,
  message, status, opened_at, last_seen_at, occurrence_count,
  acknowledged_at?, acknowledged_by?, resolved_at?, details JSONB
```

- [ ] **Step 6: Apply, inspect, downgrade, and re-upgrade**

Run:

```bash
export NERVOUS_SYSTEM_TEST_DATABASE_URL=postgresql+psycopg://cynolycus:cynolycus_dev_only@127.0.0.1:55432/cynolycus
./.venv/bin/python -m alembic -c core/nervous_system/persistence/alembic.ini upgrade head
./.venv/bin/python -m pytest core/nervous_system/tests/test_migrations.py -q
./.venv/bin/python -m alembic -c core/nervous_system/persistence/alembic.ini downgrade base
./.venv/bin/python -m alembic -c core/nervous_system/persistence/alembic.ini upgrade head
```

Expected: tests pass; downgrade and second upgrade complete without errors.

- [ ] **Step 7: Commit**

```bash
git add core/nervous_system/persistence core/nervous_system/tests/conftest.py core/nervous_system/tests/test_migrations.py
git commit -m "feat: add nervous system PostgreSQL schema"
```

### Task 7: Repositories, unit of work, and transactional outbox

**Files:**

- Create: `core/nervous_system/persistence/uow.py`
- Create: `core/nervous_system/persistence/repositories/state.py`
- Create: `core/nervous_system/persistence/repositories/decision.py`
- Create: `core/nervous_system/persistence/repositories/execution.py`
- Create: `core/nervous_system/persistence/repositories/registry.py`
- Create: `core/nervous_system/persistence/repositories/operations.py`
- Modify: `core/nervous_system/persistence/repositories/__init__.py`
- Create: `core/nervous_system/tests/test_state_repository.py`
- Create: `core/nervous_system/tests/test_decision_repository.py`

**Interfaces:**

- Consumes: Tasks 3-6 contracts and ORM mappings.
- Produces:
  `UnitOfWork`, `StateRepository`, `DecisionRepository`,
  `ExecutionRepository`, `RegistryRepository`, and `OperationsRepository`.

- [ ] **Step 1: Write the failing as-of query test**

```python
@pytest.mark.postgres
def test_latest_valid_state_uses_available_at_and_exclusive_valid_until(pg_session):
    repo = StateRepository(pg_session)
    early = make_market_state("2026-07-29T20:30:00Z", "2026-07-30T20:30:00Z")
    future = make_market_state("2026-07-30T20:30:00Z", "2026-07-31T20:30:00Z")
    repo.save_state(early)
    repo.save_state(future)
    selected = repo.get_latest_valid_state(
        StateType.MARKET, "US", datetime.fromisoformat("2026-07-30T18:20:00+00:00")
    )
    assert selected.state_id == early.state_id
    assert repo.get_latest_valid_state(
        StateType.MARKET, "US", early.valid_until
    ) is None
```

- [ ] **Step 2: Write the failing atomic decision/outbox test**

```python
@pytest.mark.postgres
def test_decision_and_outbox_commit_together(session_factory, complete_decision_chain):
    with UnitOfWork(session_factory) as uow:
        uow.decisions.save_chain(complete_decision_chain)
        uow.operations.enqueue(
            event_type="DecisionRecordCreated",
            aggregate_type="decision_record",
            aggregate_id=complete_decision_chain.record.decision_record_id,
            payload={"strategy_id": "meta_ranker"},
        )
        uow.commit()
    assert count_rows("decision_records") == 1
    assert count_rows("outbox_events") == 1
```

- [ ] **Step 3: Run and verify repository import failures**

Run:

```bash
NERVOUS_SYSTEM_TEST_DATABASE_URL=postgresql+psycopg://cynolycus:cynolycus_dev_only@127.0.0.1:55432/cynolycus ./.venv/bin/python -m pytest core/nervous_system/tests/test_state_repository.py core/nervous_system/tests/test_decision_repository.py -q
```

Expected: FAIL because repository classes are absent.

- [ ] **Step 4: Implement exact repository signatures**

```text
StateRepository.save_state(state: StateEnvelope) -> StateEnvelope
StateRepository.get_latest_valid_state(
    state_type: StateType,
    entity_id: str,
    decision_time: datetime,
) -> StateEnvelope | None
StateRepository.get_state_as_of(
    state_type: StateType,
    entity_id: str,
    decision_time: datetime,
) -> StateEnvelope | None
StateRepository.get_states_for_snapshot(
    requests: Sequence[StateRequest],
    decision_time: datetime,
) -> tuple[StateEnvelope, ...]

DecisionRepository.save_trade_intent(intent: TradeIntent) -> None
DecisionRepository.save_policy_decision(decision: PolicyDecision) -> None
DecisionRepository.save_order_request(request: OrderRequest) -> None
DecisionRepository.save_decision_record(record: DecisionRecord) -> None
DecisionRepository.append_decision_outcome(outcome: DecisionOutcome) -> None

ExecutionRepository.append_execution_event(event: ExecutionEvent) -> None
ExecutionRepository.get_events(
    order_request_id: UUID,
) -> tuple[ExecutionEvent, ...]
ExecutionRepository.find_by_client_order_id(
    client_order_id: str,
) -> OrderRequest | None
```

Implement `get_latest_valid_state` with ordered SQL:

```python
stmt = (
    select(StateRecord)
    .where(
        StateRecord.state_type == state_type.value,
        StateRecord.entity_id == entity_id,
        StateRecord.available_at <= decision_time,
        StateRecord.valid_until > decision_time,
    )
    .order_by(StateRecord.available_at.desc(), StateRecord.generated_at.desc())
    .limit(1)
)
```

- [ ] **Step 5: Implement `UnitOfWork`**

```python
class UnitOfWork:
    def __enter__(self) -> "UnitOfWork":
        self.session = self._session_factory()
        self.states = StateRepository(self.session)
        self.decisions = DecisionRepository(self.session)
        self.executions = ExecutionRepository(self.session)
        self.registry = RegistryRepository(self.session)
        self.operations = OperationsRepository(self.session)
        return self

    def commit(self) -> None:
        self.session.commit()

    def rollback(self) -> None:
        self.session.rollback()

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is not None:
            self.rollback()
        self.session.close()
```

- [ ] **Step 6: Run repository tests**

Run:

```bash
NERVOUS_SYSTEM_TEST_DATABASE_URL=postgresql+psycopg://cynolycus:cynolycus_dev_only@127.0.0.1:55432/cynolycus ./.venv/bin/python -m pytest core/nervous_system/tests/test_state_repository.py core/nervous_system/tests/test_decision_repository.py -q
```

Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add core/nervous_system/persistence core/nervous_system/tests/test_state_repository.py core/nervous_system/tests/test_decision_repository.py
git commit -m "feat: add causal state and decision repositories"
```

### Task 8: Source registry and idempotent historical import

**Files:**

- Create: `core/nervous_system/config/legacy_sources.toml`
- Create: `core/nervous_system/data_registry/artifacts.py`
- Create: `core/nervous_system/data_registry/lineage.py`
- Create: `core/nervous_system/data_registry/parsers.py`
- Create: `core/nervous_system/data_registry/legacy_adapters.py`
- Create: `core/nervous_system/data_registry/import_legacy.py`
- Create: `core/nervous_system/tests/fixtures/legacy_records.py`
- Create: `core/nervous_system/tests/test_legacy_import.py`

**Interfaces:**

- Consumes: `RegistryRepository`, `StateRepository`, `DecisionRepository`, and
  current operational artifacts.
- Produces:
  `register_artifact(path: Path) -> SourceArtifact`,
  `import_manifest(path: Path, uow_factory, dry_run: bool) -> ImportSummary`,
  and CLI `python -m core.nervous_system.data_registry.import_legacy`.

- [ ] **Step 1: Write failing idempotency and quarantine tests**

```python
@pytest.mark.postgres
def test_import_is_idempotent_and_preserves_bad_row(pg_uow_factory, tmp_path):
    source = tmp_path / "audit.jsonl"
    source.write_text(
        '{"event":"signal_decision","module":"meta_ranker","bar":"2026-07-29T18:00:00Z","observed_at":"2026-07-29T18:00:01Z"}\n'
        '{"event":"signal_decision","module":"meta_ranker","bar":"not-a-time"}\n',
        encoding="utf-8",
    )
    manifest = write_manifest(tmp_path, source, kind="meta_signal_audit")
    first = import_manifest(manifest, pg_uow_factory, dry_run=False)
    second = import_manifest(manifest, pg_uow_factory, dry_run=False)
    assert first.imported == 1
    assert first.quarantined == 1
    assert second.imported == 0
    assert second.duplicates == 2
    assert source.read_bytes().startswith(b'{"event":"signal_decision"')
```

- [ ] **Step 2: Run and verify importer failure**

Run:

```bash
NERVOUS_SYSTEM_TEST_DATABASE_URL=postgresql+psycopg://cynolycus:cynolycus_dev_only@127.0.0.1:55432/cynolycus ./.venv/bin/python -m pytest core/nervous_system/tests/test_legacy_import.py -q
```

Expected: FAIL because the importer does not exist.

- [ ] **Step 3: Define the exact legacy source manifest**

Create `legacy_sources.toml`:

```toml
version = 1

[[source]]
kind = "account_snapshot"
glob = "Data/inference/account_snapshots/*.jsonl"
adapter = "broker_equity_snapshot"

[[source]]
kind = "meta_signal_audit"
glob = "Data/inference/meta_ranker/live_signal_audit.jsonl"
adapter = "live_signal_audit"

[[source]]
kind = "meta_closed_trade"
glob = "Data/inference/meta_ranker/closed_trades.jsonl"
adapter = "closed_trade"

[[source]]
kind = "momentum_signal_audit"
glob = "Data/inference/momentum_expansion/live_signal_audit.jsonl"
adapter = "live_signal_audit"

[[source]]
kind = "momentum_closed_trade"
glob = "Data/inference/momentum_expansion/closed_trades.jsonl"
adapter = "closed_trade"

[[source]]
kind = "htf_signal_audit"
glob = "Data/inference/multi_ticker_swing_htf/live_signal_audit.jsonl"
adapter = "live_signal_audit"

[[source]]
kind = "htf_closed_trade"
glob = "Data/inference/multi_ticker_swing_htf/closed_trades.jsonl"
adapter = "closed_trade"

[[source]]
kind = "strategy_live_state"
glob = "signals/meta_context/meta_ranker/live_state.json"
adapter = "managed_state"

[[source]]
kind = "strategy_live_state"
glob = "strategies/momentum_expansion/live/momentum_live_state.json"
adapter = "managed_state"

[[source]]
kind = "strategy_alert"
glob = "strategies/momentum_expansion/live/alerts.jsonl"
adapter = "raw_operational_event"

[[source]]
kind = "strategy_live_state"
glob = "strategies/multi_ticker_swing_htf/live/htf_live_state.json"
adapter = "managed_state"

[[source]]
kind = "dealer_signal_audit"
glob = "Data/inference/dealer_ranker/live_signal_audit.jsonl"
adapter = "live_signal_audit"

[[source]]
kind = "dealer_closed_trade"
glob = "Data/inference/dealer_ranker/closed_trades.jsonl"
adapter = "closed_trade"

[[source]]
kind = "dealer_live_state"
glob = "Data/inference/dealer_ranker/live_state.json"
adapter = "managed_state"

[[source]]
kind = "intraday_structure_state"
glob = "Data/inference/intraday_structure/*.json"
adapter = "managed_state"

[[source]]
kind = "intraday_structure_transition"
glob = "Data/inference/intraday_structure/transitions.jsonl"
adapter = "raw_operational_event"

[[source]]
kind = "shadow_audit"
glob = "Data/inference/shadow_two_sleeve/*_shadow_audit.jsonl"
adapter = "live_signal_audit"

[[source]]
kind = "shadow_state"
glob = "Data/inference/shadow_two_sleeve/*_shadow_state.json"
adapter = "managed_state"

[[source]]
kind = "legacy_position_book"
glob = "Data/inference/multi_ticker_swing/*.json"
adapter = "managed_state"

[[source]]
kind = "spy_broker_state"
glob = "Data/inference/live_runs/*/broker-state.jsonl"
adapter = "broker_state"

[[source]]
kind = "spy_execution"
glob = "Data/inference/live_runs/*/trade-events.jsonl"
adapter = "broker_trade_event"

[[source]]
kind = "spy_action"
glob = "Data/inference/live_runs/*/actions.jsonl"
adapter = "raw_operational_event"

[[source]]
kind = "spy_decision"
glob = "Data/inference/live_runs/*/decision-10m.jsonl"
adapter = "legacy_decision"

[[source]]
kind = "spy_log"
glob = "Data/inference/live_runs/*/logs.jsonl"
adapter = "raw_operational_event"

[[source]]
kind = "spy_policy"
glob = "Data/inference/live_runs/*/order-policy.jsonl"
adapter = "legacy_policy"

[[source]]
kind = "spy_session"
glob = "Data/inference/live_runs/*/session.jsonl"
adapter = "raw_operational_event"

[[source]]
kind = "spy_session_metadata"
glob = "Data/inference/live_runs/*/session_meta.json"
adapter = "raw_operational_event"

[[source]]
kind = "spy_startup_sync"
glob = "Data/inference/live_runs/*/startup-sync.jsonl"
adapter = "broker_state"

[[source]]
kind = "spy_status"
glob = "Data/inference/live_runs/*/status.jsonl"
adapter = "raw_operational_event"

[[source]]
kind = "spy_broker_order_export"
glob = "Data/inference/live_runs/*/unique_broker_orders_*.csv"
adapter = "broker_trade_event"

[[source]]
kind = "swing_session"
glob = "UI/swing_audit/swing_session_*.jsonl"
adapter = "swing_session"

[[source]]
kind = "swing_session"
glob = "UI/swing_audit/paper/swing_session_*.jsonl"
adapter = "swing_session"

[[source]]
kind = "runtime_queue"
glob = "Data/runtime/*.json"
adapter = "runtime_evidence"

[[source]]
kind = "readiness"
glob = "Data/readiness/*.json"
adapter = "runtime_evidence"
```

Option-chain `.meta.json`, research CSV, backtests, and Parquet are
intentionally excluded from operational row import.

Before implementing, compare manifest discovery with the exact operational
roots below:

```text
Data/inference/account_snapshots
Data/inference/dealer_ranker
Data/inference/intraday_structure
Data/inference/live_runs
Data/inference/meta_ranker
Data/inference/momentum_expansion
Data/inference/multi_ticker_swing
Data/inference/multi_ticker_swing_htf
Data/inference/shadow_two_sleeve
Data/readiness
Data/runtime
UI/swing_audit/swing_session_*.jsonl
UI/swing_audit/paper/swing_session_*.jsonl
signals/meta_context/meta_ranker/live_state.json
strategies/momentum_expansion/live/momentum_live_state.json
strategies/momentum_expansion/live/alerts.jsonl
strategies/multi_ticker_swing_htf/live/htf_live_state.json
```

The discovery test fails when a JSON, JSONL, or broker-order CSV under these
roots is neither matched nor explicitly excluded by an exact rule. This makes
“all available operational history” enforceable as new evidence appears.

- [ ] **Step 4: Implement artifact registration and row parsers**

`register_artifact` streams SHA-256, records byte size, and never opens files
for write. `parsers.py` yields:

```python
@dataclass(frozen=True)
class RawImportItem:
    source_path: Path
    record_locator: str
    raw_payload: dict[str, object]
```

Use `line:<1-based-number>` for JSONL, `row:<1-based-data-row>` for CSV, and
`document:1` for JSON.

- [ ] **Step 5: Implement explicit legacy adapter results**

```python
@dataclass(frozen=True)
class LegacyAdapterResult:
    target_type: str
    contract: ContractModel | None
    warnings: tuple[str, ...]
    quarantine_code: str | None
    quarantine_message: str | None
```

Rules:

- account snapshots map `captured_at_utc` to portfolio `available_at`;
- explicit broker/event timestamps remain exact;
- legacy signal `bar` becomes `as_of`, never `available_at`;
- when no reliable availability timestamp exists, set
  `contract=None`, `quarantine_code="MISSING_AVAILABLE_AT"`, and retain the raw
  payload;
- legacy managed state imports as ownership candidate evidence with
  `UNASSIGNED`, never as confirmed ownership;
- no score field populates a probability.

- [ ] **Step 6: Implement dry-run, write mode, and JSON summary**

The CLI accepts:

```text
--manifest PATH
--database-url URL
--dry-run
--limit N
--source-kind KIND
```

It prints one JSON object with:

```text
discovered_artifacts, parsed, imported, duplicates, skipped, quarantined,
source_hashes, import_run_id
```

Dry-run performs all parsing and validation but rolls back the transaction.

- [ ] **Step 7: Run tests and a repository dry run**

Run:

```bash
NERVOUS_SYSTEM_TEST_DATABASE_URL=postgresql+psycopg://cynolycus:cynolycus_dev_only@127.0.0.1:55432/cynolycus ./.venv/bin/python -m pytest core/nervous_system/tests/test_legacy_import.py -q
./.venv/bin/python -m core.nervous_system.data_registry.import_legacy --manifest core/nervous_system/config/legacy_sources.toml --database-url postgresql+psycopg://cynolycus:cynolycus_dev_only@127.0.0.1:55432/cynolycus --dry-run
```

Expected: tests pass; dry-run reports counts and does not alter source hashes.

- [ ] **Step 8: Commit**

```bash
git add core/nervous_system/config/legacy_sources.toml core/nervous_system/data_registry core/nervous_system/tests/fixtures/legacy_records.py core/nervous_system/tests/test_legacy_import.py
git commit -m "feat: import historical operational evidence"
```

### Task 9: Market-regime and sector-state adapters

**Files:**

- Create: `signals/market_regime/nervous_system_adapter.py`
- Create: `signals/market_regime/tests/test_nervous_system_adapter.py`
- Modify: `signals/market_regime/build.py`

**Interfaces:**

```python
def adapt_market_row(
    row: Mapping[str, object],
    *,
    valid_until: datetime,
    lineage: tuple[LineageRef, ...],
) -> MarketState:
    """Interface signature; required behavior is specified in the steps below."""


def adapt_sector_row(
    row: Mapping[str, object],
    *,
    valid_until: datetime,
    lineage: tuple[LineageRef, ...],
) -> SectorState:
    """Interface signature; required behavior is specified in the steps below."""


def persist_market_regime_outputs(
    daily_regime: pd.DataFrame,
    sector_state: pd.DataFrame,
    *,
    unit_of_work: NervousSystemUnitOfWork,
    valid_until_for: Callable[[datetime], datetime],
) -> tuple[int, int]:
    """Interface signature; required behavior is specified in the steps below."""
```

The implementation replaces each `NotImplementedError`; these declarations
define type and dependency boundaries only.

- [ ] **Step 1: Add failing causal-adapter tests**

Use synthetic rows matching the real columns in
`Data/shared/market_regime/daily_regime.parquet` and
`sector_state.parquet`. Assert:

- `available_at` is copied exactly from the existing producer output;
- `as_of` is the market session represented by the row;
- `generated_at` is never used as `available_at`;
- `valid_until` is supplied by the versioned freshness policy;
- the adapter preserves metric names and finite numeric values;
- no z-score, composite, rank, or heuristic becomes a probability;
- the current rule-vector output maps to `MarketRegime.UNKNOWN` with reason
  `MARKET_REGIME_UNCLASSIFIED_RULE_VECTOR`;
- a state appended after a 14:20 or 16:20 decision cannot enter that snapshot.

For sectors, use the canonical mapping in
`signals.market_regime.sector_map`; preserve the existing XLK compatibility
fallback required by frozen-model parity tests.

- [ ] **Step 2: Run the tests and observe import failure**

```bash
./.venv/bin/python -m pytest signals/market_regime/tests/test_nervous_system_adapter.py -q
```

Expected: collection fails because the adapter does not exist.

- [ ] **Step 3: Implement immutable row adapters**

Map existing values without reinterpretation:

```text
daily row session/date -> as_of
daily row available_at -> available_at
adapter call time       -> generated_at only when producer did not persist it
next configured expiry  -> valid_until
rule-vector metrics     -> metrics JSON
current label           -> UNKNOWN
```

Reject naive timestamps, non-finite metrics, duplicate sector rows for one
state key, and `valid_until <= available_at`. Preserve the original source-row
locator and artifact hash in lineage.

- [ ] **Step 4: Add optional persistence at the existing build boundary**

After `signals/market_regime/build.py` has successfully written its existing
Parquet outputs, call `persist_market_regime_outputs` only when a unit of work
is supplied by orchestration. The normal research CLI remains usable without
PostgreSQL. A database error must fail the orchestrated QA-paper job and must
not rewrite or delete the Parquet output.

- [ ] **Step 5: Verify current causal guarantees remain intact**

```bash
./.venv/bin/python -m pytest signals/market_regime/tests -q
```

Expected: adapter tests and the existing DST, staleness, and future-append
invariance tests pass.

- [ ] **Step 6: Commit**

```bash
git add signals/market_regime/nervous_system_adapter.py signals/market_regime/tests/test_nervous_system_adapter.py signals/market_regime/build.py
git commit -m "feat: publish causal market and sector states"
```

### Task 10: Durable theme-membership history and theme-state adapter

**Files:**

- Modify: `themes/dynamic_theme/config.py`
- Modify: `themes/dynamic_theme/stages/step08_memberships.py`
- Modify: `themes/dynamic_theme/pipeline.py`
- Create: `themes/dynamic_theme/nervous_system_adapter.py`
- Create: `themes/dynamic_theme/tests/test_nervous_system_adapter.py`
- Modify: `themes/dynamic_theme/tests/test_seed_and_stability.py`

**Interfaces:**

```python
def append_membership_history(
    current: pd.DataFrame,
    *,
    history_path: Path,
    as_of: date,
    generated_at: datetime,
) -> pd.DataFrame:
    """Interface signature; required behavior is specified in the steps below."""


def adapt_theme_states(
    memberships: pd.DataFrame,
    features: pd.DataFrame,
    *,
    available_at: datetime,
    valid_until: datetime,
    taxonomy_version: str,
    lineage: tuple[LineageRef, ...],
) -> tuple[ThemeState, ...]:
    """Interface signature; required behavior is specified in the steps below."""
```

- [ ] **Step 1: Add failing append-history tests**

Create two small daily membership frames. Assert that:

- day two does not remove day one;
- rerunning the same date produces one deterministic row per
  `(date, ticker, theme, taxonomy_version)`;
- historical rows are not changed by a later taxonomy;
- the compatibility output
  `ticker_theme_membership.parquet` still contains the latest current view;
- the new
  `ticker_theme_membership_history.parquet` is append-preserving;
- writes use a same-directory temporary file followed by atomic replacement.

- [ ] **Step 2: Run the focused tests and observe failure**

```bash
./.venv/bin/python -m pytest themes/dynamic_theme/tests/test_nervous_system_adapter.py themes/dynamic_theme/tests/test_seed_and_stability.py -q
```

Expected: missing history config and adapter failures.

- [ ] **Step 3: Add the history path and deterministic taxonomy version**

Add `membership_history_path` to the existing theme configuration. Compute
`taxonomy_version` from canonical JSON containing sorted theme IDs, sorted
seed members, embedding/model identifier, and relevant clustering parameters.
The hash must not depend on dictionary insertion order, local paths, or runtime
timestamps.

- [ ] **Step 4: Preserve history at the end of Step 8**

Keep writing the current compatibility file. Also append validated rows to the
history file with:

```text
as_of, available_at, generated_at, ticker, theme, membership_score,
taxonomy_version, producer_version
```

Capture `available_at` as the actual UTC completion instant after inputs are
present, not midnight of the represented date. Do not derive it from file
mtime. Keep existing historical `ticker_theme_features` append semantics.

- [ ] **Step 5: Implement theme-state adaptation**

Group rows by theme and `as_of`. Emit `ThemeState` with:

- `regime=ThemeRegime.UNKNOWN` until a calibrated/versioned classifier exists;
- sorted memberships with the original scores preserved as scores;
- breadth, momentum, crowding, and persistence metrics only when present;
- no inferred probability;
- lineage to membership and feature artifacts;
- a quality warning when features or memberships are missing.

- [ ] **Step 6: Wire optional publication after pipeline completion**

`themes/dynamic_theme/pipeline.py` publishes through a supplied unit of work.
Research execution without a unit of work remains unchanged. An orchestrated
publication failure marks the job failed and leaves existing Parquet artifacts
intact.

- [ ] **Step 7: Verify deterministic reruns**

```bash
./.venv/bin/python -m pytest themes/dynamic_theme/tests -q
```

Expected: all tests pass and two identical synthetic runs yield identical
taxonomy hashes and history rows.

- [ ] **Step 8: Commit**

```bash
git add themes/dynamic_theme/config.py themes/dynamic_theme/stages/step08_memberships.py themes/dynamic_theme/pipeline.py themes/dynamic_theme/nervous_system_adapter.py themes/dynamic_theme/tests
git commit -m "feat: preserve and publish theme state history"
```

### Task 11: Catalyst event-time and availability-time adapter

**Files:**

- Modify: `signals/catalysts/pipeline.py`
- Modify: `signals/news/schema.py`
- Modify: `signals/news/pipeline.py`
- Modify: `signals/events/schema.py`
- Modify: `signals/events/collectors.py`
- Create: `signals/catalysts/nervous_system_adapter.py`
- Create: `signals/catalysts/tests/test_nervous_system_adapter.py`
- Create: `signals/news/tests/test_availability_metadata.py`
- Create: `signals/events/tests/test_availability_metadata.py`

**Interfaces:**

```python
@dataclass(frozen=True)
class CatalystAdapterResult:
    event: CatalystEvent | None
    warnings: tuple[str, ...]
    quarantine_code: str | None
    quarantine_message: str | None


def normalize_catalyst_record(
    row: Mapping[str, object],
    *,
    source_artifact: LineageRef,
) -> CatalystAdapterResult:
    """Interface signature; required behavior is specified in the steps below."""


def aggregate_catalyst_pressure(
    events: Sequence[CatalystEvent],
    *,
    entity_id: str,
    decision_time: datetime,
    valid_until: datetime,
    config_version: str,
) -> CatalystPressure:
    """Interface signature; required behavior is specified in the steps below."""
```

- [ ] **Step 1: Add failing timestamp-semantic tests**

Cover:

1. a news item published at 13:01 and observed at 13:03;
2. an earnings date known two weeks before the event;
3. a future scheduled event whose information is currently available;
4. a legacy row containing only ambiguous `timestamp`;
5. an event observed after a decision despite an earlier publication time.

Assert separate `event_time`, `published_at`, `observed_at`, and
`available_at`. The scheduled event is eligible once the calendar observation
is available; the late-observed news is not eligible at the earlier decision;
the ambiguous legacy row is quarantined with `MISSING_AVAILABLE_AT`.

- [ ] **Step 2: Run the test and observe failure**

```bash
./.venv/bin/python -m pytest signals/catalysts/tests/test_nervous_system_adapter.py -q
```

Expected: collection fails because the adapter is missing.

- [ ] **Step 3: Preserve observation metadata for new ingestion**

Without removing the legacy `timestamp` column, add:

```text
event_time, published_at, observed_at, available_at, source_record_id,
source_artifact_hash, timestamp_semantics_version
```

`signals/news/pipeline.py` captures one aware UTC `observed_at` for each
collection batch before merge/dedup; `signals/news/schema.py` preserves source
`published_at` separately from the compatibility `timestamp`. CSV news accepts
an explicit `observed_at` column or CLI-supplied collection time and otherwise
marks availability unknown.

`signals/events/collectors.py` similarly captures collection time and
`signals/events/schema.py` preserves it. For scheduled events, `event_time` is
the occurrence and `available_at` is when the schedule was observed. Never
backfill an absent availability timestamp from event time or file mtime.

- [ ] **Step 4: Implement normalized events and pressure**

Deduplicate by source plus stable source record ID. If no ID exists, hash
canonical source fields. Preserve conflicting revisions as separate observed
versions. Aggregate only events with `available_at <= decision_time`; preserve
raw score scale and direction; leave probability fields null. Add explicit
quality warnings for missing publication time, conflicting source values, and
late observation.

- [ ] **Step 5: Publish optionally from the current pipeline**

The pipeline keeps writing:

```text
signals/catalysts/data/processed/catalyst_records.parquet
signals/catalysts/data/processed/catalyst_scores.parquet
signals/catalysts/data/processed/catalyst_feature_matrix.parquet
```

When orchestration supplies a unit of work, publish normalized events and
states after those artifacts validate. Quarantined legacy rows remain
registered evidence and are excluded from decision snapshots.

- [ ] **Step 6: Verify**

```bash
./.venv/bin/python -m pytest signals/catalysts/tests/test_nervous_system_adapter.py signals/news/tests/test_availability_metadata.py signals/events/tests/test_availability_metadata.py -q
```

Expected: all five timestamp cases and idempotent publication pass.

- [ ] **Step 7: Commit**

```bash
git add signals/catalysts/pipeline.py signals/catalysts/nervous_system_adapter.py signals/catalysts/tests/test_nervous_system_adapter.py signals/news/schema.py signals/news/pipeline.py signals/news/tests/test_availability_metadata.py signals/events/schema.py signals/events/collectors.py signals/events/tests/test_availability_metadata.py
git commit -m "feat: publish causally timed catalyst states"
```

### Task 12: Ticker, dealer, and broker-portfolio adapters

**Files:**

- Create: `signals/meta_context/meta_ranker/nervous_system_adapter.py`
- Create: `signals/meta_context/meta_ranker/tests/test_nervous_system_adapter.py`
- Create: `strategies/dealer_positioning/nervous_system_adapter.py`
- Create:
  `strategies/dealer_positioning/tests/test_nervous_system_adapter.py`
- Create: `core/nervous_system/context/readiness_adapter.py`
- Modify: `core/broker_equity_snapshot.py`
- Create: `core/nervous_system/tests/test_portfolio_adapter.py`
- Create: `core/nervous_system/tests/test_readiness_adapter.py`

**Interfaces:**

```python
def adapt_ticker_state(
    row: Mapping[str, object],
    *,
    decision_bar: datetime,
    available_at: datetime,
    valid_until: datetime,
    lineage: tuple[LineageRef, ...],
) -> TickerState:
    """Interface signature; required behavior is specified in the steps below."""


def adapt_dealer_state(
    snapshot: Mapping[str, object],
    dynamics: Mapping[str, object] | None,
    *,
    captured_at: datetime,
    valid_until: datetime,
    lineage: tuple[LineageRef, ...],
) -> DealerState:
    """Interface signature; required behavior is specified in the steps below."""


def adapt_broker_portfolio_snapshot(
    raw: Mapping[str, object],
    *,
    strategy_ownership: Mapping[str, str],
) -> PortfolioState:
    """Interface signature; required behavior is specified in the steps below."""


def adapt_readiness_status(
    *,
    ready: bool,
    reason: str,
    payload: Mapping[str, object],
    checked_at: datetime,
    max_age_hours: float,
) -> ReadinessState:
    """Interface signature; required behavior is specified in the steps below."""
```

- [ ] **Step 1: Add failing ticker tests**

Use rows matching the current Meta feature matrix. Assert that:

- `as_of` is the exact selected 4H bar;
- `available_at` is when the completed matrix became observable;
- OHLC/reference price comes from that selected bar, not the final row in a
  ticker Parquet file;
- rank and `s_combo` remain scores;
- missing bars or non-finite selected-bar closes fail closed;
- a later appended bar cannot change an earlier ticker state.

- [ ] **Step 2: Add failing dealer tests**

Use captured dealer fixture rows. Assert:

- raw captures use `captured_at` as availability;
- dynamics use their explicit `available_at`;
- a 15:45 capture can enter a 16:20 snapshot but not a 14:20 snapshot;
- `GammaLevels.timestamp` is not accepted without explicit capture metadata;
- exposure/level metrics remain metrics, not probabilities;
- stale or print-sparse option data produces quality warnings.

- [ ] **Step 3: Add failing portfolio tests**

Use the exact current account snapshot shape from
`core/broker_equity_snapshot.py`. Assert:

- `captured_at_utc` maps to `available_at`;
- broker positions, open orders, cash, equity, and buying power are preserved;
- option symbols retain OCC identity;
- ownership comes only from supplied fill-derived ownership;
- unmatched positions are `UNASSIGNED`;
- no additional broker calls are made by the adapter.

Add readiness cases for current success, stale success, missing stamp, invalid
timestamp, and `CYNOLYCUS_READINESS_REQUIRED=0`. In QA-paper, disabling the
legacy gate must produce a non-ready state with
`READINESS_DISABLED_NOT_ACCEPTED_IN_QA`; development may expose it only as a
warning. `completed_at_utc` is the state `available_at`; `checked_at` is
observation/generation time.

- [ ] **Step 4: Run focused tests and observe failure**

```bash
./.venv/bin/python -m pytest signals/meta_context/meta_ranker/tests/test_nervous_system_adapter.py strategies/dealer_positioning/tests/test_nervous_system_adapter.py core/nervous_system/tests/test_portfolio_adapter.py core/nervous_system/tests/test_readiness_adapter.py -q
```

Expected: missing-adapter failures.

- [ ] **Step 5: Implement the three adapters**

Ticker adaptation accepts only the selected row and decision bar. Dealer
adaptation refuses ambiguous capture time. Portfolio adaptation treats the
broker payload as authoritative and adds derived ownership only as an
annotation. Readiness adaptation preserves the existing latest-completed-
session check from `core/live_readiness.py`. All four attach source hashes,
producer versions, and quality.

- [ ] **Step 6: Add optional portfolio persistence**

Extend the existing snapshot writer with an optional unit of work parameter.
Keep its append-only JSONL behavior unchanged. After the local snapshot is
durable, persist `PortfolioState`; if PostgreSQL is unavailable, report the
publication error to the caller rather than silently presenting stale state.

- [ ] **Step 7: Verify adapters and existing broker snapshots**

```bash
./.venv/bin/python -m pytest signals/meta_context/meta_ranker/tests/test_nervous_system_adapter.py strategies/dealer_positioning/tests/test_nervous_system_adapter.py core/nervous_system/tests/test_portfolio_adapter.py core/nervous_system/tests/test_readiness_adapter.py -q
./.venv/bin/python -m py_compile core/broker_equity_snapshot.py
```

Expected: focused tests pass and the existing snapshot module compiles.

- [ ] **Step 8: Commit**

```bash
git add signals/meta_context/meta_ranker/nervous_system_adapter.py signals/meta_context/meta_ranker/tests/test_nervous_system_adapter.py strategies/dealer_positioning/nervous_system_adapter.py strategies/dealer_positioning/tests/test_nervous_system_adapter.py core/nervous_system/context/readiness_adapter.py core/broker_equity_snapshot.py core/nervous_system/tests/test_portfolio_adapter.py core/nervous_system/tests/test_readiness_adapter.py
git commit -m "feat: adapt ticker dealer and portfolio states"
```

### Task 13: Versioned freshness requirements and causal snapshot builder

**Files:**

- Create: `core/nervous_system/config/freshness.py`
- Create: `core/nervous_system/context/requirements.py`
- Create: `core/nervous_system/context/snapshot_builder.py`
- Create: `core/nervous_system/tests/test_snapshot_builder.py`
- Create:
  `core/nervous_system/tests/test_snapshot_future_append_invariance.py`

**Interfaces:**

```python
@dataclass(frozen=True)
class FreshnessRule:
    state_type: StateType
    required: bool
    max_age: timedelta
    fallback: MissingStateAction


@dataclass(frozen=True)
class SnapshotProfile:
    profile_id: str
    rules: tuple[FreshnessRule, ...]


class SnapshotBuilder:
    def build(
        self,
        *,
        strategy_id: str,
        entity_id: str,
        decision_time: datetime,
        decision_bar: datetime,
        profile: SnapshotProfile,
    ) -> ContextSnapshot:
        """Interface signature; required behavior is specified in the steps below."""
```

- [ ] **Step 1: Add failing eligibility and tie-break tests**

Insert multiple state versions around one decision time. Assert the selected
row is the deterministic maximum of:

```text
(available_at, generated_at, producer_version, state_id)
```

among rows satisfying:

```text
entity match
state type match
as_of <= decision_bar when the state is bar-bound
available_at <= decision_time
decision_time < valid_until
```

Assert future, expired, wrong-entity, and future-bar rows are excluded.

- [ ] **Step 2: Add failing profile tests**

Define:

```text
meta_4h_1420@1
meta_4h_1620@1
```

Both require ticker, market, sector, portfolio, and readiness state. Theme,
catalyst, and dealer are optional during shadow; dealer is ineligible at 14:20
and eligible at 16:20 only when a post-14:20 capture exists. Daily market
regime published after 16:30 must use the prior session at both decision times.

Assert required missing/expired state marks the snapshot invalid with reason
codes. Optional missing state produces warnings, never an invented neutral
state.

- [ ] **Step 3: Run tests and observe failure**

```bash
NERVOUS_SYSTEM_TEST_DATABASE_URL=postgresql+psycopg://cynolycus:cynolycus_dev_only@127.0.0.1:55432/cynolycus ./.venv/bin/python -m pytest core/nervous_system/tests/test_snapshot_builder.py core/nervous_system/tests/test_snapshot_future_append_invariance.py -q
```

Expected: missing builder/profile failures.

- [ ] **Step 4: Implement versioned profiles**

Store profiles as frozen Python data with a canonical content hash. The
profile ID and hash enter every snapshot. Compute trading-session boundaries
with the repository market-calendar helper; do not add fixed 24-hour calendar
arithmetic across weekends, holidays, or DST.

- [ ] **Step 5: Implement one-query candidate loading and pure selection**

Load all candidate envelopes in one repository call, then apply a pure
selection function. Persist:

- selected state IDs and hashes;
- rejected candidate IDs with reason codes;
- missing/stale requirements;
- profile ID/hash;
- complete snapshot canonical hash.

If the same inputs are built twice, return the existing snapshot by
deterministic idempotency key. Derive `snapshot_id` with UUIDv5 from strategy,
entity, decision time/bar, profile hash, and ordered selected state hashes.

- [ ] **Step 6: Prove future-append invariance**

The invariance test builds a snapshot, adds states with later
`available_at`/`generated_at`, rebuilds for the original decision time, and
asserts byte-identical canonical payload and hash.

- [ ] **Step 7: Verify**

```bash
NERVOUS_SYSTEM_TEST_DATABASE_URL=postgresql+psycopg://cynolycus:cynolycus_dev_only@127.0.0.1:55432/cynolycus ./.venv/bin/python -m pytest core/nervous_system/tests/test_snapshot_builder.py core/nervous_system/tests/test_snapshot_future_append_invariance.py -q
```

Expected: eligibility, freshness, idempotency, and invariance tests pass.

- [ ] **Step 8: Commit**

```bash
git add core/nervous_system/config/freshness.py core/nervous_system/context core/nervous_system/tests/test_snapshot_builder.py core/nervous_system/tests/test_snapshot_future_append_invariance.py
git commit -m "feat: build versioned causal context snapshots"
```

### Task 14: Meta Ranker pure ranking and trade-intent parity

**Files:**

- Modify: `signals/meta_context/meta_ranker/live_runner.py`
- Modify: `signals/meta_context/meta_ranker/score.py`
- Modify: `signals/meta_context/meta_ranker/nervous_system_adapter.py`
- Create: `signals/meta_context/meta_ranker/tests/test_intent_parity.py`
- Create: `signals/meta_context/meta_ranker/tests/fixtures/meta_rows.py`

**Interfaces:**

```python
def rank_meta_candidates(
    feature_matrix: pd.DataFrame,
    *,
    bar: pd.Timestamp,
    config: MetaRankingConfig,
) -> pd.DataFrame:
    """Interface signature; required behavior is specified in the steps below."""


def build_trade_intents(
    ranked: pd.DataFrame,
    *,
    decision_time: datetime,
    decision_bar: datetime,
    snapshot_id_by_ticker: Mapping[str, UUID],
    config: MetaIntentConfig,
) -> tuple[TradeIntent, ...]:
    """Interface signature; required behavior is specified in the steps below."""
```

- [ ] **Step 1: Freeze a synthetic baseline fixture**

Construct a small deterministic feature matrix covering:

- score ties;
- one missing required feature;
- one non-finite score;
- an already-held ticker;
- one ticker excluded by the current selection rules;
- a selected bar followed by a later row.

In the test, call the current ranking/selection path before refactoring and
record ticker order, `s_combo`, component scores, tie-break order, side,
reference close, and notional request.

- [ ] **Step 2: Add failing pure-function parity tests**

Assert the extracted function produces the exact baseline values for the
selected bar. Assert `TradeIntent.raw_score` receives `s_combo`,
`raw_probability is None`, and the intent lineage references the exact matrix
row and snapshot ID.

- [ ] **Step 3: Run the focused tests and capture the expected failure**

```bash
./.venv/bin/python -m pytest signals/meta_context/meta_ranker/tests/test_intent_parity.py -q
```

Expected: missing pure interfaces.

- [ ] **Step 4: Extract ranking without changing formulas**

Move current filtering, scoring, ranking, and deterministic tie-breaking into
`rank_meta_candidates`. Do not rename features, change imputation, adjust
thresholds, or recalibrate scores. Keep `score.py` responsible for the same
mathematics and make `live_runner.py` call the pure function.

- [ ] **Step 5: Build intents from selected rows**

Each intent records:

```text
strategy_id="meta_ranker"
decision_time
decision_bar
entity_id/ticker
side
requested_notional
selected-bar reference price
raw_score and score components
raw_probability=null
context_snapshot_id
model/feature/config versions
instrument_preferences
idempotency_key
```

The key is the canonical hash of strategy, bar, ticker, side, config version,
and intent ordinal. Derive `intent_id` with UUIDv5 from that key and set
`created_at=decision_time`; rebuilding the same bar must return the same
identity and hash.

- [ ] **Step 6: Remove latest-file-row reference pricing**

Replace `_ref_price(ticker)` behavior with the selected-row close. If a
compatibility caller must read ticker Parquet, require `decision_bar`, filter
to that exact bar, reject duplicates, and fail if the row is absent. Never use
`.iloc[-1]` across unfiltered data.

- [ ] **Step 7: Verify parity**

```bash
./.venv/bin/python -m pytest signals/meta_context/meta_ranker/tests/test_intent_parity.py -q
./.venv/bin/python -m pytest signals/meta_context/meta_ranker/tests -q
```

Expected: exact fixture parity, deterministic intents, and no future-row
reference-price change.

- [ ] **Step 8: Commit**

```bash
git add signals/meta_context/meta_ranker/live_runner.py signals/meta_context/meta_ranker/score.py signals/meta_context/meta_ranker/nervous_system_adapter.py signals/meta_context/meta_ranker/tests
git commit -m "refactor: emit deterministic Meta trade intents"
```

### Task 15: Deterministic permission and modulation policy

**Files:**

- Create: `core/nervous_system/config/policy.py`
- Create: `core/nervous_system/policy/reason_codes.py`
- Create: `core/nervous_system/policy/permissions.py`
- Create: `core/nervous_system/policy/rules.py`
- Create: `core/nervous_system/policy/engine.py`
- Create: `core/nervous_system/tests/test_policy_engine.py`
- Create: `core/nervous_system/tests/test_policy_properties.py`

**Interfaces:**

```python
def evaluate_policy(
    intent: TradeIntent,
    snapshot: ContextSnapshot,
    config: PolicyConfig,
) -> PolicyDecision:
    """Interface signature; required behavior is specified in the steps below."""
```

Rule order is fixed:

```text
1 environment permission
2 snapshot validity/freshness
3 operational readiness
4 instrument/structure permission
5 broker/account constraints
6 hard portfolio limits
7 liquidity and data-quality limits
8 contextual size modifiers
9 final caps and minimum executable size
```

- [ ] **Step 1: Add failing hard-veto tests**

Cover:

- `PRODUCTION_LIVE` always returns `PolicyAction.REJECT`;
- `OFF` in development/QA produces a non-executable baseline decision with
  reason `POLICY_OFF_AUDIT_ONLY`; orchestration rejects `off + submit`;
- invalid or stale required snapshot vetoes entries;
- QA-paper without paper credentials/account identity vetoes;
- unknown maximum-loss options veto;
- naked short option and uncovered ratio veto;
- duplicate idempotency key vetoes a second entry;
- readiness gate failure vetoes entry;
- a risk-reducing intent receives `PolicyAction.EXIT`; the gateway applies its
  narrow fail-operational exit permission instead of treating it as an entry.

Assert every veto has stable machine reason codes and human detail.

- [ ] **Step 2: Add failing modifier-order and cap tests**

Use base notional `1000`, a regime multiplier `0.8`, a theme multiplier `0.5`,
and a portfolio cap `300`. Assert:

```text
1000 * 0.8 * 0.5 = 400
min(400, 300) = 300 final notional
```

Assert each `PolicyModifier` records before, `MULTIPLY` or `CAP`, configured
value, after, source state ID, rule ID, and config version. A `MULTIPLY` value
may be in `[0, 1]` only during MVP; context cannot increase risk.

- [ ] **Step 3: Add property tests**

For generated valid intents/snapshots, assert:

- same inputs and config produce the same canonical decision hash;
- adding a hard veto cannot increase final size;
- worsening one risk multiplier cannot increase final size;
- final size is never negative or above requested size;
- no modifier is silently omitted from the audit trail.

- [ ] **Step 4: Run tests and observe failure**

```bash
./.venv/bin/python -m pytest core/nervous_system/tests/test_policy_engine.py core/nervous_system/tests/test_policy_properties.py -q
```

Expected: missing policy implementation.

- [ ] **Step 5: Implement versioned default policy**

Create immutable `PolicyConfig` with:

```text
policy_version
mode: off|shadow|enforce
environment
allowed_instruments
allowed_structures
required_snapshot_profile
hard limits
liquidity thresholds
context modifier thresholds
minimum order notional
```

Default contextual thresholds run in `shadow`; hard environment, structure,
idempotency, maximum-loss, and readiness safety rules are enforceable from the
first gateway run. Config content is canonically hashed and stored with every
decision.

- [ ] **Step 6: Implement pure rules and engine**

Rules receive contracts and config only—no files, clock, database, broker, or
environment reads. Accumulate all applicable hard reasons, then apply
modifiers in the fixed order only when no hard veto exists. Quantize money and
quantity using contract rules, not binary floats.

Derive `policy_decision_id` with UUIDv5 from intent ID, snapshot hash, policy
config hash, and mode. Set policy `created_at` from the intent decision time
and `expires_at` from the versioned entry window; do not call the wall clock
inside the pure evaluator.

In shadow mode, store the counterfactual policy decision while exposing
`execution_basis="BASELINE_INTENT"` to orchestration. In enforce mode, expose
`execution_basis="POLICY_FINAL"`. The decision contract itself is identical
and auditable in both modes. Off mode records the unmodified baseline for
parity but is never executable. Production-live is rejected in every mode,
including off.

- [ ] **Step 7: Verify**

```bash
./.venv/bin/python -m pytest core/nervous_system/tests/test_policy_engine.py core/nervous_system/tests/test_policy_properties.py -q
```

Expected: deterministic hard-veto, ordering, cap, exit-only, and property
tests pass.

- [ ] **Step 8: Commit**

```bash
git add core/nervous_system/config/policy.py core/nervous_system/policy core/nervous_system/tests/test_policy_engine.py core/nervous_system/tests/test_policy_properties.py
git commit -m "feat: add deterministic shared policy engine"
```

### Task 16: Portfolio exposure, overlap, ownership, and reconciliation

**Files:**

- Create: `core/nervous_system/portfolio/exposure.py`
- Create: `core/nervous_system/portfolio/ownership.py`
- Create: `core/nervous_system/portfolio/reconciliation.py`
- Create: `core/nervous_system/config/portfolio.py`
- Create: `core/nervous_system/tests/test_portfolio_exposure.py`
- Create: `core/nervous_system/tests/test_portfolio_ownership.py`

**Interfaces:**

```python
def calculate_exposure(
    portfolio: PortfolioState,
    context: ContextSnapshot,
    *,
    config: PortfolioConfig,
) -> ExposureReport:
    """Interface signature; required behavior is specified in the steps below."""


def assign_fill_ownership(
    fill: ExecutionEvent,
    order_request: OrderRequest,
) -> OwnershipRecord:
    """Interface signature; required behavior is specified in the steps below."""


def reconcile_portfolio(
    broker: PortfolioState,
    ownership: Sequence[OwnershipRecord],
) -> PortfolioReconciliation:
    """Interface signature; required behavior is specified in the steps below."""
```

Persisted portfolio results use these fields:

```text
ExposureReport:
  report_id, portfolio_state_id, snapshot_id, calculated_at,
  gross_notional, net_notional, long_notional, short_notional,
  symbol_notional, underlying_equivalent, sector_notional,
  theme_notional, factor_notional, option_greeks,
  proposed_incremental_exposure, limit_results, quality, content_hash

OwnershipRecord:
  ownership_id, account_alias, broker_position_key, strategy_id,
  decision_record_id, order_request_id, source_fill_id,
  quantity, effective_at, ended_at?, ownership_status

PortfolioReconciliation:
  reconciliation_id, portfolio_state_id, observed_at,
  matched, partial, unassigned, orphaned, quantity_mismatches,
  ownership_adjustment_ids, content_hash
```

- [ ] **Step 1: Add failing exposure tests**

Cover:

- gross, net, long, short, and per-symbol notional;
- sector exposure from the canonical sector state;
- weighted theme exposure without double-counting full notional into every
  theme;
- options delta/gamma/vega/theta when broker/quote Greeks are available;
- conservative unknown-exposure flags when Greeks are missing;
- underlying-equivalent exposure across shares and options;
- explicit correlated-factor tags showing overlap among
  `NBIS/APLD/IREN/SOXL/TQQQ/AMD`;
- proposed-order incremental exposure and post-trade limit evaluation.

- [ ] **Step 2: Add failing ownership tests**

Assert ownership is created only after a broker-confirmed fill. A submitted or
accepted order creates no position ownership. Partial fills allocate only the
filled quantity. Manual/unmatched broker positions are `UNASSIGNED`. Closing
fills decrement ownership deterministically and never produce negative owned
quantity.

- [ ] **Step 3: Run tests and observe failure**

```bash
./.venv/bin/python -m pytest core/nervous_system/tests/test_portfolio_exposure.py core/nervous_system/tests/test_portfolio_ownership.py -q
```

Expected: missing portfolio modules.

- [ ] **Step 4: Implement exposure math**

Use `Decimal` for currency/notional and explicit signed quantities. Theme
weights for a ticker are normalized across positive memberships before
allocation; preserve the unallocated bucket when no memberships exist.
Factor tags are versioned configuration, initially including semiconductor,
leveraged-index, AI-infrastructure, and single-name-underlying links needed by
the acceptance fixture. They are risk metadata, not learned correlations.

- [ ] **Step 5: Implement ownership and reconciliation**

The ownership key includes broker account, broker position identity/OCC
symbol, strategy, originating decision, and fill. Reconciliation compares
broker quantities with ownership totals and emits:

```text
MATCHED, PARTIAL, UNASSIGNED, ORPHANED_OWNERSHIP, QUANTITY_MISMATCH
```

Do not mutate broker facts to force agreement. Persist corrections as new
observations/events.

- [ ] **Step 6: Verify**

```bash
./.venv/bin/python -m pytest core/nervous_system/tests/test_portfolio_exposure.py core/nervous_system/tests/test_portfolio_ownership.py -q
```

Expected: all exposure, overlap, partial-fill, and reconciliation cases pass.

- [ ] **Step 7: Commit**

```bash
git add core/nervous_system/config/portfolio.py core/nervous_system/portfolio core/nervous_system/tests/test_portfolio_exposure.py core/nervous_system/tests/test_portfolio_ownership.py
git commit -m "feat: calculate shared exposure and fill ownership"
```

### Task 17: Option quote contracts, payoff bounds, and approved structures

**Files:**

- Create: `core/nervous_system/execution/options/quotes.py`
- Create: `core/nervous_system/execution/options/payoff.py`
- Create: `core/nervous_system/execution/options/structures.py`
- Create: `core/nervous_system/tests/test_option_payoff.py`
- Create: `core/nervous_system/tests/test_option_structures.py`

**Interfaces:**

```python
def expiry_payoff(
    legs: Sequence[OptionLeg],
    *,
    underlying_price: Decimal,
    net_debit: Decimal,
    contract_multiplier: int = 100,
) -> Decimal:
    """Interface signature; required behavior is specified in the steps below."""


def validate_structure(
    legs: Sequence[OptionLeg],
    *,
    net_price: Decimal,
    existing_holdings: Sequence[PortfolioPosition],
    available_cash: Decimal,
) -> OptionRiskProfile:
    """Interface signature; required behavior is specified in the steps below."""


def build_structure(
    structure: InstrumentFamily,
    *,
    selected_contracts: Mapping[LegRole, OptionQuote],
    quantity: int,
) -> tuple[OptionLeg, ...]:
    """Interface signature; required behavior is specified in the steps below."""
```

- [ ] **Step 1: Add failing quote-validation tests**

Reject crossed/negative markets, stale quote time, absent bid/ask, nonpositive
multiplier, invalid OCC symbol, and a contract whose expiration/strike/right
conflicts with parsed OCC identity. Keep trade-print fields separate from
bid/ask marks.

- [ ] **Step 2: Add table-driven payoff tests**

Test payoff and maximum loss at all strikes and outer intervals for:

- long call and put;
- debit and credit verticals;
- long straddle and strangle;
- fully secured short straddle and strangle, with every call share-covered and
  every put cash-secured;
- call/put butterflies and iron butterflies;
- call/put condors and iron condors;
- covered call;
- cash-secured put;
- protective put;
- collar.

Test multiple quantities and nonzero net debit/credit. Compare exact `Decimal`
values; do not use approximate floating-point assertions.

- [ ] **Step 3: Add calendar, diagonal, ratio, and roll safety tests**

Rules:

- a long calendar/diagonal is permitted only when each earlier-expiry short
  contract is fully covered by a later-expiry long contract of the same right,
  assignment exposure is represented, and total debit is bounded;
- a short calendar/diagonal is rejected;
- every short call must be covered by same-or-more long calls or existing
  shares;
- every short put must be covered by same-or-more long puts or reserved cash;
- uncovered ratio spreads and naked shorts are rejected;
- a roll is represented as a close request followed by a separately approved
  open request, never one eight-leg atomic order;
- covered calls, protective puts, and collars require existing shares because
  Alpaca does not atomically combine equities and options.

- [ ] **Step 4: Run tests and observe failure**

```bash
./.venv/bin/python -m pytest core/nervous_system/tests/test_option_payoff.py core/nervous_system/tests/test_option_structures.py -q
```

Expected: missing option modules.

- [ ] **Step 5: Implement finite payoff bounds**

For same-expiry structures, evaluate the piecewise-linear payoff at zero, all
unique strikes, and the final slope analytically as underlying tends upward.
Return explicit finite `max_profit`, `max_loss`, break-evens, collateral, and
assignment exposure. If either tail is unbounded on the loss side, return an
invalid risk profile with `UNBOUNDED_MAX_LOSS`.

For approved debit calendars/diagonals, require matched quantities and a later
long expiration. Compute a conservative finite loss bound as paid debit plus
the adverse strike-gap assignment bound and configured fees. Same-strike
calendars have zero strike-gap term. Reject ex-dividend/early-assignment cases
whose dividend or carry exposure is not modeled within collateral. Do not
claim one expiry payoff across different expirations.

- [ ] **Step 6: Implement structure templates**

Build deterministic one-to-four-leg orders for every approved structure. Sort
legs by semantic role and use explicit `BUY_TO_OPEN`, `SELL_TO_OPEN`,
`BUY_TO_CLOSE`, or `SELL_TO_CLOSE`. Validate all legs share underlying and
supported multiplier, and that same-expiry structures satisfy strike ordering.

- [ ] **Step 7: Verify**

```bash
./.venv/bin/python -m pytest core/nervous_system/tests/test_option_payoff.py core/nervous_system/tests/test_option_structures.py -q
```

Expected: the full approved suite passes and every disallowed short/ratio case
fails with a stable reason code.

- [ ] **Step 8: Commit**

```bash
git add core/nervous_system/execution/options/quotes.py core/nervous_system/execution/options/payoff.py core/nervous_system/execution/options/structures.py core/nervous_system/tests/test_option_payoff.py core/nervous_system/tests/test_option_structures.py
git commit -m "feat: validate bounded option structures"
```

### Task 18: Deterministic full-suite option selection

**Files:**

- Create: `core/nervous_system/config/options.py`
- Create: `core/nervous_system/execution/options/selector.py`
- Create: `core/nervous_system/tests/fixtures/option_chains.py`
- Create: `core/nervous_system/tests/test_option_selector.py`

**Interfaces:**

```python
def select_instrument(
    intent: TradeIntent,
    policy: PolicyDecision,
    snapshot: ContextSnapshot,
    chain: Sequence[OptionQuote],
    portfolio: PortfolioState,
    *,
    config: OptionSelectionConfig,
) -> InstrumentSelection:
    """Interface signature; required behavior is specified in the steps below."""
```

- [ ] **Step 1: Add failing chain-fitness tests**

Require:

- quote `available_at <= decision_time`;
- quote age at or below configured maximum;
- non-crossed bid/ask and positive midpoint;
- bid/ask spread at or below both absolute and percentage limits;
- minimum open interest and volume when configured;
- allowed DTE and expiration;
- broker-tradable contract;
- identical underlying and multiplier across constructed legs.

Trade bars alone do not satisfy quote fitness. Reject a chain with stale last
prints even if a recent historical trade exists.

- [ ] **Step 2: Add full-suite selection fixtures**

Build deterministic synthetic chains that produce each approved structure from
Task 17. Include ties, missing legs, illiquid legs, zero bids, bad strike
ordering, stale quotes, insufficient cash, and missing share coverage.

The Meta default preference list starts conservatively:

```text
equity, long_call_or_put, debit_vertical
```

The selector supports the full suite when an intent explicitly requests an
approved structure; Meta does not begin automatically choosing every structure
without a separately versioned strategy rule.

- [ ] **Step 3: Add deterministic scoring tests**

Candidate score components are explicit normalized measures for spread,
liquidity, DTE fit, delta-target distance, and premium/risk budget fit. Sort by:

```text
(-candidate_score, spread_percent, -open_interest, -volume, leg_symbol_tuple)
```

Assert the same chain in any input order yields the same selection and hash.
No component is called a probability.

- [ ] **Step 4: Run tests and observe failure**

```bash
./.venv/bin/python -m pytest core/nervous_system/tests/test_option_selector.py -q
```

Expected: missing selector/config failures.

- [ ] **Step 5: Implement candidate construction and policy filtering**

Construct only structures allowed by the intent preference list and policy.
Run `validate_structure` for every candidate before scoring. Apply portfolio
coverage/cash and incremental exposure limits. Record every rejected candidate
with reason codes, quote IDs, and risk profile; do not silently fall through.

- [ ] **Step 6: Implement explicit fallback**

Return one of:

```text
SELECTED_OPTION
SELECTED_EQUITY_FALLBACK
NO_ELIGIBLE_INSTRUMENT
```

Equity fallback is allowed only if the intent and policy permit it. Otherwise
return no order. The selection records the exact quote snapshot time, chosen
legs, estimated net price, maximum loss, collateral, and selection config hash.

- [ ] **Step 7: Verify**

```bash
./.venv/bin/python -m pytest core/nervous_system/tests/test_option_selector.py core/nervous_system/tests/test_option_payoff.py core/nervous_system/tests/test_option_structures.py -q
```

Expected: full-suite fixtures, deterministic ties, data-fitness vetoes, and
fallback behavior pass.

- [ ] **Step 8: Commit**

```bash
git add core/nervous_system/config/options.py core/nervous_system/execution/options/selector.py core/nervous_system/tests/fixtures/option_chains.py core/nervous_system/tests/test_option_selector.py
git commit -m "feat: select equity and option structures deterministically"
```

### Task 19: Alpaca paper broker interface and multi-leg request support

**Files:**

- Modify: `core/API/Alpaca_API/options/options_api.py`
- Create: `core/API/Alpaca_API/tests/test_options_orders.py`
- Create: `core/nervous_system/execution/broker.py`
- Create: `core/nervous_system/execution/alpaca_adapter.py`
- Create: `core/nervous_system/tests/test_alpaca_adapter.py`

**Interfaces:**

```python
class BrokerAdapter(Protocol):
    def account(self) -> BrokerAccount:
        """Interface signature; required behavior is specified in the steps below."""

    def positions(self) -> tuple[BrokerPosition, ...]:
        """Interface signature; required behavior is specified in the steps below."""

    def orders(self, *, status: str = "all") -> tuple[BrokerOrder, ...]:
        """Interface signature; required behavior is specified in the steps below."""

    def find_by_client_order_id(self, client_order_id: str) -> BrokerOrder | None:
        """Interface signature; required behavior is specified in the steps below."""

    def submit(self, request: OrderRequest) -> BrokerOrder:
        """Interface signature; required behavior is specified in the steps below."""

    def cancel(self, broker_order_id: str) -> BrokerOrder:
        """Interface signature; required behavior is specified in the steps below."""

    def replace(
        self,
        broker_order_id: str,
        replacement: OrderReplacement,
    ) -> BrokerOrder:
        """Interface signature; required behavior is specified in the steps below."""
```

- [ ] **Step 1: Add HTTP payload tests around the existing client**

Mock the existing `urllib` request boundary and assert:

- equity and single-option submits accept and send `client_order_id`;
- one-to-four-leg orders send `order_class="mleg"` and Alpaca leg objects;
- side/position-intent values map explicitly;
- quantities are whole contract quantities for multi-leg orders;
- lookup by client order ID uses the documented order endpoint/query;
- replace uses `PATCH /v2/orders/{order_id}`;
- HTTP 5xx on POST is surfaced and is not automatically retried;
- responses redact credentials from exception text.

- [ ] **Step 2: Run tests and observe failure**

```bash
./.venv/bin/python -m pytest core/API/Alpaca_API/tests/test_options_orders.py -q
```

Expected: missing arguments and multi-leg methods.

- [ ] **Step 3: Extend the existing client compatibly**

Add optional `client_order_id` to current submit methods without changing
existing callers. Add:

```python
def submit_multileg_order(
    self,
    *,
    legs: Sequence[Mapping[str, object]],
    qty: int,
    order_type: str,
    time_in_force: str,
    limit_price: Decimal | None,
    client_order_id: str,
) -> Mapping[str, object]:
    """Interface signature; required behavior is specified in the steps below."""


def get_order_by_client_order_id(
    self,
    client_order_id: str,
) -> Mapping[str, object] | None:
    """Interface signature; required behavior is specified in the steps below."""


def replace_order(
    self,
    order_id: str,
    payload: Mapping[str, object],
) -> Mapping[str, object]:
    """Interface signature; required behavior is specified in the steps below."""
```

The implementation replaces the declaration bodies. Follow the broker’s
current API response, not assumptions from equity-only behavior.

- [ ] **Step 4: Implement the inward-facing adapter**

`AlpacaPaperAdapter` validates that the configured base URL/account is paper
before submission. It translates broker payloads into contracts, preserves
unknown response fields in sanitized raw payload, and raises typed errors:

```text
BrokerRejected, BrokerUnavailable, BrokerAmbiguousSubmission,
BrokerAuthenticationError, BrokerContractError
```

It refuses `PRODUCTION_LIVE` before making an HTTP request.

- [ ] **Step 5: Add adapter tests**

Assert exact translation for accepted, rejected, partially filled, filled,
canceled, replaced, and missing orders. Assert broker IDs, client IDs,
timestamps, legs, filled quantities, and raw status are preserved.

- [ ] **Step 6: Verify**

```bash
./.venv/bin/python -m pytest core/API/Alpaca_API/tests/test_options_orders.py core/nervous_system/tests/test_alpaca_adapter.py -q
```

Expected: payload, no-POST-retry, paper-environment, and translation tests pass.

- [ ] **Step 7: Commit**

```bash
git add core/API/Alpaca_API/options/options_api.py core/API/Alpaca_API/tests/test_options_orders.py core/nervous_system/execution/broker.py core/nervous_system/execution/alpaca_adapter.py core/nervous_system/tests/test_alpaca_adapter.py
git commit -m "feat: support idempotent Alpaca paper option orders"
```

### Task 20: Always-on local and GCS execution journal

**Files:**

- Create: `core/nervous_system/execution/journal.py`
- Create: `core/nervous_system/tests/test_local_journal.py`
- Create: `core/nervous_system/tests/test_gcs_journal.py`
- Create: `core/nervous_system/tests/test_journal_hash_chain.py`

**Interfaces:**

```python
class ExecutionJournal(Protocol):
    def write(self, event: ExecutionJournalEvent) -> JournalReceipt:
        """Interface signature; required behavior is specified in the steps below."""

    def read(self, event_id: UUID) -> ExecutionJournalEvent:
        """Interface signature; required behavior is specified in the steps below."""

    def iter_events(
        self,
        *,
        account_id: str,
        after: datetime | None = None,
    ) -> Iterator[ExecutionJournalEvent]:
        """Interface signature; required behavior is specified in the steps below."""
```

Implementations:

```text
LocalAtomicJournal
GCSImmutableJournal
CompositeExecutionJournal
```

- [ ] **Step 1: Add failing local durability tests**

Use `tmp_path`. Assert each event is one canonical JSON file at:

```text
Data/execution_journal/YYYY/MM/DD/<account>/<event_time>_<event_id>.json
```

The writer must:

1. create a same-directory unique temp file;
2. write canonical UTF-8 bytes;
3. flush and `fsync` the file;
4. atomically rename to final name without overwrite;
5. `fsync` the parent directory;
6. return the content hash and final path.

Simulate failure before rename and assert no final partial file. Simulate the
same event twice and assert identical content is idempotent while conflicting
content for the same ID raises `JournalConflict`.

- [ ] **Step 2: Add failing GCS immutability tests**

Use a fake storage client. Assert object names use the same date/account/event
identity and upload with generation precondition `if_generation_match=0`.
Existing identical content is idempotent after hash verification; conflicting
content fails. Do not list the bucket to determine identity.

- [ ] **Step 3: Add failing chain tests**

Each event contains:

```text
event_id, event_time, account_id, environment, event_type, decision_id,
order_request_id, sequence_no, client_order_id, broker_order_id, payload,
previous_event_hash, event_hash, schema_version
```

Assert canonical hash linkage detects mutation, deletion between known
checkpoints, reordering, and order-chain mixing. Maintain one chain per
`order_request_id`; the submission intent is sequence one. Persist each
resulting journal receipt/hash with the matching submission attempt or
execution event in PostgreSQL.

- [ ] **Step 4: Run tests and observe failure**

```bash
./.venv/bin/python -m pytest core/nervous_system/tests/test_local_journal.py core/nervous_system/tests/test_gcs_journal.py core/nervous_system/tests/test_journal_hash_chain.py -q
```

Expected: missing journal failures.

- [ ] **Step 5: Implement local, GCS, and composite sinks**

`CompositeExecutionJournal` requires all configured sinks before an entry
submission is permitted:

- local development: local sink required;
- QA Cloud Run: GCS sink required and local ephemeral sink optional;
- PostgreSQL is not a journal sink; it is the operational authority updated
  separately by the gateway.

The journal payload excludes credentials, account secrets, and raw
authorization headers. Include complete sanitized broker response and request
identity.

- [ ] **Step 6: Verify**

```bash
./.venv/bin/python -m pytest core/nervous_system/tests/test_local_journal.py core/nervous_system/tests/test_gcs_journal.py core/nervous_system/tests/test_journal_hash_chain.py -q
```

Expected: atomicity, immutability, idempotency, and chain-tamper tests pass.

- [ ] **Step 7: Commit**

```bash
git add core/nervous_system/execution/journal.py core/nervous_system/tests/test_local_journal.py core/nervous_system/tests/test_gcs_journal.py core/nervous_system/tests/test_journal_hash_chain.py
git commit -m "feat: add durable execution backup journal"
```

### Task 21: Idempotent execution gateway and broker reconciliation

**Files:**

- Create: `core/nervous_system/execution/gateway.py`
- Create: `core/nervous_system/execution/reconciliation.py`
- Create: `core/nervous_system/execution/pending.py`
- Create: `core/nervous_system/tests/test_execution_gateway.py`
- Create: `core/nervous_system/tests/test_gateway_crash_recovery.py`
- Create: `core/nervous_system/tests/test_broker_reconciliation.py`

**Interfaces:**

```python
class ExecutionGateway:
    def submit(
        self,
        *,
        decision: DecisionRecord,
        request: OrderRequest,
    ) -> ExecutionResult:
        """Interface signature; required behavior is specified in the steps below."""

    def cancel(
        self,
        *,
        decision: DecisionRecord,
        broker_order_id: str,
    ) -> ExecutionResult:
        """Interface signature; required behavior is specified in the steps below."""

    def replace(
        self,
        *,
        decision: DecisionRecord,
        broker_order_id: str,
        replacement: OrderReplacement,
    ) -> ExecutionResult:
        """Interface signature; required behavior is specified in the steps below."""


def reconcile_broker_account(
    *,
    broker: BrokerAdapter,
    unit_of_work: NervousSystemUnitOfWork,
    journal: ExecutionJournal,
    account_id: str,
    since: datetime,
) -> ReconciliationReport:
    """Interface signature; required behavior is specified in the steps below."""
```

- [ ] **Step 1: Add failing entry-sequence tests**

Assert the exact entry path:

```text
validate DecisionRecord and OrderRequest hashes
verify environment/account/readiness
reserve idempotency key in PostgreSQL
write INTENT_TO_SUBMIT journal event
commit submission-attempt record
call broker once
write BROKER_RESPONSE journal event
persist broker order/execution event
publish outbox event in same database transaction
return result
```

No broker call occurs if the pre-submit journal or database reservation fails.

- [ ] **Step 2: Add crash-point and ambiguity tests**

Inject a crash:

1. before journal;
2. after journal/before broker;
3. during broker response;
4. after broker acceptance/before database update;
5. after database update/before caller receives result.

On restart, query the database and then broker by deterministic
`client_order_id`. Submit only when both prove no broker order exists and the
attempt is safe. Never blindly retry an ambiguous POST.

If the broker returns but the response journal write fails, do not report a
clean rejection or resubmit. Persist/return `AMBIGUOUS` when PostgreSQL is
available, retain the in-memory response for the immediate reconciliation
attempt, and resolve from broker lookup by client ID.

- [ ] **Step 3: Add exit fail-operational tests**

Risk-reducing cancels/exits attempt the broker action when PostgreSQL is down
if the broker adapter and required durable journal are healthy. Journal the
intent and response, return `RECONCILIATION_REQUIRED`, and backfill PostgreSQL
later. New entries still fail closed.

An exit may not increase absolute position, change side through zero, or open
new option exposure under the fail-operational exception.

- [ ] **Step 4: Add reconciliation tests**

Given broker orders/fills/positions absent from PostgreSQL:

- recover orders by client ID;
- append execution events without rewriting history;
- assign ownership from recovered fills;
- classify manual orders/positions as `UNASSIGNED`;
- detect broker/database quantity and status mismatches;
- recover journal-only events;
- keep broker values authoritative while recording discrepancies.

- [ ] **Step 5: Run tests and observe failure**

```bash
NERVOUS_SYSTEM_TEST_DATABASE_URL=postgresql+psycopg://cynolycus:cynolycus_dev_only@127.0.0.1:55432/cynolycus ./.venv/bin/python -m pytest core/nervous_system/tests/test_execution_gateway.py core/nervous_system/tests/test_gateway_crash_recovery.py core/nervous_system/tests/test_broker_reconciliation.py -q
```

Expected: missing gateway and reconciliation failures.

- [ ] **Step 6: Implement deterministic request identity**

Generate:

```text
order_request_id = UUIDv5(decision_id + request_hash)
client_order_id  = "cyno-" + environment_code + "-" + first_40_hex(request_hash)
```

Use two-character codes `dv`, `qp`, and `pl`, yielding exactly 48 ASCII
characters. Re-check Alpaca's current documented limit during Task 19 and make
the hash-prefix length a tested constant if the limit changes. The
same logical request returns the existing result; changed content receives a
new identity and requires a new policy decision.

- [ ] **Step 7: Implement gateway state machine**

`SubmissionAttemptStatus` transitions are:

```text
RESERVED -> JOURNALED -> SUBMITTING -> ACCEPTED|REJECTED|AMBIGUOUS
AMBIGUOUS -> ACCEPTED|REJECTED|RECONCILIATION_REQUIRED
```

After acceptance, broker lifecycle uses `ExecutionStatus`:

```text
ACCEPTED -> PARTIALLY_FILLED -> FILLED
ACCEPTED|PARTIALLY_FILLED -> CANCELED|EXPIRED
```

A replace creates a linked new `OrderRequest`; it is not an in-place terminal
status. Reject impossible backward transitions. Broker updates append
immutable events; the current order row is a projection.

- [ ] **Step 8: Implement pending intents**

Persist deferred entry as an unbound `TradeIntent` with expiry and reason—not
an OCC symbol or final order. At retry, rebuild a fresh snapshot, re-evaluate
policy, fetch a fresh chain, and produce a new decision/request linked to the
original intent.

- [ ] **Step 9: Verify**

```bash
NERVOUS_SYSTEM_TEST_DATABASE_URL=postgresql+psycopg://cynolycus:cynolycus_dev_only@127.0.0.1:55432/cynolycus ./.venv/bin/python -m pytest core/nervous_system/tests/test_execution_gateway.py core/nervous_system/tests/test_gateway_crash_recovery.py core/nervous_system/tests/test_broker_reconciliation.py -q
```

Expected: all sequence, crash, ambiguity, exit, recovery, and state-transition
tests pass with no duplicate broker submit.

- [ ] **Step 10: Commit**

```bash
git add core/nervous_system/execution/gateway.py core/nervous_system/execution/reconciliation.py core/nervous_system/execution/pending.py core/nervous_system/tests/test_execution_gateway.py core/nervous_system/tests/test_gateway_crash_recovery.py core/nervous_system/tests/test_broker_reconciliation.py
git commit -m "feat: route and reconcile orders idempotently"
```

### Task 22: Transactional outbox, jobs, and decision orchestration

**Files:**

- Create: `core/nervous_system/orchestration/events.py`
- Create: `core/nervous_system/orchestration/jobs.py`
- Create: `core/nervous_system/orchestration/outbox.py`
- Create: `core/nervous_system/orchestration/coordinator.py`
- Create: `core/nervous_system/tests/test_outbox.py`
- Create: `core/nervous_system/tests/test_decision_coordinator.py`
- Modify: `signals/meta_context/meta_ranker/run_4h_loop.py`

**Interfaces:**

```python
class DecisionCoordinator:
    def process_intent(
        self,
        intent: TradeIntent,
        *,
        policy_mode: PolicyMode,
        submit: bool,
    ) -> DecisionRecord:
        """Interface signature; required behavior is specified in the steps below."""


class JobRunner:
    def run_once(
        self,
        *,
        job_type: JobType,
        scheduled_for: datetime,
        payload: Mapping[str, object],
    ) -> JobResult:
        """Interface signature; required behavior is specified in the steps below."""
```

- [ ] **Step 1: Add failing outbox tests**

Assert a state/decision/execution write and its event are committed in one
transaction. A dispatcher claims rows with PostgreSQL
`FOR UPDATE SKIP LOCKED`, publishes to in-process handlers, records attempt
count/error, and marks success idempotently. No Kafka or external message bus
is added for the MVP.

- [ ] **Step 2: Add failing job-idempotency tests**

Key a job by `(job_type, scheduled_for, config_hash)`. Two workers cannot run
the same job concurrently. Stale leases can be recovered with an audit event.
Record scheduled/start/end times, host/revision, status, source hashes, output
IDs, counts, and exception summary.

- [ ] **Step 3: Add failing coordinator tests**

Assert the coordinator:

1. loads/builds a snapshot;
2. persists the intent;
3. evaluates and persists policy;
4. calculates exposure;
5. selects equity/options;
6. builds and persists the order request;
7. creates the complete `DecisionRecord`;
8. in dry-run returns before gateway;
9. in shadow submits the baseline intent request through the gateway while
   retaining the counterfactual policy;
10. in enforce submits only policy-approved final requests;
11. never submits production-live.

Failure in a required stage records a failed decision with stage and reason.

- [ ] **Step 4: Run tests and observe failure**

```bash
NERVOUS_SYSTEM_TEST_DATABASE_URL=postgresql+psycopg://cynolycus:cynolycus_dev_only@127.0.0.1:55432/cynolycus ./.venv/bin/python -m pytest core/nervous_system/tests/test_outbox.py core/nervous_system/tests/test_decision_coordinator.py -q
```

Expected: missing orchestration failures.

- [ ] **Step 5: Implement coordinator and job services**

Use dependency injection for repositories, clock, market calendar, broker,
quote provider, journal, and gateway. Do not import UI or module globals. A
`DecisionRecord` always stores all input/output IDs and hashes, even for a
veto, no eligible instrument, dry run, or failure.

- [ ] **Step 6: Tighten the existing 4H loop**

In `run_4h_loop.py`, keep the current bars → feeds → matrix → runner order.
Make every required stage return a typed result. Do not invoke the runner after
a failed/stale feed or matrix stage. Record the job and failure through the
orchestrator. Preserve current schedule/bar semantics.

- [ ] **Step 7: Verify**

```bash
NERVOUS_SYSTEM_TEST_DATABASE_URL=postgresql+psycopg://cynolycus:cynolycus_dev_only@127.0.0.1:55432/cynolycus ./.venv/bin/python -m pytest core/nervous_system/tests/test_outbox.py core/nervous_system/tests/test_decision_coordinator.py -q
./.venv/bin/python -m py_compile signals/meta_context/meta_ranker/run_4h_loop.py
```

Expected: atomic outbox, idempotent jobs, coordinator modes, and fail-fast loop
tests pass.

- [ ] **Step 8: Commit**

```bash
git add core/nervous_system/orchestration core/nervous_system/tests/test_outbox.py core/nervous_system/tests/test_decision_coordinator.py signals/meta_context/meta_ranker/run_4h_loop.py
git commit -m "feat: orchestrate auditable decisions and jobs"
```

### Task 23: Meta Ranker gateway cutover and current deferral preservation

**Files:**

- Modify: `signals/meta_context/meta_ranker/live_runner.py`
- Modify: `signals/meta_context/meta_ranker/options_exec.py`
- Modify: `signals/meta_context/meta_ranker/run_4h_loop.py`
- Modify: `UI/combined_server.py`
- Modify: `core/live_4h_exec.py`
- Modify: `core/tests/test_pending_open_deferral.py`
- Create: `signals/meta_context/meta_ranker/tests/test_gateway_cutover.py`
- Create: `signals/meta_context/meta_ranker/tests/test_pending_intent_refresh.py`

**Interfaces:**

Add CLI/runtime settings:

```text
--nervous-system-mode off|shadow|enforce
--environment development|qa-paper|production-live
--submit
```

Rules:

```text
off     = decision/audit only; --submit is rejected
shadow  = baseline request may reach QA-paper through gateway
enforce = policy-final request may reach QA-paper through gateway
```

- [ ] **Step 1: Add failing no-bypass tests**

Patch the coordinator and current Alpaca client. Assert:

- every Meta candidate becomes an intent;
- every current trim/full-exit action becomes an `EXIT` or `REDUCE` intent
  with the same symbol, quantity, and reason;
- `--submit` calls only `ExecutionGateway`;
- no code path in Meta calls `submit_order` or `submit_option_order` directly;
- Meta production modules no longer instantiate `AlpacaOptionsClient`;
- `off + --submit` fails fast;
- `production-live` creates a veto and performs no broker HTTP call;
- shadow and enforce decisions have complete records.

- [ ] **Step 2: Add failing deferral tests**

Preserve the concurrent 2026-07-30 fix: readiness gating occurs before
after-close deferral. Assert:

- a readiness failure does not queue an entry;
- an after-close eligible entry queues `TradeIntent`, not a final order/OCC
  contract;
- next-open retry rebuilds context, policy, quote chain, and order;
- expired or now-vetoed intent is not submitted;
- exits are not blocked or converted into pending entries.

- [ ] **Step 3: Run focused tests and observe failure**

```bash
./.venv/bin/python -m pytest signals/meta_context/meta_ranker/tests/test_gateway_cutover.py signals/meta_context/meta_ranker/tests/test_pending_intent_refresh.py core/tests/test_pending_open_deferral.py -q
```

Expected: direct execution and pending-order behavior fail the new assertions.

- [ ] **Step 4: Replace direct Meta submission**

Inject `DecisionCoordinator` into `live_runner.py`. Remove the direct Alpaca
submit path from `_execute`; convert its current trim/full-exit plan rows to
risk-reducing intents and route them through the same coordinator/gateway.
Broker account/position reads go through the inward broker adapter. Keep
ranking output and dashboard rows compatible by projecting `DecisionRecord`
fields into the current result shape.

No automated Meta entry may bypass the gateway. The old direct code may remain
only as pure non-submitting fixture support until parity tests are deleted; it
may not import credentials or call a broker. Convert `options_exec.py` into a
compatibility wrapper over the shared quote provider/selector or remove its
production import after its current routing tests have equivalent shared
selector coverage.

- [ ] **Step 5: Convert deferrals to pending intents**

Update `core/live_4h_exec.py` and Meta integration so pending records contain
intent identity, original decision bar, retry window, and deferral reason.
Reuse the current startup queue and job guard, but route refresh through the
coordinator. Preserve existing sell/exit ladder quantities and reasons through
frozen parity fixtures, then exercise their gateway path in
`test_gateway_cutover.py`.

- [ ] **Step 6: Update combined-server startup**

Default combined-server Meta configuration to:

```text
environment=qa-paper
nervous_system_mode=shadow
submit=false
```

Require explicit paper credentials and `submit=true` for paper order
submission. Do not expose any route that enables `production-live`.

- [ ] **Step 7: Run Meta and deferral regression tests**

```bash
./.venv/bin/python -m pytest signals/meta_context/meta_ranker/tests core/tests/test_pending_open_deferral.py -q
./.venv/bin/python -m py_compile signals/meta_context/meta_ranker/live_runner.py signals/meta_context/meta_ranker/options_exec.py signals/meta_context/meta_ranker/run_4h_loop.py UI/combined_server.py core/live_4h_exec.py
rg -n "AlpacaOptionsClient|submit_order|submit_option_order" signals/meta_context/meta_ranker
```

Expected: tests pass; the final search returns no direct submission call in
Meta production code.

- [ ] **Step 8: Commit**

```bash
git add signals/meta_context/meta_ranker/live_runner.py signals/meta_context/meta_ranker/options_exec.py signals/meta_context/meta_ranker/run_4h_loop.py signals/meta_context/meta_ranker/tests UI/combined_server.py core/live_4h_exec.py core/tests/test_pending_open_deferral.py
git commit -m "feat: cut Meta execution over to nervous system"
```

### Task 24: Historical replay, outcome attribution, and source-fitness gates

**Files:**

- Create: `core/nervous_system/config/source_fitness.py`
- Create: `core/nervous_system/replay/providers.py`
- Create: `core/nervous_system/replay/source_fitness.py`
- Create: `core/nervous_system/replay/runner.py`
- Create: `core/nervous_system/tests/test_replay_determinism.py`
- Create: `core/nervous_system/tests/test_source_fitness.py`
- Create: `core/nervous_system/tests/test_outcome_attribution.py`
- Modify: `strategies/intraday_structure/replay.py`

**Interfaces:**

```python
class HistoricalProvider(Protocol):
    def states_available(
        self,
        *,
        state_type: StateType,
        entity_id: str,
        decision_time: datetime,
    ) -> tuple[StateContract, ...]:
        """Interface signature; required behavior is specified in the steps below."""


def assess_option_source_fitness(
    positions: Sequence[HistoricalOptionPosition],
    *,
    underlying_returns: pd.Series,
    option_observations: pd.DataFrame,
) -> SourceFitnessReport:
    """Interface signature; required behavior is specified in the steps below."""


def replay_decisions(
    *,
    provider: HistoricalProvider,
    schedule: Sequence[ReplayDecisionPoint],
    strategy: ReplayStrategy,
    coordinator: DecisionCoordinator,
) -> ReplayReport:
    """Interface signature; required behavior is specified in the steps below."""
```

- [ ] **Step 1: Add failing replay-causality tests**

Build a small historical state set where one revision arrives after the
decision. Assert replay chooses only states with
`available_at <= decision_time < valid_until`. Append future revisions and
assert all earlier snapshot/decision hashes remain unchanged.

- [ ] **Step 2: Add failing live/replay parity tests**

Feed identical contracts to live-mode and historical providers. Assert
snapshot hash, intent hash, policy decision, exposure report, instrument
selection, and order-request hash are identical. Only execution result differs
because replay uses a simulated broker.

- [ ] **Step 3: Add failing option-data fitness tests**

Before option P&L, compute and report:

```text
bars/quotes per instrument
quote age at entry and exit
share of positions with identical entry/exit prices
share using trades versus bid/ask marks
corr(long call return, underlying return)
corr(long put return, -underlying return)
entitlement/feed metadata
```

Use test fixtures proving stale prints produce low derivative/underlying
correlation and identical prices. Assert the runner returns
`SOURCE_UNFIT_FOR_OPTION_PNL` and computes no option performance metrics.

Version the initial hard gates:

```text
minimum matched positions = 30
minimum unique sessions = 10
minimum valid bid/ask mark coverage = 95%
maximum identical entry/exit mark share = 5%
minimum corr(long call return, underlying return) = +0.70
minimum corr(long put return, -underlying return) = +0.70
warning correlation threshold = +0.85
maximum quote age = the selection config's quote-age limit
```

If sample size is below either minimum, return
`SOURCE_FITNESS_INSUFFICIENT_SAMPLE` and withhold option P&L conclusions.
Report Pearson and rank correlation, but use the versioned Pearson threshold
for the initial gate. Threshold changes require a new config hash.

- [ ] **Step 4: Add failing outcome-attribution tests**

Create outcomes at fixed horizons from broker fills or fit-for-purpose marks.
Assert:

- evaluation time is explicit;
- prices available after evaluation are excluded;
- order/fill slippage is separated from signal movement;
- strategy, snapshot, policy, selection, and execution IDs remain linked;
- revisions append a new outcome version rather than overwriting.

- [ ] **Step 5: Run tests and observe failure**

```bash
./.venv/bin/python -m pytest core/nervous_system/tests/test_replay_determinism.py core/nervous_system/tests/test_source_fitness.py core/nervous_system/tests/test_outcome_attribution.py -q
```

Expected: missing replay and fitness modules.

- [ ] **Step 6: Implement provider abstraction and replay runner**

Reuse the same snapshot, policy, exposure, selection, and coordinator code.
The replay clock advances only to scheduled decision points. Providers return
immutable evidence; they cannot inspect rows beyond the requested time.
Randomized simulated fills require an explicit seed stored in the report.

Bridge the useful time-stepping patterns from
`strategies/intraday_structure/replay.py` without making the shared replay
package depend on that strategy.

- [ ] **Step 7: Implement outcome and source-fitness persistence**

Persist fitness reports, replay run identity, source hashes, schedule, config
versions, seed, decisions, simulated execution assumptions, and outcomes.
Block only unsupported option P&L; underlying-only decision evaluation may
continue with an explicit limitation.

- [ ] **Step 8: Verify**

```bash
./.venv/bin/python -m pytest core/nervous_system/tests/test_replay_determinism.py core/nervous_system/tests/test_source_fitness.py core/nervous_system/tests/test_outcome_attribution.py -q
```

Expected: deterministic parity, future invariance, source veto, and versioned
outcome tests pass.

- [ ] **Step 9: Commit**

```bash
git add core/nervous_system/config/source_fitness.py core/nervous_system/replay core/nervous_system/tests/test_replay_determinism.py core/nervous_system/tests/test_source_fitness.py core/nervous_system/tests/test_outcome_attribution.py strategies/intraday_structure/replay.py
git commit -m "feat: replay nervous system decisions causally"
```

### Task 25: Read-only audit API, dashboard, and operational alerts

**Files:**

- Create: `core/nervous_system/orchestration/read_models.py`
- Create: `core/nervous_system/orchestration/alerts.py`
- Create: `core/nervous_system/tests/test_read_models.py`
- Create: `core/nervous_system/tests/test_alerts.py`
- Modify: `UI/meta_ranker_dashboard.py`
- Modify: `UI/combined_server.py`

**Interfaces:**

Read-only endpoints:

```text
GET /api/nervous-system/health
GET /api/nervous-system/decisions?limit=N&strategy_id=meta_ranker
GET /api/nervous-system/decisions/{decision_id}
GET /api/nervous-system/orders/{order_request_id}
GET /api/nervous-system/reconciliation/latest
GET /api/nervous-system/jobs?limit=N
GET /api/nervous-system/alerts?status=open
```

- [ ] **Step 1: Add failing projection tests**

Create repository fixtures and assert projections include:

- decision time/bar and all state IDs/versions;
- raw intent and raw score;
- policy action, reasons, modifier waterfall, final size;
- selected instrument and rejected candidates;
- order/request/client/broker IDs;
- execution timeline and fills;
- ownership and reconciliation status;
- source quality/freshness warnings;
- mode/environment and production-live veto.

Assert secrets and credential-bearing raw headers never appear.

- [ ] **Step 2: Add failing alert tests**

Emit deduplicated, persisted alerts for:

```text
required state stale/missing
database unavailable
journal unavailable
ambiguous submission
broker/database mismatch
unassigned position
failed job
future-timestamp state
invalid option source fitness
production-live attempt
```

Each alert has first/last seen, count, severity, entity, related IDs, status,
and acknowledgement metadata. Repeated detection updates the projection but
retains immutable alert events.

- [ ] **Step 3: Run tests and observe failure**

```bash
NERVOUS_SYSTEM_TEST_DATABASE_URL=postgresql+psycopg://cynolycus:cynolycus_dev_only@127.0.0.1:55432/cynolycus ./.venv/bin/python -m pytest core/nervous_system/tests/test_read_models.py core/nervous_system/tests/test_alerts.py -q
```

Expected: missing read/alert services.

- [ ] **Step 4: Implement bounded read models**

Use repository queries with indexed filters and explicit default/max limits.
Do not deserialize every state row to render a list. Detail endpoints may load
the complete immutable decision graph. Return UTC ISO-8601 timestamps and
stable reason labels.

- [ ] **Step 5: Add the dashboard audit views**

Extend the current Meta dashboard with:

```text
Nervous System Health
Latest Decision Waterfall
State Freshness
Portfolio Exposure
Orders and Fills
Reconciliation
Open Alerts
```

Display `SHADOW` or `ENFORCE` prominently. Disable/remove the real-money
toggle; a `production-live` value is display-only and must show `BLOCKED BY
MVP POLICY`. Preserve existing ranking and audit views.

- [ ] **Step 6: Register read-only routes**

Add the endpoints to the existing combined-server handler. They may not
submit, cancel, replace, acknowledge, or mutate trading state. If the database
is unavailable, health returns a degraded response and all other endpoints
return a sanitized 503.

- [ ] **Step 7: Verify**

```bash
NERVOUS_SYSTEM_TEST_DATABASE_URL=postgresql+psycopg://cynolycus:cynolycus_dev_only@127.0.0.1:55432/cynolycus ./.venv/bin/python -m pytest core/nervous_system/tests/test_read_models.py core/nervous_system/tests/test_alerts.py -q
./.venv/bin/python -m py_compile UI/meta_ranker_dashboard.py UI/combined_server.py
```

Expected: bounded projections, alert deduplication, secret redaction, and
read-only routing pass.

- [ ] **Step 8: Commit**

```bash
git add core/nervous_system/orchestration/read_models.py core/nervous_system/orchestration/alerts.py core/nervous_system/tests/test_read_models.py core/nervous_system/tests/test_alerts.py UI/meta_ranker_dashboard.py UI/combined_server.py
git commit -m "feat: expose nervous system audit views"
```

### Task 26: Cloud SQL, GCS journal, QA-paper deployment, and recovery runbook

**Files:**

- Modify: `.env.example`
- Modify: `core/nervous_system/config/runtime.py`
- Create: `scripts/cloud/nervous_system_db.py`
- Create: `scripts/cloud/verify_nervous_system_qa.py`
- Create: `core/nervous_system/tests/test_cloud_runtime_config.py`
- Modify: `docs/GCP_MIGRATION_TUTORIAL.md`
- Create: `docs/nervous_system/OPERATIONS_RUNBOOK.md`

**Interfaces:**

Runtime configuration (environment-specific validators determine which
optional values become required):

```text
CYNOLYCUS_ENVIRONMENT=development|qa-paper|production-live
CYNOLYCUS_NERVOUS_SYSTEM_MODE=off|shadow|enforce
CYNOLYCUS_DATABASE_URL
CYNOLYCUS_DB_POOL_SIZE
CYNOLYCUS_DB_MAX_OVERFLOW
CYNOLYCUS_OPERATIONAL_ROOT
CYNOLYCUS_EXECUTION_JOURNAL=local|gcs
CYNOLYCUS_EXECUTION_JOURNAL_DIR
CYNOLYCUS_EXECUTION_JOURNAL_BUCKET
CYNOLYCUS_GCP_PROJECT
CYNOLYCUS_CLOUD_SQL_INSTANCE_CONNECTION_NAME
```

- [ ] **Step 1: Add failing Cloud runtime tests**

Assert:

- development accepts TCP localhost PostgreSQL and local journal;
- QA-paper requires Cloud SQL settings, GCS journal, Alpaca paper URL/account,
  and Secret Manager-provided credentials;
- production-live always fails execution validation;
- Cloud SQL Unix socket URLs render in the tested form
  `postgresql+psycopg://ns_app:REDACTED@/cynolycus?host=/cloudsql/example-project:us-east1:cynolycus-qa`;
- secrets are never printed by config repr, validation errors, or health.

- [ ] **Step 2: Run the test and observe failure**

```bash
./.venv/bin/python -m pytest core/nervous_system/tests/test_cloud_runtime_config.py -q
```

Expected: Cloud runtime validation is missing.

- [ ] **Step 3: Implement idempotent database administration CLI**

`scripts/cloud/nervous_system_db.py` supports:

```text
create-database
upgrade-schema
schema-status
import-history --dry-run
import-history --write
verify-counts
verify-backup
```

It calls application services and Alembic; it does not embed passwords or
shell out with secrets in command arguments. Destructive downgrade/drop
operations are intentionally absent.

- [ ] **Step 4: Document the migration sequence**

Update the current GCP tutorial without discarding its in-progress edits:

1. finish the active immutable raw/GCS data foundation;
2. develop and test PostgreSQL locally with Compose;
3. create private/IP-restricted Cloud SQL PostgreSQL in the QA project;
4. create database/user via Secret Manager and least privilege;
5. enable Cloud Run’s Cloud SQL connection;
6. create an immutable GCS journal bucket with retention/versioning policy;
7. deploy schema migration as a one-shot Cloud Run job;
8. run historical import dry-run, review counts/quarantine, then write;
9. deploy app in `qa-paper`, `shadow`, `submit=false`;
10. run reconciliation and health checks;
11. enable QA-paper submission only after acceptance gates.

Do not use Database Migration Service for the local historical files: they are
recreated through the idempotent source registry/import. DMS remains a future
option only for a continuous PostgreSQL-to-PostgreSQL migration.

- [ ] **Step 5: Write the recovery runbook**

Include exact procedures for:

- PostgreSQL unavailable before entry;
- journal unavailable;
- ambiguous broker submission;
- app crash with broker order accepted;
- complete app/server outage and restart;
- Cloud SQL point-in-time recovery into a new instance;
- GCS journal verification and PostgreSQL reconstruction;
- manual broker order/position becoming `UNASSIGNED`;
- rollback from enforce to shadow;
- credential rotation;
- disabling all QA-paper entries while retaining exits/reconciliation.

Every procedure begins with read-only diagnosis and names the source of truth.

- [ ] **Step 6: Implement QA verification CLI**

The verification command checks Cloud SQL connectivity/schema revision,
journal write/read, paper broker identity, latest source freshness,
reconciliation status, production-live veto, and a no-submit synthetic
decision. It returns nonzero on any required failure and emits sanitized JSON.

- [ ] **Step 7: Verify**

```bash
./.venv/bin/python -m pytest core/nervous_system/tests/test_cloud_runtime_config.py -q
./.venv/bin/python -m py_compile scripts/cloud/nervous_system_db.py scripts/cloud/verify_nervous_system_qa.py
```

Expected: environment separation, secret redaction, Cloud SQL URL, and CLI
syntax tests pass.

- [ ] **Step 8: Commit**

```bash
git add .env.example core/nervous_system/config/runtime.py scripts/cloud/nervous_system_db.py scripts/cloud/verify_nervous_system_qa.py core/nervous_system/tests/test_cloud_runtime_config.py docs/GCP_MIGRATION_TUTORIAL.md docs/nervous_system/OPERATIONS_RUNBOOK.md
git commit -m "docs: add Cloud SQL QA paper operations"
```

### Task 27: MVP acceptance, shadow soak, and enforce promotion

**Files:**

- Create: `core/nervous_system/tests/test_mvp_acceptance.py`
- Create: `scripts/validate_nervous_system_mvp.py`
- Create: `docs/nervous_system/MVP_ACCEPTANCE.md`
- Modify: `LIVING_SUMMARY.md`

**Interfaces:**

The validation command emits one sanitized report with:

```text
schema_revision
source_artifact/import/quarantine counts
state counts and freshness by type
future-append invariance result
Meta baseline parity result
policy reason/modifier coverage
option structure/selector coverage
journal/hash-chain result
gateway crash/idempotency result
broker reconciliation result
decision completeness result
production-live veto result
test commands and revisions
```

- [ ] **Step 1: Add the end-to-end acceptance test**

Using PostgreSQL fixtures, a fake paper broker, a fake quote provider, and a
local journal:

1. publish market, sector, ticker, theme, catalyst, dealer, readiness, and
   portfolio states;
2. build a snapshot at 16:20;
3. generate a Meta intent;
4. evaluate policy;
5. calculate overlap exposure;
6. select a liquid bounded option spread;
7. build a four-leg-capable order request;
8. submit once through the gateway;
9. process partial and final fills;
10. assign ownership;
11. reconcile;
12. rebuild/replay and compare hashes;
13. render the audit read model.

Assert every persisted object links to the prior one and the final
`DecisionRecord` is complete.

- [ ] **Step 2: Add acceptance failure cases**

Assert no broker submission for:

- stale required state;
- future state;
- database outage on entry;
- journal outage;
- `production-live`;
- invalid broker paper identity;
- naked/ratio/unknown-loss options;
- stale/crossed/print-only option data;
- breached portfolio limit;
- duplicate request.

Assert a risk-reducing exit still follows the journaled fail-operational path.

- [ ] **Step 3: Run focused and broad local verification**

```bash
docker compose -f compose.nervous-system.yaml up -d postgres
NERVOUS_SYSTEM_TEST_DATABASE_URL=postgresql+psycopg://cynolycus:cynolycus_dev_only@127.0.0.1:55432/cynolycus ./.venv/bin/python -m pytest core/nervous_system/tests core/API/Alpaca_API/tests signals/market_regime/tests themes/dynamic_theme/tests signals/catalysts/tests signals/meta_context/meta_ranker/tests strategies/dealer_positioning/tests core/tests/test_pending_open_deferral.py -q
./.venv/bin/python -m compileall -q core/nervous_system signals/market_regime themes/dynamic_theme signals/catalysts signals/meta_context/meta_ranker strategies/dealer_positioning UI scripts/cloud
```

Expected: all targeted suites pass. Record exact counts and any unrelated
pre-existing failures in the acceptance document.

- [ ] **Step 4: Run historical import review**

```bash
./.venv/bin/python -m core.nervous_system.data_registry.import_legacy --manifest core/nervous_system/config/legacy_sources.toml --database-url postgresql+psycopg://cynolycus:cynolycus_dev_only@127.0.0.1:55432/cynolycus --dry-run
./.venv/bin/python -m core.nervous_system.data_registry.import_legacy --manifest core/nervous_system/config/legacy_sources.toml --database-url postgresql+psycopg://cynolycus:cynolycus_dev_only@127.0.0.1:55432/cynolycus
./.venv/bin/python scripts/validate_nervous_system_mvp.py --environment development
```

Review source-by-source discovered/imported/quarantined counts before the
write command. Rerun import and assert imported count is zero with all existing
rows classified as duplicates.

- [ ] **Step 5: Document implementation acceptance**

Populate `MVP_ACCEPTANCE.md` with actual revisions, commands, counts,
limitations, and rollback. Technical implementation acceptance requires:

```text
all MVP tests passing
zero duplicate broker submissions in crash tests
zero production-live submissions
zero missing links in accepted DecisionRecords
100% approved option structures covered by bounded-risk tests
historical import idempotent
future-append invariance passing
reconciliation clean or every discrepancy explicitly classified
```

- [ ] **Step 6: Run QA-paper shadow soak**

Deploy `qa-paper`, `shadow`, initially `submit=false`. Compare at least 20
trading sessions and at least 100 eligible Meta intents, whichever takes
longer. After no-submit validation, permit a controlled paper-submit subset.
Promotion evidence must include:

```text
exact raw ranking parity
baseline request versus shadow policy differences
freshness/missing-state rates
policy reason/modifier distributions
option selection/rejection distributions
paper rejection/partial-fill/slippage rates
journal versus PostgreSQL versus broker reconciliation
zero duplicate submissions
zero unresolved ambiguous submissions
zero production-live attempts reaching broker
```

If 100 intents cannot be reached in 20 sessions, extend the soak rather than
lowering the sample threshold.

- [ ] **Step 7: Promote only hard safety, then selected context rules**

First set `enforce` for hard environment, readiness, idempotency, bounded-loss,
liquidity, and portfolio-limit rules while contextual multipliers remain
shadow-only. After a separately reviewed comparison demonstrates expected
behavior, enable selected non-increasing context modifiers one rule at a time.
Record the policy config hash and effective timestamp for every promotion.

- [ ] **Step 8: Run the post-pass**

Answer in `MVP_ACCEPTANCE.md`:

```text
What requested work is unfinished?
Which sources remain quarantined or semantically ambiguous?
Which tests or validation did not run?
Where can an automated order still bypass the gateway?
Can any state enter a snapshot before availability?
Can any option order have unbounded or unknown maximum loss?
Can an entry occur without PostgreSQL and the required journal?
Can production-live reach a broker?
Can broker truth be reconstructed after app outage?
```

Any unsafe answer blocks acceptance.

- [ ] **Step 9: Update continuity and commit**

Append a maximum-three-line implementation result to `LIVING_SUMMARY.md`.

```bash
git add core/nervous_system/tests/test_mvp_acceptance.py scripts/validate_nervous_system_mvp.py docs/nervous_system/MVP_ACCEPTANCE.md LIVING_SUMMARY.md
git commit -m "test: accept nervous system MVP"
```

## Post-MVP rollout roadmap

The following work is intentionally outside the first enforced vertical slice.
Each phase requires its own approved design delta and implementation plan.

### Phase R1: Momentum Expansion and HTF Swing migration

- Adapt `strategies/momentum_expansion` and
  `strategies/multi_ticker_swing_htf` outputs to `TradeIntent`.
- Preserve the current readiness-before-after-close-deferral behavior.
- Prove score/signal parity on frozen fixtures and future-append invariance.
- Run shadow comparison before removing direct order paths.
- Route entries, exits, startup recovery, and pending intents through the
  gateway.
- Acceptance: repository scan shows no broker submission outside the gateway
  for migrated modules; ownership/reconciliation identifies each fill.

### Phase R2: Dealer Positioning and SPY intraday consumers

- Promote dealer state only after quote/trade source fitness is measured.
- Add dealer-specific intent and policy rules without turning heuristic levels
  into probabilities.
- Reuse shared replay while preserving exact event/observation times.
- Migrate `strategies/intraday_structure` and related SPY paths one at a time.
- Acceptance: option P&L is produced only from valid marks/fills and long
  call/put direction sanity checks pass.

### Phase R3: Calibrated context engines and controlled modulation

- Version and validate explicit market/sector/theme regime classifiers.
- Build calibrated probability models only with train-only transformations,
  walk-forward validation, and fixed out-of-sample evaluation.
- Add ablation and rule-level counterfactual reports.
- Permit only non-increasing context multipliers until robust QA-paper evidence
  justifies a separately approved expansion.
- Acceptance: each rule improves a named baseline across time/regimes without
  leakage and with adequate sample size.

### Phase R4: Additional strategies and operational hardening

- Migrate remaining news/social/catalyst consumers and execution modules.
- Add retention/partition maintenance, read replicas only if measured load
  requires them, and restore drills.
- Remove obsolete direct state/execution stores only after hash/count parity
  and rollback windows expire.
- Acceptance: one audited state/policy/execution path exists for every
  automated QA-paper strategy.

### Phase R5: Production-live design review

Production-live is not an MVP rollout step. A separate explicit user approval,
threat/risk review, broker-account isolation, credentials, limits, kill switch,
dual confirmation, incident procedures, and paper evidence package are
required. Until that review is implemented and approved,
`PRODUCTION_LIVE` remains a hard policy veto.

## External interface references

Implementation workers must re-check current official documentation before
broker or cloud integration because these APIs can change:

- Alpaca options Level 3 and multi-leg behavior:
  <https://docs.alpaca.markets/us/docs/options-level-3-trading>
- Alpaca create-order API:
  <https://docs.alpaca.markets/us/v1.4.2/reference/postorder>
- Alpaca options overview and paper eligibility:
  <https://docs.alpaca.markets/us/docs/options-trading-overview>
- Cloud Run to Cloud SQL for PostgreSQL:
  <https://docs.cloud.google.com/sql/docs/postgres/connect-run>
- Google Database Migration Service scope:
  <https://docs.cloud.google.com/database-migration/docs/overview>

## Definition of done

The MVP is done only when:

- all 27 tasks are checked and their focused tests pass;
- the exact historical operational corpus is imported or quarantined with
  reasons, while Parquet remains analytical authority;
- Meta has causal state → snapshot → intent → policy → exposure → selection →
  request → journal → paper broker → fill → ownership → outcome lineage;
- equity and every approved one-to-four-leg options structure have bounded-risk
  validation and deterministic selection support;
- PostgreSQL, immutable execution journal, and Alpaca can be reconciled after a
  process or server outage;
- no Meta automated order bypasses the gateway;
- QA-paper shadow evidence meets the stated soak gate before enforce promotion;
- production-live remains impossible through the MVP policy and UI;
- Cloud SQL/GCS deployment and recovery procedures have been exercised;
- `MVP_ACCEPTANCE.md` contains actual evidence rather than planned results.
