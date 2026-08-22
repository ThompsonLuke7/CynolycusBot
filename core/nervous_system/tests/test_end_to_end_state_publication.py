"""Publish -> snapshot -> policy, against a real database.

This is the test that was missing. `core/nervous_system/tests` had 1209 passing
tests over the pure rules and 110 skipped ones over the persistence layer, and
nothing at all that ran a state through publication, snapshot selection and
policy evaluation together. So four independent faults could each keep the Meta
Ranker from submitting a single order for two consecutive sessions
(2026-08-20/21) with a fully green suite:

1.  MARKET/SECTOR had no reachable production publisher.
2.  TICKER had no production caller at all.
3.  The liquidity rule read a metric no producer wrote.
4.  `snapshot_vetoes` gated on `stale_inputs` / `missing_inputs`, which the
    evaluator fills in for optional rules too, so an absent THEME vetoed as hard
    as an absent TICKER.

Each of 1, 2 and 4 is caught here by omission: drop the corresponding state and
the assertion at the end fails. 3 is caught by `test_liquidity_metric_is_dollars`.

Requires NERVOUS_SYSTEM_TEST_DATABASE_URL, pointing at a DISPOSABLE database.
"""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from uuid import NAMESPACE_URL, uuid4, uuid5

import pytest

from core.nervous_system.config.freshness import get_snapshot_profile
from core.nervous_system.config.policy import MVP_POLICY_CONFIG
from core.nervous_system.context.snapshot_builder import SnapshotBuilder
from core.nervous_system.contracts.enums import (
    DecisionKind,
    Direction,
    InstrumentFamily,
    MarketRegime,
    PolicyAction,
    StateType,
    TickerSetup,
)
from core.nervous_system.contracts.intent import TradeIntent
from core.nervous_system.contracts.quality import DataQualitySummary
from core.nervous_system.contracts.states import (
    MarketState,
    PortfolioState,
    ReadinessState,
    SectorState,
    TickerState,
)
from core.nervous_system.persistence.repositories.state import StateRepository
from core.nervous_system.policy.engine import evaluate_policy
from core.nervous_system.policy.reason_codes import ReasonCode

UTC = timezone.utc
TICKER = "CRWD"
SECTORS = ("XLK", "XLF")
DECISION_BAR = datetime(2026, 8, 21, 14, 0, tzinfo=UTC)
DECISION_TIME = datetime(2026, 8, 21, 18, 20, tzinfo=UTC)
PROFILE = get_snapshot_profile("meta_4h_1420@1")
# Comfortably above MVP_POLICY_CONFIG.min_liquidity_value ($5,000,000).
LIQUID = 1_555_563_808.0


# Mirror config/freshness.py MVP_POLICY_DEFAULTS. MARKET in particular carries a
# one-session lag (market_session_lag=1), so a 6h validity would expire it before
# the decision it is meant to inform — which is a fixture bug, not a rule.
_VALIDITY = {
    StateType.TICKER: timedelta(hours=6),
    StateType.PORTFOLIO: timedelta(hours=24),
    StateType.MARKET: timedelta(hours=96),
    StateType.SECTOR: timedelta(hours=96),
    StateType.READINESS: timedelta(hours=96),
}


def _envelope(*, state_type: StateType, entity_id: str, as_of: datetime, tag: str) -> dict:
    return {
        "state_id": uuid5(NAMESPACE_URL, f"e2e/{tag}/{entity_id}/{as_of.isoformat()}"),
        "state_type": state_type,
        "entity_id": entity_id,
        "as_of": as_of,
        "available_at": as_of,
        "generated_at": as_of,
        "valid_until": as_of + _VALIDITY.get(state_type, timedelta(hours=96)),
        "source_window_start": as_of,
        "source_window_end": as_of,
        "schema_version": 1,
        "producer": f"e2e.{tag}",
        "model_version": "e2e@1",
        "feature_version": "e2e@1",
        "config_version": "e2e@1",
        "lineage_ids": (f"e2e:{tag}:{entity_id}",),
        "data_quality": DataQualitySummary(),
    }


def _required_states(*, liquidity: float | None = LIQUID) -> dict[StateType, list]:
    """One of each required state, keyed by type so a test can drop one."""

    # MARKET carries a one-session lag: the profile sets market_session_lag=1.
    market_as_of = datetime(2026, 8, 20, 20, 0, tzinfo=UTC)
    metrics = {"beta_spy_60": 1.1}
    if liquidity is not None:
        metrics["dollar_volume_20d"] = liquidity

    return {
        StateType.MARKET: [
            MarketState(
                **_envelope(
                    state_type=StateType.MARKET, entity_id="US",
                    as_of=market_as_of, tag="market",
                ),
                regime=MarketRegime.NEUTRAL,
                risk_on_probability=None,
                risk_off_probability=None,
                metrics={"risk_appetite_xly_xlp_z": 0.4},
                reason_codes=("E2E",),
            )
        ],
        StateType.SECTOR: [
            SectorState(
                **_envelope(
                    state_type=StateType.SECTOR, entity_id=sector,
                    as_of=DECISION_BAR, tag="sector",
                ),
                sector_id=sector,
                sector_regime="NEUTRAL",
                rotation_rank=0.9,
            )
            for sector in SECTORS
        ],
        StateType.TICKER: [
            TickerState(
                **_envelope(
                    state_type=StateType.TICKER, entity_id=TICKER,
                    as_of=DECISION_BAR, tag="ticker",
                ),
                ticker=TICKER,
                selected_bar=DECISION_BAR,
                reference_price=190.295,
                ticker_setup=TickerSetup.UNKNOWN,
                metrics=metrics,
                transition_probabilities={},
            )
        ],
        StateType.PORTFOLIO: [
            PortfolioState(
                **_envelope(
                    state_type=StateType.PORTFOLIO, entity_id="paper",
                    as_of=DECISION_BAR, tag="portfolio",
                ),
                account_alias="paper",
                equity=761662.00,
                buying_power=1500000.00,
                cash=-348867.90,
                positions=(),
                open_order_ids=(),
                broker_observed_at=DECISION_BAR,
            )
        ],
        StateType.READINESS: [
            ReadinessState(
                **_envelope(
                    state_type=StateType.READINESS, entity_id="nightly_data_readiness",
                    as_of=DECISION_BAR, tag="readiness",
                ),
                job="nightly_data_readiness",
                status="READY",
                ready=True,
                completed_at=DECISION_BAR,
                checked_at=DECISION_BAR,
                max_age_hours=96.0,
                latest_required_session="2026-08-21",
                reason_codes=(),
            )
        ],
    }


def _intent(snapshot_id) -> TradeIntent:
    return TradeIntent(
        intent_id=uuid4(),
        strategy_id="meta_ranker",
        ticker=TICKER,
        direction=Direction.LONG,
        decision_kind=DecisionKind.ENTRY,
        raw_score=0.97,
        raw_probability=None,
        expected_return=None,
        expected_holding_period="53x4h",
        snapshot_id=snapshot_id,
        selected_bar=DECISION_BAR,
        entry_window="current-or-next-open",
        preferred_entry=None,
        invalidation=None,
        target=None,
        stop=None,
        position_size_requested=Decimal("5000"),
        instrument_preferences=(InstrumentFamily.EQUITY,),
        feature_timestamp=DECISION_BAR,
        created_at=DECISION_TIME,
        model_version="meta@1",
        feature_version="matrix@1",
        reason_codes=("E2E",),
    )


def _decide(session, states) -> tuple:
    repository = StateRepository(session)
    flat = [state for group in states.values() for state in group]
    repository.insert_states_idempotently(flat)
    session.flush()

    snapshot = SnapshotBuilder(
        repository, sector_entity_ids=SECTORS
    ).build(
        strategy_id="meta_ranker",
        entity_id=TICKER,
        decision_time=DECISION_TIME,
        decision_bar=DECISION_BAR,
        profile=PROFILE,
    )
    config = replace(MVP_POLICY_CONFIG, account_alias="paper")
    return snapshot, evaluate_policy(_intent(snapshot.snapshot_id), snapshot, config)


def test_all_required_states_published_approves_an_entry(pg_session) -> None:
    snapshot, decision = _decide(pg_session, _required_states())

    assert snapshot.valid is True
    degraded = {
        result.state_type
        for result in snapshot.requirement_results
        if result.required and result.status != "FRESH"
    }
    assert degraded == set()
    assert decision.action is not PolicyAction.REJECT, list(decision.hard_vetoes)
    assert not decision.hard_vetoes


def test_missing_optional_states_do_not_veto(pg_session) -> None:
    """THEME, DEALER and the catalyst states are absent here and must not gate.

    This is the whole point of `required=False`. Before the fix, this exact
    snapshot — every required state fresh, every optional one missing — was
    rejected with SNAPSHOT_REQUIRED_STATE_MISSING.
    """

    snapshot, decision = _decide(pg_session, _required_states())

    assert set(snapshot.missing_inputs) >= {"THEME", "DEALER"}
    assert ReasonCode.SNAPSHOT_REQUIRED_STATE_MISSING.value not in decision.hard_vetoes
    assert decision.action is not PolicyAction.REJECT


@pytest.mark.parametrize("dropped", [StateType.TICKER, StateType.MARKET, StateType.SECTOR])
def test_each_missing_required_state_vetoes(pg_session, dropped: StateType) -> None:
    """Every required state earns its place: drop one, lose the approval."""

    states = _required_states()
    states.pop(dropped)
    snapshot, decision = _decide(pg_session, states)

    assert snapshot.valid is False
    assert decision.action is PolicyAction.REJECT
    assert ReasonCode.SNAPSHOT_REQUIRED_STATE_MISSING.value in decision.hard_vetoes


def test_liquidity_metric_is_dollars_not_a_percentile(pg_session) -> None:
    """dollar_volume_20d is compared to $5,000,000, so it must be dollars.

    The Meta matrix's only liquidity field is dollar_vol_pctile_252, a rank in
    [0, 1]. Publishing that under this metric's name passes every type check and
    vetoes every name in the universe.
    """

    percentile_states = _required_states(liquidity=0.6031746031746031)
    _snapshot, decision = _decide(pg_session, percentile_states)

    assert decision.action is PolicyAction.REJECT
    assert ReasonCode.LIQUIDITY_BELOW_MINIMUM.value in decision.hard_vetoes


def test_absent_liquidity_metric_is_not_treated_as_liquid(pg_session) -> None:
    states = _required_states(liquidity=None)
    _snapshot, decision = _decide(pg_session, states)

    assert ReasonCode.LIQUIDITY_METRIC_UNKNOWN.value in decision.hard_vetoes


def test_publication_batches_past_the_postgres_parameter_limit(pg_session) -> None:
    """A full-table publication must not fail on the 65535 bind-parameter cap.

    _state_values binds 16 parameters per row, so one INSERT tops out just above
    4000 rows. The first real market-regime publication attempted 18,312 and
    raised "number of parameters must be between 0 and 65535" before writing
    anything at all.
    """

    repository = StateRepository(pg_session)
    many = [
        SectorState(
            **_envelope(
                state_type=StateType.SECTOR,
                entity_id=f"X{index:05d}",
                as_of=DECISION_BAR,
                tag="bulk",
            ),
            sector_id=f"X{index:05d}",
            sector_regime="NEUTRAL",
            rotation_rank=0.5,
        )
        for index in range(5000)
    ]
    resolved = repository.insert_states_idempotently(many)
    assert len(resolved) == len(many)
