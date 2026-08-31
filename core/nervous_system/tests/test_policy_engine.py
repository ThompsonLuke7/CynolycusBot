"""Deterministic permission and modulation policy tests (Task 15).

The evaluator is a pure function of ``(intent, snapshot, config)``.  Every test
here constructs its inputs explicitly so that a failure names one rule.
"""

from __future__ import annotations

import ast
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, UUID, uuid4, uuid5

import pytest

from core.nervous_system.config.policy import (
    MVP_POLICY_CONFIG,
    PolicyConfig,
    StructureRisk,
)
from core.nervous_system.contracts.base import content_hash
from core.nervous_system.contracts.context import ContextSnapshot, FreshnessResult
from core.nervous_system.contracts.enums import (
    AssetClass,
    DataQualitySeverity,
    DealerRegime,
    DecisionKind,
    Direction,
    InstrumentFamily,
    MarketRegime,
    ModifierOperation,
    PolicyAction,
    PolicyMode,
    RuntimeEnvironment,
    SizeUnit,
    StateType,
    ThemeRegime,
    TickerSetup,
)
from core.nervous_system.contracts.intent import TradeIntent
from core.nervous_system.contracts.quality import DataQualityIssue, DataQualitySummary
from core.nervous_system.contracts.states import (
    DealerState,
    MarketState,
    PortfolioPosition,
    PortfolioState,
    ReadinessState,
    StateContract,
    ThemeState,
    TickerState,
)
from core.nervous_system.policy.engine import (
    POLICY_DECISION_NAMESPACE,
    ExecutionBasis,
    evaluate_policy,
    execution_basis,
    is_executable,
)
from core.nervous_system.policy.reason_codes import ReasonCode, describe


UTC = timezone.utc
TICKER = "AMD"
STRATEGY = "meta_ranker"
PROFILE = "meta_4h_1420@1"
DECISION_TIME = datetime(2026, 7, 30, 18, 20, tzinfo=UTC)
DECISION_BAR = datetime(2026, 7, 30, 18, 0, tzinfo=UTC)
IDEMPOTENCY_KEY = "a1" * 32


# --------------------------------------------------------------------------
# Builders
# --------------------------------------------------------------------------


def _envelope(
    *,
    state_id: UUID,
    state_type: StateType,
    entity_id: str,
    as_of: datetime,
    available_at: datetime,
) -> dict[str, Any]:
    return {
        "state_id": state_id,
        "state_type": state_type,
        "entity_id": entity_id,
        "as_of": as_of,
        "available_at": available_at,
        "generated_at": available_at,
        "valid_until": available_at + timedelta(days=2),
        "source_window_start": as_of - timedelta(minutes=5),
        "source_window_end": as_of,
        "schema_version": 1,
        "producer": "policy-test@1",
        "model_version": "policy-test-model@1",
        "feature_version": "policy-test-features@1",
        "config_version": "policy-test-config@1",
        "lineage_ids": (f"policy-test:{state_id}",),
        "data_quality": DataQualitySummary(),
    }


def market_state(
    *,
    regime: MarketRegime = MarketRegime.NEUTRAL,
    state_id: UUID | None = None,
) -> MarketState:
    payload = _envelope(
        state_id=state_id or uuid5(NAMESPACE_URL, "policy-test/market"),
        state_type=StateType.MARKET,
        entity_id="US",
        as_of=datetime(2026, 7, 29, 20, 0, tzinfo=UTC),
        available_at=datetime(2026, 7, 29, 20, 30, tzinfo=UTC),
    )
    payload["regime"] = regime
    return MarketState(**payload)


def theme_state(
    *,
    theme_regime: ThemeRegime = ThemeRegime.NEUTRAL,
    theme_id: str = "ai_infrastructure",
    state_id: UUID | None = None,
) -> ThemeState:
    payload = _envelope(
        state_id=state_id or uuid5(NAMESPACE_URL, f"policy-test/theme/{theme_id}"),
        state_type=StateType.THEME,
        entity_id=theme_id,
        as_of=datetime(2026, 7, 29, 21, 0, tzinfo=UTC),
        available_at=datetime(2026, 7, 30, 2, 0, tzinfo=UTC),
    )
    payload.update({"theme_id": theme_id, "theme_regime": theme_regime})
    return ThemeState(**payload)


def ticker_state(
    *,
    dollar_volume: float | None = 250_000_000.0,
    metric_name: str = "dollar_volume_20d",
    state_id: UUID | None = None,
) -> TickerState:
    payload = _envelope(
        state_id=state_id or uuid5(NAMESPACE_URL, "policy-test/ticker"),
        state_type=StateType.TICKER,
        entity_id=TICKER,
        as_of=DECISION_BAR,
        available_at=DECISION_BAR + timedelta(minutes=5),
    )
    payload.update(
        {
            "ticker": TICKER,
            "selected_bar": DECISION_BAR,
            "reference_price": 100.0,
            "ticker_setup": TickerSetup.BREAKOUT,
            "metrics": {} if dollar_volume is None else {metric_name: dollar_volume},
        }
    )
    return TickerState(**payload)


def dealer_state(
    *,
    dealer_regime: DealerRegime = DealerRegime.NEUTRAL_GAMMA,
    state_id: UUID | None = None,
) -> DealerState:
    payload = _envelope(
        state_id=state_id or uuid5(NAMESPACE_URL, "policy-test/dealer"),
        state_type=StateType.DEALER,
        entity_id=TICKER,
        as_of=datetime(2026, 7, 30, 18, 5, tzinfo=UTC),
        available_at=datetime(2026, 7, 30, 18, 10, tzinfo=UTC),
    )
    payload.update(
        {"ticker": TICKER, "dealer_regime": dealer_regime, "spot": 100.0}
    )
    return DealerState(**payload)


def portfolio_state(
    *,
    account_alias: str = "paper",
    buying_power: float = 250_000.0,
    day_pl: float | None = -100.0,
    open_order_ids: tuple[str, ...] = (),
    positions: tuple[PortfolioPosition, ...] = (),
    open_orders_observed: bool = True,
    state_id: UUID | None = None,
) -> PortfolioState:
    payload = _envelope(
        state_id=state_id or uuid5(NAMESPACE_URL, "policy-test/portfolio"),
        state_type=StateType.PORTFOLIO,
        entity_id=account_alias,
        as_of=datetime(2026, 7, 30, 18, 15, tzinfo=UTC),
        available_at=datetime(2026, 7, 30, 18, 16, tzinfo=UTC),
    )
    if not open_orders_observed:
        # Mirrors core/broker_equity_snapshot.py when the open-orders read fails.
        payload["data_quality"] = DataQualitySummary(
            issues=(
                DataQualityIssue(
                    code="OPEN_ORDERS_NOT_OBSERVED",
                    severity=DataQualitySeverity.WARNING,
                    component="broker.portfolio",
                    message="open orders were not observed (FAILED)",
                    fallback_used="open_order_ids left empty",
                ),
            )
        )
    payload.update(
        {
            "account_alias": account_alias,
            "equity": 250_000.0,
            "cash": 250_000.0,
            "buying_power": buying_power,
            "day_pl": day_pl,
            "open_order_ids": open_order_ids,
            "positions": positions,
            "broker_observed_at": datetime(2026, 7, 30, 18, 15, tzinfo=UTC),
        }
    )
    return PortfolioState(**payload)


def position(
    *,
    symbol: str = "NVDA",
    market_value: float | None = 1_000.0,
) -> PortfolioPosition:
    return PortfolioPosition(
        broker_position_id=f"pos-{symbol}",
        symbol=symbol,
        underlying=symbol,
        asset_class=AssetClass.EQUITY,
        quantity=10.0,
        market_value=market_value,
    )


def readiness_state(
    *,
    ready: bool = True,
    job: str = "nightly_data_readiness",
    state_id: UUID | None = None,
) -> ReadinessState:
    payload = _envelope(
        state_id=state_id or uuid5(NAMESPACE_URL, f"policy-test/readiness/{job}"),
        state_type=StateType.READINESS,
        entity_id=job,
        as_of=datetime(2026, 7, 30, 17, 0, tzinfo=UTC),
        available_at=datetime(2026, 7, 30, 17, 1, tzinfo=UTC),
    )
    payload.update(
        {
            "job": job,
            "status": "READY" if ready else "STALE",
            "ready": ready,
            "completed_at": datetime(2026, 7, 30, 17, 0, tzinfo=UTC),
            "checked_at": datetime(2026, 7, 30, 17, 1, tzinfo=UTC),
            "max_age_hours": 96.0,
            "latest_required_session": "2026-07-29",
            "reason_codes": () if ready else ("STALE_SESSION",),
        }
    )
    return ReadinessState(**payload)


def build_snapshot(
    *,
    states: tuple[StateContract, ...] | None = None,
    snapshot_id: UUID | None = None,
    decision_time: datetime = DECISION_TIME,
    freshness_profile: str = PROFILE,
    stale_inputs: tuple[str, ...] = (),
    missing_inputs: tuple[str, ...] = (),
    requirement_results: tuple[FreshnessResult, ...] = (),
    valid: bool = True,
) -> ContextSnapshot:
    if states is None:
        states = (
            market_state(),
            ticker_state(),
            portfolio_state(),
            readiness_state(),
        )
    return ContextSnapshot.from_states(
        snapshot_id=snapshot_id or uuid5(NAMESPACE_URL, "policy-test/snapshot"),
        decision_time=decision_time,
        strategy_id=STRATEGY,
        ticker=TICKER,
        states=states,
        freshness_profile=freshness_profile,
        freshness_profile_hash="b" * 64,
        decision_bar=DECISION_BAR,
        decision_session="2026-07-30",
        stale_inputs=stale_inputs,
        missing_inputs=missing_inputs,
        requirement_results=requirement_results,
        valid=valid,
    )


def degraded_requirement(
    state_type: StateType,
    *,
    required: bool,
    status: str,
) -> FreshnessResult:
    """One evaluator verdict for a single freshness rule.

    Tests must go through this rather than setting `stale_inputs` /
    `missing_inputs` directly: those two fields are a report the evaluator fills
    in for EVERY rule, so a test that sets them by hand cannot express the
    required-vs-optional distinction that decides whether a degraded input is a
    veto or a warning.
    """

    return FreshnessResult(
        state_type=state_type,
        entity_id=TICKER,
        required=required,
        status=status,
        selected_state_id=None,
        age_seconds=None,
        max_age_seconds=3600.0,
        reason_code=status,
    )


def build_intent(
    *,
    snapshot: ContextSnapshot,
    decision_kind: DecisionKind = DecisionKind.ENTRY,
    direction: Direction = Direction.LONG,
    position_size_requested: Decimal = Decimal("1000.00"),
    instrument_preferences: tuple[InstrumentFamily, ...] = (InstrumentFamily.EQUITY,),
    idempotency_key: str = IDEMPOTENCY_KEY,
    intent_id: UUID | None = None,
    created_at: datetime | None = None,
    position_size_unit: SizeUnit | None = None,
) -> TradeIntent:
    created = created_at or snapshot.decision_time
    # Entries are a money budget; an exit is a typed quantity.
    unit = position_size_unit or (
        SizeUnit.SHARES
        if decision_kind is DecisionKind.EXIT
        else SizeUnit.NOTIONAL_USD
    )
    return TradeIntent(
        intent_id=intent_id or uuid5(NAMESPACE_URL, "policy-test/intent"),
        strategy_id=STRATEGY,
        ticker=TICKER,
        direction=direction,
        decision_kind=decision_kind,
        raw_score=0.91,
        raw_probability=None,
        expected_return=None,
        expected_holding_period="53x4h",
        snapshot_id=snapshot.snapshot_id,
        selected_bar=DECISION_BAR,
        entry_window="current-or-next-open",
        preferred_entry=Decimal("100.00"),
        invalidation=Decimal("94.00"),
        target=Decimal("112.00"),
        stop=Decimal("94.00"),
        position_size_requested=position_size_requested,
        position_size_unit=unit,
        instrument_preferences=instrument_preferences,
        feature_timestamp=DECISION_BAR,
        created_at=created,
        model_version="meta@1",
        feature_version="matrix@1",
        reason_codes=("META_TOP_DECILE",),
        score_components={"s_combo": 0.91, "s_quality": 0.72},
        config_version="meta-config@1",
        idempotency_key=idempotency_key,
    )


def build_config(**overrides: Any) -> PolicyConfig:
    base = replace(
        MVP_POLICY_CONFIG,
        mode=PolicyMode.ENFORCE,
        environment=RuntimeEnvironment.QA_PAPER,
        required_snapshot_profile=PROFILE,
        # The shipped paper config disables the daily-loss and gross-exposure
        # gates, but these tests exercise the rules themselves, so they keep the
        # figures those rules used to carry. Tests that need a gate off say so
        # explicitly.
        max_daily_loss=Decimal("2000.00"),
        max_gross_notional=Decimal("150000.00"),
    )
    return replace(base, **overrides) if overrides else base


# --------------------------------------------------------------------------
# Step 1: hard vetoes
# --------------------------------------------------------------------------


def _assert_auditable_reasons(decision) -> None:
    """Every emitted code must be a stable registry code with human detail."""

    assert decision.reason_codes, "policy decisions must always carry reason codes"
    for code in decision.reason_codes + decision.hard_vetoes:
        assert code == ReasonCode(code).value
        assert describe(code).strip(), f"{code} is missing human detail"


@pytest.mark.parametrize("mode", list(PolicyMode))
def test_production_live_always_rejects(mode: PolicyMode) -> None:
    snapshot = build_snapshot()
    intent = build_intent(snapshot=snapshot)
    config = build_config(
        mode=mode,
        environment=RuntimeEnvironment.PRODUCTION_LIVE,
        account_alias="live",
        paper_account_aliases=frozenset({"paper"}),
    )

    decision = evaluate_policy(intent, snapshot, config)

    assert decision.action is PolicyAction.REJECT
    assert decision.final_risk_budget == Decimal("0")
    assert ReasonCode.ENV_PRODUCTION_LIVE_DISABLED_MVP.value in decision.hard_vetoes
    assert decision.allowed_instruments == frozenset()
    _assert_auditable_reasons(decision)


@pytest.mark.parametrize(
    "environment",
    [RuntimeEnvironment.DEVELOPMENT, RuntimeEnvironment.QA_PAPER],
)
def test_off_mode_is_audit_only_baseline(environment: RuntimeEnvironment) -> None:
    snapshot = build_snapshot()
    intent = build_intent(snapshot=snapshot)
    config = build_config(mode=PolicyMode.OFF, environment=environment)

    decision = evaluate_policy(intent, snapshot, config)

    assert decision.action is PolicyAction.DEFER
    assert decision.hard_vetoes == ()
    assert decision.modifiers == ()
    assert decision.base_risk_budget == intent.position_size_requested
    assert decision.final_risk_budget == intent.position_size_requested
    assert ReasonCode.POLICY_OFF_AUDIT_ONLY.value in decision.reason_codes
    assert execution_basis(decision) is ExecutionBasis.AUDIT_ONLY
    _assert_auditable_reasons(decision)


def test_invalid_snapshot_vetoes_entry() -> None:
    snapshot = build_snapshot(valid=False)
    intent = build_intent(snapshot=snapshot)

    decision = evaluate_policy(intent, snapshot, build_config())

    assert decision.action is PolicyAction.REJECT
    assert ReasonCode.SNAPSHOT_INVALID.value in decision.hard_vetoes
    _assert_auditable_reasons(decision)


def test_stale_and_missing_required_state_veto_entry() -> None:
    # A required rule that the evaluator could not satisfy. `valid` is False
    # because that is what `evaluate_requirements` does for a required rule, so
    # the fixture matches a snapshot the builder could actually produce.
    snapshot = build_snapshot(
        stale_inputs=("MARKET",),
        missing_inputs=("SECTOR",),
        requirement_results=(
            degraded_requirement(StateType.MARKET, required=True, status="STALE"),
            degraded_requirement(StateType.SECTOR, required=True, status="MISSING"),
        ),
        valid=False,
    )
    intent = build_intent(snapshot=snapshot)

    decision = evaluate_policy(intent, snapshot, build_config())

    assert decision.action is PolicyAction.REJECT
    assert ReasonCode.SNAPSHOT_REQUIRED_STATE_STALE.value in decision.hard_vetoes
    assert ReasonCode.SNAPSHOT_REQUIRED_STATE_MISSING.value in decision.hard_vetoes


def test_stale_and_missing_optional_state_do_not_veto_entry() -> None:
    """An optional input degrading is a warning, not a veto.

    THEME and CATALYST_EVENT are declared `required=False,
    MissingStateAction.WARN` in every shipped profile. The evaluator still lists
    them in `stale_inputs` / `missing_inputs` because those fields report on all
    rules, and `snapshot_vetoes` used to gate on the bare presence of either
    list — so an absent THEME vetoed exactly as hard as an absent TICKER. On
    2026-08-21 the live Meta snapshot was missing THEME, THEME_MEMBERSHIP,
    CATALYST_PRESSURE and DEALER and carried a stale CATALYST_EVENT, all
    optional, which meant publishing the three genuinely-missing required states
    would not by itself have let a single order through.
    """

    snapshot = build_snapshot(
        stale_inputs=("CATALYST_EVENT",),
        missing_inputs=("THEME", "DEALER"),
        requirement_results=(
            degraded_requirement(StateType.CATALYST_EVENT, required=False, status="STALE"),
            degraded_requirement(StateType.THEME, required=False, status="MISSING"),
            degraded_requirement(StateType.DEALER, required=False, status="MISSING"),
        ),
        valid=True,
    )
    intent = build_intent(snapshot=snapshot)

    decision = evaluate_policy(intent, snapshot, build_config())

    assert ReasonCode.SNAPSHOT_REQUIRED_STATE_STALE.value not in decision.hard_vetoes
    assert ReasonCode.SNAPSHOT_REQUIRED_STATE_MISSING.value not in decision.hard_vetoes
    assert ReasonCode.SNAPSHOT_INVALID.value not in decision.hard_vetoes
    assert decision.action is not PolicyAction.REJECT


def test_snapshot_profile_mismatch_vetoes_entry() -> None:
    snapshot = build_snapshot(freshness_profile="meta_4h_1620@1")
    intent = build_intent(snapshot=snapshot)

    decision = evaluate_policy(intent, snapshot, build_config())

    assert ReasonCode.SNAPSHOT_PROFILE_MISMATCH.value in decision.hard_vetoes


def test_snapshot_lineage_mismatch_vetoes_entry() -> None:
    snapshot = build_snapshot()
    other = build_snapshot(snapshot_id=uuid5(NAMESPACE_URL, "policy-test/other"))
    intent = build_intent(snapshot=other)

    decision = evaluate_policy(intent, snapshot, build_config())

    assert ReasonCode.SNAPSHOT_LINEAGE_MISMATCH.value in decision.hard_vetoes


def test_snapshot_after_intent_decision_time_vetoes_entry() -> None:
    """A snapshot observed after the intent would be look-ahead evidence."""

    snapshot = build_snapshot()
    intent = build_intent(
        snapshot=snapshot,
        created_at=snapshot.decision_time - timedelta(minutes=1),
    )

    decision = evaluate_policy(intent, snapshot, build_config())

    assert ReasonCode.SNAPSHOT_DECISION_TIME_AFTER_INTENT.value in decision.hard_vetoes


def test_qa_paper_without_paper_account_vetoes() -> None:
    snapshot = build_snapshot()
    intent = build_intent(snapshot=snapshot)
    config = build_config(
        environment=RuntimeEnvironment.QA_PAPER,
        account_alias="brokerage-live-1",
    )

    decision = evaluate_policy(intent, snapshot, config)

    assert decision.action is PolicyAction.REJECT
    assert ReasonCode.BROKER_PAPER_ACCOUNT_REQUIRED.value in decision.hard_vetoes


def test_account_identity_mismatch_vetoes() -> None:
    snapshot = build_snapshot(
        states=(
            market_state(),
            ticker_state(),
            portfolio_state(account_alias="other_paper"),
            readiness_state(),
        )
    )
    intent = build_intent(snapshot=snapshot)
    config = build_config(paper_account_aliases=frozenset({"paper", "other_paper"}))

    decision = evaluate_policy(intent, snapshot, config)

    assert ReasonCode.BROKER_ACCOUNT_IDENTITY_MISMATCH.value in decision.hard_vetoes


def test_missing_portfolio_state_vetoes() -> None:
    snapshot = build_snapshot(
        states=(market_state(), ticker_state(), readiness_state())
    )
    intent = build_intent(snapshot=snapshot)

    decision = evaluate_policy(intent, snapshot, build_config())

    assert ReasonCode.BROKER_PORTFOLIO_STATE_MISSING.value in decision.hard_vetoes


def test_unknown_maximum_loss_structure_vetoes() -> None:
    snapshot = build_snapshot()
    intent = build_intent(
        snapshot=snapshot,
        instrument_preferences=(InstrumentFamily.CALENDAR,),
    )
    config = build_config(
        allowed_instruments=frozenset({InstrumentFamily.CALENDAR}),
    )

    decision = evaluate_policy(intent, snapshot, config)

    assert decision.action is PolicyAction.REJECT
    assert ReasonCode.STRUCTURE_UNKNOWN_MAXIMUM_LOSS.value in decision.hard_vetoes


def test_permission_alone_does_not_authorise_naked_or_ratio_structures() -> None:
    """Widening ``allowed_instruments`` cannot restore an undefined-loss family."""

    snapshot = build_snapshot()
    config = build_config(
        allowed_instruments=frozenset(
            {InstrumentFamily.STRANGLE, InstrumentFamily.ROLL}
        ),
        structure_risk={
            InstrumentFamily.STRANGLE: StructureRisk.NAKED_SHORT,
            InstrumentFamily.ROLL: StructureRisk.UNCOVERED_RATIO,
        },
    )

    naked = evaluate_policy(
        build_intent(
            snapshot=snapshot,
            instrument_preferences=(InstrumentFamily.STRANGLE,),
        ),
        snapshot,
        config,
    )
    ratio = evaluate_policy(
        build_intent(
            snapshot=snapshot, instrument_preferences=(InstrumentFamily.ROLL,)
        ),
        snapshot,
        config,
    )

    assert ReasonCode.STRUCTURE_NAKED_SHORT_OPTION.value in naked.hard_vetoes
    assert ReasonCode.STRUCTURE_UNCOVERED_RATIO.value in ratio.hard_vetoes
    assert naked.action is ratio.action is PolicyAction.REJECT


def test_instrument_family_not_permitted_vetoes() -> None:
    snapshot = build_snapshot()
    intent = build_intent(
        snapshot=snapshot,
        instrument_preferences=(InstrumentFamily.IRON_CONDOR,),
    )

    decision = evaluate_policy(intent, snapshot, build_config())

    assert ReasonCode.INSTRUMENT_FAMILY_NOT_PERMITTED.value in decision.hard_vetoes


def test_duplicate_idempotency_key_vetoes_second_entry() -> None:
    first_snapshot = build_snapshot()
    first = evaluate_policy(
        build_intent(snapshot=first_snapshot), first_snapshot, build_config()
    )
    assert first.action is PolicyAction.APPROVE

    replayed = build_snapshot(
        states=(
            market_state(),
            ticker_state(),
            portfolio_state(open_order_ids=(IDEMPOTENCY_KEY,)),
            readiness_state(),
        )
    )
    second = evaluate_policy(
        build_intent(snapshot=replayed), replayed, build_config()
    )

    assert second.action is PolicyAction.REJECT
    assert ReasonCode.BROKER_DUPLICATE_IDEMPOTENCY_KEY.value in second.hard_vetoes


def test_unobserved_open_orders_veto_entry_but_not_exit() -> None:
    """An empty open-order tuple must not be read as "no duplicate exists"."""

    snapshot = build_snapshot(
        states=(
            market_state(),
            ticker_state(),
            portfolio_state(open_orders_observed=False),
            readiness_state(),
        )
    )
    entry = evaluate_policy(
        build_intent(snapshot=snapshot), snapshot, build_config()
    )
    exit_decision = evaluate_policy(
        build_intent(snapshot=snapshot, decision_kind=DecisionKind.EXIT),
        snapshot,
        build_config(),
    )

    assert entry.action is PolicyAction.REJECT
    assert ReasonCode.BROKER_OPEN_ORDERS_NOT_OBSERVED.value in entry.hard_vetoes
    assert exit_decision.action is PolicyAction.EXIT


def test_position_without_market_value_vetoes_entry() -> None:
    """Unknown exposure must fail closed, not be counted as zero."""

    snapshot = build_snapshot(
        states=(
            market_state(),
            ticker_state(),
            portfolio_state(
                positions=(position(market_value=None), position(symbol="AVGO"))
            ),
            readiness_state(),
        )
    )
    intent = build_intent(snapshot=snapshot)

    decision = evaluate_policy(intent, snapshot, build_config())

    assert decision.action is PolicyAction.REJECT
    assert ReasonCode.PORTFOLIO_EXPOSURE_UNKNOWN.value in decision.hard_vetoes


def test_known_positions_count_toward_the_gross_notional_limit() -> None:
    snapshot = build_snapshot(
        states=(
            market_state(),
            ticker_state(),
            portfolio_state(
                positions=(
                    position(symbol="NVDA", market_value=-100_000.0),
                    position(symbol="AVGO", market_value=49_500.0),
                )
            ),
            readiness_state(),
        )
    )
    intent = build_intent(snapshot=snapshot, position_size_requested=Decimal("1000.00"))

    decision = evaluate_policy(intent, snapshot, build_config())

    # |-100000| + 49500 + 1000 = 150500 > 150000
    assert decision.action is PolicyAction.REJECT
    assert ReasonCode.PORTFOLIO_MAX_GROSS_NOTIONAL_BREACH.value in decision.hard_vetoes


def test_is_executable_gates_off_mode_and_vetoed_decisions() -> None:
    snapshot = build_snapshot()
    intent = build_intent(snapshot=snapshot)

    approved = evaluate_policy(intent, snapshot, build_config())
    off = evaluate_policy(intent, snapshot, build_config(mode=PolicyMode.OFF))
    rejected = evaluate_policy(
        intent,
        snapshot,
        build_config(environment=RuntimeEnvironment.PRODUCTION_LIVE),
    )
    exit_decision = evaluate_policy(
        build_intent(snapshot=snapshot, decision_kind=DecisionKind.EXIT),
        snapshot,
        build_config(),
    )

    assert is_executable(approved) is True
    assert is_executable(exit_decision) is True
    # OFF carries a DEFER action and a full baseline budget, so action alone
    # is not a safe gate.
    assert off.final_risk_budget == intent.position_size_requested
    assert is_executable(off) is False
    assert is_executable(rejected) is False


def test_off_mode_is_never_executable_even_carrying_an_approve_action() -> None:
    """The mode gate must hold independently of the recorded action."""

    snapshot = build_snapshot()
    approved = evaluate_policy(
        build_intent(snapshot=snapshot), snapshot, build_config()
    )
    assert is_executable(approved) is True

    smuggled = approved.model_copy(update={"mode": PolicyMode.OFF})

    assert smuggled.action is PolicyAction.APPROVE
    assert smuggled.hard_vetoes == ()
    assert execution_basis(smuggled) is ExecutionBasis.AUDIT_ONLY
    assert is_executable(smuggled) is False


def test_readiness_gate_failure_vetoes_entry() -> None:
    snapshot = build_snapshot(
        states=(
            market_state(),
            ticker_state(),
            portfolio_state(),
            readiness_state(ready=False),
        )
    )
    intent = build_intent(snapshot=snapshot)

    decision = evaluate_policy(intent, snapshot, build_config())

    assert decision.action is PolicyAction.REJECT
    assert ReasonCode.READINESS_NOT_READY.value in decision.hard_vetoes


def test_missing_readiness_state_vetoes_entry() -> None:
    snapshot = build_snapshot(
        states=(market_state(), ticker_state(), portfolio_state())
    )
    intent = build_intent(snapshot=snapshot)

    decision = evaluate_policy(intent, snapshot, build_config())

    assert ReasonCode.READINESS_STATE_MISSING.value in decision.hard_vetoes


def test_daily_loss_breach_and_liquidity_veto_entry() -> None:
    snapshot = build_snapshot(
        states=(
            market_state(),
            ticker_state(dollar_volume=10_000.0),
            portfolio_state(day_pl=-9_000.0),
            readiness_state(),
        )
    )
    intent = build_intent(snapshot=snapshot)

    decision = evaluate_policy(intent, snapshot, build_config())

    assert decision.action is PolicyAction.REJECT
    assert ReasonCode.PORTFOLIO_MAX_DAILY_LOSS_BREACH.value in decision.hard_vetoes
    assert ReasonCode.LIQUIDITY_BELOW_MINIMUM.value in decision.hard_vetoes


def test_unknown_liquidity_fails_closed() -> None:
    snapshot = build_snapshot(
        states=(
            market_state(),
            ticker_state(dollar_volume=None),
            portfolio_state(),
            readiness_state(),
        )
    )
    intent = build_intent(snapshot=snapshot)

    decision = evaluate_policy(intent, snapshot, build_config())

    assert ReasonCode.LIQUIDITY_METRIC_UNKNOWN.value in decision.hard_vetoes


def test_blocking_data_quality_vetoes_entry() -> None:
    payload = _envelope(
        state_id=uuid5(NAMESPACE_URL, "policy-test/ticker"),
        state_type=StateType.TICKER,
        entity_id=TICKER,
        as_of=DECISION_BAR,
        available_at=DECISION_BAR + timedelta(minutes=5),
    )
    payload.update(
        {
            "ticker": TICKER,
            "selected_bar": DECISION_BAR,
            "reference_price": 100.0,
            "ticker_setup": TickerSetup.BREAKOUT,
            "metrics": {"dollar_volume_20d": 250_000_000.0},
            "data_quality": DataQualitySummary(
                issues=(
                    DataQualityIssue(
                        code="BAR_GAP",
                        severity=DataQualitySeverity.CRITICAL,
                        component="feature_matrix_4h",
                        message="missing 4h bar",
                    ),
                )
            ),
        }
    )
    snapshot = build_snapshot(
        states=(
            market_state(),
            TickerState(**payload),
            portfolio_state(),
            readiness_state(),
        )
    )
    intent = build_intent(snapshot=snapshot)

    decision = evaluate_policy(intent, snapshot, build_config())

    assert ReasonCode.DATA_QUALITY_BLOCKING.value in decision.hard_vetoes


def test_insufficient_buying_power_vetoes_entry() -> None:
    snapshot = build_snapshot(
        states=(
            market_state(),
            ticker_state(),
            portfolio_state(buying_power=10.0),
            readiness_state(),
        )
    )
    intent = build_intent(snapshot=snapshot)

    decision = evaluate_policy(intent, snapshot, build_config())

    assert ReasonCode.BROKER_INSUFFICIENT_BUYING_POWER.value in decision.hard_vetoes


def test_hard_vetoes_accumulate_across_rules_in_order() -> None:
    # `valid=True` isolates the ordering assertion to the stale-required-state
    # veto: an INVALID snapshot would add SNAPSHOT_INVALID ahead of it and the
    # test would no longer prove which rule contributed which code.
    snapshot = build_snapshot(
        valid=True,
        stale_inputs=("MARKET",),
        requirement_results=(
            degraded_requirement(StateType.MARKET, required=True, status="STALE"),
        ),
    )
    intent = build_intent(
        snapshot=snapshot, instrument_preferences=(InstrumentFamily.CALENDAR,)
    )
    config = build_config(account_alias="not_paper")

    decision = evaluate_policy(intent, snapshot, config)

    assert decision.hard_vetoes.index(
        ReasonCode.SNAPSHOT_REQUIRED_STATE_STALE.value
    ) < decision.hard_vetoes.index(
        ReasonCode.INSTRUMENT_FAMILY_NOT_PERMITTED.value
    ) < decision.hard_vetoes.index(
        ReasonCode.BROKER_PAPER_ACCOUNT_REQUIRED.value
    )


def test_risk_reducing_exit_stays_operable_under_degraded_context() -> None:
    """Freshness, readiness, liquidity, and cap rules do not trap an exit."""

    snapshot = build_snapshot(
        states=(
            ticker_state(dollar_volume=1.0),
            portfolio_state(day_pl=-9_000.0),
        ),
        stale_inputs=("MARKET",),
        missing_inputs=("READINESS",),
    )
    intent = build_intent(
        snapshot=snapshot,
        decision_kind=DecisionKind.EXIT,
        position_size_requested=Decimal("25.00"),
    )

    decision = evaluate_policy(intent, snapshot, build_config())

    assert decision.action is PolicyAction.EXIT
    assert decision.hard_vetoes == ()
    assert decision.modifiers == ()
    assert decision.final_risk_budget == Decimal("25.00")
    assert ReasonCode.EXIT_RISK_REDUCING_PERMITTED.value in decision.reason_codes
    assert decision.allowed_instruments == frozenset(intent.instrument_preferences)
    _assert_auditable_reasons(decision)


def test_an_exit_share_count_is_never_treated_as_a_money_budget() -> None:
    """``position_size_requested`` is a money budget for entries and a typed
    quantity for exits. If any money rule ran on an exit, a 41-share close
    would be reinterpreted as $41 and capped down to a partial close, leaving
    the position open. The pass-through is the guarantee that cannot break.
    """

    snapshot = build_snapshot()
    intent = build_intent(
        snapshot=snapshot,
        decision_kind=DecisionKind.EXIT,
        position_size_requested=Decimal("41"),
        position_size_unit=SizeUnit.SHARES,
    )

    decision = evaluate_policy(
        intent,
        snapshot,
        # A cap far below the share count: a money rule would bite here.
        build_config(
            max_position_notional=Decimal("10.00"),
            minimum_order_notional=Decimal("1.00"),
        ),
    )

    assert decision.action is PolicyAction.EXIT
    assert decision.final_risk_budget == Decimal("41")
    assert decision.modifiers == ()


def test_a_trim_share_count_is_never_treated_as_a_money_budget() -> None:
    """The ADJUSTMENT twin of the EXIT case above, and the one that actually bit.

    A take-profit trim is denominated in SHARES, but ADJUSTMENT used to fall
    through to the money-sizing chain, where a 36-share sell was compared
    against a dollar ``minimum_order_notional`` and rejected
    ``SIZE_BELOW_MINIMUM_EXECUTABLE``. Retrying cannot help: the share count
    never grows into the dollar floor. meta_ranker's AMLX x36 was stranded from
    the 2026-08-18 bar through five pre-open flushes on exactly this.
    """

    snapshot = build_snapshot()
    intent = build_intent(
        snapshot=snapshot,
        decision_kind=DecisionKind.ADJUSTMENT,
        position_size_requested=Decimal("36"),
        position_size_unit=SizeUnit.SHARES,
    )

    decision = evaluate_policy(
        intent,
        snapshot,
        # A floor far above the share count: the old code rejected here.
        build_config(minimum_order_notional=Decimal("500.00")),
    )

    assert decision.action is PolicyAction.EXIT
    assert decision.hard_vetoes == ()
    assert decision.final_risk_budget == Decimal("36")
    assert decision.modifiers == ()
    _assert_auditable_reasons(decision)


def test_trim_stays_operable_under_degraded_context() -> None:
    """Reducing exposure must not depend on a healthy context.

    A trim carries the same narrow permission as a full exit: it lowers risk,
    so the entry-only rules (snapshot, readiness, instrument, portfolio limits,
    liquidity) do not gate it. Only environment and broker still bind.
    """

    snapshot = build_snapshot(
        states=(
            ticker_state(dollar_volume=1.0),
            portfolio_state(day_pl=-9_000.0),
        ),
        stale_inputs=("MARKET",),
        missing_inputs=("READINESS",),
    )
    intent = build_intent(
        snapshot=snapshot,
        decision_kind=DecisionKind.ADJUSTMENT,
        position_size_requested=Decimal("16"),
        position_size_unit=SizeUnit.SHARES,
    )

    decision = evaluate_policy(intent, snapshot, build_config())

    assert decision.action is PolicyAction.EXIT
    assert decision.hard_vetoes == ()


def test_exit_still_blocked_by_environment_and_account_identity() -> None:
    snapshot = build_snapshot()
    intent = build_intent(snapshot=snapshot, decision_kind=DecisionKind.EXIT)

    live = evaluate_policy(
        intent,
        snapshot,
        build_config(
            environment=RuntimeEnvironment.PRODUCTION_LIVE, account_alias="live"
        ),
    )
    wrong_account = evaluate_policy(
        intent, snapshot, build_config(account_alias="not_paper")
    )

    assert live.action is PolicyAction.REJECT
    assert (
        ReasonCode.ENV_PRODUCTION_LIVE_DISABLED_MVP.value in live.hard_vetoes
    )
    assert wrong_account.action is PolicyAction.REJECT
    assert (
        ReasonCode.BROKER_PAPER_ACCOUNT_REQUIRED.value in wrong_account.hard_vetoes
    )


def test_duplicate_exit_is_still_vetoed() -> None:
    snapshot = build_snapshot(
        states=(
            market_state(),
            ticker_state(),
            portfolio_state(open_order_ids=(IDEMPOTENCY_KEY,)),
            readiness_state(),
        )
    )
    intent = build_intent(snapshot=snapshot, decision_kind=DecisionKind.EXIT)

    decision = evaluate_policy(intent, snapshot, build_config())

    assert decision.action is PolicyAction.REJECT
    assert ReasonCode.BROKER_DUPLICATE_IDEMPOTENCY_KEY.value in decision.hard_vetoes


# --------------------------------------------------------------------------
# Step 2: modifier order and caps
# --------------------------------------------------------------------------


def test_modifier_order_and_cap_waterfall() -> None:
    snapshot = build_snapshot(
        states=(
            market_state(regime=MarketRegime.DETERIORATING),
            theme_state(theme_regime=ThemeRegime.DISTRIBUTION),
            ticker_state(),
            portfolio_state(),
            readiness_state(),
        )
    )
    intent = build_intent(snapshot=snapshot)
    config = build_config(max_position_notional=Decimal("300.00"))

    decision = evaluate_policy(intent, snapshot, config)

    assert decision.action is PolicyAction.APPROVE_REDUCED
    assert decision.base_risk_budget == Decimal("1000.00")
    assert decision.final_risk_budget == Decimal("300.00")

    market, theme, cap, quantum = decision.modifiers
    assert [m.rule_id for m in decision.modifiers] == [
        "policy.modifier.market_regime",
        "policy.modifier.theme_regime",
        "policy.cap.max_position_notional",
        "policy.cap.money_quantum",
    ]

    assert market.operation is ModifierOperation.MULTIPLY
    assert market.input_value == MarketRegime.DETERIORATING.value
    assert market.configured_value == Decimal("0.8")
    assert market.budget_before == Decimal("1000.00")
    assert market.budget_after == Decimal("800.00")
    assert market.source_state_id == snapshot.market_state.state_id
    assert market.config_version == config.config_version
    assert market.rule_version.startswith("policy.modifier.market_regime@")

    assert theme.operation is ModifierOperation.MULTIPLY
    assert theme.input_value == ThemeRegime.DISTRIBUTION.value
    assert theme.configured_value == Decimal("0.5")
    assert theme.budget_after == Decimal("400.00")
    assert theme.source_state_id == snapshot.theme_states[0].state_id

    assert cap.operation is ModifierOperation.CAP
    assert cap.configured_value == Decimal("300.00")
    assert cap.budget_before == Decimal("400.00")
    assert cap.budget_after == Decimal("300.00")
    assert cap.source_state_id is None

    assert quantum.operation is ModifierOperation.CAP
    assert quantum.budget_after == Decimal("300.00")
    _assert_auditable_reasons(decision)


def test_absent_context_records_unavailable_reason_without_modifier() -> None:
    snapshot = build_snapshot()
    decision = evaluate_policy(
        build_intent(snapshot=snapshot), snapshot, build_config()
    )

    rule_ids = [modifier.rule_id for modifier in decision.modifiers]
    assert "policy.modifier.theme_regime" not in rule_ids
    assert "policy.modifier.dealer_regime" not in rule_ids
    assert ReasonCode.CONTEXT_THEME_UNAVAILABLE.value in decision.reason_codes
    assert ReasonCode.CONTEXT_DEALER_UNAVAILABLE.value in decision.reason_codes


def test_dealer_and_data_quality_modifiers_apply_in_fixed_order() -> None:
    payload = _envelope(
        state_id=uuid5(NAMESPACE_URL, "policy-test/market"),
        state_type=StateType.MARKET,
        entity_id="US",
        as_of=datetime(2026, 7, 29, 20, 0, tzinfo=UTC),
        available_at=datetime(2026, 7, 29, 20, 30, tzinfo=UTC),
    )
    payload.update(
        {
            "regime": MarketRegime.NEUTRAL,
            "data_quality": DataQualitySummary(
                issues=(
                    DataQualityIssue(
                        code="LATE_PUBLISH",
                        severity=DataQualitySeverity.WARNING,
                        component="market_regime",
                        message="published late",
                    ),
                )
            ),
        }
    )
    snapshot = build_snapshot(
        states=(
            MarketState(**payload),
            theme_state(theme_regime=ThemeRegime.HEALTHY),
            dealer_state(dealer_regime=DealerRegime.DOWNSIDE_ACCELERATION),
            ticker_state(),
            portfolio_state(),
            readiness_state(),
        )
    )
    decision = evaluate_policy(
        build_intent(snapshot=snapshot), snapshot, build_config()
    )

    assert [m.rule_id for m in decision.modifiers] == [
        "policy.modifier.market_regime",
        "policy.modifier.theme_regime",
        "policy.modifier.dealer_regime",
        "policy.modifier.data_quality",
        "policy.cap.max_position_notional",
        "policy.cap.money_quantum",
    ]
    dealer = decision.modifiers[2]
    quality = decision.modifiers[3]
    assert dealer.configured_value == Decimal("0.5")
    assert quality.configured_value == Decimal("0.75")
    assert quality.input_value == DataQualitySeverity.WARNING.value


def test_most_conservative_theme_is_selected_deterministically() -> None:
    snapshot = build_snapshot(
        states=(
            market_state(),
            theme_state(theme_regime=ThemeRegime.HEALTHY, theme_id="zeta_theme"),
            theme_state(
                theme_regime=ThemeRegime.LIQUIDATION, theme_id="alpha_theme"
            ),
            ticker_state(),
            portfolio_state(),
            readiness_state(),
        )
    )
    decision = evaluate_policy(
        build_intent(snapshot=snapshot), snapshot, build_config()
    )

    theme = next(
        m for m in decision.modifiers if m.rule_id == "policy.modifier.theme_regime"
    )
    assert theme.input_value == ThemeRegime.LIQUIDATION.value
    assert theme.configured_value == Decimal("0.25")


def test_money_quantum_cap_removes_sub_cent_residue() -> None:
    snapshot = build_snapshot(
        states=(
            market_state(regime=MarketRegime.DETERIORATING),
            ticker_state(),
            portfolio_state(),
            readiness_state(),
        )
    )
    intent = build_intent(snapshot=snapshot, position_size_requested=Decimal("1000.01"))

    decision = evaluate_policy(intent, snapshot, build_config())

    quantum = decision.modifiers[-1]
    assert quantum.rule_id == "policy.cap.money_quantum"
    assert quantum.budget_before == Decimal("800.008")
    assert decision.final_risk_budget == Decimal("800.00")
    # Scale, not only value: downstream sizing must not inherit sub-cent digits.
    assert decision.final_risk_budget.as_tuple().exponent == -2


def test_final_budget_is_always_expressed_at_the_money_quantum() -> None:
    config = build_config()
    for size in (Decimal("1000.00"), Decimal("1000.01"), Decimal("777.77")):
        snapshot = build_snapshot(
            states=(
                market_state(regime=MarketRegime.DETERIORATING),
                theme_state(theme_regime=ThemeRegime.DETERIORATING),
                ticker_state(),
                portfolio_state(),
                readiness_state(),
            )
        )
        decision = evaluate_policy(
            build_intent(snapshot=snapshot, position_size_requested=size),
            snapshot,
            config,
        )
        assert decision.final_risk_budget.as_tuple().exponent == -2, size


def test_below_minimum_executable_size_rejects_with_zero_budget() -> None:
    snapshot = build_snapshot(
        states=(
            market_state(regime=MarketRegime.CRISIS),
            ticker_state(),
            portfolio_state(),
            readiness_state(),
        )
    )
    intent = build_intent(snapshot=snapshot, position_size_requested=Decimal("120.00"))
    config = build_config(minimum_order_notional=Decimal("100.00"))

    decision = evaluate_policy(intent, snapshot, config)

    assert decision.action is PolicyAction.REJECT
    assert decision.final_risk_budget == Decimal("0")
    assert ReasonCode.SIZE_BELOW_MINIMUM_EXECUTABLE.value in decision.hard_vetoes
    assert decision.modifiers[-1].budget_after == Decimal("30.00")


def test_unreduced_approval_keeps_the_full_request() -> None:
    snapshot = build_snapshot()
    intent = build_intent(snapshot=snapshot)

    decision = evaluate_policy(intent, snapshot, build_config())

    assert decision.action is PolicyAction.APPROVE
    assert decision.final_risk_budget == decision.base_risk_budget


def test_config_rejects_risk_increasing_multipliers() -> None:
    with pytest.raises(ValueError, match="must not increase risk"):
        build_config(
            market_regime_multipliers={
                **MVP_POLICY_CONFIG.market_regime_multipliers,
                MarketRegime.STRONG_RISK_ON: Decimal("1.25"),
            }
        )


# --------------------------------------------------------------------------
# Determinism, identity, and execution basis
# --------------------------------------------------------------------------


def test_decision_identity_is_uuid5_of_declared_inputs() -> None:
    snapshot = build_snapshot()
    intent = build_intent(snapshot=snapshot)
    config = build_config()

    decision = evaluate_policy(intent, snapshot, config)

    expected = uuid5(
        POLICY_DECISION_NAMESPACE,
        "|".join(
            (
                str(intent.intent_id),
                snapshot.content_hash,
                config.content_hash,
                config.mode.value,
            )
        ),
    )
    assert decision.policy_decision_id == expected


def test_evaluation_is_deterministic_and_clock_free() -> None:
    snapshot = build_snapshot()
    intent = build_intent(snapshot=snapshot)
    config = build_config()

    first = evaluate_policy(intent, snapshot, config)
    second = evaluate_policy(intent, snapshot, config)

    assert content_hash(first) == content_hash(second)
    assert first.created_at == intent.created_at
    assert first.expires_at == intent.created_at + config.entry_window


def test_mode_changes_identity_and_execution_basis_only() -> None:
    snapshot = build_snapshot()
    intent = build_intent(snapshot=snapshot)
    enforce = evaluate_policy(intent, snapshot, build_config(mode=PolicyMode.ENFORCE))
    shadow = evaluate_policy(intent, snapshot, build_config(mode=PolicyMode.SHADOW))

    assert execution_basis(enforce) is ExecutionBasis.POLICY_FINAL
    assert execution_basis(shadow) is ExecutionBasis.BASELINE_INTENT
    assert enforce.policy_decision_id != shadow.policy_decision_id
    assert content_hash(
        enforce, exclude={"policy_decision_id", "mode"}
    ) == content_hash(shadow, exclude={"policy_decision_id", "mode"})


def test_shadow_records_the_same_counterfactual_budget_as_enforce() -> None:
    snapshot = build_snapshot(
        states=(
            market_state(regime=MarketRegime.RISK_OFF),
            ticker_state(),
            portfolio_state(),
            readiness_state(),
        )
    )
    intent = build_intent(snapshot=snapshot)
    shadow = evaluate_policy(intent, snapshot, build_config(mode=PolicyMode.SHADOW))

    assert shadow.action is PolicyAction.APPROVE_REDUCED
    assert shadow.final_risk_budget == Decimal("500.00")
    assert execution_basis(shadow) is ExecutionBasis.BASELINE_INTENT


_PURE_MODULES = (
    "core/nervous_system/policy/engine.py",
    "core/nervous_system/policy/rules.py",
    "core/nervous_system/policy/permissions.py",
    "core/nervous_system/policy/reason_codes.py",
    "core/nervous_system/config/policy.py",
)
_FORBIDDEN_IMPORTS = {
    "os",
    "io",
    "time",
    "random",
    "socket",
    "pathlib",
    "requests",
    "sqlalchemy",
    "httpx",
    "urllib",
    "subprocess",
}
_FORBIDDEN_CALLS = {"now", "utcnow", "today", "open", "getenv", "monotonic"}


@pytest.mark.parametrize("relative_path", _PURE_MODULES)
def test_policy_modules_are_pure(relative_path: str) -> None:
    tree = ast.parse(Path(relative_path).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name.split(".")[0] not in _FORBIDDEN_IMPORTS, alias.name
        elif isinstance(node, ast.ImportFrom) and node.module:
            assert node.module.split(".")[0] not in _FORBIDDEN_IMPORTS, node.module
        elif isinstance(node, ast.Call):
            name = getattr(node.func, "attr", None) or getattr(node.func, "id", None)
            assert name not in _FORBIDDEN_CALLS, f"{relative_path} calls {name}"


def test_every_reason_code_has_human_detail() -> None:
    for code in ReasonCode:
        assert describe(code).strip()
        assert describe(code.value) == describe(code)


def test_unknown_reason_code_is_rejected() -> None:
    with pytest.raises(KeyError):
        describe("NOT_A_REAL_REASON_CODE")


def test_evaluator_rejects_foreign_uuid_namespace_collision() -> None:
    """Two different configs cannot share a decision identity."""

    snapshot = build_snapshot()
    intent = build_intent(snapshot=snapshot)
    first = evaluate_policy(intent, snapshot, build_config())
    second = evaluate_policy(
        intent, snapshot, build_config(max_position_notional=Decimal("4000.00"))
    )

    assert first.policy_decision_id != second.policy_decision_id
    assert uuid4() != first.policy_decision_id


# --------------------------------------------------------------------------
# The daily-loss circuit breaker is optional on paper, mandatory live.
# --------------------------------------------------------------------------


def test_daily_loss_breaker_is_disabled_when_no_limit_is_configured() -> None:
    """A paper account must be allowed to show its real drawdown shape."""

    snapshot = build_snapshot(
        states=(
            market_state(),
            ticker_state(),
            portfolio_state(day_pl=-250_000.0),
            readiness_state(),
        )
    )
    intent = build_intent(snapshot=snapshot)

    # Same loss that trips the breaker in
    # test_daily_loss_breach_and_liquidity_veto_entry, only far larger.
    decision = evaluate_policy(intent, snapshot, build_config(max_daily_loss=None))

    assert ReasonCode.PORTFOLIO_MAX_DAILY_LOSS_BREACH.value not in decision.hard_vetoes


def test_the_shipped_paper_config_ships_with_the_breaker_off() -> None:
    assert MVP_POLICY_CONFIG.max_daily_loss is None


def test_production_live_may_not_disable_the_daily_loss_breaker() -> None:
    """Live is real money: refuse to construct a config without the breaker."""

    with pytest.raises(ValueError, match="max_daily_loss must be set for PRODUCTION_LIVE"):
        replace(
            MVP_POLICY_CONFIG,
            environment=RuntimeEnvironment.PRODUCTION_LIVE,
            max_daily_loss=None,
        )


# --------------------------------------------------------------------------
# The gross-exposure ceiling is optional on paper, mandatory live.
#
# On paper the broker's own buying power is the real constraint, and it is
# already enforced as BROKER_INSUFFICIENT_BUYING_POWER. A second hardcoded
# ceiling only diverges from the account it is meant to describe: on
# 2026-08-19 it sat at $150,000 against a $1,066,016 book and refused every
# entry outright.
# --------------------------------------------------------------------------


def test_gross_exposure_gate_is_disabled_when_no_ceiling_is_configured() -> None:
    snapshot = build_snapshot(
        states=(
            market_state(),
            ticker_state(),
            portfolio_state(
                positions=(
                    position(symbol="NVDA", market_value=-900_000.0),
                    position(symbol="AVGO", market_value=500_000.0),
                )
            ),
            readiness_state(),
        )
    )
    intent = build_intent(snapshot=snapshot, position_size_requested=Decimal("5000.00"))

    decision = evaluate_policy(intent, snapshot, build_config(max_gross_notional=None))

    assert ReasonCode.PORTFOLIO_MAX_GROSS_NOTIONAL_BREACH.value not in decision.hard_vetoes


def test_an_uncountable_position_is_harmless_when_the_gate_is_off() -> None:
    """PORTFOLIO_EXPOSURE_UNKNOWN guards the ceiling; with no ceiling it must
    not veto entries for a reason that no longer applies."""

    snapshot = build_snapshot(
        states=(
            market_state(),
            ticker_state(),
            portfolio_state(positions=(position(symbol="NVDA", market_value=None),)),
            readiness_state(),
        )
    )
    intent = build_intent(snapshot=snapshot)

    decision = evaluate_policy(intent, snapshot, build_config(max_gross_notional=None))

    assert ReasonCode.PORTFOLIO_EXPOSURE_UNKNOWN.value not in decision.hard_vetoes


def test_buying_power_still_binds_when_the_gross_gate_is_off() -> None:
    """The paper account is meant to be constrained by its own buying power."""

    snapshot = build_snapshot(
        states=(
            market_state(),
            ticker_state(),
            portfolio_state(buying_power=1_000.0),
            readiness_state(),
        )
    )
    intent = build_intent(snapshot=snapshot, position_size_requested=Decimal("5000.00"))

    decision = evaluate_policy(intent, snapshot, build_config(max_gross_notional=None))

    assert ReasonCode.BROKER_INSUFFICIENT_BUYING_POWER.value in decision.hard_vetoes


def test_the_shipped_paper_config_ships_with_the_gross_gate_off() -> None:
    assert MVP_POLICY_CONFIG.max_gross_notional is None


def test_production_live_may_not_disable_the_gross_exposure_ceiling() -> None:
    with pytest.raises(ValueError, match="max_gross_notional must be set for PRODUCTION_LIVE"):
        replace(
            MVP_POLICY_CONFIG,
            environment=RuntimeEnvironment.PRODUCTION_LIVE,
            max_daily_loss=Decimal("2000.00"),
            max_gross_notional=None,
        )
