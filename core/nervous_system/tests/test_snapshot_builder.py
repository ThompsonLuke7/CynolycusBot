from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from uuid import UUID

import pytest

from core.calendar.us_market_calendar import is_trading_day, prev_trading_day
from core.nervous_system.config.freshness import (
    META_4H_1420_PROFILE,
    META_4H_1620_PROFILE,
    SnapshotProfile,
    FreshnessRule,
    canonical_profile_payload,
    get_snapshot_profile,
)
from core.nervous_system.context.requirements import (
    SnapshotEntityScope,
    evaluate_requirements,
)
from core.nervous_system.context.snapshot_builder import SnapshotBuilder
from core.nervous_system.contracts.base import canonical_json
from core.nervous_system.contracts.context import RejectedCandidate
from core.nervous_system.contracts.enums import (
    DealerRegime,
    Direction,
    MarketRegime,
    MissingStateAction,
    StateType,
    ThemeRegime,
    TickerSetup,
)
from core.nervous_system.contracts.quality import DataQualitySummary
from core.nervous_system.contracts.states import (
    CatalystEvent,
    CatalystPressure,
    DealerState,
    MarketState,
    PortfolioState,
    ReadinessState,
    SectorState,
    TickerState,
    ThemeMembership,
    ThemeState,
)
from core.nervous_system.persistence.repositories.state import StateRepository


UTC = timezone.utc
TICKER = "AMD"
DECISION_1420 = datetime(2026, 7, 30, 18, 20, tzinfo=UTC)
DECISION_1620 = datetime(2026, 7, 30, 20, 20, tzinfo=UTC)
DECISION_BAR = datetime(2026, 7, 30, 18, 0, tzinfo=UTC)


def _base(
    *,
    state_id: UUID,
    state_type: StateType,
    entity_id: str,
    as_of: datetime,
    available_at: datetime,
    generated_at: datetime | None = None,
    valid_until: datetime | None = None,
    producer: str = "test-producer@1",
    schema_version: int = 1,
    model_version: str = "test-model@1",
    feature_version: str = "test-features@1",
    config_version: str = "test-config@1",
) -> dict:
    return {
        "state_id": state_id,
        "state_type": state_type,
        "entity_id": entity_id,
        "as_of": as_of,
        "available_at": available_at,
        "generated_at": generated_at or available_at,
        "valid_until": valid_until or available_at + timedelta(days=2),
        "source_window_start": as_of - timedelta(minutes=5),
        "source_window_end": as_of,
        "schema_version": schema_version,
        "producer": producer,
        "model_version": model_version,
        "feature_version": feature_version,
        "config_version": config_version,
        "lineage_ids": (f"test:{state_id}",),
        "data_quality": DataQualitySummary(),
    }


def _ticker(
    state_id: UUID,
    *,
    entity_id: str = TICKER,
    as_of: datetime = datetime(2026, 7, 30, 17, 0, tzinfo=UTC),
    available_at: datetime = datetime(2026, 7, 30, 17, 5, tzinfo=UTC),
    generated_at: datetime | None = None,
    valid_until: datetime | None = None,
    producer: str = "test-producer@1",
    schema_version: int = 1,
    model_version: str = "test-model@1",
    feature_version: str = "test-features@1",
    config_version: str = "test-config@1",
) -> TickerState:
    payload = _base(
        state_id=state_id,
        state_type=StateType.TICKER,
        entity_id=entity_id,
        as_of=as_of,
        available_at=available_at,
        generated_at=generated_at,
        valid_until=valid_until,
        producer=producer,
        schema_version=schema_version,
        model_version=model_version,
        feature_version=feature_version,
        config_version=config_version,
    )
    payload.update(
        {
            "ticker": entity_id,
            "selected_bar": as_of,
            "reference_price": 100.0,
            "ticker_setup": TickerSetup.BREAKOUT,
        }
    )
    return TickerState(**payload)


def _market(
    state_id: UUID,
    *,
    entity_id: str = "US",
    as_of: datetime = datetime(2026, 7, 29, 20, 0, tzinfo=UTC),
    available_at: datetime = datetime(2026, 7, 29, 20, 30, tzinfo=UTC),
    generated_at: datetime | None = None,
    valid_until: datetime | None = None,
) -> MarketState:
    payload = _base(
        state_id=state_id,
        state_type=StateType.MARKET,
        entity_id=entity_id,
        as_of=as_of,
        available_at=available_at,
        generated_at=generated_at,
        valid_until=valid_until,
    )
    payload.update({"regime": MarketRegime.NEUTRAL, "metrics": {"version": 1.0}})
    return MarketState(**payload)


def _sector(
    state_id: UUID,
    *,
    sector_id: str = "technology",
    as_of: datetime = datetime(2026, 7, 29, 20, 0, tzinfo=UTC),
    available_at: datetime = datetime(2026, 7, 29, 20, 30, tzinfo=UTC),
    valid_until: datetime | None = None,
) -> SectorState:
    payload = _base(
        state_id=state_id,
        state_type=StateType.SECTOR,
        entity_id=sector_id,
        as_of=as_of,
        available_at=available_at,
        valid_until=valid_until,
    )
    payload.update({"sector_id": sector_id, "capital_flow_direction": Direction.UNKNOWN})
    return SectorState(**payload)


def _portfolio(
    state_id: UUID,
    *,
    account_alias: str = "paper",
    as_of: datetime = datetime(2026, 7, 30, 17, 30, tzinfo=UTC),
    available_at: datetime = datetime(2026, 7, 30, 17, 31, tzinfo=UTC),
    valid_until: datetime | None = None,
) -> PortfolioState:
    payload = _base(
        state_id=state_id,
        state_type=StateType.PORTFOLIO,
        entity_id=account_alias,
        as_of=as_of,
        available_at=available_at,
        valid_until=valid_until,
    )
    payload.update(
        {
            "account_alias": account_alias,
            "equity": 100000.0,
            "cash": 100000.0,
            "buying_power": 100000.0,
            "broker_observed_at": as_of,
        }
    )
    return PortfolioState(**payload)


def _readiness(
    state_id: UUID,
    *,
    job: str = "nightly_data_readiness",
    as_of: datetime = datetime(2026, 7, 30, 17, 0, tzinfo=UTC),
    available_at: datetime = datetime(2026, 7, 30, 17, 1, tzinfo=UTC),
    valid_until: datetime | None = None,
) -> ReadinessState:
    payload = _base(
        state_id=state_id,
        state_type=StateType.READINESS,
        entity_id=job,
        as_of=as_of,
        available_at=available_at,
        valid_until=valid_until,
    )
    payload.update(
        {
            "job": job,
            "status": "READY",
            "ready": True,
            "completed_at": as_of,
            "checked_at": available_at,
            "max_age_hours": 96.0,
            "latest_required_session": "2026-07-29",
        }
    )
    return ReadinessState(**payload)


def _dealer(
    state_id: UUID,
    *,
    available_at: datetime = datetime(2026, 7, 30, 19, 0, tzinfo=UTC),
    as_of: datetime = datetime(2026, 7, 30, 18, 59, tzinfo=UTC),
) -> DealerState:
    payload = _base(
        state_id=state_id,
        state_type=StateType.DEALER,
        entity_id=TICKER,
        as_of=as_of,
        available_at=available_at,
    )
    payload.update(
        {
            "ticker": TICKER,
            "dealer_regime": DealerRegime.NEUTRAL_GAMMA,
            "spot": 100.0,
        }
    )
    return DealerState(**payload)


def _theme_membership(
    state_id: UUID,
    *,
    ticker: str = TICKER,
    theme_id: str = "alpha",
    as_of: datetime = datetime(2026, 7, 30, 17, 0, tzinfo=UTC),
    available_at: datetime = datetime(2026, 7, 30, 17, 5, tzinfo=UTC),
    valid_until: datetime | None = None,
    effective_from: datetime | None = None,
    effective_until: datetime | None = None,
) -> ThemeMembership:
    payload = _base(
        state_id=state_id,
        state_type=StateType.THEME_MEMBERSHIP,
        entity_id=theme_id,
        as_of=as_of,
        available_at=available_at,
        valid_until=valid_until,
    )
    payload.update(
        {
            "ticker": ticker,
            "theme_id": theme_id,
            "weight": 0.75,
            "membership_version": "themes@1",
            "effective_from": effective_from or as_of,
            "effective_until": effective_until,
        }
    )
    return ThemeMembership(**payload)


def _theme_state(
    state_id: UUID,
    *,
    theme_id: str = "alpha",
    as_of: datetime = datetime(2026, 7, 30, 17, 0, tzinfo=UTC),
    available_at: datetime = datetime(2026, 7, 30, 17, 5, tzinfo=UTC),
    valid_until: datetime | None = None,
) -> ThemeState:
    payload = _base(
        state_id=state_id,
        state_type=StateType.THEME,
        entity_id=theme_id,
        as_of=as_of,
        available_at=available_at,
        valid_until=valid_until,
    )
    payload.update({"theme_id": theme_id, "theme_regime": ThemeRegime.UNKNOWN})
    return ThemeState(**payload)


def _catalyst_event(
    state_id: UUID,
    *,
    ticker: str | None = TICKER,
    as_of: datetime = datetime(2026, 7, 30, 17, 0, tzinfo=UTC),
    available_at: datetime = datetime(2026, 7, 30, 17, 5, tzinfo=UTC),
    valid_until: datetime | None = None,
) -> CatalystEvent:
    payload = _base(
        state_id=state_id,
        state_type=StateType.CATALYST_EVENT,
        entity_id=str(state_id),
        as_of=as_of,
        available_at=available_at,
        valid_until=valid_until,
    )
    payload.update(
        {
            "event_id": state_id,
            "ticker": ticker,
            "event_type": "NEWS",
            "event_time": as_of,
            "published_at": as_of,
            "observed_at": available_at,
            "source": "wire",
            "headline": "test catalyst",
            "channel": "NEWS",
        }
    )
    return CatalystEvent(**payload)


def _catalyst_pressure(
    state_id: UUID,
    *,
    scope_type: str = "TICKER",
    scope_id: str = TICKER,
    as_of: datetime = datetime(2026, 7, 30, 17, 0, tzinfo=UTC),
    available_at: datetime = datetime(2026, 7, 30, 17, 5, tzinfo=UTC),
    valid_until: datetime | None = None,
) -> CatalystPressure:
    payload = _base(
        state_id=state_id,
        state_type=StateType.CATALYST_PRESSURE,
        entity_id=scope_id,
        as_of=as_of,
        available_at=available_at,
        valid_until=valid_until,
    )
    payload.update(
        {
            "scope_type": scope_type,
            "scope_id": scope_id,
            "channel_scores": {"NEWS": 0.8},
            "aggregate_score": 0.8,
            "event_ids": (),
            "transition_probabilities": {},
        }
    )
    return CatalystPressure(**payload)


def _save(repo: StateRepository, *states) -> None:
    repo.insert_states_idempotently(states)


def _required_states(*, decision_time: datetime = DECISION_1420):
    return (
        _ticker(UUID("00000000-0000-0000-0000-000000001001")),
        _market(UUID("00000000-0000-0000-0000-000000001002")),
        _sector(UUID("00000000-0000-0000-0000-000000001003")),
        _portfolio(UUID("00000000-0000-0000-0000-000000001004")),
        _readiness(UUID("00000000-0000-0000-0000-000000001005")),
    )


class _RecordingSnapshotRepository:
    def __init__(self, candidates=()):
        self.candidates = tuple(candidates)
        self.candidate_calls = 0
        self.requested_state_types = None
        self.requested_decision_time = None
        self.saved_snapshots = []

    def get_state_candidates_for_snapshot(self, state_types, decision_time):
        self.candidate_calls += 1
        self.requested_state_types = tuple(state_types)
        self.requested_decision_time = decision_time
        return self.candidates

    def save_context_snapshot_idempotently(self, snapshot):
        self.saved_snapshots.append(snapshot)
        return snapshot


def test_profiles_are_frozen_versioned_and_have_the_same_required_surface():
    first = get_snapshot_profile("meta_4h_1420@1")
    second = get_snapshot_profile("meta_4h_1620@1")

    assert first == META_4H_1420_PROFILE
    assert second == META_4H_1620_PROFILE
    assert first.profile_hash == get_snapshot_profile("meta_4h_1420@1").profile_hash
    assert first.profile_hash != second.profile_hash
    assert {rule.state_type for rule in first.rules if rule.required} == {
        StateType.TICKER,
        StateType.MARKET,
        StateType.SECTOR,
        StateType.PORTFOLIO,
        StateType.READINESS,
    }
    assert {rule.state_type for rule in first.rules if not rule.required} == {
        StateType.THEME_MEMBERSHIP,
        StateType.THEME,
        StateType.CATALYST_EVENT,
        StateType.CATALYST_PRESSURE,
        StateType.DEALER,
    }
    max_ages = {rule.state_type: rule.max_age for rule in first.rules}
    assert max_ages[StateType.TICKER] == timedelta(hours=6)
    assert max_ages[StateType.PORTFOLIO] == timedelta(hours=24)
    assert max_ages[StateType.DEALER] == timedelta(hours=4)
    for state_type in (
        StateType.MARKET,
        StateType.SECTOR,
        StateType.READINESS,
        StateType.THEME,
        StateType.THEME_MEMBERSHIP,
        StateType.CATALYST_EVENT,
        StateType.CATALYST_PRESSURE,
    ):
        assert max_ages[state_type] == timedelta(hours=96)
    with pytest.raises((AttributeError, TypeError)):
        first.rules += (
            FreshnessRule(
                StateType.TICKER,
                True,
                timedelta(hours=1),
                MissingStateAction.REJECT,
            ),
        )


def test_profile_hash_uses_exact_max_age_microseconds():
    first = SnapshotProfile(
        profile_id="microsecond-age@1",
        rules=(
            FreshnessRule(
                StateType.TICKER,
                True,
                timedelta(seconds=1),
                MissingStateAction.REJECT,
            ),
        ),
    )
    second = SnapshotProfile(
        profile_id="microsecond-age@1",
        rules=(
            FreshnessRule(
                StateType.TICKER,
                True,
                timedelta(seconds=1, microseconds=1),
                MissingStateAction.REJECT,
            ),
        ),
    )

    assert first.profile_hash != second.profile_hash
    assert canonical_profile_payload(second)["rules"][0]["max_age_microseconds"] == 1_000_001


def test_profile_policy_metadata_is_frozen_hashed_and_strictly_validated():
    rule = FreshnessRule(
        StateType.DEALER,
        False,
        timedelta(hours=4),
        MissingStateAction.WARN,
    )
    profile = SnapshotProfile(
        profile_id="metadata@1",
        rules=(rule,),
        dealer_allowed=True,
        dealer_capture_after_et=time(14, 20),
        market_session_lag=1,
    )
    changed = SnapshotProfile(
        profile_id="metadata@1",
        rules=(rule,),
        dealer_allowed=True,
        dealer_capture_after_et=time(14, 21),
        market_session_lag=1,
    )

    assert profile.profile_hash != changed.profile_hash
    assert canonical_profile_payload(profile)["policy"] == {
        "dealer_allowed": True,
        "dealer_capture_after_et": "14:20:00.000000",
        "market_session_lag": 1,
    }
    invalid_metadata = (
        {"dealer_allowed": "yes"},
        {"dealer_allowed": False, "dealer_capture_after_et": time(14, 20)},
        {"dealer_capture_after_et": time(14, 20, tzinfo=UTC)},
        {"market_session_lag": -1},
        {"market_session_lag": True},
    )
    for metadata in invalid_metadata:
        with pytest.raises((TypeError, ValueError)):
            SnapshotProfile(profile_id="invalid-metadata@1", rules=(rule,), **metadata)


def test_snapshot_profile_rejects_duplicate_state_rules():
    rule = FreshnessRule(
        StateType.TICKER,
        True,
        timedelta(hours=1),
        MissingStateAction.REJECT,
    )
    with pytest.raises(ValueError, match="repeat"):
        SnapshotProfile(profile_id="duplicates@1", rules=(rule, rule))


def test_builder_loads_candidate_repository_exactly_once():
    repository = _RecordingSnapshotRepository()
    profile = SnapshotProfile(
        profile_id="one-load@1",
        rules=(
            FreshnessRule(
                StateType.TICKER,
                False,
                timedelta(hours=1),
                MissingStateAction.WARN,
            ),
        ),
    )

    snapshot = SnapshotBuilder(repository).build(
        strategy_id="one-load",
        entity_id=TICKER,
        decision_time=DECISION_1420,
        decision_bar=DECISION_BAR,
        profile=profile,
    )

    assert repository.candidate_calls == 1
    assert repository.requested_state_types == (StateType.TICKER,)
    assert repository.requested_decision_time == DECISION_1420
    assert repository.saved_snapshots == [snapshot]


@pytest.mark.parametrize("naive_field", ("decision_time", "decision_bar"))
def test_builder_rejects_naive_time_before_loading_candidates(naive_field):
    repository = _RecordingSnapshotRepository()
    profile = SnapshotProfile(
        profile_id="naive-input@1",
        rules=(
            FreshnessRule(
                StateType.TICKER,
                False,
                timedelta(hours=1),
                MissingStateAction.WARN,
            ),
        ),
    )
    values = {"decision_time": DECISION_1420, "decision_bar": DECISION_BAR}
    values[naive_field] = values[naive_field].replace(tzinfo=None)

    with pytest.raises(ValueError, match="timezone-aware"):
        SnapshotBuilder(repository).build(
            strategy_id="naive-input",
            entity_id=TICKER,
            profile=profile,
            **values,
        )
    assert repository.candidate_calls == 0


@pytest.mark.parametrize(
    ("strategy_id", "entity_id"),
    (("", TICKER), ("one-load", ""), (None, TICKER)),
)
def test_builder_rejects_malformed_identity_before_loading_candidates(
    strategy_id,
    entity_id,
):
    repository = _RecordingSnapshotRepository()
    profile = SnapshotProfile(
        profile_id="malformed-input@1",
        rules=(
            FreshnessRule(
                StateType.TICKER,
                False,
                timedelta(hours=1),
                MissingStateAction.WARN,
            ),
        ),
    )

    with pytest.raises(ValueError, match="non-empty"):
        SnapshotBuilder(repository).build(
            strategy_id=strategy_id,
            entity_id=entity_id,
            decision_time=DECISION_1420,
            decision_bar=DECISION_BAR,
            profile=profile,
        )
    assert repository.candidate_calls == 0


def test_decision_time_states_may_follow_decision_bar_when_available():
    decision_time = DECISION_1420
    decision_bar = DECISION_BAR
    as_of = datetime(2026, 7, 30, 18, 10, tzinfo=UTC)
    available_at = datetime(2026, 7, 30, 18, 15, tzinfo=UTC)
    profile = SnapshotProfile(
        profile_id="decision-time-states@1",
        rules=tuple(
            FreshnessRule(state_type, True, timedelta(hours=24), MissingStateAction.REJECT)
            for state_type in (
                StateType.PORTFOLIO,
                StateType.READINESS,
                StateType.DEALER,
                StateType.CATALYST_EVENT,
                StateType.CATALYST_PRESSURE,
            )
        ),
    )
    states = (
        _portfolio(
            UUID("00000000-0000-0000-0000-000000001060"),
            as_of=as_of,
            available_at=available_at,
            valid_until=available_at + timedelta(days=2),
        ),
        _readiness(
            UUID("00000000-0000-0000-0000-000000001061"),
            as_of=as_of,
            available_at=available_at,
            valid_until=available_at + timedelta(days=2),
        ),
        _dealer(
            UUID("00000000-0000-0000-0000-000000001062"),
            as_of=as_of,
            available_at=available_at,
        ),
        _catalyst_event(
            UUID("00000000-0000-0000-0000-000000001063"),
            as_of=as_of,
            available_at=available_at,
        ),
        _catalyst_pressure(
            UUID("00000000-0000-0000-0000-000000001064"),
            as_of=as_of,
            available_at=available_at,
        ),
    )
    evaluation = evaluate_requirements(
        states,
        entity_id=TICKER,
        decision_time=decision_time,
        decision_bar=decision_bar,
        profile=profile,
    )

    assert evaluation.valid is True
    assert {state.state_id for state in evaluation.selected_states} == {
        UUID("00000000-0000-0000-0000-000000001060"),
        UUID("00000000-0000-0000-0000-000000001061"),
        UUID("00000000-0000-0000-0000-000000001062"),
        UUID("00000000-0000-0000-0000-000000001063"),
        UUID("00000000-0000-0000-0000-000000001064"),
    }


def test_relevance_filters_sector_membership_theme_and_catalyst_candidates():
    decision_time = DECISION_1420
    profile = SnapshotProfile(
        profile_id="relevance@1",
        rules=(
            FreshnessRule(StateType.SECTOR, True, timedelta(hours=96), MissingStateAction.REJECT),
            FreshnessRule(
                StateType.THEME_MEMBERSHIP,
                False,
                timedelta(hours=96),
                MissingStateAction.WARN,
            ),
            FreshnessRule(StateType.THEME, False, timedelta(hours=96), MissingStateAction.WARN),
            FreshnessRule(
                StateType.CATALYST_EVENT,
                False,
                timedelta(hours=96),
                MissingStateAction.WARN,
            ),
            FreshnessRule(
                StateType.CATALYST_PRESSURE,
                False,
                timedelta(hours=96),
                MissingStateAction.WARN,
            ),
        ),
    )
    sector = _sector(UUID("00000000-0000-0000-0000-000000001070"))
    finance = _sector(UUID("00000000-0000-0000-0000-000000001071"), sector_id="finance")
    amd_membership = _theme_membership(
        UUID("00000000-0000-0000-0000-000000001072"),
        ticker=TICKER,
        theme_id="alpha",
    )
    msft_membership = _theme_membership(
        UUID("00000000-0000-0000-0000-000000001073"),
        ticker="MSFT",
        theme_id="alpha",
    )
    related_theme = _theme_state(UUID("00000000-0000-0000-0000-000000001074"), theme_id="alpha")
    unrelated_theme = _theme_state(UUID("00000000-0000-0000-0000-000000001075"), theme_id="beta")
    amd_event = _catalyst_event(UUID("00000000-0000-0000-0000-000000001076"), ticker=TICKER)
    msft_event = _catalyst_event(UUID("00000000-0000-0000-0000-000000001077"), ticker="MSFT")
    market_event = _catalyst_event(UUID("00000000-0000-0000-0000-000000001078"), ticker=None)
    amd_pressure = _catalyst_pressure(UUID("00000000-0000-0000-0000-000000001079"))
    msft_pressure = _catalyst_pressure(
        UUID("00000000-0000-0000-0000-000000001080"), scope_id="MSFT"
    )
    market_pressure = _catalyst_pressure(
        UUID("00000000-0000-0000-0000-000000001081"),
        scope_type="MARKET",
        scope_id="US",
    )
    other_market_pressure = _catalyst_pressure(
        UUID("00000000-0000-0000-0000-000000001082"),
        scope_type="MARKET",
        scope_id="EU",
    )

    evaluation = evaluate_requirements(
        (
            sector,
            finance,
            amd_membership,
            msft_membership,
            related_theme,
            unrelated_theme,
            amd_event,
            msft_event,
            market_event,
            amd_pressure,
            msft_pressure,
            market_pressure,
            other_market_pressure,
        ),
        entity_id=TICKER,
        decision_time=decision_time,
        decision_bar=DECISION_BAR,
        profile=profile,
        scope=SnapshotEntityScope(sector_entity_ids=("technology",)),
    )

    selected_ids = {state.state_id for state in evaluation.selected_states}
    assert selected_ids == {
        sector.state_id,
        amd_membership.state_id,
        related_theme.state_id,
        amd_event.state_id,
        market_event.state_id,
        amd_pressure.state_id,
        market_pressure.state_id,
    }
    rejected = {item.state_id: item.reason_code for item in evaluation.rejected_candidates}
    assert rejected[finance.state_id] == "WRONG_ENTITY"
    assert rejected[msft_membership.state_id] == "WRONG_ENTITY"
    assert rejected[unrelated_theme.state_id] == "THEME_NOT_IN_TICKER_MEMBERSHIPS"
    assert rejected[msft_event.state_id] == "WRONG_ENTITY"
    assert rejected[msft_pressure.state_id] == "WRONG_ENTITY"
    assert rejected[other_market_pressure.state_id] == "WRONG_ENTITY"


def test_required_sector_without_explicit_scope_is_missing():
    profile = SnapshotProfile(
        profile_id="sector-scope@1",
        rules=(
            FreshnessRule(StateType.SECTOR, True, timedelta(hours=96), MissingStateAction.REJECT),
        ),
    )
    evaluation = evaluate_requirements(
        (_sector(UUID("00000000-0000-0000-0000-000000001083")),),
        entity_id=TICKER,
        decision_time=DECISION_1420,
        decision_bar=DECISION_BAR,
        profile=profile,
    )
    assert evaluation.valid is False
    assert evaluation.selected_states == ()
    assert evaluation.requirement_results[0].reason_code == "MISSING_REQUIRED_STATE"


@pytest.mark.postgres
def test_prior_session_states_survive_three_day_holiday_weekend(pg_session):
    assert is_trading_day(date(2026, 7, 2)) is True
    assert is_trading_day(date(2026, 7, 3)) is False
    assert prev_trading_day(date(2026, 7, 6)) == date(2026, 7, 2)

    repo = StateRepository(pg_session)
    decision_time = datetime(2026, 7, 6, 18, 20, tzinfo=UTC)
    decision_bar = datetime(2026, 7, 6, 18, 0, tzinfo=UTC)
    prior_available = datetime(2026, 7, 2, 20, 30, tzinfo=UTC)
    prior_as_of = datetime(2026, 7, 2, 20, 0, tzinfo=UTC)
    valid_until = decision_time + timedelta(days=1)
    states = (
        _ticker(
            UUID("00000000-0000-0000-0000-000000001090"),
            as_of=datetime(2026, 7, 6, 17, 0, tzinfo=UTC),
            available_at=datetime(2026, 7, 6, 17, 5, tzinfo=UTC),
            valid_until=valid_until,
        ),
        _market(
            UUID("00000000-0000-0000-0000-000000001091"),
            as_of=prior_as_of,
            available_at=prior_available,
            valid_until=valid_until,
        ),
        _sector(
            UUID("00000000-0000-0000-0000-000000001092"),
            as_of=prior_as_of,
            available_at=prior_available,
            valid_until=valid_until,
        ),
        _portfolio(
            UUID("00000000-0000-0000-0000-000000001093"),
            as_of=datetime(2026, 7, 6, 17, 30, tzinfo=UTC),
            available_at=datetime(2026, 7, 6, 17, 31, tzinfo=UTC),
            valid_until=valid_until,
        ),
        _readiness(
            UUID("00000000-0000-0000-0000-000000001094"),
            as_of=prior_as_of,
            available_at=prior_available,
            valid_until=valid_until,
        ),
    )
    _save(repo, *states)

    snapshot = SnapshotBuilder(repo, sector_entity_ids=("technology",)).build(
        strategy_id="holiday-weekend",
        entity_id=TICKER,
        decision_time=decision_time,
        decision_bar=decision_bar,
        profile=META_4H_1420_PROFILE,
    )

    assert snapshot.valid is True
    assert snapshot.market_state is not None
    assert snapshot.market_state.state_id == states[1].state_id
    assert snapshot.readiness_state is not None
    assert snapshot.readiness_state.state_id == states[4].state_id
    assert snapshot.decision_session == "2026-07-06"


@pytest.mark.parametrize(
    ("profile", "decision_hour"),
    ((META_4H_1420_PROFILE, 18), (META_4H_1620_PROFILE, 20)),
)
@pytest.mark.parametrize(
    ("decision_date", "expected_date", "older_date", "expect_selection"),
    (
        (date(2026, 7, 30), date(2026, 7, 29), date(2026, 7, 28), True),
        (date(2026, 7, 6), date(2026, 7, 2), date(2026, 7, 1), True),
        (date(2026, 7, 7), date(2026, 7, 6), date(2026, 7, 2), False),
    ),
)
def test_meta_market_requires_exact_prior_trading_session(
    profile,
    decision_hour,
    decision_date,
    expected_date,
    older_date,
    expect_selection,
):
    decision_time = datetime.combine(
        decision_date,
        time(decision_hour, 20),
        tzinfo=UTC,
    )
    candidate_date = expected_date if expect_selection else older_date
    candidate = _market(
        UUID(
            f"00000000-0000-0000-0000-{decision_date:%m%d}{decision_hour:02d}000001"
        ),
        as_of=datetime.combine(candidate_date, time(20), tzinfo=UTC),
        available_at=datetime.combine(candidate_date, time(20, 30), tzinfo=UTC),
        valid_until=decision_time + timedelta(days=1),
    )
    unrelated_older = _market(
        UUID(
            f"00000000-0000-0000-0000-{decision_date:%m%d}{decision_hour:02d}000002"
        ),
        as_of=datetime.combine(older_date, time(20), tzinfo=UTC),
        available_at=datetime.combine(older_date, time(20, 30), tzinfo=UTC),
        valid_until=decision_time + timedelta(days=1),
    )

    evaluation = evaluate_requirements(
        (candidate, unrelated_older),
        entity_id=TICKER,
        decision_time=decision_time,
        decision_bar=decision_time - timedelta(minutes=20),
        profile=profile,
    )
    selected_market = tuple(
        state for state in evaluation.selected_states if state.state_type is StateType.MARKET
    )
    rejected = {item.state_id: item.reason_code for item in evaluation.rejected_candidates}

    assert profile.market_session_lag == 1
    if expect_selection:
        assert tuple(state.state_id for state in selected_market) == (candidate.state_id,)
        assert rejected[unrelated_older.state_id] == "MARKET_SESSION_MISMATCH"
    else:
        assert selected_market == ()
        assert rejected[candidate.state_id] == "MARKET_SESSION_MISMATCH"


def test_dealer_policy_comes_from_profile_metadata_and_survives_new_version():
    dealer_rule = FreshnessRule(
        StateType.DEALER,
        False,
        timedelta(hours=4),
        MissingStateAction.WARN,
    )
    profile_v2 = SnapshotProfile(
        profile_id="meta_4h_1620@2",
        rules=(dealer_rule,),
        dealer_allowed=True,
        dealer_capture_after_et=time(14, 20),
        market_session_lag=1,
    )
    disallowed = SnapshotProfile(
        profile_id="dealer-shadow-off@2",
        rules=(dealer_rule,),
        dealer_allowed=False,
        dealer_capture_after_et=None,
        market_session_lag=0,
    )
    boundary = _dealer(
        UUID("00000000-0000-0000-0000-000000001120"),
        as_of=datetime(2026, 7, 30, 18, 20, tzinfo=UTC),
        available_at=datetime(2026, 7, 30, 18, 21, tzinfo=UTC),
    )
    post_boundary = _dealer(
        UUID("00000000-0000-0000-0000-000000001121"),
        as_of=datetime(2026, 7, 30, 18, 21, tzinfo=UTC),
        available_at=datetime(2026, 7, 30, 18, 22, tzinfo=UTC),
    )

    evaluation = evaluate_requirements(
        (boundary, post_boundary),
        entity_id=TICKER,
        decision_time=DECISION_1620,
        decision_bar=datetime(2026, 7, 30, 20, 0, tzinfo=UTC),
        profile=profile_v2,
    )
    assert tuple(state.state_id for state in evaluation.selected_states) == (
        post_boundary.state_id,
    )
    rejected = {item.state_id: item.reason_code for item in evaluation.rejected_candidates}
    assert rejected[boundary.state_id] == "DEALER_CAPTURE_NOT_POST_BOUNDARY"

    disabled_evaluation = evaluate_requirements(
        (post_boundary,),
        entity_id=TICKER,
        decision_time=DECISION_1620,
        decision_bar=datetime(2026, 7, 30, 20, 0, tzinfo=UTC),
        profile=disallowed,
    )
    assert disabled_evaluation.selected_states == ()
    assert disabled_evaluation.rejected_candidates[0].reason_code == "DEALER_NOT_ALLOWED"


@pytest.mark.parametrize(
    ("decision_time", "capture_boundary"),
    (
        (
            datetime(2026, 10, 30, 20, 20, tzinfo=UTC),
            datetime(2026, 10, 30, 18, 20, tzinfo=UTC),
        ),
        (
            datetime(2026, 11, 2, 21, 20, tzinfo=UTC),
            datetime(2026, 11, 2, 19, 20, tzinfo=UTC),
        ),
    ),
)
def test_dealer_capture_boundary_uses_dst_safe_et_conversion(
    decision_time,
    capture_boundary,
):
    profile = SnapshotProfile(
        profile_id="dealer-dst@2",
        rules=(
            FreshnessRule(
                StateType.DEALER,
                False,
                timedelta(hours=4),
                MissingStateAction.WARN,
            ),
        ),
        dealer_allowed=True,
        dealer_capture_after_et=time(14, 20),
        market_session_lag=0,
    )
    at_boundary = _dealer(
        UUID(
            f"00000000-0000-0000-0000-{decision_time:%m%d%H}000001"
        ),
        as_of=capture_boundary,
        available_at=capture_boundary + timedelta(minutes=1),
    )
    after_boundary = _dealer(
        UUID(
            f"00000000-0000-0000-0000-{decision_time:%m%d%H}000002"
        ),
        as_of=capture_boundary + timedelta(minutes=1),
        available_at=capture_boundary + timedelta(minutes=2),
    )

    evaluation = evaluate_requirements(
        (at_boundary, after_boundary),
        entity_id=TICKER,
        decision_time=decision_time,
        decision_bar=decision_time - timedelta(minutes=20),
        profile=profile,
    )

    assert tuple(state.state_id for state in evaluation.selected_states) == (
        after_boundary.state_id,
    )
    rejected = {item.state_id: item.reason_code for item in evaluation.rejected_candidates}
    assert rejected[at_boundary.state_id] == "DEALER_CAPTURE_NOT_POST_BOUNDARY"


def test_ticker_selection_uses_exact_four_level_tie_break():
    profile = SnapshotProfile(
        profile_id="tie-break@1",
        rules=(
            FreshnessRule(
                StateType.TICKER,
                True,
                timedelta(hours=24),
                MissingStateAction.REJECT,
            ),
        ),
    )
    low_id = UUID("00000000-0000-0000-0000-000000001100")
    high_id = UUID("00000000-0000-0000-0000-000000001101")
    as_of = datetime(2026, 7, 30, 16, 0, tzinfo=UTC)
    earlier = datetime(2026, 7, 30, 17, 0, tzinfo=UTC)
    later = datetime(2026, 7, 30, 17, 10, tzinfo=UTC)

    pairs = (
        (
            _ticker(low_id, as_of=as_of, available_at=later, generated_at=later),
            _ticker(
                high_id,
                as_of=as_of,
                available_at=earlier,
                generated_at=later + timedelta(minutes=20),
                producer="test-producer@9",
            ),
            low_id,
        ),
        (
            _ticker(low_id, as_of=as_of, available_at=earlier, generated_at=later),
            _ticker(
                high_id,
                as_of=as_of,
                available_at=earlier,
                generated_at=earlier,
                producer="test-producer@9",
            ),
            low_id,
        ),
        (
            _ticker(
                low_id,
                as_of=as_of,
                available_at=earlier,
                generated_at=earlier,
                config_version="test-config@2",
            ),
            _ticker(
                high_id,
                as_of=as_of,
                available_at=earlier,
                generated_at=earlier,
                config_version="test-config@1",
            ),
            low_id,
        ),
        (
            _ticker(low_id, as_of=as_of, available_at=earlier, generated_at=earlier),
            _ticker(high_id, as_of=as_of, available_at=earlier, generated_at=earlier),
            high_id,
        ),
    )

    for first, second, expected_id in pairs:
        evaluation = evaluate_requirements(
            (first, second),
            entity_id=TICKER,
            decision_time=DECISION_1420,
            decision_bar=DECISION_BAR,
            profile=profile,
        )
        assert evaluation.selected_states[0].state_id == expected_id


def test_theme_membership_effective_window_selects_older_active_version():
    profile = SnapshotProfile(
        profile_id="membership-window@1",
        rules=(
            FreshnessRule(
                StateType.THEME_MEMBERSHIP,
                True,
                timedelta(hours=96),
                MissingStateAction.REJECT,
            ),
        ),
    )
    active = _theme_membership(
        UUID("00000000-0000-0000-0000-000000001110"),
        effective_from=DECISION_1420 - timedelta(days=1),
    )
    future = _theme_membership(
        UUID("00000000-0000-0000-0000-000000001111"),
        available_at=datetime(2026, 7, 30, 17, 15, tzinfo=UTC),
        effective_from=DECISION_1420 + timedelta(minutes=1),
    )
    expired = _theme_membership(
        UUID("00000000-0000-0000-0000-000000001112"),
        available_at=datetime(2026, 7, 30, 17, 10, tzinfo=UTC),
        effective_from=DECISION_1420 - timedelta(days=1),
        effective_until=DECISION_1420,
    )

    evaluation = evaluate_requirements(
        (active, future, expired),
        entity_id=TICKER,
        decision_time=DECISION_1420,
        decision_bar=DECISION_BAR,
        profile=profile,
    )

    assert evaluation.valid is True
    assert tuple(state.state_id for state in evaluation.selected_states) == (active.state_id,)
    rejected = {item.state_id: item.reason_code for item in evaluation.rejected_candidates}
    assert rejected[future.state_id] == "MEMBERSHIP_NOT_YET_EFFECTIVE"
    assert rejected[expired.state_id] == "MEMBERSHIP_EFFECTIVE_EXPIRED"


@pytest.mark.postgres
def test_builder_applies_causal_gates_and_exact_tie_break_without_as_of(pg_session):
    repo = StateRepository(pg_session)
    decision_time = DECISION_1420
    same_available = datetime(2026, 7, 30, 17, 10, tzinfo=UTC)
    same_generated = datetime(2026, 7, 30, 17, 15, tzinfo=UTC)
    low_id = _ticker(
        UUID("00000000-0000-0000-0000-000000001010"),
        as_of=datetime(2026, 7, 30, 16, 0, tzinfo=UTC),
        available_at=same_available,
        generated_at=same_generated,
    )
    high_id = _ticker(
        UUID("00000000-0000-0000-0000-000000001011"),
        as_of=datetime(2026, 7, 30, 15, 0, tzinfo=UTC),
        available_at=same_available,
        generated_at=same_generated,
    )
    future_bar = _ticker(
        UUID("00000000-0000-0000-0000-000000001012"),
        as_of=datetime(2026, 7, 30, 18, 5, tzinfo=UTC),
        available_at=datetime(2026, 7, 30, 18, 10, tzinfo=UTC),
    )
    expired = _ticker(
        UUID("00000000-0000-0000-0000-000000001013"),
        valid_until=decision_time,
    )
    future = _ticker(
        UUID("00000000-0000-0000-0000-000000001014"),
        available_at=decision_time + timedelta(minutes=1),
        as_of=decision_time,
        generated_at=decision_time + timedelta(minutes=1),
    )
    wrong_entity = _ticker(UUID("00000000-0000-0000-0000-000000001015"), entity_id="MSFT")
    _save(repo, low_id, high_id, future_bar, expired, future, wrong_entity)

    profile = SnapshotProfile(
        profile_id="test_ticker@1",
        rules=(
            FreshnessRule(
                StateType.TICKER,
                True,
                timedelta(hours=24),
                MissingStateAction.REJECT,
            ),
        ),
    )
    snapshot = SnapshotBuilder(repo).build(
        strategy_id="test-strategy",
        entity_id=TICKER,
        decision_time=decision_time,
        decision_bar=DECISION_BAR,
        profile=profile,
    )

    assert snapshot.valid is True
    assert snapshot.ticker_state is not None
    assert snapshot.ticker_state.state_id == high_id.state_id
    rejected = {(item.state_id, item.reason_code) for item in snapshot.rejected_candidates}
    assert (future_bar.state_id, "FUTURE_BAR") in rejected
    assert (expired.state_id, "EXPIRED") in rejected
    assert (wrong_entity.state_id, "WRONG_ENTITY") in rejected
    assert future.state_id not in snapshot.rejected_state_ids
    assert snapshot.requirement_results[0].selected_state_id == high_id.state_id


@pytest.mark.postgres
def test_profiles_require_core_states_and_keep_optional_shadow_state_absent(pg_session):
    repo = StateRepository(pg_session)
    prior_session_market = _required_states()[1]
    after_close_market = _market(
        UUID("00000000-0000-0000-0000-000000001022"),
        as_of=datetime(2026, 7, 30, 20, 30, tzinfo=UTC),
        available_at=datetime(2026, 7, 30, 20, 31, tzinfo=UTC),
    )
    _save(
        repo,
        *_required_states(),
        prior_session_market,
        after_close_market,
        _dealer(UUID("00000000-0000-0000-0000-000000001020")),
    )

    at_1420 = SnapshotBuilder(repo, sector_entity_ids=("technology",)).build(
        strategy_id="meta",
        entity_id=TICKER,
        decision_time=DECISION_1420,
        decision_bar=DECISION_BAR,
        profile=META_4H_1420_PROFILE,
    )
    assert at_1420.valid is True
    assert at_1420.dealer_state is None
    assert any(
        result.state_type is StateType.DEALER and result.status == "MISSING"
        for result in at_1420.requirement_results
    )
    assert at_1420.market_state is not None
    assert at_1420.market_state.state_id == prior_session_market.state_id
    assert at_1420.market_state.available_at < DECISION_1420

    profile_snapshot = SnapshotBuilder(repo, sector_entity_ids=("technology",)).build(
        strategy_id="meta-profile-evidence",
        entity_id=TICKER,
        decision_time=DECISION_1420,
        decision_bar=DECISION_BAR,
        profile=META_4H_1420_PROFILE,
    )
    assert {result.state_type for result in profile_snapshot.requirement_results} == {
        rule.state_type for rule in META_4H_1420_PROFILE.rules
    }
    assert profile_snapshot.theme_states == ()
    assert profile_snapshot.catalyst_pressures == ()

    missing_required = SnapshotBuilder(
        repo,
        portfolio_entity_id="unpublished-paper",
        sector_entity_ids=("technology",),
    ).build(
        strategy_id="meta-missing-required",
        entity_id=TICKER,
        decision_time=DECISION_1420,
        decision_bar=DECISION_BAR,
        profile=META_4H_1420_PROFILE,
    )
    assert missing_required.valid is False
    missing_portfolio = next(
        result
        for result in missing_required.requirement_results
        if result.state_type is StateType.PORTFOLIO
    )
    assert missing_portfolio.status == "MISSING"
    assert missing_portfolio.reason_code == "MISSING_REQUIRED_STATE"


@pytest.mark.postgres
def test_required_expiry_invalidates_and_1620_uses_only_post_1420_dealer_capture(pg_session):
    repo = StateRepository(pg_session)
    states = list(_required_states())
    states[3] = _portfolio(
        UUID("00000000-0000-0000-0000-000000001031"),
        valid_until=DECISION_1420,
    )
    _save(repo, *states)
    invalid = SnapshotBuilder(repo, sector_entity_ids=("technology",)).build(
        strategy_id="meta-expired",
        entity_id=TICKER,
        decision_time=DECISION_1420,
        decision_bar=DECISION_BAR,
        profile=META_4H_1420_PROFILE,
    )
    assert invalid.valid is False
    portfolio_result = next(
        result
        for result in invalid.requirement_results
        if result.state_type is StateType.PORTFOLIO
    )
    assert portfolio_result.status == "STALE"
    assert portfolio_result.reason_code == "EXPIRED"

    _save(
        repo,
        _portfolio(UUID("00000000-0000-0000-0000-000000001032")),
        _dealer(
            UUID("00000000-0000-0000-0000-000000001033"),
            as_of=datetime(2026, 7, 30, 18, 19, tzinfo=UTC),
            available_at=datetime(2026, 7, 30, 18, 20, tzinfo=UTC),
        ),
    )
    _save(
        repo,
        _dealer(
            UUID("00000000-0000-0000-0000-000000001035"),
            as_of=datetime(2026, 7, 30, 18, 10, tzinfo=UTC),
            available_at=datetime(2026, 7, 30, 19, 1, tzinfo=UTC),
        ),
    )
    at_1620_without_post_capture = SnapshotBuilder(
        repo,
        sector_entity_ids=("technology",),
    ).build(
        strategy_id="meta-1620-before",
        entity_id=TICKER,
        decision_time=DECISION_1620,
        decision_bar=datetime(2026, 7, 30, 20, 0, tzinfo=UTC),
        profile=META_4H_1620_PROFILE,
    )
    assert at_1620_without_post_capture.dealer_state is None

    _save(
        repo,
        _dealer(
            UUID("00000000-0000-0000-0000-000000001034"),
            as_of=datetime(2026, 7, 30, 19, 0, tzinfo=UTC),
            available_at=datetime(2026, 7, 30, 19, 1, tzinfo=UTC),
        ),
    )
    at_1620 = SnapshotBuilder(repo, sector_entity_ids=("technology",)).build(
        strategy_id="meta-1620",
        entity_id=TICKER,
        decision_time=DECISION_1620,
        decision_bar=datetime(2026, 7, 30, 20, 0, tzinfo=UTC),
        profile=META_4H_1620_PROFILE,
    )
    assert at_1620.dealer_state is not None
    assert at_1620.dealer_state.state_id == UUID("00000000-0000-0000-0000-000000001034")


@pytest.mark.postgres
def test_builder_is_idempotent_and_persists_evidence_without_owning_transaction(pg_session):
    repo = StateRepository(pg_session)
    _save(repo, *_required_states(), _ticker(UUID("00000000-0000-0000-0000-000000001040"), entity_id="MSFT"))
    builder = SnapshotBuilder(repo, sector_entity_ids=("technology",))
    first = builder.build(
        strategy_id="meta-idempotent",
        entity_id=TICKER,
        decision_time=DECISION_1420,
        decision_bar=DECISION_BAR,
        profile=META_4H_1420_PROFILE,
    )
    second = builder.build(
        strategy_id="meta-idempotent",
        entity_id=TICKER,
        decision_time=DECISION_1420,
        decision_bar=DECISION_BAR,
        profile=META_4H_1420_PROFILE,
    )

    assert second.snapshot_id == first.snapshot_id
    assert canonical_json(second) == canonical_json(first)
    assert second.content_hash == first.content_hash
    assert first.freshness_profile_hash == META_4H_1420_PROFILE.profile_hash
    assert first.decision_session == "2026-07-30"
    assert first.rejected_candidates
    assert all(isinstance(item, RejectedCandidate) for item in first.rejected_candidates)
