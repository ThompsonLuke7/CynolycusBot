from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from core.nervous_system.contracts.context import ContextSnapshot, FreshnessResult, StateRequest
from core.nervous_system.contracts.base import ContractModel, content_hash
from core.nervous_system.contracts.enums import (
    AssetClass,
    DealerRegime,
    Direction,
    MarketRegime,
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
    PortfolioPosition,
    PortfolioState,
    ReadinessState,
    SectorState,
    StateEnvelope,
    ThemeMembership,
    ThemeState,
    TickerState,
)


UTC = timezone.utc
DECISION_TIME = datetime(2026, 7, 30, 18, 20, tzinfo=UTC)


class SnapshotHashMaterial(ContractModel):
    decision_time: datetime
    strategy_id: str
    ticker: str
    freshness_profile: str
    state_hashes: tuple[str, ...]
    stale_inputs: tuple[str, ...]
    missing_inputs: tuple[str, ...]
    data_quality: DataQualitySummary
    config_version: str
    model_versions: tuple[str, ...]
    feature_versions: tuple[str, ...]
    schema_version: int


def expected_snapshot_hash(snapshot: ContextSnapshot) -> str:
    return content_hash(
        SnapshotHashMaterial(
            decision_time=snapshot.decision_time,
            strategy_id=snapshot.strategy_id,
            ticker=snapshot.ticker,
            freshness_profile=snapshot.freshness_profile,
            state_hashes=snapshot.state_hashes,
            stale_inputs=snapshot.stale_inputs,
            missing_inputs=snapshot.missing_inputs,
            data_quality=snapshot.data_quality,
            config_version=snapshot.config_version,
            model_versions=snapshot.model_versions,
            feature_versions=snapshot.feature_versions,
            schema_version=snapshot.schema_version,
        )
    )


def _envelope(state_type: StateType, entity_id: str, *, state_id: UUID | None = None) -> dict[str, Any]:
    available = datetime(2026, 7, 29, 20, 30, tzinfo=UTC)
    return {
        "state_id": state_id or uuid4(),
        "state_type": state_type,
        "entity_id": entity_id,
        "as_of": datetime(2026, 7, 29, 20, 0, tzinfo=UTC),
        "available_at": available,
        "generated_at": available,
        "valid_until": available + timedelta(days=2),
        "source_window_start": datetime(2025, 7, 29, 20, 0, tzinfo=UTC),
        "source_window_end": datetime(2026, 7, 29, 20, 0, tzinfo=UTC),
        "schema_version": 1,
        "producer": f"producer.{state_type.value.lower()}",
        "model_version": "model@1",
        "feature_version": "features@1",
        "config_version": "config@1",
        "lineage_ids": (),
        "data_quality": DataQualitySummary(),
    }


def market_state(*, state_id: UUID | None = None, **updates: Any) -> MarketState:
    payload = _envelope(StateType.MARKET, "US", state_id=state_id)
    payload.update(
        {
            "regime": MarketRegime.NEUTRAL,
            "risk_on_probability": None,
            "risk_off_probability": None,
            "metrics": {"risk_appetite_z": -0.2},
            "transition_probabilities": {},
            "reason_codes": ("MARKET_REGIME_RULES_ONLY",),
        }
    )
    payload.update(updates)
    return MarketState(**payload)


def sector_state(sector_id: str = "technology") -> SectorState:
    payload = _envelope(StateType.SECTOR, sector_id)
    payload.update(
        {
            "sector_id": sector_id,
            "sector_regime": "UNKNOWN",
            "relative_strength": None,
            "breadth": None,
            "momentum": None,
            "volatility": None,
            "rotation_rank": None,
            "rank_change": None,
            "capital_flow_direction": Direction.UNKNOWN,
            "transition_probabilities": {},
        }
    )
    return SectorState(**payload)


def theme_membership(
    theme_id: str,
    ticker: str,
    *,
    state_id: UUID | None = None,
    **updates: Any,
) -> ThemeMembership:
    payload = _envelope(StateType.THEME_MEMBERSHIP, theme_id, state_id=state_id)
    payload.update(
        {
            "ticker": ticker,
            "theme_id": theme_id,
            "weight": 0.75,
            "membership_version": "themes@1",
            "effective_from": datetime(2026, 7, 29, 20, 0, tzinfo=UTC),
            "effective_until": None,
        }
    )
    payload.update(updates)
    return ThemeMembership(**payload)


def theme_state(theme_id: str = "semiconductors") -> ThemeState:
    payload = _envelope(StateType.THEME, theme_id)
    payload.update(
        {
            "theme_id": theme_id,
            "theme_regime": ThemeRegime.HEALTHY,
            "relative_strength": None,
            "breadth": None,
            "momentum": None,
            "distribution_score": None,
            "correlation_score": None,
            "volatility_score": None,
            "catalyst_pressure": None,
            "dealer_fragility": None,
            "leadership_score": None,
            "rotation_rank": None,
            "transition_probabilities": {},
        }
    )
    return ThemeState(**payload)


def ticker_state(ticker: str = "AMD") -> TickerState:
    payload = _envelope(StateType.TICKER, ticker)
    payload.update(
        {
            "ticker": ticker,
            "selected_bar": datetime(2026, 7, 29, 20, 0, tzinfo=UTC),
            "reference_price": 100.0,
            "ticker_setup": TickerSetup.BREAKOUT,
            "trend_state": "UPTREND",
            "relative_strength_state": "LEADING",
            "support_state": "HELD",
            "volume_state": "EXPANDING",
            "reversal_state": "UNKNOWN",
            "breakdown_state": "NONE",
            "theme_alignment": None,
            "market_alignment": None,
            "dealer_alignment": None,
            "metrics": {},
            "transition_probabilities": {},
        }
    )
    return TickerState(**payload)


def catalyst_event(
    event_id: UUID | None = None,
    ticker: str | None = "AMD",
    **updates: Any,
) -> CatalystEvent:
    event_uuid = event_id or uuid4()
    payload = _envelope(StateType.CATALYST_EVENT, str(event_uuid))
    payload.update(
        {
            "event_id": event_uuid,
            "ticker": ticker,
            "event_type": "NEWS",
            "event_time": datetime(2026, 7, 29, 19, 0, tzinfo=UTC),
            "published_at": datetime(2026, 7, 29, 19, 5, tzinfo=UTC),
            "observed_at": datetime(2026, 7, 29, 19, 10, tzinfo=UTC),
            "source": "wire",
            "headline": "A concrete catalyst",
            "channel": "NEWS",
            "relation_confidence": 0.9,
            "is_direct": True,
        }
    )
    payload.update(updates)
    return CatalystEvent(**payload)


def catalyst_pressure(scope_id: str = "AMD", **updates: Any) -> CatalystPressure:
    payload = _envelope(StateType.CATALYST_PRESSURE, scope_id)
    payload.update(
        {
            "scope_type": "TICKER",
            "scope_id": scope_id,
            "channel_scores": {"NEWS": 0.8},
            "aggregate_score": 0.8,
            "event_ids": (),
            "transition_probabilities": {},
        }
    )
    payload.update(updates)
    return CatalystPressure(**payload)


def dealer_state(ticker: str = "AMD") -> DealerState:
    payload = _envelope(StateType.DEALER, ticker)
    payload.update(
        {
            "ticker": ticker,
            "dealer_regime": DealerRegime.NEUTRAL_GAMMA,
            "spot": 100.0,
            "total_gex": None,
            "call_wall": None,
            "put_wall": None,
            "nearest_magnet": None,
            "gamma_flip": None,
            "air_gap_above_score": None,
            "air_gap_below_score": None,
            "pinning_score": None,
            "acceleration_score": None,
            "metrics": {},
        }
    )
    return DealerState(**payload)


def portfolio_state(account_alias: str = "paper") -> PortfolioState:
    payload = _envelope(StateType.PORTFOLIO, account_alias)
    payload.update(
        {
            "account_alias": account_alias,
            "equity": 100_000.0,
            "cash": 50_000.0,
            "buying_power": 75_000.0,
            "day_pl": None,
            "positions": (),
            "open_order_ids": (),
            "broker_observed_at": datetime(2026, 7, 30, 18, 0, tzinfo=UTC),
        }
    )
    return PortfolioState(**payload)


def readiness_state(job: str = "nightly") -> ReadinessState:
    payload = _envelope(StateType.READINESS, job)
    payload.update(
        {
            "job": job,
            "status": "READY",
            "ready": True,
            "completed_at": datetime(2026, 7, 30, 17, 0, tzinfo=UTC),
            "checked_at": datetime(2026, 7, 30, 18, 0, tzinfo=UTC),
            "max_age_hours": 24.0,
            "latest_required_session": "2026-07-30",
            "reason_codes": (),
        }
    )
    return ReadinessState(**payload)


def test_state_rejects_nonexclusive_or_reversed_validity_window():
    state = market_state()
    with pytest.raises(ValidationError):
        state.model_copy(update={"valid_until": state.available_at})


def test_snapshot_embeds_state_and_hash_references():
    state = market_state()
    snapshot = ContextSnapshot.from_states(
        snapshot_id=uuid4(),
        decision_time=DECISION_TIME,
        strategy_id="meta_ranker",
        ticker="AMD",
        states=(state,),
        freshness_profile="meta_4h_1420@1",
    )
    assert snapshot.market_state == state
    assert snapshot.state_ids == (state.state_id,)
    assert snapshot.state_hashes == (content_hash(state, exclude={"state_id"}),)
    assert snapshot.computed_content_hash() == expected_snapshot_hash(snapshot)


def test_snapshot_rejects_naive_decision_time_as_validation_error():
    with pytest.raises(ValidationError, match="timezone-aware"):
        ContextSnapshot.from_states(
            snapshot_id=UUID("00000000-0000-0000-0000-000000000010"),
            decision_time=DECISION_TIME.replace(tzinfo=None),
            strategy_id="meta_ranker",
            ticker="AMD",
            states=(market_state(),),
            freshness_profile="test@1",
        )


def test_snapshot_normalizes_non_utc_decision_time_before_hashing():
    state = market_state(
        state_id=UUID("00000000-0000-0000-0000-000000000013")
    )
    utc_snapshot = ContextSnapshot.from_states(
        snapshot_id=UUID("00000000-0000-0000-0000-000000000014"),
        decision_time=DECISION_TIME,
        strategy_id="meta_ranker",
        ticker="AMD",
        states=(state,),
        freshness_profile="test@1",
    )
    eastern_snapshot = ContextSnapshot.from_states(
        snapshot_id=utc_snapshot.snapshot_id,
        decision_time=datetime(2026, 7, 30, 14, 20, tzinfo=timezone(timedelta(hours=-4))),
        strategy_id="meta_ranker",
        ticker="AMD",
        states=(state,),
        freshness_profile="test@1",
    )

    assert eastern_snapshot.decision_time == DECISION_TIME
    assert eastern_snapshot.content_hash == utc_snapshot.content_hash


def test_snapshot_rejects_state_unavailable_at_decision_time():
    future_available = DECISION_TIME + timedelta(seconds=1)
    state = market_state(available_at=future_available, generated_at=future_available)
    with pytest.raises(ValueError, match="available"):
        ContextSnapshot.from_states(
            snapshot_id=uuid4(),
            decision_time=DECISION_TIME,
            strategy_id="meta_ranker",
            ticker="AMD",
            states=(state,),
            freshness_profile="test@1",
        )


def test_snapshot_rejects_state_at_exclusive_validity_boundary():
    state = market_state(valid_until=DECISION_TIME)
    with pytest.raises(ValueError, match="expired"):
        ContextSnapshot.from_states(
            snapshot_id=uuid4(),
            decision_time=DECISION_TIME,
            strategy_id="meta_ranker",
            ticker="AMD",
            states=(state,),
            freshness_profile="test@1",
        )


def test_snapshot_rejects_ambiguous_duplicate_singleton_state():
    first = market_state(state_id=UUID("00000000-0000-0000-0000-000000000001"))
    second = market_state(state_id=UUID("00000000-0000-0000-0000-000000000002"))
    with pytest.raises(ValueError, match="duplicate"):
        ContextSnapshot.from_states(
            snapshot_id=uuid4(),
            decision_time=DECISION_TIME,
            strategy_id="meta_ranker",
            ticker="AMD",
            states=(first, second),
            freshness_profile="test@1",
        )


def test_snapshot_dispatches_all_concrete_state_types_and_preserves_order():
    event_one = catalyst_event(event_id=UUID("00000000-0000-0000-0000-000000000011"))
    event_two = catalyst_event(event_id=UUID("00000000-0000-0000-0000-000000000012"))
    states = (
        market_state(),
        sector_state("financials"),
        theme_membership("semiconductors", "AMD"),
        theme_state("semiconductors"),
        ticker_state(),
        event_one,
        event_two,
        catalyst_pressure(),
        dealer_state(),
        portfolio_state(),
        readiness_state(),
    )
    snapshot = ContextSnapshot.from_states(
        snapshot_id=uuid4(),
        decision_time=DECISION_TIME,
        strategy_id="meta_ranker",
        ticker="AMD",
        states=states,
        freshness_profile="test@1",
    )
    assert snapshot.sector_states == (states[1],)
    assert snapshot.theme_memberships == (states[2],)
    assert snapshot.catalyst_events == (event_one, event_two)
    expected_state_ids = {
        state.state_id for state in states if isinstance(state, StateEnvelope)
    }
    assert set(snapshot.state_ids) == expected_state_ids
    assert len(snapshot.state_hashes) == len(states)


def test_snapshot_rejects_tampered_content_hash_on_model_copy():
    state = market_state(state_id=UUID("00000000-0000-0000-0000-000000000021"))
    snapshot = ContextSnapshot.from_states(
        snapshot_id=UUID("00000000-0000-0000-0000-000000000022"),
        decision_time=DECISION_TIME,
        strategy_id="meta_ranker",
        ticker="AMD",
        states=(state,),
        freshness_profile="test@1",
    )
    with pytest.raises(ValidationError, match="content_hash"):
        snapshot.model_copy(update={"content_hash": "tampered"})

    assert snapshot.content_hash == expected_snapshot_hash(snapshot)


def test_snapshot_rejects_tampered_content_hash_from_json():
    snapshot = ContextSnapshot.from_states(
        snapshot_id=UUID("00000000-0000-0000-0000-000000000023"),
        decision_time=DECISION_TIME,
        strategy_id="meta_ranker",
        ticker="AMD",
        states=(market_state(state_id=UUID("00000000-0000-0000-0000-000000000024")),),
        freshness_profile="test@1",
    )
    tampered_json = snapshot.model_dump_json().replace(snapshot.content_hash, "tampered")

    with pytest.raises(ValidationError, match="content_hash"):
        ContextSnapshot.model_validate_json(tampered_json)


def test_equivalent_states_with_distinct_ids_have_equal_snapshot_content_hashes():
    first_state = market_state(
        state_id=UUID("00000000-0000-0000-0000-0000000000d1")
    )
    second_state = market_state(
        state_id=UUID("00000000-0000-0000-0000-0000000000d2")
    )
    first_snapshot = ContextSnapshot.from_states(
        snapshot_id=UUID("00000000-0000-0000-0000-0000000000d3"),
        decision_time=DECISION_TIME,
        strategy_id="meta_ranker",
        ticker="AMD",
        states=(first_state,),
        freshness_profile="test@1",
    )
    second_snapshot = ContextSnapshot.from_states(
        snapshot_id=UUID("00000000-0000-0000-0000-0000000000d4"),
        decision_time=DECISION_TIME,
        strategy_id="meta_ranker",
        ticker="AMD",
        states=(second_state,),
        freshness_profile="test@1",
    )
    assert first_state.state_id != second_state.state_id
    assert first_snapshot.snapshot_id != second_snapshot.snapshot_id
    assert first_snapshot.state_hashes == second_snapshot.state_hashes
    assert first_snapshot.content_hash == second_snapshot.content_hash
    assert first_snapshot.content_hash == expected_snapshot_hash(first_snapshot)


def test_snapshot_rejects_tampered_parallel_state_id_or_hash():
    state = market_state(
        state_id=UUID("00000000-0000-0000-0000-0000000000e1")
    )
    snapshot = ContextSnapshot.from_states(
        snapshot_id=UUID("00000000-0000-0000-0000-0000000000e2"),
        decision_time=DECISION_TIME,
        strategy_id="meta_ranker",
        ticker="AMD",
        states=(state,),
        freshness_profile="test@1",
    )
    with pytest.raises(ValidationError, match="state_ids"):
        snapshot.model_copy(
            update={
                "state_ids": (
                    UUID("00000000-0000-0000-0000-0000000000e3"),
                )
            }
        )
    with pytest.raises(ValidationError, match="state_hashes"):
        snapshot.model_copy(update={"state_hashes": ("tampered",)})
    with pytest.raises(ValidationError, match="state_hashes"):
        snapshot.model_copy(
            update={"market_state": state.model_copy(update={"metrics": {"changed": 1.0}})}
        )


def test_changing_state_content_changes_snapshot_content_hash():
    state = market_state(
        state_id=UUID("00000000-0000-0000-0000-0000000000f1")
    )
    changed_state = state.model_copy(
        update={"metrics": {"risk_appetite_z": 0.4}}
    )
    first_snapshot = ContextSnapshot.from_states(
        snapshot_id=UUID("00000000-0000-0000-0000-0000000000f2"),
        decision_time=DECISION_TIME,
        strategy_id="meta_ranker",
        ticker="AMD",
        states=(state,),
        freshness_profile="test@1",
    )
    changed_snapshot = ContextSnapshot.from_states(
        snapshot_id=first_snapshot.snapshot_id,
        decision_time=DECISION_TIME,
        strategy_id="meta_ranker",
        ticker="AMD",
        states=(changed_state,),
        freshness_profile="test@1",
    )
    assert first_snapshot.state_hashes != changed_snapshot.state_hashes
    assert first_snapshot.content_hash != changed_snapshot.content_hash


def test_state_and_snapshot_round_trip_without_losing_order_or_hash():
    state = market_state(state_id=UUID("00000000-0000-0000-0000-000000000031"))
    snapshot = ContextSnapshot.from_states(
        snapshot_id=UUID("00000000-0000-0000-0000-000000000032"),
        decision_time=DECISION_TIME,
        strategy_id="meta_ranker",
        ticker="AMD",
        states=(state,),
        freshness_profile="test@1",
    )
    restored = ContextSnapshot.model_validate_json(snapshot.model_dump_json())
    assert restored == snapshot
    assert restored.state_ids == (state.state_id,)
    assert restored.content_hash == snapshot.content_hash


def test_state_defaults_do_not_share_mutable_mapping_storage():
    first = market_state(metrics={})
    second = market_state(metrics={})
    assert first.metrics is not second.metrics


def test_theme_membership_is_an_envelope_and_state_ids_match_hashes():
    membership = theme_membership("semiconductors", "AMD")
    snapshot = ContextSnapshot.from_states(
        snapshot_id=UUID("00000000-0000-0000-0000-000000000061"),
        decision_time=DECISION_TIME,
        strategy_id="meta_ranker",
        ticker="AMD",
        states=(membership,),
        freshness_profile="test@1",
    )
    assert isinstance(membership, StateEnvelope)
    assert membership.state_type is StateType.THEME_MEMBERSHIP
    assert snapshot.state_ids == (membership.state_id,)
    assert len(snapshot.state_ids) == len(snapshot.state_hashes) == 1


def test_identity_excluded_state_hashes_ignore_state_uuid_but_keep_ids_distinct():
    first = market_state(state_id=UUID("00000000-0000-0000-0000-000000000071"))
    second = market_state(state_id=UUID("00000000-0000-0000-0000-000000000072"))
    assert first.state_id != second.state_id
    assert content_hash(first, exclude={"state_id"}) == content_hash(
        second, exclude={"state_id"}
    )
    assert content_hash(first) != content_hash(second)


def test_snapshot_hash_ignores_snapshot_uuid_but_keeps_snapshot_ids_distinct():
    state = market_state(state_id=UUID("00000000-0000-0000-0000-000000000081"))
    first = ContextSnapshot.from_states(
        snapshot_id=UUID("00000000-0000-0000-0000-000000000082"),
        decision_time=DECISION_TIME,
        strategy_id="meta_ranker",
        ticker="AMD",
        states=(state,),
        freshness_profile="test@1",
    )
    second = ContextSnapshot.from_states(
        snapshot_id=UUID("00000000-0000-0000-0000-000000000083"),
        decision_time=DECISION_TIME,
        strategy_id="meta_ranker",
        ticker="AMD",
        states=(state,),
        freshness_profile="test@1",
    )
    assert first.snapshot_id != second.snapshot_id
    assert first.content_hash == second.content_hash


@pytest.mark.parametrize(
    ("factory", "payload_field", "payload_value"),
    [
        (sector_state, "sector_id", "wrong-sector"),
        (theme_state, "theme_id", "wrong-theme"),
        (ticker_state, "ticker", "MSFT"),
        (dealer_state, "ticker", "MSFT"),
        (portfolio_state, "account_alias", "other-account"),
        (readiness_state, "job", "other-job"),
    ],
)
def test_payload_identity_must_match_state_envelope(factory, payload_field, payload_value):
    state = factory()
    with pytest.raises(ValidationError, match="entity_id"):
        state.model_copy(update={payload_field: payload_value})


def test_catalyst_event_observation_time_is_causal():
    with pytest.raises(ValidationError, match="observed_at"):
        catalyst_event(
            observed_at=datetime(2026, 7, 30, 20, 31, tzinfo=UTC),
            available_at=datetime(2026, 7, 30, 20, 30, tzinfo=UTC),
            generated_at=datetime(2026, 7, 30, 20, 31, tzinfo=UTC),
        )


def test_catalyst_event_identity_matches_event_id_and_is_validated_on_update():
    event = catalyst_event(
        event_id=UUID("00000000-0000-0000-0000-0000000000e4")
    )
    assert event.entity_id == str(event.event_id)
    with pytest.raises(ValidationError, match="event_id"):
        event.model_copy(
            update={
                "event_id": UUID("00000000-0000-0000-0000-0000000000e5"),
            }
        )


def test_catalyst_event_rejects_published_after_observed():
    with pytest.raises(ValidationError, match="published_at"):
        catalyst_event(
            published_at=datetime(2026, 7, 29, 19, 11, tzinfo=UTC),
            observed_at=datetime(2026, 7, 29, 19, 10, tzinfo=UTC),
        )


def test_catalyst_event_raw_directional_score_is_typed_and_json_round_trips():
    event = catalyst_event(raw_score=4.5)
    restored = CatalystEvent.model_validate_json(event.model_dump_json())

    assert event.raw_score == 4.5
    assert restored == event


def test_future_scheduled_catalyst_is_allowed_and_is_direct_is_optional():
    state = catalyst_event(
        event_time=datetime(2026, 8, 15, 13, 30, tzinfo=UTC),
        is_direct=None,
    )
    assert state.event_time > state.observed_at
    assert state.is_direct is None


def test_snapshot_rejects_ticker_mismatch_for_ticker_scoped_states():
    for state in (
        ticker_state("MSFT"),
        dealer_state("MSFT"),
        theme_membership("semiconductors", "MSFT"),
        catalyst_event(ticker="MSFT"),
        catalyst_pressure(scope_id="MSFT"),
    ):
        with pytest.raises(ValueError, match="ticker"):
            ContextSnapshot.from_states(
                snapshot_id=uuid4(),
                decision_time=DECISION_TIME,
                strategy_id="meta_ranker",
                ticker="AMD",
                states=(state,),
                freshness_profile="test@1",
            )


def test_reversed_state_input_has_same_sorted_snapshot_content():
    states = (
        market_state(state_id=UUID("00000000-0000-0000-0000-000000000091")),
        sector_state("financials").model_copy(
            update={"state_id": UUID("00000000-0000-0000-0000-000000000092")}
        ),
        theme_membership(
            "semiconductors",
            "AMD",
            state_id=UUID("00000000-0000-0000-0000-000000000093"),
        ),
        theme_state("semiconductors").model_copy(
            update={"state_id": UUID("00000000-0000-0000-0000-000000000094")}
        ),
        ticker_state().model_copy(
            update={"state_id": UUID("00000000-0000-0000-0000-000000000095")}
        ),
        catalyst_event(
            UUID("00000000-0000-0000-0000-000000000096"),
            event_time=datetime(2026, 7, 29, 21, 0, tzinfo=UTC),
        ),
        catalyst_event(
            UUID("00000000-0000-0000-0000-000000000097"),
            event_time=datetime(2026, 7, 29, 20, 0, tzinfo=UTC),
        ),
        catalyst_pressure().model_copy(
            update={"state_id": UUID("00000000-0000-0000-0000-000000000098")}
        ),
        dealer_state().model_copy(
            update={"state_id": UUID("00000000-0000-0000-0000-000000000099")}
        ),
        portfolio_state().model_copy(
            update={"state_id": UUID("00000000-0000-0000-0000-00000000009a")}
        ),
        readiness_state().model_copy(
            update={"state_id": UUID("00000000-0000-0000-0000-00000000009b")}
        ),
    )
    forward = ContextSnapshot.from_states(
        snapshot_id=UUID("00000000-0000-0000-0000-00000000009c"),
        decision_time=DECISION_TIME,
        strategy_id="meta_ranker",
        ticker="AMD",
        states=states,
        freshness_profile="test@1",
    )
    reversed_snapshot = ContextSnapshot.from_states(
        snapshot_id=forward.snapshot_id,
        decision_time=DECISION_TIME,
        strategy_id="meta_ranker",
        ticker="AMD",
        states=tuple(reversed(states)),
        freshness_profile="test@1",
    )
    assert forward.state_ids == reversed_snapshot.state_ids
    assert forward.state_hashes == reversed_snapshot.state_hashes
    assert forward.catalyst_events[0].event_time < forward.catalyst_events[1].event_time
    assert forward.content_hash == reversed_snapshot.content_hash


def test_snapshot_rejects_duplicate_state_ids_across_classes():
    state_id = UUID("00000000-0000-0000-0000-0000000000a1")
    with pytest.raises(ValueError, match="state_id"):
        ContextSnapshot.from_states(
            snapshot_id=uuid4(),
            decision_time=DECISION_TIME,
            strategy_id="meta_ranker",
            ticker="AMD",
            states=(market_state(state_id=state_id), dealer_state().model_copy(update={"state_id": state_id})),
            freshness_profile="test@1",
        )


def test_snapshot_rejects_duplicate_effective_collection_selection_but_allows_events():
    duplicate_membership = (
        theme_membership("semiconductors", "AMD", state_id=UUID("00000000-0000-0000-0000-0000000000b1")),
        theme_membership("semiconductors", "AMD", state_id=UUID("00000000-0000-0000-0000-0000000000b2")),
    )
    with pytest.raises(ValueError, match="duplicate"):
        ContextSnapshot.from_states(
            snapshot_id=uuid4(),
            decision_time=DECISION_TIME,
            strategy_id="meta_ranker",
            ticker="AMD",
            states=duplicate_membership,
            freshness_profile="test@1",
        )

    snapshot = ContextSnapshot.from_states(
        snapshot_id=uuid4(),
        decision_time=DECISION_TIME,
        strategy_id="meta_ranker",
        ticker="AMD",
        states=(
            catalyst_event(UUID("00000000-0000-0000-0000-0000000000c1")),
            catalyst_event(UUID("00000000-0000-0000-0000-0000000000c2")),
        ),
        freshness_profile="test@1",
    )
    assert len(snapshot.catalyst_events) == 2


def test_mapping_payloads_are_deep_frozen_and_json_round_trip():
    state = market_state(
        metrics={"z": -0.2},
        transition_probabilities={"RISK_ON": 0.5},
    )
    pressure = catalyst_pressure(channel_scores={"NEWS": 0.8})
    for mapping in (state.metrics, state.transition_probabilities, pressure.channel_scores):
        with pytest.raises(TypeError):
            mapping["new"] = 1.0

    restored = MarketState.model_validate_json(state.model_dump_json())
    assert restored == state
    with pytest.raises(TypeError):
        restored.metrics["new"] = 1.0


def test_state_request_and_freshness_result_have_explicit_contract_fields():
    request = StateRequest(
        state_type=StateType.TICKER,
        entity_id="AMD",
        required=True,
        bar_bound=datetime(2026, 7, 30, 18, 0, tzinfo=UTC),
    )
    result = FreshnessResult(
        state_type=StateType.TICKER,
        entity_id="AMD",
        required=True,
        status="FRESH",
        selected_state_id=UUID("00000000-0000-0000-0000-000000000041"),
        age_seconds=120.0,
        max_age_seconds=14_400.0,
        reason_code="WITHIN_PROFILE",
    )
    assert request.bar_bound == datetime(2026, 7, 30, 18, 0, tzinfo=UTC)
    assert result.status == "FRESH"


def test_portfolio_position_uses_strong_asset_and_numeric_types():
    position = PortfolioPosition(
        broker_position_id="bp-1",
        symbol="AMD",
        underlying="AMD",
        asset_class=AssetClass.EQUITY,
        quantity=10.0,
        average_entry_price=100.0,
        current_price=101.0,
        market_value=1_010.0,
        strategy_id=None,
        ownership_status="UNASSIGNED",
    )
    assert position.asset_class is AssetClass.EQUITY
    with pytest.raises(ValidationError):
        PortfolioPosition(
            broker_position_id="bp-2",
            symbol="AMD",
            underlying="AMD",
            asset_class=AssetClass.EQUITY,
            quantity=float("nan"),
            ownership_status="UNASSIGNED",
        )
