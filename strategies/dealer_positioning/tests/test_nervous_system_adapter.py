from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import math

import pytest

from core.nervous_system.contracts.base import content_hash
from core.nervous_system.contracts.context import ContextSnapshot
from core.nervous_system.contracts.enums import DataQualitySeverity, DealerRegime
from core.nervous_system.contracts.quality import LineageRef
from strategies.dealer_positioning import nervous_system_adapter as dealer_adapter
from strategies.dealer_positioning.nervous_system_adapter import adapt_dealer_state


UTC = timezone.utc
CAPTURED_AT = datetime(2026, 7, 30, 19, 45, tzinfo=UTC)  # 15:45 ET
LEVELS_AVAILABLE_AT = datetime(2026, 7, 30, 20, 0, tzinfo=UTC)
RANKING_AVAILABLE_AT = datetime(2026, 7, 30, 20, 10, tzinfo=UTC)
VALID_UNTIL = datetime(2026, 7, 31, 20, 0, tzinfo=UTC)
LINEAGE = (
    LineageRef(
        source_id="dealer-capture",
        content_hash="b" * 64,
        record_locator="dealer_level_summary:SPY:20260730:daily_week",
    ),
)
RANKING_LINEAGE = (
    LineageRef(
        source_id="dealer-ranking",
        content_hash="c" * 64,
        record_locator="dealer_swing_rankings:SPY:20260730",
    ),
)


def _snapshot(**updates: object) -> dict[str, object]:
    """Representative output of capture_historical_snapshots._summary_row."""

    row: dict[str, object] = {
        "symbol": "SPY",
        "scope": "daily_week",
        "scope_label": "daily_week",
        "snapshot_date": "2026-07-30",
        "captured_at": CAPTURED_AT.isoformat(),
        "ref_date_et": "2026-07-30",
        "chain_query_date_et": "2026-07-30",
        "fetched_at": CAPTURED_AT.isoformat(),
        # GammaLevels.timestamp is intentionally not the availability source.
        "timestamp": "2026-07-30T19:45:00+00:00",
        "spot": 640.0,
        "spot_price": 640.0,
        "expirations": '["2026-07-31"]',
        "available_expirations": '["2026-07-31"]',
        "strike_window_pct": 0.50,
        "row_count": 120,
        "total_gex": -1200.0,
        "call_wall": 650.0,
        "put_wall": 625.0,
        "nearest_magnet": 640.0,
        "magnet": 640.0,
        "gamma_flip": 638.0,
        "air_gap_above_score": 0.35,
        "air_gap_below_score": 0.55,
        "total_option_oi": 125_000.0,
        "total_option_volume": 18_000.0,
        "liquidity_pass_reason": "adv",
        "pct_to_call_wall": 0.015625,
        "pct_to_put_wall": -0.0234375,
        "pct_to_magnet": 0.0,
        "vacuum_score": 0.35,
        "pinning_score": 0.8,
        # Produced by _matrix_features, but not calibrated as a direction.
        "dealer_bias": 2500.0,
        "dealer_imbalance": -0.40,
    }
    row.update(updates)
    return row


def _dynamics(**updates: object) -> dict[str, object]:
    row: dict[str, object] = {
        "symbol": "SPY",
        "scope": "daily_week",
        "snapshot_date": "2026-07-30",
        "captured_at": CAPTURED_AT.isoformat(),
        "available_at": LEVELS_AVAILABLE_AT,
        "wall_change_1d": 2.0,
        "gex_concentration_change": -0.05,
    }
    row.update(updates)
    return row


def _ranking(**updates: object) -> dict[str, object]:
    """Representative aggregate output of build_dealer_rankings.build_rankings."""

    row: dict[str, object] = {
        "symbol": "SPY",
        "snapshot_date": "2026-07-30",
        "captured_at": CAPTURED_AT.isoformat(),
        "scope_count": 3,
        "scope_weight_sum": 1.0,
        "dealer_swing_potential_score": 72.5,
        "dealer_direction_bias": 0.35,
        "dealer_change_intensity_score": 44.0,
        "dealer_change_direction_bias": 0.15,
        "max_scope_score": 81.0,
        "max_change_scope_score": 55.0,
        "avg_vacuum_component": 0.60,
        "avg_sparse_gamma_component": 0.70,
        "avg_pinning_room_component": 0.45,
        "avg_dealer_imbalance": -0.40,
        "dealer_direction": "bullish",
        "dealer_change_direction": "neutral",
        "dealer_swing_rank": 4,
        "scope_scores_json": json.dumps(
            [
                {"scope": "daily_week", "scope_swing_potential_score": 81.0},
                {"scope": "through_month", "scope_swing_potential_score": 70.0},
                {"scope": "two_months", "scope_swing_potential_score": 60.0},
            ],
            sort_keys=True,
        ),
    }
    row.update(updates)
    return row


def _snapshot_at(state, decision_time: datetime) -> ContextSnapshot:
    return ContextSnapshot.from_states(
        snapshot_id=__import__("uuid").uuid4(),
        decision_time=decision_time,
        strategy_id="meta_ranker",
        ticker="SPY",
        states=(state,),
        freshness_profile="qa",
    )


def test_dealer_capture_and_dynamics_use_causal_availability() -> None:
    state = adapt_dealer_state(
        _snapshot(),
        _dynamics(),
        captured_at=CAPTURED_AT,
        valid_until=VALID_UNTIL,
        lineage=LINEAGE,
    )

    assert state.available_at == LEVELS_AVAILABLE_AT
    assert state.as_of == CAPTURED_AT
    assert state.metrics["wall_change_1d"] == 2.0
    assert state.metrics["dealer_bias"] == 2500.0
    assert state.metrics["vacuum_score"] == 0.35
    assert state.data_quality.is_usable


def test_1545_capture_is_eligible_at_1620_but_not_1420() -> None:
    state = adapt_dealer_state(
        _snapshot(),
        None,
        captured_at=CAPTURED_AT,
        valid_until=VALID_UNTIL,
        lineage=LINEAGE,
    )

    with pytest.raises(ValueError, match="unavailable"):
        _snapshot_at(state, datetime(2026, 7, 30, 18, 20, tzinfo=UTC))  # 14:20 ET

    snapshot = _snapshot_at(state, datetime(2026, 7, 30, 20, 20, tzinfo=UTC))  # 16:20 ET
    assert snapshot.dealer_state is not None
    assert snapshot.dealer_state.available_at == CAPTURED_AT


def test_gamma_levels_timestamp_is_never_used_without_explicit_capture_metadata() -> None:
    ambiguous = _snapshot()
    ambiguous.pop("captured_at")
    with pytest.raises(ValueError, match="captured_at"):
        adapt_dealer_state(
            ambiguous,
            None,
            captured_at=None,  # type: ignore[arg-type]
            valid_until=VALID_UNTIL,
            lineage=LINEAGE,
        )

    state = adapt_dealer_state(
        _snapshot(timestamp="2030-01-01T00:00:00+00:00"),
        None,
        captured_at=CAPTURED_AT,
        valid_until=VALID_UNTIL,
        lineage=LINEAGE,
    )
    assert state.available_at == CAPTURED_AT


def test_dealer_metrics_are_not_relabelled_as_probabilities() -> None:
    state = adapt_dealer_state(
        _snapshot(),
        _dynamics(gex_concentration_change=4.5),
        captured_at=CAPTURED_AT,
        valid_until=VALID_UNTIL,
        lineage=LINEAGE,
    )

    assert state.metrics["gex_concentration_change"] == 4.5
    assert not hasattr(state, "transition_probabilities")


def test_stale_and_print_sparse_data_are_quality_warnings() -> None:
    state = adapt_dealer_state(
        _snapshot(stale_days=5, print_count=2, expected_prints=100),
        None,
        captured_at=CAPTURED_AT,
        valid_until=VALID_UNTIL,
        lineage=LINEAGE,
    )

    codes = {issue.code for issue in state.data_quality.issues}
    assert "STALE_OPTION_DATA" in codes
    assert "PRINT_SPARSE_OPTION_DATA" in codes
    assert state.data_quality.is_usable


def test_dynamics_without_explicit_availability_are_rejected() -> None:
    with pytest.raises(ValueError, match="available_at"):
        adapt_dealer_state(
            _snapshot(),
            _dynamics(available_at=None),
            captured_at=CAPTURED_AT,
            valid_until=VALID_UNTIL,
            lineage=LINEAGE,
        )


def test_dealer_state_identity_versions_and_lineage_are_deterministic() -> None:
    first = adapt_dealer_state(
        _snapshot(),
        _dynamics(),
        captured_at=CAPTURED_AT,
        valid_until=VALID_UNTIL,
        lineage=LINEAGE,
    )
    second = adapt_dealer_state(
        _snapshot(),
        _dynamics(),
        captured_at=CAPTURED_AT,
        valid_until=VALID_UNTIL,
        lineage=LINEAGE,
    )

    assert first.state_id == second.state_id
    assert content_hash(first, exclude={"state_id"}) == content_hash(
        second, exclude={"state_id"}
    )
    assert first.producer == "strategies.dealer_positioning"
    assert first.model_version
    assert first.feature_version
    assert first.config_version
    assert json.loads(first.lineage_ids[0]) == {
        "content_hash": "b" * 64,
        "record_locator": "dealer_level_summary:SPY:20260730:daily_week",
        "source_id": "dealer-capture",
    }


@pytest.mark.parametrize("field", ["spot", "dealer_bias", "total_gex"])
def test_nonfinite_numeric_inputs_are_rejected(field: str) -> None:
    with pytest.raises(ValueError, match="finite"):
        adapt_dealer_state(
            _snapshot(**{field: math.nan}),
            None,
            captured_at=CAPTURED_AT,
            valid_until=VALID_UNTIL,
            lineage=LINEAGE,
        )


def test_naive_or_conflicting_capture_timestamps_are_rejected() -> None:
    with pytest.raises(ValueError, match="captured_at.*timezone-aware"):
        adapt_dealer_state(
            _snapshot(),
            None,
            captured_at=CAPTURED_AT.replace(tzinfo=None),  # type: ignore[arg-type]
            valid_until=VALID_UNTIL,
            lineage=LINEAGE,
        )

    with pytest.raises(ValueError, match="captured_at"):
        adapt_dealer_state(
            _snapshot(captured_at=CAPTURED_AT + timedelta(minutes=1)),
            None,
            captured_at=CAPTURED_AT,
            valid_until=VALID_UNTIL,
            lineage=LINEAGE,
        )


def test_dynamics_availability_must_be_explicit_and_timezone_aware() -> None:
    with pytest.raises(ValueError, match="available_at.*timezone-aware"):
        adapt_dealer_state(
            _snapshot(),
            _dynamics(available_at=LEVELS_AVAILABLE_AT.replace(tzinfo=None)),
            captured_at=CAPTURED_AT,
            valid_until=VALID_UNTIL,
            lineage=LINEAGE,
        )


@pytest.mark.parametrize(
    ("scope", "weight"),
    [
        ("daily_week", 0.45),
        ("through_month", 0.35),
        ("two_months", 0.20),
    ],
)
def test_explicit_canonical_scope_is_preserved(
    scope: str, weight: float
) -> None:
    state = adapt_dealer_state(
        _snapshot(scope=scope),
        None,
        captured_at=CAPTURED_AT,
        valid_until=VALID_UNTIL,
        lineage=LINEAGE,
    )

    assert state.metrics[f"selected_scope_{scope}"] == 1.0
    assert state.metrics["selected_scope_weight"] == weight
    scope_issues = [
        issue
        for issue in state.data_quality.issues
        if issue.code == "DEALER_SCOPE_SELECTED"
    ]
    assert len(scope_issues) == 1
    assert scope in scope_issues[0].message


def test_scope_only_variants_cannot_collapse() -> None:
    daily_week = adapt_dealer_state(
        _snapshot(scope="daily_week"),
        None,
        captured_at=CAPTURED_AT,
        valid_until=VALID_UNTIL,
        lineage=LINEAGE,
    )
    through_month = adapt_dealer_state(
        _snapshot(scope="through_month"),
        None,
        captured_at=CAPTURED_AT,
        valid_until=VALID_UNTIL,
        lineage=LINEAGE,
    )

    assert daily_week.state_id != through_month.state_id
    assert content_hash(daily_week, exclude={"state_id"}) != content_hash(
        through_month, exclude={"state_id"}
    )


@pytest.mark.parametrize("scope", [None, "", "monthly", ("daily_week", "through_month")])
def test_missing_ambiguous_or_noncanonical_scope_is_rejected(scope: object) -> None:
    with pytest.raises(ValueError, match="scope"):
        adapt_dealer_state(
            _snapshot(scope=scope),
            None,
            captured_at=CAPTURED_AT,
            valid_until=VALID_UNTIL,
            lineage=LINEAGE,
        )


def test_dynamics_scope_must_match_selected_snapshot_scope() -> None:
    with pytest.raises(ValueError, match="scope"):
        adapt_dealer_state(
            _snapshot(scope="daily_week"),
            _dynamics(scope="through_month"),
            captured_at=CAPTURED_AT,
            valid_until=VALID_UNTIL,
            lineage=LINEAGE,
        )


@pytest.mark.parametrize(
    ("direction", "expected"),
    [
        ("bullish", DealerRegime.UPSIDE_ACCELERATION),
        ("neutral", DealerRegime.NEUTRAL_GAMMA),
        ("bearish", DealerRegime.DOWNSIDE_ACCELERATION),
    ],
)
def test_actual_producer_dealer_direction_maps_to_regime(
    direction: str, expected: DealerRegime
) -> None:
    state = adapt_dealer_state(
        _snapshot(dealer_direction=direction),
        None,
        captured_at=CAPTURED_AT,
        valid_until=VALID_UNTIL,
        lineage=LINEAGE,
    )

    assert state.dealer_regime is expected


@pytest.mark.parametrize("field", ["dealer_regime", "regime"])
def test_explicit_contract_regime_is_preserved(field: str) -> None:
    row = _snapshot()
    row[field] = DealerRegime.PINNING.value

    state = adapt_dealer_state(
        row,
        None,
        captured_at=CAPTURED_AT,
        valid_until=VALID_UNTIL,
        lineage=LINEAGE,
    )

    assert state.dealer_regime is DealerRegime.PINNING


@pytest.mark.parametrize(
    "updates",
    [
        {"dealer_direction": "sideways"},
        {"dealer_direction": "bullish", "regime": DealerRegime.SHORT_GAMMA.value},
        {"dealer_direction": "bullish", "dealer_regime": DealerRegime.UNKNOWN.value},
    ],
)
def test_unknown_or_conflicting_regime_evidence_fails_closed(
    updates: dict[str, object]
) -> None:
    with pytest.raises(ValueError, match="dealer.*(direction|regime)|regime"):
        adapt_dealer_state(
            _snapshot(**updates),
            None,
            captured_at=CAPTURED_AT,
            valid_until=VALID_UNTIL,
            lineage=LINEAGE,
        )


def test_actual_raw_capture_without_direction_adapts_unknown_with_warning() -> None:
    state = adapt_dealer_state(
        _snapshot(),
        None,
        captured_at=CAPTURED_AT,
        valid_until=VALID_UNTIL,
        lineage=LINEAGE,
    )

    assert state.dealer_regime is DealerRegime.UNKNOWN
    issues = [
        issue
        for issue in state.data_quality.issues
        if issue.code == "DEALER_DIRECTION_UNCALIBRATED"
    ]
    assert len(issues) == 1
    assert issues[0].severity is DataQualitySeverity.WARNING
    assert "calibrated" in issues[0].message
    assert state.data_quality.is_usable


def test_raw_dealer_bias_is_metric_not_direction_evidence() -> None:
    state = adapt_dealer_state(
        _snapshot(dealer_bias=1_000_000.0, dealer_imbalance=1.0),
        None,
        captured_at=CAPTURED_AT,
        valid_until=VALID_UNTIL,
        lineage=LINEAGE,
    )

    assert state.dealer_regime is DealerRegime.UNKNOWN
    assert state.metrics["dealer_bias"] == 1_000_000.0
    assert state.metrics["dealer_imbalance"] == 1.0


def test_direction_only_variants_have_distinct_state_ids() -> None:
    bullish = adapt_dealer_state(
        _snapshot(dealer_direction="bullish"),
        None,
        captured_at=CAPTURED_AT,
        valid_until=VALID_UNTIL,
        lineage=LINEAGE,
    )
    bearish = adapt_dealer_state(
        _snapshot(dealer_direction="bearish"),
        None,
        captured_at=CAPTURED_AT,
        valid_until=VALID_UNTIL,
        lineage=LINEAGE,
    )

    assert bullish.state_id != bearish.state_id


def test_aggregate_ranking_row_is_not_a_raw_scope_capture() -> None:
    with pytest.raises(ValueError, match="scope"):
        adapt_dealer_state(
            _ranking(),
            None,
            captured_at=CAPTURED_AT,
            valid_until=VALID_UNTIL,
            lineage=RANKING_LINEAGE,
        )


def test_honest_join_combines_raw_scope_and_aggregate_direction_evidence() -> None:
    state = dealer_adapter.adapt_dealer_state_with_ranking(
        _snapshot(),
        None,
        _ranking(),
        captured_at=CAPTURED_AT,
        ranking_available_at=RANKING_AVAILABLE_AT,
        valid_until=VALID_UNTIL,
        lineage=LINEAGE,
        ranking_lineage=RANKING_LINEAGE,
    )

    assert state.ticker == "SPY"
    assert state.dealer_regime is DealerRegime.UPSIDE_ACCELERATION
    assert state.available_at == RANKING_AVAILABLE_AT
    assert state.metrics["dealer_bias"] == 2500.0
    assert state.metrics["dealer_swing_potential_score"] == 72.5
    assert state.metrics["selected_scope_daily_week"] == 1.0
    assert "DEALER_DIRECTION_UNCALIBRATED" not in {
        issue.code for issue in state.data_quality.issues
    }
    lineage = [json.loads(value) for value in state.lineage_ids]
    assert [item["source_id"] for item in lineage] == [
        "dealer-capture",
        "dealer-ranking",
    ]

    context = _snapshot_at(state, datetime(2026, 7, 30, 20, 20, tzinfo=UTC))
    assert context.dealer_state is not None
    assert context.dealer_state.state_id == state.state_id


def test_join_uses_latest_explicit_evidence_availability() -> None:
    state = dealer_adapter.adapt_dealer_state_with_ranking(
        _snapshot(),
        _dynamics(),
        _ranking(),
        captured_at=CAPTURED_AT,
        ranking_available_at=CAPTURED_AT + timedelta(minutes=5),
        valid_until=VALID_UNTIL,
        lineage=LINEAGE,
        ranking_lineage=RANKING_LINEAGE,
    )

    assert state.available_at == LEVELS_AVAILABLE_AT


@pytest.mark.parametrize(
    "ranking_updates",
    [
        {"symbol": "QQQ"},
        {"captured_at": (CAPTURED_AT + timedelta(minutes=1)).isoformat()},
        {"snapshot_date": "2026-07-31"},
        {"scope": "daily_week"},
    ],
)
def test_join_rejects_nonmatching_or_nonaggregate_ranking(
    ranking_updates: dict[str, object]
) -> None:
    with pytest.raises(ValueError, match="ranking|scope|ticker|snapshot_date|captured_at"):
        dealer_adapter.adapt_dealer_state_with_ranking(
            _snapshot(),
            None,
            _ranking(**ranking_updates),
            captured_at=CAPTURED_AT,
            ranking_available_at=RANKING_AVAILABLE_AT,
            valid_until=VALID_UNTIL,
            lineage=LINEAGE,
            ranking_lineage=RANKING_LINEAGE,
        )


def test_join_requires_aware_causal_ranking_availability() -> None:
    with pytest.raises(ValueError, match="ranking_available_at.*timezone-aware"):
        dealer_adapter.adapt_dealer_state_with_ranking(
            _snapshot(),
            None,
            _ranking(),
            captured_at=CAPTURED_AT,
            ranking_available_at=RANKING_AVAILABLE_AT.replace(tzinfo=None),
            valid_until=VALID_UNTIL,
            lineage=LINEAGE,
            ranking_lineage=RANKING_LINEAGE,
        )

    with pytest.raises(ValueError, match="ranking_available_at.*precede"):
        dealer_adapter.adapt_dealer_state_with_ranking(
            _snapshot(),
            None,
            _ranking(),
            captured_at=CAPTURED_AT,
            ranking_available_at=CAPTURED_AT - timedelta(seconds=1),
            valid_until=VALID_UNTIL,
            lineage=LINEAGE,
            ranking_lineage=RANKING_LINEAGE,
        )


def test_join_requires_separate_ranking_lineage() -> None:
    with pytest.raises(ValueError, match="ranking_lineage.*separate"):
        dealer_adapter.adapt_dealer_state_with_ranking(
            _snapshot(),
            None,
            _ranking(),
            captured_at=CAPTURED_AT,
            ranking_available_at=RANKING_AVAILABLE_AT,
            valid_until=VALID_UNTIL,
            lineage=LINEAGE,
            ranking_lineage=LINEAGE,
        )
