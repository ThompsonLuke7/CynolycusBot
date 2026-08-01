from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from hashlib import sha256
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core.nervous_system.contracts.base import content_hash
from core.nervous_system.contracts.context import ContextSnapshot
from core.nervous_system.contracts.enums import DataQualitySeverity, DealerRegime
from core.nervous_system.contracts.quality import LineageRef
from strategies.dealer_positioning import nervous_system_adapter as dealer_adapter
from strategies.dealer_positioning.nervous_system_adapter import adapt_dealer_state
from strategies.dealer_positioning.scripts.build_level_dynamics import (
    compute_summary_dynamics,
)


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
LOCAL_REPLAY_PATH_ENV = "CYNOLYCUS_DEALER_REPLAY_PATH"
LOCAL_REPLAY_EXPECTED_ROWS_ENV = "CYNOLYCUS_DEALER_REPLAY_EXPECTED_ROWS"
EXPECTED_DYNAMICS_NUMERIC_FIELDS = frozenset(
    {
        "spot",
        "call_wall",
        "put_wall",
        "gamma_flip",
        "nearest_magnet",
        "gex_concentration_index",
        "total_option_oi",
        "total_option_volume",
        "prior_gap_days",
        "third_gap_days",
        "wall_change_1d",
        "wall_change_1d_gap_days",
        "wall_change_3d",
        "wall_change_3d_gap_days",
        "gex_concentration_change",
        "gamma_flip_velocity",
        "oi_change_by_dte",
        "volume_to_prior_oi",
        "level_stability_days",
        "atr_14d",
        "distance_to_call_wall_atr",
        "distance_to_put_wall_atr",
        "iv_skew_25d",
        "iv_skew_change",
        "near_level_option_volume_share",
    }
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
        "next_magnet_below": 635.0,
        "vega_below": 630.0,
        "gamma_flip": 638.0,
        "air_gap_above_score": 0.35,
        "air_gap_below_score": 0.55,
        "total_option_oi": 125_000.0,
        "total_option_volume": 18_000.0,
        "avg_dollar_volume_20d": 75_000_000.0,
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


@pytest.fixture(scope="module")
def actual_dynamics_output() -> pd.DataFrame:
    prior_at = CAPTURED_AT - timedelta(days=1)
    summary = pd.DataFrame(
        [
            {
                "captured_at": prior_at,
                "snapshot_date": "2026-07-29",
                "symbol": "SPY",
                "scope": "daily_week",
                "spot": 639.0,
                "call_wall": 649.0,
                "put_wall": 624.0,
                "gamma_flip": 637.0,
                "nearest_magnet": 639.0,
                "gex_concentration_index": 0.50,
                "total_option_oi": 124_000.0,
                "total_option_volume": 17_000.0,
            },
            {
                "captured_at": CAPTURED_AT,
                "snapshot_date": "2026-07-30",
                "symbol": "SPY",
                "scope": "daily_week",
                "spot": 640.0,
                "call_wall": 650.0,
                "put_wall": 625.0,
                "gamma_flip": 638.0,
                "nearest_magnet": 640.0,
                "gex_concentration_index": 0.55,
                "total_option_oi": 125_000.0,
                "total_option_volume": 18_000.0,
            },
        ]
    )
    summary["captured_at"] = pd.to_datetime(summary["captured_at"], utc=True)
    summary["snapshot_date"] = pd.to_datetime(summary["snapshot_date"])

    ladder_rows: list[dict[str, object]] = []
    for captured_at, snapshot_date, spot in (
        (prior_at, "2026-07-29", 639.0),
        (CAPTURED_AT, "2026-07-30", 640.0),
    ):
        for strike, call_oi, put_oi in (
            (625.0, 100.0, 300.0),
            (650.0, 300.0, 100.0),
        ):
            ladder_rows.append(
                {
                    "captured_at": captured_at,
                    "snapshot_date": snapshot_date,
                    "symbol": "SPY",
                    "scope": "daily_week",
                    "spot": spot,
                    "strike": strike,
                    "call_oi": call_oi,
                    "put_oi": put_oi,
                    "call_volume": 10.0,
                    "put_volume": 8.0,
                    "call_iv": 0.30,
                    "put_iv": 0.35,
                    "call_delta": 0.25,
                    "put_delta": -0.25,
                }
            )
    ladder = pd.DataFrame(ladder_rows)
    ladder["captured_at"] = pd.to_datetime(ladder["captured_at"], utc=True)
    ladder["snapshot_date"] = pd.to_datetime(ladder["snapshot_date"])
    atr_lookup = {
        ("SPY", pd.Timestamp("2026-07-29")): 2.0,
        ("SPY", pd.Timestamp("2026-07-30")): 2.0,
    }
    return compute_summary_dynamics(summary, ladder, atr_lookup=atr_lookup)


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


@pytest.mark.parametrize(
    "row_captured_at",
    [None, pd.NA, pd.NaT, "", "not-a-timestamp", "2026-07-30T19:45:00"],
)
def test_raw_capture_requires_valid_row_captured_at(row_captured_at: object) -> None:
    with pytest.raises(ValueError, match="snapshot.*captured_at|captured_at"):
        adapt_dealer_state(
            _snapshot(captured_at=row_captured_at),
            None,
            captured_at=CAPTURED_AT,
            valid_until=VALID_UNTIL,
            lineage=LINEAGE,
        )


def test_raw_capture_requires_row_captured_at_key() -> None:
    row = _snapshot()
    row.pop("captured_at")

    with pytest.raises(ValueError, match="snapshot.*captured_at|captured_at"):
        adapt_dealer_state(
            row,
            None,
            captured_at=CAPTURED_AT,
            valid_until=VALID_UNTIL,
            lineage=LINEAGE,
        )


@pytest.mark.parametrize(
    "snapshot_date",
    [None, pd.NA, pd.NaT, np.float64(np.nan), "", "not-a-date", "2026-02-30"],
)
def test_raw_capture_requires_valid_snapshot_date(snapshot_date: object) -> None:
    with pytest.raises(ValueError, match="snapshot.*snapshot_date|snapshot_date"):
        adapt_dealer_state(
            _snapshot(snapshot_date=snapshot_date),
            None,
            captured_at=CAPTURED_AT,
            valid_until=VALID_UNTIL,
            lineage=LINEAGE,
        )


def test_raw_capture_requires_snapshot_date_key() -> None:
    row = _snapshot()
    row.pop("snapshot_date")

    with pytest.raises(ValueError, match="snapshot.*snapshot_date|snapshot_date"):
        adapt_dealer_state(
            row,
            None,
            captured_at=CAPTURED_AT,
            valid_until=VALID_UNTIL,
            lineage=LINEAGE,
        )


def test_raw_snapshot_date_cannot_be_later_than_capture_date() -> None:
    with pytest.raises(ValueError, match="snapshot_date.*after.*captured_at"):
        adapt_dealer_state(
            _snapshot(snapshot_date="2026-07-31"),
            None,
            captured_at=CAPTURED_AT,
            valid_until=VALID_UNTIL,
            lineage=LINEAGE,
        )


def test_historical_snapshot_date_may_precede_later_capture_availability() -> None:
    state = adapt_dealer_state(
        _snapshot(snapshot_date="2026-07-28"),
        None,
        captured_at=CAPTURED_AT,
        valid_until=VALID_UNTIL,
        lineage=LINEAGE,
    )

    assert state.as_of == CAPTURED_AT
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


def test_required_spot_null_marker_is_rejected() -> None:
    with pytest.raises(ValueError, match="finite"):
        adapt_dealer_state(
            _snapshot(spot=np.float64(np.nan), spot_price=np.float64(np.nan)),
            None,
            captured_at=CAPTURED_AT,
            valid_until=VALID_UNTIL,
            lineage=LINEAGE,
        )


@pytest.mark.parametrize(
    ("field", "null_value"),
    [
        ("next_magnet_below", np.float64(np.nan)),
        ("vega_below", np.float64(np.nan)),
        ("avg_dollar_volume_20d", np.float64(np.nan)),
        ("total_gex", pd.NA),
        ("call_wall", np.float64(np.nan)),
    ],
)
def test_actual_parquet_optional_null_markers_are_absent(
    field: str, null_value: object
) -> None:
    state = adapt_dealer_state(
        _snapshot(**{field: null_value}),
        None,
        captured_at=CAPTURED_AT,
        valid_until=VALID_UNTIL,
        lineage=LINEAGE,
    )

    assert field not in state.metrics
    if field == "total_gex":
        assert state.total_gex is None
    if field == "call_wall":
        assert state.call_wall is None


@pytest.mark.parametrize(
    "value",
    [
        "635.0",
        b"635.0",
        True,
        np.bool_(True),
        pd.array([True], dtype="boolean")[0],
    ],
    ids=["numeric-string", "bytes", "python-bool", "numpy-bool", "pandas-bool"],
)
def test_known_numeric_fields_reject_non_numeric_scalar_types(value: object) -> None:
    with pytest.raises(ValueError, match="next_magnet_below"):
        adapt_dealer_state(
            _snapshot(next_magnet_below=value),
            None,
            captured_at=CAPTURED_AT,
            valid_until=VALID_UNTIL,
            lineage=LINEAGE,
        )


@pytest.mark.parametrize(
    "value",
    [1, 1.5, np.int64(2), np.float64(2.5), Decimal("3.5")],
    ids=["python-int", "python-float", "numpy-int", "numpy-float", "decimal"],
)
def test_known_numeric_fields_accept_genuine_finite_numeric_scalars(
    value: object,
) -> None:
    state = adapt_dealer_state(
        _snapshot(next_magnet_below=value),
        None,
        captured_at=CAPTURED_AT,
        valid_until=VALID_UNTIL,
        lineage=LINEAGE,
    )

    assert state.metrics["next_magnet_below"] == float(value)


def test_nested_optional_producer_null_markers_are_absent() -> None:
    state = adapt_dealer_state(
        _snapshot(
            per_dte_levels={
                "daily_week": {
                    "next_magnet_below": np.float64(np.nan),
                    "vega_below": pd.NA,
                    "observation_time": pd.NaT,
                }
            }
        ),
        None,
        captured_at=CAPTURED_AT,
        valid_until=VALID_UNTIL,
        lineage=LINEAGE,
    )

    assert "per_dte_levels" not in state.metrics


@pytest.mark.parametrize(
    "quality_updates",
    [
        {"stale_days": np.float64(np.nan)},
        {"print_sparse": pd.NA},
        {"print_count": np.float64(np.nan), "expected_prints": 100},
        {"print_coverage": np.float64(np.nan)},
    ],
)
def test_null_quality_evidence_is_omitted(
    quality_updates: dict[str, object]
) -> None:
    state = adapt_dealer_state(
        _snapshot(**quality_updates),
        None,
        captured_at=CAPTURED_AT,
        valid_until=VALID_UNTIL,
        lineage=LINEAGE,
    )

    codes = {issue.code for issue in state.data_quality.issues}
    assert "STALE_OPTION_DATA" not in codes
    assert "PRINT_SPARSE_OPTION_DATA" not in codes
    null_fields = {
        name for name, value in quality_updates.items() if bool(pd.isna(value))
    }
    assert null_fields.isdisjoint(state.metrics)


@pytest.mark.parametrize(
    "updates",
    [
        {"next_magnet_below": np.inf},
        {"avg_dollar_volume_20d": -np.inf},
        {"per_dte_levels": {"daily_week": {"vega_below": np.inf}}},
    ],
)
def test_present_optional_infinities_still_fail(updates: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="finite"):
        adapt_dealer_state(
            _snapshot(**updates),
            None,
            captured_at=CAPTURED_AT,
            valid_until=VALID_UNTIL,
            lineage=LINEAGE,
        )


@pytest.mark.parametrize(
    ("field", "malformed_value"),
    [
        ("next_magnet_below", "not-a-number"),
        ("vega_below", {"value": 630.0}),
        ("avg_dollar_volume_20d", [75_000_000.0]),
        ("next_magnet_above", np.array([645.0])),
        ("market_cap", True),
        ("vacuum_above", np.array([np.nan])),
    ],
)
def test_present_malformed_raw_numeric_evidence_fails_closed(
    field: str, malformed_value: object
) -> None:
    with pytest.raises(ValueError, match=field):
        adapt_dealer_state(
            _snapshot(**{field: malformed_value}),
            None,
            captured_at=CAPTURED_AT,
            valid_until=VALID_UNTIL,
            lineage=LINEAGE,
        )


@pytest.mark.parametrize(
    ("field", "malformed_value"),
    [
        ("vega_below", "not-a-number"),
        ("next_magnet_below", [635.0]),
        ("gamma_density", np.array([0.5])),
        ("dealer_bias", False),
    ],
)
def test_nested_malformed_numeric_evidence_fails_closed(
    field: str, malformed_value: object
) -> None:
    with pytest.raises(ValueError, match=field):
        adapt_dealer_state(
            _snapshot(
                per_dte_levels={
                    "daily_week": {
                        field: malformed_value,
                    }
                }
            ),
            None,
            captured_at=CAPTURED_AT,
            valid_until=VALID_UNTIL,
            lineage=LINEAGE,
        )


@pytest.mark.parametrize(
    ("field", "malformed_value"),
    [
        ("wall_change_1d", "not-a-number"),
        ("gex_concentration_change", {"value": -0.05}),
        ("distance_to_call_wall_atr", [1.2]),
        ("oi_change_by_dte", np.array([100.0])),
        ("level_stability_days", True),
    ],
)
def test_present_malformed_dynamics_numeric_evidence_fails_closed(
    field: str, malformed_value: object
) -> None:
    with pytest.raises(ValueError, match=field):
        adapt_dealer_state(
            _snapshot(),
            _dynamics(**{field: malformed_value}),
            captured_at=CAPTURED_AT,
            valid_until=VALID_UNTIL,
            lineage=LINEAGE,
        )


def test_actual_dynamics_compute_numeric_surface_is_fully_declared(
    actual_dynamics_output: pd.DataFrame,
) -> None:
    actual_numeric_fields = frozenset(
        actual_dynamics_output.select_dtypes(include="number").columns
    )

    assert actual_numeric_fields == EXPECTED_DYNAMICS_NUMERIC_FIELDS
    assert actual_numeric_fields <= dealer_adapter._KNOWN_NUMERIC_FIELDS
    assert "iv_skew_25d" in actual_numeric_fields


def test_actual_dynamics_independent_stale_horizons_warn_if_either_is_stale(
    actual_dynamics_output: pd.DataFrame,
) -> None:
    row = actual_dynamics_output.iloc[-1].to_dict()
    assert row["prior_gap_stale"] is False
    assert row["third_gap_stale"] is True

    state = adapt_dealer_state(
        _snapshot(),
        row,
        captured_at=CAPTURED_AT,
        valid_until=VALID_UNTIL,
        lineage=LINEAGE,
    )

    assert "STALE_OPTION_DATA" in {
        issue.code for issue in state.data_quality.issues
    }


@pytest.mark.parametrize("field", sorted(EXPECTED_DYNAMICS_NUMERIC_FIELDS))
def test_every_actual_dynamics_numeric_field_rejects_malformed_schema_drift(
    actual_dynamics_output: pd.DataFrame,
    field: str,
) -> None:
    row = actual_dynamics_output.iloc[-1].to_dict()
    row[field] = "1.25"

    with pytest.raises(ValueError, match=field):
        adapt_dealer_state(
            _snapshot(),
            row,
            captured_at=CAPTURED_AT,
            valid_until=VALID_UNTIL,
            lineage=LINEAGE,
        )


@pytest.mark.parametrize("field", sorted(EXPECTED_DYNAMICS_NUMERIC_FIELDS))
def test_every_actual_dynamics_optional_numeric_null_is_omitted(
    actual_dynamics_output: pd.DataFrame,
    field: str,
) -> None:
    row = actual_dynamics_output.iloc[-1].to_dict()
    row[field] = np.float64(np.nan)

    state = adapt_dealer_state(
        _snapshot(),
        row,
        captured_at=CAPTURED_AT,
        valid_until=VALID_UNTIL,
        lineage=LINEAGE,
    )

    snapshot_value = _snapshot().get(field)
    if isinstance(snapshot_value, (int, float)) and not isinstance(
        snapshot_value, bool
    ):
        assert state.metrics[field] == float(snapshot_value)
    else:
        assert field not in state.metrics


@pytest.mark.parametrize(
    ("field", "malformed_value"),
    [
        ("avg_vacuum_component", "not-a-number"),
        ("max_scope_score", {"value": 81.0}),
        ("avg_gamma_density", [0.3]),
        ("dealer_change_intensity_rank", np.array([2])),
        ("avg_wall_room_component", False),
    ],
)
def test_present_malformed_ranking_numeric_evidence_fails_closed(
    field: str, malformed_value: object
) -> None:
    with pytest.raises(ValueError, match=field):
        dealer_adapter.adapt_dealer_state_with_ranking(
            _snapshot(),
            None,
            _ranking(**{field: malformed_value}),
            captured_at=CAPTURED_AT,
            ranking_available_at=RANKING_AVAILABLE_AT,
            valid_until=VALID_UNTIL,
            lineage=LINEAGE,
            ranking_lineage=RANKING_LINEAGE,
        )


def test_declared_categorical_and_expiration_payloads_are_not_metrics() -> None:
    state = adapt_dealer_state(
        _snapshot(
            expirations=["2026-07-31"],
            available_expirations=("2026-07-31",),
            nearest_level_type="Magnet",
            market_cap_bucket="mega",
            cap_tier="large",
        ),
        None,
        captured_at=CAPTURED_AT,
        valid_until=VALID_UNTIL,
        lineage=LINEAGE,
    )

    assert {
        "expirations",
        "available_expirations",
        "nearest_level_type",
        "market_cap_bucket",
        "cap_tier",
    }.isdisjoint(state.metrics)


def _replay_dealer_parquet(path: Path, *, expected_rows: int) -> int:
    assert expected_rows > 0, "expected_rows must be positive"
    assert path.is_file(), f"explicit dealer replay path does not exist: {path}"
    frame = pd.read_parquet(path)
    assert len(frame) == expected_rows

    adapted = 0
    failures: list[str] = []
    for index, series in frame.iterrows():
        row = series.to_dict()
        try:
            captured_at = pd.Timestamp(row["captured_at"])
            if captured_at.tzinfo is None:
                raise ValueError("captured_at must be timezone-aware")
        except (KeyError, TypeError, ValueError) as exc:
            failures.append(f"{index}:invalid captured_at:{exc}")
            continue
        lineage = (
            LineageRef(
                source_id="dealer-local-replay",
                content_hash=sha256(f"{path}:{index}".encode()).hexdigest(),
                record_locator=f"{path.name}:{index}",
            ),
        )
        try:
            state = adapt_dealer_state(
                row,
                None,
                captured_at=captured_at.to_pydatetime(),
                valid_until=(captured_at + timedelta(days=7)).to_pydatetime(),
                lineage=lineage,
            )
        except ValueError as exc:
            failures.append(
                f"{index}:{row.get('symbol')}:{row.get('scope')}:{exc}"
            )
            continue

        null_columns = {
            name
            for name, value in row.items()
            if pd.api.types.is_scalar(value) and bool(pd.isna(value))
        }
        assert null_columns.isdisjoint(state.metrics)
        assert all(math.isfinite(value) for value in state.metrics.values())
        adapted += 1

    assert failures == []
    assert adapted == expected_rows
    return adapted


def test_explicit_parquet_fixture_replays_exact_row_count(tmp_path: Path) -> None:
    path = tmp_path / "dealer_level_summary.parquet"
    rows = [
        _snapshot(),
        _snapshot(
            symbol="QQQ",
            scope="through_month",
            scope_label="through_month",
            next_magnet_below=np.float64(np.nan),
        ),
        _snapshot(
            symbol="IWM",
            scope="two_months",
            scope_label="two_months",
            vega_below=np.float64(np.nan),
            avg_dollar_volume_20d=np.float64(np.nan),
        ),
    ]
    pd.DataFrame(rows).to_parquet(path, index=False)

    assert _replay_dealer_parquet(path, expected_rows=3) == 3


def test_replay_rejects_zero_expected_rows_and_empty_parquet(tmp_path: Path) -> None:
    path = tmp_path / "empty_dealer_level_summary.parquet"
    pd.DataFrame().to_parquet(path, index=False)

    with pytest.raises(AssertionError, match="expected_rows must be positive"):
        _replay_dealer_parquet(path, expected_rows=0)


def test_explicit_opt_in_local_parquet_replay() -> None:
    path_value = os.environ.get(LOCAL_REPLAY_PATH_ENV)
    if path_value is None:
        pytest.skip(f"opt in by setting {LOCAL_REPLAY_PATH_ENV}")
    expected_value = os.environ.get(LOCAL_REPLAY_EXPECTED_ROWS_ENV)
    assert expected_value is not None, (
        f"{LOCAL_REPLAY_EXPECTED_ROWS_ENV} is required with {LOCAL_REPLAY_PATH_ENV}"
    )
    try:
        expected_rows = int(expected_value)
    except ValueError as exc:
        raise AssertionError(
            f"{LOCAL_REPLAY_EXPECTED_ROWS_ENV} must be an integer"
        ) from exc
    assert expected_rows > 0

    path = Path(path_value).expanduser()
    assert _replay_dealer_parquet(path, expected_rows=expected_rows) == expected_rows


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
    "dynamics_updates",
    [
        {"symbol": "QQQ"},
        {"snapshot_date": "2026-07-31"},
        {"captured_at": (CAPTURED_AT + timedelta(minutes=1)).isoformat()},
        {"scope": "through_month"},
    ],
)
def test_direct_adapter_rejects_dynamics_from_a_different_selected_capture(
    dynamics_updates: dict[str, object]
) -> None:
    with pytest.raises(
        ValueError, match="dynamics.*(ticker|snapshot_date|captured_at|scope)"
    ):
        adapt_dealer_state(
            _snapshot(),
            _dynamics(**dynamics_updates),
            captured_at=CAPTURED_AT,
            valid_until=VALID_UNTIL,
            lineage=LINEAGE,
        )


@pytest.mark.parametrize("field", ["symbol", "snapshot_date", "captured_at"])
def test_direct_adapter_requires_dynamics_identity_fields(field: str) -> None:
    dynamics = _dynamics()
    dynamics.pop(field)

    with pytest.raises(ValueError, match=f"dynamics.*{field}|dynamics.*ticker"):
        adapt_dealer_state(
            _snapshot(),
            dynamics,
            captured_at=CAPTURED_AT,
            valid_until=VALID_UNTIL,
            lineage=LINEAGE,
        )


@pytest.mark.parametrize(
    ("field", "null_value"),
    [("symbol", pd.NA), ("snapshot_date", pd.NaT), ("captured_at", pd.NaT)],
)
def test_direct_adapter_rejects_null_dynamics_identity_fields(
    field: str, null_value: object
) -> None:
    with pytest.raises(ValueError, match=f"dynamics.*{field}|dynamics.*ticker"):
        adapt_dealer_state(
            _snapshot(),
            _dynamics(**{field: null_value}),
            captured_at=CAPTURED_AT,
            valid_until=VALID_UNTIL,
            lineage=LINEAGE,
        )


@pytest.mark.parametrize("scope_value", [None, pd.NA])
def test_direct_adapter_binds_absent_dynamics_scope_to_selected_raw_scope(
    scope_value: object,
) -> None:
    dynamics = _dynamics(scope=scope_value)
    if scope_value is None:
        dynamics.pop("scope")

    state = adapt_dealer_state(
        _snapshot(scope="daily_week"),
        dynamics,
        captured_at=CAPTURED_AT,
        valid_until=VALID_UNTIL,
        lineage=LINEAGE,
    )

    assert state.metrics["selected_scope_daily_week"] == 1.0


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
    "dynamics_updates",
    [
        {"symbol": "QQQ"},
        {"snapshot_date": "2026-07-31"},
        {"captured_at": (CAPTURED_AT + timedelta(minutes=1)).isoformat()},
        {"scope": "through_month"},
    ],
)
def test_join_rejects_dynamics_from_a_different_selected_capture(
    dynamics_updates: dict[str, object]
) -> None:
    with pytest.raises(
        ValueError, match="dynamics.*(ticker|snapshot_date|captured_at|scope)"
    ):
        dealer_adapter.adapt_dealer_state_with_ranking(
            _snapshot(),
            _dynamics(**dynamics_updates),
            _ranking(),
            captured_at=CAPTURED_AT,
            ranking_available_at=RANKING_AVAILABLE_AT,
            valid_until=VALID_UNTIL,
            lineage=LINEAGE,
            ranking_lineage=RANKING_LINEAGE,
        )


def test_join_binds_missing_dynamics_scope_to_selected_raw_scope() -> None:
    dynamics = _dynamics()
    dynamics.pop("scope")

    state = dealer_adapter.adapt_dealer_state_with_ranking(
        _snapshot(scope="daily_week"),
        dynamics,
        _ranking(),
        captured_at=CAPTURED_AT,
        ranking_available_at=RANKING_AVAILABLE_AT,
        valid_until=VALID_UNTIL,
        lineage=LINEAGE,
        ranking_lineage=RANKING_LINEAGE,
    )

    assert state.metrics["selected_scope_daily_week"] == 1.0


def test_join_rejects_pre_capture_dynamics_availability_before_max_merge() -> None:
    with pytest.raises(ValueError, match="dynamics available_at.*precede"):
        dealer_adapter.adapt_dealer_state_with_ranking(
            _snapshot(),
            _dynamics(available_at=CAPTURED_AT - timedelta(seconds=1)),
            _ranking(),
            captured_at=CAPTURED_AT,
            ranking_available_at=RANKING_AVAILABLE_AT,
            valid_until=VALID_UNTIL,
            lineage=LINEAGE,
            ranking_lineage=RANKING_LINEAGE,
        )


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
