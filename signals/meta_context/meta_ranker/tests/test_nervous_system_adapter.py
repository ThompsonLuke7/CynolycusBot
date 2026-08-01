from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from core.nervous_system.contracts.base import content_hash
from core.nervous_system.contracts.quality import LineageRef
from core.nervous_system.contracts.enums import StateType
from signals.meta_context.meta_ranker.nervous_system_adapter import adapt_ticker_state


UTC = timezone.utc
DECISION_BAR = datetime(2026, 7, 30, 18, 0, tzinfo=UTC)
AVAILABLE_AT = datetime(2026, 7, 30, 20, 25, tzinfo=UTC)
VALID_UNTIL = AVAILABLE_AT + timedelta(hours=24)
LINEAGE = (
    LineageRef(
        source_id="meta-ranker-matrix",
        content_hash="a" * 64,
        record_locator="meta_ranker_matrix:row:42",
    ),
)


def _row(**updates: object) -> dict[str, object]:
    row: dict[str, object] = {
        "ticker": "ABC",
        "timestamp": DECISION_BAR,
        "open": 100.0,
        "high": 106.0,
        "low": 98.5,
        "close": 104.0,
        "rank": 2,
        "s_combo": 0.81,
        "s_upside": 0.77,
        "s_quality": 0.85,
        "ticker_setup": "BREAKOUT",
        "mom_xs_rank": 0.91,
    }
    row.update(updates)
    return row


def test_ticker_state_uses_exact_selected_bar_and_preserves_scores() -> None:
    state = adapt_ticker_state(
        _row(),
        decision_bar=DECISION_BAR,
        available_at=AVAILABLE_AT,
        valid_until=VALID_UNTIL,
        lineage=LINEAGE,
    )

    assert state.ticker == "ABC"
    assert state.as_of == DECISION_BAR
    assert state.selected_bar == DECISION_BAR
    assert state.available_at == AVAILABLE_AT
    assert state.reference_price == 104.0
    assert state.metrics["open"] == 100.0
    assert state.metrics["high"] == 106.0
    assert state.metrics["low"] == 98.5
    assert state.metrics["close"] == 104.0
    assert state.metrics["rank"] == 2.0
    assert state.metrics["s_combo"] == pytest.approx(0.81)
    assert state.metrics["s_upside"] == pytest.approx(0.77)
    assert state.transition_probabilities == {}
    assert state.data_quality.is_usable
    assert state.lineage_ids and "meta-ranker-matrix" in state.lineage_ids[0]


def test_ticker_state_has_deterministic_metadata_and_source_lineage() -> None:
    first = adapt_ticker_state(
        _row(),
        decision_bar=DECISION_BAR,
        available_at=AVAILABLE_AT,
        valid_until=VALID_UNTIL,
        lineage=LINEAGE,
    )
    second = adapt_ticker_state(
        _row(),
        decision_bar=DECISION_BAR,
        available_at=AVAILABLE_AT,
        valid_until=VALID_UNTIL,
        lineage=LINEAGE,
    )

    assert first.state_type is StateType.TICKER
    assert first.generated_at == AVAILABLE_AT
    assert first.source_window_start == DECISION_BAR
    assert first.source_window_end == DECISION_BAR
    assert first.producer == "signals.meta_context.meta_ranker"
    assert first.model_version == "meta-ranker-adapter@1"
    assert first.feature_version == "meta-ranker-matrix@1"
    assert first.config_version == "meta-ranker-ticker@1"
    assert LINEAGE[0].content_hash in first.lineage_ids[0]
    assert LINEAGE[0].record_locator in first.lineage_ids[0]
    assert first.model_dump() == second.model_dump()


def test_ticker_state_does_not_use_a_later_appended_bar() -> None:
    rows = [_row(), _row(timestamp=DECISION_BAR + timedelta(hours=4), close=999.0)]
    selected = next(row for row in rows if row["timestamp"] == DECISION_BAR)

    before = adapt_ticker_state(
        selected,
        decision_bar=DECISION_BAR,
        available_at=AVAILABLE_AT,
        valid_until=VALID_UNTIL,
        lineage=LINEAGE,
    )
    after = adapt_ticker_state(
        next(row for row in rows if row["timestamp"] == DECISION_BAR),
        decision_bar=DECISION_BAR,
        available_at=AVAILABLE_AT,
        valid_until=VALID_UNTIL,
        lineage=LINEAGE,
    )

    assert after.reference_price == 104.0
    assert after.state_id == before.state_id
    assert content_hash(after, exclude={"state_id"}) == content_hash(
        before, exclude={"state_id"}
    )


def test_ticker_state_rejects_a_row_for_a_different_decision_bar() -> None:
    with pytest.raises(ValueError, match="selected bar|decision bar"):
        adapt_ticker_state(
            _row(timestamp=DECISION_BAR + timedelta(hours=4)),
            decision_bar=DECISION_BAR,
            available_at=AVAILABLE_AT,
            valid_until=VALID_UNTIL,
            lineage=LINEAGE,
        )


def test_ticker_state_rejects_a_missing_selected_bar() -> None:
    row = _row()
    row.pop("timestamp")

    with pytest.raises(ValueError, match="timestamp|selected bar|decision bar"):
        adapt_ticker_state(
            row,
            decision_bar=DECISION_BAR,
            available_at=AVAILABLE_AT,
            valid_until=VALID_UNTIL,
            lineage=LINEAGE,
        )


@pytest.mark.parametrize(
    ("argument", "value"),
    [
        ("decision_bar", datetime(2026, 7, 30, 18, 0)),
        ("available_at", datetime(2026, 7, 30, 20, 25)),
        ("valid_until", datetime(2026, 7, 31, 20, 25)),
    ],
)
def test_ticker_state_rejects_naive_causal_timestamps(
    argument: str, value: datetime
) -> None:
    kwargs: dict[str, object] = {
        "decision_bar": DECISION_BAR,
        "available_at": AVAILABLE_AT,
        "valid_until": VALID_UNTIL,
        "lineage": LINEAGE,
    }
    kwargs[argument] = value

    with pytest.raises(ValueError, match="timezone-aware|naive"):
        adapt_ticker_state(_row(), **kwargs)


def test_ticker_state_rejects_availability_before_selected_bar() -> None:
    with pytest.raises(ValueError, match="causal|available_at|decision bar"):
        adapt_ticker_state(
            _row(),
            decision_bar=DECISION_BAR,
            available_at=DECISION_BAR - timedelta(minutes=1),
            valid_until=VALID_UNTIL,
            lineage=LINEAGE,
        )


def test_ticker_state_rejects_a_non_exclusive_validity_window() -> None:
    with pytest.raises(ValueError, match="valid_until"):
        adapt_ticker_state(
            _row(),
            decision_bar=DECISION_BAR,
            available_at=AVAILABLE_AT,
            valid_until=AVAILABLE_AT,
            lineage=LINEAGE,
        )


@pytest.mark.parametrize("close", [None, float("nan"), float("inf")])
def test_ticker_state_fails_closed_for_missing_or_non_finite_selected_close(close: object) -> None:
    row = _row()
    if close is None:
        row.pop("close")
    else:
        row["close"] = close

    with pytest.raises(ValueError, match="close"):
        adapt_ticker_state(
            row,
            decision_bar=DECISION_BAR,
            available_at=AVAILABLE_AT,
            valid_until=VALID_UNTIL,
            lineage=LINEAGE,
        )
