"""Causal publication tests for market-regime and sector-state outputs."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from core.nervous_system.contracts.context import ContextSnapshot
from core.nervous_system.contracts.enums import MarketRegime
from core.nervous_system.contracts.quality import LineageRef
from signals.market_regime import build as build_module
from signals.market_regime.nervous_system_adapter import (
    adapt_market_row,
    adapt_sector_row,
    persist_market_regime_outputs,
)


UTC = timezone.utc
AVAILABLE_AT = datetime(2026, 7, 29, 20, 30, tzinfo=UTC)
GENERATED_AT = datetime(2026, 7, 29, 21, 0, tzinfo=UTC)
VALID_UNTIL = datetime(2026, 7, 30, 20, 30, tzinfo=UTC)
LINEAGE = (
    LineageRef(
        source_id="daily-regime-artifact",
        content_hash="a" * 64,
        record_locator="daily_regime:row:42",
    ),
)


def _market_row(**updates: object) -> dict[str, object]:
    row: dict[str, object] = {
        "date": pd.Timestamp("2026-07-29"),
        "available_at": pd.Timestamp(AVAILABLE_AT),
        "generated_at": pd.Timestamp(GENERATED_AT),
        "risk_appetite_z": 0.25,
        "risk_appetite_z_n_components": 4,
        "risk_appetite_z_stale_days": 0.0,
        "breadth_z": -0.4,
        "spy_trend_state": 1.0,
        "label": "risk_on_rule_vector",
        "source_id": "daily-regime-artifact",
        "content_hash": "a" * 64,
        "record_locator": "daily_regime:row:42",
    }
    row.update(updates)
    return row


def _sector_row(**updates: object) -> dict[str, object]:
    row: dict[str, object] = {
        "date": pd.Timestamp("2026-07-29"),
        "sector_etf": "XLK",
        "available_at": pd.Timestamp(AVAILABLE_AT),
        "generated_at": pd.Timestamp(GENERATED_AT),
        "excess_21d": 0.12,
        "excess_63d": 0.24,
        "rank_21d": 0.8,
        "rank_63d": 0.6,
        "rs_accel": 0.04,
        "above_20d": 1.0,
        "above_50d": 0.0,
        "stale_days": 0.0,
        "source_id": "sector-state-artifact",
        "content_hash": "b" * 64,
        "record_locator": "sector_state:row:42",
    }
    row.update(updates)
    return row


def _valid_until(_: datetime) -> datetime:
    return VALID_UNTIL


def test_market_adapter_copies_causal_timestamps_and_preserves_metrics() -> None:
    state = adapt_market_row(_market_row(), valid_until=VALID_UNTIL, lineage=LINEAGE)

    assert state.as_of == datetime(2026, 7, 29, 20, 0, tzinfo=UTC)
    assert state.available_at == AVAILABLE_AT
    assert state.generated_at == GENERATED_AT
    assert state.valid_until == VALID_UNTIL
    assert state.metrics["risk_appetite_z"] == 0.25
    assert state.metrics["risk_appetite_z_n_components"] == 4.0
    assert state.metrics["spy_trend_state"] == 1.0
    assert "label" not in state.metrics
    assert LINEAGE[0].content_hash in state.lineage_ids[0]
    assert LINEAGE[0].record_locator in state.lineage_ids[0]


def test_market_adapter_never_substitutes_generated_at_for_available_at() -> None:
    row = _market_row(
        available_at=pd.Timestamp("2026-07-29 20:30:00", tz="UTC"),
        generated_at=pd.Timestamp("2026-07-30 02:00:00", tz="UTC"),
    )

    state = adapt_market_row(row, valid_until=VALID_UNTIL, lineage=LINEAGE)

    assert state.available_at == AVAILABLE_AT
    assert state.available_at != state.generated_at


def test_market_adapter_uses_call_time_only_when_generated_at_is_absent(monkeypatch) -> None:
    import signals.market_regime.nervous_system_adapter as adapter

    call_time = datetime(2026, 7, 29, 21, 15, tzinfo=UTC)
    monkeypatch.setattr(adapter, "_utc_now", lambda: call_time)

    row = _market_row()
    row.pop("generated_at")
    state = adapt_market_row(row, valid_until=VALID_UNTIL, lineage=LINEAGE)

    assert state.available_at == AVAILABLE_AT
    assert state.generated_at == call_time


@pytest.mark.parametrize(
    "field,value",
    [
        ("available_at", datetime(2026, 7, 29, 20, 30)),
        ("generated_at", datetime(2026, 7, 29, 21, 0)),
    ],
)
def test_market_adapter_rejects_naive_timestamps(field: str, value: object) -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        adapt_market_row(_market_row(**{field: value}), valid_until=VALID_UNTIL, lineage=LINEAGE)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_market_adapter_rejects_nonfinite_metrics(value: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        adapt_market_row(_market_row(risk_appetite_z=value), valid_until=VALID_UNTIL, lineage=LINEAGE)


def test_market_adapter_rejects_nonexclusive_validity_window() -> None:
    with pytest.raises(ValueError, match="valid_until"):
        adapt_market_row(_market_row(), valid_until=AVAILABLE_AT, lineage=LINEAGE)


def test_rule_vector_is_unknown_and_not_a_probability() -> None:
    state = adapt_market_row(_market_row(risk_appetite_z=2.5), valid_until=VALID_UNTIL, lineage=LINEAGE)

    assert state.regime is MarketRegime.UNKNOWN
    assert state.reason_codes == ("MARKET_REGIME_UNCLASSIFIED_RULE_VECTOR",)
    assert state.risk_on_probability is None
    assert state.risk_off_probability is None
    assert state.metrics["risk_appetite_z"] == 2.5


@pytest.mark.parametrize(
    ("ticker", "provided_sector", "expected"),
    [
        ("AAPL", "XLF", "XLK"),
        ("UNKNOWN_TICKER", None, "XLK"),
        (None, "XLF", "XLF"),
    ],
)
def test_sector_adapter_uses_canonical_mapping_and_compatibility_fallback(
    ticker: str | None, provided_sector: str | None, expected: str
) -> None:
    updates: dict[str, object] = {"ticker": ticker}
    if provided_sector is None:
        updates["sector_etf"] = None
    else:
        updates["sector_etf"] = provided_sector

    state = adapt_sector_row(_sector_row(**updates), valid_until=VALID_UNTIL, lineage=LINEAGE)

    assert state.sector_id == expected
    assert state.entity_id == expected


def test_sector_adapter_maps_existing_metrics_without_probability_inference() -> None:
    state = adapt_sector_row(_sector_row(), valid_until=VALID_UNTIL, lineage=LINEAGE)

    assert state.relative_strength == 0.12
    assert state.breadth == 0.5
    assert state.momentum == 0.04
    assert state.rotation_rank == 0.8
    assert state.rank_change == pytest.approx(0.2)
    assert state.transition_probabilities == {}


def test_sector_adapter_rejects_nonfinite_metric() -> None:
    with pytest.raises(ValueError, match="finite"):
        adapt_sector_row(_sector_row(excess_21d=float("nan")), valid_until=VALID_UNTIL, lineage=LINEAGE)


def test_future_appended_state_cannot_enter_1420_or_1620_snapshot() -> None:
    future = adapt_market_row(
        _market_row(
            date=pd.Timestamp("2026-07-30"),
            available_at=pd.Timestamp("2026-07-30 20:30:00", tz="UTC"),
            generated_at=pd.Timestamp("2026-07-30 21:00:00", tz="UTC"),
        ),
        valid_until=datetime(2026, 7, 31, 20, 30, tzinfo=UTC),
        lineage=LINEAGE,
    )

    for decision_time in (
        datetime(2026, 7, 30, 18, 20, tzinfo=UTC),
        datetime(2026, 7, 30, 20, 20, tzinfo=UTC),
    ):
        with pytest.raises(ValueError, match="unavailable at decision time"):
            ContextSnapshot.from_states(
                snapshot_id=__import__("uuid").uuid4(),
                decision_time=decision_time,
                strategy_id="test",
                ticker="AAPL",
                states=(future,),
                freshness_profile="test@1",
            )


class _RecordingStates:
    def __init__(self) -> None:
        self.saved: list[object] = []

    def insert_states_if_absent(self, states):
        self.saved.extend(states)
        return {str(index): state.state_id for index, state in enumerate(states)}


class _RecordingUow:
    def __init__(self) -> None:
        self.states = _RecordingStates()
        self.commits = 0
        self.rollbacks = 0

    def commit(self) -> None:
        self.commits += 1

    def rollback(self) -> None:
        self.rollbacks += 1


def test_persistence_adapts_both_outputs_and_commits_once() -> None:
    uow = _RecordingUow()

    counts = persist_market_regime_outputs(
        pd.DataFrame([_market_row()]),
        pd.DataFrame([_sector_row()]),
        unit_of_work=uow,
        valid_until_for=_valid_until,
    )

    assert counts == (1, 1)
    assert len(uow.states.saved) == 2
    assert uow.commits == 1
    assert uow.rollbacks == 0


def test_persistence_rejects_duplicate_sector_state_keys() -> None:
    uow = _RecordingUow()
    duplicate = pd.DataFrame([_sector_row(), _sector_row(record_locator="sector_state:row:43")])

    with pytest.raises(ValueError, match="duplicate sector"):
        persist_market_regime_outputs(
            pd.DataFrame([_market_row()]),
            duplicate,
            unit_of_work=uow,
            valid_until_for=_valid_until,
        )

    assert uow.states.saved == []
    assert uow.commits == 0


def test_persistence_failure_rolls_back_and_propagates() -> None:
    class FailingUow(_RecordingUow):
        def commit(self) -> None:
            self.commits += 1
            raise RuntimeError("database unavailable")

    uow = FailingUow()
    with pytest.raises(RuntimeError, match="database unavailable"):
        persist_market_regime_outputs(
            pd.DataFrame([_market_row()]),
            pd.DataFrame([_sector_row()]),
            unit_of_work=uow,
            valid_until_for=_valid_until,
        )

    assert uow.commits == 1
    assert uow.rollbacks == 1


def test_build_research_cli_without_uow_does_not_publish(monkeypatch, tmp_path: Path) -> None:
    regime = pd.DataFrame([_market_row()])
    sectors = pd.DataFrame([_sector_row()])
    monkeypatch.setattr(build_module, "build_daily_regime", lambda: regime)
    monkeypatch.setattr(build_module, "build_sector_state", lambda: sectors)
    monkeypatch.setattr(build_module, "persist_market_regime_outputs", pytest.fail)
    monkeypatch.setattr(build_module, "atomic_write_parquet", lambda df, path, **kwargs: Path(path).write_text("parquet"))

    assert build_module.main(
        [
            "--out-regime",
            str(tmp_path / "daily_regime.parquet"),
            "--out-sector-state",
            str(tmp_path / "sector_state.parquet"),
        ]
    ) == 0


def test_post_parquet_persistence_failure_propagates_and_keeps_outputs(monkeypatch, tmp_path: Path) -> None:
    regime_path = tmp_path / "daily_regime.parquet"
    sector_path = tmp_path / "sector_state.parquet"
    monkeypatch.setattr(build_module, "build_daily_regime", lambda: pd.DataFrame([_market_row()]))
    monkeypatch.setattr(build_module, "build_sector_state", lambda: pd.DataFrame([_sector_row()]))

    def write_marker(_df, path, **_kwargs):
        Path(path).write_text("published-parquet")

    monkeypatch.setattr(build_module, "atomic_write_parquet", write_marker)

    def fail_persistence(*_args, **_kwargs):
        raise RuntimeError("database unavailable")

    monkeypatch.setattr(build_module, "persist_market_regime_outputs", fail_persistence)

    with pytest.raises(RuntimeError, match="database unavailable"):
        build_module.main(
            [
                "--out-regime",
                str(regime_path),
                "--out-sector-state",
                str(sector_path),
            ],
            unit_of_work=_RecordingUow(),
            valid_until_for=_valid_until,
        )

    assert regime_path.read_text() == "published-parquet"
    assert sector_path.read_text() == "published-parquet"
