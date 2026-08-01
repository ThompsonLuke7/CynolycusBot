"""Causal publication tests for market-regime and sector-state outputs."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core.nervous_system.contracts.context import ContextSnapshot
from core.nervous_system.contracts.enums import MarketRegime
from core.nervous_system.contracts.quality import LineageRef
from signals.market_regime import build as build_module
from signals.market_regime.daily_regime import build_daily_regime
from signals.market_regime.nervous_system_adapter import (
    FINITE_ROW_PUBLICATION_POLICY,
    MARKET_REQUIRED_METRICS,
    SECTOR_REQUIRED_METRICS,
    adapt_market_row,
    adapt_sector_row,
    persist_market_regime_outputs,
)
from signals.market_regime.sector_state import build_sector_state
from signals.market_regime.tests.conftest import build_full_universe_bars, make_loader


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


def _with_lineage_attrs(frame: pd.DataFrame, *, table_name: str, content_hash: str) -> pd.DataFrame:
    frame = frame.copy()
    frame.attrs["source_id"] = f"{table_name}-artifact"
    frame.attrs["content_hash"] = content_hash
    frame.attrs["record_locators"] = {
        index: f"{table_name}:row:{index}" for index in frame.index
    }
    return frame


def _producer_market_frame(*, rows: int = 3) -> pd.DataFrame:
    values: dict[str, object] = {
        "date": [
            pd.Timestamp("2026-07-29") + pd.Timedelta(days=index)
            for index in range(rows)
        ],
        "available_at": [
            pd.Timestamp("2026-07-29 20:30:00", tz="UTC") + pd.Timedelta(days=index)
            for index in range(rows)
        ],
    }
    for metric_index, metric in enumerate(MARKET_REQUIRED_METRICS):
        values[metric] = [float(metric_index + index + 1) for index in range(rows)]
    return _with_lineage_attrs(
        pd.DataFrame(values),
        table_name="daily_regime",
        content_hash="c" * 64,
    )


def _producer_sector_frame(*, rows_per_sector: int = 3) -> pd.DataFrame:
    values: list[dict[str, object]] = []
    for sector_index, sector in enumerate(("XLK", "XLF")):
        for row_index in range(rows_per_sector):
            row: dict[str, object] = {
                "date": pd.Timestamp("2026-07-29") + pd.Timedelta(days=row_index),
                "sector_etf": sector,
                "available_at": pd.Timestamp("2026-07-29 20:30:00", tz="UTC")
                + pd.Timedelta(days=row_index),
            }
            row.update(
                {
                    metric: float(sector_index + row_index + metric_index + 1)
                    for metric_index, metric in enumerate(SECTOR_REQUIRED_METRICS)
                }
            )
            values.append(row)
    return _with_lineage_attrs(
        pd.DataFrame(values),
        table_name="sector_state",
        content_hash="d" * 64,
    )


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


@pytest.mark.parametrize("value", [True, np.bool_(True)])
def test_market_adapter_rejects_boolean_metrics(value: object) -> None:
    with pytest.raises(ValueError, match="finite numeric|boolean"):
        adapt_market_row(_market_row(risk_appetite_z=value), valid_until=VALID_UNTIL, lineage=LINEAGE)


def test_adapter_identity_is_stable_across_call_time_and_revised_source(monkeypatch) -> None:
    import signals.market_regime.nervous_system_adapter as adapter

    row = _market_row()
    row.pop("generated_at")
    first_call = datetime(2026, 7, 29, 21, 15, tzinfo=UTC)
    second_call = datetime(2026, 7, 30, 21, 15, tzinfo=UTC)
    monkeypatch.setattr(adapter, "_utc_now", lambda: first_call)
    first = adapt_market_row(row, valid_until=VALID_UNTIL, lineage=LINEAGE)
    monkeypatch.setattr(adapter, "_utc_now", lambda: second_call)
    rerun = adapt_market_row(row, valid_until=VALID_UNTIL, lineage=LINEAGE)

    assert first.generated_at == first_call
    assert rerun.generated_at == second_call
    assert rerun.state_id == first.state_id

    revised = adapt_market_row(
        row,
        valid_until=VALID_UNTIL,
        lineage=(
            LineageRef(
                source_id=LINEAGE[0].source_id,
                content_hash="e" * 64,
                record_locator=LINEAGE[0].record_locator,
            ),
        ),
    )
    assert revised.state_id != first.state_id


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


def test_required_metric_policy_matches_actual_producer_columns() -> None:
    _, bars = build_full_universe_bars(start="2021-01-04", n=25, seed_base=401)
    loader = make_loader(bars)
    regime = build_daily_regime(
        loader=loader,
        zscore_window=5,
        zscore_min_periods=5,
        component_window_short=3,
        component_window_long=5,
        excess_return_window=3,
    )
    sectors = build_sector_state(
        loader=loader,
        component_window_short=3,
        component_window_long=5,
        excess_return_window=3,
        excess_return_window_long=6,
    )

    assert set(regime.columns) - {"date", "available_at"} == set(MARKET_REQUIRED_METRICS)
    assert set(sectors.columns) - {"date", "sector_etf", "available_at"} == set(
        SECTOR_REQUIRED_METRICS
    )
    assert FINITE_ROW_PUBLICATION_POLICY


def test_real_producer_warmup_prefix_is_skipped_and_counts_are_deterministic() -> None:
    _, bars = build_full_universe_bars(start="2021-01-04", n=25, seed_base=402)
    loader = make_loader(bars)
    regime = _with_lineage_attrs(
        build_daily_regime(
            loader=loader,
            zscore_window=5,
            zscore_min_periods=5,
            component_window_short=3,
            component_window_long=5,
            excess_return_window=3,
        ),
        table_name="daily_regime",
        content_hash="f" * 64,
    )
    sectors = _with_lineage_attrs(
        build_sector_state(
            loader=loader,
            component_window_short=3,
            component_window_long=5,
            excess_return_window=3,
            excess_return_window_long=6,
        ),
        table_name="sector_state",
        content_hash="1" * 64,
    )

    market_publishable = regime[list(MARKET_REQUIRED_METRICS)].notna().all(axis=1)
    expected_market = int(market_publishable.sum())
    expected_sector = sum(
        int(group[list(SECTOR_REQUIRED_METRICS)].notna().all(axis=1).sum())
        for _, group in sectors.groupby("sector_etf", sort=False)
    )
    uow = _RecordingUow()
    counts = persist_market_regime_outputs(
        regime,
        sectors,
        unit_of_work=uow,
        valid_until_for=lambda available_at: available_at + timedelta(days=2),
    )

    assert counts == (expected_market, expected_sector)
    assert len(uow.states.saved) == expected_market + expected_sector
    assert expected_market < len(regime)
    assert expected_sector < len(sectors)


def test_post_warmup_nonfinite_gap_fails_for_market_and_sector_entities() -> None:
    market = _producer_market_frame()
    market.loc[0, list(MARKET_REQUIRED_METRICS)] = np.nan
    market.loc[2, MARKET_REQUIRED_METRICS[0]] = np.nan
    with pytest.raises(ValueError, match=FINITE_ROW_PUBLICATION_POLICY):
        persist_market_regime_outputs(
            market,
            _producer_sector_frame(rows_per_sector=1),
            unit_of_work=_RecordingUow(),
            valid_until_for=lambda available_at: available_at + timedelta(days=2),
        )

    sector = _producer_sector_frame()
    sector.loc[
        (sector["sector_etf"] == "XLF") & (sector["date"] == pd.Timestamp("2026-07-30")),
        SECTOR_REQUIRED_METRICS[0],
    ] = np.nan
    with pytest.raises(ValueError, match=FINITE_ROW_PUBLICATION_POLICY):
        persist_market_regime_outputs(
            _producer_market_frame(rows=1),
            sector,
            unit_of_work=_RecordingUow(),
            valid_until_for=lambda available_at: available_at + timedelta(days=2),
        )


def test_persistence_passes_exact_utc_available_at_to_validity_policy() -> None:
    received: list[datetime] = []

    def valid_until_for(available_at: datetime) -> datetime:
        received.append(available_at)
        assert available_at.tzinfo is UTC
        assert type(available_at) is datetime
        return available_at + timedelta(days=1)

    regime = _producer_market_frame(rows=1)
    sectors = _producer_sector_frame(rows_per_sector=1).iloc[:1].copy()
    persist_market_regime_outputs(
        regime,
        sectors,
        unit_of_work=_RecordingUow(),
        valid_until_for=valid_until_for,
    )

    assert received == [AVAILABLE_AT, AVAILABLE_AT]


def test_missing_lineage_metadata_fails_closed_without_synthetic_fallback() -> None:
    regime = _producer_market_frame(rows=1)
    regime.attrs = {}
    with pytest.raises(ValueError, match="exact.*lineage|artifact hash|record locator") as exc_info:
        persist_market_regime_outputs(
            regime,
            pd.DataFrame([_sector_row()]),
            unit_of_work=_RecordingUow(),
            valid_until_for=_valid_until,
        )
    assert "synthetic" not in str(exc_info.value).lower()


def test_lineage_attrs_round_trip_exact_artifact_hash_and_row_locator() -> None:
    regime = _producer_market_frame(rows=1)
    uow = _RecordingUow()
    persist_market_regime_outputs(
        regime,
        _producer_sector_frame(rows_per_sector=1).iloc[:1].copy(),
        unit_of_work=uow,
        valid_until_for=lambda available_at: available_at + timedelta(days=1),
    )

    assert "c" * 64 in uow.states.saved[0].lineage_ids[0]
    assert "daily_regime:row:0" in uow.states.saved[0].lineage_ids[0]


def test_repeated_publication_is_idempotent_and_revised_artifact_adds_evidence(monkeypatch) -> None:
    import signals.market_regime.nervous_system_adapter as adapter

    class IdempotentStates(_RecordingStates):
        def __init__(self) -> None:
            super().__init__()
            self.by_id: dict[object, object] = {}

        def insert_states_idempotently(self, states):
            for state in states:
                self.by_id.setdefault(state.state_id, state)
            self.saved = list(self.by_id.values())

    uow = _RecordingUow()
    uow.states = IdempotentStates()
    regime = _producer_market_frame(rows=1)
    sectors = _producer_sector_frame(rows_per_sector=1).iloc[:1].copy()
    monkeypatch.setattr(
        adapter,
        "_utc_now",
        lambda: datetime(2026, 8, 1, 12, 0, tzinfo=UTC),
    )
    persist_market_regime_outputs(
        regime,
        sectors,
        unit_of_work=uow,
        valid_until_for=lambda available_at: available_at + timedelta(days=1),
    )
    monkeypatch.setattr(
        adapter,
        "_utc_now",
        lambda: datetime(2026, 8, 2, 12, 0, tzinfo=UTC),
    )
    persist_market_regime_outputs(
        regime,
        sectors,
        unit_of_work=uow,
        valid_until_for=lambda available_at: available_at + timedelta(days=1),
    )
    assert len(uow.states.by_id) == 2

    revised_regime = regime.copy()
    revised_regime.attrs = dict(regime.attrs)
    revised_regime.attrs["content_hash"] = "e" * 64
    revised_regime.attrs["record_locators"] = {0: "daily_regime:row:0"}
    persist_market_regime_outputs(
        revised_regime,
        sectors,
        unit_of_work=uow,
        valid_until_for=lambda available_at: available_at + timedelta(days=1),
    )
    assert len(uow.states.by_id) == 3


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

    def insert_states_idempotently(self, states):
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


def test_persistence_adapts_both_outputs_without_owning_transaction() -> None:
    uow = _RecordingUow()

    counts = persist_market_regime_outputs(
        _producer_market_frame(rows=1),
        _producer_sector_frame(rows_per_sector=1).iloc[:1].copy(),
        unit_of_work=uow,
        valid_until_for=_valid_until,
    )

    assert counts == (1, 1)
    assert len(uow.states.saved) == 2
    assert uow.commits == 0
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


def test_persistence_failure_propagates_for_caller_rollback() -> None:
    class FailingUow(_RecordingUow):
        class FailingStates(_RecordingStates):
            def insert_states_idempotently(self, states):
                raise RuntimeError("database unavailable")

        def __init__(self) -> None:
            super().__init__()
            self.states = self.FailingStates()

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

    assert uow.commits == 0
    assert uow.rollbacks == 0


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


def test_build_boundary_attaches_exact_artifact_hash_and_original_row_locators(
    monkeypatch, tmp_path: Path
) -> None:
    regime_path = tmp_path / "daily_regime.parquet"
    sector_path = tmp_path / "sector_state.parquet"
    regime = pd.DataFrame([_market_row()])
    sectors = pd.DataFrame([_sector_row()])
    captured: dict[str, object] = {}

    monkeypatch.setattr(build_module, "build_daily_regime", lambda: regime)
    monkeypatch.setattr(build_module, "build_sector_state", lambda: sectors)
    monkeypatch.setattr(
        build_module,
        "atomic_write_parquet",
        lambda _df, path, **_kwargs: Path(path).write_bytes(b"parquet-artifact"),
    )

    def capture(regime_frame, sector_frame, **kwargs):
        captured["regime"] = regime_frame
        captured["sector"] = sector_frame
        captured["kwargs"] = kwargs

    monkeypatch.setattr(build_module, "persist_market_regime_outputs", capture)
    build_module.main(
        ["--out-regime", str(regime_path), "--out-sector-state", str(sector_path)],
        unit_of_work=_RecordingUow(),
        valid_until_for=_valid_until,
    )

    expected_hash = __import__("hashlib").sha256(b"parquet-artifact").hexdigest()
    published_regime = captured["regime"]
    published_sector = captured["sector"]
    assert published_regime.attrs["content_hash"] == expected_hash
    assert published_sector.attrs["content_hash"] == expected_hash
    assert published_regime.attrs["record_locators"] == {0: "daily_regime:row:0"}
    assert published_sector.attrs["record_locators"] == {0: "sector_state:row:0"}


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
