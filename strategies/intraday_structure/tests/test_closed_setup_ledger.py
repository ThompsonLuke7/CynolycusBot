from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from strategies.intraday_structure.config import IntradayStructureConfig, ReplayPolicy
from strategies.intraday_structure.engine import IntradayStructureEngine
from strategies.intraday_structure.ledger import (
    LEDGERED_FLAG,
    build_closed_setup_record,
    ledger_sink,
)
from strategies.intraday_structure.models import (
    Bar, Candidate, Direction, SetupRecord, SetupState, SetupType,
)


NOW = datetime(2026, 8, 25, 14, 0, tzinfo=timezone.utc)


def _setup(**overrides) -> SetupRecord:
    candidate = Candidate("XYZ", NOW, Direction.LONG, ("meta_ranker",), score=0.8)
    defaults = dict(
        setup_id="XYZ:long:breakout_continuation", ticker="XYZ",
        setup_type=SetupType.BREAKOUT, direction=Direction.LONG, candidate=candidate,
        state=SetupState.INVALIDATED, entry_price=100.0, entry_time=NOW,
        updated_at=NOW + timedelta(minutes=12), targets=[103.0], invalidation=99.0,
        confidence=0.7, runway_score=0.61, expected_reward_risk=1.8,
        max_favorable_excursion=1.4, max_adverse_excursion=1.0, bars_alive=12,
    )
    defaults.update(overrides)
    record = SetupRecord(**defaults)
    record.metadata.setdefault("initial_invalidation", 99.0)
    record.metadata.setdefault("exit_price", 99.0)
    record.metadata.setdefault("exit_reason", "invalidation touched")
    return record


def _policy() -> ReplayPolicy:
    return ReplayPolicy(spread_bps=8.0, slippage_bps=4.0, commission_per_share=0.0)


def test_record_prices_the_round_trip_and_stamps_its_cost_assumptions() -> None:
    record = build_closed_setup_record(_setup(), replay_policy=_policy(), engine_version="v")
    assert record is not None
    # Long 100 -> 99 is -1.00 gross; 12bps of the 100 entry is another -0.12.
    assert record.gross_points == pytest.approx(-1.0)
    assert record.net_points == pytest.approx(-1.12)
    assert record.net_return == pytest.approx(-0.0112)
    assert record.risk_points == pytest.approx(1.0)
    assert record.realized_r_after_costs == pytest.approx(-1.12)
    # A row must carry the assumptions it was priced under, so changing the
    # config later cannot silently re-price history.
    assert (record.cost_spread_bps, record.cost_slippage_bps) == (8.0, 4.0)


def test_a_short_is_signed_the_other_way() -> None:
    setup = _setup(direction=Direction.SHORT, invalidation=101.0)
    setup.metadata["initial_invalidation"] = 101.0
    setup.metadata["exit_price"] = 98.0
    record = build_closed_setup_record(setup, replay_policy=_policy(), engine_version="v")
    assert record.gross_points == pytest.approx(2.0)
    assert record.net_points == pytest.approx(1.88)


def test_a_setup_that_never_entered_is_not_a_trade() -> None:
    # Confirmed then closed before its entry bar arrived. Giving it a synthetic
    # entry would fabricate P&L, so it must produce no row at all.
    assert build_closed_setup_record(
        _setup(entry_price=None), replay_policy=_policy(), engine_version="v"
    ) is None


def test_the_ledger_records_the_decision_inputs_the_ablation_needs() -> None:
    setup = _setup()
    setup.metadata["target_level_type"] = "prior_day_high+options_call_wall"
    setup.metadata["target_level_sources"] = ["prior_day_high", "options_call_wall"]
    setup.metadata["dealer_plate"] = {"qualified": True, "score": 0.82}
    record = build_closed_setup_record(setup, replay_policy=_policy(), engine_version="v")
    assert record.target_level_sources == ["prior_day_high", "options_call_wall"]
    assert record.dealer_plate_qualified is True
    assert record.dealer_plate_score == pytest.approx(0.82)
    assert record.candidate_sources == ["meta_ranker"]
    # The five clocks stay distinct.
    assert record.candidate_timestamp is not None and record.entry_time is not None
    assert record.exit_time != record.entry_time


# --------------------------------------------------------------------------
# Engine-level: one row per closed setup, and no duplicates across archival.
# --------------------------------------------------------------------------

def _engine(rows: list) -> IntradayStructureEngine:
    config = IntradayStructureConfig(enabled=True, min_average_dollar_volume=0.0)
    return IntradayStructureEngine(config, ledger_sink=rows.append)


def test_invalidated_then_archived_to_closed_emits_exactly_one_row() -> None:
    rows: list = []
    engine = _engine(rows)
    setup = _setup()
    engine.setups[setup.setup_id] = setup
    setup.state = SetupState.RUNNING
    bar = Bar("XYZ", NOW + timedelta(minutes=12), 99.5, 99.6, 98.9, 99.0, 1000)

    engine._transition(setup, SetupState.INVALIDATED, bar, "invalidation touched", ("stop",))
    assert len(rows) == 1
    assert rows[0].terminal_state == "INVALIDATED"

    # The engine archives INVALIDATED -> CLOSED on a later bar. Keying the
    # ledger off CLOSED would double-count AND lose the real exit reason.
    engine._transition(setup, SetupState.CLOSED, bar, "terminal state archived", ("closed",))
    assert len(rows) == 1


def test_a_restart_cannot_re_emit_a_row_already_written() -> None:
    rows: list = []
    engine = _engine(rows)
    setup = _setup(state=SetupState.RUNNING)
    # The guard rides in metadata, so it survives state.json round-tripping.
    restored = SetupRecord.from_mapping(setup.to_dict())
    assert LEDGERED_FLAG not in restored.metadata

    engine.setups[setup.setup_id] = setup
    bar = Bar("XYZ", NOW + timedelta(minutes=12), 99.5, 99.6, 98.9, 99.0, 1000)
    engine._transition(setup, SetupState.INVALIDATED, bar, "invalidation touched", ("stop",))
    assert len(rows) == 1
    assert setup.metadata[LEDGERED_FLAG]

    survivor = SetupRecord.from_mapping(setup.to_dict())
    engine.setups[survivor.setup_id] = survivor
    engine._transition(survivor, SetupState.CLOSED, bar, "terminal state archived", ("closed",))
    assert len(rows) == 1


def test_a_failing_sink_cannot_kill_the_runner_thread() -> None:
    def explode(_record):
        raise OSError("disk full")

    config = IntradayStructureConfig(enabled=True, min_average_dollar_volume=0.0)
    engine = IntradayStructureEngine(config, ledger_sink=explode)
    setup = _setup(state=SetupState.RUNNING)
    engine.setups[setup.setup_id] = setup
    bar = Bar("XYZ", NOW + timedelta(minutes=12), 99.5, 99.6, 98.9, 99.0, 1000)
    engine._transition(setup, SetupState.INVALIDATED, bar, "invalidation touched", ("stop",))
    assert setup.state == SetupState.INVALIDATED


def test_sink_writes_one_json_object_per_line(tmp_path) -> None:
    import json

    path = tmp_path / "closed_setups.jsonl"
    sink = ledger_sink(path)
    record = build_closed_setup_record(_setup(), replay_policy=_policy(), engine_version="v")
    sink(record)
    sink(record)
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["setup_id"] == "XYZ:long:breakout_continuation"


def test_the_runner_points_both_sinks_at_the_configured_paths(tmp_path) -> None:
    """Wiring test: the module is only measurable if the live path writes."""
    import queue
    from dataclasses import replace as dc_replace

    from strategies.intraday_structure.runner import IntradayStructureRunner

    config = dc_replace(
        IntradayStructureConfig(enabled=True),
        state_path=str(tmp_path / "state.json"),
        transition_log_path=str(tmp_path / "transitions.jsonl"),
        signal_path=str(tmp_path / "signals.json"),
        ledger_path=str(tmp_path / "closed_setups.jsonl"),
        abstention_path=str(tmp_path / "abstentions.jsonl"),
    )
    runner = IntradayStructureRunner(config, queue.Queue())
    assert runner.engine.ledger_sink is not None
    assert runner.engine.abstention_sink is not None

    record = build_closed_setup_record(_setup(), replay_policy=config.replay, engine_version=config.version)
    runner.engine.ledger_sink(record)
    assert (tmp_path / "closed_setups.jsonl").exists()
