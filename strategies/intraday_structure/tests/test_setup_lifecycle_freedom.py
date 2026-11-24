"""Regression: a setup slot must never be held hostage.

Setup ids are unique per ticker/direction/detector, so a setup that cannot
progress blocks every later setup of its kind on that ticker. On 2026-11-23:

* 102 pre-entry setups from earlier sessions (some ARMED since July) held the
  slots of 53 tickers. QQQ's short structural rejection had been ARMED since
  11-12 at a 694 pivot, so nothing could short the 652 resistance.
* Revival ran only on candidate registration (about once a session), so a setup
  that closed at 09:50 stayed dark all day: 6.2% same-session re-arm.
* The "too old to enter" clock counted the hours an empty WATCHING shell had
  waited, so afternoon confirmations were refused.
* ``bars_held`` in the ledger counted that same pre-entry life.

Detectors are stubbed so each test drives exactly one lifecycle rule.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

import strategies.intraday_structure.engine as engine_module
from strategies.intraday_structure.config import IntradayStructureConfig, TargetPolicy
from strategies.intraday_structure.detectors.base import DetectionDecision
from strategies.intraday_structure.engine import IntradayStructureEngine
from strategies.intraday_structure.models import Bar, Candidate, Direction, SetupState, SetupType
from strategies.intraday_structure.target_manager import INVALIDATION_TOO_WIDE, TargetPlanOutcome


DAY1 = datetime(2026, 11, 20, 15, 0, tzinfo=timezone.utc)   # Fri 10:00 ET
DAY2 = datetime(2026, 11, 23, 14, 30, tzinfo=timezone.utc)  # Mon 09:30 ET
SETUP_ID = "XYZ:long:gamma_structural_level_rejection"


class _Idle:
    def evaluate(self, setup, ctx):  # noqa: ARG002
        return DetectionDecision()


class _ConfirmWhenArmed:
    def evaluate(self, setup, ctx):  # noqa: ARG002
        if setup.state == SetupState.ARMED:
            return DetectionDecision(SetupState.CONFIRMED, "CONFIRM", "stub confirmation")
        return DetectionDecision()


class _ArmFromWatching:
    def evaluate(self, setup, ctx):  # noqa: ARG002
        if setup.state == SetupState.WATCHING:
            return DetectionDecision(SetupState.ARMED, "ARM", "stub arm", pivot=ctx.bar.close - 1.0)
        return DetectionDecision()


def _engine(*, max_setup_bars: int = 180, **overrides) -> IntradayStructureEngine:
    config = IntradayStructureConfig(
        enabled=True, min_average_dollar_volume=0.0, failure_cooldown_bars=5,
        target=TargetPolicy(max_setup_bars=max_setup_bars), **overrides,
    )
    engine = IntradayStructureEngine(config)
    engine._detectors = {setup_type: _Idle() for setup_type in engine._detectors}
    engine._exhaustion = _Idle()
    return engine


def _candidate(at: datetime) -> Candidate:
    return Candidate(ticker="XYZ", timestamp=at, direction=Direction.LONG,
                     sources=("high_liquidity_universe",), score=0.5)


def _bar(at: datetime, price: float = 100.0) -> Bar:
    return Bar("XYZ", at, price, price + 0.2, price - 0.2, price, 10_000.0)


def _feed(engine: IntradayStructureEngine, start: datetime, n: int, *, price: float = 100.0) -> datetime:
    for i in range(n):
        engine.on_bar(_bar(start + timedelta(minutes=i), price))
    return start + timedelta(minutes=n)


def _warm(engine: IntradayStructureEngine, start: datetime = DAY1) -> datetime:
    engine.register_candidate(_candidate(start))
    return _feed(engine, start, 25)  # past min_history_bars, so the detector loop runs


def _reasons(engine: IntradayStructureEngine) -> list[str]:
    return [t.reason for t in engine.transitions if t.setup_id == SETUP_ID]


# --- revival on the bar path ------------------------------------------------------

def test_a_closed_setup_revives_mid_session_without_re_registration() -> None:
    engine = _engine()
    t = _warm(engine)
    setup = engine.setups[SETUP_ID]
    setup.state = SetupState.CLOSED
    setup.metadata["terminal_bar_count"] = engine._bar_counts["XYZ"]

    t = _feed(engine, t, 4)
    assert engine.setups[SETUP_ID].state == SetupState.CLOSED, "the failure cooldown still applies"
    _feed(engine, t, 2)
    revived = engine.setups[SETUP_ID]
    assert revived.state == SetupState.WATCHING
    assert revived.created_at > DAY1, "a revived shell is born at the bar that revived it"


def test_a_pruned_setup_is_recreated_on_the_next_bar() -> None:
    engine = _engine()
    t = _warm(engine)
    del engine.setups[SETUP_ID]  # what prune_settled_state does to a cooled CLOSED setup
    _feed(engine, t, 1)
    assert engine.setups[SETUP_ID].state == SetupState.WATCHING


# --- session reset ------------------------------------------------------------------

def test_a_pre_entry_setup_does_not_survive_the_night() -> None:
    engine = _engine()
    _warm(engine)
    setup = engine.setups[SETUP_ID]
    setup.state = SetupState.ARMED
    setup.state_entered_at = DAY1
    setup.pivot = 694.0

    engine.register_candidate(_candidate(DAY2 - timedelta(hours=5)))  # the daily re-seed
    assert engine.setups[SETUP_ID].state == SetupState.ARMED, "registration alone must not reset it"
    _feed(engine, DAY2, 1)

    assert "carried over from a prior session" in _reasons(engine)
    fresh = engine.setups[SETUP_ID]
    assert fresh.state == SetupState.WATCHING and fresh.pivot is None


def test_a_stale_watching_shell_is_replaced_silently() -> None:
    engine = _engine()
    _warm(engine)
    engine.setups[SETUP_ID].bars_alive = 5_000
    engine.register_candidate(_candidate(DAY2 - timedelta(hours=5)))
    before = len(engine.transitions)
    _feed(engine, DAY2, 1)
    assert engine.setups[SETUP_ID].bars_alive <= 1
    assert all(t.setup_id != SETUP_ID for t in engine.transitions[before:]), "no transition for an empty shell"


def test_a_running_position_is_not_touched_by_the_session_reset() -> None:
    engine = _engine()
    _warm(engine)
    setup = engine.setups[SETUP_ID]
    setup.state = SetupState.RUNNING
    setup.entry_price, setup.entry_time, setup.state_entered_at = 100.0, DAY1, DAY1
    setup.invalidation = 90.0
    engine.register_candidate(_candidate(DAY2 - timedelta(hours=5)))
    _feed(engine, DAY2, 1)
    assert "carried over from a prior session" not in _reasons(engine)


# --- pre-entry staleness ------------------------------------------------------------

def test_an_armed_setup_ages_out_and_frees_its_slot() -> None:
    engine = _engine(max_setup_bars=10)
    engine._detectors[SetupType.STRUCTURAL_REJECTION] = _ArmFromWatching()
    t = _warm(engine)  # armed on bar 20, the first evaluated bar
    assert engine.setups[SETUP_ID].state == SetupState.ARMED
    engine._detectors[SetupType.STRUCTURAL_REJECTION] = _Idle()

    t = _feed(engine, t, 4)  # bar 29: 9 bars since detection
    assert engine.setups[SETUP_ID].state == SetupState.ARMED
    t = _feed(engine, t, 1)  # bar 30: 10 bars
    assert engine.setups[SETUP_ID].state == SetupState.CLOSED
    assert engine.transitions[-1].evidence == ("stale_pre_entry",)
    _feed(engine, t, 5)
    assert engine.setups[SETUP_ID].state == SetupState.WATCHING, "slot comes back after the cooldown"


def test_the_staleness_clock_starts_at_detection_not_at_shell_creation() -> None:
    """The old clock refused an afternoon entry because the shell was created pre-market."""
    engine = _engine(max_setup_bars=10)
    t = _warm(engine)
    setup = engine.setups[SETUP_ID]
    setup.bars_alive = 400  # an empty shell that waited all morning
    engine._detectors[SetupType.STRUCTURAL_REJECTION] = _ArmFromWatching()
    t = _feed(engine, t, 1)
    assert setup.state == SetupState.ARMED
    assert engine._pre_entry_age(setup) == 0

    setup.state = SetupState.CONFIRMED  # confirmed on the detection bar
    engine._detectors[SetupType.STRUCTURAL_REJECTION] = _Idle()
    _feed(engine, t, 1)
    assert setup.state == SetupState.RUNNING, "a fresh thesis must be entered, whatever the shell's age"


# --- out-of-reach pivots ------------------------------------------------------------

@pytest.mark.parametrize("risk_atr, closes", [(30.0, True), (3.0, False)])
def test_a_confirmation_refused_far_from_its_pivot_closes_the_setup(monkeypatch, risk_atr, closes) -> None:
    monkeypatch.setattr(engine_module, "build_target_plan", lambda setup, ctx: TargetPlanOutcome(
        reason=INVALIDATION_TOO_WIDE, proposed_invalidation=694.11,
        proposed_risk_atr=risk_atr, max_invalidation_atr=2.0,
    ))
    engine = _engine()
    t = _warm(engine)
    engine.setups[SETUP_ID].state = SetupState.ARMED
    engine._detectors[SetupType.STRUCTURAL_REJECTION] = _ConfirmWhenArmed()
    _feed(engine, t, 1)
    setup = engine.setups[SETUP_ID]
    if closes:
        assert setup.state == SetupState.CLOSED
        assert "pivot_out_of_reach" in engine.transitions[-1].evidence
        assert setup.metadata["terminal_bar_count"] == engine._bar_counts["XYZ"], "cooldown applies"
    else:
        assert setup.state == SetupState.ARMED, "a marginal refusal keeps waiting"


# --- ledger holding period ----------------------------------------------------------

def test_bars_held_counts_from_entry_not_from_the_shell() -> None:
    records = []
    engine = _engine()
    engine.ledger_sink = records.append
    t = _warm(engine)
    setup = engine.setups[SETUP_ID]
    setup.bars_alive = 300
    setup.state = SetupState.CONFIRMED
    setup.metadata["detected_bar_count"] = engine._bar_counts["XYZ"]
    setup.invalidation = 95.0
    t = _feed(engine, t, 1)  # entry bar
    assert setup.state == SetupState.RUNNING
    t = _feed(engine, t, 3)
    from strategies.intraday_structure.models import PriceUpdate
    engine.on_price_update(PriceUpdate("XYZ", t, 90.0, "trade"))
    assert setup.state == SetupState.INVALIDATED
    assert records and records[-1].bars_held == 3
