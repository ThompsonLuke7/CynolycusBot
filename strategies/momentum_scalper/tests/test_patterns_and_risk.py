from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

from strategies.momentum_scalper.configs.v1 import PatternConfigV1, RiskConfigV1
from strategies.momentum_scalper.contracts import SetupKind
from strategies.momentum_scalper.patterns.state_machine import PatternStateMachine, hod_breakout
from strategies.momentum_scalper.portfolio.positions import MomentumPosition
from strategies.momentum_scalper.portfolio.risk import DailyRiskState, entry_rejection_reason, size_shares
from strategies.momentum_scalper.execution.entry_policy import evaluate_entry


UTC = timezone.utc


def _history() -> pd.DataFrame:
    return pd.DataFrame([
        {"timestamp": datetime(2026, 8, 10, 8, 0, tzinfo=UTC), "open": 9.0, "high": 9.5, "low": 8.9, "close": 9.4, "volume": 100},
        {"timestamp": datetime(2026, 8, 10, 8, 1, tzinfo=UTC), "open": 9.4, "high": 10.0, "low": 9.3, "close": 9.9, "volume": 100},
        {"timestamp": datetime(2026, 8, 10, 8, 2, tzinfo=UTC), "open": 9.9, "high": 10.2, "low": 9.8, "close": 10.2, "volume": 200},
    ])


def test_hod_breakout_uses_prior_high_not_current_high() -> None:
    signal = hod_breakout(_history(), "RUNR", PatternConfigV1(min_breakout_volume_ratio=1.5))

    assert signal is not None
    assert signal.kind is SetupKind.HOD_BREAKOUT
    assert signal.trigger_price == 10.0
    machine = PatternStateMachine("RUNR", PatternConfigV1(min_breakout_volume_ratio=1.5))
    signals, transitions = machine.observe(_history())
    assert signals[0].trigger_price == 10.0
    assert any(item.current.value == "TRIGGERED" for item in transitions)


def test_position_partials_then_uses_breakeven_and_red_exit() -> None:
    cfg = RiskConfigV1(partial_at_r=2.0, partial_fraction=0.5, max_hold_minutes=30)
    pos = MomentumPosition.from_fill(
        ticker="RUNR", opened_at=datetime(2026, 8, 10, 8, 0, tzinfo=UTC),
        entry_price=10.0, stop_price=9.0, quantity=10, config=cfg,
    )
    target = pos.process_bar(timestamp=datetime(2026, 8, 10, 8, 1, tzinfo=UTC), open_price=10.0, high=12.0, low=10.0, close=11.9, config=cfg)
    assert target[0].reason == "partial_at_target"
    pos.apply_exit(quantity=5, price=12.0)
    assert pos.remaining_quantity == 5 and pos.partial_taken
    red = pos.process_bar(timestamp=datetime(2026, 8, 10, 8, 2, tzinfo=UTC), open_price=12.0, high=12.1, low=10.5, close=11.0, config=cfg)
    assert red[0].reason == "first_red_after_extension"


def test_risk_size_and_fuses_are_bounded() -> None:
    cfg = RiskConfigV1(max_position_notional=1_000, max_quote_participation=0.1)
    assert size_shares(entry_price=10, stop_price=9, risk_budget_dollars=500, ask_size=40, config=cfg) == 4
    assert entry_rejection_reason(DailyRiskState(realized_r=-3.0), cfg) == "max_daily_loss"


def test_legacy_dataframe_entry_path_fails_closed() -> None:
    with pytest.raises(RuntimeError, match="retired"):
        evaluate_entry(pd.Series(dtype=float))
