"""Holiday/closure-aware option expiry guard.

Force-close any option whose expiry passes before the next tradable session, so
contracts don't auto-exercise into assigned equity over a weekend/holiday (the
2026-06-18 Juneteenth incident).
"""
from __future__ import annotations

from datetime import date, datetime
from zoneinfo import ZoneInfo

from core.calendar import next_trading_day
from strategies.multi_ticker_swing.live import position_manager as pm

_ET = ZoneInfo("America/New_York")


class _Pos:
    def __init__(self, option_symbol: str):
        self.option_symbol = option_symbol


def _bar(dt: datetime, close: float = 85.0) -> dict:
    return {"timestamp": dt, "close": close}


def test_calendar_skips_juneteenth_weekend():
    # Thu 2026-06-18 -> next session is Mon 2026-06-22 (Fri 6/19 Juneteenth).
    assert next_trading_day(date(2026, 6, 18)) == date(2026, 6, 22)


def test_closes_contract_expiring_into_holiday_weekend():
    pos = _Pos("ASTS260619C00080000")  # expiry Fri 6/19 (holiday)
    reason = pm._expiring_before_closure_exit_reason(pos, _bar(datetime(2026, 6, 18, 15, 50, tzinfo=_ET)))
    assert reason == "expiring_before_closure"


def test_does_not_close_before_eod_cutoff():
    pos = _Pos("ASTS260619C00080000")
    reason = pm._expiring_before_closure_exit_reason(pos, _bar(datetime(2026, 6, 18, 10, 0, tzinfo=_ET)))
    assert reason is None


def test_does_not_close_contract_that_survives_the_closure():
    pos = _Pos("ASTS260626C00080000")  # a week out
    reason = pm._expiring_before_closure_exit_reason(pos, _bar(datetime(2026, 6, 18, 15, 50, tzinfo=_ET)))
    assert reason is None


def test_closes_zero_dte_near_close():
    pos = _Pos("ASTS260622C00080000")  # expires today (Mon)
    reason = pm._expiring_before_closure_exit_reason(pos, _bar(datetime(2026, 6, 22, 15, 50, tzinfo=_ET)))
    assert reason == "expiring_before_closure"


def test_force_close_uses_liquidation_path():
    assert "expiring_before_closure" in pm._LIQUIDATION_CLOSE_REASONS
