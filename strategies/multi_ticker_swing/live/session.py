from __future__ import annotations

from datetime import datetime, time as time_of_day
from zoneinfo import ZoneInfo

from core.calendar import close_relative_time


ET = ZoneInfo("America/New_York")

RTH_START = time_of_day(9, 30)
RTH_END = time_of_day(16, 0)
LOG_END = time_of_day(16, 15)
CONFIRM_START = time_of_day(10, 0)
CONFIRM_END = time_of_day(15, 55)
SCAN_END_TS = time_of_day(15, 55)


def et_time(ts: datetime | None) -> time_of_day | None:
    if ts is None or not hasattr(ts, "astimezone"):
        return None
    return ts.astimezone(ET).time()


def _session_time(t: time_of_day, ts: datetime) -> time_of_day:
    """`t` (written for a 16:00 close) moved to the same distance before `ts`'s
    session close, so a 13:00 half-day ends every window three hours early."""
    return close_relative_time(t, ts.astimezone(ET).date())


def is_regular_trading_time(ts: datetime | None) -> bool:
    local_time = et_time(ts)
    return local_time is not None and RTH_START <= local_time < _session_time(RTH_END, ts)


def is_log_window(ts: datetime | None) -> bool:
    local_time = et_time(ts)
    return local_time is not None and RTH_START <= local_time < _session_time(LOG_END, ts)


def should_check_confirmation(ts: datetime | None) -> bool:
    local_time = et_time(ts)
    return local_time is None or CONFIRM_START <= local_time <= _session_time(CONFIRM_END, ts)


def should_scan_after_30m_close(ts: datetime | None) -> bool:
    local_time = et_time(ts)
    return local_time is None or local_time < _session_time(SCAN_END_TS, ts)


def entry_bucket(now_et: datetime) -> str:
    minute = 30 if now_et.minute >= 30 else 0
    return f"{now_et.hour:02d}:{minute:02d}"


def confirmation_breakout(*, direction: int, ref_high: float, ref_low: float, bar: dict) -> bool:
    high = float(bar["high"])
    low = float(bar["low"])
    open_ = float(bar["open"])
    close = float(bar["close"])
    if direction == 1:
        return high >= float(ref_high) and close > open_ and close > float(ref_high)
    return low <= float(ref_low) and close < open_ and close < float(ref_low)
