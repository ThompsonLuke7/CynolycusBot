"""Event timing helpers with explicit earnings-release anchoring."""

from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo

import pandas as pd

from signals.events.forward_guidance.config import DEFAULT_TIMEZONE


REPORT_TIMES = {"BMO", "AMC", "UNKNOWN"}


def normalize_report_time(value: object) -> str:
    raw = str(value or "UNKNOWN").strip().upper()
    if raw in {"BEFORE", "BEFORE_OPEN", "PRE", "PREMARKET", "PM"}:
        return "BMO"
    if raw in {"AFTER", "AFTER_CLOSE", "POST", "POSTMARKET", "AH"}:
        return "AMC"
    if raw not in REPORT_TIMES:
        return "UNKNOWN"
    return raw


def as_date(value: object) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert(DEFAULT_TIMEZONE)
    return ts.normalize().tz_localize(None)


def next_business_day(value: object) -> pd.Timestamp:
    return as_date(value) + pd.offsets.BDay(1)


def reaction_session(earnings_date: object, report_time: object) -> pd.Timestamp:
    """Return the trading session whose close can first include the ER reaction."""
    rt = normalize_report_time(report_time)
    d = as_date(earnings_date)
    if rt == "BMO":
        return d
    return next_business_day(d)


def session_close_timestamp(session_date: object, tz_name: str = DEFAULT_TIMEZONE) -> pd.Timestamp:
    tz = ZoneInfo(tz_name)
    d = as_date(session_date)
    close = dt.datetime(d.year, d.month, d.day, 16, 0, tzinfo=tz)
    return pd.Timestamp(close).tz_convert("UTC")


def event_available_at(earnings_date: object, report_time: object) -> pd.Timestamp:
    """Best-effort release availability timestamp used for leakage checks."""
    tz = ZoneInfo(DEFAULT_TIMEZONE)
    d = as_date(earnings_date)
    rt = normalize_report_time(report_time)
    hour = 7 if rt == "BMO" else 16
    minute = 0 if rt == "BMO" else 30
    if rt == "UNKNOWN":
        d = next_business_day(d)
        hour = 7
        minute = 0
    return pd.Timestamp(dt.datetime(d.year, d.month, d.day, hour, minute, tzinfo=tz)).tz_convert("UTC")


def event_id(ticker: str, earnings_date: object) -> str:
    return f"{str(ticker).upper()}_{as_date(earnings_date).date().isoformat()}"
