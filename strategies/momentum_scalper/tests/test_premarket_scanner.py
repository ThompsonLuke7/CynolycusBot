from __future__ import annotations

from datetime import timedelta

import pandas as pd

from strategies.momentum_scalper.configs.v1 import ScannerConfigV1
from strategies.momentum_scalper.scanners.premarket import reconstruct_premarket_snapshot


def _ts(day: str, time: str) -> pd.Timestamp:
    return pd.Timestamp(f"{day} {time}", tz="America/New_York").tz_convert("UTC")


def _bars() -> pd.DataFrame:
    rows: list[dict] = []
    for day in ("2026-08-03", "2026-08-04", "2026-08-05", "2026-08-06", "2026-08-07"):
        rows.extend([
            {"timestamp": _ts(day, "04:05"), "ticker": "RUNR", "open": 8.0, "high": 8.2, "low": 7.9, "close": 8.0, "volume": 100.0},
            {"timestamp": _ts(day, "16:00"), "ticker": "RUNR", "open": 8.0, "high": 8.5, "low": 7.8, "close": 8.0, "volume": 1_000.0},
        ])
    rows.extend([
        {"timestamp": _ts("2026-08-10", "04:05"), "ticker": "RUNR", "open": 9.0, "high": 9.0, "low": 8.9, "close": 9.0, "volume": 100_000.0},
        {"timestamp": _ts("2026-08-10", "04:06"), "ticker": "RUNR", "open": 9.0, "high": 10.0, "low": 8.9, "close": 10.0, "volume": 200_000.0},
    ])
    return pd.DataFrame(rows)


def _config(**overrides: object) -> ScannerConfigV1:
    values = dict(
        min_premarket_volume=1, min_rvol=1, min_rvol_history_sessions=5,
        min_daily_resistance_distance_pct=1, max_quote_age_seconds=5,
    )
    values.update(overrides)
    return ScannerConfigV1(**values)


def test_scanner_uses_prior_session_close_and_time_matched_rvol() -> None:
    decision = _ts("2026-08-10", "04:06")
    metadata = pd.DataFrame([{"ticker": "RUNR", "float": 5_000_000, "available_at": decision - timedelta(minutes=1)}])
    news = pd.DataFrame([{"ticker": "RUNR", "material": True, "available_at": decision - timedelta(minutes=1)}])
    quotes = pd.DataFrame([{"ticker": "RUNR", "timestamp": decision, "received_at": decision, "bid": 9.99, "ask": 10.00, "bid_size": 1000, "ask_size": 1000, "feed": "SIP"}])

    out = reconstruct_premarket_snapshot(_bars(), decision_at=decision, metadata=metadata, news=news, quotes=quotes, config=_config())

    row = out.iloc[0]
    assert bool(row["eligible"])
    assert row["prior_close"] == 8.0
    assert row["gap_pct"] == 25.0
    assert row["rvol"] == 3000.0
    assert row["rvol_history_sessions"] == 5


def test_scanner_rejects_future_metadata_and_missing_quote() -> None:
    decision = _ts("2026-08-10", "04:06")
    metadata = pd.DataFrame([{"ticker": "RUNR", "float": 5_000_000, "available_at": decision + timedelta(seconds=1)}])
    news = pd.DataFrame([{"ticker": "RUNR", "material": True, "available_at": decision - timedelta(minutes=1)}])

    out = reconstruct_premarket_snapshot(_bars(), decision_at=decision, metadata=metadata, news=news, quotes=pd.DataFrame(), config=_config())

    reasons = out.iloc[0]["rejection_reasons"]
    assert "missing_point_in_time_float" in reasons
    assert "missing_two_sided_quote" in reasons


def test_scanner_does_not_use_quote_received_after_decision() -> None:
    decision = _ts("2026-08-10", "04:06")
    metadata = pd.DataFrame([{"ticker": "RUNR", "float": 5_000_000, "available_at": decision - timedelta(minutes=1)}])
    news = pd.DataFrame([{"ticker": "RUNR", "material": True, "available_at": decision - timedelta(minutes=1)}])
    quotes = pd.DataFrame([{
        "ticker": "RUNR", "timestamp": decision - timedelta(seconds=1),
        "received_at": decision + timedelta(seconds=1), "bid": 9.99, "ask": 10.0,
    }])

    out = reconstruct_premarket_snapshot(_bars(), decision_at=decision, metadata=metadata, news=news, quotes=quotes, config=_config())

    assert "missing_two_sided_quote" in out.iloc[0]["rejection_reasons"]
