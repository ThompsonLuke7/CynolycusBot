"""Read-only source-fitness checks for momentum-scalper market feeds."""
from __future__ import annotations

import pandas as pd


def assess_market_feed(
    *,
    trades: pd.DataFrame,
    quotes: pd.DataFrame,
    statuses: pd.DataFrame | None = None,
    expected_feed: str = "SIP",
) -> dict[str, object]:
    """Summarize coverage and causal timestamp quality without network access."""

    report: dict[str, object] = {"expected_feed": expected_feed.upper()}
    for name, frame, required in (
        ("trades", trades, {"ticker", "timestamp", "price", "size"}),
        ("quotes", quotes, {"ticker", "timestamp", "bid", "ask"}),
    ):
        missing = sorted(required - set(frame.columns))
        report[f"{name}_schema_ok"] = not missing
        report[f"{name}_missing_columns"] = missing
        report[f"{name}_rows"] = int(len(frame))
        if missing or frame.empty:
            report[f"{name}_symbols"] = 0
            report[f"{name}_future_available_rows"] = 0
            continue
        available = pd.to_datetime(frame.get("received_at", frame["timestamp"]), utc=True, errors="coerce")
        event = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
        report[f"{name}_symbols"] = int(frame["ticker"].astype(str).str.upper().nunique())
        report[f"{name}_invalid_timestamps"] = int(event.isna().sum() + available.isna().sum())
        report[f"{name}_received_before_event"] = int((available < event).sum())
        if "feed" in frame.columns:
            feeds = set(frame["feed"].dropna().astype(str).str.upper())
            report[f"{name}_feeds"] = sorted(feeds)
            report[f"{name}_expected_feed_present"] = expected_feed.upper() in feeds
        else:
            report[f"{name}_feeds"] = []
            report[f"{name}_expected_feed_present"] = False
    if statuses is not None:
        report["status_rows"] = int(len(statuses))
        report["status_schema_ok"] = {"ticker", "timestamp"}.issubset(statuses.columns)
    report["fit_for_quote_aware_replay"] = bool(
        report["trades_schema_ok"]
        and report["quotes_schema_ok"]
        and report["trades_rows"] > 0
        and report["quotes_rows"] > 0
        and report["quotes_expected_feed_present"]
    )
    return report


__all__ = ["assess_market_feed"]
