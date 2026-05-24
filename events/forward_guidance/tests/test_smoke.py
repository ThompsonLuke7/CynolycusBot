"""Fixture smoke tests for events.forward_guidance.

Run with:
    python -m events.forward_guidance.tests.test_smoke
"""

from __future__ import annotations

import gzip

import numpy as np
import pandas as pd

from events.forward_guidance.data.schema import EarningsEvent
from events.forward_guidance.data.discover_events import _parse_yfinance_earnings_dates, dedupe_events
from events.forward_guidance.data.sec_client import decompress_response_bytes
from events.forward_guidance.features.market_context import compute_forward_labels, compute_market_context
from events.forward_guidance.features.nlp import extract_forward_sections, extract_structured_guidance_features
from events.forward_guidance.models.train import walk_forward_splits
from events.forward_guidance.utils.dates import reaction_session


def _bars(symbol: str, *, start: str = "2025-11-03", days: int = 130, growth: float = 0.002) -> pd.DataFrame:
    sessions = pd.bdate_range(start=start, periods=days)
    rows = []
    price = 100.0
    for i, session in enumerate(sessions):
        price *= 1.0 + growth
        open_px = price * (0.995 if i == 45 and symbol == "NVDA" else 1.0)
        close_px = price
        if i > 45 and symbol == "NVDA":
            close_px *= 1.0 + 0.004 * (i - 45)
        ts_open = pd.Timestamp(session.date().isoformat() + " 09:30", tz="America/New_York").tz_convert("UTC")
        ts_close = pd.Timestamp(session.date().isoformat() + " 15:30", tz="America/New_York").tz_convert("UTC")
        rows.append({"timestamp": ts_open, "open": open_px, "high": open_px * 1.01, "low": open_px * 0.99, "close": open_px, "volume": 1000 + i})
        rows.append({"timestamp": ts_close, "open": open_px, "high": close_px * 1.01, "low": min(open_px, close_px) * 0.99, "close": close_px, "volume": 1100 + i})
    return pd.DataFrame(rows)


def test_reaction_session_anchors() -> None:
    assert reaction_session("2026-01-05", "BMO").date().isoformat() == "2026-01-05"
    assert reaction_session("2026-01-05", "AMC").date().isoformat() == "2026-01-06"
    assert reaction_session("2026-01-05", "UNKNOWN").date().isoformat() == "2026-01-06"


def test_guidance_extraction_and_features() -> None:
    text = """
    Financial Outlook

    We are raising full year revenue guidance due to strong AI demand, backlog,
    and improved gross margin expansion. Management is confident in future demand.

    Question-and-Answer

    Analyst: Can you discuss orders?
    """
    sections = extract_forward_sections(text)
    assert "raising full year revenue guidance" in sections["forward_guidance"].lower()
    features = extract_structured_guidance_features(sections["forward_guidance"])
    assert features["guidance_revenue_raise_cut"] > 0
    assert features["ai_demand_mentions"] >= 1
    assert features["guidance_strength_score"] > 0


def test_market_context_and_labels() -> None:
    event = EarningsEvent(ticker="NVDA", earnings_date="2025-12-31", report_time="BMO", sector_etf="XLK")
    bars = {
        "NVDA": _bars("NVDA"),
        "SPY": _bars("SPY", growth=0.0005),
        "XLK": _bars("XLK", growth=0.0007),
        "QQQ": _bars("QQQ", growth=0.0006),
        "VIXY": _bars("VIXY", growth=-0.0003),
    }
    context = compute_market_context(event, bars)
    labels = compute_forward_labels(event, bars)
    assert "post_er_gap_pct" in context
    assert labels["fwd_ret_60d"] == labels["fwd_ret_60d"]
    assert labels["fwd_60d_excess_ret_vs_sector"] == labels["fwd_60d_excess_ret_vs_sector"]
    assert labels["target"] in {0.0, 1.0}


def test_walk_forward_splits_are_ordered() -> None:
    df = pd.DataFrame(
        {
            "signal_timestamp": pd.date_range("2024-01-01", periods=30, tz="UTC"),
            "target": [0, 1] * 15,
            "feature": np.arange(30),
        }
    )
    splits = walk_forward_splits(df, n_folds=5)
    assert splits
    for train_idx, val_idx in splits:
        assert train_idx.max() < val_idx.min()


def test_event_discovery_helpers() -> None:
    raw = pd.DataFrame(
        {
            "Earnings Date": [
                pd.Timestamp("2026-01-05 21:00", tz="UTC"),
                pd.Timestamp("2026-04-05 21:00", tz="UTC"),
            ],
            "EPS Estimate": [1.2, 1.4],
        }
    )
    events = _parse_yfinance_earnings_dates("MSFT", raw, start="2026-01-01", end="2026-02-01")
    assert len(events) == 1
    assert events[0].ticker == "MSFT"
    assert events[0].report_time == "AMC"

    deduped = dedupe_events(
        [
            EarningsEvent(ticker="MSFT", earnings_date="2026-01-05", report_time="UNKNOWN"),
            EarningsEvent(ticker="MSFT", earnings_date="2026-01-05", report_time="AMC"),
        ]
    )
    assert len(deduped) == 1
    assert deduped[0].report_time == "AMC"


def test_sec_gzip_bytes_are_decoded() -> None:
    compressed = gzip.compress(b'{"ok": true}')
    assert decompress_response_bytes(compressed, "gzip").decode("utf-8") == '{"ok": true}'


def run_all() -> None:
    test_reaction_session_anchors()
    test_guidance_extraction_and_features()
    test_market_context_and_labels()
    test_walk_forward_splits_are_ordered()
    test_event_discovery_helpers()
    test_sec_gzip_bytes_are_decoded()
    print("events.forward_guidance smoke tests passed")


if __name__ == "__main__":
    run_all()
