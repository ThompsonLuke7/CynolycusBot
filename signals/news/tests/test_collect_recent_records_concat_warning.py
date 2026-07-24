"""collect_recent_records() must not trigger pandas' empty/all-NA concat
FutureWarning.

2026-07-21 live audit: this fired ~16 times/session because one or more
configured sources routinely return zero rows for a given lookback window
(e.g. no fresh Fed press releases most polls), and pandas now warns when an
empty/all-NA frame is included in a concat. Fixed by filtering empty frames
out before the concat.
"""
from __future__ import annotations

import warnings
from datetime import datetime, timezone

import pandas as pd

import scripts.intraday_catalyst_poll as poll
from signals.news.schema import empty_news_frame


def _news_row(ticker: str) -> pd.DataFrame:
    ts = datetime.now(timezone.utc)
    frame = empty_news_frame()
    frame.loc[0] = {
        "record_id": f"{ticker}-{ts.isoformat()}",
        "ticker": ticker,
        "timestamp": pd.Timestamp(ts),
        "headline": f"{ticker} headline",
        "summary": "",
        "body": "",
        "url": "",
        "source": "google_news",
        "source_id": "1",
        "text": "",
        "content_hash": f"hash-{ticker}",
    }
    return frame


def test_collect_recent_records_no_future_warning_when_a_source_is_empty(monkeypatch):
    monkeypatch.setattr(poll, "fetch_google_news_rss", lambda *a, **k: _news_row("AAA"))
    monkeypatch.setattr(poll, "fetch_yfinance_news", lambda *a, **k: empty_news_frame())
    monkeypatch.setattr(poll, "fetch_finnhub_company_news", lambda *a, **k: empty_news_frame())

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = poll.collect_recent_records(
            ["AAA"], lookback_minutes=60,
            sources=["google_news", "yfinance", "finnhub"],
        )

    concat_warnings = [w for w in caught if issubclass(w.category, FutureWarning) and "concat" in str(w.message)]
    assert concat_warnings == []
    assert list(out["ticker"]) == ["AAA"]


def test_collect_recent_records_all_sources_empty_returns_empty_frame(monkeypatch):
    monkeypatch.setattr(poll, "fetch_google_news_rss", lambda *a, **k: empty_news_frame())

    out = poll.collect_recent_records(["AAA"], lookback_minutes=60, sources=["google_news"])
    assert out.empty
