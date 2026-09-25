"""Watchlist source alignment and read-only HTTP behavior."""
from __future__ import annotations

import json
import threading
import urllib.request

import pandas as pd
import pytest

from UI import universe_dashboard as watch

pytestmark = pytest.mark.safe


@pytest.fixture
def app(tmp_path, monkeypatch):
    universe = tmp_path / "universe.csv"
    pd.DataFrame([
        {"ticker": "AAA", "is_eligible": True, "market_cap_bucket": "Large",
         "sector": "Technology", "in_momentum_candidate": True, "in_swing_live": False},
        {"ticker": "BBB", "is_eligible": False, "market_cap_bucket": "Small",
         "sector": "Energy", "theme_1": "solar", "in_momentum_candidate": False,
         "in_swing_live": True},
    ]).to_csv(universe, index=False)
    discovery = tmp_path / "discovery.csv"
    pd.DataFrame([{"ticker": "AAA", "market_cap": 12_000_000_000,
                   "last_seen": "2026-06-22"}]).to_csv(discovery, index=False)
    matrix = tmp_path / "matrix.parquet"
    pd.DataFrame([
        {"timestamp": "2026-09-20T18:00:00Z", "ticker": "AAA", "mom_score": 0.1,
         "htf_score": 0.2, "news_catalyst_score": 0.3, "theme": "software"},
        {"timestamp": "2026-09-24T18:00:00Z", "ticker": "AAA", "mom_score": 0.8,
         "htf_score": 0.7, "news_catalyst_score": float("nan"), "theme": "software"},
    ]).assign(timestamp=lambda x: pd.to_datetime(x.timestamp, utc=True)).set_index(
        ["timestamp", "ticker"]).to_parquet(matrix)
    bars = tmp_path / "bars"
    bars.mkdir()
    pd.DataFrame([
        {"timestamp": "2026-09-23T04:00:00Z", "close": 10.0, "volume": 1_000_000},
        {"timestamp": "2026-09-24T04:00:00Z", "close": 12.0, "volume": 2_000_000},
    ]).assign(timestamp=lambda x: pd.to_datetime(x.timestamp, utc=True)).to_parquet(
        bars / "AAA.parquet")
    monkeypatch.setattr(watch.news_library, "index_is_current", lambda: True)
    monkeypatch.setattr(watch.news_library, "search", lambda **kwargs: {
        "rows": [{"timestamp": "2026-09-24T15:00:00Z", "headline": "News",
                  "record_catalyst_score": float("nan"), "move_1d_pct": 10.0}],
        "total": 1,
    })
    return watch.UniverseDashboardApp(universe=universe, matrix=matrix, bars=bars,
                                      discovery=discovery)


def test_filters_and_latest_source_dates(app):
    result = app.search({"eligible": ["yes"], "min_price": ["11"],
                         "min_volume": ["1200000"], "min_cap": ["10000000000"]})
    assert result["total"] == 1
    row = result["rows"][0]
    assert row["ticker"] == "AAA"
    assert row["price"] == 12
    assert row["volume"] == 2_000_000
    assert row["avg_volume_20d"] == 1_500_000
    assert row["price_at"].startswith("2026-09-24")
    assert row["momentum_score"] == 0.8
    assert row["htf_score"] == 0.7
    assert row["catalyst_score"] is None
    assert row["score_at"].startswith("2026-09-24")
    assert row["market_cap_at"] == "2026-06-22"
    assert app.search({"cap": ["Small"]})["rows"][0]["ticker"] == "BBB"
    assert app.search({"theme": ["solar"], "module": ["swing"]})["total"] == 1
    assert app.search({"sort": ["price"], "direction": ["desc"]})["rows"][0]["ticker"] == "AAA"
    with pytest.raises(ValueError, match="Invalid min_volume"):
        app.search({"min_volume": ["bad"]})


def test_detail_excludes_hindsight_and_is_json_safe(app):
    detail = app.detail("AAA")
    assert detail["news"]["total"] == 1
    assert "move_1d_pct" not in detail["news"]["rows"][0]
    payload = json.dumps(watch._json_safe(detail), allow_nan=False)
    assert '"record_catalyst_score": null' in payload
    with pytest.raises(KeyError):
        app.detail("MISSING")


def test_http_endpoints(app):
    server = watch.make_server("127.0.0.1", 0, app)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with urllib.request.urlopen(base + "/api/search?min_price=11") as response:
            assert json.load(response)["total"] == 1
        with urllib.request.urlopen(base + "/api/detail?ticker=AAA") as response:
            assert json.load(response)["stock"]["ticker"] == "AAA"
        with urllib.request.urlopen(base + "/") as response:
            assert b"Tradable Universe" in response.read()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_detail_uses_dated_cached_news_while_index_refreshes(app, tmp_path, monkeypatch):
    index = tmp_path / "news.parquet"
    pd.DataFrame([{"ticker": "AAA", "timestamp": pd.Timestamp("2026-09-24T15:00:00Z"),
                   "headline": "Cached headline", "source": "wire", "url": None,
                   "catalyst_family": "earnings", "record_catalyst_score": float("nan"),
                   "predicted_direction": "neutral", "tone": "neutral",
                   "move_1d_pct": 99.0}]).to_parquet(index)
    monkeypatch.setattr(watch.news_library, "INDEX_PATH", index)
    monkeypatch.setattr(watch.news_library, "index_is_current", lambda: False)
    detail = app.detail("AAA")
    assert detail["news_index_current"] is False
    assert detail["news"]["rows"][0]["headline"] == "Cached headline"
    assert "move_1d_pct" not in detail["news"]["rows"][0]
    json.dumps(watch._json_safe(detail), allow_nan=False)


def test_hub_state_warms_without_blocking(app, monkeypatch):
    entered = threading.Event()
    release = threading.Event()
    def slow_rows():
        entered.set()
        release.wait(timeout=2)
        return []
    monkeypatch.setattr(app, "_rows", slow_rows)
    state = app.state()
    assert state["building"] is True
    assert entered.wait(timeout=1)
    release.set()
    app._warm_thread.join(timeout=2)
