"""HTTP surface of the Library dashboard.

Runs the real server against a stubbed ``news_library`` so the endpoint
contracts, the building-state handling, and the chart's event bucketing are
covered without building the production index.
"""
from __future__ import annotations

import json
import threading
import urllib.request
from urllib.parse import urlencode

import pytest

from UI import library_dashboard


@pytest.fixture()
def server(monkeypatch):
    monkeypatch.setattr(library_dashboard.news_library, "index_is_current", lambda: True)
    monkeypatch.setattr(
        library_dashboard.news_library, "status",
        lambda: {"current": True, "rows": 5, "tickers": 2, "built_at": "2026-07-28T00:00:00+00:00",
                 "index_path": "/tmp/x.parquet"},
    )
    app = library_dashboard.LibraryDashboardApp(default_ticker="AAPL")
    srv = library_dashboard.make_server("127.0.0.1", 0, app)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", app
    srv.shutdown()


def get(base: str, path: str, **params):
    url = f"{base}{path}" + (f"?{urlencode(params)}" if params else "")
    with urllib.request.urlopen(url, timeout=10) as r:
        return r.status, r.read(), r.headers.get("Content-Type", "")


def get_json(base: str, path: str, **params):
    return json.loads(get(base, path, **params)[1])


def test_page_renders_with_shared_chrome(server):
    base, _ = server
    status, body, ctype = get(base, "/")
    text = body.decode()
    assert status == 200 and "text/html" in ctype
    assert "cyno-nav" in text                      # shared nav injected
    assert "/static/cynolycus_theme.css" in text   # shared theme linked
    assert "__DEFAULT_TICKER__" not in text        # placeholder substituted
    assert "'AAPL'" in text


def test_theme_css_is_served(server):
    base, _ = server
    status, body, ctype = get(base, "/static/cynolycus_theme.css")
    assert status == 200 and "text/css" in ctype
    assert b"--bg:#0d1117" in body


def test_state_reports_index(server):
    base, _ = server
    s = get_json(base, "/api/state")
    assert s["config"]["mode"] == "library"
    assert s["config"]["tradeable"] is False
    assert s["index"]["rows"] == 5


def test_search_passes_filters_through(server, monkeypatch):
    base, _ = server
    seen = {}

    def fake_search(**kwargs):
        seen.update(kwargs)
        return {"rows": [{"ticker": "AAPL", "headline": "hi"}], "total": 1,
                "offset": 0, "limit": 100}

    monkeypatch.setattr(library_dashboard.news_library, "search", fake_search)
    out = get_json(base, "/api/search", ticker="aapl", q="widget",
                   family="company_news", relation="direct_mention",
                   source="finnhub", origin="news", start="2026-07-01",
                   end="2026-07-28", limit="25", offset="50")
    assert out["total"] == 1 and out["building"] is False
    assert seen["ticker"] == "AAPL"          # normalized upper
    assert seen["query"] == "widget"         # q -> query
    assert seen["family"] == "company_news"
    assert seen["relation"] == "direct_mention"
    assert (seen["limit"], seen["offset"]) == (25, 50)
    # Blank params must become None, not "", so they don't filter everything out.
    out2 = get_json(base, "/api/search", ticker="AAPL", q="")
    assert seen["query"] is None and out2["total"] == 1


def test_search_reports_backend_failure(server, monkeypatch):
    base, _ = server

    def boom(**_kwargs):
        raise RuntimeError("index exploded")

    monkeypatch.setattr(library_dashboard.news_library, "search", boom)
    out = get_json(base, "/api/search", ticker="AAPL")
    assert "index exploded" in out["error"] and out["rows"] == []


def test_building_state_does_not_block_and_never_builds_in_process(server, monkeypatch):
    base, app = server
    monkeypatch.setattr(library_dashboard.news_library, "index_is_current", lambda: False)

    def fail(*_a, **_k):
        raise AssertionError("index must never be built on a request thread")

    monkeypatch.setattr(library_dashboard.news_library, "build_index", fail)
    monkeypatch.setattr(library_dashboard.news_library, "ensure_index", fail)
    launched = {}

    def fake_run(argv, **_kw):
        launched["argv"] = argv
        return type("P", (), {"returncode": 0, "stderr": "", "stdout": ""})()

    monkeypatch.setattr(library_dashboard.subprocess, "run", fake_run)

    for path in ("/api/search", "/api/price", "/api/facets", "/api/tickers"):
        out = get_json(base, path, ticker="AAPL")
        assert out["building"] is True and out["rows"] == []

    if app._build_thread:
        app._build_thread.join(timeout=10)
    assert launched["argv"][1:] == ["-m", "UI.news_library", "--force"]


def test_price_buckets_events_by_et_session_date(server, monkeypatch):
    """A 22:44 UTC headline is after-hours news on the PREVIOUS ET session."""
    base, _ = server
    monkeypatch.setattr(
        library_dashboard.news_library, "price_series",
        lambda t, days=365: {"ticker": t, "bars": [
            {"t": "2026-07-20T04:00:00+00:00", "o": 1, "h": 2, "l": 1, "c": 1.5, "v": 10},
            {"t": "2026-07-21T04:00:00+00:00", "o": 1, "h": 2, "l": 1, "c": 1.7, "v": 10},
        ]},
    )
    monkeypatch.setattr(
        library_dashboard.news_library, "event_days",
        lambda **_k: {"days": [
            {"day": "2026-07-20", "count": 2, "klass": "company_news",
             "breakdown": {"company_news": 2}, "headlines": [
                 {"headline": "late a", "source": "s", "family": "company_news"}]},
            {"day": "2026-07-21", "count": 1, "klass": "company_news",
             "breakdown": {"company_news": 1}, "headlines": [
                 {"headline": "next day", "source": "s", "family": "company_news"}]},
        ], "classes": [{"name": "company_news", "count": 3}],
            "color_by": "catalyst_family", "total": 3},
    )
    out = get_json(base, "/api/price", ticker="AAPL", days="365")
    days = {e["day"]: e["count"] for e in out["events"]}
    assert days == {"2026-07-20": 2, "2026-07-21": 1}
    assert out["events_total"] == 3
    assert out["classes"] == [{"name": "company_news", "count": 3}]
    assert out["color_by"] == "catalyst_family"
    assert out["events"] == sorted(out["events"], key=lambda e: e["day"])


def test_price_passes_color_by_through(server, monkeypatch):
    base, _ = server
    seen = {}
    monkeypatch.setattr(
        library_dashboard.news_library, "price_series",
        lambda t, days=365: {"ticker": t, "bars": [
            {"t": "2026-07-20T04:00:00+00:00", "o": 1, "h": 2, "l": 1, "c": 1.5, "v": 10}]},
    )

    def fake_event_days(**kwargs):
        seen.update(kwargs)
        return {"days": [], "classes": [], "color_by": kwargs.get("color_by"), "total": 0}

    monkeypatch.setattr(library_dashboard.news_library, "event_days", fake_event_days)
    get_json(base, "/api/price", ticker="AAPL", color_by="source")
    assert seen["color_by"] == "source"
    # Omitted -> the documented default, never None.
    get_json(base, "/api/price", ticker="AAPL")
    assert seen["color_by"] == "catalyst_family"


def test_price_clips_event_window_to_the_user_date_filter(server, monkeypatch):
    """Markers must never cover a window the table below excludes."""
    base, _ = server
    seen = {}
    monkeypatch.setattr(
        library_dashboard.news_library, "price_series",
        lambda t, days=365: {"ticker": t, "bars": [
            {"t": "2026-01-01T05:00:00+00:00", "o": 1, "h": 2, "l": 1, "c": 1.5, "v": 10}]},
    )

    def fake_event_days(**kwargs):
        seen.update(kwargs)
        return {"days": [], "classes": [], "color_by": "catalyst_family", "total": 0}

    monkeypatch.setattr(library_dashboard.news_library, "event_days", fake_event_days)
    get_json(base, "/api/price", ticker="AAPL", start="2026-06-01", end="2026-06-30")
    assert str(seen["start"]).startswith("2026-06-01")   # user start wins (later)
    assert seen["end"] == "2026-06-30"


def test_price_missing_ticker_is_reported_not_faked(server, monkeypatch):
    base, _ = server
    monkeypatch.setattr(
        library_dashboard.news_library, "price_series",
        lambda t, days=365: {"ticker": t, "bars": [], "error": f"no 1d bar cache for {t}"},
    )
    out = get_json(base, "/api/price", ticker="ZZZZ")
    assert out["bars"] == [] and out["events"] == []
    assert "no 1d bar cache" in out["error"]


def test_unknown_path_404s(server):
    base, _ = server
    with pytest.raises(urllib.error.HTTPError) as exc:
        get(base, "/api/nope")
    assert exc.value.code == 404


def test_dashboard_exposes_no_write_endpoints(server):
    """The Library is read-only: it must reject POST outright."""
    base, _ = server
    req = urllib.request.Request(f"{base}/api/search", data=b"{}", method="POST")
    with pytest.raises(urllib.error.HTTPError) as exc:
        urllib.request.urlopen(req, timeout=10)
    assert exc.value.code == 501
