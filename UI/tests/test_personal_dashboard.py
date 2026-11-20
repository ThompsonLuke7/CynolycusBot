import http.client
import json
import threading
import time
from datetime import datetime

import pytest

from core.personal_portfolio.data import ET
from UI import personal_dashboard as pdash
from UI.hub_dashboard import _adapt_personal


def _holdings(**_):
    return {"fetched_at": "2026-11-12T20:00:00+00:00", "quotes": {"SPY": {"mark": 600.0}},
            "accounts": [{"account": "…678", "type": "MARGIN", "is_day_trader": False, "closing_only": False,
                          "balances": {"liquidation_value": 10_000.0, "available_funds": 2_000.0},
                          "positions": [{"symbol": "AAA", "raw_symbol": "AAA", "kind": "equity", "quantity": 10.0,
                                         "market_value": 1_000.0, "average_price": 90.0, "open_pl": 100.0,
                                         "day_pl": 5.0, "multiplier": 1.0, "description": "AAA CORP"}]}]}


class _Fetcher:
    def __init__(self, fail=False):
        self.calls, self.fail = 0, fail

    def __call__(self, **kwargs):
        self.calls += 1
        if self.fail:
            raise RuntimeError("token expired")
        return _holdings()


def _app(tmp_path, fetcher, **kw):
    return pdash.PersonalDashboardApp(
        base=tmp_path, holdings_fetcher=fetcher, bars_refresher=lambda *a, **k: {"AAA": "current"},
        earnings_refresher=lambda *a, **k: {}, momentum_loader=lambda: {}, themes_loader=lambda t: {},
        token_status=lambda: {"message": "ok", "severity": "info"},
        clock=lambda: datetime(2026, 11, 12, 15, 0, tzinfo=ET), **kw)


def _wait(app):
    for _ in range(200):
        t = app._thread
        if t is None or not t.is_alive():
            return
        time.sleep(0.01)
    raise AssertionError("sync did not finish")


def test_state_never_calls_schwab_and_never_returns_balances(tmp_path):
    fetcher = _Fetcher()
    app = _app(tmp_path, fetcher)
    assert app.state()["state"] == "never" and fetcher.calls == 0
    app.portfolio()
    _wait(app)
    st = app.state()
    assert st["state"] == "ready" and st["n_positions"] == 1 and fetcher.calls == 1
    assert "10000" not in json.dumps(st) and "net_liq" not in json.dumps(st)
    assert "acct" not in _adapt_personal(st) and _adapt_personal(st)["state"] == "ready"


def test_sync_is_throttled_unless_forced(tmp_path):
    fetcher = _Fetcher()
    app = _app(tmp_path, fetcher, holdings_ttl=60.0)
    app.portfolio()
    _wait(app)
    out = app.portfolio()
    _wait(app)
    assert fetcher.calls == 1 and out["report"]["totals"]["net_liq"] == 10_000.0
    app.portfolio(force=True)
    _wait(app)
    assert fetcher.calls == 2
    assert (tmp_path / "equity_log.csv").read_text().splitlines()[1] == "2026-11-12,10000.0,600.0"


def test_a_failed_sync_keeps_the_last_report(tmp_path):
    fetcher = _Fetcher()
    app = _app(tmp_path, fetcher)
    app.portfolio()
    _wait(app)
    fetcher.fail = True
    app.portfolio(force=True)
    _wait(app)
    out = app.portfolio()
    assert out["status"]["state"] == "error" and "token expired" in out["status"]["error"]
    assert out["report"]["positions"][0]["symbol"] == "AAA"
    assert _adapt_personal(app.state())["state"] == "error"


@pytest.fixture
def server(tmp_path):
    srv = pdash.make_server("127.0.0.1", 0, _app(tmp_path, _Fetcher()))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv
    srv.shutdown()
    srv.server_close()


def _req(srv, method, path, body=None, headers=None):
    conn = http.client.HTTPConnection("127.0.0.1", srv.server_address[1], timeout=5)
    conn.request(method, path, body=body, headers=headers or {})
    resp = conn.getresponse()
    data = resp.read()
    conn.close()
    return resp.status, data


def test_http_routes_and_guards(server, tmp_path):
    status, page = _req(server, "GET", "/")
    assert status == 200 and b"Personal Portfolio" in page and b"cyno-nav" in page
    assert _req(server, "GET", "/api/state")[0] == 200
    assert _req(server, "GET", "/api/state", headers={"Host": "evil.example:8778"})[0] == 403
    form = {"Content-Type": "application/x-www-form-urlencoded"}
    assert _req(server, "POST", "/api/watchlist", b"ticker=AAA", form)[0] == 415
    js = {"Content-Type": "application/json"}
    status, body = _req(server, "POST", "/api/watchlist", json.dumps({"ticker": "msft", "note": "x"}), js)
    assert status == 200 and (tmp_path / "watchlist.csv").read_text().splitlines()[1].startswith("MSFT,x,")
    assert _req(server, "POST", "/api/watchlist", json.dumps({"ticker": "<b>"}), js)[0] == 400
    assert _req(server, "POST", "/api/watchlist/remove", json.dumps({"ticker": "ZZZ"}), js)[0] == 400
    assert _req(server, "POST", "/api/watchlist/remove", json.dumps({"ticker": "MSFT"}), js)[0] == 200
    assert _req(server, "POST", "/api/watchlist", b"x" * 5000, js)[0] == 400


def test_hub_adapter_before_first_sync():
    assert _adapt_personal({"state": "never"}) == {"state": "idle", "detail": "read-only · syncs when the page is open"}


def test_every_tooltip_key_and_flag_code_has_a_definition():
    import re
    from pathlib import Path

    page = pdash.PAGE
    tips = set(re.findall(r"^ (\w+):'", page.split("const TIPS={", 1)[1].split("};", 1)[0], flags=re.M))
    static = set(re.findall(r"data-tip=(\w+)", page))
    tiles = set(re.findall(r"tile\('[^']*',.*?,'(\w+)'\)", page))
    assert static and tiles and (static | tiles) - tips == set()
    src = Path(__file__).resolve().parents[2].joinpath("core/personal_portfolio/analysis.py").read_text()
    codes = set(re.findall(r'"code": "(\w+)"', src)) | set(re.findall(r'\bflag\([^\n]*?,\s*"(\w+)",', src))
    assert len(codes) == 14 and {f"flag_{c}" for c in codes} - tips == set()
    assert {"sev_high", "sev_warn", "sev_info"} <= tips


def _history_app(tmp_path):
    from core.personal_portfolio import trade_history as th
    from core.personal_portfolio.tests.test_trade_history import _opt_inst, stock, trade

    def tx_sync(now, base, full):
        inst = _opt_inst("AAA   261218C00010000", "AAA", "2026-12-18", "CALL", 10.0)
        recs = [trade("2026-11-02T15:00:00+0000", inst, 1, 1.0, "OPENING"),
                trade("2026-11-05T15:00:00+0000", inst, -1, 1.5, "CLOSING"),
                trade("2026-11-06T15:00:00+0000", stock("AAA"), 10, 100.0, "OPENING")]
        th.append_raw([{**r, "accountNumber": "…678"} for r in recs], base)   # the account _holdings() reports
        return {"mode": "backfill", "fetched": 3, "written": 3, "stored": 3, "at": now.isoformat()}

    return _app(tmp_path, _Fetcher(), transactions_syncer=tx_sync, daily_refresher=lambda *a, **k: {},
                intraday_refresher=lambda *a, **k: {})


def _wait_history(app):
    for _ in range(300):
        t = app._hist_thread
        if t is not None and not t.is_alive():
            return
        time.sleep(0.01)
    raise AssertionError("history sync did not finish")


def test_history_builds_episodes_and_reconciles_with_holdings(tmp_path):
    app = _history_app(tmp_path)
    assert app.history()["status"]["state"] == "syncing"
    _wait_history(app)
    out = app.history()
    rep = out["report"]
    assert out["status"]["state"] == "ready" and rep["transactions"]["stored"] == 3
    by = {e["key"]: e for e in rep["episodes"]}
    assert by["AAA 2026-12-18 C 10"]["pnl"] == pytest.approx(50 - 2 * 0.66) and "fills" not in by["AAA"]
    assert by["AAA"]["status"] == "open" and by["AAA"]["unrealized"] == pytest.approx(1_000 - 1_000 - 0.66)
    assert rep["reconciliation"]["ok"] and rep["summary"]["n"] == 1
    json.dumps(rep, allow_nan=False)


def test_history_chart_routes(tmp_path):
    app = _history_app(tmp_path)
    app.history()
    _wait_history(app)
    srv = pdash.make_server("127.0.0.1", 0, app)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        ep = app.history()["report"]["episodes"][0]["id"]
        status, body = _req(srv, "GET", f"/api/history/chart?id={ep}&tf=1d")
        assert status == 200 and "no daily bars" in json.loads(body)["error"]
        assert _req(srv, "GET", "/api/history/chart?id=nope")[0] == 404
        assert _req(srv, "GET", f"/api/history/chart?id={ep}&tf=5m")[0] == 400
        assert _req(srv, "GET", "/api/history/chart?ticker=%3Cb%3E")[0] == 400
        assert _req(srv, "GET", "/api/history")[0] == 200
    finally:
        srv.shutdown()
        srv.server_close()


def test_watchlist_accepts_a_pasted_list_all_or_nothing(tmp_path):
    app = _app(tmp_path, _Fetcher())
    app.add_watch("nvda, amd\nTSM", "looking for entries")
    _wait(app)
    rows = (tmp_path / "watchlist.csv").read_text().splitlines()[1:]
    assert [r.split(",")[0] for r in rows] == ["AMD", "NVDA", "TSM"] and all("looking for entries" in r for r in rows)
    with pytest.raises(ValueError):
        app.add_watch("AAPL, <b>")
    assert "AAPL" not in (tmp_path / "watchlist.csv").read_text()
