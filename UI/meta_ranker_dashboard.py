"""Meta Ranker dashboard — combo top-10 scan, paper positions + P&L, matrix freshness.

Read-only display (the 4H loop runs via combined_server's scheduler). Endpoints:
  GET  /              → HTML page (auto-refreshing)
  GET  /api/state     → snapshot JSON (account, positions, latest scan, matrix age)
  POST /api/run-loop  → fire one loop pass now ({"submit": true})  (for manual testing)

Serves on its own port via ThreadingHTTPServer, wired into UI/combined_server.py.
"""
from __future__ import annotations

import json
import logging
import math
import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from core.API.Alpaca_API.options.options_api import AlpacaOptionsClient
from signals.meta_context.meta_ranker.score import score_frame
from UI.ui_chrome import NAV_HTML, THEME_LINK, serve_theme_css
from UI.performance import module_performance

logger = logging.getLogger(__name__)
MATRIX = REPO / "signals/meta_context/meta_ranker/meta_ranker_matrix.parquet"
STATE = REPO / "signals/meta_context/meta_ranker/live_state.json"
LOOP = REPO / "signals/meta_context/meta_ranker/run_4h_loop.py"
SCAN_TTL = 60.0  # seconds to cache the (somewhat expensive) scan
ACCT_TTL = 8.0  # seconds to cache the broker account/positions fetch


def _json_safe(o: Any) -> Any:
    if o is None or isinstance(o, (bool, int, str)):
        return o
    if isinstance(o, float):
        return None if (math.isnan(o) or math.isinf(o)) else o
    if isinstance(o, (list, tuple)):
        return [_json_safe(x) for x in o]
    if isinstance(o, dict):
        return {str(k): _json_safe(v) for k, v in o.items()}
    return str(o)


class MetaRankerDashboardApp:
    def __init__(self, *, env_file: str = ".env#PAPER", live_env_file: str | None = None,
                 mode: str = "equity", submit: bool = True, top_k: int = 10) -> None:
        self.paper_env_file = env_file
        # When set, the LIVE toggle routes the loop + account view to this
        # real-money env. Off by default.
        self.live_env_file = live_env_file
        self.mode = mode
        self.submit = submit  # always submit on the selected account (no dry-run)
        self.top_k = top_k
        self._live = False
        self._client = AlpacaOptionsClient(env_file=self.env_file)
        # (ts, result, matrix_mtime) — re-read the 65 MB matrix only when the
        # file actually changes, not on every 5 s poll.
        self._scan_cache: tuple[float, dict, float | None] | None = None
        # (ts, (account, all_positions)) — cache the broker fetch so the page's
        # 5 s auto-refresh doesn't hit Alpaca synchronously on every poll.
        self._broker_cache: tuple[float, tuple[dict, list]] | None = None
        self._lock = threading.Lock()
        self._loop_running = False

    @property
    def env_file(self) -> str:
        return self.live_env_file if (self._live and self.live_env_file) else self.paper_env_file

    def set_live(self, live: bool) -> dict:
        want = bool(live) and bool(self.live_env_file)
        if want != self._live:
            self._live = want
            self._client = AlpacaOptionsClient(env_file=self.env_file)
            self._scan_cache = None
            self._broker_cache = None
        return {"live": self._live, "available": bool(self.live_env_file)}

    # ---- data ----
    def _scan(self) -> dict:
        # The matrix only changes a few times a day (readiness job + the 4H
        # loop). Key the cache on the file's mtime so repeated polls return the
        # cached result instantly instead of re-loading the 65 MB parquet.
        try:
            mtime = MATRIX.stat().st_mtime
        except OSError:
            mtime = None
        if self._scan_cache and self._scan_cache[2] == mtime:
            return self._scan_cache[1]
        try:
            import pyarrow.parquet as pq

            # Cheap pass: read only the timestamp column to find the latest
            # sufficiently-full bar, before pulling that bar's full feature rows.
            ts = pd.to_datetime(
                pq.read_table(MATRIX, columns=["timestamp"]).column("timestamp").to_pandas(),
                utc=True)
            counts = ts.value_counts()
            bar = counts[counts >= max(50, int(0.25 * counts.max()))].index.max()
            df = pd.read_parquet(MATRIX, filters=[("timestamp", "==", bar.to_pydatetime())])
            if "timestamp" not in df.columns:
                df = df.reset_index()
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            scored = score_frame(df[df["timestamp"] == bar].copy())
            top = scored.sort_values("s_combo", ascending=False).head(self.top_k)
            age_d = (datetime.now(timezone.utc) - bar.to_pydatetime()).total_seconds() / 86400.0
            out = {
                "bar": str(bar), "age_days": round(age_d, 2),
                "picks": [
                    {"rank": i + 1, "ticker": r.ticker,
                     "combo": round(float(r.s_combo), 3), "upside": round(float(r.s_upside), 3),
                     "quality": round(float(r.s_quality), 3)}
                    for i, (_, r) in enumerate(top.iterrows())
                ],
            }
        except Exception as exc:  # noqa: BLE001
            out = {"error": f"scan_failed: {exc}", "picks": []}
        self._scan_cache = (time.time(), out, mtime)
        return out

    def _fetch_broker(self) -> tuple[dict, list]:
        """Account + all positions, cached for ``ACCT_TTL`` seconds.

        The page auto-refreshes every 5 s; without this each poll fired two
        synchronous Alpaca REST calls, which made the tab hang/feel "down" when
        you returned to it. Cached raw so ``_account`` can re-filter by the
        (cheaply changing) managed set on every call.
        """
        now = time.time()
        if self._broker_cache and now - self._broker_cache[0] < ACCT_TTL:
            return self._broker_cache[1]
        a = self._client.get_account()
        pos = self._client.get_positions() or []
        account = {
            "equity": float(a.get("equity", 0) or 0),
            "cash": float(a.get("cash", 0) or 0),
            "buying_power": float(a.get("buying_power", 0) or 0),
        }
        all_positions = [
            {"symbol": p["symbol"], "qty": float(p["qty"]),
             "avg_entry": float(p.get("avg_entry_price", 0) or 0),
             "current": float(p.get("current_price", 0) or 0),
             "mv": float(p.get("market_value", 0) or 0),
             "upl": float(p.get("unrealized_pl", 0) or 0),
             "upl_pct": round(float(p.get("unrealized_plpc", 0) or 0) * 100, 2)}
            for p in pos
        ]
        self._broker_cache = (now, (account, all_positions))
        return account, all_positions

    def _account(self, managed: set[str] | None = None) -> dict:
        """Account snapshot scoped to THIS module's own book.

        The Alpaca account is shared across every module, so `positions` is
        filtered to the symbols the Meta Ranker actually owns (``managed`` from
        live_state.json). Account-level equity/cash are reported as-is, plus the
        broker totals under ``account_*`` so the hub can show the true account
        once and attribute the rest per module.
        """
        managed = managed or set()
        try:
            account, all_positions = self._fetch_broker()
            book = [p for p in all_positions if p["symbol"] in managed]
            return {
                "equity": account["equity"],
                "cash": account["cash"],
                "buying_power": account["buying_power"],
                "n_positions": len(book),
                "positions": sorted(book, key=lambda x: -x["upl"]),
                # Broker-wide context (shared account; not this module's book).
                "account_n_positions": len(all_positions),
                "account_upl": round(sum(p["upl"] for p in all_positions), 2),
            }
        except Exception as exc:  # noqa: BLE001
            return {"error": f"account_failed: {exc}", "positions": []}

    def snapshot(self) -> dict:
        state = json.loads(STATE.read_text()) if STATE.exists() else {"managed": {}, "history": []}
        managed = set(state.get("managed", {}).keys())
        account = self._account(managed)
        return {
            "ts": datetime.now(timezone.utc).isoformat(),
            "config": {"env": self.env_file, "mode": self.mode, "submit": self.submit,
                       "top_k": self.top_k, "live": self._live,
                       "live_available": bool(self.live_env_file)},
            "scan": self._scan(),
            "account": account,
            "performance": module_performance(
                "meta_ranker", open_upl=sum(float(p.get("upl", 0) or 0) for p in account.get("positions", []))),
            "managed": sorted(managed),
            "history": list(reversed(state.get("history", [])))[:20],
            "loop_running": self._loop_running,
        }

    # ---- manual loop trigger ----
    def run_loop(self) -> dict:
        """Fire one loop pass now. Always submits on the selected account."""
        with self._lock:
            if self._loop_running:
                return {"started": False, "reason": "already_running"}
            self._loop_running = True

        def _go():
            argv = [sys.executable, str(LOOP), "--mode", self.mode,
                    "--top-k", str(self.top_k), "--submit",
                    "--skip-bars", "--skip-feeds", "--skip-matrix"]
            if self._live:
                argv.append("--live")
            try:
                subprocess.run(argv, cwd=str(REPO), env={**os.environ, "PYTHONPATH": str(REPO)})
            except Exception as exc:  # noqa: BLE001
                logger.error("manual loop failed: %s", exc)
            finally:
                self._scan_cache = None
                self._broker_cache = None
                self._loop_running = False

        threading.Thread(target=_go, daemon=True, name="meta-ranker-manual-loop").start()
        return {"started": True, "live": self._live}


_PAGE = """<!doctype html><html><head><meta charset=utf-8><title>Meta Ranker</title>
__THEME_LINK__
<style>
.wrap{margin:14px 18px}
h2{margin:14px 0 6px;font-size:15px}
table{border-collapse:collapse;width:100%;margin:8px 0}
td,th{border:1px solid var(--border);padding:4px 8px;text-align:right}th{background:var(--panel2)}
td:first-child,th:first-child,td.t,th.t{text-align:left}
.controls{margin:6px 0}
.tog{display:inline-flex;align-items:center;gap:6px;color:var(--muted);font-size:12px}
</style></head><body>
__NAV_HTML__
<div class=wrap>
<h2>Meta Ranker <span id=cfg class=pill></span> <span id=age class=muted></span></h2>
<div class=controls>
  <button class=primary onclick=runLoop()>Run + SUBMIT now</button>
  <label class=tog title="Off: submit on the PAPER account (default). On: the real-money account.">
    <input type=checkbox id=live-toggle onchange=setLive(this.checked)> Real money account</label>
  <span id=msg class=muted></span>
</div>
<h2>Account</h2><div id=acct></div>
<h2>Combo top-10 (latest bar)</h2><table id=scan></table>
<h2>Open positions</h2><table id=pos></table>
<h2>Recent loop runs</h2><div id=hist class=muted></div>
</div>
<script>
function f(n,d=2){return n==null?'-':Number(n).toLocaleString(undefined,{maximumFractionDigits:d})}
async function tick(){let s=await(await fetch('/api/state')).json();
let live=!!s.config.live;
document.getElementById('cfg').textContent=s.config.mode+' · '+(live?'real money':'paper');
document.getElementById('cfg').className='pill '+(live?'live':'paper');
let lt=document.getElementById('live-toggle');lt.checked=live;lt.disabled=!s.config.live_available;
let badge=document.getElementById('cyno-live-badge');
if(badge){badge.textContent=live?'real money':'paper';badge.className='live-badge'+(live?' live':'');}
document.getElementById('age').textContent='matrix '+(s.scan.age_days)+'d old · bar '+(s.scan.bar||'-');
let a=s.account;document.getElementById('acct').innerHTML=a.error?('<span class=neg>'+a.error+'</span>'):
('equity $'+f(a.equity,0)+' · cash $'+f(a.cash,0)+' · BP $'+f(a.buying_power,0)+' · '+a.n_positions+' positions');
document.getElementById('scan').innerHTML='<tr><th>#</th><th class=t>ticker</th><th>combo</th><th>upside</th><th>quality</th></tr>'+
(s.scan.picks||[]).map(p=>'<tr><td>'+p.rank+'</td><td class=t>'+p.ticker+'</td><td>'+p.combo+'</td><td>'+p.upside+'</td><td>'+p.quality+'</td></tr>').join('');
document.getElementById('pos').innerHTML='<tr><th class=t>symbol</th><th>qty</th><th>entry</th><th>current</th><th>mkt val</th><th>unreal P/L</th><th>%</th></tr>'+
(a.positions||[]).map(p=>{let c=p.upl>=0?'pos':'neg';return '<tr><td class=t>'+p.symbol+'</td><td>'+f(p.qty,0)+'</td><td>'+f(p.avg_entry)+'</td><td>'+f(p.current)+'</td><td>$'+f(p.mv,0)+'</td><td class='+c+'>$'+f(p.upl,0)+'</td><td class='+c+'>'+p.upl_pct+'%</td></tr>'}).join('');
document.getElementById('hist').innerHTML=(s.history||[]).map(h=>h.ts+' — '+(h.mode||'equity')+' — '+(h.orders||0)+' orders').join('<br>')||'none yet';}
async function setLive(v){await fetch('/api/set-live',{method:'POST',body:JSON.stringify({live:v})});tick();}
async function runLoop(){document.getElementById('msg').textContent='launching...';
let r=await(await fetch('/api/run-loop',{method:'POST',body:JSON.stringify({})})).json();
document.getElementById('msg').textContent=r.started?('running ('+(r.live?'real money':'paper')+')'):('skipped: '+r.reason);}
tick();setInterval(tick,5000);
</script></body></html>""".replace("__THEME_LINK__", THEME_LINK).replace("__NAV_HTML__", NAV_HTML)


class MetaRankerHTTPServer(ThreadingHTTPServer):
    app: MetaRankerDashboardApp


class MetaRankerHandler(BaseHTTPRequestHandler):
    server_version = "CynolycusMetaRanker/1.0"

    def log_message(self, fmt, *a):  # noqa: A002
        return

    def _app(self) -> MetaRankerDashboardApp:
        return self.server.app  # type: ignore[attr-defined]

    def _send(self, body: bytes, status=HTTPStatus.OK, ctype="application/json; charset=utf-8"):
        self.send_response(int(status))
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        if self.path == "/" or self.path.startswith("/index"):
            self._send(_PAGE.encode("utf-8"), ctype="text/html; charset=utf-8")
        elif self.path == "/static/cynolycus_theme.css":
            serve_theme_css(self)
        elif self.path.startswith("/api/state"):
            self._send(json.dumps(_json_safe(self._app().snapshot())).encode("utf-8"))
        else:
            self._send(b'{"error":"not_found"}', status=HTTPStatus.NOT_FOUND)

    def do_POST(self):  # noqa: N802
        if self.path.startswith("/api/set-live"):
            length = int(self.headers.get("Content-Length", "0") or 0)
            payload = json.loads(self.rfile.read(length) or b"{}") if length else {}
            res = self._app().set_live(bool(payload.get("live")))
            self._send(json.dumps(res).encode("utf-8"))
            return
        if self.path.startswith("/api/run-loop"):
            length = int(self.headers.get("Content-Length", "0") or 0)
            _ = self.rfile.read(length) if length else b""
            res = self._app().run_loop()
            self._send(json.dumps(res).encode("utf-8"))
        else:
            self._send(b'{"error":"not_found"}', status=HTTPStatus.NOT_FOUND)


def make_server(host: str, port: int, app: MetaRankerDashboardApp) -> MetaRankerHTTPServer:
    srv = MetaRankerHTTPServer((host, port), MetaRankerHandler)
    srv.daemon_threads = True
    srv.app = app
    return srv
