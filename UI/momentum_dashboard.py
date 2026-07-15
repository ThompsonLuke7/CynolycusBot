"""Momentum Expansion dashboard — read-only 4H/1H scan + account + manual run.

Mirrors the Meta Ranker dashboard: shows the current expansion-ranked entry
triggers, the Alpaca account/positions, and a "Run now" button that fires one
``MomentumLiveRunner`` pass (submits on the selected account). Orders default to
paper; the real-money toggle (off by default) routes to the live env.

Endpoints:
  GET  /              → HTML page (auto-refreshing)
  GET  /api/state     → snapshot JSON (config, scan, account)
  POST /api/run-loop  → fire one evaluate_now + manage_open_positions pass
  POST /api/set-live  → toggle paper / real-money account
"""
from __future__ import annotations

import json
import logging
import math
import os
import sys
import threading
import time
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from core.API.Alpaca_API.options.options_api import AlpacaOptionsClient
from UI.ui_chrome import NAV_HTML, THEME_LINK, serve_theme_asset, serve_theme_css

logger = logging.getLogger(__name__)
SCAN_TTL = 120.0  # evaluate_now is expensive; cache the scan


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


class MomentumDashboardApp:
    def __init__(self, *, env_file: str = ".env#PAPER", live_env_file: str | None = None,
                 top_k: int = 10) -> None:
        self.paper_env_file = env_file
        self.live_env_file = live_env_file
        self.top_k = top_k
        self._live = False
        self._client = AlpacaOptionsClient(env_file=self.env_file)
        self._scan_cache: tuple[float, dict] | None = None
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
        return {"live": self._live, "available": bool(self.live_env_file)}

    # ---- data ----
    def _scan(self) -> dict:
        now = time.time()
        if self._scan_cache and now - self._scan_cache[0] < SCAN_TTL:
            return self._scan_cache[1]
        try:
            from strategies.momentum_expansion.live.runner import MomentumLiveRunner
            runner = MomentumLiveRunner(auto_trade=False)  # display only — never trades
            triggers = runner.evaluate_now() or []
            picks = [
                {"rank": int(t.get("rank", i + 1)), "ticker": t.get("ticker"),
                 "expansion": round(float(t.get("expansion_score", 0) or 0), 3),
                 "trigger": t.get("trigger_rule"), "close": t.get("bar_close")}
                for i, t in enumerate(triggers[: self.top_k])
            ]
            out = {"picks": picks, "ts": datetime.now(timezone.utc).isoformat()}
        except Exception as exc:  # noqa: BLE001
            out = {"error": f"scan_failed: {exc}", "picks": []}
        self._scan_cache = (now, out)
        return out

    def _own_book_symbols(self) -> set[str]:
        """OCC symbols the momentum option policy has opened (durable book)."""
        try:
            from strategies.momentum_expansion.policy.momentum_option_policy import (
                DEFAULT_BOOK_PATH,
            )

            if DEFAULT_BOOK_PATH.exists():
                return set(json.loads(DEFAULT_BOOK_PATH.read_text()).keys())
        except Exception as exc:  # noqa: BLE001
            logger.warning("momentum book read failed: %s", exc)
        return set()

    def _account(self) -> dict:
        """Account snapshot scoped to THIS module's own book.

        The Alpaca account is shared across every module, so `positions` is
        filtered to the option contracts momentum opened (book.json), reconciled
        against the live broker so closed/expired contracts drop off. Broker-wide
        totals are reported under ``account_*`` for the hub.
        """
        try:
            a = self._client.get_account()
            pos = self._client.get_positions() or []
            all_positions = [
                {"symbol": p["symbol"], "qty": float(p["qty"]),
                 "avg_entry": float(p.get("avg_entry_price", 0) or 0),
                 "current": float(p.get("current_price", 0) or 0),
                 "mv": float(p.get("market_value", 0) or 0),
                 "upl": float(p.get("unrealized_pl", 0) or 0),
                 "upl_pct": round(float(p.get("unrealized_plpc", 0) or 0) * 100, 2)}
                for p in pos
            ]
            own = self._own_book_symbols()
            book = [p for p in all_positions if p["symbol"] in own]
            return {"equity": float(a.get("equity", 0) or 0),
                    "cash": float(a.get("cash", 0) or 0),
                    "buying_power": float(a.get("buying_power", 0) or 0),
                    "n_positions": len(book),
                    "positions": sorted(book, key=lambda x: -x["upl"]),
                    "account_n_positions": len(all_positions),
                    "account_upl": round(sum(p["upl"] for p in all_positions), 2)}
        except Exception as exc:  # noqa: BLE001
            return {"error": f"account_failed: {exc}", "positions": []}

    def snapshot(self) -> dict:
        return {
            "ts": datetime.now(timezone.utc).isoformat(),
            "config": {"env": self.env_file, "mode": "momentum", "submit": True,
                       "top_k": self.top_k, "live": self._live,
                       "live_available": bool(self.live_env_file)},
            "scan": self._scan(),
            "account": self._account(),
            "loop_running": self._loop_running,
        }

    # ---- manual run ----
    def run_loop(self) -> dict:
        """Fire one momentum pass now: score the (uncapped) universe via the
        ExpansionRanker, check entry triggers, and route fills through this
        module's OWN order policy (MomentumOptionPolicy). This is the standalone
        Momentum harness — separate from the Meta Ranker, same base score.
        """
        with self._lock:
            if self._loop_running:
                return {"started": False, "reason": "already_running"}
            self._loop_running = True

        def _go():
            try:
                from core.API.Alpaca_API.options.options_api import AlpacaOptionsClient as _C
                from strategies.momentum_expansion.live.runner import MomentumLiveRunner

                os.environ["PYTHONPATH"] = str(REPO)
                # Shared 4H engine (option/share routing + hold-based exits), same as Meta/HTF.
                runner = MomentumLiveRunner(auto_trade=True)
                runner.run_pass(_C(env_file=self.env_file), submit=True)
            except Exception as exc:  # noqa: BLE001
                logger.error("momentum loop failed: %s", exc)
            finally:
                self._scan_cache = None
                self._loop_running = False

        threading.Thread(target=_go, daemon=True, name="momentum-manual-loop").start()
        return {"started": True, "live": self._live}


_PAGE = """<!doctype html><html><head><meta charset=utf-8><title>Momentum Expansion</title>
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
<h2>Momentum Expansion <span id=cfg class=pill></span> <span id=age class=muted></span></h2>
<div class=muted>Standalone Momentum harness — ExpansionRanker score (same as Meta's mom_score) traded via its own order policy.</div>
<div class=controls>
  <button class=primary onclick=runLoop()>Run + SUBMIT now</button>
  <label class=tog title="Off: submit on the PAPER account (default). On: the real-money account.">
    <input type=checkbox id=live-toggle onchange=setLive(this.checked)> Real money account</label>
  <span id=msg class=muted></span>
</div>
<h2>Account</h2><div id=acct></div>
<h2>Top expansion triggers</h2><table id=scan></table>
<h2>Open positions</h2><table id=pos></table>
</div>
<script>
function f(n,d=2){return n==null?'-':Number(n).toLocaleString(undefined,{maximumFractionDigits:d})}
async function tick(){let s=await(await fetch('/api/state')).json();
let live=!!s.config.live;
document.getElementById('cfg').textContent='momentum · '+(live?'real money':'paper');
document.getElementById('cfg').className='pill '+(live?'live':'paper');
let lt=document.getElementById('live-toggle');if(lt){lt.checked=live;lt.disabled=!s.config.live_available;}
let badge=document.getElementById('cyno-live-badge');
if(badge){badge.textContent=live?'real money':'paper';badge.className='live-badge'+(live?' live':'');}
document.getElementById('age').textContent=(s.scan&&s.scan.error)?s.scan.error:'';
let a=s.account;document.getElementById('acct').innerHTML=a.error?('<span class=neg>'+a.error+'</span>'):
('equity $'+f(a.equity,0)+' · cash $'+f(a.cash,0)+' · BP $'+f(a.buying_power,0)+' · '+a.n_positions+' positions');
document.getElementById('scan').innerHTML='<tr><th>#</th><th class=t>ticker</th><th>expansion</th><th class=t>trigger</th><th>close</th></tr>'+
(s.scan.picks||[]).map(p=>'<tr><td>'+p.rank+'</td><td class=t>'+p.ticker+'</td><td>'+p.expansion+'</td><td class=t>'+(p.trigger||'-')+'</td><td>'+f(p.close)+'</td></tr>').join('');
document.getElementById('pos').innerHTML='<tr><th class=t>symbol</th><th>qty</th><th>entry</th><th>current</th><th>mkt val</th><th>unreal P/L</th><th>%</th></tr>'+
(a.positions||[]).map(p=>{let c=p.upl>=0?'pos':'neg';return '<tr><td class=t>'+p.symbol+'</td><td>'+f(p.qty,0)+'</td><td>'+f(p.avg_entry)+'</td><td>'+f(p.current)+'</td><td>$'+f(p.mv,0)+'</td><td class='+c+'>$'+f(p.upl,0)+'</td><td class='+c+'>'+p.upl_pct+'%</td></tr>'}).join('');}
async function setLive(v){await fetch('/api/set-live',{method:'POST',body:JSON.stringify({live:v})});tick();}
async function runLoop(){document.getElementById('msg').textContent='launching...';
let r=await(await fetch('/api/run-loop',{method:'POST',body:JSON.stringify({})})).json();
document.getElementById('msg').textContent=r.started?('running ('+(r.live?'real money':'paper')+')'):('skipped: '+r.reason);}
tick();setInterval(tick,5000);
</script></body></html>""".replace("__THEME_LINK__", THEME_LINK).replace("__NAV_HTML__", NAV_HTML)


class MomentumHTTPServer(ThreadingHTTPServer):
    app: MomentumDashboardApp


class MomentumHandler(BaseHTTPRequestHandler):
    server_version = "CynolycusMomentum/1.0"

    def log_message(self, fmt, *a):  # noqa: A002
        return

    def _app(self) -> MomentumDashboardApp:
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
        elif self.path.startswith("/static/themes/"):
            serve_theme_asset(self, self.path)
        elif self.path.startswith("/api/state"):
            self._send(json.dumps(_json_safe(self._app().snapshot())).encode("utf-8"))
        else:
            self._send(b'{"error":"not_found"}', status=HTTPStatus.NOT_FOUND)

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", "0") or 0)
        payload = json.loads(self.rfile.read(length) or b"{}") if length else {}
        if self.path.startswith("/api/set-live"):
            res = self._app().set_live(bool(payload.get("live")))
        elif self.path.startswith("/api/run-loop"):
            res = self._app().run_loop()
        else:
            self._send(b'{"error":"not_found"}', status=HTTPStatus.NOT_FOUND)
            return
        self._send(json.dumps(res).encode("utf-8"))


def make_server(host: str, port: int, app: MomentumDashboardApp) -> MomentumHTTPServer:
    srv = MomentumHTTPServer((host, port), MomentumHandler)
    srv.daemon_threads = True
    srv.app = app
    return srv
