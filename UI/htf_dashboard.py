"""HTF Swing dashboard — read-only ranking from the HTF swing scorer.

The higher-timeframe swing strategy has no live trading loop today (only a
trained scorer + backtest), so this dashboard is display-only: it scores the
latest 4H bar of the shared feature matrix and shows the top-ranked names. No
account, no orders, no toggles.

Endpoints:
  GET  /              → HTML page (auto-refreshing)
  GET  /api/state     → snapshot JSON (latest bar + ranked picks)
"""
from __future__ import annotations

import json
import logging
import math
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

from UI.ui_chrome import NAV_HTML, THEME_LINK, serve_theme_css
from UI.performance import module_performance

logger = logging.getLogger(__name__)
# Read htf_score straight off the shared Meta matrix — the SAME per-(timestamp,
# ticker) signal the live HTF runner trades on — so the display can never drift
# from what's actually traded. (The old features_4h.parquet path was a separate,
# nightly-rebuilt cache that could sit weeks stale.)
MATRIX = REPO / "signals/meta_context/meta_ranker/meta_ranker_matrix.parquet"
BLACKLIST_PATH = REPO / "strategies/multi_ticker_swing_htf/live/blacklist.txt"
SIGNALS_LOG = REPO / "Data/inference/htf_swing/htf_signals.parquet"
SIGNAL = "htf_score"
MIN_FULL_BAR = 50  # ignore degenerate edge bars with fewer rows (matches the runner)
# Mirror the live HTF runner's eligibility so the table == the traded targets.
HTF_RANK_FLOOR = 0.85
LIQUIDITY_FLOOR = 0.6


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


class HTFDashboardApp:
    def __init__(self, *, top_k: int = 15, persist: bool = True) -> None:
        self.top_k = top_k
        # (ts, result, matrix_mtime) — re-read the large matrix only when it changes.
        self._scan_cache: tuple[float, dict, float | None] | None = None
        self._persist = persist
        self._last_persisted_bar: str | None = None

    def _blacklist(self) -> set[str]:
        try:
            if BLACKLIST_PATH.exists():
                return {ln.strip().upper() for ln in BLACKLIST_PATH.read_text().splitlines()
                        if ln.strip() and not ln.startswith("#")}
        except Exception:  # noqa: BLE001
            pass
        return set()

    def _scan(self) -> dict:
        # Cache on the matrix file's mtime (it updates a few times a day via the
        # readiness job + the Meta loop), so polls don't re-read the parquet.
        try:
            mtime = MATRIX.stat().st_mtime
        except OSError:
            mtime = None
        if self._scan_cache and self._scan_cache[2] == mtime:
            return self._scan_cache[1]
        try:
            import pandas as pd
            import pyarrow.parquet as pq

            # Cheap pass for the latest sufficiently-full bar, then read just its rows.
            ts = pd.to_datetime(
                pq.read_table(MATRIX, columns=["timestamp"]).column("timestamp").to_pandas(),
                utc=True)
            counts = ts.value_counts()
            bar = counts[counts >= max(MIN_FULL_BAR, int(0.25 * counts.max()))].index.max()
            df = pd.read_parquet(MATRIX, filters=[("timestamp", "==", bar.to_pydatetime())])
            if "timestamp" not in df.columns:
                df = df.reset_index()
            if SIGNAL not in df.columns:
                raise RuntimeError(f"matrix has no '{SIGNAL}' column — rebuild via update_meta_matrix.py")
            # Same eligibility the live runner applies, so the table == traded targets.
            cur = df.dropna(subset=[SIGNAL]).copy()
            cur["htf_rank_pct"] = cur[SIGNAL].rank(pct=True)
            elig = cur[cur["htf_rank_pct"] >= HTF_RANK_FLOOR]
            if "dollar_vol_pctile_252" in elig.columns:
                elig = elig[elig["dollar_vol_pctile_252"].fillna(0) >= LIQUIDITY_FLOOR]
            bl = self._blacklist()
            if bl:
                elig = elig[~elig["ticker"].str.upper().isin(bl)]
            top = elig.sort_values(SIGNAL, ascending=False).head(self.top_k)
            age_d = (datetime.now(timezone.utc) - bar.to_pydatetime()).total_seconds() / 86400.0
            out = {"bar": str(bar), "age_days": round(age_d, 2),
                   "picks": [{"rank": i + 1, "ticker": str(r["ticker"]),
                              "score": round(float(r[SIGNAL]), 4)}
                             for i, (_, r) in enumerate(top.iterrows())]}
        except Exception as exc:  # noqa: BLE001
            out = {"error": f"scan_failed: {exc}", "picks": []}
        self._scan_cache = (time.time(), out, mtime)
        if self._persist and out.get("picks") and not out.get("error"):
            self._persist_scan(out)
        return out

    def _persist_scan(self, out: dict) -> None:
        """Append the ranking to a parquet artifact, once per unique 4H bar.

        Signals are otherwise computed on-demand and lost; this keeps a durable
        record so live HTF picks can be analyzed/backtested after the fact.
        """
        bar = str(out.get("bar") or "")
        if not bar or bar == self._last_persisted_bar:
            return
        try:
            import pandas as pd

            rows = [
                {"captured_at": datetime.now(timezone.utc).isoformat(), "bar": bar,
                 "rank": p["rank"], "ticker": p["ticker"], "htf_swing_score": p["score"]}
                for p in out["picks"]
            ]
            new = pd.DataFrame(rows)
            SIGNALS_LOG.parent.mkdir(parents=True, exist_ok=True)
            if SIGNALS_LOG.exists():
                old = pd.read_parquet(SIGNALS_LOG)
                combined = pd.concat([old[old["bar"] != bar], new], ignore_index=True)
            else:
                combined = new
            combined.to_parquet(SIGNALS_LOG, index=False)
            self._last_persisted_bar = bar
        except Exception as exc:  # noqa: BLE001
            logger.warning("HTF signal persist failed: %s", exc)

    def snapshot(self) -> dict:
        return {"ts": datetime.now(timezone.utc).isoformat(),
                "config": {"mode": "htf_swing", "top_k": self.top_k, "tradeable": False},
                "performance": module_performance("multi_ticker_swing_htf"),
                "scan": self._scan()}


_PAGE = """<!doctype html><html><head><meta charset=utf-8><title>HTF Swing</title>
__THEME_LINK__
<style>
.wrap{margin:14px 18px}
h2{margin:14px 0 6px;font-size:15px}
table{border-collapse:collapse;width:100%;margin:8px 0;max-width:540px}
td,th{border:1px solid var(--border);padding:4px 8px;text-align:right}th{background:var(--panel2)}
td:first-child,th:first-child,td.t,th.t{text-align:left}
</style></head><body>
__NAV_HTML__
<div class=wrap>
<h2>HTF Swing <span id=cfg class=pill>read-only</span> <span id=age class=muted></span></h2>
<div class=muted>Higher-timeframe swing ranking (4H bar) — htf_score read live off the shared Meta matrix (same signal &amp; eligibility the HTF runner trades).</div>
<h2>Top ranked names</h2><table id=scan></table>
</div>
<script>
function f(n,d=4){return n==null?'-':Number(n).toLocaleString(undefined,{maximumFractionDigits:d})}
async function tick(){let s=await(await fetch('/api/state')).json();
let sc=s.scan||{};
document.getElementById('age').textContent=sc.error?sc.error:('bar '+(sc.bar||'-')+' · '+(sc.age_days)+'d old');
document.getElementById('scan').innerHTML='<tr><th>#</th><th class=t>ticker</th><th>htf score</th></tr>'+
(sc.picks||[]).map(p=>'<tr><td>'+p.rank+'</td><td class=t>'+p.ticker+'</td><td>'+f(p.score)+'</td></tr>').join('');
let badge=document.getElementById('cyno-live-badge');if(badge){badge.textContent='signals';badge.className='live-badge';}}
tick();setInterval(tick,10000);
</script></body></html>""".replace("__THEME_LINK__", THEME_LINK).replace("__NAV_HTML__", NAV_HTML)


class HTFHTTPServer(ThreadingHTTPServer):
    app: HTFDashboardApp


class HTFHandler(BaseHTTPRequestHandler):
    server_version = "CynolycusHTF/1.0"

    def log_message(self, fmt, *a):  # noqa: A002
        return

    def _app(self) -> HTFDashboardApp:
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


def make_server(host: str, port: int, app: HTFDashboardApp) -> HTFHTTPServer:
    srv = HTFHTTPServer((host, port), HTFHandler)
    srv.daemon_threads = True
    srv.app = app
    return srv
