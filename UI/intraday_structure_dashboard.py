"""Read-only dashboard/API for the deterministic Intraday Structure Engine."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from UI.ui_chrome import NAV_HTML, THEME_LINK, serve_theme_css
from strategies.intraday_structure.config import IntradayStructureConfig
from strategies.intraday_structure.models import Candidate
from strategies.intraday_structure.runner import IntradayStructureRunner


class IntradayStructureDashboardApp:
    def __init__(self, config: IntradayStructureConfig, bar_queue) -> None:
        self.runner = IntradayStructureRunner(config, bar_queue)
        self.runner.start()

    def snapshot(self) -> dict:
        return {"ts": datetime.now(timezone.utc).isoformat(), **self.runner.snapshot()}

    def add_candidate(self, payload: dict) -> dict:
        candidate = Candidate.from_mapping({
            **payload,
            "timestamp": payload.get("timestamp") or datetime.now(timezone.utc).isoformat(),
            "source": payload.get("source") or "manual_dashboard",
        })
        changed = self.runner.engine.register_candidate(candidate)
        return {"ok": True, "changed": changed, "candidate": candidate.to_dict()}

    def stop(self) -> None:
        self.runner.stop()


_PAGE = """<!doctype html><html><head><meta charset=utf-8><title>Intraday Structure</title>
__THEME_LINK__
<style>
.wrap{margin:14px 18px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:10px}
.setup{border:1px solid var(--border);background:var(--panel);padding:10px;border-radius:6px}.head{display:flex;gap:8px;align-items:center}
.ticker{font-size:17px;font-weight:800}.muted{color:var(--muted)}.evidence{font-size:11px;color:var(--muted);margin-top:6px}
table{border-collapse:collapse;width:100%}td,th{border-bottom:1px solid var(--border);padding:5px;text-align:left}
</style></head><body>__NAV_HTML__<div class=wrap>
<div class=head><h2>Intraday Structure Engine</h2><span class="pill paper">paper-only confirmation</span><span id=status class=muted></span></div>
<div class=muted>Stateful 1-minute setup confirmation. No order-submission path exists in v1.</div>
<h3>Active setups</h3><div id=setups class=grid></div><h3>Recent transitions</h3><table id=timeline></table>
</div><script>
function f(v,d=2){return v==null?'-':Number(v).toFixed(d)}
function esc(v){return String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}
async function tick(){let s=await(await fetch('/api/state')).json();
document.getElementById('status').textContent=(s.running?'running':'stopped')+' · '+s.candidate_count+' candidates';
document.getElementById('setups').innerHTML=(s.active_signals||[]).map(x=>`<div class=setup><div class=head><span class=ticker>${esc(x.ticker)}</span><span class=pill>${esc(x.state)}</span><span>${esc(x.direction)}</span></div><b>${esc(x.setup_type)}</b><div>spot ${f(x.spot)} · pivot ${f(x.pivot)} · stop ${f(x.invalidation)}</div><div>target ${f(x.active_target)} · runway ${f(x.runway_score)} · confidence ${f(x.confidence)}</div><div class=evidence>${(x.evidence||[]).map(esc).join(' · ')}</div></div>`).join('')||'<span class=muted>No active detected setups.</span>';
let rows=(s.recent_transitions||[]).slice().reverse();document.getElementById('timeline').innerHTML='<tr><th>time</th><th>ticker</th><th>setup</th><th>transition</th><th>reason</th></tr>'+rows.map(x=>`<tr><td>${esc(x.timestamp)}</td><td>${esc(x.ticker)}</td><td>${esc(x.setup_type)}</td><td>${esc(x.from_state)} → ${esc(x.to_state)}</td><td>${esc(x.reason)}</td></tr>`).join('');}
tick();setInterval(tick,5000);
</script></body></html>""".replace("__THEME_LINK__", THEME_LINK).replace("__NAV_HTML__", NAV_HTML)


class IntradayStructureHTTPServer(ThreadingHTTPServer):
    app: IntradayStructureDashboardApp


class IntradayStructureHandler(BaseHTTPRequestHandler):
    server_version = "CynolycusIntradayStructure/1.0"

    def log_message(self, fmt, *args):  # noqa: A002
        return

    def _send(self, payload: Any, *, status=HTTPStatus.OK, content_type="application/json; charset=utf-8") -> None:
        body = payload if isinstance(payload, bytes) else json.dumps(payload, allow_nan=False, default=str).encode("utf-8")
        self.send_response(int(status)); self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body))); self.send_header("Cache-Control", "no-store")
        self.end_headers(); self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        if self.path == "/" or self.path.startswith("/index"):
            self._send(_PAGE.encode("utf-8"), content_type="text/html; charset=utf-8")
        elif self.path == "/static/cynolycus_theme.css":
            serve_theme_css(self)
        elif self.path.startswith("/api/state"):
            self._send(self.server.app.snapshot())  # type: ignore[attr-defined]
        else:
            self._send({"error": "not_found"}, status=HTTPStatus.NOT_FOUND)

    def do_POST(self):  # noqa: N802
        if not self.path.startswith("/api/candidates"):
            self._send({"error": "not_found"}, status=HTTPStatus.NOT_FOUND); return
        try:
            length = int(self.headers.get("Content-Length", "0") or 0)
            payload = json.loads(self.rfile.read(length) or b"{}")
            self._send(self.server.app.add_candidate(payload))  # type: ignore[attr-defined]
        except Exception as exc:
            self._send({"ok": False, "error": str(exc)}, status=HTTPStatus.BAD_REQUEST)


def make_server(host: str, port: int, app: IntradayStructureDashboardApp) -> IntradayStructureHTTPServer:
    server = IntradayStructureHTTPServer((host, port), IntradayStructureHandler)
    server.daemon_threads = True
    server.app = app
    return server
