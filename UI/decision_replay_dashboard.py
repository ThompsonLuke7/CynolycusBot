"""Decision replay dashboard: step a position bar by bar beside what the model saw.

Pick a position (or any module + ticker + window). The page then replays it one
bar at a time and only ever shows what existed at the cursor:

  * the price bars up to the cursor (later bars stay hidden unless "show
    future bars" is ticked, which is labelled as hindsight), zoomable and
    pannable, with the module's own indicators overlaid: momentum's EMAs and
    20/60/52-week ceilings and floors on bar charts, and the intraday engine's
    VWAP, EMAs, opening range, prior-day range and its clustered level map at
    the cursor on 1-minute charts;
  * the module's decision on each bar: score, rank, every gate, the action,
    and for archived runs the full feature vector with each value's percentile
    among that run's candidates, its change since the entry decision, and the
    price levels it implies, drawn against the chart's own lines;
  * the orders the module planned, the contract it traded, and the fills;
  * the position at the cursor: gain against the entry reference, excursions,
    runs held against the horizon, distance to the exit levels, which are
    drawn from the cursor rightwards only while they apply.

Read-only. It never writes and never contacts a broker. Data assembly and
timing rules live in ``UI/decision_replay.py``; the page is
``UI/decision_replay_static/replay.html``, read per request.

Endpoints:
  GET /                                   HTML page
  GET /api/state                          hub status payload
  GET /api/positions?module&ticker&status&outcome    the picker
  GET /api/replay?id=…&tf=…               one position
  GET /api/replay?module&ticker&start&end&tf         any ticker and window
  GET /api/run?module&run_id&ticker&ref   one archived run: features, gates, picks
  GET /api/engine?ticker&at[&id]          intraday engine features + level map at one bar
  GET /api/feature_series?module&ticker&start&end&name   one archived feature over a window
"""
from __future__ import annotations

import json
import logging
import math
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from UI import decision_replay as dr
from UI.ui_chrome import NAV_HTML, THEME_LINK, serve_theme_css

logger = logging.getLogger(__name__)
DEFAULT_PORT = 8779
PAGE_PATH = Path(__file__).resolve().parent / "decision_replay_static" / "replay.html"


def _config() -> str:
    """Module list and timeframe rules for the page, embedded at render time."""
    return json.dumps({
        "modules": list(dr.KNOWN_MODULES),
        "tf_options": {m: dr.timeframe_options(m) for m in dr.KNOWN_MODULES},
        "default_tf": {m: dr.default_timeframe(m) for m in dr.KNOWN_MODULES},
    }).replace("</", "<\\/")


def render_page() -> bytes:
    page = PAGE_PATH.read_text(encoding="utf-8")
    return (page.replace("__THEME__", THEME_LINK).replace("__NAV__", NAV_HTML)
            .replace("__CONFIG__", _config())).encode("utf-8")


def engine_payload(q: dict[str, str]) -> dict[str, Any]:
    """Engine view at one bar. With a position id, the position's own setup
    supplies the candidate (its timestamp anchors the candidate VWAP)."""
    setup, direction = None, None
    if q.get("id"):
        pos = dr._find_position(q["id"])
        if pos is None:
            return {"error": f"no such position: {q['id']}"}
        setup = dr.setup_record(pos.get("setup_id"), pos.get("entry_filled_at") or pos.get("entry_submitted_at"))
        contract = dr.contract_info(pos) or {}
        direction = (setup or {}).get("direction") or {"put": "short", "call": "long"}.get(contract.get("right"))
    return dr.engine_view(ticker=q.get("ticker", ""), at=q.get("at"), setup=setup, direction=direction)


def _finite(obj: Any) -> Any:
    """``obj`` with every non-finite float replaced by None."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _finite(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_finite(v) for v in obj]
    return obj


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # noqa: A003 - quiet the stdlib access log
        logger.debug("decision-replay %s", fmt % args)

    def _send(self, body: bytes, status=HTTPStatus.OK, ctype="application/json"):
        # A client leaving mid-response is normal HTTP (the hub polls /api/state
        # with a short timeout); see UI/tests/test_dashboard_broken_pipe_guard.py.
        try:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True

    def _json(self, payload: Any, status=HTTPStatus.OK):
        # NaN/Infinity are not JSON and break the page's JSON.parse; a cached bar
        # can carry a NaN open/high/low, so those become null.
        try:
            body = json.dumps(payload, default=str, allow_nan=False)
        except ValueError:
            body = json.dumps(_finite(payload), default=str, allow_nan=False)
        self._send(body.encode(), status)

    def do_GET(self):  # noqa: N802 - stdlib interface
        parsed = urlparse(self.path)
        path = parsed.path
        q = {k: v[0].strip() for k, v in parse_qs(parsed.query).items() if v}
        try:
            if path == "/":
                self._send(render_page(), ctype="text/html; charset=utf-8")
            elif path == "/static/cynolycus_theme.css":
                serve_theme_css(self)
            elif path == "/api/state":
                self._json(dr.state())
            elif path == "/api/positions":
                self._json(dr.positions(q))
            elif path == "/api/replay":
                self._json(dr.replay(position_id=q.get("id"), module=q.get("module"),
                                     ticker=q.get("ticker"), start=q.get("start"),
                                     end=q.get("end"), tf=q.get("tf")))
            elif path == "/api/run":
                self._json(dr.run_detail(q.get("module", ""), q.get("run_id", ""),
                                         q.get("ticker", ""), q.get("ref") or None))
            elif path == "/api/engine":
                self._json(engine_payload(q))
            elif path == "/api/feature_series":
                self._json(dr.feature_series(q.get("module", ""), q.get("ticker", ""), q.get("start"),
                                             q.get("end"), q.get("name", "")))
            else:
                self._json({"error": "not_found"}, HTTPStatus.NOT_FOUND)
        except Exception as exc:  # noqa: BLE001 - a dashboard must not take the server down
            logger.exception("decision-replay request failed: %s", path)
            self._json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)


def make_server(host: str = "127.0.0.1", port: int = DEFAULT_PORT) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), Handler)


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Decision replay dashboard (read-only).")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    print(f"Decision replay: http://{args.host}:{args.port}/")
    make_server(args.host, args.port).serve_forever()


if __name__ == "__main__":
    main()
