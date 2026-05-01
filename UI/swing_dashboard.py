"""
Multi-Ticker Swing Live Dashboard.

A standalone HTTP + Server-Sent-Events dashboard that controls and visualises a
SwingLiveRunner session. Run it like the intraday SPY dashboard, but on its own
port (default 8766) and against its own HTML page (UI/swing_index.html).

  python -m UI.swing_dashboard --host 127.0.0.1 --port 8766

Routes:
  GET  /                → swing_index.html
  GET  /api/state       → snapshot JSON
  GET  /api/events      → SSE stream of session events
  POST /api/start       → start a session ({"max_entries": 5, "dry_run": true})
  POST /api/stop        → stop the running session
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import queue as queue_mod
import threading
import time
import traceback
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Cap how many of each kind we retain server-side for late-joining clients.
MAX_EVENTS_LOG  = 500   # generic event log (signals, confirmations, scans, ...)
MAX_ORDERS_LOG  = 500   # alpaca order submissions (buy/sell/dry/failed)
MAX_TRADES_LOG  = 500   # closed positions
MAX_WARMUP_LOG  = 600   # warmup progress lines


# ---------------------------------------------------------------------------
# JSON safety
# ---------------------------------------------------------------------------

def _json_safe(obj: Any) -> Any:
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, (list, tuple, set)):
        return [_json_safe(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, datetime):
        return obj.astimezone(timezone.utc).isoformat()
    if hasattr(obj, "isoformat"):
        try:
            return obj.isoformat()
        except Exception:
            return str(obj)
    return str(obj)


def _utc_iso(ts: datetime | None = None) -> str:
    return (ts or datetime.now(timezone.utc)).astimezone(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Audit writer (JSONL) — every event also goes to disk
# ---------------------------------------------------------------------------

class AuditWriter:
    def __init__(self, root: Path) -> None:
        self._root = root
        self._fh = None
        self._lock = threading.Lock()
        self._path: Path | None = None

    def start(self) -> Path:
        self._root.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self._path = self._root / f"swing_session_{ts}.jsonl"
        self._fh = open(self._path, "a", buffering=1, encoding="utf-8")
        return self._path

    def write(self, event: dict) -> None:
        if self._fh is None:
            return
        with self._lock:
            try:
                self._fh.write(json.dumps(_json_safe(event)) + "\n")
            except Exception as exc:
                logger.warning("audit write failed: %s", exc)

    def stop(self) -> None:
        with self._lock:
            if self._fh is not None:
                try:
                    self._fh.close()
                except Exception:
                    pass
                self._fh = None


# ---------------------------------------------------------------------------
# Event broker (SSE fan-out)
# ---------------------------------------------------------------------------

class EventBroker:
    def __init__(self) -> None:
        self._subs: list[queue_mod.Queue] = []
        self._lock = threading.Lock()

    def subscribe(self, max_buffer: int = 1000) -> queue_mod.Queue:
        q: queue_mod.Queue = queue_mod.Queue(maxsize=max_buffer)
        with self._lock:
            self._subs.append(q)
        return q

    def unsubscribe(self, q: queue_mod.Queue) -> None:
        with self._lock:
            if q in self._subs:
                self._subs.remove(q)

    def publish(self, event: dict) -> None:
        with self._lock:
            subs = list(self._subs)
        for q in subs:
            try:
                q.put_nowait(event)
            except queue_mod.Full:
                # Drop the oldest item to keep the queue bounded
                try:
                    q.get_nowait()
                    q.put_nowait(event)
                except Exception:
                    pass


# ---------------------------------------------------------------------------
# Snapshot store
# ---------------------------------------------------------------------------

@dataclass
class _Status:
    state: str = "idle"             # idle | warming | running | stopping | stopped | error
    started_at: str | None = None
    stopped_at: str | None = None
    error: str | None = None
    dry_run: bool = True
    max_entries: int = 5
    env_file: str = ".env"
    warmup_index: int = 0
    warmup_total: int = 0
    warmup_message: str = ""
    universe_size: int = 0
    stream_symbols: int = 0
    bar_count: int = 0
    last_bar_ts: str | None = None
    confirming_count: int = 0
    open_positions_count: int = 0


class SwingDashboardStore:
    """Server-side mirror of session state — used to seed late-joining SSE clients."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._status = _Status()
        self._positions: list[dict] = []
        self._confirming: list[dict] = []
        self._scans_recent: deque[dict] = deque(maxlen=50)
        self._events: deque[dict] = deque(maxlen=MAX_EVENTS_LOG)
        self._orders: deque[dict] = deque(maxlen=MAX_ORDERS_LOG)
        self._trades: deque[dict] = deque(maxlen=MAX_TRADES_LOG)
        self._warmup: deque[dict] = deque(maxlen=MAX_WARMUP_LOG)

    def reset_for_session(self, *, max_entries: int, dry_run: bool, env_file: str) -> None:
        with self._lock:
            self._status = _Status(
                state="warming",
                started_at=_utc_iso(),
                dry_run=dry_run,
                max_entries=max_entries,
                env_file=env_file,
            )
            self._positions.clear()
            self._confirming.clear()
            self._scans_recent.clear()
            self._events.clear()
            self._orders.clear()
            self._trades.clear()
            self._warmup.clear()

    # -- mutation helpers (called from runner thread via event sink) ----

    def set_state(self, state: str, *, error: str | None = None) -> None:
        with self._lock:
            self._status.state = state
            if state in ("stopped", "error"):
                self._status.stopped_at = _utc_iso()
                if error is not None:
                    self._status.error = error

    def set_meta(self, meta: dict) -> None:
        with self._lock:
            self._status.universe_size = int(meta.get("universe_size", 0))
            self._status.stream_symbols = int(meta.get("stream_symbols", 0))
            self._status.bar_count = int(meta.get("bar_count", 0))
            self._status.last_bar_ts = meta.get("last_bar_ts")
            self._status.confirming_count = int(meta.get("confirming_count", 0))
            self._status.open_positions_count = int(meta.get("open_positions_count", 0))

    def set_positions(self, positions: list[dict]) -> None:
        with self._lock:
            self._positions = list(positions)
            self._status.open_positions_count = len(positions)

    def set_confirming(self, confirming: list[dict]) -> None:
        with self._lock:
            self._confirming = list(confirming)
            self._status.confirming_count = len(confirming)

    def append_event(self, event: dict) -> None:
        with self._lock:
            self._events.append(event)

    def append_order(self, order: dict) -> None:
        with self._lock:
            self._orders.append(order)

    def append_trade(self, trade: dict) -> None:
        with self._lock:
            self._trades.append(trade)

    def append_scan(self, scan: dict) -> None:
        with self._lock:
            self._scans_recent.append(scan)

    def append_warmup(self, line: dict) -> None:
        with self._lock:
            self._warmup.append(line)
            if "index" in line and "total" in line:
                self._status.warmup_index = int(line["index"])
                self._status.warmup_total = int(line["total"])

    def set_warmup_message(self, message: str) -> None:
        with self._lock:
            self._status.warmup_message = message

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "status": self._status.__dict__.copy(),
                "positions": list(self._positions),
                "confirming": list(self._confirming),
                "scans_recent": list(self._scans_recent),
                "events": list(self._events),
                "orders": list(self._orders),
                "trades": list(self._trades),
                "warmup": list(self._warmup)[-30:],  # last 30 lines is enough to seed
                "ts": _utc_iso(),
            }


# ---------------------------------------------------------------------------
# Session — owns the runner thread
# ---------------------------------------------------------------------------

class SwingSession:
    def __init__(self, store: SwingDashboardStore, broker: EventBroker, audit_root: Path) -> None:
        self._store = store
        self._broker = broker
        self._audit_root = audit_root
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._runner: Any = None  # SwingLiveRunner; lazy-imported
        self._audit: AuditWriter | None = None
        self._meta_thread: threading.Thread | None = None
        self._meta_stop = threading.Event()

    # ------------------------------------------------------------------

    def is_running(self) -> bool:
        with self._lock:
            return self._thread is not None and self._thread.is_alive()

    # ------------------------------------------------------------------
    # Event sink — called from runner thread
    # ------------------------------------------------------------------

    def _publish(self, event: dict) -> None:
        # Persist to disk + fan out to SSE
        if self._audit is not None:
            self._audit.write(event)
        self._broker.publish(event)

    def _on_event(self, kind: str, payload: dict) -> None:
        ts = _utc_iso()
        evt = {"type": kind, "ts": ts, "payload": payload}

        # Update server-side store first so /api/state reflects reality immediately.
        if kind == "warmup_start":
            self._store.set_state("warming")
            self._store.set_warmup_message(payload.get("message", ""))
            self._store.append_warmup({"ts": ts, "kind": "start",
                                        "message": payload.get("message", ""),
                                        "tickers": payload.get("tickers")})
        elif kind == "warmup_progress":
            self._store.append_warmup({"ts": ts, **payload})
        elif kind == "warmup_done":
            cached = payload.get("cached", 0)
            loaded = payload.get("loaded", 0)
            gap_filled = payload.get("gap_filled", 0)
            failed = payload.get("failed", 0)
            self._store.set_warmup_message(
                f"Warmup complete · cached={cached} · loaded={loaded} "
                f"· gap-filled={gap_filled} · failed={failed}"
            )
            self._store.append_warmup({"ts": ts, "kind": "done", **payload})
        elif kind == "stream_started":
            self._store.set_state("running")
            self._store.append_event(evt)
        elif kind == "signal":
            self._store.append_event(evt)
            self._refresh_runner_state()
        elif kind == "confirmation":
            self._store.append_event(evt)
            self._refresh_runner_state()
        elif kind == "confirmation_expired":
            self._store.append_event(evt)
            self._refresh_runner_state()
        elif kind == "scan":
            sigs = payload.get("signals") or []
            if sigs:
                self._store.append_scan({"ts": ts, **payload})
                # Skip noisy zero-signal scans from the event log; keep them in scans_recent only
            self._refresh_runner_state()
        elif kind in ("order_submitted", "order_failed", "order_dry_run", "entry_skipped"):
            self._store.append_order({"ts": ts, "kind": kind, **payload})
            self._store.append_event(evt)
        elif kind == "position_opened":
            self._store.append_event(evt)
            self._refresh_runner_state()
        elif kind == "position_closed":
            self._store.append_trade({"ts": ts, **payload})
            self._store.append_event(evt)
            self._refresh_runner_state()
        elif kind == "stream_error":
            error_msg = payload.get("error", "unknown stream error")
            self._store.set_state("error", error=error_msg)
            self._store.set_warmup_message(
                f"Stream error: {error_msg}"
            )
            self._store.append_event(evt)
        elif kind == "stopped":
            self._store.set_state("stopped")
        else:
            self._store.append_event(evt)

        self._publish(evt)

    def _refresh_runner_state(self) -> None:
        runner = self._runner
        if runner is None:
            return
        try:
            self._store.set_positions(runner.snapshot_positions())
            self._store.set_confirming(runner.snapshot_confirming())
            self._store.set_meta(runner.snapshot_meta())
        except Exception as exc:
            logger.warning("snapshot refresh failed: %s", exc)

    # ------------------------------------------------------------------
    # Periodic meta-refresh (so bar_count/last_bar_ts tick on the UI)
    # ------------------------------------------------------------------

    def _meta_loop(self) -> None:
        # Periodically re-publish meta + positions + confirming so the dashboard
        # reflects per-bar P&L drift even when no entry/exit event has fired.
        while not self._meta_stop.wait(2.0):
            runner = self._runner
            if runner is None:
                continue
            try:
                meta = runner.snapshot_meta()
                positions = runner.snapshot_positions()
                confirming = runner.snapshot_confirming()
                self._store.set_meta(meta)
                self._store.set_positions(positions)
                self._store.set_confirming(confirming)
                self._broker.publish({
                    "type": "meta_tick",
                    "ts": _utc_iso(),
                    "payload": {
                        "meta": meta,
                        "positions": positions,
                        "confirming": confirming,
                    },
                })
            except Exception:
                pass

    # ------------------------------------------------------------------

    def start(self, *, max_entries: int, dry_run: bool, env_file: str) -> dict:
        if self.is_running():
            raise RuntimeError("Session already running.")

        # Lazy import to keep the dashboard importable even without xgboost installed.
        from multi_ticker_swing.live.runner import SwingLiveRunner

        self._store.reset_for_session(
            max_entries=max_entries, dry_run=dry_run, env_file=env_file,
        )

        audit = AuditWriter(self._audit_root)
        audit_path = audit.start()
        logger.info("Audit log: %s", audit_path)
        self._audit = audit

        try:
            runner = SwingLiveRunner(
                env_file=env_file,
                dry_run=dry_run,
                max_entries_per_bar=max_entries,
                event_sink=self._on_event,
            )
        except Exception as exc:
            err = f"runner_init_failed: {exc}"
            logger.error("%s\n%s", err, traceback.format_exc(limit=12))
            self._store.set_state("error", error=err)
            self._publish({"type": "error", "ts": _utc_iso(), "payload": {"error": err}})
            audit.stop()
            self._audit = None
            raise

        self._runner = runner
        self._store.set_meta({
            "universe_size": runner.universe_size,
            "stream_symbols": len(runner.stream_symbols),
            "bar_count": 0,
            "last_bar_ts": None,
            "confirming_count": 0,
            "open_positions_count": 0,
        })

        self._meta_stop.clear()
        self._meta_thread = threading.Thread(target=self._meta_loop, daemon=True)
        self._meta_thread.start()

        def _run() -> None:
            try:
                runner.start()
            except Exception as exc:
                err = f"runner_crashed: {exc}"
                logger.error("%s\n%s", err, traceback.format_exc(limit=12))
                self._store.set_state("error", error=err)
                self._publish({
                    "type": "error", "ts": _utc_iso(),
                    "payload": {"error": err, "traceback": traceback.format_exc(limit=12)},
                })
            finally:
                self._meta_stop.set()
                self._store.set_state("stopped")
                self._publish({"type": "session_ended", "ts": _utc_iso(), "payload": {}})
                if self._audit is not None:
                    self._audit.stop()
                self._audit = None

        thread = threading.Thread(target=_run, daemon=True, name="SwingLiveRunner")
        with self._lock:
            self._thread = thread
        thread.start()
        self._publish({
            "type": "session_started",
            "ts": _utc_iso(),
            "payload": {
                "max_entries": max_entries,
                "dry_run": dry_run,
                "env_file": env_file,
                "stream_symbols": len(runner.stream_symbols),
                "universe_size": runner.universe_size,
                "audit_log": str(audit_path),
            },
        })
        return self._store.snapshot()

    def stop(self) -> dict:
        runner = self._runner
        if runner is None or not self.is_running():
            self._store.set_state("stopped")
            return self._store.snapshot()
        self._store.set_state("stopping")
        try:
            runner.stop()
        except Exception as exc:
            logger.warning("runner.stop() raised: %s", exc)
        with self._lock:
            t = self._thread
        if t is not None:
            t.join(timeout=10.0)
        self._meta_stop.set()
        if self._meta_thread is not None:
            self._meta_thread.join(timeout=2.0)
        with self._lock:
            self._thread = None
        self._runner = None
        self._store.set_state("stopped")
        return self._store.snapshot()


# ---------------------------------------------------------------------------
# HTTP server
# ---------------------------------------------------------------------------

class SwingDashboardApp:
    def __init__(self, audit_root: Path) -> None:
        self.store = SwingDashboardStore()
        self.broker = EventBroker()
        self.session = SwingSession(self.store, self.broker, audit_root)

    def snapshot(self) -> dict:
        snap = self.store.snapshot()
        snap["session_alive"] = self.session.is_running()
        return snap

    def start(self, payload: dict) -> dict:
        max_entries = int(payload.get("max_entries", 5) or 5)
        if max_entries < 1:
            max_entries = 1
        # Dry-run is always ON for now (paper account safety).
        dry_run = True
        env_file = str(payload.get("env_file") or ".env")
        return self.session.start(
            max_entries=max_entries, dry_run=dry_run, env_file=env_file,
        )

    def stop(self) -> dict:
        return self.session.stop()


class SwingDashboardHTTPServer(ThreadingHTTPServer):
    app: SwingDashboardApp


class SwingDashboardHandler(BaseHTTPRequestHandler):
    server_version = "CynolycusSwingDashboard/1.0"

    # Quieter access logs
    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        if self.path.startswith("/api/events"):
            return
        super().log_message(format, *args)

    def _read_json_body(self) -> dict:
        length = int(self.headers.get("Content-Length", "0") or 0)
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        if not raw:
            return {}
        return json.loads(raw.decode("utf-8"))

    def _write_json(self, payload: dict, status: int = HTTPStatus.OK) -> None:
        body = json.dumps(_json_safe(payload)).encode("utf-8")
        self.send_response(int(status))
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _write_text(self, body: str, *, status: int = HTTPStatus.OK,
                    content_type: str = "text/plain") -> None:
        blob = body.encode("utf-8")
        self.send_response(int(status))
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.send_header("Content-Length", str(len(blob)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(blob)

    def _serve_index(self) -> None:
        root = Path(__file__).resolve().parent
        index_path = root / "swing_index.html"
        if not index_path.exists():
            self._write_text("Missing UI/swing_index.html",
                             status=HTTPStatus.NOT_FOUND)
            return
        html = index_path.read_text(encoding="utf-8")
        self._write_text(html, status=HTTPStatus.OK, content_type="text/html")

    def _app(self) -> SwingDashboardApp:
        server = self.server
        if not isinstance(server, SwingDashboardHTTPServer):
            raise RuntimeError("Handler attached to unexpected server type.")
        return server.app

    def do_GET(self) -> None:  # noqa: N802
        from urllib.parse import urlparse
        parsed = urlparse(self.path)
        app = self._app()

        if parsed.path in {"/", "/index.html", "/swing"}:
            self._serve_index()
            return

        if parsed.path == "/api/state":
            self._write_json(app.snapshot())
            return

        if parsed.path == "/api/events":
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            q = app.broker.subscribe()
            try:
                hello = {"type": "hello", "ts": _utc_iso(),
                         "payload": app.snapshot()}
                self.wfile.write(f"data: {json.dumps(_json_safe(hello))}\n\n".encode("utf-8"))
                self.wfile.flush()
                while True:
                    try:
                        event = q.get(timeout=12.0)
                    except queue_mod.Empty:
                        self.wfile.write(b": keepalive\n\n")
                        self.wfile.flush()
                        continue
                    data = json.dumps(_json_safe(event))
                    self.wfile.write(f"data: {data}\n\n".encode("utf-8"))
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                app.broker.unsubscribe(q)
            return

        self._write_json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
        from urllib.parse import urlparse
        parsed = urlparse(self.path)
        app = self._app()
        try:
            payload = self._read_json_body()
        except Exception as exc:
            self._write_json({"error": f"invalid_json: {exc}"},
                             status=HTTPStatus.BAD_REQUEST)
            return

        try:
            if parsed.path == "/api/start":
                state = app.start(payload)
                self._write_json({"ok": True, "state": state})
                return
            if parsed.path == "/api/stop":
                state = app.stop()
                self._write_json({"ok": True, "state": state})
                return
        except Exception as exc:
            self._write_json({"ok": False, "error": str(exc)},
                             status=HTTPStatus.BAD_REQUEST)
            return

        self._write_json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)


def run_server(*, host: str, port: int, audit_root: Path) -> None:
    app = SwingDashboardApp(audit_root=audit_root)
    server = SwingDashboardHTTPServer((host, int(port)), SwingDashboardHandler)
    server.daemon_threads = True
    server.app = app

    print(f"[swing-ui] Multi-Ticker Swing Live Dashboard: http://{host}:{port}")
    print("[swing-ui] Click 'Run' in the browser to start a session.")
    print(f"[swing-ui] Audit logs: {audit_root}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("[swing-ui] Shutting down...")
    finally:
        try:
            app.stop()
        except Exception:
            pass
        server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run local web dashboard for the multi-ticker swing live runner.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host interface.")
    parser.add_argument("--port", type=int, default=8766, help="HTTP port.")
    parser.add_argument(
        "--audit-root",
        default=str(Path(__file__).resolve().parent / "swing_audit"),
        help="Directory for JSONL audit logs.",
    )
    args = parser.parse_args()
    run_server(host=args.host, port=args.port, audit_root=Path(args.audit_root))


if __name__ == "__main__":
    main()
