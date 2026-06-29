"""Dealer positioning dashboard and session controller."""
from __future__ import annotations

import argparse
import json
import logging
import math
import queue as queue_mod
import threading
import traceback
from collections import deque
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from strategies.dealer_positioning.config import DealerPositioningConfig

logger = logging.getLogger(__name__)


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_safe(obj: Any) -> Any:
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, float):
        return None if math.isnan(obj) or math.isinf(obj) else obj
    if isinstance(obj, (list, tuple, set, deque)):
        return [_json_safe(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, datetime):
        return obj.astimezone(timezone.utc).isoformat()
    return str(obj)


class EventBroker:
    def __init__(self) -> None:
        self._subs: list[queue_mod.Queue] = []
        self._lock = threading.Lock()

    def subscribe(self) -> queue_mod.Queue:
        q: queue_mod.Queue = queue_mod.Queue(maxsize=1000)
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
                try:
                    q.get_nowait()
                    q.put_nowait(event)
                except Exception:
                    pass


class DealerDashboardStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state = "idle"
        self._error: str | None = None
        self._started_at: str | None = None
        self._stopped_at: str | None = None
        self._snapshot: dict[str, Any] = {}
        self._events: deque[dict] = deque(maxlen=300)

    def reset(self) -> None:
        with self._lock:
            self._state = "running"
            self._error = None
            self._started_at = _utc_iso()
            self._stopped_at = None
            self._snapshot = {}
            self._events.clear()

    def set_state(self, state: str, *, error: str | None = None) -> None:
        with self._lock:
            self._state = state
            self._error = error
            if state in {"stopped", "error"}:
                self._stopped_at = _utc_iso()

    def update_snapshot(self, snapshot: dict[str, Any]) -> None:
        with self._lock:
            self._snapshot = snapshot

    def append_event(self, event: dict) -> None:
        with self._lock:
            self._events.appendleft(event)

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "state": self._state,
                "error": self._error,
                "started_at": self._started_at,
                "stopped_at": self._stopped_at,
                "runner": _json_safe(self._snapshot),
                "events": list(self._events),
            }


class DealerDashboardApp:
    def __init__(self, *, bar_queue: queue_mod.Queue | None = None) -> None:
        self.store = DealerDashboardStore()
        self.broker = EventBroker()
        self._bar_queue = bar_queue
        self._runner = None
        self._thread: threading.Thread | None = None
        self._meta_thread: threading.Thread | None = None
        self._meta_stop = threading.Event()
        self._lock = threading.Lock()

    def snapshot(self) -> dict:
        snap = self.store.snapshot()
        snap["session_alive"] = self.is_running()
        return snap

    def is_running(self) -> bool:
        with self._lock:
            return bool(self._thread and self._thread.is_alive())

    def start(self, payload: dict) -> dict:
        if self.is_running():
            raise RuntimeError("Dealer positioning session already running.")
        from strategies.dealer_positioning.runner import DealerPositioningRunner

        config = DealerPositioningConfig.from_env()
        symbols = payload.get("symbols")
        if symbols:
            if isinstance(symbols, str):
                parsed_symbols = tuple(x.strip().upper() for x in symbols.split(",") if x.strip())
            else:
                parsed_symbols = tuple(str(x).strip().upper() for x in symbols if str(x).strip())
            config = DealerPositioningConfig(**{**config.__dict__, "symbols": parsed_symbols})
        if payload.get("poll_seconds"):
            config = DealerPositioningConfig(**{**config.__dict__, "poll_seconds": int(payload["poll_seconds"])})
        if "submit_orders" in payload:
            config = DealerPositioningConfig(**{**config.__dict__, "submit_orders": bool(payload["submit_orders"])})
        if payload.get("alpaca_env_file"):
            config = DealerPositioningConfig(**{**config.__dict__, "alpaca_env_file": str(payload["alpaca_env_file"])})

        self.store.reset()
        runner = DealerPositioningRunner(config=config, event_sink=self._on_event, bar_queue=self._bar_queue)
        self._runner = runner
        self.store.update_snapshot(runner.snapshot())

        self._meta_stop.clear()
        self._meta_thread = threading.Thread(target=self._meta_loop, daemon=True, name="DealerPositioningMeta")
        self._meta_thread.start()

        def _run() -> None:
            try:
                runner.start()
            except Exception as exc:
                err = f"runner_crashed: {exc}"
                logger.error("%s\n%s", err, traceback.format_exc(limit=12))
                self.store.set_state("error", error=err)
                self._publish({"type": "error", "ts": _utc_iso(), "payload": {"error": err}})
            finally:
                self._meta_stop.set()
                if self.store.snapshot()["state"] != "error":
                    self.store.set_state("stopped")
                self._publish({"type": "session_ended", "ts": _utc_iso(), "payload": {}})

        thread = threading.Thread(target=_run, daemon=True, name="DealerPositioningRunner")
        with self._lock:
            self._thread = thread
        thread.start()
        self._publish({"type": "session_started", "ts": _utc_iso(), "payload": {"config": runner.snapshot()["config"]}})
        return self.snapshot()

    def stop(self) -> dict:
        runner = self._runner
        if runner is not None:
            runner.stop()
        with self._lock:
            thread = self._thread
        if thread is not None:
            thread.join(timeout=10)
        self._meta_stop.set()
        if self._meta_thread is not None:
            self._meta_thread.join(timeout=2)
        self.store.set_state("stopped")
        return self.snapshot()

    def _meta_loop(self) -> None:
        while not self._meta_stop.wait(2.0):
            runner = self._runner
            if runner is None:
                continue
            try:
                snap = runner.snapshot()
                self.store.update_snapshot(snap)
                self._publish({"type": "meta_tick", "ts": _utc_iso(), "payload": snap})
            except Exception:
                pass

    def _on_event(self, event_type: str, payload: dict[str, Any]) -> None:
        event = {"type": event_type, "ts": _utc_iso(), "payload": payload}
        self.store.append_event(event)
        if self._runner is not None:
            self.store.update_snapshot(self._runner.snapshot())
        self._publish(event)

    def _publish(self, event: dict) -> None:
        self.broker.publish(event)


class DealerDashboardHTTPServer(ThreadingHTTPServer):
    app: DealerDashboardApp


class DealerDashboardHandler(BaseHTTPRequestHandler):
    server_version = "CynolycusDealerDashboard/1.0"

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        if self.path.startswith(("/api/events", "/api/state")):
            return
        super().log_message(format, *args)

    def _app(self) -> DealerDashboardApp:
        server = self.server
        if not isinstance(server, DealerDashboardHTTPServer):
            raise RuntimeError("Handler attached to unexpected server type.")
        return server.app

    def _read_json_body(self) -> dict:
        length = int(self.headers.get("Content-Length", "0") or 0)
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        return json.loads(raw.decode("utf-8")) if raw else {}

    def _write_json(self, payload: dict, status: int = HTTPStatus.OK) -> None:
        body = json.dumps(_json_safe(payload)).encode("utf-8")
        self.send_response(int(status))
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _write_text(self, body: str, *, status: int = HTTPStatus.OK, content_type: str = "text/plain") -> None:
        blob = body.encode("utf-8")
        self.send_response(int(status))
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.send_header("Content-Length", str(len(blob)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(blob)

    def do_GET(self) -> None:  # noqa: N802
        from urllib.parse import urlparse

        parsed = urlparse(self.path)
        app = self._app()
        if parsed.path in {"/", "/index.html", "/dealer"}:
            index_path = Path(__file__).resolve().parent / "dealer_positioning_index.html"
            if not index_path.exists():
                self._write_text("Missing UI/dealer_positioning_index.html", status=HTTPStatus.NOT_FOUND)
                return
            from UI.ui_chrome import NAV_HTML
            html = index_path.read_text(encoding="utf-8").replace("<!--CYNO_NAV-->", NAV_HTML)
            self._write_text(html, content_type="text/html")
            return
        if parsed.path == "/static/cynolycus_theme.css":
            from UI.ui_chrome import serve_theme_css
            serve_theme_css(self)
            return
        if parsed.path.startswith("/static/themes/"):
            from UI.ui_chrome import serve_theme_asset
            serve_theme_asset(self, parsed.path)
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
                hello = {"type": "hello", "ts": _utc_iso(), "payload": app.snapshot()}
                self.wfile.write(f"data: {json.dumps(_json_safe(hello))}\n\n".encode("utf-8"))
                self.wfile.flush()
                while True:
                    try:
                        event = q.get(timeout=12.0)
                    except queue_mod.Empty:
                        self.wfile.write(b": keepalive\n\n")
                        self.wfile.flush()
                        continue
                    self.wfile.write(f"data: {json.dumps(_json_safe(event))}\n\n".encode("utf-8"))
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                app.broker.unsubscribe(q)
            return
        self._write_json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
        from urllib.parse import urlparse

        app = self._app()
        parsed = urlparse(self.path)
        try:
            payload = self._read_json_body()
            if parsed.path == "/api/start":
                self._write_json({"ok": True, "state": app.start(payload)})
                return
            if parsed.path == "/api/stop":
                self._write_json({"ok": True, "state": app.stop()})
                return
            self._write_json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)
        except Exception as exc:
            self._write_json({"error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
    parser = argparse.ArgumentParser(description="Run dealer positioning dashboard.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8768)
    args = parser.parse_args()
    app = DealerDashboardApp()
    server = DealerDashboardHTTPServer((args.host, args.port), DealerDashboardHandler)
    server.app = app
    logger.info("Dealer positioning dashboard: http://%s:%d", args.host, args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        app.stop()
        server.shutdown()


if __name__ == "__main__":
    main()
