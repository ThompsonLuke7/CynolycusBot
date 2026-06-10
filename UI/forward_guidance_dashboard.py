"""Read-only dashboard for forward-guidance ranked opportunities."""

from __future__ import annotations

import argparse
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pandas as pd

from signals.events.forward_guidance.config import DASHBOARD_STATE_PATH, RANKED_OUTPUT_CSV, RANKED_OUTPUT_PARQUET
from signals.events.forward_guidance.utils.io import json_safe, read_json


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8776


class ForwardGuidanceDashboardApp:
    def __init__(self, *, ranked_path: Path | str | None = None, state_path: Path | str = DASHBOARD_STATE_PATH) -> None:
        self.ranked_path = Path(ranked_path) if ranked_path else None
        self.state_path = Path(state_path)

    def _load_ranked(self) -> pd.DataFrame:
        candidates = []
        if self.ranked_path:
            candidates.append(self.ranked_path)
        candidates.extend([RANKED_OUTPUT_PARQUET, RANKED_OUTPUT_CSV])
        for path in candidates:
            if not path.exists():
                continue
            if path.suffix.lower() == ".csv":
                return pd.read_csv(path)
            return pd.read_parquet(path)
        return pd.DataFrame()

    def snapshot(self) -> dict[str, Any]:
        state = read_json(self.state_path, default={}) or {}
        ranked = self._load_ranked()
        if ranked.empty:
            return {
                "status": "no_output",
                "updated_at": state.get("updated_at"),
                "count": 0,
                "opportunities": [],
                "source": str(self.ranked_path or RANKED_OUTPUT_PARQUET),
            }
        view_cols = [
            "ticker",
            "earnings_date",
            "report_time",
            "future_60d_outperformance_probability",
            "expected_return",
            "confidence",
            "holding_horizon",
            "guidance_strength_score",
            "post_er_gap_pct",
            "post_er_move_pct",
            "technical_stabilization_flag",
            "top_reason_features",
        ]
        cols = [c for c in view_cols if c in ranked.columns]
        return {
            "status": "ready",
            "updated_at": state.get("updated_at"),
            "count": int(len(ranked)),
            "source": str(self.ranked_path or RANKED_OUTPUT_PARQUET),
            "opportunities": ranked[cols].head(100).to_dict(orient="records"),
        }


class ForwardGuidanceHTTPServer(ThreadingHTTPServer):
    app: ForwardGuidanceDashboardApp


class ForwardGuidanceDashboardHandler(BaseHTTPRequestHandler):
    server_version = "CynolycusForwardGuidanceDashboard/1.0"

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        return

    def _write_json(self, payload: dict[str, Any], status: int = HTTPStatus.OK) -> None:
        body = json.dumps(json_safe(payload)).encode("utf-8")
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

    def _serve_index(self) -> None:
        index_path = Path(__file__).resolve().parent / "forward_guidance_index.html"
        if not index_path.exists():
            self._write_text("Missing UI/forward_guidance_index.html", status=HTTPStatus.NOT_FOUND)
            return
        self._write_text(index_path.read_text(encoding="utf-8"), content_type="text/html")

    def _app(self) -> ForwardGuidanceDashboardApp:
        server = self.server
        if not isinstance(server, ForwardGuidanceHTTPServer):
            raise RuntimeError("Unexpected dashboard server type.")
        return server.app

    def do_GET(self) -> None:  # noqa: N802
        from urllib.parse import urlparse

        parsed = urlparse(self.path)
        if parsed.path in {"/", "/index.html", "/forward-guidance"}:
            self._serve_index()
            return
        if parsed.path == "/api/state":
            self._write_json(self._app().snapshot())
            return
        self._write_json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)


def run_server(*, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT, ranked_path: Path | str | None = None) -> None:
    app = ForwardGuidanceDashboardApp(ranked_path=ranked_path)
    server = ForwardGuidanceHTTPServer((host, int(port)), ForwardGuidanceDashboardHandler)
    server.daemon_threads = True
    server.app = app
    print(f"[forward-guidance-ui] Dashboard: http://{host}:{port}")
    print("[forward-guidance-ui] Read-only ranked opportunities. Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("[forward-guidance-ui] Shutting down...")
    finally:
        server.server_close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Forward-guidance ranked opportunity dashboard.")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--ranked-path", default=None)
    args = parser.parse_args()
    run_server(host=args.host, port=args.port, ranked_path=args.ranked_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
