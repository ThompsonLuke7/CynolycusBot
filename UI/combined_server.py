"""
Combined dashboard server.

Runs both the intraday SPY dashboard and the multi-ticker swing dashboard
from a single process, sharing one Alpaca WebSocket connection so neither
exceeds the IEX free-tier limit of one concurrent stream.

  Intraday SPY dashboard:  http://localhost:8765  (same URL as before)
  Swing dashboard:         http://localhost:8766  (same URL as before)

Usage:
  python -m UI.combined_server [--host 127.0.0.1] [--port-intraday 8765] [--port-swing 8766] [--env .env]

The stream starts immediately at process boot with the full symbol universe
(swing tickers + context tickers + intraday-specific tickers).  Both
dashboards become interactive once the stream is running; click "Run" in
either browser tab to start the respective trading session.

Standalone mode is preserved: you can still run each dashboard separately
with `python -m UI.live_dashboard` or `python -m UI.swing_dashboard`; they
fall back to creating their own individual AlpacaBarStreamer.
"""
from __future__ import annotations

import argparse
import logging
import queue as queue_mod
import signal
import sys
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

# Intraday dashboard needs SPY + these extra symbols beyond the swing universe
_INTRADAY_EXTRA_SYMBOLS = ["SPY", "VIXY", "QQQ", "IWM", "TLT", "UUP"]

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT_INTRADAY = 8765
DEFAULT_PORT_SWING = 8766
DEFAULT_AUDIT_ROOT = str(Path(__file__).resolve().parent / "swing_audit")


def _build_symbol_union(env_file: str) -> list[str]:
    """Union of all symbols needed by both dashboards."""
    from multi_ticker_swing.config.pipeline_config import CONTEXT_TICKERS
    from multi_ticker_swing.live.universe import load_universe

    swing_universe = set(load_universe().keys())
    combined = swing_universe | set(CONTEXT_TICKERS) | set(_INTRADAY_EXTRA_SYMBOLS)
    return sorted(combined)


def run_combined(
    *,
    host: str = DEFAULT_HOST,
    port_intraday: int = DEFAULT_PORT_INTRADAY,
    port_swing: int = DEFAULT_PORT_SWING,
    env_file: str = ".env",
    audit_root: Path | None = None,
) -> None:
    if audit_root is None:
        audit_root = Path(DEFAULT_AUDIT_ROOT)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    # ------------------------------------------------------------------
    # 1. Build shared symbol universe and start the single WebSocket stream
    # ------------------------------------------------------------------
    logger.info("Building symbol universe...")
    all_symbols = _build_symbol_union(env_file)
    logger.info("Symbol universe: %d symbols", len(all_symbols))

    # Two separate queues — one per dashboard.  SharedBarStream fans each bar
    # into both queues regardless of which runner is active.
    intraday_queue: queue_mod.Queue = queue_mod.Queue(maxsize=10_000)
    swing_queue: queue_mod.Queue = queue_mod.Queue(maxsize=10_000)

    from UI.shared_stream import get_shared_bar_stream
    stream = get_shared_bar_stream()
    stream.register(intraday_queue)
    stream.register(swing_queue)
    stream.start(all_symbols, env_file=env_file)
    logger.info("Shared bar stream started.")

    # ------------------------------------------------------------------
    # 2. Intraday SPY dashboard (port 8765)
    # ------------------------------------------------------------------
    from UI.live_dashboard import DashboardApp, DashboardHandler, DashboardHTTPServer

    intraday_app = DashboardApp(bar_queue=intraday_queue)
    intraday_server = DashboardHTTPServer((host, port_intraday), DashboardHandler)
    intraday_server.daemon_threads = True
    intraday_server.app = intraday_app

    intraday_thread = threading.Thread(
        target=intraday_server.serve_forever,
        daemon=True,
        name="intraday-http",
    )
    intraday_thread.start()
    logger.info("Intraday SPY dashboard:  http://%s:%d", host, port_intraday)

    # ------------------------------------------------------------------
    # 3. Multi-ticker swing dashboard (port 8766)
    # ------------------------------------------------------------------
    from UI.swing_dashboard import (
        SwingDashboardApp,
        SwingDashboardHandler,
        SwingDashboardHTTPServer,
    )

    swing_app = SwingDashboardApp(audit_root=audit_root, bar_queue=swing_queue)
    swing_server = SwingDashboardHTTPServer((host, port_swing), SwingDashboardHandler)
    swing_server.daemon_threads = True
    swing_server.app = swing_app

    swing_thread = threading.Thread(
        target=swing_server.serve_forever,
        daemon=True,
        name="swing-http",
    )
    swing_thread.start()
    logger.info("Swing dashboard:         http://%s:%d", host, port_swing)

    print()
    print("=" * 60)
    print(f"  Intraday SPY dashboard:  http://{host}:{port_intraday}")
    print(f"  Swing dashboard:         http://{host}:{port_swing}")
    print(f"  Shared stream:           {len(all_symbols)} symbols")
    print("  Press Ctrl+C to stop.")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 4. Block main thread; handle Ctrl-C / SIGTERM gracefully
    # ------------------------------------------------------------------
    stop_evt = threading.Event()

    def _shutdown(sig: int, frame: object) -> None:
        logger.info("Shutdown signal received — stopping...")
        stop_evt.set()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    stop_evt.wait()

    logger.info("Stopping sessions...")
    try:
        intraday_app.stop()
    except Exception as exc:
        logger.warning("intraday_app.stop(): %s", exc)
    try:
        swing_app.stop()
    except Exception as exc:
        logger.warning("swing_app.stop(): %s", exc)

    logger.info("Stopping HTTP servers...")
    intraday_server.shutdown()
    swing_server.shutdown()

    logger.info("Stopping shared bar stream...")
    stream.unregister(intraday_queue)
    stream.unregister(swing_queue)
    stream.stop()

    logger.info("Combined server stopped.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run intraday SPY + swing dashboards sharing one Alpaca WebSocket.",
    )
    parser.add_argument("--host", default=DEFAULT_HOST, help="Bind address (default 127.0.0.1).")
    parser.add_argument("--port-intraday", type=int, default=DEFAULT_PORT_INTRADAY,
                        help="Port for intraday SPY dashboard (default 8765).")
    parser.add_argument("--port-swing", type=int, default=DEFAULT_PORT_SWING,
                        help="Port for swing dashboard (default 8766).")
    parser.add_argument("--env", default=".env", help="Path to .env file.")
    parser.add_argument(
        "--audit-root",
        default=DEFAULT_AUDIT_ROOT,
        help="Directory for swing JSONL audit logs.",
    )
    args = parser.parse_args()

    run_combined(
        host=args.host,
        port_intraday=args.port_intraday,
        port_swing=args.port_swing,
        env_file=args.env,
        audit_root=Path(args.audit_root),
    )


if __name__ == "__main__":
    main()
