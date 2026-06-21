"""
Combined dashboard server.

Runs both the intraday SPY dashboard and the multi-ticker swing dashboard
from a single process, sharing one Alpaca WebSocket connection so neither
exceeds the IEX free-tier limit of one concurrent stream.

  Intraday SPY dashboard:  http://localhost:8765  (same URL as before)
  Swing dashboard:         http://localhost:8766  (same URL as before)
  Dealer positioning:      http://localhost:8768

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
DEFAULT_PORT_DEALER = 8768
DEFAULT_AUDIT_ROOT = str(Path(__file__).resolve().parent / "swing_audit")


def _env_profile_exists(env_file: str, profile: str) -> bool:
    try:
        from core.API.Alpaca_API.core.config import _read_env_file, _split_env_file_profile

        env_path, _ = _split_env_file_profile(env_file)
        values = _read_env_file(env_path or ".env")
    except Exception:
        return False
    suffix = str(profile).strip().upper()
    return bool(
        values.get(f"APCA_API_KEY_ID_{suffix}")
        or values.get(f"ALPACA_API_KEY_{suffix}")
        or values.get(f"{suffix}_APCA_API_KEY_ID")
        or values.get(f"{suffix}_ALPACA_API_KEY")
    )


def _default_profile_env(env_file: str, profile: str) -> str:
    if "#" in str(env_file):
        return str(env_file)
    return f"{env_file}#{profile}"


def _run_nightly_jobs() -> None:
    """Invoke the nightly market-data + discovery pipeline (one source of truth).

    Reuses ``scripts/nightly_market_data.sh`` so the in-process scheduler and a
    system cron run exactly the same steps (CBOE snapshot, FINRA short volume,
    ticker discovery + promotion gate). The script logs to its own cron log.
    """
    import shutil
    import subprocess

    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "nightly_market_data.sh"
    bash = shutil.which("bash")
    if bash is None:
        logger.error(
            "Nightly jobs: 'bash' not found on PATH — cannot run %s. "
            "Use system cron instead, or run the script manually.",
            script,
        )
        return
    logger.info("Nightly jobs: launching %s", script)
    result = subprocess.run([bash, str(script)], cwd=str(repo_root))
    logger.info("Nightly jobs: finished (exit=%s)", result.returncode)


def _build_symbol_union(env_file: str) -> list[str]:
    """Union of all symbols needed by both dashboards."""
    from strategies.multi_ticker_swing.config.pipeline_config import CONTEXT_TICKERS
    from strategies.multi_ticker_swing.live.universe import load_universe

    swing_universe = set(load_universe().keys())
    combined = swing_universe | set(CONTEXT_TICKERS) | set(_INTRADAY_EXTRA_SYMBOLS)
    return sorted(combined)


def run_combined(
    *,
    host: str = DEFAULT_HOST,
    port_intraday: int = DEFAULT_PORT_INTRADAY,
    port_swing: int = DEFAULT_PORT_SWING,
    port_swing_live: int | None = None,
    port_dealer: int = DEFAULT_PORT_DEALER,
    env_file: str = ".env",
    paper_env_file: str | None = None,
    live_env_file: str | None = None,
    audit_root: Path | None = None,
    nightly_time: str | None = "16:30",
    catalyst_poll: bool = True,
    catalyst_poll_interval: int = 300,
) -> None:
    if audit_root is None:
        audit_root = Path(DEFAULT_AUDIT_ROOT)
    if paper_env_file is None and _env_profile_exists(env_file, "paper"):
        paper_env_file = _default_profile_env(env_file, "paper")
    if "#" not in str(env_file) and _env_profile_exists(env_file, "paper"):
        env_file = _default_profile_env(env_file, "paper")

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

    # Separate queues per dashboard. Subscriber filters keep SPY-only consumers
    # from receiving the full swing universe.
    intraday_queue: queue_mod.Queue = queue_mod.Queue(maxsize=10_000)
    swing_paper_queue: queue_mod.Queue = queue_mod.Queue(maxsize=50_000)
    swing_live_queue: queue_mod.Queue | None = queue_mod.Queue(maxsize=50_000) if live_env_file else None
    dealer_queue: queue_mod.Queue = queue_mod.Queue(maxsize=10_000)

    from UI.shared_stream import get_shared_bar_stream
    stream = get_shared_bar_stream()
    stream.register(
        intraday_queue,
        name="intraday",
        symbols=("SPY", "VIXY", "QQQ", "IWM", "TLT", "UUP"),
    )
    stream.register(swing_paper_queue, name="swing-paper")
    if swing_live_queue is not None:
        stream.register(swing_live_queue, name="swing-live")
    stream.register(dealer_queue, name="dealer-positioning", symbols=("SPY",))
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

    paper_env_file = paper_env_file or env_file
    swing_app = SwingDashboardApp(
        audit_root=audit_root / "paper" if live_env_file else audit_root,
        bar_queue=swing_paper_queue,
        default_env_file=paper_env_file,
        default_dry_run=False,
        default_real_account_policy=False,
    )
    swing_server = SwingDashboardHTTPServer((host, port_swing), SwingDashboardHandler)
    swing_server.daemon_threads = True
    swing_server.app = swing_app

    swing_thread = threading.Thread(
        target=swing_server.serve_forever,
        daemon=True,
        name="swing-http",
    )
    swing_thread.start()
    logger.info("Swing paper dashboard:   http://%s:%d", host, port_swing)

    swing_live_app = None
    swing_live_server = None
    if live_env_file:
        live_port = int(port_swing_live or (port_swing + 1))
        swing_live_app = SwingDashboardApp(
            audit_root=audit_root / "live",
            bar_queue=swing_live_queue,
            default_env_file=live_env_file,
            default_dry_run=False,
            default_real_account_policy=True,
            default_real_account_policy_state_path=(
                "Data/inference/multi_ticker_swing/real_account_book_live.json"
            ),
        )
        swing_live_server = SwingDashboardHTTPServer((host, live_port), SwingDashboardHandler)
        swing_live_server.daemon_threads = True
        swing_live_server.app = swing_live_app
        swing_live_thread = threading.Thread(
            target=swing_live_server.serve_forever,
            daemon=True,
            name="swing-live-http",
        )
        swing_live_thread.start()
        logger.info("Swing real-money dashboard: http://%s:%d", host, live_port)

    # ------------------------------------------------------------------
    # 4. Dealer positioning dashboard (port 8768)
    # ------------------------------------------------------------------
    from UI.dealer_positioning_dashboard import (
        DealerDashboardApp,
        DealerDashboardHandler,
        DealerDashboardHTTPServer,
    )

    dealer_app = DealerDashboardApp(bar_queue=dealer_queue)
    dealer_server = DealerDashboardHTTPServer((host, port_dealer), DealerDashboardHandler)
    dealer_server.daemon_threads = True
    dealer_server.app = dealer_app
    dealer_thread = threading.Thread(
        target=dealer_server.serve_forever,
        daemon=True,
        name="dealer-positioning-http",
    )
    dealer_thread.start()
    logger.info("Dealer positioning:      http://%s:%d", host, port_dealer)

    print()
    print("=" * 60)
    print(f"  Intraday SPY dashboard:  http://{host}:{port_intraday}")
    print(f"  Swing paper dashboard:   http://{host}:{port_swing}")
    if live_env_file:
        print(f"  Swing real-money:        http://{host}:{int(port_swing_live or (port_swing + 1))}")
    print(f"  Dealer positioning:      http://{host}:{port_dealer}")
    print(f"  Shared stream:           {len(all_symbols)} symbols")
    print("  Press Ctrl+C to stop.")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 5. Block main thread; handle Ctrl-C / SIGTERM gracefully
    # ------------------------------------------------------------------
    stop_evt = threading.Event()

    def _shutdown(sig: int, frame: object) -> None:
        logger.info("Shutdown signal received — stopping...")
        stop_evt.set()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # ------------------------------------------------------------------
    # 5a. Nightly jobs scheduler (CBOE/FINRA/ticker discovery after close)
    # ------------------------------------------------------------------
    nightly_scheduler = None
    if nightly_time:
        from UI.nightly_scheduler import NightlyScheduler, parse_hhmm

        try:
            hour, minute = parse_hhmm(nightly_time)
        except ValueError as exc:
            logger.error("Invalid --nightly-time %r (%s) — nightly jobs disabled", nightly_time, exc)
        else:
            nightly_scheduler = NightlyScheduler(
                _run_nightly_jobs,
                hour=hour,
                minute=minute,
                stop_event=stop_evt,
            )
            nightly_scheduler.start()
            logger.info("Nightly jobs scheduled daily at %s America/New_York (weekdays)", nightly_time)
            print(f"  Nightly jobs:            daily {nightly_time} ET (CBOE/FINRA/discovery)")

    # ------------------------------------------------------------------
    # 5b. Intraday catalyst poller (keeps the live news ledger fresh during RTH
    #     so the swing scanner can react to breaking news in real time)
    # ------------------------------------------------------------------
    poller_supervisor = None
    if catalyst_poll:
        from UI.intraday_poller import IntradayPollerSupervisor

        poller_supervisor = IntradayPollerSupervisor(
            interval=catalyst_poll_interval,
            stop_event=stop_evt,
        )
        poller_supervisor.start()
        logger.info("Intraday catalyst poller scheduled 09:30–16:00 America/New_York (weekdays)")
        print(f"  Catalyst poller:         RTH every {catalyst_poll_interval}s ET (live news ledger)")

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
    try:
        dealer_app.stop()
    except Exception as exc:
        logger.warning("dealer_app.stop(): %s", exc)
    if swing_live_app is not None:
        try:
            swing_live_app.stop()
        except Exception as exc:
            logger.warning("swing_live_app.stop(): %s", exc)

    logger.info("Stopping HTTP servers...")
    intraday_server.shutdown()
    swing_server.shutdown()
    dealer_server.shutdown()
    if swing_live_server is not None:
        swing_live_server.shutdown()

    logger.info("Stopping shared bar stream...")
    stream.unregister(intraday_queue)
    stream.unregister(swing_paper_queue)
    if swing_live_queue is not None:
        stream.unregister(swing_live_queue)
    stream.unregister(dealer_queue)
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
                        help="Port for paper swing dashboard (default 8766).")
    parser.add_argument("--port-swing-live", type=int, default=None,
                        help="Port for live swing dashboard when --live-env is set (default port-swing+1).")
    parser.add_argument("--port-dealer", type=int, default=DEFAULT_PORT_DEALER,
                        help="Port for dealer positioning dashboard (default 8768).")
    parser.add_argument("--env", default=".env",
                        help="Path to env file for the shared market-data stream.")
    parser.add_argument("--paper-env", default=None,
                        help="Path to Alpaca paper env file for paper swing orders (default --env).")
    parser.add_argument("--live-env", default=None,
                        help="Path to Alpaca real-money env file. When set explicitly, starts a protected real-money swing dashboard.")
    parser.add_argument(
        "--audit-root",
        default=DEFAULT_AUDIT_ROOT,
        help="Directory for swing JSONL audit logs.",
    )
    parser.add_argument(
        "--nightly-time",
        default="16:30",
        help="Local (America/New_York) HH:MM to run nightly CBOE/FINRA/discovery jobs (default 16:30).",
    )
    parser.add_argument(
        "--no-nightly-jobs",
        action="store_true",
        help="Disable the in-process nightly scheduler (use system cron instead).",
    )
    parser.add_argument(
        "--no-catalyst-poll",
        action="store_true",
        help="Disable the in-process intraday catalyst news poller.",
    )
    parser.add_argument(
        "--catalyst-poll-interval",
        type=int,
        default=300,
        help="Seconds between intraday catalyst polls during market hours (default 300).",
    )
    args = parser.parse_args()

    run_combined(
        host=args.host,
        port_intraday=args.port_intraday,
        port_swing=args.port_swing,
        port_swing_live=args.port_swing_live,
        port_dealer=args.port_dealer,
        env_file=args.env,
        paper_env_file=args.paper_env,
        live_env_file=args.live_env,
        audit_root=Path(args.audit_root),
        nightly_time=None if args.no_nightly_jobs else args.nightly_time,
        catalyst_poll=not args.no_catalyst_poll,
        catalyst_poll_interval=args.catalyst_poll_interval,
    )


if __name__ == "__main__":
    main()
