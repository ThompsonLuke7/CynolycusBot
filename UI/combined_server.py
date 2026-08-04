"""
Combined dashboard server.

Runs both the intraday SPY dashboard and the multi-ticker swing dashboard
from a single process, sharing one Alpaca WebSocket connection so neither
exceeds the IEX free-tier limit of one concurrent stream.

  Intraday SPY dashboard:  http://localhost:8765  (same URL as before)
  Swing dashboard:         http://localhost:8766  (same URL as before)
  Dealer positioning:      http://localhost:8768
  Dealer Ranker:           http://localhost:8773
  Library (news search):   http://localhost:8775

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
import ctypes
import ctypes.util
import gc
import logging
import os
import queue as queue_mod
import signal
import sys
import threading
import time
from pathlib import Path

logger = logging.getLogger(__name__)


def _process_rss_mb() -> float | None:
    """Resident set size of this process in MB, read from /proc (no psutil dep)."""
    try:
        with open("/proc/self/status", "r") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0  # kB -> MB
    except Exception:
        return None
    return None


def _start_memory_keeper(stop_event: threading.Event, *, interval: int = 300) -> None:
    """Background thread that fights long-run RSS growth in this pandas-heavy
    process and logs the footprint so OOM regressions are visible.

    The combined server churns per-scan DataFrames across ~900 tickers all day.
    glibc's allocator keeps freed arenas mapped (RSS never drops), which is what
    walked us into the 16GB WSL cap and the OOM-kill. Each tick we run a full GC
    and ask glibc to return free arenas to the OS via ``malloc_trim(0)``, then log
    VmRSS. This does not touch any feature/warmup data, so it cannot introduce NaNs.
    """
    try:
        libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6", use_errno=True)
        has_trim = hasattr(libc, "malloc_trim")
    except Exception:
        libc, has_trim = None, False

    def _loop() -> None:
        while not stop_event.wait(interval):
            try:
                gc.collect()
                if has_trim:
                    libc.malloc_trim(0)
                rss = _process_rss_mb()
                if rss is not None:
                    logger.info("Memory keeper: RSS=%.0f MB (gc+malloc_trim done)", rss)
            except Exception as exc:  # never let the keeper kill the server
                logger.warning("Memory keeper tick failed: %s", exc)

    threading.Thread(target=_loop, daemon=True, name="memory-keeper").start()
    rss0 = _process_rss_mb()
    logger.info(
        "Memory keeper started (every %ds, malloc_trim=%s, startup RSS=%s MB)",
        interval, has_trim, f"{rss0:.0f}" if rss0 is not None else "?",
    )

# Intraday dashboard needs SPY + these extra symbols beyond the swing universe
_INTRADAY_EXTRA_SYMBOLS = ["SPY", "VIXY", "QQQ", "IWM", "TLT", "UUP"]

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT_HUB = 8764
DEFAULT_PORT_INTRADAY = 8765
DEFAULT_PORT_SWING = 8766
DEFAULT_PORT_DEALER = 8768
DEFAULT_PORT_META = 8769
DEFAULT_PORT_MOMENTUM = 8770
DEFAULT_PORT_HTF = 8771
DEFAULT_PORT_AMETHYST = 8772
DEFAULT_PORT_DEALER_RANKER = 8773
DEFAULT_PORT_INTRADAY_STRUCTURE = 8774
DEFAULT_PORT_LIBRARY = 8775

# State book used by the swing real-account (LIVE) policy.
_SWING_LIVE_BOOK = "Data/inference/multi_ticker_swing/real_account_book_live.json"
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

    Reuses ``scripts/nightly_market_data.sh`` as the single post-close owner for
    bounded collection/enrichment (options, dealer, FINRA, discovery, priority
    news, themes, and earnings). The script logs to its historical cron log.
    """
    import shutil
    import subprocess

    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "nightly_market_data.sh"
    bash = shutil.which("bash")
    if bash is None:
        logger.error(
            "Nightly jobs: 'bash' not found on PATH — cannot run %s. "
            "Run the script manually after the close instead.",
            script,
        )
        return
    logger.info("Nightly jobs: launching %s", script)
    # Isolate the long-running child from the supervisor's terminal process
    # group.  A Ctrl-C/server shutdown should stop trading and dashboards, but
    # must not interrupt an already-running data pipeline halfway through.
    result = subprocess.run(
        [bash, str(script)],
        cwd=str(repo_root),
        start_new_session=True,
    )
    logger.info("Nightly jobs: finished (exit=%s)", result.returncode)


def _run_data_readiness(*, catch_up: bool = False) -> None:
    """Refresh the shared bars + HTF features + Meta matrix so HTF Swing,
    Momentum, and Meta Ranker all wake up caught up to the same state.

    Reuses ``scripts/nightly_data_readiness.sh`` (one source of truth). The
    scheduled run is in the evening, after the nightly job, so it never collides
    with the live session or the 15:50 Meta MOC loop.

    ``catch_up=True`` is the startup repair path taken only when the stamp is
    already stale. It deliberately overrides the live-window block: with a stale
    stamp the readiness gate rejects every entry order, so the modules trade
    nothing at all, and a heavy job during the session is strictly better than
    another dark day. The memory guard still applies.
    """
    import os
    import shutil
    import subprocess
    from core.live_job_guard import heavy_job_guard

    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "nightly_data_readiness.sh"
    bash = shutil.which("bash")
    if bash is None:
        logger.error("Data readiness: 'bash' not found on PATH — cannot run %s.", script)
        return
    with heavy_job_guard(
        "combined-server-data-readiness",
        block_live_window=not catch_up,
        min_available_mb=6144,
        min_swap_free_mb=6144,
    ) as guard:
        if not guard.ok:
            logger.warning("Data readiness: skipped (%s)", guard.reason)
            return
        env = dict(os.environ)
        if catch_up:
            # The script has the same live-window guard; lift it for the repair.
            env["ALLOW_LIVE_READINESS"] = "1"
        logger.info("Data readiness: launching %s%s", script, " (stale-stamp catch-up)" if catch_up else "")
        result = subprocess.run([bash, str(script)], cwd=str(repo_root), env=env)
        logger.info("Data readiness: finished (exit=%s)", result.returncode)


def _run_startup_queue(*, env_file: str, account_label: str) -> None:
    """Drain actions queued for "next time the server is up".

    Runs after the readiness check so a queued readiness rebuild is a no-op when
    the startup check already handled it. Order entries stay pending while the
    market is closed and are retried at the next boot.
    """
    try:
        from core import startup_queue
        from core.API.Alpaca_API.options.options_api import AlpacaOptionsClient

        queued = startup_queue.pending()
        if not queued:
            return
        logger.info("Startup queue: %d pending entr%s", len(queued), "y" if len(queued) == 1 else "ies")
        summary = startup_queue.run_pending(
            # This process talks to exactly one account — the one it was started
            # with — so entries never pick their own credentials.
            client_factory=lambda _account: AlpacaOptionsClient(env_file=env_file),
            readiness_runner=lambda: _run_data_readiness(catch_up=True),
            # Live orders require an explicit opt-in inside run_pending; this
            # server process only ever drains the account it was started with.
            allow_live=(str(account_label).strip().lower() == "live"),
        )
        logger.info("Startup queue: %s", summary)
    except Exception as exc:  # noqa: BLE001
        logger.error("Startup queue: drain failed (%s)", exc)


def _data_readiness_startup_check() -> None:
    """Run readiness at boot only when the stamp will not authorize entries.

    Without this, a server that starts after its scheduled readiness slot has
    passed (a crash-and-restart, exactly what happened on 2026-07-24) runs the
    whole session with a stale stamp and silently places no entry orders at all.
    """
    try:
        from core.live_readiness import readiness_status

        ok, reason, _payload = readiness_status()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Data readiness: startup check failed (%s) — running catch-up", exc)
        ok, reason = False, str(exc)

    if ok:
        logger.info("Data readiness: stamp is current at startup — no catch-up needed")
        return
    logger.warning("Data readiness: stamp will NOT authorize entries (%s) — running catch-up now", reason)
    _run_data_readiness(catch_up=True)


def _run_broker_equity_snapshot(*, env_file: str, account_label: str) -> None:
    """Capture read-only broker marks used by the daily briefing attribution."""
    from core.broker_equity_snapshot import capture_from_env

    try:
        result = capture_from_env(env_file=env_file, account_label=account_label)
    except Exception as exc:  # noqa: BLE001
        logger.error("Broker equity snapshot (%s) failed: %s", account_label, exc, exc_info=True)
        return
    logger.info(
        "Broker equity snapshot (%s): positions=%d -> %s",
        account_label, result["position_count"], result["path"],
    )


def _run_shadow_tracker() -> None:
    """Two-sleeve (id4 tail-rider / g284 harvester) exit-policy shadow tracker for
    Momentum/HTF/Meta — paper-only observation, no submit_order path (see
    scripts/shadow/two_sleeve_shadow_tracker.py docstring). Subprocess-isolated
    like the other loops so a failure here can't take down the combined server;
    its own file-reads are already memory-bounded against the multi-GB feature
    matrices (see that script's 2026-07-21 WSL-crash postmortem in its header).
    """
    import os
    import subprocess
    import sys

    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts/shadow/two_sleeve_shadow_tracker.py"
    argv = [sys.executable, str(script)]
    logger.info("Shadow tracker: launching %s", " ".join(argv))
    env = {**os.environ, "PYTHONPATH": str(repo_root)}
    result = subprocess.run(argv, cwd=str(repo_root), env=env)
    logger.info("Shadow tracker: finished (exit=%s)", result.returncode)


def _build_symbol_union(env_file: str) -> list[str]:
    """Union of all symbols needed by both dashboards."""
    from strategies.multi_ticker_swing.config.pipeline_config import CONTEXT_TICKERS
    from strategies.multi_ticker_swing.live.universe import load_universe

    swing_universe = set(load_universe().keys())
    combined = swing_universe | set(CONTEXT_TICKERS) | set(_INTRADAY_EXTRA_SYMBOLS)
    return sorted(combined)


def _run_meta_ranker_loop(*, mode: str, submit: bool, live: bool = False, flush: bool = False) -> None:
    """Fire one Meta Ranker 4H pass — RUNNER ONLY (read matrix -> score -> trade).

    The shared bars + Meta matrix are kept fresh out-of-band by nightly/pre-open
    readiness plus the guarded afternoon SharedDataRefresher, so this pass skips
    bars/feeds/matrix and just runs the runner (~seconds). Runs as a
    subprocess so a failure can't take down the combined server.

    flush=True is the pre-open pass: it re-ranks and submits the entries that were
    queued after yesterday's close (the pm-bar signal is the latest bar, so
    --allow-stale is expected), then exits — it manages no positions.
    """
    import os
    import subprocess
    import sys

    repo_root = Path(__file__).resolve().parents[1]
    if flush:
        script = repo_root / "signals/meta_context/meta_ranker/live_runner.py"
        argv = [sys.executable, str(script), "--mode", mode, "--flush-pending-open", "--allow-stale"]
    else:
        script = repo_root / "signals/meta_context/meta_ranker/run_4h_loop.py"
        argv = [sys.executable, str(script), "--mode", mode,
                "--skip-bars", "--skip-feeds", "--skip-matrix"]
    if submit:
        argv.append("--submit")
    if live:
        # The governed Meta path is paper-only. Rather than forwarding a flag
        # the runner now rejects, refuse here so the pass never launches.
        logger.error("Meta Ranker live mode is not supported; refusing to launch")
        return
    logger.info("Meta Ranker %s: launching %s", "pre-open flush" if flush else "loop", " ".join(argv))
    env = {**os.environ, "PYTHONPATH": str(repo_root)}
    subprocess.run(argv, cwd=str(repo_root), env=env)


def _run_htf_loop(*, mode: str, submit: bool, live: bool, flush: bool = False) -> None:
    """Fire one standalone HTF Swing pass (reads htf_score off the shared matrix).

    Runs as a subprocess so a failure can't take down the combined server. The
    shared matrix is kept fresh by the data-readiness job + the Meta loop, so this
    just reads it; schedule it a few minutes AFTER the Meta times. flush=True is the
    pre-open pass that submits yesterday's after-close-queued entries (re-ranked).
    """
    import os
    import subprocess
    import sys

    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "strategies/multi_ticker_swing_htf/live/runner.py"
    argv = [sys.executable, str(script), "--mode", mode]
    if flush:
        argv += ["--flush-pending-open", "--allow-stale"]
    if submit:
        argv.append("--submit")
    if live:
        argv.append("--live")
    logger.info("HTF Swing %s: launching %s", "pre-open flush" if flush else "loop", " ".join(argv))
    env = {**os.environ, "PYTHONPATH": str(repo_root)}
    subprocess.run(argv, cwd=str(repo_root), env=env)


def _run_momentum_loop(*, submit: bool, live: bool, flush: bool = False) -> None:
    """Fire one standalone Momentum Expansion pass (ExpansionRanker -> own policy).

    Subprocess-isolated. Reads 4H/1H bars off disk (kept fresh by data-readiness +
    the Meta loop's bar catchup), so schedule it a few minutes AFTER the Meta times.
    flush=True is the pre-open pass that submits yesterday's after-close-queued
    entries (re-ranked); the pm bar is the latest so we relax the signal-staleness
    guard (read at import via MOMENTUM_MAX_SIGNAL_STALENESS_DAYS) for that pass only.
    """
    import os
    import subprocess
    import sys

    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "strategies/momentum_expansion/live/runner.py"
    argv = [sys.executable, str(script)]
    if flush:
        argv.append("--flush-pending-open")
    if submit:
        argv.append("--submit")
    if live:
        argv.append("--live")
    logger.info("Momentum %s: launching %s", "pre-open flush" if flush else "loop", " ".join(argv))
    env = {**os.environ, "PYTHONPATH": str(repo_root)}
    if flush:
        env["MOMENTUM_MAX_SIGNAL_STALENESS_DAYS"] = "5"  # pm bar is the latest; not "stale"
    subprocess.run(argv, cwd=str(repo_root), env=env)


def _run_dealer_ranker_loop(*, submit: bool, live: bool, top_k: int, workers: int, target_notional: float) -> None:
    """Fire one near-close dealer-rank options pass.

    It refreshes option-chain snapshots, rebuilds dealer rankings, buys nearest
    ATM non-0DTE contracts for the top ranks, and then lets the shared execution
    policy manage exits. Subprocess-isolated so a Schwab/Alpaca hiccup cannot
    take down the combined server.
    """
    import os
    import subprocess
    import sys

    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "strategies" / "dealer_positioning" / "live_ranked_options.py"
    argv = [
        sys.executable,
        str(script),
        "--refresh-chain",
        "--top-k",
        str(int(top_k)),
        "--workers",
        str(int(workers)),
        "--target-notional",
        str(float(target_notional)),
    ]
    if submit:
        argv.append("--submit")
    if live:
        argv.append("--live")
    logger.info("Dealer Ranker loop: launching %s", " ".join(argv))
    env = {**os.environ, "PYTHONPATH": str(repo_root)}
    subprocess.run(argv, cwd=str(repo_root), env=env)


def _assert_ports_free(host: str, ports: dict[str, int]) -> None:
    """Fail fast if any dashboard port is already bound.

    The HTTP servers are bound one-by-one *after* the 900+ symbol universe is
    built and the Alpaca WebSocket is opened, so without this check a second
    launch wastes that startup work and then dies with an opaque server_bind()
    traceback on the first conflicting port. Worse, a prior combined_server that
    is *stopped/frozen* (SIGSTOP -> state T) still holds its sockets in LISTEN,
    so it reads as "in use" and a fresh launch crash-loops forever against it.
    Probe every port up front and refuse with a clear, actionable message.
    """
    import socket

    busy: list[str] = []
    for name, port in ports.items():
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            # Mirror ThreadingHTTPServer.allow_reuse_address so the probe sees
            # exactly what the real bind will (SO_REUSEADDR clears TIME_WAIT but
            # NOT an active LISTEN, so a live/frozen owner still trips this).
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind((host, port))
            except OSError:
                busy.append(f"{name}={port}")
    if busy:
        raise SystemExit(
            "[combined_server] refusing to start: dashboard port(s) already in "
            f"use: {', '.join(busy)}. Another combined_server is already running "
            "(possibly stopped/frozen but still holding the sockets). Find the "
            r"owner with:  ss -ltnp | grep -E ':(876[4-9]|877[0-4])'"
        )


def run_combined(
    *,
    host: str = DEFAULT_HOST,
    port_hub: int = DEFAULT_PORT_HUB,
    port_intraday: int = DEFAULT_PORT_INTRADAY,
    port_swing: int = DEFAULT_PORT_SWING,
    port_dealer: int = DEFAULT_PORT_DEALER,
    env_file: str = ".env",
    paper_env_file: str | None = None,
    live_env_file: str | None = None,
    audit_root: Path | None = None,
    nightly_time: str | None = "16:45",
    data_readiness_time: str | None = None,
    data_readiness_on_start: bool = False,
    data_readiness_startup_check: bool = True,
    startup_queue_enabled: bool = True,
    startup_queue_account: str = "paper",
    # Market-hours re-drain slots (ET). The boot drain alone cannot fire a
    # `close_position` entry when the server is started outside 9:30-16:00.
    startup_queue_market_times: str = "09:35,13:00",
    data_refresher: bool = True,
    data_refresher_interval: int = 300,
    data_refresher_feeds: bool = False,
    catalyst_poll: bool = True,
    catalyst_poll_interval: int = 300,
    port_meta: int = DEFAULT_PORT_META,
    port_momentum: int = DEFAULT_PORT_MOMENTUM,
    port_htf: int = DEFAULT_PORT_HTF,
    port_amethyst: int = DEFAULT_PORT_AMETHYST,
    port_dealer_ranker: int = DEFAULT_PORT_DEALER_RANKER,
    port_intraday_structure: int = DEFAULT_PORT_INTRADAY_STRUCTURE,
    port_library: int = DEFAULT_PORT_LIBRARY,
    library_enabled: bool = True,
    intraday_structure_enabled: bool = True,
    intraday_structure_config: str = "strategies/intraday_structure/config/intraday_structure_v1.json",
    meta_ranker_times: str = "14:20,16:20",
    meta_ranker_mode: str = "options",
    meta_ranker_live: bool = False,
    htf_times: str = "14:25,16:25",
    htf_mode: str = "options",
    htf_live: bool = False,
    momentum_times: str = "14:25,16:25",
    momentum_live: bool = False,
    dealer_ranker_time: str = "15:45",
    dealer_ranker_live: bool = False,
    dealer_ranker_workers: int = 8,
    dealer_ranker_top_k: int = 10,
    dealer_ranker_target_notional: float = 5000.0,
    start_all: bool = False,
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

    # Pre-flight: refuse to start (before the expensive universe build + Alpaca
    # WebSocket connect) if any dashboard port is already held by another — or a
    # stopped/frozen — combined_server instance.
    dashboard_ports = {
        "hub": port_hub,
        "intraday": port_intraday,
        "swing": port_swing,
        "dealer": port_dealer,
        "meta": port_meta,
        "momentum": port_momentum,
        "htf": port_htf,
        "amethyst": port_amethyst,
        "dealer_ranker": port_dealer_ranker,
    }
    if intraday_structure_enabled:
        dashboard_ports["intraday_structure"] = port_intraday_structure
    if library_enabled:
        dashboard_ports["library"] = port_library
    _assert_ports_free(host, dashboard_ports)

    # ------------------------------------------------------------------
    # 1. Build shared symbol universe and start the single WebSocket stream
    # ------------------------------------------------------------------
    logger.info("Building symbol universe...")
    all_symbols = _build_symbol_union(env_file)
    structure_config = None
    if intraday_structure_enabled:
        from strategies.intraday_structure.config import load_config as load_intraday_structure_config
        from strategies.intraday_structure.candidate_sources import LiquidityCandidateFeed

        structure_config = load_intraday_structure_config(intraday_structure_config)
        liquidity_symbols = LiquidityCandidateFeed(
            structure_config.liquidity_universe.universe_path,
            top_n=structure_config.liquidity_universe.top_n,
        ).symbols() if structure_config.liquidity_universe.enabled else []
        all_symbols = sorted(
            set(all_symbols)
            | set(structure_config.manual_watchlist)
            | set(structure_config.context_symbols)
            | set(liquidity_symbols)
        )
    logger.info("Symbol universe: %d symbols", len(all_symbols))

    # Separate queues per dashboard. Subscriber filters keep SPY-only consumers
    # from receiving the full swing universe.
    intraday_queue: queue_mod.Queue = queue_mod.Queue(maxsize=10_000)
    swing_queue: queue_mod.Queue = queue_mod.Queue(maxsize=50_000)
    dealer_queue: queue_mod.Queue = queue_mod.Queue(maxsize=10_000)
    structure_queue: queue_mod.Queue | None = queue_mod.Queue(maxsize=50_000) if intraday_structure_enabled else None

    from UI.shared_stream import get_shared_bar_stream
    stream = get_shared_bar_stream()
    stream.register(
        intraday_queue,
        name="intraday",
        symbols=("SPY", "VIXY", "QQQ", "IWM", "TLT", "UUP"),
    )
    stream.register(swing_queue, name="swing")
    stream.register(dealer_queue, name="dealer-positioning", symbols=("SPY",))
    if structure_queue is not None:
        stream.register(structure_queue, name="intraday-structure")
    stream.start(all_symbols, env_file=env_file)
    logger.info("Shared bar stream started.")

    # ------------------------------------------------------------------
    # 2. Intraday SPY dashboard (port 8765)
    # ------------------------------------------------------------------
    from UI.live_dashboard import DashboardApp, DashboardHandler, DashboardHTTPServer

    intraday_app = DashboardApp(bar_queue=intraday_queue, live_env_file=live_env_file)
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
    # One swing dashboard; orders submit to paper by default and route to the
    # real-money account only when the page's LIVE toggle is on (needs --live-env).
    swing_app = SwingDashboardApp(
        audit_root=audit_root,
        bar_queue=swing_queue,
        default_env_file=paper_env_file,
        live_env_file=live_env_file,
        live_real_account_policy_state_path=_SWING_LIVE_BOOK,
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
    logger.info("Swing dashboard:         http://%s:%d (LIVE toggle %s)",
                host, port_swing, "available" if live_env_file else "disabled")

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

    # ------------------------------------------------------------------
    # 4a. Momentum Expansion dashboard (port 8770) — read-only scan + manual run
    # ------------------------------------------------------------------
    from UI.momentum_dashboard import MomentumDashboardApp, make_server as _make_momentum_server

    momentum_app = MomentumDashboardApp(env_file=paper_env_file or env_file, live_env_file=live_env_file)
    momentum_server = _make_momentum_server(host, port_momentum, momentum_app)
    momentum_thread = threading.Thread(target=momentum_server.serve_forever, daemon=True, name="momentum-http")
    momentum_thread.start()
    logger.info("Momentum dashboard:      http://%s:%d", host, port_momentum)

    # ------------------------------------------------------------------
    # 4b. HTF Swing dashboard (port 8771) — read-only scores (no live trading)
    # ------------------------------------------------------------------
    from UI.htf_dashboard import HTFDashboardApp, make_server as _make_htf_server

    htf_app = HTFDashboardApp()
    htf_server = _make_htf_server(host, port_htf, htf_app)
    htf_thread = threading.Thread(target=htf_server.serve_forever, daemon=True, name="htf-http")
    htf_thread.start()
    logger.info("HTF Swing dashboard:     http://%s:%d", host, port_htf)

    # ------------------------------------------------------------------
    # 4c. Amethyst dealer-vision dashboard (port 8772) — read-only chart/grid
    # ------------------------------------------------------------------
    from UI.amethyst_dashboard import AmethystDashboardApp, make_server as _make_amethyst_server

    amethyst_app = AmethystDashboardApp()
    amethyst_server = _make_amethyst_server(host, port_amethyst, amethyst_app)
    amethyst_thread = threading.Thread(target=amethyst_server.serve_forever, daemon=True, name="amethyst-http")
    amethyst_thread.start()
    logger.info("Amethyst dashboard:      http://%s:%d", host, port_amethyst)

    # ------------------------------------------------------------------
    # 4d. Dealer Ranker dashboard (port 8773) — near-close rank scan + options
    # ------------------------------------------------------------------
    from UI.dealer_ranker_dashboard import DealerRankerDashboardApp, make_server as _make_dealer_ranker_server

    dealer_ranker_app = DealerRankerDashboardApp(
        env_file=paper_env_file or env_file,
        live_env_file=live_env_file,
        top_k=dealer_ranker_top_k,
        workers=dealer_ranker_workers,
        target_notional=dealer_ranker_target_notional,
    )
    dealer_ranker_server = _make_dealer_ranker_server(host, port_dealer_ranker, dealer_ranker_app)
    dealer_ranker_thread = threading.Thread(
        target=dealer_ranker_server.serve_forever,
        daemon=True,
        name="dealer-ranker-http",
    )
    dealer_ranker_thread.start()
    logger.info("Dealer Ranker dashboard: http://%s:%d", host, port_dealer_ranker)

    # ------------------------------------------------------------------
    # 4e. Intraday Structure Engine — deterministic, paper-only.
    # ------------------------------------------------------------------
    structure_app = structure_server = None
    if structure_queue is not None and structure_config is not None:
        from UI.intraday_structure_dashboard import (
            IntradayStructureDashboardApp,
            make_server as _make_intraday_structure_server,
        )

        structure_app = IntradayStructureDashboardApp(structure_config, structure_queue)
        structure_server = _make_intraday_structure_server(host, port_intraday_structure, structure_app)
        threading.Thread(
            target=structure_server.serve_forever,
            daemon=True,
            name="intraday-structure-http",
        ).start()
        logger.info("Intraday Structure:      http://%s:%d (paper-only)", host, port_intraday_structure)

    # ------------------------------------------------------------------
    # 4f. Library (port 8775) — read-only searchable news/catalyst archive with
    #     a daily price overlay. Never touches a broker; its search index is
    #     built in a subprocess (see LibraryDashboardApp) so the browse page
    #     cannot add the build's ~650MB peak to this process's RSS.
    # ------------------------------------------------------------------
    library_server = None
    if library_enabled:
        from UI.library_dashboard import LibraryDashboardApp, make_server as _make_library_server

        library_app = LibraryDashboardApp()
        library_server = _make_library_server(host, port_library, library_app)
        threading.Thread(target=library_server.serve_forever, daemon=True,
                         name="library-http").start()
        logger.info("Library dashboard:       http://%s:%d", host, port_library)

    # ------------------------------------------------------------------
    # 4g. Hub / overview dashboard (port 8764) — links + status + start-all
    # ------------------------------------------------------------------
    from UI.hub_dashboard import HubDashboardApp, make_server as _make_hub_server

    hub_app = HubDashboardApp(
        host=host, port_spy=port_intraday, port_swing=port_swing,
        port_dealer=port_dealer, port_meta=port_meta,
        port_momentum=port_momentum, port_htf=port_htf,
        port_amethyst=port_amethyst,
        port_dealer_ranker=port_dealer_ranker,
        port_intraday_structure=port_intraday_structure if intraday_structure_enabled else None,
        port_library=port_library if library_enabled else None,
    )
    hub_server = _make_hub_server(host, port_hub, hub_app)
    hub_thread = threading.Thread(target=hub_server.serve_forever, daemon=True, name="hub-http")
    hub_thread.start()
    logger.info("Hub overview:            http://%s:%d", host, port_hub)

    print()
    print("=" * 60)
    print(f"  Hub (overview):          http://{host}:{port_hub}")
    print(f"  Intraday SPY dashboard:  http://{host}:{port_intraday}")
    print(f"  Swing dashboard:         http://{host}:{port_swing}  "
          f"(real money {'available' if live_env_file else 'disabled'})")
    print(f"  HTF Swing (signals):     http://{host}:{port_htf}")
    print(f"  Momentum dashboard:      http://{host}:{port_momentum}")
    print(f"  Amethyst dashboard:      http://{host}:{port_amethyst}")
    print(f"  Dealer positioning:      http://{host}:{port_dealer}")
    print(f"  Dealer Ranker:           http://{host}:{port_dealer_ranker}")
    if intraday_structure_enabled:
        print(f"  Intraday Structure:      http://{host}:{port_intraday_structure}  (paper-only)")
    if library_enabled:
        print(f"  Library (news search):   http://{host}:{port_library}  (read-only)")
    print(f"  Shared stream:           {len(all_symbols)} symbols")
    print("  Orders default to the PAPER account; LIVE toggle is OFF by default.")
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
    # 5a. Nightly jobs scheduler (collection/enrichment after the live refresher closes)
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
            print(f"  Nightly jobs:            daily {nightly_time} ET (bounded collection/enrichment)")

    # ------------------------------------------------------------------
    # 5a1. Optional data-readiness on startup. Default is OFF: live startup should
    #      load the nightly/pre-open caches and start trading sessions, not launch
    #      a full-universe catchup/rebuild during the fragile morning window.
    # ------------------------------------------------------------------
    if data_readiness_on_start:
        threading.Thread(
            target=_run_data_readiness,
            daemon=True,
            name="data-readiness-startup",
        ).start()
        logger.info("Data readiness: running once on startup (background)")
        print("  Data readiness:          on startup (guarded, background)")
    elif data_readiness_startup_check:
        # Cheap stamp read; only launches the heavy job when entries would
        # otherwise be blocked all session.
        threading.Thread(
            target=_data_readiness_startup_check,
            daemon=True,
            name="data-readiness-startup-check",
        ).start()
        print("  Data readiness:          startup check (catch-up only if stamp is stale)")

    # ------------------------------------------------------------------
    # 5a1b. Actions queued for the next startup (see core/startup_queue.py).
    # ------------------------------------------------------------------
    if startup_queue_enabled:
        threading.Thread(
            target=_run_startup_queue,
            kwargs={"env_file": env_file, "account_label": startup_queue_account},
            daemon=True,
            name="startup-queue",
        ).start()
        print(f"  Startup queue:           draining ({startup_queue_account})")

    # ------------------------------------------------------------------
    # 5a0. Continuous background data refresher — keeps shared bars + Meta matrix
    #      fresh OUT-OF-BAND near the 4H decision windows, so startup/the open
    #      remain trading-only. Daily enrichment stays in nightly_market_data.sh.
    # ------------------------------------------------------------------
    if data_refresher:
        from UI.shared_data_refresher import SharedDataRefresher

        refresher = SharedDataRefresher(
            interval=data_refresher_interval, feeds_enabled=data_refresher_feeds,
            stop_event=stop_evt,
        )
        refresher.start()
        logger.info(
            "Shared data refresher: started (every %ds during 13:45-16:40 ET, feeds=%s)",
            data_refresher_interval, data_refresher_feeds,
        )
        print(f"  Data refresher:          every {data_refresher_interval}s ET "
              f"(bars+matrix, guarded)")

    # ------------------------------------------------------------------
    # 5a2. Optional data-readiness scheduler — only useful if you leave the server
    #      running across days. Off by default (the startup run covers the common
    #      "start it each morning" workflow).
    # ------------------------------------------------------------------
    data_readiness_scheduler = None
    if data_readiness_time:
        from UI.nightly_scheduler import NightlyScheduler, parse_hhmm

        try:
            dr_hour, dr_minute = parse_hhmm(data_readiness_time)
        except ValueError as exc:
            logger.error("Invalid --data-readiness-time %r (%s) — readiness job disabled",
                         data_readiness_time, exc)
        else:
            data_readiness_scheduler = NightlyScheduler(
                _run_data_readiness,
                hour=dr_hour,
                minute=dr_minute,
                stop_event=stop_evt,
                name="data-readiness-scheduler",
            )
            data_readiness_scheduler.start()
            logger.info("Data readiness scheduled daily at %s America/New_York (weekdays)",
                        data_readiness_time)
            print(f"  Data readiness:          daily {data_readiness_time} ET (bars+HTF+meta matrix)")

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

    # ------------------------------------------------------------------
    # 5c. Meta Ranker 4H loop (long-only swing; equity or options). The 4H bar
    #     generates the signal and we act on it, so the loop fires after EACH RTH
    #     4H bar CLOSES. The shared bars are SIP (~15-min delay), so a bar is only
    #     fully fetchable ~20 min after it closes — hence "close + 20 min":
    #       * 1st bar 10:00->14:00 ET (stamp 14:00 UTC) -> score 14:20 ET (same-day entry)
    #       * 2nd bar 14:00->16:00 ET (stamp 18:00 UTC) -> score 16:20 ET (post-close;
    #         equity entries fill at the next open — the known small "next-open" cost).
    #     To trade the 2nd bar AT the close instead, that pass would need the
    #     real-time IEX feed (the model trained on SIP, so validate first).
    #     Each pass: catch up the eligible bars -> rebuild matrix -> score + trade.
    #     Themes are refreshed weekly via `update_feeds.py --weekly` (separate).
    # ------------------------------------------------------------------
    # Meta Ranker dashboard (4th dashboard) — display + manual run trigger.
    # Always submits; account defaults to paper, LIVE toggle (off by default)
    # routes to the real-money env when one is configured.
    meta_server = meta_thread = None
    meta_env = paper_env_file or env_file
    from UI.meta_ranker_dashboard import MetaRankerDashboardApp, make_server as _make_meta_server

    meta_app = MetaRankerDashboardApp(
        env_file=meta_env, mode=meta_ranker_mode, submit=False,
    )
    meta_server = _make_meta_server(host, port_meta, meta_app)
    meta_thread = threading.Thread(target=meta_server.serve_forever, daemon=True, name="meta-ranker-http")
    meta_thread.start()
    logger.info("Meta Ranker dashboard at http://%s:%d", host, port_meta)
    print(f"  Meta Ranker dashboard:   http://{host}:{port_meta}/  "
          f"({meta_ranker_mode}/paper/read-only)")

    meta_schedulers: list = []
    if meta_ranker_times:
        from UI.nightly_scheduler import NightlyScheduler, parse_hhmm

        _tag = f"{meta_ranker_mode}/paper/SUBMIT"
        fired_times: list[str] = []
        for raw in str(meta_ranker_times).split(","):
            hhmm = raw.strip()
            if not hhmm:
                continue
            try:
                mh, mm = parse_hhmm(hhmm)
            except ValueError as exc:
                logger.error("Invalid Meta Ranker time %r (%s) — skipped", hhmm, exc)
                continue
            sched = NightlyScheduler(
                lambda: _run_meta_ranker_loop(
                    mode=meta_ranker_mode, submit=True, live=False),
                hour=mh, minute=mm, stop_event=stop_evt,
                name=f"meta-ranker-scheduler-{hhmm}",
            )
            sched.start()
            meta_schedulers.append(sched)
            fired_times.append(hhmm)
        if fired_times:
            logger.info("Meta Ranker loop scheduled at %s ET (%s)", ", ".join(fired_times), _tag)
            print(f"  Meta Ranker loop:        {', '.join(fired_times)} ET per 4H bar ({_tag})")

    # ------------------------------------------------------------------
    # 5d. Standalone base-model harnesses (HTF Swing, Momentum) — each trades
    #     its own base score (the same one feeding Meta) via its own order policy,
    #     on the same multi-time 4H cadence. Scheduled a few minutes AFTER Meta so
    #     they read the bars/matrix Meta just refreshed.
    # ------------------------------------------------------------------
    def _schedule_loop(times: str, job, *, label: str, tag: str) -> list:
        scheds: list = []
        if not times:
            return scheds
        from UI.nightly_scheduler import NightlyScheduler, parse_hhmm

        fired: list[str] = []
        for raw in str(times).split(","):
            hhmm = raw.strip()
            if not hhmm:
                continue
            try:
                hh, mm = parse_hhmm(hhmm)
            except ValueError as exc:
                logger.error("Invalid %s time %r (%s) — skipped", label, hhmm, exc)
                continue
            sched = NightlyScheduler(job, hour=hh, minute=mm, stop_event=stop_evt,
                                     name=f"{label}-scheduler-{hhmm}")
            sched.start()
            scheds.append(sched)
            fired.append(hhmm)
        if fired:
            logger.info("%s loop scheduled at %s ET (%s)", label, ", ".join(fired), tag)
            print(f"  {label} loop:{' ' * max(1, 16 - len(label))}{', '.join(fired)} ET per 4H bar ({tag})")
        return scheds

    htf_tag = f"{htf_mode}/{'LIVE' if htf_live else 'paper'}/SUBMIT"
    htf_schedulers = _schedule_loop(
        htf_times,
        lambda: _run_htf_loop(mode=htf_mode, submit=True, live=htf_live),
        label="HTF Swing", tag=htf_tag,
    )
    mom_tag = f"options/{'LIVE' if momentum_live else 'paper'}/SUBMIT"
    momentum_schedulers = _schedule_loop(
        momentum_times,
        lambda: _run_momentum_loop(submit=True, live=momentum_live),
        label="Momentum", tag=mom_tag,
    )
    dealer_ranker_tag = f"options/{'LIVE' if dealer_ranker_live else 'paper'}/SUBMIT"
    dealer_ranker_schedulers = _schedule_loop(
        dealer_ranker_time,
        lambda: _run_dealer_ranker_loop(
            submit=True,
            live=dealer_ranker_live,
            top_k=dealer_ranker_top_k,
            workers=dealer_ranker_workers,
            target_notional=dealer_ranker_target_notional,
        ),
        label="Dealer Ranker",
        tag=dealer_ranker_tag,
    )

    # Capture exact broker marks shortly after the regular and extended-session
    # closes.  This is read-only and deliberately independent of signal/order
    # paths, so reporting remains available even when entry readiness is stale.
    snapshot_schedulers: list = []
    snapshot_accounts = [("paper", meta_env)]
    if live_env_file:
        snapshot_accounts.append(("live", live_env_file))
    for account_label, snapshot_env in snapshot_accounts:
        snapshot_schedulers += _schedule_loop(
            "16:05,20:05",
            lambda env=snapshot_env, label=account_label: _run_broker_equity_snapshot(
                env_file=env, account_label=label,
            ),
            label=f"Broker snapshot {account_label}", tag="READ-ONLY",
        )

    # Re-drain the startup queue during market hours.
    #
    # The boot-time drain (5a1b above) runs once, and `close_position` entries
    # need an open market -- so a server started overnight (the normal case)
    # defers the entry and then never looks at it again until the next restart.
    # That made a queued liquidate fire only if you happened to boot between
    # 9:30 and 16:00 ET. These slots close that hole: 09:35 (after the opening
    # auction settles) with a 13:00 backstop for a server started mid-morning.
    # `run_pending` skips anything not still `pending`, so re-running is a
    # no-op once an entry has completed -- it cannot double-submit an order.
    startup_queue_schedulers: list = []
    if startup_queue_enabled:
        startup_queue_schedulers = _schedule_loop(
            startup_queue_market_times,
            lambda: _run_startup_queue(
                env_file=env_file, account_label=startup_queue_account,
            ),
            label="Startup queue (market hours)", tag="ORDERS",
        )

    # Shadow tracker: 10 min after each Momentum/HTF/Meta cycle (14:20/16:20 and
    # 14:25/16:25) so it reads live_state.json after that cycle's entries/exits
    # have landed. Read-only against those state files; never calls a broker.
    shadow_tracker_schedulers = _schedule_loop(
        "14:35,16:35",
        _run_shadow_tracker,
        label="Shadow tracker (2-sleeve)", tag="READ-ONLY",
    )

    # ------------------------------------------------------------------
    # 5e. Pre-open flush (~09:35 ET). The pm 4H bar (18:00 UTC = 2-6pm ET) only
    #     finishes AFTER the 16:00 equity close, so its entries can't fill same-day
    #     and are queued (see defer_entries_if_market_closed). Just after the 09:30
    #     open we re-rank each queue against today's fresh top-K and submit the
    #     survivors as normal day orders, then exit — matching the "next_open" fill
    #     that backtest_exits.py validated. Staggered so Meta re-scores first.
    # ------------------------------------------------------------------
    flush_schedulers: list = []
    if meta_ranker_times:
        flush_schedulers += _schedule_loop(
            "09:35",
            lambda: _run_meta_ranker_loop(mode=meta_ranker_mode, submit=True, live=False, flush=True),
            label="Meta pre-open flush", tag=_tag,
        )
    if htf_times:
        flush_schedulers += _schedule_loop(
            "09:37",
            lambda: _run_htf_loop(mode=htf_mode, submit=True, live=htf_live, flush=True),
            label="HTF pre-open flush", tag=htf_tag,
        )
    if momentum_times:
        flush_schedulers += _schedule_loop(
            "09:37",
            lambda: _run_momentum_loop(submit=True, live=momentum_live, flush=True),
            label="Momentum pre-open flush", tag=mom_tag,
        )

    # Optional boot-time equivalent of clicking Hub -> Start All. Keep this
    # paper-default so launching the combined server never implies real-money
    # order routing without the existing explicit per-module LIVE controls.
    if start_all:
        def _auto_start_all() -> None:
            logger.info("Auto Start All requested; launching startable dashboards in paper mode.")
            results = hub_app.start_all({})
            failures = [r for r in results.get("results", []) if not r.get("ok")]
            if failures:
                logger.warning("Auto Start All completed with issues: %s", failures)
            else:
                logger.info("Auto Start All completed successfully.")

        threading.Thread(target=_auto_start_all, daemon=True, name="hub-auto-start-all").start()
        print("  Auto Start All:          enabled (paper-default, background)")

    # ------------------------------------------------------------------
    # 5e. Memory keeper — periodic gc + malloc_trim to cap long-run RSS growth
    #     (this process OOM-killed at the 16GB WSL cap on 2026-06-26). Logs RSS
    #     each tick so footprint regressions are visible in the server log.
    # ------------------------------------------------------------------
    _start_memory_keeper(stop_evt, interval=300)

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
    if structure_app is not None:
        try:
            structure_app.stop()
        except Exception as exc:
            logger.warning("structure_app.stop(): %s", exc)
    logger.info("Stopping HTTP servers...")
    hub_server.shutdown()
    intraday_server.shutdown()
    swing_server.shutdown()
    dealer_server.shutdown()
    momentum_server.shutdown()
    htf_server.shutdown()
    amethyst_server.shutdown()
    dealer_ranker_server.shutdown()
    if structure_server is not None:
        structure_server.shutdown()
    if library_server is not None:
        library_server.shutdown()
    if meta_server is not None:
        meta_server.shutdown()

    logger.info("Stopping shared bar stream...")
    stream.unregister(intraday_queue)
    stream.unregister(swing_queue)
    stream.unregister(dealer_queue)
    if structure_queue is not None:
        stream.unregister(structure_queue)
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
                        help="Port for the swing dashboard (default 8766).")
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
        default="16:45",
        help="Local (America/New_York) HH:MM to run the bounded nightly collection/enrichment job "
             "after the 16:40 shared-data refresher close (default 16:45).",
    )
    parser.add_argument(
        "--no-nightly-jobs",
        action="store_true",
        help="Disable the canonical in-process nightly scheduler (manual execution only).",
    )
    parser.add_argument(
        "--readiness-on-start",
        action="store_true",
        help="Run the guarded shared-bars/HTF/Meta-matrix readiness job at server startup. "
             "Default is off so live boot stays trading-only.",
    )
    parser.add_argument(
        "--no-readiness-on-start",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--no-startup-queue",
        action="store_true",
        help="Do not drain Data/runtime/startup_queue.json at boot. By default queued "
             "actions (position closes, forced readiness rebuilds) run once at startup; "
             "order entries stay pending while the market is closed.",
    )
    parser.add_argument(
        "--startup-queue-account",
        default="paper",
        choices=("paper", "live"),
        help="Account the startup queue may act on (default paper). Entries marked "
             "account=live are skipped unless this is 'live'.",
    )
    parser.add_argument(
        "--no-readiness-startup-check",
        action="store_true",
        help="Disable the boot-time readiness-stamp check. By default the server reads the "
             "stamp at startup and, ONLY if it would not authorize entries, runs the "
             "readiness job immediately (overriding the live-window guard, since a stale "
             "stamp means the 4H modules place no entry orders at all).",
    )
    parser.add_argument(
        "--data-readiness-time",
        default="",
        help="Optional local (America/New_York) HH:MM to ALSO refresh shared bars + HTF "
             "features + Meta matrix on a daily timer (only useful if the server runs "
             "across days; the on-startup refresh already covers a daily restart). "
             "Empty = timer disabled.",
    )
    parser.add_argument(
        "--no-data-refresher",
        action="store_true",
        help="Disable the continuous background shared-bars + Meta-matrix refresher. With it "
             "off, the matrix only refreshes on startup/nightly and the runner-only 4H loops "
             "will abort on staleness during a multi-day run.",
    )
    parser.add_argument(
        "--data-refresher-interval",
        type=int,
        default=300,
        help="Seconds to sleep between background refresh cycles during the session (default 300).",
    )
    parser.add_argument(
        "--data-refresher-feeds",
        action="store_true",
        help="Also run light feed refreshes from the afternoon data refresher. Default is off; "
             "post-close nightly collection owns daily enrichment.",
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
    parser.add_argument(
        "--meta-ranker-times",
        default="14:20,16:20",
        help="Comma-separated local ET HH:MM times to fire the Meta Ranker 4H loop — one per "
             "RTH 4H bar, set to each bar's close + the ~20-min SIP delay (default "
             "'14:20,16:20'). Each pass catches up bars, rebuilds the matrix, and scores+trades. "
             "Pass '' to disable.",
    )
    parser.add_argument("--meta-ranker-mode", choices=["equity", "options"], default="options",
                        help="Meta Ranker order type (default options).")
    parser.add_argument("--meta-ranker-live", action="store_true",
                        # Rejected at startup; kept so an old invocation is not
                        # silently ignored.
                        help="Run the scheduled Meta Ranker loop against the LIVE account (default paper).")
    parser.add_argument(
        "--htf-times", default="14:25,16:25",
        help="Comma-separated ET HH:MM to fire the standalone HTF Swing loop, a few min after "
             "the Meta times so it reads the refreshed matrix (default '14:25,16:25'). '' disables.")
    parser.add_argument("--htf-mode", choices=["equity", "options"], default="options",
                        help="HTF Swing order type (default options).")
    parser.add_argument("--htf-live", action="store_true",
                        help="Run the scheduled HTF Swing loop against the LIVE account (default paper).")
    parser.add_argument(
        "--momentum-times", default="14:25,16:25",
        help="Comma-separated ET HH:MM to fire the standalone Momentum loop (default '14:25,16:25'). "
             "'' disables.")
    parser.add_argument("--momentum-live", action="store_true",
                        help="Run the scheduled Momentum loop against the LIVE account (default paper).")
    parser.add_argument("--port-hub", type=int, default=DEFAULT_PORT_HUB,
                        help="Port for the hub / overview dashboard (default 8764).")
    parser.add_argument("--port-meta", type=int, default=DEFAULT_PORT_META,
                        help="Port for the Meta Ranker dashboard (default 8769).")
    parser.add_argument("--port-momentum", type=int, default=DEFAULT_PORT_MOMENTUM,
                        help="Port for the Momentum Expansion dashboard (default 8770).")
    parser.add_argument("--port-htf", type=int, default=DEFAULT_PORT_HTF,
                        help="Port for the HTF Swing signals dashboard (default 8771).")
    parser.add_argument("--port-amethyst", type=int, default=DEFAULT_PORT_AMETHYST,
                        help="Port for the Amethyst dealer-vision dashboard (default 8772).")
    parser.add_argument("--port-dealer-ranker", type=int, default=DEFAULT_PORT_DEALER_RANKER,
                        help="Port for the Dealer Ranker dashboard (default 8773).")
    parser.add_argument("--port-intraday-structure", type=int, default=DEFAULT_PORT_INTRADAY_STRUCTURE,
                        help="Port for the Intraday Structure dashboard (default 8774).")
    parser.add_argument("--port-library", type=int, default=DEFAULT_PORT_LIBRARY,
                        help="Port for the Library news/catalyst search dashboard (default 8775).")
    parser.add_argument(
        "--library",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run the read-only Library news/catalyst search dashboard (default; "
             "use --no-library to disable).",
    )
    parser.add_argument(
        "--intraday-structure",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run the deterministic, paper-only Intraday Structure engine (default; use --no-intraday-structure to disable).",
    )
    parser.add_argument(
        "--intraday-structure-config",
        default="strategies/intraday_structure/config/intraday_structure_v1.json",
        help="Versioned Intraday Structure config JSON path.",
    )
    parser.add_argument(
        "--dealer-ranker-time",
        default="15:45",
        help="ET HH:MM to run the near-close Dealer Ranker scan+options refresh pass (default 15:45). "
             "Pass '' to disable; manual dashboard runs remain available.",
    )
    parser.add_argument("--dealer-ranker-live", action="store_true",
                        help="Run the scheduled Dealer Ranker pass against the LIVE account (default paper).")
    parser.add_argument("--dealer-ranker-workers", type=int, default=8,
                        help="Parallel dealer-chain workers for the Dealer Ranker pass (default 8).")
    parser.add_argument("--dealer-ranker-top-k", type=int, default=10,
                        help="Number of dealer-ranked names to target (default 10).")
    parser.add_argument("--dealer-ranker-target-notional", type=float, default=5000.0,
                        help="Dollar size per Dealer Ranker entry; contracts computed from premium (default 5000).")
    parser.add_argument(
        "--start-all",
        action="store_true",
        help="Automatically start every startable dashboard after boot, matching the hub Start All "
             "button's paper-default account routing.",
    )
    args = parser.parse_args()

    if args.meta_ranker_live:
        # The Meta path is governed and paper-only. Failing at startup is
        # safer than booting with a flag the runner will refuse anyway.
        print(
            "ERROR: --meta-ranker-live is not supported; the Meta Ranker path "
            "is paper-only in this MVP.",
            file=sys.stderr,
        )
        raise SystemExit(2)

    run_combined(
        host=args.host,
        port_hub=args.port_hub,
        port_intraday=args.port_intraday,
        port_swing=args.port_swing,
        port_dealer=args.port_dealer,
        env_file=args.env,
        paper_env_file=args.paper_env,
        live_env_file=args.live_env,
        audit_root=Path(args.audit_root),
        nightly_time=None if args.no_nightly_jobs else args.nightly_time,
        data_readiness_time=(args.data_readiness_time or None),
        data_readiness_on_start=bool(args.readiness_on_start),
        data_readiness_startup_check=not bool(args.no_readiness_startup_check),
        startup_queue_enabled=not bool(args.no_startup_queue),
        startup_queue_account=str(args.startup_queue_account),
        data_refresher=not args.no_data_refresher,
        data_refresher_interval=args.data_refresher_interval,
        data_refresher_feeds=bool(args.data_refresher_feeds),
        catalyst_poll=not args.no_catalyst_poll,
        catalyst_poll_interval=args.catalyst_poll_interval,
        port_meta=args.port_meta,
        port_momentum=args.port_momentum,
        port_htf=args.port_htf,
        port_amethyst=args.port_amethyst,
        port_dealer_ranker=args.port_dealer_ranker,
        port_intraday_structure=args.port_intraday_structure,
        port_library=args.port_library,
        library_enabled=bool(args.library),
        intraday_structure_enabled=bool(args.intraday_structure),
        intraday_structure_config=args.intraday_structure_config,
        meta_ranker_times=args.meta_ranker_times,
        meta_ranker_mode=args.meta_ranker_mode,
        meta_ranker_live=args.meta_ranker_live,
        htf_times=args.htf_times,
        htf_mode=args.htf_mode,
        htf_live=args.htf_live,
        momentum_times=args.momentum_times,
        momentum_live=args.momentum_live,
        dealer_ranker_time=args.dealer_ranker_time,
        dealer_ranker_live=args.dealer_ranker_live,
        dealer_ranker_workers=args.dealer_ranker_workers,
        dealer_ranker_top_k=args.dealer_ranker_top_k,
        dealer_ranker_target_notional=args.dealer_ranker_target_notional,
        start_all=args.start_all,
    )


if __name__ == "__main__":
    main()
