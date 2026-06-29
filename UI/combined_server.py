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


def _run_data_readiness() -> None:
    """Refresh the shared bars + HTF features + Meta matrix so HTF Swing,
    Momentum, and Meta Ranker all wake up caught up to the same state.

    Reuses ``scripts/nightly_data_readiness.sh`` (one source of truth). Run
    off-hours (pre-open) so it never collides with the live session, the 15:50
    Meta MOC loop, or the nightly news job.
    """
    import shutil
    import subprocess

    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "nightly_data_readiness.sh"
    bash = shutil.which("bash")
    if bash is None:
        logger.error("Data readiness: 'bash' not found on PATH — cannot run %s.", script)
        return
    logger.info("Data readiness: launching %s", script)
    result = subprocess.run([bash, str(script)], cwd=str(repo_root))
    logger.info("Data readiness: finished (exit=%s)", result.returncode)


def _build_symbol_union(env_file: str) -> list[str]:
    """Union of all symbols needed by both dashboards."""
    from strategies.multi_ticker_swing.config.pipeline_config import CONTEXT_TICKERS
    from strategies.multi_ticker_swing.live.universe import load_universe

    swing_universe = set(load_universe().keys())
    combined = swing_universe | set(CONTEXT_TICKERS) | set(_INTRADAY_EXTRA_SYMBOLS)
    return sorted(combined)


def _run_meta_ranker_loop(*, mode: str, submit: bool, live: bool) -> None:
    """Fire one Meta Ranker 4H loop pass (bars -> feeds -> matrix -> runner).

    Runs as a subprocess so a failure can't take down the combined server. Themes
    are NOT refreshed here (weekly job: `update_feeds.py --weekly`).
    """
    import os
    import subprocess
    import sys

    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "signals/meta_context/meta_ranker/run_4h_loop.py"
    argv = [sys.executable, str(script), "--mode", mode]
    if submit:
        argv.append("--submit")
    if live:
        argv.append("--live")
    logger.info("Meta Ranker loop: launching %s", " ".join(argv))
    env = {**os.environ, "PYTHONPATH": str(repo_root)}
    subprocess.run(argv, cwd=str(repo_root), env=env)


def _run_htf_loop(*, mode: str, submit: bool, live: bool) -> None:
    """Fire one standalone HTF Swing pass (reads htf_score off the shared matrix).

    Runs as a subprocess so a failure can't take down the combined server. The
    shared matrix is kept fresh by the data-readiness job + the Meta loop, so this
    just reads it; schedule it a few minutes AFTER the Meta times.
    """
    import os
    import subprocess
    import sys

    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "strategies/multi_ticker_swing_htf/live/runner.py"
    argv = [sys.executable, str(script), "--mode", mode]
    if submit:
        argv.append("--submit")
    if live:
        argv.append("--live")
    logger.info("HTF Swing loop: launching %s", " ".join(argv))
    env = {**os.environ, "PYTHONPATH": str(repo_root)}
    subprocess.run(argv, cwd=str(repo_root), env=env)


def _run_momentum_loop(*, submit: bool, live: bool) -> None:
    """Fire one standalone Momentum Expansion pass (ExpansionRanker -> own policy).

    Subprocess-isolated. Reads 4H/1H bars off disk (kept fresh by data-readiness +
    the Meta loop's bar catchup), so schedule it a few minutes AFTER the Meta times.
    """
    import os
    import subprocess
    import sys

    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "strategies/momentum_expansion/live/runner.py"
    argv = [sys.executable, str(script)]
    if submit:
        argv.append("--submit")
    if live:
        argv.append("--live")
    logger.info("Momentum loop: launching %s", " ".join(argv))
    env = {**os.environ, "PYTHONPATH": str(repo_root)}
    subprocess.run(argv, cwd=str(repo_root), env=env)


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
    nightly_time: str | None = "16:30",
    data_readiness_time: str | None = None,
    data_readiness_on_start: bool = True,
    catalyst_poll: bool = True,
    catalyst_poll_interval: int = 300,
    port_meta: int = DEFAULT_PORT_META,
    port_momentum: int = DEFAULT_PORT_MOMENTUM,
    port_htf: int = DEFAULT_PORT_HTF,
    meta_ranker_times: str = "14:20,16:20",
    meta_ranker_mode: str = "equity",
    meta_ranker_live: bool = False,
    htf_times: str = "14:25,16:25",
    htf_mode: str = "equity",
    htf_live: bool = False,
    momentum_times: str = "14:25,16:25",
    momentum_live: bool = False,
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

    # ------------------------------------------------------------------
    # 1. Build shared symbol universe and start the single WebSocket stream
    # ------------------------------------------------------------------
    logger.info("Building symbol universe...")
    all_symbols = _build_symbol_union(env_file)
    logger.info("Symbol universe: %d symbols", len(all_symbols))

    # Separate queues per dashboard. Subscriber filters keep SPY-only consumers
    # from receiving the full swing universe.
    intraday_queue: queue_mod.Queue = queue_mod.Queue(maxsize=10_000)
    swing_queue: queue_mod.Queue = queue_mod.Queue(maxsize=50_000)
    dealer_queue: queue_mod.Queue = queue_mod.Queue(maxsize=10_000)

    from UI.shared_stream import get_shared_bar_stream
    stream = get_shared_bar_stream()
    stream.register(
        intraday_queue,
        name="intraday",
        symbols=("SPY", "VIXY", "QQQ", "IWM", "TLT", "UUP"),
    )
    stream.register(swing_queue, name="swing")
    stream.register(dealer_queue, name="dealer-positioning", symbols=("SPY",))
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
    # 4c. Hub / overview dashboard (port 8764) — links + status + start-all
    # ------------------------------------------------------------------
    from UI.hub_dashboard import HubDashboardApp, make_server as _make_hub_server

    hub_app = HubDashboardApp(
        host=host, port_spy=port_intraday, port_swing=port_swing,
        port_dealer=port_dealer, port_meta=port_meta,
        port_momentum=port_momentum, port_htf=port_htf,
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
    print(f"  Dealer positioning:      http://{host}:{port_dealer}")
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
    # 5a1. Data-readiness ON STARTUP — refresh shared bars + HTF features + Meta
    #      matrix as soon as the server boots, so the three shared-universe
    #      modules (HTF/Momentum/Meta) are caught up the moment you start it each
    #      morning. This is the reliable trigger: a fixed-time cron never fires if
    #      the server isn't running yet. Runs in a background thread so dashboards
    #      come up immediately; the staleness guards hold the line until it lands.
    # ------------------------------------------------------------------
    if data_readiness_on_start:
        threading.Thread(
            target=_run_data_readiness,
            daemon=True,
            name="data-readiness-startup",
        ).start()
        logger.info("Data readiness: running once on startup (background)")
        print("  Data readiness:          on startup (bars+HTF+meta matrix, background)")

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
        env_file=meta_env, live_env_file=live_env_file, mode=meta_ranker_mode, submit=True,
    )
    meta_server = _make_meta_server(host, port_meta, meta_app)
    meta_thread = threading.Thread(target=meta_server.serve_forever, daemon=True, name="meta-ranker-http")
    meta_thread.start()
    logger.info("Meta Ranker dashboard at http://%s:%d", host, port_meta)
    print(f"  Meta Ranker dashboard:   http://{host}:{port_meta}/  "
          f"({meta_ranker_mode}/{'LIVE' if meta_ranker_live else 'paper'}/SUBMIT)")

    meta_schedulers: list = []
    if meta_ranker_times:
        from UI.nightly_scheduler import NightlyScheduler, parse_hhmm

        _tag = f"{meta_ranker_mode}/{'LIVE' if meta_ranker_live else 'paper'}/SUBMIT"
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
                    mode=meta_ranker_mode, submit=True, live=meta_ranker_live),
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
    logger.info("Stopping HTTP servers...")
    hub_server.shutdown()
    intraday_server.shutdown()
    swing_server.shutdown()
    dealer_server.shutdown()
    momentum_server.shutdown()
    htf_server.shutdown()
    if meta_server is not None:
        meta_server.shutdown()

    logger.info("Stopping shared bar stream...")
    stream.unregister(intraday_queue)
    stream.unregister(swing_queue)
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
        default="16:30",
        help="Local (America/New_York) HH:MM to run nightly CBOE/FINRA/discovery jobs (default 16:30).",
    )
    parser.add_argument(
        "--no-nightly-jobs",
        action="store_true",
        help="Disable the in-process nightly scheduler (use system cron instead).",
    )
    parser.add_argument(
        "--no-readiness-on-start",
        action="store_true",
        help="Disable the on-startup shared-bars/HTF/Meta-matrix refresh (on by default).",
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
    parser.add_argument("--meta-ranker-mode", choices=["equity", "options"], default="equity",
                        help="Meta Ranker order type (default equity).")
    parser.add_argument("--meta-ranker-live", action="store_true",
                        help="Run the scheduled Meta Ranker loop against the LIVE account (default paper).")
    parser.add_argument(
        "--htf-times", default="14:25,16:25",
        help="Comma-separated ET HH:MM to fire the standalone HTF Swing loop, a few min after "
             "the Meta times so it reads the refreshed matrix (default '14:25,16:25'). '' disables.")
    parser.add_argument("--htf-mode", choices=["equity", "options"], default="equity",
                        help="HTF Swing order type (default equity).")
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
    parser.add_argument(
        "--start-all",
        action="store_true",
        help="Automatically start every startable dashboard after boot, matching the hub Start All "
             "button's paper-default account routing.",
    )
    args = parser.parse_args()

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
        data_readiness_on_start=not args.no_readiness_on_start,
        catalyst_poll=not args.no_catalyst_poll,
        catalyst_poll_interval=args.catalyst_poll_interval,
        port_meta=args.port_meta,
        port_momentum=args.port_momentum,
        port_htf=args.port_htf,
        meta_ranker_times=args.meta_ranker_times,
        meta_ranker_mode=args.meta_ranker_mode,
        meta_ranker_live=args.meta_ranker_live,
        htf_times=args.htf_times,
        htf_mode=args.htf_mode,
        htf_live=args.htf_live,
        momentum_times=args.momentum_times,
        momentum_live=args.momentum_live,
        start_all=args.start_all,
    )


if __name__ == "__main__":
    main()
