#!/usr/bin/env python
"""Nightly ticker discovery + promotion gate.

Runs the broad market screen and catalyst discovery, merges new candidates into
the pending queue, and releases (promotes) any that meet the price / ADV /
market-cap / history minimums. Promoted names are folded into the shared
universe on the next ``build_shared_universe`` (passing ``--rebuild-universe``
does it here).

Examples
--------
    # Full nightly run (screen + catalyst), default $100M small-cap floor:
    python -m scripts.nightly_ticker_discovery --rebuild-universe

    # Catalyst-only (skip the broad Alpaca scan):
    python -m scripts.nightly_ticker_discovery --no-screen

    # Dry sample of the broad scan (first N symbols) for a quick check:
    python -m scripts.nightly_ticker_discovery --max-symbols 300
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.shared_universe.discovery import DiscoveryConfig, run_discovery
from core.shared_universe.universe import build_shared_universe, summarize_universe


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--env-file", default=".env", help="Alpaca env file/profile (default .env)")
    parser.add_argument("--no-screen", action="store_true", help="Skip the broad Alpaca market scan")
    parser.add_argument("--no-catalyst", action="store_true", help="Skip catalyst-ledger discovery")
    parser.add_argument("--max-symbols", type=int, default=None, help="Cap broad-scan symbols (debug/sampling)")
    parser.add_argument("--min-market-cap", type=float, default=None, help="Override market-cap floor (USD)")
    parser.add_argument("--min-adv", type=float, default=None, help="Override 20d avg dollar volume floor (USD)")
    parser.add_argument("--min-history-days", type=int, default=None, help="Override history-bars quarantine")
    parser.add_argument("--min-observation-days", type=int, default=None, help="Minimum calendar days in pending before promotion")
    parser.add_argument("--rebuild-universe", action="store_true", help="Rebuild shared_universe.csv after promotion")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    overrides = {}
    if args.min_market_cap is not None:
        overrides["min_market_cap"] = args.min_market_cap
    if args.min_adv is not None:
        overrides["min_avg_dollar_volume_20d"] = args.min_adv
    if args.min_history_days is not None:
        overrides["min_history_days"] = args.min_history_days
    if args.min_observation_days is not None:
        overrides["min_observation_days"] = args.min_observation_days
    cfg = DiscoveryConfig(**overrides) if overrides else DiscoveryConfig()

    summary = run_discovery(
        cfg,
        env_file=args.env_file,
        do_screen=not args.no_screen,
        do_catalyst=not args.no_catalyst,
        max_symbols=args.max_symbols,
    )

    print("\n=== Discovery summary ===")
    print(f"  discovered this run : {summary['discovered']}")
    print(f"  pending (waiting)   : {summary['pending_waiting']}")
    print(f"  promoted (total)    : {summary['promoted_total']}")
    print(f"  newly promoted      : {', '.join(summary['newly_promoted']) or '(none)'}")
    print(f"  queue file          : {summary['pending_path']}")
    print(f"  novel pending       : {summary['novel_pending']}")
    print(f"  novel queue file    : {summary['novel_pending_path']}")

    if args.rebuild_universe:
        df = build_shared_universe()
        print("\n=== Shared universe rebuilt ===")
        print(f"  {summarize_universe(df)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
