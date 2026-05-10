"""
CLI entry point for momentum_expansion.

Examples:
    # Build the weekly universe snapshot for today
    python -m momentum_expansion.main --refresh-universe

    # Pull bars for a small starter set
    python -m momentum_expansion.main --fetch --tickers AAPL MSFT NVDA

    # Build features + labels + training matrix for everything cached
    python -m momentum_expansion.main --build-features --build-labels --build-matrix

    # Bundle for Colab
    python -m momentum_expansion.main --export-colab

    # Live evaluation tick (no auto-trade)
    python -m momentum_expansion.main --live-evaluate
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from momentum_expansion.config.momentum_config import (
    CONTEXT_TICKERS,
    FEATURES_COMBINED,
    LABELS_COMBINED,
    SECTOR_ETFS,
    TRAINING_EXPORT_DIR,
    TRAINING_MATRIX,
)
from momentum_expansion.data.bars import (
    fetch_context_bars,
    fetch_daily_for_universe_scoring,
    fetch_universe_bars,
)
from momentum_expansion.data.universe import (
    filter_context_only_tickers,
    get_candidate_pool,
    list_snapshots,
    load_snapshot_for,
    write_weekly_snapshot,
)
from momentum_expansion.features.feature_matrix_4h import build_all_features_4h
from momentum_expansion.labels.expansion_labels import (
    build_all_labels_4h,
    build_training_matrix,
)


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def main() -> int:
    p = argparse.ArgumentParser(description="momentum_expansion CLI")
    p.add_argument("--log", default="INFO")
    p.add_argument("--refresh-universe", action="store_true",
                   help="Score candidate pool and write a weekly snapshot")
    p.add_argument("--fetch", action="store_true",
                   help="Pull 1H + 1D bars for tickers (and resample to 4H)")
    p.add_argument("--fetch-context", action="store_true",
                   help="Pull 1H bars for context tickers + sector ETFs")
    p.add_argument("--fetch-daily-only", action="store_true",
                   help="Pull 1D only (used for universe scoring)")
    p.add_argument("--build-features", action="store_true")
    p.add_argument("--build-labels", action="store_true")
    p.add_argument("--build-matrix", action="store_true")
    p.add_argument("--export-colab", action="store_true",
                   help="Write training_matrix to Colab bundle")
    p.add_argument("--live-evaluate", action="store_true",
                   help="One-shot: score active universe and emit alerts")
    p.add_argument("--tickers", nargs="*", default=None)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    _setup_logging(args.log)

    # Determine ticker set
    if args.tickers:
        tickers = list(args.tickers)
    else:
        snap = pd.DataFrame()
        if list_snapshots():
            snap = load_snapshot_for(pd.Timestamp.utcnow())
        tickers = snap["ticker"].astype(str).tolist() if not snap.empty else get_candidate_pool()
    candidate_tickers = filter_context_only_tickers(tickers)

    if args.refresh_universe:
        write_weekly_snapshot(as_of=pd.Timestamp.utcnow().normalize())

    if args.fetch_daily_only:
        fetch_daily_for_universe_scoring(tickers=candidate_tickers, force=args.force)

    if args.fetch_context:
        fetch_context_bars(force=args.force)

    if args.fetch:
        fetch_universe_bars(tickers=candidate_tickers, force=args.force)

    if args.build_features:
        build_all_features_4h(tickers=candidate_tickers, force=args.force)

    if args.build_labels:
        build_all_labels_4h(tickers=candidate_tickers, force=args.force)

    if args.build_matrix:
        build_training_matrix(force=args.force)

    if args.export_colab:
        from momentum_expansion.models.export_for_colab import export_training_bundle
        export_training_bundle()

    if args.live_evaluate:
        from momentum_expansion.live.runner import MomentumLiveRunner
        runner = MomentumLiveRunner(auto_trade=False)
        runner.evaluate_now()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
