"""
Multi-ticker swing trading pipeline — CLI entry point.

Stages (run in order):
  fetch       Pull 30m + 10m bars for all universe tickers
  features    Build 30m feature matrix per ticker → combined parquet
  labels      Build swing + triple-barrier labels → combined parquet
  matrix      Merge features + labels → training_matrix.parquet (NaN-dropped)
  all         Run fetch → features → labels → matrix in sequence

Training is a separate script:
  python multi_ticker_swing/models/train.py

Backtest is a separate script:
  python multi_ticker_swing/backtest/simulate.py

Usage examples (run from repo root):
  python multi_ticker_swing/main.py --stage fetch
  python multi_ticker_swing/main.py --stage features
  python multi_ticker_swing/main.py --stage labels
  python multi_ticker_swing/main.py --stage matrix
  python multi_ticker_swing/main.py --stage all
  python multi_ticker_swing/main.py --stage all --force
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Arg parsing
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Multi-ticker swing pipeline: fetch → features → labels → matrix",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument(
        "--stage",
        choices=["fetch", "features", "labels", "matrix", "all"],
        default="all",
        help=(
            "Pipeline stage to run:\n"
            "  fetch     — download raw 30m + 10m OHLCV for all tickers\n"
            "  features  — build 30m feature matrix\n"
            "  labels    — build swing + triple-barrier labels\n"
            "  matrix    — merge features + labels → training_matrix.parquet\n"
            "  all       — run all stages in order"
        ),
    )
    p.add_argument(
        "--start",
        default=None,
        help="Override fetch start date (ISO format, e.g. 2021-01-01)",
    )
    p.add_argument(
        "--end",
        default=None,
        help="Override fetch end date (ISO format, e.g. 2026-04-15)",
    )
    p.add_argument(
        "--universe",
        default=None,
        help="Path to universe CSV (default: config/universe.csv)",
    )
    p.add_argument(
        "--no-fetch-30m",
        dest="fetch_30m",
        action="store_false",
        default=True,
        help="Skip fetching 30m bars (only relevant for --stage fetch)",
    )
    p.add_argument(
        "--no-fetch-10m",
        dest="fetch_10m",
        action="store_false",
        default=True,
        help="Skip fetching 10m bars (only relevant for --stage fetch)",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Recompute and overwrite cached outputs at every stage",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Stage runners
# ---------------------------------------------------------------------------

def run_fetch(args: argparse.Namespace) -> None:
    from strategies.multi_ticker_swing.config.pipeline_config import TRAIN_START, TRAIN_END, UNIVERSE_CSV
    from strategies.multi_ticker_swing.data.fetch_data import fetch_all

    start = args.start or TRAIN_START
    end   = args.end   or TRAIN_END
    csv   = Path(args.universe) if args.universe else UNIVERSE_CSV

    logger.info("=== STAGE: fetch  (%s → %s) ===", start, end)
    fetch_all(
        start=start,
        end=end,
        universe_csv=csv,
        fetch_30m=args.fetch_30m,
        fetch_10m=args.fetch_10m,
        force=args.force,
    )
    logger.info("=== fetch complete ===")


def run_features(args: argparse.Namespace) -> None:
    from strategies.multi_ticker_swing.config.pipeline_config import UNIVERSE_CSV
    from strategies.multi_ticker_swing.features.build_features import build_all_features

    csv = Path(args.universe) if args.universe else UNIVERSE_CSV
    logger.info("=== STAGE: features ===")
    build_all_features(universe_csv=csv, force=args.force)
    logger.info("=== features complete ===")


def run_labels(args: argparse.Namespace) -> None:
    from strategies.multi_ticker_swing.config.pipeline_config import UNIVERSE_CSV
    from strategies.multi_ticker_swing.labels.build_labels import build_all_labels

    csv = Path(args.universe) if args.universe else UNIVERSE_CSV
    logger.info("=== STAGE: labels ===")
    build_all_labels(universe_csv=csv, force=args.force)
    logger.info("=== labels complete ===")


def run_matrix(args: argparse.Namespace) -> None:
    from strategies.multi_ticker_swing.labels.build_labels import build_training_matrix

    logger.info("=== STAGE: matrix ===")
    result = build_training_matrix(force=args.force)
    if result is not None:
        logger.info("Training matrix built: %d rows", len(result))
    logger.info("=== matrix complete ===")


# ---------------------------------------------------------------------------
# Post-pass verification
# ---------------------------------------------------------------------------

def verify_outputs(stage: str) -> None:
    """Check expected output files exist after each stage."""
    from strategies.multi_ticker_swing.config.pipeline_config import (
        FEATURES_COMBINED,
        LABELS_COMBINED,
        TRAINING_MATRIX,
    )

    checks: list[tuple[str, Path]] = []

    if stage in ("features", "all"):
        checks.append(("features_30m.parquet", FEATURES_COMBINED))
    if stage in ("labels", "all"):
        checks.append(("labels_30m.parquet", LABELS_COMBINED))
    if stage in ("matrix", "all"):
        checks.append(("training_matrix.parquet", TRAINING_MATRIX))

    issues: list[str] = []
    for name, path in checks:
        if not path.exists():
            issues.append(f"  MISSING: {name} → {path}")

    if issues:
        logger.warning("Post-pass verification — some outputs are missing:")
        for msg in issues:
            logger.warning(msg)
    else:
        logger.info("Post-pass verification passed — all expected outputs present.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = _parse_args()

    stage_map = {
        "fetch":    run_fetch,
        "features": run_features,
        "labels":   run_labels,
        "matrix":   run_matrix,
    }

    if args.stage == "all":
        ordered = ["fetch", "features", "labels", "matrix"]
    else:
        ordered = [args.stage]

    for s in ordered:
        try:
            stage_map[s](args)
        except Exception as exc:
            logger.error("Stage '%s' failed: %s", s, exc, exc_info=True)
            return 1

    verify_outputs(args.stage)
    return 0


if __name__ == "__main__":
    sys.exit(main())
