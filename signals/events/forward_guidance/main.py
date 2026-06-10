"""CLI entry point for the forward-guidance earnings pipeline.

Examples:
    python -m events.forward_guidance.main --stage discover-events --start 2025-01-01 --end 2026-02-01
    python -m events.forward_guidance.main --stage ingest --events-csv events.csv
    python -m events.forward_guidance.main --stage fetch-market --events-csv events.csv --limit 5
    python -m events.forward_guidance.main --stage features --events-csv events.csv
    python -m events.forward_guidance.main --stage train --model-kind xgboost
    python -m events.forward_guidance.main --stage inference --events-csv today_events.csv
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from signals.events.forward_guidance.config import EVENTS_PATH, ensure_data_dirs
from signals.events.forward_guidance.data.ingest_events import ingest_event, load_events, load_events_from_csv, write_events
from signals.events.forward_guidance.data.market_data import fetch_event_market_window
from signals.events.forward_guidance.data.schema import EarningsEvent, event_from_record


logger = logging.getLogger(__name__)


def _setup_logging(level: str) -> None:
    logging.basicConfig(level=getattr(logging, level.upper()), format="%(asctime)s %(levelname)s %(name)s %(message)s")


def _events_from_args(args: argparse.Namespace) -> list[EarningsEvent]:
    if args.events_csv:
        return load_events_from_csv(args.events_csv)
    df = load_events(EVENTS_PATH)
    if df.empty:
        raise FileNotFoundError("No events available. Pass --events-csv or run --stage ingest first.")
    return [event_from_record(row) for _, row in df.iterrows()]


def _limit(events: list[EarningsEvent], limit: int | None) -> list[EarningsEvent]:
    return events[: int(limit)] if limit else events


def stage_ingest(args: argparse.Namespace) -> None:
    events = _limit(_events_from_args(args), args.limit)
    write_events(events)
    for event in events:
        ingest_event(event, force=args.force)
    logger.info("Ingested %d events", len(events))


def stage_discover_events(args: argparse.Namespace) -> None:
    from signals.events.forward_guidance.config import DISCOVERED_EVENTS_CSV
    from signals.events.forward_guidance.data.discover_events import discover_events, write_discovered_events

    if not args.start or not args.end:
        raise ValueError("--start and --end are required for --stage discover-events")
    events = discover_events(
        start=args.start,
        end=args.end,
        source=args.discovery_source,
        tickers=args.tickers,
        limit=args.limit,
        include_funds=args.include_funds,
    )
    output = Path(args.output) if args.output else DISCOVERED_EVENTS_CSV
    df = write_discovered_events(events, csv_path=output)
    logger.info("Discovered %d events -> %s", len(df), output)


def stage_fetch_market(args: argparse.Namespace) -> None:
    events = _limit(_events_from_args(args), args.limit)
    for event in events:
        fetch_event_market_window(event, force=args.force, timeframe=args.timeframe, feed=args.feed)
    logger.info("Fetched market windows for %d events", len(events))


def stage_features(args: argparse.Namespace) -> None:
    events = _limit(_events_from_args(args), args.limit)
    from signals.events.forward_guidance.features.build_matrix import build_feature_matrix

    features, labels, matrix = build_feature_matrix(
        events,
        force=args.force,
        generate_embeddings=args.embeddings,
        generate_finbert=args.finbert,
    )
    logger.info("Built features=%d labels=%d training_rows=%d", len(features), len(labels), len(matrix))


def stage_train(args: argparse.Namespace) -> None:
    from signals.events.forward_guidance.models.train import train_models

    metrics = train_models(model_kind=args.model_kind, force=args.force)
    logger.info("Training metrics: %s", metrics)


def stage_inference(args: argparse.Namespace) -> None:
    events = _limit(_events_from_args(args), args.limit)
    from signals.events.forward_guidance.inference.daily import score_events

    ranked = score_events(events, generate_embeddings=args.embeddings, generate_finbert=args.finbert, model_path=args.model_path)
    logger.info("Ranked %d opportunities", len(ranked))


def stage_backtest(args: argparse.Namespace) -> None:
    from signals.events.forward_guidance.backtests.simulate import run_backtest

    if not args.predictions:
        raise ValueError("--predictions is required for --stage backtest")
    path = Path(args.predictions)
    df = pd.read_csv(path) if path.suffix.lower() == ".csv" else pd.read_parquet(path)
    trades, summary = run_backtest(df)
    logger.info("Backtest trades=%d summary=%s", len(trades), summary)


def main() -> int:
    parser = argparse.ArgumentParser(description="Forward-guidance earnings trader V1 pipeline.")
    parser.add_argument(
        "--stage",
        choices=["discover-events", "ingest", "fetch-market", "features", "train", "inference", "backtest", "all"],
        default="features",
    )
    parser.add_argument("--events-csv", default=None, help="CSV with ticker, earnings_date, report_time, optional sector/cik fields.")
    parser.add_argument("--predictions", default=None, help="Prediction parquet/csv for backtesting.")
    parser.add_argument("--model-kind", choices=["xgboost", "lightgbm", "both"], default="xgboost")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--start", default=None, help="Discovery start date, e.g. 2025-01-01.")
    parser.add_argument("--end", default=None, help="Discovery end date, e.g. 2026-02-01.")
    parser.add_argument("--discovery-source", choices=["sec", "yfinance", "both"], default="both")
    parser.add_argument("--tickers", nargs="*", default=None, help="Optional ticker list for discovery.")
    parser.add_argument("--include-funds", action="store_true", help="Include ETFs/funds from the reusable universe during discovery.")
    parser.add_argument("--output", default=None, help="CSV output path for discovered events.")
    parser.add_argument("--timeframe", default="30Min")
    parser.add_argument("--feed", default="IEX")
    parser.add_argument("--limit", type=int, default=None, help="Cap events for smoke runs/backfills.")
    parser.add_argument("--embeddings", action="store_true", help="Generate sentence-transformer embeddings.")
    parser.add_argument("--finbert", action="store_true", help="Generate FinBERT tone features.")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--log", default="INFO")
    args = parser.parse_args()
    _setup_logging(args.log)
    ensure_data_dirs()

    if args.stage == "all":
        for stage in ("ingest", "fetch-market", "features"):
            setattr(args, "stage", stage)
            {"ingest": stage_ingest, "fetch-market": stage_fetch_market, "features": stage_features}[stage](args)
        return 0

    stages = {
        "discover-events": stage_discover_events,
        "ingest": stage_ingest,
        "fetch-market": stage_fetch_market,
        "features": stage_features,
        "train": stage_train,
        "inference": stage_inference,
        "backtest": stage_backtest,
    }
    stages[args.stage](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
