"""CLI for unscheduled catalyst news features."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from news.config import NEWS_FEATURE_MATRIX_PATH, NEWS_RECORDS_PATH
from news.pipeline import (
    build_news_embeddings,
    build_news_features,
    build_winner_loser_libraries,
    cluster_news_embeddings,
    collect_company_news,
    collect_news_from_csv,
)


def _read_table(path: str) -> pd.DataFrame:
    p = Path(path)
    return pd.read_parquet(p) if p.suffix.lower() == ".parquet" else pd.read_csv(p)


def main() -> int:
    parser = argparse.ArgumentParser(description="Unscheduled catalyst news pipeline.")
    parser.add_argument("--stage", choices=["collect", "embed", "cluster", "label", "features"], required=True)
    parser.add_argument("--tickers", nargs="*", default=[])
    parser.add_argument("--use-backtest-universe", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--input-csv", default=None)
    parser.add_argument("--timestamps-csv", default=None)
    parser.add_argument("--bars-csv", default=None)
    parser.add_argument("--bars-per-day", type=int, default=13)
    parser.add_argument("--no-embeddings", action="store_true")
    parser.add_argument("--no-finbert", action="store_true")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    if args.stage == "collect":
        if args.input_csv:
            collect_news_from_csv(args.input_csv)
        else:
            tickers = list(args.tickers)
            if args.use_backtest_universe:
                from meta_context.backtest_inputs import build_context_backtest_universe

                tickers = build_context_backtest_universe()["ticker"].astype(str).tolist()
            if args.limit:
                tickers = tickers[: int(args.limit)]
            if not tickers or not args.start or not args.end:
                raise ValueError("--tickers or --use-backtest-universe, plus --start and --end, are required for API collection")
            collect_company_news(tickers, start=args.start, end=args.end)
    elif args.stage == "embed":
        build_news_embeddings(generate_embeddings=not args.no_embeddings, generate_finbert=not args.no_finbert)
    elif args.stage == "cluster":
        cluster_news_embeddings()
    elif args.stage == "label":
        from news.pipeline import label_news_forward_returns

        if args.bars_csv:
            bars = _read_table(args.bars_csv)
        else:
            from meta_context.backtest_inputs import build_context_backtest_universe, load_cached_30m_bars_for_universe

            universe = build_context_backtest_universe()
            bars = load_cached_30m_bars_for_universe(universe)
        label_news_forward_returns(NEWS_RECORDS_PATH, bars, bars_per_day=args.bars_per_day)
        build_winner_loser_libraries()
    else:
        if not args.timestamps_csv:
            raise ValueError("--timestamps-csv is required for --stage features")
        timestamps = _read_table(args.timestamps_csv)
        build_news_features(timestamps, output_path=Path(args.output) if args.output else NEWS_FEATURE_MATRIX_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
