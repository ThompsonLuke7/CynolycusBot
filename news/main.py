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
    classify_existing_news,
    cluster_news_embeddings,
    collect_company_news,
    collect_news_from_csv,
    refine_news_records_from_clusters,
)
from news.scoring import build_news_similarity_scores


def _read_table(path: str) -> pd.DataFrame:
    p = Path(path)
    return pd.read_parquet(p) if p.suffix.lower() == ".parquet" else pd.read_csv(p)


def main() -> int:
    parser = argparse.ArgumentParser(description="Unscheduled catalyst news pipeline.")
    parser.add_argument("--stage", choices=["collect", "collect-sec-history", "sec-full-text", "enrich-ex99", "classify", "earnings", "embed", "cluster", "refine", "label", "score", "features"], required=True)
    parser.add_argument("--tickers", nargs="*", default=[])
    parser.add_argument(
        "--sources",
        nargs="*",
        default=[
            "finnhub",
            "sec_8k",
            "sec_alpha",
            "yfinance",
            "google_news",
            "fed_rss",
            "openfda",
            "clinicaltrials",
        ],
        help="Source list. Optional add-ons: yf_options_flow (daily-only), fmp_transcripts (needs FMP_API_KEY).",
    )
    parser.add_argument("--forms", nargs="*", default=["8-K", "10-Q", "10-K"])
    parser.add_argument("--use-backtest-universe", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--input-csv", default=None)
    parser.add_argument("--timestamps-csv", default=None)
    parser.add_argument("--bars-csv", default=None)
    parser.add_argument("--bars-per-day", type=int, default=None, help="Defaults to auto-detected from bar timestamps")
    parser.add_argument("--sec-no-archives", action="store_true")
    parser.add_argument("--sec-full-text-limit", type=int, default=0)
    parser.add_argument("--sec-alpha-forms", action="store_true")
    parser.add_argument("--no-embeddings", action="store_true")
    parser.add_argument("--no-finbert", action="store_true")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    if args.stage in {"collect", "collect-sec-history"}:
        if args.input_csv:
            collect_news_from_csv(args.input_csv, output_path=Path(args.output) if args.output else NEWS_RECORDS_PATH)
        else:
            tickers = list(args.tickers)
            if args.use_backtest_universe:
                from meta_context.backtest_inputs import build_context_backtest_universe

                tickers = build_context_backtest_universe(limit=args.limit)["ticker"].astype(str).tolist()
            if args.limit:
                tickers = tickers[: int(args.limit)]
            if not tickers or not args.start or not args.end:
                raise ValueError("--tickers or --use-backtest-universe, plus --start and --end, are required for API collection")
            output_path = Path(args.output) if args.output else NEWS_RECORDS_PATH
            if args.stage == "collect-sec-history":
                from news.sec_history import collect_sec_history_resumable

                collect_sec_history_resumable(
                    tickers,
                    start=args.start,
                    end=args.end,
                    output_path=output_path,
                    full_text_limit=args.sec_full_text_limit,
                    alpha_forms=args.sec_alpha_forms,
                )
            else:
                collect_company_news(
                    tickers,
                    start=args.start,
                    end=args.end,
                    sources=args.sources,
                    sec_include_archives=not args.sec_no_archives,
                    sec_full_text_limit=args.sec_full_text_limit,
                    output_path=output_path,
                )
    elif args.stage == "sec-full-text":
        from news.sec_text import backfill_sec_full_text

        backfill_sec_full_text(
            NEWS_RECORDS_PATH,
            output_path=Path(args.output) if args.output else NEWS_RECORDS_PATH,
            forms=args.forms,
            full_text_limit=args.sec_full_text_limit or 20000,
            limit=args.limit,
        )
    elif args.stage == "enrich-ex99":
        from news.sources import enrich_sec_8k_ex99_text

        existing = pd.read_parquet(NEWS_RECORDS_PATH)
        enriched = enrich_sec_8k_ex99_text(existing)
        enriched.to_parquet(
            Path(args.output) if args.output else NEWS_RECORDS_PATH,
            index=False,
        )
        print(f"EX-99 enriched bodies: {enriched.attrs.get('ex99_enriched_count', 0)} of {len(enriched)} records")
    elif args.stage in {"classify", "earnings"}:
        classify_existing_news()
    elif args.stage == "embed":
        build_news_embeddings(generate_embeddings=not args.no_embeddings, generate_finbert=not args.no_finbert)
    elif args.stage == "cluster":
        cluster_news_embeddings()
    elif args.stage == "refine":
        refine_news_records_from_clusters()
    elif args.stage == "label":
        from news.pipeline import label_news_forward_returns

        if args.bars_csv:
            bars = _read_table(args.bars_csv)
        else:
            from meta_context.config import CONTEXT_BACKTEST_UNIVERSE_PATH
            from meta_context.backtest_inputs import build_context_backtest_universe, load_cached_30m_bars_for_universe

            universe = pd.read_csv(CONTEXT_BACKTEST_UNIVERSE_PATH) if CONTEXT_BACKTEST_UNIVERSE_PATH.exists() else build_context_backtest_universe(limit=args.limit)
            if args.limit:
                universe = universe.head(int(args.limit)).copy()
            bars = load_cached_30m_bars_for_universe(universe)
        label_news_forward_returns(NEWS_RECORDS_PATH, bars, bars_per_day=args.bars_per_day)
        build_winner_loser_libraries()
    elif args.stage == "score":
        build_news_similarity_scores()
    else:
        if not args.timestamps_csv:
            raise ValueError("--timestamps-csv is required for --stage features")
        timestamps = _read_table(args.timestamps_csv)
        build_news_features(timestamps, output_path=Path(args.output) if args.output else NEWS_FEATURE_MATRIX_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
