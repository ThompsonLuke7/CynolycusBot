"""CLI for unscheduled catalyst news features."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from signals.news.config import (
    CBOE_OPTIONS_SUMMARY_PATH,
    EARNINGS_CALENDAR_PATH,
    ECONOMIC_CALENDAR_PATH,
    FINRA_SHORT_VOLUME_PATH,
    NASDAQ_SHORT_INTEREST_PATH,
    NEWS_FEATURE_MATRIX_PATH,
    NEWS_RECORDS_PATH,
    TICKER_PROFILE_PATH,
    USASPENDING_CONTRACTS_PATH,
)
from signals.news.pipeline import (
    build_news_embeddings,
    build_news_features,
    build_winner_loser_libraries,
    classify_existing_news,
    cluster_news_embeddings,
    collect_company_news,
    collect_news_from_csv,
    refine_news_records_from_clusters,
)
from signals.news.scoring import build_news_similarity_scores


def _read_table(path: str) -> pd.DataFrame:
    p = Path(path)
    return pd.read_parquet(p) if p.suffix.lower() == ".parquet" else pd.read_csv(p)


def _load_context_universe(limit: int | None = None) -> pd.DataFrame:
    from signals.meta_context.config import CONTEXT_BACKTEST_UNIVERSE_PATH
    from signals.meta_context.backtest_inputs import build_context_backtest_universe

    shared_path = Path("Data/shared/universe/shared_universe.csv")
    if shared_path.exists():
        universe = pd.read_csv(shared_path)
        if "is_eligible" in universe.columns:
            universe = universe[universe["is_eligible"].astype(bool)].copy()
    elif CONTEXT_BACKTEST_UNIVERSE_PATH.exists():
        universe = pd.read_csv(CONTEXT_BACKTEST_UNIVERSE_PATH)
    else:
        universe = build_context_backtest_universe(limit=limit)
    if limit:
        universe = universe.head(int(limit)).copy()
    return universe


def main() -> int:
    parser = argparse.ArgumentParser(description="Unscheduled catalyst news pipeline.")
    parser.add_argument("--stage", choices=["collect", "collect-sec-history", "sec-full-text", "sec-mda", "enrich-ex99", "classify", "earnings", "embed", "cluster", "refine", "label", "score", "features", "profiles", "finra-backfill", "finra-spikes", "cboe-snapshot", "scrape-bodies", "incremental", "earnings-calendar", "economic-calendar", "nasdaq-short-interest", "usaspending-contracts"], required=True)
    parser.add_argument("--full-refit", action="store_true", help="Force a full re-embed / re-cluster (default is incremental). Use weekly.")
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
    parser.add_argument(
        "--ticker-timeout",
        type=float,
        default=20.0,
        help="Hard per-ticker timeout in seconds for isolated network stages such as earnings-calendar.",
    )
    args = parser.parse_args()

    if args.stage in {"collect", "collect-sec-history"}:
        if args.input_csv:
            collect_news_from_csv(args.input_csv, output_path=Path(args.output) if args.output else NEWS_RECORDS_PATH)
        else:
            tickers = list(args.tickers)
            if args.use_backtest_universe:
                from signals.meta_context.backtest_inputs import build_context_backtest_universe

                tickers = build_context_backtest_universe(limit=args.limit)["ticker"].astype(str).tolist()
            if args.limit:
                tickers = tickers[: int(args.limit)]
            if not tickers or not args.start or not args.end:
                raise ValueError("--tickers or --use-backtest-universe, plus --start and --end, are required for API collection")
            output_path = Path(args.output) if args.output else NEWS_RECORDS_PATH
            if args.stage == "collect-sec-history":
                from signals.news.sec_history import collect_sec_history_resumable

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
        from signals.news.sec_text import backfill_sec_full_text

        backfill_sec_full_text(
            NEWS_RECORDS_PATH,
            output_path=Path(args.output) if args.output else NEWS_RECORDS_PATH,
            forms=args.forms,
            full_text_limit=args.sec_full_text_limit or 20000,
            limit=args.limit,
        )
    elif args.stage == "sec-mda":
        from signals.news.sec_text import backfill_sec_mda

        backfill_sec_mda(
            NEWS_RECORDS_PATH,
            output_path=Path(args.output) if args.output else NEWS_RECORDS_PATH,
            forms=args.forms,
            limit=args.limit,
        )
    elif args.stage == "scrape-bodies":
        from signals.news.body_scraper import backfill_google_news_bodies

        backfill_google_news_bodies(
            NEWS_RECORDS_PATH,
            output_path=Path(args.output) if args.output else NEWS_RECORDS_PATH,
            limit=args.limit,
        )
    elif args.stage == "enrich-ex99":
        from signals.news.sources import enrich_sec_8k_ex99_text

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
        build_news_embeddings(
            generate_embeddings=not args.no_embeddings,
            generate_finbert=not args.no_finbert,
            incremental=not args.full_refit,
        )
    elif args.stage == "cluster":
        cluster_news_embeddings(incremental=not args.full_refit)
    elif args.stage == "refine":
        refine_news_records_from_clusters()
    elif args.stage == "label":
        from signals.news.pipeline import label_news_forward_returns

        if args.bars_csv:
            bars = _read_table(args.bars_csv)
        else:
            from signals.meta_context.backtest_inputs import load_cached_30m_bars_for_universe

            universe = _load_context_universe(args.limit)
            bars = load_cached_30m_bars_for_universe(universe)
        label_news_forward_returns(NEWS_RECORDS_PATH, bars, bars_per_day=args.bars_per_day, incremental=not args.full_refit)
        build_winner_loser_libraries()
    elif args.stage == "incremental":
        # Lightweight nightly pass: embed only new records, predict-assign their
        # clusters using saved KMeans, re-apply refine, label only records whose
        # forward window has matured, append libraries.
        from signals.news.pipeline import label_news_forward_returns
        from signals.meta_context.backtest_inputs import load_cached_30m_bars_for_universe

        print("=== incremental update ===")
        print("[1/4] embed (incremental)")
        build_news_embeddings(generate_embeddings=True, generate_finbert=True, incremental=True)
        print("[2/4] cluster (incremental — uses saved KMeans models)")
        cluster_news_embeddings(incremental=True)
        print("[3/4] refine (full re-apply; cheap)")
        refine_news_records_from_clusters()
        print("[4/4] label + winner/loser libraries (incremental)")
        universe = _load_context_universe(args.limit)
        bars = load_cached_30m_bars_for_universe(universe)
        label_news_forward_returns(NEWS_RECORDS_PATH, bars, bars_per_day=args.bars_per_day, incremental=True)
        build_winner_loser_libraries()
        print("=== incremental update complete ===")
    elif args.stage == "score":
        build_news_similarity_scores()
    elif args.stage == "profiles":
        from signals.news.sources import fetch_yfinance_profiles

        if args.tickers:
            tickers = list(args.tickers)
        else:
            universe = _load_context_universe(args.limit)
            tickers = universe["ticker"].astype(str).tolist()
        out_path = Path(args.output) if args.output else TICKER_PROFILE_PATH
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df = fetch_yfinance_profiles(tickers)
        df.to_parquet(out_path, index=False)
        print(f"yfinance profiles: {len(df)} rows -> {out_path}")
    elif args.stage == "finra-backfill":
        from signals.news.sources import backfill_finra_short_volume

        if not args.start or not args.end:
            raise ValueError("--start and --end required for finra-backfill")
        out_path = Path(args.output) if args.output else FINRA_SHORT_VOLUME_PATH
        df = backfill_finra_short_volume(start=args.start, end=args.end, output_path=out_path)
        print(f"finra short-volume backfill: {len(df)} rows -> {out_path}")
    elif args.stage == "finra-spikes":
        from signals.news.sources import emit_finra_short_spike_records

        records = emit_finra_short_spike_records(FINRA_SHORT_VOLUME_PATH)
        records.to_parquet(Path(args.output) if args.output else "signals/news/data/processed/finra_short_spike_records.parquet", index=False)
        print(f"finra short-volume spike records: {len(records)}")
    elif args.stage == "cboe-snapshot":
        from signals.news.sources import fetch_cboe_options_snapshot

        if args.tickers:
            tickers = list(args.tickers)
        else:
            universe = _load_context_universe(args.limit)
            tickers = universe["ticker"].astype(str).tolist()
        summary_path = Path(args.output) if args.output else CBOE_OPTIONS_SUMMARY_PATH
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_df, records_df, strike_df = fetch_cboe_options_snapshot(tickers)
        # Append to historical summary if it exists; one row per ticker per day
        if summary_path.exists():
            try:
                prior = pd.read_parquet(summary_path)
                if not prior.empty and not summary_df.empty:
                    summary_df = pd.concat([prior, summary_df], ignore_index=True)
                    summary_df = summary_df.drop_duplicates(["ticker", "snapshot_date"], keep="last")
            except Exception:
                pass
        summary_df.to_parquet(summary_path, index=False)
        records_path = Path("signals/news/data/processed/cboe_unusual_records.parquet")
        if records_path.exists():
            try:
                prior_records = pd.read_parquet(records_path)
                if not prior_records.empty and not records_df.empty:
                    records_df = pd.concat([prior_records, records_df], ignore_index=True)
                    records_df = records_df.drop_duplicates(["record_id"], keep="last")
                elif records_df.empty:
                    records_df = prior_records
            except Exception:
                pass
        records_df.to_parquet(records_path, index=False)
        strikes_path = Path("signals/news/data/processed/cboe_unusual_strikes.parquet")
        if strikes_path.exists():
            try:
                prior_strikes = pd.read_parquet(strikes_path)
                if not prior_strikes.empty and not strike_df.empty:
                    strike_df = pd.concat([prior_strikes, strike_df], ignore_index=True)
                    strike_df = strike_df.drop_duplicates(["ticker", "snapshot_date", "contract"], keep="last")
                elif strike_df.empty:
                    strike_df = prior_strikes
            except Exception:
                pass
        strike_df.to_parquet(strikes_path, index=False)
        print(f"cboe snapshot: {len(summary_df)} summary rows -> {summary_path}")
        print(f"cboe unusual catalyst records: {len(records_df)} -> {records_path}")
        print(f"cboe unusual strike rows: {len(strike_df)} -> {strikes_path}")
    elif args.stage == "earnings-calendar":
        from signals.news.sources import fetch_earnings_calendar

        if args.tickers:
            tickers = list(args.tickers)
        else:
            universe = _load_context_universe(args.limit)
            # Yahoo returns expected quote-summary 404s for ETFs and funds.
            # Keep unknown types because most discovered equities have no type
            # metadata yet; exclude only rows explicitly identified as non-stock.
            if "type" in universe.columns:
                known_type = universe["type"].fillna("").astype(str).str.casefold()
                universe = universe[~known_type.isin({"etf", "fund"})].copy()
            tickers = universe["ticker"].astype(str).tolist()
        out_path = Path(args.output) if args.output else EARNINGS_CALENDAR_PATH
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df = fetch_earnings_calendar(tickers, request_timeout_s=args.ticker_timeout)
        if out_path.exists():
            try:
                prior = pd.read_parquet(out_path)
                df = pd.concat([prior, df], ignore_index=True).drop_duplicates(["ticker", "snapshot_date"], keep="last")
            except Exception:
                pass
        df.to_parquet(out_path, index=False)
        print(f"earnings-calendar: {len(df)} rows -> {out_path}")
    elif args.stage == "economic-calendar":
        from signals.news.sources import fetch_economic_calendar

        out_path = Path(args.output) if args.output else ECONOMIC_CALENDAR_PATH
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df = fetch_economic_calendar(start=args.start, end=args.end)
        if out_path.exists():
            try:
                prior = pd.read_parquet(out_path)
                df = pd.concat([prior, df], ignore_index=True).drop_duplicates(["event", "event_date"], keep="last")
            except Exception:
                pass
        df.to_parquet(out_path, index=False)
        print(f"economic-calendar: {len(df)} rows -> {out_path}")
    elif args.stage == "nasdaq-short-interest":
        from signals.news.sources import fetch_nasdaq_short_interest

        if args.tickers:
            tickers = list(args.tickers)
        else:
            universe = _load_context_universe(args.limit)
            tickers = universe["ticker"].astype(str).tolist()
        out_path = Path(args.output) if args.output else NASDAQ_SHORT_INTEREST_PATH
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df = fetch_nasdaq_short_interest(tickers)
        if out_path.exists():
            try:
                prior = pd.read_parquet(out_path)
                df = pd.concat([prior, df], ignore_index=True).drop_duplicates(["ticker", "settlement_date"], keep="last")
            except Exception:
                pass
        df.to_parquet(out_path, index=False)
        print(f"nasdaq-short-interest: {len(df)} rows -> {out_path}")
    elif args.stage == "usaspending-contracts":
        from signals.news.sources import fetch_usaspending_contracts

        tickers = list(args.tickers) if args.tickers else None
        out_path = Path(args.output) if args.output else USASPENDING_CONTRACTS_PATH
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df = fetch_usaspending_contracts(tickers, start=args.start, end=args.end)
        if out_path.exists():
            try:
                prior = pd.read_parquet(out_path)
                df = pd.concat([prior, df], ignore_index=True).drop_duplicates(["award_id"], keep="last")
            except Exception:
                pass
        df.to_parquet(out_path, index=False)
        print(f"usaspending-contracts: {len(df)} rows -> {out_path}")
    else:
        if not args.timestamps_csv:
            raise ValueError("--timestamps-csv is required for --stage features")
        timestamps = _read_table(args.timestamps_csv)
        build_news_features(timestamps, output_path=Path(args.output) if args.output else NEWS_FEATURE_MATRIX_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
