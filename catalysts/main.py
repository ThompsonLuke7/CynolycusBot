"""CLI for the unified catalyst layer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from catalysts.config import CATALYST_FEATURE_MATRIX_PATH
from catalysts.pipeline import build_catalyst_features, build_catalyst_records, build_catalyst_scores


def _read_table(path: str) -> pd.DataFrame:
    p = Path(path)
    return pd.read_parquet(p) if p.suffix.lower() == ".parquet" else pd.read_csv(p)


def main() -> int:
    parser = argparse.ArgumentParser(description="Unified catalyst pipeline.")
    parser.add_argument("--stage", choices=["records", "scores", "features", "all", "score-live", "correlation-report"], required=True)
    parser.add_argument("--timestamps-csv", default=None)
    parser.add_argument("--output", default=None)
    # score-live options
    parser.add_argument("--ticker", default=None, help="Ticker symbol for score-live")
    parser.add_argument("--headline", default=None, help="Headline text for score-live")
    parser.add_argument("--summary", default="", help="Optional summary text for score-live")
    parser.add_argument("--body", default="", help="Optional body text for score-live")
    parser.add_argument("--source", default="finnhub", help="News source label (finnhub / sec_8-k / ...)")
    parser.add_argument("--url", default="", help="URL for the news record (optional)")
    parser.add_argument("--timestamp", default=None, help="Timestamp for the news (ISO 8601); defaults to now")
    parser.add_argument("--min-similarity", type=float, default=None, help="Override min cosine similarity (default 0.55)")
    parser.add_argument("--top-k", type=int, default=None, help="Override neighbor count for the lookup")
    args = parser.parse_args()

    if args.stage in {"records", "all"}:
        build_catalyst_records()
    if args.stage in {"scores", "all"}:
        build_catalyst_scores()
    if args.stage in {"features", "all"}:
        if not args.timestamps_csv:
            raise ValueError("--timestamps-csv is required for catalyst features")
        timestamps = _read_table(args.timestamps_csv)
        build_catalyst_features(
            timestamps,
            output_path=Path(args.output) if args.output else CATALYST_FEATURE_MATRIX_PATH,
        )
    if args.stage == "correlation-report":
        from news.correlation_report import build_correlation_report

        build_correlation_report()
        print("correlation report written to news/data/processed/correlation_report/")
    if args.stage == "score-live":
        if not args.ticker or not args.headline:
            raise ValueError("--ticker and --headline are required for score-live")
        from news.live_scoring import LiveScoringConfig, score_headline

        config = LiveScoringConfig()
        if args.min_similarity is not None:
            config.min_similarity = float(args.min_similarity)
        if args.top_k is not None:
            config.top_k = int(args.top_k)
        result = score_headline(
            ticker=args.ticker,
            timestamp=args.timestamp,
            headline=args.headline,
            summary=args.summary,
            body=args.body,
            source=args.source,
            url=args.url,
            config=config,
        )
        if args.output:
            Path(args.output).write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
        else:
            print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
