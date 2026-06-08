"""Build the per-(ticker, timestamp) news_catalyst_score signal that feeds
the meta-context matrix.

Loads all records from news_records.parquet, runs the trained catalyst
classifier on each, and aggregates to a per-(ticker, snapshot_date) summary
with:
  - news_catalyst_score          max score across that day's records
  - news_catalyst_score_mean     mean across that day's records
  - news_catalyst_count_24h      count of records in trailing 24h
  - news_catalyst_top_family     family of the highest-scoring record
  - news_catalyst_top_subtype    subtype of the highest-scoring record

Output: meta_context/data/processed/news_catalyst_signal.parquet

This parquet is one of the inputs to ``meta_context.build_matrix.build_from_paths``
via the ``news_path=`` argument; the meta-ranker (Model 6) consumes it.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from news.config import NEWS_RECORDS_PATH
from news.live_scorer import CatalystScorer


OUTPUT_PATH = Path("meta_context/data/processed/news_catalyst_signal.parquet")


def build(
    records_path: Path = Path(NEWS_RECORDS_PATH),
    output_path: Path = OUTPUT_PATH,
    *,
    batch_size: int = 1024,
    use_finbert: bool = True,
) -> pd.DataFrame:
    if not records_path.exists():
        raise SystemExit(f"news_records not found at {records_path}")

    print(f"loading {records_path}...")
    df = pd.read_parquet(records_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp", "ticker"]).copy()
    df["ticker"] = df["ticker"].astype(str).str.upper()
    print(f"  records to score: {len(df):,}")

    scorer = CatalystScorer(use_finbert=use_finbert)

    print("scoring in batches...")
    scores: list[np.ndarray] = []
    for start in range(0, len(df), batch_size):
        batch = df.iloc[start:start + batch_size].copy()
        batch_records = batch[["ticker", "timestamp", "headline", "summary", "body",
                               "source", "earnings_forward_guidance_text"]].to_dict(orient="records") \
            if "earnings_forward_guidance_text" in df.columns else \
            batch[["ticker", "timestamp", "headline", "summary", "body", "source"]].to_dict(orient="records")
        s = scorer.score(batch_records)
        scores.append(s)
        if (start // batch_size + 1) % 10 == 0:
            print(f"  scored {start + len(batch):,} / {len(df):,}", flush=True)

    df["catalyst_score"] = np.concatenate(scores)

    # Aggregate per (ticker, day)
    print("aggregating to per-(ticker, day)...")
    df["snapshot_date"] = df["timestamp"].dt.tz_convert("UTC").dt.tz_localize(None).dt.floor("D")
    grp = df.sort_values("catalyst_score", ascending=False).groupby(["ticker", "snapshot_date"])
    agg = grp.agg(
        news_catalyst_score=("catalyst_score", "max"),
        news_catalyst_score_mean=("catalyst_score", "mean"),
        news_catalyst_count=("catalyst_score", "size"),
        news_catalyst_top_family=("catalyst_family", "first"),
        news_catalyst_top_subtype=("catalyst_subtype", "first"),
    ).reset_index()

    # Pivot to (timestamp, ticker) schema compatible with meta_context.build_matrix
    agg["timestamp"] = pd.to_datetime(agg["snapshot_date"], utc=True)
    out = agg[
        [
            "timestamp",
            "ticker",
            "news_catalyst_score",
            "news_catalyst_score_mean",
            "news_catalyst_count",
            "news_catalyst_top_family",
            "news_catalyst_top_subtype",
        ]
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_path, index=False)
    print(f"saved -> {output_path}: {len(out):,} (ticker, day) rows")
    print(f"score distribution: min={out['news_catalyst_score'].min():.3f} median={out['news_catalyst_score'].median():.3f} max={out['news_catalyst_score'].max():.3f}")
    return out


if __name__ == "__main__":
    build()
