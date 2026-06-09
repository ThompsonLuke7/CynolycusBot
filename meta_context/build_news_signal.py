"""Build the per-(ticker, timestamp) news catalyst signal that feeds the
meta-context matrix.

Loads all records from news_records.parquet, runs both the binary expansion
classifier AND the multiclass trajectory classifier on each, then aggregates
to a per-(ticker, snapshot_date) summary with:

  Binary (expansion) outputs:
    news_catalyst_score          max binary score across day
    news_catalyst_score_mean     mean binary score across day
    news_catalyst_count          # of records that day
    news_catalyst_top_family     family of the highest-scoring record
    news_catalyst_top_subtype    subtype of the highest-scoring record

  Trajectory (multiclass) outputs — max P() across the day:
    news_p_bull_steady           "clean breakout" probability
    news_p_bull_volatile         "move with shakeout" probability
    news_p_v_bounce              "spike then fade" probability
    news_p_crash_stayed          "real bear catalyst" probability
    news_p_flat                  "nothing happens" probability

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
from news.live_scorer import CatalystScorer, TRAJECTORY_LABELS


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
    has_traj = scorer.trajectory_booster is not None
    print(f"binary classifier loaded; trajectory classifier loaded: {has_traj}")

    print("scoring in batches...")
    scores: list[np.ndarray] = []
    traj_chunks: list[pd.DataFrame] = []
    for start in range(0, len(df), batch_size):
        batch = df.iloc[start:start + batch_size].copy()
        cols = ["ticker", "timestamp", "headline", "summary", "body", "source"]
        if "earnings_forward_guidance_text" in df.columns:
            cols.append("earnings_forward_guidance_text")
        batch_records = batch[cols].to_dict(orient="records")
        s = scorer.score(batch_records)
        scores.append(s)
        if has_traj:
            traj_chunks.append(scorer.score_trajectory(batch_records))
        if (start // batch_size + 1) % 10 == 0:
            print(f"  scored {start + len(batch):,} / {len(df):,}", flush=True)

    df["catalyst_score"] = np.concatenate(scores)
    if has_traj:
        traj_df = pd.concat(traj_chunks, ignore_index=True)
        for col in TRAJECTORY_LABELS:
            df[f"p_{col}"] = traj_df[col].values

    # Aggregate per (ticker, day)
    print("aggregating to per-(ticker, day)...")
    df["snapshot_date"] = df["timestamp"].dt.tz_convert("UTC").dt.tz_localize(None).dt.floor("D")
    df_sorted = df.sort_values("catalyst_score", ascending=False)
    agg_kwargs = dict(
        news_catalyst_score=("catalyst_score", "max"),
        news_catalyst_score_mean=("catalyst_score", "mean"),
        news_catalyst_count=("catalyst_score", "size"),
        news_catalyst_top_family=("catalyst_family", "first"),
        news_catalyst_top_subtype=("catalyst_subtype", "first"),
    )
    if has_traj:
        for col in TRAJECTORY_LABELS:
            agg_kwargs[f"news_p_{col}"] = (f"p_{col}", "max")
            agg_kwargs[f"news_p_{col}_mean"] = (f"p_{col}", "mean")
    agg = df_sorted.groupby(["ticker", "snapshot_date"]).agg(**agg_kwargs).reset_index()

    # Pivot to (timestamp, ticker) schema compatible with meta_context.build_matrix
    agg["timestamp"] = pd.to_datetime(agg["snapshot_date"], utc=True)
    keep_cols = [
        "timestamp",
        "ticker",
        "news_catalyst_score",
        "news_catalyst_score_mean",
        "news_catalyst_count",
        "news_catalyst_top_family",
        "news_catalyst_top_subtype",
    ]
    if has_traj:
        for col in TRAJECTORY_LABELS:
            keep_cols += [f"news_p_{col}", f"news_p_{col}_mean"]
    out = agg[keep_cols]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_path, index=False)
    print(f"saved -> {output_path}: {len(out):,} (ticker, day) rows, {len(out.columns)} cols")
    print(f"score distribution: min={out['news_catalyst_score'].min():.3f} median={out['news_catalyst_score'].median():.3f} max={out['news_catalyst_score'].max():.3f}")
    if has_traj:
        print("trajectory probability distributions (max-per-day):")
        for col in TRAJECTORY_LABELS:
            c = f"news_p_{col}"
            print(f"  {c:<28} min={out[c].min():.3f}  median={out[c].median():.3f}  max={out[c].max():.3f}")
    return out


if __name__ == "__main__":
    build()
