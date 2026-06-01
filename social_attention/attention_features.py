"""Attention feature engineering for Reddit ticker mentions."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from social_attention.config import ATTENTION_FEATURES_PATH, REDDIT_MENTIONS_PATH, REDDIT_POSTS_PATH
from social_attention.io import read_table, write_table

logger = logging.getLogger(__name__)


def _clean_posts(posts: pd.DataFrame) -> pd.DataFrame:
    out = posts.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out["score"] = pd.to_numeric(out.get("praw_score", out.get("score", 0.0)), errors="coerce").fillna(
        pd.to_numeric(out.get("score", 0.0), errors="coerce")
    ).fillna(0.0)
    out["num_comments"] = pd.to_numeric(out.get("praw_num_comments", out.get("num_comments", 0.0)), errors="coerce").fillna(
        pd.to_numeric(out.get("num_comments", 0.0), errors="coerce")
    ).fillna(0.0)
    out["kind"] = out.get("kind", "").astype(str)
    out["author"] = out.get("author", "").astype(str)
    out["engagement"] = np.where(
        out["kind"].eq("submission"),
        np.log1p(out["score"].clip(lower=0.0)) + np.log1p(out["num_comments"].clip(lower=0.0)),
        np.log1p(out["score"].clip(lower=0.0)),
    )
    if "sentiment_score" not in out.columns:
        out["sentiment_score"] = np.nan
    return out.dropna(subset=["timestamp", "post_id"])


def _maybe_complete_hourly_grid(df: pd.DataFrame, max_grid_rows: int = 5_000_000) -> pd.DataFrame:
    if df.empty:
        return df
    tickers = sorted(df["ticker"].dropna().unique())
    start = df["timestamp_bucket"].min()
    end = df["timestamp_bucket"].max()
    hours = pd.date_range(start, end, freq="1h", tz="UTC")
    estimated = len(tickers) * len(hours)
    if estimated > max_grid_rows:
        logger.warning("Skipping dense hourly grid; estimated rows=%s exceeds max_grid_rows=%s", estimated, max_grid_rows)
        return df.sort_values(["ticker", "timestamp_bucket"]).reset_index(drop=True)
    idx = pd.MultiIndex.from_product([tickers, hours], names=["ticker", "timestamp_bucket"])
    dense = df.set_index(["ticker", "timestamp_bucket"]).reindex(idx).reset_index()
    fill_zero = [
        "mentions_1h",
        "submission_mentions_1h",
        "comment_mentions_1h",
        "unique_authors",
        "engagement_score",
        "avg_engagement",
        "sentiment_score",
    ]
    for col in fill_zero:
        if col in dense.columns:
            dense[col] = dense[col].fillna(0.0)
    return dense


def _same_hour_zscore(group: pd.DataFrame, window: int = 30) -> pd.Series:
    pieces = []
    for _, hour_group in group.groupby(group["timestamp_bucket"].dt.hour, sort=False):
        s = hour_group["mentions_1h"].astype(float)
        mean = s.shift(1).rolling(window, min_periods=5).mean()
        std = s.shift(1).rolling(window, min_periods=5).std().replace(0, np.nan)
        pieces.append(((s - mean) / std).rename("mention_zscore"))
    return pd.concat(pieces).sort_index()


def build_attention_features(
    posts: pd.DataFrame | None = None,
    mentions: pd.DataFrame | None = None,
    *,
    posts_path: Path | str = REDDIT_POSTS_PATH,
    mentions_path: Path | str = REDDIT_MENTIONS_PATH,
    output_path: Path | str = ATTENTION_FEATURES_PATH,
    complete_grid: bool = True,
) -> pd.DataFrame:
    posts = read_table(posts_path) if posts is None else posts.copy()
    mentions = read_table(mentions_path) if mentions is None else mentions.copy()
    if posts.empty or mentions.empty:
        out = pd.DataFrame()
        return write_table(out, output_path)

    posts = _clean_posts(posts)
    cur = mentions.merge(
        posts[["post_id", "timestamp", "kind", "author", "engagement", "sentiment_score"]],
        on="post_id",
        how="inner",
    )
    if cur.empty:
        return write_table(pd.DataFrame(), output_path)
    cur["ticker"] = cur["ticker"].astype(str).str.upper().str.replace("$", "", regex=False)
    cur["timestamp_bucket"] = cur["timestamp"].dt.floor("1h")
    cur["is_submission"] = cur["kind"].eq("submission").astype(float)
    cur["is_comment"] = cur["kind"].eq("comment").astype(float)

    bars = (
        cur.groupby(["ticker", "timestamp_bucket"], as_index=False)
        .agg(
            mentions_1h=("post_id", "count"),
            submission_mentions_1h=("is_submission", "sum"),
            comment_mentions_1h=("is_comment", "sum"),
            unique_authors=("author", "nunique"),
            engagement_score=("engagement", "sum"),
            avg_engagement=("engagement", "mean"),
            sentiment_score=("sentiment_score", "mean"),
        )
        .sort_values(["ticker", "timestamp_bucket"])
        .reset_index(drop=True)
    )
    if complete_grid:
        bars = _maybe_complete_hourly_grid(bars)

    bars = bars.sort_values(["ticker", "timestamp_bucket"]).reset_index(drop=True)
    grouped = bars.groupby("ticker", group_keys=False)
    bars["mentions_4h"] = grouped["mentions_1h"].transform(lambda s: s.rolling(4, min_periods=1).sum())
    bars["mentions_24h"] = grouped["mentions_1h"].transform(lambda s: s.rolling(24, min_periods=1).sum())
    bars["unique_authors_24h"] = grouped["unique_authors"].transform(lambda s: s.rolling(24, min_periods=1).sum())
    bars["engagement_24h"] = grouped["engagement_score"].transform(lambda s: s.rolling(24, min_periods=1).sum())
    baseline_4h = grouped["mentions_4h"].transform(lambda s: s.shift(1).rolling(20 * 24, min_periods=24).mean())
    bars["mention_acceleration"] = bars["mentions_4h"] / baseline_4h.replace(0, np.nan)
    bars["mention_zscore"] = np.nan
    for _, ticker_group in bars.groupby("ticker", sort=False):
        bars.loc[ticker_group.index, "mention_zscore"] = _same_hour_zscore(ticker_group)
    bars["rank_now"] = bars.groupby("timestamp_bucket")["mentions_1h"].rank(method="dense", ascending=False)
    prev_rank = bars[["ticker", "timestamp_bucket", "rank_now"]].copy()
    prev_rank["timestamp_bucket"] = prev_rank["timestamp_bucket"] + pd.Timedelta(hours=24)
    prev_rank = prev_rank.rename(columns={"rank_now": "rank_24h_ago"})
    bars = bars.merge(prev_rank, on=["ticker", "timestamp_bucket"], how="left")
    bars["rank_change_24h"] = bars["rank_24h_ago"] - bars["rank_now"]
    sentiment_4h = grouped["sentiment_score"].transform(lambda s: s.rolling(4, min_periods=1).mean())
    sentiment_24h = grouped["sentiment_score"].transform(lambda s: s.rolling(24, min_periods=1).mean())
    bars["sentiment_acceleration"] = sentiment_4h - sentiment_24h
    bars = bars.rename(columns={"timestamp_bucket": "timestamp"})
    return write_table(bars, output_path)
