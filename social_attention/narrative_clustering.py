"""Narrative discovery and narrative acceleration features."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from news.pipeline import parse_embedding
from social_attention.config import (
    NARRATIVE_CLUSTERS_PATH,
    NARRATIVE_FEATURES_PATH,
    REDDIT_MENTIONS_PATH,
    REDDIT_POSTS_PATH,
    SOCIAL_EMBEDDINGS_PATH,
)
from social_attention.io import read_table, write_table

STOPWORDS = {
    "about",
    "after",
    "again",
    "all",
    "also",
    "and",
    "are",
    "because",
    "but",
    "can",
    "for",
    "from",
    "have",
    "just",
    "like",
    "more",
    "not",
    "that",
    "the",
    "this",
    "with",
    "you",
    "your",
}
TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9]{2,}")


def _top_terms(texts: pd.Series, n: int = 8) -> str:
    counts: Counter[str] = Counter()
    for text in texts.fillna("").astype(str):
        for token in TOKEN_RE.findall(text.lower()):
            if token not in STOPWORDS and not token.isdigit():
                counts[token] += 1
    return " ".join(term for term, _ in counts.most_common(n))


def cluster_social_embeddings(
    embeddings: pd.DataFrame | None = None,
    *,
    embeddings_path: Path | str = SOCIAL_EMBEDDINGS_PATH,
    posts_path: Path | str = REDDIT_POSTS_PATH,
    output_path: Path | str = SOCIAL_EMBEDDINGS_PATH,
    clusters_path: Path | str = NARRATIVE_CLUSTERS_PATH,
    min_cluster_size: int = 8,
    min_samples: int | None = None,
    min_text_chars: int = 20,
) -> pd.DataFrame:
    emb = read_table(embeddings_path) if embeddings is None else embeddings.copy()
    posts = read_table(posts_path)
    if emb.empty or "embedding" not in emb.columns:
        emb["narrative_cluster_id"] = np.nan
        write_table(pd.DataFrame(), clusters_path)
        return write_table(emb, output_path)
    cur = emb.merge(posts[["post_id", "text"]], on="post_id", how="left", suffixes=("", "_post")) if not posts.empty else emb.copy()
    text_col = "text_post" if "text_post" in cur.columns else "text"
    vectors = [parse_embedding(value) for value in cur["embedding"]]
    valid_idx = [
        i
        for i, vector in enumerate(vectors)
        if vector is not None and len(str(cur.iloc[i].get(text_col, "") or "")) >= min_text_chars
    ]
    cur["narrative_cluster_id"] = -1.0
    if len(valid_idx) >= max(2, min_cluster_size):
        from sklearn.cluster import HDBSCAN

        x = np.vstack([vectors[i] for i in valid_idx])
        labels = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples).fit_predict(x)
        cur.loc[cur.index[valid_idx], "narrative_cluster_id"] = labels.astype(float)

    cluster_rows = []
    clustered = cur.loc[cur["narrative_cluster_id"].ge(0)].copy()
    for cluster_id, group in clustered.groupby("narrative_cluster_id", sort=True):
        subreddit_mix = group.get("subreddit", pd.Series("", index=group.index)).astype(str).value_counts().head(8).to_dict()
        cluster_rows.append(
            {
                "narrative_cluster_id": float(cluster_id),
                "cluster_size": int(len(group)),
                "representative_terms": _top_terms(group.get(text_col, pd.Series("", index=group.index))),
                "subreddit_mix": json.dumps(subreddit_mix, sort_keys=True),
            }
        )
    write_table(pd.DataFrame(cluster_rows), clusters_path)
    keep_cols = [c for c in emb.columns if c not in {"narrative_cluster_id"}]
    out = cur[keep_cols + ["narrative_cluster_id"]].copy()
    return write_table(out, output_path)


def build_narrative_features(
    *,
    posts_path: Path | str = REDDIT_POSTS_PATH,
    mentions_path: Path | str = REDDIT_MENTIONS_PATH,
    embeddings_path: Path | str = SOCIAL_EMBEDDINGS_PATH,
    output_path: Path | str = NARRATIVE_FEATURES_PATH,
) -> pd.DataFrame:
    posts = read_table(posts_path)
    mentions = read_table(mentions_path)
    emb = read_table(embeddings_path)
    if posts.empty or mentions.empty or emb.empty or "narrative_cluster_id" not in emb.columns:
        return write_table(pd.DataFrame(), output_path)
    posts = posts.copy()
    posts["timestamp"] = pd.to_datetime(posts["timestamp"], utc=True, errors="coerce")
    cur = mentions.merge(posts[["post_id", "timestamp"]], on="post_id", how="inner")
    cur = cur.merge(emb[["post_id", "narrative_cluster_id"]], on="post_id", how="inner")
    cur = cur.loc[cur["narrative_cluster_id"].ge(0)].copy()
    if cur.empty:
        return write_table(pd.DataFrame(), output_path)
    cur["timestamp"] = cur["timestamp"].dt.floor("1h")
    cur["ticker"] = cur["ticker"].astype(str).str.upper().str.replace("$", "", regex=False)
    hourly = (
        cur.groupby(["ticker", "narrative_cluster_id", "timestamp"], as_index=False)
        .agg(narrative_mentions_1h=("post_id", "count"))
        .sort_values(["ticker", "narrative_cluster_id", "timestamp"])
    )
    grouped = hourly.groupby(["ticker", "narrative_cluster_id"], group_keys=False)
    hourly["narrative_mentions_4h"] = grouped["narrative_mentions_1h"].transform(lambda s: s.rolling(4, min_periods=1).sum())
    hourly["narrative_mentions_24h"] = grouped["narrative_mentions_1h"].transform(lambda s: s.rolling(24, min_periods=1).sum())
    baseline = grouped["narrative_mentions_4h"].transform(lambda s: s.shift(1).rolling(20 * 24, min_periods=24).mean())
    hourly["narrative_acceleration"] = hourly["narrative_mentions_4h"] / baseline.replace(0, np.nan)
    mean = grouped["narrative_mentions_1h"].transform(lambda s: s.shift(1).rolling(30 * 24, min_periods=24).mean())
    std = grouped["narrative_mentions_1h"].transform(lambda s: s.shift(1).rolling(30 * 24, min_periods=24).std()).replace(0, np.nan)
    hourly["narrative_zscore"] = (hourly["narrative_mentions_1h"] - mean) / std
    totals = hourly.groupby(["ticker", "timestamp"], as_index=False)["narrative_mentions_24h"].sum().rename(
        columns={"narrative_mentions_24h": "ticker_narrative_mentions_24h"}
    )
    hourly = hourly.merge(totals, on=["ticker", "timestamp"], how="left")
    hourly["ticker_narrative_concentration"] = hourly["narrative_mentions_24h"] / hourly["ticker_narrative_mentions_24h"].replace(0, np.nan)
    hourly = hourly.sort_values(["ticker", "timestamp", "narrative_mentions_24h"], ascending=[True, True, False])
    top = hourly.drop_duplicates(["ticker", "timestamp"], keep="first").rename(
        columns={"narrative_cluster_id": "top_narrative_cluster_id"}
    )
    return write_table(top.reset_index(drop=True), output_path)

