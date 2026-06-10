"""FinBERT scoring for Reddit social posts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from signals.news.nlp import finbert_scores_batch
from signals.social_attention.config import REDDIT_POSTS_PATH
from signals.social_attention.io import read_table, write_table


def score_reddit_sentiment(
    posts: pd.DataFrame | None = None,
    *,
    posts_path: Path | str = REDDIT_POSTS_PATH,
    output_path: Path | str = REDDIT_POSTS_PATH,
    batch_size: int = 32,
    device: int = -1,
) -> pd.DataFrame:
    posts = read_table(posts_path) if posts is None else posts.copy()
    if posts.empty:
        return write_table(posts, output_path)
    texts = posts.get("text", pd.Series("", index=posts.index)).fillna("").astype(str).tolist()
    try:
        scores = finbert_scores_batch(texts, batch_size=batch_size, device=device)
        score_df = pd.DataFrame(scores)
        score_df["finbert_available"] = 1.0
    except Exception:
        score_df = pd.DataFrame(
            {
                "finbert_positive_score": np.nan,
                "finbert_negative_score": np.nan,
                "finbert_neutral_score": np.nan,
                "finbert_available": 0.0,
            },
            index=posts.index,
        )
    out = posts.drop(
        columns=[
            "finbert_positive_score",
            "finbert_negative_score",
            "finbert_neutral_score",
            "finbert_available",
            "sentiment_score",
        ],
        errors="ignore",
    ).reset_index(drop=True)
    out = pd.concat([out, score_df.reset_index(drop=True)], axis=1)
    out["sentiment_score"] = out["finbert_positive_score"] - out["finbert_negative_score"]
    return write_table(out, output_path)

