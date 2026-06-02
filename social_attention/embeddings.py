"""BGE embedding generation for Reddit social posts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from news.nlp import embed_texts_bge
from news.pipeline import parse_embedding
from social_attention.config import REDDIT_POSTS_PATH, SOCIAL_EMBEDDINGS_PATH
from social_attention.io import read_table, write_table


def build_social_embeddings(
    posts: pd.DataFrame | None = None,
    *,
    posts_path: Path | str = REDDIT_POSTS_PATH,
    output_path: Path | str = SOCIAL_EMBEDDINGS_PATH,
    batch_size: int = 256,
    device: str = "cpu",
) -> pd.DataFrame:
    posts = read_table(posts_path) if posts is None else posts.copy()
    if posts.empty:
        out = pd.DataFrame(columns=["post_id", "ticker", "timestamp", "embedding", "embedding_available"])
        return write_table(out, output_path)
    keep = [c for c in ["post_id", "timestamp", "kind", "subreddit", "text"] if c in posts.columns]
    out = posts[keep].copy()
    out["embedding"] = None
    out["embedding_available"] = 0.0
    texts = out.get("text", pd.Series("", index=out.index)).fillna("").astype(str).tolist()
    vectors: list[np.ndarray] = []
    try:
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            vectors.extend(embed_texts_bge(batch, device=device))
        out["embedding"] = [json.dumps(np.asarray(v, dtype=np.float32).astype(float).tolist()) for v in vectors]
        out["embedding_available"] = 1.0
    except Exception:
        out["embedding_available"] = 0.0
    return write_table(out, output_path)


__all__ = ["build_social_embeddings", "parse_embedding"]

