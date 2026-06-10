"""Step 2 — Generate ticker embeddings using the existing BGE pipeline.

Concatenates all document fields into one text block per ticker, then embeds
with bge-small-en-v1.5 (same model used by the news module).

Output: ticker_embeddings.parquet
Schema : ticker | embedding (list[float]) | date
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from dynamic_theme.config import BGE_MODEL, TICKER_DOCUMENTS_PATH, TICKER_EMBEDDINGS_PATH, ensure_outputs

logger = logging.getLogger(__name__)


def _concat_document(row: pd.Series) -> str:
    parts = [
        str(row.get("description") or ""),
        str(row.get("recent_news_summary") or ""),
        str(row.get("earnings_summary") or ""),
        str(row.get("catalyst_summary") or ""),
    ]
    return " ".join(p for p in parts if p).strip() or row.get("ticker", "")


def generate_embeddings(
    docs: pd.DataFrame | None = None,
    *,
    model_name: str = BGE_MODEL,
) -> pd.DataFrame:
    """Embed ticker documents and write ticker_embeddings.parquet."""
    ensure_outputs()

    if docs is None:
        docs = pd.read_parquet(TICKER_DOCUMENTS_PATH)

    if docs.empty:
        logger.warning("No ticker documents found — skipping embedding step")
        return pd.DataFrame(columns=["ticker", "embedding", "date"])

    texts = docs.apply(_concat_document, axis=1).tolist()
    tickers = docs["ticker"].astype(str).tolist()
    date = docs["date"].iloc[0] if "date" in docs.columns else pd.Timestamp.now().normalize()

    logger.info("Embedding %d tickers with %s ...", len(texts), model_name)

    try:
        from news.nlp import embed_texts_bge
        embeddings: np.ndarray = embed_texts_bge(texts, model_name=model_name)
    except ImportError:
        logger.error("sentence-transformers not installed. Install it with: pip install sentence-transformers")
        raise

    # store as Python lists so pyarrow can serialize them
    out = pd.DataFrame(
        {
            "ticker": tickers,
            "embedding": [emb.tolist() for emb in embeddings],
            "date": date,
        }
    )
    out.to_parquet(TICKER_EMBEDDINGS_PATH, index=False)
    logger.info("Wrote %s  rows=%d  dim=%d", TICKER_EMBEDDINGS_PATH, len(out), len(embeddings[0]))
    return out


def load_embeddings_matrix() -> tuple[list[str], np.ndarray, pd.Timestamp]:
    """Load embeddings parquet → (tickers, matrix[N, D], date)."""
    df = pd.read_parquet(TICKER_EMBEDDINGS_PATH)
    tickers = df["ticker"].astype(str).tolist()
    matrix = np.array(df["embedding"].tolist(), dtype=np.float32)
    date = pd.to_datetime(df["date"].iloc[0]) if "date" in df.columns else pd.Timestamp.now()
    return tickers, matrix, date
