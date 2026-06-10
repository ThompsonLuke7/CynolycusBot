"""Step 2 — Generate ticker embeddings using the existing BGE pipeline.

Fast path (preferred): aggregate the already-computed per-article BGE embeddings
from news/data/processed/news_embeddings.parquet by averaging all articles for
each ticker. Zero re-embedding required — costs milliseconds.

Slow path (fallback): if news_embeddings.parquet is absent or a ticker has no
article coverage, concatenate its document fields and embed fresh.

Output: ticker_embeddings.parquet
Schema : ticker | embedding (list[float]) | date
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from themes.dynamic_theme.config import BGE_MODEL, NEWS_RECORDS_PATH, TICKER_DOCUMENTS_PATH, TICKER_EMBEDDINGS_PATH, ensure_outputs

logger = logging.getLogger(__name__)

# Path to pre-computed per-article BGE embeddings from the news module
_NEWS_EMBEDDINGS_PATH = Path(__file__).resolve().parents[3] / "signals" / "news" / "data" / "processed" / "news_embeddings.parquet"


def _load_article_embeddings_by_ticker(tickers: list[str]) -> dict[str, np.ndarray]:
    """Average existing per-article embeddings into one vector per ticker.

    Returns {ticker: mean_embedding}. Tickers with no article coverage are absent.
    """
    if not _NEWS_EMBEDDINGS_PATH.exists():
        return {}

    try:
        emb_df = pd.read_parquet(_NEWS_EMBEDDINGS_PATH)
    except Exception as exc:
        logger.warning("Could not read news_embeddings.parquet: %s", exc)
        return {}

    # The news embeddings parquet may have ticker in the embeddings frame itself
    # or we need to join via news_records.
    ticker_col = next((c for c in emb_df.columns if c.lower() in ("ticker", "symbol")), None)
    emb_col = next((c for c in emb_df.columns if c.lower() in ("embedding", "vector", "emb")), None)

    if ticker_col is None or emb_col is None:
        # Try joining via news_records (record_id key)
        if NEWS_RECORDS_PATH.exists():
            try:
                records = pd.read_parquet(NEWS_RECORDS_PATH, columns=["record_id", "ticker"])
                id_col = next((c for c in emb_df.columns if c.lower() in ("record_id", "id")), None)
                if id_col and "record_id" in records.columns:
                    emb_df = emb_df.merge(records, left_on=id_col, right_on="record_id", how="inner")
                    ticker_col = "ticker"
                    # emb_col may still be missing — detect it
                    emb_col = next((c for c in emb_df.columns if c.lower() in ("embedding", "vector", "emb")), None)
            except Exception as exc:
                logger.warning("Could not join news_embeddings with news_records: %s", exc)

    if ticker_col is None or emb_col is None:
        logger.warning("news_embeddings.parquet has unexpected schema — cannot extract ticker embeddings")
        return {}

    ticker_set = set(tickers)
    emb_df[ticker_col] = emb_df[ticker_col].astype(str).str.upper()
    sub = emb_df[emb_df[ticker_col].isin(ticker_set)]

    if sub.empty:
        return {}

    result: dict[str, np.ndarray] = {}
    for ticker, grp in sub.groupby(ticker_col):
        vecs = np.array(grp[emb_col].tolist(), dtype=np.float32)
        mean_vec = vecs.mean(axis=0)
        norm = np.linalg.norm(mean_vec)
        result[str(ticker)] = mean_vec / norm if norm > 0 else mean_vec

    logger.info("Fast path: aggregated article embeddings for %d / %d tickers", len(result), len(tickers))
    return result


def _concat_document(row: pd.Series) -> str:
    parts = [
        str(row.get("description") or ""),
        str(row.get("recent_news_summary") or ""),
        str(row.get("earnings_summary") or ""),
        str(row.get("catalyst_summary") or ""),
    ]
    return " ".join(p for p in parts if p).strip() or str(row.get("ticker", ""))


def generate_embeddings(
    docs: pd.DataFrame | None = None,
    *,
    model_name: str = BGE_MODEL,
    force_fresh: bool = False,
) -> pd.DataFrame:
    """Embed ticker documents and write ticker_embeddings.parquet.

    Args:
        docs: ticker_documents DataFrame. Loaded from disk if None.
        model_name: BGE model name (used only for slow-path fresh embedding).
        force_fresh: skip the fast path and always re-embed from text documents.
    """
    ensure_outputs()

    if docs is None:
        docs = pd.read_parquet(TICKER_DOCUMENTS_PATH)

    if docs.empty:
        logger.warning("No ticker documents found — skipping embedding step")
        return pd.DataFrame(columns=["ticker", "embedding", "date"])

    tickers = docs["ticker"].astype(str).tolist()
    date = docs["date"].iloc[0] if "date" in docs.columns else pd.Timestamp.now().normalize()

    embeddings_by_ticker: dict[str, np.ndarray] = {}

    # ── Fast path: reuse pre-computed article-level embeddings ────────────────
    if not force_fresh:
        embeddings_by_ticker = _load_article_embeddings_by_ticker(tickers)

    missing = [t for t in tickers if t not in embeddings_by_ticker]

    # ── Slow path: embed fresh for any tickers without article coverage ───────
    if missing:
        logger.info("Slow path: embedding %d tickers without article coverage ...", len(missing))
        docs_missing = docs[docs["ticker"].isin(set(missing))].copy()
        texts = docs_missing.apply(_concat_document, axis=1).tolist()
        missing_tickers = docs_missing["ticker"].astype(str).tolist()

        try:
            from signals.news.nlp import embed_texts_bge
            fresh_embs: np.ndarray = embed_texts_bge(texts, model_name=model_name)
        except ImportError:
            logger.error("sentence-transformers not installed: pip install sentence-transformers")
            raise

        for ticker, emb in zip(missing_tickers, fresh_embs):
            embeddings_by_ticker[ticker] = emb

    # Preserve ticker order from docs
    final_vecs = [embeddings_by_ticker[t] for t in tickers if t in embeddings_by_ticker]
    final_tickers = [t for t in tickers if t in embeddings_by_ticker]

    out = pd.DataFrame(
        {
            "ticker": final_tickers,
            "embedding": [v.tolist() for v in final_vecs],
            "date": date,
        }
    )
    out.to_parquet(TICKER_EMBEDDINGS_PATH, index=False)
    dim = len(final_vecs[0]) if final_vecs else 0
    logger.info(
        "Wrote %s  rows=%d  dim=%d  (fast=%d, fresh=%d)",
        TICKER_EMBEDDINGS_PATH, len(out), dim,
        len(out) - len(missing), len(missing),
    )
    return out


def load_embeddings_matrix() -> tuple[list[str], np.ndarray, pd.Timestamp]:
    """Load embeddings parquet → (tickers, matrix[N, D], date)."""
    df = pd.read_parquet(TICKER_EMBEDDINGS_PATH)
    tickers = df["ticker"].astype(str).tolist()
    matrix = np.array(df["embedding"].tolist(), dtype=np.float32)
    date = pd.to_datetime(df["date"].iloc[0]) if "date" in df.columns else pd.Timestamp.now()
    return tickers, matrix, date
