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

import html
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

from themes.dynamic_theme.config import (
    BGE_MODEL,
    DAILY_BARS_DIR,
    DESC_BLEND_WEIGHT,
    NEWS_RECORDS_PATH,
    NEWS_SPAM_PATTERNS,
    TICKER_DOCUMENTS_PATH,
    TICKER_EMBEDDINGS_PATH,
    ensure_outputs,
)

logger = logging.getLogger(__name__)

# Path to pre-computed per-article BGE embeddings from the news module
_NEWS_EMBEDDINGS_PATH = Path(__file__).resolve().parents[3] / "signals" / "news" / "data" / "processed" / "news_embeddings.parquet"

# Ownership-filing / 13F / coverage spam that would otherwise dominate a
# mega-cap's averaged news embedding (compiled once from config patterns).
_SPAM_RE = re.compile("|".join(NEWS_SPAM_PATTERNS), re.I)
_HTML_TAG_RE = re.compile(r"<[^>]+>")


def _clean_text(text: str) -> str:
    """Strip HTML tags/entities so scraped markup doesn't pollute the signal."""
    if not isinstance(text, str):
        return ""
    return html.unescape(_HTML_TAG_RE.sub(" ", text)).strip()


def _is_spam(text: str) -> bool:
    return bool(_SPAM_RE.search(text or ""))

# Price co-movement: how many trading days of history to use for correlation PCA
_COMOVEMENT_LOOKBACK_DAYS = 60
_COMOVEMENT_PCA_COMPONENTS = 20
# Weight blending text vs. price vectors (0.0 = text only, 1.0 = price only)
_COMOVEMENT_WEIGHT = 0.4


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

    # Drop articles whose embedding is unavailable / degenerate.
    if "embedding_available" in sub.columns:
        sub = sub[sub["embedding_available"].fillna(True).astype(bool)]

    if sub.empty:
        return {}

    # Flag ownership-filing / 13F / HTML-only spam so it can be excluded from
    # the per-ticker average — unless a ticker has nothing else.
    text_col = next((c for c in sub.columns if c.lower() in ("text", "headline", "title", "body")), None)
    if text_col is not None:
        sub = sub.assign(_spam=sub[text_col].fillna("").map(_is_spam))
    else:
        sub = sub.assign(_spam=False)

    import json

    result: dict[str, np.ndarray] = {}
    n_filtered = 0
    for ticker, grp in sub.groupby(ticker_col):
        kept = grp[~grp["_spam"]]
        if kept.empty:  # all spam → fall back to using everything for coverage
            kept = grp
        else:
            n_filtered += len(grp) - len(kept)
        vecs_raw = [json.loads(v) if isinstance(v, str) else v for v in kept[emb_col].tolist()]
        vecs = np.array(vecs_raw, dtype=np.float32)
        mean_vec = vecs.mean(axis=0)
        norm = np.linalg.norm(mean_vec)
        result[str(ticker)] = mean_vec / norm if norm > 0 else mean_vec

    logger.info(
        "Fast path: aggregated article embeddings for %d / %d tickers (dropped %d spam articles)",
        len(result), len(tickers), n_filtered,
    )
    return result


def _concat_document(row: pd.Series) -> str:
    parts = [
        _clean_text(str(row.get("description") or "")),
        _clean_text(str(row.get("recent_news_summary") or "")),
        _clean_text(str(row.get("earnings_summary") or "")),
        _clean_text(str(row.get("catalyst_summary") or "")),
    ]
    return " ".join(p for p in parts if p).strip() or str(row.get("ticker", ""))


def _embed_descriptions(docs: pd.DataFrame, *, model_name: str = BGE_MODEL) -> dict[str, np.ndarray]:
    """Embed each ticker's cleaned business description ("what the company IS").

    This is the anchor of the text vector — it is what makes Apple cluster as
    consumer electronics rather than as institutional-ownership news. Returns
    {ticker: L2-normalized desc vector} for tickers with a non-empty description.
    """
    if "description" not in docs.columns:
        return {}
    sub = docs[["ticker", "description"]].copy()
    sub["clean"] = sub["description"].fillna("").map(_clean_text)
    sub = sub[sub["clean"].str.len() > 0]
    if sub.empty:
        return {}

    from signals.news.nlp import embed_texts_bge

    vecs = np.asarray(embed_texts_bge(sub["clean"].tolist(), model_name=model_name), dtype=np.float32)
    out: dict[str, np.ndarray] = {}
    for ticker, vec in zip(sub["ticker"].astype(str).tolist(), vecs):
        norm = np.linalg.norm(vec)
        out[ticker] = vec / norm if norm > 0 else vec
    logger.info("Embedded business descriptions for %d tickers", len(out))
    return out


def _build_comovement_vectors(
    tickers: list[str],
    as_of: pd.Timestamp | None = None,
    lookback_days: int = _COMOVEMENT_LOOKBACK_DAYS,
    n_components: int = _COMOVEMENT_PCA_COMPONENTS,
) -> dict[str, np.ndarray]:
    """Build PCA-compressed price co-movement vectors for each ticker.

    Loads the last `lookback_days` of daily close prices from the bars cache,
    computes pairwise return correlations across the universe, then projects
    each ticker's correlation row through PCA to get a low-dim behavior vector.

    Tickers with insufficient history get a zero vector (neutral — they don't
    attract or repel any cluster until they accumulate enough bars).

    Returns {ticker: behavior_vector (n_components,)}.
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    as_of_dt = (as_of or pd.Timestamp.now(tz="UTC")).tz_localize(None) if (
        as_of and as_of.tzinfo
    ) else (as_of or pd.Timestamp.now()).normalize()

    closes: dict[str, pd.Series] = {}
    for ticker in tickers:
        path = DAILY_BARS_DIR / f"{ticker}.parquet"
        if not path.exists():
            continue
        try:
            df = pd.read_parquet(path, columns=["timestamp", "close"])
            df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None).dt.normalize()
            df = df[df["timestamp"] <= as_of_dt].sort_values("timestamp")
            if len(df) >= 10:
                closes[ticker] = df.set_index("timestamp")["close"]
        except Exception:
            continue

    if len(closes) < 5:
        logger.warning("Too few tickers with bar history for co-movement PCA (%d)", len(closes))
        return {}

    # Build returns matrix: rows=dates, cols=tickers (last lookback_days only)
    price_df = pd.DataFrame(closes).ffill().dropna(how="all")
    price_df = price_df.iloc[-lookback_days:]
    returns = price_df.pct_change().dropna(how="all").fillna(0)

    if len(returns) < 10:
        logger.warning("Insufficient return history for co-movement PCA")
        return {}

    # Correlation matrix: [n_tickers_with_data, n_tickers_with_data]
    available = [t for t in tickers if t in returns.columns]
    ret_sub = returns[available].copy()
    corr = ret_sub.corr().fillna(0).values.astype(np.float32)

    # PCA on correlation rows → compressed behavior vectors
    actual_components = min(n_components, corr.shape[0] - 1)
    if actual_components < 2:
        return {}

    scaler = StandardScaler()
    corr_scaled = scaler.fit_transform(corr)
    pca = PCA(n_components=actual_components, random_state=42)
    behavior = pca.fit_transform(corr_scaled).astype(np.float32)

    # L2-normalize each vector
    norms = np.linalg.norm(behavior, axis=1, keepdims=True)
    behavior = np.where(norms > 0, behavior / norms, behavior)

    result = {}
    for i, ticker in enumerate(available):
        vec = behavior[i]
        # Pad to n_components if PCA used fewer
        if len(vec) < n_components:
            vec = np.pad(vec, (0, n_components - len(vec)))
        result[ticker] = vec

    logger.info(
        "Co-movement PCA: %d tickers × %d-day returns → %d components (%.1f%% variance explained)",
        len(available), len(returns), actual_components,
        100 * pca.explained_variance_ratio_.sum(),
    )
    return result


def generate_embeddings(
    docs: pd.DataFrame | None = None,
    *,
    model_name: str = BGE_MODEL,
    force_fresh: bool = False,
    as_of: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Embed ticker documents and write ticker_embeddings.parquet.

    The final embedding is a weighted blend of:
      - text vector  (BGE, 384-dim): "what the company IS" from news + description
      - behavior vector (PCA of 60d return correlations, 20-dim): "how it MOVES"

    Both are L2-normalized before blending. Tickers with no bar history get
    text-only embeddings (behavior weight falls back to 0 for them).

    Args:
        docs: ticker_documents DataFrame. Loaded from disk if None.
        model_name: BGE model name (used only for slow-path fresh embedding).
        force_fresh: skip the fast path and always re-embed from text documents.
        as_of: date ceiling for bar history (prevents lookahead in backfill).
    """
    ensure_outputs()

    if docs is None:
        docs = pd.read_parquet(TICKER_DOCUMENTS_PATH)

    if docs.empty:
        logger.warning("No ticker documents found — skipping embedding step")
        return pd.DataFrame(columns=["ticker", "embedding", "date"])

    tickers = docs["ticker"].astype(str).tolist()
    date = docs["date"].iloc[0] if "date" in docs.columns else pd.Timestamp.now().normalize()
    as_of = as_of or (pd.to_datetime(date) if not isinstance(date, pd.Timestamp) else date)

    # ── Text vector = description anchor blended with cleaned news ────────────
    # news_by_ticker : mean of spam-filtered article embeddings ("what it DOES")
    # desc_by_ticker : business-description embedding ("what the company IS")
    news_by_ticker: dict[str, np.ndarray] = {}
    if not force_fresh:
        news_by_ticker = _load_article_embeddings_by_ticker(tickers)

    desc_by_ticker = _embed_descriptions(docs, model_name=model_name)

    embeddings_by_ticker: dict[str, np.ndarray] = {}
    w = float(DESC_BLEND_WEIGHT)
    n_blend = n_desc_only = n_news_only = 0
    for ticker in set(tickers):
        desc = desc_by_ticker.get(ticker)
        news = news_by_ticker.get(ticker)
        if desc is not None and news is not None:
            vec = w * desc + (1.0 - w) * news
            n_blend += 1
        elif desc is not None:
            vec = desc
            n_desc_only += 1
        elif news is not None:
            vec = news
            n_news_only += 1
        else:
            continue
        norm = np.linalg.norm(vec)
        embeddings_by_ticker[ticker] = (vec / norm if norm > 0 else vec).astype(np.float32)

    logger.info(
        "Text vectors: %d desc+news blend (w_desc=%.2f), %d desc-only, %d news-only",
        n_blend, w, n_desc_only, n_news_only,
    )

    # ── Slow path: fresh concat embedding for tickers with neither source ─────
    missing = [t for t in tickers if t not in embeddings_by_ticker]
    if missing:
        logger.info("Slow path: embedding %d tickers with no description or news ...", len(missing))
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
            emb = np.asarray(emb, dtype=np.float32)
            norm = np.linalg.norm(emb)
            embeddings_by_ticker[ticker] = emb / norm if norm > 0 else emb

    # ── Price co-movement blend ───────────────────────────────────────────────
    logger.info("Building price co-movement vectors (lookback=%dd, PCA=%d) ...",
                _COMOVEMENT_LOOKBACK_DAYS, _COMOVEMENT_PCA_COMPONENTS)
    behavior_by_ticker = _build_comovement_vectors(tickers, as_of=as_of)

    text_dim = len(next(iter(embeddings_by_ticker.values()))) if embeddings_by_ticker else 384
    behavior_dim = _COMOVEMENT_PCA_COMPONENTS

    blended: dict[str, np.ndarray] = {}
    for ticker in tickers:
        if ticker not in embeddings_by_ticker:
            continue
        text_vec = embeddings_by_ticker[ticker]
        if ticker in behavior_by_ticker:
            bvec = behavior_by_ticker[ticker]
            if len(bvec) < behavior_dim:
                bvec = np.pad(bvec, (0, behavior_dim - len(bvec)))
            combined = np.concatenate([
                (1.0 - _COMOVEMENT_WEIGHT) * text_vec,
                _COMOVEMENT_WEIGHT * bvec,
            ]).astype(np.float32)
        else:
            # No bar history: pad with zeros so UMAP input dim is consistent
            combined = np.concatenate([
                text_vec,
                np.zeros(behavior_dim, dtype=np.float32),
            ]).astype(np.float32)
        # L2-normalize the final blended vector
        norm = np.linalg.norm(combined)
        blended[ticker] = combined / norm if norm > 0 else combined

    n_blended = sum(1 for t in tickers if t in behavior_by_ticker)
    logger.info("Blended text+behavior for %d / %d tickers (text-only for %d)",
                n_blended, len(tickers), len(tickers) - n_blended)

    # Preserve ticker order from docs
    final_vecs = [blended[t] for t in tickers if t in blended]
    final_tickers = [t for t in tickers if t in blended]

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
