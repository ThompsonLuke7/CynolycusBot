"""CPU-only NLP wrappers for BGE embeddings and FinBERT tone."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

import numpy as np

from news.config import DEFAULT_BGE_MODEL, DEFAULT_FINBERT_MODEL, EMBEDDINGS_DIR


def text_hash(text: str) -> str:
    return hashlib.sha256(str(text or "").encode("utf-8", errors="ignore")).hexdigest()[:20]


def embedding_path(record_id: str, content_hash: str, model_name: str = DEFAULT_BGE_MODEL) -> Path:
    slug = model_name.replace("/", "_")
    return EMBEDDINGS_DIR / slug / f"{record_id}_{content_hash}.npy"


def embed_texts_bge(
    texts: Iterable[str],
    *,
    model_name: str = DEFAULT_BGE_MODEL,
) -> np.ndarray:
    """Embed text with BGE on CPU, raising a helpful ImportError if optional deps are absent."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError("Install sentence-transformers to generate BGE news embeddings.") from exc
    model = SentenceTransformer(model_name, device="cpu")
    return np.asarray(model.encode(list(texts), normalize_embeddings=True), dtype=np.float32)


def finbert_scores(text: str, *, model_name: str = DEFAULT_FINBERT_MODEL) -> dict[str, float]:
    """Score financial tone with FinBERT on CPU."""
    try:
        from transformers import pipeline
    except ImportError as exc:
        raise ImportError("Install transformers to generate FinBERT news sentiment.") from exc
    clf = pipeline("sentiment-analysis", model=model_name, tokenizer=model_name, device=-1, truncation=True)
    rows = clf([text or ""])
    totals = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
    for row in rows:
        label = str(row.get("label", "")).lower()
        score = float(row.get("score", 0.0))
        if label in totals:
            totals[label] += score
    return {
        "finbert_positive_score": totals["positive"],
        "finbert_negative_score": totals["negative"],
        "finbert_neutral_score": totals["neutral"],
    }

