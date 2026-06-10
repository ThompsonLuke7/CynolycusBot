"""CPU-only NLP wrappers for BGE embeddings and FinBERT tone."""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import numpy as np

from signals.news.config import DEFAULT_BGE_MODEL, DEFAULT_FINBERT_MODEL, EMBEDDINGS_DIR


def text_hash(text: str) -> str:
    return hashlib.sha256(str(text or "").encode("utf-8", errors="ignore")).hexdigest()[:20]


def embedding_path(record_id: str, content_hash: str, model_name: str = DEFAULT_BGE_MODEL) -> Path:
    slug = model_name.replace("/", "_")
    return EMBEDDINGS_DIR / slug / f"{record_id}_{content_hash}.npy"


def embed_texts_bge(
    texts: Iterable[str],
    *,
    model_name: str = DEFAULT_BGE_MODEL,
    device: str | None = None,
) -> np.ndarray:
    """Embed text with BGE on CPU, raising a helpful ImportError if optional deps are absent."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError("Install sentence-transformers to generate BGE news embeddings.") from exc
    if device is None:
        try:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"
    model = SentenceTransformer(model_name, device=device)
    return np.asarray(model.encode(list(texts), normalize_embeddings=True), dtype=np.float32)


def _resolve_finbert_device(device: int | None = None) -> int:
    if device is not None:
        return int(device)
    try:
        import torch

        return 0 if torch.cuda.is_available() else -1
    except Exception:
        return -1


@lru_cache(maxsize=4)
def _finbert_pipeline(model_name: str, device: int):
    try:
        from transformers import pipeline
    except ImportError as exc:
        raise ImportError("Install transformers to generate FinBERT news sentiment.") from exc

    return pipeline("sentiment-analysis", model=model_name, tokenizer=model_name, device=device, truncation=True)


def _empty_finbert_scores() -> dict[str, float]:
    return {
        "finbert_positive_score": 0.0,
        "finbert_negative_score": 0.0,
        "finbert_neutral_score": 0.0,
    }


def _normalize_finbert_row(row: object) -> dict[str, float]:
    totals = _empty_finbert_scores()
    items = row if isinstance(row, list) else [row]
    for item in items:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label", "")).lower()
        key = f"finbert_{label}_score"
        if key in totals:
            totals[key] = float(item.get("score", 0.0))
    return totals


def finbert_scores_batch(
    texts: Iterable[str],
    *,
    model_name: str = DEFAULT_FINBERT_MODEL,
    device: int | None = None,
    batch_size: int = 32,
) -> list[dict[str, float]]:
    """Score financial tone with one cached FinBERT pipeline."""
    clf = _finbert_pipeline(model_name, _resolve_finbert_device(device))
    rows = clf([text or "" for text in texts], top_k=None, batch_size=batch_size)
    return [_normalize_finbert_row(row) for row in rows]


def finbert_scores(text: str, *, model_name: str = DEFAULT_FINBERT_MODEL, device: int | None = None) -> dict[str, float]:
    """Score financial tone for one text with the cached FinBERT pipeline."""
    return finbert_scores_batch([text], model_name=model_name, device=device)[0]
