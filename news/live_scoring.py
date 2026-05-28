"""Live-trading scoring hook for catalyst news.

Given a new headline (and optionally its summary/body), this module:

1. Classifies it into a catalyst family/subtype using the same rules as the offline pipeline.
2. Embeds the text with BGE.
3. Compares against the labeled-and-realized historical library.
4. Returns:
   - ``expected_5d_return``: weighted average forward-5d return of similar prior records
   - ``winner_similarity_max`` / ``loser_similarity_max`` / ``edge``
   - ``top_examples``: a peek at the prior records actually driving the prediction

The lookup is **leak-free**: only records whose 10-day forward-return label was
realized by the prediction timestamp contribute. For live use (timestamp ≈ now),
that means only records older than the label horizon are used.

Typical use from ``momentum_expansion/live``:

    from news.live_scoring import score_headline

    result = score_headline(
        ticker="RKLB",
        timestamp=pd.Timestamp.utcnow(),
        headline="Rocket Lab wins $200M Department of Defense contract",
        summary="...",
        source="finnhub",
    )
    if result["expected_5d_return"] > 0.05 and result["edge"] > 0.10:
        # build trade signal …
        ...

CLI form: ``python -m catalysts.main --stage score-live --ticker RKLB --headline "..."``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from news.catalyst_types import classify_catalyst_types
from news.config import (
    LABEL_HORIZON_DAYS,
    LOSER_LIBRARY_PATH,
    NEWS_LABELS_PATH,
    NEWS_EMBEDDINGS_PATH,
    NEWS_RECORDS_PATH,
    WINNER_LIBRARY_PATH,
)
from news.earnings import enrich_earnings_catalyst_fields
from news.pipeline import parse_embedding
from news.relations import classify_news_relations
from news.schema import NewsRecord


@dataclass
class LiveScoringConfig:
    label_horizon_days: int = LABEL_HORIZON_DAYS
    min_similarity: float = 0.55  # lower than offline 0.72: live calls are noisier
    top_k: int = 25
    same_family_weight: float = 1.60
    cross_family_weight: float = 0.15
    same_subtype_weight: float = 1.35
    same_relation_weight: float = 1.25
    top_examples: int = 5
    library_paths: dict[str, Path] = field(
        default_factory=lambda: {
            "winners": WINNER_LIBRARY_PATH,
            "losers": LOSER_LIBRARY_PATH,
            "records": NEWS_RECORDS_PATH,
            "embeddings": NEWS_EMBEDDINGS_PATH,
            "labels": NEWS_LABELS_PATH,
        }
    )


@dataclass
class _Library:
    df: pd.DataFrame  # holds record_id, ticker, headline, family, subtype, relation_type, fwd returns
    vectors: np.ndarray  # (N, D) row-normalized embeddings
    label_ready_ts: np.ndarray  # datetime64[ns], in UTC-naive form

    def filter_by_cutoff(self, cutoff_ts: pd.Timestamp) -> tuple[pd.DataFrame, np.ndarray]:
        """Return (df, vectors) restricted to rows whose label was realized by cutoff_ts."""
        cutoff64 = pd.Timestamp(cutoff_ts).tz_convert("UTC").tz_localize(None).to_datetime64()
        mask = self.label_ready_ts <= cutoff64
        if not mask.any():
            return self.df.iloc[0:0], np.empty((0, self.vectors.shape[1]), dtype=np.float32)
        return self.df.loc[mask], self.vectors[mask]


def _load_library(config: LiveScoringConfig) -> _Library:
    paths = config.library_paths
    records = pd.read_parquet(paths["records"])
    embeddings = pd.read_parquet(paths["embeddings"])
    labels = pd.read_parquet(paths["labels"])

    # Drop overlapping columns (catalyst_family, catalyst_subtype, text) before merge so
    # we keep the canonical news_records versions.
    emb_cols_to_drop = [c for c in ("text", "catalyst_family", "catalyst_subtype") if c in embeddings.columns]
    df = (
        records.merge(embeddings.drop(columns=emb_cols_to_drop, errors="ignore"), on=["record_id", "ticker", "timestamp"], how="inner")
        .merge(labels, on=["record_id", "ticker", "timestamp"], how="inner")
    )
    df = df.dropna(subset=["forward_5d_return"]).reset_index(drop=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")

    vectors_raw = [parse_embedding(value) for value in df["embedding"]]
    keep = [i for i, v in enumerate(vectors_raw) if v is not None]
    if not keep:
        return _Library(df=df.iloc[0:0], vectors=np.empty((0, 0), dtype=np.float32), label_ready_ts=np.asarray([], dtype="datetime64[ns]"))
    df = df.iloc[keep].reset_index(drop=True)
    mat = np.vstack([vectors_raw[i] for i in keep]).astype(np.float32)
    mat = mat / np.maximum(np.linalg.norm(mat, axis=1, keepdims=True), 1e-12)
    label_ready_ts = (df["timestamp"] + pd.Timedelta(days=int(config.label_horizon_days))).dt.tz_convert("UTC").dt.tz_localize(None).to_numpy(dtype="datetime64[ns]")
    return _Library(df=df, vectors=mat, label_ready_ts=label_ready_ts)


_LIBRARY_CACHE: dict[tuple, _Library] = {}


def _library_signature(config: LiveScoringConfig) -> tuple:
    sig = []
    for path in config.library_paths.values():
        p = Path(path)
        if p.exists():
            stat = p.stat()
            sig.append((str(p), stat.st_size, int(stat.st_mtime)))
        else:
            sig.append((str(p), 0, 0))
    sig.append(int(config.label_horizon_days))
    return tuple(sig)


def _cached_library(signature: tuple, config: LiveScoringConfig) -> _Library:
    cached = _LIBRARY_CACHE.get(signature)
    if cached is not None:
        return cached
    library = _load_library(config)
    _LIBRARY_CACHE.clear()  # keep cache small; only one live library config at a time.
    _LIBRARY_CACHE[signature] = library
    return library


def _classify_new_record(
    *,
    ticker: str,
    timestamp: pd.Timestamp,
    headline: str,
    summary: str,
    body: str,
    source: str,
    url: str,
) -> pd.DataFrame:
    record = NewsRecord(
        ticker=ticker,
        timestamp=timestamp.isoformat() if hasattr(timestamp, "isoformat") else str(timestamp),
        headline=headline,
        summary=summary,
        body=body,
        url=url,
        source=source,
    ).to_record()
    df = pd.DataFrame([record])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = classify_news_relations(df)
    df = enrich_earnings_catalyst_fields(df)
    df = classify_catalyst_types(df)
    return df


def _embed_query(text: str) -> np.ndarray | None:
    try:
        from news.nlp import embed_texts_bge

        vec = embed_texts_bge([text])
        return np.asarray(vec[0], dtype=np.float32) if len(vec) else None
    except ImportError:
        return None


def _finbert_query(text: str) -> dict[str, float] | None:
    try:
        from news.nlp import finbert_scores

        return finbert_scores(text)
    except ImportError:
        return None


def score_headline(
    *,
    ticker: str,
    timestamp: pd.Timestamp | str | None = None,
    headline: str,
    summary: str = "",
    body: str = "",
    source: str = "finnhub",
    url: str = "",
    config: LiveScoringConfig | None = None,
) -> dict[str, Any]:
    """Score a single news headline against the labeled-and-realized historical library.

    Returns a dict with:
      - ``ticker`` / ``timestamp_utc`` / ``catalyst_family`` / ``catalyst_subtype``
      - ``relation_type`` / ``impact_role`` / ``is_direct_catalyst``
      - ``finbert_positive_score`` / ``finbert_negative_score`` / ``finbert_neutral_score`` (if available)
      - ``library_size_eligible``: how many prior records contributed to the lookup
      - ``expected_5d_return`` / ``expected_max_forward_return`` / ``expected_max_drawdown``
      - ``expansion_hit_rate``: share of top-K matches that hit +10% within 10 days
      - ``winner_similarity_max`` / ``loser_similarity_max`` / ``edge`` (winner - loser)
      - ``top_examples``: top-K most similar prior records (record_id, headline, family, fwd_5d, similarity)
      - ``status``: "ok" | "no_priors" | "no_embedding_model" | "empty_library"
    """
    config = config or LiveScoringConfig()
    ts = pd.Timestamp(timestamp) if timestamp is not None else pd.Timestamp.utcnow().tz_localize("UTC")
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    ts = ts.tz_convert("UTC")

    classified = _classify_new_record(
        ticker=ticker, timestamp=ts, headline=headline, summary=summary, body=body, source=source, url=url
    )
    row = classified.iloc[0]
    family = str(row.get("catalyst_family", ""))
    subtype = str(row.get("catalyst_subtype", ""))
    relation = str(row.get("relation_type", ""))
    impact_role = str(row.get("impact_role", ""))

    library = _cached_library(_library_signature(config), config)
    if library.df.empty:
        return {
            "status": "empty_library",
            "ticker": ticker.upper(),
            "timestamp_utc": ts.isoformat(),
            "catalyst_family": family,
            "catalyst_subtype": subtype,
            "relation_type": relation,
            "impact_role": impact_role,
            "library_size_eligible": 0,
        }

    eligible_df, eligible_vecs = library.filter_by_cutoff(ts)
    if eligible_df.empty:
        return {
            "status": "no_priors",
            "ticker": ticker.upper(),
            "timestamp_utc": ts.isoformat(),
            "catalyst_family": family,
            "catalyst_subtype": subtype,
            "library_size_eligible": 0,
        }

    text_for_embedding = " ".join(s for s in (headline, summary, body) if s).strip()
    query_vec = _embed_query(text_for_embedding)
    if query_vec is None:
        return {
            "status": "no_embedding_model",
            "ticker": ticker.upper(),
            "timestamp_utc": ts.isoformat(),
            "catalyst_family": family,
            "catalyst_subtype": subtype,
            "library_size_eligible": int(len(eligible_df)),
        }
    query = query_vec / max(float(np.linalg.norm(query_vec)), 1e-12)
    sims = eligible_vecs @ query

    weights = np.maximum(sims - float(config.min_similarity), 0.0)
    eligible_family = eligible_df["catalyst_family"].astype(str).to_numpy()
    eligible_subtype = eligible_df["catalyst_subtype"].astype(str).to_numpy()
    eligible_relation = eligible_df["relation_type"].astype(str).to_numpy() if "relation_type" in eligible_df.columns else np.full(len(eligible_df), "", dtype=object)

    weights = weights * np.where(eligible_family == family, config.same_family_weight, config.cross_family_weight)
    weights = weights * np.where(eligible_subtype == subtype, config.same_subtype_weight, 1.0)
    weights = weights * np.where(eligible_relation == relation, config.same_relation_weight, 1.0)

    valid = np.flatnonzero(weights > 0)
    finbert = _finbert_query(text_for_embedding) or {}
    if len(valid) == 0:
        return {
            "status": "no_similar_priors",
            "ticker": ticker.upper(),
            "timestamp_utc": ts.isoformat(),
            "catalyst_family": family,
            "catalyst_subtype": subtype,
            "library_size_eligible": int(len(eligible_df)),
            "finbert_positive_score": finbert.get("finbert_positive_score"),
            "finbert_negative_score": finbert.get("finbert_negative_score"),
            "finbert_neutral_score": finbert.get("finbert_neutral_score"),
        }

    ranked = valid[np.argsort(weights[valid])[-int(config.top_k):]]
    ranked_weights = weights[ranked]
    fwd_5d = eligible_df.iloc[ranked]["forward_5d_return"].astype(float).to_numpy()
    max_fwd = eligible_df.iloc[ranked]["max_forward_return"].astype(float).to_numpy() if "max_forward_return" in eligible_df.columns else fwd_5d.copy()
    max_dd = eligible_df.iloc[ranked]["max_drawdown"].astype(float).to_numpy() if "max_drawdown" in eligible_df.columns else np.zeros_like(fwd_5d)
    expansion = eligible_df.iloc[ranked]["expansion_label"].astype(float).to_numpy() if "expansion_label" in eligible_df.columns else np.zeros_like(fwd_5d)

    expected_5d = float(np.average(fwd_5d, weights=ranked_weights)) if np.any(ranked_weights > 0) else float("nan")
    expected_max_fwd = float(np.average(max_fwd, weights=ranked_weights)) if np.any(ranked_weights > 0) else float("nan")
    expected_max_dd = float(np.average(max_dd, weights=ranked_weights)) if np.any(ranked_weights > 0) else float("nan")
    expansion_hit_rate = float(np.average(expansion, weights=ranked_weights)) if np.any(ranked_weights > 0) else float("nan")

    # Winner / loser library similarity (already filtered by label-ready in _prepare).
    winner_sim = float("nan")
    loser_sim = float("nan")
    winner_mask = eligible_df["expansion_label"].astype(float).to_numpy() >= 1.0 if "expansion_label" in eligible_df.columns else np.zeros(len(eligible_df), dtype=bool)
    loser_mask = eligible_df["expansion_label"].astype(float).to_numpy() <= 0.0 if "expansion_label" in eligible_df.columns else np.zeros(len(eligible_df), dtype=bool)
    if winner_mask.any():
        winner_sim = float(np.nanmax(sims[winner_mask]))
    if loser_mask.any():
        loser_sim = float(np.nanmax(sims[loser_mask]))
    edge = (winner_sim - loser_sim) if np.isfinite(winner_sim) and np.isfinite(loser_sim) else float("nan")

    top_idx = ranked[np.argsort(-ranked_weights)][: int(config.top_examples)]
    top_rows = eligible_df.iloc[top_idx]
    top_examples = [
        {
            "record_id": str(r["record_id"]),
            "ticker": str(r["ticker"]),
            "timestamp": pd.Timestamp(r["timestamp"]).isoformat(),
            "catalyst_family": str(r.get("catalyst_family", "")),
            "catalyst_subtype": str(r.get("catalyst_subtype", "")),
            "headline": str(r.get("headline", "")),
            "forward_5d_return": float(r["forward_5d_return"]) if pd.notna(r.get("forward_5d_return")) else None,
            "similarity": float(sims[i]),
        }
        for i, (_, r) in zip(top_idx, top_rows.iterrows())
    ]

    return {
        "status": "ok",
        "ticker": ticker.upper(),
        "timestamp_utc": ts.isoformat(),
        "catalyst_family": family,
        "catalyst_subtype": subtype,
        "relation_type": relation,
        "impact_role": impact_role,
        "is_direct_catalyst": float(row.get("is_direct_catalyst", 0.0) or 0.0),
        "finbert_positive_score": finbert.get("finbert_positive_score"),
        "finbert_negative_score": finbert.get("finbert_negative_score"),
        "finbert_neutral_score": finbert.get("finbert_neutral_score"),
        "library_size_eligible": int(len(eligible_df)),
        "neighbor_count": int(len(ranked)),
        "expected_5d_return": expected_5d,
        "expected_max_forward_return": expected_max_fwd,
        "expected_max_drawdown": expected_max_dd,
        "expansion_hit_rate": expansion_hit_rate,
        "winner_similarity_max": winner_sim,
        "loser_similarity_max": loser_sim,
        "edge": edge,
        "top_examples": top_examples,
    }
