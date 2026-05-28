"""Tests for the live trading scoring hook."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from news.config import LABEL_HORIZON_DAYS
from news.live_scoring import LiveScoringConfig, score_headline


def _seed_library(tmp_path) -> dict:
    """Write minimal news_records / news_embeddings / news_labels parquets for live scoring."""
    base_ts = pd.Timestamp("2026-01-01T14:00:00Z")
    # A handful of priors: half winners, half losers, all in the same family/subtype.
    rows = []
    embeddings = []
    labels = []
    for i in range(8):
        rid = f"r{i}"
        is_winner = i % 2 == 0
        # Winners aligned with [1,0,0], losers aligned with [0,1,0].
        emb_vec = [1.0, 0.0, 0.0] if is_winner else [0.0, 1.0, 0.0]
        ts = base_ts + pd.Timedelta(days=i)
        rows.append(
            {
                "record_id": rid,
                "ticker": "RKLB",
                "timestamp": ts,
                "headline": f"RKLB wins contract {i}" if is_winner else f"RKLB shares drop on dilution {i}",
                "summary": "",
                "body": "",
                "url": "",
                "source": "finnhub",
                "source_id": "",
                "text": f"RKLB news {i}",
                "content_hash": rid,
                "relation_type": "direct_mention",
                "impact_role": "direct_beneficiary" if is_winner else "direct_victim",
                "is_direct_catalyst": 1.0,
                "catalyst_family": "contract_partnership",
                "catalyst_subtype": "commercial_deal",
            }
        )
        embeddings.append(
            {
                "record_id": rid,
                "ticker": "RKLB",
                "timestamp": ts,
                "embedding": json.dumps(emb_vec),
                "finbert_positive_score": 0.7 if is_winner else 0.1,
                "finbert_negative_score": 0.1 if is_winner else 0.7,
                "finbert_neutral_score": 0.2,
            }
        )
        labels.append(
            {
                "record_id": rid,
                "ticker": "RKLB",
                "timestamp": ts,
                "forward_5d_return": 0.25 if is_winner else -0.20,
                "max_forward_return": 0.30 if is_winner else -0.05,
                "max_drawdown": -0.05 if is_winner else -0.25,
                "expansion_label": 1.0 if is_winner else 0.0,
            }
        )

    records_path = tmp_path / "news_records.parquet"
    embeddings_path = tmp_path / "news_embeddings.parquet"
    labels_path = tmp_path / "news_labels.parquet"
    pd.DataFrame(rows).to_parquet(records_path, index=False)
    pd.DataFrame(embeddings).to_parquet(embeddings_path, index=False)
    pd.DataFrame(labels).to_parquet(labels_path, index=False)
    # winners / losers paths exist but live_scoring doesn't use them directly.
    return {
        "records": records_path,
        "embeddings": embeddings_path,
        "labels": labels_path,
        "winners": tmp_path / "winners_unused.parquet",
        "losers": tmp_path / "losers_unused.parquet",
    }


def _stub_embed_query(text: str) -> np.ndarray:
    """Tiny test stub: 'win'/'contract' → [1,0,0]; 'drop'/'dilution' → [0,1,0]."""
    text_low = text.lower()
    if any(token in text_low for token in ("win", "contract", "award", "partnership")):
        return np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    if any(token in text_low for token in ("drop", "dilution", "loss", "miss")):
        return np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
    return np.asarray([0.0, 0.0, 1.0], dtype=np.float32)


def test_score_headline_picks_winners_for_positive_headline(tmp_path, monkeypatch) -> None:
    """A 'wins contract' headline should match winner priors and have positive expected return."""
    paths = _seed_library(tmp_path)
    # Clear cached library and patch embeddings model.
    import news.live_scoring as live

    live._LIBRARY_CACHE.clear()
    monkeypatch.setattr(live, "_embed_query", _stub_embed_query)
    monkeypatch.setattr(live, "_finbert_query", lambda text: None)

    config = LiveScoringConfig(
        label_horizon_days=LABEL_HORIZON_DAYS,
        min_similarity=0.30,
        top_k=10,
        library_paths={
            "records": paths["records"],
            "embeddings": paths["embeddings"],
            "labels": paths["labels"],
            "winners": paths["winners"],
            "losers": paths["losers"],
        },
    )
    # Prediction time well after all priors' label horizons have elapsed.
    pred_ts = pd.Timestamp("2026-03-01T15:00:00Z")
    result = score_headline(
        ticker="RKLB",
        timestamp=pred_ts,
        headline="RKLB wins contract",
        source="finnhub",
        config=config,
    )
    assert result["status"] == "ok"
    assert result["catalyst_family"] == "contract_partnership"
    assert result["expected_5d_return"] > 0.0, f"Positive headline must yield positive expected_5d_return, got {result['expected_5d_return']}"
    assert result["winner_similarity_max"] > result["loser_similarity_max"]
    assert result["edge"] > 0
    assert result["neighbor_count"] >= 1


def test_score_headline_excludes_priors_inside_label_horizon(tmp_path, monkeypatch) -> None:
    """If the prediction timestamp is too close to the priors, no records should be eligible."""
    paths = _seed_library(tmp_path)
    import news.live_scoring as live

    live._LIBRARY_CACHE.clear()
    monkeypatch.setattr(live, "_embed_query", _stub_embed_query)
    monkeypatch.setattr(live, "_finbert_query", lambda text: None)

    config = LiveScoringConfig(
        label_horizon_days=LABEL_HORIZON_DAYS,
        library_paths={
            "records": paths["records"],
            "embeddings": paths["embeddings"],
            "labels": paths["labels"],
            "winners": paths["winners"],
            "losers": paths["losers"],
        },
    )
    # Prediction timestamp before any prior's label could be realized.
    pred_ts = pd.Timestamp("2026-01-02T15:00:00Z")
    result = score_headline(
        ticker="RKLB",
        timestamp=pred_ts,
        headline="RKLB wins contract",
        source="finnhub",
        config=config,
    )
    assert result["status"] == "no_priors", f"Expected 'no_priors', got {result['status']}"


def test_score_headline_no_embedding_model_returns_status(tmp_path, monkeypatch) -> None:
    """When sentence-transformers isn't installed, the hook must report it cleanly."""
    paths = _seed_library(tmp_path)
    import news.live_scoring as live

    live._LIBRARY_CACHE.clear()
    monkeypatch.setattr(live, "_embed_query", lambda text: None)
    monkeypatch.setattr(live, "_finbert_query", lambda text: None)
    config = LiveScoringConfig(
        library_paths={
            "records": paths["records"],
            "embeddings": paths["embeddings"],
            "labels": paths["labels"],
            "winners": paths["winners"],
            "losers": paths["losers"],
        },
    )
    result = score_headline(
        ticker="RKLB",
        timestamp=pd.Timestamp("2026-03-01T15:00:00Z"),
        headline="RKLB wins contract",
        source="finnhub",
        config=config,
    )
    assert result["status"] == "no_embedding_model"


def run_all() -> None:
    from pathlib import Path
    from tempfile import TemporaryDirectory
    import pytest

    pytest_args = [
        "-q",
        __file__,
    ]
    raise SystemExit(pytest.main(pytest_args))


if __name__ == "__main__":
    run_all()
