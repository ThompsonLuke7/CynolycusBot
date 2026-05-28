"""Fixture smoke tests for unscheduled catalyst news."""

from __future__ import annotations

import json
import builtins

import numpy as np
import pandas as pd

from news.dedup import deduplicate_news
from news.nlp import embed_texts_bge, finbert_scores
from news.pipeline import (
    build_news_features,
    build_winner_loser_libraries,
    label_news_forward_returns,
    max_cosine_similarity,
)
from news.schema import records_from_frame


def test_news_dedup_removes_duplicate_url_and_content() -> None:
    raw = pd.DataFrame(
        [
            {"ticker": "RKLB", "timestamp": "2026-05-01T12:00:00Z", "headline": "RKLB wins contract", "url": "https://x/news?a=1"},
            {"ticker": "RKLB", "timestamp": "2026-05-01T12:00:00Z", "headline": "RKLB wins contract", "url": "https://x/news?a=2"},
        ]
    )
    news = records_from_frame(raw, source="fixture")
    assert len(deduplicate_news(news)) == 1


def test_optional_nlp_wrappers_degrade_cleanly() -> None:
    original_import = builtins.__import__

    def blocked_import(name, *args, **kwargs):
        if name in {"sentence_transformers", "transformers"}:
            raise ImportError(name)
        return original_import(name, *args, **kwargs)

    builtins.__import__ = blocked_import
    try:
        try:
            embed_texts_bge(["contract win"])
        except ImportError as exc:
            assert "sentence-transformers" in str(exc)
        try:
            finbert_scores("strong demand")
        except ImportError as exc:
            assert "transformers" in str(exc)
    finally:
        builtins.__import__ = original_import


def test_similarity_and_libraries(tmp_path) -> None:
    emb = pd.DataFrame(
        [
            {"record_id": "w", "ticker": "RKLB", "timestamp": pd.Timestamp("2026-01-01", tz="UTC"), "embedding": json.dumps([1.0, 0.0])},
            {"record_id": "l", "ticker": "RKLB", "timestamp": pd.Timestamp("2026-01-02", tz="UTC"), "embedding": json.dumps([0.0, 1.0])},
        ]
    )
    labels = pd.DataFrame(
        [
            {"record_id": "w", "ticker": "RKLB", "timestamp": pd.Timestamp("2026-01-01", tz="UTC"), "expansion_label": 1.0},
            {"record_id": "l", "ticker": "RKLB", "timestamp": pd.Timestamp("2026-01-02", tz="UTC"), "expansion_label": 0.0},
        ]
    )
    emb_path = tmp_path / "emb.parquet"
    labels_path = tmp_path / "labels.parquet"
    win_path = tmp_path / "win.parquet"
    lose_path = tmp_path / "lose.parquet"
    emb.to_parquet(emb_path, index=False)
    labels.to_parquet(labels_path, index=False)
    winners, losers = build_winner_loser_libraries(emb_path, labels_path, winner_path=win_path, loser_path=lose_path)
    assert len(winners) == 1
    assert len(losers) == 1
    assert max_cosine_similarity(np.asarray([1.0, 0.0]), [np.asarray([1.0, 0.0])]) == 1.0


def test_news_labels_and_features_are_time_aligned(tmp_path) -> None:
    news = records_from_frame(
        pd.DataFrame(
            [{"ticker": "RKLB", "timestamp": "2026-01-01T14:00:00Z", "headline": "RKLB wins contract"}]
        ),
        source="fixture",
    )
    news_path = tmp_path / "news.parquet"
    news.to_parquet(news_path, index=False)
    bars = pd.DataFrame(
        {
            "ticker": ["RKLB"] * 131,
            "timestamp": pd.date_range("2026-01-01T14:30:00Z", periods=131, freq="30min"),
            "close": np.linspace(10, 20, 131),
        }
    )
    labels = label_news_forward_returns(news_path, bars, output_path=tmp_path / "labels.parquet")
    assert labels.iloc[0]["forward_5d_return"] > 0
    emb = pd.DataFrame(
        [
            {
                "record_id": news.iloc[0]["record_id"],
                "ticker": "RKLB",
                "timestamp": pd.Timestamp("2026-01-01T14:00:00Z"),
                "text": "RKLB wins contract",
                "embedding": json.dumps([1.0, 0.0]),
                "finbert_positive_score": 0.8,
                "finbert_negative_score": 0.1,
                "finbert_neutral_score": 0.1,
                "news_cluster_id": 2.0,
            }
        ]
    )
    emb_path = tmp_path / "emb.parquet"
    emb.to_parquet(emb_path, index=False)
    # Priors must sit >= LABEL_HORIZON_DAYS before the prediction timestamp, otherwise
    # their forward-return label wouldn't be realized in time to score against.
    win = emb.copy()
    win["record_id"] = "prior_winner"
    win["timestamp"] = pd.Timestamp("2025-12-15T14:00:00Z")
    lose = emb.copy()
    lose["record_id"] = "prior_loser"
    lose["timestamp"] = pd.Timestamp("2025-12-15T14:00:00Z")
    lose["embedding"] = json.dumps([0.0, 1.0])
    win_path = tmp_path / "win.parquet"
    lose_path = tmp_path / "lose.parquet"
    win.to_parquet(win_path, index=False)
    lose.to_parquet(lose_path, index=False)
    features = build_news_features(
        pd.DataFrame([{"timestamp": "2026-01-01T15:00:00Z", "ticker": "RKLB"}]),
        news_path,
        emb_path,
        winner_path=win_path,
        loser_path=lose_path,
        output_path=tmp_path / "features.parquet",
    )
    assert features.iloc[0]["news_count_24h"] == 1.0
    assert features.iloc[0]["news_edge_score"] > 0


def run_all() -> None:
    from tempfile import TemporaryDirectory
    from pathlib import Path

    test_news_dedup_removes_duplicate_url_and_content()
    test_optional_nlp_wrappers_degrade_cleanly()
    with TemporaryDirectory() as d:
        p = Path(d)
        test_similarity_and_libraries(p)
        test_news_labels_and_features_are_time_aligned(p)
    print("news smoke tests passed")


if __name__ == "__main__":
    run_all()
