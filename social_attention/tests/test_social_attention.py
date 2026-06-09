from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from social_attention.attention_features import build_attention_features
from social_attention.embeddings import build_social_embeddings
from social_attention.labels import build_social_labels
from social_attention.narrative_clustering import build_narrative_features, cluster_social_embeddings
from social_attention.praw_enrichment import enrich_posts_with_praw
from social_attention.pullpush_client import PullPushClient
from social_attention.sentiment import score_reddit_sentiment
from social_attention.ticker_extractor import extract_mentions, extract_ticker_hits


class _Response:
    def __init__(self, payload: dict, status_code: int = 200) -> None:
        self.payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)

    def json(self) -> dict:
        return self.payload


class _Session:
    def __init__(self) -> None:
        self.calls = 0

    def get(self, url, params=None, timeout=None):
        self.calls += 1
        after = int(params["after"])
        if after <= 100:
            return _Response({"data": [{"id": "a", "created_utc": 101}, {"id": "b", "created_utc": 102}]})
        if after == 103:
            return _Response({"data": [{"id": "c", "created_utc": 104}]})
        return _Response({"data": []})


def test_pullpush_iter_search_paginates_by_created_utc() -> None:
    client = PullPushClient(session=_Session(), sleep_seconds=0.0)
    rows = list(client.iter_search(kind="comment", subreddit="stocks", after=100, before=200))
    assert [row["id"] for row in rows] == ["a", "b", "c"]


def test_praw_enrichment_with_mock_reddit() -> None:
    posts = pd.DataFrame(
        [{"post_id": "reddit_submission:abc", "reddit_id": "abc", "kind": "submission"}]
    )

    class FakeReddit:
        def submission(self, id):
            return SimpleNamespace(
                score=42,
                num_comments=7,
                upvote_ratio=0.91,
                permalink="/r/stocks/comments/abc/x/",
                author=SimpleNamespace(name="tester"),
                selftext="hello",
            )

    out = enrich_posts_with_praw(posts, reddit=FakeReddit())
    assert out.iloc[0]["praw_score"] == 42
    assert out.iloc[0]["praw_num_comments"] == 7
    assert out.iloc[0]["praw_author"] == "tester"


def test_ticker_extraction_prefers_cashtags_and_blocks_ambiguous_bare(tmp_path) -> None:
    universe = {"AI", "ASTS", "NVDA", "ON"}
    text = "I like $AI and ASTS with NVDA, but ON is just a word here."
    hits = extract_ticker_hits(text, universe=universe)
    by_ticker = {row["ticker"]: row["methods"] for row in hits}
    assert by_ticker["AI"] == "cashtag"
    assert by_ticker["ASTS"] == "bare"
    assert by_ticker["NVDA"] == "bare"
    assert "ON" not in by_ticker

    posts = pd.DataFrame(
        [{"post_id": "p1", "title": "Rocket Lab news", "selftext": "", "body": "", "text": "Rocket Lab news"}]
    )
    mentions = extract_mentions(
        posts,
        output_path=tmp_path / "mentions.parquet",
        universe={"RKLB"},
        alias_map={"RKLB": {"rocket lab"}},
    )
    assert mentions.iloc[0]["ticker"] == "RKLB"
    assert mentions.iloc[0]["methods"] == "alias"


def test_attention_features_counts_engagement_and_rank_change(tmp_path) -> None:
    posts = pd.DataFrame(
        [
            {
                "post_id": "p1",
                "timestamp": "2026-01-01T10:05:00Z",
                "kind": "submission",
                "author": "a",
                "score": 9,
                "num_comments": 4,
                "sentiment_score": 0.2,
            },
            {
                "post_id": "p2",
                "timestamp": "2026-01-01T10:10:00Z",
                "kind": "comment",
                "author": "b",
                "score": 3,
                "num_comments": 0,
                "sentiment_score": -0.1,
            },
            {
                "post_id": "p3",
                "timestamp": "2026-01-02T10:05:00Z",
                "kind": "comment",
                "author": "a",
                "score": 2,
                "num_comments": 0,
                "sentiment_score": 0.4,
            },
        ]
    )
    mentions = pd.DataFrame(
        [
            {"post_id": "p1", "ticker": "ASTS"},
            {"post_id": "p2", "ticker": "ASTS"},
            {"post_id": "p3", "ticker": "ASTS"},
        ]
    )
    features = build_attention_features(
        posts,
        mentions,
        output_path=tmp_path / "features.parquet",
        complete_grid=True,
    )
    first = features.loc[features["timestamp"].eq(pd.Timestamp("2026-01-01T10:00:00Z"))].iloc[0]
    second = features.loc[features["timestamp"].eq(pd.Timestamp("2026-01-02T10:00:00Z"))].iloc[0]
    assert first["mentions_1h"] == 2
    assert first["unique_authors"] == 2
    assert first["engagement_score"] > 0
    assert pd.notna(second["rank_24h_ago"])


def test_sentiment_degrades_cleanly(tmp_path, monkeypatch) -> None:
    posts = pd.DataFrame([{"post_id": "p1", "text": "great earnings"}])

    def boom(*args, **kwargs):
        raise ImportError("no model")

    monkeypatch.setattr("social_attention.sentiment.finbert_scores_batch", boom)
    out = score_reddit_sentiment(posts, output_path=tmp_path / "posts.parquet")
    assert out.iloc[0]["finbert_available"] == 0.0
    assert "sentiment_score" in out.columns


def test_embeddings_degrade_cleanly(tmp_path, monkeypatch) -> None:
    posts = pd.DataFrame([{"post_id": "p1", "timestamp": "2026-01-01T00:00:00Z", "text": "hello"}])

    def boom(*args, **kwargs):
        raise ImportError("no model")

    monkeypatch.setattr("social_attention.embeddings.embed_texts_bge", boom)
    out = build_social_embeddings(posts, output_path=tmp_path / "emb.parquet")
    assert out.iloc[0]["embedding_available"] == 0.0


def test_hdbscan_and_narrative_features(tmp_path) -> None:
    pytest.importorskip("sklearn")
    posts = pd.DataFrame(
        [
            {"post_id": "p1", "timestamp": "2026-01-01T10:00:00Z", "text": "space contract rocket win", "subreddit": "stocks"},
            {"post_id": "p2", "timestamp": "2026-01-01T10:05:00Z", "text": "rocket contract space award", "subreddit": "stocks"},
            {"post_id": "p3", "timestamp": "2026-01-01T11:00:00Z", "text": "chips ai datacenter demand", "subreddit": "stocks"},
            {"post_id": "p4", "timestamp": "2026-01-01T11:05:00Z", "text": "datacenter ai chip demand", "subreddit": "stocks"},
        ]
    )
    emb = pd.DataFrame(
        [
            {"post_id": "p1", "timestamp": posts.iloc[0]["timestamp"], "subreddit": "stocks", "text": posts.iloc[0]["text"], "embedding": json.dumps([1.0, 0.0])},
            {"post_id": "p2", "timestamp": posts.iloc[1]["timestamp"], "subreddit": "stocks", "text": posts.iloc[1]["text"], "embedding": json.dumps([0.99, 0.01])},
            {"post_id": "p3", "timestamp": posts.iloc[2]["timestamp"], "subreddit": "stocks", "text": posts.iloc[2]["text"], "embedding": json.dumps([0.0, 1.0])},
            {"post_id": "p4", "timestamp": posts.iloc[3]["timestamp"], "subreddit": "stocks", "text": posts.iloc[3]["text"], "embedding": json.dumps([0.01, 0.99])},
        ]
    )
    posts_path = tmp_path / "posts.parquet"
    emb_path = tmp_path / "emb.parquet"
    posts.to_parquet(posts_path, index=False)
    emb.to_parquet(emb_path, index=False)
    clustered = cluster_social_embeddings(
        emb,
        posts_path=posts_path,
        output_path=emb_path,
        clusters_path=tmp_path / "clusters.parquet",
        min_cluster_size=2,
        min_samples=1,
        min_text_chars=3,
    )
    assert clustered["narrative_cluster_id"].ge(0).any()

    mentions = pd.DataFrame(
        [{"post_id": "p1", "ticker": "RKLB"}, {"post_id": "p2", "ticker": "RKLB"}, {"post_id": "p3", "ticker": "NVDA"}]
    )
    mentions_path = tmp_path / "mentions.parquet"
    mentions.to_parquet(mentions_path, index=False)
    features = build_narrative_features(
        posts_path=posts_path,
        mentions_path=mentions_path,
        embeddings_path=emb_path,
        output_path=tmp_path / "narrative.parquet",
    )
    assert {"top_narrative_cluster_id", "narrative_mentions_1h", "ticker_narrative_concentration"}.issubset(features.columns)


def test_label_alignment_uses_next_momentum_timestamp(tmp_path) -> None:
    attention = pd.DataFrame(
        [
            {"ticker": "RKLB", "timestamp": "2026-01-01T10:00:00Z", "mentions_1h": 3},
            {"ticker": "RKLB", "timestamp": "2026-01-01T14:00:00Z", "mentions_1h": 1},
        ]
    )
    labels = pd.DataFrame(
        [
            {"ticker": "RKLB", "timestamp": "2026-01-01T09:30:00Z", "expansion_target": 0.0, "expansion_score": 0.1},
            {"ticker": "RKLB", "timestamp": "2026-01-01T13:30:00Z", "expansion_target": 1.0, "expansion_score": 0.9},
        ]
    )
    attention_path = tmp_path / "attention.parquet"
    labels_path = tmp_path / "labels.parquet"
    attention.to_parquet(attention_path, index=False)
    labels.to_parquet(labels_path, index=False)
    out = build_social_labels(
        attention_path=attention_path,
        narrative_path=tmp_path / "missing.parquet",
        momentum_labels_path=labels_path,
        labels_output_path=tmp_path / "social_labels.parquet",
        matrix_output_path=tmp_path / "matrix.parquet",
    )
    first = out.loc[out["timestamp"].eq(pd.Timestamp("2026-01-01T10:00:00Z"))].iloc[0]
    second = out.loc[out["timestamp"].eq(pd.Timestamp("2026-01-01T14:00:00Z"))]
    assert first["label_timestamp"] == pd.Timestamp("2026-01-01T13:30:00Z")
    assert first["social_spike_success"] == 1.0
    assert second["social_spike_success"].isna().all()

