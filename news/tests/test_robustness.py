"""Robustness tests: leak prevention, schema invariants, determinism."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from news.config import LABEL_HORIZON_DAYS
from news.pipeline import build_news_features, label_news_forward_returns
from news.schema import records_from_frame
from news.scoring import build_news_similarity_scores


def _fixture_news(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    return records_from_frame(df, source="fixture")


def test_winner_loser_similarity_respects_label_horizon(tmp_path) -> None:
    """A winner whose 10d label wouldn't be realized by the prediction time must not contribute."""
    pred_ts = pd.Timestamp("2026-01-15T15:00:00Z")
    news_ts = pred_ts - pd.Timedelta(minutes=30)  # news strictly before the prediction bar
    # Stale winner: timestamp >= LABEL_HORIZON_DAYS before news → label realized → admissible.
    # Fresh winner: timestamp 2 days before news → label realized after news → NOT admissible.
    news = records_from_frame(
        pd.DataFrame([{"ticker": "RKLB", "timestamp": news_ts.isoformat(), "headline": "RKLB wins contract"}]),
        source="fixture",
    )
    news_path = tmp_path / "news.parquet"
    news.to_parquet(news_path, index=False)
    emb = pd.DataFrame(
        [
            {
                "record_id": news.iloc[0]["record_id"],
                "ticker": "RKLB",
                "timestamp": news_ts,
                "text": "RKLB wins contract",
                "embedding": json.dumps([1.0, 0.0]),
                "finbert_positive_score": 0.5,
                "finbert_negative_score": 0.0,
                "finbert_neutral_score": 0.5,
                "news_cluster_id": 0.0,
            }
        ]
    )
    emb_path = tmp_path / "emb.parquet"
    emb.to_parquet(emb_path, index=False)

    base_row = {
        "record_id": "wfresh",
        "ticker": "RKLB",
        "timestamp": news_ts - pd.Timedelta(days=2),
        "text": "fresh winner",
        "embedding": json.dumps([1.0, 0.0]),
    }
    fresh_winner = pd.DataFrame([base_row])
    win_path = tmp_path / "win_fresh.parquet"
    fresh_winner.to_parquet(win_path, index=False)
    empty_loser = pd.DataFrame(columns=fresh_winner.columns)
    lose_path = tmp_path / "lose_empty.parquet"
    empty_loser.to_parquet(lose_path, index=False)

    features = build_news_features(
        pd.DataFrame([{"timestamp": pred_ts.isoformat(), "ticker": "RKLB"}]),
        news_path,
        emb_path,
        winner_path=win_path,
        loser_path=lose_path,
        output_path=tmp_path / "features_fresh.parquet",
    )
    assert pd.isna(features.iloc[0]["winner_similarity_max"]), (
        "Fresh winner (label not yet realized) must be excluded from similarity matching"
    )

    stale_row = dict(base_row)
    stale_row["record_id"] = "wstale"
    stale_row["timestamp"] = news_ts - pd.Timedelta(days=LABEL_HORIZON_DAYS + 5)
    stale_winner = pd.DataFrame([stale_row])
    stale_path = tmp_path / "win_stale.parquet"
    stale_winner.to_parquet(stale_path, index=False)

    features = build_news_features(
        pd.DataFrame([{"timestamp": pred_ts.isoformat(), "ticker": "RKLB"}]),
        news_path,
        emb_path,
        winner_path=stale_path,
        loser_path=lose_path,
        output_path=tmp_path / "features_stale.parquet",
    )
    assert not pd.isna(features.iloc[0]["winner_similarity_max"]), (
        "Stale winner (label realized) must be admissible"
    )
    assert features.iloc[0]["winner_similarity_max"] > 0.99


def test_news_similarity_scoring_excludes_unrealized_priors(tmp_path) -> None:
    """``build_news_similarity_scores`` must not consult priors whose labels weren't realized yet."""
    # Three records, all same direction & family. Without leak prevention, record 1
    # would score against record 0 and inherit its tanh-ed forward return even though
    # record 0's label isn't realized until 10 days later.
    base_ts = pd.Timestamp("2026-01-01T14:00:00Z")
    news = pd.DataFrame(
        [
            {
                "record_id": f"r{i}",
                "ticker": "RKLB",
                "timestamp": base_ts + pd.Timedelta(days=i),
                "headline": f"RKLB news {i}",
                "summary": "",
                "body": "",
                "url": "",
                "source": "finnhub",
                "source_id": "",
                "text": f"RKLB news {i}",
                "content_hash": f"h{i}",
                "relation_type": "direct_mention",
                "impact_role": "company_specific",
                "is_direct_catalyst": 1.0,
                "catalyst_family": "company_news",
                "catalyst_subtype": "general_company_news",
            }
            for i in range(3)
        ]
    )
    emb = pd.DataFrame(
        [
            {
                "record_id": f"r{i}",
                "ticker": "RKLB",
                "timestamp": base_ts + pd.Timedelta(days=i),
                "embedding": json.dumps([1.0, 0.0]),
                "finbert_positive_score": 0.5,
                "finbert_negative_score": 0.0,
                "finbert_neutral_score": 0.5,
            }
            for i in range(3)
        ]
    )
    labels = pd.DataFrame(
        [
            {
                "record_id": f"r{i}",
                "ticker": "RKLB",
                "timestamp": base_ts + pd.Timedelta(days=i),
                "forward_5d_return": 0.5,
                "max_forward_return": 0.6,
                "max_drawdown": -0.05,
                "expansion_label": 1.0,
            }
            for i in range(3)
        ]
    )
    n_path = tmp_path / "news.parquet"
    e_path = tmp_path / "emb.parquet"
    l_path = tmp_path / "labels.parquet"
    o_path = tmp_path / "scores.parquet"
    news.to_parquet(n_path, index=False)
    emb.to_parquet(e_path, index=False)
    labels.to_parquet(l_path, index=False)

    out = build_news_similarity_scores(
        news_path=str(n_path), embeddings_path=str(e_path), labels_path=str(l_path), output_path=str(o_path)
    )
    # Records are 1 day apart; the 10d label horizon means none should be able to use
    # the prior records as priors yet → all news_similarity_score must be NaN.
    assert out["news_similarity_score"].isna().all(), (
        f"Expected all NaN scores under leak prevention; got {out['news_similarity_score'].tolist()}"
    )


def test_news_similarity_scoring_uses_prior_once_horizon_elapsed(tmp_path) -> None:
    """When priors are old enough, scoring must successfully use them."""
    base_ts = pd.Timestamp("2026-01-01T14:00:00Z")
    rows = [
        # Two priors well outside the horizon plus one fresh record.
        {"day_offset": 0, "rid": "p1"},
        {"day_offset": 1, "rid": "p2"},
        {"day_offset": LABEL_HORIZON_DAYS + 5, "rid": "current"},
    ]
    news = pd.DataFrame(
        [
            {
                "record_id": r["rid"],
                "ticker": "RKLB",
                "timestamp": base_ts + pd.Timedelta(days=r["day_offset"]),
                "headline": f"RKLB {r['rid']}",
                "summary": "",
                "body": "",
                "url": "",
                "source": "finnhub",
                "source_id": "",
                "text": f"RKLB {r['rid']}",
                "content_hash": r["rid"],
                "relation_type": "direct_mention",
                "impact_role": "company_specific",
                "is_direct_catalyst": 1.0,
                "catalyst_family": "company_news",
                "catalyst_subtype": "general_company_news",
            }
            for r in rows
        ]
    )
    emb = pd.DataFrame(
        [
            {
                "record_id": r["rid"],
                "ticker": "RKLB",
                "timestamp": base_ts + pd.Timedelta(days=r["day_offset"]),
                "embedding": json.dumps([1.0, 0.0]),
                "finbert_positive_score": 0.5,
                "finbert_negative_score": 0.0,
                "finbert_neutral_score": 0.5,
            }
            for r in rows
        ]
    )
    labels = pd.DataFrame(
        [
            {
                "record_id": r["rid"],
                "ticker": "RKLB",
                "timestamp": base_ts + pd.Timedelta(days=r["day_offset"]),
                "forward_5d_return": 0.30,
                "max_forward_return": 0.40,
                "max_drawdown": -0.05,
                "expansion_label": 1.0,
            }
            for r in rows
        ]
    )
    n_path = tmp_path / "news.parquet"
    e_path = tmp_path / "emb.parquet"
    l_path = tmp_path / "labels.parquet"
    o_path = tmp_path / "scores.parquet"
    news.to_parquet(n_path, index=False)
    emb.to_parquet(e_path, index=False)
    labels.to_parquet(l_path, index=False)

    out = build_news_similarity_scores(
        news_path=str(n_path), embeddings_path=str(e_path), labels_path=str(l_path), output_path=str(o_path)
    )
    current_row = out.loc[out["record_id"] == "current"].iloc[0]
    assert pd.notna(current_row["news_similarity_score"])
    assert current_row["news_similarity_neighbor_count"] >= 1


def test_label_forward_returns_use_post_news_bars(tmp_path) -> None:
    """label_news_forward_returns must enter on the first bar strictly after the news timestamp."""
    news = records_from_frame(
        pd.DataFrame(
            [{"ticker": "RKLB", "timestamp": "2026-01-01T14:00:00Z", "headline": "headline"}]
        ),
        source="fixture",
    )
    news_path = tmp_path / "news.parquet"
    news.to_parquet(news_path, index=False)
    bars = pd.DataFrame(
        {
            "ticker": ["RKLB"] * 131,
            "timestamp": pd.date_range("2026-01-01T14:30:00Z", periods=131, freq="30min"),
            "close": np.linspace(10, 30, 131),
        }
    )
    labels = label_news_forward_returns(news_path, bars, output_path=tmp_path / "labels.parquet")
    assert len(labels) == 1
    assert labels.iloc[0]["entry_gap_days"] >= 0.0
    # First bar after 14:00 is 14:30; entry close is bars.close[0] = 10.0.
    # Forward 5d (5 * 13 bars) close = bars.close[64].
    expected = bars["close"].iloc[64] / bars["close"].iloc[0] - 1.0
    assert abs(labels.iloc[0]["forward_5d_return"] - float(expected)) < 1e-6


def run_all() -> None:
    from pathlib import Path
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as d:
        p = Path(d)
        test_winner_loser_similarity_respects_label_horizon(p)
        test_news_similarity_scoring_excludes_unrealized_priors(p)
        test_news_similarity_scoring_uses_prior_once_horizon_elapsed(p)
        test_label_forward_returns_use_post_news_bars(p)
    print("news robustness tests passed")


if __name__ == "__main__":
    run_all()
