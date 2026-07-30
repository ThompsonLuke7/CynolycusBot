"""Fixture smoke tests for unified catalyst records and features."""

from __future__ import annotations

import pandas as pd

from signals.catalysts.pipeline import build_catalyst_features, build_catalyst_records, build_catalyst_scores


def test_catalyst_records_scores_and_features(tmp_path) -> None:
    news = pd.DataFrame(
        [
            {
                "record_id": "n1",
                "ticker": "RKLB",
                "timestamp": pd.Timestamp("2026-01-01T14:00:00Z"),
                "headline": "RKLB wins contract",
                "summary": "",
                "source": "finnhub",
                "url": "",
                "relation_type": "direct_mention",
                "impact_role": "direct_beneficiary",
                "relation_confidence": 0.9,
                "is_direct_catalyst": 1.0,
            }
        ]
    )
    macro = pd.DataFrame(
        [{"event_type": "cpi", "timestamp": pd.Timestamp("2026-01-02T13:30:00Z"), "title": "CPI", "source": "fixture", "ticker": ""}]
    )
    earnings = pd.DataFrame(
        [{"event_type": "earnings", "timestamp": pd.Timestamp("2026-01-03T21:00:00Z"), "title": "earnings", "source": "fixture", "ticker": "RKLB"}]
    )
    scores = pd.DataFrame(
        [{"record_id": "n1", "ticker": "RKLB", "timestamp": pd.Timestamp("2026-01-01T14:00:00Z"), "news_similarity_score": 0.75, "news_similarity_neighbor_count": 3, "news_similarity_max": 0.9, "realized_news_score": 0.6}]
    )
    news_features = pd.DataFrame(
        [{"timestamp": pd.Timestamp("2026-01-01T15:00:00Z"), "ticker": "RKLB", "news_similarity_score": 0.75, "news_count_24h": 1.0, "direct_news_count_24h": 1.0, "hours_since_news": 1.0, "news_relation_confidence": 0.9, "news_is_direct_catalyst": 1.0}]
    )
    event_features = pd.DataFrame(
        [{"timestamp": pd.Timestamp("2026-01-01T15:00:00Z"), "ticker": "RKLB", "macro_event_next_24h": 1.0, "earnings_next_7d": 1.0}]
    )
    paths = {name: tmp_path / f"{name}.parquet" for name in ["news", "macro", "earnings", "scores", "records", "cat_scores", "news_features", "event_features", "features"]}
    news.to_parquet(paths["news"], index=False)
    macro.to_parquet(paths["macro"], index=False)
    earnings.to_parquet(paths["earnings"], index=False)
    scores.to_parquet(paths["scores"], index=False)
    news_features.to_parquet(paths["news_features"], index=False)
    event_features.to_parquet(paths["event_features"], index=False)

    records = build_catalyst_records(
        news_path=paths["news"],
        macro_path=paths["macro"],
        earnings_path=paths["earnings"],
        earnings_result_features_path=None,
        earnings_result_labels_path=None,
        output_path=paths["records"],
    )
    assert set(records["catalyst_kind"]) == {"news", "scheduled_event", "earnings"}
    cat_scores = build_catalyst_scores(catalyst_path=paths["records"], news_scores_path=paths["scores"], output_path=paths["cat_scores"])
    assert cat_scores["catalyst_score"].max() == 0.75
    features = build_catalyst_features(
        pd.DataFrame([{"timestamp": "2026-01-01T15:00:00Z", "ticker": "RKLB"}]),
        news_features_path=paths["news_features"],
        event_features_path=paths["event_features"],
        catalyst_scores_path=paths["cat_scores"],
        output_path=paths["features"],
    )
    assert features.iloc[0]["catalyst_score"] == 0.5


def run_all() -> None:
    from pathlib import Path
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as directory:
        test_catalyst_records_scores_and_features(Path(directory))
    print("catalysts smoke tests passed")


if __name__ == "__main__":
    run_all()
