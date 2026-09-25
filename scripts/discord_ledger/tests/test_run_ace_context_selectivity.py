import pandas as pd

from scripts.discord_ledger.run_ace_context_selectivity import summarize, watchlist_features


def test_watchlist_features_are_strictly_pre_decision_and_windowed() -> None:
    decision = pd.Timestamp("2026-01-05T15:00:00Z")
    events = [
        {"message_id": "old", "available_at_utc": pd.Timestamp("2025-12-28T15:00:00Z"),
         "availability_basis": "creation", "speech_act": "instruction", "has_level": True},
        {"message_id": "recent", "available_at_utc": pd.Timestamp("2026-01-05T14:00:00Z"),
         "availability_basis": "creation", "speech_act": "intention", "has_level": False},
        {"message_id": "at_decision", "available_at_utc": decision,
         "availability_basis": "creation", "speech_act": "instruction", "has_level": True},
    ]

    result = watchlist_features(events, decision)

    assert result["watchlist_24h"] is True
    assert result["watchlist_7d"] is True
    assert result["watchlist_message_ids_7d"] == ["recent"]
    assert result["watchlist_level_7d"] is False
    assert result["watchlist_intention_or_instruction_7d"] is True


def test_summary_uses_control_weights() -> None:
    rows = [
        {"row_kind": "alert", "sample_weight": 1.0, "watchlist_24h": True, "watchlist_7d": True,
         "watchlist_level_7d": False, "watchlist_intention_or_instruction_7d": False,
         "news_24h_has_article": False, "in_eligible_universe": None, "news_24h_article_count": 0,
         "news_24h_direct_catalyst_count": 0, "news_24h_max_record_catalyst_score": None,
         "news_24h_mean_p_bullish": None, "universe_status": "missing_pit_snapshot"},
        {"row_kind": "control", "sample_weight": 0.25, "watchlist_24h": True, "watchlist_7d": False,
         "watchlist_level_7d": False, "watchlist_intention_or_instruction_7d": False,
         "news_24h_has_article": False, "in_eligible_universe": None, "news_24h_article_count": 0,
         "news_24h_direct_catalyst_count": 0, "news_24h_max_record_catalyst_score": None,
         "news_24h_mean_p_bullish": None, "universe_status": "missing_pit_snapshot"},
        {"row_kind": "control", "sample_weight": 0.75, "watchlist_24h": False, "watchlist_7d": False,
         "watchlist_level_7d": False, "watchlist_intention_or_instruction_7d": False,
         "news_24h_has_article": False, "in_eligible_universe": None, "news_24h_article_count": 0,
         "news_24h_direct_catalyst_count": 0, "news_24h_max_record_catalyst_score": None,
         "news_24h_mean_p_bullish": None, "universe_status": "missing_pit_snapshot"},
    ]

    result = summarize(rows)

    assert result["control"]["effective_weight"] == 1.0
    assert result["control"]["rates"]["watchlist_24h"] == 0.25
    assert result["alert_minus_control"]["rates"]["watchlist_24h"] == 0.75
