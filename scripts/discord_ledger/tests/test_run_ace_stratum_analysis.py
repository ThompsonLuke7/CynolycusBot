from scripts.discord_ledger.run_ace_stratum_analysis import analyze, engine_summary


def _side(symbol: str, detected: bool, confirmed: bool, trades: int) -> dict:
    return {"symbol": symbol, "structure_setup_detected": detected,
            "structure_explicit_confirmation": confirmed, "structure_modeled_trade_count": trades}


def test_engine_summary_compares_paired_rates() -> None:
    pairs = [
        {"source": _side("SPY", True, True, 1), "control": _side("SPY", True, False, 0)},
        {"source": _side("AAPL", True, False, 0), "control": _side("AAPL", True, True, 1)},
    ]

    result = engine_summary(pairs)

    assert result["n"] == 2
    assert result["explicit_confirmation"]["source_rate"] == 0.5
    assert result["explicit_confirmation"]["control_rate"] == 0.5
    assert result["explicit_confirmation"]["paired_exact_p"] == 1.0


def test_analyze_assigns_spy_and_non_spy_strata() -> None:
    pairs = [{"source": _side("SPY", True, True, 1), "control": _side("SPY", True, False, 0)}]
    context = [
        {"symbol": "SPY", "row_kind": "alert", "sample_weight": 1.0, "watchlist_24h": False,
         "watchlist_7d": True, "watchlist_level_7d": False, "watchlist_intention_or_instruction_7d": False,
         "news_24h_has_article": False, "in_eligible_universe": None, "news_24h_article_count": 0,
         "news_24h_direct_catalyst_count": 0, "news_24h_max_record_catalyst_score": None,
         "news_24h_mean_p_bullish": None, "universe_status": "missing_pit_snapshot"},
        {"symbol": "SPY", "row_kind": "control", "sample_weight": 1.0, "watchlist_24h": False,
         "watchlist_7d": False, "watchlist_level_7d": False, "watchlist_intention_or_instruction_7d": False,
         "news_24h_has_article": False, "in_eligible_universe": None, "news_24h_article_count": 0,
         "news_24h_direct_catalyst_count": 0, "news_24h_max_record_catalyst_score": None,
         "news_24h_mean_p_bullish": None, "universe_status": "missing_pit_snapshot"},
    ]

    result = analyze(pairs, context)

    assert result["spy"]["engine"]["n"] == 1
    assert result["non_spy"]["engine"]["n"] == 0
    assert result["spy"]["context"]["alert"]["rows"] == 1
