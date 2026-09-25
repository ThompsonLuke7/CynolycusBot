from __future__ import annotations

from scripts.discord_ledger.strategy_audit import outcome_evidence, price_is_premium, watchlist_candidates


def _message(mid: str, text: str, created: str, *, edited: str | None = None) -> dict:
    return {
        "message_id": mid, "author_id": "ACE", "channel_name": "ace-free-watchlists",
        "text": text, "timestamp_utc": created,
        "is_edited": bool(edited), "edited_timestamp_utc": edited,
    }


def test_watchlist_matches_second_symbol_and_excludes_future_edit() -> None:
    entry = {"symbol": "RDDT", "author_id": "ACE", "available_at_utc": "2026-01-05T15:00:00Z"}
    messages = [
        _message("one", "Watchlist: APP 830 calls; RDDT 245 calls", "2026-01-05T14:00:00Z"),
        _message("two", "Watching RDDT", "2026-01-05T13:00:00Z", edited="2026-01-05T15:01:00Z"),
        _message("three", "Watching RDDT", "2026-01-05T15:01:00Z"),
    ]
    matches = watchlist_candidates(entry, messages)
    assert [row["message_id"] for row in matches] == ["one"]
    assert matches[0]["watchlist_date_et"] == "2026-01-05"


def test_percent_claim_is_not_option_price() -> None:
    assert not price_is_premium({"price": 100.0, "price_text": "AT 100", "source_quote": "half out AT 100%"})
    assert price_is_premium({"price": 1.7, "price_text": "at 1.7", "source_quote": "Closed at 1.7"})


def test_complete_claimed_cycle_requires_exact_quantities_and_no_add() -> None:
    entry = {"event_id": "e", "event_type": "entry", "price": 0.8, "price_text": ".8",
             "source_quote": "BTO 2 at .8", "quantity": 2, "speech_act": "claimed_execution",
             "symbol": "WMT", "strike": 124, "option_type": "call", "source_message_ids": ["1"]}
    first = {"event_id": "x1", "event_type": "trim", "price": 1.7, "price_text": "1.7",
             "source_quote": "STC 1 at 1.7", "quantity": 1, "speech_act": "claimed_execution",
             "symbol": "WMT", "strike": 124, "option_type": "call", "source_message_ids": ["2"]}
    second = {**first, "event_id": "x2", "price": 1.95, "price_text": "1.95",
              "source_quote": "STC 1 at 1.95", "source_message_ids": ["3"]}
    trade = {"trade_id": "t", "author_name": "ACE", "symbol": "WMT", "entry_event_id": "e",
             "entry_message_id": "1", "event_ids": ["e", "x1", "x2"],
             "link_confidence": "low", "source_message_ids": ["1", "2", "3"]}
    events = {"e": entry, "x1": first, "x2": second}
    result = outcome_evidence(trade, events)
    assert result["complete_source_claimed_quantity_cycle"] is True
    assert result["complete_cycle_reported_gross_price_change_pct"] == 128.125
    events["x2"]["quantity"] = None
    assert outcome_evidence(trade, events)["complete_source_claimed_quantity_cycle"] is False
