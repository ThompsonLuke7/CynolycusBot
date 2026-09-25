from __future__ import annotations

from scripts.discord_ledger.recover_image_management import action_kind, candidates_for, screenshot_contracts


def test_intc_screenshot_current_price_is_not_sale_fill() -> None:
    ocr = {"attachment_id": "image", "sha256": "hash",
           "ocr_text": "View INTC\nINTC $110.49 (+9.34%)\nINTC $108 Call\n$3.50\n$3.13 (845.95%)\nToday"}
    message = {"message_id": "exit", "author_id": "1079391263083733072",
               "timestamp_utc": "2026-09-17T15:03:43Z", "is_edited": False,
               "text": "INTC CALLS GONE @everyone"}
    trade = {"trade_id": "intc", "author_id": message["author_id"], "entry_event_id": "entry",
             "entry_message_id": "alert", "entry_available_at_utc": "2026-09-11T19:39:48Z",
             "symbol": "INTC", "strike": 108.0, "option_type": "call", "expiry": "2026-09-18"}
    rows = candidates_for(message, ocr, [trade])
    assert len(rows) == 1
    assert rows[0]["candidate_entry_message_ids"] == ["alert"]
    assert rows[0]["ocr_contract"]["displayed_option_price"] == 3.5
    assert rows[0]["ocr_contract"]["price_kind"] == "screenshot_displayed_current_price_not_fill"
    assert rows[0]["action_text_class"] == "gone_close_claim_needs_review"


def test_hold_language_does_not_become_confirmed_exit() -> None:
    assert action_kind("XLE GONE 2.52 I ain't selling this") == "ambiguous_gone_or_price_movement"
    assert action_kind("MSFT 615% all out") == "reported_full_close_or_stc_unspecified_quantity"
    assert screenshot_contracts("MSFT $505 Call\n$8.08\n$4.90 Today")[0]["displayed_option_price"] == 8.08


def test_stale_unspecified_expiry_is_not_linked() -> None:
    ocr = {"attachment_id": "image", "sha256": "hash", "ocr_text": "SPY $685 Call\n$1.80"}
    message = {"message_id": "exit", "author_id": "1079391263083733072",
               "timestamp_utc": "2026-03-04T15:23:31Z", "is_edited": False, "text": "I'm out"}
    old = {"trade_id": "old", "author_id": message["author_id"], "entry_event_id": "entry",
           "entry_message_id": "alert", "entry_available_at_utc": "2026-01-21T15:00:00Z",
           "symbol": "SPY", "strike": 685.0, "option_type": "call", "expiry": None}
    result = candidates_for(message, ocr, [old])
    assert result[0]["candidate_trade_ids"] == []
