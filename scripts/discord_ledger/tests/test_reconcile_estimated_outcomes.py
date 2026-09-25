from __future__ import annotations

from scripts.discord_ledger.reconcile_estimated_outcomes import reconcile, reference, screen_receipt


def test_reference_and_order_receipt_are_not_exact_fill() -> None:
    assert reference(1.5, 3.5) == 133.333
    receipt = screen_receipt("WMT order filled\n1 of 1\n$169.96\nContract sold\nLimit price\n$1.70\nNew position\n0 Contracts")
    assert receipt["screenshot_reports_filled_order"] is True
    assert receipt["screenshot_limit_price"] == 1.7
    assert receipt["screenshot_new_position_zero"] is True
    assert "lower bound" in receipt["interpretation"]


def test_gone_mark_is_evidence_but_not_full_close_result() -> None:
    trade = {"trade_id": "intc", "author_name": "ACE", "author_id": "1079391263083733072",
             "entry_event_id": "entry", "entry_message_id": "alert", "event_ids": ["entry"],
             "symbol": "INTC", "link_confidence": "medium"}
    text = {"trade_id": "intc", "reported_entry_price": 1.5,
            "reported_price_change_legs": [], "complete_source_claimed_quantity_cycle": False,
            "complete_cycle_reported_gross_price_change_pct": None, "source_message_ids": ["alert"]}
    image = {"link_status": "unique_contract_candidate", "candidate_trade_ids": ["intc"],
             "message_id": "exit", "attachment_id": "image", "posted_at_utc": "2026-09-17T15:03:43Z",
             "available_at_utc": "2026-09-17T15:03:43Z", "source_text": "INTC CALLS GONE",
             "action_text_class": "gone_close_claim_needs_review", "source_message_ids": ["alert", "exit"],
             "ocr_contract": {"displayed_option_price": 3.5}}
    ocr = {"message_id": "exit", "attachment_id": "image",
           "ocr_text": "INTC $108 Call\n$3.50\n$3.13 (845.95%) Today"}
    rows, _, summary = reconcile([trade], [text], [image], [ocr], [])
    assert rows[0]["best_close_reference"] is None
    assert rows[0]["evidence"][0]["mark_vs_alert_reference_pct"] == 133.333
    assert summary["entries_with_any_image_management_evidence"] == 1
    assert summary["entries_with_priced_image_mark_vs_alert"] == 1
    assert summary["priced_image_entries_without_close_reference"] == 1


def test_reported_full_close_mark_can_be_estimated() -> None:
    trade = {"trade_id": "aapl", "author_name": "ACE", "author_id": "1079391263083733072",
             "entry_event_id": "entry", "entry_message_id": "alert", "event_ids": ["entry"],
             "symbol": "AAPL", "link_confidence": "low"}
    text = {"trade_id": "aapl", "reported_entry_price": 1.5,
            "reported_price_change_legs": [], "complete_source_claimed_quantity_cycle": False,
            "complete_cycle_reported_gross_price_change_pct": None, "source_message_ids": ["alert"]}
    image = {"link_status": "unique_contract_candidate", "candidate_trade_ids": ["aapl"],
             "message_id": "exit", "attachment_id": "image", "posted_at_utc": "2026-09-11T13:37:08Z",
             "available_at_utc": "2026-09-11T13:37:08Z", "source_text": "all out",
             "action_text_class": "reported_full_close_or_stc_unspecified_quantity", "source_message_ids": ["alert", "exit"],
             "ocr_contract": {"displayed_option_price": 5.13}}
    ocr = {"message_id": "exit", "attachment_id": "image", "ocr_text": "AAPL $327.5 Call\n$5.13\nToday"}
    rows, _, _ = reconcile([trade], [text], [image], [ocr], [])
    assert rows[0]["best_close_reference"]["gross_reference_return_pct"] == 242.0
    assert rows[0]["best_close_reference"]["basis"] == "screenshot_mark_at_reported_full_close_not_fill"
