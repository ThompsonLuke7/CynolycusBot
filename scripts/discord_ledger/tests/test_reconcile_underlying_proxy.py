from __future__ import annotations

from scripts.discord_ledger.reconcile_underlying_proxy import build_rows


def _trade(trade_id: str = "t1") -> dict:
    return {"trade_id": trade_id, "author_name": "ACE", "entry_event_id": "entry",
            "entry_available_at_utc": "2026-01-05T15:00:30Z", "entry_message_id": "m-entry",
            "source_message_ids": ["m-entry"], "symbol": "SPY", "option_type": "call",
            "direction": "long", "strike": 600.0, "expiry": "2026-01-09", "channel_id": "c"}


def _event(event_id: str = "exit") -> dict:
    return {"event_id": event_id, "author_name": "ACE", "event_type": "exit", "trade_id": "t1",
            "available_at_utc": "2026-01-05T16:00:30Z", "timestamp_utc": "2026-01-05T16:00:30Z",
            "message_id": "m-exit", "source_message_ids": ["m-exit"], "source_quote": "SPY calls ALL OUT",
            "symbol": "SPY", "option_type": "call", "strike": 600.0, "expiry": "2026-01-09", "channel_id": "c"}


def test_canonical_close_with_prices_is_directional_proxy_winner() -> None:
    prices = {
        "t1:entry": {"target_id": "t1:entry", "price": 600.0},
        "t1:anchor:exit": {"target_id": "t1:anchor:exit", "price": 603.0},
    }
    rows, targets, detail = build_rows([_trade()], [_event()], prices)
    assert rows[0]["proxy_outcome"] == "proxy_winner"
    assert rows[0]["direction_adjusted_underlying_return_pct"] == 0.5
    assert rows[0]["management_anchor"]["anchor_scope"] == "reported_full_close"
    assert len(targets) == 2
    assert detail["summary"]["management_anchors"] == 1


def test_unlabelled_close_is_not_nearest_position_linked() -> None:
    event = _event()
    event.update({"trade_id": "different", "symbol": None, "source_quote": "ALL OUT",
                  "available_at_utc": "2026-01-08T16:00:30Z", "timestamp_utc": "2026-01-08T16:00:30Z"})
    rows, _, _ = build_rows([_trade()], [event], {})
    assert rows[0]["management_anchor"] is None
    assert rows[0]["outcome_anchor"]["anchor_type"] == "unverified_expiry_horizon"
    assert rows[0]["proxy_outcome"] == "unknown_no_proxy_price"


def test_single_same_channel_full_close_is_retained_as_very_low_estimate() -> None:
    event = _event()
    event.update({"trade_id": "different", "symbol": None, "option_type": None, "strike": None,
                  "expiry": None, "source_quote": "ALL OUT", "available_at_utc": "2026-01-05T16:00:30Z"})
    rows, _, _ = build_rows([_trade()], [event], {})
    assert rows[0]["management_anchor"]["link_confidence"] == "very_low"
    assert "unlabelled_full_close_same_channel_within_36h" in rows[0]["management_anchor"]["link_reasons"]


def test_author_is_explicit_not_hardcoded_to_ace() -> None:
    trade, event = _trade(), _event()
    trade.update({"trade_id": "ft1", "author_name": "FT", "entry_event_id": "entry", "entry_message_id": "m-entry"})
    event.update({"trade_id": "ft1", "author_name": "FT"})
    prices = {"ft1:entry": {"price": 600.0}, "ft1:anchor:exit": {"price": 597.0}}
    rows, _, detail = build_rows([trade], [event], prices, author="FT")
    assert rows[0]["proxy_outcome"] == "proxy_loser"
    assert detail["summary"]["author_name"] == "FT"
