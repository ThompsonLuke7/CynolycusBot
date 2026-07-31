"""Small, byte-stable legacy evidence payloads used by importer tests."""

from __future__ import annotations

import json


VALID_SIGNAL_LINE = {
    "event": "signal_decision",
    "module": "meta_ranker",
    "bar": "2026-07-29T18:00:00Z",
    "observed_at": "2026-07-29T18:00:01Z",
    "score": 0.97,
}

INVALID_SIGNAL_LINE = {
    "event": "signal_decision",
    "module": "meta_ranker",
    "bar": "not-a-time",
}


def signal_jsonl_bytes() -> bytes:
    return (
        json.dumps(VALID_SIGNAL_LINE, separators=(",", ":"))
        + "\n"
        + json.dumps(INVALID_SIGNAL_LINE, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def account_snapshot_payload() -> dict[str, object]:
    return {
        "account_alias": "qa-paper",
        "captured_at_utc": "2026-07-29T20:00:00Z",
        "equity": 100000.0,
        "cash": 90000.0,
        "buying_power": 90000.0,
        "positions": [
            {
                "broker_position_id": "pos-1",
                "symbol": "AMD",
                "underlying": "AMD",
                "quantity": 10,
                "current_price": 100.0,
            }
        ],
    }


def managed_state_payload() -> dict[str, object]:
    return {
        "updated_at": "2026-07-29T20:00:00Z",
        "strategy_id": "meta_ranker",
        "positions": [
            {
                "symbol": "AMD",
                "quantity": 10,
                "strategy_id": "meta_ranker",
                "ownership_status": "CONFIRMED",
                "confirmed_ownership": True,
            }
        ],
    }
