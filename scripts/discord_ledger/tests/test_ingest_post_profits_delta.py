from __future__ import annotations

from scripts.discord_ledger.ingest_post_profits_delta import CHANNEL_ID, build


def test_keeps_base_and_adds_only_new_message_ids() -> None:
    base = [{"message_id": "1", "author_id": "a", "author_name": "A", "timestamp_utc": "2026-01-01T00:00:00Z",
             "text": "same", "reply_to_message_id": None, "attachments": []}]
    supplement = [dict(base[0]), {"message_id": "2", "author_id": "b", "author_name": "B", "timestamp_utc": "2026-01-01T00:01:00Z",
                                   "text": "new", "reply_to_message_id": None, "attachments": [], "channel_id": None, "channel_name": None}]
    merged, conflicts, summary = build(base, supplement)
    assert len(merged) == 2
    assert conflicts == []
    assert summary["new_messages_added"] == 1
    assert next(row for row in merged if row["message_id"] == "2")["channel_id"] == CHANNEL_ID
