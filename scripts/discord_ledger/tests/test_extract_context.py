from scripts.discord_ledger.extract_context import ACE_ID, CHANNELS, extract_events


def test_extract_events_emits_each_classified_event_with_source_timestamp() -> None:
    message = {
        "message_id": "100", "author_id": ACE_ID,
        "channel_id": next(iter(CHANNELS)),
        "timestamp_local": "Monday, January 5, 2026 10:00 AM",
        "timestamp_utc": "2026-01-05T15:00:00Z",
        "text": "Watching SPY above 600; added SPY calls.",
        "attachments": [], "reply_to_message_id": None,
    }

    events, counts, candidates = extract_events([message])

    assert candidates == 1
    assert counts["watchlist"] == 1
    assert counts["unknown_management"] == 1
    assert {event["timestamp_utc"] for event in events} == {"2026-01-05T15:00:00Z"}
