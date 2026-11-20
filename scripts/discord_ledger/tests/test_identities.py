import json

from scripts.discord_ledger import extract_alerts, extract_post_profits
from scripts.discord_ledger.identities import (ACE_ID, ACE_SUCCESSOR_ID, CALLERS, FREE_ALERTS_CHANNEL_ID, FT_ID,
                                               SUCCESSOR_ALERTS_CHANNEL_ID)


def test_successor_account_shares_the_ace_label_only():
    assert CALLERS[ACE_ID] == CALLERS[ACE_SUCCESSOR_ID] == "ACE"
    assert CALLERS[FT_ID] == "FT"


def test_alert_extraction_reads_both_alert_channels(tmp_path):
    rows = [
        {"message_id": "1", "author_id": ACE_ID, "channel_id": FREE_ALERTS_CHANNEL_ID, "text": "SPY 600 calls .50",
         "timestamp_utc": "2026-10-01T14:00:00+00:00"},
        {"message_id": "2", "author_id": ACE_SUCCESSOR_ID, "channel_id": SUCCESSOR_ALERTS_CHANNEL_ID,
         "text": "QQQ 700 puts .80", "timestamp_utc": "2026-10-20T14:00:00+00:00"},
        {"message_id": "3", "author_id": ACE_SUCCESSOR_ID, "channel_id": "999", "text": "chat",
         "timestamp_utc": "2026-10-20T14:00:00+00:00"},
    ]
    path = tmp_path / "messages.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    kept = extract_alerts.read_normalized(path)
    assert [m["message_id"] for m in kept] == ["1", "2"]
    events, _, coverage = extract_alerts.extract(kept)
    assert coverage["target_messages"] == 2


def test_post_profit_symbols_include_successor_trades():
    trades = [{"author_id": ACE_SUCCESSOR_ID, "author_name": "Zach_ace", "symbol": "MRK"},
              {"author_id": "member", "author_name": "someone", "symbol": "ZZZ"}]
    assert extract_post_profits.valid_symbols(trades) == {"MRK"}
