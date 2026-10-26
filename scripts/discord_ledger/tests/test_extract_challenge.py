import json

from scripts.discord_ledger.extract_challenge import normalized_messages


def test_jsonl_preserves_unicode_line_separators(tmp_path):
    path = tmp_path / "messages.jsonl"
    row = {"message_id": "123", "text": "first\u2028second\u2029third", "author_id": "ace"}
    path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    messages = normalized_messages(path)
    assert len(messages) == 1
    assert messages[0]["text"] == row["text"]
