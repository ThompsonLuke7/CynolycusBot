from __future__ import annotations

from scripts.discord_ledger.index_screenshot_returns import extract


def test_ocr_return_is_snapshot_not_closed_trade() -> None:
    row = {"author_id": "1079391263083733072", "message_id": "123",
           "attachment_id": "a", "sha256": "hash", "local_path": "image.png",
           "source_message_ids": ["123"],
           "ocr_text": "Your position\nAverage cost\n$1.15\nTotal return\n-$2.00 (-1.74%)"}
    parsed = extract(row)
    assert parsed is not None
    assert parsed["ocr_total_return_percent"] == -1.74
    assert parsed["sign_class"] == "negative"
    assert parsed["snapshot_type"] == "open_position_screen"
