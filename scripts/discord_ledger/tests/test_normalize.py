import json
from pathlib import Path

from scripts.discord_ledger.normalize import normalize_export, parse_export_timestamp


HTML = """<!doctype html><html><head><title>Guild - alerts</title></head><body>
<div class="preamble"><div class="preamble__entry">Guild</div><div class="preamble__entry">VIP / alerts</div></div>
<div class="chatlog">
<div class="chatlog__message-group">
 <div id="chatlog__message-container-1466557584239759422" class="chatlog__message-container" data-message-id="1466557584239759422"><div class="chatlog__message"><div class="chatlog__message-aside"><img class="chatlog__avatar" src="avatar.png"></div><div class="chatlog__message-primary"><div class="chatlog__header"><span class="chatlog__author" data-user-id="1" title="ace_handle">ACE</span><span class="chatlog__timestamp" title="Thursday, January 29, 2026 1:16 PM">1/29/2026 1:16 PM</span></div><div class="chatlog__content chatlog__markdown">BTO SPY<br>call <span class="chatlog__edited-timestamp" title="Thursday, January 29, 2026 1:20 PM">(edited)</span></div><div class="chatlog__attachment"><a href="https://cdn.example/image.png"><img src="https://cdn.example/image.png" alt="Image attachment" title="Image: image.png (1 KB)"></a></div><div class="chatlog__embed"><div class="chatlog__embed-title">News</div><div class="chatlog__embed-description">context</div></div><div class="chatlog__reactions"><div class="chatlog__reaction" title="rocket"><img class="chatlog__emoji" alt="🚀"><span class="chatlog__reaction-count">2</span></div></div></div></div></div>
 <div id="chatlog__message-container-1466557600000000000" class="chatlog__message-container" data-message-id="1466557600000000000"><div class="chatlog__message"><div class="chatlog__message-aside"><div class="chatlog__short-timestamp" title="Thursday, January 29, 2026 1:16 PM">1:16 PM</div></div><div class="chatlog__message-primary"><div class="chatlog__reply"><div class="chatlog__reply-author" title="ace_handle">ACE</div><div class="chatlog__reply-content"><span class="chatlog__reply-link" onclick="scrollToMessage(event,'1466557584239759422')">BTO SPY call</span></div></div><div class="chatlog__content">Added</div></div></div></div>
</div></div>
<div class="postamble"><div class="postamble__entry">Exported 2 message(s)</div><div class="postamble__entry">Timezone: UTC-5</div></div>
</body></html>"""


def test_normalizes_group_inheritance_reply_and_evidence(tmp_path: Path) -> None:
    source = tmp_path / "exports"
    source.mkdir()
    source_file = source / "Guild - VIP - alerts [123].html"
    source_file.write_text(HTML, encoding="utf-8")
    output = tmp_path / "out"

    manifest = normalize_export(source, output)
    records = [json.loads(line) for line in (output / "messages.jsonl").read_text().splitlines()]

    assert manifest["totals"]["message_count"] == 2
    assert manifest["source_files"][0]["declared_count_matches_normalized"] is True
    assert records[1]["author_id"] == "1"
    assert records[1]["author_name"] == "ACE"
    assert records[1]["reply_to_message_id"] == "1466557584239759422"
    assert records[1]["reply_author"] == "ACE"
    assert records[0]["text"] == "BTO SPY\ncall"
    assert records[0]["is_edited"] is True
    assert records[0]["edited_timestamp_utc"] == "2026-01-29T18:20:00Z"
    assert records[0]["attachments"][0]["filename"] == "image.png"
    assert records[0]["embeds"][0]["title"] == "News"
    assert records[0]["reactions"] == [{"count": 2, "emoji": "🚀", "name": "rocket"}]
    assert records[0]["raw_html_sha256"]
    assert records[0]["timestamp_utc_method"] == "discord_snowflake_creation_time"


def test_empty_export_is_recorded_in_manifest(tmp_path: Path) -> None:
    source = tmp_path / "exports"
    source.mkdir()
    (source / "Guild - VIP - empty [999].html").write_text(
        """<html><head><title>Guild - empty</title></head><body><div class=preamble__entry>Guild</div><div class=preamble__entry>VIP / empty</div><div class=postamble><div class=postamble__entry>Exported 0 message(s)</div><div class=postamble__entry>Timezone: UTC-5</div></div></body></html>""",
        encoding="utf-8",
    )
    output = tmp_path / "out"

    manifest = normalize_export(source, output)

    assert manifest["totals"]["message_count"] == 0
    assert manifest["totals"]["empty_channel_count"] == 1
    assert manifest["source_files"][0]["message_count"] == 0
    assert manifest["source_files"][0]["declared_count_matches_normalized"] is True


def test_rendered_display_time_uses_new_york_dst_not_fixed_utc_minus_five() -> None:
    assert parse_export_timestamp("Wednesday, July 8, 2026 11:47 AM") == "2026-07-08T15:47:00Z"
    assert parse_export_timestamp("Thursday, January 29, 2026 1:20 PM") == "2026-01-29T18:20:00Z"
