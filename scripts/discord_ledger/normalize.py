#!/usr/bin/env python3
"""Normalize DiscordChatExporter HTML into evidence-preserving JSONL.

The input HTML is read only.  This module deliberately does not make network
requests: attachment and embed URLs remain evidence references, not downloads.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import os
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo

from bs4 import BeautifulSoup, Tag


DISCORD_EPOCH_MS = 1_420_070_400_000
FIXED_EXPORT_OFFSET = timezone(timedelta(hours=-5), name="UTC-5")
DISPLAY_ZONE = ZoneInfo("America/New_York")
MESSAGE_ID_RE = re.compile(r"chatlog__message-container-(\d+)")
CHANNEL_ID_RE = re.compile(r"\[(\d+)\]\.html$")
SCROLL_TO_MESSAGE_RE = re.compile(r"scrollToMessage\(event,\s*['\"](\d+)")
FILE_TITLE_RE = re.compile(r"^(?P<guild>.+?)\s+-\s+(?P<channel>.+)$")


@dataclass(frozen=True)
class ExportContext:
    path: Path
    source_file: str
    file_sha256: str
    file_size_bytes: int
    channel_id: str | None
    channel_name: str
    guild_name: str | None
    preamble_entries: list[str]
    footer_entries: list[str]
    timezone_footer: str | None
    declared_exported_message_count: int | None


def normalize_space(value: str | None) -> str | None:
    if value is None:
        return None
    result = re.sub(r"[ \t]+", " ", value.replace("\xa0", " ")).strip()
    return result or None


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def classes(tag: Tag | None) -> set[str]:
    if tag is None:
        return set()
    return set(tag.get("class", []))


def tag_text(tag: Tag | None, *, exclude_classes: Iterable[str] = ()) -> str | None:
    """Return readable visible text while retaining intentional line breaks."""
    if tag is None:
        return None
    excluded = set(exclude_classes)
    pieces: list[str] = []

    def visit(node: Any) -> None:
        if isinstance(node, str):
            pieces.append(node)
            return
        if not isinstance(node, Tag) or classes(node) & excluded:
            return
        if node.name == "br":
            pieces.append("\n")
            return
        if node.name == "img" and node.get("alt"):
            pieces.append(str(node["alt"]))
            return
        for child in node.children:
            visit(child)
        if node.name in {"p", "div", "li", "blockquote", "pre"}:
            pieces.append("\n")

    visit(tag)
    text = "".join(pieces).replace("\r\n", "\n").replace("\r", "\n")
    # The exporter places indentation in some emoji tags.  Trim it per line,
    # while preserving intentionally blank Markdown lines.
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text or None


def snowflake_timestamp_utc(message_id: str | None) -> str | None:
    if not message_id or not message_id.isdigit():
        return None
    milliseconds = (int(message_id) >> 22) + DISCORD_EPOCH_MS
    return datetime.fromtimestamp(milliseconds / 1000, tz=timezone.utc).isoformat(
        timespec="milliseconds"
    ).replace("+00:00", "Z")


def parse_export_timestamp(raw: str | None) -> str | None:
    """Parse local time using the zone inferred from snowflake crosschecks.

    The footer says UTC-5, but summer titles match America/New_York daylight
    time instead.  A fall-back hour with two possible UTC times stays unknown.
    """
    if not raw:
        return None
    raw = normalize_space(raw)
    if not raw:
        return None
    for pattern in ("%A, %B %d, %Y %I:%M %p", "%m/%d/%Y %I:%M %p"):
        try:
            wall = datetime.strptime(raw, pattern)
            first = wall.replace(tzinfo=DISPLAY_ZONE, fold=0)
            second = wall.replace(tzinfo=DISPLAY_ZONE, fold=1)
            if first.utcoffset() != second.utcoffset():
                return None
            parsed = first
            return parsed.astimezone(timezone.utc).isoformat(timespec="seconds").replace(
                "+00:00", "Z"
            )
        except ValueError:
            continue
    return None


def raw_container_html(source: str, message_id: str) -> str | None:
    """Slice the original HTML for one container without serializing it anew."""
    needle = f"chatlog__message-container-{message_id}"
    found = source.find(needle)
    if found < 0:
        return None
    start = source.rfind("<div", 0, found)
    if start < 0:
        return None
    depth = 0
    for match in re.finditer(r"</?div\b[^>]*>", source[start:], flags=re.IGNORECASE):
        token = match.group(0).lower()
        if token.startswith("</"):
            depth -= 1
            if depth == 0:
                return source[start : start + match.end()]
        elif not token.rstrip().endswith("/>"):
            depth += 1
    return None


def export_context(path: Path, raw: bytes, soup: BeautifulSoup) -> ExportContext:
    source_file = str(path).replace("\\", "/")
    match = CHANNEL_ID_RE.search(path.name)
    channel_id = match.group(1) if match else None
    title = tag_text(soup.title)
    title_match = FILE_TITLE_RE.match(title or "")
    entries = [tag_text(entry) or "" for entry in soup.select(".preamble__entry")]
    footer_entries = [tag_text(entry) or "" for entry in soup.select(".postamble__entry")]
    channel_name = (
        entries[1].split(" / ", 1)[-1]
        if len(entries) > 1 and " / " in entries[1]
        else (title_match.group("channel") if title_match else path.stem)
    )
    timezone_footer = next(
        (entry.split(":", 1)[1].strip() for entry in footer_entries if entry.startswith("Timezone:")),
        None,
    )
    exported_entry = next((entry for entry in footer_entries if entry.startswith("Exported ")), None)
    exported_match = re.search(r"Exported\s+([\d,]+)\s+message", exported_entry or "")
    return ExportContext(
        path=path,
        source_file=source_file,
        file_sha256=sha256_bytes(raw),
        file_size_bytes=len(raw),
        channel_id=channel_id,
        channel_name=channel_name,
        guild_name=entries[0] if entries else (title_match.group("guild") if title_match else None),
        preamble_entries=entries,
        footer_entries=footer_entries,
        timezone_footer=timezone_footer,
        declared_exported_message_count=(
            int(exported_match.group(1).replace(",", "")) if exported_match else None
        ),
    )


def attachment_metadata(container: Tag) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[tuple[str | None, str | None, str]] = set()
    selectors = ((".chatlog__attachment", "attachment"), (".chatlog__forwarded-attachments", "forwarded_attachment"))
    for selector, kind in selectors:
        for outer in container.select(selector):
            for anchor in outer.select("a[href]"):
                media = anchor.select_one("img, video, source")
                url = anchor.get("href")
                src = media.get("src") if media else None
                key = (url, src, kind)
                if key in seen:
                    continue
                seen.add(key)
                title = media.get("title") if media else None
                filename = None
                if title:
                    name_match = re.search(r"(?:Image|Video|File):\s*(.*?)\s*(?:\([^)]*\))?$", title)
                    filename = name_match.group(1) if name_match else None
                records.append(
                    {
                        "source_kind": kind,
                        "url": url,
                        "media_url": src,
                        "alt": media.get("alt") if media else None,
                        "title": title,
                        "filename": filename,
                        "is_spoiler": "chatlog__attachment--hidden" in classes(outer),
                    }
                )
    return records


def embed_metadata(container: Tag) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for embed in container.select(".chatlog__embed"):
        fields = []
        for field in embed.select(".chatlog__embed-field"):
            fields.append(
                {
                    "name": tag_text(field.select_one(".chatlog__embed-field-name")),
                    "value": tag_text(field.select_one(".chatlog__embed-field-value")),
                    "inline": "chatlog__embed-field--inline" in classes(field),
                }
            )
        links = [anchor.get("href") for anchor in embed.select("a[href]")]
        images = [image.get("src") for image in embed.select("img[src], video[src], source[src]")]
        records.append(
            {
                "text": tag_text(embed),
                "author": tag_text(embed.select_one(".chatlog__embed-author")),
                "title": tag_text(embed.select_one(".chatlog__embed-title")),
                "description": tag_text(embed.select_one(".chatlog__embed-description")),
                "footer": tag_text(embed.select_one(".chatlog__embed-footer")),
                "fields": fields,
                "urls": links,
                "media_urls": images,
            }
        )
    return records


def reaction_metadata(container: Tag) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for reaction in container.select(".chatlog__reaction"):
        count_text = tag_text(reaction.select_one(".chatlog__reaction-count"))
        try:
            count = int(count_text) if count_text else None
        except ValueError:
            count = None
        emoji = reaction.select_one("img.chatlog__emoji")
        records.append(
            {
                "name": reaction.get("title"),
                "emoji": emoji.get("alt") if emoji else tag_text(reaction),
                "count": count,
            }
        )
    return records


def forwarded_metadata(container: Tag) -> list[dict[str, Any]]:
    """Preserve the displayed content/timing of Discord forwarded-message cards."""
    records: list[dict[str, Any]] = []
    for forwarded in container.select(".chatlog__forwarded"):
        timestamp = forwarded.select_one(".chatlog__forwarded-timestamp [title]")
        raw_timestamp = timestamp.get("title") if timestamp else None
        records.append(
            {
                "text": tag_text(forwarded.select_one(".chatlog__forwarded-content")),
                "originally_sent_timestamp_raw": raw_timestamp,
                "originally_sent_timestamp_utc": parse_export_timestamp(raw_timestamp),
                "originally_sent_timestamp_utc_method": (
                    "inferred_America_New_York_from_snowflake_crosscheck" if raw_timestamp else None
                ),
            }
        )
    return records


def reply_metadata(container: Tag) -> tuple[str | None, str | None, str | None, dict[str, Any] | None]:
    reply = container.select_one(".chatlog__reply")
    if reply is None:
        return None, None, None, None
    link = reply.select_one(".chatlog__reply-link")
    target_id = None
    if link:
        onclick = link.get("onclick", "")
        match = SCROLL_TO_MESSAGE_RE.search(onclick)
        target_id = match.group(1) if match else None
    author = reply.select_one(".chatlog__reply-author")
    content = reply.select_one(".chatlog__reply-content")
    return (
        target_id,
        tag_text(author),
        tag_text(content),
        {
            "reply_author_title": author.get("title") if author else None,
            "reply_author_id": author.get("data-user-id") if author else None,
            "reply_link_onclick": link.get("onclick") if link else None,
            "reply_edited_timestamp_raw": tag_text(reply.select_one(".chatlog__reply-edited-timestamp")),
        },
    )


def message_record(
    container: Tag,
    context: ExportContext,
    raw_source: str,
    group_index: int,
    index_in_group: int,
    inherited_author: tuple[str | None, str | None] | None,
) -> dict[str, Any]:
    message_id = container.get("data-message-id")
    header_author = container.select_one(".chatlog__header .chatlog__author")
    current_author = (
        header_author.get("data-user-id") if header_author else None,
        tag_text(header_author) if header_author else None,
    )
    author_id, author_name = current_author if header_author else (inherited_author or (None, None))
    timestamp = container.select_one(".chatlog__header .chatlog__timestamp")
    short_timestamp = container.select_one(".chatlog__short-timestamp")
    timestamp_raw = timestamp.get("title") if timestamp else (short_timestamp.get("title") if short_timestamp else None)
    edited = container.select_one(".chatlog__edited-timestamp")
    edited_timestamp_raw = edited.get("title") if edited else None
    reply_id, reply_author, reply_text, reply_original = reply_metadata(container)
    content = container.select_one(".chatlog__content")
    forwarded = forwarded_metadata(container)
    original = raw_container_html(raw_source, message_id or "")
    snowflake_utc = snowflake_timestamp_utc(message_id)
    rendered_utc = parse_export_timestamp(timestamp_raw)
    return {
        "message_id": message_id,
        "channel_id": context.channel_id,
        "channel_name": context.channel_name,
        "source_file": context.source_file,
        "source_anchor": f"#chatlog__message-container-{message_id}" if message_id else None,
        "author_id": author_id,
        "author_name": author_name,
        "timestamp_raw": timestamp_raw,
        "timestamp_utc": snowflake_utc,
        "timestamp_utc_method": "discord_snowflake_creation_time" if snowflake_utc else None,
        "edited_timestamp_raw": edited_timestamp_raw,
        "edited_timestamp_utc": parse_export_timestamp(edited_timestamp_raw),
        "edited_timestamp_utc_method": "inferred_America_New_York_from_snowflake_crosscheck" if edited_timestamp_raw else None,
        "is_edited": edited is not None,
        "reply_to_message_id": reply_id,
        "reply_author": reply_author,
        "reply_text": reply_text,
        "text": tag_text(content, exclude_classes={"chatlog__edited-timestamp"})
        or ("\n\n".join(item["text"] for item in forwarded if item["text"]) or None),
        "attachments": attachment_metadata(container),
        "embeds": embed_metadata(container),
        "reactions": reaction_metadata(container),
        "raw_html_sha256": sha256_bytes(original.encode("utf-8")) if original else None,
        "original_html_metadata": {
            "container_id": container.get("id"),
            "group_index": group_index,
            "index_in_group": index_in_group,
            "has_message_header": header_author is not None,
            "author_display_title": header_author.get("title") if header_author else None,
            "avatar_url": (container.select_one(".chatlog__message-aside .chatlog__avatar") or {}).get("src"),
            "short_timestamp_raw": short_timestamp.get("title") if short_timestamp else None,
            "rendered_timestamp_utc_inferred_local_zone": rendered_utc,
            "timestamp_crosscheck_matches_snowflake_to_minute": (
                rendered_utc[:16] == snowflake_utc[:16] if rendered_utc and snowflake_utc else None
            ),
            "source_file_sha256": context.file_sha256,
            "source_timezone_footer": context.timezone_footer,
            "reply": reply_original,
            "forwarded": forwarded,
        },
    }


def normalize_file(path: Path, input_root: Path) -> tuple[ExportContext, list[dict[str, Any]]]:
    raw = path.read_bytes()
    raw_text = raw.decode("utf-8")
    soup = BeautifulSoup(raw_text, "html.parser")
    context = export_context(path, raw, soup)
    records: list[dict[str, Any]] = []
    for group_index, group in enumerate(soup.select(".chatlog__message-group")):
        inherited_author: tuple[str | None, str | None] | None = None
        for index_in_group, container in enumerate(group.select(":scope > .chatlog__message-container")):
            record = message_record(
                container, context, raw_text, group_index, index_in_group, inherited_author
            )
            if record["original_html_metadata"]["has_message_header"]:
                inherited_author = (record["author_id"], record["author_name"])
            records.append(record)
    # Use a path relative to the input folder if possible, keeping it portable
    # while retaining a direct reference to the immutable evidence file.
    for record in records:
        record["source_file"] = str(path.relative_to(input_root.parent)).replace("\\", "/")
    return context, records


def build_manifest(contexts: list[ExportContext], records: list[dict[str, Any]]) -> dict[str, Any]:
    by_channel = Counter(record["channel_id"] for record in records)
    by_author = Counter((record["author_id"], record["author_name"]) for record in records)
    timestamp_methods = Counter(record["timestamp_utc_method"] for record in records)
    missing_author = sum(1 for record in records if not record["author_id"] and not record["author_name"])
    missing_timestamps = sum(1 for record in records if not record["timestamp_utc"])
    crosscheck = [
        record["original_html_metadata"]["timestamp_crosscheck_matches_snowflake_to_minute"]
        for record in records
        if record["original_html_metadata"]["timestamp_crosscheck_matches_snowflake_to_minute"] is not None
    ]
    return {
        "schema_version": "discord-ledger-source-v1",
        "normalizer": {
            "script": "scripts/discord_ledger/normalize.py",
            "network_access": False,
            "creation_timestamp_policy": "Discord snowflake creation time is primary because rendered timestamps omit an offset.",
            "edited_timestamp_policy": "Rendered local zone inferred as America/New_York because summer snowflake times contradict fixed UTC-5; ambiguous fall-back hour remains unresolved.",
            "display_timezone_inference": "UTC-5 footer labels standard offset; creation-title versus snowflake checks show DST in summer.",
        },
        "source_files": [
            {
                "source_file": str(context.path),
                "sha256": context.file_sha256,
                "size_bytes": context.file_size_bytes,
                "channel_id": context.channel_id,
                "channel_name": context.channel_name,
                "guild_name": context.guild_name,
                "preamble_entries": context.preamble_entries,
                "footer_entries": context.footer_entries,
                "timezone_footer": context.timezone_footer,
                "declared_exported_message_count": context.declared_exported_message_count,
                "message_count": by_channel[context.channel_id],
                "declared_count_matches_normalized": (
                    context.declared_exported_message_count == by_channel[context.channel_id]
                    if context.declared_exported_message_count is not None
                    else None
                ),
            }
            for context in contexts
        ],
        "totals": {
            "file_count": len(contexts),
            "message_count": len(records),
            "empty_channel_count": sum(1 for context in contexts if by_channel[context.channel_id] == 0),
            "channels": [
                {
                    "channel_id": context.channel_id,
                    "channel_name": context.channel_name,
                    "message_count": by_channel[context.channel_id],
                }
                for context in contexts
            ],
            "authors": [
                {"author_id": author_id, "author_name": author_name, "message_count": count}
                for (author_id, author_name), count in sorted(by_author.items(), key=lambda item: (-item[1], item[0]))
            ],
        },
        "timestamp_diagnostics": {
            "creation_timestamp_methods": dict(timestamp_methods),
            "missing_author_count": missing_author,
            "missing_creation_timestamp_count": missing_timestamps,
            "rendered_vs_snowflake_comparisons": len(crosscheck),
            "rendered_vs_snowflake_matches_to_minute": sum(value is True for value in crosscheck),
            "rendered_vs_snowflake_mismatches_to_minute": sum(value is False for value in crosscheck),
        },
    }


def normalize_export(input_dir: Path, output_dir: Path) -> dict[str, Any]:
    paths = sorted(input_dir.glob("*.html"))
    if not paths:
        raise FileNotFoundError(f"No .html exports found in {input_dir}")
    contexts: list[ExportContext] = []
    records: list[dict[str, Any]] = []
    for path in paths:
        context, file_records = normalize_file(path, input_dir)
        contexts.append(context)
        records.extend(file_records)
    output_dir.mkdir(parents=True, exist_ok=True)
    messages_path = output_dir / "messages.jsonl"
    temp_messages = messages_path.with_suffix(".jsonl.tmp")
    with temp_messages.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")
    os.replace(temp_messages, messages_path)
    manifest = build_manifest(contexts, records)
    manifest_path = output_dir / "manifest.json"
    temp_manifest = manifest_path.with_suffix(".json.tmp")
    temp_manifest.write_text(json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp_manifest, manifest_path)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("VaultOfAceDiscordLogs"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("research/discord_ledger_2026-09-21/source")
    )
    args = parser.parse_args()
    manifest = normalize_export(args.input_dir, args.output_dir)
    print(
        f"Normalized {manifest['totals']['message_count']} messages from "
        f"{manifest['totals']['file_count']} HTML exports into {args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
