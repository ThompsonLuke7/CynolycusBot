#!/usr/bin/env python3
"""Fetch Discord-export attachments into an auditable, content-addressed cache.

This deliberately reads only ``cdn.discordapp.com/attachments/...`` links from
the supplied static HTML export.  It neither follows arbitrary links in the
export nor evaluates HTML/JavaScript.  A manifest row is emitted for every
reference, including references deliberately skipped because the author is not
one of the two callers under review.
"""

from __future__ import annotations

import argparse
import hashlib
import html as html_module
import json
import mimetypes
import re
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urlparse
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

from bs4 import BeautifulSoup


ACE_ID = "1079391263083733072"
FT_ID = "720901855995101225"
CALLER_IDS = {ACE_ID, FT_ID}
DISCORD_HOST = "cdn.discordapp.com"
MAX_ATTACHMENT_BYTES = 25 * 1024 * 1024
ET = ZoneInfo("America/New_York")
# Alert and challenge screenshots can contain contemporaneous contract details.
# Watchlists and the post-profit channel are retained too, but are lower-review
# priority because they are more often plans or retrospective screenshots.
CHANNEL_REVIEW_PRIORITY = {
    "1416372120388243516": 0,  # ace-free-alerts
    "1466557447094403335": 1,  # 1k-challenge
    "1443663553239449770": 2,  # ace-free-watchlists
    "1416374095456632912": 3,  # post-ur-profits
}


def is_safe_discord_attachment(url: str) -> bool:
    """Permit exactly Discord CDN attachment paths; reject all other export URLs."""
    parsed = urlparse(html_module.unescape(url))
    return (
        parsed.scheme == "https"
        and parsed.hostname == DISCORD_HOST
        and parsed.path.startswith("/attachments/")
    )


def clean_filename(value: str) -> str:
    value = Path(unquote(value)).name
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return value[:180] or "attachment"


def parse_timestamp(value: str | None) -> str | None:
    if not value:
        return None
    for fmt in ("%A, %B %d, %Y %I:%M %p", "%A, %B %d, %Y %H:%M"):
        try:
            return datetime.strptime(value, fmt).replace(tzinfo=ET).isoformat()
        except ValueError:
            pass
    return None


def channel_id_from_export(path: Path) -> str | None:
    match = re.search(r"\[(\d+)\]\.html$", path.name)
    return match.group(1) if match else None


def text_of(element: Any) -> str:
    return element.get_text(" ", strip=True) if element else ""


def attachment_links(message: Any) -> Iterable[tuple[str, str | None, str | None]]:
    """Yield unique direct attachment URLs, names, and declared media titles.

    DiscordChatExporter represents image attachments as an ``a > img`` but
    represents video attachments as a bare ``video > source``.  Both are
    source attachments and need inventorying even when video review is out of
    scope.
    """
    seen: set[str] = set()
    for attachment in message.select(":scope > .chatlog__message .chatlog__attachment"):
        media = attachment.select_one("a[href], source[src], video[src], img[src]")
        if not media:
            continue
        url = html_module.unescape(media.get("href") or media.get("src") or "")
        if not is_safe_discord_attachment(url) or url in seen:
            continue
        seen.add(url)
        titled_media = attachment.select_one("[title]")
        title = titled_media.get("title") if titled_media else None
        filename = None
        if title:
            name_match = re.search(r"(?:Image|Video|File):\s*(.*?)\s*\(", title)
            filename = name_match.group(1) if name_match else None
        if not filename:
            filename = Path(urlparse(url).path).name
        yield url, filename, title


def parse_export(path: Path) -> list[dict[str, Any]]:
    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="replace"), "html.parser")
    channel_id = channel_id_from_export(path)
    rows: list[dict[str, Any]] = []
    last_author: dict[str, str | None] = {"author_id": None, "author_name": None, "author_handle": None}
    for message in soup.select("[data-message-id]"):
        author = message.select_one(".chatlog__author[data-user-id]")
        if author:
            last_author = {
                "author_id": author.get("data-user-id"),
                "author_name": text_of(author) or None,
                "author_handle": author.get("title"),
            }
        message_id = message.get("data-message-id")
        timestamp_node = message.select_one(".chatlog__timestamp[title], .chatlog__short-timestamp[title]")
        content = message.select_one(".chatlog__content")
        for ordinal, (url, filename, title) in enumerate(attachment_links(message)):
            parts = urlparse(url).path.split("/")
            attachment_id = parts[3] if len(parts) > 3 else None
            rows.append(
                {
                    "record_type": "attachment_reference",
                    "source_export": str(path),
                    "channel_id": channel_id,
                    "message_id": message_id,
                    "author_id": last_author["author_id"],
                    "author_name": last_author["author_name"],
                    "author_handle": last_author["author_handle"],
                    "message_timestamp_et": parse_timestamp(timestamp_node.get("title") if timestamp_node else None),
                    "message_text": text_of(content) or None,
                    "attachment_ordinal": ordinal,
                    "attachment_id": attachment_id,
                    "source_url": url,
                    "declared_filename": filename,
                    "declared_title": title,
                    "is_image_hint": bool(title and title.lower().startswith("image:")),
                }
            )
    return rows


def fetch_one(row: dict[str, Any], output_dir: Path, timeout: int) -> dict[str, Any]:
    """Download with a hard size cap, hashing bytes while writing a temp file."""
    fetched_at = datetime.now(timezone.utc).isoformat()
    url = row["source_url"]
    if not is_safe_discord_attachment(url):  # Defensive: manifest input is untrusted.
        return {"fetch_status": "rejected_unsafe_url", "fetched_at_utc": fetched_at}
    filename = clean_filename(str(row.get("declared_filename") or "attachment"))
    attachment_id = clean_filename(str(row.get("attachment_id") or "unknown"))
    destination = output_dir / f"{attachment_id}_{filename}"
    temp = output_dir / f".{attachment_id}_{filename}.part"
    # A prior successful preservation is immutable evidence.  Re-hash it for
    # the current manifest instead of making an unnecessary expiring-URL call.
    if destination.exists():
        return {
            "fetch_status": "already_preserved_local",
            "fetched_at_utc": fetched_at,
            "bytes": destination.stat().st_size,
            "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
            "local_path": str(destination),
        }
    digest = hashlib.sha256()
    size = 0
    try:
        request = Request(url, headers={"User-Agent": "CynolycusBot-discord-ledger/1.0"})
        with urlopen(request, timeout=timeout) as response, temp.open("wb") as handle:
            content_type = response.headers.get_content_type()
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > MAX_ATTACHMENT_BYTES:
                    raise ValueError(f"attachment exceeds {MAX_ATTACHMENT_BYTES} byte safety cap")
                digest.update(chunk)
                handle.write(chunk)
        sha256 = digest.hexdigest()
        shutil.move(str(temp), str(destination))
        status = "downloaded"
        return {
            "fetch_status": status,
            "fetched_at_utc": fetched_at,
            "http_status": 200,
            "content_type": content_type,
            "bytes": size,
            "sha256": sha256,
            "local_path": str(destination),
        }
    except HTTPError as exc:
        temp.unlink(missing_ok=True)
        return {"fetch_status": "http_error", "fetched_at_utc": fetched_at, "http_status": exc.code, "error": str(exc)}
    except (URLError, TimeoutError, ValueError, OSError) as exc:
        temp.unlink(missing_ok=True)
        return {"fetch_status": "fetch_error", "fetched_at_utc": fetched_at, "error": str(exc)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--html-dir", type=Path, default=Path("VaultOfAceDiscordLogs"))
    parser.add_argument("--output-dir", type=Path, default=Path("research/discord_ledger_2026-09-21/attachments"))
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("research/discord_ledger_2026-09-21/attachments/manifest.jsonl"),
        help="Attachment-reference/fetch manifest.  This is intentionally separate from event annotations.",
    )
    parser.add_argument("--include-member-images", action="store_true", help="Fetch attachments from non-caller authors too.")
    parser.add_argument("--no-fetch", action="store_true", help="Only inventory references; make no network requests.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum eligible files to fetch (for staged review).")
    parser.add_argument("--timeout", type=int, default=30)
    args = parser.parse_args()
    exports = sorted(args.html_dir.glob("*.html"))
    if not exports:
        parser.error(f"no HTML exports found under {args.html_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    rows = [row for export in exports for row in parse_export(export)]
    fetched = 0
    eligible = [
        row
        for row in rows
        if args.include_member_images or row.get("author_id") in CALLER_IDS
    ]
    eligible.sort(
        key=lambda row: (
            CHANNEL_REVIEW_PRIORITY.get(str(row.get("channel_id")), 99),
            row.get("message_timestamp_et") or "",
            row.get("message_id") or "",
            row.get("attachment_ordinal") or 0,
        )
    )
    eligible_ids = {id(row): index for index, row in enumerate(eligible)}
    for row in rows:
        is_caller = row.get("author_id") in CALLER_IDS
        if args.no_fetch:
            row["fetch_status"] = "not_requested"
        elif not args.include_member_images and not is_caller:
            row["fetch_status"] = "skipped_noncaller_author"
        elif args.limit is not None and eligible_ids[id(row)] >= args.limit:
            row["fetch_status"] = "deferred_limit"
        else:
            row.update(fetch_one(row, args.output_dir, args.timeout))
            fetched += 1
    temporary = args.manifest.with_suffix(args.manifest.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    temporary.replace(args.manifest)
    summary = Counter(row.get("fetch_status") for row in rows)
    print(json.dumps({"references": len(rows), "fetched_attempts": fetched, "status_counts": summary}, default=dict, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
