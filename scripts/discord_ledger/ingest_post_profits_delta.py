"""Normalize and deduplicate a supplemental post-ur-profits Discord export."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from scripts.discord_ledger.build import read_jsonl, write_jsonl
from scripts.discord_ledger.normalize import normalize_file


CHANNEL_ID = "1416374095456632912"
CHANNEL_NAME = "📈-post-ur-profits"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def comparable(row: dict) -> tuple:
    return (row.get("author_id"), row.get("author_name"), row.get("timestamp_utc"), row.get("text"),
            row.get("reply_to_message_id"), tuple((item.get("attachment_id"), item.get("filename")) for item in row.get("attachments") or []))


def build(base: list[dict], supplemental: list[dict]) -> tuple[list[dict], list[dict], dict]:
    by_id = {row["message_id"]: row for row in base}
    if len(by_id) != len(base):
        raise ValueError("Base source has duplicate message IDs")
    new, conflicts = [], []
    for row in supplemental:
        row = dict(row)
        row["channel_id"] = CHANNEL_ID
        row["channel_name"] = CHANNEL_NAME
        existing = by_id.get(row["message_id"])
        if existing is None:
            new.append(row)
            by_id[row["message_id"]] = row
        elif comparable(existing) != comparable(row):
            conflicts.append({"message_id": row["message_id"], "base_source_file": existing.get("source_file"),
                              "supplement_source_file": row.get("source_file"), "status": "same_id_different_exported_content_not_overwritten"})
    merged = sorted(by_id.values(), key=lambda row: (row["timestamp_utc"], row["message_id"]))
    summary = {"base_message_count": len(base), "supplement_message_count": len(supplemental),
               "overlap_same_content": len(supplemental) - len(new) - len(conflicts),
               "new_messages_added": len(new), "same_id_conflicts_not_overwritten": len(conflicts),
               "merged_message_count": len(merged), "new_by_author": dict(Counter(row["author_name"] for row in new)),
               "channel_id_forced_from_user_supplied_filename": CHANNEL_ID}
    return merged, conflicts, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--html", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    context, supplement = normalize_file(args.html, args.html.parent)
    merged, conflicts, summary = build(read_jsonl(args.base), supplement)
    args.out.mkdir(parents=True, exist_ok=False)
    write_jsonl(args.out / "messages.jsonl", merged)
    write_jsonl(args.out / "new_messages.jsonl", [row for row in supplement if row["message_id"] not in {base["message_id"] for base in read_jsonl(args.base)}])
    write_jsonl(args.out / "same_id_conflicts.jsonl", conflicts)
    manifest = {"base_path": str(args.base), "base_sha256": digest(args.base), "supplement_path": str(args.html),
                "supplement_sha256": digest(args.html), "normalizer_context_channel_id": context.channel_id,
                "normalizer_context_channel_name": context.channel_name, "summary": summary}
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
