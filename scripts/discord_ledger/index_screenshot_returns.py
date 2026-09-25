"""Index OCR-reported position returns without turning screenshots into fills."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re

from scripts.discord_ledger.build import read_jsonl, write_jsonl

CALLERS = {"1079391263083733072": "ACE", "720901855995101225": "FT"}
RETURN_LINE = re.compile(r"Total\s*return\s*\n\s*([^\n]{1,100})", re.I)
PERCENT = re.compile(r"([+-]?\d[\d,]*(?:\.\d+)?)\s*%")


def extract(row: dict) -> dict | None:
    if row.get("author_id") not in CALLERS:
        return None
    text = row.get("ocr_text") or ""
    match = RETURN_LINE.search(text)
    if not match:
        return None
    display = match.group(1).strip()
    percents = PERCENT.findall(display)
    pct = float(percents[-1].replace(",", "")) if percents else None
    return {
        "message_id": row["message_id"], "author_name": CALLERS[row["author_id"]],
        "attachment_id": row["attachment_id"], "attachment_sha256": row["sha256"],
        "local_path": row["local_path"], "ocr_total_return_line": display,
        "ocr_total_return_percent": pct,
        "sign_class": "positive" if pct is not None and pct > 0 else "negative" if pct is not None and pct < 0 else "zero" if pct == 0 else "unparsed",
        "snapshot_type": "open_position_screen" if re.search(r"Your\s+position", text, re.I) else "screen_type_unclear",
        "visual_review_status": "visually_checked" if row["message_id"] in {"1466809051915616329", "1469022904585945150"} else "ocr_only_unverified",
        "source_message_ids": row["source_message_ids"],
        "limitation": "OCR of a selected posted screenshot; usually an unrealized mark, possibly repeated position, not a closed-trade win/loss or independent broker record.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ocr", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = [parsed for raw in read_jsonl(args.ocr) if (parsed := extract(raw)) is not None]
    write_jsonl(args.out, rows)
    print(json.dumps({"rows": len(rows), "by_author_sign": {f"{author}:{sign}": count
                     for (author, sign), count in Counter((r["author_name"], r["sign_class"]) for r in rows).items()}}, indent=2))


if __name__ == "__main__":
    main()
