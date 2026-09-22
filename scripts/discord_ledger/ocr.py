"""Offline CPU OCR of preserved Discord screenshots, retaining image evidence.

OCR output is a search aid. It does not by itself assert an order, fill, or P&L.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "research/discord_ledger_2026-09-21"
CALLERS = {"1079391263083733072", "720901855995101225"}
OCR_DEPS = Path("/tmp/cynolycus_discord_ocr")
_ENGINE = None


def recognize(row: dict) -> dict:
    global _ENGINE
    if _ENGINE is None:
        sys.path.insert(0, str(OCR_DEPS))
        from rapidocr_onnxruntime import RapidOCR
        _ENGINE = RapidOCR(intra_op_num_threads=1, inter_op_num_threads=1)
    path = ROOT / row["local_path"]
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != row["sha256"]:
        raise ValueError(f"Preserved attachment changed: {path}")
    output, times = _ENGINE(str(path), use_cls=False)
    lines = [{"text": item[1], "confidence": float(item[2]), "box": item[0]} for item in (output or [])]
    return {"message_id": row["message_id"], "channel_id": row["channel_id"],
            "author_id": row["author_id"], "attachment_id": row["attachment_id"],
            "attachment_ordinal": row["attachment_ordinal"], "local_path": row["local_path"],
            "sha256": digest, "source_message_ids": [row["message_id"]],
            "ocr_lines": lines, "ocr_text": "\n".join(line["text"] for line in lines),
            "ocr_model": "rapidocr_onnxruntime-1.4.4-local-CPU",
            "ocr_at_utc": datetime.now(timezone.utc).isoformat(),
            "interpretation": "unverified_machine_reading_of_exported_attachment_not_a_fill",
            "runtime_seconds": sum(times) if isinstance(times, (list, tuple)) else None}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=STUDY / "attachments/manifest.jsonl")
    parser.add_argument("--out", type=Path, default=STUDY / "attachments/ocr.jsonl")
    parser.add_argument("--limit", type=int, default=0, help="Additional images this invocation; zero means all remaining")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--include-members", action="store_true", help="Also OCR other users' execution screenshots")
    parser.add_argument("--members-only", action="store_true", help="OCR only non-caller attachments, to a separate output")
    args = parser.parse_args()
    if not OCR_DEPS.exists():
        raise RuntimeError("Isolated OCR dependency directory missing; no system package is modified")
    rows = [json.loads(line) for line in args.manifest.open(encoding="utf-8") if line.strip()]
    existing = {json.loads(line)["attachment_id"] for line in args.out.open(encoding="utf-8")} if args.out.exists() else set()
    selected = [r for r in rows if r.get("local_path") and r["attachment_id"] not in existing
                and ((r.get("author_id") not in CALLERS) if args.members_only else (args.include_members or r.get("author_id") in CALLERS)) and
                Path(r["local_path"]).suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}]
    priorities = {"1416372120388243516": 0, "1466557447094403335": 1,
                  "1443663553239449770": 2, "1416374095456632912": 3, "1521796664052940870": 4}
    selected.sort(key=lambda r: (priorities.get(r["channel_id"], 9),
                                 0 if not (r.get("message_text") or "").strip() else 1,
                                 r["message_id"], r["attachment_ordinal"]))
    if args.limit:
        selected = selected[: args.limit]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    successes, failures = 0, 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool, args.out.open("a", encoding="utf-8") as handle:
        futures = {pool.submit(recognize, r): r for r in selected}
        for future in as_completed(futures):
            row = futures[future]
            try:
                record = future.result()
                successes += 1
            except Exception as exc:
                record = {"attachment_id": row["attachment_id"], "message_id": row["message_id"],
                          "local_path": row["local_path"], "error": f"{type(exc).__name__}:{exc}",
                          "source_message_ids": [row["message_id"]], "interpretation": "ocr_failed_unresolved"}
                failures += 1
            handle.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")
            handle.flush()
    print(json.dumps({"attempted": len(selected), "successes": successes, "failures": failures,
                      "remaining_estimate": max(0, len(rows) - len(existing) - len(selected))}))


if __name__ == "__main__":
    main()
