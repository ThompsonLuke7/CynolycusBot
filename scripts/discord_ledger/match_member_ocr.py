"""Find unreviewed same-contract caller/member screenshot pairs for triage.

OCR is untrusted and approximate. Candidates are not promoted to follower fills.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta
import json
from pathlib import Path
import re

from scripts.discord_ledger.build import DEFAULT_STUDY, read_jsonl, write_jsonl

CONTRACT = re.compile(r"(?<![A-Za-z0-9])([A-Z]{2,5}W?)\s*\$?\s*([\d,]+(?:\.\d+)?)\s*(Call|Put)\b", re.I)


def dt(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def make_candidates(study: Path, run: Path) -> list[dict]:
    messages = {m["message_id"]: m for m in read_jsonl(study / "source/messages.jsonl")}
    alerts = [e for e in read_jsonl(run / "events.jsonl") if e["event_type"] == "entry" and
              e.get("symbol") and e.get("option_type") and e.get("strike") is not None]
    by_key: dict[tuple[str, float, str], list[dict]] = {}
    for alert in alerts:
        key = (alert["symbol"], float(alert["strike"]), alert["option_type"])
        by_key.setdefault(key, []).append(alert)
    rows = []
    for image in read_jsonl(study / "attachments/member_ocr.jsonl"):
        if not image.get("ocr_text"):
            continue
        msg = messages[image["message_id"]]
        found = {(m[1].upper(), float(m[2].replace(",", "")), m[3].lower()) for m in CONTRACT.finditer(image["ocr_text"])}
        # Some broker screens label SPX options SPXW, whereas the Discord alert
        # says SPX. This remains an explicit possible mapping, not a silent join.
        expanded = set(found)
        expanded.update(("SPX", strike, side) for sym, strike, side in found if sym == "SPXW")
        for key in sorted(expanded):
            for alert in by_key.get(key, []):
                if alert["author_id"] == msg["author_id"]:
                    continue
                lag = dt(msg["timestamp_utc"]) - dt(alert["timestamp_utc"])
                if not timedelta(0) <= lag <= timedelta(days=30):
                    continue
                sources = [alert["message_id"], msg["message_id"]]
                row = {"alert_event_id": alert["event_id"], "alert_message_id": alert["message_id"],
                       "alert_author": alert["author_name"], "alert_timestamp_utc": alert["timestamp_utc"],
                       "member_message_id": msg["message_id"], "member_author": msg["author_name"],
                       "member_posted_at_utc": msg["timestamp_utc"], "attachment_id": image["attachment_id"],
                       "attachment_sha256": image["sha256"], "attachment_path": image["local_path"],
                       "symbol_literal_match": key[0], "strike_literal_match": key[1], "option_type_literal_match": key[2],
                       "alert_expiry_at_post_time": alert.get("expiry"),
                       "lag_seconds": int(lag.total_seconds()),
                       "ocr_excerpt": image["ocr_text"][:600],
                       "source_message_ids": sources,
                       "status": "unreviewed_ocr_contract_candidate_not_an_execution",
                       "cautions": ["OCR can misread symbols and numbers.", "Same contract and later post do not prove copying or execution after alert.",
                                    "No screenshot-derived expiry, fill price, or P&L is promoted without visual review."]}
                row["field_provenance"] = {field: {"source_message_ids": [msg["message_id"]] if field.startswith(("member_", "attachment_", "ocr_")) else [alert["message_id"]] if field.startswith(("alert_",)) else sources,
                                                    "basis": "OCR_candidate_join_not_verified", "confidence": "low"}
                                           for field in row if field != "field_provenance"}
                rows.append(row)
    rows.sort(key=lambda x: (x["member_posted_at_utc"], x["member_message_id"], x["alert_message_id"], x["attachment_id"]))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=DEFAULT_STUDY)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = make_candidates(args.study, args.run)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out, rows)
    print(json.dumps({"candidate_pairs": len(rows), "distinct_member_messages": len({r["member_message_id"] for r in rows})}))


if __name__ == "__main__":
    main()
