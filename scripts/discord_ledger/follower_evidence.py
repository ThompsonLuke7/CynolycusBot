"""Validate manually read member screenshots against earlier source alerts.

These are observations from exported images, not independent brokerage records.
No purchase timestamp is inferred from a date-bought label.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path

from scripts.discord_ledger.build import DEFAULT_STUDY, read_jsonl, write_jsonl

ROOT = Path(__file__).resolve().parents[2]

# Values below were read visually from the preserved local images, then checked
# against OCR as a second transcription aid. The link is a plausible same-
# contract comparison, not proof the member entered because of the alert.
REVIEWED = [
    {"alert_message_id": "1438184906550415493", "screenshot_message_id": "1438195378595041322",
     "attachment_id": "1438195378456625253", "image_sha256": "4eb21c43d0b2d7c3da89be4e432ea7bb5868398748b385dd81b94956ed93cc88",
     "symbol": "GLD", "option_type": "call", "strike": 383.0, "screenshot_expiry": "2025-11-17",
     "reported_date_bought": "2025-11-12", "reported_average_cost": 3.15, "reported_contract_count": 2,
     "reported_current_price": 4.48, "reported_credit_at_close": None, "reported_realized_profit": None,
     "screenshot_state": "position_screen", "contract_match_confidence": "high"},
    {"alert_message_id": "1458529779379863624", "screenshot_message_id": "1458847295419322469",
     "attachment_id": "1458847295146557583", "image_sha256": "3ac0cb4871830857129586b4af18c3aecec4980f16dbddbeb7313b0eecf69bf8",
     "symbol": "RKLB", "option_type": "call", "strike": 90.0, "screenshot_expiry": "2026-01-16",
     "reported_date_bought": "2026-01-07", "reported_average_cost": 2.74, "reported_contract_count": 1,
     "reported_current_price": 4.05, "reported_credit_at_close": None, "reported_realized_profit": None,
     "screenshot_state": "position_screen", "contract_match_confidence": "high"},
    {"alert_message_id": "1458835995884650507", "screenshot_message_id": "1458849085313519797",
     "attachment_id": "1458849084650815642", "image_sha256": "34661ae748dc1add0f9f81c488fdeb042493062685282a4aa2df43dcebbd5b20",
     "symbol": "QQQ", "option_type": "put", "strike": 615.0, "screenshot_expiry": "2026-01-08",
     "reported_date_bought": None, "reported_average_cost": 0.38, "reported_contract_count": 5,
     "reported_current_price": None, "reported_credit_at_close": 0.60, "reported_realized_profit": 110.0,
     "screenshot_state": "closed_position_screen", "contract_match_confidence": "medium"},
    {"alert_message_id": "1461403804309393614", "screenshot_message_id": "1461439004192018493",
     "attachment_id": "1461439004036694340", "image_sha256": "080554e13fdcb0bb13443279bf503e3a061936398da3d6330a4f483c966fb701",
     "symbol": "MU", "option_type": "put", "strike": 310.0, "screenshot_expiry": "2026-01-23",
     "reported_date_bought": None, "reported_average_cost": 1.40, "reported_contract_count": 1,
     "reported_current_price": None, "reported_credit_at_close": 1.65, "reported_realized_profit": 25.0,
     "screenshot_state": "closed_position_screen", "contract_match_confidence": "high"},
    {"alert_message_id": "1464278240935018645", "screenshot_message_id": "1464279154898898975",
     "attachment_id": "1464279154676727860", "image_sha256": "d79b70a4e59126c3b53ae0a2e4cbc24294c4d6343f46dbae97907892bd6481e6",
     "symbol": "GLD", "option_type": "put", "strike": 450.0, "screenshot_expiry": "2026-01-23",
     "reported_date_bought": None, "reported_average_cost": None, "reported_contract_count": 1,
     "reported_current_price": 0.36, "reported_credit_at_close": None, "reported_realized_profit": None,
     "screenshot_state": "position_list_screen_current_price_not_cost", "contract_match_confidence": "medium"},
    {"alert_message_id": "1545072423198785576", "screenshot_message_id": "1545429146056925285",
     "attachment_id": "1545429145570377738", "image_sha256": "b1a16cccc519ddbed1555ef13670ac3f9a7fc5ef5a5d62e13faca3a172e05f50",
     "symbol": "MU", "option_type": "call", "strike": 980.0, "screenshot_expiry": "2026-09-04",
     "reported_date_bought": "2026-09-03", "reported_average_cost": 2.63, "reported_contract_count": 5,
     "reported_current_price": 23.40, "reported_credit_at_close": None, "reported_realized_profit": None,
     "screenshot_state": "position_screen", "contract_match_confidence": "medium"},
    {"alert_message_id": "1547688324146139197", "screenshot_message_id": "1547698894421360681",
     "attachment_id": "1547698894450982933", "image_sha256": "42c8d36759392ce5307f2762c7b1608cf160a68a980a1a32de331ea8f9605d18",
     "symbol": "AAPL", "option_type": "call", "strike": 327.5, "screenshot_expiry": "2026-09-11",
     "reported_date_bought": None, "reported_average_cost": 1.44, "reported_contract_count": 1,
     "reported_current_price": None, "reported_credit_at_close": 1.74, "reported_realized_profit": 30.0,
     "screenshot_state": "closed_position_screen", "contract_match_confidence": "medium"},
]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def iso_to_datetime(value: str) -> datetime:
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("Timezone required")
    return result


def build(study: Path, run: Path) -> list[dict]:
    messages = {m["message_id"]: m for m in read_jsonl(study / "source/messages.jsonl")}
    events = {e["message_id"]: e for e in read_jsonl(run / "events.jsonl") if e["event_type"] == "entry"}
    attachments = {a["attachment_id"]: a for a in read_jsonl(study / "attachments/manifest.jsonl") if a.get("local_path")}
    records = []
    for item in REVIEWED:
        aid, sid = item["alert_message_id"], item["screenshot_message_id"]
        alert, shot = events[aid], messages[sid]
        attachment = attachments[item["attachment_id"]]
        if attachment["message_id"] != sid or attachment["author_id"] == alert["author_id"]:
            raise ValueError(f"Not a member screenshot attached to {sid}")
        if attachment["sha256"] != item["image_sha256"] or digest(ROOT / attachment["local_path"]) != item["image_sha256"]:
            raise ValueError(f"Screenshot hash mismatch: {sid}")
        if any(alert.get(field) != item[field] for field in ("symbol", "option_type", "strike")):
            raise ValueError(f"Alert identity mismatch: {aid}")
        if alert.get("expiry") and alert["expiry"] != item["screenshot_expiry"]:
            raise ValueError(f"Expiry conflict: {aid}")
        lag = int((iso_to_datetime(shot["timestamp_utc"]) - iso_to_datetime(alert["timestamp_utc"])).total_seconds())
        if lag <= 0:
            raise ValueError(f"Screenshot predates alert: {sid}")
        row = {**item, "alert_event_id": alert["event_id"], "alert_trade_id": alert["trade_id"],
               "alert_author": alert["author_name"], "member_author": shot["author_name"],
               "alert_timestamp_utc": alert["timestamp_utc"], "screenshot_posted_at_utc": shot["timestamp_utc"],
               "screenshot_post_lag_seconds": lag,
               "alert_source_quoted_price": alert.get("price"),
               "alert_source_quoted_price_rule": alert.get("price_extraction_rule"),
               "alert_expiry_at_post_time": alert.get("expiry"),
               "image_path": attachment["local_path"],
               "source_message_ids": [aid, sid],
               "association_status": "same_contract_member_report_after_alert_not_causal_proof",
               "purchase_time_status": "unknown_exact_time; date_bought_only_if_screenshot_displays_it",
               "verification": "visually_read_preserved_image_and_cross_checked_OCR"}
        image_fields = {"reported_date_bought", "reported_average_cost", "reported_contract_count", "reported_current_price", "reported_credit_at_close", "reported_realized_profit", "screenshot_state", "screenshot_expiry", "image_path", "image_sha256", "attachment_id", "member_author", "screenshot_posted_at_utc", "screenshot_message_id"}
        alert_fields = {"alert_event_id", "alert_trade_id", "alert_author", "alert_timestamp_utc", "alert_message_id", "alert_source_quoted_price", "alert_source_quoted_price_rule", "alert_expiry_at_post_time"}
        row["field_provenance"] = {key: {"source_message_ids": [sid] if key in image_fields else [aid] if key in alert_fields else [aid, sid],
                                          "basis": "manual_image_read" if key in image_fields else "caller_alert" if key in alert_fields else "cross_source_comparison",
                                          "confidence": item["contract_match_confidence"]}
                                   for key in row if key != "field_provenance"}
        records.append(row)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=DEFAULT_STUDY)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    records = build(args.study, args.run)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out, records)
    messages = {m["message_id"]: m for m in read_jsonl(args.study / "source/messages.jsonl")}
    lines = ["# Visually reviewed member screenshots matched to earlier callouts", "",
             "These are source-reported observations, not an unbiased follower sample or broker-certified fills. Post lag is the interval from alert creation to screenshot post, **not** the member's execution delay. A matching contract does not prove the member acted because of that alert.", "",
             "| Caller | Contract | Alert reference | Member average cost | Difference vs alert | Post lag | Evidence |",
             "| --- | --- | ---: | ---: | ---: | ---: | --- |"]
    for row in records:
        alert_price, average = row["alert_source_quoted_price"], row["reported_average_cost"]
        difference = f"{(average / alert_price - 1) * 100:+.1f}%" if average is not None and alert_price else "unknown"
        a, s = messages[row["alert_message_id"]], messages[row["screenshot_message_id"]]
        alert_path = str(ROOT / a["source_file"]) + "#" + a["source_anchor"].lstrip("#")
        shot_path = str(ROOT / s["source_file"]) + "#" + s["source_anchor"].lstrip("#")
        contract = f"{row['symbol']} {row['strike']:g} {row['option_type']} {row['screenshot_expiry']}"
        lines.append(f"| {row['alert_author']} | {contract} | {alert_price if alert_price is not None else 'unknown'} | {average if average is not None else 'unknown'} | {difference} | {row['screenshot_post_lag_seconds'] // 60} min | [alert {row['alert_message_id']}](<{alert_path}>), [member {row['screenshot_message_id']}](<{shot_path}>), [image](<{ROOT / row['image_path']}>) |")
    lines += ["", "The GLD 450 put screenshot is a position-list screen: its $0.36 is a **displayed current price**, not an average cost. QQQ 615 put, MU 310 put, and AAPL 327.5 call show close credits and source-reported realized profit; these amounts are not generalized to other followers or trades.",
              "", "The seven reviewed cases were selected from visible member posts in the supplied export. The companion `member_ocr_candidates_v2.jsonl` contains additional unreviewed OCR matches. Other users' losing/quiet trades may be absent from the profit-posting channel."]
    args.out.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"reviewed_screenshot_links": len(records), "out": str(args.out)}))


if __name__ == "__main__":
    main()
