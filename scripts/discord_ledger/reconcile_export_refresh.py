"""Preserve prior annotations and add source-verified refreshed-export evidence.

Caller price references remain unverified. Typed plan/zone/target/mark evidence
cannot silently become fills. Previous annotations transfer only when every
cited source version is semantically unchanged. Output paths are exclusive.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from scripts.discord_ledger.build import available_at, digest, read_jsonl, write_jsonl
from scripts.discord_ledger.curate import asof_expiry, curate
from scripts.discord_ledger.stage_export_refresh import semantic_key


def annotation(fact: dict, message: dict, ordinal: int) -> dict:
    kind = fact.get("kind", "published_chart_plan")
    action, speech = "watchlist", "conditional"
    instrument, direction = "underlying_plan", "long"
    uncertainties = ["Source evidence is untrusted; interpretation confidence is not execution verification."]
    if kind == "option_buy_reference":
        action, speech, instrument = "entry", "instruction", "option"
        uncertainties.append("Buy row/reference is not a fill receipt. Published outside the regular option session; actual entry/fill time unknown.")
    elif kind == "partial_exit_claim_with_mark":
        action, speech, instrument = "trim", "claimed_execution", "option"
        uncertainties.append("Mostly out is a partial exit claim; no sale quantity or execution price is established by the mark image.")
    elif kind == "option_mark":
        action, speech, instrument = "position_update", "observation", "option"
    elif kind in {"underlying_zone_tag_claim", "underlying_target_tag_claim"}:
        speech = "observation"
        uncertainties.append("Zone/target notification is not an option entry or an executed sale.")
    row = {"event_id": f"{message['message_id']}:reviewed_export:{ordinal}",
           "message_id": message["message_id"], "author_id": message["author_id"],
           "channel_id": message["channel_id"], "timestamp_utc": message["timestamp_utc"],
           "event_type": action, "speech_act": speech, "symbol": fact["symbol"],
           "symbol_raw": fact["symbol"], "direction": direction, "instrument_type": instrument,
           "option_type": fact.get("option_type"), "strike": fact.get("strike"),
           "expiry_raw": fact.get("expiry_raw"), "expiry": None,
           "quantity": None, "price": fact.get("buy_reference"), "stop_price": None,
           "evidence_kind": kind, "source_quote": message.get("text") or "",
           "reviewed_attachment_filename": fact.get("filename"),
           "related_message_ids": [fact["plan_message_id"]] if fact.get("plan_message_id") else [],
           "reply_to_message_id": message.get("reply_to_message_id"),
           "confidence": "high", "uncertainties": uncertainties,
           "confirmed_fill_price": None, "confirmed_sale_price": None}
    if row["expiry_raw"]:
        row["expiry"], row["expiry_resolution_rule"] = asof_expiry(row["expiry_raw"], message["timestamp_utc"])
    for key in ("entry_zone", "targets", "target", "option_mark", "underlying_mark",
                "displayed_today_change_pct", "claimed_gain_pct", "suggested_management"):
        if fact.get(key) is not None:
            row[key] = fact[key]
    row["field_sources"] = {k: [message["message_id"]] for k, v in row.items()
                            if v is not None and k not in {"uncertainties", "related_message_ids", "field_sources"}}
    return row


def preserve_prior_annotations(prior_events: list[dict], prior_messages: dict, current: dict) -> list[dict]:
    result = []
    for event in prior_events:
        refs = {str(event["message_id"])}
        refs.update(str(ref) for values in event.get("field_sources", {}).values() for ref in values)
        if any(ref not in current or ref not in prior_messages or
               semantic_key(current[ref]) != semantic_key(prior_messages[ref]) for ref in refs):
            raise ValueError(f"Prior annotation needs re-review: {event['event_id']}")
        row = {k: v for k, v in event.items() if k not in {
            "field_provenance", "trade_id", "link_confidence", "link_method",
            "source_file", "source_anchor", "available_at_utc", "availability_basis",
            "source_message_ids", "author_name", "channel_name"}}
        result.append(row)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--prior-events", type=Path, required=True)
    parser.add_argument("--prior-messages", type=Path, required=True)
    parser.add_argument("--reviewed-evidence", type=Path, required=True)
    args = parser.parse_args()
    messages = {m["message_id"]: m for m in read_jsonl(args.study / "source/messages.jsonl")}
    prior = {m["message_id"]: m for m in read_jsonl(args.prior_messages)}
    events = preserve_prior_annotations(read_jsonl(args.prior_events), prior, messages)
    legacy_count = len(events)
    review = json.loads(args.reviewed_evidence.read_text())
    facts = review["plans"] + review["notifications"] + review["option_references"]
    reviewed_ids = {r["message_id"] for r in facts}
    draft = curate(args.study)
    candidates = [e for e in draft["kept"] if e["message_id"] not in prior and e["message_id"] not in reviewed_ids]
    seen, duplicates = set(), []
    for event in candidates:
        key = tuple(event.get(k) for k in ("message_id", "event_type", "symbol", "instrument_type",
                                         "option_type", "strike", "expiry", "speech_act", "price", "quantity"))
        if key in seen:
            duplicates.append({"event_id": event["event_id"], "reason": "same_source_clause_semantics_duplicate"})
            continue
        seen.add(key)
        events.append(event)
    text_count = len(events) - legacy_count
    preserved = {r["message_id"]: r for r in read_jsonl(args.study / "attachments/fetch_attempt_02.jsonl")}
    resolved = []
    for ordinal, fact in enumerate(facts):
        m = messages[fact["message_id"]]
        at, basis = available_at(m)
        if at is None:
            raise ValueError(f"Unresolved evidence availability: {m['message_id']}")
        if fact.get("plan_message_id") and m.get("reply_to_message_id") != fact["plan_message_id"]:
            raise ValueError(f"Plan reply mismatch: {m['message_id']}")
        row = dict(fact, author_id=m["author_id"], channel_id=m["channel_id"],
                   timestamp_raw=m["timestamp_raw"], timestamp_utc=m["timestamp_utc"],
                   available_at_utc=at, availability_basis=basis,
                   is_edited=m["is_edited"], edited_timestamp_utc=m.get("edited_timestamp_utc"),
                   reply_to_message_id=m.get("reply_to_message_id"), source_quote=m.get("text"),
                   source_file=m["source_file"], source_anchor=m["source_anchor"],
                   source_message_ids=[m["message_id"]], confidence="high")
        if fact.get("filename"):
            attachment = preserved[m["message_id"]]
            p = Path(attachment["local_path"])
            if attachment["fetch_status"] != "downloaded" or digest(p) != attachment["sha256"]:
                raise ValueError(f"Unverified preserved attachment: {p}")
            if fact["filename"] != attachment["declared_filename"]:
                raise ValueError(f"Attachment identity mismatch: {m['message_id']}")
            row.update(attachment_path=str(p), attachment_sha256=digest(p))
        row["field_provenance"] = {k: {"source_message_ids": [m["message_id"]],
            "basis": "visually_reviewed_attachment" if k in fact and fact.get("filename") else "export_or_authored_message",
            "confidence": "high"} for k in row}
        resolved.append(row)
        events.append(annotation(fact, m, ordinal))
    events.sort(key=lambda e: (messages[e["message_id"]]["timestamp_utc"], e["event_id"]))
    out = args.study / "curated"
    out.mkdir(exist_ok=False)
    write_jsonl(out / "events.jsonl", events)
    write_jsonl(out / "reviewed_evidence.jsonl", resolved)
    write_jsonl(out / "rejected_candidates.jsonl", draft["rejected"] + duplicates)
    summary = {"legacy_annotations_transferred_after_source_version_checks": legacy_count,
               "new_text_annotations": text_count, "reviewed_typed_annotations": len(facts),
               "total_annotations": len(events), "event_types": dict(Counter(e["event_type"] for e in events)),
               "unresolved_user_screenshot_rows": review["unresolved_user_screenshot_rows"],
               "prior_events_sha256": digest(args.prior_events), "reviewed_evidence_sha256": digest(args.reviewed_evidence)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
