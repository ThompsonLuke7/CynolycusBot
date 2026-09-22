"""Reconcile source-cited Discord claims without manufacturing broker executions.

This is offline research. Neither the HTML nor text embedded in it is executed.
Outputs are immutable run directories; reruns must select a fresh directory.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import html
import json
from pathlib import Path
import re
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STUDY = ROOT / "research/discord_ledger_2026-09-21"
CALLERS = {"1079391263083733072": "ACE", "720901855995101225": "FT"}
IDENTITY = ("symbol", "direction", "instrument_type", "option_type", "strike", "expiry")
MANAGEMENT = {"add", "trim", "exit", "stop_adjustment", "invalidation", "order_cancel", "correction", "unknown_management", "position_update"}
OPENERS = {"entry", "order_pending"}
CONTEXT = {"watchlist", "performance_claim", "position_update"}
SCHEMA = "discord_managed_ledger_v1"


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def stamp(value: str) -> datetime:
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError(f"Naive timestamp is not admissible: {value}")
    return result.astimezone(timezone.utc)


def iso(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def available_at(message: dict) -> tuple[str | None, str]:
    """Latest exported text is not assumed to have existed before its edit."""
    if not message.get("is_edited"):
        return message["timestamp_utc"], "creation_time_for_unedited_exported_message"
    edited = message.get("edited_timestamp_utc")
    if not edited:
        return None, "edited_original_version_missing_and_edit_time_unresolved"
    # Exporter renders minute precision. Its final version could have appeared
    # anywhere within that minute: use the upper bound, never the minute start.
    upper = stamp(edited).replace(second=0, microsecond=0) + timedelta(minutes=1)
    return iso(max(stamp(message["timestamp_utc"]), upper)), "conservative_end_of_exported_edit_minute_original_version_missing"


def provenance(row: dict, sources: list[str], basis: str, confidence: str = "medium") -> dict:
    return {key: {"source_message_ids": sorted(set(sources)), "basis": basis,
                  "confidence": confidence} for key in row if key != "field_provenance"}


def normalize_annotations(paths: list[Path], messages: dict[str, dict]) -> tuple[list[dict], list[dict]]:
    events, issues, seen = [], [], set()
    for path in paths:
        for raw in read_jsonl(path):
            mid = str(raw["message_id"])
            if mid not in messages:
                raise ValueError(f"Unknown message {mid} in {path}")
            m = messages[mid]
            if m["author_id"] not in CALLERS:
                raise ValueError(f"Non-caller message used as caller event: {mid}")
            if str(raw.get("author_id")) != m["author_id"]:
                raise ValueError(f"Author mismatch for {mid}")
            if str(raw.get("channel_id")) != m["channel_id"]:
                raise ValueError(f"Channel mismatch for {mid}")
            eid = str(raw["event_id"])
            if eid in seen:
                # Independent image evidence is a distinct event, never silently
                # overwrite the text annotation with the same local suffix.
                eid = f"{eid}:{path.stem}"
            if eid in seen:
                raise ValueError(f"Duplicate event ID: {eid}")
            seen.add(eid)
            e = dict(raw, event_id=eid)
            field_sources = e.get("field_sources", {})
            for key in ("symbol", "direction", "instrument_type", "option_type", "strike", "expiry", "quantity", "price", "stop_price", "event_type", "speech_act"):
                if e.get(key) is not None and not field_sources.get(key):
                    raise ValueError(f"Missing field source {eid}.{key}")
            source_ids = {mid}
            for key, refs in field_sources.items():
                if not isinstance(refs, list) or any(str(ref) not in messages for ref in refs):
                    raise ValueError(f"Invalid source IDs for {eid}.{key}: {refs}")
                source_ids.update(map(str, refs))
            e["source_message_ids"] = sorted(source_ids)
            e["timestamp_utc"] = m["timestamp_utc"]
            e["timestamp_raw"] = m["timestamp_raw"]
            e["author_name"] = m["author_name"]
            e["channel_name"] = m["channel_name"]
            e["source_file"] = m["source_file"]
            e["source_anchor"] = m["source_anchor"]
            e["is_edited"] = m["is_edited"]
            e["available_at_utc"], e["availability_basis"] = available_at(m)
            e["uncertainties"] = list(e.get("uncertainties") or [])
            if m["is_edited"]:
                e["uncertainties"].append("Original pre-edit message version is not in the export.")
            if e.get("price") is not None:
                e["uncertainties"].append("Price is a source claim/reference, not a broker-verified fill.")
            if e.get("symbol"):
                e["symbol"] = e["symbol"].upper()
            for key, refs in field_sources.items():
                for ref in refs:
                    avail, _ = available_at(messages[str(ref)])
                    if avail is None or (e["available_at_utc"] and stamp(avail) > stamp(e["available_at_utc"])):
                        issues.append({"kind": "field_source_not_available_at_event", "event_id": eid,
                                       "field": key, "source_message_ids": [mid, str(ref)]})
                        e["uncertainties"].append(f"{key} cites a source version not available at this event.")
                        e["pit_evidence_blocked"] = True
            # Every exported field has explicit message provenance. This records
            # the annotation method; it does not make its interpretation certain.
            e["field_provenance"] = provenance(e, sorted(source_ids), "source_annotation_and_export_metadata", e.get("confidence", "low"))
            for key, refs in field_sources.items():
                e["field_provenance"][key] = {"source_message_ids": refs, "basis": "annotator_interpretation_of_cited_evidence", "confidence": e.get("confidence", "low")}
            events.append(e)
    events.sort(key=lambda e: (e["available_at_utc"] or e["timestamp_utc"], e["message_id"], e["event_id"]))
    return events, issues


def compatible(trade: dict, event: dict) -> bool:
    if trade["author_id"] != event["author_id"]:
        return False
    for key in IDENTITY:
        left, right = trade.get(key), event.get(key)
        if left not in (None, "unknown") and right not in (None, "unknown") and left != right:
            return False
    return True


def full_contract(event: dict) -> bool:
    if event.get("instrument_type") == "option":
        return all(event.get(key) is not None for key in ("symbol", "option_type", "strike", "expiry"))
    return event.get("instrument_type") in {"equity", "future"} and bool(event.get("symbol"))


def make_trade(event: dict, orphan: bool = False) -> dict:
    return {"trade_id": f"trade:{event['event_id']}", "author_id": event["author_id"],
            "author_name": event["author_name"], **{k: event.get(k) for k in IDENTITY},
            "entry_event_id": None if orphan or event["event_type"] == "order_pending" else event["event_id"],
            "origin_event_id": event["event_id"], "entry_message_id": None if orphan else event["message_id"],
            "origin": "orphan_management" if orphan else "order_intent" if event["event_type"] == "order_pending" else "entry_claim_or_instruction",
            "channel_id": event["channel_id"], "event_ids": [], "context_event_ids": [],
            "source_message_ids": [], "link_confidence": "low" if orphan else event.get("confidence", "low"),
            "status": "entry_not_observed" if orphan else "pending_order" if event["event_type"] == "order_pending" else "entry_reported_unverified",
            "reported_entry_price": None if orphan else event.get("price"),
            "reported_entry_quantity": None if orphan else event.get("quantity"),
            "confirmed_fill_quantity": None, "realized_pnl": None,
            "conflicts": [], "limitations": ["No broker fill records in this export; reported executions remain unverified."],
            "first_timestamp_utc": event["timestamp_utc"], "last_timestamp_utc": event["timestamp_utc"],
            "entry_available_at_utc": None if orphan else event.get("available_at_utc")}


def apply_event(trade: dict, event: dict, method: str, confidence: str) -> None:
    trade["event_ids"].append(event["event_id"])
    trade["source_message_ids"] = sorted(set(trade["source_message_ids"] + event["source_message_ids"]))
    trade["last_timestamp_utc"] = event["timestamp_utc"]
    event["trade_id"] = trade["trade_id"]
    event["link_method"] = method
    event["link_confidence"] = confidence
    if confidence == "low":
        trade["link_confidence"] = "low"
    elif confidence == "medium" and trade["link_confidence"] == "high":
        trade["link_confidence"] = "medium"
    action, speech = event["event_type"], event["speech_act"]
    if action == "entry" and trade["entry_event_id"] is None:
        trade["entry_event_id"] = event["event_id"]
        trade["entry_message_id"] = event["message_id"]
        trade["entry_available_at_utc"] = event.get("available_at_utc")
        trade["reported_entry_price"] = event.get("price")
        trade["reported_entry_quantity"] = event.get("quantity")
        trade["status"] = "entry_reported_unverified"
    if action == "exit" and speech == "claimed_execution":
        trade["status"] = "exit_reported_unverified"
    elif action == "exit" and speech == "instruction":
        trade["status"] = "exit_instruction_no_confirmed_fill"
    elif action == "order_cancel" and speech not in {"conditional", "intention"}:
        trade["status"] = "order_cancellation_reported"
    elif action == "invalidation":
        trade["limitations"].append("Invalidation is not proof a stop or exit filled.")
    if action in {"add", "trim", "exit"} and event.get("quantity") is None:
        trade["limitations"].append("Management event lacks an explicit contract/share quantity; remaining quantity unknown.")
    if action == "correction":
        for key in IDENTITY:
            if event.get(key) is not None and event.get(key) != trade.get(key):
                trade["conflicts"].append({"field": key, "original": trade.get(key), "later_correction": event[key],
                                           "source_message_ids": [event["message_id"]],
                                           "rule": "Retain original entry knowledge; correction takes effect only at its own availability time."})
    trade["limitations"] = list(dict.fromkeys(trade["limitations"] + event.get("uncertainties", [])))


def reconcile(events: list[dict]) -> tuple[list[dict], list[dict]]:
    trades: list[dict] = []
    by_message: dict[str, list[dict]] = defaultdict(list)
    links = []
    for event in events:
        action = event["event_type"]
        e_time = event.get("available_at_utc") or event["timestamp_utc"]
        explicit_refs = list(dict.fromkeys([event.get("reply_to_message_id")] + list(event.get("related_message_ids") or [])))
        explicit = {t["trade_id"]: t for ref in explicit_refs if ref for t in by_message.get(str(ref), [])
                    if t["author_id"] == event["author_id"] and (compatible(t, event) or action == "correction")}
        candidates = []
        if len(explicit) == 1:
            candidates = list(explicit.values())
            method, conf = "explicit_message_reference", "high"
        else:
            for t in trades:
                if not event.get("symbol") or t.get("symbol") != event.get("symbol") or not compatible(t, event):
                    continue
                if t["status"] in {"exit_reported_unverified", "exit_instruction_no_confirmed_fill", "order_cancellation_reported"}:
                    continue
                if t.get("expiry") and t["expiry"] < e_time[:10]:
                    continue  # Not a claim of expiry settlement; no automatic P&L.
                if t["channel_id"] != event["channel_id"] and not (full_contract(t) and full_contract(event)):
                    continue
                candidates.append(t)
            method = "unique_compatible_contract" if full_contract(event) else "unique_symbol_in_same_author_channel"
            conf = "medium" if full_contract(event) else "low"
        # New entries are not silently netted against old entries. Only explicit
        # replies or one fully specified pending order can establish continuity.
        if action in OPENERS:
            explicit_match = len(explicit) == 1
            pending = [t for t in candidates if t["status"] == "pending_order" and full_contract(event) and full_contract(t)]
            if explicit_match:
                chosen = list(explicit.values())[0]
            elif action == "entry" and len(pending) == 1:
                chosen, method, conf = pending[0], "matching_fully_specified_pending_order", "medium"
            else:
                chosen = make_trade(event)
                trades.append(chosen)
                if candidates:
                    chosen["limitations"].append("Other compatible lifecycles exist; this entry was not silently merged with them.")
                    links.append({"event_id": event["event_id"], "candidate_trade_ids": [t["trade_id"] for t in candidates],
                                  "source_message_ids": event["source_message_ids"], "status": "possible_reentry_add_or_crosspost_unresolved", "confidence": "low"})
                method, conf = "explicit_new_entry_or_pending_order", event.get("confidence", "low")
            apply_event(chosen, event, method, conf)
            by_message[event["message_id"]].append(chosen)
        elif action in MANAGEMENT and action != "position_update":
            if len(candidates) == 1:
                chosen = candidates[0]
                apply_event(chosen, event, method, conf)
                by_message[event["message_id"]].append(chosen)
            else:
                links.append({"event_id": event["event_id"], "candidate_trade_ids": [t["trade_id"] for t in candidates],
                              "source_message_ids": event["source_message_ids"],
                              "status": "ambiguous_management" if candidates else "no_observed_matching_entry", "confidence": "low"})
                orphan = make_trade(event, orphan=True)
                if candidates:
                    orphan["limitations"].append("Multiple compatible entries: no lifecycle link has been selected.")
                trades.append(orphan)
                apply_event(orphan, event, "unresolved_management_record", "low")
                by_message[event["message_id"]].append(orphan)
        else:
            event["trade_id"] = None
            event["link_method"] = "context_or_performance_claim_not_a_fill"
            event["link_confidence"] = "low" if len(candidates) != 1 else conf
            if len(candidates) == 1:
                candidates[0]["context_event_ids"].append(event["event_id"])
            elif len(candidates) > 1:
                links.append({"event_id": event["event_id"], "candidate_trade_ids": [t["trade_id"] for t in candidates],
                              "source_message_ids": event["source_message_ids"], "status": "ambiguous_context", "confidence": "low"})
        for key in ("trade_id", "link_method", "link_confidence"):
            event["field_provenance"][key] = {"source_message_ids": event["source_message_ids"], "basis": "conservative_lifecycle_reconciliation_v1", "confidence": event["link_confidence"]}
    for t in trades:
        t["field_provenance"] = provenance(t, t["source_message_ids"], "derived_from_cited_events_with_no_broker_fill_confirmation", t["link_confidence"])
        first = next(e for e in events if e["event_id"] == t["origin_event_id"])
        for key in IDENTITY:
            t["field_provenance"][key] = {"source_message_ids": first.get("field_sources", {}).get(key, [first["message_id"]]),
                                           "basis": "origin_event_only_later_corrections_not_backfilled", "confidence": first.get("confidence", "low")}
    return trades, links


def source_link(message: dict) -> str:
    return str(ROOT / message["source_file"]) + "#" + message["source_anchor"].lstrip("#")


def research_trade(trade: dict, events: list[dict], event_map: dict[str, dict], messages: dict[str, dict], market: Any = None) -> dict:
    entry = event_map[trade["entry_event_id"] or trade["origin_event_id"]]
    at = entry.get("available_at_utc")
    eligible = []
    if at and trade.get("symbol"):
        for context in events:
            ct = context.get("available_at_utc")
            if context["symbol"] != trade["symbol"] or context["author_id"] != trade["author_id"]:
                continue
            if context["event_type"] not in CONTEXT or not ct or not (stamp(ct) < stamp(at)) or context.get("pit_evidence_blocked"):
                continue
            # Declared relevance window, not a claim that older theses expired.
            if stamp(at) - stamp(ct) <= timedelta(days=30):
                eligible.append(context)
    eligible.sort(key=lambda e: e["available_at_utc"], reverse=True)
    research = {"trade_id": trade["trade_id"], "symbol": trade["symbol"],
                "source_message_ids": entry["source_message_ids"], "as_of_utc": at,
                "original_alert_timestamp_utc": entry["timestamp_utc"],
                "pit_status": "blocked_missing_entry" if not trade["entry_event_id"] else "blocked_edited_version_time_unknown" if not at else "blocked_late_field_evidence" if entry.get("pit_evidence_blocked") else "retrospective_asof_reconstruction_not_certified_live_capture",
                "prior_context": [{"event_id": e["event_id"], "source_message_ids": e["source_message_ids"],
                                   "available_at_utc": e["available_at_utc"], "source_quote": e["source_quote"],
                                   "claim_status": "unverified_trader_context"} for e in eligible],
                "context_rule": "Same caller/symbol, preceding available version, 30 calendar days; future recaps/corrections excluded.",
                "external_catalyst_verification": "not_established_by_export; source messages are claims, not independent news confirmation",
                "underlying_market_context": None, "follower_execution": [],
                "limitations": ["No assumption that a channel alert equals the trader's or follower's fill.",
                                "Current HTML cannot reveal deleted messages or prior versions of edited messages."]}
    if at and trade["entry_event_id"] and not entry.get("pit_evidence_blocked") and trade.get("symbol") and market:
        research["underlying_market_context"] = market.lookup_context(trade["symbol"], at)
    if not trade["entry_event_id"]:
        block = "No entry was observed; follower entry cannot be reconstructed."
    elif not at or entry.get("pit_evidence_blocked"):
        block = "Entry evidence availability cannot be established without later information."
    elif entry["event_type"] == "order_pending" or entry["speech_act"] in {"conditional", "intention", "retrospective"}:
        block = "Source does not establish an actionable contemporaneous entry."
    elif trade["instrument_type"] == "option" and not full_contract(entry):
        block = "Incomplete option identity at alert time; cannot select an executable contract."
    elif trade["instrument_type"] == "option":
        block = "No validated contemporaneous historical option bid/ask quote and size series; trade bars cannot estimate executable option prices."
    elif trade["instrument_type"] in {None, "unknown", "future"}:
        block = "Instrument or supporting executable bid/ask data is unresolved."
    else:
        block = "Underlying bars are trades, not executable bid/ask quotes; no queue, spread or order-size evidence."
    for delay in (30, 60, 300):
        research["follower_execution"].append({"delay_seconds": delay,
            "earliest_submission_utc": iso(stamp(at) + timedelta(seconds=delay)) if at else None,
            "estimated_fill_price": None, "estimated_fill_quantity": None, "estimated_pnl": None,
            "status": "not_estimable_from_available_evidence", "reason": block,
            "source_message_ids": entry["source_message_ids"],
            "scenario_assumptions": "Delay starts at available exported message version; excludes unknown personal notification/order-routing lag."})
    sources = sorted(set(entry["source_message_ids"] + [ref for e in eligible for ref in e["source_message_ids"]]))
    research["field_provenance"] = provenance(research, sources, "asof_source_filter_and_explicit_execution_data_requirements")
    return research


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False, sort_keys=True) if isinstance(value, (dict, list)) else value for key, value in row.items()})


def coverage(messages: dict[str, dict], events: list[dict]) -> list[dict]:
    by_id = defaultdict(list)
    for e in events:
        by_id[e["message_id"]].append(e["event_id"])
    rows = []
    for mid, m in messages.items():
        caller = m["author_id"] in CALLERS
        text = m.get("text") or ""
        image_only = bool(m.get("attachments")) and len(re.sub(r"@\w+|\W", "", text)) < 12
        suspicious = bool(re.search(r"\b(BTO|STC|STO|BTC|bought|buying|sold|selling|trim|stop|calls?|puts?|entry|exit|position|filled|adding)\b", text, re.I))
        status = "annotated" if by_id[mid] else "other_author_preserved_not_caller_trade" if not caller else "attachment_requires_review" if image_only else "unannotated_trade_language_review_required" if suspicious else "no_extracted_trade_event"
        rows.append({"message_id": mid, "author_id": m["author_id"], "channel_id": m["channel_id"],
                     "timestamp_utc": m["timestamp_utc"], "event_ids": by_id[mid], "status": status,
                     "has_attachments": bool(m.get("attachments")), "is_edited": m["is_edited"],
                     "source_message_ids": [mid]})
    return rows


def render_report(out: Path, trades: list[dict], events: list[dict], research: list[dict], coverage_rows: list[dict], source_manifest: dict, links: list[dict], issues: list[dict]) -> None:
    by_caller = Counter(t["author_name"] for t in trades if t["entry_event_id"])
    lines = ["# Vault of Ace: auditable managed-trade evidence ledger", "",
             "Research reconstruction of the supplied export. Trader messages are unverified evidence; this is not a broker statement or a validated performance backtest.", "",
             f"- Archived messages: **{len(coverage_rows):,}**; source-derived events: **{len(events):,}**.",
             f"- Entry-associated lifecycles: **{sum(bool(t['entry_event_id']) for t in trades):,}**; order/orphan records: **{sum(not t['entry_event_id'] for t in trades):,}**.",
             f"- Caller entry counts: `{dict(by_caller)}`. ACE and FT are separate traders.",
             f"- Ambiguous/unmatched links: **{len(links):,}**; provenance/time issues: **{len(issues)}**.",
             f"- Message coverage: `{dict(Counter(r['status'] for r in coverage_rows))}`.", "",
             "## Scope and evidence limits", "",
             "All seven VIP exports contain zero messages. Empty exports do not prove no VIP trading occurred. The five populated free channels span October 2025–September 2026. Image-only content and unresolved management remain visible in coverage and attachment reports; no missing trade is treated as a breakeven.", "",
             "The export footer says UTC-5, but rendered message times follow America/New_York daylight-saving transitions. Discord snowflake creation times provide millisecond UTC chronology and were cross-checked against all rendered timestamps. Both the literal footer and rendered strings are preserved. Edited messages preserve their final visible version and edit marker; the original text/history is unavailable. Final edited text is eligible only after the conservative end of its edit minute.", "",
             "Every ledger/event field has `field_provenance` with source message IDs and its interpretation basis. High confidence means the extraction/link is clear, not that the trader's claim is verified. Prices remain reported references; fill quantities and P&L stay null. CSVs contain JSON provenance cells; the JSONL files are canonical.", "",
             "## Point-in-time research and follower estimates", "",
             "Each lifecycle has a research row and a readable case entry. Prior same-caller/symbol context is restricted to the preceding 30 days and eligible message versions. Later recaps, corrections and outcomes do not improve entry knowledge. Local historical market observations, where available, are clearly identified as retrospectively fetched and cannot certify original publication/revision history.", "",
             "Follower scenarios explicitly model 30-, 60- and 300-second delays from evidence availability. Reported alert prices and screenshots are retained as useful source claims; they are not assumed to be a later follower's fill. A numerical market-based option fill needs a timestamped contract quote/size series and a stated order rule. Missing estimates are null with per-trade reasons, never replaced by alert prices or synthetic option marks.", "",
             "## Files", "",
             "- `managed_trades.jsonl` / `.csv`: conservative lifecycle ledger, including unresolved management records.",
             "- `events.jsonl` / `.csv`: every annotated trade, management, watchlist and performance claim.",
             "- `trade_research.jsonl`: per-lifecycle as-of context and follower scenarios.",
             "- `trade_cases.md`: source-linked chronology and research for every lifecycle.",
             "- `unresolved_links.jsonl`, `message_coverage.csv`, `validation.json`: uncertainty and audit coverage.",
             "- `index.html`: local searchable ledger; `manifest.json`: input/output hashes and methodology version.", "",
             "## Method sources", "",
             "- [Discord snowflake format](https://docs.discord.com/developers/reference#snowflakes): message creation timestamp decoding.",
             "- [Alpaca historical options data](https://docs.alpaca.markets/us/docs/historical-option-data): feed distinctions; indicative quotes are not actual OPRA quotes.", "",
             "See the parent directory's `data_feasibility.md` and annotation/attachment review notes for detailed coverage findings."]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_cases(out: Path, trades: list[dict], events: list[dict], research: list[dict], messages: dict[str, dict]) -> None:
    emap, rmap = {e["event_id"]: e for e in events}, {r["trade_id"]: r for r in research}
    lines = ["# Per-trade evidence and as-of research", ""]
    for t in trades:
        r = rmap[t["trade_id"]]
        contract = " ".join(str(t[k]) for k in ("symbol", "strike", "option_type", "expiry") if t.get(k) is not None) or "Unresolved instrument"
        lines += [f"## {t['trade_id']} — {t['author_name']} — {contract}", "", f"Status: `{t['status']}`; link confidence: `{t['link_confidence']}`. Original entry fields are never backfilled with later corrections.", ""]
        for eid in t["event_ids"]:
            e = emap[eid]
            quote = e["source_quote"].replace("\n", " ")
            lines += [f"- {e['timestamp_utc']} — **{e['event_type']}** / {e['speech_act']} — [{e['message_id']}](<{source_link(messages[e['message_id']])}>) — {quote}"]
        lines += ["", f"As-of research: `{r['pit_status']}` at `{r['as_of_utc']}`.", ""]
        if r["prior_context"]:
            for c in r["prior_context"]:
                mid = c["source_message_ids"][0]
                lines += [f"- Earlier caller context ({c['available_at_utc']}): [{mid}](<{source_link(messages[mid])}>) — {c['source_quote'].replace(chr(10), ' ')}"]
        else:
            lines += ["No eligible earlier same-caller/symbol context was identified within the declared 30-day window."]
        market = r["underlying_market_context"]
        if market:
            lines += ["", "Underlying data research:", "", "```json", json.dumps(market, ensure_ascii=False, indent=2), "```"]
        lines += ["", "Follower execution: " + r["follower_execution"][0]["reason"], "", "Delays assessed: 30 / 60 / 300 seconds. Estimated prices, quantities and P&L remain unknown without supporting execution evidence.", ""]
        if t["conflicts"]:
            lines += ["Conflicts: `" + json.dumps(t["conflicts"], ensure_ascii=False) + "`", ""]
        lines += ["Limitations: " + " ".join(t["limitations"]), ""]
    (out / "trade_cases.md").write_text("\n".join(lines), encoding="utf-8")
    # Escape all source text; no source HTML or remote assets enter this viewer.
    rows = []
    for t in trades:
        timeline = "\n".join(f"{emap[eid]['timestamp_utc']} {emap[eid]['event_type']}: {emap[eid]['source_quote']} [message {emap[eid]['message_id']}]" for eid in t["event_ids"])
        columns = [t["trade_id"], t["author_name"], " ".join(str(t[k]) for k in ("symbol", "strike", "option_type", "expiry") if t.get(k) is not None), t["status"], t["link_confidence"], timeline]
        rows.append("<tr>" + "".join("<td>" + html.escape(str(c)) + "</td>" for c in columns) + "</tr>")
    page = """<!doctype html><html><head><meta charset="utf-8"><title>Discord trade evidence ledger</title><style>body{font:15px system-ui;margin:2rem;background:#10151f;color:#e3e9f1}input{padding:.7rem;width:min(80%,700px)}table{border-collapse:collapse;width:100%;margin-top:1rem}td,th{border:1px solid #465066;text-align:left;padding:.6rem;vertical-align:top}td:last-child{white-space:pre-wrap;min-width:400px}a{color:#94c9ff}</style></head><body><h1>Discord trade evidence ledger</h1><p>Unverified source claims. Unknown fills and P&amp;L remain unknown. <a href="report.md">Methodology</a> · <a href="managed_trades.csv">CSV</a> · <a href="trade_cases.md">Per-trade research</a></p><input id="q" placeholder="Filter by caller, ticker, message, action or status"><table><thead><tr><th>ID</th><th>Caller</th><th>Instrument at origin</th><th>Status</th><th>Link confidence</th><th>Evidence timeline</th></tr></thead><tbody>""" + "".join(rows) + """</tbody></table><script>document.getElementById('q').addEventListener('input',e=>{const q=e.target.value.toLowerCase();document.querySelectorAll('tbody tr').forEach(r=>r.hidden=!r.textContent.toLowerCase().includes(q));});</script></body></html>"""
    (out / "index.html").write_text(page, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=DEFAULT_STUDY)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--skip-market-context", action="store_true")
    args = parser.parse_args()
    source = args.study / "source/messages.jsonl"
    messages_list = read_jsonl(source)
    messages = {str(m["message_id"]): m for m in messages_list}
    if len(messages) != len(messages_list):
        raise ValueError("Duplicate source IDs require explicit reconciliation, not silent deduplication")
    curated_path = args.study / "curated/events.jsonl"
    paths = [curated_path] if curated_path.exists() else sorted((args.study / "annotations").glob("*.jsonl"))
    if not paths:
        raise ValueError("No annotations available")
    events, issues = normalize_annotations(paths, messages)
    trades, links = reconcile(events)
    market = None
    if not args.skip_market_context:
        from scripts.discord_ledger import market_context as market
    research = [research_trade(t, events, {e["event_id"]: e for e in events}, messages, market) for t in trades]
    coverage_rows = coverage(messages, events)
    args.out.mkdir(parents=True, exist_ok=False)
    for name, rows in (("events", events), ("managed_trades", trades), ("trade_research", research), ("unresolved_links", links), ("provenance_issues", issues)):
        write_jsonl(args.out / f"{name}.jsonl", rows)
    write_csv(args.out / "events.csv", events)
    write_csv(args.out / "managed_trades.csv", trades)
    write_csv(args.out / "message_coverage.csv", coverage_rows)
    source_manifest = json.loads((args.study / "source/manifest.json").read_text())
    render_report(args.out, trades, events, research, coverage_rows, source_manifest, links, issues)
    render_cases(args.out, trades, events, research, messages)
    checks = {"message_count": len(messages), "event_count": len(events), "trade_record_count": len(trades),
              "source_ids_resolve": True, "authors_match_source": True, "channels_match_source": True,
              "event_ids_unique": len({e["event_id"] for e in events}) == len(events),
              "all_research_rows_present": len(research) == len(trades), "future_or_unknown_evidence_issues": len(issues),
              "all_trade_fields_cited": all(set(t) - {"field_provenance"} <= set(t["field_provenance"]) for t in trades),
              "no_fabricated_fill_pnl": all(t["confirmed_fill_quantity"] is None and t["realized_pnl"] is None for t in trades),
              "unresolved_links": len(links), "coverage": dict(Counter(r["status"] for r in coverage_rows)),
              "semantic_extraction_status": "see_annotation_reviews_and_unresolved_coverage_not_a_claim_of_perfect_recall"}
    (args.out / "validation.json").write_text(json.dumps(checks, indent=2) + "\n")
    manifest = {"schema_version": SCHEMA, "created_at_utc": iso(datetime.now(timezone.utc)),
                "inputs": [{"path": str(p.resolve().relative_to(ROOT)), "sha256": digest(p)} for p in [source, args.study / "source/manifest.json", *paths]],
                "code": [{"path": str(p.relative_to(ROOT)), "sha256": digest(p)} for p in sorted(Path(__file__).parent.glob("*.py"))],
                "outputs": [{"path": p.name, "sha256": digest(p)} for p in sorted(args.out.iterdir()) if p.is_file()],
                "execution": "offline_read_only_research_no_orders", "all_source_claims_unverified": True,
                "context_lookback_days": 30, "follower_delays_seconds": [30, 60, 300]}
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
