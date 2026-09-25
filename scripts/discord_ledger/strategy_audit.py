"""Audit pre-alert watchlist context and source-reported option outcomes.

This is a research index, not a broker-fill or executable-P&L backtest. The
original export and canonical lifecycle run are read-only inputs.
"""
from __future__ import annotations

import argparse
from datetime import timedelta
import json
from pathlib import Path
import re
from zoneinfo import ZoneInfo

from scripts.discord_ledger.build import available_at, read_jsonl, stamp, write_jsonl

STUDY = Path("research/discord_ledger_2026-09-21")
MANAGEMENT = {"add", "trim", "exit", "stop_adjustment", "invalidation"}
NY = ZoneInfo("America/New_York")


def literal_symbol(text: str, symbol: str) -> bool:
    """A candidate textual mention, not semantic evidence of an idea."""
    return bool(symbol and re.search(r"(?<![A-Za-z0-9])\$?" + re.escape(symbol) + r"(?![A-Za-z0-9])", text))


def watchlist_candidates(entry: dict, messages: list[dict], *, days: int = 7) -> list[dict]:
    symbol, asof = entry.get("symbol"), entry.get("available_at_utc")
    if not symbol or not asof:
        return []
    cutoff = stamp(asof) - timedelta(days=days)
    found = []
    for message in messages:
        if message["author_id"] != entry["author_id"] or "watchlists" not in message["channel_name"]:
            continue
        when, basis = available_at(message)
        if not when or not cutoff <= stamp(when) < stamp(asof):
            continue
        if not literal_symbol(message.get("text") or "", symbol):
            continue
        text = message.get("text") or ""
        explicit_idea = bool(re.search(r"\b(?:watchlist|on watch|watching|looking at|looking for|want to enter|idea|entry|above|below|breakout|breaks?)\b", text, re.I))
        found.append({
            "message_id": message["message_id"],
            "created_at_utc": message["timestamp_utc"],
            "available_at_utc": when,
            "availability_basis": basis,
            "watchlist_date_et": stamp(when).astimezone(NY).date().isoformat(),
            "text": text,
            "candidate_type": "explicit_idea_language" if explicit_idea else "symbol_mention_only",
            "link_confidence": "medium" if explicit_idea else "low",
            "source_message_ids": [message["message_id"]],
            "caveat": "Same-caller/symbol watchlist-channel mention; not proof this idea caused the later entry.",
        })
    return sorted(found, key=lambda row: row["available_at_utc"], reverse=True)


def price_is_premium(event: dict) -> bool:
    """Reject percentage claims that a permissive extractor rendered as dollars."""
    price = event.get("price")
    if not isinstance(price, (int, float)) or price <= 0:
        return False
    text = (event.get("price_text") or "").strip()
    quote = event.get("source_quote") or ""
    if text and re.search(re.escape(text) + r"\s*%", quote, re.I):
        return False
    return True


def outcome_evidence(trade: dict, events: dict[str, dict]) -> dict:
    entry_id = trade.get("entry_event_id")
    entry = events.get(entry_id) if entry_id else None
    linked = [events[eid] for eid in dict.fromkeys(trade["event_ids"]) if eid in events]
    management = [e for e in linked if e["event_type"] in MANAGEMENT and e["event_id"] != entry_id]
    priced = [e for e in management if e["event_type"] in {"trim", "exit"} and price_is_premium(e)]
    claims = []
    for event in priced:
        if not entry or not price_is_premium(entry):
            continue
        claims.append({
            "event_id": event["event_id"], "event_type": event["event_type"],
            "reported_exit_price": event["price"],
            "reference_price_change_pct": round(100 * (event["price"] / entry["price"] - 1), 3),
            "exit_quantity_claimed": event.get("quantity"),
            "source_message_ids": sorted(set(entry["source_message_ids"] + event["source_message_ids"])),
            "confidence": "low" if trade["link_confidence"] == "low" or not event.get("strike") else "medium",
            "meaning": "Price change from source-reported entry reference to this reported exit/trim; not total trade return or follower fill.",
        })
    adds = [e for e in management if e["event_type"] == "add"]
    exits = [e for e in management if e["event_type"] in {"trim", "exit"}]
    entry_qty = entry.get("quantity") if entry else None
    known_exit_qty = sum(e.get("quantity") or 0 for e in exits)
    complete_claimed = bool(
        entry and entry["speech_act"] == "claimed_execution" and price_is_premium(entry)
        and entry_qty and not adds and exits and len(priced) == len(exits)
        and all(e["speech_act"] == "claimed_execution" and e.get("quantity") for e in exits)
        and known_exit_qty == entry_qty
        and all(e.get("symbol") in (None, entry.get("symbol")) for e in exits)
        and all(e.get("strike") in (None, entry.get("strike")) for e in exits)
        and all(e.get("option_type") in (None, entry.get("option_type")) for e in exits)
    )
    weighted_change = None
    if complete_claimed:
        weighted_change = round(100 * (sum(e["quantity"] * e["price"] for e in exits) / entry_qty / entry["price"] - 1), 3)
    return {
        "trade_id": trade["trade_id"], "author_name": trade["author_name"],
        "symbol": trade["symbol"], "entry_event_id": entry_id,
        "entry_message_id": trade.get("entry_message_id"),
        "entry_speech_act": entry.get("speech_act") if entry else None,
        "reported_entry_price": entry.get("price") if entry and price_is_premium(entry) else None,
        "reported_entry_quantity": entry_qty,
        "linked_management_event_ids": [e["event_id"] for e in management],
        "linked_exit_or_trim_event_ids": [e["event_id"] for e in exits],
        "reported_price_change_legs": claims,
        "complete_source_claimed_quantity_cycle": complete_claimed,
        "complete_cycle_reported_gross_price_change_pct": weighted_change,
        "broker_verified_pnl": None, "follower_executable_return": None,
        "link_confidence": trade["link_confidence"],
        "source_message_ids": sorted(set(trade["source_message_ids"])),
        "field_provenance": {key: {"source_message_ids": sorted(set(trade["source_message_ids"])),
                                   "basis": "linked_source_claims_with_strict_price_and_quantity_gates",
                                   "confidence": "low" if trade["link_confidence"] == "low" else "medium"}
                             for key in ("reported_price_change_legs", "complete_source_claimed_quantity_cycle",
                                         "complete_cycle_reported_gross_price_change_pct")},
    }


def build(messages: list[dict], event_rows: list[dict], trades: list[dict]) -> tuple[list[dict], list[dict], dict]:
    events = {e["event_id"]: e for e in event_rows}
    ideas, outcomes = [], []
    for trade in trades:
        if not trade.get("entry_event_id"):
            continue
        entry = events[trade["entry_event_id"]]
        candidates = watchlist_candidates(entry, messages)
        prior = candidates[0] if candidates else None
        prior_explicit = next((c for c in candidates if c["candidate_type"] == "explicit_idea_language"), None)
        row = {
            "trade_id": trade["trade_id"], "author_name": trade["author_name"],
            "symbol": entry.get("symbol"), "entry_message_id": entry["message_id"],
            "entry_available_at_utc": entry.get("available_at_utc"),
            "entry_text": entry["source_quote"],
            "prior_watchlist_candidates_7d": candidates,
            "nearest_candidate_message_id": prior["message_id"] if prior else None,
            "nearest_candidate_date_et": prior["watchlist_date_et"] if prior else None,
            "nearest_explicit_idea_message_id": prior_explicit["message_id"] if prior_explicit else None,
            "watchlist_link_status": "candidate_only" if prior else "none_found_in_export",
            "underlying_trigger_status": "unverified; requires intraday underlying features and matched non-alert intervals",
            "contract_selection_status": "contract identity is a source claim; contemporaneous option spread/depth/OI not yet joined",
            "linked_management_event_ids": [eid for eid in trade["event_ids"] if eid != trade["entry_event_id"] and events[eid]["event_type"] in MANAGEMENT],
            "source_message_ids": sorted(set(entry["source_message_ids"] + [c["message_id"] for c in candidates])),
        }
        row["field_provenance"] = {key: {"source_message_ids": row["source_message_ids"],
                                         "basis": "same_author_literal_symbol_pre_alert_watchlist_channel_7d",
                                         "confidence": "low"}
                                   for key in row}
        ideas.append(row)
        outcomes.append(outcome_evidence(trade, events))
    summary = {}
    for author in ("ACE", "FT"):
        irows = [r for r in ideas if r["author_name"] == author]
        orows = [r for r in outcomes if r["author_name"] == author]
        summary[author] = {
            "entry_lifecycles": len(irows),
            "prior_watchlist_candidate_7d": sum(bool(r["prior_watchlist_candidates_7d"]) for r in irows),
            "prior_explicit_idea_language_7d": sum(bool(r["nearest_explicit_idea_message_id"]) for r in irows),
            "linked_management": sum(bool(r["linked_management_event_ids"]) for r in irows),
            "linked_exit_or_trim": sum(bool(r["linked_exit_or_trim_event_ids"]) for r in orows),
            "paired_reported_price_legs": sum(bool(r["reported_price_change_legs"]) for r in orows),
            "complete_source_claimed_quantity_cycles": sum(r["complete_source_claimed_quantity_cycle"] for r in orows),
            "complete_cycle_trade_ids": [r["trade_id"] for r in orows if r["complete_source_claimed_quantity_cycle"]],
        }
    return ideas, outcomes, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=STUDY)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    ideas, outcomes, summary = build(read_jsonl(args.study / "source/messages.jsonl"),
                                     read_jsonl(args.run / "events.jsonl"),
                                     read_jsonl(args.run / "managed_trades.jsonl"))
    args.out.mkdir(parents=True, exist_ok=False)
    write_jsonl(args.out / "watchlist_entry_candidates.jsonl", ideas)
    write_jsonl(args.out / "outcome_evidence.jsonl", outcomes)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
