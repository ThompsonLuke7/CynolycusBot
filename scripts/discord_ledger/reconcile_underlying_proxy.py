"""Build a broad, source-cited outcome proxy for every ACE entry lifecycle.

This deliberately answers a narrower question than option P&L: after a
reported entry, did the *underlying* move in the direction of the option by
the time of the best available reported management/closure message?  It is a
way to triage winners/losses when option fills are absent, not a replacement
for them.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, time, timedelta
import json
from pathlib import Path
import re
from zoneinfo import ZoneInfo

from scripts.discord_ledger.build import read_jsonl, stamp, write_jsonl


MANAGEMENT = {"add", "trim", "exit", "stop_adjustment", "invalidation"}
FULL_CLOSE = re.compile(r"\b(?:all\s+out|fully\s+out|i[’']?m\s+out|i\s+am\s+out|closing|closed|stop(?:ped)?\s+out|stop\s+hit|sl\s+hit)\b", re.I)
LOSS = re.compile(r"\b(?:loss|stop(?:ped)?\s+out|stop\s+hit|sl\s+hit)\b", re.I)
NY = ZoneInfo("America/New_York")


def literal_symbol(text: str, symbol: str | None) -> bool:
    return bool(symbol and re.search(r"(?<![A-Za-z0-9])\$?" + re.escape(symbol) + r"(?![A-Za-z0-9])", text or "", re.I))


def option_sign(trade: dict) -> int | None:
    """Map long option right to the direction required for an underlying gain."""

    if trade.get("direction") not in {None, "long"}:
        return None
    right = trade.get("option_type")
    return 1 if right == "call" else -1 if right == "put" else None


def usable_management(event: dict, author: str) -> bool:
    return event.get("author_name") == author and event.get("event_type") in MANAGEMENT


def is_full_close(event: dict) -> bool:
    return event.get("event_type") == "exit" or bool(FULL_CLOSE.search(event.get("source_quote") or ""))


def window_end(trade: dict) -> object:
    entry = stamp(trade["entry_available_at_utc"])
    expiry = trade.get("expiry")
    if expiry:
        # Management posted very shortly after the named expiry may be a
        # delayed recap. Do not search indefinitely into later same-symbol
        # positions.
        return min(entry + timedelta(days=30), stamp(expiry + "T23:59:59Z") + timedelta(days=3))
    return entry + timedelta(days=14)


def expiry_horizon(trade: dict) -> dict | None:
    """A hypothetical expiry observation, never a claimed or inferred exit."""

    expiry = trade.get("expiry")
    if not expiry or not trade.get("entry_available_at_utc"):
        return None
    close = datetime.combine(datetime.fromisoformat(expiry).date(), time(16), tzinfo=NY).astimezone(ZoneInfo("UTC"))
    if close <= stamp(trade["entry_available_at_utc"]):
        return None
    return {"anchor_type": "unverified_expiry_horizon", "timestamp_utc": close.isoformat().replace("+00:00", "Z"),
            "source_message_ids": trade["source_message_ids"],
            "link_confidence": "not_a_management_link",
            "limitation": "No message establishes the position was held to expiry; this is a separate directional horizon only."}


def match_score(trade: dict, event: dict) -> tuple[float, list[str]] | None:
    """Score only evidence-based possible links; no bare chronological join."""

    entry_at = stamp(trade["entry_available_at_utc"])
    event_at = stamp(event.get("available_at_utc") or event["timestamp_utc"])
    if not entry_at < event_at <= window_end(trade):
        return None
    text = event.get("source_quote") or ""
    symbol = trade.get("symbol")
    event_symbol = event.get("symbol")
    literal = literal_symbol(text, symbol)
    stated_symbol = event_symbol == symbol and literal_symbol(text, event_symbol)
    same_trade = event.get("trade_id") == trade["trade_id"]
    same_contract = bool(
        event.get("symbol") == symbol
        and event.get("option_type") == trade.get("option_type")
        and event.get("strike") == trade.get("strike")
        and event.get("expiry") == trade.get("expiry")
        and all(value is not None for value in (event.get("strike"), event.get("expiry"), trade.get("strike"), trade.get("expiry")))
    )
    # An explicitly labelled event can be linked broadly. An unlabelled close
    # gets a deliberately separate, very-low-confidence tier only when it is a
    # full-close statement in the same channel shortly after the entry. This
    # is the user-requested *estimate* path, never merged with reported fills.
    temporal_fallback = (
        is_full_close(event)
        and event.get("channel_id") == trade.get("channel_id")
        and event_at - entry_at <= timedelta(hours=36)
        and trade.get("status") not in {"exit_reported_unverified", "exit_instruction_no_confirmed_fill"}
    )
    if not (same_trade or literal or same_contract or temporal_fallback):
        return None
    score, reasons = 0.0, []
    if same_trade:
        score += 100.0
        reasons.append("canonical_lifecycle_link")
    if same_contract:
        score += 60.0
        reasons.append("matching_visible_option_contract")
    if literal:
        score += 35.0
        reasons.append("symbol_literal_in_management_text")
    if stated_symbol:
        score += 10.0
        reasons.append("annotated_symbol_agrees_with_text")
    if event.get("option_type") and event.get("option_type") == trade.get("option_type"):
        score += 5.0
        reasons.append("matching_option_right")
    if event.get("channel_id") == trade.get("channel_id"):
        score += 3.0
        reasons.append("same_channel")
    if temporal_fallback and not (same_trade or literal or same_contract):
        score += 4.0
        reasons.append("unlabelled_full_close_same_channel_within_36h")
    # Prefer the closer position where evidence otherwise ties, while keeping
    # the actual time distance transparent to review.
    age_hours = (event_at - entry_at).total_seconds() / 3600
    score -= min(age_hours / 240, 5.0)
    return score, reasons


def choose_anchors(trades: list[dict], events: list[dict], author: str = "ACE") -> tuple[dict[str, dict], list[dict]]:
    """Assign each reported management message to at most one entry proxy.

    A fully explicit canonical link wins. Otherwise a literal ticker or full
    contract match is needed. This gets broad coverage without pretending an
    unlabelled "all out" post belongs to every open SPY position.
    """

    eligible = [t for t in trades if t.get("author_name") == author and t.get("entry_event_id") and t.get("entry_available_at_utc")]
    candidates: list[dict] = []
    for event in events:
        if not usable_management(event, author):
            continue
        for trade in eligible:
            scored = match_score(trade, event)
            if scored is None:
                continue
            score, reasons = scored
            candidates.append({"event_id": event["event_id"], "trade_id": trade["trade_id"], "score": round(score, 4), "reasons": reasons})
    by_event: dict[str, list[dict]] = defaultdict(list)
    for row in candidates:
        by_event[row["event_id"]].append(row)
    event_map = {event["event_id"]: event for event in events}
    assigned: dict[str, list[dict]] = defaultdict(list)
    review: list[dict] = []
    # Resolve each management message once. This deliberately leaves some
    # competing positions open rather than emitting duplicate outcomes.
    for event_id, rows in by_event.items():
        rows.sort(key=lambda row: row["score"], reverse=True)
        winner = rows[0]
        runner_up = rows[1] if len(rows) > 1 else None
        margin = winner["score"] - runner_up["score"] if runner_up else None
        explicit = "canonical_lifecycle_link" in winner["reasons"] or "matching_visible_option_contract" in winner["reasons"]
        temporal_only = winner["reasons"] == ["same_channel", "unlabelled_full_close_same_channel_within_36h"]
        # A fallback needs one uniquely closer open candidate. Any close tie
        # remains audit-visible but unassigned.
        required_margin = 1 if temporal_only else 5
        if not explicit and runner_up and margin is not None and margin < required_margin:
            review.append({"event_id": event_id, "candidate_trade_ids": [row["trade_id"] for row in rows[:5]],
                           "scores": [row["score"] for row in rows[:5]], "status": "ambiguous_proxy_link_not_assigned",
                           "source_message_ids": event_map[event_id]["source_message_ids"]})
            continue
        assigned[winner["trade_id"]].append({**winner, "margin_to_next_candidate": margin})
    anchors: dict[str, dict] = {}
    for trade in eligible:
        rows = assigned.get(trade["trade_id"], [])
        if not rows:
            continue
        # A full-close message is the preferred outcome anchor. If none is
        # present, a trim is still a useful partial-profit/loss proxy, labelled
        # accordingly rather than treated as the complete trade result.
        rows.sort(key=lambda row: stamp(event_map[row["event_id"]].get("available_at_utc") or event_map[row["event_id"]]["timestamp_utc"]))
        full = [row for row in rows if is_full_close(event_map[row["event_id"]])]
        chosen = full[-1] if full else rows[-1]
        event = event_map[chosen["event_id"]]
        confidence = "medium" if any(reason in chosen["reasons"] for reason in ("canonical_lifecycle_link", "matching_visible_option_contract")) else "very_low" if "unlabelled_full_close_same_channel_within_36h" in chosen["reasons"] else "low"
        anchors[trade["trade_id"]] = {
            "management_event_id": event["event_id"], "management_message_id": event["message_id"],
            "management_available_at_utc": event.get("available_at_utc") or event["timestamp_utc"],
            "management_event_type": event["event_type"], "management_text": event.get("source_quote"),
            "anchor_scope": "reported_full_close" if is_full_close(event) else "reported_partial_management",
            "link_score": chosen["score"], "link_margin_to_next_candidate": chosen["margin_to_next_candidate"],
            "link_reasons": chosen["reasons"], "link_confidence": confidence,
            "source_message_ids": sorted(set(trade["source_message_ids"] + event["source_message_ids"])),
            "reported_loss_wording": bool(LOSS.search(event.get("source_quote") or "")),
        }
    return anchors, review


def build_rows(trades: list[dict], events: list[dict], prices: dict[str, dict], author: str = "ACE") -> tuple[list[dict], list[dict], dict]:
    anchors, review = choose_anchors(trades, events, author)
    rows, targets = [], []
    for trade in trades:
        if trade.get("author_name") != author or not trade.get("entry_event_id") or not trade.get("entry_available_at_utc"):
            continue
        sign = option_sign(trade)
        entry_id = f"{trade['trade_id']}:entry"
        management_anchor = anchors.get(trade["trade_id"])
        horizon = None if management_anchor else expiry_horizon(trade)
        outcome_anchor = management_anchor or horizon
        targets.append({"target_id": entry_id, "trade_id": trade["trade_id"], "role": "entry",
                        "symbol": trade.get("symbol"), "timestamp_utc": trade["entry_available_at_utc"],
                        "source_message_ids": trade["source_message_ids"]})
        entry_price = prices.get(entry_id)
        exit_price = None
        if outcome_anchor:
            if management_anchor:
                anchor_id = f"{trade['trade_id']}:anchor:{management_anchor['management_event_id']}"
                role, anchor_time = "management_anchor", management_anchor["management_available_at_utc"]
            else:
                anchor_id = f"{trade['trade_id']}:expiry_horizon"
                role, anchor_time = "expiry_horizon", horizon["timestamp_utc"]
            targets.append({"target_id": anchor_id, "trade_id": trade["trade_id"], "role": role,
                            "symbol": trade.get("symbol"), "timestamp_utc": anchor_time,
                            "source_message_ids": outcome_anchor["source_message_ids"]})
            exit_price = prices.get(anchor_id)
        outcome, move, reason = "unknown_no_proxy_price", None, "underlying snapshots have not both been fetched"
        if sign is None:
            outcome, reason = "unknown_direction", "option right/direction is not sufficiently identified"
        elif not outcome_anchor:
            outcome, reason = "unknown_no_management_anchor", "no uniquely assignable literal-symbol, full-contract, or canonical management link"
        elif not entry_price or entry_price.get("price") is None or not exit_price or exit_price.get("price") is None:
            reason = "missing_underlying_snapshot_for_entry_or_management_anchor"
        else:
            raw_return = exit_price["price"] / entry_price["price"] - 1
            move = round(100 * sign * raw_return, 4)
            # A 5bp deadband makes the classification explicit; it does not
            # represent option costs, delta, theta, or executable P&L.
            prefix = "proxy" if management_anchor else "expiry_horizon"
            outcome = f"{prefix}_winner" if move > 0.05 else f"{prefix}_loser" if move < -0.05 else f"{prefix}_flat"
            reason = "direction_adjusted_underlying_change_at_reported_management_anchor" if management_anchor else "direction_adjusted_underlying_change_to_unverified_expiry_horizon"
        source_ids = sorted(set(trade["source_message_ids"] + (outcome_anchor["source_message_ids"] if outcome_anchor else [])))
        row = {
            "trade_id": trade["trade_id"], "author_name": author, "symbol": trade.get("symbol"),
            "option_type": trade.get("option_type"), "strike": trade.get("strike"), "expiry": trade.get("expiry"),
            "entry_message_id": trade.get("entry_message_id"), "entry_available_at_utc": trade["entry_available_at_utc"],
            "underlying_direction_sign": sign, "management_anchor": management_anchor, "outcome_anchor": outcome_anchor,
            "entry_underlying_snapshot": entry_price, "anchor_underlying_snapshot": exit_price,
            "direction_adjusted_underlying_return_pct": move, "proxy_outcome": outcome,
            "proxy_outcome_reason": reason, "estimated_option_pnl": None, "estimated_follower_pnl": None,
            "source_message_ids": source_ids,
            "limitations": (
                "Underlying direction proxy only: it is not an option fill, option return, broker P&L, or follower execution. "
                "A positive underlying move can still lose after option spread, theta, IV change, or an unobserved exit."
            ),
        }
        row["field_provenance"] = {key: {"source_message_ids": source_ids,
                                          "basis": "source_cited_management_link_plus_retroactive_underlying_price_proxy",
                                          "confidence": outcome_anchor["link_confidence"] if outcome_anchor else "low"}
                                   for key in row}
        rows.append(row)
    summary = {
        "author_name": author,
        "entry_lifecycles": len(rows),
        "proxy_outcomes": dict(Counter(row["proxy_outcome"] for row in rows)),
        "management_anchors": sum(row["management_anchor"] is not None for row in rows),
        "full_close_anchors": sum(bool(row["management_anchor"] and row["management_anchor"]["anchor_scope"] == "reported_full_close") for row in rows),
        "unverified_expiry_horizons": sum(bool(row["outcome_anchor"] and row["outcome_anchor"].get("anchor_type") == "unverified_expiry_horizon") for row in rows),
        "ambiguous_management_not_auto_assigned": len(review),
        "interpretation": f"All {author} entry rows are included. Proxy winner/loss labels use either a uniquely linked management anchor or a separately labelled unverified expiry horizon; neither is option P&L or overall trader profitability.",
    }
    return rows, targets, {"summary": summary, "ambiguous_links": review}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--prices", type=Path, help="Optional snapshot JSONL from fetch_underlying_proxy.py")
    parser.add_argument("--author", default="ACE")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    prices = {row["target_id"]: row for row in read_jsonl(args.prices)} if args.prices else {}
    rows, targets, detail = build_rows(read_jsonl(args.run / "managed_trades.jsonl"), read_jsonl(args.run / "events.jsonl"), prices, args.author)
    args.out.mkdir(parents=True, exist_ok=False)
    write_jsonl(args.out / "underlying_proxy_ledger.jsonl", rows)
    write_jsonl(args.out / "underlying_proxy_targets.jsonl", targets)
    write_jsonl(args.out / "ambiguous_management_links.jsonl", detail["ambiguous_links"])
    (args.out / "summary.json").write_text(json.dumps(detail["summary"], indent=2) + "\n", encoding="utf-8")
    print(json.dumps(detail["summary"], indent=2))


if __name__ == "__main__":
    main()
