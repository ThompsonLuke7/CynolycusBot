"""Summarize source-cited callout habits without interpreting returns as edge."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import json
from pathlib import Path
import re
from zoneinfo import ZoneInfo

from scripts.discord_ledger.build import read_jsonl, write_jsonl

NY = ZoneInfo("America/New_York")
SETUPS = {
    "dealer_or_expected_move": r"\b(?:dealer|DLEM|UEM|WEM|LEM|expected move|gamma)\b",
    "breakout_or_level": r"\b(?:breakout|break(?:s|ing)?\s+(?:above|below|of)|high of day|reclaim|support|resistance|above\s+\d|below\s+\d)\b",
    "earnings_or_catalyst": r"\b(?:earnings|\bER\b|catalyst|news)\b",
    "flow": r"\bflow\b",
    "hedge": r"\b(?:hedge|insurance|protection)\b",
}
RISK = {
    "lotto": r"\blotto\b",
    "explicit_high_risk": r"\bhigh risk\b",
    "size_language": r"\b(?:\d+\s*(?:cons?|contracts?|shares?)|small size|lotto size)\b",
    "stop_language": r"\b(?:stop|\bSL\b)\b",
    "target_language": r"\b(?:target|\bTP\b)\b",
    "swing_language": r"\b(?:swing|overnight|weekly expiry)\b",
}


def labels(text: str, patterns: dict[str, str]) -> list[str]:
    return [name for name, pattern in patterns.items() if re.search(pattern, text, re.I)]


def session_bin(utc: str) -> str:
    local = datetime.fromisoformat(utc.replace("Z", "+00:00")).astimezone(NY)
    minute = local.hour * 60 + local.minute
    if minute < 9 * 60 + 30 or minute >= 16 * 60:
        return "outside_regular_session"
    if minute < 10 * 60:
        return "open_0930_1000"
    if minute < 12 * 60:
        return "morning_1000_1200"
    if minute < 15 * 60 + 30:
        return "midday_1200_1530"
    return "close_1530_1600"


def build(events: list[dict], trades: list[dict], research: list[dict]) -> tuple[list[dict], dict]:
    by_trade = {t["trade_id"]: t for t in trades}
    by_research = {r["trade_id"]: r for r in research}
    features = []
    for event in events:
        if event["event_type"] != "entry":
            continue
        trade = by_trade.get(event.get("trade_id"))
        if not trade or trade["entry_event_id"] != event["event_id"]:
            continue
        research_row = by_research[trade["trade_id"]]
        prior = research_row["prior_context"]
        setup_at_alert = labels(event["source_quote"], SETUPS)
        risk_at_alert = labels(event["source_quote"], RISK)
        prior_setup = sorted({tag for row in prior for tag in labels(row["source_quote"], SETUPS)})
        source_ids = sorted(set(event["source_message_ids"] +
                                [mid for row in prior for mid in row["source_message_ids"]]))
        features.append({
            "event_id": event["event_id"], "trade_id": trade["trade_id"],
            "entry_message_id": event["message_id"],
            "author_id": event["author_id"], "author_name": event["author_name"],
            "symbol_literal": event.get("symbol"), "instrument_type": event.get("instrument_type"),
            "option_type": event.get("option_type"), "strike": event.get("strike"),
            "expiry": event.get("expiry"), "expiry_raw": event.get("expiry_raw"),
            "alert_created_at_utc": event["timestamp_utc"],
            "alert_available_at_utc": event.get("available_at_utc"),
            "session_bucket_et": session_bin(event["timestamp_utc"]),
            "speech_act": event["speech_act"], "source_claimed_price": event.get("price"),
            "source_claimed_quantity": event.get("quantity"),
            "risk_language_at_alert": risk_at_alert,
            "setup_language_at_alert": setup_at_alert,
            "prior_30d_same_caller_symbol_setup_language": prior_setup,
            "prior_30d_context_message_ids": sorted({mid for row in prior for mid in row["source_message_ids"]}),
            "market_context_status": (research_row.get("underlying_market_context") or {}).get("status"),
            "link_confidence": trade["link_confidence"],
            "source_message_ids": source_ids,
            "method": "literal_keyword_taxonomy_v1; tags identify language, not an independently validated signal",
        })
    return features, summarize(features, events, trades)


def summarize(features: list[dict], events: list[dict], trades: list[dict]) -> dict:
    result = {}
    for author in ("ACE", "FT"):
        rows = [r for r in features if r["author_name"] == author]
        ev = [e for e in events if e["author_name"] == author]
        result[author] = {
            "entry_event_count": len(rows),
            "distinct_entry_message_count": len({r["entry_message_id"] for r in rows}),
            "most_common_literal_symbols": Counter(r["symbol_literal"] for r in rows if r["symbol_literal"]).most_common(15),
            "session_buckets": dict(Counter(r["session_bucket_et"] for r in rows)),
            "option_type": dict(Counter(r["option_type"] or "unknown" for r in rows)),
            "explicit_0dte_literal": sum(bool(re.search(r"\b0\s*DTE\b", next((e["source_quote"] for e in ev if e["event_id"] == r["event_id"]), ""), re.I)) for r in rows),
            "complete_option_identity": sum(r["instrument_type"] == "option" and all(r[k] is not None for k in ("symbol_literal", "option_type", "strike", "expiry")) for r in rows),
            "risk_language_at_alert": dict(Counter(tag for r in rows for tag in r["risk_language_at_alert"])),
            "setup_language_at_alert": dict(Counter(tag for r in rows for tag in r["setup_language_at_alert"])),
            "prior_setup_language_30d": dict(Counter(tag for r in rows for tag in r["prior_30d_same_caller_symbol_setup_language"])),
            "management_event_counts": dict(Counter(e["event_type"] for e in ev if e["event_type"] in {"add", "trim", "exit", "stop_adjustment", "invalidation"})),
            "entry_lifecycles_with_at_least_one_linked_management_event": sum(bool(t["entry_event_id"] and len(t["event_ids"]) > 1) for t in trades if t["author_name"] == author),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    events = read_jsonl(args.run / "events.jsonl")
    trades = read_jsonl(args.run / "managed_trades.jsonl")
    research = read_jsonl(args.run / "trade_research.jsonl")
    features, summary = build(events, trades, research)
    args.out.mkdir(parents=True, exist_ok=False)
    write_jsonl(args.out / "signal_features.jsonl", features)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["# Source-cited callout process profile", "",
             "This profiles the language and timing of the exported free-channel callouts. It does not validate performance or show what the caller knew privately. ACE and FT are separate authors. Counts are annotated events, not all trades made by either person.", ""]
    for author, stats in summary.items():
        lines += [f"## {author}", "", f"- Entry annotations: {stats['entry_event_count']}; complete option identities: {stats['complete_option_identity']}; explicit 0DTE literals: {stats['explicit_0dte_literal']}.",
                  f"- Most frequent literal symbols: {stats['most_common_literal_symbols'][:8]}.",
                  f"- Session timing (ET): {stats['session_buckets']}.",
                  f"- Risk/tenor language: {stats['risk_language_at_alert']}.",
                  f"- Setup language in alert: {stats['setup_language_at_alert']}; in preceding same-caller/symbol context: {stats['prior_setup_language_30d']}.",
                  f"- Management annotations: {stats['management_event_counts']}; entry lifecycles with at least one conservatively linked management event: {stats['entry_lifecycles_with_at_least_one_linked_management_event']}.", ""]
    lines += ["## Reproduction hypotheses to test", "",
              "1. Separate *idea/watchlist* from actionable entry. A quoted contract and an at-price idea are not automatically a buy; preserve the later explicit alert time.",
              "2. For 0DTE/lotto callouts, test a low-premium directional-option sleeve only after reconstructing the underlying trigger and using executable quote/spread/size rules. Do not equate quoted premiums with available fills.",
              "3. For swing callouts, encode literal breakout, high-of-day, target and stop language as candidate trigger/risk features. Link each feature only to messages available before entry.",
              "4. Treat hedges, spreads and multi-leg messages as distinct strategies; they should not be mixed into a simple long-call/put result.",
              "5. Run a matched-time, matched-symbol baseline and out-of-sample period before adding any rule to paper trading. No profitability conclusion is implied by follower counts or reported wins.", "",
              "Each row in `signal_features.jsonl` lists the source message IDs and separates alert text from earlier eligible context. Keyword tags are search aids, not semantic proof; inspect the linked originals for candidate rules."]
    (args.out / "process_profile.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({author: {"entries": s["entry_event_count"], "complete_contracts": s["complete_option_identity"]} for author, s in summary.items()}))


if __name__ == "__main__":
    main()
