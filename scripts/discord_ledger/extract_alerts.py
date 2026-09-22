#!/usr/bin/env python3
"""Conservatively annotate ACE / FT alerts from the supplied Discord HTML export.

This is deliberately an extraction pass, not a reconciliation or P&L engine.  It
does not turn an alert into a broker fill and it leaves every unsupported field
null.  The raw HTML is read only; annotations can be regenerated deterministically.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

from bs4 import BeautifulSoup


CHANNEL_ID = "1416372120388243516"
ACE = "1079391263083733072"
FT = "720901855995101225"
TARGETS = {ACE: "ACE", FT: "FT"}
DISCORD_EPOCH_MS = 1420070400000
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HTML = ROOT / "VaultOfAceDiscordLogs" / (
    "THE VAULT OF ACE - FREE TRADING SECTION - "
    "🔥-ace-free-alerts [1416372120388243516].html"
)
DEFAULT_OUTPUT = ROOT / "research/discord_ledger_2026-09-21/annotations/alerts.jsonl"
DEFAULT_REVIEW = ROOT / "research/discord_ledger_2026-09-21/annotations/alerts_review.md"
DEFAULT_MESSAGES = ROOT / "research/discord_ledger_2026-09-21/source/messages.jsonl"

# These are recognized only in an explicit trade/market context.  This avoids
# interpreting arbitrary all-caps prose, percentages, or Discord mentions as a
# security.  Unknown literal tickers are retained when a contract pattern proves
# that they are the underlying.
KNOWN_MARKET_SYMBOLS = {
    "AAPL", "AMD", "AMZN", "CRWV", "CIFR", "DKNG", "GLD", "GOOG", "GOOGL",
    "HOOD", "HUT", "INTC", "IWM", "META", "MSFT", "NBIS", "NDX", "NDXP",
    "NFLX", "NVDA", "ORCL", "PANW", "QQQ", "RBRK", "RIVN", "SMCI", "SPX",
    "SPY", "TSLA", "TSLL", "UBER", "VIX", "XAU", "XLF", "XLE", "XLP",
}
NON_TICKER_CONTRACT_WORDS = {
    "ALL", "AND", "ARE", "ATM", "CALL", "CALLS", "CON", "DTE", "EXP", "FOR",
    "HERE", "IF", "IN", "IS", "LOTTO", "NOW", "OF", "ON", "OR", "PUT", "PUTS",
    "SL", "THAT", "THE", "THIS", "TO", "TODAY", "WATCH", "WITH", "YOU",
    # This source writes the company name rather than its ticker; do not silently
    # turn it into AAPL.
    "APPLE",
}
OPTION_RE = re.compile(
    r"(?<![A-Z0-9$])(?P<symbol>\$?[A-Z]{1,5})\s+"
    r"(?P<strike>\d{1,5}(?:\.\d+)?)\s*"
    r"(?P<option_type>[cCpP])(?:all|alls|ut|uts)?\b"
    r"(?:\s*(?:exp(?:\s*(?:today|tomorrow|tmw))?|exp)?)?"
    r"(?:\s*(?P<expiry>(?:\d{1,2}/\d{1,2}(?:/\d{2,4})?)|(?:0|1|2)\s*DTE|today|tomorrow|tmw))?"
    r"(?:\s*(?:@|entry\s*:?|at)?\s*(?P<price>\.\d+|\d+\.\d+))?",
    re.I,
)
OPTION_RANGE_RE = re.compile(
    r"(?<![A-Z0-9$])(?P<symbol>\$?[A-Z]{1,5})\s+"
    r"(?P<lo>\d{1,5}(?:\.\d+)?)-(?P<hi>\d{1,5}(?:\.\d+)?)\s*"
    r"(?P<option_type>[cCpP])\b(?:\s+EXP)?\s*"
    r"(?P<expiry>today|tomorrow|tmw|\d{1,2}/\d{1,2}(?:/\d{2,4})?)?",
    re.I,
)
ENTRY_PRICE_RE = re.compile(r"\bentry\s*:\s*(?P<price>\.?\d+(?:\.\d+)?)", re.I)
STOP_RE = re.compile(
    r"\b(?:s/?l|stop(?:s)?|stop loss)\s*(?:at|of|:)?\s*"
    r"(?P<price>\.?\d+(?:\.\d+)?)", re.I
)
TICKER_DOLLAR_RE = re.compile(r"(?<![A-Z0-9])\$(?P<symbol>[A-Z]{1,5})\b")
REPLY_RE = re.compile(r"scrollToMessage\(event,'(?P<id>\d+)'\)")


def snowflake_timestamp(message_id: str) -> str:
    """Return UTC from Discord's snowflake; independent of export's fixed UTC-5 UI."""
    millis = (int(message_id) >> 22) + DISCORD_EPOCH_MS
    return datetime.fromtimestamp(millis / 1000, tz=timezone.utc).isoformat()


def _text(node: Any) -> str:
    if node is None:
        return ""
    # Separator preserves authored line boundaries while omitting reply previews.
    cloned = copy.copy(node)
    for metadata in cloned.select(".chatlog__edited-timestamp"):
        metadata.decompose()
    return cloned.get_text("\n", strip=True).replace("\u200b", "")


def read_html(path: Path) -> list[dict[str, Any]]:
    """Read containers with author inheritance restricted to each message group."""
    soup = BeautifulSoup(path.read_text(encoding="utf-8"), "html.parser")
    messages: list[dict[str, Any]] = []
    for group in soup.select(".chatlog__message-group"):
        current_author_id: str | None = None
        current_author_name: str | None = None
        for container in group.select(":scope > [data-message-id]"):
            header = container.select_one(".chatlog__header")
            if header:
                author = header.select_one(".chatlog__author")
                if author:
                    current_author_id = author.get("data-user-id")
                    current_author_name = _text(author)
            if not current_author_id:
                continue
            content = container.select_one(".chatlog__content")
            reply = container.select_one(".chatlog__reply-link")
            reply_match = REPLY_RE.search(reply.get("onclick", "")) if reply else None
            timestamp = header.select_one(".chatlog__timestamp") if header else None
            messages.append(
                {
                    "message_id": container["data-message-id"],
                    "author_id": current_author_id,
                    "author_name": current_author_name,
                    # HTML titles use the archive's fixed UTC-5 display timezone.
                    "export_timestamp_fixed_utc_minus_5": timestamp.get("title") if timestamp else None,
                    "timestamp_utc": snowflake_timestamp(container["data-message-id"]),
                    "channel_id": CHANNEL_ID,
                    "text": _text(content),
                    "attachment_count": len(container.select(".chatlog__attachment, .chatlog__forwarded-attachment")),
                    "reply_to_message_id": reply_match.group("id") if reply_match else None,
                }
            )
    return messages


def read_normalized(path: Path) -> list[dict[str, Any]]:
    """Use the source normalizer when available, preserving its exact message text."""
    messages: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("channel_id") != CHANNEL_ID:
                continue
            messages.append({
                "message_id": row["message_id"], "author_id": row["author_id"],
                "author_name": row.get("author_name"), "channel_id": row["channel_id"],
                "text": row.get("text") or "", "timestamp_utc": row["timestamp_utc"],
                "attachment_count": len(row.get("attachments") or []),
                "reply_to_message_id": row.get("reply_to_message_id"),
                "export_timestamp_fixed_utc_minus_5": row.get("timestamp_raw"),
            })
    return messages


def clean_symbol(value: str | None) -> str | None:
    return value.lstrip("$").upper() if value else None


def expiry_iso(raw: str | None, utc_time: str) -> str | None:
    """Resolve a numeric month/day relative to its alert date; DTE words stay null."""
    if not raw or not re.fullmatch(r"\d{1,2}/\d{1,2}(?:/\d{2,4})?", raw):
        return None
    parts = [int(piece) for piece in raw.split("/")]
    message_date = datetime.fromisoformat(utc_time).date()
    month, day = parts[0], parts[1]
    if len(parts) == 3:
        year = parts[2] + 2000 if parts[2] < 100 else parts[2]
    else:
        # Contract dates normally lie after the alert.  The date is only inferred
        # when that convention yields a date within the next 370 days.
        year = message_date.year
        candidate = datetime(year, month, day).date()
        if candidate < message_date - timedelta(days=7):
            year += 1
    try:
        candidate = datetime(year, month, day).date()
    except ValueError:
        return None
    return candidate.isoformat() if 0 <= (candidate - message_date).days <= 370 else None


def contract_fields(message: dict[str, Any]) -> list[dict[str, Any]]:
    """Return concrete option mentions, including separate rows for strike ranges."""
    text = message["text"]
    found: list[dict[str, Any]] = []
    spans: list[tuple[int, int]] = []
    for match in OPTION_RANGE_RE.finditer(text):
        spans.append(match.span())
        for strike in (match.group("lo"), match.group("hi")):
            found.append({
                "symbol": clean_symbol(match.group("symbol")), "symbol_raw": match.group("symbol"),
                "instrument_type": "option", "option_type": "call" if match.group("option_type").lower() == "c" else "put",
                "strike": float(strike), "expiry_raw": match.group("expiry"),
                "expiry": expiry_iso(match.group("expiry"), message["timestamp_utc"]),
                "price": None, "price_text": None,
            })
    for match in OPTION_RE.finditer(text):
        if any(match.start() >= start and match.end() <= end for start, end in spans):
            continue
        raw_symbol = match.group("symbol")
        symbol = clean_symbol(raw_symbol)
        # A one-letter all-caps word in prose is never a demonstrated ticker.
        if len(symbol or "") < 2 or symbol in NON_TICKER_CONTRACT_WORDS:
            continue
        price = match.group("price")
        found.append({
            "symbol": symbol, "symbol_raw": raw_symbol, "instrument_type": "option",
            "option_type": "call" if match.group("option_type").lower() == "c" else "put",
            "strike": float(match.group("strike")), "expiry_raw": match.group("expiry"),
            "expiry": expiry_iso(match.group("expiry"), message["timestamp_utc"]),
            "price": float(price) if price else None, "price_text": price,
        })
    return found


def contextual_symbol(text: str) -> tuple[str | None, str | None]:
    for match in TICKER_DOLLAR_RE.finditer(text):
        candidate = clean_symbol(match.group("symbol"))
        if candidate and candidate not in {"SL", "ATM", "DTE"}:
            return candidate, match.group("symbol")
    upper = text.upper()
    for symbol in sorted(KNOWN_MARKET_SYMBOLS, key=len, reverse=True):
        if re.search(rf"(?<![A-Z]){re.escape(symbol)}(?![A-Z])", upper):
            return symbol, symbol
    return None, None


def classify(text: str, has_contract: bool) -> tuple[str | None, str]:
    lower = text.lower()
    if re.search(r"\bwrong\b|correct one|typo|mistake", lower):
        return "correction", "retrospective"
    if re.search(r"\b(all out|fully? exit|full exiting|closing all|closed all|out at|i'?m out|gone)\b", lower):
        return "exit", "claimed_execution"
    if re.search(r"\b(trim|taking half|take half|take (?:some )?profit|cash(?:ing)? out|down to runners)\b", lower):
        return "trim", "claimed_execution"
    if STOP_RE.search(text):
        return "stop_adjustment", "instruction"
    if re.search(r"\b(?:invalid|break(?:s|ing)? below|loses?\s+(?:it|this|\d)|rejection)\b", lower):
        return "invalidation", "conditional"
    if re.search(r"\bwatch(?:list|ing)?|high watch|looking for|might take|wait for|if .*\b(?:buy|break|cross)\b", lower):
        return "watchlist", "conditional"
    if re.search(r"\b(?:add(?:ing|s)?|(?:2nd|second) entry|averag(?:e|ing) down|re[- ]?entry)\b", lower):
        claimed = re.search(r"\b(?:i(?:'m| am)|we|my)\b", lower) is not None
        return "add", "claimed_execution" if claimed else "instruction"
    if ("%" in text or re.search(r"\b(?:printing|banger|paid|profit|made|win|wins)\b", lower)
            or (re.search(r"\bup\b", lower) and contextual_symbol(text)[0] is not None)):
        return "performance_claim", "retrospective"
    if has_contract:
        # Bare contract strings are alerts, not proof the caller was filled.
        return "entry", "claimed_execution" if re.search(r"\b(?:i(?:'m| am)|we)\s+(?:in|bought|grabbed)\b|\bgrabbed\b", lower) else "instruction"
    if re.search(r"\b(?:manage|runners?|holding|position|hedges?)\b", lower):
        return "unknown_management", "unclear"
    return None, "unclear"


def make_event(
    message: dict[str, Any], index: int, event_type: str, speech_act: str,
    fields: dict[str, Any], uncertainties: list[str], confidence: str,
) -> dict[str, Any]:
    entry_price = ENTRY_PRICE_RE.search(message["text"])
    stop = STOP_RE.search(message["text"])
    if fields.get("price") is None and entry_price:
        fields["price"] = float(entry_price.group("price"))
        fields["price_text"] = entry_price.group(0)
    if stop:
        fields["stop_price"] = float(stop.group("price"))
        fields["stop_text"] = stop.group(0)
    base = {
        "event_id": f"{message['message_id']}:{index}", "message_id": message["message_id"],
        "author_id": message["author_id"], "channel_id": message["channel_id"],
        "event_type": event_type, "speech_act": speech_act,
        "symbol": None, "symbol_raw": None, "direction": None, "instrument_type": "unknown",
        "option_type": None, "strike": None, "expiry_raw": None, "expiry": None,
        "quantity": None, "price": None, "stop_price": None, "quantity_text": None,
        "price_text": None, "stop_text": None, "rationale": None,
        "source_quote": message["text"], "field_sources": {}, "confidence": confidence,
        "uncertainties": uncertainties, "reply_to_message_id": message["reply_to_message_id"],
        "related_message_ids": [], "link_notes": None,
    }
    base.update(fields)
    # All non-null substantive extracted fields are grounded in this message.
    substantive = ("symbol", "symbol_raw", "direction", "instrument_type", "option_type", "strike",
                   "expiry_raw", "expiry", "quantity", "price", "stop_price", "quantity_text",
                   "price_text", "stop_text", "rationale")
    base["field_sources"] = {
        key: [message["message_id"]]
        for key in ("event_type", "speech_act", "source_quote", *substantive)
        if base.get(key) is not None
    }
    if message["reply_to_message_id"]:
        base["link_notes"] = "Reply target retained as provenance only; no fields inferred from reply preview."
    return base


def extract(messages: Iterable[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Counter]:
    events: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    coverage: Counter = Counter()
    for message in messages:
        if message["author_id"] not in TARGETS:
            continue
        coverage["target_messages"] += 1
        contracts = contract_fields(message)
        event_type, speech_act = classify(message["text"], bool(contracts))
        trade_like_text = bool(OPTION_RE.search(message["text"]) or OPTION_RANGE_RE.search(message["text"]))
        has_market_reference = bool(contracts) or trade_like_text or contextual_symbol(message["text"])[0] is not None
        no_text_beyond_mentions = not re.sub(r"@(everyone|here)|\s+", "", message["text"], flags=re.I)
        if message["attachment_count"] and (has_market_reference or event_type or no_text_beyond_mentions):
            unresolved.append({
                "message_id": message["message_id"], "author": TARGETS[message["author_id"]],
                "reason": "Attachment may contain trade details; no attachment OCR was used.",
                "text": message["text"], "attachments": message["attachment_count"],
            })
        if not event_type:
            continue
        mentions = contracts or [{}]
        # A generic price-only/management assertion can have a contextual symbol;
        # exact option metadata remains null unless written in this message.
        symbol, symbol_raw = contextual_symbol(message["text"])
        event_types = [(event_type, speech_act)]
        # A single source message may both place/alert an option and set its stop.
        # Keep both facts instead of reclassifying the entry as a stop-only event.
        if contracts and STOP_RE.search(message["text"]) and event_type not in {"entry", "add"}:
            entry_act = "claimed_execution" if re.search(r"\b(?:grabbed|bought|i(?:'m| am) in)\b", message["text"], re.I) else "instruction"
            event_types.append(("entry", entry_act))
        for contract_index, contract in enumerate(mentions):
            fields = dict(contract)
            fields.setdefault("symbol", symbol)
            fields.setdefault("symbol_raw", symbol_raw)
            if fields.get("instrument_type") is None:
                fields["instrument_type"] = "equity" if symbol else "unknown"
            if fields.get("instrument_type") == "option":
                fields["direction"] = "long"
            uncertainties: list[str] = []
            if event_type in {"entry", "add"} and speech_act != "claimed_execution":
                uncertainties.append("Alert/instruction is not a verified caller fill.")
            if fields.get("instrument_type") == "option" and not fields.get("expiry"):
                uncertainties.append("Option expiry is absent or non-calendar wording.")
            if not fields.get("symbol"):
                uncertainties.append("No explicit instrument identity in message text.")
            if message["attachment_count"]:
                uncertainties.append("Attachment was not OCR'd; image details intentionally unresolved.")
            confidence = "high" if contracts and not uncertainties else "medium" if fields.get("symbol") else "low"
            for type_index, (event_name, act) in enumerate(event_types):
                event_index = contract_index * len(event_types) + type_index
                events.append(make_event(message, event_index, event_name, act, fields, uncertainties, confidence))
                coverage[f"event_{event_name}"] += 1
    return events, unresolved, coverage


def write_review(path: Path, messages: list[dict[str, Any]], events: list[dict[str, Any]], unresolved: list[dict[str, Any]], coverage: Counter) -> None:
    by_author = Counter(message["author_id"] for message in messages if message["author_id"] in TARGETS)
    by_type = Counter(event["event_type"] for event in events)
    image_only = [u for u in unresolved if not u["text"].strip() or u["text"].strip() in {"@everyone", "@here"}]
    lines = [
        "# Alert extraction review", "",
        "Generated by `scripts/discord_ledger/extract_alerts.py` from the immutable free-alerts HTML export.",
        "The export's footer/display timestamp is fixed UTC-5. The extractor additionally derives UTC from Discord snowflakes; it never treats the displayed clock as daylight-saving ET.", "",
        "## Coverage", "",
        f"- Target-author messages read: {coverage['target_messages']} (ACE {by_author[ACE]}, FT {by_author[FT]}).",
        f"- Extracted events: {len(events)}. " + ", ".join(f"{key}={value}" for key, value in sorted(by_type.items())),
        f"- Attachment-relevant messages retained for review: {len(unresolved)}; image-only/mention-only among them: {len(image_only)}.",
        "- Other members are intentionally excluded as callers; reply previews are never used as author content or field evidence.", "",
        "## Interpretation controls", "",
        "- A contract alert is an instruction unless the caller explicitly claims execution; neither is a broker-verified fill.",
        "- `trim` is not rewritten as `exit`; explicit full/all-out language is needed for an exit event.",
        "- Put purchases are `direction=long`, `option_type=put`; no text is treated as a short option absent an explicit statement.",
        "- Contract ranges produce one event per stated strike; all unavailable quantity, fill, contract or image details remain null.",
        "- Messages that merely claim percentage gains remain performance claims unless their exact contract is in the same message.", "",
        "## Attachment unresolved evidence", "",
    ]
    for item in unresolved:
        excerpt = item["text"].replace("\n", " / ") or "[no text]"
        lines.append(f"- `{item['message_id']}` ({item['author']}; {item['attachments']} attachment(s)): {item['reason']} Excerpt: {excerpt[:260]}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--html", type=Path, default=DEFAULT_HTML)
    parser.add_argument("--messages", type=Path, default=DEFAULT_MESSAGES,
                        help="Normalized source messages; falls back to --html when absent.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--review", type=Path, default=DEFAULT_REVIEW)
    args = parser.parse_args()
    messages = read_normalized(args.messages) if args.messages.exists() else read_html(args.html)
    events, unresolved, coverage = extract(messages)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event, sort_keys=True) + "\n")
    write_review(args.review, messages, events, unresolved, coverage)
    print(json.dumps({"messages": coverage["target_messages"], "events": len(events), "by_type": dict(sorted(Counter(e["event_type"] for e in events).items())), "unresolved_attachments": len(unresolved)}, sort_keys=True))


if __name__ == "__main__":
    main()
