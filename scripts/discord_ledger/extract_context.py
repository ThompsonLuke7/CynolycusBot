#!/usr/bin/env python3
"""Extract source-cited ACE/FT context from the free Discord export.

This is deliberately an *evidence* extractor, not a lifecycle linker.  It
emits only statements made by the two named callers, preserves the literal
message quotation, and leaves missing option-contract terms null.  The ledger
reconciler decides later whether an event can be linked to a trade.

The preferred input is ``annotations/source/messages.jsonl`` from the export
normalizer.  A read-only HTML fallback exists so extraction can be audited
before normalization is available.  No remote attachments are fetched: an
attachment without text is reported as unresolved image-only evidence.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

try:
    from bs4 import BeautifulSoup
except ImportError:  # pragma: no cover - only relevant for the HTML fallback
    BeautifulSoup = None


ROOT = Path(__file__).resolve().parents[2]
RESEARCH = ROOT / "research" / "discord_ledger_2026-09-21"
DEFAULT_SOURCE = RESEARCH / "source" / "messages.jsonl"
DEFAULT_OUTPUT = RESEARCH / "annotations" / "context.jsonl"
DEFAULT_REVIEW = RESEARCH / "context_review.md"
RAW_DIR = ROOT / "VaultOfAceDiscordLogs"

ACE_ID = "1079391263083733072"
FT_ID = "720901855995101225"
CALLERS = {ACE_ID: "ACE", FT_ID: "FT"}
CHANNELS = {
    "1443663553239449770": "free-watchlists",
    "1521796664052940870": "free-chat",
    "1416374095456632912": "post-ur-profits",
}
CHANNEL_HINTS = tuple(CHANNELS.values())

EVENT_FIELDS = (
    "event_id", "message_id", "author_id", "channel_id", "timestamp_local",
    "timestamp_utc", "event_type", "speech_act", "symbol", "symbol_raw",
    "direction", "instrument_type", "option_type", "strike", "expiry_raw",
    "expiry", "quantity", "price", "stop_price", "quantity_text", "price_text",
    "stop_text", "rationale", "source_quote", "field_sources", "confidence",
    "uncertainties", "reply_to_message_id", "related_message_ids", "link_notes",
    "levels", "attachment_refs", "unresolved_evidence",
)

# Prevent ordinary Discord prose from becoming invented tickers.  This is a
# guard, not a ticker universe: unfamiliar uppercase strings are retained.
TICKER_STOPWORDS = {
    "ACE", "FT", "VIP", "ATM", "IMO", "FOMC", "CPI", "PPI", "GDP", "FED",
    "JPOW", "THE", "AND", "FOR", "WITH", "THIS", "THAT", "ALL", "OUT", "NOW",
    "TODAY", "TOMORROW", "WEEK", "WEEKLY", "MONTH", "MONTHLY", "GOOD", "GREAT",
    "CALL", "CALLS", "PUT", "PUTS", "LONG", "SHORT", "OPEN", "CLOSE", "STOP",
    "HIGH", "LOW", "LFG", "GOO", "PM", "AM", "ET", "EST", "UTC", "RTH",
    "BTO", "STC", "STO", "BTC", "USA", "US", "YOLO", "LOL", "NOT", "BUT",
    "IF", "THEN", "OR", "IS", "ARE", "BE", "IT", "IN", "ON", "AT", "TO",
}

# This reviewed recognition list is deliberately finite.  A bare all-caps word
# such as "OUT" is not evidence of a ticker merely because an exchange happens
# to list that symbol.  A `$SYMBOL` is always preserved verbatim, including an
# apparent typo (for example SPCX), while bare symbols must be in this list.
# It combines every stock/index/ETF/crypto symbol manually observed in the
# scoped caller messages; it does not validate a symbol or correct its spelling.
REVIEWED_BARE_SYMBOLS = {
    "AA", "AAPL", "ACHR", "AFRM", "AI", "AMD", "AMZN", "APA", "APLD", "APP", "ARBE",
    "ASTS", "AVGO", "BE", "BHP", "BIDU", "BMNR", "BTC", "CCL", "CIFR", "COIN", "CORZ",
    "COST", "CRML", "CRWV", "CVX", "CX", "DDOG", "DELL", "DIA", "EOSE", "ETH", "FEZ",
    "FANG", "FCX", "GLD", "GLW", "GME", "GOOG", "GOOGL", "HIMS", "HL", "HOOD", "IONQ",
    "IREN", "INTC", "IWM", "JOBY", "JPM", "LCID", "LEN", "LI", "LITE", "LLY", "LQD",
    "LUNR", "MA", "META", "MP", "MRVL", "MSFT", "MSTR", "MU", "MVST", "MVT", "NBIS",
    "NDXP", "NET", "NFLX", "NOW", "NVDA", "NVO", "OGN", "ONDS", "OPEN", "ORCL", "OXY",
    "PALL", "PLTR", "QCOM", "QQQ", "RBRK", "RDDT", "RDW", "RGTI", "RKLB", "RKT", "RZLV",
    "SLV", "SNDK", "SOFI", "SPX", "SPXW", "SPY", "TEM", "TSLA", "TWLO", "USAR", "UVXY",
    "VIX", "VST", "WBD", "WMB", "WMT", "WULF", "XLE", "XLF", "XLK", "XOM", "XYZ",
}


def clean_text(value: Any) -> str:
    """Normalise whitespace without changing wording/punctuation."""
    return re.sub(r"[ \t\xa0]+", " ", str(value or "")).strip()


def first_value(message: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = message.get(key)
        if value not in (None, ""):
            return value
    return None


def canonical_channel(message: dict[str, Any]) -> str | None:
    raw = first_value(message, "channel_id", "channelId")
    if raw is not None:
        return str(raw)
    name = clean_text(first_value(message, "channel_name", "channel", "source_channel"))
    lower = name.lower()
    for channel_id, hint in CHANNELS.items():
        if hint in lower:
            return channel_id
    return None


def normalise_message(message: dict[str, Any]) -> dict[str, Any]:
    """Adapt benign normalizer-schema variations to the extractor schema."""
    author = first_value(message, "author_id", "authorId", "user_id")
    author_obj = message.get("author") if isinstance(message.get("author"), dict) else {}
    author = author or author_obj.get("id")
    attachments = first_value(message, "attachments", "attachment_refs", "attachment_urls")
    if attachments is None:
        attachments = []
    elif not isinstance(attachments, list):
        attachments = [attachments]
    text = clean_text(first_value(message, "text", "content", "content_text", "message_text"))
    forwarded = clean_text(first_value(message, "forwarded_text", "forwarded_content"))
    # Forwarded content is visible source content, so include it once rather
    # than silently dropping a statement copied from a paid channel.
    if forwarded and forwarded not in text:
        text = clean_text(f"{text}\n{forwarded}")
    return {
        "message_id": str(first_value(message, "message_id", "id") or ""),
        "author_id": str(author or ""),
        "channel_id": canonical_channel(message),
        "channel_name": clean_text(first_value(message, "channel_name", "channel", "source_channel")),
        "timestamp_local": first_value(message, "timestamp_local", "local_timestamp", "timestamp"),
        "timestamp_utc": first_value(message, "timestamp_utc", "utc_timestamp"),
        "text": text,
        "reply_to_message_id": first_value(message, "reply_to_message_id", "reply_to", "reference_message_id"),
        "attachments": attachments,
        "raw": message,
    }


def parse_html_messages() -> list[dict[str, Any]]:
    """Read the three scoped exports, including Discord group author inheritance."""
    if BeautifulSoup is None:
        raise RuntimeError("BeautifulSoup is required when normalized JSONL is unavailable")
    selected = []
    for path in sorted(RAW_DIR.glob("*.html")):
        lower = path.name.lower()
        channel_id = next((cid for cid in CHANNELS if f"[{cid}]" in path.name), None)
        if channel_id is None:
            continue
        soup = BeautifulSoup(path.read_text(encoding="utf-8"), "html.parser")
        inherited_author_id = None
        for node in soup.select("[data-message-id]"):
            message_id = node.get("data-message-id", "")
            author = node.select_one(".chatlog__author")
            if author is not None:
                inherited_author_id = author.get("data-user-id")
            # A message in a Discord group has no header but still belongs to
            # the prior author in that group. Resetting only occurs at a later
            # explicitly headed message, matching the rendered export.
            reply = node.select_one(".chatlog__reply-link")
            reply_id = None
            if reply is not None:
                match = re.search(r"scrollToMessage\\(event,'(\\d+)'\\)", reply.get("onclick", ""))
                if match:
                    reply_id = match.group(1)
            content_nodes = node.select(".chatlog__content, .chatlog__forwarded-content")
            text = clean_text("\n".join(part.get_text(" ", strip=True) for part in content_nodes))
            attachments = [
                link.get("href") for link in node.select(".chatlog__attachment a[href], .chatlog__forwarded-attachments a[href]")
            ]
            timestamp = node.select_one(".chatlog__timestamp")
            timestamp_value = timestamp.get("title") if timestamp else None
            selected.append(normalise_message({
                "message_id": message_id,
                "author_id": inherited_author_id,
                "channel_id": channel_id,
                "channel_name": CHANNELS[channel_id],
                "timestamp_local": timestamp_value,
                "text": text,
                "reply_to_message_id": reply_id,
                "attachments": attachments,
            }))
    return selected


def load_messages(source: Path) -> tuple[list[dict[str, Any]], str]:
    if source.exists():
        rows = []
        with source.open(encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                if line.strip():
                    try:
                        rows.append(normalise_message(json.loads(line)))
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"Invalid JSON on {source}:{line_no}") from exc
        return rows, str(source.relative_to(ROOT))
    return parse_html_messages(), "HTML fallback (normalized source absent)"


def extract_symbols(text: str) -> list[tuple[str, str]]:
    """Return explicit ticker spelling; no symbol correction or entity lookup."""
    found: list[tuple[str, str]] = []
    for match in re.finditer(r"(?<![A-Za-z0-9])(\$?)([A-Za-z]{1,6})(?![A-Za-z0-9])", text):
        raw = match.group(0)
        symbol = match.group(2).upper()
        has_dollar = bool(match.group(1))
        # Case-folding lets an explicit `spy` in trader prose remain SPY while
        # protecting ordinary title-cased words.  Bare lowercase tokens are
        # accepted only for the conventional index/ETF names below.
        if not has_dollar and raw != raw.upper() and symbol not in {"SPY", "SPX", "QQQ", "IWM", "VIX"}:
            continue
        if (not has_dollar and symbol not in REVIEWED_BARE_SYMBOLS) or symbol in TICKER_STOPWORDS:
            continue
        if (symbol, raw) not in found:
            found.append((symbol, raw))
    return found


def parse_option(text: str) -> dict[str, Any]:
    lower = text.lower()
    opt_type = "call" if re.search(r"\b(?:calls?|c)\b", lower) else "put" if re.search(r"\b(?:puts?|p)\b", lower) else None
    strike_match = re.search(r"\b(\d{1,5}(?:\.\d+)?)\s*(?:c|p|calls?|puts?)\b", text, re.I)
    strike = float(strike_match.group(1)) if strike_match else None
    expiry_match = re.search(r"\b(?:\d+\s*)?DTE\b|\b(\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?)\b", text, re.I)
    expiry_raw = expiry_match.group(1) if expiry_match else None
    return {"option_type": opt_type, "strike": strike, "expiry_raw": expiry_match.group(0) if expiry_match else None, "expiry": None}


def parse_numbers(text: str) -> dict[str, Any]:
    price_match = re.search(r"(?:@|at|avg(?:erage)?(?:\s+(?:of|at))?)\s*\$?(\d+(?:\.\d+)?)", text, re.I)
    stop_match = re.search(r"\b(?:stop|risk|cut(?:ting)?(?:\s+at)?)\s*(?:is|at|@)?\s*\$?(\d+(?:\.\d+)?)", text, re.I)
    quantity_match = re.search(r"\b(\d+(?:\.\d+)?)\s*(?:cons?|contracts?|shares?)\b", text, re.I)
    return {
        "price": float(price_match.group(1)) if price_match else None,
        "price_text": price_match.group(0) if price_match else None,
        "stop_price": float(stop_match.group(1)) if stop_match else None,
        "stop_text": stop_match.group(0) if stop_match else None,
        "quantity": float(quantity_match.group(1)) if quantity_match else None,
        "quantity_text": quantity_match.group(0) if quantity_match else None,
    }


def levels_for_symbol(text: str, symbol: str) -> list[str]:
    """Literal nearby level clauses, preserving price/percentage wording."""
    levels = []
    for clause in re.split(r"(?<=[.!?;])|\n", text):
        if re.search(rf"(?<![A-Za-z0-9])\$?{re.escape(symbol)}(?![A-Za-z0-9])", clause, re.I) and re.search(
            r"\b(?:above|below|over|under|break|reclaim|hold|support|resistance|level|target|stop|at)\b|\$\d",
            clause,
            re.I,
        ):
            levels.append(clean_text(clause))
    return levels


def explicit_contract(text: str, symbol: str, instrument_type: str, option: dict[str, Any]) -> bool:
    """Require identity sufficient for lifecycle reconciliation, especially options."""
    if instrument_type == "equity":
        return bool(re.search(r"\b(?:shares?|stock)\b", text, re.I))
    if instrument_type == "option":
        return bool(symbol and option["option_type"] and option["strike"] is not None and option["expiry_raw"])
    return False


def speech_act(text: str, event_type: str) -> str:
    lower = text.lower()
    if re.search(r"\b(?:if|unless|would|could|wait(?:ing)? for|only if|on (?:a )?break)\b", lower):
        return "conditional"
    if event_type == "performance_claim" or re.search(r"\b(?:was|were|already|earlier|yesterday|premium|banger)\b", lower):
        return "retrospective"
    if re.search(r"\b(?:bought|added|sold|trimmed|closed|took|entered|filled|got in|all out)\b", lower):
        return "claimed_execution"
    if re.search(r"\b(?:buy|sell|add|trim|enter|exit)\b", lower):
        return "instruction"
    if re.search(r"\b(?:looking|watching|plan|will|gonna|want)\b", lower):
        return "intention"
    return "observation"


def classify(text: str, channel_id: str | None) -> list[str]:
    lower = text.lower()
    has_symbol = bool(extract_symbols(text))
    has_level = bool(re.search(r"\b(?:watch(?:list|ing)?|levels?|support|resistance|break(?:out)?|reclaim|expected move|above|below|target|fail)\b", lower))
    result: list[str] = []
    if has_level and has_symbol:
        result.append("watchlist")
    if re.search(r"\b(?:all out|closed|close(?:d)?|sold|sell(?:ing)?|exit(?:ed|ing)?)\b", lower):
        result.append("exit")
    if re.search(r"\b(?:trim(?:med)?|half out|half off|take(?:ing)? (?:some|profit))\b", lower):
        result.append("trim")
    if re.search(r"\b(?:add(?:ed|ing)?|average(?:d|ing)? down|scale(?:d|ing)? in)\b", lower):
        result.append("add")
    # 'entry at' is often a watchlist level, not an order.  Require an actual
    # execution verb rather than assuming a signal was filled.
    if re.search(r"\b(?:bought|just bought|opening|opened|initiated|beginning to enter|got in|filled|BTO)\b", text, re.I):
        result.append("entry")
    if re.search(r"\b(?:stop(?: loss)?|move(?:d)? (?:my )?stop|breakeven|SL)\b", text, re.I) and not re.search(r"non[- ]?stop|one stop", lower):
        result.append("stop_adjustment")
    if re.search(r"\b(?:invalid(?:ated|ation)|thesis (?:is )?dead|no longer valid|stop hit)\b", lower):
        result.append("invalidation")
    # A symbol-specific return/recap is a claim and remains one even where a
    # nearby phrase says half-out or sold.
    if has_symbol and re.search(r"(?:\b\d+(?:\.\d+)?%|\b(?:banger|premium|profit|runner|green|banked|winner|win|recap|review)\b)", lower):
        result.append("performance_claim")
    return list(dict.fromkeys(result))


def instrument_for(text: str, option: dict[str, Any]) -> str:
    if option["option_type"]:
        return "option"
    if re.search(r"\b(?:shares?|stock)\b", text, re.I):
        return "equity"
    return "unknown"


def event_from(message: dict[str, Any], event_type: str, symbol_pair: tuple[str, str] | None, index: int, *, unresolved: str | None = None) -> dict[str, Any]:
    text = message["text"]
    symbol, symbol_raw = symbol_pair if symbol_pair else (None, None)
    option = parse_option(text)
    instrument = instrument_for(text, option)
    numeric = parse_numbers(text)
    contract_ok = bool(symbol and explicit_contract(text, symbol, instrument, option))
    act = speech_act(text, event_type)
    uncertainties: list[str] = []
    link_notes = None
    if event_type in {"entry", "add", "trim", "exit"} and not contract_ok:
        # Do not imply an executable/filled trade where only a ticker or a
        # screenshot exists.  Retain the management declaration for review.
        unresolved = unresolved or "No explicit contract identity in source; not eligible for lifecycle linkage."
        event_type = "unknown_management"
        link_notes = "Recorded as management context only; no explicit equity shares or option strike+expiry."
    if act == "conditional" and event_type in {"entry", "add", "trim", "exit"}:
        event_type = "watchlist"
        link_notes = "Conditional level/plan; not a filled order."
    if instrument == "option" and not contract_ok:
        uncertainties.append("Option strike and/or expiry absent; option contract is unresolved.")
    if symbol is None and event_type not in {"watchlist", "unknown_management"}:
        uncertainties.append("No explicit ticker in source text.")
    if message["attachments"]:
        uncertainties.append("Attachment content was not fetched or OCRed.")
    if unresolved:
        uncertainties.append(unresolved)
    rationale = text if event_type == "watchlist" else None
    levels = levels_for_symbol(text, symbol) if event_type == "watchlist" and symbol else []
    if levels:
        rationale = levels[0] if len(levels) == 1 else text
    row: dict[str, Any] = {
        "event_id": f"{message['message_id']}:{index}",
        "message_id": message["message_id"], "author_id": message["author_id"],
        "channel_id": message["channel_id"], "timestamp_local": message["timestamp_local"],
        "timestamp_utc": message["timestamp_utc"], "event_type": event_type,
        "speech_act": act, "symbol": symbol, "symbol_raw": symbol_raw,
        "direction": "long" if event_type in {"entry", "add"} and contract_ok else None,
        "instrument_type": instrument, "option_type": option["option_type"],
        "strike": option["strike"], "expiry_raw": option["expiry_raw"], "expiry": option["expiry"],
        **numeric, "rationale": rationale, "source_quote": text or None,
        "field_sources": {}, "confidence": "high" if contract_ok else "medium" if symbol else "low",
        "uncertainties": uncertainties, "reply_to_message_id": message["reply_to_message_id"],
        "related_message_ids": [], "link_notes": link_notes, "levels": levels,
        "attachment_refs": message["attachments"], "unresolved_evidence": unresolved,
    }
    # Map every stated content field to its local evidence.  Mechanical IDs,
    # timestamps, and null values intentionally have no source mapping.
    substantive = {
        "event_type", "speech_act", "symbol", "symbol_raw", "direction", "instrument_type",
        "option_type", "strike", "expiry_raw", "quantity", "price", "stop_price",
        "quantity_text", "price_text", "stop_text", "rationale", "source_quote", "levels",
        "attachment_refs", "unresolved_evidence", "link_notes",
    }
    row["field_sources"] = {key: [message["message_id"]] for key in substantive if row.get(key) not in (None, [], "")}
    return {key: row.get(key) for key in EVENT_FIELDS}


def extract_events(messages: Iterable[dict[str, Any]]) -> tuple[list[dict[str, Any]], Counter[str], int]:
    events: list[dict[str, Any]] = []
    candidate_count = 0
    for message in messages:
        if message["author_id"] not in CALLERS or message["channel_id"] not in CHANNELS:
            continue
        text = message["text"]
        event_type = classify(text, message["channel_id"])
        # A textless attachment from a caller is preserved as an unresolved
        # source item, even though its trade meaning cannot be read.
        if not text and message["attachments"]:
            events.append(event_from(message, "unknown_management", None, 0, unresolved="Image-only/attachment-only message; visual content unresolved."))
            continue
        if event_type is None:
            continue
        candidate_count += 1
        symbols = extract_symbols(text)
        if event_type == "watchlist":
            # Per-symbol rows make PIT lookup exact.  Preserve a no-symbol
            # market-context item too, rather than inventing a ticker.
            targets = symbols or [None]
        elif event_type == "performance_claim":
            targets = symbols or [None]
        else:
            targets = symbols[:1] or [None]
        for index, symbol_pair in enumerate(targets):
            events.append(event_from(message, event_type, symbol_pair, index))
    return events, Counter(event["event_type"] for event in events), candidate_count


def write_review(path: Path, *, source_label: str, messages: list[dict[str, Any]], events: list[dict[str, Any]], candidate_count: int) -> None:
    scoped = [m for m in messages if m["channel_id"] in CHANNELS]
    caller_messages = [m for m in scoped if m["author_id"] in CALLERS]
    by_author = Counter(CALLERS.get(m["author_id"], m["author_id"] or "unattributed") for m in caller_messages)
    by_channel = Counter(CHANNELS.get(m["channel_id"], m["channel_id"] or "unknown") for m in scoped)
    event_counts = Counter(event["event_type"] for event in events)
    unresolved = sum(bool(event["unresolved_evidence"]) for event in events)
    lines = [
        "# Context extraction review",
        "",
        f"- Input: `{source_label}`.",
        f"- Scoped messages read: {len(scoped)} (channels: " + ", ".join(f"{name}={by_channel[name]}" for name in sorted(by_channel)) + ").",
        f"- Caller messages considered: {len(caller_messages)} (ACE={by_author['ACE']}, FT={by_author['FT']}).",
        f"- Candidate messages classified: {candidate_count}; context rows written: {len(events)}.",
        "- Event counts: " + ", ".join(f"{kind}={event_counts[kind]}" for kind in sorted(event_counts)) + ".",
        f"- Unresolved attachment/contract notes: {unresolved}.",
        "",
        "## Interpretation constraints",
        "",
        "- ACE and FT are separate callers. Other members' messages are outside this file even where they describe a trade.",
        "- A performance claim remains a claim; it is not an exit, fill, return verification, or lifecycle link.",
        "- Conditional breakout, support, stop, or target language is stored as watchlist context, never as a filled order.",
        "- Options require ticker, call/put, strike, and expiry before a claimed entry/exit can be lifecycle-eligible. Missing terms remain null and are recorded as unresolved management context.",
        "- Attachment URLs are evidence references only. No image was fetched or OCRed; image-only messages retain an explicit unresolved note.",
        "- `field_sources` in every JSONL row maps each non-null substantive field to its originating message ID. Future messages are not used to enrich entry context.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="normalized messages JSONL (HTML fallback if absent)")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--review", type=Path, default=DEFAULT_REVIEW)
    args = parser.parse_args()
    messages, source_label = load_messages(args.source)
    events, counts, candidate_count = extract_events(messages)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event, sort_keys=True, ensure_ascii=False) + "\n")
    write_review(args.review, source_label=source_label, messages=messages, events=events, candidate_count=candidate_count)
    print(json.dumps({"events": len(events), "event_types": counts, "output": str(args.output)}, default=dict))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
