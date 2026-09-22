#!/usr/bin/env python3
"""Extract a conservative, source-cited event ledger from the 1k challenge export.

This is deliberately an *annotation* pass, not a trade matcher.  It preserves
the author inherited by DiscordHTMLExporter message groups, makes no claim that
an alert was broker-filled, and keeps incomplete contracts incomplete.  The
reconciler owns cross-message lifecycle matching.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from bs4 import BeautifulSoup


ACE = "1079391263083733072"
FT = "720901855995101225"
CHANNEL_ID = "1466557447094403335"
DEFAULT_HTML = Path(
    "VaultOfAceDiscordLogs/THE VAULT OF ACE - FREE TRADING SECTION - "
    "1k-challenge [1466557447094403335].html"
)
DEFAULT_OUTPUT = Path("research/discord_ledger_2026-09-21/annotations/challenge.jsonl")

# This finite list is intentionally not a ticker normalizer.  It is only used
# to recognize explicit symbols mentioned in this particular export; unknown
# and malformed symbols remain null rather than being silently 'corrected'.
SYMBOLS = {
    "AA", "AAPL", "BE", "BTC", "CVNA", "CVX", "DRAM", "FCX", "GLD", "HNRG",
    "IGV", "IWM", "LLY", "MU", "NDXP", "NFLX", "ORCL", "PLG", "PLTR", "QQQ",
    "RIO", "SLV", "SOFI", "SPX", "SPXW", "SPY", "UAL", "USAR", "VIX", "WMT",
    "XLE", "XLF", "XLP",
}

EVENT_TYPES = {
    "entry", "add", "trim", "exit", "stop_adjustment", "invalidation",
    "order_pending", "order_cancel", "watchlist", "position_update",
    "performance_claim", "correction", "unknown_management",
}

# Only explicit typo/correction links are supplied here.  These are reviewable
# annotations, not a general proximity-based lifecycle matcher.
RELATED: dict[str, list[str]] = {
    "1466814781758640355": ["1466812799190700043"],
    "1468653988202287188": ["1468653878101807218"],
    "1468654487509008417": ["1468653878101807218", "1468653988202287188"],
    "1468654636310204500": ["1468654324707102852"],
    "1478800464396157039": ["1478800317121560768"],
}


def clean_text(node: Any | None) -> str:
    if node is None:
        return ""
    # Preserve message wording, while normalising rendering-only indentation.
    raw = node.get_text("", strip=False).replace("\u200b", "")
    return "\n".join(part.strip() for part in raw.splitlines() if part.strip())


def attachments(container: Any) -> list[dict[str, str | None]]:
    found: list[dict[str, str | None]] = []
    for image in container.select(".chatlog__attachment img, .chatlog__forwarded-attachment"):
        found.append({"url": image.get("src"), "title": image.get("title"), "alt": image.get("alt")})
    return found


def html_messages(path: Path) -> list[dict[str, Any]]:
    soup = BeautifulSoup(path.read_text(encoding="utf-8"), "html.parser")
    rows: list[dict[str, Any]] = []
    for group in soup.select(".chatlog__message-group"):
        # DiscordHTMLExporter omits author/header in consecutive messages.  The
        # group header is therefore authoritative for every child container.
        author = group.select_one(".chatlog__author")
        author_id = author.get("data-user-id") if author else None
        author_name = clean_text(author)
        for container in group.select(".chatlog__message-container"):
            timestamp = container.select_one(".chatlog__timestamp, .chatlog__short-timestamp")
            content = container.select_one(".chatlog__content")
            forwarded = container.select_one(".chatlog__forwarded-content")
            rows.append(
                {
                    "message_id": container.get("data-message-id"),
                    "author_id": author_id,
                    "author_name": author_name,
                    "channel_id": CHANNEL_ID,
                    "timestamp_raw": timestamp.get("title") if timestamp else None,
                    "timezone": "UTC-5",
                    "text": clean_text(content),
                    "forwarded_text": clean_text(forwarded),
                    "attachments": attachments(container),
                    "reply_to_message_id": None,
                }
            )
    return rows


def normalized_messages(path: Path) -> list[dict[str, Any]]:
    """Read the normalizer's JSONL when available, tolerating field aliases."""
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        src = json.loads(line)
        rows.append(
            {
                "message_id": str(src.get("message_id") or src.get("id")),
                "author_id": str(src.get("author_id") or src.get("author", {}).get("id") or ""),
                "author_name": src.get("author_name") or src.get("author", {}).get("name") or "",
                "channel_id": str(src.get("channel_id") or CHANNEL_ID),
                "timestamp_raw": src.get("timestamp_raw") or src.get("timestamp") or src.get("created_at"),
                "timestamp_utc": src.get("timestamp_utc"),
                "timezone": src.get("timezone") or "UTC-5",
                "text": src.get("text") or src.get("content") or "",
                "forwarded_text": src.get("forwarded_text") or "",
                "attachments": src.get("attachments") or [],
                "reply_to_message_id": src.get("reply_to_message_id"),
            }
        )
    return rows


def explicit_symbol(text: str) -> tuple[str | None, str | None]:
    for token in re.findall(r"(?<![A-Za-z])([A-Za-z]{2,5})(?![A-Za-z])", text):
        if token.upper() in SYMBOLS:
            return token.upper(), token
    return None, None


def option_fields(text: str) -> dict[str, Any]:
    symbol, symbol_raw = explicit_symbol(text)
    lower = text.lower()
    option_type = None
    if re.search(r"\b(?:call|calls|c)\b", lower):
        option_type = "call"
    elif re.search(r"\b(?:put|puts|p)\b", lower):
        option_type = "put"
    instrument = "option" if option_type or "0dte" in lower or re.search(r"\bcon(?:tract)?s?\b", lower) else "unknown"
    strike = None
    if symbol:
        strike_match = re.search(
            rf"\b{re.escape(symbol_raw)}\s+(\d{{1,5}}(?:\.\d+)?)\s*(?:call|calls|c\b|put|puts|p\b)",
            text,
            re.I,
        )
        if strike_match:
            strike = float(strike_match.group(1))
    # A corrected message such as '78 call sorry' has no symbol; retain the
    # strike as a correction fact but do not borrow a future/current symbol.
    if strike is None:
        lone_strike = re.search(r"\b(\d{1,5}(?:\.\d+)?)\s*(?:call|calls|c\b|put|puts|p\b)", text, re.I)
        if lone_strike:
            strike = float(lone_strike.group(1))
    expiry_match = re.search(r"\b(0\s*DTE|\d{1,2}/\d{1,2})\b", text, re.I)
    expiry_raw = expiry_match.group(1).replace(" ", "") if expiry_match else None
    qty_match = re.search(r"\b(?:BTO|STC|BTC|sold|sell(?:ing)?|closed|closing|add(?:ed|ing)?)\s+(\d+(?:\.\d+)?)\s*(?:con(?:tract)?s?)?\b", text, re.I)
    if not qty_match:
        qty_match = re.search(r"\b(\d+(?:\.\d+)?)\s*(?:con(?:tract)?s?)\b", text, re.I)
    quantity = float(qty_match.group(1)) if qty_match else None
    quantity_text = qty_match.group(0) if qty_match else None
    # Price is only extracted when source wording labels it or follows an
    # option contract.  It remains a claimed alert/fill, never a verified fill.
    price_match = re.search(r"\b(?:at|cost|fill(?:ed)?\s+at)\s*\$?(\d+(?:\.\d+)?)\b", text, re.I)
    if not price_match and option_type:
        price_match = re.search(r"(?:0\s*DTE|\d{1,2}/\d{1,2}|(?:call|put|c|p))\s+\$?(\.\d+|\d+\.\d+)\b", text, re.I)
    price = float(price_match.group(1)) if price_match else None
    price_text = price_match.group(0) if price_match else None
    stop_match = re.search(r"\b(?:SL|stop(?:\s+loss)?)(?:\s+(?:raised|moved|set|to|at))?\s*\$?(\d+(?:\.\d+)?)\b", text, re.I)
    stop_price = float(stop_match.group(1)) if stop_match else None
    expiry = None
    if expiry_raw and expiry_raw.lower() == "0dte":
        # 0DTE itself states the expiration is the trading date.  The local
        # date is parsed from the exporter timestamp (whose footer says UTC-5)
        # rather than guessed from the currently running machine's timezone.
        try:
            expiry = datetime.strptime(text_timestamp(text), "%Y-%m-%d").date().isoformat()
        except (TypeError, ValueError):
            expiry = None
    return {
        "symbol": symbol,
        "symbol_raw": symbol_raw,
        "direction": "long" if instrument == "option" else None,
        "instrument_type": instrument,
        "option_type": option_type,
        "strike": strike,
        "expiry_raw": expiry_raw,
        "expiry": expiry,  # M/D remains raw; only explicit 0DTE is dated.
        "quantity": quantity,
        "quantity_text": quantity_text,
        "price": price,
        "price_text": price_text,
        "stop_price": stop_price,
        "stop_text": stop_match.group(0) if stop_match else None,
    }


def text_timestamp(text: str) -> str | None:
    """Compatibility shim; populated by event_row before option parsing."""
    return _CURRENT_DATE.get("date")


_CURRENT_DATE: dict[str, str | None] = {"date": None}


def classify(text: str, has_attachment: bool) -> list[tuple[str, str]]:
    """Return reviewed category rules in precedence order; can emit multiples."""
    lower = text.lower()
    events: list[tuple[str, str]] = []
    if not text and has_attachment:
        return [("unknown_management", "unclear")]
    # These two forms were manually reviewed in the source as contract
    # corrections.  Generic apologies and 'updated position' captions are not
    # corrections and must not become lifecycle facts.
    if re.search(r"\b(?:wrong strike|typo|correction|corrected)\b", lower) or (
        "sorry" in lower and re.search(r"\b(?:strike|call|put)\b", lower)
    ):
        events.append(("correction", "retrospective"))
    if re.search(r"\b(limit order|waiting for (?:the )?fill|i(?:'m| am) not in\b|unfilled)\b", lower):
        events.append(("order_pending", "intention"))
    if re.search(r"\b(cancel(?:led)?|pull(?:ing)? (?:the )?order)\b", lower):
        events.append(("order_cancel", "claimed_execution"))
    if re.search(r"\b(watchlist|top plays|will be watching)\b", lower):
        events.append(("watchlist", "observation"))
    if re.search(r"\b(position(?:s)? update|end of day positions|my positions|cash with a hedge|sitting in cash)\b", lower):
        events.append(("position_update", "claimed_execution"))
    explicit_option_take = bool(re.search(r"\btaking\b", lower) and re.search(r"\b(?:call|put|0\s*dte|\d{1,2}/\d{1,2})\b", lower))
    if (
        re.search(r"\b(?:BTO|BTC|initiated|opened)\b", text, re.I)
        or explicit_option_take
    ) and not re.search(r"i(?:'m| am) not in\b|buying the dip", lower):
        events.append(("entry", "claimed_execution"))
    elif re.search(r"(?:^|\n)\s*(?:adding|added|re-add)\b|\bi(?:'ve| have) added\b|\badded one more\b|\badding one more\b", lower):
        events.append(("add", "claimed_execution"))
    partial = re.search(r"\b(?:half|most|\d+\s+(?:more|cons?|contracts?)|sell(?:ing)?\s+1|STC\s+1|taking\s+(?:one|\d+)|take\s+(?:one|\d+)|\d+%\s+out)\b", lower)
    exit_words = re.search(r"\b(?:STC|sold|closed|closing|all out|expire worthless)\b", lower)
    personal_sell = re.search(r"\b(?:i(?:'m| am|\s+will)?\s+(?:going to )?sell|sell (?:here|at|my)|will sell most)\b", lower)
    out_words = re.search(r"\b(?:i(?:'m| am)?\s+out|out at|out of (?:all )?(?:puts|calls|positions?))\b", lower)
    if exit_words or personal_sell or out_words or partial:
        event_type = "trim" if partial else "exit"
        events.append((event_type, "conditional" if re.search(r"\b(would|may|going to|if )\b", lower) else "claimed_execution"))
    if re.search(r"\bSL\b|break[ -]?even stop|(?:set(?:ting)?|raise(?:d|ing)?|move(?:d|ing)?|tight|hit)\s+(?:a )?stop(?:\s+loss)?|my stop\b|stop loss hit", text, re.I):
        events.append(("stop_adjustment", "claimed_execution"))
    if re.search(r"\b(?:invalid|hands off|no more|stop hit|stop loss hit)\b", lower):
        events.append(("invalidation", "claimed_execution"))
    if re.search(r"(?:\b\d+(?:\.\d+)?%\s*(?:banger|out|gains?|profit|up)?|\b\$?\d+(?:\.\d+)?\s*(?:banger|gains?|profit)\b|account\s+(?:up|is)|\b\d+[xX]\b|challenge (?:should|will|is))", text, re.I) and not re.search(r"\b(?:spent|own|giving|lines?|extension|target)\b", lower):
        events.append(("performance_claim", "retrospective"))
    # An attached meme/chart beside a normal chat message is not trade
    # evidence.  We retain only image-only messages and image captions that
    # actually describe a position/management state as unresolved evidence.
    if not events and has_attachment and re.search(
        r"\b(?:position|account|trade|calls?|puts?|hedge|cash|ITM|contract|con\b|"
        r"target|stop|challenge|banger|profit|gains?)\b",
        lower,
    ):
        events.append(("unknown_management", "unclear"))
    return events


def is_substantive(value: Any) -> bool:
    return value not in (None, "", [], {}, "unknown")


def event_row(message: dict[str, Any], event_type: str, speech_act: str, index: int) -> dict[str, Any]:
    text = message["text"]
    _CURRENT_DATE["date"] = None
    if message.get("timestamp_raw"):
        try:
            _CURRENT_DATE["date"] = datetime.strptime(
                message["timestamp_raw"], "%A, %B %d, %Y %I:%M %p"
            ).date().isoformat()
        except ValueError:
            pass
    fields = option_fields(text)
    # Text-only, unlabelled spot watchlist references aren't contracts.
    if event_type == "watchlist" and fields["instrument_type"] == "unknown":
        fields["direction"] = None
    rationale = None
    rationale_match = re.search(r"(?:because|looking for|target(?: is)?|reason(?: is)?)[^.\n]*", text, re.I)
    if rationale_match:
        rationale = rationale_match.group(0)
    row: dict[str, Any] = {
        "event_id": f"{message['message_id']}:{index}",
        "message_id": message["message_id"],
        "author_id": message["author_id"],
        "channel_id": message["channel_id"],
        "event_time_raw": message["timestamp_raw"],
        "event_time_utc": message.get("timestamp_utc"),
        "timezone": message["timezone"],
        "event_type": event_type,
        "speech_act": speech_act,
        **fields,
        "rationale": rationale,
        "source_quote": text or None,
        "field_sources": {},
        "confidence": "high" if event_type in {"entry", "exit", "add", "trim", "order_pending"} and text else "medium",
        "uncertainties": [],
        "reply_to_message_id": message.get("reply_to_message_id"),
        "related_message_ids": RELATED.get(message["message_id"], []),
        "link_notes": None,
        "attachment_refs": message["attachments"] or [],
        "forwarded_text": message["forwarded_text"] or None,
    }
    if message["message_id"] in RELATED:
        row["link_notes"] = "Explicit correction context only; lifecycle matching is deferred."
    if fields["instrument_type"] == "option" and fields["expiry_raw"] and fields["expiry"] is None:
        row["uncertainties"].append("Expiry year is not stated; expiry is retained only as source text.")
    if fields["instrument_type"] == "option" and fields["strike"] is None:
        row["uncertainties"].append("Option strike is not explicit in this message.")
    if event_type in {"entry", "add", "order_pending"}:
        row["uncertainties"].append("Discord alert/claim is not broker-fill verification.")
    if event_type in {"exit", "trim"} and fields["symbol"] is None:
        row["uncertainties"].append("Exit/trim contract is not explicit in this message.")
    if not text and message["attachments"]:
        row["confidence"] = "low"
        row["uncertainties"].append("Image-only evidence has not been inspected; no trade facts were transcribed.")
    if message["forwarded_text"]:
        row["uncertainties"].append("Forwarded content is retained separately and is not used as this message's source quote.")
    source_fields = ("event_type", "speech_act", "symbol", "symbol_raw", "direction", "instrument_type", "option_type", "strike", "expiry_raw", "quantity", "quantity_text", "price", "price_text", "stop_price", "stop_text", "rationale")
    row["field_sources"] = {key: [message["message_id"]] for key in source_fields if is_substantive(row.get(key))}
    return row


def extract(messages: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for message in messages:
        if message["author_id"] not in {ACE, FT}:
            continue
        if message["channel_id"] != CHANNEL_ID:
            continue
        categories = classify(message["text"], bool(message["attachments"]))
        for index, (event_type, speech_act) in enumerate(categories):
            if event_type not in EVENT_TYPES:
                raise ValueError(f"unsupported event type {event_type}")
            events.append(event_row(message, event_type, speech_act, index))
    return events


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--html", type=Path, default=DEFAULT_HTML)
    parser.add_argument("--messages", type=Path, help="Optional normalized messages.jsonl")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    messages = normalized_messages(args.messages) if args.messages else html_messages(args.html)
    events = extract(messages)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(event, sort_keys=True) + "\n" for event in events), encoding="utf-8")
    counts = Counter((row["author_id"], row["event_type"]) for row in events)
    print(json.dumps({"messages": len(messages), "events": len(events), "counts": {f"{a}:{t}": n for (a, t), n in sorted(counts.items())}}, indent=2))


if __name__ == "__main__":
    main()
