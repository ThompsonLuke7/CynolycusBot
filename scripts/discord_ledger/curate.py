"""Conservative, reproducible review gate for draft Discord text annotations.

Draft rule classifiers are discovery aids. This gate keeps only source-grounded
trade or plan language and records every excluded candidate with a reason.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import re
from zoneinfo import ZoneInfo

from scripts.discord_ledger.build import DEFAULT_STUDY, read_jsonl, write_jsonl

ACTUAL_ENTRY = re.compile(r"\b(?:BTO|bought|purchased|opened|filled|entered|initiated|i\s+(?:am|'m)\s+in)\b", re.I)
ALERT_ENTRY = re.compile(r"\b(?:ALERT|BUY|ENTRY|LOTTO|TRADE\s+IDEA|FREE\s+TRADE|taking\s+(?:a|the|\d+)|in\s+[A-Z$]{2,6}\s+\d)\b", re.I)
SELL = re.compile(r"\b(?:STC|sold|closed|closing|cutting|cut\s+the\s+trade|all\s+out|fully?\s+out|taking\s+full\s+profit|out\s+at)\b", re.I)
TRIM = re.compile(r"\b(?:trim(?:ming|med)?|taking\s+(?:half|some|more|\d+%|\d+\s+cons?)|sold\s+\d+\s+(?:more|cons?|contracts?)|secure\s+(?:some\s+)?profit|leave\s+(?:a\s+)?runner|half\s+out)\b", re.I)
ADD = re.compile(r"\b(?:i\s+(?:am|'m|have)\s+add(?:ing|ed)|we\s+(?:are|'re|have)\s+add(?:ing|ed)|added\s+(?:one|\d+|[A-Z]{2,6})|averag(?:e|ed|ing)\s+down|second\s+entry|2nd\s+entry)\b", re.I)
STOP = re.compile(r"(?<!non\s)\b(?:SL|stop(?:\s+loss)?|break[ -]?even\s+stop)\b", re.I)
PENDING = re.compile(r"\b(?:limit\s+order|waiting\s+for\s+(?:the\s+)?fill|not\s+in|hasn'?t\s+filled|if\s+it\s+fills)\b", re.I)
PLAN = re.compile(r"\b(?:watch(?:list|ing)?|trade\s+plan|above|below|if\s+|looking\s+for|at\s+open|tomorrow|tmw|would\s+|will\s+|could\s+|might\s+|gonna\s+|going\s+to)\b", re.I)
PERFORMANCE = re.compile(r"\b(?:profit|pnl|return|paid|made\s+\$\d+|win(?:ner)?s?)\b|(?:\bup\s+\d+(?:\.\d+)?%|\b\d+(?:\.\d+)?%\s+(?:up|gain)|\+\s*\d+(?:\.\d+)?%)", re.I)
MARKET_PLAN = re.compile(r"\b(?:support|resistance|target|break|reclaim|hold(?:s|ing)?|supply|demand|volume|VWAP|levels?|setups?|catalyst|bulls?|bears?)\b", re.I)
CONTRACT = re.compile(r"(?<![A-Za-z0-9])\$?[A-Z]{2,6}\s+\$?\d{1,5}(?:\.\d+)?\s*(?:[CP]\b|calls?\b|puts?\b)", re.I)
SOURCE_SYMBOL = re.compile(r"(?<![A-Za-z0-9])\$?([A-Z]{2,6})(?![A-Za-z0-9])")
NON_MARKET = re.compile(r"\b(?:VIP\s+spots?|subscribe|subscription|membership|dm\s+to\s+join|keno|casino|cash\s+account|i\s+cannot\s+(?:buy|sell)|market\s+closed\s+(?:monday|today)|in\s+for\s+another\s+day)\b", re.I)
INVALIDATION = re.compile(r"\b(?:invalidat(?:e|ed|ion)|breaks?\s+(?:below|under)|loses?\s+(?:the\s+)?level)\b", re.I)
RECAP = re.compile(r"\b(?:recap|review|summary|open\s+positions|yesterday'?s|last\s+week|was\s+out\s+at|bangers?\s+to\s+close)\b", re.I)
STOP_MOVED = re.compile(r"\b(?:raise|raised|raising|move|moved|set|setting|breakeven|break[ -]?even)\b.{0,30}\b(?:stop|SL)\b|\b(?:stop|SL)\b.{0,25}\b(?:move|moved|raise|raised|raising|to)\b", re.I)
SOURCE_STOPWORDS = {"ZERO", "COST", "TMW", "TOMORROW", "FIRST", "CONS", "WILL", "CAN", "ITM", "NEAR", "START", "GOING", "WENT", "ENTER", "DEC", "JAN", "FEB", "MAR", "APR", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "OF", "SL", "ALERT", "ENTRY", "LOTTO", "HIGH", "RISK", "CALL", "PUT", "BTO", "STC", "NOW", "THIS", "THAT", "WE", "YOU", "ALL", "OUT", "AND", "FOR", "THE", "WITH", "HERO"}
VALID_SYMBOLS = {p.stem.upper() for p in (Path(__file__).resolve().parents[2] / "Data/shared/bars/1d").glob("*.parquet")} | {"SPX", "SPXW", "NDX", "NDXP", "VIX", "IBIT", "USO"}


def source_contracts(text: str) -> list[tuple[re.Match[str], str, float, str]]:
    found = []
    for match in CONTRACT.finditer(text):
        name = match[0].split()[0].lstrip("$").upper()
        if name in SOURCE_STOPWORDS or not (name in VALID_SYMBOLS or match[0].lstrip().startswith("$") or match[0].split()[0].isupper()):
            continue
        terms = re.search(r"(\d+(?:\.\d+)?)\s*(calls?|puts?|[CP])\b\s*$", match[0], re.I)
        if terms:
            found.append((match, name, float(terms[1]), "put" if terms[2].lower().startswith("p") else "call"))
    return found


def source_price(text: str, contract: re.Match[str] | None) -> tuple[float | None, str | None, str | None]:
    """Extract a quoted premium, never a stop or underlying target."""
    explicit = re.search(r"\b(?:entry|filled|avg(?:erage)?\s+cost)\s*(?::|@|at)?\s*\$?(\.\d+|\d+(?:\.\d+)?)\b", text, re.I)
    if explicit:
        return float(explicit[1]), explicit[0], "explicit_entry_or_average_cost_label"
    for at in re.finditer(r"(?:@|\bat)\s*\$?(\.\d+|\d+(?:\.\d+)?)\b", text, re.I):
        prefix = text[max(0, at.start() - 36):at.start()]
        if re.search(r"\b(?:stop|SL|target|TP|below|above|close)\b[^\n]{0,25}$", prefix, re.I):
            continue
        return float(at[1]), at[0], "quoted_at_price"
    if contract:
        tail = text[contract.end():]
        trailing = re.match(r"\s*(?:(?:\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?\s*[CP]?|\d+\s*DTE|\d+\s*(?:cons?|contracts?))\s*)?(\.\d+|\d+\.\d+)\b", tail, re.I)
        if trailing:
            return float(trailing[1]), trailing[1], "premium_immediately_after_contract"
    return None, None, None


def source_stop(text: str) -> tuple[float | None, str | None]:
    match = re.search(r"\b(?:stop(?:s|\s+loss)?(?:\s+on\s+this\s+contract)?|SL)\s*(?:(?:at|to|:|=)\s*)?\$?(\.\d+|\d+(?:\.\d+)?)\b", text, re.I)
    return (float(match[1]), match[0]) if match else (None, None)


def asof_expiry(raw: str | None, timestamp_utc: str) -> tuple[str | None, str | None]:
    """Resolve only near calendar expiries; retain the literal text separately."""
    if not raw:
        return None, None
    local_date = datetime.fromisoformat(timestamp_utc.replace("Z", "+00:00")).astimezone(ZoneInfo("America/New_York")).date()
    if re.fullmatch(r"0\s*DTE|today", raw, re.I):
        return local_date.isoformat(), "same_local_calendar_day_from_0DTE_literal"
    match = re.fullmatch(r"(\d{1,2})[/-](\d{1,2})(?:[/-](\d{2,4}))?", raw)
    if not match:
        return None, None
    month, day = int(match[1]), int(match[2])
    if match[3]:
        year = int(match[3])
        if year < 100:
            year += 2000
    else:
        year = local_date.year
    try:
        candidate = local_date.replace(year=year, month=month, day=day)
        if not match[3] and candidate < local_date:
            candidate = candidate.replace(year=year + 1)
    except ValueError:
        return None, None
    if candidate < local_date or (candidate - local_date).days > (370 if match[3] else 60):
        return None, None
    return candidate.isoformat(), "explicit_year" if match[3] else "nearest_future_month_day_within_60d_inferred_year"


def decision(draft: dict, source: dict) -> tuple[dict | None, str]:
    text = source.get("text") or ""
    cleaned = re.sub(r"@(?:everyone|here)|\s+", "", text, flags=re.I)
    if not cleaned:
        return None, "image_or_mention_only; handled_by_attachment_review"
    symbol = draft.get("symbol")
    if symbol:
        raw = draft.get("symbol_raw") or symbol
        if not re.search(rf"(?<![A-Za-z0-9])\$?{re.escape(raw)}(?![A-Za-z0-9])", text):
            return None, "parsed_symbol_not_literal_in_authored_text"
        # Single ordinary words such as 'be' are not ticker evidence.
        if raw.islower() and not re.search(rf"(?<!\w)\${re.escape(raw)}\b", text, re.I) and not CONTRACT.search(text):
            return None, "lowercase_word_not_explicit_ticker"
        if symbol.upper() in SOURCE_STOPWORDS:
            return None, "ordinary_word_parsed_as_symbol"
        if draft.get("strike") is not None and draft.get("instrument_type") == "option":
            strike_text = re.escape(str(float(draft["strike"])).rstrip("0").rstrip("."))
            direct = re.search(rf"(?<!\w)\$?{re.escape(raw)}\s+\$?{strike_text}\s*(?:calls?|puts?|[CP])\b", text, re.I)
            range_match = re.search(rf"(?<!\w)\$?{re.escape(raw)}\s+(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)\s*(?:calls?|puts?|[CP])\b", text, re.I)
            in_range = bool(range_match and float(draft["strike"]) in {float(range_match[1]), float(range_match[2])})
            if not direct and not in_range:
                return None, "contract_terms_not_cooccurring_in_source"
    if symbol and symbol.upper() not in VALID_SYMBOLS and not re.search(rf"\${re.escape(symbol)}\b", text, re.I):
        # Preserve a literal contract for later typo/delisting review, but
        # never pretend it is a priceable known underlying.
        draft = dict(draft)
        draft.setdefault("uncertainties", []).append("Literal symbol is absent from the current local universe; possible typo or delisting. No canonical correction inferred.")
    contracts = source_contracts(text)
    has_contract = bool(contracts)
    has_symbol = bool(symbol and (str(symbol).upper() in {"SPY", "QQQ", "IWM", "SPX", "NDX", "AAPL", "NVDA", "TSLA", "MSFT", "ORCL", "GOOG", "GOOGL", "MA", "V", "WMT", "HUT", "MU", "LITE", "SLV", "GLD"} or re.search(r"\$[A-Z]{2,6}\b", text)))
    action = None
    speech = None
    if re.search(r"\b(?:watchlist\s*:|top\s+plays\s+this\s+week|(?:free\s+)?trade\s+ideas?\b|lotto\s+(?:friday\s+)?trade\s+ideas?)", text, re.I) and not re.search(r"\b(?:BTO|bought|i(?:'ve|\s+have)\s+just\s+bought)\b", text, re.I):
        action, speech = "watchlist", "conditional"
    elif re.search(r"\b(?:new\s+avg|average\s+cost)\b", text, re.I) and not re.search(r"\b(?:BTO|buy|bought|sold|STC)\b", text, re.I):
        action, speech = "position_update", "observation"
    elif RECAP.search(text) or re.search(r"\bCLOSED\s*:\s*\n", text, re.I):
        if re.search(r"\bopen\s+positions\b", text, re.I):
            action, speech = "position_update", "retrospective"
        elif PERFORMANCE.search(text) or re.search(r"\b(?:%|out|profit|paid|win)\b", text, re.I):
            action, speech = "performance_claim", "retrospective"
    elif PENDING.search(text):
        action, speech = "order_pending", "intention"
    elif re.search(r"\b(?:cancelled?|pull(?:ed|ing)?\s+the\s+order)\b", text, re.I) and (has_contract or has_symbol or source.get("reply_to_message_id")):
        action, speech = "order_cancel", "claimed_execution"
    elif TRIM.search(text) and (has_contract or has_symbol or source.get("reply_to_message_id") or re.search(r"\b(?:trim|taking\s+half|sold\s+\d+\s+more)\b", text, re.I)):
        action, speech = "trim", "instruction" if re.search(r"\b(?:if|will|would|going\s+to|gonna)\b", text, re.I) else "claimed_execution"
    elif SELL.search(text) and not re.search(r"\b(?:closing\s+up|sold\s+off|market\s+closed)\b", text, re.I) and (has_contract or has_symbol or source.get("reply_to_message_id") or len(text) < 70):
        action, speech = ("trim" if re.search(r"\b(?:STC\s+\d+|sold\s+\d+\s+(?:cons?|contracts?|shares?))\b", text, re.I) and not re.search(r"\b(?:all|full|fully|last)\b", text, re.I) else "exit"), "intention" if re.search(r"\b(?:will|would|going\s+to|gonna|in\s+a\s+(?:sec|minute))\b", text, re.I) else "claimed_execution"
    elif ADD.search(text) and (has_contract or has_symbol or source.get("reply_to_message_id")):
        action, speech = "add", "claimed_execution" if not re.search(r"\b(?:gonna|going\s+to|will|might)\b", text, re.I) else "intention"
    elif STOP.search(text) and not (PLAN.search(text) and not re.search(r"\b(?:SL\s*[\d.$]|stop\s*(?:at|to|raised|moved|set)\s*[\d.$]|break[ -]?even\s+stop)\b", text, re.I)) and (STOP_MOVED.search(text) or (not has_contract and not re.search(r"\b(?:non\s+stop|stop\s+wins|stop\s+bangers|stop\s+loss\s+on\s+every\s+trade)\b", text, re.I) and (has_symbol or source.get("reply_to_message_id") or re.search(r"\bSL\s*[\d.$]", text, re.I)))):
        action, speech = "stop_adjustment", "instruction"
    elif INVALIDATION.search(text) and (has_contract or has_symbol):
        action, speech = "invalidation", "conditional"
    elif (has_contract or has_symbol or source.get("reply_to_message_id")) and re.search(r"\b(?:coming\s+in|in\s+the\s+money|ITM)\b|\d+(?:\.\d+)?%", text, re.I) and not (ACTUAL_ENTRY.search(text) or re.search(r"\b(?:BUY|BTO|ENTRY)\b", text, re.I)):
        action, speech = ("watchlist", "observation") if re.search(r"\bcoming\s+in\b", text, re.I) else ("performance_claim", "retrospective")
    elif has_contract and PERFORMANCE.search(text) and not ACTUAL_ENTRY.search(text) and not re.search(r"\b(?:BUY|BTO)\b", text, re.I):
        action, speech = "performance_claim", "retrospective"
    elif has_contract and (ACTUAL_ENTRY.search(text) or ALERT_ENTRY.search(text) or (STOP.search(text) and len(text) < 180)):
        action, speech = "entry", "claimed_execution" if ACTUAL_ENTRY.search(text) and not re.search(r"\b(?:would|will|at\s+open|tomorrow|tmw|gonna|going\s+to)\b", text, re.I) else "instruction"
    elif has_contract and not PLAN.search(text) and not PERFORMANCE.search(text) and not SELL.search(text) and len(text) < 140:
        action, speech = "entry", "instruction"
    elif has_contract or (has_symbol and MARKET_PLAN.search(text)):
        action, speech = "watchlist", "conditional" if PLAN.search(text) else "observation"
    elif PERFORMANCE.search(text) and (has_symbol or source.get("reply_to_message_id")):
        action, speech = "performance_claim", "retrospective"
    if not action:
        return None, "no_grounded_trade_or_market_plan_clause"
    if NON_MARKET.search(text) and action not in {"watchlist", "performance_claim"}:
        return None, "non_market_or_promotional_context"
    # Require quote to be authored text, never a preview or OCR from another
    # message silently substituted for the caller's own words.
    e = dict(draft)
    e["source_quote"] = text
    e["event_type"], e["speech_act"] = action, speech
    e["reply_to_message_id"] = source.get("reply_to_message_id")
    e["timestamp_utc"] = source["timestamp_utc"]
    e["timestamp_raw"] = source["timestamp_raw"]
    e["is_edited"] = source["is_edited"]
    contract = next((item for item in contracts if item[1] == str(e.get("symbol") or "").upper()
                     and (e.get("strike") is None or item[2] == float(e["strike"]))),
                    contracts[0] if len(contracts) == 1 else None)
    if contract and e.get("instrument_type") != "multi_leg_option_strategy":
        e["symbol"] = contract[1]
        e["symbol_raw"] = contract[1]
        e["instrument_type"] = "option"
        e["option_type"] = contract[3]
        e["strike"] = contract[2]
        if action == "entry" and e.get("direction") in {None, "unknown"}:
            e["direction"] = "long"
    if re.search(r"\b(?:condor|straddle|strangle|vertical|credit\s+spread|debit\s+spread)\b", text, re.I):
        e.setdefault("uncertainties", []).append("Message describes a multi-leg strategy; this single annotation does not specify net position or all leg executions.")
        e["instrument_type"] = "multi_leg_option_strategy"
        e["direction"] = None
    if e.get("expiry_raw") and e.get("instrument_type") == "option":
        inferred_expiry, rule = asof_expiry(e["expiry_raw"], source["timestamp_utc"])
        if inferred_expiry:
            e["expiry"] = inferred_expiry
            e["expiry_resolution_rule"] = rule
            if rule == "nearest_future_month_day_within_60d_inferred_year":
                e.setdefault("uncertainties", []).append("Expiry year inferred from source M/D and alert date; not explicitly written.")
    if action in {"watchlist", "performance_claim", "position_update", "stop_adjustment", "invalidation", "order_pending"}:
        # A plan level or recap number is not an entry/exit fill price.
        e["price"] = None
        e["price_text"] = None
    elif action in {"entry", "add", "trim", "exit"}:
        e["price"], e["price_text"], e["price_extraction_rule"] = source_price(text, contract[0] if contract else None)
        if e.get("instrument_type") == "option" and e.get("price") is not None and e["price"] > 100 and e["price_text"] and not re.search(r"\d+\.\d+", e["price_text"]):
            e.setdefault("uncertainties", []).append("Large integer after option contract is not safely identifiable as an option premium; literal retained without numeric price.")
            e["price"] = None
            e["price_extraction_rule"] = "rejected_ambiguous_large_integer"
    if action in {"entry", "add", "trim", "exit", "stop_adjustment"}:
        e["stop_price"], e["stop_text"] = source_stop(text)
    if action in {"watchlist", "performance_claim", "position_update"}:
        e["quantity"] = None
        e["quantity_text"] = None
    if action != "stop_adjustment":
        # Keep an explicit stop in an entry/plan too; never conflate it with a
        # reported execution price.
        pass
    # On unqualified claims, 'unknown' is materially safer than option just
    # because the source says 'con' in an unrelated word or mentions 0DTE.
    if not has_contract and e.get("instrument_type") == "option" and not re.search(r"\b(?:option|call|put|con(?:s|tracts?)?|0DTE)\b", text, re.I):
        e["instrument_type"] = "unknown"
        e["option_type"] = None
        e["strike"] = None
        e["expiry"] = None
    if not has_contract and e.get("strike") is not None and not re.search(r"\b\d+(?:\.\d+)?\s*(?:calls?|puts?|[CP])\b", text, re.I):
        e["strike"] = None
    if action in {"entry", "add", "trim", "exit"} and not (has_contract or has_symbol or source.get("reply_to_message_id")):
        e.setdefault("uncertainties", []).append("Instrument identity is not explicit; lifecycle link may remain unresolved.")
    if draft.get("event_type") != action or draft.get("speech_act") != speech:
        e.setdefault("uncertainties", []).append(f"Curated from draft {draft.get('event_type')}/{draft.get('speech_act')} using reviewed grammar.")
    e["confidence"] = "medium" if has_contract and action in {"entry", "add"} else "low"
    fields = ("event_type", "speech_act", "symbol", "symbol_raw", "direction", "instrument_type", "option_type", "strike", "expiry_raw", "expiry", "quantity", "quantity_text", "price", "price_text", "stop_price", "stop_text", "rationale", "source_quote", "reply_to_message_id")
    e["field_sources"] = {key: [e["message_id"]] for key in fields if e.get(key) is not None}
    e["curation_method"] = "source_literal_and_conservative_speech_act_grammar_v1"
    return e, "kept"


def supplement(message: dict) -> dict | None:
    """Recover explicit execution/contract syntax missed by draft classifiers."""
    if message["author_id"] not in {"1079391263083733072", "720901855995101225"}:
        return None
    text = message.get("text") or ""
    if not text or NON_MARKET.search(text) or RECAP.search(text) or re.search(r"\bCLOSED\s*:", text, re.I):
        return None
    lower = text.lower()
    if re.search(r"\b(?:BTO|buy\s+to\s+open)\b", text, re.I):
        action, speech = "entry", "claimed_execution"
    elif re.search(r"\b(?:STC|sell\s+to\s+close)\b", text, re.I):
        action, speech = "trim", "claimed_execution"  # Full close is not established by STC alone.
    elif re.search(r"\b(?:my\s+stop\s+hit|stop\s+hit|stopped\s+out)\b", text, re.I):
        action, speech = "exit", "claimed_execution"
    elif re.search(r"\bi\s+sold\s+most\s+of\b", text, re.I):
        action, speech = "trim", "claimed_execution"
    elif re.search(r"\b(?:i\s+would\s+sell|you\s+can\s+exit|if\s+.*\bexit)\b", text, re.I):
        action, speech = "exit", "conditional"
    elif re.search(r"\b(?:i\s+am\s+going\s+to\s+buy|going\s+to\s+buy|gonna\s+buy)\b", text, re.I):
        action, speech = "order_pending", "intention"
    else:
        contract = re.search(r"(?<!\w)(\$?[A-Za-z]{2,6})\s+\$?(\d+(?:\.\d+)?)\s*[.]?\s*(calls?|puts?|[CP])\b", text, re.I)
        if contract and contract[1].lstrip("$").upper() in SOURCE_STOPWORDS:
            contract = None
        if not contract or PERFORMANCE.search(text) or re.search(r"\d+(?:\.\d+)?%|\b(?:if|would|could|watch(?:ing)?|looking|plan|above|below|tomorrow|tmw|next\s+week|coming\s+in|in\s+the\s+money)\b", text, re.I):
            alternate = re.search(r"\$?(\d+(?:\.\d+)?)\s*(calls?|puts?|[CP])\s+on\s+(\$?[A-Z]{2,6})\b", text, re.I)
            if not alternate or alternate[3].lstrip("$").upper() in SOURCE_STOPWORDS:
                return None
        action, speech = "entry", "instruction"
    symbol = None
    contract = re.search(r"(?<!\w)(\$?[A-Za-z]{2,6})\s+\$?(\d+(?:\.\d+)?)\s*[.]?\s*(calls?|puts?|[CP])\b", text, re.I)
    if contract and contract[1].lstrip("$").upper() not in SOURCE_STOPWORDS:
        symbol = contract[1].lstrip("$").upper()
    alternate = re.search(r"\$?(\d+(?:\.\d+)?)\s*(calls?|puts?|[CP])\s+on\s+(\$?[A-Z]{2,6})\b", text, re.I)
    if not symbol and alternate and alternate[3].lstrip("$").upper() not in SOURCE_STOPWORDS:
        symbol = alternate[3].lstrip("$").upper()
    if not symbol:
        ticker = re.search(r"\b(?:BTO|STC|buy\s+to\s+open|sell\s+to\s+close)(?:\s+\d+)?(?:\s+(?:cons?|contracts?|shares?))?\s+(\$?[A-Z]{2,6})\b", text, re.I)
        if ticker and ticker[1].lstrip("$").upper() not in SOURCE_STOPWORDS:
            symbol = ticker[1].lstrip("$").upper()
    if not symbol:
        cash = re.search(r"(?<!\w)\$([A-Z]{2,6})\b", text)
        if cash:
            symbol = cash[1]
    qty = re.search(r"\b(?:BTO|STC)\s+(\d+(?:\.\d+)?)\s*(?:cons?|contracts?|shares?)?\b", text, re.I)
    if not qty:
        qty = re.search(r"\b(\d+)\s*(?:cons?|contracts?|shares?)\b", text, re.I)
    matched_contracts = source_contracts(text)
    quoted_price, quoted_price_text, price_rule = source_price(text, matched_contracts[0][0] if matched_contracts else None)
    stop_price, stop_text = source_stop(text)
    expiry = re.search(r"\b(0\s*DTE|\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?)\b", text, re.I)
    option_type = None
    strike = None
    if contract:
        typ = contract[3].lower()
        option_type = "put" if typ.startswith("p") else "call"
        strike = float(contract[2])
    elif alternate and symbol == alternate[3].lstrip("$").upper():
        typ = alternate[2].lower()
        option_type = "put" if typ.startswith("p") else "call"
        strike = float(alternate[1])
    elif re.search(r"\b(?:calls?|puts?)\b", text, re.I):
        option_type = "put" if re.search(r"\bputs?\b", text, re.I) else "call"
    instrument = "option" if option_type or re.search(r"\bcons?|contracts?|0\s*DTE\b", text, re.I) else "unknown"
    mid = message["message_id"]
    e = {"event_id": f"{mid}:supplement:0", "message_id": mid, "author_id": message["author_id"],
         "channel_id": message["channel_id"], "timestamp_utc": message["timestamp_utc"],
         "timestamp_raw": message["timestamp_raw"], "is_edited": message["is_edited"],
         "event_type": action, "speech_act": speech,
         "symbol": symbol, "symbol_raw": symbol, "direction": "long" if action in {"entry", "order_pending"} and instrument == "option" else None,
         "instrument_type": instrument, "option_type": option_type, "strike": strike,
         "expiry_raw": expiry[1] if expiry else None, "expiry": None,
         "quantity": float(qty[1]) if qty else None, "quantity_text": qty[0] if qty else None,
         "price": quoted_price if action in {"entry", "trim", "exit"} else None,
         "price_text": quoted_price_text if action in {"entry", "trim", "exit"} else None,
         "price_extraction_rule": price_rule if action in {"entry", "trim", "exit"} else None,
         "stop_price": stop_price, "stop_text": stop_text, "rationale": None, "source_quote": text,
         "reply_to_message_id": message.get("reply_to_message_id"), "related_message_ids": [],
         "link_notes": "Recovered from explicit source syntax missed by draft classifier.",
         "confidence": "medium" if contract or symbol else "low",
         "uncertainties": ["Supplementary syntax extraction; manually review contract and lifecycle before promotion."]}
    if e["expiry_raw"]:
        e["expiry"], e["expiry_resolution_rule"] = asof_expiry(e["expiry_raw"], message["timestamp_utc"])
        if e.get("expiry_resolution_rule") == "nearest_future_month_day_within_60d_inferred_year":
            e["uncertainties"].append("Expiry year inferred from M/D and alert date; not written in source.")
    if e.get("instrument_type") == "option" and e.get("price") is not None and e["price"] > 100 and e.get("price_text") and not re.search(r"\d+\.\d+", e["price_text"]):
        e["uncertainties"].append("Large integer after option contract is not safely identifiable as an option premium; literal retained without numeric price.")
        e["price"] = None
        e["price_extraction_rule"] = "rejected_ambiguous_large_integer"
    e["field_sources"] = {key: [mid] for key, value in e.items() if value is not None and key not in {"field_sources", "uncertainties", "related_message_ids"}}
    return e


def curate(study: Path) -> dict:
    messages = {m["message_id"]: m for m in read_jsonl(study / "source/messages.jsonl")}
    draft_paths = [study / "annotations" / f"{name}.jsonl" for name in ("alerts", "challenge", "context")]
    kept, rejected, seen = [], [], set()
    for path in draft_paths:
        for raw in read_jsonl(path):
            mid = str(raw.get("message_id"))
            if mid not in messages:
                rejected.append({"event_id": raw.get("event_id"), "message_id": mid, "reason": "missing_source_message"})
                continue
            e, reason = decision(raw, messages[mid])
            if e is None:
                rejected.append({"event_id": raw.get("event_id"), "message_id": mid, "reason": reason})
                continue
            if e["event_id"] in seen:
                rejected.append({"event_id": e["event_id"], "message_id": mid, "reason": "duplicate_candidate_across_drafts"})
                continue
            seen.add(e["event_id"])
            kept.append(e)
    annotated_messages = {e["message_id"] for e in kept}
    recovered = 0
    for mid, message in messages.items():
        if mid in annotated_messages:
            continue
        extra = supplement(message)
        if extra:
            kept.append(extra)
            seen.add(extra["event_id"])
            recovered += 1
    kept.sort(key=lambda e: (e["timestamp_utc"], e["message_id"], e["event_id"]))
    return {"kept": kept, "rejected": rejected,
            "counts": {"kept": len(kept), "rejected": len(rejected), "supplement_recovered": recovered,
                       "event_types": dict(Counter(e["event_type"] for e in kept)),
                       "rejection_reasons": dict(Counter(e["reason"] for e in rejected))}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=DEFAULT_STUDY)
    args = parser.parse_args()
    result = curate(args.study)
    curated = args.study / "curated"
    curated.mkdir(exist_ok=True)
    for name, rows in (("events.jsonl", result["kept"]), ("rejected_candidates.jsonl", result["rejected"])):
        path = curated / name
        temp = path.with_suffix(path.suffix + ".tmp")
        write_jsonl(temp, rows)
        temp.replace(path)
    (curated / "summary.json").write_text(json.dumps(result["counts"], indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["counts"], indent=2))


if __name__ == "__main__":
    main()
