"""Find management evidence missed when Discord close posts carry screenshot prices.

Rows are review candidates. A displayed option price is a screenshot mark, not
an execution; 'gone' can mean sold or moved sharply and is kept ambiguous.
"""
from __future__ import annotations

import argparse
from collections import defaultdict, Counter
from datetime import datetime, timedelta
import json
from pathlib import Path
import re
from zoneinfo import ZoneInfo

from scripts.discord_ledger.build import available_at, read_jsonl, write_jsonl
from scripts.discord_ledger.curate import source_contracts

ACE_ID = "1079391263083733072"
NY = ZoneInfo("America/New_York")
CONTRACT = re.compile(r"\b(?P<symbol>[A-Z]{2,5})\s*\$?\s*(?P<strike>[\d,]+(?:\.\d+)?)\s*(?P<side>Call|Put)s?\b", re.I)
PRICE_LINE = re.compile(r"^\s*\$(\d+(?:\.\d+)?)\s*$")
EXIT = re.compile(r"\b(?:all\s+out|i[’']?m\s+out|i\s+am\s+out|closed|closing|stc|sold\s+all|stop\s+hit|sl\s+hit)\b", re.I)
PARTIAL = re.compile(r"\b(?:half\s+out|\d+%\s+out|trim|sold\s+\d+|taking\s+half)\b", re.I)
GONE = re.compile(r"\bgone\b", re.I)
HOLD = re.compile(r"\b(?:not\s+selling|ain[’']t\s+selling|we\s+swinging|still\s+holding)\b", re.I)


def action_kind(text: str) -> str | None:
    if PARTIAL.search(text):
        return "reported_partial_close"
    if EXIT.search(text):
        return "reported_full_close_or_stc_unspecified_quantity"
    if GONE.search(text):
        return "ambiguous_gone_or_price_movement" if HOLD.search(text) else "gone_close_claim_needs_review"
    return None


def screenshot_contracts(ocr_text: str) -> list[dict]:
    lines = ocr_text.splitlines()
    found = []
    for index, line in enumerate(lines):
        for match in CONTRACT.finditer(line):
            symbol = match["symbol"].upper()
            strike = float(match["strike"].replace(",", ""))
            side = match["side"].lower()
            # A current-price screenshot commonly places the displayed mark on
            # the immediately following line. Order confirmations are different.
            next_line = lines[index + 1] if index + 1 < len(lines) else ""
            mark_match = PRICE_LINE.fullmatch(next_line)
            is_order_screen = bool(re.search(r"order\s*(?:filled|executed)|contract\s*sold", ocr_text, re.I))
            found.append({"symbol": symbol, "strike": strike, "option_type": side,
                          "displayed_option_price": float(mark_match[1]) if mark_match and not is_order_screen else None,
                          "price_kind": "screenshot_displayed_current_price_not_fill" if mark_match and not is_order_screen else None,
                          "screen_kind": "order_confirmation_ocr" if is_order_screen else "option_price_screen",
                          "contract_ocr_line": line, "price_ocr_line": next_line if mark_match else None})
    return list({(r["symbol"], r["strike"], r["option_type"], r["displayed_option_price"]): r for r in found}.values())


def candidates_for(message: dict, ocr: dict, trades: list[dict]) -> list[dict]:
    if message.get("author_id") != ACE_ID:
        return []
    action = action_kind(message.get("text") or "")
    if not action:
        return []
    when, basis = available_at(message)
    if not when:
        return []
    post_day = datetime.fromisoformat(when.replace("Z", "+00:00")).astimezone(NY).date().isoformat()
    post_time = datetime.fromisoformat(when.replace("Z", "+00:00"))
    all_contracts = screenshot_contracts(ocr.get("ocr_text") or "")
    explicit_contracts = {(symbol, strike, side) for _, symbol, strike, side in source_contracts(message.get("text") or "")}
    explicit_symbols = {c["symbol"] for c in all_contracts
                        if re.search(r"(?<![A-Za-z0-9])\$?" + re.escape(c["symbol"]) + r"(?![A-Za-z0-9])",
                                     message.get("text") or "", re.I)}
    rows = []
    for contract in all_contracts:
        if explicit_contracts and (contract["symbol"], contract["strike"], contract["option_type"]) not in explicit_contracts:
            continue
        if explicit_symbols and contract["symbol"] not in explicit_symbols:
            continue
        symbol = "SPX" if contract["symbol"] == "SPXW" else contract["symbol"]
        matches = [trade for trade in trades
                   if trade["author_id"] == ACE_ID and trade.get("entry_event_id")
                   and trade.get("symbol") == symbol and trade.get("strike") == contract["strike"]
                   and trade.get("option_type") == contract["option_type"]
                   and trade.get("entry_available_at_utc") and trade["entry_available_at_utc"] < when
                   and post_time - datetime.fromisoformat(trade["entry_available_at_utc"].replace("Z", "+00:00")) <= timedelta(days=30 if trade.get("expiry") else 7)
                   and (not trade.get("expiry") or trade["expiry"] >= post_day)]
        rows.append({
            "message_id": message["message_id"], "posted_at_utc": message["timestamp_utc"],
            "available_at_utc": when, "availability_basis": basis,
            "source_text": message.get("text") or "", "action_text_class": action,
            "attachment_id": ocr["attachment_id"], "attachment_sha256": ocr["sha256"],
            "ocr_contract": contract, "candidate_trade_ids": [trade["trade_id"] for trade in matches],
            "candidate_entry_message_ids": [trade["entry_message_id"] for trade in matches],
            "link_status": "unique_contract_candidate" if len(matches) == 1 else "multiple_contract_candidates" if matches else "no_entry_candidate",
            "link_confidence": "medium" if len(matches) == 1 and action.startswith("reported_") else "low",
            "source_message_ids": sorted(set([message["message_id"]] + [trade["entry_message_id"] for trade in matches])),
            "limitation": "OCR may be wrong; screenshot current price is not a sale fill; close text and exact execution quantity require visual/context review.",
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    messages = {m["message_id"]: m for m in read_jsonl(args.study / "source/messages.jsonl")}
    trades = read_jsonl(args.run / "managed_trades.jsonl")
    rows = []
    for ocr in read_jsonl(args.study / "attachments/ocr.jsonl"):
        message = messages.get(ocr["message_id"])
        if message:
            rows.extend(candidates_for(message, ocr, trades))
    args.out.mkdir(parents=True, exist_ok=False)
    write_jsonl(args.out / "image_management_candidates.jsonl", rows)
    summary = {"rows": len(rows), "distinct_messages": len({r["message_id"] for r in rows}),
               "link_status": dict(Counter(r["link_status"] for r in rows)),
               "action_text_class": dict(Counter(r["action_text_class"] for r in rows)),
               "rows_with_displayed_option_price": sum(r["ocr_contract"]["displayed_option_price"] is not None for r in rows),
               "warning": "Discovery index; not a revised trade win rate or execution ledger."}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
