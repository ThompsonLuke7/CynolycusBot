"""Index claimed results in the Discord ``post-ur-profits`` channel.

The channel is a result-claim stream, not an execution ledger. This parser
creates one source-cited row per visible ACE contract/result claim and keeps
member posts separately. It never converts a claim into broker P&L.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import timedelta
import json
from pathlib import Path
import re

from scripts.discord_ledger.build import read_jsonl, stamp, write_jsonl


CHANNEL_ID = "1416374095456632912"
ACE_ID = "1079391263083733072"
FT_ID = "720901855995101225"
CALLERS = {ACE_ID: "ACE", FT_ID: "FT"}
CONTRACT = re.compile(r"(?<![A-Za-z0-9])\$?(?P<symbol>[A-Z]{2,6})\s+(?P<strike>\d+(?:\.\d+)?)\s*(?P<right>[CP]|calls?|puts?)\b", re.I)
GENERIC = re.compile(r"(?<![A-Za-z0-9])\$?(?P<symbol>[A-Z]{2,6})\s+(?:(?P<right>calls?|puts?)\s+)?(?P<pct>[+-]?\d+(?:\.\d+)?)%", re.I)
ARROW = re.compile(r"\$?(?P<entry>\d+(?:\.\d+)?)\s*(?:→|->)\s*\$?(?P<exit>\d+(?:\.\d+)?)")
PCT = re.compile(r"(?P<pct>[+-]?\d+(?:\.\d+)?)%")
LOSS = re.compile(r"\b(?:loss|failed|stop(?:ped)?\s+out|red)\b", re.I)


def valid_symbols(trades: list[dict]) -> set[str]:
    return {trade["symbol"] for trade in trades if trade.get("author_name") in set(CALLERS.values()) and trade.get("symbol")}


def parse_claims(message: dict, symbols: set[str]) -> list[dict]:
    text = message.get("text") or ""
    contracts = list(CONTRACT.finditer(text))
    rows: list[dict] = []
    covered: list[tuple[int, int]] = []
    for index, match in enumerate(contracts):
        symbol = match["symbol"].upper()
        if symbol not in symbols:
            continue
        end = contracts[index + 1].start() if index + 1 < len(contracts) else len(text)
        chunk = text[match.start():end]
        arrow = ARROW.search(chunk)
        percentages = list(PCT.finditer(chunk))
        pct = float(percentages[-1]["pct"]) if percentages else None
        if LOSS.search(chunk) and (pct is None or pct > 0):
            pct = -abs(pct) if pct is not None else None
        rows.append({"symbol": symbol, "strike": float(match["strike"]),
                     "option_type": "put" if match["right"].lower().startswith("p") else "call",
                     "reported_entry_price": float(arrow["entry"]) if arrow else None,
                     "reported_exit_price": float(arrow["exit"]) if arrow else None,
                     "reported_return_pct": pct,
                     "claim_text": chunk.strip(), "parse_kind": "contract_claim"})
        covered.append((match.start(), end))
    # Also retain a non-contract result (e.g. "IWM puts 200%"), but do not
    # duplicate a symbol already represented by a contract-specific claim.
    contract_symbols = {row["symbol"] for row in rows}
    for match in GENERIC.finditer(text):
        symbol = match["symbol"].upper()
        if symbol not in symbols or symbol in contract_symbols:
            continue
        right = match["right"].lower() if match["right"] else None
        local = text[match.start():min(len(text), match.end() + 80)]
        pct = float(match["pct"])
        if LOSS.search(local):
            pct = -abs(pct)
        rows.append({"symbol": symbol, "strike": None,
                     "option_type": "call" if right and right.startswith("call") else "put" if right and right.startswith("put") else None,
                     "reported_entry_price": None, "reported_exit_price": None, "reported_return_pct": pct,
                     "claim_text": local.strip(), "parse_kind": "symbol_percent_claim"})
    return rows


def link_claim(claim: dict, message: dict, trades: list[dict], author: str) -> tuple[str, list[str]]:
    at = stamp(message["timestamp_utc"])
    candidates = []
    for trade in trades:
        if trade.get("author_name") != author or not trade.get("entry_event_id") or trade.get("symbol") != claim["symbol"]:
            continue
        entry = stamp(trade["entry_available_at_utc"])
        if not entry < at <= entry + timedelta(days=30):
            continue
        if claim["option_type"] and trade.get("option_type") and claim["option_type"] != trade["option_type"]:
            continue
        if claim["strike"] is not None and trade.get("strike") is not None and claim["strike"] != trade["strike"]:
            continue
        candidates.append(trade)
    if len(candidates) == 1:
        return "unique_prior_entry_candidate", [candidates[0]["trade_id"]]
    if not candidates:
        return "no_prior_entry_in_export", []
    return "ambiguous_prior_entries", [row["trade_id"] for row in candidates]


def build(messages: list[dict], trades: list[dict]) -> tuple[list[dict], list[dict], dict]:
    symbols = valid_symbols(trades)
    ace_rows, member_rows = [], []
    for message in messages:
        if message.get("channel_id") != CHANNEL_ID:
            continue
        text = message.get("text") or ""
        if message.get("author_id") in CALLERS:
            for claim in parse_claims(message, symbols):
                status, candidate_ids = link_claim(claim, message, trades, CALLERS[message["author_id"]])
                source_ids = [message["message_id"]]
                row = {"claim_id": f"{message['message_id']}:{len(ace_rows)}", "message_id": message["message_id"],
                       "timestamp_utc": message["timestamp_utc"], "author_name": CALLERS[message["author_id"]], **claim,
                       "link_status": status, "candidate_trade_ids": candidate_ids,
                       "broker_verified_pnl": None, "source_message_ids": source_ids,
                       "limitation": "ACE result claim from post-ur-profits; not a broker statement, complete trade inventory, or follower fill."}
                row["field_provenance"] = {key: {"source_message_ids": source_ids,
                                                  "basis": "literal_post_ur_profits_claim_parser",
                                                  "confidence": "low"}
                                           for key in row}
                ace_rows.append(row)
        elif re.search(r"\b(?:profit|gain|up\s+\$?|made\s+\$|banger|green|loss|red)\b|\+\d+(?:\.\d+)?%", text, re.I):
            member_rows.append({"message_id": message["message_id"], "timestamp_utc": message["timestamp_utc"],
                                "author_name": message.get("author_name"), "text": text,
                                "has_attachments": bool(message.get("attachments")), "source_message_ids": [message["message_id"]],
                                "interpretation": "Member-reported result language; no causality or comparable execution inferred."})
    summary = {"channel_id": CHANNEL_ID, "result_claim_rows_by_caller": dict(Counter(row["author_name"] for row in ace_rows)),
               "claim_return_sign_by_caller": {name: dict(Counter("positive" if row["reported_return_pct"] and row["reported_return_pct"] > 0 else "negative" if row["reported_return_pct"] and row["reported_return_pct"] < 0 else "unspecified" for row in ace_rows if row["author_name"] == name)) for name in CALLERS.values()},
               "claim_link_status_by_caller": {name: dict(Counter(row["link_status"] for row in ace_rows if row["author_name"] == name)) for name in CALLERS.values()},
               "member_result_language_messages": len(member_rows),
               "interpretation": "This indexes results claims present in the export; it is not a population performance calculation."}
    return ace_rows, member_rows, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--messages", type=Path, help="Optional merged normalized messages JSONL")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    message_path = args.messages or args.study / "source/messages.jsonl"
    ace, members, summary = build(read_jsonl(message_path), read_jsonl(args.run / "managed_trades.jsonl"))
    summary["messages_path"] = str(message_path)
    args.out.mkdir(parents=True, exist_ok=False)
    write_jsonl(args.out / "caller_result_claims.jsonl", ace)
    write_jsonl(args.out / "ace_result_claims.jsonl", [row for row in ace if row["author_name"] == "ACE"])
    write_jsonl(args.out / "ft_result_claims.jsonl", [row for row in ace if row["author_name"] == "FT"])
    write_jsonl(args.out / "member_result_language.jsonl", members)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
