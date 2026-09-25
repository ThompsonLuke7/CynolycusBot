"""Reconcile reported option outcomes with explicitly tiered estimates.

This audit estimates a reference price change, not trader account P&L or a
follower fill. Every estimate retains entry and management message IDs.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re

from scripts.discord_ledger.build import read_jsonl, write_jsonl

FULL_WORDING = re.compile(r"\b(?:all\s+out|i[’']?m\s+out|i\s+am\s+out|closed|closing|sold\s+all|stop\s+hit|sl\s+hit)\b", re.I)
LOSS_WORDING = re.compile(r"\b(?:sold\s+with\s+(?:a\s+)?loss|closed\s+(?:with\s+)?(?:a\s+)?loss|stop\s+hit|sl\s+hit|stopped\s+out)\b", re.I)
LIMIT_PRICE = re.compile(r"Limit\s*price\s*\n\s*\$(\d+(?:\.\d+)?)", re.I)
ZERO_POSITION = re.compile(r"New\s*position\s*\n\s*[0O]\s*Contracts?\b", re.I)
FILLED = re.compile(r"order\s*(?:filled|executed)|contract\s*sold", re.I)


def reference(entry_price: float | None, later_price: float | None) -> float | None:
    return round(100 * (later_price / entry_price - 1), 3) if entry_price and later_price and entry_price > 0 and later_price > 0 else None


def screen_receipt(ocr_text: str) -> dict:
    filled = bool(FILLED.search(ocr_text))
    limit = LIMIT_PRICE.search(ocr_text) if filled else None
    return {"screenshot_reports_filled_order": filled,
            "screenshot_limit_price": float(limit[1]) if limit else None,
            "screenshot_new_position_zero": bool(ZERO_POSITION.search(ocr_text)) if filled else None,
            "interpretation": "Order-screen OCR is a source claim; sell limit is a lower bound on filled execution price, not necessarily the actual fill."}


def reconcile(trades: list[dict], text_outcomes: list[dict], image_candidates: list[dict],
              ocr_rows: list[dict], messages: list[dict]) -> tuple[list[dict], list[dict], dict]:
    outcomes = {row["trade_id"]: row for row in text_outcomes}
    ocr = {(row["message_id"], row["attachment_id"]): row for row in ocr_rows}
    images: dict[str, list[dict]] = defaultdict(list)
    for row in image_candidates:
        if row["link_status"] == "unique_contract_candidate":
            images[row["candidate_trade_ids"][0]].append(row)
    result = []
    for trade in trades:
        if trade["author_name"] != "ACE" or not trade.get("entry_event_id"):
            continue
        text = outcomes[trade["trade_id"]]
        entry_price = text["reported_entry_price"]
        evidence = []
        for leg in text["reported_price_change_legs"]:
            evidence.append({"kind": "reported_text_exit_or_trim_price", "event_id": leg["event_id"],
                             "message_ids": leg["source_message_ids"], "price": leg["reported_exit_price"],
                             "reference_return_pct": leg["reference_price_change_pct"],
                             "action": leg["event_type"], "price_role": "reported_execution_or_reference_not_broker_verified"})
        for image in sorted(images.get(trade["trade_id"], []), key=lambda row: row["available_at_utc"]):
            contract = image["ocr_contract"]
            source = ocr.get((image["message_id"], image["attachment_id"]))
            receipt = screen_receipt(source["ocr_text"]) if source else screen_receipt("")
            mark = contract["displayed_option_price"]
            explicit_full = bool(FULL_WORDING.search(image["source_text"]))
            evidence.append({"kind": "image_management_post", "message_ids": image["source_message_ids"],
                             "attachment_id": image["attachment_id"], "posted_at_utc": image["posted_at_utc"],
                             "action_text_class": image["action_text_class"],
                             "explicit_full_close_wording": explicit_full,
                             "displayed_mark": mark,
                             "mark_vs_alert_reference_pct": reference(entry_price, mark),
                             "mark_is_not_sale_fill": True,
                             **receipt})
        complete = text["complete_source_claimed_quantity_cycle"]
        best = None
        if complete:
            best = {"basis": "all_claimed_quantities_accounted_for_by_text_prices",
                    "gross_reference_return_pct": text["complete_cycle_reported_gross_price_change_pct"],
                    "source_message_ids": text["source_message_ids"],
                    "confidence": "low" if trade["link_confidence"] == "low" else "medium"}
        else:
            # A text-price exit is preferred. When its quantity is missing the
            # number is only an alert-to-exit reference, not full-cycle P&L.
            text_exit = next((leg for leg in evidence if leg["kind"] == "reported_text_exit_or_trim_price" and leg["action"] == "exit"), None)
            if text_exit:
                best = {"basis": "reported_text_exit_price_quantity_unresolved",
                        "gross_reference_return_pct": text_exit["reference_return_pct"],
                        "source_message_ids": text_exit["message_ids"], "confidence": "low"}
            else:
                text_sale_with_zero = next((leg for leg in evidence if leg["kind"] == "reported_text_exit_or_trim_price"
                                            and any(screen["kind"] == "image_management_post"
                                                    and screen["screenshot_reports_filled_order"]
                                                    and screen["screenshot_new_position_zero"]
                                                    and leg["event_id"].split(":", 1)[0] in screen["message_ids"]
                                                    for screen in evidence)), None)
                if text_sale_with_zero:
                    best = {"basis": "reported_text_sale_price_plus_image_zero_position",
                            "gross_reference_return_pct": text_sale_with_zero["reference_return_pct"],
                            "source_message_ids": text_sale_with_zero["message_ids"], "confidence": "low"}
            if best is None:
                # An order-confirmation image whose new position reads zero
                # provides an executed-sale lower bound, not exact P&L.
                confirmed = next((leg for leg in evidence if leg["kind"] == "image_management_post"
                                  and leg["screenshot_reports_filled_order"] and leg["screenshot_new_position_zero"]
                                  and leg["screenshot_limit_price"] and entry_price), None)
                if confirmed:
                    best = {"basis": "screenshot_filled_sell_limit_lower_bound_position_zero",
                            "gross_reference_return_pct": reference(entry_price, confirmed["screenshot_limit_price"]),
                            "source_message_ids": confirmed["message_ids"], "confidence": "low"}
                if best is None:
                    full_mark = next((leg for leg in evidence if leg["kind"] == "image_management_post"
                                      and leg["explicit_full_close_wording"] and leg["mark_vs_alert_reference_pct"] is not None), None)
                    if full_mark:
                        best = {"basis": "screenshot_mark_at_reported_full_close_not_fill",
                                "gross_reference_return_pct": full_mark["mark_vs_alert_reference_pct"],
                                "source_message_ids": full_mark["message_ids"], "confidence": "low"}
        source_ids = sorted(set(text["source_message_ids"] + [mid for leg in evidence for mid in leg["message_ids"]]))
        row = {"trade_id": trade["trade_id"], "author_name": "ACE", "symbol": trade["symbol"],
               "entry_message_id": trade["entry_message_id"], "entry_reference_price": entry_price,
               "evidence": evidence, "best_close_reference": best,
               "estimated_trade_pnl": None, "estimated_follower_pnl": None,
               "source_message_ids": source_ids,
               "limitations": "Selected public-channel evidence; missing/partial exits and adds prevent portfolio P&L and population win rate."}
        row["field_provenance"] = {key: {"source_message_ids": source_ids,
                                         "basis": "reported_prices_and_unique_contract_image_candidate_reconciliation",
                                         "confidence": "low"}
                                   for key in row}
        result.append(row)
    linked_messages = {event_id.split(":", 1)[0] for trade in trades for event_id in trade.get("event_ids", [])}
    image_messages = {row["message_id"] for row in image_candidates}
    unlinked_losses = []
    for message in messages:
        if message.get("author_name") != "ACE" or message["message_id"] in image_messages or message["message_id"] in linked_messages:
            continue
        text = message.get("text") or ""
        if LOSS_WORDING.search(text):
            unlinked_losses.append({"message_id": message["message_id"],
                                    "timestamp_utc": message["timestamp_utc"],
                                    "source_text": text,
                                    "source_message_ids": [message["message_id"]],
                                    "interpretation": "Reported loss/stop text; not assigned to a unique trade without contract evidence."})
    priced_image_rows = [row for row in result if any(
        leg["kind"] == "image_management_post" and leg["mark_vs_alert_reference_pct"] is not None
        for leg in row["evidence"])]
    priced_image_without_close = [row for row in priced_image_rows if row["best_close_reference"] is None]
    summary = {"ACE_entry_lifecycles": len(result),
               "best_close_reference_count": sum(row["best_close_reference"] is not None for row in result),
               "by_best_basis": dict(Counter(row["best_close_reference"]["basis"] for row in result if row["best_close_reference"])),
               "favorable_best_reference_count": sum(row["best_close_reference"] is not None and row["best_close_reference"]["gross_reference_return_pct"] > 0 for row in result),
               "unfavorable_best_reference_count": sum(row["best_close_reference"] is not None and row["best_close_reference"]["gross_reference_return_pct"] < 0 for row in result),
               "entries_with_any_image_management_evidence": sum(any(e["kind"] == "image_management_post" for e in row["evidence"]) for row in result),
               "entries_with_priced_image_mark_vs_alert": len(priced_image_rows),
               "priced_image_entries_without_close_reference": len(priced_image_without_close),
               "unassigned_reported_loss_messages": len(unlinked_losses),
               "interpretation": "Coverage of selected estimable public evidence, not the trader's profitability or an unbiased win rate."}
    return result, unlinked_losses, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--strategy-audit", type=Path, required=True)
    parser.add_argument("--image-recovery", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows, losses, summary = reconcile(
        read_jsonl(args.run / "managed_trades.jsonl"),
        read_jsonl(args.strategy_audit / "outcome_evidence.jsonl"),
        read_jsonl(args.image_recovery / "image_management_candidates.jsonl"),
        read_jsonl(args.study / "attachments/ocr.jsonl"),
        read_jsonl(args.study / "source/messages.jsonl"))
    args.out.mkdir(parents=True, exist_ok=False)
    write_jsonl(args.out / "estimated_trade_outcomes.jsonl", rows)
    write_jsonl(args.out / "unlinked_reported_losses.jsonl", losses)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
