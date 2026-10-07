#!/usr/bin/env python
"""Settle closed-trade rows that were written before their fill was known.

A module that books a close from the order's acknowledgement writes
`exit_fill_price: null, realized_pnl: null` when the fill has not landed yet.
The row is never revisited, so one of two things is then true and the ledger
cannot say which:

  * the order filled a moment later (SPY260928C00765000 on 2026-09-28: booked
    14:51, filled 15:06), and the row is a real trade with no P&L; or
  * the order never filled and a later order closed the position, which wrote
    its own priced row (SPY261001C00765000 on 2026-10-01), and the unpriced row
    is a second record of one close.

The broker FILL journal settles it by order id. A row is touched only when it
is unpriced, carries a real order id, and falls inside the period the journal
covers:

  * fills under that order id, selling the same symbol for the same quantity:
    the row is priced from them;
  * no fills under that order id, and the same module has a later priced close
    of the same symbol that day: the row is moved to
    `closed_trades.quarantine.jsonl`, exactly as written.

Anything else, including every row older than the journal, is left alone.
Changed files are copied to a timestamped `.bak` first and each change is
appended to `<module>/ledger_repairs.jsonl`.

Default is a dry run; pass --apply to write.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from core.live_4h_exec import route_for_symbol  # noqa: E402
from scripts.repair_route_mislabeled_exits import (  # noqa: E402
    _dump, _lines, broker_fills_by_order, rewrite,
)

_UUID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I)


def classify(row: dict, rows: list[dict], fills: list[dict], covered) -> tuple[str, dict | None]:
    """("price", corrected row) | ("duplicate", None) | ("leave", None)."""
    order_id, symbol = str(row.get("order_id") or ""), row.get("order_symbol")
    if row.get("realized_pnl") is not None or not _UUID.match(order_id):
        return "leave", None
    if not covered[0] <= str(row.get("ts")) <= covered[1]:
        return "leave", None
    basis, qty = row.get("entry_avg_price"), float(row.get("qty") or 0)
    if fills:
        sold = sum(float(f["qty"]) for f in fills)
        if (any(f.get("symbol") != symbol or str(f.get("side")).lower() != "sell" for f in fills)
                or not math.isclose(sold, qty, abs_tol=1e-8) or not basis):
            return "leave", None
        price = sum(float(f["qty"]) * float(f["price"]) for f in fills) / sold
        mult = 100.0 if route_for_symbol(symbol) == "option" else 1.0
        return "price", {**row, "exit_fill_price": round(price, 6),
                         "realized_pnl": round((price - float(basis)) * qty * mult, 2),
                         "fill_gain": round(price / float(basis) - 1.0, 6)}
    later = [r for r in rows if r.get("order_symbol") == symbol and r.get("realized_pnl") is not None
             and str(r.get("ts"))[:10] == str(row.get("ts"))[:10] and str(r.get("ts")) > str(row.get("ts"))]
    return ("duplicate", None) if later else ("leave", None)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", default=str(REPO / "Data/inference"))
    ap.add_argument("--account", default="paper")
    ap.add_argument("--apply", action="store_true", help="write (default is a dry run)")
    args = ap.parse_args(argv)
    root = Path(args.root)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    by_order = broker_fills_by_order(root / "broker_reconciliation" / args.account / "fills.jsonl")
    times = sorted(f["transaction_time"] for fills in by_order.values() for f in fills)
    if not times:
        raise SystemExit("the broker fill journal is empty")
    covered = (times[0], times[-1])

    total = 0
    for ledger in sorted(root.glob("*/closed_trades.jsonl")):
        module, lines = ledger.parent.name, _lines(ledger)
        rows = [json.loads(line) if line.strip() else {} for line in lines]
        edits: dict[int, dict | None] = {}
        audit, quarantined = [], []
        for index, row in enumerate(rows):
            verdict, fixed = classify(row, rows, by_order.get(str(row.get("order_id")), []), covered)
            if verdict == "price":
                marker = {"reason": "priced_from_broker_fills", "repaired_at": stamp}
                edits[index] = {**fixed, "ledger_repair": marker}
                print(f"  PRICE     {module} {row['ts'][:16]} {row['order_symbol']} qty {row['qty']:g}: "
                      f"exit {fixed['exit_fill_price']:g}, realized {fixed['realized_pnl']:+,.2f}")
            elif verdict == "duplicate":
                edits[index] = None
                quarantined.append(lines[index])
                print(f"  DUPLICATE {module} {row['ts'][:16]} {row['order_symbol']} qty {row['qty']:g}: "
                      f"order {row['order_id'][:8]} never filled; a later close is booked")
            else:
                continue
            audit.append({"repaired_at": stamp, "module": module, "action": verdict,
                          "order_id": row.get("order_id"), "order_symbol": row.get("order_symbol"),
                          "original": row, "corrected": fixed})
        total += len(edits)
        if not edits or not args.apply:
            continue
        backup = rewrite(ledger, edits, stamp)
        if quarantined:
            with ledger.with_name("closed_trades.quarantine.jsonl").open("a") as fh:
                fh.write("".join(line + "\n" for line in quarantined))
        with (root / module / "ledger_repairs.jsonl").open("a") as fh:
            fh.write("".join(_dump(record) + "\n" for record in audit))
        print(f"  wrote {module}/closed_trades.jsonl  ({len(edits)} row(s); backup {backup.name})")

    print(f"\n{total} row(s) settled; journal covers {covered[0][:16]} to {covered[1][:16]}")
    if total and not args.apply:
        print("DRY RUN — nothing written. Re-run with --apply to write.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
