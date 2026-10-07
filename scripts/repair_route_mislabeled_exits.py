#!/usr/bin/env python
"""Re-book share sales that the exit ledger recorded as option sales (x100).

Meta's options-mode pass submits one plan holding option and share rows and,
until 2026-10-04, labelled every sell in it "option". A share sale on that pass
was therefore booked with the x100 contract multiplier in three places:

  * `Data/inference/<module>/closed_trades.jsonl`  (route, realized_pnl)
  * `Data/inference/<module>/exit_fills.jsonl`     (proceeds, realized_pnl)
  * `Data/inference/broker_reconciliation/<account>/order_registry.jsonl` (route)

The registry row is why the broker-fill certificate never flagged these: it
multiplies broker proceeds by 100 for an "option" owner, so both sides of its
comparison were inflated together.

Rule: a row is repaired only when its route says "option", its order symbol is
not an OCC contract, AND the broker FILL journal proves the sale: same order
id, a share symbol, the same quantity and the same average price. The corrected
P&L is recomputed from the row's own prices and must equal recorded/100 to the
cent. Anything that cannot be proven that way is reported and left untouched
(for example the unpriced 2026-08-27 / 09-02 rows, which carry no order id and
predate the fill journal).

Nothing is deleted. Each changed file is first copied to a timestamped `.bak`,
the corrected row keeps its original values under `ledger_repair`, and one
audit record per order is appended to `<module>/ledger_repairs.jsonl`. Every
other line is kept byte-for-byte in its original order.

Default is a dry run; pass --apply to write.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from core.live_4h_exec import route_for_symbol  # noqa: E402

REASON = "share_sale_booked_as_option_x100"


def _lines(path: Path) -> list[str]:
    return path.read_text().splitlines() if path.exists() else []


def _dump(record: dict) -> str:
    # Same serialisation as core.live_state.append_once.
    return json.dumps(record, default=str, allow_nan=False)


def broker_fills_by_order(journal: Path) -> dict[str, list[dict]]:
    by_order: dict[str, list[dict]] = {}
    for line in _lines(journal):
        if line.strip():
            activity = json.loads(line)["activity"]
            by_order.setdefault(str(activity.get("order_id")), []).append(activity)
    return by_order


def verify_against_broker(row: dict, fills: list[dict]) -> tuple[dict | None, str]:
    """Return (evidence, "") when the broker proves this row was a share sale."""
    symbol, qty, exit_px, basis, pnl = (row.get("order_symbol"), row.get("qty"),
        row.get("exit_fill_price"), row.get("entry_avg_price"), row.get("realized_pnl"))
    if pnl is None or exit_px is None or not basis or not qty:
        return None, "unpriced row"
    if not fills:
        return None, "order id not in the broker fill journal"
    if any(f.get("symbol") != symbol or str(f.get("side")).lower() != "sell" for f in fills):
        return None, "broker fills are not sells of this symbol"
    broker_qty = sum(float(f["qty"]) for f in fills)
    broker_proceeds = sum(float(f["qty"]) * float(f["price"]) for f in fills)
    if not math.isclose(broker_qty, float(qty), abs_tol=1e-8):
        return None, f"broker qty {broker_qty} != ledger qty {qty}"
    if not math.isclose(broker_proceeds / broker_qty, float(exit_px), rel_tol=1e-4):
        return None, f"broker avg price {broker_proceeds / broker_qty} != ledger {exit_px}"
    corrected = round(float(pnl) / 100.0, 2)
    recomputed = round((float(exit_px) - float(basis)) * float(qty), 2)
    if not math.isclose(corrected, recomputed, abs_tol=0.01):
        return None, f"recorded/100 {corrected} != per-share P&L {recomputed}"
    return {"broker_qty": broker_qty, "broker_proceeds": round(broker_proceeds, 6),
            "activity_ids": [f["id"] for f in fills], "corrected_pnl": corrected}, ""


def _marker(stamp: str, **original) -> dict:
    return {"reason": REASON, "repaired_at": stamp, **original}


def plan_module(ledger: Path, fills_by_order: dict[str, list[dict]], stamp: str):
    """Decide the closed_trades edits for one module."""
    edits: dict[int, dict] = {}      # line index -> corrected row
    orders: dict[str, dict] = {}     # order id -> audit record
    skipped: list[tuple[dict, str]] = []
    for index, line in enumerate(_lines(ledger)):
        if not line.strip():
            continue
        row = json.loads(line)
        symbol = row.get("order_symbol")
        if row.get("route") != "option" or route_for_symbol(symbol) != "equity":
            continue
        oid = str(row.get("order_id") or "")
        evidence, why = verify_against_broker(row, fills_by_order.get(oid, []))
        if evidence is None:
            skipped.append((row, why))
            continue
        corrected = {**row, "route": "equity", "realized_pnl": evidence["corrected_pnl"],
                     "ledger_repair": _marker(stamp, original_route=row["route"],
                                              original_realized_pnl=row["realized_pnl"])}
        edits[index] = corrected
        orders[oid] = {"repaired_at": stamp, "reason": REASON, "module": ledger.parent.name,
                       "order_id": oid, "order_symbol": symbol, "qty": row.get("qty"),
                       "original": {"route": row["route"], "realized_pnl": row["realized_pnl"]},
                       "corrected": {"route": "equity", "realized_pnl": evidence["corrected_pnl"]},
                       "broker_evidence": {k: evidence[k] for k in
                                           ("broker_qty", "broker_proceeds", "activity_ids")}}
    return edits, orders, skipped


def plan_exit_fills(path: Path, orders: dict[str, dict], stamp: str) -> dict[int, dict]:
    edits: dict[int, dict] = {}
    totals: dict[str, float] = {}
    for index, line in enumerate(_lines(path)):
        if not line.strip():
            continue
        row = json.loads(line)
        oid = str(row.get("order_id") or "")
        if oid not in orders or "ledger_repair" in row:
            continue
        proceeds = round(float(row["proceeds"]) / 100.0, 6)
        pnl = None if row.get("realized_pnl") is None else round(float(row["realized_pnl"]) / 100.0, 2)
        totals[oid] = totals.get(oid, 0.0) + proceeds
        edits[index] = {**row, "proceeds": proceeds, "realized_pnl": pnl,
                        "ledger_repair": _marker(stamp, original_proceeds=row["proceeds"],
                                                 original_realized_pnl=row.get("realized_pnl"))}
    for oid, total in totals.items():
        broker = orders[oid]["broker_evidence"]["broker_proceeds"]
        if not math.isclose(total, broker, abs_tol=0.01):
            raise SystemExit(f"exit_fills for {oid}: corrected proceeds {total} != broker {broker}")
    return edits


def plan_registry(path: Path, orders: dict[str, dict], stamp: str) -> dict[int, dict]:
    edits: dict[int, dict] = {}
    for index, line in enumerate(_lines(path)):
        if not line.strip():
            continue
        row = json.loads(line)
        if str(row.get("order_id")) in orders and row.get("route") == "option":
            edits[index] = {**row, "route": "equity",
                            "ledger_repair": _marker(stamp, original_route="option")}
    return edits


def rewrite(path: Path, edits: dict[int, dict | None], stamp: str) -> Path:
    """Replace the edited lines in place, under the same lock append_once takes.

    An edit of None removes the line; the caller keeps the original elsewhere.
    """
    with path.with_suffix(path.suffix + ".lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        lines = path.read_text().splitlines()
        backup = path.with_suffix(f"{path.suffix}.{stamp}.bak")
        shutil.copy2(path, backup)
        for index, record in edits.items():
            lines[index] = None if record is None else _dump(record)
        lines = [line for line in lines if line is not None]
        fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            with os.fdopen(fd, "w") as fh:
                fh.write("".join(line + "\n" for line in lines))
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(name, path)
        finally:
            if os.path.exists(name):
                os.unlink(name)
    return backup


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", default=str(REPO / "Data/inference"))
    ap.add_argument("--account", default="paper")
    ap.add_argument("--apply", action="store_true",
                    help="write the corrected ledgers (default is a dry run)")
    args = ap.parse_args(argv)
    root = Path(args.root)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    recon = root / "broker_reconciliation" / args.account
    fills_by_order = broker_fills_by_order(recon / "fills.jsonl")

    writes: list[tuple[Path, dict[int, dict]]] = []
    all_orders: dict[str, dict] = {}
    for ledger in sorted(root.glob("*/closed_trades.jsonl")):
        edits, orders, skipped = plan_module(ledger, fills_by_order, stamp)
        module = ledger.parent.name
        for row, why in skipped:
            print(f"  SKIP  {module} {row.get('ts', '')[:10]} {row.get('order_symbol')}: {why}")
        if not edits:
            continue
        for oid, rec in orders.items():
            print(f"  FIX   {module} {rec['order_symbol']:<6} qty {rec['qty']:g}: realized_pnl "
                  f"{rec['original']['realized_pnl']:,.2f} -> {rec['corrected']['realized_pnl']:,.2f}"
                  f"  (order {oid[:8]}, {len(rec['broker_evidence']['activity_ids'])} broker fills)")
        writes.append((ledger, edits))
        fill_edits = plan_exit_fills(ledger.with_name("exit_fills.jsonl"), orders, stamp)
        if fill_edits:
            writes.append((ledger.with_name("exit_fills.jsonl"), fill_edits))
        all_orders.update(orders)

    registry_edits = plan_registry(recon / "order_registry.jsonl", all_orders, stamp)
    if registry_edits:
        writes.append((recon / "order_registry.jsonl", registry_edits))

    before = sum(o["original"]["realized_pnl"] for o in all_orders.values())
    after = sum(o["corrected"]["realized_pnl"] for o in all_orders.values())
    print(f"\n{len(all_orders)} order(s): recorded {before:,.2f} -> corrected {after:,.2f}; "
          f"{sum(len(e) for _, e in writes)} row(s) across {len(writes)} file(s)")
    if not all_orders:
        return 0
    if not args.apply:
        print("DRY RUN — nothing written. Re-run with --apply to write.")
        return 0

    for path, edits in writes:
        backup = rewrite(path, edits, stamp)
        print(f"  wrote {path.relative_to(root)}  ({len(edits)} row(s); backup {backup.name})")
    by_module: dict[str, list[dict]] = {}
    for rec in all_orders.values():
        by_module.setdefault(rec["module"], []).append(rec)
    for module, records in by_module.items():
        with (root / module / "ledger_repairs.jsonl").open("a") as fh:
            for rec in records:
                fh.write(_dump(rec) + "\n")
        print(f"  audit {module}/ledger_repairs.jsonl  (+{len(records)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
