#!/usr/bin/env python
"""Put a closed trade's P&L in the ledger of the module that opened the position.

The paper account nets each symbol across modules, so a module can sell shares
or contracts another module bought. The account total is right either way; the
per-module totals are wrong by exactly that trade. Known on 2026-10-07:

  * NTSK260918C00015000 x33, +$3,960, in Meta's ledger. HTF bought all 39
    (Meta's own entry was logged `id=?` and never reached the broker).
  * ABCL x123, +$68.88, in Momentum's ledger. Meta bought them a day earlier.
  * Nine Intraday Structure contracts the 30m swing adopted and sold as
    `restored_unknown_expiring`.
  * Exits no ledger holds at all: an Intraday contract closed through
    `core.startup_queue`, and one that expired worthless after its claim was
    dropped.

Ownership follows the ENTRY order. Every action names the entry order, and the
row is moved or written only when the records agree:

  1. The broker says the entry is a filled buy of the symbol, filled before the
     exit, for at least the exit's quantity.
  2. Exactly one module is on record as having sent that entry, and it is the
     destination. The records read, in order: the module's saved exit evidence
     (`exit_orders/*.json`, which carries `entry_order_id`), a server-log line
     that names both the module and the order id (`<module> pending-open:
     submitted buy ... id=<order id>`), and, for Intraday Structure, its own
     `intraday execution: BUY <symbol> x<qty> filled @ <price>` line within 15
     seconds of the broker's fill time with the same quantity and price.
  3. `--move`: the source ledger holds exactly one row for that exit (matched by
     exit order id, or by symbol, quantity, price and time where the ledger
     records no order id) and its entry price is the entry order's fill price.
     `--book`: no ledger holds a row for that exit.
  4. The destination ledger does not already hold it.

    --move SRC=DST:EXIT_ORDER_ID:ENTRY_ORDER_ID
    --book DST:EXIT_ORDER_ID:ENTRY_ORDER_ID
    --book-expiry DST:ENTRY_ORDER_ID      (the broker's OPEXP activity is the exit)

A moved row keeps every field. Its `module` becomes the destination, the
original module and the evidence are recorded inline under `ledger_repair`, the
matching `exit_fills.jsonl` rows and the order registry's owner move with it,
and each touched file is copied to a timestamped `.bak` first. Both modules get
a line in `ledger_repairs.jsonl`.

Default is a dry run against the live paper account; pass --apply to write.
Arguments can be read from a file, one per line, with `@path`.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import math
import re
import shutil
import sys
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from core.live_4h_exec import closed_trade_record, parse_occ_expiry, route_for_symbol  # noqa: E402

REASON = "reattributed_to_entry_owner"
PRICE_TOL = 0.005
ROW_MATCH_SECONDS = 300
INTRADAY_LOG_SECONDS = 15
INTRADAY = "intraday_structure"
_ET = ZoneInfo("America/New_York")
_OCC = re.compile(r"^([A-Z]{1,6})\d{6}[CP]\d{8}$")
_INTRADAY_BUY = re.compile(
    r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),\d+ .*intraday execution: BUY (\S+) x(\d+) filled @ ([\d.]+)")


class Refused(SystemExit):
    """The records do not prove the change; nothing is written."""


def _utc(value) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _lines(path: Path) -> list[str]:
    return [line for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in _lines(path)]


def filled_order(read_order, order_id: str, side: str) -> dict:
    """The broker's record of a filled `side` order, as plain numbers."""
    try:
        order = read_order(order_id) or {}
    except Exception as exc:  # noqa: BLE001 - an order the broker cannot show proves nothing
        raise Refused(f"order {order_id} could not be read from the broker ({exc})") from exc
    qty = float(order.get("filled_qty") or 0)
    if str(order.get("side")).lower() != side or qty <= 0 or not order.get("filled_avg_price"):
        raise Refused(f"order {order_id} is not a filled {side} at the broker "
                      f"(side={order.get('side')}, filled={order.get('filled_qty')})")
    return {"id": str(order_id), "symbol": str(order["symbol"]), "qty": qty,
            "price": float(order["filled_avg_price"]), "filled_at": _utc(order["filled_at"])}


def entry_order_owners(entry: dict, root: Path, logs_dir: Path) -> dict[str, str]:
    """{module: the record that names it} for every module on record as sending `entry`."""
    owners: dict[str, str] = {}
    modules = sorted(p.name for p in root.iterdir() if (p / "closed_trades.jsonl").exists())
    for module in modules:
        for path in (root / module / "exit_orders").glob("*.json"):
            if (json.loads(path.read_text()).get("state") or {}).get("entry_order_id") == entry["id"]:
                owners.setdefault(module, f"exit evidence {path.name[:12]}")
                break
    fill_et = entry["filled_at"].tz_convert(_ET)
    for day in (fill_et, fill_et - timedelta(days=1)):
        log = logs_dir / f"server_{day.strftime('%Y%m%d')}.log"
        if not log.exists():
            continue
        with log.open(errors="replace") as fh:
            for number, line in enumerate(fh, 1):
                where = f"{log.name}:{number}"
                if entry["id"] in line:
                    for module in modules:
                        if re.search(rf"\b{re.escape(module)}\b", line):
                            owners.setdefault(module, where)
                    continue
                hit = _INTRADAY_BUY.match(line) if "intraday execution: BUY" in line else None
                if hit and hit.group(2) == entry["symbol"]:
                    logged = pd.Timestamp(hit.group(1), tz=_ET)
                    lag = abs((logged - fill_et).total_seconds())
                    if (lag <= INTRADAY_LOG_SECONDS and float(hit.group(3)) == entry["qty"]
                            and math.isclose(float(hit.group(4)), entry["price"], abs_tol=PRICE_TOL)):
                        owners.setdefault(INTRADAY, where)
    return owners


def require_owner(entry: dict, dest: str, root: Path, logs_dir: Path) -> str:
    owners = entry_order_owners(entry, root, logs_dir)
    if set(owners) != {dest}:
        raise Refused(f"entry {entry['id'][:8]} ({entry['symbol']}): on record as sent by "
                      f"{sorted(owners) or 'no module'}, not only {dest}")
    return owners[dest]


def row_is_exit(row: dict, exit_: dict) -> bool:
    """Does this ledger row describe the broker's `exit_` order?"""
    if row.get("order_id"):
        return str(row["order_id"]) == exit_["id"]
    try:
        return (row.get("order_symbol") == exit_["symbol"]
                and math.isclose(float(row["qty"]), exit_["qty"], abs_tol=1e-8)
                and math.isclose(float(row["exit_fill_price"]), exit_["price"], abs_tol=PRICE_TOL)
                and abs((_utc(row["ts"]) - exit_["filled_at"]).total_seconds()) <= ROW_MATCH_SECONDS)
    except (KeyError, TypeError, ValueError):
        return False


def holders(root: Path, exit_: dict) -> dict[str, list[int]]:
    """{module: [line indexes]} of every ledger row describing `exit_`."""
    out: dict[str, list[int]] = {}
    for path in sorted(root.glob("*/closed_trades.jsonl")):
        hits = [i for i, row in enumerate(_rows(path)) if row_is_exit(row, exit_)]
        if hits:
            out[path.parent.name] = hits
    return out


def check_entry_and_exit(entry: dict, exit_: dict) -> None:
    if entry["symbol"] != exit_["symbol"]:
        raise Refused(f"entry is {entry['symbol']} but the exit is {exit_['symbol']}")
    if entry["filled_at"] >= exit_["filled_at"]:
        raise Refused(f"{exit_['symbol']}: entry {entry['id'][:8]} filled after the exit")
    if entry["qty"] + 1e-8 < exit_["qty"]:
        raise Refused(f"{exit_['symbol']}: entry filled {entry['qty']:g}, exit sold {exit_['qty']:g}")


def plan_move(spec: str, read_order, root: Path, logs_dir: Path) -> dict:
    modules, _, orders = spec.partition(":")
    source, _, dest = modules.partition("=")
    exit_id, _, entry_id = orders.partition(":")
    if not (source and dest and exit_id and entry_id) or source == dest:
        raise Refused(f"cannot read --move {spec!r}")
    exit_, entry = filled_order(read_order, exit_id, "sell"), filled_order(read_order, entry_id, "buy")
    check_entry_and_exit(entry, exit_)
    held = holders(root, exit_)
    if set(held) != {source} or len(held[source]) != 1:
        raise Refused(f"exit {exit_id[:8]}: expected one row in {source}, found {held or 'none'}")
    row = _rows(root / source / "closed_trades.jsonl")[held[source][0]]
    basis = row.get("entry_avg_price")
    if not basis or not math.isclose(float(basis), entry["price"], abs_tol=PRICE_TOL):
        raise Refused(f"exit {exit_id[:8]}: row basis {basis} is not entry {entry_id[:8]}'s "
                      f"fill price {entry['price']:g}")
    evidence = require_owner(entry, dest, root, logs_dir)
    return {"action": "move", "source": source, "dest": dest, "row": row,
            "exit": exit_, "entry": entry, "evidence": evidence}


def plan_book(spec: str, read_order, root: Path, logs_dir: Path) -> dict:
    dest, _, orders = spec.partition(":")
    exit_id, _, entry_id = orders.partition(":")
    if not (dest and exit_id and entry_id):
        raise Refused(f"cannot read --book {spec!r}")
    exit_, entry = filled_order(read_order, exit_id, "sell"), filled_order(read_order, entry_id, "buy")
    check_entry_and_exit(entry, exit_)
    held = holders(root, exit_)
    if held:
        raise Refused(f"exit {exit_id[:8]} is already booked in {sorted(held)}; use --move")
    evidence = require_owner(entry, dest, root, logs_dir)
    return {"action": "book", "dest": dest, "exit": exit_, "entry": entry, "evidence": evidence,
            "row": _new_row(dest, entry, qty=exit_["qty"], price=exit_["price"],
                            when=exit_["filled_at"], reason="booked_from_broker_fill",
                            order_id=exit_["id"], outcome="exit_filled")}


def plan_expiry(spec: str, read_order, read_activities, root: Path, logs_dir: Path) -> dict:
    dest, _, entry_id = spec.partition(":")
    if not (dest and entry_id):
        raise Refused(f"cannot read --book-expiry {spec!r}")
    entry = filled_order(read_order, entry_id, "buy")
    symbol, expiry = entry["symbol"], parse_occ_expiry(entry["symbol"])
    if expiry is None:
        raise Refused(f"{symbol} is not an option contract")
    expired = [a for a in read_activities(symbol, expiry)
               if a.get("symbol") == symbol and a.get("activity_type") == "OPEXP"
               and a.get("status") == "executed"]
    if (len(expired) != 1 or not math.isclose(-float(expired[0].get("qty") or 0), entry["qty"])
            or float(expired[0].get("net_amount") or 0) != 0):
        raise Refused(f"{symbol}: no single worthless-expiry activity for {entry['qty']:g} "
                      f"contracts (found {[(a.get('qty'), a.get('net_amount')) for a in expired]})")
    for path in sorted(root.glob("*/closed_trades.jsonl")):
        if any(r.get("order_symbol") == symbol and (
                r.get("entry_order_id") == entry_id
                or r.get("activity_id") == expired[0]["id"]) for r in _rows(path)):
            raise Refused(f"{symbol}: {path.parent.name} already books this position")
    evidence = require_owner(entry, dest, root, logs_dir)
    when = pd.Timestamp(expired[0]["date"], tz=_ET) + pd.Timedelta(hours=16)
    row = _new_row(dest, entry, qty=entry["qty"], price=0.0, when=when.tz_convert("UTC"),
                   reason="expired_worthless", order_id=None, outcome="expired_worthless")
    row["activity_id"] = expired[0]["id"]
    return {"action": "book", "dest": dest, "entry": entry, "evidence": evidence, "row": row,
            "exit": {"id": f"OPEXP {expired[0]['id']}", "symbol": symbol}}


def _new_row(dest: str, entry: dict, *, qty, price, when, reason, order_id, outcome) -> dict:
    symbol = entry["symbol"]
    route = route_for_symbol(symbol)
    mult = 100.0 if route == "option" else 1.0
    occ = _OCC.match(symbol)
    row = closed_trade_record(
        module=dest, bar=when.isoformat(), ticker=occ.group(1) if occ else symbol,
        order_symbol=symbol, route=route, qty=qty, exit_reason=reason,
        entry_avg_price=entry["price"], exit_fill_price=price,
        realized_pnl=round((price - entry["price"]) * qty * mult, 2),
        entry_state={"entry_order_id": entry["id"], "entry_fill_price": entry["price"],
                     "entry_filled_qty": entry["qty"],
                     "entry_filled_at": entry["filled_at"].isoformat()},
        order_id=order_id)
    row["settle_outcome"] = outcome
    return row


@contextmanager
def _locked(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(path.suffix + ".lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        yield


def _backup(path: Path, stamp: str) -> str | None:
    if not path.exists():
        return None
    backup = path.with_suffix(f"{path.suffix}.{stamp}.bak")
    if not backup.exists():
        shutil.copy2(path, backup)
    return backup.name


def _write(path: Path, lines: list[str]) -> None:
    path.write_text("".join(line + "\n" for line in lines))


def apply_plan(plan: dict, root: Path, account: str, stamp: str) -> None:
    dest_ledger = root / plan["dest"] / "closed_trades.jsonl"
    repair = {"repaired_at": stamp, "reason": REASON if plan["action"] == "move" else plan["row"]["exit_reason"],
              "entry_order_id": plan["entry"]["id"], "entry_owner_evidence": plan["evidence"]}
    row = dict(plan["row"])
    if plan["action"] == "move":
        source_dir = root / plan["source"]
        source_ledger = source_dir / "closed_trades.jsonl"
        repair["from_module"] = plan["source"]
        with _locked(source_ledger):
            lines = _lines(source_ledger)
            at = [i for i, line in enumerate(lines) if json.loads(line) == plan["row"]]
            if len(at) != 1:
                raise Refused(f"{source_ledger} changed since it was read; this row was not moved")
            repair["source_backup"] = _backup(source_ledger, stamp)
            _write(source_ledger, lines[:at[0]] + lines[at[0] + 1:])
        row.update(module=plan["dest"], order_id=row.get("order_id") or plan["exit"]["id"])
        fills = source_dir / "exit_fills.jsonl"
        moved = [l for l in _lines(fills) if json.loads(l).get("order_id") == plan["exit"]["id"]]
        if moved:
            with _locked(fills):
                _backup(fills, stamp)
                _write(fills, [l for l in _lines(fills) if l not in moved])
            dest_fills = root / plan["dest"] / "exit_fills.jsonl"
            with _locked(dest_fills):
                _backup(dest_fills, stamp)
                _write(dest_fills, _lines(dest_fills) + [
                    json.dumps({**json.loads(l), "module": plan["dest"]}, default=str) for l in moved])
        registry = root / "broker_reconciliation" / account / "order_registry.jsonl"
        owned = [json.loads(l) for l in _lines(registry)]
        if any(r.get("order_id") == plan["exit"]["id"] for r in owned):
            with _locked(registry):
                _backup(registry, stamp)
                _write(registry, [json.dumps(
                    {**r, "module": plan["dest"], "ledger_repair": repair}
                    if r.get("order_id") == plan["exit"]["id"] else r, default=str) for r in owned])
        with (source_dir / "ledger_repairs.jsonl").open("a") as fh:
            fh.write(json.dumps({**repair, "module": plan["source"], "to_module": plan["dest"],
                                 "row": plan["row"]}, default=str, allow_nan=False) + "\n")
    row["ledger_repair"] = repair
    with _locked(dest_ledger):
        repair["dest_backup"] = _backup(dest_ledger, stamp)
        with dest_ledger.open("a") as fh:
            fh.write(json.dumps(row, default=str, allow_nan=False) + "\n")
    with (root / plan["dest"] / "ledger_repairs.jsonl").open("a") as fh:
        fh.write(json.dumps({**repair, "module": plan["dest"], "action": plan["action"],
                             "row": row}, default=str, allow_nan=False) + "\n")


def main(argv: list[str] | None = None, *, read_order=None, read_activities=None) -> int:
    # `@file` reads arguments from a file, one per line: order ids are long.
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], fromfile_prefix_chars="@")
    ap.add_argument("--root", default=str(REPO / "Data/inference"))
    ap.add_argument("--logs", default=str(REPO / "logs/live_server"))
    ap.add_argument("--account", default="paper")
    ap.add_argument("--env-file", default=".env#PAPER")
    ap.add_argument("--move", action="append", default=[], metavar="SRC=DST:EXIT_ID:ENTRY_ID")
    ap.add_argument("--book", action="append", default=[], metavar="DST:EXIT_ID:ENTRY_ID")
    ap.add_argument("--book-expiry", action="append", default=[], metavar="DST:ENTRY_ID")
    ap.add_argument("--apply", action="store_true", help="write (default is a dry run)")
    args = ap.parse_args(argv)
    root, logs_dir = Path(args.root), Path(args.logs)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    if read_order is None:
        from core.API.Alpaca_API.options.options_api import AlpacaOptionsClient

        client = AlpacaOptionsClient(env_file=args.env_file)
        read_order = client.get_order

        def read_activities(symbol, expiry):
            return client.get_account_activities(
                activity_types="OPEXP", after=f"{expiry.isoformat()}T00:00:00Z",
                until=f"{(expiry + timedelta(days=4)).isoformat()}T00:00:00Z", page_size=100) or []

    plans, refused = [], 0
    jobs = ([(plan_move, s, (read_order, root, logs_dir)) for s in args.move]
            + [(plan_book, s, (read_order, root, logs_dir)) for s in args.book]
            + [(plan_expiry, s, (read_order, read_activities, root, logs_dir)) for s in args.book_expiry])
    for build, spec, deps in jobs:
        try:
            plan = build(spec, *deps)
        except Refused as exc:
            print(f"  REFUSED {spec}: {exc}")
            refused += 1
            continue
        plans.append(plan)
        row = plan["row"]
        origin = f"{plan['source']} -> " if plan["action"] == "move" else "(unbooked) -> "
        print(f"  {plan['action'].upper():<5} {origin}{plan['dest']}: {row['order_symbol']} x{row['qty']:g} "
              f"{row['entry_avg_price']:g} -> {row['exit_fill_price']:g}  pnl {row['realized_pnl']:+,.2f}  "
              f"[entry {plan['entry']['id'][:8]} per {plan['evidence']}]")

    print(f"\n{len(plans)} change(s) proven, {refused} refused")
    if not args.apply or not plans:
        if plans:
            print("DRY RUN — nothing written. Re-run with --apply to write.")
        return 1 if refused else 0
    if refused:
        raise SystemExit("some changes were refused; fix or drop them before --apply")
    for plan in plans:
        apply_plan(plan, root, args.account, stamp)
    print(f"wrote {len(plans)} change(s); backups end .{stamp}.bak")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
