#!/usr/bin/env python
"""Set each module's recorded position size to what its entry actually filled.

Two defects left managed state disagreeing with the broker on 2026-10-05, on
71 of 221 positions:

  * Under-claimed (56). `mark_entry_unconfirmed` read a new order about a
    second after sending it and kept that first, partial reading as the size
    (MYGN: ordered 1,264, read 51, filled 1,264). Sells are capped at the
    recorded size, so the exit would sell 51 and strand the rest unowned.
  * Over-claimed (15, Meta). Entries from before fills were read at all kept
    the PLANNED quantity while the governed path submitted a reduced order
    (ACVA: planned 480, filled 180). The exit sells what the broker has and
    then waits forever for a remainder that never existed.

`core.live_4h_exec.settle_entry_fill` stops new entries doing this. This script
repairs the positions already open.

Rule. A position is resized only when the broker's own records prove the size:

  1. exactly one book claims the symbol;
  2. its entry is identified: the buy order in the broker FILL journal that
     starts within two minutes of the state's `entry_submitted_at`; or, for a
     state with no entry clock, the only journal buy on or after `entry_bar`;
     or, for an entry older than the journal, the state's `entry_filled_qty`;
  3. no other buy of the symbol follows that entry; and
  4. entry fill minus every later sell of the symbol equals the broker's
     current quantity exactly.

The size is then set to the broker quantity. Anything that fails a step is
reported and left untouched. Each state file is copied to a timestamped `.bak`
first and every change is appended to `<module>/ledger_repairs.jsonl`.

A symbol two books claim cannot be settled by arithmetic alone: the broker nets
the position and says nothing about whose it is. For those the operator names
each claimant's entry order, taken from the server log, with

    --assign TTAN=multi_ticker_swing_htf:91@<order id>,meta_ranker:34@<order id>
    --assign TECX=multi_ticker_swing_htf:164@<order id>,meta_ranker:0

Every claimant must be named. Each order is read back from the broker and must
be a filled buy of that symbol for exactly the stated quantity, and the
quantities must add up to the broker position. A claimant given 0 has no order
behind it (Meta's 2026-09-02 entries were logged with `id=?` and never reached
the broker) and its claim is removed.

Default is a dry run against the live paper account; pass --apply to write.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from core.live_state import load_state, module_state_lock, save_state  # noqa: E402
from core.orphan_positions import (  # noqa: E402
    MANAGED_STATE_PATHS, SWING_BOOK_PATH, intraday_structure_symbols, managed_symbols,
    spy_daytrader_symbols,
)

REASON = "size_set_from_entry_fill"
ENTRY_MATCH_SECONDS = 120


def _utc(value) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def orders_from_fills(fills: list[dict]) -> list[dict]:
    """Collapse FILL activities into one record per order, oldest first."""
    orders: dict[str, dict] = {}
    for fill in fills:
        order = orders.setdefault(str(fill["order_id"]), {
            "order_id": str(fill["order_id"]), "side": str(fill["side"]).lower(),
            "qty": 0.0, "start": _utc(fill["transaction_time"])})
        order["qty"] += float(fill["qty"])
        order["start"] = min(order["start"], _utc(fill["transaction_time"]))
    return sorted(orders.values(), key=lambda o: o["start"])


def proven_size(st: dict, orders: list[dict], broker_qty: float) -> tuple[float | None, str]:
    """(size, evidence) when the broker's records prove it, else (None, why not)."""
    buys = [o for o in orders if o["side"] == "buy"]
    submitted = st.get("entry_submitted_at")
    entry = None
    if submitted:
        near = [o for o in buys
                if abs((o["start"] - _utc(submitted)).total_seconds()) <= ENTRY_MATCH_SECONDS]
        if len(near) > 1:
            return None, "more than one buy order at the entry time"
        entry = near[0] if near else None
        if entry is None and buys:
            return None, "journal buys do not match the recorded entry time"
    elif st.get("entry_bar"):
        after = [o for o in buys if o["start"] >= _utc(st["entry_bar"])]
        if len(after) != 1 or len(buys) != 1:
            return None, "no single journal buy identifies this entry"
        entry = after[0]

    if entry is not None:
        total, since, source = entry["qty"], entry["start"], f"order {entry['order_id'][:8]}"
    else:
        total = st.get("entry_filled_qty")
        if total is None or not submitted:
            return None, "entry is not in the journal and state records no fill"
        total, since, source = float(total), _utc(submitted), "state entry_filled_qty"

    later = [o for o in orders if o["start"] > since and o is not entry]
    if any(o["side"] == "buy" for o in later):
        return None, "another buy follows the entry"
    sold = sum(o["qty"] for o in later if o["side"] == "sell")
    if not math.isclose(total - sold, broker_qty, abs_tol=1e-8):
        return None, f"entry {total:g} - later sells {sold:g} != broker {broker_qty:g}"
    return broker_qty, f"{source} filled {total:g}, later sells {sold:g}"


def state_qty(ticker: str, st: dict) -> tuple[str, str, float]:
    """(size key, order symbol, recorded quantity), read the way `managed_symbols` does."""
    route = st.get("route", "option")
    key = "contracts" if route == "option" else "shares"
    symbol = st.get("occ") if route == "option" else st.get("symbol", ticker)
    qty = st.get("remaining_qty")
    return key, symbol, float((st.get(key) or 0) if qty is None else qty)


def other_books_symbols() -> set[str]:
    """Contracts held by the books that are not `managed` state files."""
    out = {s.upper() for s in intraday_structure_symbols()} | {s.upper() for s in spy_daytrader_symbols()}
    book = json.loads(SWING_BOOK_PATH.read_text()) if SWING_BOOK_PATH.exists() else {}
    out |= {str(p["option_symbol"]).upper() for p in book.get("positions") or []
            if isinstance(p, dict) and p.get("option_symbol")}
    return out


def broker_quantities(args) -> tuple[dict[str, float], str]:
    if args.snapshot:
        rows = [json.loads(line) for line in Path(args.snapshot).read_text().splitlines() if line.strip()]
        return ({p["symbol"]: float(p["qty"]) for p in rows[-1]["positions"]},
                f"snapshot {rows[-1].get('captured_at_et')}")
    from core.API.Alpaca_API.options.options_api import AlpacaOptionsClient

    positions = AlpacaOptionsClient(env_file=args.env_file).get_positions() or []
    return ({p["symbol"]: float(p["qty"]) for p in positions},
            f"broker read {datetime.now(timezone.utc).isoformat(timespec='seconds')}")


def parse_assignments(specs: list[str]) -> dict[str, dict[str, tuple[float, str | None]]]:
    """{symbol: {module: (qty, order id or None)}} from --assign arguments."""
    out: dict[str, dict[str, tuple[float, str | None]]] = {}
    for spec in specs:
        symbol, _, rest = spec.partition("=")
        for part in rest.split(","):
            module, _, tail = part.partition(":")
            qty, _, order_id = tail.partition("@")
            if not (symbol and module and qty):
                raise SystemExit(f"cannot read --assign {spec!r}")
            if float(qty) > 0 and not order_id:
                raise SystemExit(f"{symbol}: {module} is given {qty} but no entry order")
            out.setdefault(symbol, {})[module] = (float(qty), order_id or None)
    return out


def verify_assignment(symbol, shares, claimants, held, read_order) -> None:
    """Refuse unless the named orders account for the whole broker position."""
    if set(shares) != set(claimants):
        raise SystemExit(f"{symbol}: name every claimant {sorted(claimants)}, got {sorted(shares)}")
    if not math.isclose(sum(q for q, _ in shares.values()), held, abs_tol=1e-8):
        raise SystemExit(f"{symbol}: assigned quantities do not add up to broker {held:g}")
    for module, (qty, order_id) in shares.items():
        if qty == 0:
            continue
        order = read_order(order_id) or {}
        filled = float(order.get("filled_qty") or 0)
        if (order.get("symbol") != symbol or str(order.get("side")).lower() != "buy"
                or not math.isclose(filled, qty, abs_tol=1e-8)):
            raise SystemExit(f"{symbol}: order {order_id} is not a filled buy of {qty:g} {symbol} "
                             f"(broker says {order.get('side')} {filled:g} {order.get('symbol')})")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", default=str(REPO / "Data/inference"))
    ap.add_argument("--account", default="paper")
    ap.add_argument("--env-file", default=".env#PAPER")
    ap.add_argument("--snapshot", help="read positions from a broker snapshot file, not the broker")
    ap.add_argument("--assign", action="append", default=[],
                    metavar="SYMBOL=MODULE:QTY@ORDER_ID,...",
                    help="settle a symbol two books claim; see the module docstring")
    ap.add_argument("--apply", action="store_true", help="write (default is a dry run)")
    args = ap.parse_args(argv)
    root = Path(args.root)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    assignments = parse_assignments(args.assign)

    broker, as_of = broker_quantities(args)
    fills_by_symbol: dict[str, list[dict]] = {}
    journal = root / "broker_reconciliation" / args.account / "fills.jsonl"
    for line in journal.read_text().splitlines():
        if line.strip():
            activity = json.loads(line)["activity"]
            fills_by_symbol.setdefault(activity["symbol"], []).append(activity)
    claimants = managed_symbols()
    elsewhere = other_books_symbols()
    # The broker nets a symbol across books, so a shared one is right when the
    # claims add up to it, not when any single claim equals it.
    claimed_total: dict[str, float] = {}
    for module, path in MANAGED_STATE_PATHS.items():
        for ticker, st in load_state(path).get("managed", {}).items():
            if isinstance(st, dict):
                _, symbol, recorded = state_qty(ticker, st)
                claimed_total[symbol] = claimed_total.get(symbol, 0.0) + recorded

    plans: dict[str, list[dict]] = {}
    unproven = 0
    for module, path in MANAGED_STATE_PATHS.items():
        for ticker, st in load_state(path).get("managed", {}).items():
            if not isinstance(st, dict) or isinstance(st.get("exit_pending"), dict):
                continue
            key, symbol, recorded = state_qty(ticker, st)
            held = broker.get(symbol, 0.0)
            if not symbol or held <= 0 or math.isclose(recorded, held, abs_tol=1e-8):
                continue
            if symbol in assignments and symbol.upper() not in elsewhere:
                continue  # settled below, from the operator's named orders
            shared = len(claimants.get(symbol, [])) != 1 or symbol.upper() in elsewhere
            if shared and math.isclose(claimed_total.get(symbol, 0.0), held, abs_tol=1e-8):
                continue
            if shared:
                print(f"  SHARED  {module} {ticker}: state {recorded:g}, broker {held:g}, "
                      f"claimed by {claimants.get(symbol)} — not resized here")
                unproven += 1
                continue
            size, evidence = proven_size(st, orders_from_fills(fills_by_symbol.get(symbol, [])), held)
            if size is None:
                print(f"  UNPROVEN {module} {ticker}: state {recorded:g}, broker {held:g} ({evidence})")
                unproven += 1
                continue
            plans.setdefault(module, []).append({
                "ticker": ticker, "symbol": symbol, "key": key, "from": recorded, "to": size,
                "entry_bar": st.get("entry_bar"), "evidence": evidence})
            print(f"  RESIZE  {module} {ticker}: {recorded:g} -> {size:g}  ({evidence})")

    if assignments:
        from core.API.Alpaca_API.options.options_api import AlpacaOptionsClient

        client = AlpacaOptionsClient(env_file=args.env_file)
        for symbol, shares in assignments.items():
            held = broker.get(symbol, 0.0)
            verify_assignment(symbol, shares, claimants.get(symbol, []), held, client.get_order)
            for module, (qty, order_id) in shares.items():
                managed = load_state(MANAGED_STATE_PATHS[module]).get("managed", {})
                ticker, st = next((t, v) for t, v in managed.items()
                                  if isinstance(v, dict) and state_qty(t, v)[1] == symbol)
                key, _, recorded = state_qty(ticker, st)
                if isinstance(st.get("exit_pending"), dict):
                    raise SystemExit(f"{symbol}: {module} has an exit in flight; settle that first")
                if math.isclose(recorded, qty, abs_tol=1e-8):
                    print(f"  KEEP    {module} {ticker}: {recorded:g}  (order {order_id[:8]} filled {qty:g})")
                    continue
                evidence = (f"order {order_id[:8]} filled {qty:g}" if qty else
                            "no entry order behind this claim; the position is the other claimant's")
                plans.setdefault(module, []).append({
                    "ticker": ticker, "symbol": symbol, "key": key, "from": recorded, "to": qty,
                    "entry_bar": st.get("entry_bar"), "evidence": evidence})
                print(f"  {'RESIZE ' if qty else 'REMOVE '} {module} {ticker}: {recorded:g} -> {qty:g}  ({evidence})")

    total = sum(len(v) for v in plans.values())
    print(f"\n{total} position(s) to change, {unproven} left for review; positions from {as_of}")
    if not args.apply or not total:
        if total:
            print("DRY RUN — nothing written. Re-run with --apply to write.")
        return 0

    for module, changes in plans.items():
        path = MANAGED_STATE_PATHS[module]
        with module_state_lock(module) as acquired:
            if not acquired:
                print(f"  {module}: state is locked by a running pass; skipped, run again")
                continue
            state = load_state(path)
            backup = path.with_suffix(f"{path.suffix}.{stamp}.bak")
            shutil.copy2(path, backup)
            done = []
            for change in changes:
                st = state["managed"].get(change["ticker"])
                if (not isinstance(st, dict) or st.get("entry_bar") != change["entry_bar"]
                        or not math.isclose(state_qty(change["ticker"], st)[2], change["from"], abs_tol=1e-8)):
                    print(f"  {module} {change['ticker']}: changed since it was read; skipped")
                    continue
                if change["to"] == 0:
                    del state["managed"][change["ticker"]]
                else:
                    st[change["key"]] = change["to"]
                    if "remaining_qty" in st:
                        st["remaining_qty"] = change["to"]
                done.append({"repaired_at": stamp, "reason": REASON, "module": module,
                             "positions_as_of": as_of, "state_backup": backup.name, **change})
            save_state(path, state)
            with (root / module / "ledger_repairs.jsonl").open("a") as fh:
                for record in done:
                    fh.write(json.dumps(record, default=str, allow_nan=False) + "\n")
            print(f"  wrote {path}: {len(done)} changed (backup {backup.name})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
