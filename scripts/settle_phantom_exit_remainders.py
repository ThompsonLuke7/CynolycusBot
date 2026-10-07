#!/usr/bin/env python
"""Settle a full exit whose "remaining" quantity was already sold by an earlier order.

`reconcile_pending_exit` sizes a position from managed state. When state claims
more than the broker holds, the full exit sells what is actually there, the
broker goes flat, and the difference is left as a remainder nothing can ever
sell. The exit then stays `unresolved` ("disappearance requires activity
evidence") and logs an ERROR on every risk pass. Two Meta positions were stuck
this way on 2026-10-04:

  * HTFL: 169 shares; 27 trimmed 2026-08-14, 142 sold 2026-10-02. State still
    carried the pre-trim 169.
  * NTSK260918C00015000: state claimed 39 contracts, but Meta's 2026-09-02 entry
    never reached the broker (order id "?"). The only 39 contracts in the account
    were HTF's. HTF trimmed 6 on 09-17; Meta's risk pass sold the other 33 on
    09-18 and waited for 6 more that no one held.

The evidence the reconciler is waiting for exists: the earlier sell. This
script takes that order id from the operator and refuses unless ALL of these
hold: the pending exit is completely filled; the broker holds none of the
symbol in its latest snapshot; each named order is a sell of the same symbol
that is ALREADY BOOKED in a module's closed-trade ledger, dated after the
position's entry and before the exit; and the named quantities add up to the
remainder exactly.

It books nothing: every unit must already be in a ledger. A sell that is in no
ledger is a separate accounting question and is refused here.

Applying it (a) marks the exit's saved evidence complete, because
`recover_pending_exits` would otherwise restore the position on the next pass,
and (b) removes the position from managed state. The state file is copied to a
timestamped `.bak` first and one audit record is appended to
`<module>/ledger_repairs.jsonl`.

Run with no --settle to list stuck exits. Default is a dry run; pass --apply.

    scripts/settle_phantom_exit_remainders.py --module meta_ranker \\
        --state signals/meta_context/meta_ranker/live_state.json \\
        --settle HTFL=<trim order id> --apply
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

from core.live_4h_exec import route_for_symbol  # noqa: E402
from core.live_state import load_state, module_state_lock, save_state  # noqa: E402
from core.order_reconciliation import read_evidence, save_evidence  # noqa: E402

REASON = "phantom_exit_remainder"
STUCK = "exit_evidence_incomplete"


def _rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _utc(value) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def stuck_exits(managed: dict) -> dict[str, dict]:
    return {ticker: st for ticker, st in managed.items()
            if isinstance(st, dict) and isinstance(st.get("exit_pending"), dict)
            and st.get("reconciliation_required") == STUCK}


def broker_holds(root: Path, account: str, symbol: str) -> tuple[bool, str]:
    snapshots = sorted((root / "account_snapshots").glob(f"broker_equity_*_{account}.jsonl"))
    if not snapshots:
        raise SystemExit("no broker snapshot to prove the account is flat")
    latest = _rows(snapshots[-1])[-1]
    held = any(p.get("symbol") == symbol and float(p.get("qty") or 0) != 0
               for p in latest.get("positions", []))
    return held, str(latest.get("captured_at_et"))


def booked_sales(root: Path) -> dict[str, tuple[str, dict]]:
    """Every closed-trade row with a broker order id, keyed by that id."""
    out: dict[str, tuple[str, dict]] = {}
    for path in sorted(root.glob("*/closed_trades.jsonl")):
        for row in _rows(path):
            oid = str(row.get("order_id") or "")
            if oid and oid != "?":
                out[oid] = (path.parent.name, row)
    return out


def explain(st: dict, symbol: str, order_ids: list[str], booked) -> list[dict]:
    pending = st["exit_pending"]
    window = (_utc(st.get("entry_bar")), _utc(pending.get("submitted_ts")))
    out = []
    for oid in order_ids:
        if oid == pending.get("order_id"):
            raise SystemExit(f"{oid} is the exit itself, not an earlier sell")
        if oid not in booked:
            raise SystemExit(f"{oid} is not booked in any module ledger; settle that first")
        owner, row = booked[oid]
        if row.get("order_symbol") != symbol or str(row.get("side")).lower() != "sell":
            raise SystemExit(f"{oid} is not a sell of {symbol}")
        if not window[0] <= _utc(row["ts"]) <= window[1]:
            raise SystemExit(f"{oid} at {row['ts']} is outside the position's life {window}")
        out.append({"order_id": oid, "qty": float(row["qty"]), "booked_in": owner,
                    "at": row["ts"], "realized_pnl": row.get("realized_pnl")})
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--module", required=True)
    ap.add_argument("--state", required=True, help="the module's managed-state JSON")
    ap.add_argument("--settle", action="append", default=[], metavar="TICKER=ORDER_ID[,ORDER_ID]",
                    help="earlier sell order(s) that account for the remainder")
    ap.add_argument("--root", default=str(REPO / "Data/inference"))
    ap.add_argument("--account", default="paper")
    ap.add_argument("--apply", action="store_true", help="write (default is a dry run)")
    args = ap.parse_args(argv)
    root, state_path, module = Path(args.root), Path(args.state), args.module
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    stuck = stuck_exits(load_state(state_path).get("managed", {}))
    if not args.settle:
        for ticker, st in stuck.items():
            p = st["exit_pending"]
            print(f"  STUCK {module} {ticker}: exit {p.get('order_id')} filled "
                  f"{p.get('confirmed_qty')} of position {p.get('position_qty')}")
        print(f"{len(stuck)} stuck exit(s). Name the earlier sell with --settle to settle one.")
        return 0

    booked = booked_sales(root)
    plans = []
    for spec in args.settle:
        ticker, _, ids = spec.partition("=")
        order_ids = [x for x in ids.split(",") if x]
        st = stuck.get(ticker)
        if st is None or not order_ids:
            raise SystemExit(f"{ticker}: not a stuck exit in {state_path}, or no order id given")
        pending = st["exit_pending"]
        route = pending.get("route")
        symbol = st.get("occ") if route == "option" else st.get("symbol", ticker)
        if route != route_for_symbol(symbol):
            raise SystemExit(f"{ticker}: route {route!r} contradicts symbol {symbol}")
        sold, requested = float(pending.get("confirmed_qty") or 0), float(pending.get("qty") or 0)
        if pending.get("status") != "filled" or not math.isclose(sold, requested) or sold <= 0:
            raise SystemExit(f"{ticker}: the exit order is not completely filled")
        if pending.get("order_id") not in booked:
            raise SystemExit(f"{ticker}: the exit's own fill is not in the ledger")
        remainder = float(pending["position_qty"]) - sold
        held, as_of = broker_holds(root, args.account, symbol)
        if held:
            raise SystemExit(f"{ticker}: broker still holds {symbol} as of {as_of}")
        items = explain(st, symbol, order_ids, booked)
        explained = sum(i["qty"] for i in items)
        if remainder <= 0 or not math.isclose(explained, remainder, abs_tol=1e-8):
            raise SystemExit(f"{ticker}: remainder {remainder} but named sells total {explained}")
        plans.append((ticker, st, symbol, route, remainder, items, as_of))
        print(f"  SETTLE {module} {ticker} ({symbol}): position {pending['position_qty']:g} = "
              f"exit {sold:g} + earlier {explained:g}; broker flat as of {as_of[:16]}")
        for i in items:
            print(f"      {i['order_id'][:8]}  sold {i['qty']:g} at {i['at'][:19]}, "
                  f"booked in {i['booked_in']} ({i['realized_pnl']:+,.2f})")

    if not args.apply:
        print("DRY RUN — nothing written. Re-run with --apply to write.")
        return 0

    with module_state_lock(module) as acquired:
        if not acquired:
            raise SystemExit(f"{module} state is locked by a running pass; try again")
        state = load_state(state_path)
        for ticker, st, *_ in plans:
            live = state["managed"].get(ticker, {})
            if (live.get("exit_pending") or {}).get("order_id") != st["exit_pending"]["order_id"]:
                raise SystemExit(f"{ticker}: state changed since it was read; nothing written")
        backup = state_path.with_suffix(f"{state_path.suffix}.{stamp}.bak")
        shutil.copy2(state_path, backup)
        for ticker, st, symbol, route, remainder, items, as_of in plans:
            exit_oid = st["exit_pending"]["order_id"]
            evidence = read_evidence(root, module, exit_oid)
            if evidence is not None:
                size_key = "contracts" if route == "option" else "shares"
                evidence["state"].update({"remaining_qty": 0.0, size_key: 0.0})
                evidence["state"].pop("reconciliation_required", None)
                evidence.update(complete=True, result="closed",
                                ledger_repair={"reason": REASON, "repaired_at": stamp})
                save_evidence(root, module, exit_oid, evidence)
            del state["managed"][ticker]
            record = {"repaired_at": stamp, "reason": REASON, "module": module, "ticker": ticker,
                      "order_symbol": symbol, "exit_order_id": exit_oid,
                      "position_qty": st["exit_pending"]["position_qty"],
                      "sold_by_exit": st["exit_pending"]["confirmed_qty"], "remainder": remainder,
                      "explained_by": items, "broker_flat_as_of": as_of,
                      "state_backup": backup.name}
            with (root / module / "ledger_repairs.jsonl").open("a") as fh:
                fh.write(json.dumps(record, default=str, allow_nan=False) + "\n")
            print(f"  settled {ticker}: evidence marked complete, removed from managed state")
        save_state(state_path, state)
        print(f"  wrote {state_path}  (backup {backup.name})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
