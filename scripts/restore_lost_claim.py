#!/usr/bin/env python
"""Put a position back under the module that opened it.

A module's managed state can stop naming a position it still holds (see
`core.orphan_positions`). Four Meta equities were unowned this way on
2026-10-05: BE, AMLX, SGMT and TENX. In each case Meta had sized a take-profit
trim that never filled, state was decremented anyway, and the later horizon
exit sold the reduced size and dropped the claim, leaving the trim-sized
remainder at the broker with nothing to exit it.

Restoring the claim lets the module finish the exit it already decided, and
books the result in its own ledger. A startup-queue close would also flatten
the position, but books it nowhere.

A claim is restored only when the records prove whose lot it is:

  1. no book claims the symbol;
  2. the broker holds it, and the whole broker quantity is restored;
  3. the module's closed-trade ledger has a priced sell of the symbol at the
     broker's average entry price (the same lot), and no other module's does;
  4. the module's signal audit planned a buy of the symbol on the stated bar.

The restored entry carries that entry bar, the broker's average entry price,
`trimmed: true` (the trim is why a remainder exists), and `runs_held` equal to
the number of passes the module has logged since the entry, which is what the
counter would read had the claim never been lost. Equities only.

It also carries `exit_reconciled_ids`: every exit order this position already
settled. Without it the restore undoes itself. The engine keeps each exit
order's evidence on disk and replays it onto any position with the same entry
bar that does not list the order as reconciled, which is how a crash between a
fill and a state save is recovered. The first version of this tool left the
list off; on 2026-10-07 at 09:33 Meta's risk pass replayed the 09-24, 09-28 and
10-02 horizon exits onto the restored AMLX, SGMT and TENX, found them filled,
and dropped all three claims again. (BE survived only because its exit predates
the evidence files.)

Default is a dry run against the live paper account; pass --apply to write.

    scripts/restore_lost_claim.py --module meta_ranker \\
        --state signals/meta_context/meta_ranker/live_state.json \\
        --restore "BE=2026-07-20 14:00:00+00:00" --apply
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
from core.orphan_positions import claimed_symbols  # noqa: E402

REASON = "lost_claim_restored"


def _utc(value) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def same_lot_sellers(root: Path, symbol: str, basis: float) -> dict[str, list[str]]:
    """{module: [order ids]} of priced sells of `symbol` booked at exactly this basis."""
    out: dict[str, list[str]] = {}
    for path in sorted(root.glob("*/closed_trades.jsonl")):
        for row in _rows(path):
            price = row.get("entry_avg_price")
            if (row.get("order_symbol") == symbol and row.get("realized_pnl") is not None
                    and price and math.isclose(float(price), basis, rel_tol=1e-6)):
                out.setdefault(path.parent.name, []).append(str(row.get("order_id")))
    return out


def settled_exit_ids(root: Path, module: str, symbol: str, entry_bar) -> list[str]:
    """Exit orders of `symbol` this module has already settled for this position.

    Read from the engine's own exit evidence (`<module>/exit_orders/*.json`),
    matched on symbol and entry bar: exactly the records
    `core.order_reconciliation.recover_pending_exits` would replay.
    """
    out: list[str] = []
    for path in sorted((root / module / "exit_orders").glob("*.json")):
        record = json.loads(path.read_text())
        saved = record.get("state") or {}
        order_id = (saved.get("exit_pending") or {}).get("order_id")
        if (order_id and record.get("symbol") == symbol
                and str(saved.get("entry_bar")) == str(entry_bar) and order_id not in out):
            out.append(str(order_id))
    return out


def audit_history(root: Path, module: str, symbol: str, entry_bar) -> tuple[bool, int]:
    """(a buy was planned on the entry bar, passes logged since it)."""
    entry, planned, later = _utc(entry_bar), False, set()
    path = root / module / "live_signal_audit.jsonl"
    with path.open() as fh:
        for line in fh:
            if '"order_plan"' not in line:
                continue
            try:
                record = json.loads(line)
            except ValueError:
                continue  # a torn audit line is not evidence either way
            if record.get("event") != "order_plan" or not record.get("bar"):
                continue
            bar = _utc(record["bar"])
            if bar > entry:
                later.add(bar)
            elif bar == entry:
                rows = (record.get("plan") or []) + (record.get("planned") or [])
                planned = planned or any(r.get("symbol") == symbol and
                                         str(r.get("side")).lower() == "buy" for r in rows)
    return planned, len(later)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--module", required=True)
    ap.add_argument("--state", required=True, help="the module's managed-state JSON")
    ap.add_argument("--restore", action="append", default=[], required=True,
                    metavar="SYMBOL=ENTRY_BAR", help="symbol and the bar the module entered on")
    ap.add_argument("--root", default=str(REPO / "Data/inference"))
    ap.add_argument("--env-file", default=".env#PAPER")
    ap.add_argument("--apply", action="store_true", help="write (default is a dry run)")
    args = ap.parse_args(argv)
    root, state_path, module = Path(args.root), Path(args.state), args.module
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    from core.API.Alpaca_API.options.options_api import AlpacaOptionsClient

    broker = {p["symbol"]: p for p in AlpacaOptionsClient(env_file=args.env_file).get_positions() or []}
    claimed = claimed_symbols(strict=True)

    entries: dict[str, dict] = {}
    for spec in args.restore:
        symbol, _, entry_bar = spec.partition("=")
        if not entry_bar or route_for_symbol(symbol) != "equity":
            raise SystemExit(f"{spec!r}: need EQUITY_SYMBOL=ENTRY_BAR")
        if symbol.upper() in claimed:
            raise SystemExit(f"{symbol}: a book already claims it")
        held = broker.get(symbol)
        if held is None or float(held["qty"]) <= 0:
            raise SystemExit(f"{symbol}: the broker does not hold it")
        qty, basis = float(held["qty"]), float(held["avg_entry_price"])
        sellers = same_lot_sellers(root, symbol, basis)
        if set(sellers) != {module}:
            raise SystemExit(f"{symbol}: ledger sells at basis {basis:g} are by "
                             f"{sorted(sellers) or 'no module'}, not only {module}")
        planned, passes = audit_history(root, module, symbol, entry_bar)
        if not planned:
            raise SystemExit(f"{symbol}: {module} planned no buy on {entry_bar}")
        settled = settled_exit_ids(root, module, symbol, entry_bar)
        entries[symbol] = {
            "route": "equity", "symbol": symbol, "shares": qty, "runs_held": passes,
            "bars_out": passes, "trimmed": True, "entry_bar": str(entry_bar),
            "entry_avg_price": basis, "exit_reconciled_ids": settled,
        }
        print(f"  RESTORE {module} {symbol}: {qty:g} sh @ {basis:g}, entered {entry_bar}, "
              f"{passes} passes since; same-lot sell {sellers[module][0][:8]} in its ledger; "
              f"{len(settled)} settled exit(s) marked reconciled")

    if not args.apply:
        print("DRY RUN — nothing written. Re-run with --apply to write.")
        return 0

    with module_state_lock(module) as acquired:
        if not acquired:
            raise SystemExit(f"{module} state is locked by a running pass; try again")
        state = load_state(state_path)
        clash = sorted(set(entries) & set(state.get("managed", {})))
        if clash:
            raise SystemExit(f"{clash} appeared in state since it was read; nothing written")
        backup = state_path.with_suffix(f"{state_path.suffix}.{stamp}.bak")
        shutil.copy2(state_path, backup)
        state.setdefault("managed", {}).update(entries)
        save_state(state_path, state)
        with (root / module / "ledger_repairs.jsonl").open("a") as fh:
            for symbol, entry in entries.items():
                fh.write(json.dumps({"repaired_at": stamp, "reason": REASON, "module": module,
                                     "state_backup": backup.name, "restored": entry},
                                    default=str, allow_nan=False) + "\n")
        print(f"  wrote {state_path}: {len(entries)} claim(s) restored (backup {backup.name})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
