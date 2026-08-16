#!/usr/bin/env python
"""Move test-fixture rows out of the live realized-P&L ledgers.

Before the repo-root `conftest.py` guard existed (added 2026-08-04), several
tests called `execute_plan(module=...)` without a `ledger_root`, so
`record_exit_realized_pnl` appended synthetic exits to the REAL
`Data/inference/<module>/closed_trades.jsonl`. The guard stops new pollution;
this removes the residue it left behind, which as of 2026-08-15 was 40 rows in
dealer_ranker and 9 in meta_ranker — enough that dealer_ranker's ledger read as
61 closed trades when only 21 were real.

Rows are QUARANTINED, never deleted: the original file is copied to a timestamped
`.bak`, removed rows are written to `closed_trades.quarantine.jsonl` next to the
ledger, and the cleaned ledger keeps every surviving row byte-for-byte in its
original order.

Discriminator: a real broker fill always carries a UUID `order_id` (Alpaca).
The test doubles build theirs as f"{symbol}-{side}" (e.g. "AAA260724C00010000-sell",
"OLD-sell"), so a non-UUID `order_id` identifies a synthetic row exactly. Rows
that carry no `order_id` at all are LEFT ALONE — that is a real exit whose
submission response lacked an id, not a fixture.

Default is a dry run; pass --apply to write.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LEDGER_GLOB = "Data/inference/*/closed_trades.jsonl"
_UUID = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I
)


def is_synthetic(record: dict) -> bool:
    """True when a ledger row was written by a test double, not a broker fill.

    Conservative on purpose: anything that is not clearly synthetic is kept. A
    row with no `order_id` is a real exit we failed to get an id for, and a row
    with a realized P&L came from a real fill price, so neither is ever removed.
    """
    if not isinstance(record, dict):
        return False
    if record.get("realized_pnl") is not None:
        return False
    order_id = record.get("order_id")
    if order_id in (None, ""):
        return False
    return not _UUID.match(str(order_id))


def scan(path: Path) -> tuple[list[str], list[str], list[str]]:
    """Split a ledger's raw lines into (kept, removed, unparseable)."""
    kept: list[str] = []
    removed: list[str] = []
    bad: list[str] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            bad.append(line)          # e.g. the audit_gap repair marker — always kept
            kept.append(line)
            continue
        (removed if is_synthetic(rec) else kept).append(line)
    return kept, removed, bad


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true",
                    help="write the cleaned ledgers (default is a dry run)")
    ap.add_argument("--root", default=str(REPO), help="repository root")
    args = ap.parse_args(argv)

    root = Path(args.root)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    total_removed = 0

    for path in sorted(root.glob(LEDGER_GLOB)):
        kept, removed, bad = scan(path)
        if bad:
            print(f"{path.relative_to(root)}: {len(bad)} unparseable line(s) preserved")
        if not removed:
            print(f"{path.relative_to(root)}: clean ({len(kept)} rows)")
            continue
        total_removed += len(removed)
        tickers = sorted({json.loads(r).get("ticker") for r in removed})
        print(f"{path.relative_to(root)}: {len(removed)} synthetic row(s) "
              f"{tickers} -> quarantine; {len(kept)} real row(s) kept")
        if not args.apply:
            continue
        backup = path.with_suffix(f".jsonl.{stamp}.bak")
        shutil.copy2(path, backup)
        quarantine = path.with_name("closed_trades.quarantine.jsonl")
        with quarantine.open("a") as fh:
            for line in removed:
                fh.write(line + "\n")
        path.write_text("".join(line + "\n" for line in kept))
        print(f"    backup   -> {backup.relative_to(root)}")
        print(f"    removed  -> {quarantine.relative_to(root)}")

    if not args.apply:
        print(f"\nDRY RUN — {total_removed} row(s) would be quarantined. "
              f"Re-run with --apply to write.")
    else:
        print(f"\nQuarantined {total_removed} row(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
