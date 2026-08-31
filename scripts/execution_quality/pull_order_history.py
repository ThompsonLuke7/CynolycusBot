"""Stage 0a — pull the paper account's full order history to a local cache.

WHY: no entry submit/fill timestamp is persisted anywhere in the repo. The
`order_plan` audit line carries no wall clock and `closed_trades.entry_bar` is
the SIGNAL bar, not the fill. The broker is the only source for T_submit and
T_fill on entries, and it is authoritative.

Read-only: GET /v2/orders only. Nothing here can submit, modify, or cancel.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.API.Alpaca_API.options.options_api import AlpacaOptionsClient

OUT = REPO_ROOT / "research/execution_quality/data/order_history.jsonl"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--after", default="2026-06-01T00:00:00Z")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    client = AlpacaOptionsClient()
    seen: dict[str, dict] = {}
    cursor = args.after
    pages = 0
    while True:
        batch = client.get_orders(
            status="all", limit=500, direction="asc", after=cursor, nested="true"
        )
        if not isinstance(batch, list) or not batch:
            break
        pages += 1
        new = 0
        for o in batch:
            oid = str(o.get("id"))
            if oid not in seen:
                seen[oid] = o
                new += 1
        last = batch[-1].get("submitted_at") or batch[-1].get("created_at")
        print(f"page {pages}: {len(batch)} rows, {new} new, through {last}", flush=True)
        # Advance strictly, else the same page repeats forever.
        if new == 0 or last is None or last == cursor:
            break
        cursor = last

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(seen.values(), key=lambda r: str(r.get("submitted_at") or ""))
    with out.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} orders -> {out}")


if __name__ == "__main__":
    main()
