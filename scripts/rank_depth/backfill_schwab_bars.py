"""Fill the Alpaca OPRA hole with Schwab bars, for contracts already selected.

`rank_top3_option_paths.jsonl` recorded which contract each top-3 signal would
have bought, then failed to price 283 of them because they had not expired yet
and Alpaca refuses live contracts without the OPRA agreement. The contract
choice is already made and is not revisited here -- only the price path is
fetched, from the source that actually serves it.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from core.API.Schwab_API.option_bars import fetch_daily_bars  # noqa: E402
from core.API.Schwab_API.schwab_client import SchwabClient  # noqa: E402

DATA = REPO / "research/execution_quality/data"
SRC = DATA / "rank_top3_option_paths.jsonl"
OUT = DATA / "rank_top3_option_paths_schwab.jsonl"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sleep", type=float, default=0.15)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    todo = []
    for line in SRC.open():
        r = json.loads(line)
        if r.get("skip") or not r.get("occ"):
            continue
        if int(r.get("n_bars") or 0) > 0:
            continue                      # Alpaca already priced it
        todo.append(r)
    if args.limit:
        todo = todo[:args.limit]
    print(f"contracts Alpaca could not price: {len(todo)}")

    client = SchwabClient().client
    ok = 0
    with OUT.open("w", encoding="utf-8") as fh:
        for i, r in enumerate(todo, 1):
            bars, status = fetch_daily_bars(
                client, r["occ"], date.fromisoformat(r["signal_date"]),
                date.fromisoformat(r["expiry"]))
            r = {**r, "bars": bars, "n_bars": len(bars),
                 "bars_status": status, "bars_source": "schwab"}
            fh.write(json.dumps(r) + "\n")
            ok += int(bool(bars))
            if i % 25 == 0:
                print(f"  {i}/{len(todo)}  priced={ok}")
            time.sleep(args.sleep)
    print(f"\npriced {ok} / {len(todo)} -> {OUT}")


if __name__ == "__main__":
    main()
