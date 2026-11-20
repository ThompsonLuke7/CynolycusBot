"""Draw a fixed-seed audit sample for hand-labelling the rule extractor.

Writes <out>/audit_calls.jsonl (N per tier, to estimate precision) and
<out>/audit_misses.jsonl (ticker-tagged posts WITH a fetched body that produced no
call, to estimate what the rules miss). Labels are added by hand afterwards:
  calls:  label = correct | wrong_direction | not_a_call | wrong_ticker
  misses: label = no_call | missed_call
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from scripts.afterhour.collect import read_jsonl, write_jsonl


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--extract-dir", type=Path, required=True, help="dir holding calls.jsonl and posts.jsonl")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--per-tier", type=int, default=20)
    ap.add_argument("--misses", type=int, default=40)
    ap.add_argument("--seed", type=int, default=20261116)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    calls = read_jsonl(args.extract_dir / "calls.jsonl")
    posts = read_jsonl(args.extract_dir / "posts.jsonl")
    sample = []
    for tier in ("A", "B", "C"):
        pool = [c for c in calls if c["tier"] == tier]
        sample += rng.sample(pool, min(args.per_tier, len(pool)))
    with_call = {c["post_id"] for c in calls}
    pool = [p for p in posts if p["has_detail"] and p["tickers"] and p["post_id"] not in with_call]
    misses = rng.sample(pool, min(args.misses, len(pool)))
    args.out.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out / "audit_calls.jsonl", [{**c, "label": None} for c in sample])
    write_jsonl(args.out / "audit_misses.jsonl", [{k: p[k] for k in ("username", "post_id", "share_url", "posted_at_utc",
                                                                      "title", "body", "tickers")} | {"label": None}
                                                   for p in misses])
    print(json.dumps({"calls": len(sample), "misses": len(misses), "seed": args.seed}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
