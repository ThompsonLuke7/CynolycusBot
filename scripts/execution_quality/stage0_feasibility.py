"""Stage 0 — can the data answer the question at all?

Three checks, each able to kill or reshape the study (AGENTS.md: validate a
source BEFORE building a pipeline on it):

  A. Entry-fill recovery. Match each closed-trade ledger row to the broker's
     entry order(s) and report the match rate. Without T_fill there is no
     entry-timing metric.
  B. Ledger integrity. Why `entry_bar` is null on a fifth to a third of rows.
  C. (separate script) 1-minute bar coverage on the traded names.

Read-only. Consumes the cached order history; touches no live endpoint.
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MODULES = ["momentum_expansion", "multi_ticker_swing_htf", "meta_ranker", "dealer_ranker"]
ORDERS = REPO_ROOT / "research/execution_quality/data/order_history.jsonl"


def P(s):
    if not s:
        return None
    s = str(s).replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        return None


def load_orders():
    rows = [json.loads(l) for l in ORDERS.open() if l.strip()]
    for r in rows:
        r["_sub"] = P(r.get("submitted_at") or r.get("created_at"))
        r["_fill"] = P(r.get("filled_at"))
    return rows


def main() -> None:
    orders = load_orders()
    by_symbol = defaultdict(list)
    for o in orders:
        by_symbol[str(o.get("symbol"))].append(o)
    for v in by_symbol.values():
        v.sort(key=lambda r: r["_sub"] or datetime.min)
    by_id = {str(o["id"]): o for o in orders}

    earliest = min(o["_sub"] for o in orders if o["_sub"])
    print(f"order history: {len(orders)} orders, earliest {earliest.isoformat()}")
    print("NOTE: retention floor is the earliest order above, not the --after asked for.\n")

    overall = Counter()
    matched_rows = []
    for m in MODULES:
        path = REPO_ROOT / f"Data/inference/{m}/closed_trades.jsonl"
        led = [json.loads(l) for l in path.open() if l.strip()]
        c = Counter()
        for row in led:
            c["rows"] += 1
            sym = row.get("order_symbol")
            exit_ts = P(row.get("ts"))
            exit_oid = str(row.get("order_id") or "")
            if not row.get("entry_bar"):
                c["null_entry_bar"] += 1

            exit_order = by_id.get(exit_oid)
            if exit_order is None:
                c["exit_order_not_in_history"] += 1

            # Entry candidates: filled buys of this symbol before the exit.
            cands = [
                o for o in by_symbol.get(str(sym), [])
                if o.get("side") == "buy" and o.get("status") == "filled"
                and o["_fill"] is not None
                and (exit_ts is None or o["_fill"] <= exit_ts + timedelta(minutes=5))
            ]
            if not cands:
                c["no_entry_fill_found"] += 1
                continue

            # An entry is a ladder: consecutive rungs inside a short window.
            last = cands[-1]["_fill"]
            group = [o for o in cands if last - o["_fill"] <= timedelta(minutes=30)]
            qty = sum(float(o.get("filled_qty") or 0) for o in group)
            notional = sum(
                float(o.get("filled_qty") or 0) * float(o.get("filled_avg_price") or 0)
                for o in group
            )
            vwap = notional / qty if qty else float("nan")
            led_px = row.get("entry_avg_price")
            ok_px = (
                led_px is not None and qty > 0
                and abs(vwap - float(led_px)) <= max(0.02, 0.02 * abs(float(led_px)))
            )
            c["entry_fill_found"] += 1
            c["price_agrees" if ok_px else "price_disagrees"] += 1
            matched_rows.append({
                "module": m,
                "ticker": row.get("ticker"),
                "order_symbol": sym,
                "route": row.get("route"),
                "exit_reason": row.get("exit_reason"),
                "ledger_entry_bar": row.get("entry_bar"),
                "ledger_entry_avg_price": led_px,
                "entry_submitted_at": min(o["_sub"] for o in group).isoformat(),
                "entry_filled_at": max(o["_fill"] for o in group).isoformat(),
                "entry_fill_vwap": vwap,
                "entry_qty": qty,
                "entry_rungs": len(group),
                "price_agrees": ok_px,
                "exit_ts": row.get("ts"),
                "exit_order_id": exit_oid,
                "exit_filled_at": (exit_order or {}).get("filled_at"),
                "exit_submitted_at": (exit_order or {}).get("submitted_at"),
                "exit_fill_price": row.get("exit_fill_price"),
                "realized_pnl": row.get("realized_pnl"),
            })
        overall.update(c)
        n = c["rows"]
        print(f"{m:24s} rows={n:3d}  entry_fill_found={c['entry_fill_found']:3d} "
              f"({c['entry_fill_found']/n:5.1%})  price_agrees={c['price_agrees']:3d}  "
              f"no_fill={c['no_entry_fill_found']:2d}  null_entry_bar={c['null_entry_bar']:2d}  "
              f"exit_oid_missing={c['exit_order_not_in_history']:2d}")

    n = overall["rows"]
    print(f"\nPOOLED rows={n} entry_fill_found={overall['entry_fill_found']} "
          f"({overall['entry_fill_found']/n:.1%}) price_agrees={overall['price_agrees']} "
          f"({overall['price_agrees']/n:.1%})")

    out = REPO_ROOT / "research/execution_quality/data/stage0_entry_match.jsonl"
    with out.open("w", encoding="utf-8") as fh:
        for r in matched_rows:
            fh.write(json.dumps(r) + "\n")
    print(f"wrote {len(matched_rows)} matched rows -> {out}")


if __name__ == "__main__":
    main()
