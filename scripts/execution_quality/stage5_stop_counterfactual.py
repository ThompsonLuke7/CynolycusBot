"""Stage 5 — what a tighter stop would have done. Counterfactual, not a change.

Stage 4C4: a stopped position reaches +0.32 ATR, travels to -0.55 ATR before the
stop fires, and recovers only 0.30 ATR in the next three days. That says "too
loose" but not "how much tighter", which is what a decision needs.

Method. Replay every CLOSED lifecycle on its own 1-minute underlying path and
exit at whichever comes first: a stop k*ATR below the entry, or the exit that
actually happened. The counterfactual can therefore only ever exit EARLIER, and
is applied to ALL positions rather than only the ones that really stopped —
applying it to the stopped subset alone would select on the outcome.

Underlying only. Option P&L is levered off this and is not the timing metric
(retraction rule).
"""
from __future__ import annotations

import json
import math
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts/execution_quality"))
from stage3_metrics import P, window  # noqa: E402

DATA = REPO_ROOT / "research/execution_quality/data"
SPLIT = "2026-08-15"
GRID = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0, None]  # None = the policy as it stands


def med(v):
    v = [x for x in v if x is not None and isinstance(x, (int, float)) and math.isfinite(x)]
    return float(np.median(v)) if v else float("nan")


def simulate(r, k):
    """Realised underlying move in ATR under a k*ATR stop; k=None is actual."""
    t = r["ticker"]
    fill, exit_t = P(r["first_entry_fill"]), P(r.get("exit_last_fill"))
    a, p_fill = r.get("atr"), r.get("u_fill")
    if not (exit_t and a and math.isfinite(a) and p_fill and math.isfinite(p_fill)):
        return None
    sign = -1 if str(r.get("signal_side", "long")).lower() in ("short", "sell") else 1
    if k is None:
        return r.get("realized_move_atr")
    w = window(t, fill, exit_t)
    if w is None or len(w) < 2:
        return r.get("realized_move_atr")
    level = p_fill - sign * k * a
    hit = (w["low"] <= level) if sign > 0 else (w["high"] >= level)
    if bool(hit.any()):
        return -k  # stopped: the loss is the stop distance (fills at the level)
    return r.get("realized_move_atr")


def main() -> None:
    rows = [json.loads(l) for l in (DATA / "stage3_trade_metrics.jsonl").open()]
    rows = [r for r in rows if r.get("module") and r.get("realized_move_atr") is not None]
    print(f"closed lifecycles replayed: {len(rows)}\n")

    for split in ("explore", "holdout", "all"):
        sel = ([r for r in rows if (r["first_entry_fill"][:10] < SPLIT) == (split == "explore")]
               if split != "all" else rows)
        print("=" * 92)
        print(f"SPLIT = {split}   n = {len(sel)}   (underlying move in ATR units)")
        print("=" * 92)
        print(f"{'stop':>8s} {'median':>9s} {'mean':>9s} {'win rate':>9s} "
              f"{'p10':>8s} {'p90':>8s} {'stopped':>8s}")
        for k in GRID:
            vals, stopped = [], 0
            for r in sel:
                v = simulate(r, k)
                if v is None or not math.isfinite(v):
                    continue
                if k is not None and abs(v + k) < 1e-9:
                    stopped += 1
                vals.append(v)
            if not vals:
                continue
            label = "actual" if k is None else f"{k:.2f} ATR"
            print(f"{label:>8s} {med(vals):9.3f} {float(np.mean(vals)):9.3f} "
                  f"{np.mean([1.0 if v > 0 else 0.0 for v in vals]):9.1%} "
                  f"{float(np.percentile(vals, 10)):8.3f} {float(np.percentile(vals, 90)):8.3f} "
                  f"{stopped/len(vals):8.0%}")
        print()

    print("=" * 92)
    print("Per module (all closed), median underlying move in ATR")
    print("=" * 92)
    hdr = "  ".join(f"{('actual' if k is None else f'{k:g}A'):>7s}" for k in GRID)
    print(f"{'module':24s} {'n':>4s}  {hdr}")
    for m in sorted({r["module"] for r in rows}):
        sel = [r for r in rows if r["module"] == m]
        if len(sel) < 20:
            print(f"{m:24s} {len(sel):4d}   underpowered (n<20)")
            continue
        cells = []
        for k in GRID:
            vals = [simulate(r, k) for r in sel]
            vals = [v for v in vals if v is not None and math.isfinite(v)]
            cells.append(f"{med(vals):7.3f}")
        print(f"{m:24s} {len(sel):4d}  " + "  ".join(cells))


if __name__ == "__main__":
    main()
