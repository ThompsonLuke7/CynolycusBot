"""Stage 5b — profit protection, the rule the giveback evidence actually implies.

Stage 5a killed the obvious reading of Stage 4C4. A stopped trade does travel far
past its peak, but tightening the INITIAL stop is worse at every level tested:
across all closed lifecycles the mean and win rate fall monotonically as the stop
tightens, because the same tight stop that cuts a loser also cuts the winners.
Stage 4C4 looked otherwise only because it conditioned on trades that actually
stopped, which is selection on the outcome.

What the evidence does support is different. Median MFE during the hold is
0.35 ATR and median realised is -0.01 ATR: positions reach a profit and round-trip
it. That asks for a rule that does nothing until a position IS profitable, and
then protects the gain -- leaving the initial risk untouched.

Rule tested: once MFE >= ARM (in ATR), exit if price retraces GIVEBACK of the
peak gain. Replayed on the 1-minute underlying path; can only ever exit EARLIER
than the actual exit. Underlying only.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts/execution_quality"))
from stage3_metrics import P, window  # noqa: E402

DATA = REPO_ROOT / "research/execution_quality/data"
SPLIT = "2026-08-15"


def med(v):
    v = [x for x in v if x is not None and isinstance(x, (int, float)) and math.isfinite(x)]
    return float(np.median(v)) if v else float("nan")


def simulate(r, arm, give):
    """Realised underlying move in ATR under an armed give-back rule."""
    t = r["ticker"]
    fill, exit_t = P(r["first_entry_fill"]), P(r.get("exit_last_fill"))
    a, p_fill = r.get("atr"), r.get("u_fill")
    if not (exit_t and a and math.isfinite(a) and p_fill and math.isfinite(p_fill)):
        return None
    actual = r.get("realized_move_atr")
    if arm is None:
        return actual
    w = window(t, fill, exit_t)
    if w is None or len(w) < 2:
        return actual
    sign = -1 if str(r.get("signal_side", "long")).lower() in ("short", "sell") else 1
    hi = (w["high"].to_numpy(dtype=float) if sign > 0 else -w["low"].to_numpy(dtype=float))
    lo = (w["low"].to_numpy(dtype=float) if sign > 0 else -w["high"].to_numpy(dtype=float))
    base = p_fill * sign
    peak = -1e18
    for i in range(len(hi)):
        peak = max(peak, hi[i])
        gain = (peak - base) / a
        if gain >= arm:
            trigger = peak - give * (peak - base)
            if lo[i] <= trigger:
                return float((trigger - base) / a)
    return actual


def main() -> None:
    rows = [json.loads(l) for l in (DATA / "stage3_trade_metrics.jsonl").open()]
    rows = [r for r in rows if r.get("module") and r.get("realized_move_atr") is not None]
    grid = [(None, None)] + [(a, g) for a in (0.5, 1.0, 1.5) for g in (0.25, 0.4, 0.5)]

    for split in ("explore", "holdout"):
        sel = [r for r in rows if (r["first_entry_fill"][:10] < SPLIT) == (split == "explore")]
        print("=" * 88)
        print(f"SPLIT = {split}   n = {len(sel)}   (underlying move, ATR)")
        print("=" * 88)
        print(f"{'arm':>6s} {'giveback':>9s} {'median':>9s} {'mean':>9s} "
              f"{'win rate':>9s} {'p90':>8s} {'triggered':>10s}")
        for arm, give in grid:
            vals, trig = [], 0
            for r in sel:
                v = simulate(r, arm, give)
                if v is None or not math.isfinite(v):
                    continue
                if arm is not None and abs(v - (r.get("realized_move_atr") or 0)) > 1e-9:
                    trig += 1
                vals.append(v)
            if not vals:
                continue
            lbl = ("actual", "") if arm is None else (f"{arm:.2f}", f"{give:.0%}")
            print(f"{lbl[0]:>6s} {lbl[1]:>9s} {med(vals):9.3f} {float(np.mean(vals)):9.3f} "
                  f"{np.mean([1.0 if v > 0 else 0.0 for v in vals]):9.1%} "
                  f"{float(np.percentile(vals, 90)):8.3f} {trig/len(vals):10.0%}")
        print()

    print("=" * 88)
    print("Best candidate (arm 1.0 ATR, give back 40%) per module, all closed")
    print("=" * 88)
    print(f"{'module':24s} {'n':>4s} {'actual med':>11s} {'rule med':>9s} "
          f"{'actual mean':>12s} {'rule mean':>10s} {'triggered':>10s}")
    for m in sorted({r["module"] for r in rows}):
        sel = [r for r in rows if r["module"] == m]
        if len(sel) < 20:
            continue
        act = [r["realized_move_atr"] for r in sel]
        new, trig = [], 0
        for r in sel:
            v = simulate(r, 1.0, 0.4)
            if v is None or not math.isfinite(v):
                continue
            if abs(v - r["realized_move_atr"]) > 1e-9:
                trig += 1
            new.append(v)
        print(f"{m:24s} {len(sel):4d} {med(act):11.3f} {med(new):9.3f} "
              f"{float(np.mean(act)):12.3f} {float(np.mean(new)):10.3f} {trig/len(sel):10.0%}")


if __name__ == "__main__":
    main()
