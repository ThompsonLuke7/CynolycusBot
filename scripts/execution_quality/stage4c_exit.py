"""Stage 4C — exit policy: giveback vs prematurity, by module and exit reason
(pre-registered C1-C5)."""
from __future__ import annotations

import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
DATA = REPO_ROOT / "research/execution_quality/data"
SPLIT = "2026-08-15"


def med(v):
    v = [x for x in v if x is not None and isinstance(x, (int, float)) and math.isfinite(x)]
    return float(np.median(v)) if v else float("nan")


def reason_of(r):
    rs = r.get("ledger_exit_reasons") or []
    if not rs:
        return "unknown"
    t = str(rs[0] or "")
    if t.startswith("take_profit"):
        return "take_profit"
    if t.startswith("stop") or "stop" in t:
        return "stop"
    if t.startswith("trail"):
        return "trail"
    for k in ("horizon", "expiry", "expiring", "dropped", "underlying"):
        if t.startswith(k):
            return k
    return t.split("_")[0] or "unknown"


def main() -> None:
    rows = [json.loads(l) for l in (DATA / "stage3_trade_metrics.jsonl").open()]
    rows = [r for r in rows if r.get("module") and r.get("giveback_atr") is not None]

    print("=" * 104)
    print("C1/C2 — per module: what the position reached, what we kept, what it did after we left")
    print("=" * 104)
    print(f"{'module':24s} {'n':>4s} {'MFE':>7s} {'MAE':>7s} {'realized':>9s} "
          f"{'giveback':>9s} {'eff':>7s} {'prem1d':>7s} {'prem3d':>7s} {'prem10d':>8s}")
    for m in sorted({r["module"] for r in rows}):
        s = [r for r in rows if r["module"] == m]
        if len(s) < 20:
            print(f"{m:24s} {len(s):4d}   underpowered (n<20)")
            continue
        print(f"{m:24s} {len(s):4d} {med([r['mfe_hold_atr'] for r in s]):7.3f} "
              f"{med([r['mae_hold_atr'] for r in s]):7.3f} "
              f"{med([r.get('realized_move_atr') for r in s]):+9.3f} "
              f"{med([r['giveback_atr'] for r in s]):9.3f} "
              f"{med([r.get('hold_efficiency') for r in s]):+7.2f} "
              f"{med([r.get('prematurity_1d_atr') for r in s]):7.3f} "
              f"{med([r.get('prematurity_3d_atr') for r in s]):7.3f} "
              f"{med([r.get('prematurity_10d_atr') for r in s]):8.3f}")

    print("\n" + "=" * 104)
    print("C4 — by exit reason (a stop and a take-profit want opposite fixes)")
    print("=" * 104)
    print(f"{'exit reason':16s} {'n':>4s} {'MFE':>7s} {'realized':>9s} {'giveback':>9s} "
          f"{'prem1d':>7s} {'prem3d':>7s} {'prem10d':>8s} {'verdict':>28s}")
    for reason, _ in Counter(reason_of(r) for r in rows).most_common():
        s = [r for r in rows if reason_of(r) == reason]
        if len(s) < 20:
            print(f"{reason:16s} {len(s):4d}   underpowered (n<20)")
            continue
        gb = med([r["giveback_atr"] for r in s])
        p3 = med([r.get("prematurity_3d_atr") for r in s])
        verdict = ("too tight — widen" if p3 > gb * 1.5 else
                   "too loose — tighten/trail" if gb > p3 * 1.5 else
                   "balanced at 3d")
        print(f"{reason:16s} {len(s):4d} {med([r['mfe_hold_atr'] for r in s]):7.3f} "
              f"{med([r.get('realized_move_atr') for r in s]):+9.3f} {gb:9.3f} "
              f"{med([r.get('prematurity_1d_atr') for r in s]):7.3f} {p3:7.3f} "
              f"{med([r.get('prematurity_10d_atr') for r in s]):8.3f} {verdict:>28s}")

    print("\n" + "=" * 104)
    print("Holdout check on the pooled direction (giveback vs prematurity_3d)")
    print("=" * 104)
    for split in ("explore", "holdout"):
        s = [r for r in rows if (r["first_entry_fill"][:10] < SPLIT) == (split == "explore")]
        gb = med([r["giveback_atr"] for r in s])
        p3 = med([r.get("prematurity_3d_atr") for r in s])
        print(f"  {split:8s} n={len(s):4d}  giveback={gb:.3f}  prematurity_3d={p3:.3f}  "
              f"ratio={p3/gb if gb else float('nan'):.2f}  "
              f"realized={med([r.get('realized_move_atr') for r in s]):+.3f}")

    print("\n" + "=" * 104)
    print("Time-to-peak vs hold: is the exit even in the right neighbourhood?")
    print("=" * 104)
    for m in sorted({r["module"] for r in rows}):
        s = [r for r in rows if r["module"] == m and r.get("time_to_peak_min") is not None]
        if len(s) < 20:
            continue
        frac = [r["time_to_peak_min"] / r["hold_minutes"]
                for r in s if r.get("hold_minutes")]
        print(f"  {m:24s} n={len(s):4d} median time-to-peak={med([r['time_to_peak_min'] for r in s]):9.0f}m "
              f"median hold={med([r.get('hold_minutes') for r in s]):9.0f}m  "
              f"peak at {med(frac):.0%} of the hold  "
              f"peak in first 10% of hold: {np.mean([1.0 if f < 0.1 else 0.0 for f in frac]):.0%}")


if __name__ == "__main__":
    main()
