"""Stage 4B — entry timing, and the counterfactual delay grid (pre-registered B1-B4)."""
from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
DATA = REPO_ROOT / "research/execution_quality/data"
sys.path.insert(0, str(REPO_ROOT / "scripts/execution_quality"))
from stage3_metrics import P, atr_at, bars, price_at, window, WAIT_WINDOW  # noqa: E402

SPLIT = "2026-08-15"


def med(v):
    v = [x for x in v if x is not None and isinstance(x, (int, float)) and math.isfinite(x)]
    return float(np.median(v)) if v else float("nan")


def main() -> None:
    rows = [json.loads(l) for l in (DATA / "stage3_trade_metrics.jsonl").open()]
    rows = [r for r in rows if r.get("module")]

    print("=" * 100)
    print("B1/B2 — early vs late, by module (phase_error > 0 means we entered AFTER the move began)")
    print("=" * 100)
    print(f"{'module':24s} {'n':>4s} {'%late':>6s} {'phase med':>10s} "
          f"{'missed_leg(late)':>17s} {'pre_adverse(early)':>19s} {'entry_vs_oracle':>16s}")
    for m in sorted({r["module"] for r in rows}):
        s = [r for r in rows if r["module"] == m and r.get("phase_error_min") is not None]
        if len(s) < 20:
            print(f"{m:24s} {len(s):4d}   underpowered (n<20)")
            continue
        late = [r for r in s if r["phase_error_min"] >= 0]
        early = [r for r in s if r["phase_error_min"] < 0]
        print(f"{m:24s} {len(s):4d} {len(late)/len(s):6.0%} "
              f"{med([r['phase_error_min'] for r in s]):10.1f} "
              f"{med([r.get('missed_leg_atr') for r in late]):17.3f} "
              f"{med([r.get('pre_entry_adverse_atr') for r in early]):19.3f} "
              f"{med([r.get('entry_vs_oracle_atr') for r in s]):16.3f}")

    print("\n" + "=" * 100)
    print("B4 — overnight-deferred entries (afternoon 18:00Z bar) vs same-session entries")
    print("=" * 100)
    joined = [r for r in rows if r.get("signal_available_at") and r.get("signal_to_fill_min") is not None]
    defer = [r for r in joined if r["signal_to_fill_min"] > 120]
    same = [r for r in joined if r["signal_to_fill_min"] <= 120]
    for name, g in (("same-session", same), ("overnight-deferred", defer)):
        cl = [r for r in g if r.get("realized_move_atr") is not None]
        print(f"  {name:20s} n={len(g):4d}  entry_vs_oracle={med([r.get('entry_vs_oracle_atr') for r in g]):+.3f}  "
              f"entry_slip={med([r.get('entry_slip_atr') for r in g]):+.3f}  "
              f"mfe_hold={med([r.get('mfe_hold_atr') for r in g]):.3f}  "
              f"realized={med([r.get('realized_move_atr') for r in cl]):+.3f} (n={len(cl)})")

    print("\n" + "=" * 100)
    print("B3 — counterfactual delay grid: what if the entry had waited N trading minutes?")
    print("     Re-prices every joined entry off the SAME availability stamp. Underlying only.")
    print("=" * 100)
    grid = [0, 5, 15, 30, 60, 120, 240]
    for split in ("explore", "holdout"):
        sel = [r for r in joined
               if (r["first_entry_fill"][:10] < SPLIT) == (split == "explore")]
        print(f"\n  split={split}  n={len(sel)}")
        print(f"  {'delay':>7s} {'fill vs oracle (ATR)':>21s} {'vs actual':>10s} "
              f"{'fwd 1d MFE from fill':>21s} {'unfilled':>9s}")
        # One COMMON subset across every delay: a grid where each row is priced
        # off whichever delays happen to have data is not a comparison.
        priced: dict[int, dict[int, tuple[float, float]]] = {g: {} for g in grid}
        for i, r in enumerate(sel):
            t, avail = r["ticker"], P(r["signal_available_at"])
            sign = -1 if str(r.get("signal_side", "long")).lower() in ("short", "sell") else 1
            a = r.get("atr")
            o = r.get("oracle_entry_px")
            if not (a and math.isfinite(a)) or not (o and math.isfinite(o)):
                continue
            # Generous wall-clock span: N trading minutes can straddle sessions,
            # so a fixed multiple of N is not enough at the 14:00 ET decision.
            w = window(t, avail, avail + timedelta(minutes=max(grid) * 6 + 6000))
            if w is None:
                continue
            for g in grid:
                if len(w) <= g:
                    continue
                px = float(w["close"].iloc[g])
                start = w.index[g].to_pydatetime()
                wf = window(t, start, start + timedelta(minutes=390 * 6 + 3000))
                if wf is None or len(wf) < 30:
                    continue
                wf = wf.iloc[:390]
                fw = ((float(wf["high"].max()) - px) if sign > 0
                      else (px - float(wf["low"].min()))) / a
                priced[g][i] = ((px - o) * sign / a, fw)
        common = set.intersection(*[set(priced[g]) for g in grid]) if grid else set()
        print(f"  common subset across all delays: n={len(common)}")
        base = None
        for g in grid:
            vs = [priced[g][i][0] for i in common]
            fw = [priced[g][i][1] for i in common]
            v = med(vs)
            if base is None:
                base = v
            print(f"  {g:5d}m {v:21.3f} {v - base:+10.3f} {med(fw):21.3f} "
                  f"{len(sel) - len(common):9d}")


if __name__ == "__main__":
    main()
