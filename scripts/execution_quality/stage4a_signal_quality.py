"""Stage 4A — does the ranking carry information? (pre-registered: 05_preregistration.md)

The traded rows cannot answer this: the order policy selected them. This runs on
the signal spine — every ranked target, traded or not — and compares each
module's top of book against two controls:

  * WITHIN-DECISION control: the same module's lower-ranked names on the SAME
    bar. Holds the day, the tape and the universe fixed, so it isolates rank.
  * DRIFT control: every ranked name pooled. In an up-tape everything rises;
    "our picks went up" is not evidence without this.

Read-only.
"""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
DATA = REPO_ROOT / "research/execution_quality/data"
SPLIT = "2026-08-15"


def load(split: str | None):
    rows = []
    for line in (DATA / "stage3_signal_metrics.jsonl").open():
        r = json.loads(line)
        if r.get("mfe_1d_atr") is None:
            continue
        d = r["available_at"][:10]
        if split == "explore" and d >= SPLIT:
            continue
        if split == "holdout" and d < SPLIT:
            continue
        rows.append(r)
    return rows


def med(v):
    v = [x for x in v if x is not None and math.isfinite(x)]
    return float(np.median(v)) if v else float("nan")


def boot_diff(a, b, n=4000, seed=7):
    """Bootstrap CI for median(a) - median(b)."""
    a = np.array([x for x in a if x is not None and math.isfinite(x)])
    b = np.array([x for x in b if x is not None and math.isfinite(x)])
    if len(a) < 5 or len(b) < 5:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    d = [np.median(rng.choice(a, len(a), True)) - np.median(rng.choice(b, len(b), True))
         for _ in range(n)]
    return float(np.median(d)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def main() -> None:
    for split in ("explore", "holdout"):
        rows = load(split)
        print(f"\n{'='*94}\nSPLIT = {split}   n = {len(rows)}\n{'='*94}")

        for h in ("1d", "3d", "10d"):
            key = f"mfe_{h}_atr"
            ret = f"ret_{h}_atr"
            print(f"\n--- horizon {h} ---")
            print(f"{'module':24s} {'n':>4s} {'top3 MFE':>9s} {'rest MFE':>9s} "
                  f"{'diff [95% CI]':>26s} {'top3 ret':>9s} {'rest ret':>9s}")
            for m in sorted({r["module"] for r in rows}):
                sub = [r for r in rows if r["module"] == m and r.get(key) is not None]
                if len(sub) < 20:
                    print(f"{m:24s} {len(sub):4d}   underpowered (n<20)")
                    continue
                # within-decision: rank 1-3 vs the rest of the SAME bar
                top, rest = [], []
                bybar = defaultdict(list)
                for r in sub:
                    bybar[r["bar"]].append(r)
                for _, g in bybar.items():
                    g = [x for x in g if x.get("rank") is not None]
                    if len(g) < 4:
                        continue
                    g.sort(key=lambda x: x["rank"])
                    top += g[:3]
                    rest += g[3:]
                if len(top) < 10 or len(rest) < 10:
                    print(f"{m:24s} {len(sub):4d}   too few paired decisions")
                    continue
                d, lo, hi = boot_diff([r[key] for r in top], [r[key] for r in rest])
                sig = "" if (math.isnan(lo) or lo * hi <= 0) else "  *"
                print(f"{m:24s} {len(sub):4d} {med([r[key] for r in top]):9.3f} "
                      f"{med([r[key] for r in rest]):9.3f} "
                      f"{d:+8.3f} [{lo:+.3f},{hi:+.3f}]{sig:>3s} "
                      f"{med([r.get(ret) for r in top]):9.3f} "
                      f"{med([r.get(ret) for r in rest]):9.3f}")

        # drift control: what did an average ranked name do?
        print(f"\n--- drift control (all ranked names pooled) ---")
        for h in ("1d", "3d", "10d"):
            k, rk = f"mfe_{h}_atr", f"ret_{h}_atr"
            print(f"  {h:4s} median MFE={med([r.get(k) for r in rows]):.3f} ATR   "
                  f"median MAE={med([r.get(f'mae_{h}_atr') for r in rows]):.3f} ATR   "
                  f"median ret={med([r.get(rk) for r in rows]):+.3f} ATR   "
                  f"share ret>0 = {np.mean([1.0 if (r.get(rk) or 0) > 0 else 0.0 for r in rows]):.1%}")

        # score decile monotonicity, pooled within module (rank-normalised)
        print(f"\n--- does SCORE order forward 3d MFE? (within-module score quintile) ---")
        for m in sorted({r["module"] for r in rows}):
            sub = [r for r in rows if r["module"] == m
                   and r.get("mfe_3d_atr") is not None and r.get("score") is not None]
            if len(sub) < 40:
                print(f"  {m:24s} underpowered ({len(sub)})")
                continue
            sub.sort(key=lambda r: r["score"])
            q = max(1, len(sub) // 5)
            cells = [sub[i * q:(i + 1) * q] for i in range(5)]
            meds = [med([r["mfe_3d_atr"] for r in c]) for c in cells]
            sp = np.corrcoef(
                [r["score"] for r in sub],
                [r["mfe_3d_atr"] for r in sub])[0, 1]
            print(f"  {m:24s} n={len(sub):4d} quintiles(low→high)="
                  f"[{', '.join(f'{x:.2f}' for x in meds)}]  pearson r={sp:+.3f}")


if __name__ == "__main__":
    main()
