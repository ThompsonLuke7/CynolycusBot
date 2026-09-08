"""Does restricting to tight option markets rescue the long-call expression?

Break-even round-trip cost was 11.9% of premium. Measured cost on the traded
universe was 16.2%. This asks whether a subset of names clears the bar, using a
much larger contract sample drawn from the SIGNAL spine (every ranked target)
rather than only the ~200 that were actually traded -- the traded set put just
3-6 contracts under a 10% spread, which cannot be bucketed.

Cost per bucket is set from each name's OWN measured spread rather than a single
pooled number, because that is the whole point of the filter. Entry pays half the
spread, exit pays half plus the measured extra slippage the 30m module shows
beyond mid (its realised exit was +12.2% against a 25.1% quoted spread, i.e.
roughly half the spread plus ~0.5pp).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts/thesis_test"))
from validate_and_simulate import shares_return, validate  # noqa: E402

DATA = REPO / "research/execution_quality/data"
LIQ = json.loads((DATA / "option_liquidity_by_name.json").read_text())


def sim(row, hold_days, prem_stop, entry_c, exit_c):
    bars = row["bars"]
    raw = float(bars[0]["c"])
    if raw <= 0:
        return None
    entry = raw * (1.0 + entry_c)
    path = bars[1:hold_days + 1]
    if not path:
        return None
    for b in path:
        if prem_stop is not None and float(b["l"]) <= raw * (1.0 - prem_stop):
            return (raw * (1.0 - prem_stop) * (1.0 - exit_c)) / entry - 1.0
    return (float(path[-1]["c"]) * (1.0 - exit_c)) / entry - 1.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", default=str(DATA / "thesis_paths_liquid.jsonl"))
    args = ap.parse_args()
    rows = [json.loads(l) for l in Path(args.paths).open() if l.strip()]
    good = []
    for r in rows:
        if not r.get("n_bars"):
            continue
        ok, _ = validate(r)
        s = LIQ.get(r["ticker"], {}).get("spread_med")
        if ok and s is not None:
            good.append({**r, "_spread": float(s)})
    print(f"entries fetched: {len(rows)}   with bars: {sum(1 for r in rows if r.get('n_bars'))}"
          f"   usable + liquidity read: {len(good)}\n")
    if not good:
        return

    sp = np.array([g["_spread"] for g in good])
    print("spread distribution of the usable contracts:")
    for q in (5, 10, 25, 50, 75):
        print(f"  p{q:<2d} {np.percentile(sp, q):5.1%}")
    print()

    buckets = [
        ("top 5%  (tightest)", np.percentile(sp, 5)),
        ("top 10%", np.percentile(sp, 10)),
        ("top 25%", np.percentile(sp, 25)),
        ("under 10% spread", 0.10),
        ("under 15% spread", 0.15),
        ("ALL", 1.0),
    ]
    holds = [5, 8, 10, 13, 15, 20]
    stops = [("-39% (live)", 0.39), ("none", None)]

    for label, thr in buckets:
        sel = [g for g in good if g["_spread"] <= thr]
        if len(sel) < 8:
            print(f"{label:22s} n={len(sel):3d}  too few to report\n")
            continue
        med_sp = float(np.median([g["_spread"] for g in sel]))
        print("=" * 92)
        print(f"{label}   n={len(sel)}   threshold {thr:.1%}   median name spread {med_sp:.1%}")
        print("=" * 92)
        print(f"{'hold':>6s} " + " ".join(f"{s:>15s}" for s, _ in stops) + f" {'SHARES':>12s}")
        for h in holds:
            cells = []
            for _, stp in stops:
                v = []
                for g in sel:
                    # per-name cost: half the spread in, half plus the measured
                    # extra slippage out
                    ec = g["_spread"] / 2.0
                    xc = g["_spread"] / 2.0 + 0.005
                    x = sim(g, h, stp, ec, xc)
                    if x is not None and np.isfinite(x):
                        v.append(x)
                v = np.array(v)
                cells.append(f"{1000 * v.sum():15,.0f}" if len(v) else f"{'-':>15s}")
            sh = [shares_return(g, h) for g in sel]
            sh = np.array([x for x in sh if x is not None and np.isfinite(x)]) - 0.0005
            cells.append(f"{1000 * sh.sum():12,.0f}" if len(sh) else f"{'-':>12s}")
            print(f"{h:5d}d " + " ".join(cells))
        print()


if __name__ == "__main__":
    main()
