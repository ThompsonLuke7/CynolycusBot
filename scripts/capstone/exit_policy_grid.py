"""
Meta ranker exit-policy grid → research/capstone/exit_policy_grid.csv (committed).

Why this exists
---------------
`signals/meta_context/meta_ranker/backtest_exits.py` reads its scored input from
`/tmp/meta_scored.parquet`, which does not survive a reboot — so the numbers
behind fig11 could not be regenerated without first re-running
`build_meta_scored_from_oof.py`. This script runs the full policy grid ONCE and
persists the results as a small committed CSV that the figures read, so the
figure set is reproducible from the repo alone.

What it measures
----------------
Entries: top-10 by clean walk-forward OOF `s_combo`, holdout 2025-07-01+.
Paths:   **SHARES ONLY** — 4H stock close/high/low from Data/shared/bars/4h.
         There is no option-premium path anywhere in this harness. The live
         ExecPolicy's stop/trail ride `gain`, which IS premium for option
         positions, so a 50% stop that almost never binds on a stock binds
         constantly on a decaying OTM call. Every stop/trail number here is
         therefore a LOWER bound on how often those rules fire live.

Grid: the three policies the paper cites, the deployed live policy (stop 50% +
trail 35%, wired 2026-07-12), stop/trail sensitivity, and the scale-out
fraction x target grid.

Usage:
  PYTHONPATH=. .venv/bin/python scripts/capstone/exit_policy_grid.py
  PYTHONPATH=. .venv/bin/python scripts/capstone/exit_policy_grid.py --quick   # baselines only
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "research" / "capstone" / "exit_policy_grid.csv"
SCORED = Path("/tmp/meta_scored.parquet")

sys.path.insert(0, str(REPO / "signals/meta_context/meta_ranker"))


def build_grid(quick: bool = False) -> list[tuple[str, str, dict]]:
    """(group, policy_name, simulate kwargs)."""
    g: list[tuple[str, str, dict]] = [
        ("baseline", "rank drop-out g=0 (old live)", dict(grace=0)),
        ("baseline", "target +20% full exit", dict(target=0.20, scale_frac=1.0, grace=None)),
        ("baseline", "scale 50%@+20% + horizon25", dict(target=0.20, scale_frac=0.5, horizon=25, grace=None)),
        ("deployed", "LIVE: stop50 + trail35 + scale50@+20 + hz25",
         dict(stop=0.50, trail=0.35, target=0.20, scale_frac=0.5, horizon=25, grace=None)),
    ]
    if quick:
        return g
    # stop / trail sensitivity on top of the deployed shape
    for stop in (0.15, 0.25, 0.50):
        g.append(("stop_sens", f"stop {int(stop*100)}% + scale50@+20 + hz25",
                  dict(stop=stop, target=0.20, scale_frac=0.5, horizon=25, grace=None)))
    for trail in (0.15, 0.25, 0.35):
        g.append(("trail_sens", f"trail {int(trail*100)}% + scale50@+20 + hz25",
                  dict(trail=trail, target=0.20, scale_frac=0.5, horizon=25, grace=None)))
    # scale-out fraction x target grid (no stop, ride remainder to horizon 25)
    for frac in (0.25, 0.50, 0.75, 1.00):
        for tgt in (0.10, 0.15, 0.20, 0.30, 0.50):
            g.append(("scaleout_grid", f"scale {int(frac*100)}%@+{int(tgt*100)}% + horizon25",
                      dict(target=tgt, scale_frac=frac, horizon=25, grace=None)))
    return g


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="baselines + deployed only")
    args = ap.parse_args()

    if not SCORED.exists():
        raise SystemExit(
            f"{SCORED} missing — regenerate first:\n"
            "  PYTHONPATH=. .venv/bin/python scripts/capstone/build_meta_scored_from_oof.py"
        )
    from backtest_exits import HOLDOUT, _load_member, simulate  # noqa: E402

    member = _load_member()
    print(f"holdout {HOLDOUT.date()}+  ticker-bars={len(member):,}  tickers={member.ticker.nunique()}")
    print("price paths: SHARES (4H stock OHLC) — no option premium in this harness\n")

    rows = []
    for group, name, kw in build_grid(args.quick):
        s = simulate(member, **kw)
        if not s:
            continue
        s.update(group=group, policy=name,
                 stop=kw.get("stop"), trail=kw.get("trail"),
                 target=kw.get("target"), scale_frac=kw.get("scale_frac"),
                 horizon=kw.get("horizon"), grace=kw.get("grace"))
        rows.append(s)
        print(f"  [{group:13s}] {name:46s} n={s['n']:5d} mean={s['mean']*100:+6.2f}% "
              f"med={s['median']*100:+6.2f}% win={s['win']*100:4.1f}% hold={s['avg_hold']:5.1f}")

    df = pd.DataFrame(rows)[
        ["group", "policy", "n", "mean", "median", "win", "std", "ret_std", "avg_hold",
         "ret_per_bar", "stop", "trail", "target", "scale_frac", "horizon", "grace"]
    ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"\nwrote {OUT.relative_to(REPO)}  ({len(df)} policies)")


if __name__ == "__main__":
    main()
