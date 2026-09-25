"""Step 2b -- sweep the live exit policy's two free parameters on real underlying paths.

Step 2 established the levels: on momentum's top-3 the SUPERSEDED config returned 0.091R,
the ACTUAL live policy returns 0.416R, and simply holding 30 sessions returns 0.662R. So the
change the repo shipped was worth ~+0.33R/trade and there is still ~-0.25R/trade between the
live policy and no stop at all.

This sweeps the two knobs that gap turns on, holding everything else at the live values
(no trail, +30% trim of 16%):

    underlying_stop_atr  in {1.0 .. 4.0, and OFF}
    horizon_bars         in {10 .. 40 sessions}

Paths are the cached Stage 3 entries (equity, real bars, R units, 20bp round trip). Reported
with the full distribution, because a mean alone hides exactly what a stop does to the tail.

SCOPE, stated because it decides how far this travels: this is the EQUITY path. On an option
the stop also prevents riding a contract toward zero, which a share path cannot represent, and
the option premium path does not exist historically (2026-07 retraction). Treat this as the
answer for the equity sleeve and as the DIRECTIONAL input for options.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "research/rl_reintroduction_2026-09-23"))

import panel as P  # noqa: E402

PATHS = REPO / "research/rl_reintroduction_2026-09-23/data/stage3_paths.parquet"
COST_RT = 0.002
TAKE_PROFIT = 0.30
SCALE_FRAC = 0.16
STOPS = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, None]      # None = no stop
HORIZONS = [10, 15, 20, 27, 30, 40]
LIVE = (1.5, 27)


def run(paths: pd.DataFrame, stop_atr: float | None, horizon: int) -> pd.DataFrame:
    out = []
    for eid, g in paths.groupby("eid", sort=False):
        g = g.sort_values("offset")
        rc = g["r_close"].to_numpy(float)
        rh = g["r_high"].to_numpy(float)
        rl = g["r_low"].to_numpy(float)
        n = len(rc)
        if n < 2:
            continue
        atr = float(g["atr_pct"].iloc[0])
        cst = COST_RT / (P.K_RISK * atr)
        stop_level = (-stop_atr / P.K_RISK) if stop_atr is not None else -np.inf
        tp_level = TAKE_PROFIT / (P.K_RISK * atr)
        trimmed, exit_r, exit_day = None, None, None
        for k in range(1, n):
            if rl[k] <= stop_level:
                exit_r, exit_day = stop_level, k
                break
            if trimmed is None and rh[k] >= tp_level:
                trimmed = tp_level
            if k >= horizon:
                exit_r, exit_day = rc[k], k
                break
        if exit_r is None:
            exit_r, exit_day = rc[n - 1], n - 1
        r = (SCALE_FRAC * trimmed + (1 - SCALE_FRAC) * exit_r) if trimmed is not None else exit_r
        out.append({"eid": eid, "arm": g["arm"].iloc[0], "R": r - cst, "day": exit_day,
                    "stopped": bool(rl[1:n].min() <= stop_level) if stop_atr else False})
    return pd.DataFrame(out)


def main() -> None:
    if not PATHS.exists():
        raise SystemExit("run research/rl_reintroduction_2026-09-23/03_exit_oracle.py first")
    paths = pd.read_parquet(PATHS)
    print(f"paths {len(paths):,} rows over {paths['eid'].nunique():,} entries")
    print("HOLDING FIXED at the live values: no trail, take_profit 0.30, scale_frac 0.16.\n")

    out: dict = {"live": {"stop_atr": LIVE[0], "horizon": LIVE[1]}}
    grid = []
    for s in STOPS:
        for h in HORIZONS:
            res = run(paths, s, h)
            for arm, g in res.groupby("arm"):
                r = g["R"].to_numpy(float)
                grid.append({"stop_atr": ("off" if s is None else s), "horizon": h, "arm": arm,
                             "n": len(r), "mean_R": float(r.mean()),
                             "median_R": float(np.median(r)),
                             "p5_R": float(np.percentile(r, 5)),
                             "p95_R": float(np.percentile(r, 95)),
                             "share_gt_3R": float((r >= 3).mean()),
                             "share_lt_neg1R": float((r <= -1).mean()),
                             "stop_rate": float(g["stopped"].mean()),
                             "median_day": float(g["day"].median())})
        print(f"  swept stop={s}", flush=True)
    gd = pd.DataFrame(grid)
    out["grid"] = gd.to_dict("records")

    for arm in ("mom_score", "htf_score"):
        a = gd[gd["arm"] == arm]
        print("\n" + "=" * 78 + f"\n{arm}: MEAN R by stop width x horizon\n" + "=" * 78)
        print(a.pivot_table(index="stop_atr", columns="horizon", values="mean_R",
                            sort=False).round(3).to_string())
        print(f"\n{arm}: share of trades reaching +3R")
        print(a.pivot_table(index="stop_atr", columns="horizon", values="share_gt_3R",
                            sort=False).round(4).to_string())
        print(f"\n{arm}: share of trades worse than -1R")
        print(a.pivot_table(index="stop_atr", columns="horizon", values="share_lt_neg1R",
                            sort=False).round(4).to_string())
        live = a[(a["stop_atr"] == LIVE[0]) & (a["horizon"] == LIVE[1])]
        if len(live):
            lv = live.iloc[0]
            best = a.loc[a["mean_R"].idxmax()]
            print(f"\n  LIVE cell (stop {LIVE[0]} ATR, horizon {LIVE[1]}): "
                  f"mean {lv['mean_R']:.3f}R, +3R {lv['share_gt_3R']:.1%}, "
                  f"<-1R {lv['share_lt_neg1R']:.1%}, stop rate {lv['stop_rate']:.1%}")
            print(f"  BEST cell (stop {best['stop_atr']}, horizon {int(best['horizon'])}): "
                  f"mean {best['mean_R']:.3f}R, +3R {best['share_gt_3R']:.1%}, "
                  f"<-1R {best['share_lt_neg1R']:.1%}")
            print(f"  difference: {best['mean_R'] - lv['mean_R']:+.3f}R per trade, "
                  f"bought with {best['share_lt_neg1R'] - lv['share_lt_neg1R']:+.1%} "
                  "more trades below -1R")
            out[f"{arm}_live"] = lv.to_dict()
            out[f"{arm}_best"] = best.to_dict()

    (HERE / "data").mkdir(exist_ok=True)
    (HERE / "data" / "stop_width_sweep.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"\nwrote {HERE / 'data' / 'stop_width_sweep.json'}")


if __name__ == "__main__":
    main()
