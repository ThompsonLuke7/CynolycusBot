"""Causal fuzzy-zone range fade on 1-minute RTH bars (research sketch, not live code).

Zones are built only from swing pivots that are already confirmed (pivot bar + K
bars later), clustered with a fuzzy tolerance. A fade is taken only when an
active support zone sits below AND an active resistance zone sits above price
(a "range" is visible), price tests one edge, and the test bar closes back
inside. Entry is the NEXT bar's open; target is the near edge of the opposite
zone; stop is beyond the far edge of the tested zone; flat at 15:55 ET.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Params:
    k: int = 5                 # pivot half-window (bars); pivot known at i + k
    tol_pct: float = 0.0012    # pivots within this fraction are the same zone
    pad_pct: float = 0.0003    # zone edge padding
    min_touches: int = 2
    stop_pct: float = 0.0010   # stop beyond far edge of tested zone
    min_width_cost_mult: float = 3.0
    max_width_pct: float = 0.015
    cost_bps_rt: float = 12.0  # module default: 8 spread + 4 slippage, round trip
    start: str = "10:00"
    last_entry: str = "15:30"
    flat: str = "15:55"


def load_jsonl(path: str, symbol: str) -> pd.DataFrame:
    rows = [json.loads(l) for l in open(path) if f'"{symbol}"' in l]
    df = pd.DataFrame(rows)
    df = df[df.symbol == symbol]
    df["t"] = pd.to_datetime(df.timestamp).dt.tz_convert("America/New_York")
    return df.drop_duplicates("t").sort_values("t").set_index("t")[["open", "high", "low", "close", "volume"]]


def zones_from(pivots: list[float], p: Params, ref: float) -> list[tuple[float, float, int]]:
    if not pivots:
        return []
    out = []
    srt = sorted(pivots)
    grp = [srt[0]]
    for x in srt[1:]:
        if x - grp[-1] <= ref * p.tol_pct:
            grp.append(x)
        else:
            out.append(grp)
            grp = [x]
    out.append(grp)
    pad = ref * p.pad_pct
    return [(min(g) - pad, max(g) + pad, len(g)) for g in out if len(g) >= p.min_touches]


def run_day(day: pd.DataFrame, p: Params) -> list[dict]:
    day = day.between_time("09:30", "15:59")
    h, l, o, c = (day[x].to_numpy() for x in ("high", "low", "open", "close"))
    idx = day.index
    n = len(day)
    known: list[float] = []
    trades: list[dict] = []
    pos = None
    t_start, t_last, t_flat = (pd.Timestamp(x).time() for x in (p.start, p.last_entry, p.flat))
    for i in range(n):
        j = i - p.k  # pivot candidate now confirmable
        if j - p.k >= 0:
            if h[j] == h[j - p.k:j + p.k + 1].max():
                known.append(h[j])
            if l[j] == l[j - p.k:j + p.k + 1].min():
                known.append(l[j])
        now = idx[i].time()
        if pos is not None:
            exit_px = None
            if pos["side"] == 1:
                if l[i] <= pos["stop"]:
                    exit_px, why = min(o[i], pos["stop"]), "stop"
                elif h[i] >= pos["target"]:
                    exit_px, why = max(o[i], pos["target"]), "target"
            else:
                if h[i] >= pos["stop"]:
                    exit_px, why = max(o[i], pos["stop"]), "stop"
                elif l[i] <= pos["target"]:
                    exit_px, why = min(o[i], pos["target"]), "target"
            if exit_px is None and now >= t_flat:
                exit_px, why = c[i], "eod"
            if exit_px is not None:
                gross = pos["side"] * (exit_px - pos["entry"]) / pos["entry"]
                trades.append({**pos, "exit_t": str(idx[i]), "exit": exit_px, "why": why,
                               "gross_bps": gross * 1e4, "net_bps": gross * 1e4 - p.cost_bps_rt})
                pos = None
            continue
        if not (t_start <= now <= t_last) or i + 1 >= n:
            continue
        zs = zones_from(known, p, c[i])
        below = [z for z in zs if z[1] <= c[i] + c[i] * p.pad_pct]
        above = [z for z in zs if z[0] >= c[i] - c[i] * p.pad_pct]
        if not below or not above:
            continue
        sup = max(below, key=lambda z: z[1])
        res = min(above, key=lambda z: z[0])
        width = (res[0] - sup[1]) / c[i]
        if width * 1e4 < p.min_width_cost_mult * p.cost_bps_rt or width > p.max_width_pct:
            continue
        side = 0
        if l[i] <= sup[1] and c[i] > sup[0] and c[i] > o[i]:
            side, stop, target = 1, sup[0] * (1 - p.stop_pct), res[0]
        elif h[i] >= res[0] and c[i] < res[1] and c[i] < o[i]:
            side, stop, target = -1, res[1] * (1 + p.stop_pct), sup[1]
        if side:
            entry = o[i + 1]
            if (side == 1 and entry >= target) or (side == -1 and entry <= target):
                continue
            pos = {"side": side, "signal_t": str(idx[i]), "entry_t": str(idx[i + 1]), "entry": entry,
                   "stop": stop, "target": target, "zone": [round(sup[0], 2), round(sup[1], 2)] if side == 1 else [round(res[0], 2), round(res[1], 2)],
                   "opp_zone": [round(res[0], 2), round(res[1], 2)] if side == 1 else [round(sup[0], 2), round(sup[1], 2)]}
            # entry bar itself is managed from the next iteration (fill happens at its open)
    return trades


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl")
    ap.add_argument("--symbol", default="QQQ")
    ap.add_argument("--cost-bps", type=float, default=12.0)
    a = ap.parse_args()
    p = Params(cost_bps_rt=a.cost_bps)
    df = load_jsonl(a.jsonl, a.symbol)
    tr = run_day(df, p)
    for t in tr:
        print(f"{t['signal_t'][11:16]} {'LONG ' if t['side']==1 else 'SHORT'} entry {t['entry']:.2f} @ {t['entry_t'][11:16]} "
              f"zone {t['zone']} -> target {t['target']:.2f} stop {t['stop']:.2f} | exit {t['exit']:.2f} @ {t['exit_t'][11:16]} {t['why']:6s} "
              f"gross {t['gross_bps']:+.1f} bps net {t['net_bps']:+.1f} bps")
    if tr:
        g = np.array([t["gross_bps"] for t in tr]); nb = np.array([t["net_bps"] for t in tr])
        print(f"n={len(tr)} wins={int((g>0).sum())} sum gross {g.sum():+.1f} bps  sum net {nb.sum():+.1f} bps (cost {p.cost_bps_rt} bps RT)")


if __name__ == "__main__":
    main()
