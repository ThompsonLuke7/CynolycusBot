"""Where is the momentum payoff? Next-month return by 12-1 momentum rank bucket inside the
liquid-300, point-in-time universe (2026-10-04).

Explains 13: a model fitted to the whole cross-section learns little if the payoff sits only in the
extreme top ranks. For each weekly decision date 2019+, stocks are bucketed by momentum rank; the
number shown is the bucket's mean forward 21-session return minus the liquid-300 mean that date,
averaged over dates, with a 6-month block-bootstrap CI (labels overlap).

    .venv/bin/python research/long_horizon_discount_2026-09-25/14_momentum_profile.py
"""
from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
_spec = spec_from_file_location("rb", HERE / "05_rotation_backtest.py")
rb = module_from_spec(_spec)
_spec.loader.exec_module(rb)
_spec = spec_from_file_location("wf", HERE / "03_walkforward.py")
wf = module_from_spec(_spec)
_spec.loader.exec_module(wf)
pb = rb.pb
HOLD = 21
BUCKETS = [(1, 5), (6, 10), (11, 20), (21, 40), (41, 60), (61, 100), (101, 200), (201, 300)]


def main() -> None:
    pb.BARS, pb.OUT = pb.PIT_BARS, pb.PIT_OUT
    p = pd.read_parquet(pb.OUT, columns=["date", "ticker", "mom_12_1", "dv_rank", "rv_63"])
    p = p[(p["dv_rank"] <= 300) & (p["date"] >= rb.START)].reset_index(drop=True)
    O_df, C_df = rb.price_matrices(sorted(set(p["ticker"]) | set(rb.BENCH)),
                                   pb.PIT_OUT.parent / "px_open_pit.parquet", pb.PIT_OUT.parent / "px_close_pit.parquet")
    C_df = C_df.ffill()
    O_df = O_df.fillna(C_df.shift(1))
    cal, col = C_df.index, {t: i for i, t in enumerate(C_df.columns)}
    e = cal.searchsorted(p["date"].values, side="right")
    x = e + HOLD - 1
    ok = x < len(cal)
    j = p["ticker"].map(col).values
    p = p[ok].assign(fwd=C_df.values[x[ok], j[ok]] / O_df.values[e[ok], j[ok]] - 1)
    p["xs"] = p["fwd"] - p.groupby("date")["fwd"].transform("mean")
    p["rank"] = p.groupby("date")["mom_12_1"].rank(ascending=False, method="first")
    p["vol_pct"] = p.groupby("date")["rv_63"].rank(pct=True)
    lines = [f"=== next-{HOLD}-session return by 12-1 momentum rank | liquid-300, point-in-time | "
             f"{p.date.nunique()} weekly dates {p.date.min().date()}..{p.date.max().date()} ==="]
    lines.append(f"{'momentum rank':>13s} {'mean xs/month':>13s} {'[95% 6mo-block CI]':>20s} {'median xs':>9s} "
                 f"{'hit vs group':>12s} {'P(top decile)':>13s} {'vol pct':>7s}")
    p["top_dec"] = p.groupby("date")["fwd"].rank(pct=True) >= 0.9
    for lo, hi in BUCKETS:
        b = p[p["rank"].between(lo, hi)]
        m = b.groupby("date")["xs"].mean()
        ci = wf.block_ci(m)
        lines.append(f"{f'{lo}-{hi}':>13s} {m.mean():+13.2%} [{ci[0]:+7.2%}, {ci[1]:+7.2%}] {b['xs'].median():+9.2%} "
                     f"{(b['xs'] > 0).mean():12.0%} {b['top_dec'].mean():13.0%} {b['vol_pct'].mean():7.2f}")
    txt = "\n".join(lines)
    print(txt)
    (HERE / "14_results.txt").write_text(txt)


if __name__ == "__main__":
    main()
