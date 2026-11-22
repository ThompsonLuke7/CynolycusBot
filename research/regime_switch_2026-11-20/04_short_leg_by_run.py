"""Decomposition of 01's long/short book: what did each short run (a stretch of DOWN state) make?

No new hypothesis: same QQQ bars, detectors and timing as 01_regime_study.py. A run starts at the
open after the first DOWN close and ends at the open after the first UP close. Its short P&L is
-(exit open / entry open - 1) (gross, no cost). A run "overlaps a dump" when any of its days
falls inside a >= 10% close-to-close drawdown (peak to trough) from 01's episode finder.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
_spec = importlib.util.spec_from_file_location("rs01", HERE / "01_regime_study.py")
rs01 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rs01)


def runs(d: pd.DataFrame, down: pd.Series, in_dump: np.ndarray, det: str) -> list[dict]:
    o, idx = d["open"].to_numpy(), d.index
    s = down.to_numpy()
    out, i, n = [], 200, len(s)
    while i < n:
        if s[i] == 1.0:
            j = i
            while j + 1 < n and s[j + 1] == 1.0:
                j += 1
            e_i, x_i = i + 1, j + 2           # enter next open, exit at the open after the first UP close
            if e_i >= n:
                break
            x_i = min(x_i, n - 1)
            out.append({"detector": det, "flag_close": idx[i].date(), "entry": idx[e_i].date(), "exit": idx[x_i].date(),
                        "sessions": x_i - e_i, "short_ret": -(o[x_i] / o[e_i] - 1),
                        "dump": bool(in_dump[i:j + 1].any()), "open_at_end": j + 1 >= n})
            i = j + 1
        i += 1
    return out


def main() -> None:
    q = rs01.load("QQQ")
    st = rs01.states(q)
    in_dump = np.zeros(len(q), bool)
    for e in rs01.drawdown_episodes(q["close"]):
        in_dump[e["peak_i"]: e["trough_i"] + 1] = True
    rows = []
    for det in ("D0_fast", "D1_sma50", "D2_ema20_50"):
        rows += runs(q, st[det], in_dump, det)
    r = pd.DataFrame(rows)
    r.to_csv(OUT / "short_runs.csv", index=False)
    g = r.groupby(["detector", "dump"]).agg(runs=("short_ret", "size"), win=("short_ret", lambda s: (s > 0).mean()),
                                            mean=("short_ret", "mean"),
                                            compounded=("short_ret", lambda s: (1 + s).prod() - 1),
                                            median_sessions=("sessions", "median"))
    g.to_csv(OUT / "short_runs_summary.csv")
    print(g.round(3).to_string())


if __name__ == "__main__":
    main()
