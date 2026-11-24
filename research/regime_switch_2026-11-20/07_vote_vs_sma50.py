"""Does a several-signal regime vote do better than the 50-day alone?

The question (user, 2026-11-22): "i dont want to just use 50ma i want to use a various collection,
maybe even an ML model for market regime".

Pre-registered 2026-11-22 ~02:28 ET, before any result was computed. No new thresholds: the vote is
the one fixed in 05_intraday_structure_by_regime.py (7 bearish votes; BEAR = votes >= 4). The
stricter BEAR5 (votes >= 5) is the only added variant and is reported whatever it shows.

Data   QQQ daily bars from 01 (adjusted; only used for moving-average comparisons, which are
       scale-invariant) and daily_regime.parquet (starts 2020-07-27). Window: sessions from
       2020-08-03 to 2026-11-19 so every detector is compared on the same days.
State  The vote for session s is known at the close of s-1, so it is stored as the state at that
       close, and every detector uses the timing of 01/04: decide at the close, trade the next open.
Tests  (1) detection on each QQQ drawdown >= 10% in the window (sessions from peak to first flag, share
       of the drawdown already gone), flips and false flags per year (01's t4);
       (2) short runs one by one, split dump vs false alarm (04's runs);
       (3) daily books long / long-flat / long-short with 2 bp per unit turnover (01's book) and the
       circular-shift placebo (01's 1,000 draws, minimum shift 60 sessions).
Limits About six years with ~4 corrections. The regime table's z-scores are rolling and stamped
       21:30 UTC on the day, but the historical rows were built in one backfill, not captured live.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
START = pd.Timestamp("2020-08-03")


def _load(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rs01 = _load("rs01", "01_regime_study.py")
rs04 = _load("rs04", "04_short_leg_by_run.py")
rs05 = _load("rs05", "05_intraday_structure_by_regime.py")


def main() -> None:
    q = rs01.load("QQQ")
    st = rs01.states(q)
    sessions = q.index[q.index > START - pd.Timedelta(days=10)]
    nxt = rs01.LAST_SESSION + pd.offsets.BDay(1)  # 11-20: its vote is the state at the 11-19 close
    v = rs05.regime_votes(pd.DatetimeIndex(list(sessions[1:]) + [nxt]))
    # vote known before session s  ->  state at the close of the session before s
    closes = pd.Series(q.index, index=q.index)
    prev_close = closes.shift(1).reindex(v.index)
    votes = pd.Series(v["votes"].to_numpy(), index=prev_close.to_numpy())
    # the 11-20 vote (from the 11-19 close) maps to the last close in the data
    votes = votes[votes.index.notna()]
    if rs01.LAST_SESSION + pd.offsets.BDay(1) in v.index:
        votes.loc[rs01.LAST_SESSION] = v.at[rs01.LAST_SESSION + pd.offsets.BDay(1), "votes"]
    votes = votes[~votes.index.duplicated(keep="last")].reindex(q.index)

    s = pd.DataFrame(index=q.index)
    s["D1_sma50"] = st["D1_sma50"]
    s["BEAR4"] = (votes >= 4).astype(float).where(votes.notna())
    s["BEAR5"] = (votes >= 5).astype(float).where(votes.notna())
    keep = s.index >= START
    d, s = q.loc[keep], s.loc[keep]
    if s.isna().any().any():
        raise ValueError("missing state inside the window:\n" + s[s.isna().any(axis=1)].to_string())

    # (1) detection. 01's t4 skips episodes whose peak index < 200 and annualises from row 200, so pad
    # the frame with 200 warm-up rows of the full history and keep the states NaN there.
    pad = q.loc[: START].iloc[-201:-1]
    dd = pd.concat([pad, d])
    ss = pd.concat([pd.DataFrame(np.nan, index=pad.index, columns=s.columns), s])
    old = rs01.DETECTORS
    rs01.DETECTORS = tuple(s.columns)
    det_rows, fa_rows = rs01.t4("QQQ", dd, ss)
    # (3) books with placebo
    sub = dd.iloc[200:]
    base = rs01.book(sub, pd.Series(False, index=sub.index), "long")
    book_rows = [{"detector": "-", "mode": "long", **rs01.stats(base)}]
    for det in s.columns:
        x = ss[det].iloc[200:].astype(bool)
        for mode in ("long_flat", "long_short"):
            real = rs01.stats(rs01.book(sub, x, mode))
            arr = x.to_numpy()
            sh, md = [], []
            for _ in range(rs01.PLACEBO_DRAWS):
                k = int(rs01.RNG.integers(rs01.MIN_SHIFT, len(arr) - rs01.MIN_SHIFT))
                p = rs01.stats(rs01.book(sub, pd.Series(np.roll(arr, k), index=x.index), mode))
                sh.append(p["sharpe"]); md.append(p["mdd"])
            book_rows.append({"detector": det, "mode": mode, **real,
                              "down_share": float(x.mean()),
                              "placebo_sharpe_pct": float((np.array(sh) < real["sharpe"]).mean()),
                              "placebo_mdd_pct": float((np.array(md) < real["mdd"]).mean())})
    rs01.DETECTORS = old

    # (2) short runs
    in_dump = np.zeros(len(dd), bool)
    for e in rs01.drawdown_episodes(dd["close"]):
        in_dump[e["peak_i"]: e["trough_i"] + 1] = True
    runs = []
    for det in s.columns:
        runs += rs04.runs(dd, ss[det], in_dump, det)
    r = pd.DataFrame(runs)
    g = r.groupby(["detector", "dump"]).agg(runs=("short_ret", "size"), win=("short_ret", lambda z: (z > 0).mean()),
                                            mean=("short_ret", "mean"),
                                            compounded=("short_ret", lambda z: (1 + z).prod() - 1),
                                            median_sessions=("sessions", "median"))

    pd.DataFrame(det_rows).to_csv(OUT / "vote_detection.csv", index=False)
    pd.DataFrame(fa_rows).to_csv(OUT / "vote_flip_rates.csv", index=False)
    pd.DataFrame(book_rows).to_csv(OUT / "vote_books.csv", index=False)
    r.to_csv(OUT / "vote_short_runs.csv", index=False)
    g.to_csv(OUT / "vote_short_runs_summary.csv")
    print(f"window {d.index[0].date()}..{d.index[-1].date()} sessions={len(d)}")


if __name__ == "__main__":
    main()
