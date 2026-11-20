"""Where the $627k went: module, route, month, market beta, leverage, exercise, outage.

Reads the ledger from 01_build_ledger.py. Everything here is descriptive (what
happened), not a counterfactual; 03_counterfactuals.py asks what would have.

    PYTHONPATH=. .venv/bin/python research/account_postmortem_2026-11-03/02_attribution.py
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("Data/research/account_postmortem_20261103")
ET = "America/New_York"
pd.set_option("display.width", 220)
pd.set_option("display.max_rows", 200)


def index_closes(symbols=("SPY", "QQQ", "IWM")) -> pd.DataFrame:
    out = {}
    for s in symbols:
        d = pd.read_parquet(f"Data/shared/bars/1d/{s}.parquet")
        tcol = next(c for c in d.columns if "time" in c.lower() or "date" in c.lower())
        idx = pd.to_datetime(d[tcol], utc=True).dt.tz_convert(ET).dt.normalize() + pd.Timedelta(hours=16)
        out[s] = pd.Series(d["close"].to_numpy(), index=idx)
    return pd.DataFrame(out).sort_index()


def close_at(closes: pd.DataFrame, ts: pd.Timestamp) -> pd.Series:
    return closes[closes.index <= ts].iloc[-1]


def server_coverage() -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """[first, last] timestamp of every server log, as ET timestamps."""

    spans = []
    for path in sorted(glob.glob("logs/live_server/server_*.log")):
        first = last = None
        with open(path, errors="replace") as fh:
            for line in fh:
                if line[:4] == "2026" and line[4] == "-":
                    stamp = line[:19]
                    first = first or stamp
                    last = stamp
        if first:
            spans.append((pd.Timestamp(first, tz=ET), pd.Timestamp(last, tz=ET)))
    return spans


def session_was_managed(day: pd.Timestamp, spans) -> bool:
    open_, close = day.normalize() + pd.Timedelta(hours=9, minutes=35), day.normalize() + pd.Timedelta(hours=15, minutes=55)
    return any(a <= open_ + pd.Timedelta(hours=3) and b >= open_ for a, b in spans) or any(
        a <= close and b >= close - pd.Timedelta(hours=2) for a, b in spans)


def main() -> None:
    c = pd.read_parquet(OUT / "ledger_contrib.parquet")
    a = pd.read_parquet(OUT / "ledger_account.parquet")
    ep = pd.read_parquet(OUT / "ledger_episodes.parquet")
    ep["exercised"] = ep.episode.str.contains("T23:59:00")
    total = a.d_equity.sum()
    print(f"equity change 07-15 -> 11-02: {total:,.0f} (plus -$7,506 from the 07-09 $1,000,000 deposit to the first snapshot)")

    print("\n== by module x route (realized + unrealized, $) ==")
    t = ep.pivot_table(index="module", columns="route", values="pnl", aggfunc="sum", fill_value=0)
    t["total"] = t.sum(axis=1)
    t["episodes"] = ep.groupby("module").size()
    t["win%"] = ep.groupby("module").pnl.apply(lambda s: (s > 0).mean() * 100).round(1)
    print(t.sort_values("total").round(0).to_string())
    print(f"other (fees/div/merger): {c[c.symbol == '__other__'].contrib.sum():,.0f}")

    print("\n== by month of the interval ==")
    c["month"] = c.t1.dt.strftime("%Y-%m")
    print(c.pivot_table(index="month", columns="route", values="contrib", aggfunc="sum", fill_value=0).assign(
        total=lambda x: x.sum(axis=1)).round(0).to_string())

    print("\n== options by fate ==")
    opt = ep[ep.route == "option"].copy()
    nontrade = pd.DataFrame(json.loads((OUT / "broker_nontrade_20260701_20261103.json").read_text()))
    expired = set(nontrade[nontrade.activity_type == "OPEXP"].symbol)
    exercised = set(nontrade[nontrade.activity_type == "OPEXC"].symbol)
    opt["fate"] = np.where(opt.symbol.isin(exercised), "exercised", np.where(opt.symbol.isin(expired), "expired_worthless", "sold/open"))
    print(opt.groupby("fate").pnl.agg(["size", "sum", lambda s: (s > 0).mean()]).round(2).to_string())
    print("option episodes held into expiry day:", int(opt.fate.ne("sold/open").sum()))

    print("\n== exercise-delivered stock lots ==")
    lots = ep[ep.exercised]
    print(lots[["symbol", "module", "buy_cost", "sell_proceeds", "pnl", "last_fill_ts"]].round(0).to_string())
    print(f"delivered notional {lots.buy_cost.sum():,.0f}; P&L of the delivered stock {lots.pnl.sum():,.0f}")

    print("\n== market beta vs selection (per interval) ==")
    closes = index_closes()
    rows = []
    for _, r in a.iterrows():
        c0, c1 = close_at(closes, r.t0), close_at(closes, r.t1)
        ret = c1 / c0 - 1
        rows.append({"t1": r.t1, "pnl": r.d_equity, "gross": r.gross_t0, "eq": r.equity_t0,
                     **{f"r_{k}": v for k, v in ret.items()}})
    m = pd.DataFrame(rows)
    m = m[m.gross > 0]
    m["book_ret"] = m.pnl / m.gross
    for idx in ("SPY", "QQQ", "IWM"):
        x, y = m[f"r_{idx}"].to_numpy(), m.book_ret.to_numpy()
        beta = np.cov(x, y)[0, 1] / np.var(x, ddof=1)
        alpha = y.mean() - beta * x.mean()
        corr = np.corrcoef(x, y)[0, 1]
        market = (beta * m[f"r_{idx}"] * m.gross).sum()
        print(f"{idx}: book beta {beta:.2f} corr {corr:.2f}; market part {market:,.0f}, "
              f"selection (alpha x gross) {(alpha * m.gross).sum():,.0f}, mean gross ${m.gross.mean():,.0f}")
    print(f"index moves 07-15..11-02: {(close_at(closes, a.t1.iloc[-1]) / close_at(closes, a.t0.iloc[0]) - 1).round(4).to_dict()}")

    print("\n== leverage ==")
    a["gross_x"] = a.gross_t0 / a.equity_t0
    a["ret"] = a.d_equity / a.equity_t0
    for lo, hi in ((0, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 9)):
        s = a[(a.gross_x >= lo) & (a.gross_x < hi)]
        print(f"gross {lo:.1f}-{hi:.1f}x: {len(s):2d} intervals, P&L {s.d_equity.sum():>10,.0f}, "
              f"mean interval return {s.ret.mean() * 100:+.2f}%, book return {(s.d_equity / s.gross_t0).mean() * 100:+.2f}%")

    print("\n== unmanaged sessions (server not running) ==")
    spans = server_coverage()
    a["managed"] = [all(session_was_managed(d, spans) for d in pd.bdate_range(r.t0.normalize() + pd.Timedelta(days=1), r.t1.normalize(), tz=ET))
                    for _, r in a.iterrows()]
    um = a[~a.managed]
    print(um[["t0", "t1", "d_equity", "gross_x"]].assign(t0=lambda x: x.t0.dt.strftime("%m-%d"), t1=lambda x: x.t1.dt.strftime("%m-%d")).round(2).to_string())
    print(f"P&L over intervals containing an unmanaged session: {um.d_equity.sum():,.0f} of {total:,.0f}")

    print("\n== worst 15 episodes ==")
    print(ep.sort_values("pnl")[["symbol", "module", "route", "entry_ts", "buy_cost", "pnl"]].head(15)
          .assign(entry_ts=lambda x: x.entry_ts.dt.strftime("%m-%d %H:%M")).round(0).to_string())
    print("\n== concentration: share of total loss from the worst N episodes ==")
    losses = ep.pnl.sort_values()
    for n in (10, 25, 50, 100):
        print(f"  worst {n}: {losses.head(n).sum():,.0f} ({losses.head(n).sum() / total * 100:.0f}% of the net)")
    m.to_parquet(OUT / "interval_market.parquet", index=False)
    a.to_parquet(OUT / "ledger_account_annotated.parquet", index=False)


if __name__ == "__main__":
    main()
