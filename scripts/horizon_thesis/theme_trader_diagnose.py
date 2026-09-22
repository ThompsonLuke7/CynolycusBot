"""Diagnose the implausible theme-trader result before anyone believes it.

The backtest reported +2,450% total / 74% annual alpha for the leader arm. Results
that large in this repo have twice turned out to be corporate-action artifacts in the
UNADJUSTED daily cache, so this checks the three ways it could be fake:

  1. CONCENTRATION  what share of total P&L comes from the top handful of positions?
  2. ARTIFACTS      do the biggest winners contain a >=4x overnight ratio move ANYWHERE
                    in the holding window? (the backtest and the §7 event study both
                    screened only the ENTRY day -- a split inside the hold was never
                    filtered)
  3. TRADEABILITY   what is the dollar volume and price of the names being bought?
                    A +7% pop screen over 1,135 names selects microcaps that cannot be
                    bought at the open in size.

Prints the worst offenders by name so they can be checked by hand.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts/horizon_thesis"))

from theme_cohesion_leadlag import POP_THRESHOLD, THEMES  # noqa: E402

BARS_1D = REPO / "Data/shared/bars/1d"
HOLD = 5


def main() -> None:
    th = pd.read_parquet(THEMES, columns=["date", "ticker", "primary_theme"]).dropna()
    th["date"] = pd.to_datetime(th["date"])
    tickers = sorted(th["ticker"].unique())
    op, cl, vol = {}, {}, {}
    for t in tickers:
        p = BARS_1D / f"{t}.parquet"
        if not p.exists():
            continue
        d = pd.read_parquet(p, columns=["timestamp", "open", "close", "volume"])
        if len(d) < 150:
            continue
        idx = (pd.to_datetime(d["timestamp"], utc=True).dt.tz_convert("America/New_York")
               .dt.normalize().dt.tz_localize(None))
        d = d.set_index(idx).sort_index()
        op[t], cl[t], vol[t] = d["open"], d["close"], d["volume"]
    opens, closes, volumes = pd.DataFrame(op), pd.DataFrame(cl), pd.DataFrame(vol)
    rets = closes.pct_change(fill_method=None)
    sessions = closes.index
    dollar_vol = (closes * volumes).rolling(20).mean()

    # every leader-pop trade at HOLD, entered next open
    rows = []
    pos = {s: i for i, s in enumerate(sessions)}
    for d, g in th.groupby("date"):
        if d not in pos or pos[d] + 1 + HOLD >= len(sessions):
            continue
        row = rets.loc[d]
        for t in g["ticker"]:
            if t not in row.index or not row[t] >= POP_THRESHOLD:
                continue
            e, x = sessions[pos[d] + 1], sessions[pos[d] + 1 + HOLD]
            o, c = opens.at[e, t], closes.at[x, t]
            if pd.isna(o) or pd.isna(c) or o <= 0:
                continue
            win = closes.loc[e:x, t].dropna()
            ratio = float((win / win.shift(1)).max()) if len(win) > 2 else np.nan
            rows.append(dict(ticker=t, entry=e, ret=c / o - 1.0, pop=float(row[t]),
                             max_overnight_ratio=ratio,
                             dollar_vol=float(dollar_vol.at[d, t]) if t in dollar_vol.columns else np.nan,
                             price=float(closes.at[d, t])))
    tr = pd.DataFrame(rows).dropna(subset=["ret"])
    print(f"leader-pop trades at {HOLD}d: {len(tr):,}  mean {tr['ret'].mean() * 100:+.2f}%  "
          f"median {tr['ret'].median() * 100:+.2f}%  win {(tr['ret'] > 0).mean():.1%}")

    s = tr["ret"].sort_values(ascending=False)
    tot = tr["ret"].sum()
    print(f"\n1. CONCENTRATION: top 1% of trades = {s.head(max(1, len(s) // 100)).sum() / tot:.0%} "
          f"of summed return; top 5% = {s.head(len(s) // 20).sum() / tot:.0%}")

    susp = tr[tr["max_overnight_ratio"] >= 4.0]
    print(f"\n2. ARTIFACTS: {len(susp)} trades contain a >=4x overnight ratio INSIDE the hold "
          f"({len(susp) / len(tr):.3%} of trades) but contribute "
          f"{susp['ret'].sum() / tot:.1%} of summed return")
    if len(susp):
        print(susp.nlargest(10, "ret")[["ticker", "entry", "ret", "max_overnight_ratio", "price"]]
              .to_string(index=False))
    print("\n   biggest winners overall:")
    print(tr.nlargest(12, "ret")[["ticker", "entry", "ret", "pop", "max_overnight_ratio", "price", "dollar_vol"]]
          .to_string(index=False))

    print("\n3. TRADEABILITY of the names bought:")
    for lo, hi, lab in ((0, 1e6, "<$1M/day"), (1e6, 1e7, "$1-10M"), (1e7, 1e8, "$10-100M"),
                        (1e8, np.inf, ">$100M")):
        m = tr["dollar_vol"].between(lo, hi)
        if m.any():
            print(f"   {lab:10s} {m.mean():6.1%} of trades  mean ret {tr.loc[m, 'ret'].mean() * 100:+7.2f}%")
    print(f"   median price ${tr['price'].median():.2f}; share under $5: {(tr['price'] < 5).mean():.1%}")
    liquid = tr[(tr["dollar_vol"] >= 1e7) & (tr["price"] >= 5)]
    print(f"\n   LIQUID SUBSET ($10M+/day, $5+): {len(liquid):,} trades, "
          f"mean {liquid['ret'].mean() * 100:+.2f}%, median {liquid['ret'].median() * 100:+.2f}% "
          f"(vs all-trades mean {tr['ret'].mean() * 100:+.2f}%)")


if __name__ == "__main__":
    main()
