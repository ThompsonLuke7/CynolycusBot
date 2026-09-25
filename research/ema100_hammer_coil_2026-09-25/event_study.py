"""Event study: does "uptrend pullback to 100 EMA + hammer + coil" have edge on liquid names?

Motivated by AMD 2026-09-03 (low 440.5 tagged EMA100 445.0, hammer, then +38% by 09-24).
Question: is that a repeatable, tradable pattern or a memorable single case?

Data: Data/shared/bars/1d (UNADJUSTED, survivor-shaped; see project notes). Splits are
handled by dropping any event whose entry->exit window contains a |1d return| > 35%.
Universe per date: top-N by trailing 60d median dollar volume (per-date rank, so the
2026Q3 IEX->SIP volume switch does not shift the cut).

Timing: signal computed on the close of day t; entry at open t+1; exit at close t+h.
Returns are excess vs SPY over the same open(t+1)->close(t+h) window. EMAs/ATR use data
through t only.

  PYTHONPATH=. .venv/bin/python research/ema100_hammer_coil_2026-09-25/event_study.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
BARS = REPO / "Data/shared/bars/1d"
OUT = Path(__file__).resolve().parent
TOP_N = 300
HORIZONS = (5, 10, 15)
SPLIT_GUARD = 0.35
N_BOOT = 2000
SEED = 7


def load(sym: str) -> pd.DataFrame | None:
    p = BARS / f"{sym}.parquet"
    if not p.exists():
        return None
    b = pd.read_parquet(p, columns=["timestamp", "open", "high", "low", "close", "volume"])
    b["date"] = pd.to_datetime(b["timestamp"], utc=True).dt.tz_convert("America/New_York").dt.normalize().dt.tz_localize(None)
    return b.drop_duplicates("date").set_index("date").sort_index().drop(columns="timestamp")


def features(b: pd.DataFrame) -> pd.DataFrame:
    c, o, h, l = b["close"], b["open"], b["high"], b["low"]
    f = pd.DataFrame(index=b.index)
    for n in (20, 50, 100, 200):
        f[f"e{n}"] = c.ewm(span=n, adjust=False, min_periods=n).mean()
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    f["atr10"] = tr.rolling(10).mean()
    f["atr50"] = tr.rolling(50).mean()
    f["dv60"] = (c * b["volume"]).rolling(60).median()
    rng = (h - l).replace(0, np.nan)
    body = (c - o).abs()
    lower = np.minimum(o, c) - l
    # Trend: stacked 50>100>200, 100 EMA rising over 20d.
    f["uptrend"] = (f.e50 > f.e100) & (f.e100 > f.e200) & (f.e100 > f.e100.shift(20))
    # Pullback into the 100 EMA and closing back above it.
    f["touch100"] = (l <= f.e100 * 1.005) & (c > f.e100)
    f["hammer"] = (lower >= 2 * body) & (lower >= 0.5 * rng) & ((c - l) / rng >= 0.6)
    # Coil: short-term true range compressed vs 50d.
    f["coil"] = f.atr10 / f.atr50 < 0.85
    # Prior leader: >40% gain over the prior 126 sessions (AMD-2026-like base after a run).
    f["leader"] = c / c.shift(126) - 1 > 0.40
    # Forward excess returns filled in later (need SPY); keep raw pieces.
    f["open_next"] = o.shift(-1)
    for hz in HORIZONS:
        f[f"close_fwd{hz}"] = c.shift(-hz)
        daily = c.pct_change().abs()
        f[f"maxjump{hz}"] = daily[::-1].rolling(hz, min_periods=1).max()[::-1].shift(-1)
    return f


def boot_ci(x: pd.Series, months: pd.Series, rng: np.random.Generator) -> tuple[float, float]:
    g = pd.DataFrame({"x": x.values, "m": months.values}).groupby("m")["x"]
    sums, cnts = g.sum().values, g.count().values
    k = len(sums)
    if k < 5:
        return (np.nan, np.nan)
    idx = rng.integers(0, k, size=(N_BOOT, k))
    means = sums[idx].sum(1) / cnts[idx].sum(1)
    return tuple(np.percentile(means, [2.5, 97.5]))


def main() -> None:
    spy = features(load("SPY"))
    rows = []
    for p in sorted(BARS.glob("*.parquet")):
        sym = p.stem
        b = load(sym)
        if b is None or len(b) < 260:
            continue
        f = features(b)
        f["ticker"] = sym
        rows.append(f.dropna(subset=["e200", "dv60"]))
    panel = pd.concat(rows).reset_index()
    panel["dv_rank"] = panel.groupby("date")["dv60"].rank(ascending=False)
    panel = panel[panel.dv_rank <= TOP_N].copy()
    spy = spy.reindex(panel["date"].unique())
    for hz in HORIZONS:
        spy_ret = spy[f"close_fwd{hz}"] / spy["open_next"] - 1
        r = panel[f"close_fwd{hz}"] / panel["open_next"] - 1
        panel[f"xs{hz}"] = r - panel["date"].map(spy_ret)
        panel.loc[panel[f"maxjump{hz}"] > SPLIT_GUARD, f"xs{hz}"] = np.nan

    arms = {
        "A all liquid days": pd.Series(True, index=panel.index),
        "B uptrend": panel.uptrend,
        "C uptrend+touch100": panel.uptrend & panel.touch100,
        "D C+hammer": panel.uptrend & panel.touch100 & panel.hammer,
        "E C+coil": panel.uptrend & panel.touch100 & panel.coil,
        "F C+hammer+coil": panel.uptrend & panel.touch100 & panel.hammer & panel.coil,
        "G C+leader": panel.uptrend & panel.touch100 & panel.leader,
        "H F+leader": panel.uptrend & panel.touch100 & panel.hammer & panel.coil & panel.leader,
        "I C top50": panel.uptrend & panel.touch100 & (panel.dv_rank <= 50),
        "J D top50": panel.uptrend & panel.touch100 & panel.hammer & (panel.dv_rank <= 50),
        "K F top50": panel.uptrend & panel.touch100 & panel.hammer & panel.coil & (panel.dv_rank <= 50),
    }
    rng = np.random.default_rng(SEED)
    month = panel["date"].dt.to_period("M")
    out = []
    for name, m in arms.items():
        for hz in HORIZONS:
            x = panel.loc[m, f"xs{hz}"].dropna()
            lo, hi = boot_ci(x, month[x.index], rng)
            out.append({"arm": name, "h": hz, "n": len(x), "n_tickers": panel.loc[x.index, "ticker"].nunique(),
                        "mean_xs_%": 100 * x.mean(), "median_xs_%": 100 * x.median(),
                        "hit_%": 100 * (x > 0).mean(), "ci_lo_%": 100 * lo, "ci_hi_%": 100 * hi,
                        "p90_%": 100 * x.quantile(0.9)})
    res = pd.DataFrame(out).round(2)
    res.to_csv(OUT / "event_study_results.csv", index=False)
    print(f"panel {panel.date.min().date()}..{panel.date.max().date()}  top{TOP_N} by 60d $vol")
    print(res.to_string(index=False))

    ev = panel[arms["F C+hammer+coil"]][["date", "ticker", "xs5", "xs10", "xs15"]]
    ev.to_csv(OUT / "events_F.csv", index=False)
    print("\nAMD events in any arm C..F:")
    amd = panel[(panel.ticker == "AMD") & arms["C uptrend+touch100"]]
    print(amd[["date", "hammer", "coil", "xs5", "xs10", "xs15"]].round(3).to_string(index=False))
    # Era split for arm F to check stability.
    for name in ("C uptrend+touch100", "F C+hammer+coil"):
        x = panel.loc[arms[name], ["date", "xs10"]].dropna()
        print(f"\n{name} xs10 by year:")
        print(x.groupby(x.date.dt.year)["xs10"].agg(["count", "mean", lambda s: (s > 0).mean()]).round(4).to_string())


if __name__ == "__main__":
    main()
