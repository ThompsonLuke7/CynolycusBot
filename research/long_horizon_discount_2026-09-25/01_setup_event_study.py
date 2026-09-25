"""Feasibility: does the "dip below 200SMA + TMO oversold pivot" entry earn
excess return over 63/90/126-session holds versus matched controls?

Cheap, standalone check BEFORE building any long-horizon (Roth) model.
Entry = next session OPEN after the signal close; exit = close H sessions later.
Excess = stock return minus SPY over the identical window.

Known biases (see README): bar cache is survivor-shaped (inflates dip-buying
LEVELS most of all, since dips that went to zero are missing) and unadjusted
(2:1/3:1 splits are masked below by a split-shape rule).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
BARS = REPO / "Data" / "shared" / "bars" / "1d"
OUT = Path(__file__).resolve().parent
HOLDS = (63, 90, 126)
TOP_N_DV = 400      # per-date rank on trailing-60d median dollar volume (PIT).
# Absolute $ floors are unusable: the cache's volume is IEX-only until 2026Q3,
# then SIP (~40x larger), so a fixed floor admits a different universe per era.
MIN_PX = 10.0
TMO_OS = -9.0       # ThinkOrSwim-style oversold line ~ -0.7*length (length 14)
LOOKBACK = 10       # dip / oversold must occur within this many sessions
DEDUPE = 20         # at most one event per ticker per 20 sessions, per arm
SEED = 7


def tmo(df: pd.DataFrame, length=14, calc=5, smooth=3):
    """Vectorised twin of spy_intraday custom_indicators.add_tmo."""
    c, o = df["close"], df["open"]
    data = sum(np.sign(c - o.shift(i)).fillna(0) for i in range(length))
    main = data.ewm(span=calc, adjust=False).mean().ewm(span=smooth, adjust=False).mean()
    return main, main.ewm(span=smooth, adjust=False).mean()


def split_mask(df: pd.DataFrame, hold: int) -> pd.Series:
    """Mask windows containing a split-shaped gap: price ratio <=0.7 or >=1.4
    with dollar volume roughly preserved (<2.5x its trailing median)."""
    ratio = df["open"] / df["close"].shift(1)
    dv = df["close"] * df["volume"]
    dv_ratio = dv / dv.shift(1).rolling(20, min_periods=3).median()
    hit = ((ratio <= 0.7) | (ratio >= 1.4)) & (dv_ratio < 2.5)
    bad = np.zeros(len(df), bool)
    for p in np.flatnonzero(hit.to_numpy()):
        bad[max(0, p - hold - 1):p + 1] = True
    return pd.Series(bad, index=df.index)


def load(t: str) -> pd.DataFrame | None:
    p = BARS / f"{t}.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p).sort_values("timestamp").reset_index(drop=True)
    df["date"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("America/New_York").dt.normalize().dt.tz_localize(None)
    return df.set_index("date")[["open", "high", "low", "close", "volume"]]


def features(df: pd.DataFrame, spy: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=df.index)
    c = df["close"]
    sma200 = c.rolling(200).mean()
    main, sig = tmo(df)
    f["dv"] = (c * df["volume"]).rolling(60).median()
    f["px"] = c
    f["below200_recent"] = (df["low"] < sma200).rolling(LOOKBACK).max().astype(bool)
    f["below200_now"] = c < sma200
    f["sma200_rising"] = sma200 > sma200.shift(60)
    f["dd52"] = c / c.rolling(252).max() - 1
    f["near52low"] = c <= 1.15 * c.rolling(252).min()
    f["os_recent"] = (main <= TMO_OS).rolling(LOOKBACK).max().astype(bool)
    f["tmo_turn"] = (main > sig) & (main.shift(1) <= sig.shift(1))
    f["valid"] = sma200.notna() & c.rolling(252).max().notna()
    nxt_open = df["open"].shift(-1)
    s_open = spy["open"].reindex(df.index).shift(-1)
    for h in HOLDS:
        r = df["close"].shift(-h) / nxt_open - 1
        rs = spy["close"].reindex(df.index).shift(-h) / s_open - 1
        mdd = (df["low"][::-1].rolling(h, min_periods=h).min()[::-1].shift(-1) / nxt_open - 1)
        bad = split_mask(df, h)
        f[f"ret_{h}"] = r.where(~bad)
        f[f"xs_{h}"] = (r - rs).where(~bad)
        f[f"mdd_{h}"] = mdd.where(~bad)
    return f


def dedupe(ev: pd.DataFrame) -> pd.DataFrame:
    keep, last = [], {}
    for i, (t, d) in enumerate(zip(ev["ticker"], ev["pos"])):
        if t not in last or d - last[t] >= DEDUPE:
            keep.append(i)
            last[t] = d
    return ev.iloc[keep]


def month_boot_diff(a: pd.DataFrame, b: pd.DataFrame, col: str, n=2000):
    """Bootstrap CI for mean(a)-mean(b) resampling signal MONTHS (events cluster)."""
    rng = np.random.default_rng(SEED)
    ma = a.groupby("month")[col].agg(["sum", "count"])
    mb = b.groupby("month")[col].agg(["sum", "count"])
    months = ma.index.union(mb.index)
    ma, mb = ma.reindex(months, fill_value=0), mb.reindex(months, fill_value=0)
    out = []
    for _ in range(n):
        idx = rng.integers(0, len(months), len(months))
        sa, ca = ma["sum"].values[idx].sum(), ma["count"].values[idx].sum()
        sb, cb = mb["sum"].values[idx].sum(), mb["count"].values[idx].sum()
        if ca and cb:
            out.append(sa / ca - sb / cb)
    return np.percentile(out, [2.5, 97.5])


def main():
    uni = pd.read_csv(REPO / "Data" / "shared" / "universe" / "shared_universe.csv")
    stocks = sorted(uni.loc[uni["type"] == "Stock", "ticker"].dropna().unique())
    spy = load("SPY")
    rows = []
    for t in stocks:
        df = load(t)
        if df is None or len(df) < 400:
            continue
        f = features(df, spy)
        f["ticker"], f["pos"] = t, np.arange(len(f))
        rows.append(f[f["valid"] & (f["px"] >= MIN_PX)])
    panel = pd.concat(rows)
    panel = panel[panel.groupby(level=0)["dv"].rank(ascending=False) <= TOP_N_DV]
    panel["month"] = panel.index.to_period("M")
    panel["year"] = panel.index.year
    print(f"panel: {len(panel):,} stock-days, {panel.ticker.nunique()} tickers, "
          f"{panel.index.min().date()}..{panel.index.max().date()}")

    arms = {
        "all_days (baseline)": pd.Series(True, index=panel.index).to_numpy(),
        "discount_only (dipped<200, no TMO turn)": (panel.below200_recent & ~panel.tmo_turn).to_numpy(),
        "tmo_turn_only (no dip)": (panel.tmo_turn & ~panel.below200_recent & panel.os_recent).to_numpy(),
        "SETUP dip<200 + OS + TMO turn": (panel.below200_recent & panel.os_recent & panel.tmo_turn).to_numpy(),
        "SETUP + rising 200 (uptrend pullback)": (panel.below200_recent & panel.os_recent & panel.tmo_turn & panel.sma200_rising).to_numpy(),
        "discount + rising 200, no turn (control)": (panel.below200_recent & panel.sma200_rising & ~panel.tmo_turn).to_numpy(),
        "near 52w low + OS + TMO turn": (panel.near52low & panel.os_recent & panel.tmo_turn).to_numpy(),
    }
    events = {}
    for name, m in arms.items():
        ev = panel[m].reset_index().sort_values(["ticker", "date"])
        events[name] = dedupe(ev).set_index("date") if "baseline" not in name else panel[m].sample(frac=0.2, random_state=SEED)

    lines = []
    for h in HOLDS:
        base = events["all_days (baseline)"].dropna(subset=[f"xs_{h}"])
        lines.append(f"\n=== hold {h} sessions (entry next open) ===")
        lines.append(f"{'arm':44s} {'n':>6s} {'tickers':>7s} {'mean_ret':>9s} {'med_ret':>8s} "
                     f"{'mean_xsSPY':>10s} {'hit>SPY':>8s} {'mean_MDD':>9s}  xs_vs_baseline [95% month-boot]")
        for name, ev in events.items():
            e = ev.dropna(subset=[f"xs_{h}"])
            if e.empty:
                continue
            diff = e[f"xs_{h}"].mean() - base[f"xs_{h}"].mean()
            ci = month_boot_diff(e, base, f"xs_{h}") if "baseline" not in name else (np.nan, np.nan)
            lines.append(f"{name:44s} {len(e):6d} {e.ticker.nunique():7d} {e[f'ret_{h}'].mean():+9.2%} "
                         f"{e[f'ret_{h}'].median():+8.2%} {e[f'xs_{h}'].mean():+10.2%} "
                         f"{(e[f'xs_{h}'] > 0).mean():8.1%} {e[f'mdd_{h}'].mean():+9.2%}  "
                         f"{diff:+.2%} [{ci[0]:+.2%}, {ci[1]:+.2%}]")
    # stability by year for the headline arm vs its no-turn control at 90d
    lines.append("\n=== by signal year, hold 90, mean excess vs SPY ===")
    tab = {}
    for name in ["all_days (baseline)", "discount + rising 200, no turn (control)",
                 "SETUP dip<200 + OS + TMO turn", "SETUP + rising 200 (uptrend pullback)"]:
        e = events[name].dropna(subset=["xs_90"])
        tab[name[:28]] = e.groupby("year")["xs_90"].agg(lambda s: f"{s.mean():+.1%} (n={len(s)})")
    lines.append(pd.DataFrame(tab).to_string())
    # concentration check: share of SETUP excess from top-10 events
    e = events["SETUP + rising 200 (uptrend pullback)"].dropna(subset=["xs_90"])
    top = e["xs_90"].sort_values(ascending=False)
    lines.append(f"\nconcentration (uptrend SETUP, 90d): top-10 events carry "
                 f"{top.head(10).sum() / top.sum():.0%} of summed excess; "
                 f"trimmed-5% mean {top.iloc[len(top)//20:-len(top)//20 or None].mean():+.2%}")
    txt = "\n".join(lines)
    print(txt)
    (OUT / "01_results.txt").write_text(txt)
    events["SETUP + rising 200 (uptrend pullback)"].reset_index()[
        ["date", "ticker", "ret_63", "xs_63", "ret_90", "xs_90", "ret_126", "xs_126", "mdd_90"]
    ].to_csv(OUT / "01_setup_uptrend_events.csv", index=False)


if __name__ == "__main__":
    main()
