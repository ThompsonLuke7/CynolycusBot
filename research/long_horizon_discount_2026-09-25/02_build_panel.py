"""Weekly point-in-time panel for the long-horizon (Roth) ranker.

Decision time = the close of the last session of each week. Entry = the next session's OPEN.
Label = forward return to the close H sessions later, minus SPY over the identical window.

Features are continuous, so thresholds are LEARNED rather than hand-set. For example,
"came within ~5% of the 200 SMA" is min_low_vs_sma200_20, and the model picks the cut.

Inputs (all point-in-time):
  Data/research/bars_1d_sip_adj/            (SIP volume, split+div adjusted, 2016+)
  Data/research/fundamentals/sec_quarterly_facts.parquet   (as-first-reported, available_at=filed)
  signals/news/data/processed/ticker_earnings_calendar.parquet (EPS surprise; see caveat)
Output: Data/research/long_horizon/panel_weekly.parquet

--pit (added 2026-10-03) builds panel_weekly_pit.parquet from the point-in-time universe:
every security in Data/research/bars_1d_sip_adj_pit (scripts/research_data/build_pit_universe.py),
including delisted names and earlier lives of reused tickers (ids like BBBY~1). The top-1000 rank is
then taken among everything that traded that week. Fundamentals and earnings join on today's
tickers only, so they are NaN for dead names; use the PIT panel for price-based arms.

Caveats recorded in README:
  * The universe is today's list, so delisted names are absent and dip-buying is flattered.
  * yfinance eps_estimate is the consensus as Yahoo stores it now and may not be the
    point-in-time consensus. surprise_* features are flagged as possibly leaky.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
BARS = REPO / "Data" / "research" / "bars_1d_sip_adj"
FUND = REPO / "Data" / "research" / "fundamentals" / "sec_quarterly_facts.parquet"
CAL = REPO / "signals" / "news" / "data" / "processed" / "ticker_earnings_calendar.parquet"
UNIVERSE_CSV = REPO / "Data" / "shared" / "universe" / "shared_universe.csv"
OUT = REPO / "Data" / "research" / "long_horizon" / "panel_weekly.parquet"
PIT_BARS = REPO / "Data" / "research" / "bars_1d_sip_adj_pit"
PIT_SECURITIES = REPO / "Data" / "research" / "pit_universe" / "securities.parquet"
PIT_OUT = REPO / "Data" / "research" / "long_horizon" / "panel_weekly_pit.parquet"
HOLDS = (63, 126)
TOP_N_DV = 1000
ETF_SET = {"SPY", "QQQ", "IWM", "DIA", "VTI", "VOO", "RSP", "MDY", "VUG", "SCHG", "MGK"}


def load_bars(t: str) -> pd.DataFrame | None:
    p = BARS / f"{t}.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    df["date"] = (pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("America/New_York")
                  .dt.normalize().dt.tz_localize(None))
    return df.set_index("date")[["open", "high", "low", "close", "volume"]].sort_index()


def tmo(df: pd.DataFrame, length=14, calc=5, smooth=3):
    c, o = df["close"], df["open"]
    data = sum(np.sign(c - o.shift(i)).fillna(0) for i in range(length))
    main = data.ewm(span=calc, adjust=False).mean().ewm(span=smooth, adjust=False).mean()
    return main, main.ewm(span=smooth, adjust=False).mean()


def tech_features(df: pd.DataFrame, spy: pd.DataFrame) -> pd.DataFrame:
    c, lo = df["close"], df["low"]
    sma200, sma50 = c.rolling(200).mean(), c.rolling(50).mean()
    hi252, lo252 = c.rolling(252).max(), c.rolling(252).min()
    main, sig = tmo(df)
    lr = np.log(c).diff()
    spy_c = spy["close"].reindex(df.index)
    f = pd.DataFrame(index=df.index)
    f["dist_sma200"] = c / sma200 - 1
    f["min_low_vs_sma200_20"] = (lo / sma200 - 1).rolling(20).min()   # fuzzy "tagged the 200"
    f["dist_sma50"] = c / sma50 - 1
    f["sma200_slope_60"] = sma200 / sma200.shift(60) - 1
    f["sma50_vs_sma200"] = sma50 / sma200 - 1
    f["dd_52w_high"] = c / hi252 - 1
    f["up_52w_low"] = c / lo252 - 1
    f["dd_max_63"] = c / c.rolling(63).max() - 1                         # size of the recent pullback
    # "Compounder" profile, backward-looking only (the AMZN-2023+ shape): how
    # persistently the long trend has held, and how deep the worst recent break was.
    ema200 = c.ewm(span=200, adjust=False, min_periods=200).mean()
    f["dist_ema200"] = c / ema200 - 1
    f["min_low_vs_ema200_20"] = (lo / ema200 - 1).rolling(20).min()
    f["pct_above_sma200_252"] = (c > sma200).astype(float).where(sma200.notna()).rolling(252, min_periods=126).mean()
    f["sma200_slope_252"] = sma200 / sma200.shift(252) - 1
    f["max_dd_504"] = (c / c.rolling(504, min_periods=252).max() - 1).rolling(504, min_periods=252).min()
    for n in (5, 21, 63, 126):
        f[f"ret_{n}"] = c / c.shift(n) - 1
    f["mom_12_1"] = c.shift(21) / c.shift(252) - 1
    f["rs_spy_63"] = f["ret_63"] - (spy_c / spy_c.shift(63) - 1)
    f["rv_21"] = lr.rolling(21).std() * np.sqrt(252)
    f["rv_63"] = lr.rolling(63).std() * np.sqrt(252)
    f["rv_ratio"] = f["rv_21"] / f["rv_63"]
    f["tmo_main"] = main
    f["tmo_min_10"] = main.rolling(10).min()
    f["tmo_minus_sig"] = main - sig
    turn = (main > sig) & (main.shift(1) <= sig.shift(1))
    f["days_since_tmo_turn"] = turn.astype(int).groupby(turn.cumsum()).cumcount().where(turn.cumsum() > 0)
    f["vol_trend"] = df["volume"].rolling(20).mean() / df["volume"].rolling(120).mean()
    f["dv60"] = (c * df["volume"]).rolling(60).median()
    # labels: next-open entry, close exit, minus SPY over the identical window
    nxt_open, s_open = df["open"].shift(-1), spy["open"].reindex(df.index).shift(-1)
    for h in HOLDS:
        r = c.shift(-h) / nxt_open - 1
        rs = spy_c.shift(-h) / s_open - 1
        f[f"fwd_ret_{h}"] = r
        f[f"fwd_xs_{h}"] = r - rs
    f["valid"] = sma200.notna() & hi252.notna()
    return f


def fundamentals_asof() -> pd.DataFrame:
    """One row per (ticker, filing): the latest-quarter features available from that filing on."""
    fa = pd.read_parquet(FUND)
    wide = fa.pivot_table(index=["ticker", "period_end"], columns="metric", values="value", aggfunc="first")
    avail = fa.groupby(["ticker", "period_end"])["available_at"].max()
    q = wide.join(avail).reset_index().sort_values(["ticker", "period_end"])
    if "gross_profit" not in q:
        q["gross_profit"] = np.nan
    if "cost_of_revenue" in q:
        q["gross_profit"] = q["gross_profit"].fillna(q["revenue"] - q["cost_of_revenue"])
    q["gm"] = q["gross_profit"] / q["revenue"]
    q["opm"] = q["operating_income"] / q["revenue"]
    rows = []
    for t, g in q.groupby("ticker"):
        g = g.set_index("period_end").sort_index()
        # year-ago quarter by DATE (period_end - ~364d), robust to missing quarters
        prev = g.reindex(g.index - pd.Timedelta(days=364), method="nearest",
                         tolerance=pd.Timedelta(days=20))
        prev.index = g.index
        out = pd.DataFrame(index=g.index)
        out["rev_yoy"] = g["revenue"] / prev["revenue"] - 1
        out["gm_q"] = g["gm"]
        out["gm_yoy_chg"] = g["gm"] - prev["gm"]
        out["opm_q"] = g["opm"]
        out["opm_yoy_chg"] = g["opm"] - prev["opm"]
        out["eps_yoy"] = (g["eps_diluted"] - prev["eps_diluted"]) / prev["eps_diluted"].abs().clip(lower=0.05)
        out["ni_positive"] = (g["net_income"] > 0).astype(float).where(g["net_income"].notna())
        out["rev_yoy_prev"] = out["rev_yoy"].shift(1)
        out["rev_accel"] = out["rev_yoy"] - out["rev_yoy_prev"]
        ttm = g["revenue"].rolling(4, min_periods=4).sum()
        out["rev_ttm_yoy"] = ttm / ttm.shift(4) - 1
        out["log_rev_ttm"] = np.log(ttm.where(ttm > 0))
        # Available only when every input row has been filed. The year-ago rows were
        # filed earlier by construction, so this row's own filing date binds.
        out["available_at"] = g["available_at"]
        out["ticker"] = t
        rows.append(out.reset_index())
    f = pd.concat(rows, ignore_index=True).dropna(subset=["available_at"])
    return f.sort_values("available_at")


def surprise_asof() -> pd.DataFrame:
    cal = pd.read_parquet(CAL)
    cal = cal[cal["reported_eps"].notna()].copy()
    cal["date"] = pd.to_datetime(cal["date"]).dt.tz_localize(None).dt.normalize()
    cal = cal.sort_values(["ticker", "date"]).drop_duplicates(["ticker", "date"])
    cal["surprise_last"] = cal["surprise_pct"].clip(-200, 200)
    cal["surprise_mean4"] = cal.groupby("ticker")["surprise_last"].transform(lambda s: s.rolling(4, min_periods=2).mean())
    cal["earn_date"] = cal["date"]
    return cal[["ticker", "date", "earn_date", "surprise_last", "surprise_mean4"]].sort_values("date")


def main() -> None:
    global BARS, OUT
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pit", action="store_true", help="point-in-time universe (delisted names included)")
    if ap.parse_args().pit:
        BARS, OUT = PIT_BARS, PIT_OUT
        sec = pd.read_parquet(PIT_SECURITIES)
        tickers = sorted(sec.loc[sec["kind"] == "stock", "sec_id"])
    else:
        uni = pd.read_csv(UNIVERSE_CSV)
        tickers = sorted(uni.loc[uni["type"].fillna("Stock") != "ETF", "ticker"].dropna().astype(str).str.upper())
    spy = load_bars("SPY")
    weekly = spy.index.to_series().groupby(spy.index.to_period("W")).max()
    weekly = pd.DatetimeIndex(weekly.values)
    frames = []
    for t in tickers:
        if t in ETF_SET:
            continue
        df = load_bars(t)
        if df is None or len(df) < 300:
            continue
        f = tech_features(df, spy)
        f = f[f["valid"]].reindex(weekly.intersection(f.index)).dropna(subset=["dist_sma200"])
        f["ticker"] = t
        frames.append(f)
    panel = pd.concat(frames).rename_axis("date").reset_index()
    # No price floor: adjusted history embeds FUTURE reverse splits (FCEL 2019 at $0.20
    # reads ~$50), so a floor or a price feature would leak distress. Dollar volume is
    # adjustment-invariant, so the rank alone defines the tradable universe.
    panel["dv_rank"] = panel.groupby("date")["dv60"].rank(ascending=False)
    panel = panel[panel["dv_rank"] <= TOP_N_DV].drop(columns=["valid"])

    # market context (same for every row on a date; useful only via interactions)
    sc = spy["close"]
    mkt = pd.DataFrame({"spy_dist_sma200": sc / sc.rolling(200).mean() - 1,
                        "spy_ret_63": sc / sc.shift(63) - 1})
    panel = panel.merge(mkt, left_on="date", right_index=True, how="left")

    panel = panel.sort_values("date")
    fund = fundamentals_asof()
    panel = pd.merge_asof(panel, fund.drop(columns=["period_end"]).rename(columns={"available_at": "fund_avail"}),
                          left_on="date", right_on="fund_avail", by="ticker", direction="backward")
    panel["days_since_filing"] = (panel["date"] - panel["fund_avail"]).dt.days
    panel.loc[panel["days_since_filing"] > 200, [c for c in fund.columns if c not in ("ticker", "available_at", "period_end")]] = np.nan
    if CAL.exists():
        sur = surprise_asof()
        panel = pd.merge_asof(panel, sur.drop(columns=["date"]), left_on="date", right_on="earn_date",
                              by="ticker", direction="backward")
        panel["days_since_earnings"] = (panel["date"] - panel["earn_date"]).dt.days
        stale = panel["days_since_earnings"] > 200
        panel.loc[stale, ["surprise_last", "surprise_mean4"]] = np.nan
    OUT.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(OUT, index=False)
    print(f"panel {panel.shape}, tickers {panel.ticker.nunique()}, dates {panel.date.nunique()} "
          f"{panel.date.min().date()}..{panel.date.max().date()}")
    print("null rates:\n" + panel.isna().mean().sort_values(ascending=False).head(25).round(3).to_string())


if __name__ == "__main__":
    main()
