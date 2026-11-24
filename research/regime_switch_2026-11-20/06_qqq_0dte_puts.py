"""User's trade, priced with real options: when the regime is bearish and the open is red, buy a QQQ
0DTE (or 1DTE) put and sell it an hour later, or at the close, or when it doubles.

The question (user, 2026-11-22): "since it's so volatile it seems better to just trade 0 DTE or 1 DTE
expiry options on it, so why couldn't we just use those ... the backtest would need to be like the
last month or two on 1 min bars."

Pre-registered 2026-11-22 ~02:27 ET, before any option P&L was computed. (The fitness check below
was run first on six days, which is allowed: it looks at whether prices track, not at P&L.)

Data
  Underlying  Alpaca SIP 1-minute QQQ, adjustment=raw (strikes are raw prices; the adjusted 5-minute
              bars used in 03 sit ~2% below raw in 2024 and would pick the wrong strike),
              2024-02-01..2026-11-20, Data/research/regime_switch_2026-11-20/bars_1m_raw.
              Sessions need the 09:30 and 15:59 bars and >= 380 regular-session bars (half days out).
  Options     Alpaca /v1beta1/options/bars 1Min (trade prints), expired contracts only, via
              research/options_lab/chain_cache.py (cached under Data/options_history/bars/1Min).
              No historical quotes exist (quotes endpoint 404), so prices are trade prints.
Fitness check (run before this file, on 6 days): 5-minute log returns of the ATM 0DTE put vs QQQ
  correlate -0.70 to -0.89 on liquid days. Repeated here on every contract-day at 1-minute
  resolution; days with corr > -0.3 or fewer than 100 regular-session bars are reported as
  "stale" and the results are shown with and without them.
Signals (known before entry)
  GAP    open / prior 15:59 close - 1 <= -0.5%; enter 09:35
  RED30  09:59 close / 09:30 open - 1 <= -0.5%; enter 10:00
  ALL    every session; enter 09:35 (control: what buying the put costs on an ordinary day)
Regime  7-vote market regime at the PRIOR close, from 05_intraday_structure_by_regime.py;
  BEAR = votes >= 4. Cells: all, BEAR, not BEAR; descriptive: last two months
  (2026-09-21..11-20) and the selloff (2026-10-15..11-20).
Instrument  QQQ put, strike = underlying at entry rounded to the nearest $1 (ATM).
  0DTE = expiring that session; 1DTE = expiring the next session (dropped when that expiry is
  after 2026-11-20, because unexpired contracts return 403).
Fills  entry = open of the first option bar starting in [entry, entry+2 min]; exits = open of the
  first bar in [exit, exit+3 min]. No bar in the window = missing (counted, never filled in).
  Costs 2% of premium per side (primary); 5% per side reported as a sensitivity.
Exits  H60 (entry + 60 min), EOD (15:50), BRACKET (+100% target / -50% stop on premium, time exit
  15:50; when one bar spans both levels the stop is assumed first; level fills are assumed, which
  flatters stops in a fast tape).
Reported  per cell: n, mean and median premium return, win rate, share that doubled (H60/EOD),
  t = mean / (sd / sqrt(n)) (one trade per day), MDE ~= 2.8 x SE, and the matching QQQ short
  return for the same window. 1DTE is secondary.
Limits  2024-02..2026-11 holds about three corrections (2024-07/08, 2025-02..04, 2026-10..11). BEAR
  sessions are clustered, so the effective sample is a handful of spells. Option returns are
  right-skewed: the mean is carried by a few days, which is why median and win rate are shown.
"""
from __future__ import annotations

import importlib.util
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from research.options_lab import chain_cache as cc

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OUT = HERE / "out"
RAW1M = REPO / "Data" / "research" / "regime_switch_2026-11-20" / "bars_1m_raw" / "QQQ.parquet"
ET = "America/New_York"
TH = -0.005
COST, COST_HI = 0.02, 0.05
TARGET, STOP = 1.0, -0.5
LAST_EXPIRY = pd.Timestamp("2026-11-20")
LAST2M = (pd.Timestamp("2026-09-21"), pd.Timestamp("2026-11-20"))
SELLOFF = (pd.Timestamp("2026-10-15"), pd.Timestamp("2026-11-20"))
EOD = "15:50"

_spec = importlib.util.spec_from_file_location("rs05", HERE / "05_intraday_structure_by_regime.py")
rs05 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rs05)
log = logging.getLogger("qqq_0dte")


def underlying_sessions() -> dict[pd.Timestamp, pd.DataFrame]:
    d = pd.read_parquet(RAW1M)
    ts = pd.to_datetime(d["timestamp"], utc=True).dt.tz_convert(ET)
    d = d.assign(ts=ts, day=ts.dt.normalize().dt.tz_localize(None), hm=ts.dt.strftime("%H:%M"))
    d = d[(d["hm"] >= "09:30") & (d["hm"] < "16:00")]
    if d.duplicated("ts").any():
        raise ValueError("duplicate QQQ minutes")
    out = {}
    for day, g in d.groupby("day"):
        g = g.set_index("hm")
        if len(g) >= 380 and "09:30" in g.index and "15:59" in g.index:
            out[day] = g[["ts", "open", "high", "low", "close"]]
    return out


def osi(expiry: pd.Timestamp, strike: int) -> str:
    return f"QQQ{expiry:%y%m%d}P{strike * 1000:08d}"


def option_bars(symbol: str, day: pd.Timestamp) -> pd.DataFrame:
    try:
        b = cc.fetch_bars([symbol], "1Min", f"{day:%Y-%m-%d}T13:00:00Z", f"{day:%Y-%m-%d}T21:00:00Z")
    except Exception as exc:  # noqa: BLE001 - recorded, never filled in
        log.warning("fetch failed %s %s: %s", symbol, day.date(), exc)
        return pd.DataFrame()
    if b.empty:
        return b
    b = b.assign(ts=pd.to_datetime(b["t"], utc=True).dt.tz_convert(ET)).sort_values("ts")
    b["hm"] = b["ts"].dt.strftime("%H:%M")
    return b[(b["hm"] >= "09:30") & (b["hm"] < "16:00")].set_index("hm")


def first_open(b: pd.DataFrame, start: str, window_min: int) -> tuple[float, str] | tuple[None, None]:
    t0 = pd.Timestamp(f"2000-01-01 {start}")
    for k in range(window_min + 1):
        hm = (t0 + pd.Timedelta(minutes=k)).strftime("%H:%M")
        if hm in b.index:
            return float(b.at[hm, "o"]), hm
    return None, None


def fitness(b: pd.DataFrame, u: pd.DataFrame) -> tuple[int, float]:
    j = pd.concat([b["c"].rename("o"), u["close"].rename("u")], axis=1, join="inner")
    r = np.log(j).diff().dropna()
    c = float(np.corrcoef(r["o"], r["u"])[0, 1]) if len(r) > 20 else np.nan
    return len(b), c


def trade(b: pd.DataFrame, u: pd.DataFrame, entry: str) -> dict:
    px, e_hm = first_open(b, entry, 2)
    if px is None or px <= 0:
        return {"status": "no_entry_print"}
    e_ts = pd.Timestamp(f"2000-01-01 {entry}")
    h60 = (e_ts + pd.Timedelta(minutes=60)).strftime("%H:%M")
    rec = {"status": "ok", "entry_px": px, "entry_hm": e_hm}
    for name, hm in (("H60", h60), ("EOD", EOD)):
        x, _ = first_open(b, hm, 3)
        rec[f"px_{name}"] = x
        rec[f"u_{name}"] = -(u.at[hm, "open"] / u.at[entry, "open"] - 1) if hm in u.index and entry in u.index else np.nan
    # bracket on premium, from the bar after entry to the EOD exit
    after = b[(b.index > e_hm) & (b.index < EOD)]
    br = None
    for _, r in after.iterrows():
        if r["l"] <= px * (1 + STOP):
            br = px * (1 + STOP)
            break
        if r["h"] >= px * (1 + TARGET):
            br = px * (1 + TARGET)
            break
    rec["px_BRACKET"] = br if br is not None else rec["px_EOD"]
    return rec


def net(entry: float, exit_: float | None, cost: float) -> float:
    if exit_ is None or not np.isfinite(exit_):
        return np.nan
    return exit_ * (1 - cost) / (entry * (1 + cost)) - 1


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    ss = underlying_sessions()
    days = sorted(ss)
    v = rs05.regime_votes(pd.DatetimeIndex(days[1:]))
    rows, fit_rows = [], []
    started = time.time()
    for i in range(1, len(days)):
        day, prev = days[i], days[i - 1]
        u = ss[day]
        gap = u.at["09:30", "open"] / ss[prev].at["15:59", "close"] - 1
        od30 = u.at["09:59", "close"] / u.at["09:30", "open"] - 1
        base = {"day": day, "gap": gap, "od30": od30, "votes": v.at[day, "votes"], "BEAR": v.at[day, "BEAR"]}
        nxt = days[i + 1] if i + 1 < len(days) else None
        plans = [("ALL", "09:35")]
        if gap <= TH:
            plans.append(("GAP", "09:35"))
        if od30 <= TH:
            plans.append(("RED30", "10:00"))
        for rule, entry in plans:
            spot = u.at[entry, "open"]
            strike = int(round(spot))
            for tenor, exp in (("0DTE", day), ("1DTE", nxt)):
                if exp is None or exp > LAST_EXPIRY:
                    continue
                if tenor == "1DTE" and rule == "ALL" and base["BEAR"] != 1:
                    continue  # 1DTE control only where it is compared (BEAR sessions)
                sym = osi(exp, strike)
                b = option_bars(sym, day)
                if b.empty:
                    rows.append({**base, "rule": rule, "tenor": tenor, "symbol": sym, "status": "no_bars"})
                    continue
                n_bars, corr = fitness(b, u)
                fit_rows.append({"day": day, "symbol": sym, "tenor": tenor, "rth_bars": n_bars, "corr_1m": corr})
                rows.append({**base, "rule": rule, "tenor": tenor, "symbol": sym, "spot": spot, "strike": strike,
                             "rth_bars": n_bars, "corr_1m": corr, **trade(b, u, entry)})
        if i % 100 == 0:
            print(f"  {i}/{len(days)} sessions, {time.time() - started:.0f}s", flush=True)

    t = pd.DataFrame(rows)
    ok = t["status"] == "ok"
    for name in ("H60", "EOD", "BRACKET"):
        t[f"ret_{name}"] = [net(e, x, COST) if o else np.nan for e, x, o in zip(t.get("entry_px"), t.get(f"px_{name}"), ok)]
        t[f"ret_{name}_hi"] = [net(e, x, COST_HI) if o else np.nan for e, x, o in zip(t.get("entry_px"), t.get(f"px_{name}"), ok)]
    t["stale"] = (t["rth_bars"] < 100) | (t["corr_1m"] > -0.3) | t["corr_1m"].isna()
    t.to_csv(OUT / "qqq_puts_trades.csv", index=False)
    pd.DataFrame(fit_rows).to_csv(OUT / "qqq_puts_fitness.csv", index=False)

    out = []
    cells = {"all": lambda d: d["BEAR"].notna(), "BEAR": lambda d: d["BEAR"] == 1, "not BEAR": lambda d: d["BEAR"] == 0,
             "last 2 months": lambda d: d["day"].between(*LAST2M),
             "last 2 months, BEAR": lambda d: d["day"].between(*LAST2M) & (d["BEAR"] == 1),
             "selloff 10-15..11-20": lambda d: d["day"].between(*SELLOFF)}
    for (rule, tenor), g in t.groupby(["rule", "tenor"]):
        for cell, f in cells.items():
            for clean in (False, True):
                sub = g[f(g)]
                n_try = len(sub)
                sub = sub[sub["status"] == "ok"]
                if clean:
                    sub = sub[~sub["stale"]]
                for ex in ("H60", "EOD", "BRACKET"):
                    r = sub[f"ret_{ex}"].dropna()
                    if len(r) < 3:
                        continue
                    se = r.std(ddof=1) / np.sqrt(len(r))
                    uq = sub[f"u_{ex if ex != 'BRACKET' else 'EOD'}"]
                    out.append({"rule": rule, "tenor": tenor, "cell": cell, "excl_stale": clean, "exit": ex,
                                "sessions": n_try, "n": len(r), "mean": r.mean(), "median": r.median(),
                                "win": (r > 0).mean(), "doubled": (r >= 1.0).mean(), "t": r.mean() / se,
                                "mde": 2.8 * se, "mean_cost5": sub[f"ret_{ex}_hi"].mean(),
                                "sum_per_1usd": r.sum(), "best": r.max(), "worst": r.min(),
                                "qqq_short_mean": uq.mean()})
    res = pd.DataFrame(out)
    res.to_csv(OUT / "qqq_puts_summary.csv", index=False)

    # Added after the first results (descriptive, not pre-registered): is the BEAR-day result one
    # episode? Windows are the corrections in the sample, named by calendar, not by outcome.
    def episode(d: pd.Timestamp) -> str:
        if d >= LAST2M[0]:
            return "D last 2 months"
        if pd.Timestamp("2024-07-01") <= d <= pd.Timestamp("2024-09-30"):
            return "A 2024 Jul-Sep"
        if pd.Timestamp("2025-02-15") <= d <= pd.Timestamp("2025-05-31"):
            return "B 2025 Feb-May"
        return "C other"
    e = t[ok & (t["tenor"] == "0DTE") & (t["BEAR"] == 1)].copy()
    e["episode"] = e["day"].map(episode)
    e["excl_last2m"] = e["episode"] != "D last 2 months"
    ep_rows = []
    for (rule, grp), g in list(e.groupby(["rule", "episode"])) + [((r_, "ALL except last 2 months"), g_)
                                                               for r_, g_ in e[e["excl_last2m"]].groupby("rule")]:
        ep_rows.append({"rule": rule, "episode": grp, "n": len(g), "h60_mean": g["ret_H60"].mean(),
                        "h60_median": g["ret_H60"].median(), "eod_mean": g["ret_EOD"].mean(),
                        "eod_median": g["ret_EOD"].median(), "eod_win": (g["ret_EOD"] > 0).mean(),
                        "bracket_mean": g["ret_BRACKET"].mean(), "qqq_short_eod": g["u_EOD"].mean()})
    pd.DataFrame(ep_rows).to_csv(OUT / "qqq_puts_by_episode.csv", index=False)
    f = pd.DataFrame(fit_rows)
    print(f"sessions={len(days) - 1} trades={len(t)} ok={int(ok.sum())} "
          f"status={t['status'].value_counts().to_dict()} stale={int(t.loc[ok, 'stale'].sum())} "
          f"median corr_1m={f['corr_1m'].median():.2f} median rth bars={f['rth_bars'].median():.0f} "
          f"elapsed={time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
