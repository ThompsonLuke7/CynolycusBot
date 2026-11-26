"""E4: "good news sold" (F9) and other short-history context (PLAN.md E4 + "E4 specifics",
pre-registered 2026-11-25 ~00:59 ET before running).

Reports: shared earnings calendar (signals/news/data/processed/ticker_earnings_calendar.parquet),
finite surprise_pct, beat >= +5%, miss <= -5%. Reaction = close(d+1)/close(d-1) - 1 minus SPY over
the same window (covers before-open and after-close reports), known at the close of d+1. Prior close
>= $5; |reaction| > 40% dropped and counted. Returns: split/dividend-adjusted SIP research bars
(Data/research/bars_1d_sip_adj, through 2026-09-24), then live 1d-cache daily returns (returns are
spliced, never price levels). Universe = today's calendar tickers (survivor-shaped).

Daily factors at the close of t from reports known by t within the last 60 sessions:
GNS (mean beat reaction, >= 30 beats), BNB (mean miss reaction, >= 15), GNS_tech (XLK-assigned
tickers by the latest sector snapshot, >= 10 beats; secondary, the mapping is today's).
(a) Episode lead: GNS < 0 on >= 10 of the 20 sessions before the price-vote flag (3 of 4 price
    votes) of each Nasdaq-100 drawdown >= 10% since 2016-06; base rate = share of days GNS < 0.
    Shadow candidate if it leads in >= 2 of the 3 named episodes (2024-07, 2025-02, 2026-10).
(b) Predictive (secondary): GNS + BNB added to E3's base for T1, clipped logistic, walk-forward from
    the factor's span; 2020s only; 100 block-permutation draws give the null/MDE. Reported only.

Data-validity fix after the first run (2026-11-25 ~01:10 ET): the shared calendar's REPORTED EPS
ends 2026-09-24 (its nightly stage writes only next-date snapshot rows), so the 60-session window
then only lost reports and the Oct-Nov 2026 values came from a dwindling stale sample. A factor is
now NaN when its newest report is more than STALE_SESSIONS old. Follow-up (labelled, PLAN.md E4
follow-up): the relative "weak" rule (below the trailing-252 20th percentile) and a per-year split
of the predictive gain.

    PYTHONPATH=. .venv/bin/python -W ignore research/regime_experiments_2026-11-24/e4_good_news_sold.py
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e3_factor_completeness as e3  # noqa: E402
import regime_common as rc  # noqa: E402

CAL = rc.REPO / "signals" / "news" / "data" / "processed" / "ticker_earnings_calendar.parquet"
BARS = rc.REPO / "Data" / "research" / "bars_1d_sip_adj"
SECTORS = rc.REPO / "Data" / "shared" / "market_regime" / "sector_assignments.parquet"
SPLICE_AFTER = pd.Timestamp("2026-09-24")
WINDOW = 60
STALE_SESSIONS = 10
SEED = 20261125
NAMED = {"2024-07": ("2024-07-01", "2024-08-31"), "2025-02": ("2025-02-01", "2025-04-30"),
         "2026-10": ("2026-09-15", "2026-11-30")}


def daily_returns(ticker: str) -> pd.DataFrame | None:
    from strategies.momentum_expansion.data.load_bars import load_1d

    f = BARS / f"{ticker}.parquet"
    if not f.exists():
        return None
    a = pd.read_parquet(f, columns=["timestamp", "close"])
    a.index = pd.to_datetime(a["timestamp"], utc=True).dt.tz_convert("America/New_York").dt.normalize().dt.tz_localize(None)
    a = a[~a.index.duplicated(keep="last")].sort_index()
    out = pd.DataFrame({"close": a["close"], "ret": a["close"].pct_change()})
    try:
        b = load_1d(ticker)
    except Exception:
        b = None
    if b is not None and len(b):
        bi = pd.DatetimeIndex(b.index).tz_convert("America/New_York").normalize().tz_localize(None)
        bc = pd.Series(b["close"].to_numpy(), index=bi)
        bc = bc[~bc.index.duplicated(keep="last")].sort_index()
        br = bc.pct_change()
        tail = br[br.index > SPLICE_AFTER]
        if len(tail):
            out = pd.concat([out, pd.DataFrame({"close": bc.reindex(tail.index), "ret": tail})])
    return out


def reactions() -> tuple[pd.DataFrame, dict]:
    cal = pd.read_parquet(CAL, columns=["ticker", "date", "surprise_pct"])
    cal = cal.dropna(subset=["surprise_pct"])
    cal["date"] = pd.to_datetime(cal["date"])
    cal = cal[(cal["date"] >= "2016-01-15")].drop_duplicates(["ticker", "date"])
    spy = daily_returns("SPY")
    sessions = spy.index
    last = sessions[-1]
    cal = cal[cal["date"] < last]
    rows, stats = [], {"reports_in": len(cal), "no_bars": 0, "no_window": 0, "px_lt_5": 0, "extreme": 0}
    spy_cum = (1 + spy["ret"].fillna(0)).cumprod()
    for tk, g in cal.groupby("ticker"):
        d = daily_returns(tk)
        if d is None or len(d) < 30:
            stats["no_bars"] += len(g)
            continue
        cum = (1 + d["ret"].fillna(0)).cumprod()
        idx = d.index
        for _, r in g.iterrows():
            pos = idx.searchsorted(r["date"])  # first session >= report date
            if pos < 1 or pos + 1 >= len(idx) or idx[pos] - r["date"] > pd.Timedelta(days=4):
                stats["no_window"] += 1
                continue
            d0, d1 = idx[pos - 1], idx[pos + 1]
            if d["close"].iloc[pos - 1] < 5:
                stats["px_lt_5"] += 1
                continue
            if d0 not in spy_cum.index or d1 not in spy_cum.index:
                stats["no_window"] += 1
                continue
            react = cum.iloc[pos + 1] / cum.iloc[pos - 1] - 1 - (spy_cum[d1] / spy_cum[d0] - 1)
            if abs(react) > 0.40:
                stats["extreme"] += 1
                continue
            rows.append({"ticker": tk, "report_date": r["date"], "known_at": d1, "surprise": r["surprise_pct"],
                         "reaction": react})
    out = pd.DataFrame(rows)
    stats["reports_used"] = len(out)
    return out, stats


def daily_factor(rx: pd.DataFrame, sessions: pd.DatetimeIndex, mask: pd.Series, min_n: int) -> pd.Series:
    sub = rx[mask].copy()
    pos = sessions.searchsorted(sub["known_at"])
    s = np.bincount(pos, weights=sub["reaction"], minlength=len(sessions))
    c = np.bincount(pos, minlength=len(sessions))
    ss = pd.Series(s[:len(sessions)]).rolling(WINDOW, min_periods=1).sum().to_numpy()
    cc = pd.Series(c[:len(sessions)]).rolling(WINDOW, min_periods=1).sum().to_numpy()
    i = np.arange(len(sessions))
    last = np.maximum.accumulate(np.where(c[:len(sessions)] > 0, i, -10**9))
    fresh = (i - last) <= STALE_SESSIONS
    return pd.Series(np.where((cc >= min_n) & fresh, ss / np.maximum(cc, 1), np.nan), index=sessions)


def episodes(close: pd.Series, start: str) -> list[tuple]:
    c = close.loc[start:].dropna()
    peak, peak_d, in_ep, eps, tr, tr_d = c.iloc[0], c.index[0], False, [], None, None
    for d, v in c.items():
        if not in_ep:
            if v > peak:
                peak, peak_d = v, d
            elif v / peak - 1 <= -0.10:
                in_ep, tr, tr_d = True, v, d
        else:
            if v < tr:
                tr, tr_d = v, d
            if v >= peak:
                eps.append((peak_d, tr_d, tr / peak - 1))
                in_ep, peak, peak_d = False, v, d
    if in_ep:
        eps.append((peak_d, tr_d, tr / peak - 1))
    return eps


def main() -> int:
    warnings.simplefilter("ignore")
    rc.OUT.mkdir(exist_ok=True)
    rx, stats = reactions()
    rx.to_parquet(rc.OUT / "e4_reactions.parquet")
    print("reactions", stats, flush=True)
    p, _ = rc.latest_panel()
    sessions = p.loc["2016-01-01":].index
    sec = pd.read_parquet(SECTORS).sort_values("snapshot_month").drop_duplicates("ticker", keep="last")
    tech = set(sec.loc[sec["sector_etf"] == "XLK", "ticker"])
    f = pd.DataFrame({"gns": daily_factor(rx, sessions, rx["surprise"] >= 5, 30),
                      "bnb": daily_factor(rx, sessions, rx["surprise"] <= -5, 15),
                      "gns_tech": daily_factor(rx, sessions, (rx["surprise"] >= 5) & rx["ticker"].isin(tech), 10)})
    f.to_parquet(rc.OUT / "e4_daily_factors.parquet")
    print("factor coverage", f.notna().mean().round(3).to_dict(), "GNS<0 share", round((f["gns"] < 0).mean(), 3), flush=True)
    # (a) episode lead
    v = rc.lh_votes(p, "ndx")
    flag = v[["fast_down", "below_sma50", "spx_trend", "rv"]].sum(axis=1, min_count=4) >= 3
    base_rate = {k: float((f[k].dropna() < 0).mean()) for k in f}
    ep_rows = []
    for pk, tr, dd in episodes(p["ndx"], "2016-06-01"):
        after = flag.loc[pk:tr]
        fl = after.index[after.to_numpy()][0] if after.any() else None
        row = {"peak": str(pk.date()), "trough": str(tr.date()), "drawdown": round(dd, 3),
               "price_flag": str(fl.date()) if fl is not None else None}
        if fl is not None:
            w = f.loc[:fl].iloc[-21:-1]
            for k in f:
                row[f"{k}_neg_days_of_20"] = int((w[k] < 0).sum())
                row[f"{k}_mean_pre"] = float(w[k].mean())
                row[f"{k}_leads"] = bool((w[k] < 0).sum() >= 10)
        row["named"] = next((n for n, (a, b) in NAMED.items() if a <= str(pk.date()) <= b), None)
        ep_rows.append(row)
    ep = pd.DataFrame(ep_rows)
    ep.to_csv(rc.OUT / "e4_episodes.csv", index=False)
    named = ep.dropna(subset=["named"])
    lead_named = {k: int(named.get(f"{k}_leads", pd.Series(dtype=bool)).fillna(False).sum()) for k in f}
    # (b) predictive, secondary
    F, fam, y, _ = e3.build_features("ndx")
    F = F.join(f[["gns", "bnb"]], how="left")
    cols = fam["base"] + ["gns", "bnb"]
    sub, ys, years, test_years, lost = e3.span_setup(F, y, cols)
    dec = rc.decade_of(sub.index).to_numpy()
    Xa, Xb = sub[cols].to_numpy(), sub[fam["base"]].to_numpy()
    pb = e3.walk_forward(Xb, ys, years, test_years, winsor=True)
    pa = e3.walk_forward(Xa, ys, years, test_years, winsor=True)
    real = e3.decade_improvement(ys, pb, pa, dec)
    rng = np.random.default_rng(SEED)
    nulls = []
    for _ in range(100):
        Xp = Xa.copy()
        idx = e3.block_permute_rows(len(Xa), rng)
        Xp[:, -2:] = Xa[idx][:, -2:]
        nulls.append(e3.decade_improvement(ys, pb, e3.walk_forward(Xp, ys, years, test_years, winsor=True), dec))
    nul = pd.DataFrame(nulls)
    pred = {d: {"improvement": real[d], "null_p95": float(nul[d].quantile(0.95))} for d in real}
    summary = {"reaction_stats": stats, "factor_base_rate_negative": base_rate, "named_episode_leads": lead_named,
               "SHADOW_CANDIDATE": {k: lead_named[k] >= 2 for k in f}, "predictive_secondary": pred,
               "predictive_span": {"first_row": str(sub.index[0].date()), "first_test_year": test_years[0],
                                   "rows_lost": lost}}
    # follow-up (labelled): relative weak rule and per-year predictive split
    q = f.rolling(252, min_periods=252).quantile(0.20)
    weak = (f < q).where(q.notna() & f.notna())
    fu = []
    for _, r in ep.iterrows():
        row = {"peak": r["peak"], "named": r["named"], "price_flag": r["price_flag"]}
        if isinstance(r["price_flag"], str):
            w = weak.loc[:r["price_flag"]].iloc[-21:-1]
            for k in f:
                row[f"{k}_weak_days"] = int((w[k] == 1).sum())
                row[f"{k}_evaluable"] = bool(w[k].notna().sum() >= 15)
                row[f"{k}_leads"] = bool((w[k] == 1).sum() >= 10)
        fu.append(row)
    fu = pd.DataFrame(fu)
    fu.to_csv(rc.OUT / "e4_followup_relative_rule.csv", index=False)
    nm = fu.dropna(subset=["named"])
    ll = lambda yy, pp: -(yy * np.log(np.clip(pp, 1e-6, 1 - 1e-6)) + (1 - yy) * np.log(1 - np.clip(pp, 1e-6, 1 - 1e-6)))
    ok = ~np.isnan(pb) & ~np.isnan(ys)
    per_year = pd.Series(ll(ys[ok], pb[ok]) - ll(ys[ok], pa[ok]), index=sub.index[ok])
    summary["followup"] = {
        "base_rate_weak": {k: float((weak[k] == 1).sum() / weak[k].notna().sum()) for k in f},
        "named_leads": {k: int(nm[f"{k}_leads"].fillna(False).astype(bool).sum()) for k in f},
        "named_evaluable": {k: int(nm[f"{k}_evaluable"].fillna(False).astype(bool).sum()) for k in f},
        "all_leads": {k: f"{int(fu[f'{k}_leads'].fillna(False).astype(bool).sum())}/{int(fu[f'{k}_evaluable'].fillna(False).astype(bool).sum())}" for k in f},
        "per_year_millinats": (per_year.groupby(per_year.index.year).mean() * 1000).round(1).to_dict(),
        "last_report_known_at": str(rx["known_at"].max().date()),
    }
    (rc.OUT / "e4_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    pd.set_option("display.width", 250)
    print(ep.to_string(index=False))
    print(json.dumps(summary, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
