"""E0: long-history daily data for the regime experiments (see PLAN.md).

Fetches index, volatility, rate, dollar and commodity series (Yahoo) and, when a free FRED
API key is set (FRED_API_KEY in the environment or .env), credit/macro series from FRED.
Raw downloads are immutable: each run writes a new dated folder and never overwrites an
earlier one. A normalized panel is written next to them.

    .venv/bin/python research/regime_experiments_2026-11-24/e0_fetch_long_history.py

Time rules
  * Market series (Yahoo, and FRED market-priced series such as spreads and yields) are known at
    that session's close: available_at = session date 16:30 ET.
  * The session in progress is DROPPED: a row dated today is kept only after 17:00 ET.
  * FRED macro releases with a publication lag (NFCI weekly, ICSA weekly claims) get an explicit
    availability lag (see FRED_SERIES) instead of their reference date. Revised series (NFCI) are
    flagged: the value here is the latest vintage, not the one known at the time.
Fitness checks (fail fast): known crash-day returns, no duplicate dates, increasing dates,
  gap report, total-return index >= price index drift.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import sys
import urllib.request
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "Data" / "research" / "regime_40y"
ET = ZoneInfo("America/New_York")

YAHOO = {
    # equity indexes
    "^GSPC": "spx", "^SP500TR": "spx_tr", "^NDX": "ndx", "^IXIC": "nasdaq_comp", "^RUT": "rut",
    # volatility
    "^VIX": "vix", "^VIX3M": "vix3m",
    # rates (Yahoo quotes these x10 for ^TNX/^FVX in some eras; normalised below)
    "^TNX": "ust10y", "^FVX": "ust5y", "^IRX": "tbill3m",
    # dollar, commodities, credit ETF
    "DX-Y.NYB": "dxy", "CL=F": "wti", "HYG": "hyg", "IEF": "ief",
    # sector SPDRs (1998-12 onward) and equal weight, for breadth / dispersion
    "XLK": "xlk", "XLF": "xlf", "XLE": "xle", "XLV": "xlv", "XLY": "xly", "XLP": "xlp",
    "XLI": "xli", "XLB": "xlb", "XLU": "xlu", "RSP": "rsp",
}
# series id -> (column, availability lag in calendar days after the reference date, revised?)
FRED_SERIES = {
    "BAA10Y": ("baa_10y_spread", 1, False),        # Moody's Baa - 10y, daily, 1986-
    "BAMLH0A0HYM2": ("hy_oas", 1, False),          # ICE BofA HY OAS, daily, 1996-12-
    "T10Y3M": ("curve_10y3m", 1, False),           # 10y - 3m, daily, 1982-
    "DFF": ("fed_funds", 1, False),                # effective fed funds, daily, 1954-
    "NFCI": ("nfci", 5, True),                     # Chicago Fed NFCI, weekly (Fri ref, Wed release)
    "ICSA": ("initial_claims", 5, False),          # weekly claims (Sat ref, Thu release)
}
CRASH_CHECKS = {("spx", "1987-10-19"): -0.204, ("spx", "2008-10-15"): -0.090, ("spx", "2020-03-16"): -0.120}


def now_et() -> dt.datetime:
    return dt.datetime.now(ET)


def last_complete_session(now: dt.datetime) -> dt.date:
    return now.date() if now.time() >= dt.time(17, 0) else now.date() - dt.timedelta(days=1)


def fetch_yahoo(symbol: str) -> pd.DataFrame:
    import yfinance as yf

    d = yf.Ticker(symbol).history(period="max", interval="1d", auto_adjust=False, timeout=30)
    if d.empty:
        raise RuntimeError(f"Yahoo returned no rows for {symbol}")
    d.index = pd.DatetimeIndex(d.index.tz_localize(None) if d.index.tz is not None else d.index).normalize()
    return d[["Open", "High", "Low", "Close", "Volume"]].rename(columns=str.lower)


def _fred_key() -> str | None:
    key = os.environ.get("FRED_API_KEY")
    env = REPO / ".env"
    if not key and env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("FRED_API_KEY="):
                key = line.split("=", 1)[1].strip().strip('"').strip("'")
    return key or None


def fetch_fred(series_id: str, key: str) -> pd.Series:
    url = ("https://api.stlouisfed.org/fred/series/observations"
           f"?series_id={series_id}&api_key={key}&file_type=json")
    req = urllib.request.Request(url, headers={"User-Agent": "research-client"})
    payload = json.loads(urllib.request.urlopen(req, timeout=60).read())
    obs = pd.DataFrame(payload["observations"])
    s = pd.to_numeric(obs["value"], errors="coerce")
    s.index = pd.to_datetime(obs["date"])
    return s.dropna()


def check_frame(name: str, d: pd.DataFrame) -> dict:
    if d.index.duplicated().any():
        raise ValueError(f"{name}: duplicate dates")
    if not d.index.is_monotonic_increasing:
        raise ValueError(f"{name}: dates not increasing")
    gaps = d.index.to_series().diff().dt.days
    return {"rows": len(d), "first": str(d.index[0].date()), "last": str(d.index[-1].date()),
            "max_gap_days": int(gaps.max()) if len(gaps) > 1 else 0,
            "gaps_over_5d": int((gaps > 5).sum())}


def main() -> int:
    now = now_et()
    cutoff = pd.Timestamp(last_complete_session(now))
    run_dir = ROOT / "raw" / now.strftime("%Y%m%dT%H%M")
    if run_dir.exists():
        raise SystemExit(f"{run_dir} exists; raw downloads are immutable")
    run_dir.mkdir(parents=True)
    manifest = {"fetched_at_et": now.isoformat(), "last_complete_session": str(cutoff.date()),
                "series": {}, "skipped": {}}

    closes = {}
    for sym, col in YAHOO.items():
        try:
            d = fetch_yahoo(sym)
        except Exception as exc:  # recorded, never filled in
            manifest["skipped"][sym] = f"{type(exc).__name__}: {exc}"[:200]
            print(f"  {sym:9s} SKIPPED {manifest['skipped'][sym]}", flush=True)
            continue
        dropped = int((d.index > cutoff).sum())
        d = d[d.index <= cutoff]
        d.to_parquet(run_dir / f"yahoo_{col}.parquet")
        manifest["series"][col] = {"source": f"yahoo:{sym}", "dropped_incomplete_rows": dropped, **check_frame(col, d)}
        closes[col] = d["close"]
        print(f"  {sym:9s} -> {col:12s} {manifest['series'][col]}", flush=True)

    key = _fred_key()
    fred_cols = {}
    if key is None:
        manifest["skipped"]["FRED"] = "no FRED_API_KEY (free at fred.stlouisfed.org/docs/api/api_key.html)"
        print("  FRED skipped: set FRED_API_KEY (free) to add credit spreads before 2007 and macro series", flush=True)
    else:
        for sid, (col, lag, revised) in FRED_SERIES.items():
            try:
                s = fetch_fred(sid, key)
            except Exception as exc:
                manifest["skipped"][sid] = f"{type(exc).__name__}"
                continue
            s.to_frame(col).to_parquet(run_dir / f"fred_{col}.parquet")
            fred_cols[col] = (s, lag)
            manifest["series"][col] = {"source": f"fred:{sid}", "availability_lag_days": lag,
                                       "latest_vintage_not_pit": revised,
                                       **check_frame(col, s.to_frame(col))}

    # fitness checks
    for (col, day), expected in CRASH_CHECKS.items():
        r = closes[col].pct_change().get(pd.Timestamp(day))
        if r is None or abs(r - expected) > 0.01:
            raise ValueError(f"fitness: {col} {day} return {r} != {expected}")
    if "spx_tr" in closes:
        both = pd.concat([closes["spx"], closes["spx_tr"]], axis=1, join="inner").dropna()
        drift = (both.iloc[-1, 1] / both.iloc[0, 1]) / (both.iloc[-1, 0] / both.iloc[0, 0])
        if drift < 1.0:
            raise ValueError("fitness: total-return index grew slower than the price index")
        manifest["checks_spx_tr_over_price"] = float(drift)

    # normalized panel on the S&P session calendar; nothing is forward-filled here
    cal = closes["spx"].index
    panel = pd.DataFrame(index=cal)
    for col, s in closes.items():
        panel[col] = s.reindex(cal)
    for col in ("ust10y", "ust5y"):
        if col in panel:  # Yahoo quotes some eras x10; yields above 25% are impossible
            panel[col] = panel[col].where(panel[col] < 25, panel[col] / 10)
    for col, (s, lag) in fred_cols.items():
        avail = s.copy()
        avail.index = avail.index + pd.Timedelta(days=lag)
        panel[col] = avail.reindex(cal, method="ffill")  # carried only from its availability date
        panel[f"{col}_age_days"] = (cal.to_series() - avail.index.to_series().reindex(cal, method="ffill")).dt.days.to_numpy()
    panel.index.name = "date"
    panel["available_at"] = (panel.index + pd.Timedelta(hours=16, minutes=30)).tz_localize(ET).tz_convert("UTC")
    out = ROOT / f"panel_{now.strftime('%Y%m%dT%H%M')}.parquet"
    panel.to_parquet(out)
    manifest["panel"] = {"path": str(out.relative_to(REPO)), "rows": len(panel),
                         "first": str(panel.index[0].date()), "last": str(panel.index[-1].date()),
                         "null_share": panel.drop(columns="available_at").isna().mean().round(3).to_dict()}
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    print(f"panel {out.relative_to(REPO)} rows={len(panel)} {panel.index[0].date()}..{panel.index[-1].date()}")
    print(f"skipped: {list(manifest['skipped'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
