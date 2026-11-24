"""Are the intraday structure module's exits what keeps it from riding moves?

The question (user, 2026-11-23): "for intraday structure, why is it closing itself so quickly?"

Code facts established before this script (strategies/intraday_structure):
  * features.atr = mean of the last 14 ONE-MINUTE true ranges (~0.05% of price).
  * detectors/exhaustion.py closes a running setup when price is 0.25 of that ATR on the wrong side of
    session VWAP ("extension conditions deteriorated"): 1,528 of the 1,611 such exits in the ledger.
  * target_manager._tighten_structure_stop trails the stop to the last-5-bar swing -/+ 0.35 ATR.
  * Median live hold 2-5 minutes; median risk 0.14% of price; modeled costs 12 bp = 0.86 R.

Pre-registered 2026-11-23 00:40 ET, before any counterfactual was computed.

Data   Setups: Data/inference/intraday_structure/closed_setups.jsonl (2026-08-27..11-20).
       Path:   Data/archive/intraday_1m/bars_YYYYMMDD.jsonl, the module's own streamed 1-minute bars
               (same feed the engine saw). Sessions without an archive file are skipped and counted.
Same entries, different exits (gross, on the underlying, from the ledger entry price):
  ACTUAL      the engine's own exit (ledger gross_return)
  STRUCT      keep only the INITIAL structural stop (ledger initial_invalidation); no exhaustion
              overlay, no trailing; exit at the 15:55 close if never stopped. A bar whose range
              crosses the stop exits at the stop, or at that bar's open if it opened through it.
  T30, T60    exit at the close of the bar 30 / 60 minutes after entry (no stop)
  EOD         exit at the 15:55 close (no stop)
  A time exit needs a bar within 5 minutes of its target time, else the setup is "missing" for that
  exit (counted, never filled in). Entries after 15:25 ET are excluded from T30/T60/EOD/STRUCT.
Fitness check  the archive's close at the entry minute vs the ledger entry price (median abs gap).
Reported  mean / median gross return, win rate, net of 12 bp, by direction, by the 7-vote BEAR state
  (05), and session-clustered t (session means, as in 05) with MDE ~= 2.8 x SE.
Limits    one period (Aug-Nov 2026). IEX-streamed bars are sparse on thin names: exits fall back to
  the nearest bar within 5 minutes. Overlapping setups on the same ticker are not de-duplicated
  (the ledger counts them separately, so this does too).
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OUT = HERE / "out"
LEDGER = REPO / "Data" / "inference" / "intraday_structure" / "closed_setups.jsonl"
ARCHIVE = REPO / "Data" / "archive" / "intraday_1m"
ET = "America/New_York"
COST = 0.0012
TOL = pd.Timedelta(minutes=5)
EOD_HM = "15:55"
LAST_ENTRY_HM = "15:25"

_spec = importlib.util.spec_from_file_location("rs05", HERE / "05_intraday_structure_by_regime.py")
rs05 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rs05)


def load_day(path: Path, tickers: set[str]) -> dict[str, pd.DataFrame]:
    keep = []
    with path.open() as fh:
        for line in fh:
            # cheap prefilter before json parsing; the symbol field is always present
            i = line.find('"symbol":"')
            if i < 0:
                continue
            sym = line[i + 10: line.find('"', i + 10)]
            if sym in tickers:
                keep.append(json.loads(line))
    if not keep:
        return {}
    d = pd.DataFrame(keep)
    d["ts"] = pd.to_datetime(d["timestamp"], utc=True)
    d = d.sort_values(["symbol", "ts", "arrival_at"]).drop_duplicates(["symbol", "ts"], keep="last")
    return {s: g.set_index("ts")[["open", "high", "low", "close"]] for s, g in d.groupby("symbol")}


def bar_near(b: pd.DataFrame, t: pd.Timestamp) -> pd.Series | None:
    """First bar at or after t within TOL (bar-start timestamps)."""
    j = b.index.searchsorted(t)
    if j < len(b) and b.index[j] - t <= TOL:
        return b.iloc[j]
    return None


def counterfactuals(s: pd.Series, b: pd.DataFrame) -> dict:
    side = 1.0 if s["direction"] == "long" else -1.0
    e_ts, e_px = s["entry_ts"], float(s["entry_price"])
    rec: dict = {}
    eb = bar_near(b, e_ts)
    rec["entry_gap"] = abs(float(eb["close"]) / e_px - 1) if eb is not None else np.nan
    day = e_ts.tz_convert(ET).normalize()
    eod_ts = (day + pd.Timedelta(hours=15, minutes=55)).tz_convert("UTC")
    if e_ts.tz_convert(ET).strftime("%H:%M") > LAST_ENTRY_HM:
        return rec
    for name, mins in (("T30", 30), ("T60", 60)):
        x = bar_near(b, e_ts + pd.Timedelta(minutes=mins))
        rec[f"ret_{name}"] = side * (float(x["close"]) / e_px - 1) if x is not None else np.nan
    x = bar_near(b, eod_ts)
    rec["ret_EOD"] = side * (float(x["close"]) / e_px - 1) if x is not None else np.nan
    # STRUCT: initial stop only, from the bar after the entry minute to the EOD bar
    stop = float(s["initial_invalidation"])
    path = b[(b.index > e_ts) & (b.index <= eod_ts)]
    exit_px = None
    for _, r in path.iterrows():
        if side > 0 and r["low"] <= stop:
            exit_px = min(stop, float(r["open"]))
            break
        if side < 0 and r["high"] >= stop:
            exit_px = max(stop, float(r["open"]))
            break
    if exit_px is None and x is not None:
        exit_px = float(x["close"])
    rec["ret_STRUCT"] = side * (exit_px / e_px - 1) if exit_px is not None else np.nan
    rec["struct_stopped"] = exit_px is not None and x is not None and exit_px != float(x["close"])
    return rec


def cell(df: pd.DataFrame, col: str) -> dict:
    x = df[["session", col]].dropna()
    if len(x) < 5:
        return {"n": len(x)}
    sm = x.groupby("session")[col].mean()
    se = sm.std(ddof=1) / np.sqrt(len(sm))
    return {"n": len(x), "sessions": len(sm), "gross_mean": x[col].mean(), "gross_median": x[col].median(),
            "win": (x[col] > 0).mean(), "net_mean": x[col].mean() - COST, "t_sessions": sm.mean() / se,
            "mde_sessions": 2.8 * se}


def main() -> None:
    a = pd.DataFrame([json.loads(x) for x in LEDGER.open()])
    a["entry_ts"] = pd.to_datetime(a["entry_time"], utc=True)
    a["session"] = a["entry_ts"].dt.tz_convert(ET).dt.normalize().dt.tz_localize(None)
    v = rs05.regime_votes(pd.DatetimeIndex(sorted(a["session"].unique())))
    a = a.join(v[["votes", "BEAR"]], on="session")
    a["ret_ACTUAL"] = a["gross_return"]

    rows, skipped = [], 0
    for day, g in a.groupby("session"):
        p = ARCHIVE / f"bars_{day:%Y%m%d}.jsonl"
        if not p.exists():
            skipped += len(g)
            continue
        bars = load_day(p, set(g["ticker"]))
        for idx, s in g.iterrows():
            b = bars.get(s["ticker"])
            rows.append({"idx": idx, **(counterfactuals(s, b) if b is not None and len(b) else {})})
    cf = pd.DataFrame(rows).set_index("idx")
    t = a.join(cf, how="inner")
    t.drop(columns=["metadata", "evidence", "warnings", "targets", "target_level_sources", "candidate_sources"],
           errors="ignore").to_csv(OUT / "is_exit_counterfactual_trades.csv", index=False)

    out = []
    exits = ("ACTUAL", "STRUCT", "T30", "T60", "EOD")
    for d in ("long", "short"):
        for name, m in (("all", t["BEAR"].notna()), ("BEAR", t["BEAR"] == 1), ("not BEAR", t["BEAR"] == 0)):
            sub = t[m & (t["direction"] == d)]
            # compare every exit on the SAME setups: those with all five returns present
            both = sub.dropna(subset=[f"ret_{e}" for e in exits])
            for e in exits:
                out.append({"direction": d, "cell": name, "exit": e, **cell(both, f"ret_{e}")})
    res = pd.DataFrame(out)
    res.to_csv(OUT / "is_exit_counterfactual.csv", index=False)
    print(f"setups={len(a)} in archived sessions={len(t)} skipped(no archive)={skipped} "
          f"entry gap median={t['entry_gap'].median():.5f} p90={t['entry_gap'].quantile(0.9):.5f} "
          f"complete={int(t.dropna(subset=[f'ret_{e}' for e in exits]).shape[0])} "
          f"struct stopped share={t['struct_stopped'].mean():.2f}")


if __name__ == "__main__":
    main()
