"""Would the intraday structure module have made money if a market-regime vote told it which side to trade?

The question (user, 2026-11-22): build a regime from several signals, not just the 50-day; when it
turns bearish, prioritise shorts; the intraday structure module "would be good at this but needs to
know when to go short".

This needs no backtest. The module ran live (paper) from 2026-08-27 and logged every setup it
confirmed, long and short, with the engine's own entry/exit. Those rows were written in real time,
before anyone knew how the selloff would go, so nothing here is fit to the outcome.

Pre-registered 2026-11-22 ~02:25 ET, before any split below was computed.

Data
  A. Data/inference/intraday_structure/closed_setups.jsonl: 3,114 confirmed setups, 678 tickers,
     2026-08-27..11-20. Outcome = net_return on the UNDERLYING after the module's modeled costs
     (8 bp spread + 4 bp slippage). These are model outcomes, not fills.
  B. Data/inference/intraday_structure/closed_trades.jsonl: real paper OPTION fills (0-4 DTE) placed
     by the same module, 2026-09-01..11-18. Outcome = realized_pnl ($) and premium return
     exit_fill / entry_fill - 1. Rows without both fills are excluded and counted.
Regime vote (all known at the PRIOR session's close; the regime table row must have
  available_at <= 09:30 ET of the session). Seven bearish votes, thresholds fixed at round values,
  none tuned:
     q_d0      QQQ close < EMA20 and EMA20 below its value 5 sessions earlier
     q_d1      QQQ close < SMA50
     spy_trend daily_regime.spy_trend_state == -1 (SPY EMA20 < EMA50)
     breadth   breadth_z < 0
     appetite  risk_appetite_z < 0
     credit    credit_risk_z < 0 (HYG/IEI)
     vol       spy_rv20_z > 1
  BEAR = votes >= 4 (majority). Secondary: each single vote.
Tests
  T1 mean net_return by direction x BEAR. Setups on one day are correlated, so inference is on
     session means: average the setups of each (session, direction), then t over sessions,
     MDE ~= 2.8 x SE.
  T2 gated book: shorts only on BEAR sessions, longs only on non-BEAR sessions, against
     (a) everything the module took, (b) longs only, (c) shorts only, and (d) a placebo that keeps
     the same number of BEAR sessions but assigns them at random (1,000 draws).
  T3 the same split on the real option fills (B).
  Descriptive: the selloff window 2026-10-15..11-20, and the single-vote splits.
Limits: one window (Aug-Nov 2026) with one selloff; regime days are clustered in time, so the
  effective sample is a few regime spells, not hundreds of days. Nothing here certifies an edge.
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
LEDGER = REPO / "Data" / "inference" / "intraday_structure"
REGIME = REPO / "Data" / "shared" / "market_regime" / "daily_regime.parquet"
ET = "America/New_York"
VOTES = ("q_d0", "q_d1", "spy_trend", "breadth", "appetite", "credit", "vol")
BEAR_MIN = 4
SELLOFF = (pd.Timestamp("2026-10-15"), pd.Timestamp("2026-11-20"))
RNG = np.random.default_rng(20261122)

_spec = importlib.util.spec_from_file_location("rs01", HERE / "01_regime_study.py")
rs01 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rs01)


def regime_votes(sessions: pd.DatetimeIndex) -> pd.DataFrame:
    """Votes known before the open of each session in ``sessions`` (tz-naive dates)."""
    q = rs01.load("QQQ")
    st = rs01.states(q)
    prior = pd.Series(q.index, index=q.index).shift(-1)  # session after each QQQ close
    qv = pd.DataFrame({"q_d0": st["D0_fast"], "q_d1": st["D1_sma50"]})
    qv.index = prior.reindex(qv.index).to_numpy()
    qv = qv[qv.index.notna()]
    # the last QQQ close (11-19) feeds the 11-20 session, which has no daily bar in 01's data
    qv.loc[rs01.LAST_SESSION + pd.offsets.BDay(1)] = [st["D0_fast"].iloc[-1], st["D1_sma50"].iloc[-1]]

    r = pd.read_parquet(REGIME)
    r["available_at"] = pd.to_datetime(r["available_at"], utc=True)
    opens = pd.DataFrame({"session": sessions})
    opens["open_utc"] = (opens["session"] + pd.Timedelta(hours=9, minutes=30)).dt.tz_localize(ET).dt.tz_convert("UTC")
    m = pd.merge_asof(opens.sort_values("open_utc"), r.sort_values("available_at"),
                      left_on="open_utc", right_on="available_at", direction="backward")
    stale = (m["open_utc"] - m["available_at"]).dt.days
    if (stale > 4).any():
        raise ValueError(f"regime row older than 4 days for {m.loc[stale > 4, 'session'].tolist()}")
    v = pd.DataFrame(index=m["session"])
    v["spy_trend"] = (m["spy_trend_state"] == -1).astype(float).to_numpy()
    v["breadth"] = (m["breadth_z"] < 0).astype(float).to_numpy()
    v["appetite"] = (m["risk_appetite_z"] < 0).astype(float).to_numpy()
    v["credit"] = (m["credit_risk_z"] < 0).astype(float).to_numpy()
    v["vol"] = (m["spy_rv20_z"] > 1).astype(float).to_numpy()
    v["regime_row_date"] = m["date"].to_numpy()
    v = v.join(qv, how="left")
    if v[list(VOTES)].isna().any().any():
        raise ValueError("missing vote inputs:\n" + v[v[list(VOTES)].isna().any(axis=1)].to_string())
    v["votes"] = v[list(VOTES)].sum(axis=1)
    v["BEAR"] = (v["votes"] >= BEAR_MIN).astype(float)
    return v


def session_t(df: pd.DataFrame, col: str) -> dict:
    """Mean per setup, plus inference on session means."""
    if df.empty:
        return {"setups": 0}
    s = df.groupby("session")[col].mean()
    se = s.std(ddof=1) / np.sqrt(len(s)) if len(s) > 1 else np.nan
    return {"setups": len(df), "sessions": len(s), "mean_per_setup": df[col].mean(),
            "win": (df[col] > 0).mean(), "sum": df[col].sum(), "session_mean": s.mean(),
            "t_sessions": s.mean() / se if se else np.nan, "mde_sessions": 2.8 * se}


def gated(df: pd.DataFrame, bear: pd.Series) -> pd.Series:
    b = df["session"].map(bear)
    keep = ((df["direction"] == "short") & (b == 1)) | ((df["direction"] == "long") & (b == 0))
    return df.loc[keep, "net_return"]


def main() -> None:
    a = pd.DataFrame([json.loads(x) for x in (LEDGER / "closed_setups.jsonl").open()])
    a["entry_ts"] = pd.to_datetime(a["entry_time"], utc=True)
    a["session"] = a["entry_ts"].dt.tz_convert(ET).dt.normalize().dt.tz_localize(None)
    sessions = pd.DatetimeIndex(sorted(a["session"].unique()))
    v = regime_votes(sessions)
    v.to_csv(OUT / "regime_votes_by_session.csv")
    a = a.join(v[list(VOTES) + ["votes", "BEAR"]], on="session")

    rows = []
    for d in ("long", "short"):
        for cell, mask in (("all", a["BEAR"].notna()), ("BEAR", a["BEAR"] == 1), ("not BEAR", a["BEAR"] == 0),
                           ("selloff 10-15..11-20", a["session"].between(*SELLOFF))):
            rows.append({"source": "setups", "direction": d, "cell": cell,
                         **session_t(a[mask & (a["direction"] == d)], "net_return")})
        for vote in VOTES:
            for val, name in ((1.0, "bearish"), (0.0, "not")):
                rows.append({"source": "setups", "direction": d, "cell": f"{vote}={name}",
                             **session_t(a[(a[vote] == val) & (a["direction"] == d)], "net_return")})

    # T2 gated book vs placebo
    bear = v["BEAR"]
    real = gated(a, bear)
    n_bear = int(bear.sum())
    draws = []
    for _ in range(1000):
        fake = pd.Series(0.0, index=bear.index)
        fake.iloc[RNG.choice(len(bear), n_bear, replace=False)] = 1.0
        draws.append(gated(a, fake).mean())
    draws = np.array(draws)
    books = {"all taken": a["net_return"], "longs only": a.loc[a.direction == "long", "net_return"],
             "shorts only": a.loc[a.direction == "short", "net_return"], "regime-gated": real}
    book_rows = [{"book": k, "setups": len(x), "mean_per_setup": x.mean(), "win": (x > 0).mean(), "sum": x.sum()}
                 for k, x in books.items()]
    book_rows.append({"book": "placebo gate (random BEAR days)", "setups": np.nan, "mean_per_setup": draws.mean(),
                      "win": np.nan, "sum": np.nan, "real_pct_vs_placebo": float((draws < real.mean()).mean())})
    pd.DataFrame(book_rows).to_csv(OUT / "is_gated_books.csv", index=False)

    # T3 option fills
    b = pd.DataFrame([json.loads(x) for x in (LEDGER / "closed_trades.jsonl").open()])
    b = b[b["module"] == "intraday_structure"]
    n_all = len(b)
    # ledger-repaired rows (2026-10-09 repair) lost entry_submitted_at, so their entry session is unknown
    b = b[b["entry_fill_price"].notna() & b["exit_fill_price"].notna() & (b["entry_fill_price"] > 0)
          & b["entry_submitted_at"].notna()].copy()
    b["direction"] = b["direction"].replace({-1: "short", 1: "long", "-1": "short", "1": "long"})
    b["session"] = pd.to_datetime(b["entry_submitted_at"], utc=True, format="mixed").dt.tz_convert(ET).dt.normalize().dt.tz_localize(None)
    vb = regime_votes(pd.DatetimeIndex(sorted(b["session"].unique())))
    b = b.join(vb[["votes", "BEAR"]], on="session")
    b["prem_ret"] = b["exit_fill_price"] / b["entry_fill_price"] - 1
    for d in ("long", "short"):
        for cell, mask in (("all", b["BEAR"].notna()), ("BEAR", b["BEAR"] == 1), ("not BEAR", b["BEAR"] == 0),
                           ("selloff 10-15..11-20", b["session"].between(*SELLOFF))):
            sub = b[mask & (b["direction"] == d)]
            r = session_t(sub, "prem_ret")
            rows.append({"source": "option fills", "direction": d, "cell": cell, **r,
                         "pnl_usd": sub["realized_pnl"].sum(), "median_dte": sub["dte_at_entry"].median()})
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "is_by_regime.csv", index=False)
    b[["session", "ticker", "order_symbol", "direction", "dte_at_entry", "entry_fill_price", "exit_fill_price",
       "prem_ret", "realized_pnl", "exit_reason", "votes", "BEAR"]].to_csv(OUT / "is_option_fills.csv", index=False)
    print(f"setups={len(a)} sessions={len(sessions)} BEAR sessions={n_bear}; option fills used {len(b)}/{n_all}")


if __name__ == "__main__":
    main()
