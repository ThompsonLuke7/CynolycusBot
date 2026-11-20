"""Replay the account under risk rules that only use information available at the time.

Unit: the trade *episode* from 01_build_ledger.py, with its interval-by-interval
P&L as it actually happened. A rule decides, at each episode's entry, using the
counterfactual account as of the last snapshot before that entry (and regime
rows whose `available_at` precedes it), whether the episode happens at all.
Rejected episodes contribute nothing; accepted ones contribute exactly what they
did. Portfolio actions (kill switch, hedge) act at a snapshot on what the
counterfactual book holds then.

Validity check: the rule that accepts everything must reproduce the actual
equity path to the dollar.

What this cannot do (state it with every result):
  * entries the broker refused for buying power in reality cannot be added
    back, so rules that free buying power are slightly flattered;
  * every threshold here is evaluated on one 3.5-month path of one account; a
    rule that "works" here is a hypothesis, not a validated control. Results
    are reported as a grid, never as a single picked best.

    PYTHONPATH=. .venv/bin/python research/account_postmortem_2026-11-03/03_counterfactuals.py
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

OUT = Path("Data/research/account_postmortem_20261103")
ET = "America/New_York"
LIQ_COST = {"equity": 0.0025, "option": 0.10}  # half-spread-ish exit cost as a share of MV
pd.set_option("display.width", 220)


@dataclass
class State:
    equity: float
    peak: float
    gross: float
    t0: pd.Timestamp


def load():
    c = pd.read_parquet(OUT / "ledger_contrib.parquet")
    a = pd.read_parquet(OUT / "ledger_account.parquet").reset_index(drop=True)
    ep = pd.read_parquet(OUT / "ledger_episodes.parquet").set_index("episode")
    m = pd.read_parquet(OUT / "interval_market.parquet")
    a = a.merge(m[["t1", "r_SPY", "r_QQQ", "r_IWM"]], on="t1", how="left")
    reg = pd.read_parquet("Data/shared/market_regime/daily_regime.parquet")
    reg["available_at"] = pd.to_datetime(reg.available_at, utc=True).dt.tz_convert(ET)
    reg = reg.sort_values("available_at")
    iwm = pd.read_parquet("Data/shared/bars/1d/IWM.parquet")
    iwm["ts"] = pd.to_datetime(iwm.timestamp, utc=True).dt.tz_convert(ET).dt.normalize() + pd.Timedelta(hours=16)
    iwm = iwm.set_index("ts").close.sort_index()
    iwm_trend = (iwm / iwm.rolling(50, min_periods=50).mean() - 1)
    return c, a, ep, reg, iwm_trend


def simulate(c, a, ep, *, accept: Callable[[pd.Series, State], bool] = lambda e, s: True,
             scale: Callable[[pd.Series], float] = lambda e: 1.0,
             kill: Callable[[State], bool] = lambda s: False,
             hedge: Callable[[State], float] = lambda s: 0.0) -> dict:
    first_t0 = a.t0.iloc[0]
    decided: dict[str, float] = {}  # episode -> weight (0 rejected)
    for e_id, e in ep.iterrows():
        if e.entry_ts <= first_t0 or e_id.endswith("@pre"):
            decided[e_id] = scale(e)
    st = State(equity=a.equity_t0.iloc[0], peak=a.equity_t0.iloc[0], gross=a.gross_t0.iloc[0], t0=first_t0)
    killed = False
    path, rejected = [], 0
    by_t1 = {t1: g for t1, g in c.groupby("t1")}
    for _, row in a.iterrows():
        st.t0 = row.t0
        g = by_t1[row.t1]
        if not killed and kill(st):
            killed = True
            open_rows = g[(g.mv_t0 > 0) & g.episode.map(lambda e: decided.get(e, 0) > 0)]
            cost = sum(r.mv_t0 * LIQ_COST[r.route] * decided[r.episode] for r in open_rows.itertuples())
            st.equity -= cost
            for e_id in set(open_rows.episode):
                decided[e_id] = 0.0
        new_eps = [e for e in g.episode.unique() if e not in decided and e in ep.index]
        for e_id in new_eps:
            ok = (not killed) and accept(ep.loc[e_id], st)
            decided[e_id] = scale(ep.loc[e_id]) if ok else 0.0
            rejected += 0 if ok else 1
        w = g.episode.map(lambda e: 1.0 if e == "__other__" else decided.get(e, 1.0))
        pnl = float((g.contrib * w).sum())
        h = hedge(st)
        pnl += -h * st.gross * float(row.r_IWM if pd.notna(row.r_IWM) else 0.0)
        st.equity += pnl
        st.peak = max(st.peak, st.equity)
        st.gross = float((g.mv_t1.clip(lower=0) * w).sum())
        path.append((row.t1, st.equity, st.gross))
    p = pd.DataFrame(path, columns=["t1", "equity", "gross"])
    dd = (p.equity / p.equity.cummax() - 1).min()
    return {"final": p.equity.iloc[-1], "max_dd": dd, "rejected": rejected,
            "max_gross_x": (p.gross / p.equity).max(), "path": p}


def main() -> None:
    c, a, ep, reg, iwm_trend = load()
    actual_final = a.equity_t1.iloc[-1]
    base = simulate(c, a, ep)
    assert abs(base["final"] - actual_final) < 1_000, (base["final"], actual_final)
    start = a.equity_t0.iloc[0]

    def regime_at(ts: pd.Timestamp) -> pd.Series | None:
        r = reg[reg.available_at <= ts]
        return None if r.empty else r.iloc[-1]

    def trend_at(ts: pd.Timestamp) -> float:
        s = iwm_trend[iwm_trend.index <= ts].dropna()
        return float(s.iloc[-1]) if len(s) else 0.0

    def dd(s: State) -> float:
        return 1 - s.equity / s.peak

    rows = []

    def run(family: str, name: str, **kw):
        r = simulate(c, a, ep, **kw)
        rows.append({"family": family, "rule": name, "final_equity": r["final"],
                     "vs_actual": r["final"] - actual_final, "return": r["final"] / start - 1,
                     "max_dd": r["max_dd"], "max_gross_x": r["max_gross_x"], "entries_refused": r["rejected"]})
        return r

    run("actual", "replay (accept everything)")
    for cap in (1.0, 1.25, 1.5, 2.0):
        run("gross cap", f"refuse entries at gross >= {cap:.2f}x", accept=lambda e, s, cap=cap: s.gross / s.equity < cap)
    for x in (0.10, 0.15, 0.20, 0.25, 0.30):
        run("exits-only freeze", f"freeze entries forever once DD >= {x:.0%}",
            accept=lambda e, s, x=x, state={"hit": False}: not (state.__setitem__("hit", state["hit"] or dd(s) >= x) or state["hit"]))
    for x in (0.10, 0.20):
        run("exits-only freeze", f"freeze while DD >= {x:.0%} (re-arm below {x / 2:.0%})",
            accept=lambda e, s, x=x, state={"on": False}: not (
                state.__setitem__("on", (dd(s) >= x) or (state["on"] and dd(s) > x / 2)) or state["on"]))
    for x in (0.15, 0.20, 0.25, 0.30):
        run("kill switch", f"liquidate all + stop at DD >= {x:.0%}", kill=lambda s, x=x: dd(s) >= x)
    run("route", "no options (equity episodes only)", accept=lambda e, s: e.route == "equity")
    for cap in (10_000, 25_000):
        run("position size", f"cap any episode at ${cap:,.0f} cost (scaled)",
            scale=lambda e, cap=cap: min(1.0, cap / e.buy_cost) if e.buy_cost > 0 else 1.0)
    for name, test in (
        ("IWM below its 50-day average", lambda s: trend_at(s.t0) < 0),
        ("breadth_z < -1", lambda s: (r := regime_at(s.t0)) is not None and r.breadth_z < -1),
        ("spy_rv20_z > 1", lambda s: (r := regime_at(s.t0)) is not None and r.spy_rv20_z > 1),
        ("risk_appetite_z < -1 (as live: stale credit leg after 08-06)", lambda s: (r := regime_at(s.t0)) is not None and r.risk_appetite_z < -1),
    ):
        run("regime freeze", f"refuse entries while {name}", accept=lambda e, s, test=test: not test(s))
    for h in (1.0, 2.65):
        run("beta hedge", f"always short IWM = {h:g} x gross", hedge=lambda s, h=h: h)
        run("beta hedge", f"short IWM = {h:g} x gross while IWM < 50d avg", hedge=lambda s, h=h: h if trend_at(s.t0) < 0 else 0.0)
    run("combined", "gross cap 1.0x + $10k name cap + no options",
        accept=lambda e, s: e.route == "equity" and s.gross / s.equity < 1.0,
        scale=lambda e: min(1.0, 10_000 / e.buy_cost) if e.buy_cost > 0 else 1.0)
    run("combined", "gross cap 1.0x + freeze at DD 20%",
        accept=lambda e, s, state={"hit": False}: s.gross / s.equity < 1.0 and not (
            state.__setitem__("hit", state["hit"] or dd(s) >= 0.20) or state["hit"]))

    res = pd.DataFrame(rows)
    res.to_csv(OUT / "counterfactuals.csv", index=False)
    show = res.copy()
    for col in ("final_equity", "vs_actual"):
        show[col] = show[col].map(lambda v: f"{v:,.0f}")
    for col in ("return", "max_dd"):
        show[col] = show[col].map(lambda v: f"{v:+.1%}")
    show["max_gross_x"] = show.max_gross_x.map(lambda v: f"{v:.2f}")
    print(f"start {start:,.0f} (07-15)  actual {actual_final:,.0f} (11-02)  replay check {base['final']:,.0f}")
    print(show.to_string(index=False))

    # Stability: the same rules on each half of the window, starting from the actual equity then.
    mid = a.t1.iloc[len(a) // 2]
    print(f"\nsplit at {mid:%Y-%m-%d}: P&L saved vs actual in each half (path-dependent rules restart each half)")
    for half, mask in (("first", a.t1 <= mid), ("second", a.t1 > mid)):
        ah = a[mask].reset_index(drop=True)
        ch = c[c.t1.isin(ah.t1)]
        act = simulate(ch, ah, ep)["final"]
        out = []
        for name, kw in (
            ("gross cap 1.0x", dict(accept=lambda e, s: s.gross / s.equity < 1.0)),
            ("no options", dict(accept=lambda e, s: e.route == "equity")),
            ("$10k name cap", dict(scale=lambda e: min(1.0, 10_000 / e.buy_cost) if e.buy_cost > 0 else 1.0)),
            ("IWM-trend freeze", dict(accept=lambda e, s: not trend_at(s.t0) < 0)),
            ("hedge 1x always", dict(hedge=lambda s: 1.0)),
        ):
            out.append(f"{name}: {simulate(ch, ah, ep, **kw)['final'] - act:+,.0f}")
        print(f"  {half} half: " + "; ".join(out))


if __name__ == "__main__":
    main()
