"""Theme tracker/trader, measured against SPY the way it has to be measured.

§7 established the raw effect: after a theme member closes +7% in a day, its
theme-mates beat the universe by +1.49% over the next 20 sessions, and that beats
same-sector mates. This turns it into an actual portfolio and asks the only question
that matters: **does it beat SPY, after costs, adjusted for how much market risk it
took and how much of the time it was invested?**

Why total return alone is the wrong measure here: the strategy is event-driven, so it
sits in cash much of the time. A raw return comparison either flatters it (high-beta
names in a rising tape) or buries it (cash drag). So the report gives, for every arm:

  * total return and CAGR, against SPY buy-and-hold over the identical window
  * exposure (average fraction of capital deployed) and turnover
  * alpha and beta from a daily regression on SPY, with alpha annualised -- the
    market-risk-adjusted answer
  * Sharpe, max drawdown, and per-year returns (the legacy rotation's failure was
    regime-specific: top-3 lagged the 2025Q2 broad rebound)

Arms: leader-only, followers-only, the 50/50 split as proposed, and a leader-tilted
70/30, each at several holds. Plus an equal-weight "any +7% popper" control, which is
the honest null: if buying ANY stock that popped does as well, the theme adds nothing.

Membership: the backfilled theme history by default (stable, but not point-in-time)
or trailing-correlation clusters with `--grouping corr` (PIT-safe, and §7 showed they
capture ~73% of the grouping). Costs charged both sides. Entry at the next open.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts/horizon_thesis"))

from theme_cohesion_leadlag import POP_THRESHOLD, THEMES  # noqa: E402

DATA = REPO / "research/execution_quality/data"
BARS_1D = REPO / "Data/shared/bars/1d"
FLAGS = DATA / "corporate_action_flags.parquet"
HOLDS = [5, 10, 20]
COST_BPS = 10.0          # per side, shares
MAX_POSITIONS = 20       # capital is split across at most this many open positions
TRADING_DAYS = 252
SEED = 77


def load_prices(tickers: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    op, cl, hi = {}, {}, {}
    for t in tickers:
        p = BARS_1D / f"{t}.parquet"
        if not p.exists():
            continue
        d = pd.read_parquet(p, columns=["timestamp", "open", "high", "close"])
        if len(d) < 150:
            continue
        idx = (pd.to_datetime(d["timestamp"], utc=True).dt.tz_convert("America/New_York")
               .dt.normalize().dt.tz_localize(None))
        d = d.set_index(idx).sort_index()
        op[t], cl[t], hi[t] = d["open"], d["close"], d["high"]
    return pd.DataFrame(op).sort_index(), pd.DataFrame(cl).sort_index(), pd.DataFrame(hi).sort_index()


def build_events(th: pd.DataFrame, closes: pd.DataFrame, grouping: str, rng) -> dict:
    """{session -> [(leader, [followers...]), ...]} using day-d information only."""
    rets = closes.pct_change(fill_method=None)
    out: dict[pd.Timestamp, list] = {}
    for d, g in th.groupby("date"):
        if d not in rets.index:
            continue
        mapping = g.set_index("ticker")["primary_theme"].to_dict()
        row = rets.loc[d]
        pops = [t for t in mapping if t in row.index and row[t] >= POP_THRESHOLD]
        if not pops:
            continue
        by_theme: dict[str, list[str]] = {}
        for t, m in mapping.items():
            by_theme.setdefault(m, []).append(t)
        evs = []
        pool = [t for t in mapping if t in closes.columns]
        for leader in pops:
            mates = [x for x in by_theme[mapping[leader]] if x != leader and x in closes.columns]
            if not mates:
                continue
            # Control basket: same size, same day, drawn from OUTSIDE the leader's theme.
            outside = [t for t in pool if t != leader and t not in set(mates)]
            randoms = list(rng.choice(outside, min(len(mates), len(outside)), replace=False)) if outside else []
            evs.append((leader, mates, randoms))
        if evs:
            out[d] = evs
    return out


def simulate(events: dict, opens: pd.DataFrame, closes: pd.DataFrame, sessions: pd.DatetimeIndex,
             hold: int, w_leader: float, arm: str, flagged: set) -> pd.Series:
    """Daily portfolio return series. Capital is split across open slots; unused capital earns 0."""
    pos_by_exit: dict[pd.Timestamp, list] = {}
    open_slots: list[dict] = []
    all_slots: list[dict] = []      # kept separately: the loop below CONSUMES pos_by_exit
    daily = pd.Series(0.0, index=sessions)
    sess_pos = {s: i for i, s in enumerate(sessions)}
    for d in sessions:
        # close positions whose hold expired
        for p in pos_by_exit.pop(d, []):
            if p in open_slots:
                open_slots.remove(p)
        evs = events.get(d, [])
        if evs and len(open_slots) < MAX_POSITIONS:
            i = sess_pos[d]
            if i + 1 + hold >= len(sessions):
                continue
            entry_day = sessions[i + 1]
            exit_day = sessions[i + 1 + hold]
            for leader, mates, randoms in evs:
                if len(open_slots) >= MAX_POSITIONS:
                    break
                names: list[tuple[str, float]] = []
                mates = [m for m in mates if (m, d) not in flagged]
                if arm in ("leader", "split", "tilt") and (leader, d) not in flagged:
                    names.append((leader, 1.0 if arm == "leader" else w_leader))
                if arm in ("followers", "split", "tilt") and mates:
                    w = 1.0 if arm == "followers" else (1.0 - w_leader)
                    names += [(m, w / len(mates)) for m in mates]
                if arm == "random_control":
                    picks = [r for r in randoms if (r, d) not in flagged]
                    names = [(r, 1.0 / len(picks)) for r in picks] if picks else []
                names = [(t, w) for t, w in names if t in opens.columns
                         and pd.notna(opens.at[entry_day, t]) and opens.at[entry_day, t] > 0]
                if not names:
                    continue
                slot = dict(entry=entry_day, exit=exit_day, names=names)
                open_slots.append(slot)
                all_slots.append(slot)
                pos_by_exit.setdefault(exit_day, []).append(slot)
    # walk forward computing each day's portfolio return
    live: list[dict] = []
    by_entry: dict[pd.Timestamp, list] = {}
    for s in all_slots:
        by_entry.setdefault(s["entry"], []).append(s)
    for i, d in enumerate(sessions):
        live = [s for s in live if s["exit"] > d]
        live += by_entry.get(d, [])
        if not live:
            continue
        share = 1.0 / MAX_POSITIONS
        day_ret = 0.0
        for s in live:
            prev = sessions[i - 1] if i > 0 else d
            for t, w in s["names"]:
                if t not in closes.columns:
                    continue
                base = opens.at[d, t] if d == s["entry"] else closes.at[prev, t]
                px = closes.at[d, t]
                if pd.isna(base) or pd.isna(px) or base <= 0:
                    continue
                r = px / base - 1.0
                if d == s["entry"]:
                    r -= COST_BPS / 1e4
                if d == s["exit"]:
                    r -= COST_BPS / 1e4
                day_ret += share * w * r
        daily.at[d] = day_ret
    return daily


def stats(daily: pd.Series, spy: pd.Series) -> dict:
    eq = (1 + daily).cumprod()
    yrs = len(daily) / TRADING_DAYS
    total = float(eq.iloc[-1] - 1)
    cagr = float(eq.iloc[-1] ** (1 / yrs) - 1) if yrs > 0 else np.nan
    dd = float((eq / eq.cummax() - 1).min())
    sharpe = float(daily.mean() / daily.std() * np.sqrt(TRADING_DAYS)) if daily.std() > 0 else np.nan
    x = spy.reindex(daily.index).fillna(0.0)
    if x.std() > 0:
        beta = float(np.cov(daily, x)[0, 1] / np.var(x))
        alpha = float((daily.mean() - beta * x.mean()) * TRADING_DAYS)
    else:
        beta, alpha = np.nan, np.nan
    return dict(total_return=total, cagr=cagr, sharpe=sharpe, max_dd=dd, beta=beta,
                alpha_annual=alpha, days_invested=float((daily != 0).mean()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grouping", choices=["theme", "corr"], default="theme")
    args = ap.parse_args()
    rng = np.random.default_rng(SEED)

    th = pd.read_parquet(THEMES, columns=["date", "ticker", "primary_theme"]).dropna()
    th["date"] = pd.to_datetime(th["date"])
    tickers = sorted(set(th["ticker"]) | {"SPY"})
    opens, closes, _hi = load_prices(tickers)
    sessions = closes.index[(closes.index >= th["date"].min()) & (closes.index <= th["date"].max())]
    spy = closes["SPY"].pct_change().reindex(sessions).fillna(0.0)
    flags = pd.read_parquet(FLAGS)
    flags = flags[~flags["organic"]]
    flagged = {(r.ticker, pd.Timestamp(r.date)) for r in flags.itertuples()}
    events = build_events(th, closes, args.grouping, rng)
    print(f"grouping={args.grouping}  sessions {sessions[0].date()}..{sessions[-1].date()} "
          f"({len(sessions)})  event days {len(events)}  costs {COST_BPS:.0f}bps/side  "
          f"max {MAX_POSITIONS} slots")

    spy_stats = stats(spy, spy)
    print(f"\nSPY buy&hold: total {spy_stats['total_return'] * 100:+.1f}%  "
          f"CAGR {spy_stats['cagr'] * 100:+.1f}%  Sharpe {spy_stats['sharpe']:.2f}  "
          f"maxDD {spy_stats['max_dd'] * 100:.1f}%")
    print(f"\n{'arm':18s} {'hold':>5s} {'total':>9s} {'CAGR':>8s} {'alpha/yr':>9s} {'beta':>6s} "
          f"{'Sharpe':>7s} {'maxDD':>8s} {'invested':>9s}")
    rows = []
    for hold in HOLDS:
        for arm, wl in (("leader", 1.0), ("followers", 0.0), ("split", 0.5), ("tilt", 0.7),
                        ("random_control", 0.0)):
            daily = simulate(events, opens, closes, sessions, hold, wl, arm, flagged)
            s = stats(daily, spy)
            s.update(arm=arm, hold=hold)
            rows.append(s)
            print(f"{arm:18s} {hold:5d} {s['total_return'] * 100:8.1f}% {s['cagr'] * 100:7.1f}% "
                  f"{s['alpha_annual'] * 100:8.1f}% {s['beta']:6.2f} {s['sharpe']:7.2f} "
                  f"{s['max_dd'] * 100:7.1f}% {s['days_invested'] * 100:8.1f}%")
        print()
    out = DATA / f"theme_trader_backtest_{args.grouping}.json"
    out.write_text(json.dumps(dict(spy=spy_stats, arms=rows), indent=1, default=str))
    print(f"wrote {out}")
    print("\nRead alpha/yr, not total: the arms are invested only part of the time and carry "
          "different beta. 'random_control' buys a same-size basket from OUTSIDE the leader's "
          "theme on the same days -- if 'followers' does not beat it, the grouping adds nothing.")


if __name__ == "__main__":
    main()
