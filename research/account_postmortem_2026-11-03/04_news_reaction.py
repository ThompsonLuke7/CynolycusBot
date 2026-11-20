"""Was there bad news we could have acted on, and would acting have helped?

For every trade episode (options mapped to their underlying), take the first
strongly negative FinBERT record on that ticker during the hold, and replay
"exit at the next account snapshot after it" (exit cost as in 03). The rule runs
on every episode, winners included, so its cost is counted, not only its saves.

Two clocks:
  * available_at: when our pipeline actually had the record (nightly/weekly
    batch collection, median ~22 h after publication) - what the live system
    could have done;
  * timestamp + 15 min: a real-time feed - the upper bound of reacting faster.

Placebo: the same number of triggered episodes exited at a random time inside
their hold. With a negative per-trade edge an early exit helps on average; news
skill is the rule's saving beyond that.

Also: did we buy into bad news (a strongly negative record available in the 72 h
before entry)?

    PYTHONPATH=. .venv/bin/python research/account_postmortem_2026-11-03/04_news_reaction.py
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("Data/research/account_postmortem_20261103")
ET = "America/New_York"
OCC = re.compile(r"^([A-Z]{1,6})\d{6}[CP]\d{8}$")
NEG = 0.5  # finbert_negative - finbert_positive
LIQ = {"equity": 0.0025, "option": 0.10}
pd.set_option("display.width", 220)


def underlying(sym: str) -> str:
    m = OCC.match(sym)
    return m.group(1) if m else sym


def load_news(tickers: set[str]) -> pd.DataFrame:
    fb = pd.read_parquet(OUT / "news_finbert_snapshot_20261103.parquet",
                         columns=["record_id", "ticker", "timestamp", "finbert_positive_score",
                                  "finbert_negative_score", "finbert_available"])
    fb = fb[fb.ticker.isin(tickers) & fb.finbert_available.fillna(False)]
    tm = pd.read_parquet(OUT / "news_records_times_snapshot_20261103.parquet", columns=["record_id", "available_at", "headline"])
    n = fb.merge(tm, on="record_id", how="left")
    n["ts"] = pd.to_datetime(n.timestamp, utc=True, errors="coerce").dt.tz_convert(ET)
    n["avail"] = pd.to_datetime(n.available_at, utc=True, errors="coerce").dt.tz_convert(ET)
    n = n[n.ts >= "2026-07-01"]
    n["net_neg"] = n.finbert_negative_score - n.finbert_positive_score
    return n


def main() -> None:
    c = pd.read_parquet(OUT / "ledger_contrib.parquet")
    ep = pd.read_parquet(OUT / "ledger_episodes.parquet")
    snaps = np.sort(pd.read_parquet(OUT / "ledger_account.parquet").t1.unique())
    end = pd.Timestamp(snaps[-1])
    ep["ticker"] = ep.symbol.map(underlying)
    by_ep = {e: g.sort_values("t1") for e, g in c[c.episode.isin(ep.episode)].groupby("episode")}
    ep["exit_ts"] = [by_ep[e].loc[by_ep[e].mv_t1.abs() > 0, "t1"].max() if e in by_ep and (by_ep[e].mv_t1.abs() > 0).any() else r.last_fill_ts
                     for e, r in ep.set_index("episode").iterrows()]
    ep["exit_ts"] = pd.to_datetime(ep.exit_ts).fillna(ep.last_fill_ts)
    ep = ep[ep.episode.isin(by_ep)]
    news = load_news(set(ep.ticker))
    bad = news[news.net_neg >= NEG]
    print(f"episodes {len(ep)}; FinBERT records on those tickers since 07-01: {len(news):,}; strongly negative: {len(bad):,}")
    lag_h = ((bad.avail - bad.ts).dt.total_seconds() / 3600).median()
    print(f"median availability lag of a negative record: {lag_h:.1f} h after its timestamp")

    def exit_after(e_id: str, trigger: pd.Timestamp, route: str) -> float:
        """Episode P&L if closed at the first snapshot after `trigger`."""
        g = by_ep[e_id]
        after = g[g.t1 > trigger]
        if after.empty:
            return float(g.contrib.sum())
        cut = after.t1.iloc[0]
        kept = g[g.t1 <= cut]
        mv = float(kept.mv_t1.iloc[-1])
        return float(kept.contrib.sum()) - abs(mv) * LIQ[route]

    rng = np.random.default_rng(20261103)
    for clock, col, lag in (("pipeline (available_at)", "avail", pd.Timedelta(0)), ("real-time (timestamp+15m)", "ts", pd.Timedelta(minutes=15))):
        rows = []
        for r in ep.itertuples():
            hits = bad[(bad.ticker == r.ticker) & (bad[col] + lag > r.entry_ts) & (bad[col] + lag < r.exit_ts)]
            if hits.empty:
                continue
            first = hits.sort_values(col).iloc[0]
            t = first[col] + lag
            cfp = exit_after(r.episode, t, r.route)
            span = (r.exit_ts - r.entry_ts).total_seconds()
            placebo = np.mean([exit_after(r.episode, r.entry_ts + pd.Timedelta(seconds=rng.uniform(0, span)), r.route) for _ in range(20)])
            rows.append({"episode": r.episode, "ticker": r.ticker, "module": r.module, "route": r.route,
                         "actual": r.pnl, "news_exit": cfp, "placebo_exit": placebo, "trigger": t,
                         "headline": str(first.headline)[:90]})
        t = pd.DataFrame(rows)
        saved = (t.news_exit - t.actual).sum()
        placebo = (t.placebo_exit - t.actual).sum()
        print(f"\n== clock: {clock} ==")
        print(f"episodes with a strongly negative record during the hold: {len(t)} of {len(ep)} "
              f"(losers among them: {(t.actual < 0).mean():.0%}; all episodes: {(ep.pnl < 0).mean():.0%})")
        print(f"actual P&L of those episodes {t.actual.sum():,.0f}; exit-on-news {t.news_exit.sum():,.0f} "
              f"-> saved {saved:+,.0f}; random-time exit saved {placebo:+,.0f}; news skill {saved - placebo:+,.0f}")
        print(t.groupby("route").apply(lambda g: pd.Series({"n": len(g), "saved": (g.news_exit - g.actual).sum(),
                                                             "placebo": (g.placebo_exit - g.actual).sum()}), include_groups=False).round(0).to_string())
        t["saved"] = t.news_exit - t.actual
        print("largest saves:")
        print(t.sort_values("saved", ascending=False)[["ticker", "module", "route", "actual", "news_exit", "trigger", "headline"]]
              .head(6).assign(trigger=lambda x: x.trigger.dt.strftime("%m-%d %H:%M")).round(0).to_string(index=False))
        t.to_csv(OUT / f"news_exit_{col}.csv", index=False)

    print("\n== bought into bad news (strongly negative record available in the 72 h before entry) ==")
    flags = []
    for r in ep.itertuples():
        prior = bad[(bad.ticker == r.ticker) & (bad.avail <= r.entry_ts) & (bad.avail > r.entry_ts - pd.Timedelta(hours=72))]
        flags.append(not prior.empty)
    ep["bad_before"] = flags
    s = ep.groupby("bad_before").pnl.agg(["size", "sum", "mean", lambda x: (x > 0).mean()])
    s.columns = ["episodes", "pnl", "mean_pnl", "win_rate"]
    print(s.round(2).to_string())
    a, b = ep[ep.bad_before].pnl, ep[~ep.bad_before].pnl
    se = np.sqrt(a.var() / max(len(a), 1) + b.var() / max(len(b), 1))
    print(f"difference in mean P&L {a.mean() - b.mean():+,.0f} per episode (SE {se:,.0f}); "
          f"minimum detectable difference at 80% power ~ {2.8 * se:,.0f}")


if __name__ == "__main__":
    main()
