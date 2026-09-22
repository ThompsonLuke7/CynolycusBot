"""Does the leader -> follower effect survive in names you can actually trade?

The §7 result (+1.49% follower excess at 20 sessions) and the theme-trader backtest
(+74%/yr alpha) were both measured on the full 1,135-name theme universe with
corporate-action flags screened ONLY on the entry day. The trader diagnostic showed
what that buys:

  * 40% of trades sit in names under $1M/day, and they carry the return
  * the top 1% of trades are 66% of it
  * 11 trades with a >=4x overnight jump INSIDE the hold are 7.6% of it (WOLF, SBET)

So this re-runs the event study with the two screens that were missing:

  LIQUIDITY   leader and followers need 20-day dollar volume >= $10M and price >= $5
              on the decision day -- a name that can absorb an order at the open
  ARTIFACTS   any name whose closes move >=4x or <=1/4x overnight ANYWHERE inside the
              holding window is dropped, not just on the entry day

and reports the MEDIAN alongside the mean, because the diagnostic showed the mean is
a tail statistic here (median trade return was 0.00%).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts/horizon_thesis"))

from theme_cohesion_leadlag import HOLDS, MIN_MEMBERS, POP_THRESHOLD, THEMES, day_cluster_boot  # noqa: E402

DATA = REPO / "research/execution_quality/data"
BARS_1D = REPO / "Data/shared/bars/1d"
UNIVERSE = REPO / "Data/shared/universe/shared_universe.csv"
MIN_DOLLAR_VOL = 1e7
MIN_PRICE = 5.0
ARTIFACT_RATIO = 4.0
SEED = 83


def main() -> None:
    rng = np.random.default_rng(SEED)
    th = pd.read_parquet(THEMES, columns=["date", "ticker", "primary_theme"]).dropna()
    th["date"] = pd.to_datetime(th["date"])
    op, cl, vo = {}, {}, {}
    for t in sorted(th["ticker"].unique()):
        p = BARS_1D / f"{t}.parquet"
        if not p.exists():
            continue
        d = pd.read_parquet(p, columns=["timestamp", "open", "close", "volume"])
        if len(d) < 150:
            continue
        idx = (pd.to_datetime(d["timestamp"], utc=True).dt.tz_convert("America/New_York")
               .dt.normalize().dt.tz_localize(None))
        d = d.set_index(idx).sort_index()
        op[t], cl[t], vo[t] = d["open"], d["close"], d["volume"]
    opens, closes, vols = pd.DataFrame(op), pd.DataFrame(cl), pd.DataFrame(vo)
    rets = closes.pct_change(fill_method=None)
    dvol = (closes * vols).rolling(20).mean()
    ratio = closes / closes.shift(1)
    # a name is artifact-hit on day d if an extreme overnight ratio happens on d
    artifact_day = (ratio >= ARTIFACT_RATIO) | (ratio <= 1.0 / ARTIFACT_RATIO)
    sessions = closes.index
    pos = {s: i for i, s in enumerate(sessions)}
    sec = {}
    u = pd.read_csv(UNIVERSE)
    col = next((c for c in ("sector", "sector_id") if c in u.columns), None)
    if col:
        sec = {str(t): str(s) for t, s in zip(u["ticker"], u[col]) if pd.notna(s)}

    # forward open->close returns per hold
    fwd = {}
    for h in HOLDS:
        ex = closes.shift(-(h - 1))
        fwd[h] = (ex / opens.shift(-1) - 1.0)

    rows = []
    for d, g in th.groupby("date"):
        if d not in pos or pos[d] + 1 + max(HOLDS) >= len(sessions):
            continue
        liquid = set(dvol.columns[(dvol.loc[d] >= MIN_DOLLAR_VOL) & (closes.loc[d] >= MIN_PRICE)])
        if len(liquid) < 50:
            continue
        mapping = {t: m for t, m in g.set_index("ticker")["primary_theme"].items() if t in liquid}
        if len(mapping) < MIN_MEMBERS * 2:
            continue
        row = rets.loc[d]
        pops = [t for t in mapping if t in row.index and row[t] >= POP_THRESHOLD]
        if not pops:
            continue
        by_theme: dict[str, list[str]] = {}
        for t, m in mapping.items():
            by_theme.setdefault(m, []).append(t)
        shuf = dict(zip(list(mapping), rng.permutation(list(mapping.values()))))
        by_shuf: dict[str, list[str]] = {}
        for t, m in shuf.items():
            by_shuf.setdefault(m, []).append(t)
        by_sec: dict[str, list[str]] = {}
        for t in mapping:
            if t in sec:
                by_sec.setdefault(sec[t], []).append(t)

        for leader in pops:
            mates = [x for x in by_theme[mapping[leader]] if x != leader]
            if len(mates) < MIN_MEMBERS - 1:
                continue
            s_mates = [x for x in by_shuf[shuf[leader]] if x != leader]
            secpool = [x for x in by_sec.get(sec.get(leader, ""), []) if x != leader and x not in set(mates)]
            sec_mates = list(rng.choice(secpool, min(len(mates), len(secpool)), replace=False)) if secpool else []
            for h in HOLDS:
                e, x = pos[d] + 1, pos[d] + h
                win = artifact_day.iloc[e:x + 1]
                clean = set(win.columns[~win.any()])
                f = fwd[h].loc[d]
                uni = f[[c for c in liquid if c in f.index and c in clean]].dropna()
                if len(uni) < 50:
                    continue
                for tag, grp in (("theme", mates), ("sector", sec_mates), ("shuffled", s_mates)):
                    cols = [m for m in grp if m in f.index and m in clean]
                    vals = f[cols].dropna() if cols else pd.Series(dtype=float)
                    if len(vals) < MIN_MEMBERS - 1:
                        continue
                    rows.append(dict(day=d, hold=h, tag=tag,
                                     excess=float(vals.mean() - uni.mean()),
                                     med_excess=float(vals.median() - uni.median())))

    df = pd.DataFrame(rows)
    if df.empty:
        print("no events survived the liquidity screen")
        return
    t_all = df[df["tag"] == "theme"]
    print(f"LIQUID universe only (>= ${MIN_DOLLAR_VOL / 1e6:.0f}M/day, >= ${MIN_PRICE:.0f}), "
          f"artifacts screened across the whole hold")
    print(f"events: {t_all['day'].nunique()} days, {len(t_all) // len(HOLDS)} leader-events per hold "
          f"(was 24,865 unfiltered)\n")
    print(f"{'hold':>5s} {'theme mean':>22s} {'theme median':>13s} {'sector':>10s} {'shuffled':>10s}")
    out = []
    for h in HOLDS:
        sub = df[df["hold"] == h]
        t = sub[sub["tag"] == "theme"]
        s = sub[sub["tag"] == "sector"]
        sh = sub[sub["tag"] == "shuffled"]
        if t.empty:
            continue
        lo, hi = day_cluster_boot(t["excess"].to_numpy(), t["day"].to_numpy(), rng)
        print(f"{h:5d} {t['excess'].mean() * 100:8.3f}% [{lo * 100:+6.3f},{hi * 100:+6.3f}] "
              f"{t['med_excess'].mean() * 100:12.3f}% "
              f"{s['excess'].mean() * 100 if len(s) else float('nan'):9.3f}% "
              f"{sh['excess'].mean() * 100 if len(sh) else float('nan'):9.3f}%")
        out.append(dict(hold=h, theme=float(t["excess"].mean()), ci=[lo, hi],
                        theme_median=float(t["med_excess"].mean()),
                        sector=float(s["excess"].mean()) if len(s) else None,
                        shuffled=float(sh["excess"].mean()) if len(sh) else None,
                        n=int(len(t))))
    p = DATA / "leadlag_liquidity_filtered.json"
    p.write_text(json.dumps(out, indent=1, default=str))
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
