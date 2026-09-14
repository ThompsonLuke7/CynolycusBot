"""Sector control for the leader -> follower result.

`theme_cohesion_leadlag.py` part B found that after a theme member pops >= 7% in a
day, the OTHER members beat the universe by +0.30% at 5 sessions and +1.49% at 20,
against a shuffled-theme control of ~0. Its sector control silently produced NaN: the
script looked for a `sector_id` column, but the shared universe names it `sector`.

That control is the one that matters. "Stocks near a stock that just popped go up"
is only a THEME edge if theme-mates beat plain SECTOR-mates of the same leader. This
reruns the identical event study with three follower definitions:

  theme     the leader's theme-mates (as before)
  sector    size-matched draw from the leader's sector, excluding theme-mates
  shuffled  theme labels permuted across tickers, group sizes preserved

Same mechanics as the original: event detected on day d, every position entered at
the OPEN of d+1, excess measured against that day's whole scored universe,
corporate-action-flagged windows dropped, CIs clustered on the event day.
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

from theme_cohesion_leadlag import (HOLDS, MIN_MEMBERS, POP_THRESHOLD, THEMES,  # noqa: E402
                                    day_cluster_boot, load_returns)

DATA = REPO / "research/execution_quality/data"
FLAGS = DATA / "corporate_action_flags.parquet"
UNIVERSE = REPO / "Data/shared/universe/shared_universe.csv"
SEED = 67


def sector_map() -> dict[str, str]:
    u = pd.read_csv(UNIVERSE)
    col = next((c for c in ("sector", "sector_id", "gics_sector") if c in u.columns), None)
    if not col:
        raise SystemExit("no sector column in shared_universe.csv")
    print(f"sector column: '{col}'")
    return {str(t): str(s) for t, s in zip(u["ticker"], u[col]) if pd.notna(s)}


def main() -> None:
    rng = np.random.default_rng(SEED)
    th = pd.read_parquet(THEMES, columns=["date", "ticker", "primary_theme"]).dropna()
    th["date"] = pd.to_datetime(th["date"])
    R, F = load_returns(sorted(th["ticker"].unique()))
    sec = sector_map()
    flags = pd.read_parquet(FLAGS)
    flags = flags[~flags["organic"]]
    flagged = {(r.ticker, pd.Timestamp(r.date)) for r in flags.itertuples()}
    print(f"returns {R.shape[0]} sessions x {R.shape[1]} tickers; "
          f"{sum(1 for t in R.columns if t in sec)} tickers have a sector")

    rows: list[dict] = []
    for d, g in th.groupby("date"):
        if d not in R.index:
            continue
        mapping = g.set_index("ticker")["primary_theme"].to_dict()
        day_ret = R.loc[d]
        pops = [t for t in mapping if t in day_ret.index and day_ret[t] >= POP_THRESHOLD]
        if not pops:
            continue
        by_theme: dict[str, list[str]] = {}
        for t, m in mapping.items():
            by_theme.setdefault(m, []).append(t)
        shuffled = dict(zip(list(mapping), rng.permutation(list(mapping.values()))))
        by_shuffled: dict[str, list[str]] = {}
        for t, m in shuffled.items():
            by_shuffled.setdefault(m, []).append(t)
        by_sector: dict[str, list[str]] = {}
        for t in mapping:
            if t in sec:
                by_sector.setdefault(sec[t], []).append(t)

        for leader in pops:
            mates = [x for x in by_theme[mapping[leader]] if x != leader]
            if len(mates) < MIN_MEMBERS - 1:
                continue
            s_mates = [x for x in by_shuffled[shuffled[leader]] if x != leader]
            pool = [x for x in by_sector.get(sec.get(leader, ""), []) if x != leader and x not in set(mates)]
            sec_mates = list(rng.choice(pool, min(len(mates), len(pool)), replace=False)) if pool else []
            for h in HOLDS:
                fwd = F[h].loc[d]
                uni = fwd.dropna()
                if len(uni) < 50:
                    continue
                for tag, group in (("theme", mates), ("sector", sec_mates), ("shuffled", s_mates)):
                    cols = [m for m in group if m in fwd.index and (m, d) not in flagged]
                    vals = fwd[cols].dropna() if cols else pd.Series(dtype=float)
                    if len(vals) < MIN_MEMBERS - 1:
                        continue
                    rows.append(dict(day=d, hold=h, tag=tag, n=len(vals),
                                     excess=float(vals.mean() - uni.mean())))

    df = pd.DataFrame(rows)
    print(f"\nevents: {df[df.tag == 'theme']['day'].nunique()} days, "
          f"{len(df[df.tag == 'theme'])//len(HOLDS) if len(df) else 0} leader-events per hold\n")
    print(f"{'hold':>5s} {'theme':>22s} {'sector':>22s} {'shuffled':>10s} {'theme-sector':>22s}")
    out = []
    for h in HOLDS:
        sub = df[df["hold"] == h]
        t = sub[sub["tag"] == "theme"]
        s = sub[sub["tag"] == "sector"]
        sh = sub[sub["tag"] == "shuffled"]
        if t.empty or s.empty:
            continue
        tlo, thi = day_cluster_boot(t["excess"].to_numpy(), t["day"].to_numpy(), rng)
        slo, shi = day_cluster_boot(s["excess"].to_numpy(), s["day"].to_numpy(), rng)
        # paired on the same event days
        m = t.merge(s, on=["day", "hold"], suffixes=("_t", "_s"))
        diff = (m["excess_t"] - m["excess_s"]).to_numpy()
        dlo, dhi = day_cluster_boot(diff, m["day"].to_numpy(), rng)
        print(f"{h:5d} {t['excess'].mean() * 100:8.3f}% [{tlo * 100:+6.3f},{thi * 100:+6.3f}] "
              f"{s['excess'].mean() * 100:8.3f}% [{slo * 100:+6.3f},{shi * 100:+6.3f}] "
              f"{sh['excess'].mean() * 100:9.3f}% "
              f"{diff.mean() * 100:8.3f}pp [{dlo * 100:+6.3f},{dhi * 100:+6.3f}]")
        out.append(dict(hold=h, theme=float(t["excess"].mean()), theme_ci=[tlo, thi],
                        sector=float(s["excess"].mean()), sector_ci=[slo, shi],
                        shuffled=float(sh["excess"].mean()) if len(sh) else None,
                        paired_diff=float(diff.mean()), paired_ci=[dlo, dhi], n_pairs=int(len(m))))
    print("\n  theme-sector > 0 with a CI clear of zero = the TAXONOMY adds something a sector map does not")
    p = DATA / "leadlag_sector_control.json"
    p.write_text(json.dumps(out, indent=1, default=str))
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
