"""Do themes actually work the way the thesis says: a group that moves together,
with a leader and sympathy followers?

Nothing measured so far tests that. The Meta ablation asked a different question --
"do theme FEATURES help a cross-sectional ranker" (answer: no, they hurt). This asks
the original hypothesis directly, in three parts:

  A. COHESION      do theme-mates co-move more than random groups of the same size,
                   measured FORWARD (the next 20 sessions) so the answer is not
                   circular? The embeddings blend text with a 60-day trailing
                   co-movement vector, so trailing cohesion is guaranteed by
                   construction and only forward cohesion is evidence.

  B. LEAD-LAG      when one member pops (a big up day), do the others follow over the
                   NEXT sessions -- entered at the next open, so it is tradeable?
                   Controls: shuffled theme labels (same group sizes) and same-sector
                   mates, because "tech stocks move together" is not a theme edge.

  C. ALLOCATION    the 50/50 split actually proposed: half the capital on the leader,
                   half spread across the followers, held 5/10/20 sessions, against
                   leader-only, followers-only, and an equal-weight universe draw.

Timing: an event is detected on day d using only day-d data; every position is
entered at the OPEN of d+1. Same-day sympathy is reported separately and is NOT
counted as tradeable. Corporate-action-flagged windows are dropped. CIs cluster on
the event day, since one day's events share the tape.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

DATA = REPO / "research/execution_quality/data"
BARS_1D = REPO / "Data/shared/bars/1d"
THEMES = REPO / "themes/dynamic_theme/outputs/ticker_theme_features_history.parquet"
FLAGS = DATA / "corporate_action_flags.parquet"
UNIVERSE = REPO / "Data/shared/universe/shared_universe.csv"

POP_THRESHOLD = 0.07          # a "leader pop": member's 1-day return >= +7%
MIN_MEMBERS = 5
HOLDS = [1, 3, 5, 10, 20]
COHESION_WINDOW = 20
N_SHUFFLE = 20
SEED = 31


def load_returns(tickers: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Wide daily close-to-close returns, and next-open-to-close-h returns per hold."""
    rets, fwd = {}, {}
    for t in tickers:
        p = BARS_1D / f"{t}.parquet"
        if not p.exists():
            continue
        d = pd.read_parquet(p, columns=["timestamp", "open", "close"])
        if len(d) < 120:
            continue
        idx = (pd.to_datetime(d["timestamp"], utc=True).dt.tz_convert("America/New_York")
               .dt.normalize().dt.tz_localize(None))
        d = d.set_index(idx).sort_index()
        rets[t] = d["close"].pct_change()
        o, c = d["open"], d["close"]
        for h in HOLDS:
            fwd.setdefault(h, {})[t] = (c.shift(-h) / o.shift(-1) - 1.0)
    R = pd.DataFrame(rets).sort_index()
    F = {h: pd.DataFrame(v).reindex(R.index) for h, v in fwd.items()}
    return R, F


def cohesion(R: pd.DataFrame, members: dict[str, list[str]], dates: list[pd.Timestamp], rng) -> dict:
    """Mean pairwise forward correlation: themes vs size-matched random groups."""
    real, rand = [], []
    all_names = list(R.columns)
    for d in dates:
        win = R.loc[d:].iloc[1:COHESION_WINDOW + 1]
        if len(win) < COHESION_WINDOW - 2:
            continue
        for _theme, mem in members.items():
            cols = [m for m in mem if m in win.columns and win[m].notna().sum() >= COHESION_WINDOW - 4]
            if len(cols) < MIN_MEMBERS:
                continue
            c = win[cols].corr().to_numpy()
            iu = np.triu_indices_from(c, k=1)
            real.append(np.nanmean(c[iu]))
            pick = rng.choice(all_names, len(cols), replace=False)
            c2 = win[list(pick)].corr().to_numpy()
            rand.append(np.nanmean(c2[np.triu_indices_from(c2, k=1)]))
    return dict(theme=float(np.nanmean(real)), random=float(np.nanmean(rand)),
                n=len(real), diff=float(np.nanmean(real) - np.nanmean(rand)))


def day_cluster_boot(values: np.ndarray, days: np.ndarray, rng, n=2000) -> tuple[float, float]:
    uniq, inv = np.unique(days, return_inverse=True)
    groups = [values[inv == i] for i in range(len(uniq))]
    out = np.empty(n)
    for i in range(n):
        pick = rng.integers(0, len(groups), len(groups))
        out[i] = np.concatenate([groups[j] for j in pick]).mean()
    return tuple(np.percentile(out, [2.5, 97.5]))


def main() -> None:
    rng = np.random.default_rng(SEED)
    th = pd.read_parquet(THEMES, columns=["date", "ticker", "primary_theme"]).dropna()
    th["date"] = pd.to_datetime(th["date"])
    tickers = sorted(th["ticker"].unique())
    print(f"themes: {th['primary_theme'].nunique()} over {len(tickers)} tickers, "
          f"{th['date'].min().date()} .. {th['date'].max().date()}")
    R, F = load_returns(tickers)
    print(f"daily returns: {R.shape[0]} sessions x {R.shape[1]} tickers")

    flags = pd.read_parquet(FLAGS)
    flags = flags[~flags["organic"]]
    flagged = {(r.ticker, pd.Timestamp(r.date)) for r in flags.itertuples()}

    sector = {}
    if UNIVERSE.exists():
        u = pd.read_csv(UNIVERSE)
        if {"ticker", "sector_id"} <= set(u.columns):
            sector = dict(zip(u["ticker"].astype(str), u["sector_id"]))

    # membership snapshot per month-end date present in the theme history
    th["ym"] = th["date"].dt.to_period("M")
    snap_dates = th.groupby("ym")["date"].max()
    results: dict = {}

    # ---- A. forward cohesion
    sample = [d for d in snap_dates.tolist() if d in R.index]
    members_by_date = {d: th[th["date"] == d].groupby("primary_theme")["ticker"].apply(list).to_dict()
                       for d in sample}
    coh = []
    for d in sample:
        coh.append(cohesion(R, members_by_date[d], [d], rng))
    A = dict(theme=float(np.nanmean([c["theme"] for c in coh])),
             random=float(np.nanmean([c["random"] for c in coh])),
             months=len(coh), groups=int(np.sum([c["n"] for c in coh])))
    A["diff"] = A["theme"] - A["random"]
    print(f"\nA. FORWARD COHESION over {COHESION_WINDOW} sessions, {A['months']} monthly snapshots, "
          f"{A['groups']} theme-months")
    print(f"   mean pairwise correlation: theme {A['theme']:.4f} vs size-matched random "
          f"{A['random']:.4f}   diff {A['diff']:+.4f}")
    results["cohesion"] = A

    # ---- B. leader pop -> follower follow-through
    th_by_date = {d: g.set_index("ticker")["primary_theme"].to_dict()
                  for d, g in th.groupby("date") if d in R.index}
    rows = []
    shuffled_rows = []
    sector_rows = []
    for d, mapping in th_by_date.items():
        if d not in R.index:
            continue
        day_ret = R.loc[d]
        pops = [t for t, m in mapping.items() if t in day_ret.index and day_ret[t] >= POP_THRESHOLD]
        if not pops:
            continue
        groups: dict[str, list[str]] = {}
        for t, m in mapping.items():
            groups.setdefault(m, []).append(t)
        shuffle_map = dict(zip(list(mapping), rng.permutation(list(mapping.values()))))
        sgroups: dict[str, list[str]] = {}
        for t, m in shuffle_map.items():
            sgroups.setdefault(m, []).append(t)
        for leader in pops:
            for tag, gmap, store in (("theme", groups, rows), ("shuffled", sgroups, shuffled_rows)):
                mem = [x for x in gmap.get(gmap and (mapping if tag == "theme" else shuffle_map)[leader], [])
                       if x != leader]
                if len(mem) < MIN_MEMBERS - 1:
                    continue
                for h in HOLDS:
                    if h not in F:
                        continue
                    fol = F[h].loc[d, [m for m in mem if m in F[h].columns]]
                    fol = fol[[(m, d) not in flagged for m in fol.index]]
                    uni = F[h].loc[d].dropna()
                    if fol.notna().sum() < MIN_MEMBERS - 1 or len(uni) < 50:
                        continue
                    store.append(dict(day=d, hold=h, leader=leader,
                                      follower_ret=float(fol.mean()),
                                      excess=float(fol.mean() - uni.mean()),
                                      leader_fwd=float(F[h].loc[d, leader]) if leader in F[h].columns else np.nan,
                                      n_followers=int(fol.notna().sum())))
            if sector:
                sec = sector.get(leader)
                mem = [t for t in mapping if sector.get(t) == sec and t != leader]
                for h in HOLDS:
                    if h not in F or len(mem) < MIN_MEMBERS - 1:
                        continue
                    fol = F[h].loc[d, [m for m in mem if m in F[h].columns]].dropna()
                    uni = F[h].loc[d].dropna()
                    if len(fol) < MIN_MEMBERS - 1 or len(uni) < 50:
                        continue
                    sector_rows.append(dict(day=d, hold=h, excess=float(fol.mean() - uni.mean())))

    B = pd.DataFrame(rows)
    S = pd.DataFrame(shuffled_rows)
    SEC = pd.DataFrame(sector_rows)
    print(f"\nB. LEADER POP (>= {POP_THRESHOLD:.0%} in a day) -> FOLLOWERS, entered next open")
    print(f"   {B['leader'].nunique() if len(B) else 0} distinct leaders, "
          f"{B['day'].nunique() if len(B) else 0} event days")
    print(f"   {'hold':>5s} {'events':>7s} {'follower excess':>16s} {'95% CI':>20s} "
          f"{'shuffled':>10s} {'sector':>9s}")
    b_rows = []
    for h in HOLDS:
        sub = B[B["hold"] == h]
        if sub.empty:
            continue
        lo, hi = day_cluster_boot(sub["excess"].to_numpy(), sub["day"].to_numpy(), rng)
        sh = S[S["hold"] == h]["excess"].mean() if len(S) else np.nan
        se = SEC[SEC["hold"] == h]["excess"].mean() if len(SEC) else np.nan
        print(f"   {h:5d} {len(sub):7d} {sub['excess'].mean() * 100:15.3f}% "
              f"[{lo * 100:+7.3f},{hi * 100:+7.3f}] {sh * 100:9.3f}% {se * 100:8.3f}%")
        b_rows.append(dict(hold=h, n=int(len(sub)), excess=float(sub["excess"].mean()),
                           ci=[float(lo), float(hi)], shuffled=float(sh), sector=float(se)))
    results["leadlag"] = b_rows

    # ---- C. the 50/50 allocation
    print(f"\nC. ALLOCATION on leader-pop events (entered next open)")
    print(f"   {'hold':>5s} {'50/50':>9s} {'leader only':>12s} {'followers only':>15s} {'universe':>10s}")
    c_rows = []
    for h in HOLDS:
        sub = B[(B["hold"] == h)].dropna(subset=["leader_fwd"])
        if sub.empty:
            continue
        half = 0.5 * sub["leader_fwd"] + 0.5 * sub["follower_ret"]
        uni_mean = sub["follower_ret"] - sub["excess"]
        lo, hi = day_cluster_boot(half.to_numpy(), sub["day"].to_numpy(), rng)
        print(f"   {h:5d} {half.mean() * 100:8.3f}% {sub['leader_fwd'].mean() * 100:11.3f}% "
              f"{sub['follower_ret'].mean() * 100:14.3f}% {uni_mean.mean() * 100:9.3f}%   "
              f"50/50 CI [{lo * 100:+.3f},{hi * 100:+.3f}]")
        c_rows.append(dict(hold=h, n=int(len(sub)), half=float(half.mean()),
                           leader=float(sub["leader_fwd"].mean()),
                           followers=float(sub["follower_ret"].mean()),
                           universe=float(uni_mean.mean()), ci=[float(lo), float(hi)]))
    results["allocation"] = c_rows
    out = DATA / "theme_cohesion_leadlag.json"
    out.write_text(json.dumps(results, indent=1, default=str))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
