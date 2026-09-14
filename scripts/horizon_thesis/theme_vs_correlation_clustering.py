"""Does the theme machinery beat plain trailing-correlation clustering?

Themes co-move forward far more than random groups (0.387 vs 0.217) and more than
sector groups (0.233). But the theme embedding is text BLENDED WITH a 60-day trailing
co-movement vector, and correlation is persistent, so some of that forward cohesion
could come from the trailing correlation alone -- no news, no Claude, no taxonomy.

This builds the cheap statistical alternative at each monthly snapshot: cluster the
SAME tickers on their trailing 60-session correlations into the SAME number of groups,
then measure forward 20-session cohesion exactly as before.

  theme >> correlation-clusters   the text/taxonomy adds real grouping information
  theme ~= correlation-clusters   the grouping is just co-movement; the news +
                                  Claude pipeline is not what makes it work

Strictly point-in-time: the trailing window ends at the snapshot date, the forward
window starts the session after it.
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

from theme_cohesion_leadlag import COHESION_WINDOW, MIN_MEMBERS, THEMES, load_returns  # noqa: E402

DATA = REPO / "research/execution_quality/data"
TRAIL_WINDOW = 60
SEED = 53


def mean_pairwise(win: pd.DataFrame, cols: list[str]) -> float:
    c = win[cols].corr().to_numpy()
    return float(np.nanmean(c[np.triu_indices_from(c, k=1)]))


def main() -> None:
    from sklearn.cluster import KMeans

    rng = np.random.default_rng(SEED)
    th = pd.read_parquet(THEMES, columns=["date", "ticker", "primary_theme"]).dropna()
    th["date"] = pd.to_datetime(th["date"])
    R, _F = load_returns(sorted(th["ticker"].unique()))
    th["ym"] = th["date"].dt.to_period("M")
    snaps = [d for d in th.groupby("ym")["date"].max().tolist() if d in R.index]
    print(f"{len(snaps)} monthly snapshots; trailing {TRAIL_WINDOW}d clustering vs themes")

    theme_v, clust_v, sizes_t, sizes_c = [], [], [], []
    for n_done, d in enumerate(snaps, 1):
        fwd = R.loc[d:].iloc[1:COHESION_WINDOW + 1]
        trail = R.loc[:d].iloc[-TRAIL_WINDOW:]
        if len(fwd) < COHESION_WINDOW - 2 or len(trail) < TRAIL_WINDOW - 5:
            continue
        day = th[th["date"] == d]
        pool = [t for t in day["ticker"].unique()
                if t in fwd.columns and fwd[t].notna().sum() >= COHESION_WINDOW - 4
                and t in trail.columns and trail[t].notna().sum() >= TRAIL_WINDOW - 10]
        if len(pool) < 100:
            continue
        groups = {k: [t for t in v if t in pool]
                  for k, v in day.groupby("primary_theme")["ticker"].apply(list).to_dict().items()}
        groups = {k: v for k, v in groups.items() if len(v) >= MIN_MEMBERS}
        if not groups:
            continue
        corr = trail[pool].corr().fillna(0.0)
        k = len(groups)
        km = KMeans(n_clusters=k, n_init=4, random_state=SEED).fit(corr.to_numpy())
        lab = pd.Series(km.labels_, index=pool)
        for _t, mem in groups.items():
            theme_v.append(mean_pairwise(fwd, mem))
            sizes_t.append(len(mem))
        for c in range(k):
            mem = lab[lab == c].index.tolist()
            if len(mem) >= MIN_MEMBERS:
                clust_v.append(mean_pairwise(fwd, mem))
                sizes_c.append(len(mem))
        if n_done % 10 == 0:
            print(f"  {n_done}/{len(snaps)} snapshots", flush=True)

    out = dict(theme=float(np.nanmean(theme_v)), corr_clusters=float(np.nanmean(clust_v)),
               n_theme=len(theme_v), n_clusters=len(clust_v),
               median_size_theme=float(np.median(sizes_t)), median_size_cluster=float(np.median(sizes_c)))
    out["theme_minus_clusters"] = out["theme"] - out["corr_clusters"]
    print(f"\nforward {COHESION_WINDOW}-session mean pairwise correlation:")
    print(f"  theme groups                {out['theme']:.4f}  (n={out['n_theme']}, median size {out['median_size_theme']:.0f})")
    print(f"  trailing-correlation groups {out['corr_clusters']:.4f}  (n={out['n_clusters']}, median size {out['median_size_cluster']:.0f})")
    print(f"\n  theme - correlation clustering = {out['theme_minus_clusters']:+.4f}")
    print("  (~0 means the news/Claude taxonomy is not what creates the grouping)")
    p = DATA / "theme_vs_correlation_clustering.json"
    p.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
