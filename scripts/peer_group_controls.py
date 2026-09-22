"""Is the peer grouping anything more than a beta tilt?

Forward-20d mean pairwise correlation for each correlation group, against
size-matched controls drawn from the SAME screened pool:

  random        - any names at all
  beta-decile   - names from the group's modal trailing-beta decile
  sector        - names from the group's modal sector (REQUIRES sector data)

If the groups only beat `random`, they are a beta/liquidity tilt with a nice
name. The beta-decile arm is the one that settles that.

SECTOR ARM: the shared universe snapshot's `sector` column is NOT usable --
on the 2026-09-11 snapshot it is 68% NaN and 32% the literal string "Unknown",
i.e. two non-values. The canonical `signals.market_regime.sector_map.SECTOR_MAP`
covers 95/2903 names (3.3%), and the empirical assignments parquet does not
exist. So the sector arm is DISABLED unless a real mapping is supplied; an
earlier run of this script reported a sector control of 0.0804 that was
meaningless for exactly this reason, and it is retracted. The guard below
refuses to emit the arm rather than print a number that looks like evidence.
"""
from __future__ import annotations
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/home/luket/repos/CynolycusBot")
sys.path.insert(0, str(REPO))

from core.shared_universe.universe import load_universe_as_of  # noqa: E402
from signals.peer_structure.correlation_groups import (  # noqa: E402
    PeerGroupConfig, build_peer_groups, load_screened_returns,
)

rng = np.random.default_rng(11)
WIN = 20

uni = load_universe_as_of("2026-09-11T23:59:00Z")
tickers = sorted(uni["ticker"].astype(str).unique())
_NON_VALUES = {"nan", "none", "", "?", "unknown", "na", "n/a"}
sector_of: dict[str, str] = {}
for col in ("sector", "gics_sector"):
    if col not in uni.columns:
        continue
    candidate = {
        str(t): str(v)
        for t, v in zip(uni["ticker"].astype(str), uni[col].astype(str))
        if str(v).strip().lower() not in _NON_VALUES
    }
    # A "sector" column with one or two values is not a sector column. Refuse
    # it rather than draw a control from an arbitrary bucket.
    if len(set(candidate.values())) >= 4 and len(candidate) >= 0.5 * len(uni):
        sector_of = candidate
        print(f"sector control keyed on '{col}': "
              f"{len(candidate)} names, {len(set(candidate.values()))} sectors")
        break
    print(f"sector column '{col}' unusable: {len(candidate)} named, "
          f"{len(set(candidate.values()))} distinct — arm DISABLED")
SECTOR_OK = bool(sector_of)

cfg = PeerGroupConfig()
full = load_screened_returns(tickers, as_of_session=pd.Timestamp("2026-09-15"), config=cfg)

def fwd(members, start_i, R=full, win=WIN):
    cols = [m for m in members if m in R.columns]
    if len(cols) < 3:
        return np.nan
    w = R.iloc[start_i + 1: start_i + 1 + win][cols]
    if len(w) < win // 2:
        return np.nan
    c = w.corr().to_numpy()
    v = c[np.triu_indices_from(c, k=1)]
    return float(np.nanmean(v)) if v.size else np.nan

rows = []
for asof_str in ["2026-05-30", "2026-06-30", "2026-07-31"]:
    asof = pd.Timestamp(asof_str)
    R = load_screened_returns(tickers, as_of_session=asof, config=cfg)
    groups = build_peer_groups(R, as_of_session=asof, config=cfg, universe_version="ctrl")
    idx = full.index.searchsorted(asof)
    pool = [c for c in R.columns if c in full.columns]

    # trailing beta vs the equal-weight pool, for the beta-decile control
    trail = R[pool].tail(cfg.lookback_sessions)
    mkt = trail.mean(axis=1)
    var = float(mkt.var())
    beta = {t: float(trail[t].cov(mkt) / var) if var > 0 else np.nan for t in pool}
    bser = pd.Series(beta).dropna()
    decile = pd.qcut(bser.rank(method="first"), 10, labels=False)
    by_decile = {d: list(bser.index[decile == d]) for d in range(10)}

    by_sector: dict[str, list[str]] = {}
    for t in pool:
        by_sector.setdefault(sector_of.get(t, "?"), []).append(t)

    g_v, r_v, s_v, b_v = [], [], [], []
    for g in groups:
        n = len(g.members)
        g_v.append(fwd(list(g.members), idx))
        r_v.append(fwd(list(rng.choice(pool, size=min(n, len(pool)), replace=False)), idx))

        if SECTOR_OK:
            sec = Counter(sector_of.get(m, "?") for m in g.members).most_common(1)[0][0]
            cand = by_sector.get(sec, [])
            s_v.append(fwd(list(rng.choice(cand, size=min(n, len(cand)), replace=False)), idx)
                       if len(cand) >= 3 else np.nan)
        else:
            s_v.append(np.nan)

        dec = Counter(int(decile[m]) for m in g.members if m in decile.index).most_common(1)
        cand = by_decile.get(dec[0][0], []) if dec else []
        b_v.append(fwd(list(rng.choice(cand, size=min(n, len(cand)), replace=False)), idx)
                   if len(cand) >= 3 else np.nan)

    rows.append((asof_str, len(groups), np.nanmean(g_v), np.nanmean(r_v),
                 np.nanmean(s_v), np.nanmean(b_v)))

def _cell(v):
    return "     n/a" if v != v else f"{v:>9.4f}"


print(f"\n{'as_of':<12}{'grp':>5}{'group':>9}{'random':>9}{'sector':>9}{'beta-dec':>10}")
for a, n, g, r, sec_v, b in rows:
    print(f"{a:<12}{n:>5}{g:>9.4f}{r:>9.4f}{_cell(sec_v)}{b:>10.4f}")
m = [np.nanmean([r[i] for r in rows]) if i != 4 or SECTOR_OK else float("nan")
     for i in (2, 3, 4, 5)]
print(f"{'MEAN':<12}{'':>5}{m[0]:>9.4f}{m[1]:>9.4f}{_cell(m[2])}{m[3]:>10.4f}")
print(f"\nedge vs random      {m[0]-m[1]:+.4f}")
print(f"edge vs beta-decile {m[0]-m[3]:+.4f}")
if SECTOR_OK:
    print(f"edge vs sector      {m[0]-m[2]:+.4f}")
else:
    print("edge vs sector      UNAVAILABLE (no usable sector mapping)")
print("\nCaveats: membership is the 2026-09-11 snapshot (survivorship; both arms")
print("draw from the same screened pool, so the CONTRAST is fair, the LEVELS are")
print("not); 3 as-of dates, no confidence intervals -- directional, not a test.")
