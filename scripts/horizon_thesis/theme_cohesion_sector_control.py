"""Sector control for the theme-cohesion number.

`theme_cohesion_leadlag.py` part A compares theme groups against SIZE-MATCHED RANDOM
groups, and themes win by a wide margin. That comparison is too easy: a random group
mixes sectors, so ordinary sector structure alone would produce the same gap. The
question that decides whether themes are worth anything is:

    do theme-mates co-move more than SAME-SECTOR names do?

Same forward 20-session window, same monthly snapshots, same pairwise-correlation
mechanic — only the grouping changes. Sector ids come from the shared universe when
present, otherwise from the momentum matrix's `sector_id` feature.
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
UNIVERSE = REPO / "Data/shared/universe/shared_universe.csv"
MOM_MATRIX = REPO / "strategies/momentum_expansion/data/processed/training_matrix_4h.parquet"
SEED = 41


def sector_map(tickers: list[str]) -> dict[str, str]:
    if UNIVERSE.exists():
        u = pd.read_csv(UNIVERSE)
        col = next((c for c in ("sector_id", "sector", "gics_sector") if c in u.columns), None)
        if col:
            m = {str(t): str(s) for t, s in zip(u["ticker"], u[col]) if pd.notna(s)}
            if len(m) > 100:
                print(f"sector source: {UNIVERSE.name} column '{col}' ({len(m)} tickers)")
                return m
    df = pd.read_parquet(MOM_MATRIX, columns=["sector_id"])
    last = df.groupby(level=1)["sector_id"].last()
    m = {str(t): str(s) for t, s in last.items() if pd.notna(s)}
    print(f"sector source: momentum matrix sector_id ({len(m)} tickers)")
    return m


def mean_pairwise(win: pd.DataFrame, cols: list[str]) -> float:
    c = win[cols].corr().to_numpy()
    return float(np.nanmean(c[np.triu_indices_from(c, k=1)]))


def main() -> None:
    rng = np.random.default_rng(SEED)
    th = pd.read_parquet(THEMES, columns=["date", "ticker", "primary_theme"]).dropna()
    th["date"] = pd.to_datetime(th["date"])
    tickers = sorted(th["ticker"].unique())
    R, _F = load_returns(tickers)
    sec = sector_map(tickers)
    th["ym"] = th["date"].dt.to_period("M")
    snaps = [d for d in th.groupby("ym")["date"].max().tolist() if d in R.index]
    print(f"{len(snaps)} monthly snapshots, {R.shape[1]} tickers with returns")

    theme_v, sector_v, rand_v, sector_sizematched = [], [], [], []
    for d in snaps:
        win = R.loc[d:].iloc[1:COHESION_WINDOW + 1]
        if len(win) < COHESION_WINDOW - 2:
            continue
        day = th[th["date"] == d]
        groups = day.groupby("primary_theme")["ticker"].apply(list).to_dict()
        pool = [t for t in day["ticker"] if t in win.columns and win[t].notna().sum() >= COHESION_WINDOW - 4]
        by_sec: dict[str, list[str]] = {}
        for t in pool:
            if t in sec:
                by_sec.setdefault(sec[t], []).append(t)
        for _theme, mem in groups.items():
            cols = [m for m in mem if m in pool]
            if len(cols) < MIN_MEMBERS:
                continue
            theme_v.append(mean_pairwise(win, cols))
            rand_v.append(mean_pairwise(win, list(rng.choice(pool, len(cols), replace=False))))
            # size-matched sector draw: same count, drawn from ONE sector
            big = [s for s, v in by_sec.items() if len(v) >= len(cols)]
            if big:
                s = big[int(rng.integers(0, len(big)))]
                sector_sizematched.append(mean_pairwise(win, list(rng.choice(by_sec[s], len(cols), replace=False))))
        for _s, v in by_sec.items():
            if len(v) >= MIN_MEMBERS:
                sector_v.append(mean_pairwise(win, v))

    out = dict(theme=float(np.nanmean(theme_v)), sector_full=float(np.nanmean(sector_v)),
               sector_sizematched=float(np.nanmean(sector_sizematched)), random=float(np.nanmean(rand_v)),
               n_theme=len(theme_v), n_sector=len(sector_v), n_sector_sm=len(sector_sizematched))
    out["theme_minus_sector_sm"] = out["theme"] - out["sector_sizematched"]
    print(f"\nforward {COHESION_WINDOW}-session mean pairwise correlation:")
    print(f"  theme groups          {out['theme']:.4f}  (n={out['n_theme']})")
    print(f"  sector, size-matched  {out['sector_sizematched']:.4f}  (n={out['n_sector_sm']})")
    print(f"  sector, whole sector  {out['sector_full']:.4f}  (n={out['n_sector']})")
    print(f"  random                {out['random']:.4f}  (n={out['n_theme']})")
    print(f"\n  theme - size-matched sector = {out['theme_minus_sector_sm']:+.4f}")
    print("  (positive = themes group more tightly than sector membership alone)")
    p = DATA / "theme_cohesion_sector_control.json"
    p.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
