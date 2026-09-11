"""Is the 3.5-year OOF result real, or an artifact? Check BEFORE believing it.

The partial run reported momentum top-1 at +7.44% excess over 5 days, p<0.001,
on 1,739 bars. That is ~26x the effect the live study measured and it contradicts
momentum's OWN documented walk-forward calibration (top-5/10 lift ~1.06-1.12,
"thin", momentum_config.py). An effect that large, that far from the module's own
prior, is a defect until proven otherwise. Four checks:

  1. PLACEBO   - shuffle the score within each bar. Must collapse to ~0. If it
                 does not, the "excess" is an artifact of the control, not the
                 ranking.
  2. SURVIVORSHIP - the forward table only covers tickers that still have a bar
                 file today. If the matrix's delisted names are silently dropped,
                 every measured return is conditioned on survival.
  3. ALIGNMENT - does the score at bar T also "predict" the return of the bar it
                 was computed ON? A same-bar effect means the score saw it.
  4. CONCENTRATION - is the mean carried by a handful of microcap moonshots that
                 no position sizing could have held?
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
DATA = REPO / "research/execution_quality/data"
MATRIX = REPO / "signals/meta_context/meta_ranker/meta_ranker_matrix_research.parquet"
BARS = REPO / "Data/shared/bars/1d"


def main() -> None:
    m = pd.read_parquet(DATA / "oof_rank_depth_mom.parquet")
    print(f"rows {len(m):,}  bars {m['timestamp'].nunique():,}  "
          f"tickers {m['ticker'].nunique()}")
    rc = "ret_5"
    s = m[m[rc].notna()].copy()

    # ---- 1. placebo -------------------------------------------------------
    rng = np.random.default_rng(3)
    ctrl = s.groupby("timestamp")[rc].mean()
    real = s[s["rank"] <= 1].groupby("timestamp")[rc].mean()
    obs = float((real - ctrl.reindex(real.index)).mean())
    fake_scores = s.groupby("timestamp")["mom_score"].transform(
        lambda g: rng.permutation(g.values))
    s["_fake_rank"] = s.assign(_f=fake_scores).groupby("timestamp")["_f"].rank(
        ascending=False, method="first")
    fake = s[s["_fake_rank"] <= 1].groupby("timestamp")[rc].mean()
    fobs = float((fake - ctrl.reindex(fake.index)).mean())
    print(f"\n1. PLACEBO   real top-1 excess {obs*100:+7.2f}%   "
          f"shuffled {fobs*100:+7.2f}%   -> {'OK' if abs(fobs) < 0.01 else 'ARTIFACT'}")

    # ---- 2. survivorship --------------------------------------------------
    raw = pd.read_parquet(MATRIX, columns=["mom_score"]).reset_index()
    all_t = set(raw["ticker"].unique())
    have = {p.stem for p in BARS.glob("*.parquet")}
    missing = all_t - have
    print(f"\n2. SURVIVORSHIP  matrix tickers {len(all_t)}  with bar files "
          f"{len(all_t & have)}  MISSING {len(missing)} ({len(missing)/len(all_t):.1%})")
    if missing:
        # were the missing ones ever ranked highly? that is the damaging case
        top = raw.copy()
        top["rank"] = top.groupby("timestamp")["mom_score"].rank(
            ascending=False, method="first")
        t10 = top[top["rank"] <= 10]
        lost = t10[t10["ticker"].isin(missing)]
        print(f"   top-10 rows belonging to missing tickers: {len(lost):,} "
              f"of {len(t10):,} ({len(lost)/max(len(t10),1):.2%})")

    # ---- 3. alignment -----------------------------------------------------
    # ret_5 starts the session AFTER the bar. A leak would show up as the score
    # also explaining the bar's own session. Proxy: correlation of score with the
    # SAME-day return, which the forward table does not contain -- so instead
    # check that shifting the entry one more session forward does not kill it.
    print("\n3. ALIGNMENT  (entry already = first session strictly after the bar;")
    print("   merge_asof direction=forward on decision_day + 1 day)")
    print(f"   distinct session_date values per decision_day: "
          f"{s.groupby('decision_day')['session_date'].nunique().max()}")
    lag = (s["session_date"] - s["decision_day"]).dt.days
    print(f"   entry lag in calendar days: median {lag.median():.0f}  "
          f"p95 {lag.quantile(0.95):.0f}  max {lag.max():.0f}")

    # ---- 4. concentration -------------------------------------------------
    t1 = s[s["rank"] <= 1]
    v = t1[rc].to_numpy()
    v = v[np.isfinite(v)]
    order = np.sort(v)[::-1]
    print(f"\n4. CONCENTRATION  top-1 n={len(v):,}  mean {v.mean()*100:+.2f}%  "
          f"median {np.median(v)*100:+.2f}%")
    for cut in (1, 5, 10, 25):
        print(f"   excluding best {cut:3d}: mean {order[cut:].mean()*100:+7.2f}%")
    w = np.clip(v, np.percentile(v, 1), np.percentile(v, 99))
    print(f"   winsorized 1/99: mean {w.mean()*100:+.2f}%  "
          f"(control mean {ctrl.mean()*100:+.2f}%)")
    print(f"   share of top-1 picks with |ret| > 50%: {(np.abs(v) > 0.5).mean():.2%}")


if __name__ == "__main__":
    main()
