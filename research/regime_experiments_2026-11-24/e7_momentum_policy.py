"""E7-M: does a regime-aware order policy beat the current one for the momentum slot?
(PLAN.md "E7-M", pre-registered 2026-11-25 ~00:56 ET before running.)

Stage 3 out-of-fold scores of L2_20__pit, validation bars only (bars before FINAL_TEST_START minus
the 20-session embargo; the sealed test is never loaded into any statistic). At every 4H bar the
book buys the top 10 by score; each trade earns fret_20 minus the matrix cost (absolute, not
SPY-relative: the live module is long-only and unhedged).

Regime: long-history Nasdaq vote (regime_common) from the last session strictly before the bar's
session. Secondary: the live 7-vote (build_daily_regime + classify_frame, by available_at).
Policies: P0 constant; P1 live multipliers by state; P2 no entries when vote >= 4.
Primary metric: R = sum(w*r) / sum(w), return per unit of exposure.
Control: circular shift of the daily regime series (>= 60 sessions, 1,000 draws).
CI: block bootstrap over ceil(20/5)=4-week blocks of bars (Stage 3's blocks), 1,000 draws.
Decision: a profile goes to the E8 paper shadow only if R(P) - R(P0) > 0 AND it is at or above the
shuffled control's 95th percentile.

    PYTHONPATH=. .venv/bin/python -W ignore research/regime_experiments_2026-11-24/e7_momentum_policy.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import regime_common as rc  # noqa: E402

from strategies.momentum_expansion.ablation import eval_stage2 as E  # noqa: E402
from strategies.momentum_expansion.ablation.build_stage3_pit import EXPORT_DIR  # noqa: E402
from strategies.momentum_expansion.ablation.export_stage2 import FINAL_TEST_START  # noqa: E402

ARM = "L2_20__pit"
H = 20
K = 10
DRAWS = 1000
SEED = 20261125
RESULTS = rc.REPO / "stage3_results"
MATRIX = EXPORT_DIR / "stage3_matrix.parquet"


def load_validation() -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray, np.ndarray]:
    panel = E.load_panel(RESULTS, MATRIX, [ARM])
    test_start = pd.Timestamp(FINAL_TEST_START, tz="UTC")
    cut = test_start - pd.Timedelta(days=E.embargo_days(H))
    p = E.before(panel, cut)
    del panel
    if p.bar_ts.max() >= test_start:
        raise RuntimeError("sealed test bars reached the analysis")
    score = p.score[ARM]
    ret = (p.col[f"fret_{H}"] - p.col["cost"]).astype(np.float64)
    valid = np.isfinite(score) & np.isfinite(p.col[f"fret_{H}"])
    order, pos = E._positions(p.bar, score, valid)
    top = order[pos < K]
    r_top = E._bar_mean(p.bar, p.n_bars, ret, top)
    r_all = E._bar_mean(p.bar, p.n_bars, ret, np.flatnonzero(valid))
    thin = np.bincount(p.bar[np.flatnonzero(valid)], minlength=p.n_bars) < E.MIN_PAIRS
    r_top[thin], r_all[thin] = np.nan, np.nan
    return p.bar_ts, r_top, r_all, E._blocks(p.bar_ts, H)


def daily_states() -> tuple[pd.DataFrame, pd.DataFrame]:
    fr, _ = rc.build_frame("ndx")
    lh = fr[["vote7", "state"]].dropna().loc["2017-01-01":]
    from signals.market_regime.daily_regime import build_daily_regime
    from signals.market_regime.regime_vote import classify_frame

    live = build_daily_regime()
    cls = classify_frame(live)
    lv = pd.DataFrame({"vote7": cls["regime_vote_count"].to_numpy(), "state": cls["regime"].to_numpy(),
                       "available_at": pd.to_datetime(live["available_at"], utc=True).to_numpy()},
                      index=pd.to_datetime(live["date"])).dropna(subset=["vote7"])
    return lh, lv


def map_to_bars(bar_ts: pd.DatetimeIndex, daily: pd.DataFrame, by_available_at: bool) -> np.ndarray:
    """Index into ``daily`` of the row in force at each bar (-1 when none)."""
    if by_available_at:
        avail = pd.DatetimeIndex(daily["available_at"])
        return np.searchsorted(avail.asi8, bar_ts.asi8, side="right") - 1
    sess = bar_ts.tz_convert("America/New_York").normalize().tz_localize(None)
    return np.searchsorted(daily.index.asi8, sess.asi8, side="left") - 1  # strictly before the session


def weights(states: np.ndarray, votes: np.ndarray, policy: str) -> np.ndarray:
    if policy == "P0":
        return np.ones(len(states))
    if policy == "P1":
        return np.array([rc.LIVE_SIZE.get(s, np.nan) for s in states])
    return np.where(votes >= 4, 0.0, 1.0)


def ratio(w: np.ndarray, r: np.ndarray) -> float:
    ok = np.isfinite(w) & np.isfinite(r)
    return float((w[ok] * r[ok]).sum() / w[ok].sum()) if w[ok].sum() > 0 else np.nan


def analyse(bar_ts, r_top, r_all, blocks, daily: pd.DataFrame, by_avail: bool, label: str, rng) -> dict:
    ix = map_to_bars(bar_ts, daily, by_avail)
    keep = (ix >= 0) & np.isfinite(r_top)
    ix, r, ra, bl, ts = ix[keep], r_top[keep], r_all[keep], blocks[keep], bar_ts[keep]
    st, vt = daily["state"].to_numpy(), daily["vote7"].to_numpy()
    out = {"label": label, "bars": int(keep.sum()), "first": str(ts[0].date()), "last": str(ts[-1].date()),
           "R_P0": ratio(np.ones(len(r)), r)}
    # descriptive: by state, top-10 net return, market (all-name) return and selection (top - all)
    by = pd.DataFrame({"state": st[ix], "r_top": r, "r_all": ra, "sel": r - ra})
    out["by_state"] = by.groupby("state").agg(bars=("r_top", "size"), r_top=("r_top", "mean"), r_top_sd=("r_top", "std"),
                                              r_top_p10=("r_top", lambda x: x.quantile(0.10)),
                                              r_all=("r_all", "mean"), selection=("sel", "mean")).round(5).to_dict("index")
    out["by_year"] = by.assign(year=ts.year).groupby("year")[["r_top", "r_all", "sel"]].mean().round(5).to_dict("index")
    ub = np.unique(bl, return_inverse=True)[1]
    nb = ub.max() + 1
    n = len(st)
    for pol in ("P1", "P2"):
        w = weights(st[ix], vt[ix], pol)
        d = ratio(w, r) - out["R_P0"]
        shifts = rng.integers(60, n - 60, DRAWS)
        null = np.array([ratio(weights(np.roll(st, s)[ix], np.roll(vt, s)[ix], pol), r) - out["R_P0"] for s in shifts])
        boot = np.empty(DRAWS)
        for k in range(DRAWS):
            pick = rng.integers(0, nb, nb)
            m = np.concatenate([np.flatnonzero(ub == b) for b in pick])
            boot[k] = ratio(w[m], r[m]) - ratio(np.ones(len(m)), r[m])
        out[pol] = {"R": ratio(w, r), "delta": d, "exposure": float(np.nanmean(w)),
                    "delta_ci": [float(np.nanpercentile(boot, 2.5)), float(np.nanpercentile(boot, 97.5))],
                    "null_p95": float(np.percentile(null, 95)), "placebo_pct": float((null < d).mean() * 100),
                    "SHADOW": bool(d > 0 and d >= np.percentile(null, 95))}
    return out


def main() -> int:
    rc.OUT.mkdir(exist_ok=True)
    bar_ts, r_top, r_all, blocks = load_validation()
    print("validation bars", len(bar_ts), bar_ts[0], bar_ts[-1], flush=True)
    lh, lv = daily_states()
    rng = np.random.default_rng(SEED)
    res = {"arm": ARM, "k": K, "h": H,
           "lh_vote": analyse(bar_ts, r_top, r_all, blocks, lh, False, "long-history vote (prior session)", rng),
           "live_vote": analyse(bar_ts, r_top, r_all, blocks, lv, True, "live 7-vote (by available_at)", rng)}
    (rc.OUT / "e7_momentum_summary.json").write_text(json.dumps(res, indent=2, default=str))
    print(json.dumps(res, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
