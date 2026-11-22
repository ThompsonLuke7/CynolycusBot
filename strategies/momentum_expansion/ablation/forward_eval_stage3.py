"""Forward (live) evaluation of the frozen Stage 3 momentum pick's shadow scores.

Rule fixed 2026-11-20, BEFORE any forward outcome existed
(research/execution_quality/30_stage3_forward_prereg.md):

  rows      momentum decision-state archive runs decided after the parity fix
            (``pre_parity_fix`` False), first run per decision bar, bars with at
            least 200 scored names, outcome status "ok" at the horizon;
  score     the shadow score residualised within the bar on the percentile ranks
            of daily ATR%, 60-bar beta, price and bar dollar volume (the
            archive's stand-ins for Stage 3's ATR/beta/px/ADV controls);
  outcome   SPY-excess return from the executable entry
            (decision_state_outcomes: fret_h - spy_fret_h), demeaned within the
            bar's ATR-decile (the L2 construction);
  metric    mean within-bar Spearman IC at h = 10 and 20 sessions, week-block
            bootstrap CI, with its MDE printed alongside.

Interpretation is fixed too. Validation needed about 3,500 bars to reach an MDE of 0.010
for an effect of about +0.009, so a forward read is EVIDENCE only once its MDE is at or
below 0.03 (about 12 months of bars). Before that it is monitoring: it can expose a
sign flip or a bug, but it cannot confirm or refute the pick. The live momentum
score is reported on the same rows for comparison.

    .venv/bin/python -m strategies.momentum_expansion.ablation.forward_eval_stage3
"""
from __future__ import annotations

import argparse
import glob
import math
from pathlib import Path

import numpy as np
import pandas as pd

from core.decision_state_archive import FEATURE_PREFIX, load_decision_state
from strategies.momentum_expansion.ablation.bootstrap import week_block_bootstrap_ci

REPO = Path(__file__).resolve().parents[3]
SHADOW = REPO / "Data/research/decision_state_shadow/stage3_L2_20"
OUTCOMES = REPO / "Data/research/decision_state_outcomes/momentum_expansion"
CONTROLS = ("daily_atr_pct", "beta_spy_60", "close", "dollar_volume")
MIN_NAMES = 200
EVIDENCE_MDE = 0.03


def _residual(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    A = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    return y - A @ beta


def bar_ics(frame: pd.DataFrame, h: int) -> pd.DataFrame:
    """One row per decision bar: IC of the neutral shadow score and of the live score."""
    out = []
    for bar, g in frame.groupby("bar_ts"):
        g = g[g[f"status_{h}"] == "ok"].dropna(subset=["stage3_score", f"fret_{h}", f"spy_fret_{h}",
                                                         *[FEATURE_PREFIX + c for c in CONTROLS]])
        if len(g) < MIN_NAMES:
            continue
        ranks = np.column_stack([g[FEATURE_PREFIX + c].rank(pct=True).to_numpy() for c in CONTROLS])
        neutral = _residual(g["stage3_score"].rank(pct=True).to_numpy(), ranks)
        xret = (g[f"fret_{h}"] - g[f"spy_fret_{h}"]).to_numpy()
        dec = pd.qcut(g[FEATURE_PREFIX + "daily_atr_pct"].rank(method="first"), 10, labels=False).to_numpy()
        l2 = xret - pd.Series(xret).groupby(dec).transform("mean").to_numpy()
        live = g["live_score"].to_numpy()
        ok_live = np.isfinite(live)
        out.append({
            "bar_ts": bar, "n": len(g),
            "ic_stage3_neutral": pd.Series(neutral).corr(pd.Series(l2), method="spearman"),
            "ic_live": (pd.Series(live[ok_live]).corr(pd.Series(l2[ok_live]), method="spearman")
                        if ok_live.sum() >= 50 else np.nan),
        })
    return pd.DataFrame(out)


def summarize(ics: pd.DataFrame, col: str, h: int) -> dict:
    v = ics.dropna(subset=[col])
    if len(v) < 2:
        return {"metric": col, "h": h, "bars": len(v), "mean": np.nan, "ci": [np.nan, np.nan], "mde": np.nan}
    weeks = pd.to_datetime(v["bar_ts"], utc=True).dt.tz_convert(None).dt.to_period("W").astype("int64") // max(1, math.ceil(h / 5))
    b = week_block_bootstrap_ci(v[col].reset_index(drop=True), weeks.reset_index(drop=True), n_boot=1000, alpha=0.05)
    se = (b["ci_hi"] - b["ci_lo"]) / (2 * 1.96) if np.isfinite(b["ci_hi"]) else np.nan
    return {"metric": col, "h": h, "bars": len(v), "mean": b["point"], "ci": [b["ci_lo"], b["ci_hi"]],
            "mde": 2.8 * se if np.isfinite(se) else np.nan}


def load_rows() -> pd.DataFrame:
    shadow = [pd.read_parquet(p) for p in sorted(glob.glob(str(SHADOW / "*.parquet")))]
    if not shadow:
        return pd.DataFrame()
    s = pd.concat(shadow, ignore_index=True)
    s = s[~s["pre_parity_fix"]]
    s["decision_ts"] = pd.to_datetime(s["decision_ts"], utc=True)
    s = s.sort_values("decision_ts").drop_duplicates(["bar_ts", "ticker"], keep="first")
    outs = [pd.read_parquet(p) for p in sorted(glob.glob(str(OUTCOMES / "*.parquet")))]
    if s.empty or not outs:
        return pd.DataFrame()
    o = pd.concat(outs, ignore_index=True)
    feats = load_decision_state(module="momentum_expansion",
                                columns=["run_id", "ticker", *[FEATURE_PREFIX + c for c in CONTROLS]])
    return (s.merge(o.drop(columns=["module", "decision_ts"], errors="ignore"), on=["run_id", "ticker"])
             .merge(feats, on=["run_id", "ticker"], how="left"))


def main() -> int:
    argparse.ArgumentParser(description=__doc__.split("\n")[0]).parse_args()
    rows = load_rows()
    if rows.empty:
        print("forward eval: no post-fix shadow rows with outcomes yet")
        return 0
    for h in (10, 20):
        ics = bar_ics(rows, h)
        for col in ("ic_stage3_neutral", "ic_live"):
            r = summarize(ics, col, h)
            verdict = ("EVIDENCE" if np.isfinite(r["mde"]) and r["mde"] <= EVIDENCE_MDE
                       else "monitoring only (MDE above %.2f)" % EVIDENCE_MDE)
            print(f"h={h:2d} {col:18s} bars={r['bars']:4d} IC={r['mean']:+.4f} "
                  f"CI=[{r['ci'][0]:+.4f}, {r['ci'][1]:+.4f}] MDE={r['mde']:.4f} -> {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
