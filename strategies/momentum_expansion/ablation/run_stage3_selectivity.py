"""Checklist 1.1 for the Stage 3 momentum-slot pick: net-of-cost selectivity curve.

VALIDATION ONLY (bars before the sealed test minus each horizon's embargo). The
question: can the pick's small ranking edge (neutral IC +0.009) be turned into a
long-only selection that beats SPY after costs, at some depth, holding period
and weighting? Every construction is compared against the same construction on
the pick's noise-trained twin (labels shuffled within the bar), because
noise-trained models carry vol/beta tilts that look like excess return
(project_topk_null_control).

Per bar and horizon h (sessions), for each selection:
  own_net   mean(fret_h - spy_fret_h - cost) of the selected names: what a long
            book returns over SPY after the assumed round-trip cost (price bands,
            run_subset_eval.COST_BANDS), entered at the next bar's open;
  vs_univ   mean(fret_h) of the selection minus the bar's equal-weight universe
            (gross selection value, before cost);
  - perm    the same metric minus the twin's.
Bars are overlapping entries (one tranche per 4H bar), the evaluator's convention.
Week-block bootstrap CIs; per-year means. Returns winsorised 1/99 within the bar.

A construction picked from this table is exploratory and must be frozen before
the test window is opened (about 2027-06).

    .venv/bin/python -m strategies.momentum_expansion.ablation.run_stage3_selectivity
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from strategies.momentum_expansion.ablation import eval_stage2 as E
from strategies.momentum_expansion.ablation import eval_stage3 as S3
from strategies.momentum_expansion.ablation.export_stage2 import FINAL_TEST_START

logger = logging.getLogger(__name__)
PICK, TWIN = "L2_20__pit", "L2_20__pit__perm"
DEPTHS = {"k1": 1, "k3": 3, "k5": 5, "k10": 10, "k20": 20, "k50": 50, "p5": 0.05, "p10": 0.10, "p20": 0.20}
WEIGHTS = ("equal", "inv_vol")
HORIZONS = (10, 20)


def _selection(p: E.Panel, score: np.ndarray, valid: np.ndarray, depth: float) -> np.ndarray:
    order, pos = E._positions(p.bar, score, valid)
    if depth >= 1:
        return order[pos < int(depth)]
    n = np.bincount(p.bar[order], minlength=p.n_bars)
    return order[pos < np.maximum(1, np.floor(depth * n[p.bar[order]])).astype(int)]


def _bar_wmean(p: E.Panel, x: np.ndarray, rows: np.ndarray, w: np.ndarray | None) -> np.ndarray:
    if w is None:
        return E._bar_mean(p.bar, p.n_bars, x, rows)
    b = p.bar[rows]
    with np.errstate(invalid="ignore", divide="ignore"):
        return (np.bincount(b, (w[rows] * x[rows]).astype(np.float64), minlength=p.n_bars)
                / np.bincount(b, w[rows].astype(np.float64), minlength=p.n_bars))


def run(*, results: Path, manifest_path: Path = S3.MANIFEST, out_dir: Path = S3.RESULTS_DIR,
        n_boot: int = 500, test_start: str = FINAL_TEST_START) -> pd.DataFrame:
    manifest = json.loads(manifest_path.read_text())
    arms = [a for a in manifest["arms"] if a["name"] in (PICK, TWIN)]
    results_dir = E.unpack(results)
    t0 = pd.Timestamp(test_start, tz="UTC")
    panel = E.load_panel(results_dir, manifest_path.parent / manifest["matrix"], [PICK, TWIN])
    panel = E.before(panel, t0)                     # the test window is never read
    panel, _ = S3.with_neutral(S3.winsorise_returns(panel), arms)
    ranks = {c: np.nan_to_num(E._group_rank(panel.bar, panel.col[c], pct=True), nan=0.5)
             for c in E.RESID_CONTROLS}
    atr = panel.col["daily_atr_pct"].astype(np.float64)
    inv_vol = np.where(np.isfinite(atr) & (atr > 0), 1.0 / np.clip(atr, 0.005, None), np.nan)
    fold = panel.fold[PICK]
    variants = {"raw": (PICK, TWIN), "neutral": (S3.neutral_name(PICK), S3.neutral_name(TWIN))}
    rows = []
    for h in HORIZONS:
        ret = panel.col[f"fret_{h}"].astype(np.float64)
        own = ret - panel.col[f"spy_fret_{h}"] - panel.col["cost"]
        base_valid = np.isfinite(ret) & np.isfinite(panel.col[f"spy_fret_{h}"])
        val = E.partitions(panel.bar_ts, h, t0)["validation"]
        thin = np.bincount(panel.bar[np.flatnonzero(base_valid)], minlength=panel.n_bars) < E.MIN_PAIRS
        univ_rows = np.flatnonzero(base_valid)
        ew = E._bar_mean(panel.bar, panel.n_bars, ret, univ_rows)
        univ_own = E._bar_mean(panel.bar, panel.n_bars, own, univ_rows)
        mask = val & ~thin
        u = E.summarize(univ_own, panel.bar_ts, h, fold, mask, n_boot=n_boot)
        rows.append({"h": h, "score": "universe", "weight": "equal", "depth": "all", "metric": "own_net", **u})
        for vname, (a, twin) in variants.items():
            for wname in WEIGHTS:
                w = None if wname == "equal" else inv_vol
                for dname, depth in DEPTHS.items():
                    per = {}
                    for who, arm in (("pick", a), ("twin", twin)):
                        s = panel.score[arm]
                        valid = base_valid & np.isfinite(s) & (np.isfinite(inv_vol) if w is not None else True)
                        sel = _selection(panel, s, valid, depth)
                        per[who] = {
                            "own_net": _bar_wmean(panel, own, sel, w),
                            "vs_univ": _bar_wmean(panel, ret, sel, w) - ew,
                            "atr_pct": E._bar_mean(panel.bar, panel.n_bars, ranks["daily_atr_pct"], sel),
                        }
                    ids = {"h": h, "score": vname, "weight": wname, "depth": dname}
                    for metric in ("own_net", "vs_univ"):
                        rows.append({**ids, "metric": metric,
                                     **E.summarize(per["pick"][metric], panel.bar_ts, h, fold, mask, n_boot=n_boot)})
                        rows.append({**ids, "metric": f"{metric}-perm",
                                     **E.summarize(per["pick"][metric] - per["twin"][metric], panel.bar_ts, h,
                                                   fold, mask, n_boot=n_boot)})
                    rows.append({**ids, "metric": "atr_pctile", "mean": float(np.nanmean(per["pick"]["atr_pct"][mask]))})
                logger.info("h=%d %s/%s done", h, vname, wname)
    table = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "selectivity_validation.csv", index=False)
    return table


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", type=Path, default=S3.DEFAULT_RESULTS)
    ap.add_argument("--n-boot", type=int, default=500)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    table = run(results=args.results, n_boot=args.n_boot)
    print(table[table["metric"] == "own_net"].sort_values("mean", ascending=False).head(12)
          [["h", "score", "weight", "depth", "mean", "ci_lo", "ci_hi", "n_bars"]].to_string(index=False))


if __name__ == "__main__":
    main()
