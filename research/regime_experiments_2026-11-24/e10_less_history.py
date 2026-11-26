"""E10: is all that history hurting us? Rolling windows vs expanding (PLAN.md section 5,
pre-registered 2026-11-25 21:31 ET before running).

Same features, spans, purge and walk-forward as E3/E3b; only the training window changes:
expanding (all history, the current design) vs the most recent 10, 5 or 3 calendar years.
Models: clipped logistic on base; clipped logistic on base + long-span families; LightGBM on base +
long-span families. Targets: T1 (Nasdaq 20d <= -5%), T2 (60d <= -10%). HMM-R K=2 sizing books per
window (E9 execution).
Improvement = log-loss(expanding) - log-loss(rolling) (positive = less history is better), per
decade, with a 60-session block-bootstrap 95% CI over all years.
Decision: less history HELPS if a rolling window beats expanding in >= 2 of the last 3 decades AND
its all-years CI is above 0. The 2020s cell is reported separately ("unprecedented times").

    OMP_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python -W ignore research/regime_experiments_2026-11-24/e10_less_history.py
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e3_factor_completeness as e3  # noqa: E402
import e3b_null_audit as e3b  # noqa: E402
import e9_regime_model as e9  # noqa: E402
import regime_common as rc  # noqa: E402

SEED = 20261125
WINDOWS = (None, 10, 5, 3)
BOOT = 1000
BLOCK = 60


def ll(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def block_ci(d: np.ndarray, rng) -> tuple[float, float]:
    n = len(d)
    nb = int(np.ceil(n / BLOCK))
    out = np.empty(BOOT)
    for k in range(BOOT):
        idx = ((rng.integers(0, n, nb)[:, None] + np.arange(BLOCK)[None, :]) % n).ravel()[:n]
        out[k] = d[idx].mean()
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def main() -> int:
    warnings.simplefilter("ignore")
    e3.LGB_PARAMS = {**e3.LGB_PARAMS, "n_jobs": 4}
    rng = np.random.default_rng(SEED)
    rows = []
    for tname, h, thr in (("T1", 20, -0.05), ("T2", 60, -0.10)):
        F, fam, y, _ = e3b.features_for("ndx", h, thr)
        long_cols = fam["base"] + [c for k in e9.LONG_FAMS for c in fam[k]]
        for mname, cols, model in (("lr_base", fam["base"], "lr"), ("lr_long", long_cols, "lr"),
                                   ("lgb_long", long_cols, "lgb")):
            sub, ys, years, test_years, _ = e3.span_setup(F, y, cols)
            X = sub[cols].to_numpy()
            dec = rc.decade_of(sub.index).to_numpy()
            preds = {w: e3.walk_forward(X, ys, years, test_years, model=model, purge=h, winsor=(model == "lr"),
                                        window_years=w) for w in WINDOWS}
            ok = ~np.isnan(ys)
            for w in WINDOWS:
                ok &= ~np.isnan(preds[w])
            base_ll = ll(ys[ok], preds[None][ok])
            for w in WINDOWS[1:]:
                d = base_ll - ll(ys[ok], preds[w][ok])
                by_dec = pd.Series(d).groupby(dec[ok]).mean()
                lo, hi = block_ci(d, rng)
                have = [x for x in ("2000s", "2010s", "2020s") if x in by_dec.index]
                wins = int(sum(by_dec[x] > 0 for x in have))
                rows.append({"target": tname, "model": mname, "window_years": w, "first_test_year": test_years[0],
                             "all_years": float(d.mean()), "ci_lo": lo, "ci_hi": hi,
                             **{f"dec_{k}": float(v) for k, v in by_dec.items()},
                             "decade_wins_last3": wins, "LESS_HISTORY_HELPS": bool(wins >= 2 and lo > 0)})
                print(rows[-1], flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(rc.OUT / "e10_windows.csv", index=False)
    # HMM sizing books by window
    p, _ = rc.latest_panel()
    r = (np.log(p["ndx"]).diff() * 100).dropna()
    weights = {}
    for w in WINDOWS:
        ph, _ = e9.hmm_walk_forward(r.to_numpy()[:, None], r.index, 2, 1996, window_years=w)
        s = pd.Series(ph, index=r.index).dropna()
        weights[f"hmm_w{w or 'exp'}_size"] = 1 - 0.75 * s
        weights[f"hmm_w{w or 'exp'}_longflat"] = (s <= 0.5).astype(float)
    start = max(v.index[0] for v in weights.values())
    books, crashes = e9.books_on_span(weights, start, p.index[-1])
    hm = pd.DataFrame([{"book": k, **{kk: (str(vv) if isinstance(vv, tuple) else vv) for kk, vv in v.items()}}
                       for k, v in books.items()])
    hm.to_csv(rc.OUT / "e10_hmm_books.csv", index=False)
    crashes.to_csv(rc.OUT / "e10_hmm_crashes.csv", index=False)
    (rc.OUT / "e10_summary.json").write_text(json.dumps(
        {"any_helps": bool(res["LESS_HISTORY_HELPS"].any()), "hmm_span_start": str(start.date()),
         "verdicts": res[["target", "model", "window_years", "all_years", "ci_lo", "ci_hi", "decade_wins_last3",
                          "LESS_HISTORY_HELPS"]].to_dict("records")}, indent=2, default=str))
    pd.set_option("display.width", 250)
    print(res.round(4).to_string(index=False))
    print(hm[["book", "cagr", "sharpe", "mdd", "invested", "turnover_per_yr"]].round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
