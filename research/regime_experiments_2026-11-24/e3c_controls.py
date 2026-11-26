"""E3c: does E3's test machinery work? Known-answer controls (added 2026-11-25 ~01:15 ET, after E3/E3b,
to answer "are we running the experiments correctly?").

Uses E3's exact walk-forward, spans, clipped logistic, permutation null and decision rule on T1
(Nasdaq-100 20d drawdown <= -5%), on E3's long-span window (rows where the vix family exists).
  POSITIVE controls: a synthetic family built FROM THE FUTURE LABEL plus noise,
      x = s * (y - mean y) / sd(y) + N(0, 1) smoothed with a 5-session mean (so it is persistent
      like real factors), at strengths s in {0.05, 0.1, 0.2, 0.4}. These are leaks by design and
      exist only to measure detection: a working test must flag the stronger ones.
  NEGATIVE controls: 20 families of pure noise with the same smoothing. A working test should
      flag about none (the per-decade false-positive rate is <= 5%, and the rule needs 2 of 3).
Reported: pass/fail by the E3 rule per control, standalone and incremental improvements, and the
empirical detection threshold (the smallest strength reliably flagged).

    OMP_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python -W ignore research/regime_experiments_2026-11-24/e3c_controls.py
"""
from __future__ import annotations

import json
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e3_factor_completeness as e3  # noqa: E402
import regime_common as rc  # noqa: E402

SEED = 20261126
PERMS = int(os.environ.get("E3C_PERMS", 100))
WORKERS = int(os.environ.get("E3C_WORKERS", 2))
STRENGTHS = (0.05, 0.1, 0.2, 0.4)
N_NOISE = 20


def _job(args):
    X, y, years, test_years, perm_cols, seed = args
    warnings.simplefilter("ignore")
    Xp = X.copy()
    if seed is not None:
        idx = e3.block_permute_rows(len(X), np.random.default_rng(seed))
        Xp[:, perm_cols] = X[idx][:, perm_cols]
    return e3.walk_forward(Xp, y, years, test_years, winsor=True)


def test_family(sub: pd.DataFrame, ys, years, test_years, base_cols, x: np.ndarray, pool, rng, dec) -> dict:
    Xb = sub[base_cols].to_numpy()
    Xa = np.column_stack([Xb, x])
    jobs = [(Xb, ys, years, test_years, [], None), (Xa, ys, years, test_years, [], None)]
    jobs += [(Xa, ys, years, test_years, [Xa.shape[1] - 1], int(s)) for s in rng.integers(0, 2**31, PERMS)]
    out = list(pool.map(_job, jobs))
    pb, pa, nulls = out[0], out[1], out[2:]
    real = e3.decade_improvement(ys, pb, pa, dec)
    n = pd.DataFrame([e3.decade_improvement(ys, pb, q, dec) for q in nulls])
    have = [d for d in ("2000s", "2010s", "2020s") if d in real]
    passes = sum(real[d] > n[d].quantile(0.95) for d in have)
    return {"inc_all": real["all"], "passes": int(passes), "PASS": passes >= 2,
            **{f"inc_{d}": real[d] for d in have}, **{f"p95_{d}": float(n[d].quantile(0.95)) for d in have}}


def main() -> int:
    rc.OUT.mkdir(exist_ok=True)
    F, fam, y, _ = e3.build_features("ndx")
    cols = fam["base"] + fam["vix"]  # E3's long-span window (vix family present)
    sub, ys, years, test_years, _ = e3.span_setup(F, y, cols)
    dec = rc.decade_of(sub.index).to_numpy()
    base_cols = fam["base"]
    rng = np.random.default_rng(SEED)
    yz = np.where(np.isnan(ys), 0.0, (ys - np.nanmean(ys)) / np.nanstd(ys))
    rows = []
    with ProcessPoolExecutor(WORKERS) as pool:
        for s in STRENGTHS:
            raw = s * yz + rng.standard_normal(len(ys))
            x = pd.Series(raw).rolling(5, min_periods=1).mean().to_numpy()
            ok = ~np.isnan(ys)
            corr = float(np.corrcoef(x[ok], ys[ok])[0, 1])
            r = test_family(sub, ys, years, test_years, base_cols, x, pool, rng, dec)
            rows.append({"control": f"positive_s{s}", "corr_with_label": corr, **r})
            print(rows[-1], flush=True)
        for k in range(N_NOISE):
            x = pd.Series(rng.standard_normal(len(ys))).rolling(5, min_periods=1).mean().to_numpy()
            r = test_family(sub, ys, years, test_years, base_cols, x, pool, rng, dec)
            rows.append({"control": f"noise_{k}", "corr_with_label": np.nan, **r})
            print(rows[-1]["control"], r["PASS"], round(r["inc_all"] * 1000, 2), flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(rc.OUT / "e3c_controls.csv", index=False)
    noise = t[t.control.str.startswith("noise")]
    summary = {"positive": t[t.control.str.startswith("positive")][["control", "corr_with_label", "inc_all", "passes", "PASS"]].to_dict("records"),
               "noise_false_positives": int(noise["PASS"].sum()), "noise_n": len(noise),
               "noise_decade_pass_rate": float(noise["passes"].sum() / (3 * len(noise)))}
    (rc.OUT / "e3c_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
