"""Combine per-module pyramiding results, apply BH-FDR across the WHOLE
family, and print the capital-control table. See HYPOTHESES.md.

  PYTHONPATH=. .venv/bin/python -m research.pyramid_lab.aggregate
  PYTHONPATH=. .venv/bin/python -m research.pyramid_lab.aggregate --tags primary,blended
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from strategies.momentum_expansion.ablation.bootstrap import bh_fdr

from research.pyramid_lab import arms as A

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def load(out_dir: Path, tags: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    per, stat = [], []
    for tag in tags:
        per += [pd.read_csv(p) for p in sorted(out_dir.glob(f"*_{tag}_period_metrics.csv"))]
        stat += [pd.read_csv(p) for p in sorted(out_dir.glob(f"*_{tag}_weekly_diff.csv"))]
    if not per:
        raise SystemExit(f"no results for tags={tags} under {out_dir} -- run run_study.py first")
    return pd.concat(per, ignore_index=True), pd.concat(stat, ignore_index=True)


def build_family(stat: pd.DataFrame) -> pd.DataFrame:
    """The pre-registered FDR family: every (arm x module x walk-forward
    period) cell in which the module's stream is actually live.

    Excluded, and reported as excluded:
      * ``period == "full"`` — a pooled read over all periods, not a
        walk-forward period; it is reported separately and never used to claim
        an effect.
      * cells with zero baseline trades — the module's OOF stream does not
        cover that period at all, so there is no test there (meta starts
        2024-09-16). Including them would pad the family with guaranteed-null
        p=1.0 rows and make BH-FDR artificially permissive.
    """
    fam = stat[(stat["period"] != "full") & (stat["baseline_trades"] > 0)].copy()
    fam["q_fdr"] = bh_fdr(fam["p_value"])
    fam["survives_fdr"] = fam["q_fdr"] <= 0.10
    # Same cells, capital-matched statistic (see run_study.py). Same family
    # size, so this is not an extra multiple-comparison burden -- it is the
    # honest version of the same test.
    fam["cm_q_fdr"] = bh_fdr(fam["cm_p_value"])
    fam["cm_survives_fdr"] = fam["cm_q_fdr"] <= 0.10
    fam["pnl_delta"] = fam["arm_pnl"] - fam["baseline_pnl"]
    fam["rpd_delta"] = fam["arm_rpd"] - fam["baseline_rpd"]
    fam["sharpe_delta"] = fam["arm_sharpe"] - fam["baseline_sharpe"]
    fam["deployed_ratio"] = fam["arm_avg_deployed"] / fam["baseline_avg_deployed"]
    # Capital-matched P&L: rescale the arm so its average deployed capital
    # equals the baseline's. Under basis='entry' every lot scales linearly
    # (asserted in tests/test_engine.py::test_capital_linearity_under_entry_basis),
    # so this is exact, not an approximation.
    fam["arm_pnl_capital_matched"] = fam["arm_pnl"] / fam["deployed_ratio"]
    fam["capital_matched_delta"] = fam["arm_pnl_capital_matched"] - fam["baseline_pnl"]
    return fam


def arm_summary(fam: pd.DataFrame) -> pd.DataFrame:
    """Per (module, arm) cross-period consistency — the antidote to 'wins in 2
    of 7 periods with flipping signs'."""
    rows = []
    for (mod, arm), g in fam.groupby(["module", "arm"]):
        rows.append({
            "module": mod, "arm": arm, "periods": len(g),
            "periods_pnl_up": int((g["pnl_delta"] > 0).sum()),
            "periods_rpd_up": int((g["rpd_delta"] > 0).sum()),
            "periods_sharpe_up": int((g["sharpe_delta"] > 0).sum()),
            "periods_capital_matched_up": int((g["capital_matched_delta"] > 0).sum()),
            "periods_survive_fdr": int(g["survives_fdr"].sum()),
            "periods_survive_fdr_capital_matched": int(g["cm_survives_fdr"].sum()),
            "mean_deployed_ratio": round(float(g["deployed_ratio"].mean()), 3),
            "sum_pnl_delta": round(float(g["pnl_delta"].sum()), 0),
            "sum_capital_matched_delta": round(float(g["capital_matched_delta"].sum()), 0),
            "mean_rpd_delta": round(float(g["rpd_delta"].mean()), 4),
            "mean_sharpe_delta": round(float(g["sharpe_delta"].mean()), 4),
        })
    return pd.DataFrame(rows).sort_values(["module", "mean_rpd_delta"], ascending=[True, False])


def run(out_dir: Path = RESULTS_DIR, tags: tuple[str, ...] = ("primary",)) -> dict:
    period, stat = load(out_dir, list(tags))
    fam = build_family(stat)
    summ = arm_summary(fam)

    fam.to_csv(out_dir / f"fdr_family_{'_'.join(tags)}.csv", index=False)
    summ.to_csv(out_dir / f"arm_summary_{'_'.join(tags)}.csv", index=False)

    n_excluded_full = int((stat["period"] == "full").sum())
    n_excluded_dead = int(((stat["period"] != "full") & (stat["baseline_trades"] <= 0)).sum())
    out = {
        "tags": list(tags),
        "arms_evaluated": sorted(stat["arm"].unique().tolist()),
        "modules": sorted(stat["module"].unique().tolist()),
        "combinations_tried_total_rows": int(len(stat)),
        "excluded_pooled_full_rows": n_excluded_full,
        "excluded_dead_period_rows": n_excluded_dead,
        "fdr_family_size": int(len(fam)),
        "n_survive_fdr_q10": int(fam["survives_fdr"].sum()),
        "n_survive_fdr_q10_capital_matched": int(fam["cm_survives_fdr"].sum()),
        "n_nominal_p_lt_010": int((fam["p_value"] < 0.10).sum()),
        "n_nominal_p_lt_010_capital_matched": int((fam["cm_p_value"] < 0.10).sum()),
        "n_cells_pnl_up": int((fam["pnl_delta"] > 0).sum()),
        "n_cells_rpd_up": int((fam["rpd_delta"] > 0).sum()),
        "n_cells_capital_matched_up": int((fam["capital_matched_delta"] > 0).sum()),
        "n_cells_sharpe_up": int((fam["sharpe_delta"] > 0).sum()),
        "median_deployed_ratio": round(float(fam["deployed_ratio"].median()), 3),
    }
    (out_dir / f"summary_{'_'.join(tags)}.json").write_text(json.dumps(out, indent=2))

    pd.set_option("display.width", 240)
    pd.set_option("display.max_columns", 60)
    print("\n=== FDR family ===")
    print(json.dumps(out, indent=2))
    print("\n=== per (module, arm) cross-period consistency ===")
    print(summ.to_string(index=False))
    print("\n=== cells surviving BH-FDR q<=0.10 ===")
    surv = fam[fam["survives_fdr"]]
    cols = ["module", "arm", "period", "point", "ci_lo", "ci_hi", "p_value", "q_fdr",
            "baseline_pnl", "arm_pnl", "deployed_ratio", "baseline_rpd", "arm_rpd",
            "capital_matched_delta"]
    print(surv[cols].to_string(index=False) if len(surv) else "(none)")
    return {"family": fam, "summary": summ, "meta": out}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    ap.add_argument("--tags", default="primary")
    args = ap.parse_args()
    run(args.out_dir, tuple(t for t in args.tags.split(",") if t))


if __name__ == "__main__":
    main()
