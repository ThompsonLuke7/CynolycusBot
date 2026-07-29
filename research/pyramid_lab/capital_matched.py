"""Capital-matched check + blended-cost-basis sensitivity (HYPOTHESES.md).

Capital-matched: re-run the best arms with the INITIAL entry notional scaled
down by ``baseline_avg_deployed / arm_avg_deployed`` so the arm's average
deployed capital matches the baseline's, and report whether the edge survives.
This is EMPIRICAL verification of the analytic rescale used in
``aggregate.build_family`` -- sizing is linear under ``basis='entry'``
(unit-tested), so the two must agree to machine precision. If they ever
disagree, the analytic capital-matched column in the FDR table is wrong.

Blended basis: the pre-registered SECONDARY sensitivity, in which the stop /
take-profit / horizon are keyed to the blended cost of open lots instead of the
original entry. Unlike the primary grid this genuinely changes exit timing, so
it is the one place a capital-matched comparison is not arithmetically trivial.

  PYTHONPATH=. .venv/bin/python -m research.pyramid_lab.capital_matched --module momentum
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from research.pyramid_lab import arms as A
from research.pyramid_lab.engine import BasePolicy
from research.pyramid_lab.run_study import RESULTS_DIR, run_module

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("pyramid_lab.capital_matched")

# The 3 best primary arms by mean return-per-dollar-deployed delta, averaged
# across all three modules (research/pyramid_lab/results/arm_summary_primary.csv).
BEST_ARMS = ("L30_a100_m2", "L20_a100_m2", "L30_a100_m1")


def _full_row(out_dir: Path, module: str, arm: str) -> pd.Series:
    p = pd.read_csv(out_dir / f"{module}_primary_period_metrics.csv")
    return p[(p["arm"] == arm) & (p["period"] == "full")].iloc[0]


def run(module: str, *, out_dir: Path = RESULTS_DIR, best=BEST_ARMS) -> dict:
    prim = A.primary_arms()
    base_dep = float(_full_row(out_dir, module, A.BASELINE_ARM)["avg_deployed"])
    base_pnl = float(_full_row(out_dir, module, A.BASELINE_ARM)["total_pnl_net"])

    rows = []
    for arm in best:
        row = _full_row(out_dir, module, arm)
        ratio = float(row["avg_deployed"]) / base_dep
        scaled_notional = BasePolicy().target_notional / ratio
        logger.info("[%s/%s] full-span deployed ratio=%.4f -> capital-matched notional $%.2f",
                    module, arm, ratio, scaled_notional)
        res = run_module(module, out_dir=out_dir, arms={A.BASELINE_ARM: prim[A.BASELINE_ARM], arm: prim[arm]},
                         notional=scaled_notional, tag=f"capmatch_{arm}", save_positions=False)
        cm = res["period"]
        cm_arm = cm[(cm["arm"] == arm) & (cm["period"] == "full")].iloc[0]
        rows.append({
            "module": module, "arm": arm,
            "baseline_avg_deployed": round(base_dep, 2),
            "unscaled_arm_avg_deployed": round(float(row["avg_deployed"]), 2),
            "deployed_ratio": round(ratio, 4),
            "capital_matched_notional": round(scaled_notional, 2),
            "capital_matched_avg_deployed": round(float(cm_arm["avg_deployed"]), 2),
            "baseline_pnl": round(base_pnl, 2),
            "unscaled_arm_pnl": round(float(row["total_pnl_net"]), 2),
            "capital_matched_arm_pnl": round(float(cm_arm["total_pnl_net"]), 2),
            "capital_matched_delta": round(float(cm_arm["total_pnl_net"]) - base_pnl, 2),
            "analytic_rescale_pnl": round(float(row["total_pnl_net"]) / ratio, 2),
            "capital_matched_sharpe": cm_arm["sharpe_daily"],
            "baseline_sharpe": _full_row(out_dir, module, A.BASELINE_ARM)["sharpe_daily"],
            "capital_matched_maxdd": cm_arm["max_dd_dollars"],
            "baseline_maxdd": _full_row(out_dir, module, A.BASELINE_ARM)["max_dd_dollars"],
        })

    df = pd.DataFrame(rows)
    # linearity verification: the empirical rescale must equal the analytic one
    df["linearity_abs_err"] = (df["capital_matched_arm_pnl"] - df["analytic_rescale_pnl"]).abs()
    df["deployed_match_abs_err"] = (df["capital_matched_avg_deployed"] - df["baseline_avg_deployed"]).abs()
    df.to_csv(out_dir / f"{module}_capital_matched.csv", index=False)
    pd.set_option("display.width", 240)
    pd.set_option("display.max_columns", 40)
    print(df.to_string(index=False))
    return {"table": df}


def run_blended(module: str, *, out_dir: Path = RESULTS_DIR, best=BEST_ARMS) -> None:
    arms = {A.BASELINE_ARM: A.primary_arms()[A.BASELINE_ARM], **A.blended_basis_arms(list(best))}
    run_module(module, out_dir=out_dir, arms=arms, tag="blended", save_positions=False)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--module", required=True)
    ap.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    ap.add_argument("--mode", default="capital_matched", choices=["capital_matched", "blended"])
    args = ap.parse_args()
    if args.mode == "blended":
        run_blended(args.module, out_dir=args.out_dir)
    else:
        out = run(args.module, out_dir=args.out_dir)
        print(json.dumps({"max_linearity_abs_err": float(out["table"]["linearity_abs_err"].max()),
                          "max_deployed_match_abs_err": float(out["table"]["deployed_match_abs_err"].max())},
                         indent=2))


if __name__ == "__main__":
    main()
