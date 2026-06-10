from __future__ import annotations

import argparse
import itertools
from copy import deepcopy
from pathlib import Path

import pandas as pd

from itransformer_train import build_arg_parser, run_training


def _parse_float_list(value: str) -> list[float]:
    if not value:
        return []
    out: list[float] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(float(part))
    return out


def main() -> None:
    base = build_arg_parser()
    parser = argparse.ArgumentParser(parents=[base], add_help=True, conflict_handler="resolve")
    parser.set_defaults(label_mode="continuation", monitor_metric="top_decile_mae")
    parser.add_argument(
        "--alpha_grid",
        type=str,
        default="1,2,3",
        help="comma-separated cont_weight_alpha values",
    )
    parser.add_argument(
        "--power_grid",
        type=str,
        default="1,2,3",
        help="comma-separated cont_weight_power values",
    )
    parser.add_argument(
        "--beta_grid",
        type=str,
        default="1.0,0.5,0.2",
        help="comma-separated huber_beta values",
    )
    parser.add_argument(
        "--out_csv",
        type=str,
        default=None,
        help="optional path to save grid results as CSV",
    )
    args = parser.parse_args()

    alpha_grid = _parse_float_list(args.alpha_grid)
    power_grid = _parse_float_list(args.power_grid)
    beta_grid = _parse_float_list(args.beta_grid)
    if not alpha_grid or not power_grid or not beta_grid:
        raise ValueError("alpha_grid, power_grid, and beta_grid must be non-empty.")

    rows: list[dict] = []
    combos = list(itertools.product(alpha_grid, power_grid, beta_grid))
    print(f"[grid] combos={len(combos)}")

    for i, (alpha, power, beta) in enumerate(combos, start=1):
        run_args = deepcopy(args)
        run_args.cont_weight_alpha = alpha
        run_args.cont_weight_power = power
        run_args.huber_beta = beta
        run_args.monitor_metric = "wmae"

        print(f"\n[grid] {i}/{len(combos)} alpha={alpha} power={power} beta={beta}")
        results = run_training(run_args, return_predictions=False)
        if not results:
            print("[grid] no results returned")
            continue

        for side, res in results.items():
            val_metrics = res.get("val_metrics", {})
            test_metrics = res.get("test_metrics", {})
            train_metrics = res.get("train_metrics", {})
            row = {
                "side": side,
                "alpha": alpha,
                "power": power,
                "beta": beta,
                "train_loss": train_metrics.get("loss"),
                "train_wmae": train_metrics.get("wmae"),
                "train_top_decile_mae": train_metrics.get("top_decile_mae"),
                "train_peak_f1": train_metrics.get("peak_f1"),
                "train_neg_peak_f1": train_metrics.get("neg_peak_f1"),
                "val_loss": val_metrics.get("loss"),
                "val_wmae": val_metrics.get("wmae"),
                "val_top_decile_mae": val_metrics.get("top_decile_mae"),
                "val_peak_f1": val_metrics.get("peak_f1"),
                "val_neg_peak_f1": val_metrics.get("neg_peak_f1"),
                "test_loss": test_metrics.get("loss"),
                "test_wmae": test_metrics.get("wmae"),
                "test_top_decile_mae": test_metrics.get("top_decile_mae"),
                "test_peak_f1": test_metrics.get("peak_f1"),
                "test_neg_peak_f1": test_metrics.get("neg_peak_f1"),
            }
            rows.append(row)

    if not rows:
        print("[grid] no rows to report")
        return

    df = pd.DataFrame(rows)
    sort_metric = f"val_{args.monitor_metric}"
    sort_ascending = True
    if args.monitor_metric.endswith("f1") and not args.monitor_metric.startswith("neg_"):
        sort_ascending = False
    df = df.sort_values(["side", sort_metric], ascending=[True, sort_ascending])
    print(f"\n[grid] top results by side ({sort_metric})")
    for side in df["side"].unique():
        top = df[df["side"] == side].head(5)
        print(f"\n{side.upper()}")
        print(top.to_string(index=False))

    if args.out_csv:
        out_path = Path(args.out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"[grid] wrote {out_path}")


if __name__ == "__main__":
    main()
