"""Analytics/report system for replay trades."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from strategies.momentum_scalper.configs.settings import PLOTS_OUTPUT_DIR, ensure_data_dirs


def build_report(trades: pd.DataFrame, output_dir: Path = PLOTS_OUTPUT_DIR) -> dict[str, pd.DataFrame]:
    ensure_data_dirs()
    output_dir.mkdir(parents=True, exist_ok=True)
    if trades.empty:
        summary = pd.DataFrame([{"trades": 0}])
    else:
        ordered = trades.sort_values("entry_timestamp").copy()
        ordered["equity_curve"] = ordered["pnl_pct"].cumsum()
        summary = pd.DataFrame(
            [
                {
                    "trades": len(ordered),
                    "win_rate": float((ordered["pnl_pct"] > 0).mean()),
                    "expectancy_pct": float(ordered["pnl_pct"].mean()),
                    "right_tail_capture": float(ordered.get("right_tail_capture", pd.Series(dtype=float)).mean()),
                    "average_giveback": float(ordered.get("average_giveback", pd.Series(dtype=float)).mean()),
                }
            ]
        )
        ordered.to_csv(output_dir / "equity_curve.csv", index=False)
    reports = {
        "summary": summary,
        "setup_win_rates": trades.groupby("pattern")["pnl_pct"].agg(["count", "mean"]) if not trades.empty and "pattern" in trades else pd.DataFrame(),
        "MFE_MAE": trades[["MFE", "MAE"]] if not trades.empty and {"MFE", "MAE"}.issubset(trades.columns) else pd.DataFrame(),
        "best_setups": trades.nlargest(20, "pnl_pct") if not trades.empty else pd.DataFrame(),
        "worst_setups": trades.nsmallest(20, "pnl_pct") if not trades.empty else pd.DataFrame(),
        "scanner_rank_vs_outcome": trades.groupby("rank")["pnl_pct"].mean().reset_index() if not trades.empty and "rank" in trades else pd.DataFrame(),
    }
    for name, frame in reports.items():
        frame.to_csv(output_dir / f"{name}.csv", index=True)
    return reports


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate scalper replay report")
    parser.add_argument("trades", type=Path)
    parser.add_argument("--output-dir", type=Path, default=PLOTS_OUTPUT_DIR)
    args = parser.parse_args()
    reports = build_report(pd.read_parquet(args.trades), args.output_dir)
    print(reports["summary"].to_string(index=False))


if __name__ == "__main__":
    main()
