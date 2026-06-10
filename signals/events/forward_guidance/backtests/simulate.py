"""Event-level post-earnings strategy backtest."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from signals.events.forward_guidance.config import BACKTEST_CONFIG, BACKTEST_RESULTS_DIR, PRIMARY_PROBABILITY
from signals.events.forward_guidance.utils.io import write_dataframe, write_json

logger = logging.getLogger(__name__)


def _trade_return(row: pd.Series, hold_days: int, atr_trail_mult: float | None) -> tuple[float, str]:
    ret_col = f"fwd_ret_{hold_days}d"
    ret = float(row.get(ret_col, np.nan))
    if not np.isfinite(ret):
        return float("nan"), "missing_forward_return"
    if atr_trail_mult is not None:
        atr_pct = float(row.get("atr_pct_14", np.nan))
        drawdown = float(row.get("max_drawdown", np.nan))
        if np.isfinite(atr_pct) and np.isfinite(drawdown):
            stop_ret = -abs(atr_trail_mult * atr_pct)
            if drawdown <= stop_ret:
                return stop_ret, "atr_trail"
    return ret, "time"


def select_trades(predictions: pd.DataFrame, cfg: dict[str, Any] | None = None) -> pd.DataFrame:
    config = dict(BACKTEST_CONFIG)
    if cfg:
        config.update(cfg)
    df = predictions.copy()
    if PRIMARY_PROBABILITY not in df.columns:
        raise ValueError(f"Predictions must include {PRIMARY_PROBABILITY}.")
    mask = df[PRIMARY_PROBABILITY] >= float(config["prob_threshold"])
    if config.get("require_bad_reaction", True) and "bad_initial_reaction_flag" in df.columns:
        mask &= df["bad_initial_reaction_flag"].fillna(0).astype(float) > 0
    if "guidance_strength_score" in df.columns:
        mask &= df["guidance_strength_score"].fillna(0).astype(float) >= float(config["min_guidance_strength"])
    if config.get("require_stabilization", True) and "technical_stabilization_flag" in df.columns:
        mask &= df["technical_stabilization_flag"].fillna(0).astype(float) > 0
    trades = df.loc[mask].copy()
    hold_days = int(config["hold_days"])
    atr_trail_mult = config.get("atr_trail_mult")
    returns = trades.apply(lambda row: _trade_return(row, hold_days, atr_trail_mult), axis=1)
    if len(returns):
        trades["trade_return"] = [x[0] for x in returns]
        trades["exit_reason"] = [x[1] for x in returns]
    else:
        trades["trade_return"] = []
        trades["exit_reason"] = []
    trades["hold_days"] = hold_days
    trades["position_size_pct"] = float(config["position_size_pct"])
    trades = trades.loc[trades["trade_return"].notna()].sort_values("signal_timestamp" if "signal_timestamp" in trades else "earnings_date")
    return trades.reset_index(drop=True)


def summarize_trades(trades: pd.DataFrame, *, initial_equity: float = 100_000.0) -> dict[str, Any]:
    if trades.empty:
        return {
            "trades": 0,
            "cagr": None,
            "max_drawdown": None,
            "expectancy": None,
            "hit_rate": None,
            "sector_exposure": {},
        }
    equity = float(initial_equity)
    peak = equity
    max_dd = 0.0
    equity_values = []
    for _, row in trades.iterrows():
        pnl = equity * float(row.get("position_size_pct", 0.05)) * float(row["trade_return"])
        equity += pnl
        peak = max(peak, equity)
        max_dd = min(max_dd, equity / peak - 1.0)
        equity_values.append(equity)
    returns = trades["trade_return"].astype(float)
    start = pd.to_datetime(trades.iloc[0].get("signal_timestamp") or trades.iloc[0].get("earnings_date"), utc=True, errors="coerce")
    end = pd.to_datetime(trades.iloc[-1].get("signal_timestamp") or trades.iloc[-1].get("earnings_date"), utc=True, errors="coerce")
    years = max((end - start).days / 365.25, 1 / 365.25) if pd.notna(start) and pd.notna(end) else 1.0
    cagr = (equity / initial_equity) ** (1 / years) - 1 if equity > 0 else -1.0
    sector_exposure = {}
    if "sector_etf" in trades.columns:
        sector_exposure = trades["sector_etf"].fillna("UNKNOWN").value_counts(normalize=True).to_dict()
    return {
        "trades": int(len(trades)),
        "ending_equity": float(equity),
        "cagr": float(cagr),
        "max_drawdown": float(max_dd),
        "expectancy": float(returns.mean()),
        "hit_rate": float((returns > 0).mean()),
        "avg_win": float(returns.loc[returns > 0].mean()) if (returns > 0).any() else None,
        "avg_loss": float(returns.loc[returns <= 0].mean()) if (returns <= 0).any() else None,
        "sector_exposure": sector_exposure,
    }


def run_backtest(
    predictions: pd.DataFrame,
    *,
    cfg: dict[str, Any] | None = None,
    output_dir: Path | str = BACKTEST_RESULTS_DIR,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    trades = select_trades(predictions, cfg)
    summary = summarize_trades(trades)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_dataframe(trades, out_dir / "forward_guidance_trades.parquet")
    write_json(summary, out_dir / "forward_guidance_summary.json")
    return trades, summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Backtest forward-guidance ranked opportunities.")
    parser.add_argument("--predictions", required=True, help="Parquet/CSV file with prediction probabilities and labels.")
    parser.add_argument("--threshold", type=float, default=float(BACKTEST_CONFIG["prob_threshold"]))
    parser.add_argument("--hold-days", type=int, default=int(BACKTEST_CONFIG["hold_days"]))
    parser.add_argument("--log", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log.upper()), format="%(asctime)s %(levelname)s %(message)s")
    path = Path(args.predictions)
    df = pd.read_csv(path) if path.suffix.lower() == ".csv" else pd.read_parquet(path)
    trades, summary = run_backtest(df, cfg={"prob_threshold": args.threshold, "hold_days": args.hold_days})
    print(summary)
    logger.info("Wrote %d trades to %s", len(trades), BACKTEST_RESULTS_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
