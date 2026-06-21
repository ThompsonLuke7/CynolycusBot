from __future__ import annotations

import argparse
from functools import lru_cache
from pathlib import Path

import pandas as pd


def _snapshot_index(snapshot_dir: Path) -> pd.DataFrame:
    rows: list[dict] = []
    for path in sorted(snapshot_dir.glob("gamma_ladder_*.csv")):
        frame = pd.read_csv(path, nrows=1)
        if frame.empty or "timestamp" not in frame:
            continue
        ts = pd.to_datetime(frame["timestamp"].iloc[0], utc=True, errors="coerce")
        if pd.notna(ts):
            rows.append({"timestamp": ts, "path": str(path)})
    return pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)


@lru_cache(maxsize=512)
def _read_snapshot(path: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    for col in frame.columns:
        if col not in {"timestamp", "symbol"}:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame


def _nearest_snapshot(index: pd.DataFrame, ts: pd.Timestamp) -> Path | None:
    eligible = index[index["timestamp"] <= ts]
    if eligible.empty:
        return None
    return Path(str(eligible.iloc[-1]["path"]))


def _attach_contract_metrics(trades: pd.DataFrame, snapshot_dir: Path) -> pd.DataFrame:
    index = _snapshot_index(snapshot_dir)
    enriched: list[dict] = []
    for _, trade in trades.iterrows():
        ts = pd.Timestamp(trade["timestamp"])
        path = _nearest_snapshot(index, ts)
        row = trade.to_dict()
        row.update(
            {
                "option_oi": float("nan"),
                "option_volume": float("nan"),
                "option_delta_abs": float("nan"),
                "option_iv": float("nan"),
                "option_gamma": float("nan"),
                "spread_pct": float("nan"),
                "spread_available": False,
            }
        )
        if path is None:
            enriched.append(row)
            continue
        ladder = _read_snapshot(str(path))
        strike = float(trade["target"])
        nearest_idx = (ladder["strike"] - strike).abs().idxmin()
        side = "call" if str(trade["direction"]) == "long" else "put"
        row["option_oi"] = float(ladder.loc[nearest_idx, f"{side}_oi"])
        row["option_volume"] = float(ladder.loc[nearest_idx, f"{side}_volume"])
        row["option_delta_abs"] = abs(float(ladder.loc[nearest_idx, f"{side}_delta"]))
        row["option_iv"] = float(ladder.loc[nearest_idx, f"{side}_iv"])
        row["option_gamma"] = float(ladder.loc[nearest_idx, f"{side}_gamma"])
        enriched.append(row)
    return pd.DataFrame(enriched)


def _filtered_summary(trades: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    oi_thresholds = [0, 100, 500, 1000, 2500, 5000]
    volume_thresholds = [0, 100, 500, 1000, 2500, 5000, 10000]
    delta_ranges = [(0.0, 1.0), (0.05, 0.95), (0.10, 0.90), (0.20, 0.80), (0.30, 0.70), (0.40, 0.65)]
    for policy, group in trades.groupby("policy"):
        for oi in oi_thresholds:
            for volume in volume_thresholds:
                for dlo, dhi in delta_ranges:
                    kept = group[
                        (group["option_oi"] >= oi)
                        & (group["option_volume"] >= volume)
                        & (group["option_delta_abs"] >= dlo)
                        & (group["option_delta_abs"] <= dhi)
                    ].copy()
                    if kept.empty:
                        continue
                    wins = kept["pnl_points"] > 0
                    rows.append(
                        {
                            "policy": policy,
                            "trades": int(len(kept)),
                            "win_rate": float(wins.mean()),
                            "pnl_points": float(kept["pnl_points"].sum()),
                            "avg_points": float(kept["pnl_points"].mean()),
                            "min_oi": oi,
                            "min_volume": volume,
                            "delta_min": dlo,
                            "delta_max": dhi,
                            "kept_long": int(kept["direction"].eq("long").sum()),
                            "kept_short": int(kept["direction"].eq("short").sum()),
                        }
                    )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["pnl_points", "avg_points"], ascending=False).reset_index(drop=True)


def run(*, trades_path: Path, snapshot_dir: Path, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    trades = pd.read_csv(trades_path, parse_dates=["timestamp", "exit_time"])
    enriched = _attach_contract_metrics(trades, snapshot_dir)
    summary = _filtered_summary(enriched)
    output_dir.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(output_dir / "SPY_policy_experiment_trades_contract_metrics_2026-06-12.csv", index=False)
    summary.to_csv(output_dir / "SPY_contract_filter_experiment_summary_2026-06-12.csv", index=False)
    return enriched, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze dealer-positioning contract OI/volume/delta filters.")
    parser.add_argument("--trades", default="Data/dealer_positioning/reports/SPY_policy_experiment_trades_2026-06-12.csv")
    parser.add_argument("--snapshots", default="Data/dealer_positioning/SPY/snapshots")
    parser.add_argument("--output-dir", default="Data/dealer_positioning/reports")
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()
    enriched, summary = run(trades_path=Path(args.trades), snapshot_dir=Path(args.snapshots), output_dir=Path(args.output_dir))
    print("spread_available", bool(enriched["spread_available"].any()))
    print(summary.head(int(args.top)).to_string(index=False))


if __name__ == "__main__":
    main()
