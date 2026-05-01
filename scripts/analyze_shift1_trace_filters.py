from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
TRACE = (
    REPO_ROOT
    / "Data/models/ga_xgboost/10min_shift1/analysis/phase4_1m_bodyclose_l42_s15"
    / "best_phase4_asym_long_break_prev_stop_1m_body_and_close_short_break_prev_stop_1m_body_and_close_cooldown_cluster_longmax4_shortmax4_test_trades.csv"
)
OUT_DIR = REPO_ROOT / "Data/models/ga_xgboost/10min_shift1/analysis/trace_filter_diagnostics"


def _score(df: pd.DataFrame) -> dict[str, float]:
    long = df[df["side"] == "long"]
    short = df[df["side"] == "short"]
    return {
        "total_trades": float(len(df)),
        "total_ev_atr": float(df["outcome_atr"].mean()) if len(df) else float("nan"),
        "long_trades": float(len(long)),
        "short_trades": float(len(short)),
        "long_ev_atr": float(long["outcome_atr"].mean()) if len(long) else float("nan"),
        "short_ev_atr": float(short["outcome_atr"].mean()) if len(short) else float("nan"),
        "long_win_rate": float((long["outcome"] == "tp").mean()) if len(long) else float("nan"),
        "short_win_rate": float((short["outcome"] == "tp").mean()) if len(short) else float("nan"),
    }


def main() -> None:
    df = pd.read_csv(TRACE)
    df["entry_p_edge"] = np.where(df["side"] == "long", df["p_long"] - df["p_short"], df["p_short"] - df["p_long"])
    setup_ts = pd.to_datetime(df["setup_bar_time"], utc=True)
    entry_ts = pd.to_datetime(df["entry_time"], utc=True)
    df["entry_lag_min"] = (entry_ts - setup_ts).dt.total_seconds() / 60.0

    rows: list[dict[str, float | str]] = []
    long_p_ranges = [(0.0, 1.01), (0.0, 0.2), (0.0, 0.35), (0.42, 1.01), (0.42, 0.7)]
    short_p_ranges = [(0.0, 1.01), (0.0, 0.35), (0.1, 0.42), (0.2, 1.01), (0.35, 1.01)]
    edge_mins = [-999.0, -0.2, -0.1, 0.0, 0.05]
    max_lags = [10.0, 6.0, 4.0, 2.0]
    max_opp_values = [999.0, 0.35, 0.25, 0.15]

    for long_lo, long_hi in long_p_ranges:
        for short_lo, short_hi in short_p_ranges:
            for edge_min in edge_mins:
                for max_lag in max_lags:
                    for max_opp in max_opp_values:
                        long_mask = (
                            (df["side"] == "long")
                            & (df["p_long"] >= long_lo)
                            & (df["p_long"] < long_hi)
                            & (df["p_short"] <= max_opp)
                            & (df["entry_lag_min"] <= max_lag)
                        )
                        short_mask = (
                            (df["side"] == "short")
                            & (df["p_short"] >= short_lo)
                            & (df["p_short"] < short_hi)
                            & (df["p_long"] <= max_opp)
                            & (df["entry_lag_min"] <= max_lag)
                        )
                        mask = (long_mask | short_mask) & (df["entry_p_edge"] >= edge_min)
                        x = df[mask]
                        if len(x) < 200:
                            continue
                        row = {
                            "long_p_range": f"[{long_lo},{long_hi})",
                            "short_p_range": f"[{short_lo},{short_hi})",
                            "edge_min": float(edge_min),
                            "max_lag_min": float(max_lag),
                            "max_opp": float(max_opp),
                        }
                        row.update(_score(x))
                        rows.append(row)

    out = pd.DataFrame(rows).sort_values("total_ev_atr", ascending=False)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "shift1_trace_filter_sweep.csv"
    json_path = OUT_DIR / "shift1_trace_filter_summary.json"
    out.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps({"rows": len(out), "best": out.head(30).to_dict(orient="records")}, indent=2))
    print(f"[trace-filter] wrote {csv_path}")
    print(f"[trace-filter] wrote {json_path}")
    print(out.head(25).to_string(index=False))


if __name__ == "__main__":
    main()
