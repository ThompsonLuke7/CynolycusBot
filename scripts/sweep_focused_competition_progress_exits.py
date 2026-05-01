from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from sweep_progress_exit_on_traces import _load_execution, _outcome_for_trade, _metrics


OUT_DIR = Path("Data/models/ga_xgboost/model_competition_phase4_focused/progress_exit_sweep")
FOCUSED_ROOT = Path("Data/models/ga_xgboost/model_competition_phase4_focused")
OLD_BASELINE_TRACE = Path(
    "Data/models/ga_xgboost/10min/analysis/"
    "phase4_1m_oof_best_bodyclose_bodyclose_l42_s15_full_1m_train/"
    "best_phase4_asym_long_break_prev_stop_1m_body_and_close_short_break_prev_stop_1m_body_and_close_cooldown_cluster_longmax4_shortmax4_test_trades.csv"
)


def _parse_run_params(name: str) -> tuple[int, float, float]:
    match = re.search(r"_h(?P<h>\d+)_tp(?P<tp>\d+\.\d+)_sl(?P<sl>\d+\.\d+)$", name)
    if not match:
        return 12, 1.0, 0.8
    return int(match.group("h")), float(match.group("tp")), float(match.group("sl"))


def _trace_paths() -> dict[str, Path]:
    paths = {}
    if OLD_BASELINE_TRACE.exists():
        paths["old_saved_nonshift_baseline"] = OLD_BASELINE_TRACE
    for path in sorted(FOCUSED_ROOT.glob("*/best_phase4_*_trades.csv")):
        paths[path.parent.name] = path
    return paths


def _evaluate_trace(name: str, trace_path: Path, execution: pd.DataFrame) -> pd.DataFrame:
    horizon_bars, tp_atr, sl_atr = _parse_run_params(name)
    trace = pd.read_csv(trace_path)
    trace["entry_time"] = pd.to_datetime(trace["entry_time"], utc=True, errors="coerce").dt.tz_convert("America/New_York")
    trace["side"] = trace["side"].astype(str).str.lower()

    policies: list[tuple[str, int | None, float, str]] = [("current_exit", None, 0.0, "continuous")]
    for arm in (4, 6, 8, 10, 12):
        for floor in (-0.30, -0.20, -0.10, 0.0, 0.10, 0.20, 0.30):
            policies.append((f"progress_arm{arm}_floor{floor:+.2f}", arm, floor, "continuous"))

    rows = []
    for policy_name, arm_bars, floor_atr, check in policies:
        replay_rows = [
            _outcome_for_trade(
                trade,
                execution,
                horizon_bars=horizon_bars,
                tp_atr=tp_atr,
                sl_atr=sl_atr,
                progress_arm_bars=arm_bars,
                progress_floor_atr=floor_atr,
                progress_check=check,
            )
            for _, trade in trace.iterrows()
        ]
        replay = trace.copy()
        replay["new_outcome_atr"] = [r["outcome_atr"] for r in replay_rows]
        replay["new_outcome"] = [r["outcome"] for r in replay_rows]
        replay["new_exit_time"] = [r["exit_time"] for r in replay_rows]
        replay = replay.dropna(subset=["new_outcome_atr"])
        rows.append(
            {
                "trace_name": name,
                "trace_path": str(trace_path),
                "base_horizon_bars": horizon_bars,
                "base_tp_atr": tp_atr,
                "base_sl_atr": sl_atr,
                "policy": policy_name,
                "progress_arm_bars": arm_bars if arm_bars is not None else np.nan,
                "progress_floor_atr": floor_atr,
                **_metrics(replay),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    execution = _load_execution()
    traces = _trace_paths()
    if not traces:
        raise SystemExit("no traces found")
    summary = pd.concat([_evaluate_trace(name, path, execution) for name, path in traces.items()], ignore_index=True)
    summary = summary.sort_values(["ev_atr", "trades"], ascending=[False, False])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT_DIR / "focused_progress_exit_sweep_summary.csv", index=False)

    cols = [
        "trace_name",
        "policy",
        "trades",
        "ev_atr",
        "win_rate",
        "long_ev_atr",
        "short_ev_atr",
        "tp",
        "sl",
        "progress_exit",
        "timeout",
    ]
    print(summary.head(30)[cols].to_string(index=False))
    print(f"\nwrote {OUT_DIR / 'focused_progress_exit_sweep_summary.csv'}")


if __name__ == "__main__":
    main()
