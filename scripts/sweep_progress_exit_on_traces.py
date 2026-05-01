from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


EXECUTION_1M = Path("Data/raw/spy/spy_intraday_1min_runtime_rth_cache.parquet")
OUT_DIR = Path("Data/models/ga_xgboost/10min_shift1/analysis/progress_exit_sweep")

TRACE_PATHS = {
    "old_nonshift_baseline": Path(
        "Data/models/ga_xgboost/10min/analysis/"
        "phase4_1m_oof_best_bodyclose_bodyclose_l42_s15_full_1m_train/"
        "best_phase4_asym_long_break_prev_stop_1m_body_and_close_short_break_prev_stop_1m_body_and_close_cooldown_cluster_longmax4_shortmax4_test_trades.csv"
    ),
    "shift1_lag2_l42_s20": Path(
        "Data/models/ga_xgboost/10min_shift1/analysis/phase4_lag2_l42_s20/"
        "best_phase4_asym_long_break_prev_stop_1m_body_and_close_short_break_prev_stop_1m_body_and_close_cooldown_cluster_longmax4_shortmax4_test_trades.csv"
    ),
}


def _load_execution() -> pd.DataFrame:
    df = pd.read_parquet(EXECUTION_1M)
    ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce").dt.tz_convert("America/New_York")
    out = df.copy()
    out.index = pd.DatetimeIndex(ts)
    out = out.loc[out.index.notna()].sort_index()
    for col in ("open", "high", "low", "close"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _outcome_for_trade(
    row: pd.Series,
    execution: pd.DataFrame,
    *,
    horizon_bars: int,
    tp_atr: float,
    sl_atr: float,
    progress_arm_bars: int | None,
    progress_floor_atr: float,
    progress_check: str,
) -> dict[str, float | str]:
    side = str(row["side"]).lower()
    entry = float(row["entry_price"])
    atr = float(row["atr"])
    entry_time = pd.Timestamp(row["entry_time"])
    if not np.isfinite(entry) or not np.isfinite(atr) or atr <= 0:
        return {"outcome_atr": np.nan, "outcome": "invalid", "exit_time": pd.NaT}

    end_time = entry_time + pd.Timedelta(minutes=10 * int(horizon_bars))
    left = execution.index.searchsorted(entry_time, side="right")
    right = execution.index.searchsorted(end_time, side="right")
    path = execution.iloc[left:right]
    if path.empty:
        return {"outcome_atr": np.nan, "outcome": "empty", "exit_time": pd.NaT}

    if side == "long":
        tp_px = entry + float(tp_atr) * atr
        sl_px = entry - float(sl_atr) * atr
        floor_px = entry + float(progress_floor_atr) * atr
    else:
        tp_px = entry - float(tp_atr) * atr
        sl_px = entry + float(sl_atr) * atr
        floor_px = entry - float(progress_floor_atr) * atr

    arm_time = None
    if progress_arm_bars is not None:
        arm_time = entry_time + pd.Timedelta(minutes=10 * int(progress_arm_bars))

    last_close = np.nan
    last_time = path.index[-1]
    for ts, bar in path.iterrows():
        hi = float(bar["high"])
        lo = float(bar["low"])
        close = float(bar["close"])
        last_close = close

        if side == "long":
            if lo <= sl_px:
                return {"outcome_atr": -float(sl_atr), "outcome": "sl", "exit_time": ts}
            if hi >= tp_px:
                return {"outcome_atr": float(tp_atr), "outcome": "tp", "exit_time": ts}
            progress_hit = close <= floor_px
            progress_ret = (close - entry) / atr
        else:
            if hi >= sl_px:
                return {"outcome_atr": -float(sl_atr), "outcome": "sl", "exit_time": ts}
            if lo <= tp_px:
                return {"outcome_atr": float(tp_atr), "outcome": "tp", "exit_time": ts}
            progress_hit = close >= floor_px
            progress_ret = (entry - close) / atr

        if arm_time is not None and ts >= arm_time and progress_hit:
            if progress_check == "once" and ts > arm_time + pd.Timedelta(minutes=1):
                continue
            return {"outcome_atr": float(progress_ret), "outcome": "progress_exit", "exit_time": ts}

    if side == "long":
        timeout_ret = (float(last_close) - entry) / atr
    else:
        timeout_ret = (entry - float(last_close)) / atr
    return {"outcome_atr": float(timeout_ret), "outcome": "timeout", "exit_time": last_time}


def _metrics(df: pd.DataFrame) -> dict[str, float]:
    out = pd.to_numeric(df["new_outcome_atr"], errors="coerce")
    long = out[df["side"].eq("long")]
    short = out[df["side"].eq("short")]
    return {
        "trades": float(len(df)),
        "ev_atr": float(out.mean()),
        "sum_atr": float(out.sum()),
        "win_rate": float((out > 0).mean()),
        "long_trades": float(len(long)),
        "long_ev_atr": float(long.mean()) if len(long) else np.nan,
        "short_trades": float(len(short)),
        "short_ev_atr": float(short.mean()) if len(short) else np.nan,
        "tp": float(df["new_outcome"].eq("tp").sum()),
        "sl": float(df["new_outcome"].eq("sl").sum()),
        "progress_exit": float(df["new_outcome"].eq("progress_exit").sum()),
        "timeout": float(df["new_outcome"].eq("timeout").sum()),
    }


def _evaluate_trace(
    name: str,
    trace_path: Path,
    execution: pd.DataFrame,
    *,
    horizon_bars: int = 12,
    tp_atr: float = 1.0,
    sl_atr: float = 0.8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    trace = pd.read_csv(trace_path)
    trace["entry_time"] = pd.to_datetime(trace["entry_time"], utc=True, errors="coerce").dt.tz_convert("America/New_York")
    trace["side"] = trace["side"].astype(str).str.lower()

    rows = []
    traces = []
    policies: list[tuple[str, int | None, float, str]] = [("current_exit", None, 0.0, "continuous")]
    for arm in (4, 6, 8, 10):
        for floor in (-0.20, -0.10, 0.0, 0.10, 0.20):
            policies.append((f"progress_arm{arm}_floor{floor:+.2f}", arm, floor, "continuous"))

    for policy_name, arm_bars, floor_atr, check in policies:
        replay_rows = []
        for _, trade in trace.iterrows():
            outcome = _outcome_for_trade(
                trade,
                execution,
                horizon_bars=horizon_bars,
                tp_atr=tp_atr,
                sl_atr=sl_atr,
                progress_arm_bars=arm_bars,
                progress_floor_atr=floor_atr,
                progress_check=check,
            )
            replay_rows.append(outcome)
        replay = trace.copy()
        replay["new_outcome_atr"] = [r["outcome_atr"] for r in replay_rows]
        replay["new_outcome"] = [r["outcome"] for r in replay_rows]
        replay["new_exit_time"] = [r["exit_time"] for r in replay_rows]
        replay = replay.dropna(subset=["new_outcome_atr"])
        row = {
            "trace_name": name,
            "policy": policy_name,
            "progress_arm_bars": arm_bars if arm_bars is not None else np.nan,
            "progress_floor_atr": floor_atr,
            **_metrics(replay),
        }
        rows.append(row)
        replay["trace_name"] = name
        replay["policy"] = policy_name
        traces.append(replay)

    return pd.DataFrame(rows), pd.concat(traces, ignore_index=True)


def main() -> None:
    execution = _load_execution()
    all_rows = []
    all_traces = []
    for name, path in TRACE_PATHS.items():
        rows, traces = _evaluate_trace(name, path, execution)
        all_rows.append(rows)
        all_traces.append(traces)

    summary = pd.concat(all_rows, ignore_index=True)
    summary = summary.sort_values(["trace_name", "ev_atr"], ascending=[True, False])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT_DIR / "progress_exit_sweep_summary.csv", index=False)
    pd.concat(all_traces, ignore_index=True).to_csv(OUT_DIR / "progress_exit_sweep_traces.csv", index=False)

    for name in TRACE_PATHS:
        print(f"\n{name}")
        cols = [
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
        print(summary[summary["trace_name"].eq(name)].head(12)[cols].to_string(index=False))
    print(f"\nwrote {OUT_DIR / 'progress_exit_sweep_summary.csv'}")


if __name__ == "__main__":
    main()
