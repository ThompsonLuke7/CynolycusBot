from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Policy.regime_filter import StickyRegimeConfig, add_sticky_trend_regime
from scripts.sweep_live_thresholds_post_0401 import (
    _load_decisions_from_signal_frame,
    _load_one_min,
    _metrics,
    _run_one,
    _to_et,
)


DEFAULT_SIGNAL_FRAME = Path(
    "Data/models/ga_xgboost/10min_shift1/analysis/phase4_1m_bodyclose_l42_s15/phase4_signal_frame.parquet"
)
DEFAULT_ONE_MIN = Path("Data/raw/spy/spy_intraday_1min_runtime_rth_cache.parquet")
DEFAULT_RUN_ROOT = Path("Data/inference/live_runs")
DEFAULT_MAY4_DECISIONS = DEFAULT_RUN_ROOT / "20260504_074102_live_spy" / "decision-10m.jsonl"
DEFAULT_SYMBOL = "SPY"


SIM_KW: dict[str, Any] = {
    "exit_opp_long_thr": 0.40,
    "exit_opp_short_thr": 0.75,
    "setup_max_bars": 3,
    "cutoff_hhmm": "13:00",
    "new_entry_cutoff_hhmm": "15:00",
    "entry_quote_mode": "mid",
    "exit_quote_mode": "bid",
    "quote_spread_bps": 0.0,
    "stop_loss_pct": 1.0,
    "no_progress_minutes": 0,
    "no_progress_mfe_pct": 0.0,
    "trail_arm_pct": 1.0,
    "trail_giveback_pct": 0.20,
    "time_decay_minutes": 60,
    "time_decay_progress_pct": 0.5,
    "scalp_enabled": False,
    "scalp_long_thr": 0.30,
    "scalp_short_thr": 0.55,
    "scalp_setup_max_bars": 1,
    "scalp_min_signal_range_atr": 0.35,
    "scalp_require_reversal_close": True,
    "candidate_enabled": False,
    "candidate_long_thr": 0.30,
    "candidate_short_thr": 0.55,
    "candidate_opposite_max": 0.15,
    "candidate_setup_max_bars": 1,
    "candidate_min_signal_range_atr": 0.35,
    "candidate_long_enabled": True,
    "candidate_short_enabled": True,
    "candidate_start_hhmm": "09:30",
    "candidate_end_hhmm": "16:00",
}


def _parse_floats(text: str) -> list[float]:
    return [float(x.strip()) for x in str(text).split(",") if x.strip()]


def _parse_ints(text: str) -> list[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def _as_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out


def _et_index(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        ts_col = next((c for c in ("timestamp", "date", "datetime", "index") if c in out.columns), None)
        if ts_col is None:
            raise ValueError("Frame needs a DatetimeIndex or timestamp column.")
        out[ts_col] = pd.to_datetime(out[ts_col], utc=True, errors="coerce")
        out = out.dropna(subset=[ts_col]).set_index(ts_col)
    idx = pd.to_datetime(out.index, utc=True, errors="coerce")
    out = out.loc[pd.notna(idx)].copy()
    out.index = pd.DatetimeIndex(idx[pd.notna(idx)]).tz_convert("America/New_York")
    return out.sort_index()


def _load_regimes(signal_frame: Path) -> pd.DataFrame:
    frame = _et_index(pd.read_parquet(signal_frame))
    for col in ("open", "high", "low", "close", "ema_fast", "ema_slow"):
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = add_sticky_trend_regime(frame, config=StickyRegimeConfig())
    return frame[["trend_regime"]].reset_index().rename(columns={frame.index.name or "index": "timestamp"})


def _with_regime(decisions: pd.DataFrame, signal_frame: Path) -> pd.DataFrame:
    regimes = _load_regimes(signal_frame)
    out = decisions.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce").dt.tz_convert("America/New_York")
    regimes["timestamp"] = pd.to_datetime(regimes["timestamp"], utc=True, errors="coerce").dt.tz_convert("America/New_York")
    out = out.merge(regimes, on="timestamp", how="left")
    out["trend_regime"] = out["trend_regime"].fillna("neutral")
    return out


def _rank_percentile(values: pd.Series, history: pd.Series) -> pd.Series:
    hist = pd.to_numeric(history, errors="coerce").dropna().to_numpy(dtype=float)
    hist = np.sort(hist[np.isfinite(hist)])
    out = np.full(len(values), np.nan, dtype=float)
    if hist.size == 0:
        return pd.Series(out, index=values.index)
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(arr)
    out[finite] = np.searchsorted(hist, arr[finite], side="right") / hist.size
    return pd.Series(out, index=values.index)


def _rolling_percentile(values: pd.Series, *, window: int, min_periods: int) -> pd.Series:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    out = np.full(arr.shape[0], np.nan, dtype=float)
    for idx, value in enumerate(arr):
        if not np.isfinite(value):
            continue
        start = max(0, idx - int(window))
        hist = arr[start:idx]
        hist = hist[np.isfinite(hist)]
        if hist.size < min_periods:
            continue
        out[idx] = np.searchsorted(np.sort(hist), value, side="right") / hist.size
    return pd.Series(out, index=values.index)


def _regime_fixed_percentile(decisions: pd.DataFrame, *, sim_start: pd.Timestamp) -> pd.Series:
    out = np.full(len(decisions), np.nan, dtype=float)
    calibration = decisions[decisions["timestamp"] < sim_start].copy()
    fallback = np.sort(pd.to_numeric(calibration["p_enter_short"], errors="coerce").dropna().to_numpy(dtype=float))
    by_regime: dict[str, np.ndarray] = {}
    for regime, group in calibration.groupby("trend_regime"):
        values = pd.to_numeric(group["p_enter_short"], errors="coerce").dropna().to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if values.size:
            by_regime[str(regime)] = np.sort(values)
    for idx, row in decisions.iterrows():
        value = float(row.get("p_enter_short", np.nan))
        if not math.isfinite(value):
            continue
        hist = by_regime.get(str(row.get("trend_regime", "neutral")), fallback)
        if hist.size:
            out[idx] = np.searchsorted(hist, value, side="right") / hist.size
    return pd.Series(out, index=decisions.index)


def _add_normalized_columns(
    decisions: pd.DataFrame,
    *,
    sim_start: pd.Timestamp,
    windows: list[int],
    min_periods: int,
) -> pd.DataFrame:
    out = decisions.copy()
    calibration = out[out["timestamp"] < sim_start]
    out["p_short_fixed_history_pct"] = _rank_percentile(out["p_enter_short"], calibration["p_enter_short"])
    out["p_short_regime_history_pct"] = _regime_fixed_percentile(out, sim_start=sim_start)
    for window in windows:
        out[f"p_short_rolling_{window}_pct"] = _rolling_percentile(
            out["p_enter_short"],
            window=window,
            min_periods=min_periods,
        )
    return out


def _run_short_only(
    *,
    label: str,
    decisions: pd.DataFrame,
    one_min: pd.DataFrame,
    short_column: str,
    short_thr: float,
) -> dict[str, Any]:
    sim_decisions = decisions.copy()
    sim_decisions["p_enter_short"] = pd.to_numeric(sim_decisions[short_column], errors="coerce")
    sim_decisions = sim_decisions.dropna(subset=["p_enter_short", "p_enter_long"]).reset_index(drop=True)
    events = _run_one(
        decisions=sim_decisions,
        one_min=one_min,
        long_thr=2.0,
        short_thr=float(short_thr),
        **SIM_KW,
    )
    metrics = _metrics(events)
    short_returns = [float(e["return"]) for e in events if e.get("side") == "short"]
    metrics.update(
        {
            "label": label,
            "short_column": short_column,
            "short_threshold": float(short_thr),
            "decision_rows": int(len(sim_decisions)),
            "avg_short_return": float(np.nanmean(short_returns)) if short_returns else float("nan"),
            "sum_short_return": float(np.nansum(short_returns)) if short_returns else 0.0,
        }
    )
    return metrics


def _simulate_signal_window(args: argparse.Namespace) -> pd.DataFrame:
    sim_start = _to_et(args.start)
    decisions = _load_decisions_from_signal_frame(
        signal_frame=Path(args.signal_frame),
        prob_frame=None,
        start=args.history_start,
        prob_source=args.prob_source,
    )
    decisions = _with_regime(decisions, Path(args.signal_frame))
    decisions = _add_normalized_columns(
        decisions,
        sim_start=sim_start,
        windows=_parse_ints(args.windows),
        min_periods=int(args.min_periods),
    )
    sim_decisions = decisions[decisions["timestamp"] >= sim_start].copy().reset_index(drop=True)
    if str(args.end or "").strip():
        sim_end = _to_et(args.end)
        sim_decisions = sim_decisions[sim_decisions["timestamp"] <= sim_end].copy().reset_index(drop=True)
    else:
        sim_end = sim_decisions["timestamp"].max()
    one_min = _load_one_min(Path(args.one_min), args.start, sim_end)
    one_min = one_min[one_min["timestamp"] >= sim_decisions["timestamp"].min() - pd.Timedelta(minutes=10)].copy()

    rows: list[dict[str, Any]] = []
    for thr in _parse_floats(args.raw_short_thresholds):
        rows.append(
            _run_short_only(
                label="raw_short_only",
                decisions=sim_decisions,
                one_min=one_min,
                short_column="p_enter_short",
                short_thr=thr,
            )
        )
    norm_columns = ["p_short_fixed_history_pct", "p_short_regime_history_pct"]
    norm_columns.extend([f"p_short_rolling_{w}_pct" for w in _parse_ints(args.windows)])
    for col in norm_columns:
        for thr in _parse_floats(args.percentile_thresholds):
            rows.append(
                _run_short_only(
                    label=col.replace("p_short_", "").replace("_pct", ""),
                    decisions=sim_decisions,
                    one_min=one_min,
                    short_column=col,
                    short_thr=thr,
                )
            )
    out = pd.DataFrame(rows)
    out.insert(0, "source", "signal_frame")
    out.insert(1, "first_decision", sim_decisions["timestamp"].min())
    out.insert(2, "last_decision", sim_decisions["timestamp"].max())
    return out


def _simulate_recent_live(args: argparse.Namespace) -> pd.DataFrame:
    live = _load_live_decisions(Path(args.run_root), args.live_start, symbol=str(args.symbol))
    if live.empty:
        return pd.DataFrame()
    if str(args.live_end or "").strip():
        live_end = _to_et(args.live_end)
        live = live[live["timestamp"] <= live_end].copy()
    else:
        live_end = live["timestamp"].max()
    calibration = _load_decisions_from_signal_frame(
        signal_frame=Path(args.signal_frame),
        prob_frame=None,
        start=args.history_start,
        prob_source=args.prob_source,
    )
    fixed = _rank_percentile(live["p_enter_short"], calibration["p_enter_short"])
    live = live.copy().reset_index(drop=True)
    live["p_short_fixed_history_pct"] = fixed.reset_index(drop=True)
    one_min = _load_one_min(Path(args.one_min), args.live_start, live_end)
    one_min = one_min[one_min["timestamp"] >= live["timestamp"].min() - pd.Timedelta(minutes=10)].copy()
    if one_min.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for thr in _parse_floats(args.raw_short_thresholds):
        rows.append(
            _run_short_only(
                label="raw_short_only",
                decisions=live,
                one_min=one_min,
                short_column="p_enter_short",
                short_thr=thr,
            )
        )
    for thr in _parse_floats(args.percentile_thresholds):
        rows.append(
            _run_short_only(
                label="fixed_history",
                decisions=live,
                one_min=one_min,
                short_column="p_short_fixed_history_pct",
                short_thr=thr,
            )
        )
    out = pd.DataFrame(rows)
    out.insert(0, "source", "recent_live")
    out.insert(1, "first_decision", live["timestamp"].min())
    out.insert(2, "last_decision", live["timestamp"].max())
    return out


def _load_live_decisions(run_root: Path, start: str, *, symbol: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    start_ts = _to_et(start)
    symbol_key = str(symbol or "").strip().upper()
    for run_idx, run_dir in enumerate(sorted(run_root.glob("*_spy"))):
        path = run_dir / "decision-10m.jsonl"
        if not path.exists():
            continue
        for line_idx, line in enumerate(path.read_text().splitlines()):
            if not line.strip():
                continue
            rec = json.loads(line)
            payload = rec.get("payload", {})
            bar = payload.get("bar", {}) or {}
            row_symbol = str(payload.get("symbol") or bar.get("symbol") or "").strip().upper()
            if row_symbol != symbol_key:
                continue
            ts = _to_et(payload.get("timestamp") or bar.get("timestamp"))
            if pd.isna(ts) or ts < start_ts:
                continue
            rows.append(
                {
                    "timestamp": ts,
                    "available_ts": ts + pd.Timedelta(minutes=10),
                    "run": run_dir.name,
                    "run_idx": run_idx,
                    "line_idx": line_idx,
                    "open": _as_float(bar.get("open")),
                    "high": _as_float(bar.get("high")),
                    "low": _as_float(bar.get("low")),
                    "close": _as_float(bar.get("close")),
                    "atr": _as_float((payload.get("policy_state") or {}).get("atr")),
                    "p_enter_long": _as_float(bar.get("p_enter_long")),
                    "p_enter_short": _as_float(bar.get("p_enter_short")),
                    "p_neutral": _as_float(bar.get("p_swing_setup_neutral")),
                }
            )
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows).sort_values(["timestamp", "run_idx", "line_idx"])
    out = out.drop_duplicates(subset=["timestamp"], keep="last").sort_values("timestamp")
    return out.reset_index(drop=True)


def _load_live_probe(path: Path, *, symbol: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    symbol_key = str(symbol or "").strip().upper()
    if not path.exists():
        return pd.DataFrame()
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        payload = rec.get("payload", {})
        bar = payload.get("bar", {}) or {}
        row_symbol = str(payload.get("symbol") or bar.get("symbol") or "").strip().upper()
        if row_symbol != symbol_key:
            continue
        ts = _to_et(payload.get("timestamp") or bar.get("timestamp"))
        if pd.isna(ts):
            continue
        rows.append(
            {
                "timestamp": ts,
                "open": float(bar.get("open", np.nan)),
                "high": float(bar.get("high", np.nan)),
                "low": float(bar.get("low", np.nan)),
                "close": float(bar.get("close", np.nan)),
                "p_enter_long": float(bar.get("p_enter_long", np.nan)),
                "p_enter_short": float(bar.get("p_enter_short", np.nan)),
            }
        )
    return pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)


def _probe_may4(args: argparse.Namespace) -> pd.DataFrame:
    live = _load_live_probe(Path(args.may4_decisions), symbol=str(args.symbol))
    if live.empty:
        return live
    history = _load_decisions_from_signal_frame(
        signal_frame=Path(args.signal_frame),
        prob_frame=None,
        start=args.history_start,
        prob_source=args.prob_source,
    )
    live_recent = _load_live_decisions(Path(args.run_root), args.live_start, symbol=str(args.symbol))
    live["short_fixed_history_pct"] = _rank_percentile(live["p_enter_short"], history["p_enter_short"])
    if not live_recent.empty:
        live["short_recent_live_pct"] = _rank_percentile(live["p_enter_short"], live_recent["p_enter_short"])
    else:
        live["short_recent_live_pct"] = np.nan
    recent_history = history.tail(780)
    live["short_recent_20d_signal_pct"] = _rank_percentile(live["p_enter_short"], recent_history["p_enter_short"])
    live["next_30m_close_change"] = live["close"].shift(-3) - live["close"]
    return live[live["p_enter_short"] >= float(args.probe_min_short)].copy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe raw vs normalized short probability thresholds.")
    parser.add_argument("--signal-frame", default=str(DEFAULT_SIGNAL_FRAME))
    parser.add_argument("--prob-source", default="blend", choices=["blend", "full", "test", "oof"])
    parser.add_argument("--one-min", default=str(DEFAULT_ONE_MIN))
    parser.add_argument("--run-root", default=str(DEFAULT_RUN_ROOT))
    parser.add_argument("--history-start", default="2020-12-01T00:00:00-05:00")
    parser.add_argument("--start", default="2026-01-02T00:00:00-05:00")
    parser.add_argument("--end", default="")
    parser.add_argument("--live-start", default="2026-04-01T00:00:00-04:00")
    parser.add_argument("--live-end", default="2026-05-01T23:59:00-04:00")
    parser.add_argument("--windows", default="195,780")
    parser.add_argument("--min-periods", type=int, default=80)
    parser.add_argument("--raw-short-thresholds", default="0.30,0.40,0.65")
    parser.add_argument("--percentile-thresholds", default="0.85,0.90,0.95")
    parser.add_argument("--may4-decisions", default=str(DEFAULT_MAY4_DECISIONS))
    parser.add_argument("--symbol", default=DEFAULT_SYMBOL)
    parser.add_argument("--probe-min-short", type=float, default=0.15)
    parser.add_argument("--skip-signal", action="store_true")
    parser.add_argument("--skip-live", action="store_true")
    args = parser.parse_args()

    summaries: list[pd.DataFrame] = []
    if not args.skip_signal:
        summaries.append(_simulate_signal_window(args))
    if not args.skip_live:
        live_summary = _simulate_recent_live(args)
        if not live_summary.empty:
            summaries.append(live_summary)
    if summaries:
        summary = pd.concat(summaries, ignore_index=True)
        sort_cols = ["source", "sum_short_return", "avg_short_return", "short_trades"]
        summary = summary.sort_values(sort_cols, ascending=[True, False, False, False])
        cols = [
            "source",
            "label",
            "short_column",
            "short_threshold",
            "decision_rows",
            "first_decision",
            "last_decision",
            "trades",
            "short_trades",
            "avg_short_return",
            "sum_short_return",
            "median_return",
            "win_rate",
            "avg_mfe",
        ]
        print("\n=== Short-only threshold comparison ===")
        print(summary[cols].to_string(index=False))

    probe = _probe_may4(args)
    if not probe.empty:
        cols = [
            "timestamp",
            "close",
            "p_enter_long",
            "p_enter_short",
            "short_fixed_history_pct",
            "short_recent_20d_signal_pct",
            "short_recent_live_pct",
            "next_30m_close_change",
        ]
        print("\n=== 2026-05-04 short probability spike probe ===")
        print(probe[cols].to_string(index=False))


if __name__ == "__main__":
    main()
