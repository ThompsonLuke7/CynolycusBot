from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from strategies.spy_intraday.Models.ga_xgboost.analyze_phase4_triggers import (  # noqa: E402
    _apply_conflict_cooldown_cluster,
    _expand_recent_setup,
    _load_execution_1m,
    _post_setup_policy_to_internal,
    _post_setup_side_candidates,
    _session_reset_mask,
)
from strategies.spy_intraday.Models.ga_xgboost.swing_label_weights import compute_wilder_atr  # noqa: E402


DEFAULT_ANALYSIS_DIR = (
    "Data/models/ga_xgboost/10min/analysis/"
    "phase4_1m_oof_focused_trigger_sweep_l42_s15_full_1m_train"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep option profit-lock arm/floor thresholds using SPY OHLC as a "
            "0DTE option value proxy. This does not require option quote history."
        )
    )
    parser.add_argument(
        "--signal-frame",
        default=f"{DEFAULT_ANALYSIS_DIR}/phase4_signal_frame.parquet",
        help="Phase4 signal frame parquet.",
    )
    parser.add_argument(
        "--scoreboard",
        default=f"{DEFAULT_ANALYSIS_DIR}/phase4_trigger_scoreboard.json",
        help="Phase4 trigger scoreboard JSON used to select/rebuild entries.",
    )
    parser.add_argument("--split", default="oof", choices=["oof", "test"], help="Signal split to evaluate.")
    parser.add_argument(
        "--variant",
        default=None,
        help="Optional trigger variant. Defaults to the best available row by total_ev_atr for --split.",
    )
    parser.add_argument("--mode", default=None, help="Optional trigger mode matching the scoreboard row.")
    parser.add_argument("--long-setup-threshold", type=float, default=None)
    parser.add_argument("--short-setup-threshold", type=float, default=None)
    parser.add_argument("--cooldown-bars", type=int, default=None)
    parser.add_argument("--post-setup-max-bars", type=int, default=None)
    parser.add_argument("--one-per-setup-cluster", action="store_true", default=None)
    parser.add_argument("--no-one-per-setup-cluster", action="store_false", dest="one_per_setup_cluster")
    parser.add_argument(
        "--use-1m-execution",
        action="store_true",
        help="Use 1-minute SPY bars for trigger rebuilding and exit path simulation.",
    )
    parser.add_argument(
        "--execution-1m-path",
        default="Data/raw/spy/spy_intraday_1min.parquet",
        help="1-minute SPY OHLC parquet with timestamp/open/high/low/close columns.",
    )
    parser.add_argument(
        "--premium-atr-mult",
        type=float,
        default=2.5,
        help=(
            "Option premium proxy as entry SPY ATR multiple. Today's 679C entry "
            "was roughly 1.59 / 0.61 = 2.6 ATR, so 2.5 is a local approximation."
        ),
    )
    parser.add_argument(
        "--premium-fixed",
        type=float,
        default=None,
        help="Optional fixed SPY-point premium proxy. Overrides --premium-atr-mult.",
    )
    parser.add_argument("--take-profit-pct", type=float, default=1.0, help="+100%% take profit by default.")
    parser.add_argument("--stop-loss-pct", type=float, default=0.8, help="-80%% stop loss by default.")
    parser.add_argument(
        "--take-profit-values",
        default=None,
        help="Optional comma-separated TP sweep values, e.g. 1.0,2.0,3.0,none.",
    )
    parser.add_argument(
        "--stop-loss-values",
        default=None,
        help="Optional comma-separated stop-loss sweep values, e.g. 0.5,0.8,1.0.",
    )
    parser.add_argument(
        "--arm-values",
        default="0.4,0.5,0.6,0.75,1.0,1.25",
        help="Comma-separated profit-lock arming thresholds as option-premium return fractions.",
    )
    parser.add_argument(
        "--floor-values",
        default="0.0,0.1,0.2,0.3,0.4,0.5,0.75",
        help="Comma-separated profit-lock floor thresholds as option-premium return fractions.",
    )
    parser.add_argument(
        "--include-no-lock",
        action="store_true",
        help="Include a TP/SL only baseline with profit lock disabled.",
    )
    parser.add_argument(
        "--include-trailing",
        action="store_true",
        help="Include true trailing-giveback exits after the arm threshold is reached.",
    )
    parser.add_argument(
        "--trailing-arm-values",
        default="0.5,0.75,1.0,1.5,2.0",
        help="Comma-separated arming thresholds for trailing exits.",
    )
    parser.add_argument(
        "--trailing-giveback-values",
        default="0.25,0.5,0.75,1.0,1.5",
        help="Comma-separated giveback from best profit, in option-premium return fractions.",
    )
    parser.add_argument(
        "--exit-hhmm",
        default="15:40",
        help="Same-session forced exit cutoff in the signal-frame timezone.",
    )
    parser.add_argument(
        "--horizon-bars",
        type=int,
        default=39,
        help="Maximum 10-minute bars to hold before timeout/forced proxy close.",
    )
    parser.add_argument(
        "--out",
        default=f"{DEFAULT_ANALYSIS_DIR}/phase4_spy_proxy_profit_lock_sweep.csv",
        help="Output CSV summary path.",
    )
    parser.add_argument(
        "--events-out",
        default=None,
        help="Optional per-trade event CSV path. Can be large.",
    )
    return parser.parse_args()


def _parse_float_list(raw: str) -> list[float]:
    out: list[float] = []
    for part in str(raw or "").split(","):
        part = part.strip()
        if part:
            out.append(float(part))
    return out


def _parse_optional_float_list(raw: str) -> list[float | None]:
    out: list[float | None] = []
    for part in str(raw or "").split(","):
        part = part.strip().lower()
        if not part:
            continue
        if part in {"none", "nan", "uncapped"}:
            out.append(None)
        else:
            out.append(float(part))
    return out


def _load_scoreboard(path: Path, split: str, variant: str | None, mode: str | None) -> pd.Series:
    rows = pd.DataFrame(json.loads(path.read_text()))
    rows = rows[rows.get("available", True).eq(True) & rows["split"].eq(split)].copy()
    if variant is not None:
        rows = rows[rows["variant"].astype(str).eq(str(variant))]
    if mode is not None:
        rows = rows[rows["mode"].astype(str).eq(str(mode))]
    if rows.empty:
        raise ValueError(f"No scoreboard row found for split={split!r} variant={variant!r} mode={mode!r}")
    rows = rows.sort_values(
        ["total_ev_atr", "long_ev_atr", "short_ev_atr", "total_trades_per_day"],
        ascending=[False, False, False, True],
    )
    return rows.iloc[0]


def _trigger_col(policy_name: str, side: str) -> str | None:
    trigger_names = {
        "H": "trigger_H",
        "G": "trigger_G",
        "E": "trigger_E",
        "I": "trigger_I",
        "post_setup_trigger_A_break": "trigger_A",
        "post_setup_trigger_B_break_momentum": "trigger_B",
        "post_setup_trigger_C_break_body": "trigger_C",
        "post_setup_trigger_E_break_ema": "trigger_E",
        "post_setup_trigger_F_body_momentum": "trigger_F",
        "post_setup_trigger_G_ema_slope": "trigger_G",
        "post_setup_trigger_H_body_ema": "trigger_H",
        "post_setup_trigger_I_break_or_ema": "trigger_I",
        "post_setup_trigger_J_close_cross_fast_ema": "trigger_J",
    }
    prefix = trigger_names.get(policy_name)
    return f"{prefix}_{side}" if prefix is not None else None


def _asym_policy_names(variant: str) -> tuple[str, str] | None:
    prefix = "asym_long_"
    marker = "_short_"
    if not variant.startswith(prefix) or marker not in variant:
        return None
    rest = variant[len(prefix) :]
    long_name, short_name = rest.split(marker, 1)
    return long_name, short_name


def _build_entries(
    feature_df: pd.DataFrame,
    row: pd.Series,
    *,
    split: str,
    long_setup_threshold: float,
    short_setup_threshold: float,
    cooldown_bars: int,
    post_setup_max_bars: int,
    one_per_setup_cluster: bool,
    execution_1m: pd.DataFrame | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    p_suffix = "oof_train" if split == "oof" else "test"
    p_long = pd.to_numeric(feature_df[f"p_long_{p_suffix}"], errors="coerce").to_numpy(dtype=float)
    p_short = pd.to_numeric(feature_df[f"p_short_{p_suffix}"], errors="coerce").to_numpy(dtype=float)
    eval_mask = feature_df[f"is_{split}"].fillna(False).to_numpy(dtype=bool)
    finite = np.isfinite(p_long) & np.isfinite(p_short)
    long_raw_setup = eval_mask & finite & (p_long >= float(long_setup_threshold))
    short_raw_setup = eval_mask & finite & (p_short >= float(short_setup_threshold))

    variant = str(row["variant"])
    asym_names = _asym_policy_names(variant)
    if asym_names is None:
        long_policy_name = variant
        short_policy_name = variant
    else:
        long_policy_name, short_policy_name = asym_names
    max_entry_lag_minutes = row.get("max_entry_lag_minutes", None)
    if pd.isna(max_entry_lag_minutes):
        max_entry_lag_minutes = None
    elif max_entry_lag_minutes is not None:
        max_entry_lag_minutes = float(max_entry_lag_minutes)

    def _internal_policy(name: str) -> str:
        mapped = _post_setup_policy_to_internal(name)
        if mapped == "trigger_close" and str(name).startswith("break_prev_stop"):
            return str(name)
        if mapped == "trigger_close" and str(name) in {"next_open", "next_open_direction", "micro_reversal_1m"}:
            return str(name)
        return mapped

    long_candidate, long_prices, long_times, _, _ = _post_setup_side_candidates(
        feature_df,
        long_raw_setup,
        eval_mask,
        side="long",
        policy=_internal_policy(long_policy_name),
        trigger_col=_trigger_col(long_policy_name, "long"),
        max_bars=post_setup_max_bars,
        execution_1m=execution_1m,
        max_entry_lag_minutes=max_entry_lag_minutes,
    )
    short_candidate, short_prices, short_times, _, _ = _post_setup_side_candidates(
        feature_df,
        short_raw_setup,
        eval_mask,
        side="short",
        policy=_internal_policy(short_policy_name),
        trigger_col=_trigger_col(short_policy_name, "short"),
        max_bars=post_setup_max_bars,
        execution_1m=execution_1m,
        max_entry_lag_minutes=max_entry_lag_minutes,
    )
    active_long = _expand_recent_setup(long_raw_setup, post_setup_max_bars) & eval_mask
    active_short = _expand_recent_setup(short_raw_setup, post_setup_max_bars) & eval_mask
    long_entries, short_entries, _ = _apply_conflict_cooldown_cluster(
        long_candidate,
        short_candidate,
        active_long,
        active_short,
        cooldown_bars=cooldown_bars,
        one_per_setup_cluster=one_per_setup_cluster,
        session_reset=_session_reset_mask(feature_df.index),
    )
    return long_entries, short_entries, long_prices, short_prices, long_times, short_times


def _parse_exit_time(hhmm: str) -> tuple[int, int]:
    hour_raw, minute_raw = str(hhmm).split(":", 1)
    return int(hour_raw), int(minute_raw)


def _session_cutoff(ts: pd.Timestamp, *, hhmm: str) -> pd.Timestamp:
    hour, minute = _parse_exit_time(hhmm)
    return pd.Timestamp(ts).normalize() + pd.Timedelta(hours=hour, minutes=minute)


def _path_after_entry(
    feature_df: pd.DataFrame,
    *,
    entry_idx: int,
    entry_ts: pd.Timestamp,
    execution_1m: pd.DataFrame | None,
    exit_hhmm: str,
    horizon_bars: int,
) -> tuple[list[pd.Timestamp], np.ndarray, np.ndarray, np.ndarray]:
    cutoff = _session_cutoff(entry_ts, hhmm=exit_hhmm)
    max_end = entry_ts + pd.Timedelta(minutes=10 * max(1, int(horizon_bars)))
    end_ts = min(cutoff, max_end)

    if execution_1m is not None:
        idx = execution_1m.index
        left = idx.searchsorted(entry_ts, side="right")
        right = idx.searchsorted(end_ts, side="right")
        path = execution_1m.iloc[left:right]
        if not path.empty:
            return (
                list(path.index),
                pd.to_numeric(path["high"], errors="coerce").to_numpy(dtype=float),
                pd.to_numeric(path["low"], errors="coerce").to_numpy(dtype=float),
                pd.to_numeric(path["close"], errors="coerce").to_numpy(dtype=float),
            )

    end_idx = min(len(feature_df) - 1, entry_idx + max(1, int(horizon_bars)))
    if isinstance(feature_df.index, pd.DatetimeIndex):
        right = entry_idx + 1
        while (
            right <= end_idx
            and feature_df.index[right].normalize() == entry_ts.normalize()
            and feature_df.index[right] <= end_ts
        ):
            right += 1
        valid = np.arange(entry_idx + 1, right, dtype=int)
    else:
        valid = np.arange(entry_idx + 1, end_idx + 1)
    if valid.size == 0:
        return [], np.array([]), np.array([]), np.array([])
    return (
        list(feature_df.index[valid]) if isinstance(feature_df.index, pd.DatetimeIndex) else [pd.Timestamp("NaT")] * len(valid),
        pd.to_numeric(feature_df["high"], errors="coerce").to_numpy(dtype=float)[valid],
        pd.to_numeric(feature_df["low"], errors="coerce").to_numpy(dtype=float)[valid],
        pd.to_numeric(feature_df["close"], errors="coerce").to_numpy(dtype=float)[valid],
    )


def _simulate_one(
    *,
    side: str,
    entry: float,
    premium: float,
    times: list[pd.Timestamp],
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    take_profit_pct: float | None,
    stop_loss_pct: float,
    arm_pct: float | None,
    floor_pct: float | None,
    exit_style: str = "floor_lock",
    trail_giveback_pct: float | None = None,
) -> dict[str, object] | None:
    if not np.isfinite(entry) or not np.isfinite(premium) or premium <= 0 or closes.size == 0:
        return None

    best_move = 0.0
    worst_move = 0.0
    final_move = 0.0
    exit_reason = "timeout"
    exit_time = times[-1] if times else pd.NaT
    exit_move = 0.0
    tp_enabled = take_profit_pct is not None and np.isfinite(float(take_profit_pct)) and float(take_profit_pct) > 0.0
    lock_enabled = arm_pct is not None

    for ts, hi, lo, close in zip(times, highs, lows, closes):
        if side == "long":
            favorable = hi - entry
            adverse = lo - entry
            close_move = close - entry
        else:
            favorable = entry - lo
            adverse = entry - hi
            close_move = entry - close

        best_move = max(best_move, float(favorable))
        worst_move = min(worst_move, float(adverse))
        final_move = float(close_move)

        if adverse <= -float(stop_loss_pct) * premium:
            exit_reason = "stop_loss"
            exit_time = ts
            exit_move = -float(stop_loss_pct) * premium
            break
        if tp_enabled and best_move >= float(take_profit_pct) * premium:
            exit_reason = "take_profit"
            exit_time = ts
            exit_move = float(take_profit_pct) * premium
            break
        if lock_enabled and best_move >= float(arm_pct) * premium:
            if exit_style == "trailing":
                giveback = 0.0 if trail_giveback_pct is None else float(trail_giveback_pct)
                trail_move = best_move - giveback * premium
                if floor_pct is not None:
                    trail_move = max(trail_move, float(floor_pct) * premium)
                if adverse <= trail_move:
                    exit_reason = "trailing_stop"
                    exit_time = ts
                    exit_move = trail_move
                    break
            elif floor_pct is not None and adverse <= float(floor_pct) * premium:
                exit_reason = "profit_lock"
                exit_time = ts
                exit_move = float(floor_pct) * premium
                break
    else:
        exit_move = final_move

    return {
        "exit_reason": exit_reason,
        "exit_time": str(exit_time),
        "outcome_pct": float(exit_move / premium),
        "mfe_pct": float(best_move / premium),
        "mae_pct": float(worst_move / premium),
    }


def _simulate_grid(
    feature_df: pd.DataFrame,
    *,
    long_entries: np.ndarray,
    short_entries: np.ndarray,
    long_prices: np.ndarray,
    short_prices: np.ndarray,
    long_times: np.ndarray,
    short_times: np.ndarray,
    execution_1m: pd.DataFrame | None,
    premium_atr_mult: float,
    premium_fixed: float | None,
    exit_specs: list[dict[str, object]],
    exit_hhmm: str,
    horizon_bars: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    high = pd.to_numeric(feature_df["high"], errors="coerce").to_numpy(dtype=float)
    low = pd.to_numeric(feature_df["low"], errors="coerce").to_numpy(dtype=float)
    close = pd.to_numeric(feature_df["close"], errors="coerce").to_numpy(dtype=float)
    atr = compute_wilder_atr(high, low, close, length=14)
    all_event_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    entry_specs: list[dict[str, object]] = []
    for side, entries, prices, times in (
        ("long", long_entries, long_prices, long_times),
        ("short", short_entries, short_prices, short_times),
    ):
        for idx in np.flatnonzero(entries):
            entry = float(prices[idx]) if np.isfinite(prices[idx]) else float(close[idx])
            entry_ts = times[idx] if pd.notna(times[idx]) else feature_df.index[idx]
            premium = float(premium_fixed) if premium_fixed is not None else float(atr[idx]) * float(premium_atr_mult)
            path_times, path_highs, path_lows, path_closes = _path_after_entry(
                feature_df,
                entry_idx=int(idx),
                entry_ts=pd.Timestamp(entry_ts),
                execution_1m=execution_1m,
                exit_hhmm=exit_hhmm,
                horizon_bars=horizon_bars,
            )
            entry_specs.append(
                {
                    "side": side,
                    "idx": int(idx),
                    "entry": float(entry),
                    "entry_ts": pd.Timestamp(entry_ts),
                    "premium": float(premium),
                    "times": path_times,
                    "highs": path_highs,
                    "lows": path_lows,
                    "closes": path_closes,
                }
            )

    for exit_spec in exit_specs:
        label = str(exit_spec["regime"])
        arm_pct = exit_spec.get("arm_pct")
        floor_pct = exit_spec.get("floor_pct")
        trail_giveback_pct = exit_spec.get("trail_giveback_pct")
        take_profit_pct = exit_spec.get("take_profit_pct")
        stop_loss_pct = float(exit_spec["stop_loss_pct"])
        exit_style = str(exit_spec.get("exit_style", "floor_lock"))
        event_rows: list[dict[str, object]] = []
        for spec in entry_specs:
            result = _simulate_one(
                side=str(spec["side"]),
                entry=float(spec["entry"]),
                premium=float(spec["premium"]),
                times=spec["times"],  # type: ignore[arg-type]
                highs=spec["highs"],  # type: ignore[arg-type]
                lows=spec["lows"],  # type: ignore[arg-type]
                closes=spec["closes"],  # type: ignore[arg-type]
                take_profit_pct=take_profit_pct,  # type: ignore[arg-type]
                stop_loss_pct=stop_loss_pct,
                arm_pct=arm_pct if arm_pct is None else float(arm_pct),
                floor_pct=floor_pct if floor_pct is None else float(floor_pct),
                exit_style=exit_style,
                trail_giveback_pct=trail_giveback_pct if trail_giveback_pct is None else float(trail_giveback_pct),
            )
            if result is None:
                continue
            row = {
                "regime": label,
                "arm_pct": np.nan if arm_pct is None else float(arm_pct),
                "floor_pct": np.nan if floor_pct is None else float(floor_pct),
                "trail_giveback_pct": np.nan if trail_giveback_pct is None else float(trail_giveback_pct),
                "take_profit_pct": np.nan if take_profit_pct is None else float(take_profit_pct),
                "stop_loss_pct": float(stop_loss_pct),
                "exit_style": str(exit_style),
                "side": str(spec["side"]),
                "entry_time": str(spec["entry_ts"]),
                "entry_price": float(spec["entry"]),
                "premium_proxy": float(spec["premium"]),
                **result,
            }
            event_rows.append(row)
            all_event_rows.append(row)

        events = pd.DataFrame(event_rows)
        if events.empty:
            continue
        outcome = pd.to_numeric(events["outcome_pct"], errors="coerce")
        summary: dict[str, object] = {
            "regime": label,
            "arm_pct": np.nan if arm_pct is None else float(arm_pct),
            "floor_pct": np.nan if floor_pct is None else float(floor_pct),
            "trail_giveback_pct": np.nan if trail_giveback_pct is None else float(trail_giveback_pct),
            "take_profit_pct": np.nan if take_profit_pct is None else float(take_profit_pct),
            "stop_loss_pct": float(stop_loss_pct),
            "exit_style": str(exit_style),
            "trades": int(len(events)),
            "mean_outcome_pct": float(outcome.mean()),
            "median_outcome_pct": float(outcome.median()),
            "total_outcome_pct": float(outcome.sum()),
            "win_rate": float((outcome > 0).mean()),
            "avg_mfe_pct": float(pd.to_numeric(events["mfe_pct"], errors="coerce").mean()),
            "avg_mae_pct": float(pd.to_numeric(events["mae_pct"], errors="coerce").mean()),
        }
        for reason, count in events["exit_reason"].value_counts().items():
            summary[f"{reason}_count"] = int(count)
        for side in ("long", "short"):
            side_events = events[events["side"].eq(side)]
            side_outcome = pd.to_numeric(side_events["outcome_pct"], errors="coerce")
            summary[f"{side}_trades"] = int(len(side_events))
            summary[f"{side}_mean_outcome_pct"] = float(side_outcome.mean()) if not side_events.empty else np.nan
            summary[f"{side}_win_rate"] = float((side_outcome > 0).mean()) if not side_events.empty else np.nan
        summary_rows.append(summary)

    summary_df = pd.DataFrame(summary_rows).fillna(
        {
            key: 0
            for key in [
                "take_profit_count",
                "stop_loss_count",
                "profit_lock_count",
                "trailing_stop_count",
                "timeout_count",
            ]
        }
    )
    if not summary_df.empty:
        summary_df["rank_mean"] = summary_df["mean_outcome_pct"].rank(method="min", ascending=False)
        summary_df = summary_df.sort_values(
            ["mean_outcome_pct", "win_rate", "total_outcome_pct"],
            ascending=[False, False, False],
        ).reset_index(drop=True)
    return summary_df, pd.DataFrame(all_event_rows)


def main() -> None:
    args = _parse_args()
    signal_path = REPO_ROOT / args.signal_frame
    scoreboard_path = REPO_ROOT / args.scoreboard
    feature_df = pd.read_parquet(signal_path).sort_index()
    selected = _load_scoreboard(scoreboard_path, args.split, args.variant, args.mode)

    long_threshold = float(args.long_setup_threshold if args.long_setup_threshold is not None else selected["long_setup_threshold"])
    short_threshold = float(args.short_setup_threshold if args.short_setup_threshold is not None else selected["short_setup_threshold"])
    cooldown_bars = int(args.cooldown_bars if args.cooldown_bars is not None else selected["cooldown_bars"])
    post_setup_max_bars = int(
        args.post_setup_max_bars if args.post_setup_max_bars is not None else selected.get("post_setup_max_bars", 4)
    )
    if args.one_per_setup_cluster is None:
        one_per_setup_cluster = bool(selected.get("one_per_setup_cluster", False))
    else:
        one_per_setup_cluster = bool(args.one_per_setup_cluster)

    execution_1m = _load_execution_1m(args, feature_df.index)
    long_entries, short_entries, long_prices, short_prices, long_times, short_times = _build_entries(
        feature_df,
        selected,
        split=str(args.split),
        long_setup_threshold=long_threshold,
        short_setup_threshold=short_threshold,
        cooldown_bars=cooldown_bars,
        post_setup_max_bars=post_setup_max_bars,
        one_per_setup_cluster=one_per_setup_cluster,
        execution_1m=execution_1m,
    )

    take_profit_values = (
        _parse_optional_float_list(args.take_profit_values)
        if args.take_profit_values is not None
        else [float(args.take_profit_pct)]
    )
    stop_loss_values = (
        _parse_float_list(args.stop_loss_values)
        if args.stop_loss_values is not None
        else [float(args.stop_loss_pct)]
    )
    exit_specs: list[dict[str, object]] = []
    if bool(args.include_no_lock):
        for tp in take_profit_values:
            for sl in stop_loss_values:
                tp_label = "none" if tp is None else f"{tp:.2f}"
                exit_specs.append(
                    {
                        "regime": f"tp_{tp_label}_sl_{sl:.2f}_no_lock",
                        "take_profit_pct": tp,
                        "stop_loss_pct": float(sl),
                        "arm_pct": None,
                        "floor_pct": None,
                        "trail_giveback_pct": None,
                        "exit_style": "none",
                    }
                )
    for tp in take_profit_values:
        for sl in stop_loss_values:
            for arm in _parse_float_list(args.arm_values):
                for floor in _parse_float_list(args.floor_values):
                    if floor < arm and (tp is None or floor < float(tp)):
                        tp_label = "none" if tp is None else f"{tp:.2f}"
                        exit_specs.append(
                            {
                                "regime": f"tp_{tp_label}_sl_{sl:.2f}_floor_arm_{arm:.2f}_floor_{floor:.2f}",
                                "take_profit_pct": tp,
                                "stop_loss_pct": float(sl),
                                "arm_pct": float(arm),
                                "floor_pct": float(floor),
                                "trail_giveback_pct": None,
                                "exit_style": "floor_lock",
                            }
                        )
    if bool(args.include_trailing):
        for tp in take_profit_values:
            for sl in stop_loss_values:
                for arm in _parse_float_list(args.trailing_arm_values):
                    for giveback in _parse_float_list(args.trailing_giveback_values):
                        tp_label = "none" if tp is None else f"{tp:.2f}"
                        exit_specs.append(
                            {
                                "regime": f"tp_{tp_label}_sl_{sl:.2f}_trail_arm_{arm:.2f}_gb_{giveback:.2f}",
                                "take_profit_pct": tp,
                                "stop_loss_pct": float(sl),
                                "arm_pct": float(arm),
                                "floor_pct": None,
                                "trail_giveback_pct": float(giveback),
                                "exit_style": "trailing",
                            }
                        )

    summary, events = _simulate_grid(
        feature_df,
        long_entries=long_entries,
        short_entries=short_entries,
        long_prices=long_prices,
        short_prices=short_prices,
        long_times=long_times,
        short_times=short_times,
        execution_1m=execution_1m,
        premium_atr_mult=float(args.premium_atr_mult),
        premium_fixed=args.premium_fixed,
        exit_specs=exit_specs,
        exit_hhmm=str(args.exit_hhmm),
        horizon_bars=int(args.horizon_bars),
    )

    out_path = REPO_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_path, index=False)
    if args.events_out:
        events_path = REPO_ROOT / args.events_out
        events_path.parent.mkdir(parents=True, exist_ok=True)
        events.to_csv(events_path, index=False)

    print(
        "selected="
        f"{selected['split']} {selected['variant']} {selected['mode']} "
        f"long_thr={long_threshold} short_thr={short_threshold} "
        f"cooldown={cooldown_bars} max_bars={post_setup_max_bars} "
        f"one_per_cluster={one_per_setup_cluster}"
    )
    print(f"entries long={int(long_entries.sum())} short={int(short_entries.sum())}")
    show_cols = [
        "regime",
        "exit_style",
        "take_profit_pct",
        "stop_loss_pct",
        "arm_pct",
        "floor_pct",
        "trail_giveback_pct",
        "trades",
        "mean_outcome_pct",
        "win_rate",
        "take_profit_count",
        "stop_loss_count",
        "profit_lock_count",
        "trailing_stop_count",
        "timeout_count",
        "long_mean_outcome_pct",
        "short_mean_outcome_pct",
    ]
    print(summary[[col for col in show_cols if col in summary.columns]].head(15).to_string(index=False))
    print(f"summary_csv={out_path}")
    if args.events_out:
        print(f"events_csv={REPO_ROOT / args.events_out}")


if __name__ == "__main__":
    main()
