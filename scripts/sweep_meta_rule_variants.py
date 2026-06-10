from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.API.Alpaca_API.inference.live_inference import LiveMetaXGBAgent
from scripts.replay_meta_independent import _load_meta_matrix, _normalize_bounds, _score_exit


@dataclass(frozen=True)
class RegimeConfig:
    name: str
    entry_rule: str = "threshold"  # threshold|score
    exit_rule: str = "hybrid"  # hybrid|threshold|score
    entry_prob_mode: str = "raw"
    exit_prob_mode: str = "raw"
    score_mode: str = "entry_diff"  # entry_diff|enter_minus_exit|net_edge
    entry_score_threshold: float = 0.10
    exit_score_threshold: float = 0.00
    require_entry_threshold: bool = True
    entry_threshold: float | None = None
    exit_threshold: float | None = None
    min_hold_bars: int = 2
    soft_exit_confirm_bars: int = 2
    urgent_exit_prob: float = 0.85
    urgent_exit_delta: float = 0.30
    exit_entry_delta: float = 0.15
    opposite_dominance_delta: float = 0.0
    max_hold_bars: int | None = None
    same_side_reentry_cooldown_bars: int = 0
    profit_protect_arm_atr: float | None = None
    profit_protect_giveback_long: float | None = None
    profit_protect_giveback_short: float | None = None
    entry_fill: str = "next_open"  # next_open


@dataclass
class SideRuntime:
    active: bool = False
    intent_active: bool = False
    signal_row: pd.Series | None = None
    pending_exit: bool = False
    pending_exit_reason: str | None = None
    soft_confirm: int = 0
    entry_price: float = float("nan")
    entry_atr: float = float("nan")
    favorable_anchor: float = float("nan")
    entry_ref_price: float = float("nan")
    exit_ema: float | None = None
    cooldown_until_idx: int = -1

    def clear_trade_state(self) -> None:
        self.active = False
        self.intent_active = False
        self.signal_row = None
        self.pending_exit = False
        self.pending_exit_reason = None
        self.soft_confirm = 0
        self.entry_price = float("nan")
        self.entry_atr = float("nan")
        self.favorable_anchor = float("nan")
        self.entry_ref_price = float("nan")
        self.exit_ema = None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep alternative meta entry/exit regimes on cached 10m meta probabilities and 1m execution bars."
        )
    )
    parser.add_argument(
        "--meta-matrix",
        default="Data/inference/spy/10min/debug_matrices_warmup/spy/live_meta_matrix_on_trace_ts_live_2026_03_24.parquet",
        help="Cached 10m meta matrix parquet.",
    )
    parser.add_argument(
        "--one-min-data",
        default="Data/raw/spy/spy_intraday_1min_live_2026_03_24.parquet",
        help="Raw 1m parquet used for entry/exit execution timing and swing measurements.",
    )
    parser.add_argument(
        "--model-root",
        default="Data/models/meta_xgboost/10min",
        help="Meta model root directory.",
    )
    parser.add_argument("--symbol", default="SPY", help="Symbol label.")
    parser.add_argument(
        "--start",
        default=None,
        help="Optional UTC start timestamp. Omit to use the full overlap between the cached meta matrix and 1m data.",
    )
    parser.add_argument(
        "--end",
        default=None,
        help="Optional UTC end timestamp. Omit to use the full overlap between the cached meta matrix and 1m data.",
    )
    parser.add_argument("--tz", default="America/New_York", help="Display timezone for the cached meta matrix loader.")
    parser.add_argument(
        "--summary-out",
        default="Data/inference/spy/10min/meta/meta_rule_variant_experiment_summary.csv",
        help="Summary CSV path.",
    )
    parser.add_argument(
        "--trades-out",
        default="Data/inference/spy/10min/meta/meta_rule_variant_experiment_trades.csv",
        help="Per-trade metrics CSV path.",
    )
    parser.add_argument(
        "--events-out",
        default="Data/inference/spy/10min/meta/meta_rule_variant_experiment_events.csv",
        help="Combined events CSV path.",
    )
    return parser.parse_args()


def _load_one_min(path: Path, *, symbol: str, start: pd.Timestamp | None, end: pd.Timestamp | None) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "timestamp" not in df.columns:
        raise ValueError(f"1m data at {path} must contain a timestamp column.")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).copy()
    if "symbol" in df.columns:
        df = df[df["symbol"].astype(str).str.upper() == str(symbol).upper()].copy()
    if start is not None:
        df = df[df["timestamp"] >= start]
    if end is not None:
        df = df[df["timestamp"] <= end]
    if df.empty:
        raise ValueError("1m data is empty after filtering.")
    return df.sort_values("timestamp").reset_index(drop=True)


def _resolve_overlap_bounds(
    meta_path: Path,
    one_min_path: Path,
    *,
    user_start: pd.Timestamp | None,
    user_end: pd.Timestamp | None,
) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    if user_start is not None or user_end is not None:
        return user_start, user_end
    meta_df = pd.read_parquet(meta_path, columns=["timestamp"])
    one_df = pd.read_parquet(one_min_path, columns=["timestamp"])
    meta_ts = pd.to_datetime(meta_df["timestamp"], utc=True, errors="coerce").dropna()
    one_ts = pd.to_datetime(one_df["timestamp"], utc=True, errors="coerce").dropna()
    if meta_ts.empty or one_ts.empty:
        return user_start, user_end
    start = max(meta_ts.min(), one_ts.min())
    end = min(meta_ts.max(), one_ts.max())
    return start, end


def _ema_series(values: np.ndarray, alpha: float) -> np.ndarray:
    return pd.Series(values, dtype=float).ewm(alpha=float(alpha), adjust=False).mean().to_numpy(dtype=float)


def _ema_step(prev: float | None, value: float, alpha: float) -> float:
    if not np.isfinite(value):
        return float(prev) if prev is not None and np.isfinite(prev) else float("nan")
    if prev is None or not np.isfinite(prev):
        return float(value)
    return float(alpha) * float(value) + (1.0 - float(alpha)) * float(prev)


def _entry_prob_modes(entry_long_raw: np.ndarray, entry_short_raw: np.ndarray) -> dict[str, dict[str, np.ndarray]]:
    return {
        "raw": {"long": entry_long_raw, "short": entry_short_raw},
        "ema_0p20": {
            "long": _ema_series(entry_long_raw, 0.20),
            "short": _ema_series(entry_short_raw, 0.20),
        },
        "ema_0p35": {
            "long": _ema_series(entry_long_raw, 0.35),
            "short": _ema_series(entry_short_raw, 0.35),
        },
    }


def _thresholds_for_regime(base_thresholds: dict[str, float], regime: RegimeConfig) -> dict[str, float]:
    thresholds = {key: float(value) for key, value in base_thresholds.items()}
    if regime.entry_threshold is not None:
        thresholds["enter_long"] = float(regime.entry_threshold)
        thresholds["enter_short"] = float(regime.entry_threshold)
    if regime.exit_threshold is not None:
        thresholds["exit_long"] = float(regime.exit_threshold)
        thresholds["exit_short"] = float(regime.exit_threshold)
    return thresholds


def _validity_flags_threshold(
    *,
    p_enter_long: float,
    p_enter_short: float,
    thr_enter_long: float,
    thr_enter_short: float,
    opposite_dominance_delta: float,
) -> tuple[bool, bool]:
    long_ready = np.isfinite(p_enter_long) and p_enter_long >= thr_enter_long
    short_ready = np.isfinite(p_enter_short) and p_enter_short >= thr_enter_short
    long_margin = (p_enter_long - thr_enter_long) if long_ready else -np.inf
    short_margin = (p_enter_short - thr_enter_short) if short_ready else -np.inf
    long_invalidated = bool(short_ready and short_margin > long_margin + float(opposite_dominance_delta))
    short_invalidated = bool(long_ready and long_margin > short_margin + float(opposite_dominance_delta))
    return bool(long_ready and not long_invalidated), bool(short_ready and not short_invalidated)


def _side_score(
    *,
    side: str,
    enter_long: float,
    enter_short: float,
    exit_long: float,
    exit_short: float,
    mode: str,
) -> float:
    if side == "long":
        enter_side = enter_long
        enter_opp = enter_short
        exit_side = exit_long
    else:
        enter_side = enter_short
        enter_opp = enter_long
        exit_side = exit_short

    if mode == "entry_diff":
        return float(enter_side - enter_opp) if np.isfinite(enter_side) and np.isfinite(enter_opp) else float("nan")
    if mode == "enter_minus_exit":
        return float(enter_side - exit_side) if np.isfinite(enter_side) and np.isfinite(exit_side) else float("nan")
    if mode == "net_edge":
        candidates = [value for value in (enter_opp, exit_side) if np.isfinite(value)]
        if not np.isfinite(enter_side) or not candidates:
            return float("nan") if not np.isfinite(enter_side) else float(enter_side)
        return float(enter_side - max(candidates))
    raise ValueError(f"Unknown score mode: {mode}")


def _entry_ready_from_score(
    *,
    side: str,
    regime: RegimeConfig,
    thresholds: dict[str, float],
    enter_long_sig: float,
    enter_short_sig: float,
    exit_long_sig: float,
    exit_short_sig: float,
) -> bool:
    enter_side = enter_long_sig if side == "long" else enter_short_sig
    enter_thr = thresholds["enter_long"] if side == "long" else thresholds["enter_short"]
    score = _side_score(
        side=side,
        enter_long=enter_long_sig,
        enter_short=enter_short_sig,
        exit_long=exit_long_sig,
        exit_short=exit_short_sig,
        mode=regime.score_mode,
    )
    if not np.isfinite(score) or score < float(regime.entry_score_threshold):
        return False
    if not regime.require_entry_threshold:
        return True
    return bool(np.isfinite(enter_side) and enter_side >= float(enter_thr))


def _profit_protect_hit(
    *,
    side: str,
    state: SideRuntime,
    bar_open: float,
    bar_high: float,
    bar_low: float,
    regime: RegimeConfig,
) -> bool:
    arm_atr = regime.profit_protect_arm_atr
    if arm_atr is None or not np.isfinite(arm_atr) or float(arm_atr) <= 0.0:
        return False
    if not (np.isfinite(state.entry_price) and np.isfinite(state.entry_atr) and state.entry_atr > 0.0):
        return False
    if side == "long" and np.isfinite(bar_high):
        state.favorable_anchor = max(float(state.favorable_anchor), float(bar_high)) if np.isfinite(state.favorable_anchor) else float(bar_high)
    elif side == "short" and np.isfinite(bar_low):
        state.favorable_anchor = min(float(state.favorable_anchor), float(bar_low)) if np.isfinite(state.favorable_anchor) else float(bar_low)

    if side == "long":
        if not np.isfinite(state.favorable_anchor):
            return False
        mfe_atr = (float(state.favorable_anchor) - float(state.entry_price)) / float(state.entry_atr)
        realized_run_atr = (
            (float(bar_open) - float(state.entry_price)) / float(state.entry_atr)
            if np.isfinite(bar_open)
            else float("nan")
        )
        giveback_limit = regime.profit_protect_giveback_long
    else:
        if not np.isfinite(state.favorable_anchor):
            return False
        mfe_atr = (float(state.entry_price) - float(state.favorable_anchor)) / float(state.entry_atr)
        realized_run_atr = (
            (float(state.entry_price) - float(bar_open)) / float(state.entry_atr)
            if np.isfinite(bar_open)
            else float("nan")
        )
        giveback_limit = regime.profit_protect_giveback_short

    giveback_atr = (
        float(mfe_atr) - float(realized_run_atr)
        if np.isfinite(mfe_atr) and np.isfinite(realized_run_atr)
        else float("nan")
    )
    return bool(
        np.isfinite(mfe_atr)
        and mfe_atr >= float(arm_atr)
        and giveback_limit is not None
        and np.isfinite(giveback_limit)
        and np.isfinite(giveback_atr)
        and giveback_atr >= float(giveback_limit)
    )


def _build_regimes() -> list[RegimeConfig]:
    return [
        RegimeConfig(name="baseline_next_open_hybrid"),
        RegimeConfig(
            name="fast_exit_next_open_hybrid",
            soft_exit_confirm_bars=1,
            urgent_exit_prob=0.80,
            urgent_exit_delta=0.20,
        ),
        RegimeConfig(
            name="fast_exit_next_open_hybrid_cd6",
            soft_exit_confirm_bars=1,
            urgent_exit_prob=0.80,
            urgent_exit_delta=0.20,
            same_side_reentry_cooldown_bars=6,
        ),
        RegimeConfig(
            name="ema20_next_open_hybrid",
            entry_prob_mode="ema_0p20",
            exit_prob_mode="ema_0p20",
        ),
        RegimeConfig(
            name="threshold_exit_next_open_55",
            exit_rule="threshold",
            exit_threshold=0.55,
        ),
        RegimeConfig(
            name="diff_entry_fast_exit_next_open_cd6",
            entry_rule="score",
            score_mode="entry_diff",
            entry_score_threshold=0.15,
            soft_exit_confirm_bars=1,
            urgent_exit_prob=0.80,
            urgent_exit_delta=0.20,
            same_side_reentry_cooldown_bars=6,
        ),
        RegimeConfig(
            name="net_edge_entry_exit_next_open",
            entry_rule="score",
            exit_rule="score",
            score_mode="net_edge",
            entry_score_threshold=0.05,
            exit_score_threshold=0.00,
        ),
        RegimeConfig(
            name="ema20_net_edge_entry_exit_next_open",
            entry_rule="score",
            exit_rule="score",
            entry_prob_mode="ema_0p20",
            exit_prob_mode="ema_0p20",
            score_mode="net_edge",
            entry_score_threshold=0.05,
            exit_score_threshold=0.00,
        ),
        RegimeConfig(
            name="profit_protect_next_open",
            profit_protect_arm_atr=2.0,
            profit_protect_giveback_long=0.75,
            profit_protect_giveback_short=1.0,
        ),
        RegimeConfig(
            name="hybrid_entry55_next_open",
            entry_threshold=0.55,
        ),
    ]


def _simulate_regime(
    *,
    regime: RegimeConfig,
    meta_df: pd.DataFrame,
    one_min: pd.DataFrame,
    symbol: str,
    entry_prob_modes: dict[str, dict[str, np.ndarray]],
    base_thresholds: dict[str, float],
    long_agent: LiveMetaXGBAgent,
    short_agent: LiveMetaXGBAgent,
) -> pd.DataFrame:
    long_agent._reset_trade_state()
    short_agent._reset_trade_state()

    thresholds = _thresholds_for_regime(base_thresholds, regime)
    long_state = SideRuntime()
    short_state = SideRuntime()
    states = {"long": long_state, "short": short_state}
    sign_map = {"long": 1, "short": -1}
    exit_alpha = 0.20 if regime.exit_prob_mode == "ema_0p20" else (0.35 if regime.exit_prob_mode == "ema_0p35" else None)

    one_ts = one_min["timestamp"].to_numpy(dtype="datetime64[ns]")
    meta_index = meta_df.index.to_list()
    events: list[dict[str, object]] = []

    for idx, (_, row) in enumerate(meta_df.iterrows()):
        ts = pd.Timestamp(row.name).tz_convert("UTC")
        next_ts = pd.Timestamp(meta_index[idx + 1]).tz_convert("UTC") if idx + 1 < len(meta_index) else (ts + pd.Timedelta(minutes=10))
        decision_ts = ts + pd.Timedelta(minutes=10)
        next_decision_ts = next_ts + pd.Timedelta(minutes=10)
        start_pos = int(np.searchsorted(one_ts, decision_ts.to_datetime64(), side="left"))
        end_pos = int(np.searchsorted(one_ts, next_decision_ts.to_datetime64(), side="left"))
        interval = one_min.iloc[start_pos:end_pos]

        p_enter_long_raw = float(entry_prob_modes["raw"]["long"][idx])
        p_enter_short_raw = float(entry_prob_modes["raw"]["short"][idx])
        p_enter_long_sig = float(entry_prob_modes[regime.entry_prob_mode]["long"][idx])
        p_enter_short_sig = float(entry_prob_modes[regime.entry_prob_mode]["short"][idx])

        work_row = row.copy()
        work_row["p_enter_long_oof"] = p_enter_long_raw
        work_row["p_enter_short_oof"] = p_enter_short_raw

        p_exit_long_raw = _score_exit(long_agent, work_row, side="long") if long_state.active else float("nan")
        p_exit_short_raw = _score_exit(short_agent, work_row, side="short") if short_state.active else float("nan")

        if exit_alpha is not None:
            long_state.exit_ema = _ema_step(long_state.exit_ema, p_exit_long_raw, exit_alpha) if long_state.active else None
            short_state.exit_ema = _ema_step(short_state.exit_ema, p_exit_short_raw, exit_alpha) if short_state.active else None
            p_exit_long_sig = float(long_state.exit_ema) if long_state.active and long_state.exit_ema is not None else float("nan")
            p_exit_short_sig = float(short_state.exit_ema) if short_state.active and short_state.exit_ema is not None else float("nan")
        else:
            p_exit_long_sig = p_exit_long_raw
            p_exit_short_sig = p_exit_short_raw
            if not long_state.active:
                long_state.exit_ema = None
            if not short_state.active:
                short_state.exit_ema = None

        if regime.entry_rule == "threshold":
            long_valid_signal, short_valid_signal = _validity_flags_threshold(
                p_enter_long=p_enter_long_sig,
                p_enter_short=p_enter_short_sig,
                thr_enter_long=float(thresholds["enter_long"]),
                thr_enter_short=float(thresholds["enter_short"]),
                opposite_dominance_delta=float(regime.opposite_dominance_delta),
            )
        else:
            long_valid_signal = _entry_ready_from_score(
                side="long",
                regime=regime,
                thresholds=thresholds,
                enter_long_sig=p_enter_long_sig,
                enter_short_sig=p_enter_short_sig,
                exit_long_sig=p_exit_long_sig,
                exit_short_sig=p_exit_short_sig,
            )
            short_valid_signal = _entry_ready_from_score(
                side="short",
                regime=regime,
                thresholds=thresholds,
                enter_long_sig=p_enter_long_sig,
                enter_short_sig=p_enter_short_sig,
                exit_long_sig=p_exit_long_sig,
                exit_short_sig=p_exit_short_sig,
            )

        if not long_state.active:
            if long_valid_signal and idx > long_state.cooldown_until_idx:
                long_state.intent_active = True
                long_state.signal_row = row.copy()
                long_state.entry_ref_price = float(row.get("open", np.nan))
            else:
                long_state.intent_active = False
                long_state.signal_row = None
                long_state.entry_ref_price = float("nan")

        if not short_state.active:
            if short_valid_signal and idx > short_state.cooldown_until_idx:
                short_state.intent_active = True
                short_state.signal_row = row.copy()
                short_state.entry_ref_price = float(row.get("open", np.nan))
            else:
                short_state.intent_active = False
                short_state.signal_row = None
                short_state.entry_ref_price = float("nan")

        exit_inputs = {
            "long": {
                "enter_sig": p_enter_long_sig,
                "exit_sig": p_exit_long_sig,
                "enter_thr": float(thresholds["enter_long"]),
                "exit_thr": float(thresholds["exit_long"]),
                "opp_enter_sig": p_enter_short_sig,
                "opp_exit_sig": p_exit_short_sig,
            },
            "short": {
                "enter_sig": p_enter_short_sig,
                "exit_sig": p_exit_short_sig,
                "enter_thr": float(thresholds["enter_short"]),
                "exit_thr": float(thresholds["exit_short"]),
                "opp_enter_sig": p_enter_long_sig,
                "opp_exit_sig": p_exit_long_sig,
            },
        }

        for side, agent in (("long", long_agent), ("short", short_agent)):
            state = states[side]
            if not state.active:
                state.soft_confirm = 0
                continue

            info = exit_inputs[side]
            bars_held = int(agent._state.bars_since_entry)
            hold_ready = bool(bars_held >= int(regime.min_hold_bars))
            max_hold_hit = bool(regime.max_hold_bars is not None and bars_held >= int(regime.max_hold_bars))
            do_exit = False
            exit_reason: str | None = None

            if regime.exit_rule == "hybrid":
                soft_exit_condition = bool(
                    hold_ready
                    and np.isfinite(info["enter_sig"])
                    and info["enter_sig"] < info["enter_thr"]
                )
                state.soft_confirm = state.soft_confirm + 1 if soft_exit_condition else 0
                urgent_prob_hit = bool(
                    hold_ready
                    and np.isfinite(info["exit_sig"])
                    and info["exit_sig"] >= float(regime.urgent_exit_prob)
                )
                urgent_delta_hit = bool(
                    hold_ready
                    and np.isfinite(info["exit_sig"])
                    and np.isfinite(info["enter_sig"])
                    and (info["exit_sig"] - info["enter_sig"]) >= float(regime.urgent_exit_delta)
                )
                if max_hold_hit:
                    do_exit = True
                    exit_reason = "max_hold"
                elif urgent_prob_hit:
                    do_exit = True
                    exit_reason = "urgent_exit_prob"
                elif urgent_delta_hit:
                    do_exit = True
                    exit_reason = "urgent_exit_delta"
                elif state.soft_confirm >= int(regime.soft_exit_confirm_bars):
                    do_exit = True
                    exit_reason = "soft_exit"
            elif regime.exit_rule == "threshold":
                entry_still_supports = bool(
                    np.isfinite(info["enter_sig"])
                    and info["enter_sig"] >= info["enter_thr"]
                    and (
                        not np.isfinite(info["exit_sig"])
                        or (info["exit_sig"] - info["enter_sig"]) < float(regime.exit_entry_delta)
                    )
                )
                exit_threshold_hit = bool(
                    hold_ready
                    and np.isfinite(info["exit_sig"])
                    and info["exit_sig"] >= info["exit_thr"]
                )
                do_exit = bool(max_hold_hit or (exit_threshold_hit and not entry_still_supports))
                exit_reason = "max_hold" if max_hold_hit else ("threshold_exit" if do_exit else None)
            elif regime.exit_rule == "score":
                score = _side_score(
                    side=side,
                    enter_long=p_enter_long_sig,
                    enter_short=p_enter_short_sig,
                    exit_long=p_exit_long_sig,
                    exit_short=p_exit_short_sig,
                    mode=regime.score_mode,
                )
                score_soft_exit = bool(
                    hold_ready
                    and np.isfinite(score)
                    and score <= float(regime.exit_score_threshold)
                )
                state.soft_confirm = state.soft_confirm + 1 if score_soft_exit else 0
                if max_hold_hit:
                    do_exit = True
                    exit_reason = "max_hold"
                elif state.soft_confirm >= int(regime.soft_exit_confirm_bars):
                    do_exit = True
                    exit_reason = "score_exit"
            else:
                raise ValueError(f"Unknown exit rule: {regime.exit_rule}")

            action = 0 if do_exit else sign_map[side]
            agent._advance_state(action=action, row=work_row)
            if do_exit:
                state.pending_exit = True
                state.pending_exit_reason = exit_reason

        entered_long_this_interval = False
        entered_short_this_interval = False
        for bar in interval.itertuples(index=False):
            bar_ts = pd.Timestamp(bar.timestamp)
            bar_open = float(getattr(bar, "open", np.nan))
            bar_high = float(getattr(bar, "high", np.nan))
            bar_low = float(getattr(bar, "low", np.nan))

            for side, state, agent in (
                ("long", long_state, long_agent),
                ("short", short_state, short_agent),
            ):
                if state.pending_exit and state.active and np.isfinite(bar_open):
                    state.active = False
                    state.intent_active = False
                    state.signal_row = None
                    state.pending_exit = False
                    state.soft_confirm = 0
                    state.cooldown_until_idx = idx + max(0, int(regime.same_side_reentry_cooldown_bars))
                    exit_reason = state.pending_exit_reason or "exit"
                    state.pending_exit_reason = None
                    state.entry_price = float("nan")
                    state.entry_atr = float("nan")
                    state.favorable_anchor = float("nan")
                    state.entry_ref_price = float("nan")
                    state.exit_ema = None
                    events.append(
                        {
                            "regime": regime.name,
                            "timestamp": bar_ts,
                            "symbol": symbol,
                            "event": f"exit_{side}",
                            "price": bar_open,
                            "reason": exit_reason,
                        }
                    )

            if (
                long_state.intent_active
                and not long_state.active
                and not entered_long_this_interval
                and long_state.signal_row is not None
                and np.isfinite(bar_open)
            ):
                fill_price = float(bar_open)
                long_state.active = True
                long_state.intent_active = False
                entered_long_this_interval = True
                long_agent._set_trade_entry(position=1, row=long_state.signal_row, entry_price=fill_price)
                long_state.entry_price = fill_price
                long_state.entry_atr = float(long_state.signal_row.get("atr", np.nan))
                long_state.favorable_anchor = fill_price
                long_state.entry_ref_price = fill_price
                long_state.exit_ema = None
                events.append(
                    {
                        "regime": regime.name,
                        "timestamp": bar_ts,
                        "symbol": symbol,
                        "event": "enter_long",
                        "price": fill_price,
                        "reason": regime.entry_fill,
                    }
                )

            if (
                short_state.intent_active
                and not short_state.active
                and not entered_short_this_interval
                and short_state.signal_row is not None
                and np.isfinite(bar_open)
            ):
                fill_price = float(bar_open)
                short_state.active = True
                short_state.intent_active = False
                entered_short_this_interval = True
                short_agent._set_trade_entry(position=-1, row=short_state.signal_row, entry_price=fill_price)
                short_state.entry_price = fill_price
                short_state.entry_atr = float(short_state.signal_row.get("atr", np.nan))
                short_state.favorable_anchor = fill_price
                short_state.entry_ref_price = fill_price
                short_state.exit_ema = None
                events.append(
                    {
                        "regime": regime.name,
                        "timestamp": bar_ts,
                        "symbol": symbol,
                        "event": "enter_short",
                        "price": fill_price,
                        "reason": regime.entry_fill,
                    }
                )

            if _profit_protect_hit(
                side="long",
                state=long_state,
                bar_open=bar_open,
                bar_high=bar_high,
                bar_low=bar_low,
                regime=regime,
            ):
                if long_state.active and np.isfinite(bar_open):
                    long_state.clear_trade_state()
                    long_state.cooldown_until_idx = idx + max(0, int(regime.same_side_reentry_cooldown_bars))
                    long_agent._reset_trade_state()
                    events.append(
                        {
                            "regime": regime.name,
                            "timestamp": bar_ts,
                            "symbol": symbol,
                            "event": "exit_long",
                            "price": bar_open,
                            "reason": "profit_protect",
                        }
                    )

            if _profit_protect_hit(
                side="short",
                state=short_state,
                bar_open=bar_open,
                bar_high=bar_high,
                bar_low=bar_low,
                regime=regime,
            ):
                if short_state.active and np.isfinite(bar_open):
                    short_state.clear_trade_state()
                    short_state.cooldown_until_idx = idx + max(0, int(regime.same_side_reentry_cooldown_bars))
                    short_agent._reset_trade_state()
                    events.append(
                        {
                            "regime": regime.name,
                            "timestamp": bar_ts,
                            "symbol": symbol,
                            "event": "exit_short",
                            "price": bar_open,
                            "reason": "profit_protect",
                        }
                    )

    if not one_min.empty:
        last_bar = one_min.iloc[-1]
        last_ts = pd.Timestamp(last_bar["timestamp"])
        last_price = float(last_bar.get("close", np.nan))
        if long_state.active and np.isfinite(last_price):
            long_state.clear_trade_state()
            long_agent._reset_trade_state()
            events.append(
                {
                    "regime": regime.name,
                    "timestamp": last_ts,
                    "symbol": symbol,
                    "event": "exit_long",
                    "price": last_price,
                    "reason": "forced_end",
                }
            )
        if short_state.active and np.isfinite(last_price):
            short_state.clear_trade_state()
            short_agent._reset_trade_state()
            events.append(
                {
                    "regime": regime.name,
                    "timestamp": last_ts,
                    "symbol": symbol,
                    "event": "exit_short",
                    "price": last_price,
                    "reason": "forced_end",
                }
            )

    events_df = pd.DataFrame(events)
    if events_df.empty:
        return events_df
    events_df["timestamp"] = pd.to_datetime(events_df["timestamp"], utc=True, errors="coerce")
    return events_df.sort_values(["timestamp", "event"]).reset_index(drop=True)


def _build_trades(events: pd.DataFrame, meta_df: pd.DataFrame) -> pd.DataFrame:
    if "timestamp" in meta_df.columns:
        meta_lookup = meta_df[["timestamp", "atr"]].copy().reset_index(drop=True)
    else:
        meta_lookup = meta_df.reset_index(drop=False).rename(columns={meta_df.index.name or "index": "timestamp"})
        meta_lookup = meta_lookup[["timestamp", "atr"]].copy()
    meta_lookup["timestamp"] = pd.to_datetime(meta_lookup["timestamp"], utc=True, errors="coerce")
    meta_lookup = meta_lookup.sort_values("timestamp").rename(columns={"atr": "entry_atr"})
    meta_lookup = meta_lookup[["timestamp", "entry_atr"]].copy()

    trades: list[dict[str, object]] = []
    for regime, regime_events in events.groupby("regime", sort=False):
        open_trade: dict[str, dict[str, object] | None] = {"long": None, "short": None}
        for row in regime_events.sort_values("timestamp").itertuples(index=False):
            event = str(row.event)
            side = "long" if event.endswith("long") else "short"
            ts = pd.Timestamp(row.timestamp)
            price = float(row.price)
            if event.startswith("enter_"):
                open_trade[side] = {
                    "regime": regime,
                    "side": side,
                    "symbol": row.symbol,
                    "entry_timestamp": ts,
                    "entry_price": price,
                    "entry_reason": getattr(row, "reason", None),
                }
            elif event.startswith("exit_") and open_trade[side] is not None:
                rec = dict(open_trade[side] or {})
                rec["exit_timestamp"] = ts
                rec["exit_price"] = price
                rec["exit_reason"] = getattr(row, "reason", None)
                trades.append(rec)
                open_trade[side] = None

    trades_df = pd.DataFrame(trades)
    if trades_df.empty:
        return trades_df

    trades_df = pd.merge_asof(
        trades_df.sort_values("entry_timestamp"),
        meta_lookup.sort_values("timestamp"),
        left_on="entry_timestamp",
        right_on="timestamp",
        direction="backward",
        tolerance=pd.Timedelta("10min"),
    ).drop(columns=["timestamp"])

    trades_df["hold_minutes"] = (
        (pd.to_datetime(trades_df["exit_timestamp"], utc=True) - pd.to_datetime(trades_df["entry_timestamp"], utc=True))
        .dt.total_seconds()
        .div(60.0)
    )
    trades_df["realized_price_move"] = np.where(
        trades_df["side"].eq("long"),
        pd.to_numeric(trades_df["exit_price"], errors="coerce") - pd.to_numeric(trades_df["entry_price"], errors="coerce"),
        pd.to_numeric(trades_df["entry_price"], errors="coerce") - pd.to_numeric(trades_df["exit_price"], errors="coerce"),
    )
    trades_df["return_frac"] = np.where(
        trades_df["side"].eq("long"),
        pd.to_numeric(trades_df["exit_price"], errors="coerce") / pd.to_numeric(trades_df["entry_price"], errors="coerce") - 1.0,
        (
            pd.to_numeric(trades_df["entry_price"], errors="coerce")
            - pd.to_numeric(trades_df["exit_price"], errors="coerce")
        )
        / pd.to_numeric(trades_df["entry_price"], errors="coerce"),
    )
    trades_df["realized_atr_capture"] = trades_df["realized_price_move"] / pd.to_numeric(trades_df["entry_atr"], errors="coerce")
    trades_df["winner"] = trades_df["realized_price_move"] > 0.0
    return trades_df


def _attach_swing_metrics(trades: pd.DataFrame, one_min: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return trades

    ts_arr = one_min["timestamp"].to_numpy(dtype="datetime64[ns]")
    high_arr = pd.to_numeric(one_min["high"], errors="coerce").to_numpy(dtype=float)
    low_arr = pd.to_numeric(one_min["low"], errors="coerce").to_numpy(dtype=float)

    mfe_price_move: list[float] = []
    mae_price_move: list[float] = []
    capture_ratio: list[float] = []
    giveback_price_move: list[float] = []
    mfe_atr: list[float] = []
    giveback_atr: list[float] = []

    for row in trades.itertuples(index=False):
        entry_ts = pd.Timestamp(row.entry_timestamp).to_datetime64()
        exit_ts = pd.Timestamp(row.exit_timestamp).to_datetime64()
        entry_price = float(row.entry_price)
        exit_price = float(row.exit_price)
        entry_atr = float(row.entry_atr) if np.isfinite(row.entry_atr) else float("nan")

        start_idx = int(np.searchsorted(ts_arr, entry_ts, side="left"))
        end_idx = int(np.searchsorted(ts_arr, exit_ts, side="left"))
        window_high = high_arr[start_idx:end_idx]
        window_low = low_arr[start_idx:end_idx]

        if row.side == "long":
            best_price = np.nanmax(window_high) if window_high.size and np.isfinite(window_high).any() else float("nan")
            worst_price = np.nanmin(window_low) if window_low.size and np.isfinite(window_low).any() else float("nan")
            if not np.isfinite(best_price):
                best_price = exit_price
            else:
                best_price = max(float(best_price), exit_price)
            if not np.isfinite(worst_price):
                worst_price = entry_price
            mfe_move = float(best_price - entry_price)
            mae_move = float(entry_price - worst_price)
        else:
            best_price = np.nanmin(window_low) if window_low.size and np.isfinite(window_low).any() else float("nan")
            worst_price = np.nanmax(window_high) if window_high.size and np.isfinite(window_high).any() else float("nan")
            if not np.isfinite(best_price):
                best_price = exit_price
            else:
                best_price = min(float(best_price), exit_price)
            if not np.isfinite(worst_price):
                worst_price = entry_price
            mfe_move = float(entry_price - best_price)
            mae_move = float(worst_price - entry_price)

        realized_move = float(row.realized_price_move)
        giveback_move = float(mfe_move - realized_move) if np.isfinite(mfe_move) and np.isfinite(realized_move) else float("nan")
        ratio = (
            float(realized_move / mfe_move)
            if np.isfinite(mfe_move) and mfe_move > 0.0 and np.isfinite(realized_move)
            else float("nan")
        )
        mfe_price_move.append(mfe_move)
        mae_price_move.append(mae_move)
        capture_ratio.append(ratio)
        giveback_price_move.append(giveback_move)
        mfe_atr.append(mfe_move / entry_atr if np.isfinite(entry_atr) and entry_atr > 0.0 else float("nan"))
        giveback_atr.append(giveback_move / entry_atr if np.isfinite(entry_atr) and entry_atr > 0.0 else float("nan"))

    out = trades.copy()
    out["mfe_price_move"] = mfe_price_move
    out["mae_price_move"] = mae_price_move
    out["capture_ratio"] = capture_ratio
    out["giveback_price_move"] = giveback_price_move
    out["mfe_atr"] = mfe_atr
    out["giveback_atr"] = giveback_atr
    return out


def _equity_metrics(events: pd.DataFrame, one_min: pd.DataFrame) -> dict[str, float]:
    ev = events.sort_values("timestamp").reset_index(drop=True).copy()
    one = one_min.sort_values("timestamp").reset_index(drop=True).copy()
    long_on = False
    short_on = False
    long_eq = 1.0
    short_eq = 1.0
    net_1x = 1.0
    buy_hold = 1.0
    ev_idx = 0

    for i in range(len(one) - 1):
        ts = pd.Timestamp(one.loc[i, "timestamp"])
        while ev_idx < len(ev) and pd.Timestamp(ev.loc[ev_idx, "timestamp"]) <= ts:
            event = str(ev.loc[ev_idx, "event"])
            if event == "enter_long":
                long_on = True
            elif event == "exit_long":
                long_on = False
            elif event == "enter_short":
                short_on = True
            elif event == "exit_short":
                short_on = False
            ev_idx += 1

        open_i = float(one.loc[i, "open"])
        open_n = float(one.loc[i + 1, "open"])
        if not (np.isfinite(open_i) and np.isfinite(open_n) and open_i > 0.0 and open_n > 0.0):
            continue
        ret = open_n / open_i - 1.0
        buy_hold *= 1.0 + ret
        if long_on:
            long_eq *= 1.0 + ret
        if short_on:
            short_eq *= 1.0 - ret
        if long_on and not short_on:
            net_1x *= 1.0 + ret
        elif short_on and not long_on:
            net_1x *= 1.0 - ret

    return {
        "long_equity_end": float(long_eq),
        "short_equity_end": float(short_eq),
        "combined_full_gross_end": float(long_eq + short_eq - 1.0),
        "net_1x_end": float(net_1x),
        "buy_hold_end": float(buy_hold),
    }


def _trade_metrics(trades: pd.DataFrame, *, side: str | None = None) -> dict[str, float]:
    df = trades.copy()
    if side is not None:
        df = df[df["side"].eq(side)].copy()
    if df.empty:
        return {
            "closed_trades": 0.0,
            "win_rate": float("nan"),
            "avg_price_move": float("nan"),
            "median_price_move": float("nan"),
            "avg_return_frac": float("nan"),
            "avg_atr_capture": float("nan"),
            "median_atr_capture": float("nan"),
            "avg_mfe_atr": float("nan"),
            "avg_capture_ratio": float("nan"),
            "median_capture_ratio": float("nan"),
            "pct_capture_ge_0_50": float("nan"),
            "avg_giveback_atr": float("nan"),
            "avg_hold_minutes": float("nan"),
        }

    capture = pd.to_numeric(df["capture_ratio"], errors="coerce")
    return {
        "closed_trades": float(len(df)),
        "win_rate": float(pd.to_numeric(df["winner"], errors="coerce").mean()),
        "avg_price_move": float(pd.to_numeric(df["realized_price_move"], errors="coerce").mean()),
        "median_price_move": float(pd.to_numeric(df["realized_price_move"], errors="coerce").median()),
        "avg_return_frac": float(pd.to_numeric(df["return_frac"], errors="coerce").mean()),
        "avg_atr_capture": float(pd.to_numeric(df["realized_atr_capture"], errors="coerce").mean()),
        "median_atr_capture": float(pd.to_numeric(df["realized_atr_capture"], errors="coerce").median()),
        "avg_mfe_atr": float(pd.to_numeric(df["mfe_atr"], errors="coerce").mean()),
        "avg_capture_ratio": float(capture.mean()),
        "median_capture_ratio": float(capture.median()),
        "pct_capture_ge_0_50": float((capture >= 0.50).mean()),
        "avg_giveback_atr": float(pd.to_numeric(df["giveback_atr"], errors="coerce").mean()),
        "avg_hold_minutes": float(pd.to_numeric(df["hold_minutes"], errors="coerce").mean()),
    }


def _event_reason_counts(events: pd.DataFrame) -> dict[str, float]:
    exits = events[events["event"].astype(str).str.startswith("exit_")].copy()
    reasons = {
        "soft_exit",
        "urgent_exit_prob",
        "urgent_exit_delta",
        "threshold_exit",
        "score_exit",
        "max_hold",
        "profit_protect",
        "forced_end",
    }
    return {f"exit_reason_{reason}_count": float(int(exits["reason"].astype(str).eq(reason).sum())) for reason in sorted(reasons)}


def _regime_summary(
    *,
    regime: RegimeConfig,
    regime_events: pd.DataFrame,
    regime_trades: pd.DataFrame,
    one_min: pd.DataFrame,
    base_thresholds: dict[str, float],
) -> dict[str, float | str]:
    thresholds = _thresholds_for_regime(base_thresholds, regime)
    equity = _equity_metrics(regime_events, one_min)
    overall_trade = _trade_metrics(regime_trades)
    long_trade = _trade_metrics(regime_trades, side="long")
    short_trade = _trade_metrics(regime_trades, side="short")

    return {
        "regime": regime.name,
        "entry_rule": regime.entry_rule,
        "exit_rule": regime.exit_rule,
        "entry_prob_mode": regime.entry_prob_mode,
        "exit_prob_mode": regime.exit_prob_mode,
        "score_mode": regime.score_mode,
        "entry_threshold": float(thresholds["enter_long"]),
        "exit_threshold": float(thresholds["exit_long"]),
        "entry_score_threshold": float(regime.entry_score_threshold),
        "exit_score_threshold": float(regime.exit_score_threshold),
        "min_hold_bars": float(regime.min_hold_bars),
        "soft_exit_confirm_bars": float(regime.soft_exit_confirm_bars),
        "urgent_exit_prob": float(regime.urgent_exit_prob),
        "urgent_exit_delta": float(regime.urgent_exit_delta),
        "max_hold_bars": float(regime.max_hold_bars) if regime.max_hold_bars is not None else np.nan,
        "same_side_reentry_cooldown_bars": float(regime.same_side_reentry_cooldown_bars),
        "profit_protect_arm_atr": float(regime.profit_protect_arm_atr) if regime.profit_protect_arm_atr is not None else np.nan,
        "profit_protect_giveback_long": float(regime.profit_protect_giveback_long) if regime.profit_protect_giveback_long is not None else np.nan,
        "profit_protect_giveback_short": float(regime.profit_protect_giveback_short) if regime.profit_protect_giveback_short is not None else np.nan,
        "long_entries": float(int(regime_events["event"].eq("enter_long").sum())),
        "long_exits": float(int(regime_events["event"].eq("exit_long").sum())),
        "short_entries": float(int(regime_events["event"].eq("enter_short").sum())),
        "short_exits": float(int(regime_events["event"].eq("exit_short").sum())),
        **equity,
        **{f"overall_{key}": value for key, value in overall_trade.items()},
        **{f"long_{key}": value for key, value in long_trade.items()},
        **{f"short_{key}": value for key, value in short_trade.items()},
        **_event_reason_counts(regime_events),
    }


def main() -> None:
    args = _parse_args()
    requested_start, requested_end = _normalize_bounds(args.start, args.end)
    start, end = _resolve_overlap_bounds(
        Path(args.meta_matrix),
        Path(args.one_min_data),
        user_start=requested_start,
        user_end=requested_end,
    )

    meta_df = _load_meta_matrix(Path(args.meta_matrix), start=start, end=end, tz=args.tz)
    one_min = _load_one_min(Path(args.one_min_data), symbol=args.symbol, start=start, end=end)

    base_agent = LiveMetaXGBAgent(model_root=Path(args.model_root), precomputed_base_frame=meta_df)
    replay_short_agent = LiveMetaXGBAgent(model_root=Path(args.model_root), precomputed_base_frame=meta_df)
    entry_long_raw = base_agent._entry_long.predict_frame(meta_df)
    entry_short_raw = base_agent._entry_short.predict_frame(meta_df)
    base_thresholds = base_agent.last_thresholds() or {
        "enter_long": np.nan,
        "enter_short": np.nan,
        "exit_long": np.nan,
        "exit_short": np.nan,
    }
    entry_prob_modes = _entry_prob_modes(entry_long_raw, entry_short_raw)
    regimes = _build_regimes()

    all_events: list[pd.DataFrame] = []
    all_trades: list[pd.DataFrame] = []
    summary_rows: list[dict[str, float | str]] = []

    for regime in regimes:
        regime_events = _simulate_regime(
            regime=regime,
            meta_df=meta_df,
            one_min=one_min,
            symbol=args.symbol,
            entry_prob_modes=entry_prob_modes,
            base_thresholds=base_thresholds,
            long_agent=base_agent,
            short_agent=replay_short_agent,
        )
        if regime_events.empty:
            continue
        regime_trades = _attach_swing_metrics(_build_trades(regime_events, meta_df), one_min)
        all_events.append(regime_events)
        if not regime_trades.empty:
            all_trades.append(regime_trades)
        summary_rows.append(
            _regime_summary(
                regime=regime,
                regime_events=regime_events,
                regime_trades=regime_trades,
                one_min=one_min,
                base_thresholds=base_thresholds,
            )
        )

    if not summary_rows:
        raise SystemExit("No regimes produced any events.")

    summary = pd.DataFrame(summary_rows).sort_values(
        ["combined_full_gross_end", "net_1x_end", "overall_avg_capture_ratio"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    summary["rank_combined_full_gross"] = summary["combined_full_gross_end"].rank(method="min", ascending=False)
    summary["rank_net_1x"] = summary["net_1x_end"].rank(method="min", ascending=False)
    summary["rank_capture_ratio"] = summary["overall_avg_capture_ratio"].rank(method="min", ascending=False)

    events = pd.concat(all_events, ignore_index=True).sort_values(["regime", "timestamp", "event"]).reset_index(drop=True)
    trades = (
        pd.concat(all_trades, ignore_index=True).sort_values(["regime", "entry_timestamp", "side"]).reset_index(drop=True)
        if all_trades
        else pd.DataFrame()
    )

    summary_out = Path(args.summary_out)
    trades_out = Path(args.trades_out)
    events_out = Path(args.events_out)
    for path in (summary_out, trades_out, events_out):
        path.parent.mkdir(parents=True, exist_ok=True)

    summary.to_csv(summary_out, index=False)
    if not trades.empty:
        trades.to_csv(trades_out, index=False)
    events.to_csv(events_out, index=False)

    top_cols = [
        "regime",
        "combined_full_gross_end",
        "net_1x_end",
        "overall_closed_trades",
        "overall_win_rate",
        "overall_avg_atr_capture",
        "overall_avg_capture_ratio",
        "overall_avg_giveback_atr",
        "exit_reason_soft_exit_count",
        "exit_reason_urgent_exit_prob_count",
        "exit_reason_urgent_exit_delta_count",
        "exit_reason_threshold_exit_count",
        "exit_reason_score_exit_count",
        "exit_reason_max_hold_count",
        "exit_reason_profit_protect_count",
    ]
    print("Top regimes by combined_full_gross_end:\n")
    print(summary[top_cols].head(8).to_string(index=False))
    print("\nTop regimes by overall_avg_capture_ratio:\n")
    print(
        summary.sort_values(
            ["overall_avg_capture_ratio", "combined_full_gross_end"],
            ascending=[False, False],
        )[top_cols].head(8).to_string(index=False)
    )
    print(f"\nsummary_csv={summary_out}")
    print(f"trades_csv={trades_out}")
    print(f"events_csv={events_out}")


if __name__ == "__main__":
    main()
