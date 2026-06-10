from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from scripts.sweep_phase4_spy_proxy_adaptive_exits import (  # noqa: E402
    _build_trade_specs,
    _simulate_strategy_events,
    _summarize,
)
from scripts.sweep_phase4_spy_proxy_profit_locks import (  # noqa: E402
    DEFAULT_ANALYSIS_DIR,
    _build_entries,
    _load_scoreboard,
    _parse_float_list,
)
from strategies.spy_intraday.Models.ga_xgboost.analyze_phase4_triggers import _load_execution_1m  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare current adaptive exit against opposite-side setup/confirmation/proba exits."
    )
    parser.add_argument("--signal-frame", default=f"{DEFAULT_ANALYSIS_DIR}/phase4_signal_frame.parquet")
    parser.add_argument("--scoreboard", default=f"{DEFAULT_ANALYSIS_DIR}/phase4_trigger_scoreboard.json")
    parser.add_argument("--split", default="test", choices=["oof", "test"])
    parser.add_argument("--variant", default=None)
    parser.add_argument("--mode", default=None)
    parser.add_argument("--long-setup-threshold", type=float, default=None)
    parser.add_argument("--short-setup-threshold", type=float, default=None)
    parser.add_argument("--cooldown-bars", type=int, default=None)
    parser.add_argument("--post-setup-max-bars", type=int, default=None)
    parser.add_argument("--execution-1m-path", default="Data/raw/spy/spy_intraday_1min.parquet")
    parser.add_argument("--proxy-mode", choices=["black_scholes", "atr"], default="black_scholes")
    parser.add_argument("--exit-hhmm", default="15:40")
    parser.add_argument("--horizon-bars", type=int, default=39)
    parser.add_argument("--bs-iv-floor", type=float, default=0.12)
    parser.add_argument("--bs-iv-ceil", type=float, default=0.90)
    parser.add_argument("--bs-iv-mult", type=float, default=1.50)
    parser.add_argument("--bs-iv-shock", type=float, default=0.20)
    parser.add_argument("--bs-min-dte-minutes", type=float, default=1.0)
    parser.add_argument("--bs-strike-round", type=float, default=1.0)
    parser.add_argument("--premium-atr-mult", type=float, default=2.5)
    parser.add_argument("--opposite-proba-thresholds", default="0.30,0.40,0.50,0.60,0.70,0.80")
    parser.add_argument(
        "--giveback-pcts",
        default="0.20,0.30,0.40,0.50",
        help=(
            "Comma-separated option-profit giveback amounts for opposite-proba-after-giveback exits. "
            "Example: 0.30 exits after giving back 30 percentage points from the trade's best option return."
        ),
    )
    parser.add_argument(
        "--soft-counts",
        default="2,3",
        help="Comma-separated consecutive 10m opposite-probability bar counts for soft opposite exits.",
    )
    parser.add_argument("--position-modes", default="hedged")
    parser.add_argument(
        "--out",
        default=f"{DEFAULT_ANALYSIS_DIR}/phase4_opposite_exit_compare_summary.csv",
    )
    parser.add_argument(
        "--events-out",
        default=f"{DEFAULT_ANALYSIS_DIR}/phase4_opposite_exit_compare_events.csv",
    )
    return parser.parse_args()


def _position_modes(raw: str) -> list[str]:
    out: list[str] = []
    for item in str(raw or "hedged").split(","):
        value = item.strip().lower()
        if not value:
            continue
        if value in {"hedge", "current", "both"}:
            value = "hedged"
        if value in {"one", "one_position", "single_position"}:
            value = "single"
        if value not in {"hedged", "single"}:
            raise ValueError(f"Unknown position mode: {item!r}")
        if value not in out:
            out.append(value)
    return out or ["hedged"]


def _parse_int_list(raw: str) -> list[int]:
    out: list[int] = []
    for item in str(raw or "").split(","):
        value = item.strip()
        if not value:
            continue
        parsed = int(value)
        if parsed not in out:
            out.append(parsed)
    return out


def _current_best_spec() -> dict[str, object]:
    return {
        "name": "current_adaptive_arm200_gb25_stale80_opp60",
        "style": "late_hybrid",
        "sl": 1.0,
        "arm": 2.0,
        "gb": 0.25,
        "stale_bars": 8,
        "progress": 1.0,
        "opp_thr": 0.60,
    }


def _event_from_trade(
    trade: dict[str, object],
    *,
    regime: str,
    exit_style: str,
    exit_pos: int,
    exit_reason: str,
) -> dict[str, object] | None:
    path_times = trade.get("path_times")
    close_path = trade.get("close_path")
    favorable_path = trade.get("favorable_path")
    adverse_path = trade.get("adverse_path")
    path_close = trade.get("path_close")
    if not (
        isinstance(path_times, pd.DatetimeIndex)
        and isinstance(close_path, np.ndarray)
        and isinstance(favorable_path, np.ndarray)
        and isinstance(adverse_path, np.ndarray)
        and isinstance(path_close, np.ndarray)
        and close_path.size > 0
    ):
        return None
    exit_pos = int(np.clip(exit_pos, 0, close_path.size - 1))
    premium = float(trade["premium"])
    if not np.isfinite(premium) or premium <= 0.0:
        return None
    fav = favorable_path[: exit_pos + 1]
    adv = adverse_path[: exit_pos + 1]
    best_pct = float(np.nanmax(fav)) if fav.size else 0.0
    worst_pct = float(np.nanmin(adv)) if adv.size else 0.0
    outcome_pct = float(close_path[exit_pos])
    return {
        "regime": regime,
        "exit_style": exit_style,
        "side": str(trade["side"]),
        "entry_time": str(trade["entry_time"]),
        "exit_time": str(path_times[exit_pos]),
        "entry_price": float(trade["entry"]),
        "exit_price": float(path_close[exit_pos]) if exit_pos < path_close.size else np.nan,
        "entry_premium_proxy": premium,
        "peak_premium_proxy": premium * (1.0 + max(0.0, best_pct)),
        "exit_premium_proxy": premium * (1.0 + outcome_pct),
        "entry_prob": float(trade.get("entry_prob", np.nan)),
        "entry_opp_prob": float(trade.get("entry_opp_prob", np.nan)),
        "trend_aligned": bool(trade.get("trend_aligned", False)),
        "trend_spread_atr": float(trade.get("trend_spread_atr", np.nan)),
        "vol_ratio": float(trade.get("vol_ratio", np.nan)),
        "proxy_mode": str(trade.get("proxy_mode", "")),
        "strike": float(trade.get("strike", np.nan)),
        "entry_iv": float(trade.get("entry_iv", np.nan)),
        "outcome_pct": outcome_pct,
        "mfe_pct": best_pct,
        "mae_pct": worst_pct,
        "exit_reason": exit_reason,
    }


def _simulate_opposite_10m_setup(
    trades: list[dict[str, object]],
    *,
    p_long: np.ndarray,
    p_short: np.ndarray,
    long_threshold: float,
    short_threshold: float,
    position_mode: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    open_until: pd.Timestamp | None = None
    blocked = 0
    ordered = sorted(trades, key=lambda trade: pd.Timestamp(trade["entry_time"]))
    for trade in ordered:
        entry_time = pd.Timestamp(trade["entry_time"])
        if position_mode == "single" and open_until is not None and entry_time < open_until:
            blocked += 1
            continue
        side = str(trade["side"])
        path_idx = trade.get("path_idx")
        if not isinstance(path_idx, np.ndarray) or path_idx.size == 0:
            continue
        threshold = short_threshold if side == "long" else long_threshold
        opp = p_short if side == "long" else p_long
        exit_pos = path_idx.size - 1
        reason = "timeout"
        seen_10m: set[int] = set()
        for pos, idx in enumerate(path_idx):
            idx_i = int(idx)
            if idx_i in seen_10m:
                continue
            seen_10m.add(idx_i)
            if idx_i <= int(trade["entry_idx"]):
                continue
            value = float(opp[idx_i]) if idx_i < opp.size else np.nan
            if np.isfinite(value) and value >= float(threshold):
                exit_pos = pos
                reason = "opposite_10m_setup"
                break
        event = _event_from_trade(
            trade,
            regime="opposite_10m_setup_threshold",
            exit_style="opposite_10m_setup",
            exit_pos=exit_pos,
            exit_reason=reason,
        )
        if event is None:
            continue
        event["position_mode"] = position_mode
        rows.append(event)
        if position_mode == "single":
            open_until = pd.Timestamp(event["exit_time"])
    for row in rows:
        row["candidate_trades"] = len(ordered)
        row["skipped_overlap_candidates"] = blocked
    return rows


def _simulate_opposite_confirmed(
    trades: list[dict[str, object]],
    *,
    long_entry_times: pd.DatetimeIndex,
    short_entry_times: pd.DatetimeIndex,
    position_mode: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    open_until: pd.Timestamp | None = None
    blocked = 0
    ordered = sorted(trades, key=lambda trade: pd.Timestamp(trade["entry_time"]))
    for trade in ordered:
        entry_time = pd.Timestamp(trade["entry_time"])
        if position_mode == "single" and open_until is not None and entry_time < open_until:
            blocked += 1
            continue
        side = str(trade["side"])
        path_times = trade.get("path_times")
        if not isinstance(path_times, pd.DatetimeIndex) or path_times.empty:
            continue
        opp_times = short_entry_times if side == "long" else long_entry_times
        left = opp_times.searchsorted(entry_time, side="right")
        next_opp = opp_times[left] if left < len(opp_times) else None
        exit_pos = len(path_times) - 1
        reason = "timeout"
        if next_opp is not None:
            pos = path_times.searchsorted(pd.Timestamp(next_opp), side="left")
            if pos < len(path_times):
                exit_pos = int(pos)
                reason = "opposite_1m_confirmed"
        event = _event_from_trade(
            trade,
            regime="opposite_1m_confirmed_entry",
            exit_style="opposite_1m_confirmed",
            exit_pos=exit_pos,
            exit_reason=reason,
        )
        if event is None:
            continue
        event["position_mode"] = position_mode
        rows.append(event)
        if position_mode == "single":
            open_until = pd.Timestamp(event["exit_time"])
    for row in rows:
        row["candidate_trades"] = len(ordered)
        row["skipped_overlap_candidates"] = blocked
    return rows


def _simulate_opposite_proba(
    trades: list[dict[str, object]],
    *,
    p_long: np.ndarray,
    p_short: np.ndarray,
    threshold: float,
    position_mode: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    open_until: pd.Timestamp | None = None
    blocked = 0
    ordered = sorted(trades, key=lambda trade: pd.Timestamp(trade["entry_time"]))
    for trade in ordered:
        entry_time = pd.Timestamp(trade["entry_time"])
        if position_mode == "single" and open_until is not None and entry_time < open_until:
            blocked += 1
            continue
        side = str(trade["side"])
        path_idx = trade.get("path_idx")
        if not isinstance(path_idx, np.ndarray) or path_idx.size == 0:
            continue
        opp = p_short if side == "long" else p_long
        exit_pos = path_idx.size - 1
        reason = "timeout"
        for pos, idx in enumerate(path_idx):
            idx_i = int(idx)
            if idx_i <= int(trade["entry_idx"]):
                continue
            value = float(opp[idx_i]) if idx_i < opp.size else np.nan
            if np.isfinite(value) and value >= float(threshold):
                exit_pos = int(pos)
                reason = f"opposite_proba_{threshold:.2f}"
                break
        event = _event_from_trade(
            trade,
            regime=f"opposite_proba_{threshold:.2f}",
            exit_style="opposite_proba",
            exit_pos=exit_pos,
            exit_reason=reason,
        )
        if event is None:
            continue
        event["position_mode"] = position_mode
        rows.append(event)
        if position_mode == "single":
            open_until = pd.Timestamp(event["exit_time"])
    for row in rows:
        row["candidate_trades"] = len(ordered)
        row["skipped_overlap_candidates"] = blocked
    return rows


def _simulate_opposite_proba_giveback(
    trades: list[dict[str, object]],
    *,
    p_long: np.ndarray,
    p_short: np.ndarray,
    threshold: float,
    giveback_pct: float,
    position_mode: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    open_until: pd.Timestamp | None = None
    blocked = 0
    ordered = sorted(trades, key=lambda trade: pd.Timestamp(trade["entry_time"]))
    for trade in ordered:
        entry_time = pd.Timestamp(trade["entry_time"])
        if position_mode == "single" and open_until is not None and entry_time < open_until:
            blocked += 1
            continue
        side = str(trade["side"])
        path_idx = trade.get("path_idx")
        close_path = trade.get("close_path")
        favorable_path = trade.get("favorable_path")
        if not (
            isinstance(path_idx, np.ndarray)
            and isinstance(close_path, np.ndarray)
            and isinstance(favorable_path, np.ndarray)
            and path_idx.size > 0
        ):
            continue
        opp = p_short if side == "long" else p_long
        exit_pos = path_idx.size - 1
        reason = "timeout"
        for pos, idx in enumerate(path_idx):
            idx_i = int(idx)
            if idx_i <= int(trade["entry_idx"]):
                continue
            value = float(opp[idx_i]) if idx_i < opp.size else np.nan
            best_so_far = float(np.nanmax(favorable_path[: pos + 1]))
            current_profit = float(close_path[pos])
            gave_back = bool(current_profit <= best_so_far - float(giveback_pct))
            if np.isfinite(value) and value >= float(threshold) and gave_back:
                exit_pos = int(pos)
                reason = f"opposite_proba_{threshold:.2f}_giveback_{giveback_pct:.2f}"
                break
        event = _event_from_trade(
            trade,
            regime=f"opposite_proba_{threshold:.2f}_giveback_{giveback_pct:.2f}",
            exit_style="opposite_proba_giveback",
            exit_pos=exit_pos,
            exit_reason=reason,
        )
        if event is None:
            continue
        event["position_mode"] = position_mode
        rows.append(event)
        if position_mode == "single":
            open_until = pd.Timestamp(event["exit_time"])
    for row in rows:
        row["candidate_trades"] = len(ordered)
        row["skipped_overlap_candidates"] = blocked
    return rows


def _simulate_opposite_proba_soft_count(
    trades: list[dict[str, object]],
    *,
    p_long: np.ndarray,
    p_short: np.ndarray,
    threshold: float,
    soft_count: int,
    position_mode: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    open_until: pd.Timestamp | None = None
    blocked = 0
    ordered = sorted(trades, key=lambda trade: pd.Timestamp(trade["entry_time"]))
    required = max(1, int(soft_count))
    for trade in ordered:
        entry_time = pd.Timestamp(trade["entry_time"])
        if position_mode == "single" and open_until is not None and entry_time < open_until:
            blocked += 1
            continue
        side = str(trade["side"])
        path_idx = trade.get("path_idx")
        if not isinstance(path_idx, np.ndarray) or path_idx.size == 0:
            continue
        opp = p_short if side == "long" else p_long
        exit_pos = path_idx.size - 1
        reason = "timeout"
        seen_10m: set[int] = set()
        consecutive = 0
        for pos, idx in enumerate(path_idx):
            idx_i = int(idx)
            if idx_i in seen_10m:
                continue
            seen_10m.add(idx_i)
            if idx_i <= int(trade["entry_idx"]):
                continue
            value = float(opp[idx_i]) if idx_i < opp.size else np.nan
            if np.isfinite(value) and value >= float(threshold):
                consecutive += 1
            else:
                consecutive = 0
            if consecutive >= required:
                exit_pos = int(pos)
                reason = f"opposite_proba_{threshold:.2f}_soft{required}"
                break
        event = _event_from_trade(
            trade,
            regime=f"opposite_proba_{threshold:.2f}_soft{required}",
            exit_style="opposite_proba_soft_count",
            exit_pos=exit_pos,
            exit_reason=reason,
        )
        if event is None:
            continue
        event["position_mode"] = position_mode
        rows.append(event)
        if position_mode == "single":
            open_until = pd.Timestamp(event["exit_time"])
    for row in rows:
        row["candidate_trades"] = len(ordered)
        row["skipped_overlap_candidates"] = blocked
    return rows


def main() -> None:
    args = _parse_args()
    feature_df = pd.read_parquet(REPO_ROOT / args.signal_frame).sort_index()
    try:
        selected = _load_scoreboard(REPO_ROOT / args.scoreboard, args.split, args.variant, args.mode)
    except ValueError:
        if args.split == "oof":
            raise
        selected = _load_scoreboard(REPO_ROOT / args.scoreboard, "oof", args.variant, args.mode).copy()
        selected["split"] = f"oof_thresholds_for_{args.split}"

    long_threshold = float(args.long_setup_threshold if args.long_setup_threshold is not None else selected["long_setup_threshold"])
    short_threshold = float(args.short_setup_threshold if args.short_setup_threshold is not None else selected["short_setup_threshold"])
    cooldown_bars = int(args.cooldown_bars if args.cooldown_bars is not None else selected["cooldown_bars"])
    post_setup_max_bars = int(
        args.post_setup_max_bars if args.post_setup_max_bars is not None else selected.get("post_setup_max_bars", 4)
    )
    one_per_setup_cluster = bool(selected.get("one_per_setup_cluster", False))
    execution_1m = _load_execution_1m(args, feature_df.index)
    long_entries, short_entries, long_prices, short_prices, long_times, short_times = _build_entries(
        feature_df,
        selected,
        split=args.split,
        long_setup_threshold=long_threshold,
        short_setup_threshold=short_threshold,
        cooldown_bars=cooldown_bars,
        post_setup_max_bars=post_setup_max_bars,
        one_per_setup_cluster=one_per_setup_cluster,
        execution_1m=execution_1m,
    )
    trades = _build_trade_specs(
        feature_df,
        split=args.split,
        long_entries=long_entries,
        short_entries=short_entries,
        long_prices=long_prices,
        short_prices=short_prices,
        long_times=long_times,
        short_times=short_times,
        premium_atr_mult=float(args.premium_atr_mult),
        exit_hhmm=str(args.exit_hhmm),
        horizon_bars=int(args.horizon_bars),
        proxy_mode=str(args.proxy_mode),
        bs_iv_floor=float(args.bs_iv_floor),
        bs_iv_ceil=float(args.bs_iv_ceil),
        bs_iv_mult=float(args.bs_iv_mult),
        bs_iv_shock=float(args.bs_iv_shock),
        bs_min_dte_minutes=float(args.bs_min_dte_minutes),
        bs_strike_round=float(args.bs_strike_round),
        execution_1m=execution_1m,
    )
    p_suffix = "oof_train" if args.split == "oof" else "test"
    p_long = pd.to_numeric(feature_df[f"p_long_{p_suffix}"], errors="coerce").to_numpy(dtype=float)
    p_short = pd.to_numeric(feature_df[f"p_short_{p_suffix}"], errors="coerce").to_numpy(dtype=float)
    market = {
        "high": pd.to_numeric(feature_df["high"], errors="coerce").to_numpy(dtype=float),
        "low": pd.to_numeric(feature_df["low"], errors="coerce").to_numpy(dtype=float),
        "close": pd.to_numeric(feature_df["close"], errors="coerce").to_numpy(dtype=float),
        "p_long": p_long,
        "p_short": p_short,
        "index": feature_df.index,
    }
    long_confirm_times = pd.DatetimeIndex(pd.to_datetime(long_times[long_entries], utc=True, errors="coerce")).dropna().sort_values()
    short_confirm_times = pd.DatetimeIndex(pd.to_datetime(short_times[short_entries], utc=True, errors="coerce")).dropna().sort_values()

    event_rows: list[dict[str, object]] = []
    for mode in _position_modes(args.position_modes):
        event_rows.extend(_simulate_strategy_events(market, trades, _current_best_spec(), position_mode=mode))
        event_rows.extend(
            _simulate_opposite_10m_setup(
                trades,
                p_long=p_long,
                p_short=p_short,
                long_threshold=long_threshold,
                short_threshold=short_threshold,
                position_mode=mode,
            )
        )
        event_rows.extend(
            _simulate_opposite_confirmed(
                trades,
                long_entry_times=long_confirm_times,
                short_entry_times=short_confirm_times,
                position_mode=mode,
            )
        )
        for threshold in _parse_float_list(args.opposite_proba_thresholds):
            event_rows.extend(
                _simulate_opposite_proba(
                    trades,
                    p_long=p_long,
                    p_short=p_short,
                    threshold=float(threshold),
                    position_mode=mode,
                )
            )
            for giveback_pct in _parse_float_list(args.giveback_pcts):
                event_rows.extend(
                    _simulate_opposite_proba_giveback(
                        trades,
                        p_long=p_long,
                        p_short=p_short,
                        threshold=float(threshold),
                        giveback_pct=float(giveback_pct),
                        position_mode=mode,
                    )
                )
            for soft_count in _parse_int_list(args.soft_counts):
                event_rows.extend(
                    _simulate_opposite_proba_soft_count(
                        trades,
                        p_long=p_long,
                        p_short=p_short,
                        threshold=float(threshold),
                        soft_count=int(soft_count),
                        position_mode=mode,
                    )
                )

    events = pd.DataFrame(event_rows)
    summary = _summarize(events)
    out_path = REPO_ROOT / args.out
    events_path = REPO_ROOT / args.events_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_path, index=False)
    events.to_csv(events_path, index=False)

    cols = [
        "position_mode",
        "regime",
        "exit_style",
        "trades",
        "candidate_trades",
        "skipped_overlap_candidates",
        "mean_outcome_pct",
        "median_outcome_pct",
        "total_outcome_pct",
        "win_rate",
        "p90_outcome_pct",
        "p95_outcome_pct",
        "p99_outcome_pct",
        "avg_mfe_pct",
        "adaptive_trail_count",
        "time_decay_count",
        "opposite_signal_count",
        "opposite_10m_setup_count",
        "opposite_1m_confirmed_count",
        "timeout_count",
        "long_trades",
        "long_mean_outcome_pct",
        "short_trades",
        "short_mean_outcome_pct",
    ]
    print(
        f"selected={selected['split']} {selected['variant']} {selected['mode']} "
        f"thresholds long={long_threshold:.2f} short={short_threshold:.2f} "
        f"entries long={int(long_entries.sum())} short={int(short_entries.sum())} trades={len(trades)}"
    )
    print(summary[[c for c in cols if c in summary.columns]].to_string(index=False))
    print(f"summary_csv={out_path}")
    print(f"events_csv={events_path}")


if __name__ == "__main__":
    main()
