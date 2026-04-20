from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Policy.replay_option_proxy import ReplayOptionPriceProxy
from scripts.compare_entry_overlay_policies import (
    DEFAULT_ANALYSIS_DIR,
    DEFAULT_EVENTS_OUT,
    DEFAULT_ONE_MIN,
    DEFAULT_OUT,
    DEFAULT_SIGNAL_FRAME,
    PendingSetup,
    Position,
    _as_float,
    _close_event,
    _filter_window,
    _load_one_min,
    _load_signal_frame,
    _metrics,
    _sim_symbol,
    _triggered,
    _update_setups,
)


DEFAULT_SUMMARY_OUT = DEFAULT_ANALYSIS_DIR / "phase4_continuation_exit_policy_compare_summary.csv"
DEFAULT_EVENTS_OUT = DEFAULT_ANALYSIS_DIR / "phase4_continuation_exit_policy_compare_events.csv"
DEFAULT_FAILED_SETUPS_OUT = DEFAULT_ANALYSIS_DIR / "phase4_failed_setup_continuation_labels.csv"


POLICY_SETUP_MAP = {
    "baseline_current": "test_a_swing_only_l035_s065",
    "opposite_exit_requires_1m_confirm": "test_a_swing_only_l035_s065",
    "continuation_hold_filter": "test_a_swing_only_l035_s065",
    "continuation_entry_overlay": "test_c_continuation_overlay_l035_s065",
}


def _policy_setup_name(policy: str) -> str:
    return POLICY_SETUP_MAP.get(policy, "test_a_swing_only_l035_s065")


def _same_expiry_today(symbol: str, ts: pd.Timestamp) -> bool:
    parts = str(symbol or "").split("_")
    return len(parts) >= 3 and parts[2] == ts.strftime("%y%m%d")


def _favorable_continuation_context(
    *,
    side: str,
    latest_decision: dict[str, Any] | None,
    row: dict[str, Any],
    opposite_confirmed: bool,
) -> bool:
    """A soft hold filter, not a hard regime entry filter.

    It only says: if we are long in a bullish trend, or short in a bearish
    trend, and the opposite 1m reversal did not confirm yet, give the trade
    room instead of exiting on the raw opposite probability spike.
    """
    if latest_decision is None or opposite_confirmed:
        return False
    regime = str(latest_decision.get("trend_regime", "neutral"))
    close = _as_float(row.get("close"))
    ema_fast = _as_float(latest_decision.get("ema_fast"))
    ema_slow = _as_float(latest_decision.get("ema_slow"))
    if not all(math.isfinite(x) for x in (close, ema_fast, ema_slow)):
        return False
    if side == "long":
        return bool(regime == "bullish" and close >= ema_fast and ema_fast >= ema_slow)
    if side == "short":
        return bool(regime == "bearish" and close <= ema_fast and ema_fast <= ema_slow)
    return False


def _run_policy(
    *,
    policy: str,
    decisions: pd.DataFrame,
    one_min: pd.DataFrame,
    setup_max_bars: int,
    cutoff_hhmm: str,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    proxy = ReplayOptionPriceProxy(
        tz_name="America/New_York",
        expiry_hhmm="15:40",
        iv_floor=0.12,
        iv_ceiling=0.90,
        iv_multiplier=1.50,
        min_dte_minutes=1.0,
    )
    setup_policy = _policy_setup_name(policy)
    decision_records = decisions.copy().reset_index(drop=True).to_dict("records")
    decision_idx = 0
    latest_decision: dict[str, Any] | None = None
    latest_probs = {"long": float("nan"), "short": float("nan")}
    latest_atr = float("nan")
    pending: dict[str, PendingSetup | None] = {"long": None, "short": None}
    above = {"long": False, "short": False}
    peak = {"long": float("nan"), "short": float("nan")}
    pos: Position | None = None
    events: list[dict[str, Any]] = []
    counters = {
        "swing_setups": 0,
        "countertrend_vetoes": 0,
        "continuation_setups": 0,
        "opposite_exits_blocked": 0,
        "time_decay_exits_blocked": 0,
        "opposite_confirmed_exits": 0,
    }
    last_day = None

    for row in one_min.to_dict("records"):
        ts = row["timestamp"]
        if last_day is not None and ts.date() != last_day:
            pending = {"long": None, "short": None}
            above = {"long": False, "short": False}
            peak = {"long": float("nan"), "short": float("nan")}
        last_day = ts.date()

        proxy.update_bar("SPY", row)
        while decision_idx < len(decision_records) and decision_records[decision_idx]["available_ts"] <= ts:
            drow = pd.Series(decision_records[decision_idx])
            latest_decision = dict(decision_records[decision_idx])
            latest_probs["long"] = _as_float(drow.get("p_enter_long"))
            latest_probs["short"] = _as_float(drow.get("p_enter_short"))
            latest_atr = _as_float(drow.get("atr_proxy"))
            updates = _update_setups(
                drow,
                policy=setup_policy,
                pending=pending,
                above=above,
                peak=peak,
                setup_max_bars=setup_max_bars,
            )
            for key, val in updates.items():
                counters[key] = counters.get(key, 0) + int(val)
            decision_idx += 1

        close = _as_float(row.get("close"))
        opposite_setup = None
        opposite_confirmed = False
        if pos is not None:
            opp_side = "short" if pos.side == "long" else "long"
            opposite_setup = pending.get(opp_side)
            opposite_confirmed = bool(
                opposite_setup is not None
                and opposite_setup.start_ts <= ts < opposite_setup.expires_at
                and _triggered(opposite_setup, row)
            )

        if pos is not None:
            premium = proxy.price(pos.symbol, mode="mid")
            if math.isfinite(premium):
                if premium > pos.best_premium:
                    pos.best_premium = float(premium)
                    pos.best_seen_at = ts
                ret = premium / pos.entry_premium - 1.0
                best_ret = pos.best_premium / pos.entry_premium - 1.0
                minutes_since_best = (ts - pos.best_seen_at).total_seconds() / 60.0
                opp_prob = latest_probs["short"] if pos.side == "long" else latest_probs["long"]
                hold_context = _favorable_continuation_context(
                    side=pos.side,
                    latest_decision=latest_decision,
                    row=row,
                    opposite_confirmed=opposite_confirmed,
                )
                reason = None
                if best_ret >= 2.0 and ((pos.best_premium - premium) / pos.entry_premium) >= 0.25:
                    reason = "adaptive_trail"
                elif (
                    policy in {"opposite_exit_requires_1m_confirm", "continuation_hold_filter", "continuation_entry_overlay"}
                    and best_ret >= 1.0
                    and opposite_confirmed
                ):
                    reason = "opposite_1m_confirmed"
                    counters["opposite_confirmed_exits"] += 1
                elif minutes_since_best >= 80 and best_ret < 1.0 and ret <= 0.0:
                    if policy in {"continuation_hold_filter", "continuation_entry_overlay"} and hold_context:
                        counters["time_decay_exits_blocked"] += 1
                    else:
                        reason = "time_decay"
                elif best_ret >= 1.0 and math.isfinite(opp_prob) and opp_prob >= 0.60:
                    if policy in {"opposite_exit_requires_1m_confirm", "continuation_hold_filter", "continuation_entry_overlay"}:
                        counters["opposite_exits_blocked"] += 1
                    else:
                        reason = "opposite_signal"
                elif ts.hour == 15 and ts.minute >= 40 and _same_expiry_today(pos.symbol, ts):
                    reason = "same_day_eod"
                if reason:
                    events.append(_close_event(pos, ts, close, premium, reason))
                    pos = None

        if pos is None:
            live_setups = [
                setup
                for setup in (pending.get("long"), pending.get("short"))
                if setup is not None and setup.start_ts <= ts < setup.expires_at
            ]
            triggered = [setup for setup in live_setups if _triggered(setup, row)]
            if triggered:
                if len(triggered) > 1:
                    triggered.sort(key=lambda s: s.prob - s.threshold, reverse=True)
                setup = triggered[0]
                right = "C" if setup.side == "long" else "P"
                strike = round(close + latest_atr) if setup.side == "long" else round(close - latest_atr)
                symbol = _sim_symbol(ts, right, strike, cutoff_hhmm)
                premium = proxy.price(symbol, mode="mid")
                if math.isfinite(premium) and premium > 0:
                    pos = Position(
                        policy=policy,
                        side=setup.side,
                        entry_ts=ts,
                        entry_spot=float(close),
                        entry_premium=float(premium),
                        symbol=symbol,
                        best_premium=float(premium),
                        best_seen_at=ts,
                        entry_prob=setup.prob,
                        threshold=setup.threshold,
                        regime=setup.regime,
                        entry_kind=setup.entry_kind,
                        source_side=setup.source_side,
                    )
                    pending[setup.side] = None

    if pos is not None and len(one_min):
        last = one_min.iloc[-1]
        premium = proxy.price(pos.symbol, mode="mid")
        if math.isfinite(premium):
            events.append(_close_event(pos, last["timestamp"], _as_float(last["close"]), premium, "window_end"))
    return events, counters


def _failed_setup_labels(decisions: pd.DataFrame, *, setup_max_bars: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    decisions = decisions.copy().reset_index(drop=True)
    for idx, row in decisions.iterrows():
        regime = str(row.get("trend_regime", "neutral"))
        if regime not in {"bullish", "bearish"}:
            continue
        if regime == "bullish":
            side = "short"
            trend_side = "long"
            prob = _as_float(row.get("p_enter_short"))
            threshold = 0.65
            ref = _as_float(row.get("low"))
            future = decisions.iloc[idx + 1 : idx + setup_max_bars + 5]
            if not (math.isfinite(prob) and prob >= threshold and math.isfinite(ref)) or future.empty:
                continue
            confirmed = bool((future["low"] <= ref).any())
            entry_close = _as_float(row.get("close"))
            atr = _as_float(row.get("atr_proxy"))
            max_cont = (_as_float(future["high"].max()) - entry_close) / atr if math.isfinite(atr) and atr > 0 else float("nan")
            max_adverse = (entry_close - _as_float(future["low"].min())) / atr if math.isfinite(atr) and atr > 0 else float("nan")
        else:
            side = "long"
            trend_side = "short"
            prob = _as_float(row.get("p_enter_long"))
            threshold = 0.65
            ref = _as_float(row.get("high"))
            future = decisions.iloc[idx + 1 : idx + setup_max_bars + 5]
            if not (math.isfinite(prob) and prob >= threshold and math.isfinite(ref)) or future.empty:
                continue
            confirmed = bool((future["high"] >= ref).any())
            entry_close = _as_float(row.get("close"))
            atr = _as_float(row.get("atr_proxy"))
            max_cont = (entry_close - _as_float(future["low"].min())) / atr if math.isfinite(atr) and atr > 0 else float("nan")
            max_adverse = (_as_float(future["high"].max()) - entry_close) / atr if math.isfinite(atr) and atr > 0 else float("nan")
        failed = not confirmed
        rows.append(
            {
                "timestamp": row.get("timestamp"),
                "regime": regime,
                "setup_side": side,
                "trend_side": trend_side,
                "setup_prob": prob,
                "setup_threshold": threshold,
                "failed_confirmation": failed,
                "future_continuation_atr": max_cont,
                "future_adverse_atr": max_adverse,
                "continuation_label": bool(failed and math.isfinite(max_cont) and max_cont >= 1.0 and max_cont >= max_adverse),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare continuation-aware exit/hold/entry policies.")
    parser.add_argument("--signal-frame", default=str(DEFAULT_SIGNAL_FRAME))
    parser.add_argument("--one-min", default=str(DEFAULT_ONE_MIN))
    parser.add_argument("--recent-months", type=int, default=3)
    parser.add_argument("--window-scope", choices=["recent", "full", "both"], default="both")
    parser.add_argument("--setup-max-bars", type=int, default=4)
    parser.add_argument("--cutoff-hhmm", default="13:00")
    parser.add_argument(
        "--policies",
        default="baseline_current,opposite_exit_requires_1m_confirm,continuation_hold_filter,continuation_entry_overlay",
    )
    parser.add_argument("--out", default=str(DEFAULT_SUMMARY_OUT))
    parser.add_argument("--events-out", default=str(DEFAULT_EVENTS_OUT))
    parser.add_argument("--failed-setups-out", default=str(DEFAULT_FAILED_SETUPS_OUT))
    args = parser.parse_args()

    signal = _load_signal_frame(Path(args.signal_frame))
    if signal.empty:
        raise SystemExit("No signal rows found.")
    full_start = signal["timestamp"].min()
    full_end = signal["timestamp"].max()
    recent_start = full_end - pd.DateOffset(months=int(args.recent_months))
    if args.window_scope == "recent":
        windows = [("recent", recent_start, full_end)]
    elif args.window_scope == "full":
        windows = [("full", full_start, full_end)]
    else:
        windows = [("recent", recent_start, full_end), ("full", full_start, full_end)]

    summaries: list[dict[str, Any]] = []
    all_events: list[dict[str, Any]] = []
    label_frames: list[pd.DataFrame] = []
    policies = [x.strip() for x in str(args.policies).split(",") if x.strip()]

    for window_label, start_ts, end_ts in windows:
        decisions = _filter_window(signal, start_ts, end_ts)
        if decisions.empty:
            continue
        one_min = _load_one_min(Path(args.one_min), decisions["timestamp"].min(), decisions["timestamp"].max())
        one_min = one_min[one_min["timestamp"] >= decisions["timestamp"].min() - pd.Timedelta(minutes=10)].reset_index(drop=True)
        if one_min.empty:
            continue
        labels = _failed_setup_labels(decisions, setup_max_bars=int(args.setup_max_bars))
        if not labels.empty:
            labels.insert(0, "window", window_label)
            label_frames.append(labels)
        for policy in policies:
            events, counters = _run_policy(
                policy=policy,
                decisions=decisions,
                one_min=one_min,
                setup_max_bars=int(args.setup_max_bars),
                cutoff_hhmm=str(args.cutoff_hhmm),
            )
            summaries.append(_metrics(events, policy=policy, label=window_label, decisions=decisions, counters=counters))
            for event in events:
                all_events.append({"window": window_label, **event})

    summary_df = pd.DataFrame(summaries).sort_values(
        ["window", "sum_return", "avg_return", "trades"],
        ascending=[True, False, False, False],
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(out, index=False)
    events_out = Path(args.events_out)
    pd.DataFrame(all_events).to_csv(events_out, index=False)
    failed_out = Path(args.failed_setups_out)
    failed_df = pd.concat(label_frames, ignore_index=True) if label_frames else pd.DataFrame()
    failed_df.to_csv(failed_out, index=False)
    print(f"[continuation-exit-compare] wrote {out}")
    print(f"[continuation-exit-compare] wrote {events_out}")
    print(f"[continuation-exit-compare] wrote {failed_out}")
    if not summary_df.empty:
        print(summary_df.to_string(index=False))
    if not failed_df.empty:
        by_window = failed_df.groupby("window").agg(
            failed_setups=("failed_confirmation", "sum"),
            failed_setup_rows=("failed_confirmation", "count"),
            continuation_labels=("continuation_label", "sum"),
            avg_failed_cont_atr=("future_continuation_atr", "mean"),
        )
        print("\nFailed setup continuation label sketch:")
        print(by_window.to_string())


if __name__ == "__main__":
    main()
