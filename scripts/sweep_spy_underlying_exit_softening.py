from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from strategies.spy_intraday.Policy.order_policy import (
    PHASE4_SWING_SETUP_BODYCLOSE_BODYCLOSE_V1,
    OptionOrderPolicy,
    OptionOrderPolicyConfig,
)
from strategies.spy_intraday.Policy.replay_option_proxy import ReplayOptionPriceProxy
from scripts.analyze_confirmed_entry_trade_quality import (
    _extract_fill,
    _load_live_decision_signal_frame,
    _pair_trades,
    _quiet_logger,
    _summary_rows,
)
from scripts.compare_entry_overlay_policies import (
    DEFAULT_ANALYSIS_DIR,
    DEFAULT_ONE_MIN,
    DEFAULT_SIGNAL_FRAME,
    _load_one_min,
    _load_signal_frame,
)


DEFAULT_SUMMARY_OUT = DEFAULT_ANALYSIS_DIR / "phase4_underlying_exit_softening_sweep_summary.csv"
DEFAULT_TRADES_OUT = DEFAULT_ANALYSIS_DIR / "phase4_underlying_exit_softening_sweep_trades.csv"
DEFAULT_EVENTS_OUT = DEFAULT_ANALYSIS_DIR / "phase4_underlying_exit_softening_sweep_events.csv"


@dataclass(frozen=True)
class ExitVariant:
    name: str
    setup_failure_enabled: bool
    setup_failure_buffer_atr: float
    no_progress_enabled: bool
    no_progress_minutes: int
    no_progress_atr: float


TARGETED_VARIANTS = [
    ExitVariant(
        name="option_bracket_only",
        setup_failure_enabled=False,
        setup_failure_buffer_atr=0.10,
        no_progress_enabled=False,
        no_progress_minutes=10,
        no_progress_atr=0.20,
    ),
    ExitVariant(
        name="current_fixed",
        setup_failure_enabled=True,
        setup_failure_buffer_atr=0.10,
        no_progress_enabled=True,
        no_progress_minutes=10,
        no_progress_atr=0.20,
    ),
    ExitVariant(
        name="soft_020_20_020",
        setup_failure_enabled=True,
        setup_failure_buffer_atr=0.20,
        no_progress_enabled=True,
        no_progress_minutes=20,
        no_progress_atr=0.20,
    ),
    ExitVariant(
        name="soft_020_20_035",
        setup_failure_enabled=True,
        setup_failure_buffer_atr=0.20,
        no_progress_enabled=True,
        no_progress_minutes=20,
        no_progress_atr=0.35,
    ),
    ExitVariant(
        name="soft_035_30_035",
        setup_failure_enabled=True,
        setup_failure_buffer_atr=0.35,
        no_progress_enabled=True,
        no_progress_minutes=30,
        no_progress_atr=0.35,
    ),
    ExitVariant(
        name="soft_050_30_050",
        setup_failure_enabled=True,
        setup_failure_buffer_atr=0.50,
        no_progress_enabled=True,
        no_progress_minutes=30,
        no_progress_atr=0.50,
    ),
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep softer underlying-based exits on the live-style SPY order policy replay."
    )
    parser.add_argument("--signal-frame", default=str(DEFAULT_SIGNAL_FRAME))
    parser.add_argument("--one-min", default=str(DEFAULT_ONE_MIN))
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--summary-out", default=str(DEFAULT_SUMMARY_OUT))
    parser.add_argument("--trades-out", default=str(DEFAULT_TRADES_OUT))
    parser.add_argument("--events-out", default=str(DEFAULT_EVENTS_OUT))
    return parser.parse_args()


def _replay_variant(
    *,
    variant: ExitVariant,
    signals: pd.DataFrame,
    one_min: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    proxy = ReplayOptionPriceProxy(
        tz_name="America/New_York",
        expiry_hhmm="15:40",
        iv_floor=0.12,
        iv_ceiling=0.90,
        iv_multiplier=1.50,
        min_dte_minutes=1.0,
    )
    policy = OptionOrderPolicy(
        OptionOrderPolicyConfig(
            underlying="SPY",
            submit_orders=False,
            meta_intrabar_entry_policy=PHASE4_SWING_SETUP_BODYCLOSE_BODYCLOSE_V1,
            meta_intrabar_execution_enabled=True,
            meta_intrabar_breakout_entry_only=True,
            meta_intrabar_setup_max_bars=4,
            meta_intrabar_setup_bar_minutes=10,
            meta_intrabar_max_confirmation_age_minutes=30,
            meta_intrabar_ref_chase_atr=0.50,
            meta_intrabar_long_setup_threshold=0.35,
            meta_intrabar_short_setup_threshold=0.65,
            meta_hard_stop_atr=0.0,
            meta_setup_failure_exit_enabled=variant.setup_failure_enabled,
            meta_setup_failure_buffer_atr=variant.setup_failure_buffer_atr,
            meta_no_progress_exit_enabled=variant.no_progress_enabled,
            meta_no_progress_exit_minutes=variant.no_progress_minutes,
            meta_no_progress_exit_atr=variant.no_progress_atr,
            option_exit_policy="option_adaptive_trail_v1",
            option_exit_take_profit_pct=0.0,
            option_exit_stop_loss_pct=1.0,
            option_exit_profit_lock_arm_pct=2.0,
            option_exit_profit_lock_floor_pct=0.25,
            option_exit_trailing_arm_pct=2.0,
            option_exit_trailing_giveback_pct=0.25,
            option_exit_time_decay_minutes=80,
            option_exit_time_decay_progress_pct=1.0,
            option_exit_opposite_prob=0.60,
            option_exit_opposite_profit_pct=0.60,
            option_exit_quote_mode="bid",
        )
    )
    policy.set_contract_price_provider(proxy.price)

    idx_1m = 0
    fills: list[dict[str, Any]] = []
    first_available = pd.Timestamp(signals["available_ts"].iloc[0])
    while idx_1m < len(one_min) and pd.Timestamp(one_min.iloc[idx_1m]["timestamp"]) < first_available:
        bar = one_min.iloc[idx_1m].to_dict()
        proxy.update_bar("SPY", bar)
        policy.prefill_1m_bar(bar=bar)
        idx_1m += 1

    for idx, row in signals.iterrows():
        available_ts = pd.Timestamp(row["available_ts"])
        next_available = (
            pd.Timestamp(signals.iloc[idx + 1]["available_ts"])
            if idx + 1 < len(signals)
            else available_ts + pd.Timedelta(minutes=10)
        )
        bar_payload = {
            "timestamp": row["timestamp"],
            "open": row["open"],
            "high": row["high"],
            "low": row["low"],
            "close": row["close"],
            "ema_fast": row.get("ema_fast"),
            "ema_slow": row.get("ema_slow"),
            "p_enter_long": row["p_enter_long"],
            "p_enter_short": row["p_enter_short"],
            "thr_enter_long": 0.35,
            "thr_enter_short": 0.65,
            "thr_exit_long": 1.0,
            "thr_exit_short": 1.0,
            "p_exit_long": 0.0,
            "p_exit_short": 0.0,
            "trend_regime": row.get("trend_regime"),
            "signal_atr": row.get("atr_proxy"),
        }
        result_10m = policy.on_decision(action=0.0, closed_bar=bar_payload, logger=_quiet_logger)
        for wrapper in result_10m.get("orders") or []:
            event = _extract_fill(
                wrapper,
                ts=available_ts,
                spot=float(row["close"]),
                policy=policy,
            )
            if event:
                fills.append(event)

        while idx_1m < len(one_min):
            bar = one_min.iloc[idx_1m].to_dict()
            bar_ts = pd.Timestamp(bar["timestamp"])
            if bar_ts < available_ts:
                proxy.update_bar("SPY", bar)
                policy.prefill_1m_bar(bar=bar)
                idx_1m += 1
                continue
            if bar_ts >= next_available:
                break
            proxy.update_bar("SPY", bar)
            result_1m = policy.on_1m_bar(bar=bar, logger=_quiet_logger)
            for wrapper in result_1m.get("orders") or []:
                event = _extract_fill(
                    wrapper,
                    ts=bar_ts,
                    spot=float(bar["close"]),
                    policy=policy,
                )
                if event:
                    fills.append(event)
            idx_1m += 1

    events = pd.DataFrame(fills)
    if not events.empty:
        events = events.sort_values("timestamp").reset_index(drop=True)
        events.insert(0, "variant", variant.name)
    trades = _pair_trades(events, one_min) if not events.empty else pd.DataFrame()
    if not trades.empty:
        trades.insert(0, "variant", variant.name)
    summary = _summary_rows(trades)
    summary.insert(0, "variant", variant.name)
    summary.insert(1, "setup_failure_enabled", variant.setup_failure_enabled)
    summary.insert(2, "setup_failure_buffer_atr", variant.setup_failure_buffer_atr)
    summary.insert(3, "no_progress_enabled", variant.no_progress_enabled)
    summary.insert(4, "no_progress_minutes", variant.no_progress_minutes)
    summary.insert(5, "no_progress_atr", variant.no_progress_atr)
    print(
        f"[sweep-soft-exits] {variant.name}: "
        f"trades={int(summary.loc[summary['bucket'] == 'all', 'trades'].iloc[0])} "
        f"avg_return_pct={float(summary.loc[summary['bucket'] == 'all', 'avg_return_pct'].iloc[0]):.4f} "
        f"win_rate={float(summary.loc[summary['bucket'] == 'all', 'win_rate'].iloc[0]):.4f}",
        flush=True,
    )
    return events, trades, summary


def main() -> None:
    args = _parse_args()
    report_start = pd.Timestamp(args.start, tz="America/New_York") if args.start else None
    report_end = pd.Timestamp(args.end, tz="America/New_York") if args.end else None

    historical = _load_signal_frame(Path(args.signal_frame)).sort_values("timestamp").reset_index(drop=True)
    live = _load_live_decision_signal_frame(start=report_start, end=report_end)
    all_signals = pd.concat([historical, live], ignore_index=True, sort=False)
    if all_signals.empty:
        raise SystemExit("No signals available from historical frame or live decision logs.")
    signals = all_signals.sort_values("available_ts").drop_duplicates(
        subset=["available_ts"],
        keep="last",
    ).reset_index(drop=True)
    if report_start is not None:
        warmup_start = report_start - pd.Timedelta(days=5)
        signals = signals[signals["available_ts"] >= warmup_start].copy()
    if report_end is not None:
        signals = signals[signals["available_ts"] <= report_end].copy()
    if signals.empty:
        raise SystemExit("No signal rows after filtering.")
    one_min = _load_one_min(
        Path(args.one_min),
        start=pd.Timestamp(signals["timestamp"].min()) - pd.Timedelta(days=2),
        end=pd.Timestamp(signals["available_ts"].max()),
    ).sort_values("timestamp").reset_index(drop=True)

    events_frames: list[pd.DataFrame] = []
    trades_frames: list[pd.DataFrame] = []
    summary_frames: list[pd.DataFrame] = []
    for variant in TARGETED_VARIANTS:
        events, trades, summary = _replay_variant(
            variant=variant,
            signals=signals,
            one_min=one_min,
        )
        if not events.empty:
            events_frames.append(events)
        if not trades.empty:
            trades_frames.append(trades)
        summary_frames.append(summary)

    summary_df = pd.concat(summary_frames, ignore_index=True)
    events_df = pd.concat(events_frames, ignore_index=True) if events_frames else pd.DataFrame()
    trades_df = pd.concat(trades_frames, ignore_index=True) if trades_frames else pd.DataFrame()

    summary_out = Path(args.summary_out)
    trades_out = Path(args.trades_out)
    events_out = Path(args.events_out)
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    trades_out.parent.mkdir(parents=True, exist_ok=True)
    events_out.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_out, index=False)
    trades_df.to_csv(trades_out, index=False)
    events_df.to_csv(events_out, index=False)
    print(f"[sweep-soft-exits] wrote {summary_out}", flush=True)
    print(f"[sweep-soft-exits] wrote {trades_out}", flush=True)
    print(f"[sweep-soft-exits] wrote {events_out}", flush=True)


if __name__ == "__main__":
    main()
