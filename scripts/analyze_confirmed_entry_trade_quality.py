from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Policy.order_policy import (  # noqa: E402
    PHASE4_SWING_SETUP_BODYCLOSE_BODYCLOSE_V1,
    OptionOrderPolicy,
    OptionOrderPolicyConfig,
)
from Policy.replay_option_proxy import ReplayOptionPriceProxy, parse_occ_option_symbol  # noqa: E402
from scripts.compare_entry_overlay_policies import (  # noqa: E402
    DEFAULT_ANALYSIS_DIR,
    DEFAULT_ONE_MIN,
    DEFAULT_SIGNAL_FRAME,
    _load_one_min,
    _load_signal_frame,
)


DEFAULT_EVENTS_OUT = DEFAULT_ANALYSIS_DIR / "phase4_confirmed_entry_trade_quality_events.csv"
DEFAULT_TRADES_OUT = DEFAULT_ANALYSIS_DIR / "phase4_confirmed_entry_trade_quality_trades.csv"
DEFAULT_SUMMARY_OUT = DEFAULT_ANALYSIS_DIR / "phase4_confirmed_entry_trade_quality_summary.csv"
LIVE_RUNS_DIR = ROOT / "Data/inference/live_runs"


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _quiet_logger(_msg: str) -> None:
    return


def _extract_fill(wrapper: dict[str, Any], *, ts: pd.Timestamp, spot: float, policy: OptionOrderPolicy) -> dict[str, Any] | None:
    response = wrapper.get("response") if isinstance(wrapper, dict) else None
    if not isinstance(response, dict):
        return None
    payload = response.get("payload") if isinstance(response.get("payload"), dict) else {}
    order_like = response.get("response") if isinstance(response.get("response"), dict) else {}
    status = str(order_like.get("status") or "").strip().lower()
    if status != "filled":
        return None
    symbol = str(wrapper.get("symbol") or payload.get("symbol") or "").strip()
    if not symbol:
        return None
    filled_avg = _as_float(order_like.get("filled_avg_price"))
    if not math.isfinite(filled_avg):
        return None
    qty = int(round(_as_float(order_like.get("filled_qty") or payload.get("qty") or 0.0)))
    side = str(payload.get("side") or "").strip().lower()
    intent = str(payload.get("intent") or "").strip().lower()
    parsed = parse_occ_option_symbol(symbol)
    exposure = "long" if parsed and parsed.right == "C" else "short"
    structure = policy._meta_entry_structure.get(exposure) if side == "buy" else None
    event: dict[str, Any] = {
        "timestamp": ts,
        "symbol": symbol,
        "option_side": exposure,
        "order_side": side,
        "intent": intent,
        "qty": qty,
        "premium": float(filled_avg),
        "spot": float(spot) if math.isfinite(spot) else float("nan"),
    }
    if isinstance(structure, dict):
        setup_ts = structure.get("setup_ts")
        entry_ts = structure.get("entry_ts")
        signal_atr = _as_float(structure.get("signal_atr"))
        event.update(
            {
                "setup_ts": setup_ts,
                "entry_ts": entry_ts,
                "setup_ref": _as_float(structure.get("ref")),
                "setup_high": _as_float(structure.get("signal_high")),
                "setup_low": _as_float(structure.get("signal_low")),
                "setup_close": _as_float(structure.get("signal_close")),
                "setup_atr": signal_atr,
                "setup_prob": _as_float(structure.get("prob")),
                "setup_threshold": _as_float(structure.get("threshold")),
                "entry_kind": structure.get("entry_kind"),
                "source_side": structure.get("source_side"),
                "confirmation_delay_min": (
                    (pd.Timestamp(entry_ts) - pd.Timestamp(setup_ts)).total_seconds() / 60.0
                    if isinstance(entry_ts, datetime) and isinstance(setup_ts, datetime)
                    else float("nan")
                ),
            }
        )
    return event


def _pair_trades(events: pd.DataFrame, one_min: pd.DataFrame) -> pd.DataFrame:
    lots: dict[str, deque[dict[str, Any]]] = defaultdict(deque)
    trades: list[dict[str, Any]] = []
    if events.empty:
        return pd.DataFrame()
    one_min = one_min.sort_values("timestamp").reset_index(drop=True)
    for event in events.sort_values("timestamp").to_dict("records"):
        symbol = str(event.get("symbol") or "")
        if str(event.get("order_side")) == "buy":
            lots[symbol].append(event)
            continue
        if str(event.get("order_side")) != "sell" or not lots[symbol]:
            continue
        entry = lots[symbol].popleft()
        entry_ts = pd.Timestamp(entry["timestamp"])
        exit_ts = pd.Timestamp(event["timestamp"])
        path = one_min[(one_min["timestamp"] >= entry_ts) & (one_min["timestamp"] <= exit_ts)].copy()
        side = str(entry.get("option_side") or "")
        entry_spot = _as_float(entry.get("spot"))
        setup_atr = _as_float(entry.get("setup_atr"))
        if side == "long":
            mfe_spot = float(path["high"].max() - entry_spot) if not path.empty and math.isfinite(entry_spot) else float("nan")
            mae_spot = float(path["low"].min() - entry_spot) if not path.empty and math.isfinite(entry_spot) else float("nan")
        else:
            mfe_spot = float(entry_spot - path["low"].min()) if not path.empty and math.isfinite(entry_spot) else float("nan")
            mae_spot = float(entry_spot - path["high"].max()) if not path.empty and math.isfinite(entry_spot) else float("nan")
        trades.append(
            {
                "symbol": symbol,
                "side": side,
                "entry_time": entry_ts,
                "exit_time": exit_ts,
                "hold_minutes": (exit_ts - entry_ts).total_seconds() / 60.0,
                "entry_spot": entry_spot,
                "exit_spot": _as_float(event.get("spot")),
                "entry_premium": _as_float(entry.get("premium")),
                "exit_premium": _as_float(event.get("premium")),
                "return_pct": (
                    (_as_float(event.get("premium")) / _as_float(entry.get("premium")) - 1.0) * 100.0
                    if math.isfinite(_as_float(entry.get("premium"))) and _as_float(entry.get("premium")) > 0.0
                    else float("nan")
                ),
                "spot_move": (
                    _as_float(event.get("spot")) - entry_spot
                    if side == "long"
                    else entry_spot - _as_float(event.get("spot"))
                ),
                "mfe_spot": mfe_spot,
                "mae_spot": mae_spot,
                "mfe_atr": mfe_spot / setup_atr if math.isfinite(setup_atr) and setup_atr > 0.0 else float("nan"),
                "mae_atr": mae_spot / setup_atr if math.isfinite(setup_atr) and setup_atr > 0.0 else float("nan"),
                "setup_ts": entry.get("setup_ts"),
                "setup_ref": entry.get("setup_ref"),
                "setup_high": entry.get("setup_high"),
                "setup_low": entry.get("setup_low"),
                "setup_close": entry.get("setup_close"),
                "setup_atr": setup_atr,
                "setup_prob": entry.get("setup_prob"),
                "setup_threshold": entry.get("setup_threshold"),
                "confirmation_delay_min": entry.get("confirmation_delay_min"),
                "entry_kind": entry.get("entry_kind"),
                "source_side": entry.get("source_side"),
            }
        )
    return pd.DataFrame(trades)


def _summary_rows(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(
            [
                {
                    "bucket": "all",
                    "trades": 0,
                    "avg_return_pct": float("nan"),
                    "median_return_pct": float("nan"),
                    "win_rate": float("nan"),
                    "avg_confirmation_delay_min": float("nan"),
                    "avg_hold_minutes": float("nan"),
                    "avg_mfe_atr": float("nan"),
                    "avg_mae_atr": float("nan"),
                }
            ]
        )
    rows: list[dict[str, Any]] = []
    for bucket, frame in [("all", trades)] + list(trades.groupby("side")) + list(trades.groupby("entry_kind")):
        rows.append(
            {
                "bucket": bucket,
                "trades": int(len(frame)),
                "avg_return_pct": float(pd.to_numeric(frame["return_pct"], errors="coerce").mean()),
                "median_return_pct": float(pd.to_numeric(frame["return_pct"], errors="coerce").median()),
                "win_rate": float((pd.to_numeric(frame["return_pct"], errors="coerce") > 0).mean()),
                "avg_confirmation_delay_min": float(pd.to_numeric(frame["confirmation_delay_min"], errors="coerce").mean()),
                "avg_hold_minutes": float(pd.to_numeric(frame["hold_minutes"], errors="coerce").mean()),
                "avg_mfe_atr": float(pd.to_numeric(frame["mfe_atr"], errors="coerce").mean()),
                "avg_mae_atr": float(pd.to_numeric(frame["mae_atr"], errors="coerce").mean()),
            }
        )
    return pd.DataFrame(rows)


def _load_live_decision_signal_frame(
    *,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if not LIVE_RUNS_DIR.exists():
        return pd.DataFrame()
    for decision_path in sorted(LIVE_RUNS_DIR.glob("*_live_spy/decision-10m.jsonl")):
        run_name = decision_path.parent.name
        with decision_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                payload = record.get("payload") if isinstance(record, dict) else None
                if not isinstance(payload, dict):
                    continue
                bar = payload.get("bar")
                if not isinstance(bar, dict):
                    continue
                setup_ts = pd.to_datetime(bar.get("timestamp"), utc=True, errors="coerce")
                decision_ts = pd.to_datetime(payload.get("timestamp"), utc=True, errors="coerce")
                if pd.isna(setup_ts):
                    continue
                setup_ts = pd.Timestamp(setup_ts).tz_convert("America/New_York")
                available_ts = setup_ts + pd.Timedelta(minutes=10)
                if not pd.isna(decision_ts):
                    decision_ts = pd.Timestamp(decision_ts).tz_convert("America/New_York")
                if start is not None and available_ts < start - pd.Timedelta(days=5):
                    continue
                if end is not None and available_ts > end + pd.Timedelta(days=1):
                    continue
                rows.append(
                    {
                        "run": run_name,
                        "timestamp": setup_ts,
                        "available_ts": available_ts,
                        "decision_ts": decision_ts,
                        "open": _as_float(bar.get("open")),
                        "high": _as_float(bar.get("high")),
                        "low": _as_float(bar.get("low")),
                        "close": _as_float(bar.get("close")),
                        "ema_fast": float("nan"),
                        "ema_slow": float("nan"),
                        "atr_proxy": float("nan"),
                        "p_enter_long": _as_float(bar.get("p_enter_long")),
                        "p_enter_short": _as_float(bar.get("p_enter_short")),
                        "trend_regime": None,
                    }
                )
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows).sort_values(["available_ts", "run"]).drop_duplicates(
        subset=["available_ts"],
        keep="last",
    )
    return frame.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay confirmed-entry trade quality using the live order policy in simulated mode.")
    parser.add_argument("--signal-frame", default=str(DEFAULT_SIGNAL_FRAME))
    parser.add_argument("--one-min", default=str(DEFAULT_ONE_MIN))
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--events-out", default=str(DEFAULT_EVENTS_OUT))
    parser.add_argument("--trades-out", default=str(DEFAULT_TRADES_OUT))
    parser.add_argument("--summary-out", default=str(DEFAULT_SUMMARY_OUT))
    args = parser.parse_args()

    report_start = pd.Timestamp(args.start, tz="America/New_York") if args.start else None
    report_end = pd.Timestamp(args.end, tz="America/New_York") if args.end else None
    historical = _load_signal_frame(Path(args.signal_frame)).sort_values("timestamp").reset_index(drop=True)
    live = _load_live_decision_signal_frame(start=report_start, end=report_end)
    all_signals = pd.concat([historical, live], ignore_index=True, sort=False)
    if all_signals.empty:
        raise SystemExit("No signals available from historical frame or live decision logs.")
    all_signals = all_signals.sort_values("available_ts").drop_duplicates(
        subset=["available_ts"],
        keep="last",
    ).reset_index(drop=True)
    signals = all_signals
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
    )

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
            meta_setup_failure_exit_enabled=True,
            meta_setup_failure_buffer_atr=0.10,
            meta_no_progress_exit_enabled=True,
            meta_no_progress_exit_minutes=10,
            meta_no_progress_exit_atr=0.20,
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
            option_exit_quote_mode="bid",
        )
    )
    policy.set_contract_price_provider(proxy.price)

    signals = signals.sort_values("available_ts").reset_index(drop=True)
    one_min = one_min.sort_values("timestamp").reset_index(drop=True)

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
                spot=_as_float(row["close"]),
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
                    spot=_as_float(bar.get("close")),
                    policy=policy,
                )
                if event:
                    fills.append(event)
            idx_1m += 1

    events = pd.DataFrame(fills)
    if not events.empty:
        events = events.sort_values("timestamp").reset_index(drop=True)
    trades = _pair_trades(events, one_min)
    if report_start is not None and not events.empty:
        events = events[events["timestamp"] >= report_start].reset_index(drop=True)
    if report_end is not None and not events.empty:
        events = events[events["timestamp"] <= report_end].reset_index(drop=True)
    if report_start is not None and not trades.empty:
        trades = trades[trades["entry_time"] >= report_start].reset_index(drop=True)
    if report_end is not None and not trades.empty:
        trades = trades[trades["entry_time"] <= report_end].reset_index(drop=True)
    summary = _summary_rows(trades)

    events_out = Path(args.events_out)
    trades_out = Path(args.trades_out)
    summary_out = Path(args.summary_out)
    events_out.parent.mkdir(parents=True, exist_ok=True)
    trades_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    events.to_csv(events_out, index=False)
    trades.to_csv(trades_out, index=False)
    summary.to_csv(summary_out, index=False)

    print(
        {
            "events_out": str(events_out),
            "trades_out": str(trades_out),
            "summary_out": str(summary_out),
            "events": int(len(events)),
            "trades": int(len(trades)),
        }
    )


if __name__ == "__main__":
    main()
