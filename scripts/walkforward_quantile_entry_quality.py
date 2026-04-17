from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.walkforward_quantile_thresholds import DEFAULT_ANALYSIS_DIR, _load_signal_frame


DEFAULT_SIGNAL_FRAME = DEFAULT_ANALYSIS_DIR / "phase4_signal_frame.parquet"
DEFAULT_SUMMARY_OUT = DEFAULT_ANALYSIS_DIR / "phase4_walkforward_quantile_entry_quality_summary.csv"
DEFAULT_WEEKLY_OUT = DEFAULT_ANALYSIS_DIR / "phase4_walkforward_quantile_entry_quality_weekly.csv"
DEFAULT_EVENTS_OUT = DEFAULT_ANALYSIS_DIR / "phase4_walkforward_quantile_entry_quality_events.csv"


def _trading_days(frame: pd.DataFrame) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(frame["timestamp"]).dt.normalize().unique()).sort_values()


def _quantile_candidates(
    frame: pd.DataFrame,
    quantiles: list[float],
    col: str,
    *,
    floor: float,
    ceil: float,
) -> list[tuple[float, float]]:
    probs = pd.to_numeric(frame[col], errors="coerce").dropna()
    out: list[tuple[float, float]] = []
    seen: set[float] = set()
    for q in quantiles:
        value = float(probs.quantile(q))
        value = max(float(floor), min(float(ceil), value))
        key = round(value, 4)
        if np.isfinite(value) and key not in seen:
            seen.add(key)
            out.append((float(q), max(0.01, min(0.99, value))))
    return out


def _score_thresholds(
    frame: pd.DataFrame,
    *,
    long_thr: float,
    short_thr: float,
    horizon_bars: int,
    cooldown_bars: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = frame.reset_index(drop=True)
    events: list[dict[str, Any]] = []
    next_allowed = 0
    n = len(rows)
    closes = pd.to_numeric(rows["close"], errors="coerce").to_numpy(float)
    highs = pd.to_numeric(rows["high"], errors="coerce").to_numpy(float)
    lows = pd.to_numeric(rows["low"], errors="coerce").to_numpy(float)
    p_long = pd.to_numeric(rows["p_enter_long"], errors="coerce").to_numpy(float)
    p_short = pd.to_numeric(rows["p_enter_short"], errors="coerce").to_numpy(float)
    times = rows["timestamp"].to_numpy()
    tr = np.maximum(highs - lows, 0.01)
    atr = pd.Series(tr).rolling(20, min_periods=3).mean().to_numpy(float)
    for i in range(n):
        if i < next_allowed:
            continue
        if not (np.isfinite(closes[i]) and np.isfinite(p_long[i]) and np.isfinite(p_short[i])):
            continue
        long_ready = p_long[i] >= long_thr
        short_ready = p_short[i] >= short_thr
        if not (long_ready or short_ready):
            continue
        long_margin = p_long[i] - long_thr if long_ready else -np.inf
        short_margin = p_short[i] - short_thr if short_ready else -np.inf
        side = "long" if long_margin >= short_margin else "short"
        j = min(n - 1, i + int(horizon_bars))
        if j <= i:
            continue
        ref_atr = atr[i] if np.isfinite(atr[i]) and atr[i] > 0 else max(float(tr[i]), 0.01)
        raw_move = closes[j] - closes[i]
        ev_atr = raw_move / ref_atr if side == "long" else -raw_move / ref_atr
        events.append(
            {
                "entry_time": pd.Timestamp(times[i]),
                "exit_time": pd.Timestamp(times[j]),
                "side": side,
                "entry_price": closes[i],
                "exit_price": closes[j],
                "p_enter_long": p_long[i],
                "p_enter_short": p_short[i],
                "long_threshold": long_thr,
                "short_threshold": short_thr,
                "horizon_bars": int(j - i),
                "outcome_atr": float(ev_atr),
            }
        )
        next_allowed = i + max(1, int(cooldown_bars))
    outcomes = np.array([e["outcome_atr"] for e in events], dtype=float)
    metrics = {
        "trades": int(len(events)),
        "sum_ev_atr": float(np.nansum(outcomes)) if len(outcomes) else 0.0,
        "mean_ev_atr": float(np.nanmean(outcomes)) if len(outcomes) else float("nan"),
        "median_ev_atr": float(np.nanmedian(outcomes)) if len(outcomes) else float("nan"),
        "win_rate": float(np.nanmean(outcomes > 0)) if len(outcomes) else float("nan"),
        "long_trades": int(sum(1 for e in events if e["side"] == "long")),
        "short_trades": int(sum(1 for e in events if e["side"] == "short")),
    }
    return metrics, events


def _select_week_thresholds(
    cal: pd.DataFrame,
    *,
    quantiles: list[float],
    horizon_bars: int,
    cooldown_bars: int,
    min_cal_trades: int,
    long_floor: float,
    short_floor: float,
    long_ceil: float,
    short_ceil: float,
) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    long_candidates = _quantile_candidates(cal, quantiles, "p_enter_long", floor=long_floor, ceil=long_ceil)
    short_candidates = _quantile_candidates(cal, quantiles, "p_enter_short", floor=short_floor, ceil=short_ceil)
    for lq, lthr in long_candidates:
        for sq, sthr in short_candidates:
            metrics, _ = _score_thresholds(
                cal,
                long_thr=lthr,
                short_thr=sthr,
                horizon_bars=horizon_bars,
                cooldown_bars=cooldown_bars,
            )
            row = {
                "long_quantile": lq,
                "short_quantile": sq,
                "long_threshold": lthr,
                "short_threshold": sthr,
                **metrics,
            }
            if row["trades"] < min_cal_trades:
                continue
            if best is None or (
                row["sum_ev_atr"],
                row["mean_ev_atr"],
                row["win_rate"],
            ) > (
                best["sum_ev_atr"],
                best["mean_ev_atr"],
                best["win_rate"],
            ):
                best = row
    if best is None:
        raise ValueError("No candidate met min_cal_trades.")
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description="Fast weekly walk-forward quantile threshold entry-quality sweep.")
    parser.add_argument("--signal-frame", default=str(DEFAULT_SIGNAL_FRAME))
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--lookback-days", type=int, default=30)
    parser.add_argument("--quantiles", default="0.60,0.70,0.75,0.80,0.85,0.90,0.95")
    parser.add_argument("--horizon-bars", type=int, default=12)
    parser.add_argument("--cooldown-bars", type=int, default=4)
    parser.add_argument("--min-cal-trades", type=int, default=5)
    parser.add_argument("--long-floor", type=float, default=0.01)
    parser.add_argument("--short-floor", type=float, default=0.01)
    parser.add_argument("--long-ceil", type=float, default=0.99)
    parser.add_argument("--short-ceil", type=float, default=0.99)
    parser.add_argument("--summary-out", default=str(DEFAULT_SUMMARY_OUT))
    parser.add_argument("--weekly-out", default=str(DEFAULT_WEEKLY_OUT))
    parser.add_argument("--events-out", default=str(DEFAULT_EVENTS_OUT))
    args = parser.parse_args()

    quantiles = [float(x.strip()) for x in str(args.quantiles).split(",") if x.strip()]
    frame = _load_signal_frame(Path(args.signal_frame), args.start, args.end)
    days = _trading_days(frame)
    week_starts = pd.date_range(days[int(args.lookback_days)], days[-1], freq="W-MON", tz=days.tz)
    starts = []
    for anchor in week_starts:
        future = days[days >= anchor]
        if len(future):
            start = pd.Timestamp(future[0])
            if start not in starts:
                starts.append(start)
    if not starts:
        raise SystemExit("No walk-forward weeks.")

    weekly_rows: list[dict[str, Any]] = []
    wf_events: list[dict[str, Any]] = []
    for i, week_start in enumerate(starts):
        week_end = starts[i + 1] if i + 1 < len(starts) else frame["timestamp"].max() + pd.Timedelta(minutes=10)
        prior_days = days[days < week_start][-int(args.lookback_days) :]
        if len(prior_days) < int(args.lookback_days):
            continue
        cal_start = pd.Timestamp(prior_days[0])
        cal = frame[(frame["timestamp"] >= cal_start) & (frame["timestamp"] < week_start)]
        week = frame[(frame["timestamp"] >= week_start) & (frame["timestamp"] < week_end)]
        if cal.empty or week.empty:
            continue
        selected = _select_week_thresholds(
            cal,
            quantiles=quantiles,
            horizon_bars=int(args.horizon_bars),
            cooldown_bars=int(args.cooldown_bars),
            min_cal_trades=int(args.min_cal_trades),
            long_floor=float(args.long_floor),
            short_floor=float(args.short_floor),
            long_ceil=float(args.long_ceil),
            short_ceil=float(args.short_ceil),
        )
        week_metrics, week_events = _score_thresholds(
            week,
            long_thr=float(selected["long_threshold"]),
            short_thr=float(selected["short_threshold"]),
            horizon_bars=int(args.horizon_bars),
            cooldown_bars=int(args.cooldown_bars),
        )
        weekly_rows.append(
            {
                "week_start": week_start,
                "week_end": week_end,
                "cal_start": cal_start,
                "long_quantile": selected["long_quantile"],
                "short_quantile": selected["short_quantile"],
                "long_threshold": selected["long_threshold"],
                "short_threshold": selected["short_threshold"],
                "cal_trades": selected["trades"],
                "cal_sum_ev_atr": selected["sum_ev_atr"],
                "cal_mean_ev_atr": selected["mean_ev_atr"],
                **{f"week_{k}": v for k, v in week_metrics.items()},
            }
        )
        for event in week_events:
            wf_events.append({"week_start": week_start, **event})

    eval_start = starts[0]
    eval_frame = frame[frame["timestamp"] >= eval_start]
    summary_rows = []
    wf_metrics = {
        "trades": len(wf_events),
        "sum_ev_atr": float(np.nansum([e["outcome_atr"] for e in wf_events])) if wf_events else 0.0,
        "mean_ev_atr": float(np.nanmean([e["outcome_atr"] for e in wf_events])) if wf_events else float("nan"),
        "median_ev_atr": float(np.nanmedian([e["outcome_atr"] for e in wf_events])) if wf_events else float("nan"),
        "win_rate": float(np.nanmean([e["outcome_atr"] > 0 for e in wf_events])) if wf_events else float("nan"),
        "long_trades": int(sum(1 for e in wf_events if e["side"] == "long")),
        "short_trades": int(sum(1 for e in wf_events if e["side"] == "short")),
    }
    summary_rows.append({"regime": "walkforward_quantile", **wf_metrics})
    for name, lthr, sthr in [
        ("static_current_035_065", 0.35, 0.65),
        ("static_old_042_015", 0.42, 0.15),
        ("static_sym_050_050", 0.50, 0.50),
        ("static_long035_short050", 0.35, 0.50),
    ]:
        metrics, _ = _score_thresholds(
            eval_frame,
            long_thr=lthr,
            short_thr=sthr,
            horizon_bars=int(args.horizon_bars),
            cooldown_bars=int(args.cooldown_bars),
        )
        summary_rows.append({"regime": name, "long_threshold": lthr, "short_threshold": sthr, **metrics})

    summary_df = pd.DataFrame(summary_rows)
    weekly_df = pd.DataFrame(weekly_rows)
    events_df = pd.DataFrame(wf_events)
    for out in [Path(args.summary_out), Path(args.weekly_out), Path(args.events_out)]:
        out.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(args.summary_out, index=False)
    weekly_df.to_csv(args.weekly_out, index=False)
    events_df.to_csv(args.events_out, index=False)
    print(
        f"[entry-quality] rows={len(frame):,} weeks={len(weekly_df):,} eval_start={eval_start} "
        f"horizon_bars={int(args.horizon_bars)}"
    )
    print(summary_df.sort_values("sum_ev_atr", ascending=False).to_string(index=False))
    print("[entry-quality] weekly tail")
    print(
        weekly_df[
            [
                "week_start",
                "long_quantile",
                "short_quantile",
                "long_threshold",
                "short_threshold",
                "week_trades",
                "week_sum_ev_atr",
                "week_win_rate",
            ]
        ]
        .tail(12)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
