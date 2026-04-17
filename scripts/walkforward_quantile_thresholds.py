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

from scripts.sweep_live_thresholds_post_0401 import _load_one_min, _metrics, _run_one


DEFAULT_ANALYSIS_DIR = Path(
    "Data/models/ga_xgboost/10min/analysis/"
    "phase4_1m_oof_focused_trigger_sweep_l42_s15_full_1m_train"
)
DEFAULT_SIGNAL_FRAME = DEFAULT_ANALYSIS_DIR / "phase4_signal_frame.parquet"
DEFAULT_ONE_MIN = Path("Data/raw/spy/1m_train.parquet")
DEFAULT_SUMMARY_OUT = DEFAULT_ANALYSIS_DIR / "phase4_walkforward_quantile_threshold_summary.csv"
DEFAULT_WEEKLY_OUT = DEFAULT_ANALYSIS_DIR / "phase4_walkforward_quantile_threshold_weekly.csv"
DEFAULT_EVENTS_OUT = DEFAULT_ANALYSIS_DIR / "phase4_walkforward_quantile_threshold_events.csv"


def _to_et_index(values: Any) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(pd.to_datetime(values, utc=True, errors="coerce"))
    return idx.tz_convert("America/New_York")


def _load_signal_frame(path: Path, start: str | None, end: str | None) -> pd.DataFrame:
    df = pd.read_parquet(path).copy()
    if "timestamp" in df.columns:
        idx = _to_et_index(df["timestamp"])
    else:
        idx = _to_et_index(df.index)
    df.index = idx
    df = df[~df.index.isna()].sort_index()

    long_prob = df.get("p_long_test")
    if long_prob is None:
        long_prob = pd.Series(np.nan, index=df.index)
    long_prob = pd.to_numeric(long_prob, errors="coerce").combine_first(
        pd.to_numeric(df.get("p_long_oof_train"), errors="coerce")
    )
    short_prob = df.get("p_short_test")
    if short_prob is None:
        short_prob = pd.Series(np.nan, index=df.index)
    short_prob = pd.to_numeric(short_prob, errors="coerce").combine_first(
        pd.to_numeric(df.get("p_short_oof_train"), errors="coerce")
    )
    out = pd.DataFrame(
        {
            "timestamp": df.index,
            "available_ts": df.index + pd.Timedelta(minutes=10),
            "open": pd.to_numeric(df["open"], errors="coerce"),
            "high": pd.to_numeric(df["high"], errors="coerce"),
            "low": pd.to_numeric(df["low"], errors="coerce"),
            "close": pd.to_numeric(df["close"], errors="coerce"),
            "atr": pd.to_numeric(df.get("atr"), errors="coerce") if "atr" in df.columns else np.nan,
            "p_enter_long": long_prob,
            "p_enter_short": short_prob,
        }
    ).dropna(subset=["timestamp", "open", "high", "low", "close", "p_enter_long", "p_enter_short"])
    if start:
        start_ts = pd.Timestamp(start)
        if start_ts.tzinfo is None:
            start_ts = start_ts.tz_localize("America/New_York")
        else:
            start_ts = start_ts.tz_convert("America/New_York")
        out = out[out["timestamp"] >= start_ts]
    if end:
        end_ts = pd.Timestamp(end)
        if end_ts.tzinfo is None:
            end_ts = end_ts.tz_localize("America/New_York")
        else:
            end_ts = end_ts.tz_convert("America/New_York")
        out = out[out["timestamp"] <= end_ts]
    return out.reset_index(drop=True)


def _trading_days(decisions: pd.DataFrame) -> list[pd.Timestamp]:
    days = pd.DatetimeIndex(pd.to_datetime(decisions["timestamp"]).dt.normalize().unique()).sort_values()
    return [pd.Timestamp(day) for day in days]


def _week_starts(days: list[pd.Timestamp], lookback_days: int) -> list[pd.Timestamp]:
    if len(days) <= lookback_days:
        return []
    first = days[lookback_days]
    last = days[-1]
    # Use Monday anchors; if Monday is not a trading day, the week uses the
    # first available trading day >= that anchor.
    anchors = pd.date_range(first.normalize(), last.normalize(), freq="W-MON", tz=first.tz)
    starts: list[pd.Timestamp] = []
    day_index = pd.DatetimeIndex(days)
    for anchor in anchors:
        future = day_index[day_index >= anchor]
        if len(future) == 0:
            continue
        start = pd.Timestamp(future[0])
        if start not in starts:
            starts.append(start)
    if starts and starts[0] > first:
        starts.insert(0, first)
    elif not starts:
        starts.append(first)
    return starts


def _slice_by_ts(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    return df[(df["timestamp"] >= start) & (df["timestamp"] < end)].reset_index(drop=True)


def _candidate_thresholds(window: pd.DataFrame, quantiles: list[float], *, side: str) -> list[tuple[float, float]]:
    col = "p_enter_long" if side == "long" else "p_enter_short"
    probs = pd.to_numeric(window[col], errors="coerce").dropna()
    out: list[tuple[float, float]] = []
    for q in quantiles:
        if probs.empty:
            continue
        value = float(probs.quantile(q))
        if np.isfinite(value):
            out.append((float(q), max(0.01, min(0.99, value))))
    # Preserve quantile order while de-duping rounded threshold values.
    seen: set[float] = set()
    deduped: list[tuple[float, float]] = []
    for q, value in out:
        key = round(value, 4)
        if key in seen:
            continue
        seen.add(key)
        deduped.append((q, value))
    return deduped


def _select_best_thresholds(
    *,
    cal_decisions: pd.DataFrame,
    cal_one_min: pd.DataFrame,
    quantiles: list[float],
    setup_max_bars: int,
    cutoff_hhmm: str,
    min_cal_trades: int,
) -> dict[str, Any]:
    long_candidates = _candidate_thresholds(cal_decisions, quantiles, side="long")
    short_candidates = _candidate_thresholds(cal_decisions, quantiles, side="short")
    rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for long_q, long_thr in long_candidates:
        for short_q, short_thr in short_candidates:
            events = _run_one(
                decisions=cal_decisions,
                one_min=cal_one_min,
                long_thr=float(long_thr),
                short_thr=float(short_thr),
                setup_max_bars=setup_max_bars,
                cutoff_hhmm=cutoff_hhmm,
            )
            metrics = _metrics(events)
            row = {
                "long_quantile": long_q,
                "short_quantile": short_q,
                "long_threshold": long_thr,
                "short_threshold": short_thr,
                **metrics,
            }
            rows.append(row)
            if int(metrics["trades"]) < int(min_cal_trades):
                continue
            if best is None:
                best = row
                continue
            key = (row["sum_return"], row["avg_return"], row["win_rate"], -abs(row["trades"] - 8))
            best_key = (
                best["sum_return"],
                best["avg_return"],
                best["win_rate"],
                -abs(best["trades"] - 8),
            )
            if key > best_key:
                best = row
    if best is None and rows:
        best = sorted(rows, key=lambda r: (r["sum_return"], r["avg_return"], r["trades"]), reverse=True)[0]
    if best is None:
        raise ValueError("No threshold candidates available for calibration window.")
    best = dict(best)
    best["candidate_count"] = len(rows)
    return best


def _run_static(
    *,
    decisions: pd.DataFrame,
    one_min: pd.DataFrame,
    long_thr: float,
    short_thr: float,
    setup_max_bars: int,
    cutoff_hhmm: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    events = _run_one(
        decisions=decisions,
        one_min=one_min,
        long_thr=long_thr,
        short_thr=short_thr,
        setup_max_bars=setup_max_bars,
        cutoff_hhmm=cutoff_hhmm,
    )
    return _metrics(events), events


def main() -> None:
    parser = argparse.ArgumentParser(description="Weekly walk-forward quantile threshold experiment.")
    parser.add_argument("--signal-frame", default=str(DEFAULT_SIGNAL_FRAME))
    parser.add_argument("--one-min", default=str(DEFAULT_ONE_MIN))
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--lookback-days", type=int, default=30)
    parser.add_argument("--quantiles", default="0.70,0.85,0.95")
    parser.add_argument("--min-cal-trades", type=int, default=3)
    parser.add_argument("--setup-max-bars", type=int, default=4)
    parser.add_argument("--cutoff-hhmm", default="13:00")
    parser.add_argument("--summary-out", default=str(DEFAULT_SUMMARY_OUT))
    parser.add_argument("--weekly-out", default=str(DEFAULT_WEEKLY_OUT))
    parser.add_argument("--events-out", default=str(DEFAULT_EVENTS_OUT))
    args = parser.parse_args()

    quantiles = [float(x.strip()) for x in str(args.quantiles).split(",") if x.strip()]
    decisions = _load_signal_frame(Path(args.signal_frame), args.start, args.end)
    if decisions.empty:
        raise SystemExit("No signal rows loaded.")
    all_days = _trading_days(decisions)
    starts = _week_starts(all_days, int(args.lookback_days))
    if not starts:
        raise SystemExit("Not enough trading days for walk-forward.")
    eval_start = starts[0]
    eval_end = decisions["timestamp"].max() + pd.Timedelta(minutes=10)
    one_min = _load_one_min(Path(args.one_min), str(decisions["timestamp"].min()), decisions["timestamp"].max())
    one_min_eval = one_min[(one_min["timestamp"] >= eval_start) & (one_min["timestamp"] <= eval_end)].reset_index(drop=True)

    weekly_rows: list[dict[str, Any]] = []
    wf_events: list[dict[str, Any]] = []
    day_index = pd.DatetimeIndex(all_days)

    for i, week_start in enumerate(starts):
        next_start = starts[i + 1] if i + 1 < len(starts) else eval_end
        prior_days = day_index[day_index < week_start][-int(args.lookback_days) :]
        if len(prior_days) < int(args.lookback_days):
            continue
        cal_start = pd.Timestamp(prior_days[0])
        cal_end = week_start
        cal_decisions = _slice_by_ts(decisions, cal_start, cal_end)
        week_decisions = _slice_by_ts(decisions, week_start, pd.Timestamp(next_start))
        if cal_decisions.empty or week_decisions.empty:
            continue
        cal_one_min = _slice_by_ts(one_min, cal_start, cal_end)
        week_one_min = _slice_by_ts(one_min, week_start, pd.Timestamp(next_start))
        if cal_one_min.empty or week_one_min.empty:
            continue

        selected = _select_best_thresholds(
            cal_decisions=cal_decisions,
            cal_one_min=cal_one_min,
            quantiles=quantiles,
            setup_max_bars=int(args.setup_max_bars),
            cutoff_hhmm=str(args.cutoff_hhmm),
            min_cal_trades=int(args.min_cal_trades),
        )
        if (len(weekly_rows) + 1) % 25 == 0:
            print(
                f"[walkforward] calibrated {len(weekly_rows) + 1} weeks through {week_start.date()}",
                flush=True,
            )
        week_events = _run_one(
            decisions=week_decisions,
            one_min=week_one_min,
            long_thr=float(selected["long_threshold"]),
            short_thr=float(selected["short_threshold"]),
            setup_max_bars=int(args.setup_max_bars),
            cutoff_hhmm=str(args.cutoff_hhmm),
        )
        week_metrics = _metrics(week_events)
        weekly_rows.append(
            {
                "week_start": week_start,
                "week_end": next_start,
                "cal_start": cal_start,
                "cal_end": cal_end,
                "cal_trades": selected["trades"],
                "cal_sum_return": selected["sum_return"],
                "cal_avg_return": selected["avg_return"],
                "cal_win_rate": selected["win_rate"],
                "long_quantile": selected["long_quantile"],
                "short_quantile": selected["short_quantile"],
                "long_threshold": selected["long_threshold"],
                "short_threshold": selected["short_threshold"],
                "candidate_count": selected["candidate_count"],
                **{f"week_{k}": v for k, v in week_metrics.items()},
            }
        )
        for event in week_events:
            wf_events.append(
                {
                    "week_start": week_start,
                    "long_threshold": selected["long_threshold"],
                    "short_threshold": selected["short_threshold"],
                    "long_quantile": selected["long_quantile"],
                    "short_quantile": selected["short_quantile"],
                    **event,
                }
            )

    if not weekly_rows:
        raise SystemExit("No weekly walk-forward rows produced.")

    weekly_df = pd.DataFrame(weekly_rows)
    events_df = pd.DataFrame(wf_events)
    wf_metrics = _metrics(wf_events)
    static_rows: list[dict[str, Any]] = [
        {"regime": "walkforward_quantile", **wf_metrics},
    ]
    eval_decisions = decisions[decisions["timestamp"] >= eval_start].reset_index(drop=True)
    static_pairs = [
        ("static_current_035_065", 0.35, 0.65),
        ("static_old_042_015", 0.42, 0.15),
        ("static_sym_050_050", 0.50, 0.50),
        ("static_long035_short050", 0.35, 0.50),
    ]
    for name, long_thr, short_thr in static_pairs:
        metrics, _events = _run_static(
            decisions=eval_decisions,
            one_min=one_min_eval,
            long_thr=long_thr,
            short_thr=short_thr,
            setup_max_bars=int(args.setup_max_bars),
            cutoff_hhmm=str(args.cutoff_hhmm),
        )
        static_rows.append({"regime": name, "long_threshold": long_thr, "short_threshold": short_thr, **metrics})

    summary_df = pd.DataFrame(static_rows)
    summary_df["eval_start"] = eval_start
    summary_df["eval_end"] = eval_end
    summary_df["weeks"] = len(weekly_df)
    summary_df["lookback_days"] = int(args.lookback_days)
    summary_df["quantiles"] = ",".join(str(q) for q in quantiles)

    summary_out = Path(args.summary_out)
    weekly_out = Path(args.weekly_out)
    events_out = Path(args.events_out)
    for path in (summary_out, weekly_out, events_out):
        path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_out, index=False)
    weekly_df.to_csv(weekly_out, index=False)
    events_df.to_csv(events_out, index=False)

    print(
        f"[walkforward] decisions={len(decisions):,} one_min={len(one_min):,} "
        f"weeks={len(weekly_df):,} eval={eval_start}..{eval_end}"
    )
    print(f"[walkforward] wrote summary={summary_out}")
    print(summary_df.sort_values("sum_return", ascending=False).to_string(index=False))
    print("[walkforward] weekly threshold tail")
    print(
        weekly_df[
            [
                "week_start",
                "long_quantile",
                "short_quantile",
                "long_threshold",
                "short_threshold",
                "week_trades",
                "week_sum_return",
                "week_win_rate",
            ]
        ]
        .tail(12)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
