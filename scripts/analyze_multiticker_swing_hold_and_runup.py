from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from scripts.analyze_multiticker_swing_20260608_forensics import build_trade_table, parse_audit


ET = ZoneInfo("America/New_York")
DEFAULT_OUT = Path("UI/swing_audit/forensics_20260608")
RAW_5M_DIR = Path("strategies/multi_ticker_swing/data/raw/5m")
FRIDAY_AUDIT = Path("UI/swing_audit/paper/swing_session_20260605T122235Z.jsonl")
MONDAY_AUDIT = Path("UI/swing_audit/paper/swing_session_20260608T121654Z.jsonl")


def _num(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _ts(value: Any) -> pd.Timestamp:
    return pd.to_datetime(value, utc=True, errors="coerce")


def parse_expiry(symbol: str) -> pd.Timestamp | None:
    if not symbol:
        return None
    text = str(symbol).upper()
    digits = ""
    for ch in text:
        if ch.isdigit():
            digits += ch
            if len(digits) == 6:
                break
        else:
            digits = ""
    if len(digits) != 6:
        return None
    try:
        return pd.Timestamp(f"20{digits[:2]}-{digits[2:4]}-{digits[4:6]}", tz=ET)
    except ValueError:
        return None


def load_raw_5m(ticker: str) -> pd.DataFrame:
    path = RAW_5M_DIR / f"{ticker.upper()}.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    if "timestamp" not in df.columns and df.index.name:
        df = df.reset_index()
    df.columns = [str(c).lower() for c in df.columns]
    if "timestamp" not in df.columns:
        return pd.DataFrame()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["timestamp", "close"]).sort_values("timestamp")


def _mark_rows(audit: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    with audit.open("r", encoding="utf-8") as fh:
        for line in fh:
            event = json.loads(line)
            if event.get("type") != "position_bar_5m":
                continue
            payload = event.get("payload") or {}
            pos = payload.get("position") or {}
            bar = payload.get("bar") or {}
            rows.append(
                {
                    "event_ts": _ts(event.get("ts")),
                    "bar_ts": pd.to_datetime(bar.get("time"), unit="s", utc=True, errors="coerce"),
                    "ticker": str(payload.get("ticker") or pos.get("ticker") or "").upper(),
                    "direction": int(pos.get("direction") or 0),
                    "entry_time": _ts(pos.get("entry_time")),
                    "entry_price": _num(pos.get("entry_price")),
                    "underlying_close": _num(bar.get("close")),
                    "option_entry_price": _num(pos.get("option_entry_price")),
                    "option_last_price": _num(pos.get("option_last_price")),
                    "underlying_pnl_pct": _num(pos.get("pnl_pct")),
                }
            )
    return pd.DataFrame(rows)


def friday_to_monday(friday: Path, monday: Path) -> pd.DataFrame:
    fri_parsed = parse_audit(friday)
    mon_parsed = parse_audit(monday)
    fri_trades = build_trade_table(fri_parsed)
    mon_trades = build_trade_table(mon_parsed)
    fri_marks = _mark_rows(friday)
    mon_marks = _mark_rows(monday)

    late_fri = fri_trades[
        (~fri_trades["restored"].fillna(False))
        & pd.to_datetime(fri_trades["entry_time"], utc=True, errors="coerce").dt.tz_convert(ET).dt.weekday.eq(4)
    ].copy()
    late_fri["entry_time_et"] = pd.to_datetime(late_fri["entry_time"], utc=True).dt.tz_convert(ET)
    late_fri = late_fri[late_fri["entry_time_et"].dt.time >= pd.Timestamp("15:00", tz=ET).time()].copy()

    rows: list[dict[str, Any]] = []
    for _, trade in late_fri.iterrows():
        symbol = trade["symbol"]
        ticker = trade["ticker"]
        direction = int(trade["direction"])
        entry_time = _ts(trade["entry_time"])
        fri_path = fri_marks[
            (fri_marks["ticker"].eq(ticker))
            & (fri_marks["direction"].eq(direction))
            & (fri_marks["entry_time"].eq(entry_time))
        ].sort_values("bar_ts")
        mon_row = mon_trades[mon_trades["symbol"].astype(str).eq(str(symbol))]
        mon_path = mon_marks[
            (mon_marks["ticker"].eq(ticker))
            & (mon_marks["direction"].eq(direction))
            & (mon_marks["entry_time"].eq(entry_time))
        ].sort_values("bar_ts")
        fri_last = fri_path.iloc[-1] if not fri_path.empty else None
        mon_first = mon_path.iloc[0] if not mon_path.empty else None
        mon_last = mon_path.iloc[-1] if not mon_path.empty else None
        if mon_row.empty and mon_first is None:
            continue
        exp = parse_expiry(symbol)
        entry_opt = _num(trade.get("entry_option"))
        fri_opt = _num(fri_last.get("option_last_price")) if fri_last is not None else _num(trade.get("exit_option_or_mark"))
        mon_open_opt = _num(mon_first.get("option_last_price")) if mon_first is not None else None
        mon_last_opt = _num(mon_last.get("option_last_price")) if mon_last is not None else None
        if mon_last_opt is None and not mon_row.empty:
            mon_last_opt = _num(mon_row.iloc[0].get("exit_option_or_mark"))
        rows.append(
            {
                "ticker": ticker,
                "symbol": symbol,
                "side": trade["side"],
                "entry_time_et": trade["entry_time_et"],
                "expiry": None if exp is None else exp.date().isoformat(),
                "dte_entry": None if exp is None else (exp.date() - trade["entry_time_et"].date()).days,
                "entry_option": entry_opt,
                "friday_last_option": fri_opt,
                "monday_first_option": mon_open_opt,
                "monday_last_or_exit_option": mon_last_opt,
                "fri_to_mon_open_ret_pct": None
                if not fri_opt or mon_open_opt is None
                else (mon_open_opt / fri_opt - 1.0) * 100.0,
                "entry_to_mon_last_ret_pct": None
                if not entry_opt or mon_last_opt is None
                else (mon_last_opt / entry_opt - 1.0) * 100.0,
                "friday_underlying_mark_pct": None if fri_last is None else (_num(fri_last.get("underlying_pnl_pct")) or 0.0) * 100.0,
                "monday_underlying_mark_pct": None
                if mon_last is None
                else (_num(mon_last.get("underlying_pnl_pct")) or 0.0) * 100.0,
                "monday_status": None if mon_row.empty else mon_row.iloc[0].get("status"),
            }
        )
    return pd.DataFrame(rows)


def add_runup_metrics(trades: pd.DataFrame) -> pd.DataFrame:
    out = trades.copy()
    blank_metrics = {
        "signed_run_from_day_open_pct": None,
        "abs_run_from_day_open_pct": None,
        "signed_run_30m_pct": None,
        "signed_run_60m_pct": None,
        "signed_run_120m_pct": None,
        "prev_day_signed_ret_pct": None,
        "entry_vs_day_high_pct": None,
        "entry_vs_day_low_pct": None,
    }
    metrics: list[dict[str, Any]] = []
    for _, row in out.iterrows():
        ticker = row["ticker"]
        bars = load_raw_5m(ticker)
        entry = _ts(row["entry_time"])
        direction = int(row["direction"])
        if bars.empty or pd.isna(entry):
            metrics.append(blank_metrics.copy())
            continue
        bars_et = bars.copy()
        bars_et["ts_et"] = bars_et["timestamp"].dt.tz_convert(ET)
        entry_et = entry.tz_convert(ET)
        day = bars_et[bars_et["ts_et"].dt.date.eq(entry_et.date())].copy()
        before = day[day["timestamp"] <= entry].copy()
        if before.empty:
            metrics.append(blank_metrics.copy())
            continue
        day_open = _num(day.iloc[0].get("open"))
        entry_close = _num(before.iloc[-1].get("close"))
        prior_30 = before[before["timestamp"] >= entry - pd.Timedelta(minutes=30)]
        prior_60 = before[before["timestamp"] >= entry - pd.Timedelta(minutes=60)]
        prior_120 = before[before["timestamp"] >= entry - pd.Timedelta(minutes=120)]
        prior_day = bars_et[bars_et["ts_et"].dt.date.lt(entry_et.date())].tail(78)
        def signed_ret(start: float | None, end: float | None) -> float | None:
            if not start or end is None:
                return None
            return direction * (end / start - 1.0) * 100.0
        metrics.append(
            {
                "signed_run_from_day_open_pct": signed_ret(day_open, entry_close),
                "abs_run_from_day_open_pct": None if not day_open or entry_close is None else (entry_close / day_open - 1.0) * 100.0,
                "signed_run_30m_pct": signed_ret(_num(prior_30.iloc[0].get("open")) if not prior_30.empty else None, entry_close),
                "signed_run_60m_pct": signed_ret(_num(prior_60.iloc[0].get("open")) if not prior_60.empty else None, entry_close),
                "signed_run_120m_pct": signed_ret(_num(prior_120.iloc[0].get("open")) if not prior_120.empty else None, entry_close),
                "prev_day_signed_ret_pct": signed_ret(
                    _num(prior_day.iloc[0].get("open")) if not prior_day.empty else None,
                    _num(prior_day.iloc[-1].get("close")) if not prior_day.empty else None,
                ),
                "entry_vs_day_high_pct": None if entry_close is None else (entry_close / before["high"].max() - 1.0) * 100.0,
                "entry_vs_day_low_pct": None if entry_close is None else (entry_close / before["low"].min() - 1.0) * 100.0,
            }
        )
    return pd.concat([out.reset_index(drop=True), pd.DataFrame(metrics)], axis=1)


def audit_runup_metrics(audit: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    with audit.open("r", encoding="utf-8") as fh:
        for line in fh:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("type") != "position_chart_seed":
                continue
            payload = event.get("payload") or {}
            bars = payload.get("pre_entry_bars") or []
            direction = int(payload.get("direction") or 0)
            entry_price = _num(payload.get("entry_price"))
            raw_entry_time = payload.get("entry_time")
            entry_time = (
                pd.to_datetime(raw_entry_time, unit="s", utc=True, errors="coerce")
                if isinstance(raw_entry_time, (int, float))
                else _ts(raw_entry_time)
            )
            if not bars or not direction or entry_price is None or pd.isna(entry_time):
                continue
            frame = pd.DataFrame(bars)
            frame["timestamp"] = pd.to_datetime(frame["time"], unit="s", utc=True, errors="coerce")
            frame = frame.dropna(subset=["timestamp"]).sort_values("timestamp")
            if frame.empty:
                continue
            for col in ["open", "high", "low", "close"]:
                frame[col] = pd.to_numeric(frame[col], errors="coerce")
            before = frame[frame["timestamp"] <= entry_time]
            if before.empty:
                before = frame

            def signed_ret(start: float | None, end: float | None) -> float | None:
                if not start or end is None:
                    return None
                return direction * (end / start - 1.0) * 100.0

            prior_30 = before[before["timestamp"] >= entry_time - pd.Timedelta(minutes=30)]
            prior_60 = before[before["timestamp"] >= entry_time - pd.Timedelta(minutes=60)]
            prior_120 = before[before["timestamp"] >= entry_time - pd.Timedelta(minutes=120)]
            day_open = _num(before.iloc[0].get("open"))
            rows.append(
                {
                    "ticker": payload.get("ticker"),
                    "direction": direction,
                    "entry_time": entry_time,
                    "symbol": payload.get("option_symbol"),
                    "audit_signed_run_from_day_open_pct": signed_ret(day_open, entry_price),
                    "audit_abs_run_from_day_open_pct": None if not day_open else (entry_price / day_open - 1.0) * 100.0,
                    "audit_signed_run_30m_pct": signed_ret(
                        _num(prior_30.iloc[0].get("open")) if not prior_30.empty else None,
                        entry_price,
                    ),
                    "audit_signed_run_60m_pct": signed_ret(
                        _num(prior_60.iloc[0].get("open")) if not prior_60.empty else None,
                        entry_price,
                    ),
                    "audit_signed_run_120m_pct": signed_ret(
                        _num(prior_120.iloc[0].get("open")) if not prior_120.empty else None,
                        entry_price,
                    ),
                    "audit_entry_vs_seen_high_pct": (entry_price / before["high"].max() - 1.0) * 100.0,
                    "audit_entry_vs_seen_low_pct": (entry_price / before["low"].min() - 1.0) * 100.0,
                }
            )
    return pd.DataFrame(rows).drop_duplicates(["ticker", "direction", "symbol"], keep="first")


def summarize_filter(df: pd.DataFrame, name: str, mask: pd.Series) -> dict[str, Any]:
    kept = df[mask].copy()
    blocked = df[~mask].copy()
    return {
        "filter": name,
        "kept_trades": len(kept),
        "blocked_trades": len(blocked),
        "kept_pnl": kept["option_pnl_dollars"].sum(),
        "blocked_pnl": blocked["option_pnl_dollars"].sum(),
        "kept_median_ret": kept["option_ret_pct"].median(),
        "blocked_median_ret": blocked["option_ret_pct"].median(),
        "kept_win_rate": (kept["option_pnl_dollars"] > 0).mean() if len(kept) else None,
        "blocked_win_rate": (blocked["option_pnl_dollars"] > 0).mean() if len(blocked) else None,
    }


def run(out_dir: Path, friday: Path, monday: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    weekend = friday_to_monday(friday, monday)
    weekend.to_csv(out_dir / "friday_to_monday_hold_impact.csv", index=False)

    monday_trades = pd.read_csv(out_dir / "trade_forensics_fresh.csv")
    monday_trades["entry_time"] = pd.to_datetime(monday_trades["entry_time"], utc=True, errors="coerce")
    monday_trades = add_runup_metrics(monday_trades)
    audit_metrics = audit_runup_metrics(monday)
    if not audit_metrics.empty:
        monday_trades = monday_trades.merge(
            audit_metrics,
            on=["ticker", "direction", "symbol"],
            how="left",
            suffixes=("", "_audit_seed"),
        )
        for raw_col, audit_col in [
            ("signed_run_from_day_open_pct", "audit_signed_run_from_day_open_pct"),
            ("abs_run_from_day_open_pct", "audit_abs_run_from_day_open_pct"),
            ("signed_run_30m_pct", "audit_signed_run_30m_pct"),
            ("signed_run_60m_pct", "audit_signed_run_60m_pct"),
            ("signed_run_120m_pct", "audit_signed_run_120m_pct"),
            ("entry_vs_day_high_pct", "audit_entry_vs_seen_high_pct"),
            ("entry_vs_day_low_pct", "audit_entry_vs_seen_low_pct"),
        ]:
            if audit_col in monday_trades:
                monday_trades[raw_col] = monday_trades[raw_col].combine_first(monday_trades[audit_col])
    monday_trades.to_csv(out_dir / "monday_fresh_runup_metrics.csv", index=False)

    rows = []
    rows.append(summarize_filter(monday_trades, "baseline_all_fresh", pd.Series(True, index=monday_trades.index)))
    rows.append(summarize_filter(monday_trades, "block_after_14_30_et", monday_trades["entry_time"].dt.tz_convert(ET).dt.strftime("%H:%M") < "14:30"))
    rows.append(summarize_filter(monday_trades, "block_after_15_00_et", monday_trades["entry_time"].dt.tz_convert(ET).dt.strftime("%H:%M") < "15:00"))
    rows.append(summarize_filter(monday_trades, "p_dir_ge_0_85", monday_trades["p_dir"] >= 0.85))
    rows.append(summarize_filter(monday_trades, "p_dir_ge_0_90", monday_trades["p_dir"] >= 0.90))
    rows.append(summarize_filter(monday_trades, "spread_le_18pct", monday_trades["entry_spread_pct_mid"] <= 18.0))
    rows.append(summarize_filter(monday_trades, "spread_le_25pct", monday_trades["entry_spread_pct_mid"] <= 25.0))
    rows.append(
        summarize_filter(
            monday_trades,
            "avoid_already_ran_signed_day_open_gt_2pct",
            monday_trades["signed_run_from_day_open_pct"].fillna(0.0) <= 2.0,
        )
    )
    rows.append(
        summarize_filter(
            monday_trades,
            "avoid_already_ran_60m_gt_1pct",
            monday_trades["signed_run_60m_pct"].fillna(0.0) <= 1.0,
        )
    )
    rows.append(
        summarize_filter(
            monday_trades,
            "combo_p90_spread18_pre1430",
            (monday_trades["p_dir"] >= 0.90)
            & (monday_trades["entry_spread_pct_mid"] <= 18.0)
            & (monday_trades["entry_time"].dt.tz_convert(ET).dt.strftime("%H:%M") < "14:30"),
        )
    )
    filt = pd.DataFrame(rows)
    filt.to_csv(out_dir / "monday_fresh_filter_experiment.csv", index=False)

    if not weekend.empty:
        print("friday to monday")
        print(
            weekend[
                [
                    "ticker",
                    "side",
                    "entry_time_et",
                    "expiry",
                    "dte_entry",
                    "friday_last_option",
                    "monday_first_option",
                    "fri_to_mon_open_ret_pct",
                    "entry_to_mon_last_ret_pct",
                ]
            ]
            .round(3)
            .to_string(index=False)
        )
    print("\nfilter experiments")
    print(filt.round(3).to_string(index=False))
    print("\nrunup bins")
    data = monday_trades.copy()
    data["signed_run_from_day_open_pct"] = pd.to_numeric(
        data["signed_run_from_day_open_pct"], errors="coerce"
    )
    data = data.dropna(subset=["signed_run_from_day_open_pct"])
    if data.empty:
        print("no runup metrics available")
        return
    data["run_bin"] = pd.cut(
        data["signed_run_from_day_open_pct"],
        [-999, -2, 0, 1, 2, 4, 999],
        labels=["against<-2", "against-2..0", "with0..1", "with1..2", "with2..4", "with>4"],
    )
    print(
        data.groupby("run_bin", observed=False)
        .agg(
            trades=("symbol", "count"),
            pnl=("option_pnl_dollars", "sum"),
            med_ret=("option_ret_pct", "median"),
            med_under=("underlying_ret_pct", "median"),
            win=("option_pnl_dollars", lambda s: (s > 0).mean()),
        )
        .round(3)
        .reset_index()
        .to_string(index=False)
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--friday", type=Path, default=FRIDAY_AUDIT)
    parser.add_argument("--monday", type=Path, default=MONDAY_AUDIT)
    args = parser.parse_args()
    run(args.out, args.friday, args.monday)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
