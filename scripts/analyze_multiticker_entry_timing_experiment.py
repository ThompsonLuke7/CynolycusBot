from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_AUDITS = [
    Path("UI/swing_audit/swing_session_20260528T120501Z.jsonl"),
    Path("UI/swing_audit/swing_session_20260529T120845Z.jsonl"),
]
OUT_DIR = Path("Data/analysis/multi_ticker_swing_live/experiments")


def _ts(value: Any) -> pd.Timestamp:
    if value is None or value == "":
        return pd.NaT
    return pd.to_datetime(value, utc=True)


def _bar_ts(value: Any) -> pd.Timestamp:
    if value is None:
        return pd.NaT
    return pd.to_datetime(value, unit="s", utc=True)


def _load_events(path: Path) -> list[tuple[pd.Timestamp, str, dict[str, Any]]]:
    events: list[tuple[pd.Timestamp, str, dict[str, Any]]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            events.append((_ts(obj.get("ts")), obj.get("type") or obj.get("event") or "", obj.get("payload") or {}))
    return events


def _option_sell_fills(events: list[tuple[pd.Timestamp, str, dict[str, Any]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for ts, typ, payload in events:
        if typ != "order_submitted" or payload.get("side") != "sell":
            continue
        verification = payload.get("verification") or {}
        order = verification.get("order") or {}
        fill = order.get("filled_avg_price")
        rows.append(
            {
                "ts": ts,
                "ticker": payload.get("ticker"),
                "option_symbol": payload.get("option_symbol"),
                "fill": float(fill) if fill not in (None, "") else math.nan,
                "status": verification.get("status"),
            }
        )
    return rows


def _events_to_frames(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    events = _load_events(path)
    sell_fills = _option_sell_fills(events)
    opens: list[dict[str, Any]] = []
    closes: list[dict[str, Any]] = []
    position_bars: list[dict[str, Any]] = []
    chart_seeds: dict[tuple[str, str], list[dict[str, Any]]] = {}

    for ts, typ, payload in events:
        if typ == "position_opened":
            opens.append({"audit_date": path.stem, "ts": ts, **payload})
        elif typ == "position_closed":
            closes.append({"audit_date": path.stem, "ts": ts, **payload})
        elif typ == "position_bar_5m":
            bar = payload.get("bar") or {}
            pos = payload.get("position") or {}
            position_bars.append(
                {
                    "audit_date": path.stem,
                    "ticker": payload.get("ticker"),
                    "ts": _bar_ts(bar.get("time")),
                    "open": bar.get("open"),
                    "high": bar.get("high"),
                    "low": bar.get("low"),
                    "close": bar.get("close"),
                    "pos_entry_time": pos.get("entry_time"),
                    "pos_entry_price": pos.get("entry_price"),
                    "pos_dir": pos.get("direction"),
                }
            )
        elif typ == "position_chart_seed" and not payload.get("restored"):
            key = (str(payload.get("ticker")), str(payload.get("option_symbol") or ""))
            chart_seeds[key] = payload.get("pre_entry_bars") or []

    bars = pd.DataFrame(position_bars)
    if not bars.empty:
        for col in ["open", "high", "low", "close", "pos_entry_price"]:
            bars[col] = pd.to_numeric(bars[col], errors="coerce")

    closed_rows: list[dict[str, Any]] = []
    for close in closes:
        option_symbol = close.get("option_symbol")
        close_ts = close["ts"]
        open_candidates = [o for o in opens if o.get("option_symbol") == option_symbol and o["ts"] <= close_ts]
        open_row = max(open_candidates, key=lambda row: row["ts"]) if open_candidates else {}
        sell_candidates = [
            s
            for s in sell_fills
            if s.get("option_symbol") == option_symbol
            and s["ts"] >= close_ts - pd.Timedelta(minutes=2)
            and s["ts"] <= close_ts + pd.Timedelta(minutes=2)
        ]
        sell = sell_candidates[0] if sell_candidates else {}
        direction = int(close.get("direction") or 0)
        entry = float(close.get("entry_price") or open_row.get("entry_price") or math.nan)
        exit_price = float(close.get("exit_price") or close.get("last_price") or math.nan)
        option_entry = float(close.get("option_entry_price") or open_row.get("option_entry_price") or math.nan)
        option_exit = float(sell.get("fill") or close.get("option_last_price") or math.nan)
        qty = float(close.get("qty") or open_row.get("qty") or 1.0)
        is_fresh = isinstance(close.get("option_entry_meta"), dict)
        seed_key = (str(close.get("ticker")), str(option_symbol or ""))
        closed_rows.append(
            {
                "audit_date": close["audit_date"],
                "ticker": close.get("ticker"),
                "direction": direction,
                "closed_ts": close_ts,
                "entry_time": _ts(close.get("entry_time")),
                "is_fresh": is_fresh,
                "entry_underlying": entry,
                "exit_underlying": exit_price,
                "underlying_signed_ret_pct": direction * (exit_price - entry) / entry * 100
                if math.isfinite(entry) and entry
                else math.nan,
                "option_symbol": option_symbol,
                "option_entry_price": option_entry,
                "option_exit_price": option_exit,
                "qty": qty,
                "option_pnl_dollars": (option_exit - option_entry) * 100 * qty
                if math.isfinite(option_entry) and math.isfinite(option_exit)
                else math.nan,
                "option_ret_pct": (option_exit / option_entry - 1) * 100
                if math.isfinite(option_entry) and option_entry > 0 and math.isfinite(option_exit)
                else math.nan,
                "stock_100sh_pnl": direction * (exit_price - entry) * 100
                if math.isfinite(entry) and math.isfinite(exit_price)
                else math.nan,
                "exit_reason": close.get("exit_reason"),
                "bars_held": close.get("bars_held"),
                "best_price": close.get("best_price"),
                "option_best_price": close.get("option_best_price"),
                "atr_at_entry": close.get("atr_at_entry"),
                "pre_entry_bars": chart_seeds.get(seed_key, []),
            }
        )
    closed = pd.DataFrame(closed_rows)
    return closed, bars, pd.DataFrame(sell_fills)


def _context_features(pre_bars: list[dict[str, Any]], direction: int, entry: float, atr: float) -> dict[str, Any]:
    bars = pd.DataFrame(pre_bars or [])
    if bars.empty:
        return {
            "pre_bars": 0,
            "pre_run_atr": math.nan,
            "pre_directional_streak": 0,
            "confirm_body_frac": math.nan,
            "confirm_range_atr": math.nan,
            "extended_entry": False,
        }
    for col in ["open", "high", "low", "close"]:
        bars[col] = pd.to_numeric(bars[col], errors="coerce")
    recent = bars.tail(6)
    if direction == 1:
        pre_run = entry - recent["low"].min()
        dir_candle = recent["close"] > recent["open"]
    else:
        pre_run = recent["high"].max() - entry
        dir_candle = recent["close"] < recent["open"]
    streak = 0
    for is_dir in reversed(list(dir_candle.fillna(False))):
        if is_dir:
            streak += 1
        else:
            break
    last = bars.iloc[-1]
    bar_range = float(last["high"] - last["low"]) if pd.notna(last["high"]) and pd.notna(last["low"]) else math.nan
    body = abs(float(last["close"] - last["open"])) if pd.notna(last["close"]) and pd.notna(last["open"]) else math.nan
    body_frac = body / bar_range if math.isfinite(bar_range) and bar_range > 0 else math.nan
    run_atr = pre_run / atr if math.isfinite(pre_run) and math.isfinite(atr) and atr > 0 else math.nan
    range_atr = bar_range / atr if math.isfinite(bar_range) and math.isfinite(atr) and atr > 0 else math.nan
    extended = (math.isfinite(run_atr) and run_atr >= 0.75) or streak >= 3 or (
        math.isfinite(body_frac) and body_frac <= 0.25 and math.isfinite(run_atr) and run_atr >= 0.4
    )
    return {
        "pre_bars": int(len(bars)),
        "pre_run_atr": run_atr,
        "pre_directional_streak": int(streak),
        "confirm_body_frac": body_frac,
        "confirm_range_atr": range_atr,
        "extended_entry": bool(extended),
    }


def _entry_variants(row: pd.Series, bars: pd.DataFrame, max_wait_bars: int = 12) -> dict[str, Any] | None:
    entry_time = row["entry_time"]
    ticker = row["ticker"]
    trade_bars = bars[
        (bars["ticker"] == ticker)
        & (pd.to_datetime(bars["pos_entry_time"], utc=True) == entry_time)
    ].sort_values("ts")
    if trade_bars.empty:
        return None

    direction = int(row["direction"])
    entry = float(row["entry_underlying"])
    exit_price = float(row["exit_underlying"])
    atr = float(row["atr_at_entry"]) if pd.notna(row["atr_at_entry"]) else math.nan
    threshold = 0.25 * atr if math.isfinite(atr) and atr > 0 else 0.005 * entry
    probe = trade_bars.head(max_wait_bars)
    first4 = trade_bars.head(4)

    if direction == 1:
        mae_20 = entry - first4["low"].min()
        mfe_20 = first4["high"].max() - entry
        pull_touch = probe[probe["low"] <= entry - threshold]
        pull_entry = entry - threshold if not pull_touch.empty else math.nan
        reclaim_entry = math.nan
        reclaim_ts = pd.NaT
        seen_pullback = False
        prev_close = math.nan
        red_streak = 0
        max_red_streak = 0
        for _, bar in probe.iterrows():
            if bar["close"] < bar["open"]:
                red_streak += 1
            else:
                red_streak = 0
            max_red_streak = max(max_red_streak, red_streak)
            if bar["low"] <= entry - threshold or bar["close"] <= entry - threshold / 2:
                seen_pullback = True
            if (
                seen_pullback
                and math.isfinite(prev_close)
                and bar["close"] > prev_close
                and bar["close"] > bar["open"]
            ):
                reclaim_entry = float(bar["close"])
                reclaim_ts = bar["ts"]
                break
            prev_close = float(bar["close"])
    else:
        mae_20 = first4["high"].max() - entry
        mfe_20 = entry - first4["low"].min()
        pull_touch = probe[probe["high"] >= entry + threshold]
        pull_entry = entry + threshold if not pull_touch.empty else math.nan
        reclaim_entry = math.nan
        reclaim_ts = pd.NaT
        seen_pullback = False
        prev_close = math.nan
        green_streak = 0
        max_red_streak = 0
        for _, bar in probe.iterrows():
            if bar["close"] > bar["open"]:
                green_streak += 1
            else:
                green_streak = 0
            max_red_streak = max(max_red_streak, green_streak)
            if bar["high"] >= entry + threshold or bar["close"] >= entry + threshold / 2:
                seen_pullback = True
            if (
                seen_pullback
                and math.isfinite(prev_close)
                and bar["close"] < prev_close
                and bar["close"] < bar["open"]
            ):
                reclaim_entry = float(bar["close"])
                reclaim_ts = bar["ts"]
                break
            prev_close = float(bar["close"])

    context = _context_features(row.get("pre_entry_bars") or [], direction, entry, atr)
    immediate_ret = direction * (exit_price - entry) / entry * 100
    pullback_ret = direction * (exit_price - pull_entry) / pull_entry * 100 if math.isfinite(pull_entry) else math.nan
    reclaim_ret = direction * (exit_price - reclaim_entry) / reclaim_entry * 100 if math.isfinite(reclaim_entry) else math.nan
    conditional_pull_ret = pullback_ret if context["extended_entry"] and math.isfinite(pullback_ret) else immediate_ret
    conditional_pull_taken = (not context["extended_entry"]) or math.isfinite(pullback_ret)
    conditional_reclaim_ret = reclaim_ret if context["extended_entry"] and math.isfinite(reclaim_ret) else immediate_ret
    conditional_reclaim_taken = (not context["extended_entry"]) or math.isfinite(reclaim_ret)

    return {
        "audit_date": row["audit_date"],
        "ticker": ticker,
        "direction": direction,
        "entry_time": entry_time,
        "exit_reason": row["exit_reason"],
        "entry_underlying": entry,
        "exit_underlying": exit_price,
        "atr_at_entry": atr,
        "current_underlying_ret_pct": immediate_ret,
        "current_option_ret_pct": row["option_ret_pct"],
        "mae_first20m_pct": mae_20 / entry * 100,
        "mfe_first20m_pct": mfe_20 / entry * 100,
        "adverse_gt_favorable_20m": bool(mae_20 > mfe_20),
        "pullback_025atr_filled": math.isfinite(pull_entry),
        "pullback_025atr_entry": pull_entry,
        "pullback_025atr_ret_pct": pullback_ret,
        "reclaim_after_pullback_filled": math.isfinite(reclaim_entry),
        "reclaim_after_pullback_entry": reclaim_entry,
        "reclaim_after_pullback_ts": reclaim_ts,
        "reclaim_after_pullback_ret_pct": reclaim_ret,
        "post_entry_opposite_streak": int(max_red_streak),
        "conditional_pull_taken": bool(conditional_pull_taken),
        "conditional_pull_ret_pct": conditional_pull_ret if conditional_pull_taken else math.nan,
        "conditional_reclaim_taken": bool(conditional_reclaim_taken),
        "conditional_reclaim_ret_pct": conditional_reclaim_ret if conditional_reclaim_taken else math.nan,
        **context,
    }


def _summarize_policy(df: pd.DataFrame, ret_col: str, taken_col: str | None = None) -> dict[str, Any]:
    scope = df[df[taken_col]] if taken_col else df
    return {
        "trades": int(len(scope)),
        "coverage": float(len(scope) / len(df)) if len(df) else math.nan,
        "avg_underlying_ret_pct": float(scope[ret_col].mean()) if len(scope) else math.nan,
        "median_underlying_ret_pct": float(scope[ret_col].median()) if len(scope) else math.nan,
        "win_rate": float((scope[ret_col] > 0).mean()) if len(scope) else math.nan,
    }


def run(audits: list[Path]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    closed_frames: list[pd.DataFrame] = []
    bars_frames: list[pd.DataFrame] = []
    for audit in audits:
        if not audit.exists():
            continue
        closed, bars, _ = _events_to_frames(audit)
        closed_frames.append(closed)
        bars_frames.append(bars)

    closed_all = pd.concat(closed_frames, ignore_index=True) if closed_frames else pd.DataFrame()
    bars_all = pd.concat(bars_frames, ignore_index=True) if bars_frames else pd.DataFrame()
    closed_all.drop(columns=["pre_entry_bars"], errors="ignore").to_csv(
        OUT_DIR / "multiticker_20260528_20260529_closed_performance_rebuilt.csv", index=False
    )

    fresh = closed_all[closed_all.get("is_fresh", False)].copy()
    variant_rows: list[dict[str, Any]] = []
    for _, row in fresh.iterrows():
        variant = _entry_variants(row, bars_all)
        if variant:
            variant_rows.append(variant)
    variants = pd.DataFrame(variant_rows)
    variants.to_csv(OUT_DIR / "multiticker_20260528_20260529_entry_policy_variants.csv", index=False)

    summaries: list[dict[str, Any]] = []
    for side_name, side_df in [
        ("all", variants),
        ("calls", variants[variants["direction"] == 1] if not variants.empty else variants),
        ("puts", variants[variants["direction"] == -1] if not variants.empty else variants),
    ]:
        if side_df.empty:
            continue
        for policy_name, ret_col, taken_col in [
            ("current_5m_breakout", "current_underlying_ret_pct", None),
            ("pure_pullback_025atr", "pullback_025atr_ret_pct", "pullback_025atr_filled"),
            ("pure_reclaim_after_pullback", "reclaim_after_pullback_ret_pct", "reclaim_after_pullback_filled"),
            ("conditional_extended_else_current_pullback", "conditional_pull_ret_pct", "conditional_pull_taken"),
            ("conditional_extended_else_current_reclaim", "conditional_reclaim_ret_pct", "conditional_reclaim_taken"),
        ]:
            summary = _summarize_policy(side_df, ret_col, taken_col)
            summaries.append({"side": side_name, "policy": policy_name, **summary})
    pd.DataFrame(summaries).to_csv(
        OUT_DIR / "multiticker_20260528_20260529_entry_policy_variant_summary.csv", index=False
    )

    restored = closed_all[~closed_all.get("is_fresh", False)].copy()
    if not restored.empty:
        restored["stock_wins_option_loses"] = (restored["stock_100sh_pnl"] > 0) & (restored["option_pnl_dollars"] < 0)
        restored["option_intraday_capture_pct"] = (
            (restored["option_exit_price"] - restored["option_entry_price"])
            / restored["option_entry_price"]
            * 100
        )
        restored.to_csv(OUT_DIR / "multiticker_20260528_20260529_restored_option_damage.csv", index=False)

    print("wrote", OUT_DIR / "multiticker_20260528_20260529_entry_policy_variants.csv")
    if summaries:
        print(pd.DataFrame(summaries).round(3).to_string(index=False))
    if not restored.empty:
        print("\nrestored damage")
        print(
            restored.groupby(["audit_date", "direction"])
            .agg(
                trades=("ticker", "count"),
                option_pnl=("option_pnl_dollars", "sum"),
                stock_100sh_pnl=("stock_100sh_pnl", "sum"),
                option_win_rate=("option_pnl_dollars", lambda s: (s > 0).mean()),
                stock_win_rate=("stock_100sh_pnl", lambda s: (s > 0).mean()),
                stock_wins_option_loses=("stock_wins_option_loses", "mean"),
            )
            .round(3)
            .to_string()
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("audits", nargs="*", type=Path, default=DEFAULT_AUDITS)
    args = parser.parse_args()
    run(args.audits)


if __name__ == "__main__":
    main()
