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

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from core.shared_plotting import (
    DEFAULT_THEME,
    apply_mpl_defaults,
    apply_time_ticks,
    compute_time_ticks,
    plot_candles_from_frame,
    save_figure,
    style_figure,
    time_to_position,
)


ET = ZoneInfo("America/New_York")
DEFAULT_AUDIT = Path("UI/swing_audit/paper/swing_session_20260608T121654Z.jsonl")
DEFAULT_OUT = Path("UI/swing_audit/forensics_20260608")
RAW_5M_DIR = Path("strategies/multi_ticker_swing/data/raw/5m")
RAW_30M_DIR = Path("strategies/multi_ticker_swing/data/raw/30m")


def _num(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _ts(value: Any) -> pd.Timestamp:
    return pd.to_datetime(value, utc=True, errors="coerce")


def _bar_ts(value: Any) -> pd.Timestamp:
    return pd.to_datetime(value, unit="s", utc=True, errors="coerce")


def _entry_key(payload: dict[str, Any]) -> tuple[str, int, str]:
    return (
        str(payload.get("ticker") or "").upper(),
        int(payload.get("direction") or 0),
        str(payload.get("entry_time") or ""),
    )


def _load_raw_bars(ticker: str, frame: str) -> pd.DataFrame:
    path = (RAW_5M_DIR if frame == "5m" else RAW_30M_DIR) / f"{ticker.upper()}.parquet"
    if not path.exists():
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    df = pd.read_parquet(path)
    if "timestamp" not in df.columns and df.index.name:
        df = df.reset_index()
    df.columns = [str(c).lower() for c in df.columns]
    if "timestamp" not in df.columns:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    keep = [c for c in ["timestamp", "open", "high", "low", "close", "volume"] if c in df.columns]
    return df[keep].dropna(subset=["timestamp", "open", "high", "low", "close"]).sort_values("timestamp")


def parse_audit(path: Path) -> dict[str, Any]:
    active: list[dict[str, Any]] = []
    closed: list[dict[str, Any]] = []
    removed: list[dict[str, Any]] = []
    orders: list[dict[str, Any]] = []
    signals: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    confirmations: list[dict[str, Any]] = []
    position_marks: list[dict[str, Any]] = []
    chart_bars: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            if not line.strip():
                continue
            event = json.loads(line)
            typ = event.get("type")
            payload = event.get("payload") or {}
            event_ts = _ts(event.get("ts"))

            if typ == "signal":
                signals.append({"event_ts": event_ts, "line": line_no, **payload})
            elif typ == "risk_profile_policy_decision":
                decisions.append({"event_ts": event_ts, "line": line_no, **payload})
            elif typ == "confirmation":
                confirmations.append({"event_ts": event_ts, "line": line_no, **payload})
            elif typ == "order_submitted":
                orders.append({"event_ts": event_ts, "line": line_no, **payload})
            elif typ == "position_opened":
                active.append({"open_event_ts": event_ts, "open_line": line_no, **payload})
            elif typ in {"position_closed", "broker_position_missing"}:
                match_idx = None
                sym = payload.get("option_symbol")
                key = _entry_key(payload)
                for idx, pos in enumerate(active):
                    if sym and pos.get("option_symbol") == sym:
                        match_idx = idx
                        break
                if match_idx is None:
                    for idx, pos in enumerate(active):
                        if _entry_key(pos) == key:
                            match_idx = idx
                            break
                base = active.pop(match_idx) if match_idx is not None else {}
                row = {**base, **payload, "close_event": typ, "close_event_ts": event_ts, "close_line": line_no}
                if typ == "position_closed":
                    closed.append(row)
                else:
                    removed.append(row)
            elif typ == "position_bar_5m":
                pos = payload.get("position") or {}
                bar = payload.get("bar") or {}
                position_marks.append(
                    {
                        "event_ts": event_ts,
                        "ticker": str(payload.get("ticker") or pos.get("ticker") or "").upper(),
                        "direction": int(pos.get("direction") or 0),
                        "entry_time": _ts(pos.get("entry_time")),
                        "bar_ts": _bar_ts(bar.get("time")),
                        "underlying_open": _num(bar.get("open")),
                        "underlying_high": _num(bar.get("high")),
                        "underlying_low": _num(bar.get("low")),
                        "underlying_close": _num(bar.get("close")),
                        "option_last_price": _num(pos.get("option_last_price")),
                        "option_best_price": _num(pos.get("option_best_price")),
                        "underlying_pnl_pct": _num(pos.get("pnl_pct")),
                        "bars_held": _num(pos.get("bars_held")),
                    }
                )
            elif typ == "position_chart_seed":
                ticker = str(payload.get("ticker") or "").upper()
                for bar in payload.get("pre_entry_bars") or []:
                    chart_bars.append(
                        {
                            "ticker": ticker,
                            "timestamp": _bar_ts(bar.get("time")),
                            "open": _num(bar.get("open")),
                            "high": _num(bar.get("high")),
                            "low": _num(bar.get("low")),
                            "close": _num(bar.get("close")),
                            "volume": _num(bar.get("volume")),
                        }
                    )

    return {
        "active": active,
        "closed": closed,
        "removed": removed,
        "orders": pd.DataFrame(orders),
        "signals": pd.DataFrame(signals),
        "decisions": pd.DataFrame(decisions),
        "confirmations": pd.DataFrame(confirmations),
        "marks": pd.DataFrame(position_marks),
        "chart_bars": pd.DataFrame(chart_bars),
    }


def _order_fill_price(row: pd.Series) -> float | None:
    verification = row.get("verification") if isinstance(row.get("verification"), dict) else {}
    order = verification.get("order") if isinstance(verification.get("order"), dict) else {}
    return _num(order.get("filled_avg_price")) or _num((row.get("response") or {}).get("filled_avg_price"))


def _order_status(row: pd.Series) -> str | None:
    verification = row.get("verification") if isinstance(row.get("verification"), dict) else {}
    response = row.get("response") if isinstance(row.get("response"), dict) else {}
    return verification.get("status") or response.get("status")


def _latest_context(
    frame: pd.DataFrame,
    ticker: str,
    direction: int,
    ts: pd.Timestamp,
    ts_col: str,
) -> pd.Series | None:
    if frame.empty or ts_col not in frame.columns:
        return None
    data = frame.copy()
    data["ticker"] = data["ticker"].astype(str).str.upper()
    data[ts_col] = pd.to_datetime(data[ts_col], utc=True, errors="coerce")
    pool = data[
        (data["ticker"].eq(ticker.upper()))
        & (pd.to_numeric(data.get("direction"), errors="coerce") == float(direction))
        & (data[ts_col] <= ts)
    ].sort_values(ts_col)
    return None if pool.empty else pool.iloc[-1]


def _matching_marks(marks: pd.DataFrame, pos: dict[str, Any]) -> pd.DataFrame:
    if marks.empty:
        return marks
    entry = _ts(pos.get("entry_time"))
    ticker = str(pos.get("ticker") or "").upper()
    direction = int(pos.get("direction") or 0)
    out = marks[
        (marks["ticker"].astype(str).str.upper().eq(ticker))
        & (pd.to_numeric(marks["direction"], errors="coerce") == float(direction))
        & (pd.to_datetime(marks["entry_time"], utc=True, errors="coerce") == entry)
    ].copy()
    return out.sort_values("bar_ts")


def _sell_fill_for_symbol(orders: pd.DataFrame, symbol: str, after: pd.Timestamp | None = None) -> pd.Series | None:
    if orders.empty:
        return None
    data = orders[orders["option_symbol"].astype(str).eq(str(symbol)) & orders["side"].astype(str).eq("sell")].copy()
    if after is not None:
        data = data[pd.to_datetime(data["event_ts"], utc=True, errors="coerce") >= after]
    if data.empty:
        return None
    data["status"] = data.apply(_order_status, axis=1)
    filled = data[data["status"].eq("filled")]
    return (filled if not filled.empty else data).sort_values("event_ts").iloc[-1]


def build_trade_table(parsed: dict[str, Any]) -> pd.DataFrame:
    orders = parsed["orders"]
    marks = parsed["marks"]
    signals = parsed["signals"]
    decisions = parsed["decisions"]
    confirmations = parsed["confirmations"]
    positions = [*parsed["closed"], *parsed["removed"], *parsed["active"]]
    rows: list[dict[str, Any]] = []

    for pos in positions:
        ticker = str(pos.get("ticker") or "").upper()
        direction = int(pos.get("direction") or 0)
        symbol = str(pos.get("option_symbol") or "")
        entry_time = _ts(pos.get("entry_time"))
        entry_px_option = _num(pos.get("option_entry_price"))
        entry_px_underlying = _num(pos.get("entry_price"))
        restored = bool(pos.get("restored_from_broker"))
        meta = pos.get("option_entry_meta") if isinstance(pos.get("option_entry_meta"), dict) else {}
        risk = meta.get("risk_profile_policy") if isinstance(meta.get("risk_profile_policy"), dict) else {}
        quote = meta.get("entry_quote") if isinstance(meta.get("entry_quote"), dict) else {}
        confirmation = meta.get("confirmation_metrics") if isinstance(meta.get("confirmation_metrics"), dict) else {}
        mark_path = _matching_marks(marks, pos)
        latest_mark = mark_path.iloc[-1] if not mark_path.empty else None
        close_event = pos.get("close_event")
        close_ts = _ts(pos.get("close_event_ts"))
        sell = _sell_fill_for_symbol(orders, symbol, after=entry_time)
        exit_px_option = None
        exit_status = None
        if sell is not None:
            exit_px_option = _order_fill_price(sell)
            exit_status = _order_status(sell)
        if close_event is None and latest_mark is not None:
            exit_px_option = _num(latest_mark.get("option_last_price"))
            exit_status = "open_mark"
        if exit_px_option is None:
            exit_px_option = _num(pos.get("option_last_price"))
        if exit_px_option is None and latest_mark is not None:
            exit_px_option = _num(latest_mark.get("option_last_price"))

        option_pnl = None
        option_ret = None
        if entry_px_option and exit_px_option is not None:
            option_pnl = (exit_px_option - entry_px_option) * 100.0 * (_num(pos.get("qty")) or 1.0)
            option_ret = exit_px_option / entry_px_option - 1.0

        final_underlying_ret = _num(pos.get("exit_pnl_pct"))
        if final_underlying_ret is None and latest_mark is not None:
            final_underlying_ret = _num(latest_mark.get("underlying_pnl_pct"))

        opt_marks = pd.to_numeric(mark_path.get("option_last_price", pd.Series(dtype=float)), errors="coerce").dropna()
        under_marks = pd.to_numeric(mark_path.get("underlying_pnl_pct", pd.Series(dtype=float)), errors="coerce").dropna()
        signal = _latest_context(signals, ticker, direction, entry_time, "event_ts")
        decision = _latest_context(decisions, ticker, direction, entry_time, "event_ts")
        confirm = _latest_context(confirmations, ticker, direction, entry_time, "event_ts")

        reason_bits: list[str] = []
        if final_underlying_ret is not None and final_underlying_ret < 0:
            reason_bits.append("underlying_moved_against_entry")
        if (
            final_underlying_ret is not None
            and option_ret is not None
            and abs(final_underlying_ret) < 0.005
            and option_ret < -0.20
        ):
            reason_bits.append("option_crushed_on_flat_underlying")
        spread_pct = _num(quote.get("spread_pct_mid"))
        if spread_pct is not None and spread_pct > 0.25:
            reason_bits.append("wide_entry_spread")
        if signal is not None and _num(signal.get("p_dir")) is not None and _num(signal.get("p_dir")) < 0.85:
            reason_bits.append("lowish_model_confidence")
        if risk.get("profile") == "defensive" and direction == 1:
            reason_bits.append("defensive_long_allowed")
        if close_event == "broker_position_missing":
            reason_bits.append("exit_order_unverified_but_broker_removed")

        rows.append(
            {
                "ticker": ticker,
                "symbol": symbol,
                "side": "call" if direction == 1 else "put",
                "direction": direction,
                "restored": restored,
                "restore_source": pos.get("restore_source"),
                "status": close_event or "open",
                "entry_time": entry_time,
                "entry_time_et": entry_time.tz_convert(ET) if pd.notna(entry_time) else pd.NaT,
                "exit_time": close_ts,
                "exit_time_et": close_ts.tz_convert(ET) if pd.notna(close_ts) else pd.NaT,
                "exit_reason": pos.get("exit_reason") or pos.get("reason"),
                "exit_order_status": exit_status,
                "entry_underlying": entry_px_underlying,
                "exit_underlying_or_mark": _num(pos.get("exit_price"))
                or (None if latest_mark is None else _num(latest_mark.get("underlying_close"))),
                "underlying_ret_pct": None if final_underlying_ret is None else final_underlying_ret * 100.0,
                "entry_option": entry_px_option,
                "exit_option_or_mark": exit_px_option,
                "option_ret_pct": None if option_ret is None else option_ret * 100.0,
                "option_pnl_dollars": option_pnl,
                "option_mfe_pct": None if opt_marks.empty or not entry_px_option else (opt_marks.max() / entry_px_option - 1.0) * 100.0,
                "option_mae_pct": None if opt_marks.empty or not entry_px_option else (opt_marks.min() / entry_px_option - 1.0) * 100.0,
                "underlying_mfe_pct": None if under_marks.empty else under_marks.max() * 100.0,
                "underlying_mae_pct": None if under_marks.empty else under_marks.min() * 100.0,
                "bars_marked": int(len(mark_path)),
                "p_dir": None if signal is None else _num(signal.get("p_dir")),
                "ev_score": None if signal is None else _num(signal.get("ev_score")),
                "risk_profile": risk.get("profile") or (None if decision is None else decision.get("profile")),
                "risk_reason": (None if decision is None else decision.get("reason")) or risk.get("reason"),
                "qqq_ret_16_pct": None if risk.get("qqq_ret_16") is None else _num(risk.get("qqq_ret_16")) * 100.0,
                "rel_str_qqq_4": risk.get("rel_str_qqq_4"),
                "stock_beta_bucket": risk.get("stock_beta_bucket"),
                "beta_like_spy_64": risk.get("beta_like_spy_64"),
                "is_high_beta": risk.get("is_high_beta"),
                "entry_bid": quote.get("bid"),
                "entry_ask": quote.get("ask"),
                "entry_mid": quote.get("mid"),
                "entry_spread_pct_mid": None if spread_pct is None else spread_pct * 100.0,
                "fill_vs_mid_pct": None
                if not entry_px_option or not _num(quote.get("mid"))
                else (entry_px_option / _num(quote.get("mid")) - 1.0) * 100.0,
                "dte": meta.get("dte"),
                "theta": (meta.get("greeks") or {}).get("theta") if isinstance(meta.get("greeks"), dict) else None,
                "delta": (meta.get("greeks") or {}).get("delta") if isinstance(meta.get("greeks"), dict) else None,
                "confirmation_body_frac": confirmation.get("body_frac") or (None if confirm is None else confirm.get("body_frac")),
                "confirmation_range_atr": confirmation.get("range_atr") or (None if confirm is None else confirm.get("range_atr")),
                "diagnosis": "; ".join(reason_bits),
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["restored", "option_pnl_dollars"], ascending=[True, True])
    return out


def _bars_for_chart(ticker: str, chart_bars: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    raw = _load_raw_bars(ticker, "5m")
    pieces = []
    if not raw.empty:
        pieces.append(raw)
    if not chart_bars.empty:
        cb = chart_bars[chart_bars["ticker"].astype(str).str.upper().eq(ticker.upper())].copy()
        if not cb.empty:
            pieces.append(cb)
    if not pieces:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    out = pd.concat(pieces, ignore_index=True)
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out = out.dropna(subset=["timestamp", "open", "high", "low", "close"]).sort_values("timestamp")
    out = out.drop_duplicates("timestamp", keep="last")
    return out[out["timestamp"].between(start, end)].copy()


def plot_trade(
    trade: pd.Series,
    parsed: dict[str, Any],
    out_dir: Path,
) -> Path | None:
    ticker = str(trade["ticker"]).upper()
    entry = _ts(trade["entry_time"])
    exit_ts = _ts(trade["exit_time"]) if pd.notna(trade.get("exit_time")) else pd.NaT
    end_anchor = exit_ts if pd.notna(exit_ts) else pd.Timestamp("2026-06-08 16:05", tz=ET).tz_convert("UTC")
    start = entry - pd.Timedelta(hours=2)
    end = end_anchor + pd.Timedelta(minutes=20)
    bars = _bars_for_chart(ticker, parsed["chart_bars"], start, end)
    if bars.empty:
        return None
    marks = parsed["marks"].copy()
    marks = marks[
        (marks["ticker"].astype(str).str.upper().eq(ticker))
        & (pd.to_numeric(marks["direction"], errors="coerce") == float(trade["direction"]))
        & (pd.to_datetime(marks["entry_time"], utc=True, errors="coerce") == entry)
    ].sort_values("bar_ts")
    signals = parsed["signals"].copy()
    if not signals.empty:
        signals["event_ts"] = pd.to_datetime(signals["event_ts"], utc=True, errors="coerce")
        signals = signals[
            (signals["ticker"].astype(str).str.upper().eq(ticker))
            & (pd.to_numeric(signals["direction"], errors="coerce") == float(trade["direction"]))
            & signals["event_ts"].between(start, end)
        ].sort_values("event_ts")

    theme = DEFAULT_THEME
    apply_mpl_defaults(theme, font_size=9)
    fig = plt.figure(figsize=(16, 10), constrained_layout=True)
    gs = fig.add_gridspec(3, 1, height_ratios=[2.0, 1.0, 0.9])
    ax_price = fig.add_subplot(gs[0])
    ax_opt = fig.add_subplot(gs[1], sharex=ax_price)
    ax_meta = fig.add_subplot(gs[2], sharex=ax_price)
    style_figure(fig, [ax_price, ax_opt, ax_meta], theme)

    candle = plot_candles_from_frame(
        ax_price,
        bars.set_index("timestamp", drop=False),
        compressed=True,
        theme=theme,
        width=0.65,
    )
    index = pd.DatetimeIndex(bars["timestamp"])
    tick_pos, tick_labels = compute_time_ticks(index, candle.x, max_ticks=10, fmt="%H:%M")
    for ax in (ax_price, ax_opt, ax_meta):
        apply_time_ticks(ax, tick_pos, tick_labels, color=theme.muted_text, fontsize=8)

    def xpos(ts: pd.Timestamp) -> float:
        return float(time_to_position(index, pd.Series([ts])).iloc[0])

    entry_x = xpos(entry)
    exit_x = xpos(end_anchor)
    direction = int(trade["direction"])
    entry_price = _num(trade["entry_underlying"])
    exit_price = _num(trade["exit_underlying_or_mark"])
    color = theme.loss if (_num(trade["option_pnl_dollars"]) or 0.0) < 0 else theme.win
    ax_price.axvline(entry_x, color=theme.blue, lw=1.2, ls="--", alpha=0.9)
    ax_price.axvline(exit_x, color=color, lw=1.2, ls="--", alpha=0.9)
    if entry_price is not None:
        ax_price.scatter([entry_x], [entry_price], marker="^" if direction == 1 else "v", s=110, color=theme.blue, zorder=6)
    if exit_price is not None:
        ax_price.scatter([exit_x], [exit_price], marker="X", s=100, color=color, zorder=6)
    ax_price.set_ylabel("Underlying")

    if not marks.empty:
        mark_x = time_to_position(index, pd.to_datetime(marks["bar_ts"], utc=True))
        ax_opt.plot(mark_x, marks["option_last_price"], color=theme.warning, lw=1.8, marker="o", ms=3, label="option mark")
        ax_meta.plot(mark_x, pd.to_numeric(marks["underlying_pnl_pct"], errors="coerce") * 100.0, color=theme.blue, lw=1.5, label="underlying PnL %")
    ax_opt.axhline(float(trade["entry_option"]), color=theme.neutral, lw=0.9, ls=":", label="entry option")
    if pd.notna(trade.get("exit_option_or_mark")):
        ax_opt.axhline(float(trade["exit_option_or_mark"]), color=color, lw=0.9, ls="--", label="exit/mark option")
    ax_opt.set_ylabel("Option price")
    ax_opt.legend(loc="upper left")

    if not signals.empty:
        sx = time_to_position(index, signals["event_ts"])
        ax_meta.scatter(sx, pd.to_numeric(signals["p_dir"], errors="coerce") * 100.0, color=theme.long if direction == 1 else theme.short, s=24, label="p_dir %")
    ax_meta.axhline(0, color=theme.neutral, lw=0.8, alpha=0.7)
    ax_meta.set_ylabel("Model / PnL")
    ax_meta.legend(loc="upper left")

    title = (
        f"{ticker} {trade['side']} {trade['symbol']} | "
        f"option {trade['option_ret_pct']:+.1f}% / ${trade['option_pnl_dollars']:+.0f} | "
        f"underlying {trade['underlying_ret_pct']:+.2f}% | {trade['risk_profile']} | {trade['status']}"
    )
    ax_price.set_title(title, loc="left", fontsize=12, weight="bold")
    ax_meta.set_xlabel("2026-06-08 ET")
    stem = f"{ticker}_{trade['side']}_{trade['symbol']}_{int(abs(float(trade['option_pnl_dollars']))):05d}".replace("/", "_")
    out = out_dir / f"{stem}.png"
    save_figure(fig, out, dpi=160, tight=False, close=True)
    return out


def write_outputs(audit: Path, out_dir: Path, worst_n: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    parsed = parse_audit(audit)
    trades = build_trade_table(parsed)
    trades.to_csv(out_dir / "trade_forensics_all.csv", index=False)
    fresh = trades[~trades["restored"].fillna(False)].copy()
    fresh.to_csv(out_dir / "trade_forensics_fresh.csv", index=False)
    restored = trades[trades["restored"].fillna(False)].copy()
    restored.to_csv(out_dir / "trade_forensics_restored.csv", index=False)

    closed_or_marked = fresh.dropna(subset=["option_pnl_dollars"]).copy()
    worst = closed_or_marked.sort_values("option_pnl_dollars").head(worst_n)
    worst.to_csv(out_dir / "worst_fresh_trades.csv", index=False)
    top_restored = (
        restored.dropna(subset=["option_pnl_dollars"])
        .sort_values("option_pnl_dollars", ascending=False)
        .head(worst_n)
    )
    top_restored.to_csv(out_dir / "top_restored_winners.csv", index=False)
    chart_rows = []
    charts_dir = out_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    for _, row in worst.iterrows():
        chart = plot_trade(row, parsed, charts_dir)
        chart_rows.append({"ticker": row["ticker"], "symbol": row["symbol"], "chart": str(chart) if chart else None})
    pd.DataFrame(chart_rows).to_csv(out_dir / "worst_fresh_trade_charts.csv", index=False)

    summary_rows = []
    for label, frame in [
        ("fresh_all", fresh),
        ("fresh_calls", fresh[fresh["direction"].eq(1)]),
        ("fresh_puts", fresh[fresh["direction"].eq(-1)]),
        ("restored_all", trades[trades["restored"].fillna(False)]),
    ]:
        vals = pd.to_numeric(frame["option_pnl_dollars"], errors="coerce").dropna()
        rets = pd.to_numeric(frame["option_ret_pct"], errors="coerce").dropna()
        underlying = pd.to_numeric(frame["underlying_ret_pct"], errors="coerce").dropna()
        summary_rows.append(
            {
                "bucket": label,
                "trades": int(len(frame)),
                "pnl_sum": vals.sum(),
                "pnl_median": vals.median(),
                "option_ret_median_pct": rets.median(),
                "win_rate": (vals > 0).mean() if len(vals) else None,
                "underlying_ret_median_pct": underlying.median(),
                "wide_entry_spread_count": int((pd.to_numeric(frame["entry_spread_pct_mid"], errors="coerce") > 25.0).sum()),
                "flat_underlying_option_crush_count": int(frame["diagnosis"].astype(str).str.contains("option_crushed_on_flat_underlying").sum()),
                "direction_wrong_count": int(frame["diagnosis"].astype(str).str.contains("underlying_moved_against_entry").sum()),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "summary.csv", index=False)

    by_profile = (
        fresh.groupby(["risk_profile", "side"], dropna=False)
        .agg(
            trades=("symbol", "count"),
            pnl_sum=("option_pnl_dollars", "sum"),
            pnl_median=("option_pnl_dollars", "median"),
            option_ret_median_pct=("option_ret_pct", "median"),
            underlying_ret_median_pct=("underlying_ret_pct", "median"),
            wide_spreads=("entry_spread_pct_mid", lambda s: int((pd.to_numeric(s, errors="coerce") > 25.0).sum())),
        )
        .reset_index()
    )
    by_profile.to_csv(out_dir / "fresh_by_profile_side.csv", index=False)
    print("summary")
    print(summary.round(3).to_string(index=False))
    print("\nworst fresh trades")
    cols = [
        "ticker",
        "side",
        "risk_profile",
        "status",
        "exit_reason",
        "option_pnl_dollars",
        "option_ret_pct",
        "underlying_ret_pct",
        "entry_spread_pct_mid",
        "fill_vs_mid_pct",
        "p_dir",
        "diagnosis",
    ]
    print(worst[cols].round(3).to_string(index=False))
    print("\ntop restored winners")
    print(top_restored[cols].round(3).to_string(index=False))
    print(f"\nwrote {out_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Forensic charts and tables for the 2026-06-08 multi-ticker swing session.")
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--worst-n", type=int, default=12)
    args = parser.parse_args()
    write_outputs(args.audit, args.out, args.worst_n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
