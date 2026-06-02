from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Rectangle


DEFAULT_AUDITS = [
    Path("UI/swing_audit/swing_session_20260528T120501Z.jsonl"),
    Path("UI/swing_audit/swing_session_20260529T120845Z.jsonl"),
]
OUT_DIR = Path("Data/analysis/multi_ticker_swing_live/signal_plots")
RAW_30M_DIR = Path("multi_ticker_swing/data/raw/30m")


def _ts(value: Any) -> pd.Timestamp:
    if value is None:
        return pd.NaT
    return pd.to_datetime(value, utc=True)


def _bar_ts(value: Any) -> pd.Timestamp:
    if value is None:
        return pd.NaT
    return pd.to_datetime(value, unit="s", utc=True)


def _load_audit(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    signals: list[dict[str, Any]] = []
    positions: list[dict[str, Any]] = []
    bars5: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            typ = obj.get("type") or obj.get("event") or ""
            payload = obj.get("payload") or {}
            event_ts = _ts(obj.get("ts"))
            if typ == "signal":
                signals.append(
                    {
                        "audit": path.stem,
                        "ts": event_ts,
                        "ticker": payload.get("ticker"),
                        "direction": int(payload.get("direction") or 0),
                        "p_dir": payload.get("p_dir"),
                        "ev_score": payload.get("ev_score"),
                        "ref_high": payload.get("ref_high"),
                        "ref_low": payload.get("ref_low"),
                        "atr": payload.get("atr"),
                    }
                )
            elif typ in {"position_opened", "position_closed", "entry_skipped", "position_close_failed"}:
                positions.append(
                    {
                        "audit": path.stem,
                        "ts": event_ts,
                        "event": typ,
                        "ticker": payload.get("ticker"),
                        "direction": int(payload.get("direction") or 0),
                        "entry_price": payload.get("entry_price"),
                        "exit_price": payload.get("exit_price"),
                        "exit_reason": payload.get("exit_reason"),
                        "reason": payload.get("reason"),
                        "option_symbol": payload.get("option_symbol"),
                    }
                )
            elif typ == "position_bar_5m":
                bar = payload.get("bar") or {}
                bars5.append(
                    {
                        "audit": path.stem,
                        "ticker": payload.get("ticker"),
                        "ts": _bar_ts(bar.get("time")),
                        "open": bar.get("open"),
                        "high": bar.get("high"),
                        "low": bar.get("low"),
                        "close": bar.get("close"),
                        "volume": bar.get("volume"),
                    }
                )
            elif typ == "position_chart_seed":
                ticker = payload.get("ticker")
                for bar in payload.get("pre_entry_bars") or []:
                    bars5.append(
                        {
                            "audit": path.stem,
                            "ticker": ticker,
                            "ts": _bar_ts(bar.get("time")),
                            "open": bar.get("open"),
                            "high": bar.get("high"),
                            "low": bar.get("low"),
                            "close": bar.get("close"),
                            "volume": bar.get("volume"),
                        }
                    )

    sig = pd.DataFrame(signals)
    pos = pd.DataFrame(positions)
    bars = pd.DataFrame(bars5)
    if not bars.empty:
        for col in ["open", "high", "low", "close", "volume"]:
            bars[col] = pd.to_numeric(bars[col], errors="coerce")
        bars = bars.dropna(subset=["ts", "ticker", "open", "high", "low", "close"])
        bars = bars.drop_duplicates(["ticker", "ts"]).sort_values(["ticker", "ts"])
    return sig, pos, bars


def _load_raw_30m(ticker: str) -> pd.DataFrame:
    path = RAW_30M_DIR / f"{ticker}.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    if "timestamp" in df.columns:
        df = df.set_index("timestamp")
    df.index = pd.to_datetime(df.index, utc=True)
    df = df.rename_axis("ts").reset_index()
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df[["ts", "open", "high", "low", "close", "volume"]].dropna(subset=["ts", "open", "high", "low", "close"])


def _aggregate_5m_to_30m(bars5: pd.DataFrame) -> pd.DataFrame:
    if bars5.empty:
        return pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume"])
    b = bars5.copy()
    b["bucket"] = b["ts"].dt.floor("30min")
    out = (
        b.sort_values("ts")
        .groupby("bucket", as_index=False)
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
        )
        .rename(columns={"bucket": "ts"})
    )
    return out


def _bars_for_ticker(ticker: str, bars5: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    raw = _load_raw_30m(ticker)
    raw = raw[(raw["ts"] >= start) & (raw["ts"] <= end)] if not raw.empty else raw
    audit = bars5[bars5["ticker"] == ticker] if not bars5.empty else pd.DataFrame()
    audit30 = _aggregate_5m_to_30m(audit)
    audit30 = audit30[(audit30["ts"] >= start) & (audit30["ts"] <= end)] if not audit30.empty else audit30
    if raw.empty:
        return audit30.sort_values("ts")
    if audit30.empty:
        return raw.sort_values("ts")
    combined = pd.concat([raw, audit30], ignore_index=True).sort_values("ts")
    return combined.drop_duplicates("ts", keep="last").sort_values("ts")


def _draw_candles(ax: plt.Axes, bars: pd.DataFrame) -> None:
    if bars.empty:
        return
    x = mdates.date2num(bars["ts"].dt.tz_convert("America/New_York").dt.tz_localize(None))
    width = 0.012
    for xpos, row in zip(x, bars.itertuples(index=False), strict=False):
        color = "#0f9d76" if row.close >= row.open else "#d94c4c"
        ax.vlines(xpos, row.low, row.high, color=color, linewidth=1.1, alpha=0.95)
        body_low = min(row.open, row.close)
        body_h = max(abs(row.close - row.open), 0.01)
        ax.add_patch(Rectangle((xpos - width / 2, body_low), width, body_h, color=color, alpha=0.85))
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    ax.grid(True, alpha=0.25)


def _nearest_close(bars: pd.DataFrame, ts: pd.Timestamp) -> float:
    if bars.empty:
        return math.nan
    idx = bars["ts"].searchsorted(ts, side="right") - 1
    if idx < 0:
        idx = 0
    return float(bars.iloc[idx]["close"])


def _forward_return(bars: pd.DataFrame, ts: pd.Timestamp, direction: int, horizon_bars: int) -> float:
    if bars.empty or direction == 0:
        return math.nan
    idx = bars["ts"].searchsorted(ts, side="right") - 1
    if idx < 0:
        idx = 0
    exit_idx = min(idx + horizon_bars, len(bars) - 1)
    entry = float(bars.iloc[idx]["close"])
    exit_price = float(bars.iloc[exit_idx]["close"])
    return direction * (exit_price - entry) / entry * 100 if entry else math.nan


def _plot_ticker(ticker: str, bars: pd.DataFrame, signals: pd.DataFrame, positions: pd.DataFrame, out: Path) -> None:
    if bars.empty:
        return
    fig, ax = plt.subplots(figsize=(15, 7))
    _draw_candles(ax, bars)
    local_dates = bars["ts"].dt.tz_convert("America/New_York").dt.tz_localize(None)
    for _, sig in signals.iterrows():
        price = _nearest_close(bars, sig["ts"])
        xpos = mdates.date2num(sig["ts"].tz_convert("America/New_York").tz_localize(None))
        if sig["direction"] == 1:
            ax.scatter(xpos, price, marker="^", s=110, color="#1f77b4", zorder=5)
            label_y = price * 1.005
        else:
            ax.scatter(xpos, price, marker="v", s=110, color="#b00020", zorder=5)
            label_y = price * 0.995
        ax.text(
            xpos,
            label_y,
            f"{'L' if sig['direction'] == 1 else 'S'} p={sig['p_dir']:.2f}",
            fontsize=8,
            ha="center",
            va="bottom" if sig["direction"] == 1 else "top",
        )
        if pd.notna(sig.get("ref_high")):
            ax.hlines(sig["ref_high"], xmin=xpos - 0.02, xmax=xpos + 0.08, color="#1f77b4", linestyle=":", alpha=0.5)
        if pd.notna(sig.get("ref_low")):
            ax.hlines(sig["ref_low"], xmin=xpos - 0.02, xmax=xpos + 0.08, color="#b00020", linestyle=":", alpha=0.5)

    for _, pos in positions.iterrows():
        price = pos.get("entry_price") if pos["event"] == "position_opened" else pos.get("exit_price")
        if pd.isna(price):
            price = _nearest_close(bars, pos["ts"])
        xpos = mdates.date2num(pos["ts"].tz_convert("America/New_York").tz_localize(None))
        if pos["event"] == "position_opened":
            ax.scatter(xpos, price, marker="o", s=75, facecolors="none", edgecolors="black", linewidths=1.7, zorder=6)
            ax.text(xpos, price, "open", fontsize=7, ha="left", va="bottom")
        elif pos["event"] == "position_closed":
            ax.scatter(xpos, price, marker="x", s=90, color="black", zorder=6)
            ax.text(xpos, price, f"close {pos.get('exit_reason') or ''}", fontsize=7, ha="left", va="top")
        elif pos["event"] == "position_close_failed":
            ax.scatter(xpos, price, marker="X", s=110, color="#ff7f0e", zorder=6)
            ax.text(xpos, price, "close failed", fontsize=7, ha="left", va="top")

    ax.set_title(f"{ticker} 30m Candles With Live Multi-Ticker Swing Signals")
    ax.set_ylabel("Underlying price")
    if len(local_dates):
        ax.set_xlim(local_dates.min(), local_dates.max())
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def run(audits: list[Path], tickers: list[str], days_before: int, days_after: int) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sig_frames: list[pd.DataFrame] = []
    pos_frames: list[pd.DataFrame] = []
    bar_frames: list[pd.DataFrame] = []
    for audit in audits:
        if not audit.exists():
            continue
        sig, pos, bars5 = _load_audit(audit)
        sig_frames.append(sig)
        pos_frames.append(pos)
        bar_frames.append(bars5)
    signals = pd.concat(sig_frames, ignore_index=True) if sig_frames else pd.DataFrame()
    positions = pd.concat(pos_frames, ignore_index=True) if pos_frames else pd.DataFrame()
    bars5 = pd.concat(bar_frames, ignore_index=True) if bar_frames else pd.DataFrame()
    if tickers:
        tickers_set = {t.upper() for t in tickers}
    else:
        tickers_set = set(str(t).upper() for t in signals["ticker"].dropna().unique())
    rows: list[dict[str, Any]] = []
    for ticker in sorted(tickers_set):
        sig_t = signals[signals["ticker"].str.upper() == ticker].copy() if not signals.empty else pd.DataFrame()
        pos_t = positions[positions["ticker"].str.upper() == ticker].copy() if not positions.empty else pd.DataFrame()
        if sig_t.empty and pos_t.empty:
            continue
        anchor_ts = pd.concat([sig_t.get("ts", pd.Series(dtype="datetime64[ns, UTC]")), pos_t.get("ts", pd.Series(dtype="datetime64[ns, UTC]"))]).dropna()
        if anchor_ts.empty:
            continue
        start = anchor_ts.min() - pd.Timedelta(days=days_before)
        end = anchor_ts.max() + pd.Timedelta(days=days_after)
        bars = _bars_for_ticker(ticker, bars5, start, end)
        _plot_ticker(ticker, bars, sig_t, pos_t, OUT_DIR / f"{ticker}_signals_30m.png")
        for _, sig in sig_t.iterrows():
            rows.append(
                {
                    "ticker": ticker,
                    "audit": sig["audit"],
                    "signal_ts": sig["ts"],
                    "direction": sig["direction"],
                    "p_dir": sig["p_dir"],
                    "ev_score": sig["ev_score"],
                    "entry_close": _nearest_close(bars, sig["ts"]),
                    "ret_1x30m_pct": _forward_return(bars, sig["ts"], sig["direction"], 1),
                    "ret_2x30m_pct": _forward_return(bars, sig["ts"], sig["direction"], 2),
                    "ret_4x30m_pct": _forward_return(bars, sig["ts"], sig["direction"], 4),
                    "ret_8x30m_pct": _forward_return(bars, sig["ts"], sig["direction"], 8),
                    "bars_available": len(bars),
                }
            )
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT_DIR / "signal_forward_returns_30m.csv", index=False)
    if not summary.empty:
        agg_rows = []
        for horizon in ["ret_1x30m_pct", "ret_2x30m_pct", "ret_4x30m_pct", "ret_8x30m_pct"]:
            values = summary[horizon].dropna()
            agg_rows.append(
                {
                    "horizon": horizon,
                    "signals": int(values.shape[0]),
                    "avg_signed_ret_pct": values.mean(),
                    "median_signed_ret_pct": values.median(),
                    "directional_win_rate": (values > 0).mean(),
                }
            )
        pd.DataFrame(agg_rows).to_csv(OUT_DIR / "signal_forward_returns_30m_summary.csv", index=False)
        print(pd.DataFrame(agg_rows).round(4).to_string(index=False))
    print("wrote plots to", OUT_DIR)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", action="append", type=Path, dest="audits")
    parser.add_argument("--tickers", nargs="*", default=["MDB", "PAAS", "IBM", "GDX", "CCL"])
    parser.add_argument("--days-before", type=int, default=1)
    parser.add_argument("--days-after", type=int, default=1)
    args = parser.parse_args()
    run(args.audits or DEFAULT_AUDITS, args.tickers, args.days_before, args.days_after)


if __name__ == "__main__":
    main()
