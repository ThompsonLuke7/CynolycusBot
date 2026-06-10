from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from Data.plots.plots import _extract_ohlc, _plot_candles
from strategies.spy_intraday.Policy.regime_filter import StickyRegimeConfig, add_sticky_trend_regime


DEFAULT_ANALYSIS_DIR = Path(
    "Data/models/ga_xgboost/10min/analysis/"
    "phase4_1m_oof_focused_trigger_sweep_l42_s15_full_1m_train"
)
DEFAULT_SIGNAL_FRAME = DEFAULT_ANALYSIS_DIR / "phase4_signal_frame.parquet"
DEFAULT_OUT = DEFAULT_ANALYSIS_DIR / "phase4_last_2_month_regime_diagnostic.png"
DEFAULT_SEGMENTS_OUT = DEFAULT_ANALYSIS_DIR / "phase4_last_2_month_neutral_segments.csv"

REGIME_COLORS = {
    "bullish": "#22c55e",
    "bearish": "#ef4444",
    "neutral": "#2563eb",
}
REGIME_BG_ALPHA = {
    "bullish": 0.18,
    "bearish": 0.18,
    "neutral": 0.14,
}

BULL_RULES = [
    ("fast_gt_slow", "fast>slow"),
    ("close_ge_fast", "close>=fast"),
    ("slope_up", "strong slope up"),
]
BEAR_RULES = [
    ("fast_lt_slow", "fast<slow"),
    ("close_le_fast", "close<=fast"),
    ("slope_down", "strong slope down"),
]


@dataclass
class Segment:
    regime: str
    start_i: int
    end_i: int
    start_ts: pd.Timestamp
    end_ts: pd.Timestamp

    @property
    def bars(self) -> int:
        return self.end_i - self.start_i + 1

    @property
    def midpoint(self) -> float:
        return (self.start_i + self.end_i) / 2.0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot recent 10m candles with trend-regime background and neutral-rule diagnostics."
    )
    parser.add_argument("--signal-frame", default=str(DEFAULT_SIGNAL_FRAME))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--segments-out", default=str(DEFAULT_SEGMENTS_OUT))
    parser.add_argument("--months", type=int, default=2)
    parser.add_argument("--days", type=int, default=None)
    parser.add_argument("--min-neutral-label-bars", type=int, default=4)
    parser.add_argument("--max-neutral-labels", type=int, default=14)
    return parser.parse_args()


def _load_frame(path: Path, *, months: int, days: int | None) -> pd.DataFrame:
    df = pd.read_parquet(path).sort_index()
    if not isinstance(df.index, pd.DatetimeIndex):
        if "timestamp" not in df.columns:
            raise ValueError("Signal frame must have a DatetimeIndex or timestamp column.")
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()
    idx = pd.to_datetime(df.index, utc=True, errors="coerce")
    df = df.loc[pd.notna(idx)].copy()
    df.index = pd.DatetimeIndex(idx[pd.notna(idx)]).tz_convert("America/New_York")

    for col in ("open", "high", "low", "close", "ema_fast", "ema_slow"):
        if col not in df.columns:
            raise KeyError(f"Missing required column: {col}")
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["open", "high", "low", "close", "ema_fast", "ema_slow"]).copy()
    if days is not None:
        cutoff = df.index.max() - pd.Timedelta(days=max(1, int(days)))
    else:
        cutoff = df.index.max() - pd.DateOffset(months=max(1, int(months)))
    df = df[df.index >= cutoff].copy()
    if df.empty:
        raise ValueError("No rows found in selected date window.")
    return df


def _add_rules(df: pd.DataFrame) -> pd.DataFrame:
    return add_sticky_trend_regime(df, config=StickyRegimeConfig())


def _segments(df: pd.DataFrame) -> list[Segment]:
    regimes = df["trend_regime"].astype(str).to_numpy()
    segments: list[Segment] = []
    start = 0
    for i in range(1, len(df)):
        if regimes[i] != regimes[start]:
            segments.append(Segment(regimes[start], start, i - 1, df.index[start], df.index[i - 1]))
            start = i
    segments.append(Segment(regimes[start], start, len(df) - 1, df.index[start], df.index[-1]))
    return segments


def _rule_names(rows: pd.DataFrame, rules: list[tuple[str, str]], *, present: bool) -> list[str]:
    names: list[str] = []
    for col, label in rules:
        share = float(rows[col].mean()) if len(rows) else 0.0
        if present and share >= 0.5:
            names.append(label)
        elif not present and share < 0.5:
            names.append(label)
    return names


def _neutral_segment_summary(df: pd.DataFrame, segments: list[Segment]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for seg in segments:
        if seg.regime != "neutral":
            continue
        chunk = df.iloc[seg.start_i : seg.end_i + 1]
        bull_hits = _rule_names(chunk, BULL_RULES, present=True)
        bull_missing = _rule_names(chunk, BULL_RULES, present=False)
        bear_hits = _rule_names(chunk, BEAR_RULES, present=True)
        bear_missing = _rule_names(chunk, BEAR_RULES, present=False)
        bull_score = float(chunk["bull_rule_count"].mean())
        bear_score = float(chunk["bear_rule_count"].mean())
        spread_ok_share = float(chunk["ema_spread_ok"].mean())
        lean = "bull-lean" if bull_score > bear_score else ("bear-lean" if bear_score > bull_score else "mixed")
        rows.append(
            {
                "start_ts": seg.start_ts,
                "end_ts": seg.end_ts,
                "bars": seg.bars,
                "lean": lean,
                "avg_bull_rules": bull_score,
                "avg_bear_rules": bear_score,
                "spread_ok_share": spread_ok_share,
                "bull_rules_present": ", ".join(bull_hits) or "-",
                "bull_rules_missing": ", ".join(bull_missing) or "-",
                "bear_rules_present": ", ".join(bear_hits) or "-",
                "bear_rules_missing": ", ".join(bear_missing) or "-",
                "start_close": float(chunk["close"].iloc[0]),
                "end_close": float(chunk["close"].iloc[-1]),
            }
        )
    return pd.DataFrame(rows)


def _format_xticks(ax: plt.Axes, index: pd.DatetimeIndex, *, every_days: int = 3) -> None:
    dates = pd.Series(index.date, index=np.arange(len(index)))
    starts = dates.ne(dates.shift(1))
    daily_positions = dates.index[starts.to_numpy()].to_numpy()
    chosen = daily_positions[:: max(1, every_days)]
    labels = [index[int(i)].strftime("%m/%d") for i in chosen]
    ax.set_xticks(chosen)
    ax.set_xticklabels(labels, rotation=0, ha="center")


def _draw_regime_background(ax: plt.Axes, segments: list[Segment]) -> None:
    for seg in segments:
        color = REGIME_COLORS.get(seg.regime, "#64748b")
        alpha = REGIME_BG_ALPHA.get(seg.regime, 0.14)
        ax.axvspan(seg.start_i - 0.5, seg.end_i + 0.5, color=color, alpha=alpha, linewidth=0, zorder=0)


def _draw_regime_ribbon(ax: plt.Axes, segments: list[Segment], *, y: float, height: float) -> None:
    for seg in segments:
        color = REGIME_COLORS.get(seg.regime, "#64748b")
        ax.add_patch(
            Rectangle(
                (seg.start_i - 0.5, y),
                seg.bars,
                height,
                facecolor=color,
                edgecolor="none",
                alpha=0.95,
                zorder=2.4,
            )
        )


def _draw_session_lines(ax: plt.Axes, index: pd.DatetimeIndex) -> None:
    sessions = pd.Series(index.date, index=np.arange(len(index)))
    starts = sessions.ne(sessions.shift(1))
    for pos in sessions.index[starts.to_numpy()].to_numpy():
        if pos > 0:
            ax.axvline(pos - 0.5, color="#94a3b8", alpha=0.18, linewidth=0.7, zorder=0.5)


def _neutral_label(rows: pd.DataFrame) -> str:
    bull_score = float(rows["bull_rule_count"].mean())
    bear_score = float(rows["bear_rule_count"].mean())
    if bull_score >= bear_score:
        hits = _rule_names(rows, BULL_RULES, present=True)
        missing = _rule_names(rows, BULL_RULES, present=False)
        prefix = f"N bull {bull_score:.1f}/3"
    else:
        hits = _rule_names(rows, BEAR_RULES, present=True)
        missing = _rule_names(rows, BEAR_RULES, present=False)
        prefix = f"N bear {bear_score:.1f}/3"
    hit_text = "+".join(hits) if hits else "none"
    miss_text = "+".join(missing) if missing else "none"
    spread_text = "spread ok" if float(rows["ema_spread_ok"].mean()) >= 0.5 else "spread thin"
    return f"{prefix} {spread_text}\nhit: {hit_text}\nmiss: {miss_text}"


def _plot_rule_strip(ax: plt.Axes, df: pd.DataFrame) -> None:
    pos = np.arange(len(df))
    rule_rows = [
        ("Bull: fast>slow", "fast_gt_slow", "#22c55e"),
        ("Bull: close>=fast", "close_ge_fast", "#22c55e"),
        ("Bull: slope up", "slope_up", "#22c55e"),
        ("Spread >= 0.15 ATR", "ema_spread_ok", "#2563eb"),
        ("Bear: fast<slow", "fast_lt_slow", "#ef4444"),
        ("Bear: close<=fast", "close_le_fast", "#ef4444"),
        ("Bear: slope down", "slope_down", "#ef4444"),
    ]
    for y, (_label, col, color) in enumerate(rule_rows):
        values = df[col].to_numpy(dtype=bool)
        ax.scatter(pos[values], np.full(values.sum(), y), marker="s", s=7, color=color, alpha=0.7, linewidth=0)
    ax.set_yticks(np.arange(len(rule_rows)))
    ax.set_yticklabels([label for label, _col, _color in rule_rows], fontsize=8)
    ax.set_ylim(-0.7, len(rule_rows) - 0.3)
    ax.grid(axis="x", color="#cbd5e1", alpha=0.24, linewidth=0.6)
    ax.grid(axis="y", color="#cbd5e1", alpha=0.14, linewidth=0.5)
    ax.set_title("Rule checks: colored square means that rule is true on that 10min bar", fontsize=10)


def plot(df: pd.DataFrame, segments: list[Segment], neutral_summary: pd.DataFrame, args: argparse.Namespace) -> None:
    pos, open_y, high_y, low_y, close_y = _extract_ohlc(df)
    fig, (ax_price, ax_rules) = plt.subplots(
        2,
        1,
        figsize=(34, 15),
        sharex=True,
        gridspec_kw={"height_ratios": [5.6, 1.35]},
        constrained_layout=True,
    )

    _draw_regime_background(ax_price, segments)
    _draw_session_lines(ax_price, df.index)
    _plot_candles(
        ax_price,
        pos,
        open_y,
        high_y,
        low_y,
        close_y,
        up_color="#16a34a",
        down_color="#dc2626",
        wick_color="#475569",
        width=0.65,
    )
    ax_price.plot(pos, df["ema_fast"].to_numpy(dtype=float), color="#2563eb", linewidth=1.0, alpha=0.85, label="EMA fast")
    ax_price.plot(pos, df["ema_slow"].to_numpy(dtype=float), color="#f59e0b", linewidth=1.0, alpha=0.85, label="EMA slow")

    price_min = float(np.nanmin(low_y))
    price_max = float(np.nanmax(high_y))
    price_span = max(1.0, price_max - price_min)
    label_y = price_max + price_span * 0.045
    _draw_regime_ribbon(ax_price, segments, y=price_max + price_span * 0.012, height=price_span * 0.018)
    neutral_segments = [seg for seg in segments if seg.regime == "neutral" and seg.bars >= int(args.min_neutral_label_bars)]
    neutral_segments = sorted(neutral_segments, key=lambda seg: seg.bars, reverse=True)[: int(args.max_neutral_labels)]
    neutral_segments = sorted(neutral_segments, key=lambda seg: seg.start_i)
    for seg in neutral_segments:
        chunk = df.iloc[seg.start_i : seg.end_i + 1]
        ax_price.text(
            seg.midpoint,
            label_y,
            _neutral_label(chunk),
            ha="center",
            va="bottom",
            fontsize=7,
            color="#0f172a",
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "#f8fafc", "edgecolor": "#94a3b8", "alpha": 0.86},
            zorder=6,
        )

    ax_price.set_ylabel("SPY 10min")
    ax_price.set_title(
        (
            f"Last {args.days if args.days is not None else str(args.months) + ' months'}: "
            "10min bars with trend-regime background"
            f" | {df.index.min():%Y-%m-%d} to {df.index.max():%Y-%m-%d}"
        ),
        fontsize=15,
    )
    ax_price.grid(color="#cbd5e1", alpha=0.24, linewidth=0.7)
    ax_price.set_ylim(price_min - price_span * 0.03, price_max + price_span * 0.17)

    counts = df["trend_regime"].value_counts().to_dict()
    ax_price.text(
        0.01,
        0.98,
        (
            "Regime rules: bullish = fast EMA > slow EMA + close >= fast EMA + slow EMA slope up. "
            "Bearish = inverse. Sticky filter requires EMA spread, slope strength, confirmation bars, and hysteresis.\n"
            f"Bars: bullish {counts.get('bullish', 0)}, bearish {counts.get('bearish', 0)}, neutral {counts.get('neutral', 0)}. "
            f"Neutral labels show the dominant side's hit/missing checks; full neutral segment table saved to CSV."
        ),
        transform=ax_price.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color="#0f172a",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#f8fafc", "edgecolor": "#cbd5e1", "alpha": 0.86},
    )

    legend_handles = [
        Patch(facecolor=REGIME_COLORS["bullish"], alpha=0.45, label="Bullish background"),
        Patch(facecolor=REGIME_COLORS["bearish"], alpha=0.45, label="Bearish background"),
        Patch(facecolor=REGIME_COLORS["neutral"], alpha=0.45, label="Neutral background"),
        Line2D([0], [0], color="#2563eb", linewidth=1.4, label="EMA fast"),
        Line2D([0], [0], color="#f59e0b", linewidth=1.4, label="EMA slow"),
        Patch(facecolor="#16a34a", label="Bull candle"),
        Patch(facecolor="#dc2626", label="Bear candle"),
    ]
    ax_price.legend(handles=legend_handles, loc="upper left", ncol=4, fontsize=8)

    _draw_regime_background(ax_rules, segments)
    _draw_session_lines(ax_rules, df.index)
    _plot_rule_strip(ax_rules, df)
    _format_xticks(ax_rules, df.index, every_days=4)
    ax_rules.set_xlim(-1, len(df))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    df = _add_rules(_load_frame(Path(args.signal_frame), months=args.months, days=args.days))
    segments = _segments(df)
    neutral_summary = _neutral_segment_summary(df, segments)
    segments_out = Path(args.segments_out)
    segments_out.parent.mkdir(parents=True, exist_ok=True)
    neutral_summary.to_csv(segments_out, index=False)
    plot(df, segments, neutral_summary, args)
    print(f"[regime-plot] wrote {args.out}")
    print(f"[regime-plot] wrote {args.segments_out}")
    print(
        neutral_summary.sort_values("bars", ascending=False)
        .head(12)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
