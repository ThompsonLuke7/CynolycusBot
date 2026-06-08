"""
Example-trade visualizations: overlay leakage-free OOF signals on REAL 4H price
bars so you can see how each model's picks would have played out live.

For each model we take the highest-score name at a set of timestamps spread
across the OOF history (top-1 "live pick" per chosen bar), load that ticker's
actual 4H OHLCV from Data/shared/bars/4h, and draw the anatomy of the trade:
entry, forward window, and the realized outcome envelope (max-favorable,
max-adverse, close) taken directly from the OOF row -- no re-inference.

Output: <project>/.../oof_eval/example_trades.png  (3x3 grid per model)
Run:    python meta_context/oof_example_trades.py
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
BARS_4H = REPO / "Data/shared/bars/4h"


@dataclass
class Spec:
    name: str
    oof: Path
    out_dir: Path
    fwd_bars: int                 # forward window to draw on the x-axis
    fav_col: str                  # realized max-favorable return col
    adv_col: str                  # realized max-adverse return col
    close_col: str                # realized close return col
    n_examples: int = 9
    pre_bars: int = 40


SPECS = [
    Spec(
        name="momentum_expansion",
        oof=REPO / "momentum_expansion/data/training_import/bundle/oof_preds.parquet",
        out_dir=REPO / "momentum_expansion/plots/output/oof_eval",
        fwd_bars=25, fav_col="fwd_max_return", adv_col="fwd_max_drawdown",
        close_col="fwd_close_return",
    ),
    Spec(
        name="multi_ticker_swing_htf",
        oof=REPO / "multi_ticker_swing_htf/data/bundle/oof_preds.parquet",
        out_dir=REPO / "multi_ticker_swing_htf/plots/oof_eval",
        fwd_bars=38, fav_col="fwd_best_high_return", adv_col="fwd_worst_low_return",
        close_col="fwd_close_return",
    ),
]


def _load_bars(ticker: str) -> pd.DataFrame | None:
    p = BARS_4H / f"{ticker}.parquet"
    if not p.exists():
        return None
    b = pd.read_parquet(p)
    b["timestamp"] = pd.to_datetime(b["timestamp"], utc=True)
    return b.set_index("timestamp").sort_index()


def _pick_examples(df: pd.DataFrame, spec: Spec) -> list:
    """Top-1 score at timestamps evenly spread across the OOF date range."""
    ts = df.index.get_level_values("timestamp")
    bars = ts.unique().sort_values()
    picks = []
    seen = set()
    # one example per evenly-spaced bar across the whole OOF history, so the
    # panel spans regimes rather than clustering at the start.
    targets = bars[np.linspace(0, len(bars) - 1, spec.n_examples).astype(int)]
    for bar in targets:
        g = df.xs(bar, level="timestamp").sort_values("score", ascending=False)
        for ticker, row in g.iterrows():
            if ticker in seen:
                continue
            if (BARS_4H / f"{ticker}.parquet").exists():
                picks.append((ticker, pd.Timestamp(bar), row))
                seen.add(ticker)
                break
    return picks


def _panel(ax, ticker, entry_ts, row, spec):
    bars = _load_bars(ticker)
    if bars is None or entry_ts not in bars.index:
        # nearest bar
        if bars is not None:
            i = bars.index.searchsorted(entry_ts)
            if i >= len(bars.index):
                ax.set_visible(False); return
            entry_ts = bars.index[i]
        else:
            ax.set_visible(False); return
    i = bars.index.get_loc(entry_ts)
    lo = max(0, i - spec.pre_bars)
    hi = min(len(bars), i + spec.fwd_bars + 1)
    win = bars.iloc[lo:hi]
    entry_px = float(bars["close"].iloc[i])
    x = np.arange(len(win))
    xi = i - lo
    ax.plot(x, win["close"].values, color="#333333", lw=1.0)
    ax.axvline(xi, color="#1565C0", lw=1.2)
    ax.axvspan(xi, len(win) - 1, color="#1565C0", alpha=0.06)
    ax.axhline(entry_px, color="#888888", ls=":", lw=0.8)
    fav = row.get(spec.fav_col); adv = row.get(spec.adv_col); cl = row.get(spec.close_col)
    if pd.notna(fav):
        ax.axhline(entry_px * (1 + fav), color="#2E7D32", ls="--", lw=0.8)
    if pd.notna(adv):
        ax.axhline(entry_px * (1 + adv), color="#C62828", ls="--", lw=0.8)
    if pd.notna(cl):
        ax.axhline(entry_px * (1 + cl), color="#1565C0", ls="-", lw=0.8)
    parts = [f"{ticker} {pd.Timestamp(entry_ts).date()}", f"score={float(row['score']):.3f}"]
    if pd.notna(cl):
        parts.append(f"close {cl:+.1%}")
    if pd.notna(fav):
        parts.append(f"max {fav:+.1%}")
    if pd.notna(adv):
        parts.append(f"adv {adv:+.1%}")
    ax.set_title("  ".join(parts), fontsize=7.5)
    ax.tick_params(labelsize=6)
    ax.set_xticks([])


def run(spec: Spec):
    df = pd.read_parquet(spec.oof)
    picks = _pick_examples(df, spec)
    n = len(picks)
    if n == 0:
        print(f"  {spec.name}: no plottable examples"); return
    rows = int(np.ceil(n / 3))
    fig, axes = plt.subplots(rows, 3, figsize=(13, 3.2 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[n:]:
        ax.set_visible(False)
    for ax, (ticker, ts, row) in zip(axes, picks):
        _panel(ax, ticker, ts, row, spec)
    fig.suptitle(
        f"{spec.name}: example top-1 picks on real 4H prices "
        f"(grey=price, blue=entry & forward window; dashed green/red = realized "
        f"max-favorable / max-adverse, solid blue = realized close)",
        fontsize=9,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    spec.out_dir.mkdir(parents=True, exist_ok=True)
    out = spec.out_dir / "example_trades.png"
    fig.savefig(out, dpi=120); plt.close(fig)
    print(f"  {spec.name}: wrote {out} ({n} examples)")


def run_theme():
    """Theme model is basket-level: build a cumulative theme index from
    theme_return_1d and show top-1 theme picks vs their realized 20d excess."""
    oof = pd.read_parquet(REPO / "theme_expansion/models/bundle/oof_preds.parquet")
    oof["date"] = pd.to_datetime(oof["date"])
    daily = pd.read_parquet(REPO / "theme_expansion/outputs/theme_daily.parquet")
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values(["theme", "date"])
    # cumulative index per theme from daily returns
    daily["cum"] = daily.groupby("theme")["theme_return_1d"].transform(
        lambda s: (1 + s.fillna(0)).cumprod()
    )
    idx = {t: g.set_index("date")["cum"] for t, g in daily.groupby("theme")}

    out_dir = REPO / "theme_expansion/outputs/plots/oof_eval"
    dates = np.sort(oof["date"].unique())
    targets = dates[np.linspace(0, len(dates) - 1, 9).astype(int)]
    picks = []
    seen = set()
    for d in targets:
        g = oof[oof["date"] == d].sort_values("score", ascending=False)
        for _, row in g.iterrows():
            t = row["theme"]
            if t in seen or t not in idx:
                continue
            picks.append((t, pd.Timestamp(d), row)); seen.add(t); break

    fig, axes = plt.subplots(3, 3, figsize=(13, 9.6))
    axes = axes.ravel()
    for ax in axes[len(picks):]:
        ax.set_visible(False)
    for ax, (theme, d, row) in zip(axes, picks):
        s = idx[theme]
        i = s.index.searchsorted(d)
        if i >= len(s):
            ax.set_visible(False); continue
        lo = max(0, i - 40); hi = min(len(s), i + 21)
        win = s.iloc[lo:hi]
        base = float(s.iloc[i])
        x = np.arange(len(win)); xi = i - lo
        ax.plot(x, (win / base).values, color="#333333", lw=1.0)
        ax.axvline(xi, color="#1565C0", lw=1.2)
        ax.axvspan(xi, len(win) - 1, color="#1565C0", alpha=0.06)
        ax.axhline(1.0, color="#888888", ls=":", lw=0.8)
        y = row["y"]
        ax.set_title(f"{theme} {pd.Timestamp(d).date()}  score={row['score']:.3f}  "
                     f"20d excess {y:+.1%}", fontsize=7.5)
        ax.tick_params(labelsize=6); ax.set_xticks([])
    fig.suptitle("theme_expansion: example top-1 theme picks (cumulative theme "
                 "index, base=1 at entry; blue=entry & forward 20d; title shows "
                 "realized excess vs benchmark)", fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "example_trades.png", dpi=120); plt.close(fig)
    print(f"  theme_expansion: wrote {out_dir/'example_trades.png'} ({len(picks)} examples)")


def main():
    for spec in SPECS:
        print(f"=== {spec.name} ===")
        run(spec)
    print("=== theme_expansion ===")
    run_theme()


if __name__ == "__main__":
    main()
