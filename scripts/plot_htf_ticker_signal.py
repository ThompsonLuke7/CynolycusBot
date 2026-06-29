"""
Per-ticker HTF Swing signal view.

The HTF Swing live policy (strategies/multi_ticker_swing_htf/live/runner.py) is a
long-only top-K harness: each 4H bar it ranks the universe by ``htf_score`` and
holds the names whose cross-sectional rank clears ``--htf-rank-floor`` (default
0.85), exiting on take-profit / horizon / falling out of the top-K. There was no
single-name plotter, so this draws, for one ticker:

  * top: 4H candles with a marker on every bar where the name was rank-eligible
    (htf_xs_rank >= floor) — i.e. the bars the top-K policy could have held it;
  * bottom: htf_score and htf_xs_rank over time with the floor line.

Scores come from the Meta-Ranker matrix (it already carries htf_score +
htf_xs_rank per (timestamp, ticker)); bars from Data/shared/bars/4h.

  PYTHONPATH=. python scripts/plot_htf_ticker_signal.py --ticker WDC
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

MATRIX = Path("signals/meta_context/meta_ranker/meta_ranker_matrix.parquet")
BARS_4H = Path("Data/shared/bars/4h")
OUT_DIR = Path("strategies/multi_ticker_swing_htf/plots/ticker_signal")


def _candles(ax, bars: pd.DataFrame) -> None:
    x = range(len(bars))
    for i, (_, b) in zip(x, bars.iterrows()):
        up = b["close"] >= b["open"]
        color = "#26a69a" if up else "#ef5350"
        ax.plot([i, i], [b["low"], b["high"]], color=color, lw=0.6, zorder=1)
        lo, hi = sorted((b["open"], b["close"]))
        ax.add_patch(plt.Rectangle((i - 0.3, lo), 0.6, max(hi - lo, 1e-6),
                                   color=color, zorder=2))


def main() -> int:
    ap = argparse.ArgumentParser(description="Per-ticker HTF Swing top-K signal view.")
    ap.add_argument("--ticker", default="WDC")
    ap.add_argument("--start", default="2025-10-01")
    ap.add_argument("--end", default="2026-07-01")
    ap.add_argument("--rank-floor", type=float, default=0.85,
                    help="htf_xs_rank eligibility floor (matches runner --htf-rank-floor).")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()
    t = args.ticker.upper()

    mat = pd.read_parquet(MATRIX, columns=["htf_score", "htf_xs_rank"])
    if t not in mat.index.get_level_values("ticker"):
        print(f"{t}: not in meta matrix"); return 1
    sc = mat.xs(t, level="ticker").sort_index().loc[args.start:args.end]

    bp = BARS_4H / f"{t}.parquet"
    if not bp.exists():
        print(f"{t}: no 4H bars at {bp}"); return 1
    bars = pd.read_parquet(bp)
    bars.columns = [c.lower() for c in bars.columns]
    if "timestamp" in bars.columns:
        bars = bars.set_index("timestamp")
    bars.index = pd.to_datetime(bars.index, utc=True)
    bars = bars.sort_index().loc[args.start:args.end]

    # Align scores onto bar timestamps.
    sc = sc.reindex(bars.index)
    eligible = sc["htf_xs_rank"] >= args.rank_floor
    n_elig = int(eligible.sum())

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(15, 8), height_ratios=[3, 1], sharex=True)
    _candles(ax1, bars)
    elig_idx = [i for i, e in enumerate(eligible.values) if bool(e)]
    if elig_idx:
        ax1.scatter(elig_idx, bars["high"].values[elig_idx] * 1.02,
                    marker="v", color="#3b82f6", s=40, zorder=3,
                    label=f"htf rank ≥ {args.rank_floor} (top-K eligible)")
    ax1.set_title(
        f"{t} — 4H candles vs HTF Swing rank "
        f"({n_elig}/{len(bars)} bars top-K eligible, floor {args.rank_floor})",
        fontweight="bold")
    ax1.set_ylabel("price"); ax1.legend(loc="upper left"); ax1.grid(alpha=0.2)

    ax2.plot(range(len(sc)), sc["htf_xs_rank"].values, color="#7c3aed", lw=1.0,
             label="htf_xs_rank (cross-sec pct)")
    ax2.plot(range(len(sc)), sc["htf_score"].values, color="#06b6d4", lw=0.8,
             alpha=0.8, label="htf_score (raw)")
    ax2.axhline(args.rank_floor, color="#9ca3af", ls="--", lw=0.8)
    ax2.set_ylim(0, 1.02); ax2.set_ylabel("score"); ax2.grid(alpha=0.2)
    ax2.legend(loc="upper left", ncol=2)

    # x ticks as dates
    step = max(1, len(bars) // 12)
    ticks = list(range(0, len(bars), step))
    ax2.set_xticks(ticks)
    ax2.set_xticklabels([bars.index[i].strftime("%b %d") for i in ticks], rotation=45)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"htf_signal_{t}.png"
    fig.tight_layout(); fig.savefig(out, dpi=130, facecolor="white"); plt.close(fig)
    print(f"wrote {out}  ({n_elig} eligible bars)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
