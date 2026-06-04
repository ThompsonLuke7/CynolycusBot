"""Regenerate the LWLG bullish + AAL bearish catalyst example plots using
the post-rerun news corpus.

Looks for the LWLG 2026-03-11 Tower Semi pact catalyst (newly captured via
Google News RSS) for the bullish example, falls back to the 2025-07-11 record
if the new one isn't present. AAL bearish example uses the EX-99-enriched 8-K.

Output: catalysts/data/processed/example_plots/*_refreshed_*.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from momentum_expansion.config.momentum_config import RAW_1D_DIR

OUT_DIR = Path("catalysts/data/processed/example_plots")
NEWS_RECORDS = Path("news/data/processed/news_records.parquet")
BARS_DIR = RAW_1D_DIR


def load_bars_window(ticker: str, event_ts: pd.Timestamp, before: int = 15, after: int = 40) -> tuple[pd.DataFrame, int]:
    b = pd.read_parquet(BARS_DIR / f"{ticker}.parquet")
    b["t"] = pd.to_datetime(b["timestamp"], utc=True).dt.tz_convert(None)
    ev = event_ts.tz_convert(None).normalize() if event_ts.tzinfo else event_ts.normalize()
    win = b[(b["t"] >= ev - pd.Timedelta(days=before * 2)) & (b["t"] <= ev + pd.Timedelta(days=after * 2))].copy()
    win = win.sort_values("t").reset_index(drop=True)
    if len(win) == 0:
        raise ValueError(f"no bars for {ticker} around {event_ts}")
    idx_ev = int((win["t"] - ev).abs().idxmin())
    lo = max(0, idx_ev - before)
    hi = min(len(win), idx_ev + after + 1)
    return win.iloc[lo:hi].reset_index(drop=True), idx_ev - lo


def headline(text: str, n: int = 180) -> str:
    t = " ".join(str(text).split())
    return t[:n] + ("..." if len(t) > n else "")


def plot_example(rec: pd.Series, color: str, label: str, fname: str) -> Path:
    bars, evi = load_bars_window(rec["ticker"], pd.to_datetime(rec["timestamp"], utc=True))
    event_close = float(bars.loc[evi, "close"])

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})
    ax = axes[0]
    for _, r in bars.iterrows():
        c = "#2ca02c" if r["close"] >= r["open"] else "#d62728"
        ax.vlines(r["t"], r["low"], r["high"], color=c, linewidth=0.8, alpha=0.7)
        ax.add_patch(plt.Rectangle((mdates.date2num(r["t"]) - 0.3, min(r["open"], r["close"])),
                                   0.6, abs(r["close"] - r["open"]) + 1e-9,
                                   color=c, alpha=0.6, linewidth=0))
    ax.plot(bars["t"], bars["close"], color="black", linewidth=0.9, alpha=0.5)
    ev_t = bars.loc[evi, "t"]
    ax.axvline(ev_t, color=color, linewidth=1.6, alpha=0.85)
    ax.scatter([ev_t], [event_close], color=color, zorder=5, s=70, edgecolor="black", linewidth=0.8,
               label=f"catalyst @ ${event_close:.2f}")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_ylabel("price ($)")
    title = (
        f"{label}: {rec['ticker']}  |  family={rec.get('catalyst_family', '?')}  subtype={rec.get('catalyst_subtype', '?')}\n"
        f"date: {pd.to_datetime(rec['timestamp']).strftime('%Y-%m-%d %H:%M UTC')}   source: {rec.get('source', '?')}\n"
        f'"{headline(rec.get("headline", "") or rec.get("text", ""))}"'
    )
    ax.set_title(title, loc="left", fontsize=10)
    ax.grid(True, alpha=0.25)

    ax2 = axes[1]
    pre = bars.iloc[: evi + 1]
    fwd = bars.iloc[evi:]
    ax2.plot(pre["t"], pre["close"] / event_close - 1.0, color="gray", linewidth=1.2, alpha=0.55, label="pre-catalyst")
    ax2.plot(fwd["t"], fwd["close"] / event_close - 1.0, color=color, linewidth=2.0, label="post-catalyst cumulative return")
    ax2.axhline(0, color="black", linewidth=0.6, alpha=0.5)
    ax2.axvline(ev_t, color=color, linewidth=1.4, alpha=0.6)
    for n_days, mark in [(1, "1d"), (5, "5d"), (10, "10d"), (20, "20d"), (40, "40d")]:
        if evi + n_days < len(bars):
            rr = bars.iloc[evi + n_days]
            ret = rr["close"] / event_close - 1.0
            ax2.scatter([rr["t"]], [ret], color=color, s=35, zorder=5, edgecolor="black", linewidth=0.6)
            ax2.annotate(f"{mark}: {ret * 100:+.1f}%", xy=(rr["t"], ret),
                         xytext=(4, 6), textcoords="offset points", fontsize=8)
    ax2.set_ylabel("return from event close")
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x * 100:+.0f}%"))
    ax2.grid(True, alpha=0.25)
    ax2.legend(loc="upper left", fontsize=9)

    body = rec.get("body", "") or ""
    if body:
        body_clean = " ".join(str(body).split())
        for anchor in ("FOR RELEASE:", "RELEASE:", "AMERICAN AIRLINES", "Company"):
            pos = body_clean.find(anchor)
            if 0 < pos < 600:
                body_clean = body_clean[pos:]
                break
        # Append below the figure as a separate caption line (does not affect axes layout)
        import textwrap
        wrapped = textwrap.fill(body_clean[:400], width=120)
        fig.text(0.5, -0.04, f"body preview (EX-99 / news prose): {wrapped}",
                 ha="center", va="top", fontsize=8, family="monospace", color="#444")

    ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    fig.suptitle(f"Catalyst module — {label.lower()} (post-rerun)", fontsize=12, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    out = OUT_DIR / fname
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> int:
    nr = pd.read_parquet(NEWS_RECORDS)
    nr["ts"] = pd.to_datetime(nr["timestamp"], utc=True)

    # ---- Bullish: LWLG Tower Semi pact, March 2026 ----
    lwlg = nr[(nr["ticker"] == "LWLG") & (nr["ts"] >= "2026-03-01") & (nr["ts"] <= "2026-03-31")].copy()
    if not lwlg.empty:
        # prefer headlines that mention Tower or GlobalFoundries (the actual catalyst)
        mask_catalyst = lwlg["headline"].fillna("").str.contains("Tower|GlobalFoundries|pact|deal|partnership|GF", case=False, regex=True)
        candidate = lwlg[mask_catalyst].sort_values("ts").head(1)
        if candidate.empty:
            candidate = lwlg.sort_values("ts").head(1)
        bull = candidate.iloc[0]
        bull_path = plot_example(bull, "#1f77b4", "BULLISH CATALYST",
                                 f"catalyst_refreshed_bullish_LWLG_{bull['ts'].strftime('%Y-%m-%d')}.png")
        print(f"wrote {bull_path}")
    else:
        print("LWLG March 2026 catalyst not in news_records — collection may not have captured it")

    # ---- Bearish: AAL 2025-01-23 earnings ----
    aal = nr[(nr["ticker"] == "AAL") & (nr["ts"] >= "2025-01-23") & (nr["ts"] <= "2025-01-24")].copy()
    if not aal.empty:
        # prefer the 8-K with EX-99 enriched body
        aal = aal.copy()
        aal["body_len"] = aal["body"].fillna("").str.len()
        bear = aal.sort_values("body_len", ascending=False).iloc[0]
        bear_path = plot_example(bear, "#b7410e", "BEARISH CATALYST",
                                 f"catalyst_refreshed_bearish_AAL_{bear['ts'].strftime('%Y-%m-%d')}.png")
        print(f"wrote {bear_path}")
    else:
        print("AAL 2025-01-23 catalyst missing from news_records")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
