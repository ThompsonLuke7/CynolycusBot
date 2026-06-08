"""
Completed-window batch: SETTLED recent performance of the actual exported models.

Unlike recent_inference_test.py (signals too fresh to have finished their forward
window), this only scores entry bars whose FULL forward label window has already
elapsed, so realized returns are final. It scores the shipped booster on a recent
run of settled entry bars, takes top-N each bar, and measures the realized
forward return straight from price bars -- a true recent out-of-sample backtest.

Forward windows: momentum 25 x 4H (~10 td), HTF 38 x 4H (~15 td), theme 20 td.

Outputs: <project>/.../oof_eval/completed_window.png + completed_window_summary.json
Run:     python meta_context/completed_window_batch.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb

REPO = Path(__file__).resolve().parents[1]
BARS_4H = REPO / "Data/shared/bars/4h"
MOM_FEATURES = REPO / "momentum_expansion/data/processed/features_4h.parquet"

TOP_N = 10
N_ENTRY_BARS = 40       # recent settled entry bars (ticker models)
N_ENTRY_DATES = 18      # recent settled entry dates (theme)

_BAR_CACHE: dict[str, pd.DataFrame | None] = {}


def _bars(t):
    if t not in _BAR_CACHE:
        p = BARS_4H / f"{t}.parquet"
        if p.exists():
            b = pd.read_parquet(p)
            b["timestamp"] = pd.to_datetime(b["timestamp"], utc=True)
            _BAR_CACHE[t] = b.set_index("timestamp").sort_index()
        else:
            _BAR_CACHE[t] = None
    return _BAR_CACHE[t]


def _fwd(t, entry_ts, w):
    """Realized close/high/low return over w bars forward from entry_ts."""
    b = _bars(t)
    if b is None:
        return None
    i = b.index.searchsorted(entry_ts)
    if i >= len(b) or b.index[i] != entry_ts:
        return None
    if i + w >= len(b):
        return None
    e = float(b["close"].iloc[i])
    seg = b.iloc[i: i + w + 1]
    return {
        "close": float(b["close"].iloc[i + w] / e - 1),
        "fav": float(seg["high"].max() / e - 1),
        "adv": float(seg["low"].min() / e - 1),
    }


def run_ticker_model(name, model_path, feats, recent, w, out_dir, label):
    bars = recent.index.get_level_values("timestamp").unique().sort_values()
    # settled entry bars: need a SPY reference with w forward bars available
    spy = _bars("SPY")
    settled = [b for b in bars if (spy is not None
               and spy.index.searchsorted(b) + w < len(spy)
               and (spy.index.searchsorted(b) < len(spy) and spy.index[min(spy.index.searchsorted(b), len(spy)-1)] == b))]
    settled = settled[-N_ENTRY_BARS:]
    if not settled:
        print(f"  {name}: no settled entry bars"); return None
    bst = xgb.Booster(); bst.load_model(str(model_path))

    pick_rets, by_bar, scatter = [], [], []
    for b in settled:
        cs = recent.xs(b, level="timestamp")
        sc = bst.predict(xgb.DMatrix(cs[feats], missing=np.nan, feature_names=feats))
        cs = cs.assign(score=sc).sort_values("score", ascending=False)
        bar_rets = []
        for ticker in cs.index[:TOP_N * 3]:   # headroom for names lacking fwd bars
            fwd = _fwd(ticker, b, w)
            if fwd is None:
                continue
            bar_rets.append(fwd["close"])
            scatter.append((float(cs.loc[ticker, "score"]), fwd["close"]))
            if len(bar_rets) >= TOP_N:
                break
        if not bar_rets:
            continue
        sp = _fwd("SPY", b, w)
        by_bar.append({"bar": b, "top_mean": float(np.mean(bar_rets)),
                       "spy": sp["close"] if sp else np.nan})
        pick_rets.extend(bar_rets)

    pr = np.array(pick_rets)
    bb = pd.DataFrame(by_bar)
    summary = {
        "name": name, "forward_bars": w, "n_entry_bars": len(bb), "n_picks": int(len(pr)),
        "entry_range": [str(bb["bar"].min()), str(bb["bar"].max())] if len(bb) else [],
        "pick_mean": float(pr.mean()), "pick_median": float(np.median(pr)),
        "pick_win_rate": float((pr > 0).mean()),
        "top_mean_per_bar": float(bb["top_mean"].mean()) if len(bb) else None,
        "spy_mean_per_bar": float(bb["spy"].mean()) if len(bb) else None,
        "excess_vs_spy": float((bb["top_mean"] - bb["spy"]).mean()) if len(bb) else None,
        "bar_win_rate_vs_spy": float((bb["top_mean"] > bb["spy"]).mean()) if len(bb) else None,
    }
    _plot(out_dir, name, label, w, pr, bb, scatter, summary)
    return summary


def _plot(out_dir, name, label, w, pr, bb, scatter, summary):
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    ax = axes[0]
    ax.hist(pr * 100, bins=30, color="#4C72B0", alpha=0.85)
    ax.axvline(pr.mean() * 100, color="#2E7D32", ls="--", lw=1.3, label=f"mean {pr.mean():+.1%}")
    ax.axvline(np.median(pr) * 100, color="#C44E52", ls="--", lw=1.3, label=f"median {np.median(pr):+.1%}")
    ax.axvline(0, color="k", lw=0.7)
    ax.set_xlabel(f"realized {w}-bar fwd close return (%)"); ax.set_ylabel("top-N picks")
    ax.set_title(f"Settled returns  (win {summary['pick_win_rate']:.0%})"); ax.legend(fontsize=7)

    ax = axes[1]
    if len(bb):
        ax.plot(bb["bar"], bb["top_mean"] * 100, color="#2E7D32", lw=1.4, marker="o", ms=3, label="top-N mean")
        ax.plot(bb["bar"], bb["spy"] * 100, color="#1565C0", lw=1.2, ls="--", label="SPY")
        ax.axhline(0, color="k", lw=0.7)
        ax.tick_params(axis="x", labelrotation=30, labelsize=6)
    ax.set_ylabel(f"{w}-bar fwd return (%)")
    ax.set_title(f"Per entry bar vs SPY  (beat {summary['bar_win_rate_vs_spy']:.0%})"); ax.legend(fontsize=7)

    ax = axes[2]
    if scatter:
        s = np.array(scatter)
        ax.scatter(s[:, 0], s[:, 1] * 100, s=10, alpha=0.4, color="#55A868")
        ax.axhline(0, color="k", lw=0.7)
    ax.set_xlabel("model score"); ax.set_ylabel(f"realized {w}-bar return (%)")
    ax.set_title("Score vs realized (settled)")

    fig.suptitle(f"{label} — completed-window batch ({summary['n_entry_bars']} settled entry bars, "
                 f"{summary['entry_range'][0][:10]}→{summary['entry_range'][1][:10]})", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_dir / "completed_window.png", dpi=120); plt.close(fig)
    (out_dir / "completed_window_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"  {name}: {summary['n_entry_bars']} bars | top-N mean {summary['top_mean_per_bar']:+.2%} "
          f"vs SPY {summary['spy_mean_per_bar']:+.2%} (excess {summary['excess_vs_spy']:+.2%}, "
          f"beat SPY {summary['bar_win_rate_vs_spy']:.0%}) | pick win {summary['pick_win_rate']:.0%}")


def run_theme(w_days=20):
    feats = json.load(open(REPO / "theme_expansion/models/bundle/eval_metrics.json"))["feature_columns"]
    lab = pd.read_parquet(REPO / "theme_expansion/outputs/theme_signal_labels.parquet")
    lab["date"] = pd.to_datetime(lab["date"])
    daily = pd.read_parquet(REPO / "theme_expansion/outputs/theme_daily.parquet")
    daily["date"] = pd.to_datetime(daily["date"]); daily = daily.sort_values(["theme", "date"])
    daily["cum"] = daily.groupby("theme")["theme_return_1d"].transform(lambda s: (1 + s.fillna(0)).cumprod())
    idx = {t: g.set_index("date")["cum"] for t, g in daily.groupby("theme")}
    db = pd.read_parquet(REPO / "theme_expansion/outputs/daily_bars.parquet", columns=["date", "ticker", "close"])
    spy = db[db["ticker"] == "SPY"].copy(); spy["date"] = pd.to_datetime(spy["date"])
    spy = spy.set_index("date")["close"].sort_index()

    bst = xgb.Booster(); bst.load_model(str(REPO / "theme_expansion/models/bundle/theme_xgb.json"))
    all_dates = np.sort(lab["date"].unique())
    # settled: at least w_days trading days of theme history after the date
    ref = next(iter(idx.values())).index
    settled = [d for d in all_dates if ref.searchsorted(d) + w_days < len(ref)][-N_ENTRY_DATES:]

    pick_rets, by_bar, scatter = [], [], []
    for d in settled:
        cs = lab[lab["date"] == d].copy()
        dummies = pd.get_dummies(cs[["category", "benchmark", "signal_market_playbook"]],
                                 prefix=["category", "benchmark", "signal_market_playbook"])
        num = [f for f in feats if f in cs.columns]
        X = pd.concat([cs[num].reset_index(drop=True), dummies.reset_index(drop=True)], axis=1)
        X = X.reindex(columns=feats, fill_value=0).astype(float)
        cs = cs.assign(score=bst.predict(xgb.DMatrix(X, missing=np.nan, feature_names=feats)))
        cs = cs.sort_values("score", ascending=False)
        rets = []
        for _, r in cs.head(TOP_N * 3).iterrows():
            s = idx.get(r["theme"])
            if s is None:
                continue
            i = s.index.searchsorted(d)
            if i >= len(s) or i + w_days >= len(s):
                continue
            ret = float(s.iloc[i + w_days] / s.iloc[i] - 1)
            rets.append(ret); scatter.append((float(r["score"]), ret))
            if len(rets) >= TOP_N:
                break
        if not rets:
            continue
        j = spy.index.searchsorted(d)
        spy_ret = float(spy.iloc[j + w_days] / spy.iloc[j] - 1) if j + w_days < len(spy) else np.nan
        by_bar.append({"bar": pd.Timestamp(d), "top_mean": float(np.mean(rets)), "spy": spy_ret})
        pick_rets.extend(rets)

    pr = np.array(pick_rets); bb = pd.DataFrame(by_bar)
    summary = {
        "name": "theme_expansion", "forward_bars": f"{w_days}td", "n_entry_bars": len(bb), "n_picks": int(len(pr)),
        "entry_range": [str(bb["bar"].min()), str(bb["bar"].max())] if len(bb) else [],
        "pick_mean": float(pr.mean()), "pick_median": float(np.median(pr)),
        "pick_win_rate": float((pr > 0).mean()),
        "top_mean_per_bar": float(bb["top_mean"].mean()) if len(bb) else None,
        "spy_mean_per_bar": float(bb["spy"].mean()) if len(bb) else None,
        "excess_vs_spy": float((bb["top_mean"] - bb["spy"]).mean()) if len(bb) else None,
        "bar_win_rate_vs_spy": float((bb["top_mean"] > bb["spy"]).mean()) if len(bb) else None,
    }
    _plot(REPO / "theme_expansion/outputs/plots/oof_eval", "theme_expansion",
          "Theme expansion", w_days, pr, bb, scatter, summary)
    return summary


def main():
    htf_feat = json.load(open(REPO / "multi_ticker_swing_htf/data/bundle/eval_metrics.json"))["feature_columns"]
    mom_feat = json.load(open(REPO / "momentum_expansion/data/training_import/bundle/eval_metrics.json"))["feature_columns"]
    union = sorted(set(htf_feat) | set(mom_feat))
    import pyarrow.parquet as pq
    tcol = pd.to_datetime(pq.read_table(MOM_FEATURES, columns=["timestamp"]).column("timestamp").to_pandas(), utc=True)
    cut = tcol.max() - pd.Timedelta(days=60)
    print(f"loading recent 60d of momentum features (>= {cut.date()}) ...")
    recent = pd.read_parquet(MOM_FEATURES, filters=[("timestamp", ">=", cut.to_pydatetime())])
    if "timestamp" not in recent.columns:
        recent = recent.reset_index()
    recent["timestamp"] = pd.to_datetime(recent["timestamp"], utc=True)
    recent = recent.set_index(["timestamp", "ticker"]).sort_index()
    print(f"recent rows {len(recent)}  bars {recent.index.get_level_values('timestamp').nunique()}")

    print("=== momentum_expansion ===")
    run_ticker_model("momentum_expansion",
                     REPO / "momentum_expansion/data/training_import/bundle/expansion_xgb.json",
                     mom_feat, recent, 25,
                     REPO / "momentum_expansion/plots/output/oof_eval", "Momentum expansion")
    print("=== multi_ticker_swing_htf ===")
    run_ticker_model("multi_ticker_swing_htf",
                     REPO / "multi_ticker_swing_htf/data/bundle/htf_swing_xgb.json",
                     htf_feat, recent, 38,
                     REPO / "multi_ticker_swing_htf/plots/oof_eval", "HTF swing")
    print("=== theme_expansion ===")
    run_theme(20)


if __name__ == "__main__":
    main()
