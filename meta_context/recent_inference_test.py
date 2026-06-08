"""
Recent-week inference test: run the ACTUAL exported models on the most recent
week of real data, pick what they would have flagged ~a week ago, and show how
those picks have performed since (real 4H prices / theme index).

This differs from the OOF backtest: it uses the shipped booster (not per-fold
models) on out-of-sample recent bars, so it doubles as a sanity check that the
artifact works live -- especially the re-exported HTF model.

Caveat: signals this recent have NOT completed their forward label window, so we
report realized-SO-FAR return (entry -> latest bar) plus max-favorable /
max-adverse excursion since entry, benchmarked against SPY/QQQ.

Outputs: <project>/.../oof_eval/recent_inference.png + recent_leaderboard.csv
Run:     python meta_context/recent_inference_test.py
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

ENTRY_LOOKBACK_DAYS = 7
CONTEXT_DAYS = 30
TOP_N = 6


def _load_booster(model_path: Path, feats: list[str]):
    bst = xgb.Booster()
    bst.load_model(str(model_path))
    return bst


def _bars(ticker: str) -> pd.DataFrame | None:
    p = BARS_4H / f"{ticker}.parquet"
    if not p.exists():
        return None
    b = pd.read_parquet(p)
    b["timestamp"] = pd.to_datetime(b["timestamp"], utc=True)
    return b.set_index("timestamp").sort_index()


def _excursions(ticker: str, entry_ts, latest_ts):
    b = _bars(ticker)
    if b is None or entry_ts not in b.index:
        return None
    seg = b.loc[entry_ts:latest_ts]
    if len(seg) < 2:
        return None
    e = float(seg["close"].iloc[0])
    return {
        "entry_px": e,
        "ret_since": float(seg["close"].iloc[-1] / e - 1),
        "max_fav": float(seg["high"].max() / e - 1),
        "max_adv": float(seg["low"].min() / e - 1),
        "seg": seg,
    }


def _bench_path(ticker: str, bar_index) -> pd.Series | None:
    b = _bars(ticker)
    if b is None:
        return None
    s = b["close"].reindex(bar_index, method="ffill")
    return s / s.iloc[0] - 1


def run_ticker_model(name, model_path, feats, recent, out_dir, label):
    bars = recent.index.get_level_values("timestamp").unique().sort_values()
    latest = bars[-1]
    entry = bars[bars >= latest - pd.Timedelta(days=ENTRY_LOOKBACK_DAYS)][0]
    bst = _load_booster(model_path, feats)

    cs = recent.xs(entry, level="timestamp")
    X = cs[feats]
    cs = cs.assign(score=bst.predict(xgb.DMatrix(X, missing=np.nan, feature_names=feats)))
    ranked = cs.sort_values("score", ascending=False)

    # realized-so-far for the top picks
    rows = []
    for ticker in ranked.index:
        ex = _excursions(ticker, entry, latest)
        if ex is None:
            continue
        rows.append({"ticker": ticker, "score": float(ranked.loc[ticker, "score"]), **{k: ex[k] for k in ("ret_since", "max_fav", "max_adv")}})
        if len(rows) >= TOP_N:
            break
    lb = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    lb.to_csv(out_dir / "recent_leaderboard.csv", index=False)

    # ---- figure: basket vs benchmarks + per-pick price panels
    bar_index = bars[bars >= entry]
    fig = plt.figure(figsize=(13, 8.5))
    gs = fig.add_gridspec(3, 3)

    ax0 = fig.add_subplot(gs[0, :])
    pick_paths = []
    for t in lb["ticker"]:
        ex = _excursions(t, entry, latest)
        if ex is None:
            continue
        s = ex["seg"]["close"].reindex(bar_index, method="ffill")
        pick_paths.append(s / s.iloc[0] - 1)
    if pick_paths:
        basket = pd.concat(pick_paths, axis=1).mean(axis=1)
        ax0.plot(basket.index, basket.values * 100, color="#2E7D32", lw=2.0, label=f"top-{TOP_N} basket")
    for bench, col in [("SPY", "#1565C0"), ("QQQ", "#9C27B0")]:
        bp = _bench_path(bench, bar_index)
        if bp is not None:
            ax0.plot(bp.index, bp.values * 100, lw=1.2, color=col, ls="--", label=bench)
    ax0.axhline(0, color="k", lw=0.7)
    ax0.set_ylabel("% return since entry")
    ax0.set_title(f"{name}: top-{TOP_N} picks @ {pd.Timestamp(entry).date()} vs benchmarks "
                  f"(through {pd.Timestamp(latest).date()})")
    ax0.legend(fontsize=7, loc="upper left")

    for i, t in enumerate(lb["ticker"][:6]):
        ax = fig.add_subplot(gs[1 + i // 3, i % 3])
        b = _bars(t)
        i_e = b.index.get_loc(entry)
        i_l = b.index.searchsorted(latest, side="right") - 1  # last bar on/before latest
        win = b.iloc[max(0, i_e - 20): i_l + 1]
        xi = list(win.index).index(entry)
        x = np.arange(len(win))
        ax.plot(x, win["close"].values, color="#333333", lw=1.0)
        ax.axvline(xi, color="#1565C0", lw=1.2)
        ax.axvspan(xi, len(win) - 1, color="#1565C0", alpha=0.06)
        r = lb.iloc[i]
        ax.set_title(f"{t}  s={r['score']:.2f}  since {r['ret_since']:+.1%} "
                     f"(↑{r['max_fav']:+.0%}/↓{r['max_adv']:+.0%})", fontsize=7.5)
        ax.set_xticks([]); ax.tick_params(labelsize=6)

    fig.suptitle(f"{label} — recent-week inference test (exported model on real 4H data)", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_dir / "recent_inference.png", dpi=120)
    plt.close(fig)
    print(f"  {name}: entry {pd.Timestamp(entry).date()} latest {pd.Timestamp(latest).date()} | "
          f"top-{TOP_N} avg since-entry {lb['ret_since'].mean():+.2%} | wrote {out_dir/'recent_inference.png'}")
    return lb


def run_theme():
    feats = json.load(open(REPO / "theme_expansion/models/bundle/eval_metrics.json"))["feature_columns"]
    lab = pd.read_parquet(REPO / "theme_expansion/outputs/theme_signal_labels.parquet")
    lab["date"] = pd.to_datetime(lab["date"])
    latest = lab["date"].max()
    entry = lab.loc[lab["date"] >= latest - pd.Timedelta(days=ENTRY_LOOKBACK_DAYS), "date"].min()
    cs = lab[lab["date"] == entry].copy()

    dummies = pd.get_dummies(cs[["category", "benchmark", "signal_market_playbook"]],
                             prefix=["category", "benchmark", "signal_market_playbook"])
    num = [f for f in feats if f in cs.columns]
    X = pd.concat([cs[num].reset_index(drop=True), dummies.reset_index(drop=True)], axis=1)
    X = X.reindex(columns=feats, fill_value=0).astype(float)
    bst = xgb.Booster(); bst.load_model(str(REPO / "theme_expansion/models/bundle/theme_xgb.json"))
    cs = cs.assign(score=bst.predict(xgb.DMatrix(X, missing=np.nan, feature_names=feats)))
    ranked = cs.sort_values("score", ascending=False)

    daily = pd.read_parquet(REPO / "theme_expansion/outputs/theme_daily.parquet")
    daily["date"] = pd.to_datetime(daily["date"]); daily = daily.sort_values(["theme", "date"])
    daily["cum"] = daily.groupby("theme")["theme_return_1d"].transform(lambda s: (1 + s.fillna(0)).cumprod())
    idx = {t: g.set_index("date")["cum"] for t, g in daily.groupby("theme")}

    out_dir = REPO / "theme_expansion/outputs/plots/oof_eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, picks = [], []
    for theme in ranked["theme"]:
        if theme not in idx:
            continue
        s = idx[theme]
        if entry not in s.index or latest not in s.index:
            # use nearest available within range
            seg = s.loc[entry:latest]
            if len(seg) < 2:
                continue
        else:
            seg = s.loc[entry:latest]
        ret = float(seg.iloc[-1] / seg.iloc[0] - 1)
        rows.append({"theme": theme, "score": float(ranked.set_index("theme").loc[theme, "score"]), "ret_since": ret})
        picks.append((theme, seg))
        if len(rows) >= TOP_N:
            break
    lb = pd.DataFrame(rows)
    lb.to_csv(out_dir / "recent_leaderboard.csv", index=False)

    fig = plt.figure(figsize=(13, 8.5))
    gs = fig.add_gridspec(3, 3)
    ax0 = fig.add_subplot(gs[0, :])
    for theme, seg in picks:
        ax0.plot(seg.index, (seg / seg.iloc[0] - 1).values * 100, lw=1.0, alpha=0.7, label=theme)
    # SPY benchmark from daily_bars
    db = pd.read_parquet(REPO / "theme_expansion/outputs/daily_bars.parquet", columns=["date", "ticker", "close"])
    spy = db[db["ticker"] == "SPY"].set_index(pd.to_datetime(db[db["ticker"] == "SPY"]["date"]))["close"].sort_index()
    spy = spy.loc[entry:latest]
    if len(spy) > 1:
        ax0.plot(spy.index, (spy / spy.iloc[0] - 1).values * 100, color="k", ls="--", lw=1.4, label="SPY")
    ax0.axhline(0, color="k", lw=0.7)
    ax0.set_ylabel("% return since entry")
    ax0.set_title(f"theme_expansion: top-{TOP_N} theme picks @ {pd.Timestamp(entry).date()} "
                  f"(through {pd.Timestamp(latest).date()})")
    ax0.legend(fontsize=6, loc="upper left", ncol=2)
    for i, (theme, seg) in enumerate(picks[:6]):
        ax = fig.add_subplot(gs[1 + i // 3, i % 3])
        ax.plot(np.arange(len(seg)), (seg / seg.iloc[0]).values, color="#333333", lw=1.0)
        ax.axhline(1.0, color="#888", ls=":", lw=0.8)
        ax.set_title(f"{theme}  s={lb.iloc[i]['score']:.3f}  since {lb.iloc[i]['ret_since']:+.1%}", fontsize=7.5)
        ax.set_xticks([]); ax.tick_params(labelsize=6)
    fig.suptitle("theme_expansion — recent-week inference test (exported model)", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_dir / "recent_inference.png", dpi=120); plt.close(fig)
    print(f"  theme: entry {pd.Timestamp(entry).date()} latest {pd.Timestamp(latest).date()} | "
          f"top-{TOP_N} avg since-entry {lb['ret_since'].mean():+.2%} | wrote {out_dir/'recent_inference.png'}")
    return lb


def main():
    htf_feat = json.load(open(REPO / "multi_ticker_swing_htf/data/bundle/eval_metrics.json"))["feature_columns"]
    mom_feat = json.load(open(REPO / "momentum_expansion/data/training_import/bundle/eval_metrics.json"))["feature_columns"]
    union = sorted(set(htf_feat) | set(mom_feat) | {"timestamp", "ticker"})

    import pyarrow.parquet as pq
    tcol = pd.to_datetime(pq.read_table(MOM_FEATURES, columns=["timestamp"]).column("timestamp").to_pandas(), utc=True)
    cut = tcol.max() - pd.Timedelta(days=CONTEXT_DAYS)
    print(f"loading recent {CONTEXT_DAYS}d of momentum features (>= {cut.date()}) ...")
    recent = pd.read_parquet(MOM_FEATURES, filters=[("timestamp", ">=", cut.to_pydatetime())])
    if "timestamp" not in recent.columns:
        recent = recent.reset_index()
    recent["timestamp"] = pd.to_datetime(recent["timestamp"], utc=True)
    recent = recent.set_index(["timestamp", "ticker"]).sort_index()
    print(f"recent rows: {len(recent)}  bars: {recent.index.get_level_values('timestamp').nunique()}")

    print("=== momentum_expansion ===")
    run_ticker_model("momentum_expansion",
                     REPO / "momentum_expansion/data/training_import/bundle/expansion_xgb.json",
                     mom_feat, recent,
                     REPO / "momentum_expansion/plots/output/oof_eval", "Momentum expansion")
    print("=== multi_ticker_swing_htf ===")
    run_ticker_model("multi_ticker_swing_htf",
                     REPO / "multi_ticker_swing_htf/data/bundle/htf_swing_xgb.json",
                     htf_feat, recent,
                     REPO / "multi_ticker_swing_htf/plots/oof_eval", "HTF swing")
    print("=== theme_expansion ===")
    run_theme()


if __name__ == "__main__":
    main()
