"""
OOF (out-of-fold) model evaluation + leakage-free backtest plots.

The colab training bundles each ship an ``oof_preds.parquet`` containing the
walk-forward / out-of-fold model ``score``, the regression target ``y``, and
(for the swing/momentum models) the realized forward returns for each row.
Because these predictions are out-of-fold, selecting on ``score`` and reading
the realized forward return is a leakage-free estimate of live performance --
no re-inference or retraining required.

This script produces, per model, into its project plots folder:
  - decile_lift.png        mean realized outcome by score decile
  - ic_over_time.png       rolling cross-sectional rank IC (Spearman) per period
  - selection_backtest.png cumulative top-N selection vs universe vs bottom-N
  - calibration.png        binned predicted score vs realized target
  - score_hist.png         score distribution
  - feature_importance.png top-20 features
  - summary.json           headline metrics

Run:  python meta_context/oof_model_eval.py
(Generates all three models; edit MODELS below to subset.)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[1]


@dataclass
class ModelSpec:
    name: str
    bundle_dir: Path
    out_dir: Path
    # how to recover the period (cross-section) key
    period_from_index: bool          # True -> index level 0 is the period
    period_col: str | None           # column name when not from index
    # realized-return columns to show in the decile table (besides y)
    realized_cols: list[str] = field(default_factory=list)
    # column used as the per-period "return" for the selection backtest
    ret_col: str = "y"
    top_n: int = 5                   # names selected per period (long-only leg)
    label: str = ""


MODELS = [
    ModelSpec(
        name="theme_expansion",
        bundle_dir=REPO / "theme_expansion/models/bundle",
        out_dir=REPO / "theme_expansion/outputs/plots/oof_eval",
        period_from_index=False,
        period_col="date",
        realized_cols=[],
        ret_col="y",
        top_n=5,
        label="Theme excess vs benchmark, 20d fwd",
    ),
    ModelSpec(
        name="multi_ticker_swing_htf",
        bundle_dir=REPO / "multi_ticker_swing_htf/data/bundle",
        out_dir=REPO / "multi_ticker_swing_htf/plots/oof_eval",
        period_from_index=True,
        period_col=None,
        realized_cols=["fwd_best_high_return", "fwd_worst_low_return", "fwd_close_return"],
        ret_col="fwd_close_return",
        top_n=10,
        label="HTF swing score (4H)",
    ),
    ModelSpec(
        name="momentum_expansion",
        bundle_dir=REPO / "momentum_expansion/data/training_import/bundle",
        out_dir=REPO / "momentum_expansion/plots/output/oof_eval",
        period_from_index=True,
        period_col=None,
        realized_cols=["fwd_max_return", "fwd_close_return", "fwd_max_drawdown", "fwd_max_alpha"],
        ret_col="fwd_close_return",
        top_n=5,
        label="Expansion survival score (4H)",
    ),
]


def _load(spec: ModelSpec) -> pd.DataFrame:
    df = pd.read_parquet(spec.bundle_dir / "oof_preds.parquet")
    if spec.period_from_index:
        df = df.reset_index()
        df = df.rename(columns={df.columns[0]: "period"})
    else:
        df = df.rename(columns={spec.period_col: "period"})
    df["period"] = pd.to_datetime(df["period"], utc=False, errors="coerce")
    df = df.dropna(subset=["score", "y", "period"])
    return df


def _decile_lift(df: pd.DataFrame, spec: ModelSpec, ax) -> pd.DataFrame:
    d = df.copy()
    d["decile"] = pd.qcut(d["score"].rank(method="first"), 10, labels=False) + 1
    cols = ["y"] + [c for c in spec.realized_cols if c in d.columns]
    table = d.groupby("decile")[cols].mean()
    x = table.index.values
    ax.bar(x, table["y"].values, color="#4C72B0", alpha=0.85)
    ax.axhline(0, color="k", lw=0.7)
    ax.set_xlabel("score decile (10 = highest score)")
    ax.set_ylabel(f"mean realized target (y)\n{spec.label}")
    ax.set_title(f"{spec.name}: decile lift")
    ax.set_xticks(x)
    return table


def _ic_over_time(df: pd.DataFrame, ax) -> dict:
    def _ic(g):
        if g["score"].nunique() < 3 or len(g) < 5:
            return np.nan
        return spearmanr(g["score"], g["y"]).correlation

    ic = df.groupby("period")[["score", "y"]].apply(_ic).dropna()
    if ic.empty:
        return {}
    roll = ic.rolling(20, min_periods=5).mean()
    ax.plot(ic.index, ic.values, color="#bbbbbb", lw=0.6, label="per-period IC")
    ax.plot(roll.index, roll.values, color="#C44E52", lw=1.6, label="20-period rolling mean")
    ax.axhline(ic.mean(), color="#2E7D32", ls="--", lw=1.0, label=f"mean={ic.mean():.3f}")
    ax.axhline(0, color="k", lw=0.7)
    ax.set_ylabel("cross-sectional rank IC (Spearman)")
    ax.set_title("Information coefficient over time")
    ax.legend(fontsize=7, loc="upper left")
    n = len(ic)
    t_stat = ic.mean() / (ic.std(ddof=1) / np.sqrt(n)) if ic.std(ddof=1) > 0 else np.nan
    return {
        "ic_mean": float(ic.mean()),
        "ic_std": float(ic.std(ddof=1)),
        "ic_t_stat": float(t_stat),
        "ic_hit_rate_pos": float((ic > 0).mean()),
        "n_periods": int(n),
    }


def _selection_backtest(df: pd.DataFrame, spec: ModelSpec, ax) -> dict:
    """Per period: long top-N by score, universe mean, bottom-N. Cumulate the
    realized ret_col additively (overlapping fwd windows -> illustrative, not
    a compounding tradeable curve)."""
    ret = spec.ret_col if spec.ret_col in df.columns else "y"
    rows_top, rows_bot, rows_uni = [], [], []
    for period, g in df.groupby("period"):
        g = g.sort_values("score", ascending=False)
        n = min(spec.top_n, len(g))
        if n == 0:
            continue
        rows_top.append((period, g[ret].head(n).mean()))
        rows_bot.append((period, g[ret].tail(n).mean()))
        rows_uni.append((period, g[ret].mean()))
    top = pd.Series(dict(rows_top)).sort_index()
    bot = pd.Series(dict(rows_bot)).sort_index()
    uni = pd.Series(dict(rows_uni)).sort_index()
    ax.plot(top.index, top.cumsum(), color="#2E7D32", lw=1.6, label=f"top-{spec.top_n} by score")
    ax.plot(uni.index, uni.cumsum(), color="#888888", lw=1.2, label="universe mean")
    ax.plot(bot.index, bot.cumsum(), color="#C44E52", lw=1.2, label=f"bottom-{spec.top_n} by score")
    ax.axhline(0, color="k", lw=0.7)
    ax.set_ylabel(f"cumulative realized {ret}\n(additive, overlapping windows)")
    ax.set_title(f"Selection backtest (top-{spec.top_n} per period)")
    ax.legend(fontsize=7, loc="upper left")
    return {
        "ret_col": ret,
        "top_mean_per_period": float(top.mean()),
        "universe_mean_per_period": float(uni.mean()),
        "bottom_mean_per_period": float(bot.mean()),
        "top_minus_universe": float(top.mean() - uni.mean()),
        "top_minus_bottom": float(top.mean() - bot.mean()),
        "top_win_rate_vs_0": float((top > 0).mean()),
    }


def _calibration(df: pd.DataFrame, ax):
    d = df.copy()
    d["bin"] = pd.qcut(d["score"].rank(method="first"), 20, labels=False)
    g = d.groupby("bin").agg(score=("score", "mean"), y=("y", "mean"))
    ax.scatter(g["score"], g["y"], s=18, color="#4C72B0")
    lo = min(g["score"].min(), g["y"].min())
    hi = max(g["score"].max(), g["y"].max())
    ax.plot([lo, hi], [lo, hi], color="k", ls="--", lw=0.8, label="ideal (y=score)")
    ax.set_xlabel("mean predicted score (ventile)")
    ax.set_ylabel("mean realized target")
    ax.set_title("Calibration")
    ax.legend(fontsize=7)


def _score_hist(df: pd.DataFrame, ax):
    ax.hist(df["score"].values, bins=80, color="#4C72B0", alpha=0.85)
    ax.set_xlabel("score")
    ax.set_ylabel("count")
    ax.set_title("Score distribution")


def _feature_importance(spec: ModelSpec, ax):
    fp = spec.bundle_dir / "feature_importance.csv"
    if not fp.exists():
        ax.set_visible(False)
        return
    fi = pd.read_csv(fp)
    num = fi.select_dtypes("number")
    if num.empty:
        ax.set_visible(False)
        return
    val = num.columns[0]
    feat = [c for c in fi.columns if c not in num.columns]
    feat = feat[0] if feat else fi.columns[0]
    top = fi.sort_values(val, ascending=False).head(20).iloc[::-1]
    ax.barh(top[feat].astype(str), top[val].values, color="#55A868")
    ax.set_title(f"Top-20 feature importance ({val})")
    ax.tick_params(axis="y", labelsize=7)


def run(spec: ModelSpec) -> dict:
    spec.out_dir.mkdir(parents=True, exist_ok=True)
    df = _load(spec)
    summary: dict = {
        "name": spec.name,
        "n_rows": int(len(df)),
        "n_periods": int(df["period"].nunique()),
        "date_min": str(df["period"].min()),
        "date_max": str(df["period"].max()),
        "overall_spearman": float(spearmanr(df["score"], df["y"]).correlation),
    }

    fig, ax = plt.subplots(figsize=(7, 4.2))
    table = _decile_lift(df, spec, ax)
    fig.tight_layout(); fig.savefig(spec.out_dir / "decile_lift.png", dpi=120); plt.close(fig)
    table.to_csv(spec.out_dir / "decile_table.csv")
    top_d = float(table["y"].iloc[-1]); bot_d = float(table["y"].iloc[0])
    summary["decile10_mean_y"] = top_d
    summary["decile1_mean_y"] = bot_d
    summary["decile_spread"] = top_d - bot_d
    summary["decile_monotonic_spearman"] = float(
        spearmanr(table.index, table["y"]).correlation
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    summary.update(_ic_over_time(df, ax))
    fig.tight_layout(); fig.savefig(spec.out_dir / "ic_over_time.png", dpi=120); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    summary.update(_selection_backtest(df, spec, ax))
    fig.tight_layout(); fig.savefig(spec.out_dir / "selection_backtest.png", dpi=120); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4.5))
    _calibration(df, ax)
    fig.tight_layout(); fig.savefig(spec.out_dir / "calibration.png", dpi=120); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    _score_hist(df, ax)
    fig.tight_layout(); fig.savefig(spec.out_dir / "score_hist.png", dpi=120); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    _feature_importance(spec, ax)
    fig.tight_layout(); fig.savefig(spec.out_dir / "feature_importance.png", dpi=120); plt.close(fig)

    with open(spec.out_dir / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    return summary


def main():
    allsum = {}
    for spec in MODELS:
        print(f"=== {spec.name} ===")
        try:
            s = run(spec)
            allsum[spec.name] = s
            for k, v in s.items():
                print(f"  {k}: {v}")
        except Exception as exc:  # keep going across models
            import traceback
            print(f"  FAILED: {exc}")
            traceback.print_exc()
    with open(REPO / "meta_context/oof_eval_summary.json", "w") as fh:
        json.dump(allsum, fh, indent=2)
    print("\nWrote combined summary -> meta_context/oof_eval_summary.json")


if __name__ == "__main__":
    main()
