"""
Standalone-skill assessment: is each model's edge real selection skill (worth
feeding a metabreaker / ensemble) or just market beta in a good regime?

Two corrections over the naive backtest:
  1. NON-OVERLAPPING windows  -- entries spaced >= forward-window apart, so each
     trade is an independent sample (no autocorrelation inflating significance).
  2. MARKET-NEUTRAL alpha     -- cross-sectionally demean each period's returns
     (subtract the universe mean), so the common market move / beta cancels and
     what's left is pure ranking skill. Also report the top-vs-bottom decile
     long-short spread (market-neutral by construction).

Source: the leakage-free OOF predictions (full 2022-2026 history, all regimes).
Theme's target is already excess-vs-benchmark, so it is already market-neutral.

Cross-model: correlation of the per-window alpha series tells us whether the
modules diversify (ensemble helps) or overlap (use the best one standalone).

Outputs: meta_context/skill_assessment/{skill.png, skill_summary.json}
Run:     python meta_context/skill_assessment.py
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "meta_context/skill_assessment"
TOP_N = 10


@dataclass
class M:
    name: str
    oof: Path
    period_from_index: bool
    period_col: str | None
    ret_col: str          # realized return column (already-excess for theme)
    already_excess: bool
    w_step: int           # window length in periods (non-overlap stride)
    periods_per_year: float


MODELS = [
    M("momentum", REPO / "momentum_expansion/data/training_import/bundle/oof_preds.parquet",
      True, None, "fwd_close_return", False, 25, 252 / 10),
    M("htf_swing", REPO / "multi_ticker_swing_htf/data/bundle/oof_preds.parquet",
      True, None, "fwd_close_return", False, 38, 252 / 15),
    M("theme", REPO / "theme_expansion/models/bundle/oof_preds.parquet",
      False, "date", "y", True, 20, 252 / 20),
]


def _load(m: M) -> pd.DataFrame:
    df = pd.read_parquet(m.oof)
    if m.period_from_index:
        df = df.reset_index().rename(columns={df.reset_index().columns[0]: "period"})
        df = df.rename(columns={df.columns[0]: "period"})
    else:
        df = df.rename(columns={m.period_col: "period"})
    df["period"] = pd.to_datetime(df["period"])
    keep = ["period", "score", m.ret_col]
    df = df[[c for c in keep if c in df.columns]].dropna()
    return df.rename(columns={m.ret_col: "ret"})


def assess(m: M):
    df = _load(m)
    periods = np.sort(df["period"].unique())
    entries = periods[:: m.w_step]            # non-overlapping
    rows = []
    for p in entries:
        g = df[df["period"] == p]
        if len(g) < 20:
            continue
        g = g.sort_values("score", ascending=False)
        r = g["ret"].to_numpy()
        mkt = 0.0 if m.already_excess else r.mean()   # demean (market-neutral)
        n = min(TOP_N, len(g) // 5)
        top_alpha = r[:n].mean() - mkt
        # decile long-short
        d = max(1, len(g) // 10)
        ls = r[:d].mean() - r[-d:].mean()
        rows.append({"period": p, "top_alpha": top_alpha, "ls_spread": ls})
    res = pd.DataFrame(rows).set_index("period")
    a = res["top_alpha"].to_numpy()
    ls = res["ls_spread"].to_numpy()
    n = len(a)

    def stats(x):
        sd = x.std(ddof=1)
        t = x.mean() / (sd / np.sqrt(len(x))) if sd > 0 else np.nan
        sharpe = (x.mean() / sd) * np.sqrt(m.periods_per_year) if sd > 0 else np.nan
        return dict(mean=float(x.mean()), t_stat=float(t), hit=float((x > 0).mean()),
                    ann_sharpe=float(sharpe), ann_return=float(x.mean() * m.periods_per_year))

    summary = {
        "name": m.name, "n_nonoverlap_windows": int(n),
        "window_len_periods": m.w_step,
        "date_range": [str(res.index.min()), str(res.index.max())],
        "top_alpha": stats(a),
        "long_short": stats(ls),
    }
    return summary, res


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    summaries, series = {}, {}
    for m in MODELS:
        s, res = assess(m)
        summaries[m.name] = s
        series[m.name] = res
        ta = s["top_alpha"]
        print(f"{m.name:10s} | windows {s['n_nonoverlap_windows']:3d} | "
              f"top-N alpha/window {ta['mean']:+.3%} (t={ta['t_stat']:.2f}, hit {ta['hit']:.0%}, "
              f"ann~{ta['ann_return']:+.1%}, Sharpe {ta['ann_sharpe']:.2f}) | "
              f"L/S spread {s['long_short']['mean']:+.3%} (t={s['long_short']['t_stat']:.2f})")

    # cross-model alpha correlation (momentum vs htf share the 4H timeline).
    # Correlation does not need non-overlap, so use EVERY period for alignment.
    def _full_alpha(m: M) -> pd.Series:
        df = _load(m)
        out = {}
        for p, g in df.groupby("period"):
            if len(g) < 20:
                continue
            g = g.sort_values("score", ascending=False)
            r = g["ret"].to_numpy()
            mkt = 0.0 if m.already_excess else r.mean()
            n = min(TOP_N, len(g) // 5)
            out[p] = r[:n].mean() - mkt
        return pd.Series(out)

    corr = None
    fm = {m.name: _full_alpha(m) for m in MODELS if m.name in ("momentum", "htf_swing")}
    j = pd.concat([fm["momentum"].rename("mom"), fm["htf_swing"].rename("htf")], axis=1).dropna()
    if len(j) >= 5:
        corr = float(j["mom"].corr(j["htf"]))
    summaries["_cross_model"] = {
        "momentum_vs_htf_alpha_corr": corr,
        "note": "theme trades baskets on a daily clock (separate universe) -> inherently diversifying",
    }
    print(f"\nmomentum vs htf alpha correlation: {corr}")

    # ---- plots
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    ax = axes[0]
    for name, res in series.items():
        ax.plot(res.index, res["top_alpha"].cumsum() * 100, lw=1.6, marker="o", ms=2, label=name)
    ax.axhline(0, color="k", lw=0.7)
    ax.set_title("Cumulative market-neutral alpha\n(non-overlapping windows)")
    ax.set_ylabel("cumulative top-N alpha (%, additive)")
    ax.legend(fontsize=8); ax.tick_params(axis="x", labelrotation=30, labelsize=6)

    ax = axes[1]
    names = [m.name for m in MODELS]
    tvals = [summaries[n]["top_alpha"]["t_stat"] for n in names]
    lsv = [summaries[n]["long_short"]["t_stat"] for n in names]
    x = np.arange(len(names)); ww = 0.38
    ax.bar(x - ww / 2, tvals, ww, label="top-N alpha t", color="#4C72B0")
    ax.bar(x + ww / 2, lsv, ww, label="L/S spread t", color="#55A868")
    for thr in (2, -2):
        ax.axhline(thr, color="#C44E52", ls="--", lw=0.8)
    ax.axhline(0, color="k", lw=0.7)
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=8)
    ax.set_title("Skill significance (t-stat)\n|t|>2 dashed"); ax.legend(fontsize=8)

    ax = axes[2]
    if len(j) >= 5:
        ax.scatter(j["mom"] * 100, j["htf"] * 100, s=14, alpha=0.6, color="#9C27B0")
        ax.axhline(0, color="k", lw=0.7); ax.axvline(0, color="k", lw=0.7)
        ax.set_xlabel("momentum alpha (%)"); ax.set_ylabel("htf alpha (%)")
        ax.set_title(f"Module alpha co-movement\ncorr={corr:.2f}")
    fig.suptitle("Standalone skill: market-neutral alpha on non-overlapping windows (OOF, 2022-2026)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT / "skill.png", dpi=120); plt.close(fig)
    (OUT / "skill_summary.json").write_text(json.dumps(summaries, indent=2))
    print(f"\nwrote {OUT/'skill.png'} and skill_summary.json")


if __name__ == "__main__":
    main()
