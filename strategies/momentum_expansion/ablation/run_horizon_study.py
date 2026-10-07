"""Which holding horizon should the 4H models target? (training-free)

Question (2026-10-05): the Stage-2 plan proposed 10 and 20 trading days. Is that
right, or should horizon be searched like a hyper-parameter on Colab?

It can be read off the data without training. For each horizon h (1..60
sessions) the vol-matched executable target L2(h) is built for every row, and
three fixed, a-priori signal families are scored against it:

  trend      long-term position: above 52w low / 200DMA, EMA stack, 6-12m return
  reversal   short-term weakness: MINUS 1-5 day return and RSI ("pivot / swing low")
  pullback   both at once: a leader that is short-term weak
  score      the deployed momentum model (reference)

Per horizon: mean per-bar rank IC; an annualised information ratio from
NON-overlapping bars (every h-th session, averaged over all h phase offsets);
the top-quintile vol-matched excess per trade (what a round trip has to pay
for); and the rank overlap between targets at different horizons (two models are
only worth keeping if their targets differ).

Signs are fixed before looking (trend +, weakness +). Nothing is fit, so no
horizon is "tuned"; Stage 2 then trains only the horizons this picks.

Panel: the momentum decision panel (last 4H bar per session, 2022-11..2026-05,
1,080 survivor-shaped names). Entry next open, exit close h sessions later,
SPY-excess, corporate-action guarded.

Usage:
    .venv/bin/python -m strategies.momentum_expansion.ablation.run_horizon_study
"""
from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from strategies.momentum_expansion.ablation.label_arms import vol_bucket
from strategies.momentum_expansion.ablation.run_feature_screen import (
    DAILY_DIR,
    MIN_NAMES_PER_BAR,
    RESULTS_DIR,
    _gauss_rank,
    build_panel,
    exec_forward_returns,
)

logger = logging.getLogger(__name__)

HORIZONS = (1, 2, 3, 5, 10, 15, 20, 30, 40, 60)
# (column, sign): sign +1 means "higher value = more of the family's property".
FAMILIES: dict[str, list[tuple[str, int]]] = {
    "trend": [("dist_to_52w_low_atr", 1), ("daily_dist_200dma_atr", 1), ("daily_ema_stack", 1),
              ("d_ret_126", 1), ("d_ret_252_21", 1)],
    "reversal": [("ret_1", -1), ("ret_3", -1), ("ret_5", -1), ("daily_ret_5", -1), ("rsi_14", -1)],
}
SIGNALS = ["trend", "reversal", "pullback", "score"]


def _fwd(ticker: str) -> pd.DataFrame | None:
    path = DAILY_DIR / f"{ticker}.parquet"
    if not path.exists():
        return None
    d = pd.read_parquet(path, columns=["timestamp", "open", "close", "volume"])
    d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True)
    d = d.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    if len(d) < 60:
        return None
    out = pd.DataFrame({f"fret_{h}": r for h, r in exec_forward_returns(d, HORIZONS).items()})
    out["date"] = d["timestamp"].dt.tz_convert("America/New_York").dt.normalize().dt.tz_localize(None)
    out["ticker"] = ticker
    return out


def build(model: str = "momentum") -> pd.DataFrame:
    cols = sorted({c for fam in FAMILIES.values() for c, _ in fam} | {"daily_atr_pct", "adx_14", "score"})
    panel = build_panel(model)[["timestamp", "ticker", "date", *cols]]
    tickers = sorted(panel["ticker"].unique())
    with ProcessPoolExecutor(max_workers=5) as ex:
        fwd = pd.concat([b for b in ex.map(_fwd, tickers, chunksize=32) if b is not None], ignore_index=True)
    spy = _fwd("SPY").drop(columns="ticker").rename(columns={f"fret_{h}": f"spy_{h}" for h in HORIZONS})
    panel = panel.merge(fwd, on=["ticker", "date"], how="left").merge(spy, on="date", how="left")
    dec = vol_bucket(panel, vol_col="daily_atr_pct", n=10)
    for h in HORIZONS:
        x = panel[f"fret_{h}"] - panel[f"spy_{h}"]
        panel[f"L2_{h}"] = x - x.groupby([panel["timestamp"], dec]).transform("mean")
    return panel.drop(columns=[c for c in panel.columns if c.startswith(("fret_", "spy_"))])


def _signals(g: pd.DataFrame) -> np.ndarray:
    """[n, len(SIGNALS)] Gaussian-rank composites for one bar."""
    fam = {}
    for name, members in FAMILIES.items():
        z = _gauss_rank(g[[c for c, _ in members]].to_numpy(np.float64)) * np.array([s for _, s in members])
        fam[name] = np.nanmean(z, axis=1)
    fam["pullback"] = fam["trend"] + fam["reversal"]
    fam["score"] = g["score"].to_numpy(np.float64)
    return _gauss_rank(np.column_stack([fam[s] for s in SIGNALS]))


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    ok = ~(np.isnan(a) | np.isnan(b))
    if ok.sum() < 30:
        return np.nan
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def per_bar(panel: pd.DataFrame, *, split: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(ic[bar x (signal,h[,bucket])], top-quintile L2 per bar, label-overlap per bar)."""
    lab = [f"L2_{h}" for h in HORIZONS]
    ic_rows, top_rows, ov_rows, idx = [], [], [], []
    for ts, g in panel.groupby("timestamp", sort=True):
        if len(g) < MIN_NAMES_PER_BAR:
            continue
        S = _signals(g)
        Y = g[lab].to_numpy(np.float64)
        Yr = _gauss_rank(Y)
        rec, top = {}, {}
        buckets = {"all": np.ones(len(g), dtype=bool)}
        if split:
            q = pd.qcut(g[split].rank(method="first"), 3, labels=["low", "mid", "high"]).to_numpy()
            buckets.update({f"{split}_{k}": q == k for k in ("low", "high")})
        for bname, m in buckets.items():
            for i, sig in enumerate(SIGNALS):
                for j, h in enumerate(HORIZONS):
                    rec[(sig, h, bname)] = _corr(S[m, i], Yr[m, j])
        for i, sig in enumerate(SIGNALS):
            s = S[:, i]
            hi = s >= np.nanquantile(s, 0.8)
            for j, h in enumerate(HORIZONS):
                top[(sig, h)] = float(np.nanmean(Y[hi, j]))
        ov_rows.append({(a, b): _corr(Yr[:, i], Yr[:, j])
                        for i, a in enumerate(HORIZONS) for j, b in enumerate(HORIZONS) if i < j})
        ic_rows.append(rec); top_rows.append(top); idx.append(ts)
    def mk(rows: list[dict]) -> pd.DataFrame:
        df = pd.DataFrame(rows, index=pd.DatetimeIndex(idx))
        df.columns = pd.MultiIndex.from_tuples(df.columns)
        return df

    return mk(ic_rows), mk(top_rows), mk(ov_rows)


def annualised_ir(ic: pd.Series, h: int) -> float:
    """IR of one bet per h sessions, from non-overlapping bars, mean over phase offsets."""
    x = ic.to_numpy()
    irs = []
    for o in range(h):
        s = x[o::h]
        s = s[~np.isnan(s)]
        if len(s) >= 8 and s.std(ddof=1) > 0:
            irs.append(s.mean() / s.std(ddof=1) * np.sqrt(252.0 / h))
    return float(np.mean(irs)) if irs else np.nan


def summarize(ic: pd.DataFrame, top: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sig in SIGNALS:
        for h in HORIZONS:
            s = ic[(sig, h, "all")]
            rows.append({"signal": sig, "h": h, "ic": s.mean(), "ir_ann": annualised_ir(s, h),
                         "pct_bars_pos": (s > 0).mean(),
                         "top_q_excess_pct": 100 * top[(sig, h)].mean(),
                         "top_q_excess_per_day_bp": 1e4 * top[(sig, h)].mean() / h})
    return pd.DataFrame(rows)


def run() -> dict[str, pd.DataFrame]:
    out_dir = RESULTS_DIR / "horizon"
    out_dir.mkdir(parents=True, exist_ok=True)
    panel = build()
    logger.info("panel %d rows, %d bars", len(panel), panel["timestamp"].nunique())
    ic, top, ov = per_bar(panel, split="adx_14")
    summ = summarize(ic, top)
    overlap = ov.mean().unstack().round(2)
    split = ic.mean().rename("ic").rename_axis(["signal", "h", "bucket"]).reset_index()
    summ.to_csv(out_dir / "horizon_summary.csv", index=False)
    overlap.to_csv(out_dir / "label_overlap.csv")
    split.to_csv(out_dir / "ic_by_adx.csv", index=False)
    ic.columns = [f"{a}|{b}|{c}" for a, b, c in ic.columns]
    ic.to_parquet(out_dir / "ic_per_bar.parquet")
    return {"summary": summ, "overlap": overlap, "split": split}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    res = run()
    pd.set_option("display.width", 200)
    s = res["summary"]
    for col in ("ic", "ir_ann", "top_q_excess_pct", "top_q_excess_per_day_bp"):
        print(f"\n== {col} ==")
        print(s.pivot(index="signal", columns="h", values=col).reindex(SIGNALS).round(3).to_string())
    print("\n== rank overlap between targets L2(h1) vs L2(h2) ==")
    print(res["overlap"].to_string())
    sp = res["split"]
    print("\n== IC by ADX tercile (low = range-bound, high = trending) ==")
    print(sp[sp.h.isin([3, 5, 10, 20, 40])].pivot_table(index=["signal", "bucket"], columns="h", values="ic")
          .reindex(SIGNALS, level=0).round(3).to_string())


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# Causal swing-pivot event study (what the HTF "pivot detector" is meant to find)
# ---------------------------------------------------------------------------
# HTF's label marks a pivot with 3 bars on each side, i.e. it needs 3 FUTURE 4H
# bars. The tradable version is the same pivot at the first bar it is knowable:
# 3 bars after the low/high printed. This asks what the forward vol-matched
# return is from that confirmation bar -- no future selection.

BARS_4H_DIR = DAILY_DIR.parent / "4h"
PIVOT_SIDE = 3  # bars each side, as PIVOT_LABEL_CONFIG


def _pivots(ticker: str) -> pd.DataFrame | None:
    path = BARS_4H_DIR / f"{ticker}.parquet"
    if not path.exists():
        return None
    b = pd.read_parquet(path, columns=["timestamp", "high", "low", "close"])
    b["timestamp"] = pd.to_datetime(b["timestamp"], utc=True)
    b = b.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    if len(b) < 60:
        return None
    w = 2 * PIVOT_SIDE + 1
    centre_low, centre_high = b["low"].shift(PIVOT_SIDE), b["high"].shift(PIVOT_SIDE)
    conf_low = centre_low.eq(b["low"].rolling(w).min())      # known at this bar's close
    conf_high = centre_high.eq(b["high"].rolling(w).max())
    # "confirmed today": at this bar or the one before it (two 4H bars per session)
    out = pd.DataFrame({
        "timestamp": b["timestamp"],
        "pivot_low": (conf_low | conf_low.shift(1, fill_value=False)).astype(np.int8),
        "pivot_high": (conf_high | conf_high.shift(1, fill_value=False)).astype(np.int8),
        # how far price has already bounced off the low by the time it is knowable
        "bounce_pct": np.where(conf_low, b["close"] / centre_low - 1.0, np.nan),
    })
    out["ticker"] = ticker
    return out


def pivot_event_study(n_boot: int = 1000) -> pd.DataFrame:
    from strategies.momentum_expansion.ablation.bootstrap import week_block_bootstrap_ci
    from strategies.momentum_expansion.ablation.run_subset_eval import _biweek

    cols = ["daily_atr_pct", "adx_14", "daily_dist_200dma_atr", "score"]
    panel = build()
    extra = build_panel("momentum")[["timestamp", "ticker", "daily_dist_200dma_atr"]]
    panel = panel.merge(extra, on=["timestamp", "ticker"], how="left", suffixes=("", "_x"))
    tickers = sorted(panel["ticker"].unique())
    with ProcessPoolExecutor(max_workers=5) as ex:
        piv = pd.concat([p for p in ex.map(_pivots, tickers, chunksize=32) if p is not None], ignore_index=True)
    panel = panel.merge(piv, on=["timestamp", "ticker"], how="left")
    adx_hi = panel.groupby("timestamp")["adx_14"].rank(pct=True)
    subsets = {
        "all": pd.Series(True, index=panel.index),
        "range-bound (ADX bottom third)": adx_hi <= 1 / 3,
        "trending (ADX top third)": adx_hi >= 2 / 3,
        "above 200DMA": panel["daily_dist_200dma_atr"] > 0,
        "below 200DMA": panel["daily_dist_200dma_atr"] <= 0,
    }
    rows = []
    for event in ("pivot_low", "pivot_high"):
        for sname, mask in subsets.items():
            sub = panel[mask.fillna(False)]
            for h in (1, 2, 3, 5, 10, 20):
                y = f"L2_{h}"
                g = sub.dropna(subset=[y, event]).groupby("timestamp")
                diff = g.apply(lambda x: x.loc[x[event] == 1, y].mean() - x.loc[x[event] == 0, y].mean()
                               if (x[event] == 1).sum() >= 5 else np.nan, include_groups=False).dropna()
                b = week_block_bootstrap_ci(diff.reset_index(drop=True),
                                            _biweek(pd.Series(diff.index)).reset_index(drop=True),
                                            n_boot=n_boot, alpha=0.05)
                se = (b["ci_hi"] - b["ci_lo"]) / 3.92
                rows.append({"event": event, "subset": sname, "h": h, "excess_pct": 100 * b["point"],
                             "ci_lo": 100 * b["ci_lo"], "ci_hi": 100 * b["ci_hi"], "mde_pct": 100 * 2.8 * se,
                             "bars": len(diff), "flag_rate": float((sub[event] == 1).mean())})
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS_DIR / "horizon" / "pivot_event_study.csv", index=False)
    bounce = panel.loc[panel["pivot_low"] == 1, "bounce_pct"].dropna()
    logger.info("pivot low: median bounce already made by confirmation %.2f%% (n=%d)", 100 * bounce.median(), len(bounce))
    return out
