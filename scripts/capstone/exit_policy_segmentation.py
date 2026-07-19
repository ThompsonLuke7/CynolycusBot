"""
Phase 1+2 of the segmented entry/exit-policy study: build a data-derived
segmentation grid and re-evaluate the FIXED policy frontier (current-live /
deployed / g284 / id4) per segment instead of in aggregate.

Discipline notes (see research/capstone/exit_policy_segmentation_*.md):
  - The 4 policies are FIXED from prior rounds — nothing is re-searched here.
  - Cohort definitions use ONLY information available before/at entry:
      * tail-propensity clusters are fit on bars strictly BEFORE 2025-07-01
        (pre-VAL), so they are a static ticker attribute over the whole
        val/test window. AXTI/MU/SNDK must fall out of the clustering, they
        are not hand-picked.
      * entry-feature buckets come from meta_ranker_matrix entry-time columns.
      * regime comes from the matrix's own entry-time regime_* features.
  - Any policy-by-segment map is SELECTED on VAL and evaluated frozen on TEST
    (single read) by the companion writeup, not tuned on test.

Same val/test split as the whole thread:
  VAL  2025-07-01 -> 2026-01-15
  TEST 2026-01-15 -> 2026-05-15

Usage:
  PYTHONPATH=. .venv/bin/python scripts/capstone/build_meta_scored_from_oof.py
  PYTHONPATH=. .venv/bin/python scripts/capstone/exit_policy_segmentation.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "signals/meta_context/meta_ranker"))
import backtest_exits as be  # noqa: E402

VAL_START = pd.Timestamp("2025-07-01", tz="UTC")
VAL_END = pd.Timestamp("2026-01-15", tz="UTC")
TEST_END = pd.Timestamp("2026-05-15", tz="UTC")
TOPK = 10
MAX_RANK_KEPT = 40          # keep deeper ranks for the phase-3 admission study
MIN_PREVAL_BARS = 250       # ~6 months of 4H bars to qualify for clustering
POTENTIAL_HORIZON = 60      # bars used for the policy-independent MFE potential

OUT_DIR = REPO / "research/capstone/segmentation"

OOF_SOURCES = {
    "momentum": REPO / "strategies/momentum_expansion/models/expansion_v1/oof_preds.parquet",
    "htf": REPO / "strategies/multi_ticker_swing_htf/models/oof_preds.parquet",
}

POLICIES = {
    "current-live": dict(stop=None, trail=None, target=None, scale_frac=1.0, horizon=None, grace=0),
    "deployed": dict(stop=0.50, trail=0.35, target=0.20, scale_frac=0.5, horizon=25, grace=None),
    "g284": dict(stop=0.59, trail=None, target=0.07, scale_frac=1.0, horizon=60, grace=None),
    "id4": dict(stop=0.39, trail=None, target=0.30, scale_frac=0.16, horizon=53, grace=None),
}

_BAR_CACHE: dict[str, pd.DataFrame | None] = {}


def _bars(ticker: str) -> pd.DataFrame | None:
    if ticker not in _BAR_CACHE:
        _BAR_CACHE[ticker] = be._ticker_path(ticker, None)
    return _BAR_CACHE[ticker]


# ---------------------------------------------------------------- streams
def load_stream(module: str) -> pd.DataFrame:
    """[timestamp, ticker, score, rk] for rk <= MAX_RANK_KEPT, full history."""
    if module == "meta":
        df = pd.read_parquet(be.SCORED).dropna(subset=["s_combo"])
        score_col = "s_combo"
    else:
        df = pd.read_parquet(OOF_SOURCES[module]).reset_index().dropna(subset=["score"])
        score_col = "score"
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["rk"] = df.groupby("timestamp")[score_col].rank(ascending=False, method="first")
    df = df.rename(columns={score_col: "score"})
    return df.loc[df["rk"] <= MAX_RANK_KEPT, ["timestamp", "ticker", "score", "rk"]]


def member_from_stream(stream: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp,
                       topk: int = TOPK) -> pd.DataFrame:
    df = stream[(stream["timestamp"] >= start) & (stream["timestamp"] < end)].copy()
    df["in_top"] = df["rk"] <= topk
    return df[["timestamp", "ticker", "in_top"]]


# ---------------------------------------------------------------- trades
def all_trades(member: pd.DataFrame, **cfg) -> pd.DataFrame:
    """Identical mechanic to exit_policy_cross_module.simulate(), one row per trade,
    plus a policy-independent forward MFE 'potential' over POTENTIAL_HORIZON bars."""
    stop, trail, target, scale_frac, horizon, grace = (cfg.get(k) for k in
        ("stop", "trail", "target", "scale_frac", "horizon", "grace"))
    scale_frac = scale_frac if scale_frac is not None else 1.0
    rows = []
    for ticker, g in member.groupby("ticker"):
        g = g.sort_values("timestamp")
        bars = _bars(ticker)
        if bars is None:
            continue
        m = g.set_index("timestamp")["in_top"].reindex(bars.index).fillna(False).astype(bool).values
        close, high, low = bars["close"].values, bars["high"].values, bars["low"].values
        n = len(bars)
        i = 0
        while i < n - 1:
            if not m[i]:
                i += 1
                continue
            entry = close[i]
            if entry <= 0:
                i += 1
                continue
            peak = entry
            realized = 0.0
            remaining = 1.0
            trimmed = False
            out_ct = 0
            j = i + 1
            exit_ret = None
            while j < n and (j - i) <= be.MAX_HOLD:
                peak = max(peak, high[j])
                lo_ret = low[j] / entry - 1
                hi_ret = high[j] / entry - 1
                if stop is not None and lo_ret <= -stop:
                    exit_ret = -stop
                    break
                if trail is not None and low[j] <= peak * (1 - trail):
                    exit_ret = peak * (1 - trail) / entry - 1
                    break
                if target is not None and not trimmed and hi_ret >= target:
                    if scale_frac >= 1.0:
                        exit_ret = target
                        break
                    realized += scale_frac * target
                    remaining = 1.0 - scale_frac
                    trimmed = True
                out_ct = out_ct + 1 if not m[j] else 0
                if grace is not None and out_ct > grace:
                    exit_ret = close[j] / entry - 1
                    break
                if horizon is not None and (j - i) >= horizon:
                    exit_ret = close[j] / entry - 1
                    break
                j += 1
            if exit_ret is None:
                jj = min(j, n - 1)
                exit_ret = close[jj] / entry - 1
            total = realized + remaining * exit_ret
            hi_end = min(i + POTENTIAL_HORIZON, n - 1)
            potential = high[i + 1:hi_end + 1].max() / entry - 1 if hi_end > i else np.nan
            rows.append(dict(ticker=ticker, entry_ts=bars.index[i], ret=total,
                             bars_held=min(j, n - 1) - i, potential=potential))
            i = min(j, n - 1) + 1
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- cohorts
def tail_propensity_cohorts(tickers: list[str], rng_seed: int = 7) -> pd.DataFrame:
    """Cluster tickers by PRE-VAL (before 2025-07-01) return/MFE distribution shape.
    Returns [ticker, tail_cohort, mfe95, mfe_med, vol_4h, n_preval_bars]."""
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    feats = []
    for t in tickers:
        b = _bars(t)
        if b is None:
            feats.append(dict(ticker=t, n_preval_bars=0))
            continue
        b = b[b.index < VAL_START]
        nb = len(b)
        if nb < MIN_PREVAL_BARS:
            feats.append(dict(ticker=t, n_preval_bars=nb))
            continue
        close = b["close"].values
        high = b["high"].values
        # forward 60-bar MFE from each pre-val bar (entirely inside pre-val data)
        fwd_max = pd.Series(high[::-1]).rolling(POTENTIAL_HORIZON, min_periods=5).max()[::-1].shift(-1).values
        with np.errstate(divide="ignore", invalid="ignore"):
            mfe = fwd_max / close - 1
        mfe = mfe[np.isfinite(mfe)]
        rets = np.diff(np.log(np.clip(close, 1e-9, None)))
        rets = rets[np.isfinite(rets)]
        if len(mfe) < 100 or len(rets) < 100:
            feats.append(dict(ticker=t, n_preval_bars=nb))
            continue
        feats.append(dict(
            ticker=t, n_preval_bars=nb,
            mfe95=float(np.quantile(mfe, 0.95)),
            mfe_med=float(np.median(mfe)),
            vol_4h=float(np.std(rets)),
            ret_skew=float(pd.Series(rets).skew()),
        ))
    df = pd.DataFrame(feats)
    ok = df.dropna(subset=["mfe95", "mfe_med", "vol_4h", "ret_skew"]).copy()
    # rank-transform: the raw features are heavy-tailed (a handful of bad-bar
    # artifact tickers otherwise capture entire clusters), cluster on
    # distribution position instead of magnitude
    R = ok[["mfe95", "mfe_med", "vol_4h", "ret_skew"]].rank(pct=True)
    X = StandardScaler().fit_transform(R)
    km = KMeans(n_clusters=3, n_init=20, random_state=rng_seed).fit(X)
    ok["cluster"] = km.labels_
    order = ok.groupby("cluster")["mfe95"].mean().sort_values().index.tolist()
    names = {order[0]: "grinder", order[1]: "moderate", order[2]: "explosive"}
    ok["tail_cohort"] = ok["cluster"].map(names)
    df = df.merge(ok[["ticker", "tail_cohort"]], on="ticker", how="left")
    df["tail_cohort"] = df["tail_cohort"].fillna("young")
    return df


def regime_series(matrix_ts: pd.DataFrame) -> pd.DataFrame:
    """Entry-time regime label per 4H timestamp from the matrix's own regime features."""
    r = matrix_ts.copy()
    cond_riskoff = r["regime_vix_high"] > 0.5
    cond_bull = (r["regime_spy_trend"] > 0) & ~cond_riskoff
    r["regime"] = np.where(cond_riskoff, "riskoff", np.where(cond_bull, "bull", "chop"))
    return r[["timestamp", "regime"]]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- streams
    streams = {m: load_stream(m) for m in ("momentum", "htf", "meta")}
    for m, s in streams.items():
        s.to_parquet(OUT_DIR / f"stream_{m}.parquet", index=False)
        print(f"stream {m}: rows={len(s)} ts {s['timestamp'].min().date()}->{s['timestamp'].max().date()}")

    # ---- per-trade tables for every module x policy x window
    trade_frames = []
    for module, stream in streams.items():
        for window, (s, e) in {"val": (VAL_START, VAL_END), "test": (VAL_END, TEST_END)}.items():
            member = member_from_stream(stream, s, e)
            for pname, cfg in POLICIES.items():
                tr = all_trades(member, **cfg)
                tr["module"], tr["window"], tr["policy"] = module, window, pname
                trade_frames.append(tr)
                print(f"  {module:9s} {window:4s} {pname:12s} n={len(tr)}")
    trades = pd.concat(trade_frames, ignore_index=True)

    # ---- cohort axes
    all_tickers = sorted(set().union(*[set(s["ticker"]) for s in streams.values()]))
    print(f"\nclustering tail propensity for {len(all_tickers)} tickers (pre-val bars only)...")
    cohorts = tail_propensity_cohorts(all_tickers)
    cohorts.to_csv(OUT_DIR / "ticker_tail_cohorts.csv", index=False)
    print(cohorts["tail_cohort"].value_counts().to_string())
    probe = cohorts[cohorts["ticker"].isin(["AXTI", "MU", "SNDK", "WDC", "NVDA", "AAPL", "KO"])]
    print("probe tickers:\n" + probe[["ticker", "tail_cohort", "mfe95", "n_preval_bars"]].to_string(index=False))

    uni = pd.read_csv(REPO / "Data/shared/universe/shared_universe.csv",
                      usecols=["ticker", "cap_tier", "avg_dollar_volume_20d"])
    themes = pd.read_csv(REPO / "theme_expansion/data/theme_map_v4.csv",
                         usecols=["ticker", "theme_1"])

    matrix = pd.read_parquet(
        REPO / "signals/meta_context/meta_ranker/meta_ranker_matrix.parquet",
        columns=["mom_xs_rank", "dollar_vol_pctile_252", "regime_spy_trend",
                 "regime_vix_high", "theme"]).reset_index()
    matrix["timestamp"] = pd.to_datetime(matrix["timestamp"], utc=True)

    ts_regime = regime_series(
        matrix.drop_duplicates("timestamp")[["timestamp", "regime_spy_trend", "regime_vix_high"]])
    print("\nregime bar counts:\n" + ts_regime["regime"].value_counts().to_string())

    # ---- tag trades
    trades = trades.rename(columns={"entry_ts": "timestamp"})
    trades["timestamp"] = pd.to_datetime(trades["timestamp"], utc=True)
    trades = trades.merge(cohorts[["ticker", "tail_cohort"]], on="ticker", how="left")
    trades = trades.merge(uni, on="ticker", how="left")
    trades = trades.merge(themes, on="ticker", how="left")
    trades = trades.merge(matrix[["timestamp", "ticker", "mom_xs_rank", "dollar_vol_pctile_252"]],
                          on=["timestamp", "ticker"], how="left")
    trades = trades.merge(ts_regime, on="timestamp", how="left")

    trades["mom_xs_q"] = pd.cut(trades["mom_xs_rank"], [0, .2, .4, .6, .8, 1.0],
                                labels=["q1_low", "q2", "q3", "q4", "q5_high"], include_lowest=True)
    trades["dvol_bucket"] = pd.cut(trades["dollar_vol_pctile_252"], [0, 1 / 3, 2 / 3, 1.0],
                                   labels=["low_liq", "mid_liq", "high_liq"], include_lowest=True)
    trades["capture"] = np.where(trades["potential"] >= 0.20,
                                 trades["ret"] / trades["potential"], np.nan)
    trades.to_csv(OUT_DIR / "trades_all_policies.csv", index=False)
    print(f"\nsaved {len(trades)} trades -> {OUT_DIR / 'trades_all_policies.csv'}")

    # ---- phase 2 matrix
    def agg(g: pd.DataFrame) -> pd.Series:
        return pd.Series(dict(
            n=len(g), mean=g["ret"].mean(), median=g["ret"].median(),
            win=(g["ret"] > 0).mean(), total=g["ret"].sum(),
            rpb=(g["ret"] / np.maximum(g["bars_held"], 1)).mean(),
            hold=g["bars_held"].mean(),
            capture=g["capture"].mean(),
            n_tail=int((g["potential"] >= 0.5).sum()),
        ))

    mats = []
    for axis in ("tail_cohort", "cap_tier", "regime", "mom_xs_q", "dvol_bucket", "theme_1"):
        m = (trades.dropna(subset=[axis])
             .groupby(["module", "window", "policy", axis], observed=True)
             .apply(agg, include_groups=False).reset_index()
             .rename(columns={axis: "segment"}))
        m["axis"] = axis
        mats.append(m)
    matrix_out = pd.concat(mats, ignore_index=True)
    matrix_out.to_csv(OUT_DIR / "phase2_policy_by_segment.csv", index=False)

    # cell sizes for the writeup
    cells = (matrix_out[matrix_out["policy"] == "deployed"]
             .pivot_table(index=["axis", "segment"], columns=["module", "window"],
                          values="n", aggfunc="first"))
    cells.to_csv(OUT_DIR / "cell_sizes.csv")
    print("\ncell sizes (deployed policy):")
    print(cells.to_string())


if __name__ == "__main__":
    main()
