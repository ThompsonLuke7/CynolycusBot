"""
Entry-quality filter analysis for the Meta Ranker deployed exit policy.

Question: among entries into Meta's top-10, can ENTRY-TIME features (nothing
forward-looking) separate the bottom decile (losers) from the rest, well
enough that dropping them improves the deployed policy's aggregate stats?

Same val/test split as exit_policy_random_search.py / exit_policy_guided_search.py:
  VAL  2025-07-01 -> 2026-01-15   (screen features, pick a rule)
  TEST 2026-01-15 -> 2026-05-14   (frozen: apply the val-picked rule once)

Only entry-time-safe columns from meta_ranker_matrix.parquet are used —
fwd_*/meta_good/meta_upside/trend_persistence/meta_label are excluded (those
are the training labels, known only after the fact).

Usage:
  PYTHONPATH=. .venv/bin/python scripts/capstone/exit_policy_entry_quality.py
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

MATRIX = REPO / "signals/meta_context/meta_ranker/meta_ranker_matrix.parquet"
OUT = REPO / "research/capstone/exit_policy_entry_quality_trades.csv"

DEPLOYED = dict(stop=0.50, trail=0.35, target=0.20, scale_frac=0.5, horizon=25, grace=None)

LEAK_COLS = {"meta_label", "fwd_close_return", "fwd_max_drawdown", "fwd_atr_adj_return",
             "trend_persistence", "trade_quality", "meta_good", "meta_upside",
             "fwd_max_return", "fwd_max_alpha"}

_BAR_CACHE: dict[str, pd.DataFrame | None] = {}


def _bars(ticker: str) -> pd.DataFrame | None:
    if ticker not in _BAR_CACHE:
        _BAR_CACHE[ticker] = be._ticker_path(ticker, None)
    return _BAR_CACHE[ticker]


def all_trades(member: pd.DataFrame, **cfg) -> list[dict]:
    """Same exit mechanic as backtest_exits.simulate(), but returns one row per trade."""
    stop, trail, target, scale_frac, horizon, grace = (cfg.get(k) for k in
        ("stop", "trail", "target", "scale_frac", "horizon", "grace"))
    scale_frac = scale_frac if scale_frac is not None else 1.0
    out = []
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
            entry_ts = bars.index[i]
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
            out.append(dict(ticker=ticker, entry_ts=entry_ts, ret=total, bars_held=min(j, n - 1) - i))
            i = min(j, n - 1) + 1
    return out


def _load_member_window(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    df = pd.read_parquet(be.SCORED).dropna(subset=["s_combo"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["rk"] = df.groupby("timestamp")["s_combo"].rank(ascending=False, method="first")
    df = df[(df["timestamp"] >= start) & (df["timestamp"] < end)]
    df["in_top"] = df["rk"] <= be.TOPK
    return df[["timestamp", "ticker", "in_top", "rk", "s_combo"]]


def main() -> None:
    val_member = _load_member_window(VAL_START, VAL_END)
    test_member = _load_member_window(VAL_END, TEST_END)

    print("extracting per-trade returns under the DEPLOYED policy (val + test)...")
    val_trades = pd.DataFrame(all_trades(val_member, **DEPLOYED))
    test_trades = pd.DataFrame(all_trades(test_member, **DEPLOYED))
    print(f"val trades: {len(val_trades)}   test trades: {len(test_trades)}")

    matrix = pd.read_parquet(MATRIX)
    feat_cols = [c for c in matrix.columns if c not in LEAK_COLS
                 and c not in ("theme", "date", "sector_id", "market_cap_bucket", "asset_type", "is_etf")]
    matrix_feats = matrix[feat_cols].reset_index()  # timestamp, ticker, <features>

    def attach_features(trades: pd.DataFrame) -> pd.DataFrame:
        trades = trades.rename(columns={"entry_ts": "timestamp"})
        trades["timestamp"] = pd.to_datetime(trades["timestamp"], utc=True)
        out = trades.merge(matrix_feats, on=["timestamp", "ticker"], how="left")
        return out

    val_trades = attach_features(val_trades)
    test_trades = attach_features(test_trades)
    numeric_feats = [c for c in feat_cols if pd.api.types.is_numeric_dtype(matrix_feats[c])]

    print(f"\n{len(numeric_feats)} candidate entry-time features. Missingness in val trades:")
    miss = val_trades[numeric_feats].isna().mean().sort_values(ascending=False)
    print(miss[miss > 0.3].to_string() if (miss > 0.3).any() else "  (none >30% missing)")

    # ---- univariate screen on VAL: does the feature separate bottom-decile losers? ----
    val = val_trades.dropna(subset=["ret"]).copy()
    bottom_cut = val["ret"].quantile(0.10)
    val["is_loser"] = val["ret"] <= bottom_cut
    print(f"\nVAL: n={len(val)}, bottom-decile cutoff ret<={bottom_cut:.3f} "
          f"({val['is_loser'].sum()} trades), overall mean={val['ret'].mean():.4f}")

    rows = []
    for feat in numeric_feats:
        sub = val.dropna(subset=[feat])
        if len(sub) < 50 or sub[feat].nunique() < 5:
            continue
        rho = sub[[feat, "ret"]].corr(method="spearman").iloc[0, 1]
        # point-biserial-ish: mean feature value, losers vs rest
        lo = sub.loc[sub["is_loser"], feat].mean()
        hi = sub.loc[~sub["is_loser"], feat].mean()
        rows.append(dict(feature=feat, n=len(sub), spearman_vs_ret=rho,
                          mean_losers=lo, mean_others=hi, gap=hi - lo))
    screen = pd.DataFrame(rows).sort_values("spearman_vs_ret", key=lambda s: s.abs(), ascending=False)
    print("\n=== top 15 entry-time features by |Spearman(feature, trade_ret)| on VAL ===")
    print(screen.head(15).to_string(index=False))

    # ---- pick the strongest feature, define a val-only threshold rule, freeze it on test ----
    top = screen.iloc[0]
    feat = top["feature"]
    direction = "low" if top["spearman_vs_ret"] > 0 else "high"
    val_valid = val.dropna(subset=[feat])
    if direction == "low":
        thresh = val_valid[feat].quantile(0.20)
        rule = f"{feat} <= {thresh:.4f} (val 20th pct) -> SKIP"
        val_skip = val_valid[feat] <= thresh
    else:
        thresh = val_valid[feat].quantile(0.80)
        rule = f"{feat} >= {thresh:.4f} (val 80th pct) -> SKIP"
        val_skip = val_valid[feat] >= thresh

    def report(name, trades_df, skip_mask):
        kept = trades_df[~skip_mask]
        skipped = trades_df[skip_mask]
        print(f"  {name}: baseline n={len(trades_df)} mean={trades_df['ret'].mean():.4f} "
              f"win={((trades_df['ret']>0).mean()):.3f}  |  "
              f"filtered n={len(kept)} mean={kept['ret'].mean():.4f} win={((kept['ret']>0).mean()):.3f}  |  "
              f"skipped(n={len(skipped)}) mean={skipped['ret'].mean():.4f}")

    print(f"\n=== val-selected rule: {rule} ===")
    report("VAL ", val_valid, val_skip)

    test_valid = test_trades.dropna(subset=[feat, "ret"])
    if direction == "low":
        test_skip = test_valid[feat] <= thresh
    else:
        test_skip = test_valid[feat] >= thresh
    report("TEST", test_valid, test_skip)

    val_trades.to_csv(REPO / "research/capstone/exit_policy_entry_quality_val_trades.csv", index=False)
    test_trades.to_csv(REPO / "research/capstone/exit_policy_entry_quality_test_trades.csv", index=False)
    screen.to_csv(REPO / "research/capstone/exit_policy_entry_quality_feature_screen.csv", index=False)
    print("\nsaved trades + feature screen to research/capstone/")


if __name__ == "__main__":
    main()
