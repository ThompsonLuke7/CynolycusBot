"""
Two-sleeve exit policy (id4 tail-rider + g284 harvester) + approximate spread
overlay, run independently on each of Momentum/HTF/Meta's OWN top-10 stream.

Round 1 (two_sleeve_backtest.py) used mom_xs_rank as the sleeve-split feature,
but that's a Meta-specific derived column (Momentum/HTF's own top-10 entries
would show near-constant mom_xs_rank since it's what selects them — no
discriminative power). A quick check confirmed simple rank-within-top-10 does
NOT transfer either (Spearman(rank, MFE) on val: momentum -0.004, htf +0.074,
meta +0.067 — near zero everywhere, and for meta it's the OPPOSITE of useful:
top-3 by rank has LOWER tail rate than 4-10). So each module gets its own
feature screen against its OWN feature matrix (momentum/htf: features_4h.parquet,
115 cols; meta: meta_ranker_matrix.parquet), same methodology as before:
Spearman(entry-time feature, 60-bar MFE) on val, pick the winner, val-only
threshold, frozen test application. No feature is assumed to transfer.

Same val/test split and same options-spread caveats as two_sleeve_backtest.py
(no historical options-chain data — approximate payoff model, magnitude not
trustworthy, shape/win-rate more defensible than any specific number).

Usage:
  PYTHONPATH=. .venv/bin/python scripts/capstone/two_sleeve_cross_module.py
"""
from __future__ import annotations

import gc
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "signals/meta_context/meta_ranker"))
import backtest_exits as be  # noqa: E402

VAL_START = pd.Timestamp("2025-07-01", tz="UTC")
VAL_END = pd.Timestamp("2026-01-15", tz="UTC")
TEST_END = pd.Timestamp("2026-05-15", tz="UTC")
TOPK = 10

TAIL_POLICY = dict(stop=0.39, trail=None, target=0.30, scale_frac=0.16, horizon=53, grace=None)
HARVEST_POLICY = dict(stop=0.59, trail=None, target=0.07, scale_frac=1.0, horizon=60, grace=None)

OOF_SOURCES = {
    "momentum": REPO / "strategies/momentum_expansion/models/expansion_v1/oof_preds.parquet",
    "htf": REPO / "strategies/multi_ticker_swing_htf/models/oof_preds.parquet",
}
FEATURE_MATRICES = {
    "momentum": REPO / "strategies/momentum_expansion/data/processed/features_4h.parquet",
    "htf": REPO / "strategies/multi_ticker_swing_htf/data/processed/features_4h.parquet",
    "meta": REPO / "signals/meta_context/meta_ranker/meta_ranker_matrix.parquet",
}
LEAK_COLS = {"meta_label", "fwd_close_return", "fwd_max_drawdown", "fwd_atr_adj_return",
             "trend_persistence", "trade_quality", "meta_good", "meta_upside",
             "fwd_max_return", "fwd_max_alpha", "y", "target", "htf_top_swing_target",
             "fwd_best_high_return", "fwd_worst_low_return", "long_persistence",
             "short_persistence", "expansion_target", "expansion_score",
             "earnings_in_fwd_window"}
NON_FEATURE_COLS = {"open", "high", "low", "close", "volume", "theme", "date",
                     "sector_id", "market_cap_bucket", "asset_type", "is_etf"}

_BAR_CACHE: dict[str, pd.DataFrame | None] = {}


def _bars(ticker: str) -> pd.DataFrame | None:
    if ticker not in _BAR_CACHE:
        _BAR_CACHE[ticker] = be._ticker_path(ticker, None)
    return _BAR_CACHE[ticker]


def _mem_check(label: str) -> float:
    """Print available RAM; return it in GB. WSL has crashed twice loading these
    exact matrices unpruned (momentum's is 4.0GB on disk / htf's 1.5GB) — every
    heavy step checkpoints here so a problem is visible before the VM dies."""
    with open("/proc/meminfo") as f:
        info = {ln.split(":")[0]: ln.split()[1] for ln in f if ":" in ln}
    avail_gb = int(info["MemAvailable"]) / 1024 / 1024
    print(f"  [mem] {label}: {avail_gb:.1f} GB available")
    return avail_gb


def load_needed_rows(path: Path, columns: list[str], needed_keys: set[tuple]) -> pd.DataFrame:
    """Read a (timestamp, ticker)-indexed parquet file WITHOUT ever materializing
    it whole. Momentum's features_4h.parquet is 4.0GB on disk / 8.4M rows across
    9 row groups (~475MB each uncompressed per group); htf's is 1.5GB / 3.0M rows.
    A plain pd.read_parquet() of either, done twice in the same process, is the
    likely cause of the 2026-07-21 WSL OOM crash. This reads one row group at a
    time, immediately drops every row not in needed_keys (~2-4k of several
    million), and frees the row group before reading the next — peak memory is
    bounded to ~1 row group of pruned columns, not the whole file."""
    pf = pq.ParquetFile(path)
    read_cols = ["timestamp", "ticker"] + [c for c in columns if c not in ("timestamp", "ticker")]
    chunks = []
    for i in range(pf.num_row_groups):
        tbl = pf.read_row_group(i, columns=read_cols)
        df = tbl.to_pandas()
        del tbl
        if df.index.names and df.index.names[0] is not None:
            df = df.reset_index()  # pyarrow auto-restores the stored (timestamp, ticker) index
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        keys = list(zip(df["timestamp"], df["ticker"]))
        mask = pd.Series(keys, index=df.index).isin(needed_keys)
        if mask.any():
            chunks.append(df[mask].copy())
        del df, keys, mask
        gc.collect()
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=read_cols)


def load_member(module: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    if module == "meta":
        df = pd.read_parquet(be.SCORED).dropna(subset=["s_combo"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        score_col = "s_combo"
    else:
        df = pd.read_parquet(OOF_SOURCES[module]).reset_index()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.dropna(subset=["score"])
        score_col = "score"
    df["rk"] = df.groupby("timestamp")[score_col].rank(ascending=False, method="first")
    df = df[(df["timestamp"] >= start) & (df["timestamp"] < end)]
    df["in_top"] = df["rk"] <= TOPK
    return df[["timestamp", "ticker", "in_top"]]


def mfe_entries(member: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ticker, g in member[member["in_top"]].groupby("ticker"):
        bars = _bars(ticker)
        if bars is None:
            continue
        for ts in g["timestamp"]:
            if ts not in bars.index:
                continue
            pos = bars.index.get_loc(ts)
            if isinstance(pos, slice):
                pos = pos.start
            n = len(bars)
            H = min(60, n - 1 - pos)
            if H < 10:
                continue
            entry = bars["close"].values[pos]
            if entry <= 0:
                continue
            mfe = (bars["high"].values[pos + 1:pos + 1 + H] / entry - 1).max()
            rows.append(dict(ticker=ticker, timestamp=ts, mfe=mfe))
    return pd.DataFrame(rows)


def feature_columns_for(module: str) -> list[str]:
    """Schema-only (no data read) so this is free even for the 4GB files."""
    names = pq.ParquetFile(FEATURE_MATRICES[module]).schema.names
    return [c for c in names if c not in LEAK_COLS and c not in NON_FEATURE_COLS
            and c not in ("timestamp", "ticker")]


def screen_split_feature(module: str, feat_matrix: pd.DataFrame, val_mfe: pd.DataFrame,
                          feat_cols: list[str]) -> tuple[str, float, float]:
    """Returns (feature_name, spearman, val_threshold). Threshold = top-tercile.
    feat_matrix is already the small, pre-filtered (needed rows only) frame."""
    val_mfe = val_mfe.merge(feat_matrix, on=["timestamp", "ticker"], how="left")
    numeric = [c for c in feat_cols if pd.api.types.is_numeric_dtype(val_mfe[c])]
    rows = []
    for feat in numeric:
        sub = val_mfe.dropna(subset=[feat])
        if len(sub) < 50 or sub[feat].nunique() < 5:
            continue
        rho = sub[[feat, "mfe"]].corr(method="spearman").iloc[0, 1]
        rows.append((feat, rho, sub[feat].quantile(2 / 3) if rho > 0 else sub[feat].quantile(1 / 3)))
    rows.sort(key=lambda t: abs(t[1]), reverse=True)
    top = rows[0]
    print(f"  [{module}] top 5 split-feature candidates:")
    for feat, rho, _ in rows[:5]:
        print(f"    {feat:28s} spearman={rho:+.3f}")
    return top


def two_sleeve_trades(member: pd.DataFrame, feat_lookup: dict, split_thresh: float,
                       split_positive: bool) -> list[dict]:
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
            fval = feat_lookup.get((entry_ts, ticker))
            if fval is None:
                sleeve = "harvest"
            else:
                sleeve = "tail" if ((fval >= split_thresh) == split_positive) else "harvest"
            cfg = TAIL_POLICY if sleeve == "tail" else HARVEST_POLICY
            stop, trail, target, scale_frac, horizon, grace = (cfg.get(k) for k in
                ("stop", "trail", "target", "scale_frac", "horizon", "grace"))
            scale_frac = scale_frac if scale_frac is not None else 1.0
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
            out.append(dict(ticker=ticker, entry_ts=entry_ts, sleeve=sleeve,
                             ret=total, bars_held=min(j, n - 1) - i))
            i = min(j, n - 1) + 1
    return out


def bull_call_debit_spread(r, width_pct, debit_frac=0.45):
    frac = np.clip(r / width_pct, 0.0, 1.0)
    return (frac - debit_frac) / debit_frac


def bull_put_credit_spread(r, short_buffer=0.05, protect_width=0.15, credit_frac=0.30):
    loss_frac = np.clip((-short_buffer - r) / protect_width, 0.0, 1.0)
    credit = credit_frac * protect_width
    max_loss = protect_width - credit
    pnl = credit - loss_frac * protect_width
    return pnl / max_loss if max_loss > 0 else 0.0


def stats(s: pd.Series) -> dict:
    return dict(n=len(s), mean=s.mean(), median=s.median(), win=(s > 0).mean(), total=s.sum())


MIN_AVAIL_GB = 4.0  # abort before a module if free memory drops below this


def main() -> None:
    _mem_check("startup")
    all_rows = []
    for module in ("momentum", "htf", "meta"):
        print(f"\n{'='*70}\n{module.upper()}")
        avail = _mem_check(f"before {module}")
        if avail < MIN_AVAIL_GB:
            print(f"  ABORTING remaining modules: only {avail:.1f} GB available "
                  f"(< {MIN_AVAIL_GB} GB safety floor). Re-run after freeing memory.")
            break

        val_member = load_member(module, VAL_START, VAL_END)
        test_member = load_member(module, VAL_END, TEST_END)
        val_mfe = mfe_entries(val_member)
        print(f"  val entries: {len(val_mfe)}")

        needed_keys = set(zip(val_member.loc[val_member["in_top"], "timestamp"],
                               val_member.loc[val_member["in_top"], "ticker"]))
        needed_keys |= set(zip(test_member.loc[test_member["in_top"], "timestamp"],
                                test_member.loc[test_member["in_top"], "ticker"]))

        feat_cols = feature_columns_for(module)
        print(f"  reading {len(needed_keys)} needed entry-rows out of "
              f"{pq.ParquetFile(FEATURE_MATRICES[module]).metadata.num_rows:,} "
              f"({len(feat_cols)} candidate feature cols), one row-group at a time...")
        feat_matrix = load_needed_rows(FEATURE_MATRICES[module], feat_cols, needed_keys)
        _mem_check(f"after bounded read for {module}")

        feat_name, rho, thresh = screen_split_feature(module, feat_matrix, val_mfe, feat_cols)
        split_positive = rho > 0
        print(f"  -> using {feat_name} (spearman={rho:+.3f}), "
              f"{'top' if split_positive else 'bottom'}-tercile threshold={thresh:.4f}")

        feat_lookup = dict(zip(zip(feat_matrix["timestamp"], feat_matrix["ticker"]),
                                feat_matrix[feat_name]))
        del feat_matrix
        gc.collect()

        for label, member in (("VAL", val_member), ("TEST", test_member)):
            trades = two_sleeve_trades(member, feat_lookup, thresh, split_positive)
            df = pd.DataFrame(trades)
            tail = df[df["sleeve"] == "tail"]
            harv = df[df["sleeve"] == "harvest"].copy()
            harv["call_spread"] = harv["ret"].apply(
                lambda r: bull_call_debit_spread(r, HARVEST_POLICY["target"]))
            harv["put_spread"] = harv["ret"].apply(bull_put_credit_spread)

            s_tail, s_harv = stats(tail["ret"]), stats(harv["ret"])
            s_call, s_put = stats(harv["call_spread"]), stats(harv["put_spread"])
            print(f"  [{label:4s}] tail n={s_tail['n']:4d} mean={s_tail['mean']*100:6.2f}% "
                  f"win={s_tail['win']*100:5.1f}%  |  harvest n={s_harv['n']:4d} "
                  f"mean={s_harv['mean']*100:6.2f}% win={s_harv['win']*100:5.1f}%  "
                  f"|  harvest+call win={s_call['win']*100:5.1f}%  harvest+put win={s_put['win']*100:5.1f}%")
            all_rows.append(dict(module=module, split_feature=feat_name, split_spearman=rho,
                                  window=label, tail_n=s_tail["n"], tail_mean=s_tail["mean"],
                                  tail_win=s_tail["win"], harvest_n=s_harv["n"],
                                  harvest_mean=s_harv["mean"], harvest_win=s_harv["win"],
                                  harvest_call_mean=s_call["mean"], harvest_call_win=s_call["win"],
                                  harvest_put_mean=s_put["mean"], harvest_put_win=s_put["win"]))

        # per-ticker OHLCV cache doesn't need to persist across modules (different
        # universes); drop it so it can't accumulate over a 3-module run.
        _BAR_CACHE.clear()
        gc.collect()
        _mem_check(f"after {module}")

    out = pd.DataFrame(all_rows)
    out.to_csv(REPO / "research/capstone/two_sleeve_cross_module.csv", index=False)
    print(f"\nsaved research/capstone/two_sleeve_cross_module.csv ({len(out)} rows)")


if __name__ == "__main__":
    main()
