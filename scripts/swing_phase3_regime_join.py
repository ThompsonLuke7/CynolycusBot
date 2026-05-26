"""Phase 3: join regime / context features into the trades dataset.

For each trade's entry timestamp, look up context features from cached parquets:
  - QQQ 10m close (proxy for tech/large-cap broad market)
  - IWM 10m close (small-cap)
  - VIXY 10m close (vol)
  - 30m and 4h returns on each

QQQ and IWM cached only since 2026-04-27, but the trade window starts 5/14, so OK.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _load_10m(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").drop_duplicates("timestamp")
    return df.set_index("timestamp")


def _last_bar_at_or_before(idx: pd.DatetimeIndex, ts: pd.Timestamp) -> pd.Timestamp | None:
    pos = idx.searchsorted(ts, side="right") - 1
    if pos < 0:
        return None
    return idx[pos]


def _add_context_returns(
    trades: pd.DataFrame,
    name: str,
    bars: pd.DataFrame,
    lookbacks_min: dict[str, int],
) -> pd.DataFrame:
    """For each trade row, attach `{name}_close` and `{name}_ret_{k}` cols."""
    closes = bars["close"]
    idx = bars.index

    closes_at = np.full(len(trades), np.nan)
    ret_cols = {k: np.full(len(trades), np.nan) for k in lookbacks_min}

    for i, row in enumerate(trades.itertuples(index=False)):
        ts_raw = getattr(row, "open_event_ts")
        if not isinstance(ts_raw, str):
            continue
        try:
            ts = pd.Timestamp(ts_raw).tz_convert("UTC")
        except Exception:
            continue
        b_now = _last_bar_at_or_before(idx, ts)
        if b_now is None:
            continue
        c_now = float(closes.loc[b_now])
        closes_at[i] = c_now
        for label, mins in lookbacks_min.items():
            b_then = _last_bar_at_or_before(idx, ts - pd.Timedelta(minutes=mins))
            if b_then is None:
                continue
            c_then = float(closes.loc[b_then])
            if c_then > 0:
                ret_cols[label][i] = (c_now - c_then) / c_then

    trades = trades.copy()
    trades[f"{name}_close"] = closes_at
    for label, arr in ret_cols.items():
        trades[f"{name}_ret_{label}"] = arr
    return trades


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trades", default="local_artifacts/swing_analysis_20260525/trades.parquet")
    p.add_argument("--out", default="local_artifacts/swing_analysis_20260525/trades_with_regime.parquet")
    args = p.parse_args()

    trades = pd.read_parquet(args.trades)
    print(f"Loaded {len(trades)} trades")

    qqq = _load_10m("Data/raw/qqq/qqq_10min_live_runtime.parquet")
    iwm = _load_10m("Data/raw/iwm/iwm_10min_live_runtime.parquet")
    vix = _load_10m("Data/raw/vix/vixy_10min_live_runtime.parquet")

    lookbacks = {"30m": 30, "1h": 60, "4h": 240}

    trades = _add_context_returns(trades, "qqq", qqq, lookbacks)
    trades = _add_context_returns(trades, "iwm", iwm, lookbacks)
    trades = _add_context_returns(trades, "vix", vix, lookbacks)

    # Derived: QQQ vs IWM relative strength (1h)
    trades["qqq_iwm_relstr_1h"] = trades["qqq_ret_1h"] - trades["iwm_ret_1h"]

    # Whether trade direction aligns with QQQ short-term direction
    trades["dir_aligned_qqq_30m"] = (
        ((trades["direction"] == 1) & (trades["qqq_ret_30m"] > 0))
        | ((trades["direction"] == -1) & (trades["qqq_ret_30m"] < 0))
    )
    trades["dir_aligned_qqq_1h"] = (
        ((trades["direction"] == 1) & (trades["qqq_ret_1h"] > 0))
        | ((trades["direction"] == -1) & (trades["qqq_ret_1h"] < 0))
    )

    # Coverage check
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    trades.to_parquet(out_path, index=False)
    trades.to_csv(out_path.with_suffix(".csv"), index=False)
    print(f"Wrote {len(trades)} rows -> {out_path}")
    for col in ["qqq_ret_30m", "iwm_ret_30m", "vix_ret_30m", "qqq_iwm_relstr_1h"]:
        cov = trades[col].notna().sum()
        print(f"  {col}: {cov}/{len(trades)} ({100*cov/len(trades):.0f}%)")


if __name__ == "__main__":
    main()
