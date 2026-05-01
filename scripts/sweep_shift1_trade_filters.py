from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd


ANALYSIS_DIR = Path("Data/models/ga_xgboost/10min_shift1/analysis/phase4_1m_bodyclose_l42_s15")
TRADES_PATH = (
    ANALYSIS_DIR
    / "best_phase4_asym_long_break_prev_stop_1m_body_and_close_short_break_prev_stop_1m_body_and_close_cooldown_cluster_longmax4_shortmax4_test_trades.csv"
)
SIGNAL_FRAME_PATH = ANALYSIS_DIR / "phase4_signal_frame.parquet"
OUT_DIR = Path("Data/models/ga_xgboost/10min_shift1/analysis/trade_filter_experiments")


def _load() -> pd.DataFrame:
    frame = pd.read_parquet(SIGNAL_FRAME_PATH).sort_index()
    if frame.index.tz is None:
        frame.index = frame.index.tz_localize("America/New_York")

    trades = pd.read_csv(TRADES_PATH)
    for col in ("setup_bar_time", "entry_time", "exit_time"):
        trades[col] = pd.to_datetime(trades[col], utc=True, errors="coerce").dt.tz_convert(frame.index.tz)
    trades = trades.dropna(subset=["setup_bar_time", "entry_time", "outcome_atr"]).sort_values("entry_time").copy()

    join_cols = [
        "close",
        "ema_fast",
        "ema_slow",
        "fast_above_slow",
        "fast_below_slow",
        "above_vwap",
        "below_vwap",
        "ret_1",
        "mom_up",
        "mom_dn",
        "p_long_test",
        "p_short_test",
        "long_setup_test",
        "short_setup_test",
    ]
    setup_features = frame[join_cols].copy()
    setup_features.columns = [f"setup_{c}" for c in setup_features.columns]
    trades = trades.join(setup_features, on="setup_bar_time")
    trades["entry_lag_min"] = (trades["entry_time"] - trades["setup_bar_time"]).dt.total_seconds() / 60.0
    trades["session"] = trades["entry_time"].dt.date
    trades["side_p"] = np.where(trades["side"].eq("long"), trades["p_long"], trades["p_short"])
    trades["opp_p"] = np.where(trades["side"].eq("long"), trades["p_short"], trades["p_long"])
    trades["side_edge"] = np.where(trades["side"].eq("long"), trades["p_long"] - trades["p_short"], trades["p_short"] - trades["p_long"])
    return trades


def _trend_mask(trades: pd.DataFrame, policy: str) -> pd.Series:
    if policy == "any":
        return pd.Series(True, index=trades.index)
    is_long = trades["side"].eq("long")
    is_short = trades["side"].eq("short")
    if policy == "ema_aligned":
        return (is_long & trades["setup_fast_above_slow"].fillna(False)) | (is_short & trades["setup_fast_below_slow"].fillna(False))
    if policy == "not_ema_opposed":
        return (is_long & ~trades["setup_fast_below_slow"].fillna(False)) | (is_short & ~trades["setup_fast_above_slow"].fillna(False))
    if policy == "vwap_aligned":
        return (is_long & trades["setup_above_vwap"].fillna(False)) | (is_short & trades["setup_below_vwap"].fillna(False))
    if policy == "momentum_aligned":
        return (is_long & trades["setup_mom_up"].fillna(False)) | (is_short & trades["setup_mom_dn"].fillna(False))
    if policy == "momentum_and_ema":
        long_ok = trades["setup_mom_up"].fillna(False) & trades["setup_fast_above_slow"].fillna(False)
        short_ok = trades["setup_mom_dn"].fillna(False) & trades["setup_fast_below_slow"].fillna(False)
        return (is_long & long_ok) | (is_short & short_ok)
    if policy == "momentum_not_ema_opposed":
        long_ok = trades["setup_mom_up"].fillna(False) & ~trades["setup_fast_below_slow"].fillna(False)
        short_ok = trades["setup_mom_dn"].fillna(False) & ~trades["setup_fast_above_slow"].fillna(False)
        return (is_long & long_ok) | (is_short & short_ok)
    raise ValueError(policy)


def _apply_cooldowns(trades: pd.DataFrame, same_side_minutes: int, daily_side_cap: int) -> pd.DataFrame:
    if trades.empty:
        return trades
    kept = []
    last_by_side: dict[str, pd.Timestamp] = {}
    daily_counts: dict[tuple[object, str], int] = {}
    for idx, row in trades.sort_values("entry_time").iterrows():
        side = str(row["side"])
        session = row["session"]
        key = (session, side)
        if daily_counts.get(key, 0) >= daily_side_cap:
            continue
        last = last_by_side.get(side)
        if last is not None:
            lag = (row["entry_time"] - last).total_seconds() / 60.0
            if lag < same_side_minutes:
                continue
        kept.append(idx)
        last_by_side[side] = row["entry_time"]
        daily_counts[key] = daily_counts.get(key, 0) + 1
    return trades.loc[kept].copy()


def _metrics(df: pd.DataFrame) -> dict[str, float]:
    out = pd.to_numeric(df["outcome_atr"], errors="coerce")
    long = out[df["side"].eq("long")]
    short = out[df["side"].eq("short")]
    return {
        "trades": float(len(df)),
        "ev_atr": float(out.mean()) if len(out) else float("nan"),
        "sum_atr": float(out.sum()) if len(out) else 0.0,
        "win_rate": float((out > 0).mean()) if len(out) else float("nan"),
        "long_trades": float(len(long)),
        "long_ev_atr": float(long.mean()) if len(long) else float("nan"),
        "short_trades": float(len(short)),
        "short_ev_atr": float(short.mean()) if len(short) else float("nan"),
    }


def main() -> None:
    trades = _load()
    rows = []
    baseline = _metrics(trades)
    rows.append({"policy": "baseline_no_filter", **baseline})

    trend_policies = [
        "any",
        "ema_aligned",
        "not_ema_opposed",
        "vwap_aligned",
        "momentum_aligned",
        "momentum_and_ema",
        "momentum_not_ema_opposed",
    ]
    min_side_ps = [0.0, 0.25, 0.42, 0.60]
    min_edges = [-1.0, 0.0, 0.20, 0.35]
    max_opp_ps = [1.0, 0.50, 0.35]
    max_lags = [10.0, 4.0, 2.0]
    same_side_cooldowns = [0, 60, 120]
    daily_caps = [99, 3, 1]

    for trend, min_side_p, min_edge, max_opp_p, max_lag, cooldown, cap in itertools.product(
        trend_policies,
        min_side_ps,
        min_edges,
        max_opp_ps,
        max_lags,
        same_side_cooldowns,
        daily_caps,
    ):
        mask = (
            _trend_mask(trades, trend)
            & trades["side_p"].ge(min_side_p)
            & trades["side_edge"].ge(min_edge)
            & trades["opp_p"].le(max_opp_p)
            & trades["entry_lag_min"].le(max_lag)
        )
        filtered = _apply_cooldowns(trades[mask].copy(), cooldown, cap)
        if len(filtered) < 80:
            continue
        rows.append(
            {
                "policy": "post_trace_filter",
                "trend": trend,
                "min_side_p": min_side_p,
                "min_edge": min_edge,
                "max_opp_p": max_opp_p,
                "max_lag_min": max_lag,
                "same_side_cooldown_min": cooldown,
                "daily_side_cap": cap,
                **_metrics(filtered),
            }
        )

    out = pd.DataFrame(rows).sort_values(["ev_atr", "trades"], ascending=[False, False])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_DIR / "shift1_trade_filter_sweep.csv", index=False)
    out.head(50).to_csv(OUT_DIR / "shift1_trade_filter_sweep_top50.csv", index=False)
    print(out.head(25).to_string(index=False))
    print(f"wrote {OUT_DIR / 'shift1_trade_filter_sweep.csv'}")


if __name__ == "__main__":
    main()
