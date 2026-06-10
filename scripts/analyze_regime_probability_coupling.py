from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from strategies.spy_intraday.Policy.regime_filter import StickyRegimeConfig, add_sticky_trend_regime


DEFAULT_ANALYSIS_DIR = Path(
    "Data/models/ga_xgboost/10min/analysis/"
    "phase4_1m_oof_focused_trigger_sweep_l42_s15_full_1m_train"
)
DEFAULT_SIGNAL_FRAME = DEFAULT_ANALYSIS_DIR / "phase4_signal_frame.parquet"
DEFAULT_SUMMARY_OUT = DEFAULT_ANALYSIS_DIR / "phase4_regime_probability_coupling_summary.csv"
DEFAULT_BINS_OUT = DEFAULT_ANALYSIS_DIR / "phase4_regime_probability_coupling_bins.csv"


def _combine(df: pd.DataFrame, primary: str, fallback: str) -> pd.Series:
    one = pd.to_numeric(df[primary], errors="coerce") if primary in df.columns else pd.Series(index=df.index, dtype=float)
    two = pd.to_numeric(df[fallback], errors="coerce") if fallback in df.columns else pd.Series(index=df.index, dtype=float)
    return one.combine_first(two)


def _corr(a: pd.Series, b: pd.Series) -> float:
    aligned = pd.concat([a, b], axis=1).dropna()
    if len(aligned) < 20 or aligned.iloc[:, 0].nunique() < 2 or aligned.iloc[:, 1].nunique() < 2:
        return float("nan")
    return float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1]))


def _load(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path).sort_index()
    if not isinstance(df.index, pd.DatetimeIndex):
        if "timestamp" not in df.columns:
            raise ValueError("Signal frame must have DatetimeIndex or timestamp column.")
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()
    idx = pd.to_datetime(df.index, utc=True, errors="coerce")
    df = df.loc[pd.notna(idx)].copy()
    df.index = pd.DatetimeIndex(idx[pd.notna(idx)]).tz_convert("America/New_York")
    for col in ("open", "high", "low", "close", "ema_fast", "ema_slow"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["timestamp"] = df.index
    df["p_long"] = _combine(df, "p_long_test", "p_long_oof_train")
    df["p_short"] = _combine(df, "p_short_test", "p_short_oof_train")
    df["target_long"] = _combine(df, "long_setup_test", "long_setup_oof")
    df["target_short"] = _combine(df, "short_setup_test", "short_setup_oof")
    df = add_sticky_trend_regime(df, config=StickyRegimeConfig())
    for bars in (4, 12, 24):
        df[f"fwd_ret_{bars}"] = df["close"].shift(-bars) / df["close"] - 1.0
        df[f"fwd_short_ret_{bars}"] = -df[f"fwd_ret_{bars}"]
    return df.dropna(subset=["p_long", "p_short", "close", "trend_regime"]).copy()


def _summarize_window(df: pd.DataFrame, *, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for regime, group in df.groupby("trend_regime", sort=True):
        for side in ("long", "short"):
            p_col = f"p_{side}"
            t_col = f"target_{side}"
            fwd_col = "fwd_ret_12" if side == "long" else "fwd_short_ret_12"
            probs = pd.to_numeric(group[p_col], errors="coerce")
            target = pd.to_numeric(group[t_col], errors="coerce")
            fwd = pd.to_numeric(group[fwd_col], errors="coerce")
            threshold = 0.35 if side == "long" else 0.65
            rows.append(
                {
                    "window": label,
                    "regime": regime,
                    "side": side,
                    "rows": int(len(group)),
                    "target_rate": float(target.mean()),
                    "p_mean": float(probs.mean()),
                    "p_median": float(probs.median()),
                    "p_q75": float(probs.quantile(0.75)),
                    "p_q90": float(probs.quantile(0.90)),
                    "p_q95": float(probs.quantile(0.95)),
                    "share_above_current_threshold": float((probs >= threshold).mean()),
                    "corr_prob_to_target": _corr(probs, target),
                    "corr_prob_to_12bar_direction": _corr(probs, fwd),
                    "avg_12bar_direction_return": float(fwd.mean()),
                }
            )
    return rows


def _bin_window(df: pd.DataFrame, *, label: str, bins: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for regime, regime_group in df.groupby("trend_regime", sort=True):
        for side in ("long", "short"):
            p_col = f"p_{side}"
            t_col = f"target_{side}"
            fwd_col = "fwd_ret_12" if side == "long" else "fwd_short_ret_12"
            group = regime_group.dropna(subset=[p_col, t_col]).copy()
            if len(group) < bins * 5 or group[p_col].nunique() < bins:
                continue
            group["prob_bin"] = pd.qcut(group[p_col], q=bins, labels=False, duplicates="drop")
            max_bin = int(group["prob_bin"].max())
            for prob_bin, chunk in group.groupby("prob_bin", sort=True):
                probs = pd.to_numeric(chunk[p_col], errors="coerce")
                target = pd.to_numeric(chunk[t_col], errors="coerce")
                fwd = pd.to_numeric(chunk[fwd_col], errors="coerce")
                rows.append(
                    {
                        "window": label,
                        "regime": regime,
                        "side": side,
                        "prob_bin": int(prob_bin),
                        "is_top_bin": bool(int(prob_bin) == max_bin),
                        "rows": int(len(chunk)),
                        "p_min": float(probs.min()),
                        "p_mean": float(probs.mean()),
                        "p_max": float(probs.max()),
                        "target_rate": float(target.mean()),
                        "avg_12bar_direction_return": float(fwd.mean()),
                    }
                )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze probability magnitude by sticky regime.")
    parser.add_argument("--signal-frame", default=str(DEFAULT_SIGNAL_FRAME))
    parser.add_argument("--recent-months", type=int, default=3)
    parser.add_argument("--bins", type=int, default=5)
    parser.add_argument("--summary-out", default=str(DEFAULT_SUMMARY_OUT))
    parser.add_argument("--bins-out", default=str(DEFAULT_BINS_OUT))
    args = parser.parse_args()

    df = _load(Path(args.signal_frame))
    end = df.index.max()
    recent_start = end - pd.DateOffset(months=int(args.recent_months))
    windows = {
        "full": df,
        f"recent_{int(args.recent_months)}m": df[df.index >= recent_start].copy(),
    }

    summary_rows: list[dict[str, Any]] = []
    bin_rows: list[dict[str, Any]] = []
    for label, window_df in windows.items():
        summary_rows.extend(_summarize_window(window_df, label=label))
        bin_rows.extend(_bin_window(window_df, label=label, bins=int(args.bins)))

    summary = pd.DataFrame(summary_rows)
    bins_df = pd.DataFrame(bin_rows)
    summary_out = Path(args.summary_out)
    bins_out = Path(args.bins_out)
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_out, index=False)
    bins_df.to_csv(bins_out, index=False)

    print(f"[regime-prob-coupling] wrote {summary_out}")
    print(f"[regime-prob-coupling] wrote {bins_out}")
    print("\nsummary:")
    print(summary.to_string(index=False))
    if not bins_df.empty:
        print("\ntop probability bins:")
        top = bins_df[bins_df["is_top_bin"]].sort_values(["window", "regime", "side"])
        print(top.to_string(index=False))


if __name__ == "__main__":
    main()
