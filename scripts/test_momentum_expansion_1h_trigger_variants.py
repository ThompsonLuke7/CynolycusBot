from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from momentum_expansion.data.load_bars import load_1h


DEFAULT_MATRIX = Path("momentum_expansion/data/processed/training_matrix_4h.parquet")
DEFAULT_PREDS = Path("momentum_expansion/data/processed/label_model_experiment/holdout_predictions.parquet")
DEFAULT_OUT = Path("momentum_expansion/data/processed/trigger_variant_experiment")


@dataclass(frozen=True)
class Trigger:
    policy: str
    ts: pd.Timestamp
    price: float


def _atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(length).mean()


def _with_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [c.lower() for c in out.columns]
    out["ema10"] = out["close"].ewm(span=10, adjust=False).mean()
    out["ema20"] = out["close"].ewm(span=20, adjust=False).mean()
    out["atr14"] = _atr(out, 14)
    out["vol20"] = out["volume"].rolling(20).mean()
    out["prev_high"] = out["high"].shift(1)
    out["prev_close"] = out["close"].shift(1)
    return out


def _is_rth_index(index: pd.DatetimeIndex) -> np.ndarray:
    ny = index.tz_convert("America/New_York")
    minutes = ny.hour * 60 + ny.minute
    return np.asarray((minutes >= 9 * 60 + 30) & (minutes <= 16 * 60))


def _first_trigger(policy: str, full: pd.DataFrame, start_ts: pd.Timestamp, max_hours: int) -> Trigger | None:
    horizon_end = start_ts + pd.Timedelta(hours=max_hours)
    pos = full.index.searchsorted(start_ts, side="right")
    window = full.iloc[pos:].loc[:horizon_end]
    if window.empty:
        return None
    window = window.loc[_is_rth_index(window.index)]
    if window.empty:
        return None

    if policy == "next_1h_open":
        row = window.iloc[0]
        return Trigger(policy=policy, ts=window.index[0], price=float(row["open"]))

    for ts, row in window.iterrows():
        if not np.isfinite(row["atr14"]) or row["atr14"] <= 0:
            continue
        bullish_body = row["close"] > row["open"]
        up_close = row["close"] > row["prev_close"]
        if policy == "break_touch_prev_high":
            ok = row["high"] >= row["prev_high"]
            price = row["prev_high"]
        elif policy == "break_body_prev_high":
            ok = bool((row["high"] >= row["prev_high"]) and bullish_body)
            price = row["prev_high"]
        elif policy == "break_body_close_prev_high":
            ok = bool(bullish_body and (row["close"] > row["prev_high"]))
            price = row["close"]
        elif policy == "volume_body_1p5":
            ok = bool(bullish_body and up_close and row["volume"] >= 1.5 * row["vol20"])
            price = row["close"]
        elif policy == "volume_body_2p0":
            ok = bool(bullish_body and up_close and row["volume"] >= 2.0 * row["vol20"])
            price = row["close"]
        elif policy == "ema10_reclaim":
            prior = full.loc[:ts].tail(9).iloc[:-1]
            ok = bool((prior["close"] < prior["ema10"]).any() and row["close"] > row["ema10"] and bullish_body)
            price = row["close"]
        elif policy == "ema20_reclaim":
            prior = full.loc[:ts].tail(9).iloc[:-1]
            ok = bool((prior["close"] < prior["ema20"]).any() and row["close"] > row["ema20"] and bullish_body)
            price = row["close"]
        elif policy == "flag_breakout_4":
            prior = full.loc[:ts].tail(5).iloc[:-1]
            if len(prior) < 4:
                ok = False
            else:
                flag_high = prior["high"].max()
                flag_range = prior["high"].max() - prior["low"].min()
                ok = bool(flag_range <= 1.5 * row["atr14"] and row["close"] > flag_high + 0.25 * row["atr14"])
            price = row["close"]
        elif policy == "pullback_ema10_reclaim":
            prior = full.loc[:ts].tail(13).iloc[:-1]
            if len(prior) < 12:
                ok = False
            else:
                trend_ok = bool((prior["ema10"].tail(5) > prior["ema20"].tail(5)).all())
                look_high = prior["high"].iloc[-11:-1].max()
                pullback_atr = (look_high - prior["low"].tail(3).min()) / row["atr14"]
                ok = bool(trend_ok and 0.4 <= pullback_atr <= 2.5 and row["close"] > row["ema10"])
            price = row["close"]
        elif policy == "current_any":
            # Approximate the current live rule set: pullback, flag breakout,
            # volume confirmation, or EMA20 reclaim.
            nested = [
                "pullback_ema10_reclaim",
                "flag_breakout_4",
                "volume_body_1p5",
                "ema20_reclaim",
            ]
            for nested_policy in nested:
                nested_trig = _first_trigger(nested_policy, full.loc[:ts], ts - pd.Timedelta(nanoseconds=1), 1)
                if nested_trig is not None:
                    return Trigger(policy=policy, ts=ts, price=nested_trig.price)
            continue
        else:
            raise ValueError(f"unknown policy: {policy}")

        if ok and np.isfinite(price) and price > 0:
            return Trigger(policy=policy, ts=ts, price=float(price))
    return None


def _forward_outcome(full: pd.DataFrame, entry_ts: pd.Timestamp, entry_price: float, horizon_bars: int) -> dict[str, float]:
    start = full.index.searchsorted(entry_ts, side="right")
    future = full.iloc[start : start + horizon_bars]
    future = future.loc[_is_rth_index(future.index)]
    if future.empty or not np.isfinite(entry_price) or entry_price <= 0:
        return {"fwd_max_return": np.nan, "fwd_close_return": np.nan, "fwd_max_drawdown": np.nan}
    return {
        "fwd_max_return": float(future["high"].max() / entry_price - 1.0),
        "fwd_close_return": float(future["close"].iloc[-1] / entry_price - 1.0),
        "fwd_max_drawdown": float(max(0.0, 1.0 - future["low"].min() / entry_price)),
    }


def _summarize(events: pd.DataFrame, total_candidates: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for policy, g in events.groupby("policy"):
        winners = g.loc[g["fwd_close_return"] > 0, "fwd_close_return"]
        losers = g.loc[g["fwd_close_return"] <= 0, "fwd_close_return"]
        rows.append(
            {
                "policy": policy,
                "trades": int(len(g)),
                "trigger_rate": float(len(g) / max(total_candidates, 1)),
                "avg_fwd_max_return": float(g["fwd_max_return"].mean()),
                "median_fwd_max_return": float(g["fwd_max_return"].median()),
                "avg_fwd_close_return": float(g["fwd_close_return"].mean()),
                "avg_drawdown": float(g["fwd_max_drawdown"].mean()),
                "pct_gt_10": float((g["fwd_max_return"] >= 0.10).mean()),
                "pct_gt_20": float((g["fwd_max_return"] >= 0.20).mean()),
                "pct_gt_40": float((g["fwd_max_return"] >= 0.40).mean()),
                "pct_dd_gt_15": float((g["fwd_max_drawdown"] > 0.15).mean()),
                "avg_close_winner": float(winners.mean()) if len(winners) else np.nan,
                "avg_close_loser": float(losers.mean()) if len(losers) else np.nan,
                "close_win_rate": float((g["fwd_close_return"] > 0).mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["pct_gt_20", "avg_fwd_max_return"], ascending=False)


def run(
    matrix_path: Path,
    preds_path: Path,
    out_dir: Path,
    top_n: int,
    max_hours: int,
    horizon_bars: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    preds = pd.read_parquet(preds_path)
    pred_col = "EXP_C_xsec_score_regression"
    if pred_col not in preds.columns:
        raise ValueError(f"Missing {pred_col} in {preds_path}")

    base = preds[[pred_col]].copy().dropna()
    ranked = (
        base.reset_index()
        .sort_values(["timestamp", pred_col], ascending=[True, False])
        .groupby("timestamp")
        .head(top_n)
        .reset_index(drop=True)
    )
    policies = [
        "next_1h_open",
        "break_touch_prev_high",
        "break_body_prev_high",
        "break_body_close_prev_high",
        "volume_body_1p5",
        "volume_body_2p0",
        "ema10_reclaim",
        "ema20_reclaim",
        "flag_breakout_4",
        "pullback_ema10_reclaim",
        "current_any",
    ]

    cache: dict[str, pd.DataFrame | None] = {}
    events: list[dict[str, object]] = []
    for row in ranked.itertuples(index=False):
        ts = pd.Timestamp(row.timestamp)
        ticker = str(row.ticker)
        if ticker not in cache:
            try:
                cache[ticker] = _with_indicators(load_1h(ticker))
            except FileNotFoundError:
                cache[ticker] = None
        bars = cache[ticker]
        if bars is None or bars.empty:
            continue
        for policy in policies:
            trig = _first_trigger(policy, bars, ts, max_hours=max_hours)
            if trig is None:
                continue
            outcome = _forward_outcome(bars, trig.ts, trig.price, horizon_bars=horizon_bars)
            if not np.isfinite(outcome["fwd_max_return"]):
                continue
            events.append(
                {
                    "timestamp": ts,
                    "ticker": ticker,
                    "prediction": float(getattr(row, pred_col)),
                    "policy": policy,
                    "entry_ts": trig.ts,
                    "entry_price": trig.price,
                    **outcome,
                }
            )

    events_df = pd.DataFrame(events)
    if events_df.empty:
        raise RuntimeError("No trigger events generated")
    summary = _summarize(events_df, total_candidates=len(ranked))
    examples = (
        events_df[events_df["ticker"].isin(["DELL", "MU", "AAOI"])]
        .groupby(["policy", "ticker"])
        .agg(
            trades=("ticker", "size"),
            avg_fwd_max_return=("fwd_max_return", "mean"),
            pct_gt_20=("fwd_max_return", lambda s: float((s >= 0.20).mean())),
            avg_drawdown=("fwd_max_drawdown", "mean"),
        )
        .reset_index()
        .sort_values(["ticker", "pct_gt_20", "avg_fwd_max_return"], ascending=[True, False, False])
    )

    events_df.to_csv(out_dir / "trigger_events.csv", index=False)
    summary.to_csv(out_dir / "trigger_summary.csv", index=False)
    examples.to_csv(out_dir / "example_ticker_trigger_summary.csv", index=False)
    (out_dir / "metadata.json").write_text(
        json.dumps(
            {
                "matrix": str(matrix_path),
                "predictions": str(preds_path),
                "prediction_column": pred_col,
                "top_n_per_4h_bar": int(top_n),
                "max_confirmation_hours": int(max_hours),
                "forward_horizon_1h_rth_bars": int(horizon_bars),
                "ranked_candidates": int(len(ranked)),
                "trigger_events": int(len(events_df)),
            },
            indent=2,
            default=str,
        )
    )

    print("Trigger summary")
    print(summary.to_string(index=False))
    print()
    print("DELL/MU/AAOI trigger summary")
    print(examples.to_string(index=False))
    print()
    print(f"Wrote {out_dir}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--preds", type=Path, default=DEFAULT_PREDS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--max-hours", type=int, default=4)
    parser.add_argument("--horizon-bars", type=int, default=70)
    args = parser.parse_args()
    run(args.matrix, args.preds, args.out, args.top_n, args.max_hours, args.horizon_bars)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
