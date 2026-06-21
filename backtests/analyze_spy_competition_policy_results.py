from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "backtests" / "20260619_spy_competition"
SPLIT = pd.Timestamp("2026-05-12T00:00:00Z")


def _load_events(name: str, filters: dict[str, object] | None = None) -> pd.DataFrame:
    frame = pd.read_csv(OUT / name)
    if filters:
        for column, value in filters.items():
            frame = frame[frame[column] == value]
    frame = frame.copy()
    frame["entry_ts"] = pd.to_datetime(frame["entry_ts"], utc=True)
    frame["exit_ts"] = pd.to_datetime(frame["exit_ts"], utc=True)
    return frame.sort_values("entry_ts").reset_index(drop=True)


def _event_summary(policy: str, split: str, frame: pd.DataFrame) -> dict[str, object]:
    option_return = pd.to_numeric(frame["return"], errors="coerce")
    direction = np.where(frame["side"].eq("long"), 1.0, -1.0)
    underlying_return = direction * (
        pd.to_numeric(frame["exit_spot"], errors="coerce")
        / pd.to_numeric(frame["entry_spot"], errors="coerce")
        - 1.0
    )
    cumulative = option_return.fillna(0.0).cumsum()
    drawdown = cumulative - cumulative.cummax()
    daily = option_return.groupby(frame["entry_ts"].dt.date).sum()
    return {
        "policy": policy,
        "split": split,
        "trades": int(len(frame)),
        "long_trades": int(frame["side"].eq("long").sum()),
        "short_trades": int(frame["side"].eq("short").sum()),
        "option_sum_return": float(option_return.sum()),
        "option_mean_return": float(option_return.mean()),
        "option_median_return": float(option_return.median()),
        "option_win_rate": float((option_return > 0.0).mean()),
        "option_max_sum_drawdown": float(drawdown.min()),
        "positive_day_rate": float((daily > 0.0).mean()),
        "underlying_directional_sum_return": float(underlying_return.sum()),
        "underlying_directional_mean_return": float(underlying_return.mean()),
        "underlying_directional_win_rate": float((underlying_return > 0.0).mean()),
    }


def _horizon_rows(
    frame: pd.DataFrame,
    *,
    model: str,
    side: str,
    threshold: float,
    split: str,
) -> list[dict[str, object]]:
    score = pd.to_numeric(frame[f"{model}_{side}"], errors="coerce")
    selected = frame[score >= threshold]
    sign = 1.0 if side == "long" else -1.0
    rows: list[dict[str, object]] = []
    for bars, minutes in ((1, 10), (3, 30), (6, 60), (12, 120)):
        values = sign * pd.to_numeric(selected[f"fwd_ret_{bars}"], errors="coerce")
        rows.append(
            {
                "model": model,
                "side": side,
                "split": split,
                "threshold": threshold,
                "horizon_minutes": minutes,
                "signals": int(values.notna().sum()),
                "mean_directional_return": float(values.mean()),
                "median_directional_return": float(values.median()),
                "directional_win_rate": float((values > 0.0).mean()),
            }
        )
    return rows


def main() -> None:
    policies = [
        (
            "active",
            "train",
            _load_events("policy_train_active_events.csv"),
        ),
        (
            "active",
            "holdout",
            _load_events("policy_holdout_active_events.csv"),
        ),
        (
            "candidate_long_next_open",
            "train",
            _load_events(
                "policy_train_long_exit_events.csv",
                {
                    "stop_loss_pct": 1.0,
                    "trail_arm_pct": 1.0,
                    "trail_giveback_pct": 0.35,
                    "time_decay_minutes": 60,
                },
            ),
        ),
        (
            "candidate_long_next_open",
            "holdout",
            _load_events(
                "policy_holdout_long_selected_events.csv",
                {"trail_giveback_pct": 0.35},
            ),
        ),
        (
            "candidate_short_next_open",
            "train",
            _load_events(
                "policy_train_short_geometry_events.csv",
                {
                    "short_threshold": 0.95,
                    "short_entry_end_hhmm": "13:30",
                    "trigger_mode": "next_open",
                    "strike_atr_mult": 0.0,
                },
            ),
        ),
        (
            "candidate_short_next_open",
            "holdout",
            _load_events("policy_holdout_short_selected_events.csv"),
        ),
        (
            "hybrid_candidate_long_active_short",
            "train",
            _load_events("policy_train_hybrid_events.csv"),
        ),
        (
            "hybrid_candidate_long_active_short",
            "holdout",
            _load_events("policy_holdout_hybrid_events.csv"),
        ),
    ]
    comparison = pd.DataFrame(
        [_event_summary(policy, split, events) for policy, split, events in policies]
    )
    comparison.to_csv(OUT / "policy_comparison.csv", index=False)

    reasons: list[pd.DataFrame] = []
    for policy, split, events in policies:
        grouped = (
            events.groupby(["side", "reason"], dropna=False)["return"]
            .agg(["count", "sum", "mean", "median"])
            .reset_index()
        )
        grouped.insert(0, "split", split)
        grouped.insert(0, "policy", policy)
        reasons.append(grouped)
    pd.concat(reasons, ignore_index=True).to_csv(
        OUT / "policy_reason_breakdown.csv", index=False
    )

    predictions = pd.read_parquet(OUT / "prediction_frame.parquet")
    prediction_splits = {
        "train": predictions[predictions.index < SPLIT],
        "holdout": predictions[predictions.index >= SPLIT],
    }
    horizons: list[dict[str, object]] = []
    for split, frame in prediction_splits.items():
        horizons.extend(
            _horizon_rows(
                frame, model="candidate", side="long", threshold=0.85, split=split
            )
        )
        horizons.extend(
            _horizon_rows(
                frame, model="candidate", side="short", threshold=0.95, split=split
            )
        )
        horizons.extend(
            _horizon_rows(
                frame, model="active", side="long", threshold=0.5, split=split
            )
        )
        horizons.extend(
            _horizon_rows(
                frame, model="active", side="short", threshold=0.5, split=split
            )
        )
    pd.DataFrame(horizons).to_csv(OUT / "policy_underlying_horizons.csv", index=False)

    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
