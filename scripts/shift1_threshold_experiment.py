from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _resolve_repo_root() -> Path:
    try:
        return Path(__file__).resolve().parents[1]
    except NameError:
        return Path.cwd()


REPO_ROOT = _resolve_repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    fbeta_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from Data.retrieve_data import normalize_ticker


def _load_split_indices(ticker: str, dataset_name: str, x_filename: str) -> dict[str, np.ndarray]:
    clean = normalize_ticker(ticker)
    split_dir = (
        REPO_ROOT
        / "Data"
        / "processed"
        / clean.lower()
        / "splits"
        / dataset_name
        / Path(x_filename).stem
    )
    return {
        "train": np.load(split_dir / "train_idx.npy"),
        "val": np.load(split_dir / "val_idx.npy"),
        "test": np.load(split_dir / "test_idx.npy"),
    }


def _dilate(mask_in: np.ndarray, window: int) -> np.ndarray:
    mask_bool = mask_in.astype(bool)
    out = mask_bool.copy()
    for shift in range(1, max(0, int(window)) + 1):
        out[shift:] |= mask_bool[:-shift]
        out[:-shift] |= mask_bool[shift:]
    return out


def _metrics(y_true: np.ndarray, probs: np.ndarray, threshold: float, event_window: int) -> dict[str, float]:
    finite = np.isfinite(probs)
    y = y_true[finite].astype(int)
    p = probs[finite].astype(float)
    pred = (p >= float(threshold)).astype(int)
    out = {
        "threshold": float(threshold),
        "n": float(y.size),
        "positives": float((y == 1).sum()),
        "predicted": float(pred.sum()),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "f0_5": float(fbeta_score(y, pred, beta=0.5, zero_division=0)),
        "f2": float(fbeta_score(y, pred, beta=2.0, zero_division=0)),
    }
    if len(np.unique(y)) > 1:
        out["auc"] = float(roc_auc_score(y, p))
        out["average_precision"] = float(average_precision_score(y, p))
    else:
        out["auc"] = float("nan")
        out["average_precision"] = float("nan")

    if event_window > 0:
        actual = y.astype(bool)
        predicted = pred.astype(bool)
        actual_dilated = _dilate(actual, event_window)
        pred_dilated = _dilate(predicted, event_window)
        pred_hits = int((predicted & actual_dilated).sum())
        actual_hits = int((actual & pred_dilated).sum())
        event_precision = pred_hits / max(int(predicted.sum()), 1)
        event_recall = actual_hits / max(int(actual.sum()), 1)
        event_f1 = (
            2.0 * event_precision * event_recall / (event_precision + event_recall)
            if event_precision + event_recall > 0
            else 0.0
        )
        out.update(
            {
                "event_precision": float(event_precision),
                "event_recall": float(event_recall),
                "event_f1": float(event_f1),
                "pred_event_hits": float(pred_hits),
                "actual_event_hits": float(actual_hits),
            }
        )
    return out


def _sweep_side(
    side: str,
    y: np.ndarray,
    probs: np.ndarray,
    train_val_idx: np.ndarray,
    test_idx: np.ndarray,
    thresholds: np.ndarray,
    event_window: int,
) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for split_name, idx in (("oof", train_val_idx), ("test", test_idx)):
        for threshold in thresholds:
            row = _metrics(y[idx], probs[idx], float(threshold), event_window)
            row["side"] = side
            row["split"] = split_name
            rows.append(row)
    return pd.DataFrame(rows)


def _best_rows(sweep: pd.DataFrame, min_predictions: int) -> pd.DataFrame:
    rows = []
    oof = sweep[(sweep["split"] == "oof") & (sweep["predicted"] >= min_predictions)].copy()
    test = sweep[sweep["split"] == "test"].copy()
    for side in ("long", "short"):
        side_oof = oof[oof["side"] == side]
        for objective in ("f0_5", "f1", "event_f1", "precision"):
            if side_oof.empty:
                continue
            best = side_oof.sort_values(
                [objective, "event_f1", "precision", "predicted"],
                ascending=[False, False, False, True],
            ).iloc[0]
            test_row = test[(test["side"] == side) & np.isclose(test["threshold"], best["threshold"])]
            merged = best.to_dict()
            merged["objective"] = objective
            if not test_row.empty:
                for key, value in test_row.iloc[0].to_dict().items():
                    if key not in {"side", "split", "threshold"}:
                        merged[f"test_{key}"] = value
            rows.append(merged)
    return pd.DataFrame(rows)


def _select_threshold(
    best: pd.DataFrame,
    sweep: pd.DataFrame,
    *,
    side: str,
    objective: str,
    min_threshold: float,
    min_predictions: int,
) -> float:
    candidates = sweep[
        (sweep["side"] == side)
        & (sweep["split"] == "oof")
        & (sweep["threshold"] >= float(min_threshold))
        & (sweep["predicted"] >= int(min_predictions))
    ].copy()
    if not candidates.empty and objective in candidates.columns:
        row = candidates.sort_values(
            [objective, "event_f1", "event_precision", "predicted"],
            ascending=[False, False, False, True],
        ).iloc[0]
        return float(row["threshold"])

    side_best = best[(best["side"] == side) & (best["objective"] == objective)]
    if side_best.empty:
        side_best = best[(best["side"] == side) & (best["objective"] == "event_f1")]
    if side_best.empty:
        side_best = best[(best["side"] == side) & (best["objective"] == "f1")]
    return float(side_best.iloc[0]["threshold"])


def _plot_predictions(
    out_path: Path,
    index: pd.DatetimeIndex,
    y_long: np.ndarray,
    y_short: np.ndarray,
    p_long: np.ndarray,
    p_short: np.ndarray,
    test_idx: np.ndarray,
    long_threshold: float,
    short_threshold: float,
    tail_bars: int,
) -> None:
    idx = np.sort(test_idx)[-max(50, int(tail_bars)) :]
    x = index[idx]
    fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
    configs = [
        ("Long", axes[0], y_long[idx], p_long[idx], long_threshold, "#2563eb"),
        ("Short", axes[1], y_short[idx], p_short[idx], short_threshold, "#dc2626"),
    ]
    for label, ax, y, p, threshold, color in configs:
        ax.plot(x, p, color=color, linewidth=1.25, label=f"{label} probability")
        ax.axhline(threshold, color=color, linestyle="--", linewidth=1.0, label=f"threshold {threshold:.2f}")
        events = y.astype(bool)
        signals = np.isfinite(p) & (p >= threshold)
        ax.scatter(x[events], np.full(events.sum(), 1.03), marker="v", s=28, color="black", label="actual spike")
        ax.scatter(x[signals], p[signals], marker="o", s=18, facecolors="none", edgecolors=color, label="signal")
        ax.set_ylim(-0.03, 1.08)
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.2)
        ax.legend(loc="upper left", ncols=4, fontsize=8)
    axes[-1].set_xlabel("Test timestamp")
    fig.suptitle("SPY 10min_shift1 probabilities vs shifted spike labels")
    fig.autofmt_xdate()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Threshold sweep and probability plot for SPY shift1 model.")
    parser.add_argument("--ticker", default="SPY")
    parser.add_argument("--dataset-name", default="10min_shift1")
    parser.add_argument("--x-filename", default="X_10min_shift1_tree.parquet")
    parser.add_argument("--label-mode", choices=("swing",), default="swing")
    parser.add_argument("--threshold-start", type=float, default=0.05)
    parser.add_argument("--threshold-end", type=float, default=0.95)
    parser.add_argument("--threshold-step", type=float, default=0.01)
    parser.add_argument("--event-window", type=int, default=1)
    parser.add_argument("--min-oof-predictions", type=int, default=50)
    parser.add_argument("--selection-objective", default="event_f1")
    parser.add_argument("--selection-min-threshold", type=float, default=0.50)
    parser.add_argument("--selection-min-oof-predictions", type=int, default=500)
    parser.add_argument("--current-long-threshold", type=float, default=0.35)
    parser.add_argument("--current-short-threshold", type=float, default=0.65)
    parser.add_argument("--tail-bars", type=int, default=900)
    parser.add_argument(
        "--out-dir",
        default="Data/models/ga_xgboost/10min_shift1/analysis/threshold_experiment",
    )
    args = parser.parse_args()

    clean = normalize_ticker(args.ticker)
    dataset_dir = REPO_ROOT / "Data" / "processed" / clean.lower() / "datasets" / args.dataset_name
    y_df = pd.read_csv(dataset_dir / "y.csv")
    y_df["date"] = pd.to_datetime(y_df["date"], errors="raise", utc=True).dt.tz_convert(
        "America/New_York"
    )
    y_df = y_df.set_index("date")
    y_long = pd.to_numeric(y_df["long_swing_label"], errors="coerce").fillna(0).to_numpy(dtype=int)
    y_short = pd.to_numeric(y_df["short_swing_label"], errors="coerce").fillna(0).to_numpy(dtype=int)

    model_dir = REPO_ROOT / "Data" / "models" / "ga_xgboost" / args.dataset_name
    p_long_df = pd.read_parquet(model_dir / "long" / "swing" / "p_long_probs.parquet")
    p_short_df = pd.read_parquet(model_dir / "short" / "swing" / "p_short_probs.parquet")
    p_long = pd.to_numeric(p_long_df["p_long_oof_train"].combine_first(p_long_df["p_long_test"]), errors="coerce").to_numpy(dtype=float)
    p_short = pd.to_numeric(p_short_df["p_short_oof_train"].combine_first(p_short_df["p_short_test"]), errors="coerce").to_numpy(dtype=float)
    if len(y_df) != len(p_long_df) or len(y_df) != len(p_short_df):
        raise ValueError("Label and probability row counts do not match.")

    splits = _load_split_indices(args.ticker, args.dataset_name, args.x_filename)
    train_val_idx = np.sort(np.concatenate([np.sort(splits["train"]), np.sort(splits["val"])]))
    test_idx = np.sort(splits["test"])
    thresholds = np.round(
        np.arange(float(args.threshold_start), float(args.threshold_end) + 1e-9, float(args.threshold_step)),
        6,
    )

    sweep = pd.concat(
        [
            _sweep_side("long", y_long, p_long, train_val_idx, test_idx, thresholds, args.event_window),
            _sweep_side("short", y_short, p_short, train_val_idx, test_idx, thresholds, args.event_window),
        ],
        ignore_index=True,
    )
    out_dir = REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    sweep_path = out_dir / "shift1_threshold_sweep.csv"
    sweep.to_csv(sweep_path, index=False)

    best = _best_rows(sweep, int(args.min_oof_predictions))
    current_rows = []
    for side, threshold in (
        ("long", float(args.current_long_threshold)),
        ("short", float(args.current_short_threshold)),
    ):
        side_sweep = sweep[(sweep["side"] == side) & np.isclose(sweep["threshold"], threshold)]
        if not side_sweep.empty:
            for _, row in side_sweep.iterrows():
                current_rows.append(row.to_dict() | {"objective": "current"})
    best = pd.concat([best, pd.DataFrame(current_rows)], ignore_index=True)
    best_path = out_dir / "shift1_threshold_best_summary.csv"
    best.to_csv(best_path, index=False)

    selected = {
        side: _select_threshold(
            best,
            sweep,
            side=side,
            objective=str(args.selection_objective),
            min_threshold=float(args.selection_min_threshold),
            min_predictions=int(args.selection_min_oof_predictions),
        )
        for side in ("long", "short")
    }

    plot_path = out_dir / "shift1_prediction_threshold_plot.png"
    _plot_predictions(
        plot_path,
        pd.DatetimeIndex(y_df.index),
        y_long,
        y_short,
        p_long,
        p_short,
        test_idx,
        selected["long"],
        selected["short"],
        int(args.tail_bars),
    )

    payload = {
        "dataset_name": args.dataset_name,
        "probability_source": {
            "long": str(model_dir / "long" / "swing" / "p_long_probs.parquet"),
            "short": str(model_dir / "short" / "swing" / "p_short_probs.parquet"),
        },
        "label_source": str(dataset_dir / "y.csv"),
        "selected_thresholds": selected,
        "selection": {
            "objective": str(args.selection_objective),
            "min_threshold": float(args.selection_min_threshold),
            "min_oof_predictions": int(args.selection_min_oof_predictions),
        },
        "outputs": {
            "sweep": str(sweep_path),
            "best_summary": str(best_path),
            "plot": str(plot_path),
        },
    }
    summary_path = out_dir / "shift1_threshold_experiment_summary.json"
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    show_cols = [
        "side",
        "objective",
        "split",
        "threshold",
        "precision",
        "recall",
        "f1",
        "event_precision",
        "event_recall",
        "event_f1",
        "predicted",
        "test_precision",
        "test_recall",
        "test_f1",
        "test_event_precision",
        "test_event_recall",
        "test_event_f1",
        "test_predicted",
    ]
    print(best[[col for col in show_cols if col in best.columns]].to_string(index=False))


if __name__ == "__main__":
    main()
