import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from xgboost import XGBClassifier


def _resolve_repo_root() -> Path:
    try:
        return Path(__file__).resolve().parents[2]
    except NameError:
        return Path.cwd()


REPO_ROOT = _resolve_repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from Data.load_data import (  # noqa: E402
    get_ticker_plots_dir,
    get_ticker_processed_base_dir,
    load_split_indices,
)
from Data.retrieve_data import normalize_ticker  # noqa: E402


def _load_feature_frame(
    ticker: str,
    dataset_name: str,
) -> tuple[pd.DataFrame, list[str], Path]:
    clean = normalize_ticker(ticker)
    processed_dir = get_ticker_processed_base_dir(clean)
    dataset_dir = processed_dir / "datasets" / dataset_name
    X_path = dataset_dir / "X.parquet"
    if not X_path.exists():
        raise FileNotFoundError(f"Missing {X_path}")

    X_df = pd.read_parquet(X_path)
    features_path = dataset_dir / "features.txt"
    if features_path.exists():
        feature_cols = [
            line.strip() for line in features_path.read_text().splitlines() if line.strip()
        ]
        missing = [c for c in feature_cols if c not in X_df.columns]
        if missing:
            raise KeyError(f"Missing feature columns in X.parquet: {', '.join(missing)}")
        X_df = X_df[feature_cols]
    else:
        feature_cols = list(X_df.columns)

    return X_df, feature_cols, dataset_dir


def _load_model_and_mask(model_dir: Path) -> tuple[XGBClassifier, np.ndarray]:
    model_path = model_dir / "xgb_model.json"
    mask_path = model_dir / "best_mask.npy"
    if not model_path.exists() or not mask_path.exists():
        raise FileNotFoundError(f"Missing model artifacts in {model_dir}")

    model = XGBClassifier()
    model.load_model(model_path)
    mask = np.load(mask_path).astype(bool)
    return model, mask


def _select_features(X: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if mask.ndim != 1:
        raise ValueError("best_mask.npy must be a 1D array.")
    if X.shape[1] != mask.size:
        raise ValueError(
            f"Feature count mismatch: X has {X.shape[1]} cols, mask has {mask.size}."
        )
    return X[:, mask]


def plot_model_inference(
    X_df: pd.DataFrame,
    long_probs: np.ndarray | None,
    short_probs: np.ndarray | None,
    *,
    threshold: float = 0.6,
    title: str | None = None,
    save_path: str | None = None,
) -> None:
    pos = np.arange(len(X_df))
    has_ohlc = all(c in X_df.columns for c in ("open", "high", "low", "close"))
    close_y = X_df["close"].to_numpy() if "close" in X_df.columns else None

    fig, (ax_price, ax_prob) = plt.subplots(
        2,
        1,
        figsize=(18, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1]},
    )

    if has_ohlc:
        open_y = X_df["open"].to_numpy()
        high_y = X_df["high"].to_numpy()
        low_y = X_df["low"].to_numpy()
        wick_color = "#4a4a4a"
        up_color = "#1976D2"
        down_color = "#E53935"
        up = close_y >= open_y
        down = ~up

        ax_price.vlines(pos, low_y, high_y, color=wick_color, linewidth=1.0, zorder=1)
        ax_price.bar(
            pos[up],
            close_y[up] - open_y[up],
            width=0.8,
            bottom=open_y[up],
            color=up_color,
            edgecolor="none",
            zorder=1.2,
        )
        ax_price.bar(
            pos[down],
            close_y[down] - open_y[down],
            width=0.8,
            bottom=open_y[down],
            color=down_color,
            edgecolor="none",
            zorder=1.2,
        )
        marker_offset = np.nanmedian(high_y - low_y)
        if not np.isfinite(marker_offset) or marker_offset <= 0:
            marker_offset = np.nanmax(high_y) * 0.002
        long_y = low_y - marker_offset * 0.6
        short_y = high_y + marker_offset * 0.6
    elif close_y is not None:
        ax_price.plot(pos, close_y, color="#1f77b4", linewidth=1.6, label="Close")
        marker_offset = np.nanmedian(np.abs(np.diff(close_y)))
        if not np.isfinite(marker_offset) or marker_offset <= 0:
            marker_offset = np.nanmax(close_y) * 0.002
        long_y = close_y - marker_offset * 2
        short_y = close_y + marker_offset * 2
    else:
        raise ValueError("X.parquet must include close (or open/high/low/close) to plot.")

    if long_probs is not None:
        long_mask = long_probs >= threshold
        if long_mask.any():
            ax_price.scatter(
                pos[long_mask],
                long_y[long_mask],
                color="#1565C0",
                marker="^",
                s=60,
                label=f"LONG prob ≥ {threshold:.2f}",
                zorder=2,
            )
    if short_probs is not None:
        short_mask = short_probs >= threshold
        if short_mask.any():
            ax_price.scatter(
                pos[short_mask],
                short_y[short_mask],
                color="#FB8C00",
                marker="v",
                s=60,
                label=f"SHORT prob ≥ {threshold:.2f}",
                zorder=2,
            )

    ax_price.set_ylabel("Price")
    ax_price.legend(loc="upper left")
    ax_price.set_title(title or "Model Inference (Window)")

    if long_probs is not None:
        ax_prob.plot(long_probs, label="LONG P(class=1)", color="#1565C0", linewidth=1.5)
    if short_probs is not None:
        ax_prob.plot(
            short_probs, label="SHORT P(class=1)", color="#FB8C00", linewidth=1.5
        )
    ax_prob.axhline(threshold, color="#1f77b4", linestyle="--", label="Threshold")
    ax_prob.set_ylim(0, 1.02)
    ax_prob.set_title("Model Probabilities (Window)")
    ax_prob.legend(loc="upper right")
    ax_prob.set_xlabel("Bar")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=200)
        print(f"Saved plot to {save_path}")
    plt.show()


def get_default_plot_path(ticker: str, model_name: str) -> Path:
    slug = normalize_ticker(ticker).lower()
    plots_dir = get_ticker_plots_dir(slug)
    filename = f"{slug}_{model_name}_inference.png"
    return plots_dir / filename


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot model inference signals over price bars."
    )
    parser.add_argument("--ticker", default="$SPY")
    parser.add_argument("--dataset", default="15min")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--split", choices=["all", "train", "val", "test"], default="test")
    parser.add_argument("--tail", type=int, default=None)
    parser.add_argument("--save", default=None)
    args = parser.parse_args()

    X_df, feature_cols, _ = _load_feature_frame(args.ticker, args.dataset)

    if args.split != "all":
        splits = load_split_indices(args.ticker, args.dataset)
        X_df = X_df.iloc[splits[args.split]]

    if args.tail:
        X_df = X_df.tail(args.tail)

    X = X_df.to_numpy(dtype=np.float32)

    repo_root = _resolve_repo_root()
    model_root = repo_root / "Data" / "models" / args.model_name

    long_probs = None
    short_probs = None

    long_dir = model_root / "long"
    if long_dir.exists():
        long_model, long_mask = _load_model_and_mask(long_dir)
        long_probs = long_model.predict_proba(_select_features(X, long_mask))[:, 1]

    short_dir = model_root / "short"
    if short_dir.exists():
        short_model, short_mask = _load_model_and_mask(short_dir)
        short_probs = short_model.predict_proba(_select_features(X, short_mask))[:, 1]

    if long_probs is None and short_probs is None:
        raise FileNotFoundError(f"No model artifacts found under {model_root}")

    title = f"{normalize_ticker(args.ticker)} | {args.model_name} | split={args.split}"
    save_path = args.save or str(get_default_plot_path(args.ticker, args.model_name))
    plot_model_inference(
        X_df,
        long_probs,
        short_probs,
        threshold=args.threshold,
        title=title,
        save_path=save_path,
    )


if __name__ == "__main__":
    main()
