# run_train.py
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_MODELS_ROOT = Path(__file__).resolve().parents[1]
if str(_MODELS_ROOT) not in sys.path:
    sys.path.insert(0, str(_MODELS_ROOT))

from iTransformer.itransformer_train import build_arg_parser, run_training

from Data.plots.plots import _load_plot_frame, plot_model_inference
from Data.retrieve_data import normalize_ticker


def _plot_regression_inference(
    plot_df: pd.DataFrame,
    preds: np.ndarray,
    actual: np.ndarray,
    *,
    title: str,
    save_path: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Plot skipped: {exc}")
        return

    preds = np.asarray(preds).reshape(-1)
    actual = np.asarray(actual).reshape(-1)
    pos = np.arange(len(plot_df))

    has_ohlc = all(c in plot_df.columns for c in ("open", "high", "low", "close"))
    close_y = plot_df["close"].to_numpy() if "close" in plot_df.columns else None

    fig, (ax_price, ax_pred) = plt.subplots(
        2,
        1,
        figsize=(18, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1]},
    )

    if has_ohlc:
        open_y = plot_df["open"].to_numpy()
        high_y = plot_df["high"].to_numpy()
        low_y = plot_df["low"].to_numpy()
        valid_mask = (
            np.isfinite(open_y)
            & np.isfinite(high_y)
            & np.isfinite(low_y)
            & np.isfinite(close_y)
        )
        wick_color = "#4a4a4a"
        up_color = "#1976D2"
        down_color = "#E53935"
        up = close_y >= open_y
        up_mask = up & valid_mask
        down_mask = (~up) & valid_mask
        ax_price.vlines(
            pos[valid_mask],
            low_y[valid_mask],
            high_y[valid_mask],
            color=wick_color,
            linewidth=1.0,
            zorder=1,
        )
        ax_price.bar(
            pos[up_mask],
            close_y[up_mask] - open_y[up_mask],
            width=0.8,
            bottom=open_y[up_mask],
            color=up_color,
            edgecolor="none",
            zorder=1.2,
        )
        ax_price.bar(
            pos[down_mask],
            close_y[down_mask] - open_y[down_mask],
            width=0.8,
            bottom=open_y[down_mask],
            color=down_color,
            edgecolor="none",
            zorder=1.2,
        )
    elif close_y is not None:
        ax_price.plot(pos, close_y, color="#1f77b4", linewidth=1.6, label="Close")
    else:
        print("Plot skipped: missing close/ohlc in plot frame.")
        plt.close(fig)
        return

    ax_price.set_ylabel("Price")
    ax_price.set_title(title)

    ax_pred.plot(pos, actual, color="#1f77b4", linewidth=1.6, label="Actual")
    ax_pred.plot(pos, preds, color="#FB8C00", linewidth=1.4, alpha=0.85, label="Pred")
    ax_pred.set_ylabel("Target")
    ax_pred.legend(loc="upper right")

    plot_index = plot_df.index if isinstance(plot_df.index, pd.DatetimeIndex) else None
    if plot_index is not None:
        dates = pd.Series(plot_index)
        day_start = dates.dt.normalize().ne(dates.dt.normalize().shift())
        tick_positions = pos[day_start.to_numpy()]
        tick_labels = dates[day_start].dt.strftime("%Y-%m-%d").to_list()
        if len(tick_positions) > 25:
            step = int(np.ceil(len(tick_positions) / 25))
            tick_positions = tick_positions[::step]
            tick_labels = tick_labels[::step]
        ax_pred.set_xticks(tick_positions)
        ax_pred.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=9)
    ax_pred.set_xlabel("Session")

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"Saved plot to {save_path}")


def _plot_side(
    *,
    side: str,
    task: str,
    result: dict,
    plot_df: pd.DataFrame,
    ticker: str,
    label_mode: str,
) -> None:
    test_pred = result.get("test_pred")
    test_true = result.get("test_true")
    if test_pred is None or test_true is None:
        print(f"Skip plotting for {side}: missing predictions.")
        return

    model_name = f"itransformer_{label_mode}_{side}"
    slug = normalize_ticker(ticker).lower()
    save_path = (
        Path("Data")
        / "processed"
        / slug
        / "plots"
        / f"{slug}_{model_name}_inference.png"
    )

    if task == "binary":
        long_probs = test_pred if side in {"long", "up"} else None
        short_probs = test_pred if side in {"short", "down"} else None
        long_actual = test_true if side in {"long", "up"} else None
        short_actual = test_true if side in {"short", "down"} else None
        plot_model_inference(
            plot_df,
            long_probs,
            short_probs,
            long_actual=long_actual,
            short_actual=short_actual,
            long_label_name="LONG",
            short_label_name="SHORT",
            title=f"{ticker} | iTransformer {side.upper()}",
            save_path=save_path,
        )
    else:
        _plot_regression_inference(
            plot_df,
            test_pred,
            test_true,
            title=f"{ticker} | iTransformer {side.upper()}",
            save_path=save_path,
        )


def _infer_task(args) -> str:
    if args.x_path is not None or args.y_path is not None:
        return args.task
    mode = (args.label_mode or "").strip().lower()
    if mode in {"mfe", "mae", "mfe_mae", "exhaustion"}:
        return "regression"
    return "binary"


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    results = run_training(args, return_predictions=True)
    if not results:
        print("No prediction outputs returned; nothing to save.")
        return

    x_path = None
    if args.x_path is None:
        x_filename = args.x_filename or f"X_{args.dataset_name}_tree.parquet"
        dataset_dir = (
            Path("Data")
            / "processed"
            / (args.ticker or "$SPY").replace("$", "").lower()
            / "datasets"
            / args.dataset_name
        )
        candidate = dataset_dir / x_filename
        if candidate.exists():
            x_path = candidate
    else:
        x_path = Path(args.x_path)

    task = _infer_task(args)

    for side, result in results.items():
        test_indices = result.get("test_indices")
        plot_df = _load_plot_frame(args.ticker, test_indices, x_path=x_path)
        if plot_df is None:
            print(f"Skip plotting for {side}: no plot frame found.")
            continue
        _plot_side(
            side=side,
            task=task,
            result=result,
            plot_df=plot_df,
            ticker=args.ticker,
            label_mode=args.label_mode,
        )


if __name__ == "__main__":
    main()
