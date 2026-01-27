from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from Data.load_data import get_ticker_processed_base_dir
from Data.retrieve_data import normalize_ticker
from Features.feature_matrix import _collect_label_columns
from Features.feature_sets.custom_indicators import add_fractal_pivots
from Features.label_generations import add_all_labels


def _resolve_plot_frame_path(*, ticker: str, dataset_name: str) -> Path:
    clean = normalize_ticker(ticker)
    dataset_dir = get_ticker_processed_base_dir(clean) / "datasets" / dataset_name
    return dataset_dir / "plot_frame.parquet"


def _load_plot_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing plot_frame.parquet at {path}")
    df = pd.read_parquet(path)
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("plot_frame.parquet must have a DatetimeIndex")
    if not df.index.is_monotonic_increasing:
        df = df.sort_index()
    return df


def generate_labels(
    *,
    ticker: str,
    dataset_name: str,
    plot_frame_path: Path | None = None,
) -> Path:
    plot_path = plot_frame_path or _resolve_plot_frame_path(
        ticker=ticker, dataset_name=dataset_name
    )
    df = _load_plot_frame(plot_path)

    df = add_fractal_pivots(df)
    df = add_all_labels(df)

    label_cols = _collect_label_columns(df)
    if not label_cols:
        raise ValueError("No label columns were generated.")

    labels_df = df[label_cols].copy()

    clean = normalize_ticker(ticker)
    dataset_dir = get_ticker_processed_base_dir(clean) / "datasets" / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    y_parquet = dataset_dir / "y.parquet"
    y_csv = dataset_dir / "y.csv"
    labels_df.to_parquet(y_parquet, index=False)
    labels_df.to_csv(y_csv, index=True)
    return y_parquet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recompute labels from plot_frame.parquet and save y.parquet/y.csv."
    )
    parser.add_argument("--ticker", type=str, default="$SPY")
    parser.add_argument("--dataset", type=str, default="15min")
    parser.add_argument(
        "--plot-frame",
        type=str,
        default=None,
        help="Override plot_frame.parquet path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot_path = Path(args.plot_frame) if args.plot_frame else None
    out_path = generate_labels(
        ticker=args.ticker,
        dataset_name=args.dataset,
        plot_frame_path=plot_path,
    )
    print(f"Wrote {out_path} and y.csv")


if __name__ == "__main__":
    main()
