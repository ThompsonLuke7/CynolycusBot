import argparse
import sys
from pathlib import Path

import pandas as pd

def get_default_plot_path(ticker: str, data_dir: Path) -> Path:
    return _get_default_plot_path(ticker, data_dir, "continuation")


def _resolve_repo_root() -> Path:
    try:
        return Path(__file__).resolve().parents[2]
    except NameError:
        return Path.cwd()


REPO_ROOT = _resolve_repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Data.plots.plots import get_default_plot_path as _get_default_plot_path
from Data.plots.plots import plot_continuation_strength
from Data.retrieve_data import normalize_ticker


def _resolve_plot_frame(ticker: str, dataset_name: str) -> Path:
    slug = normalize_ticker(ticker).lower()
    return (
        _resolve_repo_root()
        / "Data"
        / "processed"
        / slug
        / "datasets"
        / dataset_name
        / "plot_frame.parquet"
    )


def _resolve_labels_path(ticker: str, dataset_name: str) -> Path:
    slug = normalize_ticker(ticker).lower()
    return (
        _resolve_repo_root()
        / "Data"
        / "processed"
        / slug
        / "datasets"
        / dataset_name
        / "y.parquet"
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Plot continuation labels for a dataset plot frame."
    )
    ap.add_argument("--ticker", type=str, default="$SPY")
    ap.add_argument("--dataset", type=str, default="15min")
    ap.add_argument("--save_path", type=str, default=None)
    ap.add_argument("--long_col", type=str, default="cont_strength_long")
    ap.add_argument("--short_col", type=str, default="cont_strength_short")
    ap.add_argument("--leg_col", type=str, default="atr_leg_label")
    ap.add_argument("--momentum_col", type=str, default="trend_phase_m")
    ap.add_argument("--accel_col", type=str, default="trend_phase_a")
    ap.add_argument("--exit_long_col", type=str, default="trend_phase_exit_long")
    ap.add_argument("--exit_short_col", type=str, default="trend_phase_exit_short")
    ap.add_argument("--candidate_long_col", type=str, default="tb_cont_candidate_long")
    ap.add_argument("--candidate_short_col", type=str, default="tb_cont_candidate_short")
    ap.add_argument("--label_col", type=str, default="tb_cont_label")
    ap.add_argument("--min_cont_strength", type=float, default=0.35)
    ap.add_argument("--no_accel_filter", action="store_true")
    ap.add_argument("--leg_reset_grace_bars", type=int, default=0)
    ap.add_argument("--leg_reset_counter_atr", type=float, default=0.0)
    ap.add_argument("--leg_reset_requires_both", action="store_true")
    ap.add_argument("--tail", type=int, default=200)
    ap.add_argument("--no_y_labels", action="store_true")
    args = ap.parse_args()

    plot_path = _resolve_plot_frame(args.ticker, args.dataset)
    if not plot_path.exists():
        raise FileNotFoundError(f"Missing plot_frame.parquet at {plot_path}")

    df = pd.read_parquet(plot_path)
    if not args.no_y_labels:
        y_path = _resolve_labels_path(args.ticker, args.dataset)
        if not y_path.exists():
            raise FileNotFoundError(f"Missing y.parquet at {y_path}")
        y_df = pd.read_parquet(y_path)
        if df.index.equals(y_df.index):
            df = df.join(y_df, how="left")
        else:
            common = df.index.intersection(y_df.index)
            if not common.empty:
                df = df.loc[common].join(y_df.loc[common], how="left")
            else:
                min_len = min(len(df), len(y_df))
                df = df.iloc[:min_len].copy()
                y_df = y_df.iloc[:min_len].copy()
                y_df.index = df.index
                df = df.join(y_df, how="left")
    slug = normalize_ticker(args.ticker).lower()
    data_dir = _resolve_repo_root() / "Data" / "processed" / slug
    save_path = args.save_path or get_default_plot_path(args.ticker, data_dir)

    plot_continuation_strength(
        df,
        strength_long_col=args.long_col,
        strength_short_col=args.short_col,
        leg_label_col=args.leg_col,
        momentum_col=args.momentum_col,
        accel_col=args.accel_col,
        exit_long_col=args.exit_long_col,
        exit_short_col=args.exit_short_col,
        candidate_long_col=args.candidate_long_col,
        candidate_short_col=args.candidate_short_col,
        label_col=args.label_col,
        min_cont_strength=args.min_cont_strength,
        use_accel_filter=not bool(args.no_accel_filter),
        leg_reset_grace_bars=args.leg_reset_grace_bars,
        leg_reset_counter_atr=args.leg_reset_counter_atr,
        leg_reset_requires_both=bool(args.leg_reset_requires_both),
        tail=args.tail,
        save_path=save_path,
    )


if __name__ == "__main__":
    main()
