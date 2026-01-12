import argparse
import datetime as dt

from API.Alpaca_API.fetch_intraday import fetch_intraday
from Data.load_data import (
    ensure_ticker_dirs,
    get_ticker_data_dir,
    get_ticker_processed_base_dir,
    load_ticker_parquet,
    resolve_intraday_parquet_path,
)
from Data.plots.all_labels_plot import get_default_plot_path as get_all_labels_plot_path
from Data.plots.all_labels_plot import plot_all_labels
from Data.plots.atr_swing_plot import get_default_plot_path as get_atr_swing_plot_path
from Data.plots.atr_swing_plot import plot_atr_swing_signals
from Data.plots.continuation_plot import (
    get_default_plot_path as get_continuation_plot_path,
)
from Data.plots.continuation_plot import plot_continuation_signals
from Data.plots.leg_segmentation_plot import (
    get_default_plot_path as get_leg_plot_path,
)
from Data.plots.leg_segmentation_plot import plot_leg_segmentation_signals
from Data.plots.swing_state_machine_plot import (
    get_default_plot_path as get_swing_state_plot_path,
)
from Data.plots.swing_state_machine_plot import plot_swing_state_machine_signals
from Features import data_pipeline
from Features.custom_indicators import add_fractal_pivots
from Features.label_generations import (
    add_all_labels,
    add_atr_continuation_entry_labels,
    add_atr_leg_segmentation_labels,
    add_atr_pivot_swing_labels,
    add_pivot_swing_state_machine,
)
from Features.multi_timeframe_features import ensure_time_index, resample_ohlcv
from Features.training_matrix import build_training_matrix, clean_training_matrix


def parse_args():
    parser = argparse.ArgumentParser(description="Run feature engineering pipeline.")
    parser.add_argument(
        "--ticker",
        default="$SPY",
        help='Ticker symbol to fetch and process (default: "$SPY").',
    )
    parser.add_argument(
        "--use-cached",
        dest="use_cached",
        action="store_true",
        default=True,
        help="Use cached features if available (default: True).",
    )
    parser.add_argument(
        "--no-use-cached",
        dest="use_cached",
        action="store_false",
        help="Force recompute of features/labels and overwrite cache.",
    )
    parser.add_argument(
        "--save-processed",
        dest="save_processed",
        action="store_true",
        default=True,
        help="Persist processed features/labels to parquet (default: True).",
    )
    parser.add_argument(
        "--no-save-processed",
        dest="save_processed",
        action="store_false",
        help="Skip writing processed outputs.",
    )
    parser.add_argument(
        "--save-plot",
        dest="save_plot_path",
        default=None,
        help="Path to save the ATR swing plot PNG (default: Data/plots/atr_swing_plot.png).",
    )
    parser.add_argument(
        "--timeframe",
        default="1Hour",
        help='Alpaca timeframe (e.g., "1Hour", "1Day"). Defaults to "1Hour".',
    )
    parser.add_argument(
        "--start",
        default=None,
        help="Fetch start time (ISO string). Defaults to now - 200 days.",
    )
    parser.add_argument(
        "--end",
        default=None,
        help="Fetch end time (ISO string). Defaults to now.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10000,
        help="Max bars to fetch from Alpaca (default: 10000).",
    )
    parser.add_argument(
        "--adjustment",
        default="raw",
        help='Alpaca adjustment mode: "raw", "split", or "all".',
    )
    parser.add_argument(
        "--refresh-data",
        dest="refresh_data",
        action="store_true",
        default=True,
        help="Fetch latest data from Alpaca before running (default: True).",
    )
    parser.add_argument(
        "--no-refresh-data",
        dest="refresh_data",
        action="store_false",
        help="Skip data fetch if existing raw files are present.",
    )
    parser.add_argument(
        "--label-mode",
        default="leg",
        choices=["leg", "swing"],
        help='Label mode for feature engineering and splits (default: "leg").',
    )
    parser.add_argument(
        "--train-frac",
        type=float,
        default=0.92,
        help="Train split fraction (default: 0.7).",
    )
    parser.add_argument(
        "--val-frac",
        type=float,
        default=0.00,
        help="Validation split fraction (default: 0.15).",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Only plot labels (no feature engineering or data pipeline).",
    )
    parser.add_argument(
        "--plot-timeframe",
        default="15T",
        help='Plot timeframe (e.g., "15T", "15m", "1H"). Defaults to "15T".',
    )
    parser.add_argument(
        "--plot-type",
        default="atr_swing",
        choices=[
            "atr_swing",
            "all_labels",
            "leg",
            "continuation",
            "swing_state_machine",
        ],
        help="Which plot to render (default: atr_swing).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    def _normalize_plot_timeframe(timeframe: str) -> str:
        tf = timeframe.strip().lower()
        if tf.endswith("min"):
            minutes = int(tf.replace("min", "") or "1")
            return f"{minutes}T"
        if tf.endswith("m"):
            minutes = int(tf.replace("m", "") or "1")
            return f"{minutes}T"
        if tf.endswith("hour"):
            hours = int(tf.replace("hour", "") or "1")
            return f"{hours}H"
        if tf.endswith("h"):
            hours = int(tf.replace("h", "") or "1")
            return f"{hours}H"
        if tf.endswith("day"):
            days = int(tf.replace("day", "") or "1")
            return f"{days}D"
        if tf.endswith("d"):
            days = int(tf.replace("d", "") or "1")
            return f"{days}D"
        return timeframe

    def _dataset_name_from_label_timeframe(label_timeframe: str) -> str:
        tf = _normalize_plot_timeframe(label_timeframe).lower()
        if tf.endswith("t"):
            return f"{tf[:-1]}min"
        if tf.endswith("h"):
            return f"{tf[:-1]}h"
        if tf.endswith("d"):
            return f"{tf[:-1]}d"
        return tf

    data_dir = get_ticker_data_dir(args.ticker)
    ensure_ticker_dirs(args.ticker)

    raw_path = resolve_intraday_parquet_path(args.ticker)
    raw_missing = not raw_path.exists()
    refresh_data = args.refresh_data or raw_missing
    use_cached = args.use_cached
    if refresh_data:
        if raw_missing and not args.refresh_data:
            print("Raw data missing; fetching intraday data.")
        if use_cached:
            print("Refresh requested; forcing use_cached=False.")
        use_cached = False
        now_utc = dt.datetime.now(dt.timezone.utc)
        default_start = now_utc - dt.timedelta(days=100)
        start = args.start or default_start
        end = args.end
        fetch_intraday(
            ticker=args.ticker,
            start=start,
            end=end,
            timeframe=args.timeframe,
            limit=args.limit,
            adjustment=args.adjustment,
        )

    if args.plot_only:
        df = load_ticker_parquet(args.ticker)
        df = ensure_time_index(df)
        rule = _normalize_plot_timeframe(args.plot_timeframe)
        if rule != "1T":
            df = resample_ohlcv(df, rule)

        plot_save_path = args.save_plot_path
        if plot_save_path is None:
            if args.plot_type == "atr_swing":
                plot_save_path = get_atr_swing_plot_path(args.ticker, data_dir)
            elif args.plot_type == "leg":
                plot_save_path = get_leg_plot_path(args.ticker, data_dir)
            elif args.plot_type == "continuation":
                plot_save_path = get_continuation_plot_path(args.ticker, data_dir)
            elif args.plot_type == "swing_state_machine":
                plot_save_path = get_swing_state_plot_path(args.ticker, data_dir)
            elif args.plot_type == "all_labels":
                plot_save_path = get_all_labels_plot_path(args.ticker, data_dir)

        if args.plot_type == "atr_swing":
            df = add_fractal_pivots(df)
            df = add_atr_pivot_swing_labels(df)
            plot_atr_swing_signals(df, save_path=str(plot_save_path))
        elif args.plot_type == "leg":
            df = add_atr_leg_segmentation_labels(df)
            plot_leg_segmentation_signals(df, save_path=str(plot_save_path))
        elif args.plot_type == "continuation":
            df = add_fractal_pivots(df)
            df = add_atr_continuation_entry_labels(df)
            plot_continuation_signals(df, save_path=str(plot_save_path))
        elif args.plot_type == "swing_state_machine":
            df = add_fractal_pivots(df)
            df = add_pivot_swing_state_machine(df)
            plot_swing_state_machine_signals(df, save_path=str(plot_save_path))
        elif args.plot_type == "all_labels":
            df = add_fractal_pivots(df)
            df = add_all_labels(df, swing_state_machine_kwargs={})
            plot_all_labels(df, save_path=str(plot_save_path))
        raise SystemExit(0)

    label_timeframe = "15T"
    dataset_name = _dataset_name_from_label_timeframe(label_timeframe)
    processed_base_dir = get_ticker_processed_base_dir(args.ticker)
    dataset_dir = processed_base_dir / "datasets" / dataset_name
    dataset_exists = (dataset_dir / "X.parquet").exists() and (
        dataset_dir / "y.parquet"
    ).exists()

    used_cache = use_cached and dataset_exists
    if used_cache:
        print(f"Using cached dataset at {dataset_dir}")
    else:
        parquet_path = resolve_intraday_parquet_path(args.ticker)
        df = build_training_matrix(
            parquet_path=parquet_path,
            ticker=args.ticker,
            label_timeframe=label_timeframe,
        )
        clean_training_matrix(
            df,
            save_outputs=args.save_processed,
            ticker=args.ticker,
            dataset_name=dataset_name,
        )

    if args.save_processed:
        dataset_exists = True

    if args.save_processed or used_cache:
        data_pipeline.main(
            ticker=args.ticker,
            dataset_name=dataset_name,
            label_mode=args.label_mode,
            train_frac=args.train_frac,
            val_frac=args.val_frac,
        )
    else:
        print("save_processed=False and no cached dataset; skipping data pipeline.")
