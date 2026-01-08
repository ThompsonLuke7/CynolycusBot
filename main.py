import argparse
import datetime as dt

from API.Alpaca_API.fetch_intraday import fetch_intraday
from Data.load_data import ensure_ticker_dirs, get_ticker_data_dir, load_ticker_parquet
from Data.plots.all_labels_plot import plot_all_labels
from Data.plots.atr_swing_plot import plot_atr_swing_signals
from Data.plots.continuation_plot import plot_continuation_signals
from Data.plots.leg_segmentation_plot import plot_leg_segmentation_signals
from Data.plots.swing_state_machine_plot import plot_swing_state_machine_signals
from Features import data_pipeline, feature_engineering, test_leakage
from Features.custom_indicators import add_fractal_pivots
from Features.label_generations import (
    add_all_labels,
    add_atr_continuation_entry_labels,
    add_atr_leg_segmentation_labels,
    add_atr_pivot_swing_labels,
    add_pivot_swing_state_machine,
)
from Features.multi_timeframe_features import (
    DEFAULT_TIMEFRAMES,
    ensure_time_index,
    resample_ohlcv,
)


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
        default=1000,
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

    data_dir = get_ticker_data_dir(args.ticker)
    data_dir_missing = not data_dir.exists()
    ensure_ticker_dirs(args.ticker)

    refresh_data = args.refresh_data or data_dir_missing
    use_cached = args.use_cached
    if refresh_data:
        if use_cached:
            print("Refresh requested; forcing use_cached=False.")
        use_cached = False
        now_utc = dt.datetime.now(dt.timezone.utc)
        default_start = now_utc - dt.timedelta(days=200)
        start = args.start or default_start
        end = args.end
        fetch_intraday(
            ticker=args.ticker,
            start=start,
            end=end,
            timeframe=args.timeframe,
            limit=200,
            adjustment=args.adjustment,
        )

    if args.plot_only:
        df = load_ticker_parquet(args.ticker)
        df = ensure_time_index(df)
        rule = _normalize_plot_timeframe(args.plot_timeframe)
        if rule != "1T":
            df = resample_ohlcv(df, rule)

        if args.plot_type == "atr_swing":
            df = add_fractal_pivots(df)
            df = add_atr_pivot_swing_labels(df)
            plot_atr_swing_signals(df, save_path=args.save_plot_path)
        elif args.plot_type == "leg":
            df = add_atr_leg_segmentation_labels(df)
            plot_leg_segmentation_signals(df, save_path=args.save_plot_path)
        elif args.plot_type == "continuation":
            df = add_fractal_pivots(df)
            df = add_atr_continuation_entry_labels(df)
            plot_continuation_signals(df, save_path=args.save_plot_path)
        elif args.plot_type == "swing_state_machine":
            df = add_fractal_pivots(df)
            df = add_pivot_swing_state_machine(df)
            plot_swing_state_machine_signals(df, save_path=args.save_plot_path)
        elif args.plot_type == "all_labels":
            df = add_fractal_pivots(df)
            df = add_all_labels(df, swing_state_machine_kwargs={})
            plot_all_labels(df, save_path=args.save_plot_path)
        raise SystemExit(0)

    tf_norm = args.timeframe.lower().strip()
    multi_timeframes = None
    if tf_norm.endswith("min") and tf_norm.replace("min", "") in {"1", ""}:
        multi_timeframes = DEFAULT_TIMEFRAMES

    feature_engineering.main(
        ticker=args.ticker,
        use_cached=use_cached,
        save_processed=args.save_processed,
        save_plot_path=args.save_plot_path,
        label_mode=args.label_mode,
        multi_timeframes=multi_timeframes,
    )
    data_pipeline.main(
        ticker=args.ticker,
        label_mode=args.label_mode,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
    )
    # test_leakage.main()
