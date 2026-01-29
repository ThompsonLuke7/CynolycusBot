import argparse
import datetime as dt
import math

from API.Alpaca_API.fetch_intraday import fetch_intraday
from Data.load_data import (
    ensure_ticker_dirs,
    get_ticker_data_dir,
    get_ticker_processed_base_dir,
    load_ticker_parquet,
    resolve_intraday_parquet_path,
)
from Data.plots.plots import plot_selected_label_plots
from Features import data_pipeline
from Features.feature_sets.custom_indicators import add_fractal_pivots
from Features.label_generations import (
    add_all_labels,
    add_atr_continuation_strength_labels,
    add_atr_leg_segmentation_labels,
    add_atr_pivot_swing_labels,
    add_bars_to_exhaustion_label,
    add_mfe_mae_labels,
    add_pivot_swing_state_machine,
)
from Features.multi_timeframe_features import ensure_time_index, resample_ohlcv
from Features.feature_matrix import build_feature_matrices, build_feature_matrix, clean_feature_matrix


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
        help="Path to save a single plot PNG (defaults to Data/processed/<ticker>/plots/...).",
    )
    parser.add_argument(
        "--timeframe",
        default="1Hour",
        help='Alpaca timeframe (e.g., "1Hour", "1Day"). Defaults to "1Hour".',
    )
    parser.add_argument(
        "--raw-parquet",
        default=None,
        help="Override raw parquet path (e.g., Data/raw/spy/spy_intraday_15min.parquet).",
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
        default=100000,
        help="Max bars to fetch from Alpaca (default: 100000).",
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
        default=False,
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
        "--model",
        default="all",
        choices=["all", "Tree", "LSTM"],
        help='Model feature set to build/use: "all", "Tree", or "LSTM" (default: "LSTM").',
    )
    parser.add_argument(
        "--train-frac",
        type=float,
        default=0.75,
        help="Train split fraction (default: 0.).",
    )
    parser.add_argument(
        "--val-frac",
        type=float,
        default=0.15,
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
        help=(
            "Which plot(s) to render (default: atr_swing). "
            "Use comma-separated values or 'all'."
        ),
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

    raw_path = resolve_intraday_parquet_path(args.ticker, parquet_path=args.raw_parquet)
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
        target_bars = int(args.limit)
        minutes_per_trading_day = 390
        trading_days = math.ceil(target_bars / minutes_per_trading_day)
        calendar_days = math.ceil(trading_days * 7 / 5)
        default_start = now_utc - dt.timedelta(days=calendar_days)
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
        df = load_ticker_parquet(args.ticker, parquet_path=args.raw_parquet)
        df = ensure_time_index(df)
        rule = _normalize_plot_timeframe(args.plot_timeframe)
        if rule != "1T":
            df = resample_ohlcv(df, rule)
        raw_plot_types = [p.strip() for p in str(args.plot_type).split(",") if p.strip()]
        if not raw_plot_types:
            raw_plot_types = ["atr_swing"]
        canonical_plot_types: set[str] = set()
        for plot_type in raw_plot_types:
            key = plot_type.lower()
            if key == "atr":
                key = "atr_swing"
            elif key == "leg":
                key = "leg_segmentation"
            elif key in {"state_machine", "swing_state"}:
                key = "swing_state_machine"
            elif key in {"mfe", "mae"}:
                key = "mfe_mae"
            elif key == "exhaustion":
                key = "bars_to_exhaustion"
            canonical_plot_types.add(key)

        if "all" in canonical_plot_types or "all_labels" in canonical_plot_types:
            df = add_fractal_pivots(df)
            df = add_all_labels(
                df,
                swing_state_machine_kwargs={},
                mfe_mae_kwargs={},
                bars_to_exhaustion_kwargs={},
            )
        else:
            needs_pivots = any(
                p in {"atr_swing", "continuation", "swing_state_machine", "bars_to_exhaustion"}
                for p in canonical_plot_types
            )
            if needs_pivots:
                df = add_fractal_pivots(df)

            if "atr_swing" in canonical_plot_types:
                df = add_atr_pivot_swing_labels(df)
            if "continuation" in canonical_plot_types:
                df = add_atr_continuation_strength_labels(df)
            if "swing_state_machine" in canonical_plot_types:
                df = add_pivot_swing_state_machine(df)
            if "leg_segmentation" in canonical_plot_types:
                df = add_atr_leg_segmentation_labels(df)
            if "mfe_mae" in canonical_plot_types:
                df = add_mfe_mae_labels(df)
            if "bars_to_exhaustion" in canonical_plot_types:
                # bars_to_exhaustion masks to leg direction by default; ensure leg labels exist
                if "leg_up_label" not in df.columns or "leg_down_label" not in df.columns:
                    df = add_atr_leg_segmentation_labels(df)
                df = add_bars_to_exhaustion_label(df)

        save_paths = None
        if args.save_plot_path:
            if len(raw_plot_types) == 1 and "all" not in canonical_plot_types:
                save_paths = {raw_plot_types[0]: args.save_plot_path}
            else:
                print("save_plot_path ignored because multiple plot types were requested.")

        plot_selected_label_plots(
            df,
            plot_types=args.plot_type,
            save_paths=save_paths,
            ticker=args.ticker,
            data_dir=data_dir,
        )
        raise SystemExit(0)

    label_timeframe = "15T"
    dataset_name = _dataset_name_from_label_timeframe(label_timeframe)
    selected_models = ("Tree", "LSTM") if args.model == "all" else (args.model,)
    selected_model_keys = [model.strip().lower() for model in selected_models]
    processed_base_dir = get_ticker_processed_base_dir(args.ticker)
    dataset_dir = processed_base_dir / "datasets" / dataset_name
    x_files = [
        dataset_dir / f"X_{dataset_name}_{model_key}.parquet"
        for model_key in selected_model_keys
    ]
    y_file = dataset_dir / "y.parquet"
    dataset_exists = all(x_file.exists() for x_file in x_files) and y_file.exists()

    used_cache = use_cached and dataset_exists
    if used_cache:
        print(f"Using cached dataset at {dataset_dir}")
    else:
        parquet_path = resolve_intraday_parquet_path(
            args.ticker, parquet_path=args.raw_parquet
        )

        model_dfs = build_feature_matrices(
            parquet_path=parquet_path,
            ticker=args.ticker,
            label_timeframe=label_timeframe,
            models=selected_models,
        )
        first = True
        align_index = None
        for model_name, df in model_dfs.items():
            model_key = model_name.strip().lower()
            x_file = f"X_{dataset_name}_{model_key}.parquet"  # X_15min_tree.parquet

            cleaned, _, _ = clean_feature_matrix(
                df,
                save_outputs=args.save_processed,
                ticker=args.ticker,
                dataset_name=dataset_name,   # <-- SAME folder
                x_filename=x_file,           # <-- separate X files
                write_y=first,
                align_index=align_index,
            )
            first = False
            if align_index is None:
                align_index = cleaned.index

    if args.save_processed:
        dataset_exists = True

    if args.save_processed or used_cache:
        for model_key in selected_model_keys:
            data_pipeline.main(
                ticker=args.ticker,
                dataset_name=dataset_name,
                label_mode=args.label_mode,
                train_frac=args.train_frac,
                val_frac=args.val_frac,
                x_filename=f"X_{dataset_name}_{model_key}.parquet",
        )
    else:
        print("save_processed=False and no cached dataset; skipping data pipeline.")
