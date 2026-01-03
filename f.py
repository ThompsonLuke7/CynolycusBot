import argparse

from Features import feature_engineering, test_leakage


def parse_args():
    parser = argparse.ArgumentParser(description="Run feature engineering pipeline.")
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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    feature_engineering.main(
        use_cached=args.use_cached,
        save_processed=args.save_processed,
        save_plot_path=args.save_plot_path,
    )
    # test_leakage.main()
