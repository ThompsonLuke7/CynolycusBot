import argparse
import sys
from pathlib import Path

from sklearn.isotonic import spearmanr


def _resolve_repo_root() -> Path:
    try:
        return Path(__file__).resolve().parents[2]
    except NameError:
        return Path.cwd()


REPO_ROOT = _resolve_repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Data.plots.plots import plot_bilstm_inference_vs_actual  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot BiLSTM inference vs actual targets on a split."
    )
    parser.add_argument("--ticker", default="$SPY")
    parser.add_argument("--dataset", default="15min")
    parser.add_argument("--model-name", default="mabilstm")
    parser.add_argument(
        "--label-mode",
        default="mfe_mae",
        choices=["mfe_mae", "mfe", "mae", "swing", "leg"],
    )
    parser.add_argument("--seq-len", type=int, default=30)
    parser.add_argument("--x-file", default=None)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument(
        "--side",
        default="long,short",
        help="Comma-separated sides to plot (e.g., long,short or long).",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--tail", type=int, default=None)
    parser.add_argument("--save", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--quintile-bins", type=int, default=5)
    parser.add_argument(
        "--no-quintile-test",
        dest="quintile_test",
        action="store_false",
        help="Disable the quintile usefulness test output.",
    )
    args = parser.parse_args()

    sides = tuple(s.strip() for s in str(args.side).split(",") if s.strip())
    if not sides:
        sides = ("long", "short")

    plot_bilstm_inference_vs_actual(
        ticker=args.ticker,
        dataset_name=args.dataset,
        model_name=args.model_name,
        label_mode=args.label_mode,
        seq_len=args.seq_len,
        x_filename=args.x_file,
        split=args.split,
        sides=sides,
        batch_size=args.batch_size,
        tail=args.tail,
        save_path=args.save,
        device=args.device,
        quintile_bins=args.quintile_bins,
        print_quintile_test=args.quintile_test,
    )
    


if __name__ == "__main__":
    main()
