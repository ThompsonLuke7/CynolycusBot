from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

from Data.load_data import get_ticker_processed_base_dir
from Data.retrieve_data import normalize_ticker
from strategies.spy_intraday.Policy import inference_pipeline as ip


def _resolve_repo_root() -> Path:
    try:
        return Path(__file__).resolve().parents[3]
    except NameError:
        return Path.cwd()


REPO_ROOT = _resolve_repo_root()


def _run_cmd(args: list[str]) -> None:
    print(f"[run_inference] Running: {' '.join(args)}")
    proc = subprocess.Popen(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="")
    ret = proc.wait()
    if ret != 0:
        raise subprocess.CalledProcessError(ret, args)


def _artifacts_exist(model_root: Path, label_dirs: list[str]) -> bool:
    for label_dir in label_dirs:
        for side in ("long", "short"):
            artifact_dir = model_root / side / label_dir
            mask_path = artifact_dir / "best_mask.npy"
            model_path = artifact_dir / "xgb_model.json"
            if not (mask_path.exists() and model_path.exists()):
                return False
    return True


def _normalize_ga_label_dir(token: str) -> str:
    value = token.strip().lower()
    if value in {"pivot", "pivots"}:
        return "pivots"
    if value == "tb":
        return "tb"
    if value == "swing":
        return "swing"
    return value


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference on new raw data using existing GA-XGB models."
    )
    parser.add_argument("--ticker", default="$SPY")
    parser.add_argument("--raw-parquet", required=True)
    parser.add_argument("--label-timeframe", default="15min")
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--models", default="Tree")
    parser.add_argument(
        "--ga-model-root",
        default="Data/models/ga_xgboost/15min",
        help="Root folder with trained GA-XGB models (long/short).",
    )
    parser.add_argument(
        "--ga-feature-root",
        default=None,
        help="Processed dataset root used for GA-XGB training (for feature list).",
    )
    parser.add_argument(
        "--ga-label-dirs",
        default="pivots,tb",
        help="Comma-separated label dirs to write probs under (e.g. pivots,tb).",
    )
    parser.add_argument(
        "--train-if-missing",
        action="store_true",
        help="Train GA-XGB models under inference root if artifacts are missing.",
    )
    parser.add_argument(
        "--full-fit-ga",
        action="store_true",
        help="Full-fit GA-XGB when training is required.",
    )
    parser.add_argument("--refresh-masks", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument(
        "--model-path",
        default="Data/outputs/agent/ppo_model.pt",
        help="RL policy checkpoint path.",
    )
    parser.add_argument("--plot-tail", type=int, default=200)
    parser.add_argument("--plot-random-window", action="store_true")
    parser.add_argument("--plot-seed", type=int, default=None)
    parser.add_argument("--plot-out", default=None)
    parser.add_argument("--trace-out", default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    raw_parquet = Path(args.raw_parquet)
    if not raw_parquet.exists():
        raise SystemExit(f"Missing raw parquet: {raw_parquet}")

    label_timeframe = ip._normalize_label_timeframe(args.label_timeframe)
    dataset_name = args.dataset_name or ip._dataset_name_from_label_timeframe(label_timeframe)
    ticker = args.ticker
    slug = normalize_ticker(ticker).lower()
    inference_root = REPO_ROOT / "Data" / "inference" / slug / dataset_name
    processed_root = inference_root
    split_root = inference_root / "splits"
    stats_root = inference_root / "stats"
    model_root = inference_root / "models"
    plots_root = inference_root / "plots"
    for path in (processed_root, split_root, stats_root, model_root, plots_root):
        path.mkdir(parents=True, exist_ok=True)

    model_list = tuple(m.strip() for m in args.models.split(",") if m.strip())
    if not model_list:
        raise SystemExit("No models specified.")

    print("[run_inference] Building processed features/labels...")
    ip._build_processed_from_raw(
        raw_parquet=raw_parquet,
        ticker=ticker,
        dataset_name=dataset_name,
        label_timeframe=label_timeframe,
        models=model_list,
        save_processed=True,
        processed_root=processed_root,
    )

    label_dirs = [
        _normalize_ga_label_dir(s)
        for s in args.ga_label_dirs.split(",")
        if s.strip()
    ]
    ga_model_root = Path(args.ga_model_root)
    if not _artifacts_exist(ga_model_root, label_dirs):
        if not args.train_if_missing:
            raise SystemExit(
                f"GA-XGB artifacts missing in {ga_model_root}. "
                "Re-run with --train-if-missing to create them."
            )
        train_cmd = [
            sys.executable,
            str(REPO_ROOT / "strategies" / "spy_intraday" / "Policy" / "train_inference_models.py"),
            "--processed-root",
            str(processed_root),
            "--model-root",
            str(model_root),
            "--dataset-name",
            dataset_name,
            "--label-modes",
            ",".join(label_dirs),
        ]
        if args.full_fit_ga:
            train_cmd.append("--full-fit")
        if args.refresh_masks:
            train_cmd.append("--refresh-masks")
        _run_cmd(train_cmd)
        ga_model_root = model_root / "ga_xgboost" / dataset_name

    feature_root = (
        Path(args.ga_feature_root)
        if args.ga_feature_root
        else get_ticker_processed_base_dir(normalize_ticker(ticker))
    )
    feature_list_path = (
        feature_root
        / "datasets"
        / dataset_name
        / f"features_X_{dataset_name}_tree.txt"
    )
    plot_frame_path = processed_root / "datasets" / dataset_name / "plot_frame.parquet"
    ip._run_ga_xgb_inference(
        processed_root=processed_root,
        dataset_name=dataset_name,
        x_filename=f"X_{dataset_name}_tree.parquet",
        plot_frame_path=plot_frame_path,
        feature_list_path=feature_list_path,
        model_root=ga_model_root,
        label_dirs=label_dirs,
        output_model_root=model_root,
    )

    print("[run_inference] Building agent matrix...")
    agent_csv = ip._write_agent_matrix_csv(
        ticker=ticker,
        dataset_name=dataset_name,
        drop_na=True,
        processed_root=processed_root,
        model_root=model_root / "ga_xgboost" / dataset_name,
        output_root=inference_root,
        include_pivot_probs="pivots" in label_dirs,
        include_tb_probs="tb" in label_dirs,
    )
    print(f"[run_inference] Agent matrix saved to {agent_csv}")

    if args.skip_eval:
        return

    model_path = Path(args.model_path)
    if not model_path.exists():
        raise SystemExit(f"Missing model checkpoint: {model_path}")

    plot_out = args.plot_out
    if plot_out is None:
        plots_root.mkdir(parents=True, exist_ok=True)
        plot_out = str(plots_root / "agent_actions_vs_price.png")
    trace_out = args.trace_out
    if trace_out is None:
        trace_out = str(inference_root / "agent" / "agent_trace.csv")
    ip._run_agent_eval(
        agent_csv=agent_csv,
        model_path=model_path,
        plot_tail=args.plot_tail,
        plot_random_window=args.plot_random_window,
        plot_seed=args.plot_seed,
        plot_out=plot_out,
        trace_out=trace_out,
    )


if __name__ == "__main__":
    main()
