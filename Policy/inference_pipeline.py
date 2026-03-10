from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from Data.load_data import get_ticker_processed_base_dir
from Data.retrieve_data import normalize_ticker
from Features import data_pipeline
from Features.feature_matrix import build_feature_matrices, clean_feature_matrix
from Features.feature_matrix_regime import AgentFeatureConfig, build_agent_feature_matrix
from Features.feature_scaling import save_normalization_stats


def _resolve_repo_root() -> Path:
    try:
        return Path(__file__).resolve().parents[1]
    except NameError:
        return Path.cwd()


REPO_ROOT = _resolve_repo_root()


def _normalize_ga_label_dir(token: str) -> str:
    value = token.strip().lower()
    if value in {"pivot", "pivots"}:
        return "pivots"
    if value == "tb":
        return "tb"
    if value == "swing":
        return "swing"
    return value


def _label_dir_to_train_mode(label_dir: str) -> str:
    token = _normalize_ga_label_dir(label_dir)
    if token == "pivots":
        return "pivot"
    if token == "swing":
        return "swing"
    if token == "tb":
        return "tb"
    return token


def _normalize_label_timeframe(label_timeframe: str) -> str:
    tf = label_timeframe.strip().lower()
    if tf.endswith("min"):
        minutes = int(tf.replace("min", "") or "1")
        return f"{minutes}min"
    if tf.endswith("t"):
        minutes = int(tf[:-1] or "1")
        return f"{minutes}min"
    if tf.endswith("m"):
        minutes = int(tf.replace("m", "") or "1")
        return f"{minutes}min"
    if tf.endswith("hour"):
        hours = int(tf.replace("hour", "") or "1")
        return f"{hours}h"
    if tf.endswith("h"):
        hours = int(tf.replace("h", "") or "1")
        return f"{hours}h"
    if tf.endswith("day"):
        days = int(tf.replace("day", "") or "1")
        return f"{days}d"
    if tf.endswith("d"):
        days = int(tf.replace("d", "") or "1")
        return f"{days}d"
    return label_timeframe


def _dataset_name_from_label_timeframe(label_timeframe: str) -> str:
    tf = _normalize_label_timeframe(label_timeframe).lower()
    if tf.endswith("t"):
        return f"{tf[:-1]}min"
    if tf.endswith("h"):
        return f"{tf[:-1]}h"
    if tf.endswith("d"):
        return f"{tf[:-1]}d"
    return tf


def _infer_vix_fetch_timeframe(label_timeframe: str) -> str:
    tf = _normalize_label_timeframe(label_timeframe).lower()
    if tf.endswith("min"):
        n = int(tf.replace("min", "") or "1")
        return f"{n}Min"
    if tf.endswith("h"):
        n = int(tf[:-1] or "1")
        return f"{n}Hour"
    return "15Min"


def _infer_fetch_window_from_raw(raw_parquet: Path) -> tuple[str, str | None]:
    try:
        raw_df = pd.read_parquet(raw_parquet)
        if "timestamp" in raw_df.columns:
            ts = pd.to_datetime(raw_df["timestamp"], utc=True, errors="coerce")
        else:
            ts = pd.to_datetime(raw_df.index, utc=True, errors="coerce")
        ts = ts[ts.notna()]
        if len(ts) == 0:
            return "2021-01-01T00:00:00Z", None
        start_ts = ts.min() - pd.Timedelta(days=2)
        end_ts = ts.max() + pd.Timedelta(days=1)
        return (
            start_ts.isoformat().replace("+00:00", "Z"),
            end_ts.isoformat().replace("+00:00", "Z"),
        )
    except Exception:
        return "2021-01-01T00:00:00Z", None


def _ensure_vix_parquet(
    *,
    raw_parquet: Path,
    vix_parquet_path: Path,
    vix_ticker: str,
    label_timeframe: str,
    fetch_if_missing: bool,
) -> None:
    if vix_parquet_path.exists():
        return
    if not fetch_if_missing:
        return
    try:
        from API.Alpaca_API.market_data.fetch_intraday import fetch_intraday

        tf = _infer_vix_fetch_timeframe(label_timeframe)
        start, end = _infer_fetch_window_from_raw(raw_parquet)
        vix_parquet_path.parent.mkdir(parents=True, exist_ok=True)
        print(
            f"[inference_pipeline] Missing VIX parquet at {vix_parquet_path}; "
            f"fetching {vix_ticker} {tf}..."
        )
        fetch_intraday(
            ticker=vix_ticker,
            start=start,
            end=end,
            timeframe=tf,
            limit=100000,
            adjustment="raw",
            save_path=str(vix_parquet_path),
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[inference_pipeline] VIX fetch skipped/failed: {exc}")


def _run_cmd(args: list[str]) -> None:
    print(f"[inference_pipeline] Running: {' '.join(args)}")
    proc = subprocess.Popen(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        cwd=str(REPO_ROOT),
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="")
    ret = proc.wait()
    if ret != 0:
        raise subprocess.CalledProcessError(ret, args)


def _build_processed_from_raw(
    *,
    raw_parquet: Path,
    ticker: str,
    dataset_name: str,
    label_timeframe: str,
    models: tuple[str, ...],
    save_processed: bool,
    processed_root: Path,
    vix_parquet_path: Path,
    vix_ticker: str,
    vix_fetch_if_missing: bool,
) -> None:
    feature_timeframes = {
        "30m": "30min",
        "1h": "1h",
        "4h": "4h",
        "1d": "1d",
    }
    _ensure_vix_parquet(
        raw_parquet=raw_parquet,
        vix_parquet_path=vix_parquet_path,
        vix_ticker=vix_ticker,
        label_timeframe=label_timeframe,
        fetch_if_missing=vix_fetch_if_missing,
    )

    model_dfs = build_feature_matrices(
        parquet_path=raw_parquet,
        ticker=ticker,
        label_timeframe=label_timeframe,
        feature_timeframes=feature_timeframes,
        models=models,
        include_vix_features=True,
        vix_ticker=vix_ticker,
        vix_parquet_path=vix_parquet_path,
    )
    align_index = None
    first = True
    for model_name, df in model_dfs.items():
        model_key = model_name.strip().lower()
        x_filename = f"X_{dataset_name}_{model_key}.parquet"
        cleaned, _, _ = clean_feature_matrix(
            df,
            save_outputs=save_processed,
            output_dir=processed_root,
            ticker=ticker,
            dataset_name=dataset_name,
            x_filename=x_filename,
            write_y=first,
            align_index=align_index,
        )
        first = False
        if align_index is None:
            align_index = cleaned.index


def _run_ga_xgb(
    *,
    label_mode: str,
    refresh_masks: bool,
    processed_root: Path,
    split_root: Path,
    stats_root: Path,
    model_root: Path,
    full_fit: bool,
) -> None:
    cmd = [sys.executable, str(REPO_ROOT / "Models" / "ga_xgboost" / "train.py")]
    cmd += ["--label-mode", label_mode]
    cmd += ["--processed-root", str(processed_root)]
    cmd += ["--split-root", str(split_root)]
    cmd += ["--stats-root", str(stats_root)]
    cmd += ["--model-root", str(model_root)]
    if full_fit:
        cmd.append("--full-fit")
    if refresh_masks:
        cmd.append("--refresh-masks")
    _run_cmd(cmd)


def _load_feature_list(features_path: Path) -> list[str]:
    if not features_path.exists():
        raise FileNotFoundError(f"Missing feature list: {features_path}")
    return [line.strip() for line in features_path.read_text().splitlines() if line.strip()]


def _predict_ga_xgb(
    *,
    x_df: pd.DataFrame,
    feature_list: list[str],
    model_dir: Path,
    side: str,
    label_dir: str,
) -> np.ndarray:
    x_aligned = x_df.reindex(columns=feature_list)
    artifact_dir = model_dir / side / label_dir
    mask_path = artifact_dir / "best_mask.npy"
    model_path = artifact_dir / "xgb_model.json"
    if not (mask_path.exists() and model_path.exists()):
        raise FileNotFoundError(f"Missing GA-XGB artifacts under {artifact_dir}")

    mask = np.load(mask_path).astype(bool)
    if mask.size != len(feature_list):
        raise ValueError(
            f"Mask length {mask.size} does not match feature list {len(feature_list)}"
        )

    x_selected = x_aligned.to_numpy(dtype=np.float32)[:, mask]
    dmat = xgb.DMatrix(x_selected)
    model = xgb.Booster()
    model.load_model(str(model_path))
    probs = model.predict(dmat)
    return probs.astype(np.float32)


def _save_probs(
    *,
    probs: np.ndarray,
    index: pd.Index,
    output_dir: Path,
    prefix: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / f"{prefix}_full.npy", probs.astype(np.float32))
    df = pd.DataFrame({f"{prefix}_full": probs.astype(np.float32)}, index=index)
    df.to_parquet(output_dir / f"{prefix}_probs.parquet")


def _run_ga_xgb_inference(
    *,
    processed_root: Path,
    dataset_name: str,
    x_filename: str,
    plot_frame_path: Path,
    feature_list_path: Path,
    model_root: Path,
    label_dirs: list[str],
    output_model_root: Path,
) -> None:
    dataset_dir = processed_root / "datasets" / dataset_name
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Missing dataset dir: {dataset_dir}")

    x_df = pd.read_parquet(dataset_dir / x_filename)
    plot_df = pd.read_parquet(plot_frame_path)
    if len(x_df) != len(plot_df):
        raise ValueError(
            f"X rows ({len(x_df)}) do not match plot_frame rows ({len(plot_df)})"
        )
    feature_list = _load_feature_list(feature_list_path)

    for label_dir in label_dirs:
        long_probs = _predict_ga_xgb(
            x_df=x_df,
            feature_list=feature_list,
            model_dir=model_root,
            side="long",
            label_dir=label_dir,
        )
        short_probs = _predict_ga_xgb(
            x_df=x_df,
            feature_list=feature_list,
            model_dir=model_root,
            side="short",
            label_dir=label_dir,
        )
        long_out = (
            output_model_root / "ga_xgboost" / dataset_name / "long" / label_dir
        )
        short_out = (
            output_model_root / "ga_xgboost" / dataset_name / "short" / label_dir
        )
        _save_probs(probs=long_probs, index=plot_df.index, output_dir=long_out, prefix="p_long")
        _save_probs(probs=short_probs, index=plot_df.index, output_dir=short_out, prefix="p_short")


def _write_agent_matrix_csv(
    *,
    ticker: str,
    dataset_name: str,
    drop_na: bool,
    processed_root: Path,
    model_root: Path,
    output_root: Path,
    include_pivot_probs: bool,
    include_tb_probs: bool,
    pivot_label_dir: str,
    tb_label_dir: str,
    vix_parquet_path: Path,
) -> Path:
    cfg = AgentFeatureConfig(
        ticker=ticker,
        dataset_name=dataset_name,
        drop_na=drop_na,
        processed_root=processed_root,
        model_root=model_root,
        pivot_label_dir=pivot_label_dir,
        tb_label_dir=tb_label_dir,
        include_pivot_probs=include_pivot_probs,
        include_tb_probs=include_tb_probs,
        vix_parquet_path=str(vix_parquet_path),
    )
    df = build_agent_feature_matrix(config=cfg)
    out_dir = output_root / "agent"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "agent_matrix.csv"
    df.to_csv(out_path, index=False)
    df.to_parquet(out_dir / "agent_matrix.parquet", index=False)
    return out_path


def _run_agent_eval(
    *,
    agent_csv: Path,
    model_path: Path,
    plot_tail: int,
    plot_random_window: bool,
    plot_seed: int | None,
    plot_out: str | None,
    trace_out: str | None = None,
) -> None:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "Policy" / "Agent" / "run_eval.py"),
        "--data-csv",
        str(agent_csv),
        "--model-path",
        str(model_path),
        "--plot-tail",
        str(plot_tail),
    ]
    if plot_random_window:
        cmd.append("--plot-random-window")
    if plot_seed is not None:
        cmd += ["--plot-seed", str(plot_seed)]
    if plot_out:
        cmd += ["--plot-out", plot_out]
    if trace_out:
        cmd += ["--trace-out", trace_out]
    _run_cmd(cmd)


def _resolve_cli_path(path_value: str) -> Path:
    p = Path(path_value)
    if p.is_absolute():
        return p
    return REPO_ROOT / p


def _run_meta_replay_eval(
    *,
    ticker: str,
    dataset_name: str,
    replay_data_path: Path,
    meta_model_root: Path,
    ga_model_root: Path,
    pivot_label_dir: str,
    tb_label_dir: str,
    interval: int,
    start: str | None,
    end: str | None,
    regular_only: bool,
    prepend_split_test_warmup: bool,
    plot_out: str | None,
    trace_out: str | None,
) -> None:
    cmd = [
        sys.executable,
        "-m",
        "API.Alpaca_API.runners.replay_runner",
        "--data-path",
        str(replay_data_path),
        "--symbols",
        normalize_ticker(ticker),
        "--inference-mode",
        "meta",
        "--interval",
        str(int(interval)),
        "--meta-model-root",
        str(meta_model_root),
        "--ga-model-root",
        str(ga_model_root),
        "--ga-dataset-name",
        str(dataset_name),
        "--ga-pivot-label-dir",
        str(pivot_label_dir),
        "--ga-tb-label-dir",
        str(tb_label_dir),
    ]
    if start:
        cmd += ["--start", str(start)]
    if end:
        cmd += ["--end", str(end)]
    if regular_only:
        cmd.append("--regular-only")
    if not prepend_split_test_warmup:
        cmd.append("--no-prepend-split-test-warmup")
    if plot_out:
        cmd += ["--plot-out", str(plot_out)]
    if trace_out:
        cmd += ["--trace-out", str(trace_out)]
    _run_cmd(cmd)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="End-to-end inference pipeline for new raw intraday data."
    )
    parser.add_argument("--ticker", default="SPY")
    parser.add_argument(
        "--raw-parquet",
        default="Data/raw/spy/inference_buffer_10min.parquet",
        help="Path to the new raw intraday parquet (e.g. 15m bars).",
    )
    parser.add_argument("--label-timeframe", default="10T")
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--models", default="Tree", help="Comma-separated models.")
    parser.add_argument("--train-frac", type=float, default=0.75)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--refresh-masks", action="store_true")
    parser.add_argument("--skip-ga", action="store_true")
    parser.add_argument(
        "--full-fit-ga",
        action="store_true",
        help="Train GA-XGB masks on the full dataset for live inference.",
    )
    parser.add_argument(
        "--ga-model-root",
        default="Data/models/ga_xgboost/10min",
        help="Root folder with trained GA-XGB models (long/short).",
    )
    parser.add_argument(
        "--ga-feature-root",
        default=None,
        help="Processed dataset root used for GA-XGB training (for feature list).",
    )
    parser.add_argument(
        "--ga-label-dirs",
        default="swing,tb",
        help="Comma-separated label dirs to write probs under (e.g. pivots,tb).",
    )
    parser.add_argument(
        "--eval-mode",
        choices=["ppo", "meta", "none"],
        default="meta",
        help="Evaluation mode: ppo policy eval, replayed meta eval, or none.",
    )
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
    parser.add_argument(
        "--meta-model-root",
        default="Data/models/meta_xgboost/10min",
        help="Meta-XGB model root used by replay eval mode.",
    )
    parser.add_argument(
        "--meta-ga-model-root",
        default="Data/models/ga_xgboost/10min",
        help="Optional GA model root for replay meta eval (defaults to --ga-model-root).",
    )
    parser.add_argument(
        "--meta-pivot-label-dir",
        default="swing",
        help="Pivot label dir used by replay meta eval.",
    )
    parser.add_argument(
        "--meta-tb-label-dir",
        default="tb",
        help="TB label dir used by replay meta eval.",
    )
    parser.add_argument(
        "--meta-replay-data-path",
        default="Data/raw/spy/inference_buffer_1m.parquet",
        help="Replay source (1m parquet/csv) for meta eval mode.",
    )
    parser.add_argument(
        "--meta-replay-start",
        default="2026-01-30T00:00:00Z",
        help="Optional ISO UTC start for replay meta eval.",
    )
    parser.add_argument(
        "--meta-replay-end",
        default="2026-03-05T00:00:00Z",
        help="Optional ISO UTC end for replay meta eval.",
    )
    parser.add_argument(
        "--meta-replay-regular-only",
        action="store_true",
        help="Filter replay bars to regular hours for meta eval mode.",
    )
    parser.add_argument(
        "--meta-interval",
        type=int,
        default=10,
        help="Replay aggregation interval for meta eval mode.",
    )
    parser.add_argument(
        "--meta-prepend-split-test-warmup",
        action="store_true",
        help="Prepend split-test warmup bars in replay meta eval (default disabled).",
    )
    parser.add_argument(
        "--vix-parquet",
        default="Data/raw/vix/vixy_15min_buffer.parquet",
        help="Intraday VIX parquet path used by feature_matrix during inference.",
    )
    parser.add_argument(
        "--vix-ticker",
        default="VIXY",
        help="Ticker used when fetching missing intraday VIX parquet.",
    )
    parser.add_argument(
        "--vix-fetch-if-missing",
        dest="vix_fetch_if_missing",
        action="store_true",
        default=True,
        help="Auto-fetch intraday VIX data when --vix-parquet is missing (default).",
    )
    parser.add_argument(
        "--no-vix-fetch",
        dest="vix_fetch_if_missing",
        action="store_false",
        help="Do not auto-fetch intraday VIX data.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    raw_parquet = Path(args.raw_parquet)
    if not raw_parquet.exists():
        raise SystemExit(f"Missing raw parquet: {raw_parquet}")

    label_timeframe = _normalize_label_timeframe(args.label_timeframe)
    dataset_name = args.dataset_name or _dataset_name_from_label_timeframe(label_timeframe)
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
    vix_parquet_path = Path(args.vix_parquet)
    if not vix_parquet_path.is_absolute():
        vix_parquet_path = REPO_ROOT / vix_parquet_path

    model_list = tuple(m.strip() for m in args.models.split(",") if m.strip())
    if not model_list:
        raise SystemExit("No models specified.")

    print("[inference_pipeline] Building processed features/labels...")
    _build_processed_from_raw(
        raw_parquet=raw_parquet,
        ticker=ticker,
        dataset_name=dataset_name,
        label_timeframe=label_timeframe,
        models=model_list,
        save_processed=True,
        processed_root=processed_root,
        vix_parquet_path=vix_parquet_path,
        vix_ticker=args.vix_ticker,
        vix_fetch_if_missing=bool(args.vix_fetch_if_missing),
    )

    print("[inference_pipeline] Building splits + scaler stats...")
    for model_name in model_list:
        model_key = model_name.strip().lower()
        x_filename = f"X_{dataset_name}_{model_key}.parquet"
        dataset_dir = processed_root / "datasets" / dataset_name
        X = pd.read_parquet(dataset_dir / x_filename)
        splits = data_pipeline.chronological_split_indices(
            len(X),
            train_frac=args.train_frac,
            val_frac=args.val_frac,
        )
        stats = data_pipeline.fit_scaler_on_train(X, splits["train"])
        x_stem = Path(x_filename).stem
        data_pipeline.save_split_indices(split_root, dataset_name, splits, x_stem)
        save_normalization_stats(
            stats_root,
            stats,
            filename=f"norm_stats_{dataset_name}_{x_stem}_train.json",
        )

    label_dirs = [
        _normalize_ga_label_dir(s)
        for s in args.ga_label_dirs.split(",")
        if s.strip()
    ]
    pivot_label_dir = next((lbl for lbl in label_dirs if lbl in {"swing", "pivots"}), "pivots")
    tb_label_dir = "tb"
    if not args.skip_ga:
        if args.full_fit_ga:
            print("[inference_pipeline] Training GA-XGB full-fit masks...")
            for label_dir in label_dirs:
                train_mode = _label_dir_to_train_mode(label_dir)
                if train_mode not in {"pivot", "swing", "tb"}:
                    continue
                _run_ga_xgb(
                    label_mode=train_mode,
                    refresh_masks=args.refresh_masks,
                    processed_root=processed_root,
                    split_root=split_root,
                    stats_root=stats_root,
                    model_root=model_root,
                    full_fit=True,
                )
        else:
            print("[inference_pipeline] Running GA-XGB inference using existing models...")
            ga_model_root = _resolve_cli_path(args.ga_model_root)
            if not ga_model_root.exists():
                raise SystemExit(f"Missing GA-XGB model root: {ga_model_root}")
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
            _run_ga_xgb_inference(
                processed_root=processed_root,
                dataset_name=dataset_name,
                x_filename=f"X_{dataset_name}_tree.parquet",
                plot_frame_path=plot_frame_path,
                feature_list_path=feature_list_path,
                model_root=ga_model_root,
                label_dirs=label_dirs,
                output_model_root=model_root,
            )

    print("[inference_pipeline] Building agent matrix...")
    agent_csv = _write_agent_matrix_csv(
        ticker=ticker,
        dataset_name=dataset_name,
        drop_na=True,
        processed_root=processed_root,
        model_root=model_root / "ga_xgboost" / dataset_name,
        output_root=inference_root,
        include_pivot_probs=any(lbl in {"pivots", "swing"} for lbl in label_dirs),
        include_tb_probs="tb" in label_dirs,
        pivot_label_dir=pivot_label_dir,
        tb_label_dir=tb_label_dir,
        vix_parquet_path=vix_parquet_path,
    )
    print(f"[inference_pipeline] Agent matrix saved to {agent_csv}")

    eval_mode = "none" if args.skip_eval else str(args.eval_mode).strip().lower()
    if eval_mode == "none":
        return

    if eval_mode == "ppo":
        model_path = Path(args.model_path)
        if not model_path.exists():
            raise SystemExit(f"Missing model checkpoint: {model_path}")

        print("[inference_pipeline] Running PPO policy evaluation...")
        plot_out = args.plot_out
        if plot_out is None:
            plots_root.mkdir(parents=True, exist_ok=True)
            plot_out = str(plots_root / "agent_actions_vs_price.png")
        trace_out = args.trace_out
        if trace_out is None:
            trace_out = str(inference_root / "agent" / "agent_trace.csv")
        _run_agent_eval(
            agent_csv=agent_csv,
            model_path=model_path,
            plot_tail=args.plot_tail,
            plot_random_window=args.plot_random_window,
            plot_seed=args.plot_seed,
            plot_out=plot_out,
            trace_out=trace_out,
        )
        return

    if eval_mode == "meta":
        replay_data_path = _resolve_cli_path(args.meta_replay_data_path)
        if not replay_data_path.exists():
            raise SystemExit(f"Missing replay data path: {replay_data_path}")
        meta_model_root = _resolve_cli_path(args.meta_model_root)
        if not meta_model_root.exists():
            raise SystemExit(f"Missing meta model root: {meta_model_root}")
        ga_model_root_value = args.meta_ga_model_root or args.ga_model_root
        ga_model_root = _resolve_cli_path(ga_model_root_value)
        if not ga_model_root.exists():
            raise SystemExit(f"Missing GA model root for meta replay: {ga_model_root}")

        print("[inference_pipeline] Running Meta-XGB replay evaluation...")
        plot_out = args.plot_out
        if plot_out is None:
            plots_root.mkdir(parents=True, exist_ok=True)
            plot_out = str(plots_root / "meta_entries_exits_probs.png")
        trace_out = args.trace_out
        if trace_out is None:
            trace_out = str(inference_root / "meta" / "meta_trace.csv")
        _run_meta_replay_eval(
            ticker=ticker,
            dataset_name=dataset_name,
            replay_data_path=replay_data_path,
            meta_model_root=meta_model_root,
            ga_model_root=ga_model_root,
            pivot_label_dir=str(args.meta_pivot_label_dir),
            tb_label_dir=str(args.meta_tb_label_dir),
            interval=int(args.meta_interval),
            start=args.meta_replay_start,
            end=args.meta_replay_end,
            regular_only=bool(args.meta_replay_regular_only),
            prepend_split_test_warmup=bool(args.meta_prepend_split_test_warmup),
            plot_out=plot_out,
            trace_out=trace_out,
        )
        return

    raise SystemExit(f"Unsupported eval mode: {eval_mode}")


if __name__ == "__main__":
    main()
