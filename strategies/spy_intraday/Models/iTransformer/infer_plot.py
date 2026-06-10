from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_MODELS_ROOT = Path(__file__).resolve().parents[1]
if str(_MODELS_ROOT) not in sys.path:
    sys.path.insert(0, str(_MODELS_ROOT))

from Data.load_data import get_ticker_processed_split_dir  # noqa: E402
from Data.retrieve_data import normalize_ticker  # noqa: E402
from Data.plots.plots import _load_plot_frame, plot_model_inference  # noqa: E402
from iTransformer.itransformer_dataset import SplitIndex, WindowedTimeSeries  # noqa: E402
from iTransformer.itransformer_model import iTransformerEncoder  # noqa: E402


def _select_target(label_mode: str, side: str) -> tuple[str, bool]:
    mode = (label_mode or "").strip().lower()
    side = (side or "").strip().lower()
    if mode == "swing":
        return ("long_swing_label" if side == "long" else "short_swing_label"), True
    if mode == "leg":
        return ("leg_up_label" if side == "long" else "leg_down_label"), True
    if mode == "continuation":
        return ("long_cont_label" if side == "long" else "short_cont_label"), True
    if mode == "mfe":
        return ("mfe_up_atr" if side == "long" else "mfe_down_atr"), False
    if mode == "mae":
        return ("mae_down_atr" if side == "long" else "mae_up_atr"), False
    if mode == "mfe_mae":
        return ("mfe_up_atr" if side == "long" else "mfe_down_atr"), False
    if mode == "exhaustion":
        return "bars_to_exhaustion", False
    raise ValueError(f"Unsupported label_mode: {label_mode}")


def _load_split_index(ticker: str, dataset_name: str, x_filename: str) -> SplitIndex:
    clean = normalize_ticker(ticker)
    split_root = get_ticker_processed_split_dir(clean)
    split_dir = split_root / dataset_name / Path(x_filename).stem
    train_idx = np.load(split_dir / "train_idx.npy")
    val_idx = np.load(split_dir / "val_idx.npy")
    test_idx = np.load(split_dir / "test_idx.npy")
    if train_idx.size == 0 or val_idx.size == 0 or test_idx.size == 0:
        raise ValueError("Split indices are empty.")
    train_end = int(train_idx[-1] + 1)
    val_end = int(val_idx[-1] + 1)
    return SplitIndex(train_end=train_end, val_end=val_end)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load iTransformer artifacts and plot inference without retraining."
    )
    parser.add_argument("--ticker", default="SPY")
    parser.add_argument("--dataset", dest="dataset_name", default="15min")
    parser.add_argument(
        "--label-mode",
        default="swing",
        choices=["swing", "leg", "continuation", "mfe", "mae", "mfe_mae", "exhaustion"],
    )
    parser.add_argument("--side", default="long", choices=["long", "short"])
    parser.add_argument("--x-filename", default=None)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    slug = normalize_ticker(args.ticker).lower()
    dataset_name = args.dataset_name
    x_filename = args.x_filename or f"X_{dataset_name}_lstm.parquet"

    dataset_dir = Path("Data") / "processed" / slug / "datasets" / dataset_name
    x_path = dataset_dir / x_filename
    y_path = dataset_dir / "y.parquet"
    if not x_path.exists() or not y_path.exists():
        raise FileNotFoundError(f"Missing {x_filename} or y.parquet in {dataset_dir}")

    target_col, is_binary = _select_target(args.label_mode, args.side)
    y_df = pd.read_parquet(y_path)
    if target_col not in y_df.columns:
        raise KeyError(f"Missing label column '{target_col}' in {y_path.name}")

    X_df = pd.read_parquet(x_path)
    X = X_df.to_numpy(dtype=np.float32)
    y = y_df[target_col].to_numpy()

    if is_binary:
        y = y.astype(np.float32)
    else:
        y = y.astype(np.float32)

    model_root = Path("Data") / "models" / "itransformer"
    seq_len = args.seq_len
    if seq_len is None:
        meta_path = (
            model_root
            / f"{slug}_{dataset_name}_{args.label_mode}_{args.side}_seq64_meta.json"
        )
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            seq_len = int(meta["seq_len"])
        else:
            # try to find any matching meta for this label/side
            candidates = sorted(
                model_root.glob(
                    f"{slug}_{dataset_name}_{args.label_mode}_{args.side}_seq*_meta.json"
                )
            )
            if not candidates:
                raise FileNotFoundError(
                    f"No meta found for {slug}/{dataset_name}/{args.label_mode}/{args.side}."
                )
            meta = json.loads(candidates[-1].read_text())
            seq_len = int(meta["seq_len"])
    else:
        meta = json.loads(
            (
                model_root
                / f"{slug}_{dataset_name}_{args.label_mode}_{args.side}_seq{seq_len}_meta.json"
            ).read_text()
        )

    model_path = (
        model_root
        / f"{slug}_{dataset_name}_{args.label_mode}_{args.side}_seq{seq_len}.pth"
    )
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model file: {model_path}")

    mask_path = (
        model_root
        / f"{slug}_{dataset_name}_{args.label_mode}_{args.side}_seq{seq_len}_mask.npy"
    )
    if mask_path.exists():
        mask = np.load(mask_path).astype(bool)
        X = X[:, mask]

    split_idx = _load_split_index(args.ticker, dataset_name, x_filename)
    ds_test = WindowedTimeSeries(X, y, seq_len, "test", split_idx)
    dl_test = DataLoader(ds_test, batch_size=256, shuffle=False)

    model = iTransformerEncoder(
        seq_len=seq_len,
        num_variates=X.shape[1],
        d_model=int(meta["d_model"]),
        n_heads=int(meta["n_heads"]),
        n_layers=int(meta["n_layers"]),
        d_ff=int(meta["d_ff"]),
        dropout=float(meta["dropout"]),
        use_var_embedding=bool(meta.get("use_var_embedding", False)),
        out_dim=1,
    )
    state = torch.load(model_path, map_location=args.device)
    model.load_state_dict(state)
    model.to(args.device)
    model.eval()

    preds = []
    for xb, _, _ in dl_test:
        xb = xb.to(args.device)
        with torch.no_grad():
            out = model(xb)
        preds.append(out.detach().cpu().numpy())
    preds = np.concatenate(preds, axis=0).reshape(-1)

    plot_df = _load_plot_frame(args.ticker, ds_test.targets, x_path=x_path)
    actual = y[ds_test.targets]

    if is_binary:
        probs = 1.0 / (1.0 + np.exp(-preds))
        long_probs = probs if args.side == "long" else None
        short_probs = probs if args.side == "short" else None
        long_actual = actual if args.side == "long" else None
        short_actual = actual if args.side == "short" else None
        plot_model_inference(
            plot_df,
            long_probs,
            short_probs,
            long_actual=long_actual,
            short_actual=short_actual,
            long_label_name="LONG",
            short_label_name="SHORT",
            threshold=args.threshold,
            title=f"{args.ticker} | iTransformer {args.label_mode} {args.side}",
        )
    else:
        raise ValueError(
            "Regression plotting is not implemented in infer_plot.py. "
            "Use run_train.py or extend this script."
        )


if __name__ == "__main__":
    main()
