# itransformer_train.py
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ga_itransformer import GAITransformerFeatureSelector
from itransformer_dataset import SplitIndex, WindowedTimeSeries
from itransformer_model import iTransformerEncoder


def set_seed(seed: int = 42) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class EarlyStopper:
    def __init__(self, patience: int = 6, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best = None
        self.bad = 0

    def step(self, val: float) -> bool:
        if self.best is None or val < self.best - self.min_delta:
            self.best = val
            self.bad = 0
            return False
        self.bad += 1
        return self.bad >= self.patience


@torch.no_grad()
def evaluate(model, loader, loss_fn, task: str, y_mu=None, y_std=None, device="cpu"):
    model.eval()
    total_loss = 0.0
    n = 0
    mae_real = 0.0
    correct = 0
    total = 0

    for x, y, _ in loader:
        x = x.to(device)
        y = y.to(device)
        out = model(x)
        if task == "binary":
            yb = y.view(-1, 1)
            loss = loss_fn(out, yb)
            probs = torch.sigmoid(out)
            preds = (probs >= 0.5).float()
            correct += (preds == yb).sum().item()
            total += yb.numel()
        else:
            yt = y.view_as(out)
            loss = loss_fn(out, yt)
            if y_mu is not None and y_std is not None:
                pred_real = out * y_std + y_mu
                y_real = yt * y_std + y_mu
                mae_real += torch.abs(pred_real - y_real).sum().item()

        bs = x.shape[0]
        total_loss += loss.item() * bs
        n += bs

    metrics = {"loss": total_loss / max(n, 1)}
    if task == "binary":
        metrics["acc"] = correct / max(total, 1)
    else:
        if y_mu is not None and y_std is not None:
            metrics["mae_real"] = mae_real / max(n, 1)
    return metrics


@torch.no_grad()
def predict_on_loader(model, loader, device="cpu") -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds = []
    targets = []
    for x, y, _ in loader:
        x = x.to(device)
        out = model(x)
        preds.append(out.detach().cpu().numpy())
        targets.append(y.detach().cpu().numpy())
    if not preds:
        return np.empty((0, 1), dtype=np.float32), np.empty((0, 1), dtype=np.float32)
    return np.concatenate(preds, axis=0), np.concatenate(targets, axis=0)


def _resolve_repo_root() -> Path:
    try:
        return Path(__file__).resolve().parents[2]
    except NameError:
        return Path.cwd()


REPO_ROOT = _resolve_repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Data.load_data import (  # noqa: E402
    get_ticker_processed_base_dir,
    get_ticker_processed_split_dir,
    get_ticker_processed_stats_dir,
)
from Data.retrieve_data import normalize_ticker  # noqa: E402
from Features.feature_scaling import apply_scaler_from_stats  # noqa: E402


def _select_target(side: str, y_long: np.ndarray, y_short: np.ndarray) -> np.ndarray:
    side = side.strip().lower()
    if side in ("long", "up"):
        return y_long
    if side in ("short", "down"):
        return y_short
    raise ValueError(f"Unknown side: {side}")


def _is_regression_label_mode(label_mode: str) -> bool:
    return label_mode in ("mfe", "mae", "mfe_mae")


def _load_norm_stats(stats_dir: Path, dataset_name: str, x_filename: str) -> dict | None:
    x_stem = Path(x_filename).stem
    stats_path = stats_dir / f"norm_stats_{dataset_name}_{x_stem}_train.json"
    if not stats_path.exists():
        return None
    return json.loads(stats_path.read_text())


def _load_split_indices_for_xfile(
    ticker: str,
    dataset_name: str,
    x_filename: str,
) -> dict[str, np.ndarray]:
    clean = normalize_ticker(ticker)
    split_dir = get_ticker_processed_split_dir(clean) / dataset_name / Path(x_filename).stem
    train_path = split_dir / "train_idx.npy"
    val_path = split_dir / "val_idx.npy"
    test_path = split_dir / "test_idx.npy"
    missing = [p.name for p in (train_path, val_path, test_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing split files in {split_dir}: {', '.join(missing)}")
    return {
        "train": np.load(train_path),
        "val": np.load(val_path),
        "test": np.load(test_path),
    }


def _infer_split_index(
    n: int, train_idx: np.ndarray, val_idx: np.ndarray, test_idx: np.ndarray
) -> SplitIndex:
    train_idx = np.sort(train_idx)
    val_idx = np.sort(val_idx)
    test_idx = np.sort(test_idx)
    if train_idx.size == 0 or val_idx.size == 0:
        raise ValueError("Train/val splits are empty.")
    train_end = int(train_idx[-1] + 1)
    val_end = int(val_idx[-1] + 1)
    if not np.array_equal(train_idx, np.arange(0, train_end)):
        raise ValueError("Train split indices are not contiguous.")
    if not np.array_equal(val_idx, np.arange(train_end, val_end)):
        raise ValueError("Val split indices are not contiguous.")
    if not np.array_equal(test_idx, np.arange(val_end, n)):
        raise ValueError("Test split indices are not contiguous.")
    return SplitIndex(train_end=train_end, val_end=val_end)


def _load_repo_full_dataset(
    *,
    ticker: str,
    dataset_name: str,
    label_mode: str,
    x_filename: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, SplitIndex]:
    clean = normalize_ticker(ticker)
    processed_dir = get_ticker_processed_base_dir(clean)
    dataset_dir = processed_dir / "datasets" / dataset_name
    x_path = dataset_dir / x_filename
    y_path = dataset_dir / "y.parquet"

    if not x_path.exists() or not y_path.exists():
        raise FileNotFoundError(f"Missing {x_filename} or y.parquet in {dataset_dir}")

    X_df = pd.read_parquet(x_path)
    stats_dir = get_ticker_processed_stats_dir(clean)
    stats = _load_norm_stats(stats_dir, dataset_name, x_filename)
    if stats:
        X_df = apply_scaler_from_stats(X_df, stats)
    else:
        x_stem = Path(x_filename).stem
        stats_path = stats_dir / f"norm_stats_{dataset_name}_{x_stem}_train.json"
        print(f"No scaler stats found at {stats_path}; using raw features.")
    X = X_df.to_numpy(dtype=np.float32)

    y_df = pd.read_parquet(y_path)
    if label_mode == "swing":
        long_col, short_col = "long_swing_label", "short_swing_label"
    elif label_mode == "leg":
        long_col, short_col = "leg_up_label", "leg_down_label"
    elif label_mode == "mfe":
        long_col, short_col = "mfe_up_atr", "mfe_down_atr"
    elif label_mode == "mae":
        long_col, short_col = "mae_down_atr", "mae_up_atr"
    elif label_mode == "mfe_mae":
        long_col, short_col = "mfe_up_atr", "mfe_down_atr"
    else:
        raise ValueError(f"Unknown label_mode: {label_mode}")

    missing_cols = [c for c in (long_col, short_col) if c not in y_df.columns]
    if missing_cols:
        raise KeyError(f"Missing label columns in {y_path.name}: {', '.join(missing_cols)}")

    y_long = y_df[long_col].to_numpy()
    y_short = y_df[short_col].to_numpy()
    y_long = np.nan_to_num(y_long, nan=0.0, posinf=0.0, neginf=0.0)
    y_short = np.nan_to_num(y_short, nan=0.0, posinf=0.0, neginf=0.0)

    splits = _load_split_indices_for_xfile(clean, dataset_name, x_filename)
    split_idx = _infer_split_index(len(X), splits["train"], splits["val"], splits["test"])
    return X, y_long, y_short, split_idx


def _apply_feature_mask(X: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    if mask is None:
        return X
    return X[:, mask.astype(bool)]


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--x_path", type=str, default=None)  # npy: (N, C)
    ap.add_argument("--y_path", type=str, default=None)  # npy: (N,) or (N,1)
    ap.add_argument("--task", type=str, choices=["regression", "binary"], default="regression")

    ap.add_argument("--ticker", type=str, default="$SPY")
    ap.add_argument("--dataset_name", type=str, default="15min")
    ap.add_argument("--x_filename", type=str, default=None)
    ap.add_argument("--label_mode", type=str, default="mfe")
    ap.add_argument("--sides", type=str, default="long,short")

    ap.add_argument("--seq_len", type=int, default=64)
    ap.add_argument("--train_frac", type=float, default=0.8)
    ap.add_argument("--val_frac", type=float, default=0.1)

    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--n_heads", type=int, default=4)
    ap.add_argument("--n_layers", type=int, default=3)
    ap.add_argument("--d_ff", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--use_var_embedding", action="store_true")

    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)

    # GA flags
    ap.add_argument("--use_ga", action="store_true")
    ap.add_argument("--ga_population_size", type=int, default=8)
    ap.add_argument("--ga_generations", type=int, default=12)
    ap.add_argument("--ga_crossover_rate", type=float, default=0.5)
    ap.add_argument("--ga_mutation_rate", type=float, default=0.01)
    ap.add_argument("--ga_max_features", type=int, default=80)
    ap.add_argument("--ga_selection", type=str, default="tournament")
    ap.add_argument("--ga_tournament_k", type=int, default=3)
    ap.add_argument("--ga_feature_penalty", type=float, default=0.0)
    ap.add_argument("--ga_epochs", type=int, default=6)
    ap.add_argument("--ga_batch_size", type=int, default=256)
    ap.add_argument("--ga_lr", type=float, default=2e-4)
    ap.add_argument("--ga_weight_decay", type=float, default=1e-2)
    ap.add_argument("--ga_clip", type=float, default=1.0)
    return ap


def run_training(args: argparse.Namespace, *, return_predictions: bool = False) -> dict:

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    use_repo = args.x_path is None or args.y_path is None
    if use_repo:
        dataset_name = args.dataset_name
        x_filename = args.x_filename or f"X_{dataset_name}_lstm.parquet"
        X, y_long, y_short, split_idx = _load_repo_full_dataset(
            ticker=args.ticker,
            dataset_name=dataset_name,
            label_mode=args.label_mode,
            x_filename=x_filename,
        )
        task = "regression" if _is_regression_label_mode(args.label_mode) else "binary"
        sides = [s.strip() for s in args.sides.split(",") if s.strip()]
    else:
        X = np.load(args.x_path).astype(np.float32)
        y = np.load(args.y_path).astype(np.float32)
        if y.ndim == 1:
            y = y[:, None]
        y_long = y_short = y
        task = args.task
        sides = ["long"]
        N = len(X)
        train_end = int(N * args.train_frac)
        val_end = int(N * (args.train_frac + args.val_frac))
        split_idx = SplitIndex(train_end=train_end, val_end=val_end)

    results: dict[str, dict] = {}

    for side in sides:
        y_raw = _select_target(side, y_long, y_short)
        if task == "binary":
            y_raw = (y_raw > 0).astype(np.float32)

        if y_raw.ndim == 1:
            y_raw = y_raw[:, None]
        y_raw = y_raw.astype(np.float32)

        # Normalize y for regression (train stats only).
        y_mu = y_std = None
        y_train = y_raw[: split_idx.train_end]
        if task == "regression":
            y_mu = y_train.mean(axis=0, keepdims=True)
            y_std = y_train.std(axis=0, keepdims=True) + 1e-8
            y_scaled = (y_raw - y_mu) / y_std
        else:
            y_scaled = y_raw

        feature_mask = None
        ga_score = None
        if args.use_ga:
            selector = GAITransformerFeatureSelector(
                population_size=args.ga_population_size,
                generations=args.ga_generations,
                crossover_rate=args.ga_crossover_rate,
                mutation_rate=args.ga_mutation_rate,
                max_features=args.ga_max_features,
                selection=args.ga_selection,
                tournament_k=args.ga_tournament_k,
                random_state=args.seed,
                fitness_metric="acc" if task == "binary" else "neg_val_loss",
                feature_penalty=args.ga_feature_penalty,
                seq_len=args.seq_len,
                d_model=args.d_model,
                n_heads=args.n_heads,
                n_layers=args.n_layers,
                d_ff=args.d_ff,
                dropout=args.dropout,
                use_var_embedding=args.use_var_embedding,
                batch_size=args.ga_batch_size,
                epochs=args.ga_epochs,
                learning_rate=args.ga_lr,
                weight_decay=args.ga_weight_decay,
                clip=args.ga_clip,
            )

            X_train = X[: split_idx.train_end]
            y_train = y_raw[: split_idx.train_end]
            X_val = X[split_idx.train_end : split_idx.val_end]
            y_val = y_raw[split_idx.train_end : split_idx.val_end]
            selector.fit(X_train, y_train, X_val, y_val, task=task)
            feature_mask = selector.best_mask_
            ga_score = selector.best_score_
            if feature_mask is not None:
                print(
                    f"[GA-iTransformer] {side} selected {int(feature_mask.sum())}/{X.shape[1]} features"
                )
                model_dir = REPO_ROOT / "Data" / "models" / "itransformer"
                model_dir.mkdir(parents=True, exist_ok=True)
                slug = normalize_ticker(args.ticker).lower()
                mask_path = (
                    model_dir
                    / f"{slug}_{args.dataset_name}_{args.label_mode}_{side}_seq{args.seq_len}_mask.npy"
                )
                np.save(mask_path, feature_mask.astype(np.int8))

        X_masked = _apply_feature_mask(X, feature_mask)

        ds_train = WindowedTimeSeries(X_masked, y_scaled, args.seq_len, "train", split_idx)
        ds_val = WindowedTimeSeries(X_masked, y_scaled, args.seq_len, "val", split_idx)
        ds_test = WindowedTimeSeries(X_masked, y_scaled, args.seq_len, "test", split_idx)

        dl_train = DataLoader(
            ds_train, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=0
        )
        dl_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False, num_workers=0)
        dl_test = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False, num_workers=0)

        out_dim = y_scaled.shape[1]
        model = iTransformerEncoder(
            seq_len=args.seq_len,
            num_variates=X_masked.shape[1],
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            d_ff=args.d_ff,
            dropout=args.dropout,
            use_var_embedding=args.use_var_embedding,
            out_dim=out_dim,
        ).to(device)

        if task == "binary":
            loss_fn = nn.BCEWithLogitsLoss()
        else:
            loss_fn = nn.SmoothL1Loss()

        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=0.5, patience=2
        )

        stopper = EarlyStopper(patience=args.patience, min_delta=0.0)
        best_state = None
        best_val = float("inf")

        for epoch in range(1, args.epochs + 1):
            model.train()
            total = 0.0
            n = 0
            for x, yb, _ in dl_train:
                x = x.to(device)
                yb = yb.to(device)
                opt.zero_grad(set_to_none=True)
                out = model(x)
                if task == "binary":
                    target = yb.view(-1, 1)
                    loss = loss_fn(out, target)
                else:
                    target = yb.view_as(out)
                    loss = loss_fn(out, target)
                loss.backward()
                if args.clip is not None and args.clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                opt.step()
                bs = x.size(0)
                total += loss.item() * bs
                n += bs

            train_loss = total / max(n, 1)
            val_metrics = evaluate(
                model,
                dl_val,
                loss_fn,
                task,
                y_mu=torch.tensor(y_mu).to(device) if y_mu is not None else None,
                y_std=torch.tensor(y_std).to(device) if y_std is not None else None,
                device=device,
            )
            sched.step(val_metrics["loss"])

            if val_metrics["loss"] < best_val:
                best_val = val_metrics["loss"]
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

            print(f"{side.upper()} Epoch {epoch:03d} | train_loss={train_loss:.6f} | val={val_metrics}")

            if stopper.step(val_metrics["loss"]):
                print(f"{side.upper()} Early stop at epoch {epoch}, best_val={best_val:.6f}")
                break

        if best_state is not None:
            model.load_state_dict(best_state)

        test_metrics = evaluate(
            model,
            dl_test,
            loss_fn,
            task,
            y_mu=torch.tensor(y_mu).to(device) if y_mu is not None else None,
            y_std=torch.tensor(y_std).to(device) if y_std is not None else None,
            device=device,
        )
        print(f"{side.upper()} TEST: {test_metrics}")

        preds = targets = None
        test_indices = None
        if return_predictions:
            preds, targets = predict_on_loader(model, dl_test, device=device)
            test_indices = ds_test.targets.copy()
            if task == "binary":
                probs = 1.0 / (1.0 + np.exp(-preds))
                preds_out = probs
                targets_out = targets
            else:
                preds_out = preds
                targets_out = targets
                if y_mu is not None and y_std is not None:
                    preds_out = preds_out * y_std + y_mu
                    targets_out = targets_out * y_std + y_mu
            results[side] = {
                "test_metrics": test_metrics,
                "test_indices": test_indices,
                "test_pred": preds_out,
                "test_true": targets_out,
                "feature_mask": feature_mask,
                "ga_best_score": ga_score,
            }

        # Save model + meta
        model_dir = REPO_ROOT / "Data" / "models" / "itransformer"
        model_dir.mkdir(parents=True, exist_ok=True)
        slug = normalize_ticker(args.ticker).lower()
        model_path = model_dir / f"{slug}_{args.dataset_name}_{args.label_mode}_{side}_seq{args.seq_len}.pth"
        torch.save(model.state_dict(), model_path)

        meta = {
            "ticker": args.ticker,
            "dataset_name": args.dataset_name,
            "label_mode": args.label_mode,
            "side": side,
            "seq_len": args.seq_len,
            "n_features": int(X_masked.shape[1]),
            "n_features_total": int(X.shape[1]),
            "ga_best_score": ga_score,
            "use_ga": bool(args.use_ga),
            "d_model": args.d_model,
            "n_heads": args.n_heads,
            "n_layers": args.n_layers,
            "d_ff": args.d_ff,
            "dropout": args.dropout,
            "epochs": args.epochs,
            "learning_rate": args.lr,
            "train_end": int(split_idx.train_end),
            "val_end": int(split_idx.val_end),
            "task": task,
            "test_metrics": test_metrics,
        }
        meta_path = model_dir / f"{slug}_{args.dataset_name}_{args.label_mode}_{side}_seq{args.seq_len}_meta.json"
        meta_path.write_text(json.dumps(meta, indent=2))

    return results


def main():
    ap = build_arg_parser()
    args = ap.parse_args()
    run_training(args, return_predictions=False)


if __name__ == "__main__":
    main()
