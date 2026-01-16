# train_mabilstm.py
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from mabilstm_dataset import SequenceRegressionDataset
from mabilstm_model import MABiLSTM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TICKER = "$SPY"
DATASET_NAME = "15min"
LABEL_MODE = "mfe"
MODEL_NAME = "mabilstm"
SIDES = ("long", "short")
X_FILENAME = f"X_{DATASET_NAME}_lstm.parquet"
WEIGHT_TOP_PCT = 0.2
WEIGHT_BOOST = 2.0


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


def _build_dataset(
    X: np.ndarray, y: np.ndarray, seq_len: int, sample_weights: np.ndarray | None = None
) -> SequenceRegressionDataset | None:
    if len(X) < seq_len:
        return None
    return SequenceRegressionDataset(X, y, seq_len=seq_len, sample_weights=sample_weights)


def _load_norm_stats(
    stats_dir: Path, dataset_name: str, x_filename: str
) -> dict | None:
    x_stem = Path(x_filename).stem
    stats_path = stats_dir / f"norm_stats_{dataset_name}_{x_stem}_train.json"
    if not stats_path.exists():
        return None
    return json.loads(stats_path.read_text())


def _zscore_stats(values: np.ndarray) -> tuple[float, float]:
    mean = float(np.nanmean(values))
    std = float(np.nanstd(values))
    if not np.isfinite(std) or std < 1e-8:
        std = 1.0
    if not np.isfinite(mean):
        mean = 0.0
    return mean, std


def _zscore(values: np.ndarray, mean: float, std: float) -> np.ndarray:
    return (values - mean) / std


def _sample_weights_for_targets(
    targets: np.ndarray,
    *,
    label_mode: str,
    top_pct: float = WEIGHT_TOP_PCT,
    boost: float = WEIGHT_BOOST,
    threshold: float | None = None,
) -> np.ndarray:
    weights = np.ones_like(targets, dtype=np.float32)
    if not _is_regression_label_mode(label_mode):
        return weights
    clean = targets.astype(np.float32)
    if label_mode == "mae":
        clean = np.abs(clean)
    mask = np.isfinite(clean)
    if not mask.any():
        return weights
    if threshold is None:
        threshold = float(np.quantile(clean[mask], 1.0 - top_pct))
    if not np.isfinite(threshold):
        return weights
    weights[clean >= threshold] = float(boost)
    return weights


def _weight_threshold_for_targets(
    targets: np.ndarray,
    *,
    label_mode: str,
    top_pct: float = WEIGHT_TOP_PCT,
) -> float | None:
    if not _is_regression_label_mode(label_mode):
        return None
    clean = targets.astype(np.float32)
    if label_mode == "mae":
        clean = np.abs(clean)
    mask = np.isfinite(clean)
    if not mask.any():
        return None
    threshold = float(np.quantile(clean[mask], 1.0 - top_pct))
    if not np.isfinite(threshold):
        return None
    return threshold


def _unpack_batch(
    batch: tuple[torch.Tensor, ...],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    if len(batch) == 2:
        xb, yb = batch
        wb = None
    else:
        xb, yb, wb = batch
    xb = xb.to(device)
    yb = yb.to(device)
    if wb is not None:
        wb = wb.to(device)
    return xb, yb, wb


def _update_binary_counts(logits: torch.Tensor, targets: torch.Tensor, counts: dict, threshold: float = 0.5):
    probs = torch.sigmoid(logits)
    preds = probs >= threshold
    labels = targets >= 0.5
    counts["tp"] += int(((preds == 1) & (labels == 1)).sum().item())
    counts["fp"] += int(((preds == 1) & (labels == 0)).sum().item())
    counts["fn"] += int(((preds == 0) & (labels == 1)).sum().item())
    counts["tn"] += int(((preds == 0) & (labels == 0)).sum().item())


def _metrics_from_counts(counts: dict) -> tuple[float, float]:
    total = counts["tp"] + counts["fp"] + counts["fn"] + counts["tn"]
    acc = (counts["tp"] + counts["tn"]) / max(total, 1)
    denom = 2 * counts["tp"] + counts["fp"] + counts["fn"]
    f1 = (2 * counts["tp"]) / denom if denom else 0.0
    return acc, f1


def _load_split_indices_for_xfile(
    ticker: str,
    dataset_name: str,
    x_filename: str,
) -> dict[str, np.ndarray]:
    clean = normalize_ticker(ticker)
    split_dir = (
        get_ticker_processed_split_dir(clean)
        / dataset_name
        / Path(x_filename).stem
    )
    train_path = split_dir / "train_idx.npy"
    val_path = split_dir / "val_idx.npy"
    test_path = split_dir / "test_idx.npy"

    missing = [p.name for p in (train_path, val_path, test_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing split files in {split_dir}: {', '.join(missing)}"
        )

    return {
        "train": np.load(train_path),
        "val": np.load(val_path),
        "test": np.load(test_path),
    }


def _load_dataset_splits_for_xfile(
    *,
    ticker: str,
    dataset_name: str,
    label_mode: str,
    x_filename: str,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
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
    elif label_mode in ("mfe", "mae", "mfe_mae"):
        long_col, short_col = "mfe_up_atr", "mfe_down_atr"
    else:
        raise ValueError(f"Unknown label_mode: {label_mode}")

    missing_cols = [c for c in (long_col, short_col) if c not in y_df.columns]
    if missing_cols:
        raise KeyError(
            f"Missing label columns in {y_path.name}: {', '.join(missing_cols)}"
        )

    if _is_regression_label_mode(label_mode):
        y_long = y_df[long_col].to_numpy(dtype=np.float32)
        y_short = y_df[short_col].to_numpy(dtype=np.float32)
        y_long = np.nan_to_num(y_long, nan=0.0, posinf=0.0, neginf=0.0)
        y_short = np.nan_to_num(y_short, nan=0.0, posinf=0.0, neginf=0.0)
    else:
        y_long = y_df[long_col].to_numpy(dtype=np.int64)
        y_short = y_df[short_col].to_numpy(dtype=np.int64)

    splits = _load_split_indices_for_xfile(clean, dataset_name, x_filename)
    return {name: (X[idx], y_long[idx], y_short[idx]) for name, idx in splits.items()}


def train_and_eval_side(
    side_name: str,
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    y_long_train: np.ndarray,
    y_short_train: np.ndarray,
    y_long_val: np.ndarray,
    y_short_val: np.ndarray,
    y_long_test: np.ndarray,
    y_short_test: np.ndarray,
    seq_len: int,
) -> dict:
    y_train_raw = _select_target(side_name, y_long_train, y_short_train).astype(np.float32)
    y_val_raw = _select_target(side_name, y_long_val, y_short_val).astype(np.float32)
    y_test_raw = _select_target(side_name, y_long_test, y_short_test).astype(np.float32)
    use_regression = _is_regression_label_mode(LABEL_MODE)

    if use_regression:
        target_mean, target_std = _zscore_stats(y_train_raw)
        y_train = _zscore(y_train_raw, target_mean, target_std)
        y_val = _zscore(y_val_raw, target_mean, target_std)
        y_test = _zscore(y_test_raw, target_mean, target_std)
        weight_threshold = _weight_threshold_for_targets(
            y_train_raw, label_mode=LABEL_MODE
        )
        train_weights = _sample_weights_for_targets(
            y_train_raw, label_mode=LABEL_MODE, threshold=weight_threshold
        )
        val_weights = _sample_weights_for_targets(
            y_val_raw, label_mode=LABEL_MODE, threshold=weight_threshold
        )
        test_weights = _sample_weights_for_targets(
            y_test_raw, label_mode=LABEL_MODE, threshold=weight_threshold
        )
    else:
        target_mean, target_std = 0.0, 1.0
        y_train, y_val, y_test = y_train_raw, y_val_raw, y_test_raw
        train_weights = val_weights = test_weights = None
        weight_threshold = None

    train_ds = _build_dataset(X_train, y_train, seq_len, sample_weights=train_weights)
    val_ds = _build_dataset(X_val, y_val, seq_len, sample_weights=val_weights)
    test_ds = _build_dataset(X_test, y_test, seq_len, sample_weights=test_weights)
    if train_ds is None or test_ds is None:
        raise ValueError("Train/test split too small for the requested seq_len.")

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False) if val_ds else None
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    pos = int((y_train == 1).sum())
    neg = int((y_train == 0).sum())
    print(f"\n=== {side_name.upper()} side ===")
    if use_regression:
        print(
            f"{side_name.upper()} train target mean={target_mean:.4f}, "
            f"std={target_std:.4f} (z-score stats)"
        )
    else:
        print(f"{side_name.upper()} train positives: {pos}, negatives: {neg}")

    model = MABiLSTM(input_dim=X_train.shape[1]).to(device)
    if use_regression:
        criterion = nn.SmoothL1Loss(reduction="none")
    else:
        pos_weight = torch.tensor([neg / max(pos, 1)], device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    epochs = 30
    for epoch in range(epochs):
        # ----- train -----
        model.train()
        train_loss = 0.0
        train_weight_total = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [train]"):
            xb, yb, wb = _unpack_batch(batch, device)

            optimizer.zero_grad()
            preds, _ = model(xb)
            if wb is not None:
                loss_sum = (criterion(preds, yb) * wb).sum()
                weight_sum = torch.clamp(wb.sum(), min=1.0)
                loss = loss_sum / weight_sum
                train_loss += float(loss_sum.item())
                train_weight_total += float(weight_sum.item())
            else:
                loss = criterion(preds, yb).mean()
                train_loss += loss.item() * xb.size(0)
                train_weight_total += xb.size(0)
            loss.backward()
            optimizer.step()

        if train_weight_total:
            train_loss /= train_weight_total

        # ----- validate -----
        if val_loader is None:
            print(f"Epoch {epoch+1}: train_loss={train_loss:.5f}")
        else:
            model.eval()
            val_loss = 0.0
            val_weight_total = 0.0
            counts = {"tp": 0, "fp": 0, "fn": 0, "tn": 0} if not use_regression else None
            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [val]"):
                    xb, yb, wb = _unpack_batch(batch, device)
                    preds, _ = model(xb)
                    if wb is not None:
                        loss_sum = (criterion(preds, yb) * wb).sum()
                        weight_sum = torch.clamp(wb.sum(), min=1.0)
                        loss = loss_sum / weight_sum
                        val_loss += float(loss_sum.item())
                        val_weight_total += float(weight_sum.item())
                    else:
                        loss = criterion(preds, yb).mean()
                        val_loss += loss.item() * xb.size(0)
                        val_weight_total += xb.size(0)
                    if counts is not None:
                        _update_binary_counts(preds, yb, counts)

            if val_weight_total:
                val_loss /= val_weight_total
            if use_regression:
                print(
                    f"Epoch {epoch+1}: train_loss={train_loss:.5f}, "
                    f"val_loss={val_loss:.5f}"
                )
            else:
                val_acc, val_f1 = _metrics_from_counts(counts)
                print(
                    f"Epoch {epoch+1}: train_loss={train_loss:.5f}, "
                    f"val_loss={val_loss:.5f}, val_acc={val_acc:.4f}, val_f1={val_f1:.4f}"
                )

    # ----- test -----
    model.eval()
    test_loss = 0.0
    test_weight_total = 0.0
    counts = {"tp": 0, "fp": 0, "fn": 0, "tn": 0} if not use_regression else None
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Test"):
            xb, yb, wb = _unpack_batch(batch, device)
            preds, _ = model(xb)
            if wb is not None:
                loss_sum = (criterion(preds, yb) * wb).sum()
                weight_sum = torch.clamp(wb.sum(), min=1.0)
                loss = loss_sum / weight_sum
                test_loss += float(loss_sum.item())
                test_weight_total += float(weight_sum.item())
            else:
                loss = criterion(preds, yb).mean()
                test_loss += loss.item() * xb.size(0)
                test_weight_total += xb.size(0)
            if counts is not None:
                _update_binary_counts(preds, yb, counts)

    if test_weight_total:
        test_loss /= test_weight_total
    if use_regression:
        print(f"Test: loss={test_loss:.5f}")
        test_acc, test_f1 = 0.0, 0.0
    else:
        test_acc, test_f1 = _metrics_from_counts(counts)
        print(f"Test: loss={test_loss:.5f}, acc={test_acc:.4f}, f1={test_f1:.4f}")

    model_dir = REPO_ROOT / "Data" / "models" / MODEL_NAME
    model_dir.mkdir(parents=True, exist_ok=True)
    slug = normalize_ticker(TICKER).lower()
    side = side_name.strip().lower()
    model_path = model_dir / f"{slug}_{DATASET_NAME}_{LABEL_MODE}_{side}_seq{seq_len}.pth"
    torch.save(model.state_dict(), model_path)

    meta = {
        "ticker": TICKER,
        "dataset_name": DATASET_NAME,
        "label_mode": LABEL_MODE,
        "target_side": side_name,
        "seq_len": seq_len,
        "n_features": int(X_train.shape[1]),
        "epochs": epochs,
        "learning_rate": 1e-4,
        "train_size": len(train_ds),
        "val_size": len(val_ds) if val_ds else 0,
        "test_size": len(test_ds),
        "target_mean": target_mean,
        "target_std": target_std,
        "weight_top_pct": WEIGHT_TOP_PCT if use_regression else None,
        "weight_boost": WEIGHT_BOOST if use_regression else None,
        "weight_threshold": weight_threshold if use_regression else None,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "test_f1": test_f1,
    }
    meta_path = model_dir / f"{slug}_{DATASET_NAME}_{LABEL_MODE}_{side}_seq{seq_len}_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))

    return {
        "side": side_name,
        "model_path": model_path,
        "meta_path": meta_path,
    }


def main():
    splits = _load_dataset_splits_for_xfile(
        ticker=TICKER,
        dataset_name=DATASET_NAME,
        label_mode=LABEL_MODE,
        x_filename=X_FILENAME,
    )

    X_train, y_long_train, y_short_train = splits["train"]
    X_val, y_long_val, y_short_val = splits["val"]
    X_test, y_long_test, y_short_test = splits["test"]

    print(f"Loaded train split: {X_train.shape}")
    print(f"Loaded val split:   {X_val.shape}")
    print(f"Loaded test split:  {X_test.shape}")

    seq_len = 30
    for side in SIDES:
        train_and_eval_side(
            side,
            X_train,
            X_val,
            X_test,
            y_long_train,
            y_short_train,
            y_long_val,
            y_short_val,
            y_long_test,
            y_short_test,
            seq_len,
        )


if __name__ == "__main__":
    main()
