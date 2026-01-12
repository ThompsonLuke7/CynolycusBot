# train_mabilstm.py
import json
import sys
from pathlib import Path

import numpy as np
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
LABEL_MODE = "swing"
MODEL_NAME = "mabilstm"
SIDES = ("long", "short")


def _resolve_repo_root() -> Path:
    try:
        return Path(__file__).resolve().parents[2]
    except NameError:
        return Path.cwd()


REPO_ROOT = _resolve_repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Data.load_data import load_dataset_splits  # noqa: E402
from Data.retrieve_data import normalize_ticker  # noqa: E402


def _select_target(side: str, y_long: np.ndarray, y_short: np.ndarray) -> np.ndarray:
    side = side.strip().lower()
    if side in ("long", "up", "bull"):
        return y_long
    if side in ("short", "down", "bear"):
        return y_short
    raise ValueError(f"Unknown side: {side}")


def _build_dataset(
    X: np.ndarray, y: np.ndarray, seq_len: int
) -> SequenceRegressionDataset | None:
    if len(X) < seq_len:
        return None
    return SequenceRegressionDataset(X, y, seq_len=seq_len)


def _update_binary_counts(logits: torch.Tensor, targets: torch.Tensor, counts: dict, threshold: float = 0.2):
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
    y_train = _select_target(side_name, y_long_train, y_short_train).astype(np.float32)
    y_val = _select_target(side_name, y_long_val, y_short_val).astype(np.float32)
    y_test = _select_target(side_name, y_long_test, y_short_test).astype(np.float32)

    train_ds = _build_dataset(X_train, y_train, seq_len)
    val_ds = _build_dataset(X_val, y_val, seq_len)
    test_ds = _build_dataset(X_test, y_test, seq_len)
    if train_ds is None or test_ds is None:
        raise ValueError("Train/test split too small for the requested seq_len.")

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False) if val_ds else None
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    pos = int((y_train == 1).sum())
    neg = int((y_train == 0).sum())
    print(f"\n=== {side_name.upper()} side ===")
    print(f"{side_name.upper()} train positives: {pos}, negatives: {neg}")

    model = MABiLSTM(input_dim=X_train.shape[1]).to(device)
    pos_weight = torch.tensor([neg / max(pos, 1)], device=device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    epochs = 30
    for epoch in range(epochs):
        # ----- train -----
        model.train()
        train_loss = 0.0
        for xb, yb in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [train]"):
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()
            preds, _ = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * xb.size(0)

        train_loss /= len(train_loader.dataset)

        # ----- validate -----
        if val_loader is None:
            print(f"Epoch {epoch+1}: train_loss={train_loss:.5f}")
        else:
            model.eval()
            val_loss = 0.0
            counts = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
            with torch.no_grad():
                for xb, yb in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [val]"):
                    xb, yb = xb.to(device), yb.to(device)
                    preds, _ = model(xb)
                    loss = criterion(preds, yb)
                    val_loss += loss.item() * xb.size(0)
                    _update_binary_counts(preds, yb, counts)

            val_loss /= len(val_loader.dataset)
            val_acc, val_f1 = _metrics_from_counts(counts)
            print(
                f"Epoch {epoch+1}: train_loss={train_loss:.5f}, "
                f"val_loss={val_loss:.5f}, val_acc={val_acc:.4f}, val_f1={val_f1:.4f}"
            )

    # ----- test -----
    model.eval()
    test_loss = 0.0
    counts = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    with torch.no_grad():
        for xb, yb in tqdm(test_loader, desc="Test"):
            xb, yb = xb.to(device), yb.to(device)
            preds, _ = model(xb)
            loss = criterion(preds, yb)
            test_loss += loss.item() * xb.size(0)
            _update_binary_counts(preds, yb, counts)

    test_loss /= len(test_loader.dataset)
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
    splits = load_dataset_splits(
        ticker=TICKER,
        dataset_name=DATASET_NAME,
        label_mode=LABEL_MODE,
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
