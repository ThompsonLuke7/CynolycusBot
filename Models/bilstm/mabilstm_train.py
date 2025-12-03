# train_mabilstm.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import numpy as np

from dataset_mabilstm import SequenceRegressionDataset
from model_mabilstm import MABiLSTM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def r2_score_torch(y_true, y_pred):
    y_true_mean = torch.mean(y_true)
    ss_tot = torch.sum((y_true - y_true_mean) ** 2)
    ss_res = torch.sum((y_true - y_pred) ** 2)
    return 1 - ss_res / ss_tot

def main():
    # load your preprocessed daily data
    X = np.load("X_spy_daily.npy")    # any features you built (can be OHLCV + TA)
    y = np.load("y_spy_close.npy")    # e.g. raw close or normalized close

    seq_len = 30
    dataset = SequenceRegressionDataset(X, y, seq_len=seq_len)

    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=64, shuffle=False)

    model = MABiLSTM(input_dim=X.shape[1]).to(device)

    criterion = nn.MSELoss()
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
        model.eval()
        val_loss = 0.0
        val_r2 = 0.0
        with torch.no_grad():
            for xb, yb in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [val]"):
                xb, yb = xb.to(device), yb.to(device)
                preds, _ = model(xb)
                loss = criterion(preds, yb)
                val_loss += loss.item() * xb.size(0)
                val_r2 += r2_score_torch(yb, preds) * xb.size(0)

        val_loss /= len(val_loader.dataset)
        val_r2 /= len(val_loader.dataset)

        print(
            f"Epoch {epoch+1}: train_loss={train_loss:.5f}, "
            f"val_loss={val_loss:.5f}, val_R2={val_r2:.4f}"
        )

    torch.save(model.state_dict(), "mabilstm_spy_daily.pth")

if __name__ == "__main__":
    main()
