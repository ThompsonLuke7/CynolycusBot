# itransformer_dataset.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class SplitIndex:
    train_end: int          # exclusive
    val_end: int            # exclusive (train_end < val_end <= N)
    # test is [val_end, N)


class WindowedTimeSeries(Dataset):
    """
    Builds windows over full X, but assigns samples to a split based on target time index t.
    Window: X[t-seq_len+1 : t+1], label: y[t].
    No future leakage. Val/test get full left context from earlier data.
    """
    def __init__(
        self,
        X: np.ndarray,                  # (N, C)
        y: np.ndarray,                  # (N,) or (N, K)
        seq_len: int,
        split: str,                     # "train" | "val" | "test"
        split_index: SplitIndex,
        sample_weight: Optional[np.ndarray] = None,  # (N,)
        device: Optional[str] = None,
    ):
        assert X.ndim == 2, "X must be (N, C)"
        assert y.shape[0] == X.shape[0], "y must align with X"
        assert split in {"train", "val", "test"}
        assert seq_len >= 2

        self.X = X
        self.y = y
        self.seq_len = seq_len
        self.sample_weight = sample_weight
        self.device = device

        N = X.shape[0]
        # valid targets t must have enough left history
        t_min = seq_len - 1
        t_max = N - 1

        if split == "train":
            t0, t1 = t_min, split_index.train_end - 1
        elif split == "val":
            t0, t1 = split_index.train_end, split_index.val_end - 1
        else:
            t0, t1 = split_index.val_end, t_max

        # If val/test start before t_min, clamp
        t0 = max(t0, t_min)
        self.targets = np.arange(t0, t1 + 1, dtype=np.int64)

        if len(self.targets) == 0:
            raise ValueError(f"No samples for split={split}. Check seq_len and split boundaries.")

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        t = int(self.targets[i])
        x_win = self.X[t - self.seq_len + 1 : t + 1]   # (seq_len, C)
        y_t = self.y[t]

        x = torch.from_numpy(x_win).float()
        y = torch.from_numpy(np.array(y_t)).float()

        if self.sample_weight is not None:
            w = torch.tensor(float(self.sample_weight[t]), dtype=torch.float32)
        else:
            # Return a tensor so default_collate doesn't choke on None.
            w = torch.tensor(1.0, dtype=torch.float32)

        if self.device is not None:
            x = x.to(self.device)
            y = y.to(self.device)
            if w is not None:
                w = w.to(self.device)

        return x, y, w
