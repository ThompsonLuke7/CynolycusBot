# dataset_mabilstm.py
import numpy as np
import torch
from torch.utils.data import Dataset

class SequenceRegressionDataset(Dataset):
    def __init__(self, X, y, seq_len):
        """
        X: (N, num_features)
        y: (N,)      # continuous target: close price
        We create samples: [t-seq_len+1 .. t] -> y[t]
        """
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)
        self.seq_len = seq_len

        self.max_idx = len(self.X) - 1

    def __len__(self):
        # you need seq_len elements before each label index
        return self.max_idx - self.seq_len + 1

    def __getitem__(self, idx):
        # label index
        t = idx + self.seq_len - 1
        x_window = self.X[t - self.seq_len + 1 : t + 1]      # (seq_len, num_features)
        y_target = self.y[t]                                 # scalar

        return torch.from_numpy(x_window), torch.tensor(y_target)
