# model_mabilstm.py
import torch
import torch.nn as nn

class MABiLSTM(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 1024,      # matches paper's best setting
        lstm_layers: int = 1,
        mlp_hidden_dim: int = 256,
        mlp_hidden_dim2: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_directions = 2

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
        )

        # attention: score_t = (W_q h_t) · (W_k h_t)  (simplified to linear -> scalar)
        # paper uses dot product q^T k; we implement a simple linear scoring over h_t
        self.attn_linear = nn.Linear(hidden_dim * self.num_directions, 1)

        self.dropout = nn.Dropout(dropout)

        # MLP head (regression output)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * self.num_directions, mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim2, 1),   # scalar price
        )

    def forward(self, x):
        """
        x: (batch, seq_len, input_dim)
        """
        lstm_out, _ = self.lstm(x)   # (batch, seq_len, hidden*2)

        # attention scores over time
        scores = self.attn_linear(torch.tanh(lstm_out)).squeeze(-1)  # (batch, seq_len)

        attn_weights = torch.softmax(scores, dim=1)                  # (batch, seq_len)
        attn_weights_expanded = attn_weights.unsqueeze(-1)           # (batch, seq_len, 1)

        # context vector = Σ α_t h_t
        context = (lstm_out * attn_weights_expanded).sum(dim=1)      # (batch, hidden*2)

        context = self.dropout(context)
        out = self.mlp(context)                                      # (batch, 1)

        return out.squeeze(-1), attn_weights    # outputs: (batch,), (batch, seq_len)
