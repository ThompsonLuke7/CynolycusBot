# model_mabilstm.py
import torch
import torch.nn as nn


class MABiLSTM(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,      # matches paper's best setting
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

        # Attention block.
        attn_dim = hidden_dim * self.num_directions
        self.attn_query = nn.Linear(attn_dim, attn_dim)
        self.attn_key = nn.Linear(attn_dim, attn_dim)

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
        self._init_weights()

    def _init_weights(self) -> None:
        # Explicit Xavier init to match the stated setup.
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        for name, param in self.lstm.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def forward(self, x):
        """
        x: (batch, seq_len, input_dim)
        """
        lstm_out, _ = self.lstm(x)   # (batch, seq_len, hidden*2)

        # Attention scores over time.
        query = self.attn_query(lstm_out[:, -1, :])                 # (batch, hidden*2)
        keys = self.attn_key(lstm_out)                              # (batch, seq_len, hidden*2)
        scores = (keys * query.unsqueeze(1)).sum(dim=-1)            # (batch, seq_len)
        scores = scores / (keys.size(-1) ** 0.5)

        attn_weights = torch.softmax(scores, dim=1)                  # (batch, seq_len)
        attn_weights_expanded = attn_weights.unsqueeze(-1)           # (batch, seq_len, 1)

        # Context vector = sum_t a_t * h_t.
        context = (lstm_out * attn_weights_expanded).sum(dim=1)      # (batch, hidden*2)

        context = self.dropout(context)
        out = self.mlp(context)                                      # (batch, 1)

        return out.squeeze(-1), attn_weights    # outputs: (batch,), (batch, seq_len)
