# itransformer_model.py
from __future__ import annotations

import torch
import torch.nn as nn


class iTransformerEncoder(nn.Module):
    """
    Inverted Transformer encoder for multivariate time-series.
    Treat each variable as a token; embed over the time axis.
    """

    def __init__(
        self,
        *,
        seq_len: int,
        num_variates: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        d_ff: int = 256,
        dropout: float = 0.1,
        use_var_embedding: bool = False,
        out_dim: int = 1,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        self.seq_len = int(seq_len)
        self.num_variates = int(num_variates)
        self.use_var_embedding = bool(use_var_embedding)

        self.input_proj = nn.Linear(self.seq_len, d_model)
        if self.use_var_embedding:
            self.var_embed = nn.Parameter(torch.zeros(1, self.num_variates, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            norm=nn.LayerNorm(d_model),
        )

        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, out_dim),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.input_proj.weight)
        if self.input_proj.bias is not None:
            nn.init.zeros_(self.input_proj.bias)
        if self.use_var_embedding:
            nn.init.normal_(self.var_embed, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, num_variates)
        """
        if x.ndim != 3:
            raise ValueError("Expected input shape (batch, seq_len, num_variates)")
        if x.size(1) != self.seq_len:
            raise ValueError(f"Expected seq_len={self.seq_len}, got {x.size(1)}")
        if x.size(2) != self.num_variates:
            raise ValueError(
                f"Expected num_variates={self.num_variates}, got {x.size(2)}"
            )

        x = x.transpose(1, 2)
        x = self.input_proj(x)
        if self.use_var_embedding:
            x = x + self.var_embed

        x = self.dropout(x)
        x = self.encoder(x)
        pooled = x.mean(dim=1)
        out = self.head(pooled)
        return out
