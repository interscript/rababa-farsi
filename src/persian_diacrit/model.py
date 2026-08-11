"""Persian diacritization model — char-level Transformer encoder + multi-head.

Architecture:
  - Char-level Transformer encoder (6L, 384d, same as rababa Arabic)
  - Head 1: haraqat classification (10 classes)
  - Head 2: ezafe detection (binary — is there an ezafe at this position?)

The ezafe head is the unique Persian innovation: it detects the unwritten
/e/ vowel that connects words (e.g.,ketāb-e olm = "book of science").
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn

from .constants import NUM_HARAQAT


class PersianDiacritModel(nn.Module):
    """Char-level Transformer with haraqat + ezafe heads."""

    def __init__(
        self,
        vocab_size: int = 100,
        dim: int = 384,
        layers: int = 6,
        heads: int = 6,
        ff_dim: int = 1536,
        dropout: float = 0.1,
        max_len: int = 200,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim, padding_idx=0)
        self.pos_encoding = nn.Parameter(torch.randn(1, max_len, dim) * 0.02)
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.norm = nn.LayerNorm(dim)

        # Head 1: haraqat (10 classes)
        self.haraqat_head = nn.Linear(dim, NUM_HARAQAT)

        # Head 2: ezafe (binary — 0=no ezafe, 1=ezafe)
        self.ezafe_head = nn.Linear(dim, 2)

    def forward(
        self,
        src: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (haraqat_logits, ezafe_logits).

        Args:
            src: (B, T) char IDs.
            src_key_padding_mask: (B, T) True for pad positions.

        Returns:
            haraqat_logits: (B, T, NUM_HARAQAT)
            ezafe_logits: (B, T, 2)
        """
        T = src.size(1)
        x = self.embedding(src)
        x = x + self.pos_encoding[:, :T]
        x = self.dropout(x)
        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        x = self.norm(x)

        haraqat_logits = self.haraqat_head(x)
        ezafe_logits = self.ezafe_head(x)
        return haraqat_logits, ezafe_logits

    def head_names(self) -> list[str]:
        return ["haraqat", "ezafe"]
