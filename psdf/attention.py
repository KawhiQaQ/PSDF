"""Multi-head self-attention with separate Q, K and V projections."""
from __future__ import annotations

import torch
from torch import nn


class MultiheadSelfAttention(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        if dim % heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim**-0.5
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, n_tokens, dim = x.shape
        q = self.q_proj(x).view(bsz, n_tokens, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, n_tokens, self.heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, n_tokens, self.heads, self.head_dim).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.dropout(attn.softmax(dim=-1))
        out = (attn @ v).transpose(1, 2).contiguous().view(bsz, n_tokens, dim)
        return self.out_proj(out)
