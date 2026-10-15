"""Transformer blocks and the shared ViT backbone."""
from __future__ import annotations

import torch
from torch import nn

from .attention import MultiheadSelfAttention
from .config import PSDFConfig


class TransformerBlock(nn.Module):
    def __init__(self, cfg: PSDFConfig) -> None:
        super().__init__()
        dim = cfg.embed_dim
        hidden = int(round(dim * cfg.mlp_ratio))
        self.norm1 = nn.LayerNorm(dim, eps=cfg.layer_norm_eps)
        self.attn = MultiheadSelfAttention(dim, cfg.num_heads, cfg.dropout)
        self.norm2 = nn.LayerNorm(dim, eps=cfg.layer_norm_eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class ViTBackbone(nn.Module):
    def __init__(self, cfg: PSDFConfig) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.depth)])
        self.norm = nn.LayerNorm(cfg.embed_dim, eps=cfg.layer_norm_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return self.norm(x)
