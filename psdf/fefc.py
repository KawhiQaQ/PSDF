"""Harmonic frequency-energy feature mapping."""
from __future__ import annotations

import torch
from torch import nn


class FEFC(nn.Module):
    def __init__(self, freq_bins: int, embed_dim: int) -> None:
        super().__init__()
        self.proj_harmonic = nn.Linear(freq_bins, embed_dim, bias=False)
        self.proj_out = nn.Linear(embed_dim * 2, embed_dim)
        self.act = nn.ReLU()

    def forward(self, energy: torch.Tensor) -> torch.Tensor:
        hidden = self.proj_harmonic(energy)
        harmonic = torch.cat([torch.cos(hidden), torch.sin(hidden)], dim=-1)
        return self.act(self.proj_out(harmonic))
