"""EEG spectral token extraction and channel/time identity encoding."""
from __future__ import annotations

import torch
from torch import nn

from .config import PSDFConfig
from .embeddings import sinusoidal_position_encoding
from .fefc import FEFC


class FMIE(nn.Module):
    def __init__(self, cfg: PSDFConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.segment_len = int(round(cfg.segment_seconds * cfg.fs))
        self.stride = int(round((cfg.segment_seconds - cfg.overlap_seconds) * cfg.fs))
        if self.segment_len <= 1:
            raise ValueError("segment_seconds is too small")
        if self.stride <= 0:
            raise ValueError("overlap_seconds must be smaller than segment_seconds")
        freqs = torch.fft.rfftfreq(self.segment_len, d=1.0 / cfg.fs)
        # Eq. (3): strictly positive FFT bins, including Nyquist for even N.
        freq_mask = freqs > 0
        self.register_buffer("freq_index", torch.where(freq_mask)[0], persistent=False)
        self.freq_bins = int(freq_mask.sum().item())
        self.fefc = FEFC(self.freq_bins, cfg.embed_dim)
        self.channel_embedding = nn.Embedding(cfg.n_channels, cfg.embed_dim)
        if cfg.channel_embedding_std != 1.0:
            with torch.no_grad():
                self.channel_embedding.weight.mul_(cfg.channel_embedding_std)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or not x.is_floating_point():
            raise ValueError("Expected floating EEG tensor [batch, channels, time]")
        bsz, channels, n_time = x.shape
        if channels != self.cfg.n_channels:
            raise ValueError(f"Expected {self.cfg.n_channels} EEG channels, got {channels}")
        if n_time < self.segment_len:
            raise ValueError(f"Trial length {n_time} is shorter than segment length {self.segment_len}")

        segments = x.unfold(dimension=-1, size=self.segment_len, step=self.stride)
        n_windows = segments.shape[2]
        segments = segments.contiguous().view(bsz * channels * n_windows, self.segment_len)
        spec = torch.fft.rfft(segments, dim=-1)
        spec = spec.index_select(dim=-1, index=self.freq_index.to(spec.device))
        energy = spec.abs().pow(2)
        if self.cfg.spectral_transform == 'ortho_power':
            energy = energy / self.segment_len
        elif self.cfg.spectral_transform == 'log_power':
            energy = torch.log1p(energy / self.segment_len)
        emb = self.fefc(energy).view(bsz, channels, n_windows, self.cfg.embed_dim)

        if self.cfg.normalize_content:
            emb = torch.nn.functional.layer_norm(emb, (self.cfg.embed_dim,))
        return emb.reshape(bsz, channels*n_windows, self.cfg.embed_dim)

    def add_identity(self, tokens: torch.Tensor) -> torch.Tensor:
        bsz, length, dim = tokens.shape
        channels = self.cfg.n_channels
        if dim != self.cfg.embed_dim or length % channels:
            raise ValueError('Expected channel-major content tokens')
        n_windows = length // channels
        emb = tokens.reshape(bsz, channels, n_windows, dim)
        channel_ids = torch.arange(channels, device=tokens.device)
        channel_emb = self.channel_embedding(channel_ids).view(1, channels, 1, dim)
        pos_emb = sinusoidal_position_encoding(n_windows, dim, tokens.device).view(1, 1, n_windows, dim)
        return self.dropout((emb + channel_emb + pos_emb).reshape(bsz, length, dim))
