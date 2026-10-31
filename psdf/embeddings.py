import math

import torch


def sinusoidal_position_encoding(length: int, dim: int, device: torch.device) -> torch.Tensor:
    pos = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, dim, 2, device=device, dtype=torch.float32) * (-math.log(10000.0) / dim))
    pe = torch.zeros(length, dim, device=device)
    pe[:, 0::2] = torch.sin(pos * div)
    if dim % 2 == 0:
        pe[:, 1::2] = torch.cos(pos * div)
    else:
        pe[:, 1::2] = torch.cos(pos * div[:-1])
    return pe
