from __future__ import annotations

import torch
from torch import nn


class FiLMGenerator(nn.Module):
    def __init__(self, static_dim, channels, hidden=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(static_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels * 2),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
        with torch.no_grad():
            self.mlp[-1].bias[:channels] = 1.0

    def forward(self, s):
        p = self.mlp(s)
        return p.chunk(2, dim=-1)
