from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .film import FiLMGenerator


class FiLMResBlock(nn.Module):
    def __init__(self, in_c, out_c, ks, dil, static_dim, drop=0.3):
        super().__init__()
        pad = (ks - 1) * dil
        self.c1 = nn.Conv1d(in_c, out_c, ks, padding=pad, dilation=dil)
        self.bn1 = nn.BatchNorm1d(out_c)
        self.f1 = FiLMGenerator(static_dim, out_c)
        self.c2 = nn.Conv1d(out_c, out_c, ks, padding=pad, dilation=dil)
        self.bn2 = nn.BatchNorm1d(out_c)
        self.f2 = FiLMGenerator(static_dim, out_c)
        self.drop = nn.Dropout(drop)
        self.res = nn.Conv1d(in_c, out_c, 1) if in_c != out_c else nn.Identity()

    def forward(self, x, s, disable_film: bool = False):
        T = x.size(2)
        o = self.bn1(self.c1(x)[:, :, :T])
        if not disable_film:
            g, b = self.f1(s)
            o = g.unsqueeze(-1) * o + b.unsqueeze(-1)
        o = self.drop(F.relu(o))
        o = self.bn2(self.c2(o)[:, :, :T])
        if not disable_film:
            g, b = self.f2(s)
            o = g.unsqueeze(-1) * o + b.unsqueeze(-1)
        o = self.drop(F.relu(o))
        return F.relu(o + self.res(x))
