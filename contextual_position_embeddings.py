"""
https://arxiv.org/pdf/2405.18719
"""

import torch
from torch import nn
import torch.nn.functional as F


class CoPE(nn.Module):
    def __init__(self, T, C):
        super().__init__()
        self.T = T
        self.pos_emb = nn.Parameter(torch.zeros(1, C, T))

    def forward(self, Q, A):
        B, T, C = Q.shape
        gates = F.sigmoid(A)
        pos = gates.flip(-1).cumsum(dim=-1).flip(-1).clamp(max=self.T - 1)
        pos_ceil = pos.ceil().long()
        pos_floor = pos.floor().long()
        logits_int = Q @ self.pos_emb#[..., :T]
        logits_ceil = logits_int.gather(-1, pos_ceil)
        logits_floor = logits_int.gather(-1, pos_floor)
        w = pos - pos_floor
        return logits_ceil * w + logits_floor * (1 - w)
        
