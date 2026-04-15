from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class AttentionArtifacts:
    attn: torch.Tensor  # [B, H, S, S]
    scores: torch.Tensor  # [B, H, S, S]


class TimeAwareAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_time_bins: int = 1000):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model must be divisible by n_heads, got {d_model=} {n_heads=}")
        self.n_heads = int(n_heads)
        self.d_k = d_model // n_heads

        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.time_proj = nn.Embedding(int(max_time_bins), self.n_heads)
        self.out = nn.Linear(d_model, d_model)

    def forward(
        self,
        x: torch.Tensor,
        time_deltas: torch.Tensor,
        mask: torch.Tensor | None = None,
        *,
        return_attn: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, AttentionArtifacts]:
        b, s, d = x.size()
        q = self.q(x).view(b, s, self.n_heads, self.d_k).transpose(1, 2)  # [B,H,S,Dk]
        k = self.k(x).view(b, s, self.n_heads, self.d_k).transpose(1, 2)
        v = self.v(x).view(b, s, self.n_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)  # [B,H,S,S]

        t_bins = torch.clamp(time_deltas.long(), 0, self.time_proj.num_embeddings - 1)  # [B,S]
        time_bias = self.time_proj(t_bins).transpose(1, 2).unsqueeze(2)  # [B,H,1,S]
        scores = scores + time_bias

        if mask is not None:
            # mask: [B,S] with True for valid tokens
            scores = scores.masked_fill(mask.unsqueeze(1).unsqueeze(2) == 0, -1e9)

        attn = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn, v).transpose(1, 2).contiguous().view(b, s, d)
        out = self.out(context)
        if return_attn:
            return out, AttentionArtifacts(attn=attn, scores=scores)
        return out


class ChronoFormer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_heads: int = 4,
        max_time_bins: int = 1000,
        padding_idx: int = 0,
    ):
        super().__init__()
        self.padding_idx = int(padding_idx)
        self.max_time_bins = int(max_time_bins)

        self.event_embedding = nn.Embedding(int(vocab_size), int(d_model), padding_idx=self.padding_idx)
        self.time_embedding = nn.Embedding(self.max_time_bins, int(d_model))
        self.attention = TimeAwareAttention(int(d_model), int(n_heads), max_time_bins=self.max_time_bins)
        self.norm = nn.LayerNorm(int(d_model))
        self.classifier = nn.Sequential(
            nn.Linear(int(d_model), int(d_model) // 2),
            nn.ReLU(),
            nn.Linear(int(d_model) // 2, 1),
            nn.Sigmoid(),
        )

    def encode(
        self, x: torch.Tensor, time_deltas: torch.Tensor, mask: torch.Tensor | None
    ) -> torch.Tensor:
        t_bins = torch.clamp(time_deltas.long(), 0, self.max_time_bins - 1)
        return self.event_embedding(x) + self.time_embedding(t_bins)

    def forward(
        self,
        x: torch.Tensor,
        time_deltas: torch.Tensor,
        mask: torch.Tensor | None = None,
        *,
        return_attn: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, AttentionArtifacts]:
        h = self.encode(x, time_deltas, mask)
        if return_attn:
            h, artifacts = self.attention(h, time_deltas, mask, return_attn=True)
            h = self.norm(h)
            p = self.classifier(h[:, -1, :])
            return p, artifacts
        h = self.attention(h, time_deltas, mask, return_attn=False)
        h = self.norm(h)
        return self.classifier(h[:, -1, :])

