from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class AttentionArtifacts:
    attn: torch.Tensor  # [B, H, S, S]
    scores: torch.Tensor  # [B, H, S, S]


@dataclass
class MultiLayerAttentionArtifacts:
    attn_by_layer: list[torch.Tensor]  # each [B,H,S,S]
    scores_by_layer: list[torch.Tensor]  # each [B,H,S,S]


@dataclass
class HierarchicalAttentionArtifacts:
    # Intra-bin attention: per intra layer attention matrices inside each bin
    # Shape per layer: [B, N, H, K, K]
    intra_attn_by_layer: list[torch.Tensor]
    intra_scores_by_layer: list[torch.Tensor]
    # Inter-bin attention: per inter layer attention between bins
    # Shape per layer: [B, H, N, N]
    inter_attn_by_layer: list[torch.Tensor]
    inter_scores_by_layer: list[torch.Tensor]
    # bin mapping to token positions in the original (padded) sequence
    bin_token_slices: list[tuple[int, int]]
    bin_size: int
    n_bins: int


def _cumulative_time(time_deltas: torch.Tensor) -> torch.Tensor:
    return torch.cumsum(torch.clamp(time_deltas, min=0.0), dim=1)


def _make_bins(s: int, bin_size: int) -> tuple[int, int, list[tuple[int, int]]]:
    if bin_size <= 0:
        raise ValueError("bin_size must be > 0")
    n_bins = (s + bin_size - 1) // bin_size
    padded_s = n_bins * bin_size
    slices = [(i * bin_size, min((i + 1) * bin_size, padded_s)) for i in range(n_bins)]
    return n_bins, padded_s, slices


def _pad_to_length(x: torch.Tensor, target_len: int, pad_value: float | int) -> torch.Tensor:
    b, s = x.shape[:2]
    if s == target_len:
        return x
    if s > target_len:
        return x[:, :target_len, ...]
    pad = torch.full((b, target_len - s, *x.shape[2:]), pad_value, device=x.device, dtype=x.dtype)
    return torch.cat([x, pad], dim=1)


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
        n_layers: int = 1,
        max_time_bins: int = 1000,
        padding_idx: int = 0,
    ):
        super().__init__()
        self.padding_idx = int(padding_idx)
        self.max_time_bins = int(max_time_bins)
        self.n_layers = int(n_layers)

        self.event_embedding = nn.Embedding(int(vocab_size), int(d_model), padding_idx=self.padding_idx)
        self.time_embedding = nn.Embedding(self.max_time_bins, int(d_model))
        self.attention_layers = nn.ModuleList(
            [
                TimeAwareAttention(int(d_model), int(n_heads), max_time_bins=self.max_time_bins)
                for _ in range(self.n_layers)
            ]
        )
        self.norm_layers = nn.ModuleList([nn.LayerNorm(int(d_model)) for _ in range(self.n_layers)])
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
    ) -> torch.Tensor | tuple[torch.Tensor, MultiLayerAttentionArtifacts]:
        h = self.encode(x, time_deltas, mask)
        if return_attn:
            attn_by_layer: list[torch.Tensor] = []
            scores_by_layer: list[torch.Tensor] = []
            for attn_layer, norm in zip(self.attention_layers, self.norm_layers):
                h, artifacts = attn_layer(h, time_deltas, mask, return_attn=True)
                h = norm(h)
                attn_by_layer.append(artifacts.attn)
                scores_by_layer.append(artifacts.scores)
            p = self.classifier(h[:, -1, :])
            return p, MultiLayerAttentionArtifacts(attn_by_layer=attn_by_layer, scores_by_layer=scores_by_layer)

        for attn_layer, norm in zip(self.attention_layers, self.norm_layers):
            h = attn_layer(h, time_deltas, mask, return_attn=False)
            h = norm(h)
        return self.classifier(h[:, -1, :])


class HierarchicalChronoFormer(nn.Module):
    """
    Hierarchical ChronoFormer with Intra-bin and Inter-bin attention.

    Forward signature stays compatible with training:
      forward(x, time_deltas, mask) -> probability
    """

    def __init__(
        self,
        vocab_size: int,
        *,
        d_model: int = 128,
        n_heads: int = 4,
        max_time_bins: int = 1000,
        padding_idx: int = 0,
        bin_size: int = 32,
        intra_layers: int = 1,
        inter_layers: int = 1,
        bin_pool: str = "mean",  # "mean" or "cls"
    ):
        super().__init__()
        if bin_pool not in {"mean", "cls"}:
            raise ValueError("bin_pool must be 'mean' or 'cls'")
        self.padding_idx = int(padding_idx)
        self.max_time_bins = int(max_time_bins)
        self.bin_size = int(bin_size)
        self.intra_layers = int(intra_layers)
        self.inter_layers = int(inter_layers)
        self.bin_pool = bin_pool

        self.event_embedding = nn.Embedding(int(vocab_size), int(d_model), padding_idx=self.padding_idx)
        # Dual time embeddings: relative (delta) + absolute (cumsum(delta))
        self.rel_time_embedding = nn.Embedding(self.max_time_bins, int(d_model))
        self.abs_time_embedding = nn.Embedding(self.max_time_bins, int(d_model))

        if self.bin_pool == "cls":
            self.bin_cls = nn.Parameter(torch.zeros(1, 1, int(d_model)))

        self.intra_attn = nn.ModuleList(
            [
                TimeAwareAttention(int(d_model), int(n_heads), max_time_bins=self.max_time_bins)
                for _ in range(self.intra_layers)
            ]
        )
        self.intra_norm = nn.ModuleList([nn.LayerNorm(int(d_model)) for _ in range(self.intra_layers)])

        self.inter_attn = nn.ModuleList(
            [
                TimeAwareAttention(int(d_model), int(n_heads), max_time_bins=self.max_time_bins)
                for _ in range(self.inter_layers)
            ]
        )
        self.inter_norm = nn.ModuleList([nn.LayerNorm(int(d_model)) for _ in range(self.inter_layers)])

        self.classifier = nn.Sequential(
            nn.Linear(int(d_model), int(d_model) // 2),
            nn.ReLU(),
            nn.Linear(int(d_model) // 2, 1),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor, time_deltas: torch.Tensor) -> torch.Tensor:
        rel_bins = torch.clamp(time_deltas.long(), 0, self.max_time_bins - 1)
        abs_bins = torch.clamp(_cumulative_time(time_deltas).long(), 0, self.max_time_bins - 1)
        return self.event_embedding(x) + self.rel_time_embedding(rel_bins) + self.abs_time_embedding(abs_bins)

    def forward(
        self,
        x: torch.Tensor,
        time_deltas: torch.Tensor,
        mask: torch.Tensor | None = None,
        *,
        return_attn: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, HierarchicalAttentionArtifacts]:
        b, s = x.shape
        device = x.device

        n_bins, padded_s, slices = _make_bins(s, self.bin_size)
        x_p = _pad_to_length(x, padded_s, self.padding_idx)
        t_p = _pad_to_length(time_deltas, padded_s, 0.0)
        if mask is None:
            m_p = x_p != self.padding_idx
        else:
            m_p = _pad_to_length(mask.to(dtype=torch.bool), padded_s, False)

        h_p = self.encode(x_p, t_p)  # [B,S,D]

        k = self.bin_size
        h_bins = h_p.view(b, n_bins, k, -1)
        t_bins = t_p.view(b, n_bins, k)
        m_bins = m_p.view(b, n_bins, k)

        # add CLS at start of each bin if requested
        if self.bin_pool == "cls":
            cls = self.bin_cls.expand(b, n_bins, 1, -1)
            h_bins = torch.cat([cls, h_bins], dim=2)  # [B,N,K+1,D]
            t0 = torch.zeros((b, n_bins, 1), device=device, dtype=t_bins.dtype)
            t_bins = torch.cat([t0, t_bins], dim=2)
            m1 = torch.ones((b, n_bins, 1), device=device, dtype=torch.bool)
            m_bins = torch.cat([m1, m_bins], dim=2)

        bn = b * n_bins
        kb = h_bins.shape[2]
        h_intra = h_bins.reshape(bn, kb, -1)
        t_intra = t_bins.reshape(bn, kb)
        m_intra = m_bins.reshape(bn, kb)

        intra_attn_by_layer: list[torch.Tensor] = []
        intra_scores_by_layer: list[torch.Tensor] = []
        for layer, norm in zip(self.intra_attn, self.intra_norm):
            if return_attn:
                h_intra, artifacts = layer(h_intra, t_intra, m_intra, return_attn=True)
                h_intra = norm(h_intra)
                intra_attn_by_layer.append(artifacts.attn.view(b, n_bins, artifacts.attn.shape[1], kb, kb))
                intra_scores_by_layer.append(artifacts.scores.view(b, n_bins, artifacts.scores.shape[1], kb, kb))
            else:
                h_intra = layer(h_intra, t_intra, m_intra, return_attn=False)
                h_intra = norm(h_intra)

        h_intra = h_intra.view(b, n_bins, kb, -1)

        # bin aggregation
        if self.bin_pool == "cls":
            bin_repr = h_intra[:, :, 0, :]
        else:
            w = m_bins.to(dtype=h_intra.dtype).unsqueeze(-1)
            denom = w.sum(dim=2).clamp_min(1.0)
            bin_repr = (h_intra * w).sum(dim=2) / denom

        # bin mask: exclude bins that are fully padded
        bin_mask = m_p.view(b, n_bins, k).any(dim=2)

        # bin-level relative time deltas based on cumulative time at end of each bin
        abs_time = _cumulative_time(t_p).view(b, n_bins, k)
        abs_masked = abs_time.masked_fill(~m_p.view(b, n_bins, k), 0.0)
        bin_abs = abs_masked.max(dim=2).values  # [B,N]
        bin_rel = torch.zeros_like(bin_abs)
        bin_rel[:, 1:] = torch.clamp(bin_abs[:, 1:] - bin_abs[:, :-1], min=0.0)

        h_inter = bin_repr
        inter_attn_by_layer: list[torch.Tensor] = []
        inter_scores_by_layer: list[torch.Tensor] = []
        for layer, norm in zip(self.inter_attn, self.inter_norm):
            if return_attn:
                h_inter, artifacts = layer(h_inter, bin_rel, bin_mask, return_attn=True)
                h_inter = norm(h_inter)
                inter_attn_by_layer.append(artifacts.attn)
                inter_scores_by_layer.append(artifacts.scores)
            else:
                h_inter = layer(h_inter, bin_rel, bin_mask, return_attn=False)
                h_inter = norm(h_inter)

        last_bin_idx = (bin_mask.to(dtype=torch.int64).sum(dim=1) - 1).clamp_min(0)
        last_repr = h_inter[torch.arange(b, device=device), last_bin_idx]
        p = self.classifier(last_repr)

        if return_attn:
            return p, HierarchicalAttentionArtifacts(
                intra_attn_by_layer=intra_attn_by_layer,
                intra_scores_by_layer=intra_scores_by_layer,
                inter_attn_by_layer=inter_attn_by_layer,
                inter_scores_by_layer=inter_scores_by_layer,
                bin_token_slices=slices,
                bin_size=self.bin_size,
                n_bins=n_bins,
            )
        return p

    def forward_from_embeddings(
        self,
        h: torch.Tensor,
        time_deltas: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        IG helper: caller provides token embeddings h=[B,S,D] and we run hierarchical attention.
        """
        b, s, d = h.shape
        n_bins, padded_s, _ = _make_bins(s, self.bin_size)
        h_p = _pad_to_length(h, padded_s, 0.0)
        t_p = _pad_to_length(time_deltas, padded_s, 0.0)
        if mask is None:
            m_p = torch.ones((b, padded_s), device=h.device, dtype=torch.bool)
        else:
            m_p = _pad_to_length(mask.to(dtype=torch.bool), padded_s, False)

        k = self.bin_size
        h_bins = h_p.view(b, n_bins, k, d)
        t_bins = t_p.view(b, n_bins, k)
        m_bins = m_p.view(b, n_bins, k)

        bn = b * n_bins
        h_intra = h_bins.reshape(bn, k, d)
        t_intra = t_bins.reshape(bn, k)
        m_intra = m_bins.reshape(bn, k)
        for layer, norm in zip(self.intra_attn, self.intra_norm):
            h_intra = layer(h_intra, t_intra, m_intra, return_attn=False)
            h_intra = norm(h_intra)
        h_intra = h_intra.view(b, n_bins, k, d)
        w = m_bins.to(dtype=h_intra.dtype).unsqueeze(-1)
        denom = w.sum(dim=2).clamp_min(1.0)
        bin_repr = (h_intra * w).sum(dim=2) / denom

        bin_mask = m_bins.any(dim=2)
        bin_rel = torch.zeros((b, n_bins), device=h.device, dtype=t_p.dtype)
        for layer, norm in zip(self.inter_attn, self.inter_norm):
            bin_repr = layer(bin_repr, bin_rel, bin_mask, return_attn=False)
            bin_repr = norm(bin_repr)
        last_bin_idx = (bin_mask.to(dtype=torch.int64).sum(dim=1) - 1).clamp_min(0)
        last_repr = bin_repr[torch.arange(b, device=h.device), last_bin_idx]
        return self.classifier(last_repr)


# Backwards-compatible alias: our training loop imports `ChronoFormer`.
# We now map it to the hierarchical implementation by default.
# If you need the old flat model, use `FlatChronoFormer = ChronoFormer` before this reassignment.
FlatChronoFormer = ChronoFormer
ChronoFormer = HierarchicalChronoFormer
