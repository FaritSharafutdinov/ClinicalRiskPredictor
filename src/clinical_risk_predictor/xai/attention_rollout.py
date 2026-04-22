from __future__ import annotations

import torch


def attention_rollout_single_layer(
    attn: torch.Tensor, mask: torch.Tensor | None = None, *, add_residual: bool = True
) -> torch.Tensor:
    """
    For our current baseline (single attention layer), "rollout" is:
      A_bar = mean_heads(attn)
      if add_residual: A_hat = (A_bar + I) / row_sum
    Returns:
      A_hat: [B, S, S]
    """
    if attn.dim() != 4:
        raise ValueError(f"Expected attn [B,H,S,S], got {tuple(attn.shape)}")
    a = attn.mean(dim=1)  # [B,S,S]
    if add_residual:
        b, s, _ = a.shape
        eye = torch.eye(s, device=a.device, dtype=a.dtype).unsqueeze(0).expand(b, -1, -1)
        a = a + eye
    a = a / (a.sum(dim=-1, keepdim=True) + 1e-12)

    if mask is not None:
        # zero out attention rows/cols for padded tokens for nicer visualizations
        m = mask.to(dtype=a.dtype)  # [B,S]
        a = a * m.unsqueeze(-1) * m.unsqueeze(-2)
        a = a / (a.sum(dim=-1, keepdim=True) + 1e-12)
    return a


def last_token_importance_from_rollout(rollout: torch.Tensor) -> torch.Tensor:
    """
    Uses attention from the LAST token (classifier reads h[:, -1, :]) as importance to all tokens.
    rollout: [B,S,S]  -> importance: [B,S]
    """
    if rollout.dim() != 3:
        raise ValueError(f"Expected rollout [B,S,S], got {tuple(rollout.shape)}")
    return rollout[:, -1, :]


class AttentionRollout:
    """
    Multi-layer attention rollout (Abnar & Zuidema-style), with residual connections.

    Notes:
    - Our current ChronoFormer baseline is typically a single layer, but this supports N layers.
    - For hierarchical models (inter-bin + intra-bin), see `unroll_hierarchical_to_tokens`.
    """

    def __init__(self, *, add_residual: bool = True, head_reduction: str = "mean"):
        if head_reduction not in {"mean", "max"}:
            raise ValueError("head_reduction must be 'mean' or 'max'")
        self.add_residual = bool(add_residual)
        self.head_reduction = head_reduction

    def _reduce_heads(self, attn: torch.Tensor) -> torch.Tensor:
        # attn: [B,H,S,S] -> [B,S,S]
        if self.head_reduction == "mean":
            return attn.mean(dim=1)
        return attn.max(dim=1).values

    def rollout(self, attn_by_layer: list[torch.Tensor], mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        attn_by_layer: list of [B,H,S,S] from shallow->deep.
        Returns token-level rollout matrix: [B,S,S]
        """
        if len(attn_by_layer) == 0:
            raise ValueError("attn_by_layer is empty")
        b, _, s, _ = attn_by_layer[0].shape
        device = attn_by_layer[0].device
        dtype = attn_by_layer[0].dtype
        joint = torch.eye(s, device=device, dtype=dtype).unsqueeze(0).expand(b, -1, -1)  # [B,S,S]

        for attn in attn_by_layer:
            a = self._reduce_heads(attn)  # [B,S,S]
            if self.add_residual:
                eye = torch.eye(s, device=device, dtype=dtype).unsqueeze(0).expand(b, -1, -1)
                a = a + eye
            a = a / (a.sum(dim=-1, keepdim=True) + 1e-12)
            if mask is not None:
                m = mask.to(dtype=dtype)
                a = a * m.unsqueeze(-1) * m.unsqueeze(-2)
                a = a / (a.sum(dim=-1, keepdim=True) + 1e-12)
            joint = torch.bmm(a, joint)
        return joint

    @staticmethod
    def unroll_hierarchical_to_tokens(
        inter_bin_rollout: torch.Tensor,
        intra_bin_rollout: torch.Tensor,
        bin_token_slices: list[tuple[int, int]],
        *,
        total_tokens: int,
    ) -> torch.Tensor:
        """
        Unroll a hierarchical attention structure down to token level.

        Inputs:
        - inter_bin_rollout: [B, BINS, BINS] rollout between bins
        - intra_bin_rollout: [B, BINS, L, L] rollout inside each bin (L may vary; here assumed padded to max L)
        - bin_token_slices: list of (start,end) token spans per bin in the flattened token sequence
        - total_tokens: S in the flattened representation

        Output:
        - token_level: [B, S, S]

        This is a generic utility: if your ChronoFormer variant truly uses bin hierarchies,
        you can feed its artifacts here to obtain a single global token importance map.
        """
        b, n_bins, _ = inter_bin_rollout.shape
        device = inter_bin_rollout.device
        dtype = inter_bin_rollout.dtype
        out = torch.zeros((b, total_tokens, total_tokens), device=device, dtype=dtype)

        for i in range(n_bins):
            si, ei = bin_token_slices[i]
            Li = max(ei - si, 0)
            if Li == 0:
                continue
            # intra within bin i
            out[:, si:ei, si:ei] = intra_bin_rollout[:, i, :Li, :Li]

        # distribute inter-bin mass uniformly across tokens in corresponding bins
        for i in range(n_bins):
            si, ei = bin_token_slices[i]
            Li = max(ei - si, 0)
            if Li == 0:
                continue
            for j in range(n_bins):
                sj, ej = bin_token_slices[j]
                Lj = max(ej - sj, 0)
                if Lj == 0:
                    continue
                # add coarse inter-bin attention between any token in bin i to any token in bin j
                out[:, si:ei, sj:ej] += inter_bin_rollout[:, i, j].unsqueeze(-1).unsqueeze(-1) / (Lj + 1e-12)
        out = out / (out.sum(dim=-1, keepdim=True) + 1e-12)
        return out

