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

