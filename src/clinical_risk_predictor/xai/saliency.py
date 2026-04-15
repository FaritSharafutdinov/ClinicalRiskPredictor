from __future__ import annotations

import torch


@torch.no_grad()
def _safe_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    x = x - x.min(dim=-1, keepdim=True).values
    d = x.max(dim=-1, keepdim=True).values + eps
    return x / d


def token_saliency_via_input_grads(
    model,
    x: torch.Tensor,
    time_deltas: torch.Tensor,
    mask: torch.Tensor | None,
) -> torch.Tensor:
    """
    Gradient-based saliency on tokens.
    We take gradients of the output probability w.r.t. the *combined* input representation
    (event_emb + time_emb). Then aggregate over embedding dimension with L1 norm.

    Returns:
      saliency: [B,S] (normalized to [0,1] per sample)
    """
    model.eval()
    # build differentiable input
    h = model.encode(x, time_deltas, mask)
    h = h.detach().requires_grad_(True)

    out = model.attention(h, time_deltas, mask, return_attn=False)
    out = model.norm(out)
    p = model.classifier(out[:, -1, :]).squeeze(-1)  # [B]

    grads = torch.autograd.grad(outputs=p.sum(), inputs=h, create_graph=False, retain_graph=False)[0]
    s = grads.abs().sum(dim=-1)  # [B,S]
    if mask is not None:
        s = s * mask.to(dtype=s.dtype)
    s = _safe_normalize(s)
    return s

