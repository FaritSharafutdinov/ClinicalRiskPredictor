from __future__ import annotations

from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class FidelityResult:
    base_prob: float
    prob_after_mask: float
    delta: float
    k: int


@torch.no_grad()
def mask_topk_tokens(
    x: torch.Tensor,
    mask: torch.Tensor,
    importance: torch.Tensor,
    *,
    k: int,
    pad_idx: int = 0,
) -> torch.Tensor:
    """
    Replaces top-k important *valid* tokens with PAD.
    x: [B,S], mask: [B,S] bool, importance: [B,S]
    """
    b, s = x.shape
    x2 = x.clone()
    imp = importance.clone()
    imp = imp.masked_fill(~mask, -1e9)
    k = int(min(max(k, 0), int(mask.sum(dim=1).max().item()) if mask.any() else 0))
    if k == 0:
        return x2
    topk = torch.topk(imp, k=k, dim=1).indices  # [B,k]
    for i in range(b):
        x2[i, topk[i]] = pad_idx
    return x2


@torch.no_grad()
def fidelity_drop_prob(
    model,
    x: torch.Tensor,
    time_deltas: torch.Tensor,
    mask: torch.Tensor,
    importance: torch.Tensor,
    *,
    k: int,
    pad_idx: int = 0,
) -> FidelityResult:
    """
    Simple faithfulness proxy:
      delta = p(original) - p(mask_top_k(original))
    (The larger the delta, the more "important" the masked tokens are, in a causal sense.)
    """
    p0 = model(x, time_deltas, mask).squeeze(-1)
    x_masked = mask_topk_tokens(x, mask, importance, k=k, pad_idx=pad_idx)
    p1 = model(x_masked, time_deltas, mask).squeeze(-1)
    # return mean over batch to keep API simple for quick experiments
    base = float(p0.mean().item())
    after = float(p1.mean().item())
    return FidelityResult(base_prob=base, prob_after_mask=after, delta=base - after, k=int(k))

