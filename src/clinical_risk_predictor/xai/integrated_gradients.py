from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class IGAttribution:
    # per-token scalar attribution
    token_attr: torch.Tensor  # [B,S]
    # per-token embedding attribution (useful for deeper analysis)
    embedding_attr: torch.Tensor  # [B,S,D]


@dataclass(frozen=True)
class IGResult:
    combined: IGAttribution
    event_only: IGAttribution
    time_only: IGAttribution


def _linspace_steps(steps: int, device: torch.device) -> torch.Tensor:
    # exclude 0.0 baseline and include 1.0 endpoint (common IG convention)
    return torch.linspace(0.0, 1.0, steps + 1, device=device)[1:]


def integrated_gradients_dual_embeddings(
    model,
    x: torch.Tensor,
    time_deltas: torch.Tensor,
    mask: torch.Tensor | None,
    *,
    steps: int = 32,
    baseline: str = "zero",
) -> IGResult:
    """
    Integrated Gradients on ChronoFormer input representation, with explicit split:
      h = event_emb(x) + time_emb(t_bins)

    We return:
    - combined IG (event+time together)
    - event-only IG (interpolate event while keeping time fixed)
    - time-only IG (interpolate time while keeping event fixed)

    This lets you quantify which "time signal" vs "event signal" contributes more.
    """
    model.eval()
    device = x.device

    # build event/time embeddings explicitly (support both flat and hierarchical model variants)
    t_bins = torch.clamp(time_deltas.long(), 0, model.max_time_bins - 1)
    e = model.event_embedding(x)  # [B,S,D]
    if hasattr(model, "time_embedding"):
        te = model.time_embedding(t_bins)  # [B,S,D]
    else:
        # HierarchicalChronoFormer uses dual time embeddings: relative + absolute.
        rel = model.rel_time_embedding(t_bins)
        abs_bins = torch.clamp(torch.cumsum(torch.clamp(time_deltas, min=0.0), dim=1).long(), 0, model.max_time_bins - 1)
        te = rel + model.abs_time_embedding(abs_bins)

    if baseline == "zero":
        e0 = torch.zeros_like(e)
        te0 = torch.zeros_like(te)
    elif baseline == "pad":
        pad = torch.full_like(x, int(getattr(model, "padding_idx", 0)))
        e0 = model.event_embedding(pad)
        te0 = torch.zeros_like(te)
    else:
        raise ValueError("baseline must be 'zero' or 'pad'")

    alphas = _linspace_steps(int(steps), device=device)  # [steps]

    def ig_for(
        e_base: torch.Tensor,
        te_base: torch.Tensor,
        e_target: torch.Tensor,
        te_target: torch.Tensor,
    ) -> IGAttribution:
        diff_e = e_target - e_base
        diff_te = te_target - te_base
        total_grad = torch.zeros_like(e_target)

        for a in alphas:
            h = (e_base + a * diff_e) + (te_base + a * diff_te)
            h = h.detach().requires_grad_(True)
            p = model.forward_from_embeddings(h, time_deltas, mask).squeeze(-1)  # [B]
            grads = torch.autograd.grad(p.sum(), h, retain_graph=False, create_graph=False)[0]
            total_grad = total_grad + grads

        avg_grad = total_grad / float(len(alphas))
        attr_embed = (diff_e + diff_te) * avg_grad
        attr_tok = attr_embed.abs().sum(dim=-1)
        if mask is not None:
            attr_tok = attr_tok * mask.to(dtype=attr_tok.dtype)
            attr_embed = attr_embed * mask.to(dtype=attr_embed.dtype).unsqueeze(-1)
        return IGAttribution(token_attr=attr_tok, embedding_attr=attr_embed)

    def ig_for_component(
        component: str,
    ) -> IGAttribution:
        if component == "event":
            # interpolate event only, keep time fixed at target
            diff = e - e0
            total_grad = torch.zeros_like(e)
            for a in alphas:
                h = (e0 + a * diff) + te
                h = h.detach().requires_grad_(True)
                p = model.forward_from_embeddings(h, time_deltas, mask).squeeze(-1)
                grads = torch.autograd.grad(p.sum(), h, retain_graph=False, create_graph=False)[0]
                total_grad = total_grad + grads
            avg_grad = total_grad / float(len(alphas))
            # attribution for event part only: diff_event * grad
            attr_embed = diff * avg_grad
        elif component == "time":
            diff = te - te0
            total_grad = torch.zeros_like(te)
            for a in alphas:
                h = e + (te0 + a * diff)
                h = h.detach().requires_grad_(True)
                p = model.forward_from_embeddings(h, time_deltas, mask).squeeze(-1)
                grads = torch.autograd.grad(p.sum(), h, retain_graph=False, create_graph=False)[0]
                total_grad = total_grad + grads
            avg_grad = total_grad / float(len(alphas))
            attr_embed = diff * avg_grad
        else:
            raise ValueError("component must be 'event' or 'time'")

        attr_tok = attr_embed.abs().sum(dim=-1)
        if mask is not None:
            attr_tok = attr_tok * mask.to(dtype=attr_tok.dtype)
            attr_embed = attr_embed * mask.to(dtype=attr_embed.dtype).unsqueeze(-1)
        return IGAttribution(token_attr=attr_tok, embedding_attr=attr_embed)

    combined = ig_for(e0, te0, e, te)
    event_only = ig_for_component("event")
    time_only = ig_for_component("time")
    return IGResult(combined=combined, event_only=event_only, time_only=time_only)

