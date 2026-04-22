from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from clinical_risk_predictor.config import TrainConfig
from clinical_risk_predictor.data import (
    MEDSDataset,
    TinySequenceDataset,
    create_vocab,
    load_synthea_parquet,
    make_splits,
    make_tiny_sequences,
)
from clinical_risk_predictor.train import load_model_for_inference
from clinical_risk_predictor.xai.attention_rollout import AttentionRollout
from clinical_risk_predictor.xai.integrated_gradients import integrated_gradients_dual_embeddings


def _normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    v = v.copy()
    v[v < 0] = 0
    m = v.max() if v.size else 1.0
    return v / (m + eps)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default="artifacts/chronoformer_best.pt")
    p.add_argument("--out", type=str, default="artifacts/xai/compare_xai.png")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--sample-index", type=int, default=0, help="Index within the first batch.")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-len", type=int, default=256)
    p.add_argument("--ig-steps", type=int, default=32)
    p.add_argument("--device", type=str, default=None)
    args = p.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    cfg = TrainConfig(max_len=args.max_len, batch_size=args.batch_size, device=device, epochs=1)

    if args.dry_run:
        seqs, times, labels, splits = make_tiny_sequences(
            n_patients=256, max_events=min(64, cfg.max_len), vocab_size=500, seed=cfg.seed
        )
        vocab_size = 500
        id_to_i = {pid: i for i, pid in enumerate([f"p{i:05d}" for i in range(len(labels))])}
        test_idx = [id_to_i[i] for i in splits.test_ids]
        ds = TinySequenceDataset(
            [seqs[i] for i in test_idx],
            [times[i] for i in test_idx],
            [labels[i] for i in test_idx],
            max_len=cfg.max_len,
        )
    else:
        labels_df, events_df = load_synthea_parquet()
        splits = make_splits(labels_df, seed=cfg.seed)
        vocab = create_vocab(events_df)
        vocab_size = len(vocab)
        ds = MEDSDataset(events_df, labels_df, vocab, splits.test_ids, max_len=cfg.max_len, desc="test")

    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0, pin_memory=(device == "cuda"))
    batch = next(iter(loader))
    x = batch["x"].to(device)
    t = batch["times"].to(device)
    m = batch["mask"].to(device)

    model = load_model_for_inference(args.checkpoint, vocab_size=vocab_size, config=cfg)

    idx = int(np.clip(args.sample_index, 0, x.shape[0] - 1))

    # Raw attention (last layer, mean heads, last token row)
    with torch.no_grad():
        p_hat, artifacts = model(x, t, m, return_attn=True)
    # Hierarchical: use inter-bin attention as "raw" (bin-level), then broadcast to token level.
    inter_attn_last = artifacts.inter_attn_by_layer[-1]  # [B,H,N,N]
    raw_inter = inter_attn_last.mean(dim=1)[idx, -1, :].detach().cpu().numpy()  # last bin -> all bins
    # broadcast bin importance to tokens
    raw = np.zeros((cfg.max_len,), dtype=np.float32)
    for j, (s0, s1) in enumerate(artifacts.bin_token_slices):
        s1 = min(s1, cfg.max_len)
        if s1 > s0:
            raw[s0:s1] = float(raw_inter[j])

    # Attention rollout across layers
    # Rollout on inter-bin level, then broadcast to token level for fair comparison
    inter_roll = AttentionRollout(add_residual=True, head_reduction="mean").rollout(artifacts.inter_attn_by_layer, None)
    roll_inter = inter_roll[idx, -1, :].detach().cpu().numpy()
    roll_imp = np.zeros((cfg.max_len,), dtype=np.float32)
    for j, (s0, s1) in enumerate(artifacts.bin_token_slices):
        s1 = min(s1, cfg.max_len)
        if s1 > s0:
            roll_imp[s0:s1] = float(roll_inter[j])

    # Integrated gradients (combined + event vs time)
    ig = integrated_gradients_dual_embeddings(
        model, x[idx : idx + 1], t[idx : idx + 1], m[idx : idx + 1], steps=args.ig_steps
    )
    ig_comb = ig.combined.token_attr.squeeze(0).detach().cpu().numpy()
    ig_event = ig.event_only.token_attr.squeeze(0).detach().cpu().numpy()
    ig_time = ig.time_only.token_attr.squeeze(0).detach().cpu().numpy()

    raw_n = _normalize(raw)
    roll_n = _normalize(roll_imp)
    ig_n = _normalize(ig_comb)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(4, 1, figsize=(14, 6), sharex=True)
    fig.suptitle(f"XAI comparison for one trajectory (p={float(p_hat[idx].item()):.3f})")

    axes[0].plot(raw_n, color="tab:blue")
    axes[0].set_ylabel("Raw attn")

    axes[1].plot(roll_n, color="tab:green")
    axes[1].set_ylabel("Rollout")

    axes[2].plot(ig_n, color="tab:red")
    axes[2].set_ylabel("IG (combined)")

    # show event vs time attribution strength
    axes[3].plot(_normalize(ig_event), label="IG event", color="tab:purple")
    axes[3].plot(_normalize(ig_time), label="IG time", color="tab:orange")
    axes[3].set_ylabel("IG split")
    axes[3].legend(loc="upper right")

    axes[-1].set_xlabel("Token position")
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close(fig)
    print(f"Saved: {out.resolve()}")


if __name__ == "__main__":
    main()

