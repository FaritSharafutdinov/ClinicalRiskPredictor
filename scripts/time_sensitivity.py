from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

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


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default="artifacts/chronoformer_best.pt")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-len", type=int, default=256)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--mode", type=str, default="scale", choices=["scale", "noise", "shuffle"])
    p.add_argument("--scale", type=float, default=2.0, help="Used for mode=scale")
    p.add_argument("--noise-std", type=float, default=5.0, help="Std (days) for mode=noise")
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

    with torch.no_grad():
        p0 = model(x, t, m).squeeze(-1)

    if args.mode == "scale":
        t2 = torch.clamp(t * float(args.scale), min=0.0)
    elif args.mode == "noise":
        noise = torch.randn_like(t) * float(args.noise_std)
        t2 = torch.clamp(t + noise, min=0.0)
    else:  # shuffle within each sequence
        t2 = t.clone()
        for i in range(t2.shape[0]):
            valid = m[i].nonzero(as_tuple=False).squeeze(-1)
            if valid.numel() > 1:
                perm = valid[torch.randperm(valid.numel(), device=valid.device)]
                t2[i, valid] = t2[i, perm]

    with torch.no_grad():
        p1 = model(x, t2, m).squeeze(-1)

    delta = (p1 - p0).detach().cpu().numpy()
    print("Time-aware sensitivity analysis")
    print(f"- mode={args.mode}")
    if args.mode == "scale":
        print(f"- scale={args.scale}")
    if args.mode == "noise":
        print(f"- noise_std={args.noise_std} days")
    print(f"- mean_delta={delta.mean():.6f}")
    print(f"- mean_abs_delta={np.abs(delta).mean():.6f}")
    print(f"- p0_mean={float(p0.mean().item()):.6f} p1_mean={float(p1.mean().item()):.6f}")


if __name__ == "__main__":
    main()

