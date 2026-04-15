from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

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
from clinical_risk_predictor.xai.attention_rollout import (
    attention_rollout_single_layer,
    last_token_importance_from_rollout,
)
from clinical_risk_predictor.xai.fidelity import fidelity_drop_prob
from clinical_risk_predictor.xai.saliency import token_saliency_via_input_grads


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default="artifacts/chronoformer_best.pt")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--k", type=int, default=10, help="How many tokens to mask.")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-len", type=int, default=256)
    p.add_argument("--device", type=str, default=None)
    args = p.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    cfg = TrainConfig(max_len=args.max_len, batch_size=args.batch_size, device=device, epochs=1)

    if args.dry_run:
        seqs, times, labels, splits = make_tiny_sequences(
            n_patients=256, max_events=min(64, cfg.max_len), vocab_size=500, seed=cfg.seed
        )
        vocab_size = 500
    else:
        labels_df, events_df = load_synthea_parquet()
        splits = make_splits(labels_df, seed=cfg.seed)
        vocab = create_vocab(events_df)
        vocab_size = len(vocab)

    if args.dry_run:
        id_to_i = {pid: i for i, pid in enumerate([f"p{i:05d}" for i in range(len(labels))])}
        test_idx = [id_to_i[i] for i in splits.test_ids]
        ds = TinySequenceDataset(
            [seqs[i] for i in test_idx],
            [times[i] for i in test_idx],
            [labels[i] for i in test_idx],
            max_len=cfg.max_len,
        )
    else:
        ds = MEDSDataset(events_df, labels_df, vocab, splits.test_ids, max_len=cfg.max_len, desc="test")
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0, pin_memory=(device == "cuda"))
    batch = next(iter(loader))
    x = batch["x"].to(device)
    t = batch["times"].to(device)
    m = batch["mask"].to(device)

    model = load_model_for_inference(args.checkpoint, vocab_size=vocab_size, config=cfg)

    # attention rollout importance
    with torch.no_grad():
        _, artifacts = model(x, t, m, return_attn=True)
    rollout = attention_rollout_single_layer(artifacts.attn, m, add_residual=True)
    imp_roll = last_token_importance_from_rollout(rollout)
    res_roll = fidelity_drop_prob(model, x, t, m, imp_roll, k=args.k, pad_idx=0)

    # saliency importance
    imp_sal = token_saliency_via_input_grads(model, x, t, m)
    res_sal = fidelity_drop_prob(model, x, t, m, imp_sal, k=args.k, pad_idx=0)

    print("Fidelity (mask top-k tokens; higher delta => more faithful)")
    print(f"- rollout:  k={res_roll.k} base={res_roll.base_prob:.4f} after={res_roll.prob_after_mask:.4f} delta={res_roll.delta:.4f}")
    print(f"- saliency: k={res_sal.k} base={res_sal.base_prob:.4f} after={res_sal.prob_after_mask:.4f} delta={res_sal.delta:.4f}")


if __name__ == "__main__":
    main()

