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
from clinical_risk_predictor.train import train_model, evaluate, load_model_for_inference


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=str, default="artifacts")
    p.add_argument("--dry-run", action="store_true", help="Use tiny synthetic data (no HF download).")
    p.add_argument("--max-len", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--device", type=str, default=None)
    args = p.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    cfg = TrainConfig(
        max_len=args.max_len,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        patience=args.patience,
        device=device,
        bin_size=min(32, args.max_len) if args.dry_run else min(32, args.max_len),
        intra_layers=1,
        inter_layers=1,
    )

    if args.dry_run:
        seqs, times, labels, splits = make_tiny_sequences(
            n_patients=512, max_events=min(64, cfg.max_len), vocab_size=500, seed=cfg.seed
        )
        vocab_size = 500
    else:
        labels_df, events_df = load_synthea_parquet()
        splits = make_splits(labels_df, seed=cfg.seed)
        vocab = create_vocab(events_df)
        vocab_size = len(vocab)

    loader_kw = dict(num_workers=0, pin_memory=(cfg.device == "cuda"))
    if args.dry_run:
        id_to_i = {pid: i for i, pid in enumerate([f"p{i:05d}" for i in range(len(labels))])}
        train_idx = [id_to_i[i] for i in splits.train_ids]
        val_idx = [id_to_i[i] for i in splits.val_ids]
        test_idx = [id_to_i[i] for i in splits.test_ids]
        train_ds = TinySequenceDataset([seqs[i] for i in train_idx], [times[i] for i in train_idx], [labels[i] for i in train_idx], max_len=cfg.max_len)
        val_ds = TinySequenceDataset([seqs[i] for i in val_idx], [times[i] for i in val_idx], [labels[i] for i in val_idx], max_len=cfg.max_len)
        test_ds = TinySequenceDataset([seqs[i] for i in test_idx], [times[i] for i in test_idx], [labels[i] for i in test_idx], max_len=cfg.max_len)
    else:
        train_ds = MEDSDataset(events_df, labels_df, vocab, splits.train_ids, max_len=cfg.max_len, desc="train")
        val_ds = MEDSDataset(events_df, labels_df, vocab, splits.val_ids, max_len=cfg.max_len, desc="val")
        test_ds = MEDSDataset(events_df, labels_df, vocab, splits.test_ids, max_len=cfg.max_len, desc="test")

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, **loader_kw)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, **loader_kw)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, **loader_kw)

    result = train_model(
        train_loader=train_loader,
        val_loader=val_loader,
        vocab_size=vocab_size,
        config=cfg,
        out_dir=args.out_dir,
    )
    print(f"Saved best checkpoint to: {result.checkpoint_path}")

    model = load_model_for_inference(result.checkpoint_path, vocab_size=vocab_size, config=cfg)
    test_auroc, test_f1 = evaluate(model, test_loader, device=torch.device(cfg.device))
    print(f"test_auroc={test_auroc:.4f} test_f1={test_f1:.4f}")

    # save vocab for later explanation scripts
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "vocab_size.txt").write_text(str(vocab_size), encoding="utf-8")


if __name__ == "__main__":
    main()

