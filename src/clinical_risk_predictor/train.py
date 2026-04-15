from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .config import TrainConfig
from .model import ChronoFormer


@dataclass(frozen=True)
class TrainResult:
    best_val_auroc: float
    best_epoch: int
    checkpoint_path: Path


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    preds: list[float] = []
    targets: list[float] = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="eval", leave=False):
            x = batch["x"].to(device)
            t = batch["times"].to(device)
            m = batch["mask"].to(device)
            y = batch["label"].to(device)
            out = model(x, t, m).squeeze(-1)
            preds.extend(out.detach().cpu().numpy().tolist())
            targets.extend(y.detach().cpu().numpy().tolist())

    if len(set(targets)) < 2:
        auroc = float("nan")
    else:
        auroc = float(roc_auc_score(targets, preds))
    f1 = float(f1_score(targets, [1 if p > 0.5 else 0 for p in preds]))
    return auroc, f1


def train_model(
    *,
    train_loader: DataLoader,
    val_loader: DataLoader,
    vocab_size: int,
    config: TrainConfig,
    out_dir: str | Path = "artifacts",
) -> TrainResult:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(config.seed)

    device = torch.device(config.device)
    model = ChronoFormer(
        vocab_size=vocab_size,
        d_model=config.d_model,
        n_heads=config.n_heads,
        max_time_bins=config.max_time_bins,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    criterion = nn.BCELoss()

    best_auroc = -float("inf")
    best_epoch = -1
    patience_counter = 0
    ckpt_path = out_dir / "chronoformer_best.pt"
    config.save_json(out_dir / "train_config.json")

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    for epoch in range(config.epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"ep{epoch} train", leave=False)
        total_loss = 0.0
        for batch in pbar:
            x = batch["x"].to(device, non_blocking=(device.type == "cuda"))
            t = batch["times"].to(device, non_blocking=(device.type == "cuda"))
            m = batch["mask"].to(device, non_blocking=(device.type == "cuda"))
            y = batch["label"].to(device, non_blocking=(device.type == "cuda"))

            optimizer.zero_grad(set_to_none=True)
            out = model(x, t, m).squeeze(-1)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            pbar.set_postfix(loss=float(loss.item()))

        val_auroc, val_f1 = evaluate(model, val_loader, device=device)
        print(f"ep{epoch} loss={total_loss/max(1,len(train_loader)):.4f} val_auroc={val_auroc:.4f} val_f1={val_f1:.4f}")

        if val_auroc != val_auroc:  # NaN
            patience_counter += 1
        elif val_auroc > best_auroc:
            best_auroc = val_auroc
            best_epoch = epoch
            torch.save(model.state_dict(), ckpt_path)
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= config.patience:
            print("Early stopping.")
            break

    if best_epoch < 0:
        torch.save(model.state_dict(), ckpt_path)
        best_epoch = 0
        best_auroc = float("nan")

    return TrainResult(best_val_auroc=float(best_auroc), best_epoch=int(best_epoch), checkpoint_path=ckpt_path)


def load_model_for_inference(
    checkpoint_path: str | Path,
    *,
    vocab_size: int,
    config: TrainConfig,
) -> ChronoFormer:
    device = torch.device(config.device)
    model = ChronoFormer(
        vocab_size=vocab_size,
        d_model=config.d_model,
        n_heads=config.n_heads,
        max_time_bins=config.max_time_bins,
    ).to(device)
    state = torch.load(Path(checkpoint_path), map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model

