from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import json
import torch


@dataclass(frozen=True)
class TrainConfig:
    max_len: int = 256
    batch_size: int = 128
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 1
    bin_size: int = 256
    intra_layers: int = 1
    inter_layers: int = 1
    bin_pool: str = "mean"  # "mean" or "cls"
    epochs: int = 50
    lr: float = 5e-5
    patience: int = 5
    max_time_bins: int = 1000
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def to_dict(self) -> dict:
        return asdict(self)

    def save_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

