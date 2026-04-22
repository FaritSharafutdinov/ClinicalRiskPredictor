from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

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


def _save_class_balance(labels: np.ndarray, out_dir: Path) -> Path:
    out_path = out_dir / "class_balance.png"
    zeros = int((labels == 0).sum())
    ones = int((labels == 1).sum())
    total = max(1, zeros + ones)
    pos_rate = ones / total

    plt.figure(figsize=(7, 4))
    bars = plt.bar(["Alive (0)", "Deceased (1)"], [zeros, ones], color=["#4c78a8", "#e45756"])
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, h, f"{int(h)}", ha="center", va="bottom")
    plt.title(f"Class Balance (positive rate={pos_rate:.3f})")
    plt.ylabel("Patients")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path


def _save_sequence_lengths(lengths: np.ndarray, out_dir: Path) -> Path:
    out_path = out_dir / "sequence_length_distribution.png"
    plt.figure(figsize=(8, 4))
    bins = min(60, max(10, int(np.sqrt(max(1, lengths.size)))))
    plt.hist(lengths, bins=bins, color="#4c78a8", edgecolor="black", linewidth=0.3)
    plt.title("Sequence Length Distribution")
    plt.xlabel("Number of events per patient")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path


def _save_time_deltas(gaps: np.ndarray, out_dir: Path) -> Path:
    out_path = out_dir / "time_delta_distribution.png"
    gaps = np.asarray(gaps, dtype=float)
    finite = gaps[np.isfinite(gaps)]
    if finite.size == 0:
        finite = np.array([0.0], dtype=float)
    p99 = float(np.percentile(finite, 99))
    clipped = np.clip(finite, 0.0, p99)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    bins = min(80, max(10, int(np.sqrt(max(1, clipped.size)))))
    axes[0].hist(clipped, bins=bins, color="#72b7b2", edgecolor="black", linewidth=0.3)
    axes[0].set_title(f"Time Gaps (days), clipped at p99={p99:.1f}")
    axes[0].set_xlabel("time_delta (days)")
    axes[0].set_ylabel("Count")

    axes[1].hist(np.log1p(np.maximum(finite, 0.0)), bins=bins, color="#f58518", edgecolor="black", linewidth=0.3)
    axes[1].set_title("log1p(time_delta)")
    axes[1].set_xlabel("log(1 + time_delta)")
    axes[1].set_ylabel("Count")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path


def _save_importance_overlay(
    rollout_imp: torch.Tensor,
    saliency_imp: torch.Tensor,
    mask: torch.Tensor,
    p_hat: torch.Tensor,
    out_dir: Path,
    *,
    sample_index: int,
) -> Path:
    out_path = out_dir / f"importance_overlay_sample{sample_index}.png"
    idx = int(np.clip(sample_index, 0, rollout_imp.shape[0] - 1))
    valid_len = int(mask[idx].sum().item())
    valid_len = max(valid_len, 1)

    x = np.arange(valid_len)
    y_roll = rollout_imp[idx, :valid_len].detach().cpu().numpy()
    y_sal = saliency_imp[idx, :valid_len].detach().cpu().numpy()
    prob = float(p_hat[idx].item())

    plt.figure(figsize=(12, 4))
    plt.plot(x, y_roll, label="Attention Rollout", linewidth=2.0, color="#4c78a8")
    plt.plot(x, y_sal, label="Saliency (input grads)", linewidth=2.0, color="#e45756")
    plt.title(f"Token Importance Overlay (sample={idx}, p={prob:.3f})")
    plt.xlabel("Token position")
    plt.ylabel("Importance")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path


def _parse_k_values(raw: str) -> list[int]:
    vals: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        k = int(part)
        if k > 0:
            vals.append(k)
    return sorted(set(vals))


def _save_fidelity_curve(
    model,
    x: torch.Tensor,
    t: torch.Tensor,
    m: torch.Tensor,
    imp_roll: torch.Tensor,
    imp_sal: torch.Tensor,
    out_dir: Path,
    *,
    k_values: list[int],
) -> tuple[Path, list[dict[str, float]]]:
    out_path = out_dir / "fidelity_curve.png"
    rows: list[dict[str, float]] = []
    roll_deltas: list[float] = []
    sal_deltas: list[float] = []

    for k in k_values:
        r = fidelity_drop_prob(model, x, t, m, imp_roll, k=k, pad_idx=0)
        s = fidelity_drop_prob(model, x, t, m, imp_sal, k=k, pad_idx=0)
        rows.append(
            {
                "k": int(k),
                "rollout_delta": float(r.delta),
                "saliency_delta": float(s.delta),
            }
        )
        roll_deltas.append(float(r.delta))
        sal_deltas.append(float(s.delta))

    plt.figure(figsize=(8, 4))
    plt.plot(k_values, roll_deltas, marker="o", linewidth=2.0, label="Rollout", color="#4c78a8")
    plt.plot(k_values, sal_deltas, marker="o", linewidth=2.0, label="Saliency", color="#e45756")
    plt.title("Fidelity Curve (delta probability vs top-k masked)")
    plt.xlabel("k masked tokens")
    plt.ylabel("delta = p(original) - p(masked)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path, rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate presentation-ready plots for ClinicalRiskPredictor."
    )
    parser.add_argument("--out-dir", type=str, default="presentation/figures")
    parser.add_argument("--dry-run", action="store_true", help="Use tiny synthetic data.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="Path to model checkpoint. If provided, model-based plots are generated.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument(
        "--k-values",
        type=str,
        default="1,3,5,10,20,40",
        help="Comma-separated k values for fidelity curve.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    cfg = TrainConfig(max_len=args.max_len, batch_size=args.batch_size, device=device, epochs=1)

    if args.dry_run:
        seqs, times, labels, splits = make_tiny_sequences(
            n_patients=512,
            max_events=min(64, cfg.max_len),
            vocab_size=500,
            seed=cfg.seed,
        )
        vocab_size = 500
        labels_np = np.asarray(labels, dtype=int)
        lengths_np = np.asarray([len(x) for x in seqs], dtype=int)
        gaps_np = np.asarray([g for row in times for g in row], dtype=float)

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

        labels_np = np.asarray(labels_df["label"].to_list(), dtype=int)
        lengths_np = np.asarray(events_df.group_by("patient_id").len()["len"].to_list(), dtype=int)
        gaps_np = np.asarray(events_df["time_delta"].to_list(), dtype=float)

        ds = MEDSDataset(events_df, labels_df, vocab, splits.test_ids, max_len=cfg.max_len, desc="test")

    print("Generating dataset plots...")
    p_class = _save_class_balance(labels_np, out_dir)
    p_len = _save_sequence_lengths(lengths_np, out_dir)
    p_gap = _save_time_deltas(gaps_np, out_dir)

    summary: dict[str, object] = {
        "dry_run": bool(args.dry_run),
        "out_dir": str(out_dir.resolve()),
        "dataset": {
            "n_patients": int(labels_np.size),
            "positive_rate": float(labels_np.mean()) if labels_np.size else float("nan"),
            "median_sequence_length": float(np.median(lengths_np)) if lengths_np.size else 0.0,
            "mean_time_delta": float(np.mean(gaps_np)) if gaps_np.size else 0.0,
        },
        "plots": [
            str(p_class.resolve()),
            str(p_len.resolve()),
            str(p_gap.resolve()),
        ],
    }

    checkpoint = Path(args.checkpoint) if args.checkpoint else None
    if checkpoint is not None and checkpoint.exists():
        print("Generating model/XAI plots...")
        loader = DataLoader(
            ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=(cfg.device == "cuda"),
        )
        batch = next(iter(loader))
        x = batch["x"].to(device)
        t = batch["times"].to(device)
        m = batch["mask"].to(device)

        model = load_model_for_inference(checkpoint, vocab_size=vocab_size, config=cfg)
        with torch.no_grad():
            p_hat, artifacts = model(x, t, m, return_attn=True)
        rollout = attention_rollout_single_layer(artifacts.attn, m, add_residual=True)
        imp_roll = last_token_importance_from_rollout(rollout)
        imp_sal = token_saliency_via_input_grads(model, x, t, m)

        p_imp = _save_importance_overlay(
            imp_roll,
            imp_sal,
            m,
            p_hat.squeeze(-1) if p_hat.dim() > 1 else p_hat,
            out_dir,
            sample_index=args.sample_index,
        )
        k_values = _parse_k_values(args.k_values)
        if not k_values:
            k_values = [1, 3, 5, 10, 20]
        p_fid, fid_rows = _save_fidelity_curve(
            model,
            x,
            t,
            m,
            imp_roll,
            imp_sal,
            out_dir,
            k_values=k_values,
        )
        summary["plots"].extend([str(p_imp.resolve()), str(p_fid.resolve())])
        summary["fidelity"] = fid_rows
    else:
        print("Checkpoint not provided or not found. Skipping model/XAI plots.")

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved summary: {summary_path.resolve()}")
    print(f"Done. Plots directory: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
