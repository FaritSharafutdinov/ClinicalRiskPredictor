## ClinicalRiskPredictor

This project implements a **ChronoFormer** baseline for clinical risk prediction (baseline task: binary mortality) from a sequence of clinical events (condition codes) and **inter-event time gaps**. On top of the baseline model, it includes an **XAI** module:

- **Attention Rollout** (aggregate attention → token importance for the last position)
- **Saliency** (gradient-based token importance on input embeddings)
- **Fidelity** (mask top-k important tokens → probability drop)

The baseline data setup follows `baseline.ipynb`: Hugging Face `richardyoung/synthea-575k-patients` (parquet).

### Installation

Python **3.10–3.12** is recommended (PyTorch may not provide prebuilt wheels for Python 3.13 on your platform).

```bash
pip install -r requirements.txt
pip install -e .
```

### Training

- **Dry-run (no dataset download, quick smoke test):**

```bash
python scripts/train.py --dry-run --epochs 2 --batch-size 64 --max-len 64
```

- **Full run on Synthea (slow and RAM-heavy):**

```bash
python scripts/train.py --epochs 50 --batch-size 128 --max-len 256 --out-dir artifacts
```

The best checkpoint is saved to `artifacts/chronoformer_best.pt`.

### Explanations (heatmaps)

Generates two images (rollout and saliency over token positions) for one sample from the test split.

```bash
python scripts/explain.py --dry-run --checkpoint artifacts/chronoformer_best.pt --out-dir artifacts/xai --sample-index 0
```

Outputs: `artifacts/xai/rollout_sample0.png`, `artifacts/xai/saliency_sample0.png`.

### Fidelity

Measures how much the predicted probability drops after masking the top-k important tokens.

```bash
python scripts/fidelity.py --dry-run --checkpoint artifacts/chronoformer_best.pt --k 10
```

### Advanced XAI (next-step)

- **Integrated Gradients (IG)** (with event vs time attribution split): use `scripts/visualize_xai.py`
- **Multi-layer Attention Rollout**: supported via `AttentionRollout` (works for 1+ layers)
- **Time-aware sensitivity analysis**: perturb \(\Delta t\) and measure prediction change

```bash
python scripts/visualize_xai.py --dry-run --checkpoint artifacts/chronoformer_best.pt --out artifacts/xai/compare_xai.png
python scripts/time_sensitivity.py --dry-run --checkpoint artifacts/chronoformer_best.pt --mode scale --scale 2.0
```

