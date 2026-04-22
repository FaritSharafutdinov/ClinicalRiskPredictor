# Presentation Plots

This folder contains tooling for generating defense-ready charts for the project presentation.

## Script

- `generate_plots.py` builds dataset-level and (optionally) model/XAI-level charts.

## Quick Start

Dry-run mode (no external dataset download, no checkpoint required):

```bash
python presentation/generate_plots.py --dry-run --out-dir presentation/figures
```

With a trained checkpoint (adds XAI overlay and fidelity curve):

```bash
python presentation/generate_plots.py \
  --dry-run \
  --checkpoint artifacts/chronoformer_best.pt \
  --out-dir presentation/figures
```

Full dataset mode (Synthea parquet via HF, slow/heavy):

```bash
python presentation/generate_plots.py \
  --checkpoint artifacts/chronoformer_best.pt \
  --out-dir presentation/figures_full
```

## Output files

Always generated:

- `class_balance.png`
- `sequence_length_distribution.png`
- `time_delta_distribution.png`
- `summary.json`

Generated only if checkpoint is available:

- `importance_overlay_sample{idx}.png`
- `fidelity_curve.png`
