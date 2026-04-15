from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def save_token_importance_heatmap(
    importance: np.ndarray,
    *,
    title: str,
    out_path: str | Path,
    y_label: str = "sample",
    x_label: str = "token position",
    figsize: tuple[int, int] = (14, 2),
) -> Path:
    """
    importance: [S] or [B,S]
    Produces a simple heatmap over positions (tokens).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    imp = np.asarray(importance)
    if imp.ndim == 1:
        imp = imp[None, :]

    plt.figure(figsize=figsize)
    plt.imshow(imp, aspect="auto", cmap="viridis", interpolation="nearest")
    plt.colorbar()
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path

