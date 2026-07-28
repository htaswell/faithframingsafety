"""Figures for the ordinal pipeline.

Produces one grid of score-distribution bar charts per (model, prompt) cell.
Colours come from the default matplotlib cycle so that figures stay legible in
greyscale print and do not depend on hard-coded values.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend, safe on servers and in CI
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def plot_score_distributions(
    df: pd.DataFrame,
    columns: dict[str, str],
    metrics: list[str],
    identities: list[str],
    score_levels: list[int],
    output_dir: str | Path,
    file_format: str = "png",
    dpi: int = 200,
) -> list[Path]:
    """Save one distribution figure per (model, prompt) cell. Returns the paths."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    colors = {identity: palette[i % len(palette)] for i, identity in enumerate(identities)}

    written: list[Path] = []
    for (model, prompt), cell in df.groupby([columns["model"], columns["prompt"]], dropna=False):
        n_metrics = len(metrics)
        n_cols = min(4, n_metrics)
        n_rows = int(np.ceil(n_metrics / n_cols))
        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows), squeeze=False
        )
        fig.suptitle(f"Prompt {prompt} — {model}", fontsize=14, fontweight="bold")

        x = np.arange(len(score_levels))
        present = [i for i in identities if (cell[columns["group"]] == i).any()]
        width = 0.8 / max(len(present), 1)

        for idx, metric in enumerate(metrics):
            ax = axes[idx // n_cols][idx % n_cols]
            for i, identity in enumerate(present):
                values = cell.loc[cell[columns["group"]] == identity, metric].dropna()
                total = len(values)
                pcts = [
                    100.0 * (values == level).sum() / total if total else 0.0
                    for level in score_levels
                ]
                offset = (i - (len(present) - 1) / 2) * width
                ax.bar(
                    x + offset,
                    pcts,
                    width,
                    label=str(identity).capitalize(),
                    color=colors[identity],
                    edgecolor="white",
                )
            ax.set_title(metric, fontsize=11)
            ax.set_xticks(x)
            ax.set_xticklabels([str(level) for level in score_levels])
            ax.set_xlabel("Score")
            ax.set_ylabel("% of responses")
            ax.set_ylim(0, 100)
            ax.grid(axis="y", alpha=0.3)

        for idx in range(n_metrics, n_rows * n_cols):
            axes[idx // n_cols][idx % n_cols].axis("off")

        axes[0][0].legend(fontsize=9)
        fig.tight_layout()

        safe_model = str(model).replace("/", "-").replace(" ", "_")
        path = output_dir / f"distribution_{safe_model}_prompt{prompt}.{file_format}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        written.append(path)

    return written
