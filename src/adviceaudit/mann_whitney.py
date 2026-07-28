"""Pipeline B, step 2: ordinal comparison via Mann-Whitney U.

Compares ordinal 0-3 annotation scores between groups within each
(model, prompt) cell, using the Mann-Whitney U test for independent samples
with rank-biserial correlation as the effect size.

A comparison is flagged ``meaningful_raw`` when p < alpha and |r| >= the effect
threshold, reproducing the original decision rule. ``meaningful_adj`` applies
the same rule to Benjamini-Hochberg adjusted p-values and is the stricter,
recommended criterion when many comparisons are run.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from .io_utils import (
    load_config,
    read_table,
    require_columns,
    write_excel_sheets,
    write_manifest,
    write_table,
)
from .stats_utils import add_fdr_correction, mann_whitney_pair

OUTPUT_COLUMNS = [
    "model",
    "prompt",
    "metric",
    "identity_1",
    "identity_2",
    "n_1",
    "n_2",
    "mean_1",
    "mean_2",
    "median_1",
    "median_2",
    "u_statistic",
    "effect_size",
    "p_value",
    "p_value_adj",
    "significant_adj",
    "meaningful_raw",
    "meaningful_adj",
    "correction_family",
    "n_tests_in_family",
    "note",
]


def compare_groups(
    df: pd.DataFrame,
    columns: dict[str, str],
    metrics: list[str],
    identities: list[str],
    min_n: int = 2,
) -> pd.DataFrame:
    """Run all pairwise Mann-Whitney U tests and return a tidy results table.

    Every identity in ``identities`` is compared against every other one,
    within each (model, prompt) cell, for each metric.
    """
    require_columns(
        df,
        [columns["model"], columns["prompt"], columns["group"], *metrics],
        "Mann-Whitney comparison",
    )

    rows = []
    for (model, prompt), cell in df.groupby([columns["model"], columns["prompt"]], dropna=False):
        present = set(cell[columns["group"]].dropna().unique())
        cell_identities = [i for i in identities if i in present]
        for metric in metrics:
            for identity_1, identity_2 in combinations(sorted(cell_identities, key=str), 2):
                x1 = cell.loc[cell[columns["group"]] == identity_1, metric]
                x2 = cell.loc[cell[columns["group"]] == identity_2, metric]
                stats = mann_whitney_pair(x1, x2, min_n=min_n)
                rows.append(
                    {
                        "model": model,
                        "prompt": prompt,
                        "metric": metric,
                        "identity_1": identity_1,
                        "identity_2": identity_2,
                        **stats,
                    }
                )

    return pd.DataFrame(rows)


def score_distribution(
    df: pd.DataFrame,
    columns: dict[str, str],
    metrics: list[str],
    identities: list[str],
    score_levels: list[int],
) -> pd.DataFrame:
    """Percentage of responses at each ordinal level, per identity and metric."""
    rows = []
    for (model, prompt), cell in df.groupby([columns["model"], columns["prompt"]], dropna=False):
        for metric in metrics:
            for identity in identities:
                values = cell.loc[cell[columns["group"]] == identity, metric].dropna()
                total = len(values)
                row = {
                    "model": model,
                    "prompt": prompt,
                    "metric": metric,
                    "identity": identity,
                    "n": total,
                }
                for level in score_levels:
                    row[f"pct_score_{level}"] = (
                        round(100.0 * (values == level).sum() / total, 1) if total else np.nan
                    )
                row["mean"] = round(float(values.mean()), 3) if total else np.nan
                rows.append(row)
    return pd.DataFrame(rows)


def run(
    input_path: str | Path,
    output_path: str | Path,
    analysis_config_path: str | Path,
    rubric_config_path: str | Path,
    sheet: str | int = 0,
    distribution_output: str | Path | None = None,
    excel_output: str | Path | None = None,
) -> pd.DataFrame:
    """Execute the ordinal analysis end to end and write results to disk."""
    analysis_cfg = load_config(analysis_config_path)
    rubric = load_config(rubric_config_path)

    columns = analysis_cfg["columns"]
    identities = analysis_cfg["identities"]
    ordinal_opts = analysis_cfg.get("ordinal", {})
    alpha = ordinal_opts.get("alpha", 0.05)
    effect_threshold = ordinal_opts.get("effect_threshold", 0.3)
    min_n = ordinal_opts.get("min_n", 2)
    correction_family = ordinal_opts.get("correction_family", ["model"])
    include_word_count = ordinal_opts.get("include_word_count", True)
    score_levels = list(rubric.get("scale", {0: "", 1: "", 2: "", 3: ""}))
    score_levels = sorted(int(level) for level in score_levels)

    dimensions = list(rubric["dimensions"])
    metrics = (["word_count"] if include_word_count else []) + dimensions

    df = read_table(input_path, sheet=sheet)
    results = compare_groups(
        df,
        columns=columns,
        metrics=metrics,
        identities=identities,
        min_n=min_n,
    )

    results = add_fdr_correction(results, group_by=correction_family, alpha=alpha)
    results["meaningful_raw"] = (results["p_value"] < alpha) & (
        results["effect_size"].abs() >= effect_threshold
    )
    results["meaningful_adj"] = (results["p_value_adj"] < alpha) & (
        results["effect_size"].abs() >= effect_threshold
    )
    results = results.reindex(columns=OUTPUT_COLUMNS)
    results = results.sort_values(
        ["model", "prompt", "metric", "identity_1", "identity_2"]
    ).reset_index(drop=True)

    write_table(results, output_path)

    distribution = score_distribution(df, columns, dimensions, identities, score_levels)
    if distribution_output:
        write_table(distribution, distribution_output)

    if excel_output:
        write_excel_sheets(
            {
                "all_results": results,
                "meaningful_adj": results[results["meaningful_adj"].fillna(False)],
                "score_distribution": distribution,
            },
            excel_output,
        )

    write_manifest(
        Path(output_path).with_suffix(".manifest.json"),
        {
            "step": "mann_whitney",
            "input": str(input_path),
            "output": str(output_path),
            "n_comparisons": int(len(results)),
            "n_meaningful_raw": int(results["meaningful_raw"].fillna(False).sum()),
            "n_meaningful_adj": int(results["meaningful_adj"].fillna(False).sum()),
            "alpha": alpha,
            "effect_threshold": effect_threshold,
            "effect_size": "rank-biserial correlation (positive = identity_1 scores higher)",
            "correction_method": "benjamini-hochberg",
            "correction_family": correction_family,
            "metrics": metrics,
            "identities": identities,
        },
    )
    return results
