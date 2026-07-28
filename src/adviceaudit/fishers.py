"""Pipeline A, step 2: binary keyword-presence comparison via Fisher's exact test.

Each response is reduced to present (>=1 keyword hit) or absent (0 hits) per
category. Within every (model, prompt) cell, every identity is compared
against every other identity, pairwise, on presence rates. p-values are
Benjamini-Hochberg corrected within each correction family (by default,
per model).
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import pandas as pd

from .io_utils import (
    load_config,
    read_table,
    require_columns,
    write_excel_sheets,
    write_manifest,
    write_table,
)
from .stats_utils import add_fdr_correction, fisher_pair

OUTPUT_COLUMNS = [
    "model",
    "prompt",
    "category",
    "identity_1",
    "identity_2",
    "n_1",
    "n_2",
    "n_present_1",
    "n_present_2",
    "pct_present_1",
    "pct_present_2",
    "pct_point_diff",
    "odds_ratio",
    "odds_ratio_haldane",
    "p_value",
    "p_value_adj",
    "significant_adj",
    "correction_family",
    "n_tests_in_family",
    "note",
]


def binarize(df: pd.DataFrame, count_columns: list[str], suffix: str = "__bin") -> pd.DataFrame:
    """Add a 0/1 presence column for each count column."""
    out = df.copy()
    for column in count_columns:
        out[column + suffix] = (out[column].fillna(0) > 0).astype(int)
    return out


def compare_presence(
    df: pd.DataFrame,
    columns: dict[str, str],
    categories: list[str],
    identities: list[str],
    count_prefix: str = "count_",
    min_n: int = 3,
) -> pd.DataFrame:
    """Run all pairwise Fisher's exact tests and return a tidy results table.

    Every identity in ``identities`` is compared against every other one,
    within each (model, prompt) cell, for each category.
    """
    count_columns = [f"{count_prefix}{c}" for c in categories]
    require_columns(
        df,
        [columns["model"], columns["prompt"], columns["group"], *count_columns],
        "Fisher's exact test",
    )
    binned = binarize(df, count_columns)

    rows = []
    for (model, prompt), cell in binned.groupby([columns["model"], columns["prompt"]], dropna=False):
        present = set(cell[columns["group"]].dropna().unique())
        cell_identities = [i for i in identities if i in present]
        for identity_1, identity_2 in combinations(sorted(cell_identities, key=str), 2):
            cell_1 = cell.loc[cell[columns["group"]] == identity_1]
            cell_2 = cell.loc[cell[columns["group"]] == identity_2]
            for category in categories:
                bin_col = f"{count_prefix}{category}__bin"
                x1 = cell_1[bin_col].dropna()
                x2 = cell_2[bin_col].dropna()
                stats = fisher_pair(
                    n_present_1=int(x1.sum()),
                    n_total_1=len(x1),
                    n_present_2=int(x2.sum()),
                    n_total_2=len(x2),
                    min_n=min_n,
                )
                rows.append(
                    {
                        "model": model,
                        "prompt": prompt,
                        "category": category,
                        "identity_1": identity_1,
                        "identity_2": identity_2,
                        **stats,
                    }
                )

    return pd.DataFrame(rows)


def run(
    input_path: str | Path,
    output_path: str | Path,
    analysis_config_path: str | Path,
    lexicon_config_path: str | Path,
    sheet: str | int = 0,
    excel_output: str | Path | None = None,
) -> pd.DataFrame:
    """Execute the Fisher's exact step end to end and write results to disk."""
    analysis_cfg = load_config(analysis_config_path)
    lexicon_cfg = load_config(lexicon_config_path)

    columns = analysis_cfg["columns"]
    identities = analysis_cfg["identities"]
    lexicon_opts = analysis_cfg.get("lexicon", {})
    fisher_opts = analysis_cfg.get("fisher", {})
    prefix = lexicon_opts.get("column_prefix", "count_")
    min_n = fisher_opts.get("min_n", 3)
    alpha = fisher_opts.get("alpha", 0.05)
    correction_family = fisher_opts.get("correction_family", ["model"])

    categories = list(lexicon_cfg["categories"])

    df = read_table(input_path, sheet=sheet)
    results = compare_presence(
        df,
        columns=columns,
        categories=categories,
        identities=identities,
        count_prefix=prefix,
        min_n=min_n,
    )

    results = add_fdr_correction(results, group_by=correction_family, alpha=alpha)
    results = results.reindex(columns=OUTPUT_COLUMNS)
    results = results.sort_values(
        ["model", "prompt", "category", "identity_1", "identity_2"]
    ).reset_index(drop=True)

    write_table(results, output_path)

    if excel_output:
        write_excel_sheets(
            {
                "all_results": results,
                "significant_only": results[results["significant_adj"].fillna(False)],
            },
            excel_output,
        )

    n_significant = int(results["significant_adj"].fillna(False).sum())
    write_manifest(
        Path(output_path).with_suffix(".manifest.json"),
        {
            "step": "fishers_exact",
            "input": str(input_path),
            "output": str(output_path),
            "n_comparisons": int(len(results)),
            "n_significant_adj": n_significant,
            "alpha": alpha,
            "min_n": min_n,
            "correction_method": "benjamini-hochberg",
            "correction_family": correction_family,
            "identities": identities,
        },
    )
    return results
