"""Statistical helpers shared by the lexicon and LLM-annotation pipelines."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu
from statsmodels.stats.multitest import multipletests


def rank_biserial(u_statistic: float, n1: int, n2: int) -> float:
    """Rank-biserial correlation as an effect size for Mann-Whitney U.

    Parameters
    ----------
    u_statistic:
        The U statistic for **sample 1**, i.e. the value returned by
        ``scipy.stats.mannwhitneyu(x1, x2)`` with ``x1`` passed first.
    n1, n2:
        Sample sizes of group 1 and group 2.

    Returns
    -------
    float
        Effect size in [-1, 1]. **Positive means group 1 tends to score
        higher than group 2**; negative means group 2 scores higher.

    Notes
    -----
    Sign convention: ``r = 2U / (n1 * n2) - 1``. Some references (and an
    earlier version of this analysis) use ``1 - 2U / (n1 * n2)``, which has
    identical magnitude but the opposite sign. Because thresholds are applied
    to ``|r|``, significance decisions are unchanged either way, but the
    *direction* reported here follows the convention above.
    """
    if n1 <= 0 or n2 <= 0:
        return float("nan")
    return 2.0 * u_statistic / (n1 * n2) - 1.0


def mann_whitney_pair(x1, x2, min_n: int = 2) -> dict:
    """Two-sided Mann-Whitney U test with rank-biserial effect size.

    Returns a dict of results; ``p_value`` and ``effect_size`` are NaN when
    either group has fewer than ``min_n`` observations.
    """
    x1 = pd.Series(x1).dropna().to_numpy(dtype=float)
    x2 = pd.Series(x2).dropna().to_numpy(dtype=float)
    n1, n2 = len(x1), len(x2)

    result = {
        "n_1": n1,
        "n_2": n2,
        "mean_1": float(np.mean(x1)) if n1 else np.nan,
        "mean_2": float(np.mean(x2)) if n2 else np.nan,
        "median_1": float(np.median(x1)) if n1 else np.nan,
        "median_2": float(np.median(x2)) if n2 else np.nan,
        "u_statistic": np.nan,
        "p_value": np.nan,
        "effect_size": np.nan,
        "note": "",
    }

    if n1 < min_n or n2 < min_n:
        result["note"] = f"skipped: fewer than {min_n} observations in a group"
        return result

    if np.all(x1 == x1[0]) and np.all(x2 == x2[0]) and x1[0] == x2[0]:
        result["note"] = "constant in both groups"

    u_stat, p_value = mannwhitneyu(x1, x2, alternative="two-sided")
    result["u_statistic"] = float(u_stat)
    result["p_value"] = float(p_value)
    result["effect_size"] = rank_biserial(u_stat, n1, n2)
    return result


def fisher_pair(
    n_present_1: int, n_total_1: int, n_present_2: int, n_total_2: int, min_n: int = 3
) -> dict:
    """Two-sided Fisher's exact test on a 2x2 presence/absence table.

    The table is ``[[present_1, absent_1], [present_2, absent_2]]``.
    ``odds_ratio_haldane`` adds 0.5 to every cell so that a finite estimate is
    available when a cell is zero (the unadjusted odds ratio is then 0 or inf).
    """
    result = {
        "n_1": n_total_1,
        "n_2": n_total_2,
        "n_present_1": n_present_1,
        "n_present_2": n_present_2,
        "pct_present_1": 100.0 * n_present_1 / n_total_1 if n_total_1 else np.nan,
        "pct_present_2": 100.0 * n_present_2 / n_total_2 if n_total_2 else np.nan,
        "odds_ratio": np.nan,
        "odds_ratio_haldane": np.nan,
        "p_value": np.nan,
        "note": "",
    }
    result["pct_point_diff"] = result["pct_present_1"] - result["pct_present_2"]

    if n_total_1 < min_n or n_total_2 < min_n:
        result["note"] = f"skipped: fewer than {min_n} texts in a group"
        return result

    a, b = n_present_1, n_total_1 - n_present_1
    c, d = n_present_2, n_total_2 - n_present_2

    odds_ratio, p_value = fisher_exact([[a, b], [c, d]], alternative="two-sided")
    result["odds_ratio"] = float(odds_ratio)
    result["odds_ratio_haldane"] = float(((a + 0.5) * (d + 0.5)) / ((b + 0.5) * (c + 0.5)))
    result["p_value"] = float(p_value)
    if n_present_1 == 0 and n_present_2 == 0:
        result["note"] = "category absent in both groups"
    return result


def add_fdr_correction(
    results: pd.DataFrame,
    group_by: list[str],
    alpha: float = 0.05,
    p_col: str = "p_value",
    method: str = "fdr_bh",
) -> pd.DataFrame:
    """Add Benjamini-Hochberg adjusted p-values within each correction family.

    ``group_by`` defines the correction family (e.g. ``["axis", "model"]``).
    Rows with a missing p-value are left untouched and excluded from the
    correction. Pass an empty list to correct across the whole table at once.
    """
    results = results.copy()
    adj_col = f"{p_col}_adj"
    results[adj_col] = np.nan
    results["significant_adj"] = False
    results["correction_family"] = ""
    results["n_tests_in_family"] = np.nan

    valid = results[p_col].notna()
    if not valid.any():
        return results

    if group_by:
        # groupby(by=[x]) with a single key will yield 1-tuples in a future
        # pandas; passing the bare key when there is only one avoids the
        # deprecation warning and keeps scalar family labels.
        by = group_by[0] if len(group_by) == 1 else group_by
        families = results[valid].groupby(by, dropna=False).groups.items()
    else:
        families = [("all", results.index[valid])]

    for key, idx in families:
        idx = pd.Index(idx)
        pvals = results.loc[idx, p_col].to_numpy(dtype=float)
        rejected, p_adjusted, _, _ = multipletests(pvals, alpha=alpha, method=method)
        results.loc[idx, adj_col] = p_adjusted
        results.loc[idx, "significant_adj"] = rejected
        label = key if isinstance(key, str) else " | ".join(str(k) for k in key)
        results.loc[idx, "correction_family"] = label
        results.loc[idx, "n_tests_in_family"] = len(idx)

    return results
