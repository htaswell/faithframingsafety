"""Tests for the statistical helpers used by both pipelines."""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import mannwhitneyu

from adviceaudit.stats_utils import (
    add_fdr_correction,
    fisher_pair,
    mann_whitney_pair,
    rank_biserial,
)


# --- rank-biserial effect size -------------------------------------------


def test_rank_biserial_bounds():
    assert rank_biserial(0, 5, 5) == -1.0
    assert rank_biserial(25, 5, 5) == 1.0
    assert rank_biserial(12.5, 5, 5) == 0.0


def test_rank_biserial_sign_positive_when_group1_higher():
    """Positive r must mean group 1 scores higher. This is the documented
    convention and is the opposite sign to the formula 1 - 2U/(n1*n2)."""
    x1, x2 = [3, 3, 3, 2], [0, 0, 1, 0]
    u, _ = mannwhitneyu(x1, x2, alternative="two-sided")
    assert rank_biserial(u, len(x1), len(x2)) > 0


def test_rank_biserial_sign_negative_when_group2_higher():
    x1, x2 = [0, 0, 1, 0], [3, 3, 3, 2]
    u, _ = mannwhitneyu(x1, x2, alternative="two-sided")
    assert rank_biserial(u, len(x1), len(x2)) < 0


def test_rank_biserial_handles_empty_group():
    assert np.isnan(rank_biserial(0, 0, 5))


# --- Mann-Whitney ---------------------------------------------------------


def test_mann_whitney_detects_separation():
    result = mann_whitney_pair([3, 3, 3, 3, 2], [0, 0, 0, 1, 0])
    assert result["p_value"] < 0.05
    assert result["effect_size"] > 0.9
    assert result["n_1"] == 5


def test_mann_whitney_skips_small_groups():
    result = mann_whitney_pair([1], [2, 3, 4], min_n=2)
    assert np.isnan(result["p_value"])
    assert "skipped" in result["note"]


def test_mann_whitney_drops_missing_values():
    result = mann_whitney_pair([1, 2, np.nan], [3, 4, 5])
    assert result["n_1"] == 2


def test_mann_whitney_reports_summary_statistics():
    result = mann_whitney_pair([0, 2, 4], [1, 1, 1])
    assert result["mean_1"] == pytest.approx(2.0)
    assert result["median_1"] == pytest.approx(2.0)
    assert result["median_2"] == pytest.approx(1.0)


# --- Fisher's exact -------------------------------------------------------


def test_fisher_perfect_separation_is_significant():
    result = fisher_pair(n_present_1=10, n_total_1=10, n_present_2=0, n_total_2=10)
    assert result["p_value"] < 0.001
    assert result["pct_present_1"] == 100.0
    assert result["pct_present_2"] == 0.0


def test_fisher_no_difference_is_not_significant():
    result = fisher_pair(n_present_1=5, n_total_1=10, n_present_2=5, n_total_2=10)
    assert result["p_value"] == pytest.approx(1.0)
    assert result["odds_ratio"] == pytest.approx(1.0)


def test_fisher_haldane_is_finite_when_a_cell_is_zero():
    """The plain odds ratio is infinite here; the Haldane version is not."""
    result = fisher_pair(n_present_1=10, n_total_1=10, n_present_2=0, n_total_2=10)
    assert np.isinf(result["odds_ratio"])
    assert np.isfinite(result["odds_ratio_haldane"])


def test_fisher_skips_small_groups():
    result = fisher_pair(n_present_1=1, n_total_1=2, n_present_2=1, n_total_2=10, min_n=3)
    assert np.isnan(result["p_value"])
    assert "skipped" in result["note"]


def test_fisher_flags_category_absent_in_both():
    result = fisher_pair(n_present_1=0, n_total_1=10, n_present_2=0, n_total_2=10)
    assert result["note"] == "category absent in both groups"


def test_fisher_percentage_point_difference():
    result = fisher_pair(n_present_1=8, n_total_1=10, n_present_2=3, n_total_2=10)
    assert result["pct_point_diff"] == pytest.approx(50.0)


# --- multiple-comparison correction ---------------------------------------


def test_fdr_correction_raises_p_values():
    df = pd.DataFrame({"axis": ["a"] * 4, "model": ["m"] * 4, "p_value": [0.01, 0.02, 0.03, 0.04]})
    out = add_fdr_correction(df, group_by=["axis", "model"])
    assert (out["p_value_adj"] >= out["p_value"]).all()


def test_fdr_correction_is_applied_within_families():
    """Splitting tests into separate families leaves p-values untouched;
    pooling them into one family penalises the smaller p-value."""
    separate = pd.DataFrame({"axis": ["a", "b"], "model": ["m", "m"], "p_value": [0.01, 0.04]})
    out = add_fdr_correction(separate, group_by=["axis", "model"])
    assert out["p_value_adj"].tolist() == pytest.approx([0.01, 0.04])
    assert set(out["n_tests_in_family"]) == {1}

    together = pd.DataFrame({"axis": ["a", "a"], "model": ["m", "m"], "p_value": [0.01, 0.04]})
    out = add_fdr_correction(together, group_by=["axis", "model"])
    assert out["p_value_adj"].tolist() == pytest.approx([0.02, 0.04])
    assert set(out["n_tests_in_family"]) == {2}


def test_fdr_correction_does_not_inflate_tied_p_values():
    """Benjamini-Hochberg leaves a set of identical p-values unchanged.
    This is expected behaviour and distinguishes BH from Bonferroni."""
    df = pd.DataFrame({"axis": ["a", "a"], "model": ["m", "m"], "p_value": [0.04, 0.04]})
    out = add_fdr_correction(df, group_by=["axis", "model"])
    assert out["p_value_adj"].tolist() == pytest.approx([0.04, 0.04])


def test_fdr_correction_ignores_missing_p_values():
    df = pd.DataFrame({"axis": ["a"] * 3, "model": ["m"] * 3, "p_value": [0.01, np.nan, 0.04]})
    out = add_fdr_correction(df, group_by=["axis", "model"])
    assert np.isnan(out.loc[1, "p_value_adj"])
    assert out.loc[1, "significant_adj"] == False  # noqa: E712
    assert out["n_tests_in_family"].max() == 2


def test_fdr_correction_across_whole_table():
    """group_by=[] pools every row into a single correction family."""
    df = pd.DataFrame({"axis": ["a", "b"], "model": ["m", "n"], "p_value": [0.01, 0.04]})
    out = add_fdr_correction(df, group_by=[])
    assert out["p_value_adj"].tolist() == pytest.approx([0.02, 0.04])
    assert set(out["n_tests_in_family"]) == {2}


def test_fdr_correction_handles_all_missing():
    df = pd.DataFrame({"axis": ["a"], "model": ["m"], "p_value": [np.nan]})
    out = add_fdr_correction(df, group_by=["axis", "model"])
    assert np.isnan(out.loc[0, "p_value_adj"])
