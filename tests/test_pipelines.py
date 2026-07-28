"""End-to-end tests: both pipelines run start to finish on synthetic data.

Pipeline B runs in mock mode, so the whole suite executes with no API key.
"""

import pandas as pd
import pytest
import yaml

from adviceaudit import fishers, lexicon_counts, mann_whitney
from adviceaudit.annotate import run as annotate_run

ANALYSIS_CONFIG = {
    "columns": {
        "id": "ID",
        "text": "text",
        "model": "model",
        "prompt": "prompt_number",
        "group": "identity",
    },
    "identities": ["christian", "jewish", "muslim", "male", "female"],
    "lexicon": {"count_mode": "words", "column_prefix": "count_"},
    "fisher": {"alpha": 0.05, "min_n": 3, "correction_family": ["model"]},
    "annotation": {"judge_model": "test-model", "neutralize_pronouns": True, "max_workers": 2},
    "ordinal": {
        "alpha": 0.05,
        "effect_threshold": 0.3,
        "min_n": 2,
        "include_word_count": True,
        "correction_family": ["model"],
    },
}

LEXICON_CONFIG = {
    "categories": {
        "Exit": ["leave", "walk away"],
        "Religious": ["pray", "god", "rabbi"],
    }
}

RUBRIC_CONFIG = {
    "role_description": "You are a text classifier.",
    "scale": {0: "Absent.", 1: "Low.", 2: "Moderate.", 3: "High."},
    "dimensions": {
        "Empathy": {"definition": "empathy", "examples": []},
        "Religion": {"definition": "religious framing", "examples": []},
    },
}

TEXTS = {
    "christian": "You should pray about this and consider whether to walk away.",
    "jewish": "Speak to your rabbi, and pray before you leave.",
    "muslim": "Pray on this and consider whether to leave or walk away.",
    "male": "Consider whether to leave; it may help to walk away.",
    "female": "You could leave, or pray on it before you walk away.",
}


@pytest.fixture
def workspace(tmp_path):
    """A temporary directory holding configs and a synthetic input table."""
    for name, config in [
        ("analysis.yaml", ANALYSIS_CONFIG),
        ("lexicon.yaml", LEXICON_CONFIG),
        ("rubric.yaml", RUBRIC_CONFIG),
    ]:
        (tmp_path / name).write_text(yaml.safe_dump(config), encoding="utf-8")

    rows = []
    for model in ["model_a", "model_b"]:
        for prompt in [1, 2]:
            for identity, text in TEXTS.items():
                # Six per cell: enough that a perfectly separated category
                # survives BH correction (Fisher p = 0.0022 at 6 vs 6).
                for i in range(6):
                    rows.append(
                        {
                            "ID": f"{model}_{prompt}_{identity}_{i}",
                            "model": model,
                            "prompt_number": prompt,
                            "identity": identity,
                            "text": text if i % 2 == 0 else text + " Take your time.",
                        }
                    )
    (tmp_path / "input.csv").write_text(pd.DataFrame(rows).to_csv(index=False), encoding="utf-8")
    return tmp_path


# --- Pipeline A -----------------------------------------------------------


def test_lexicon_pipeline_end_to_end(workspace):
    counts_path = workspace / "counts.csv"
    counted = lexicon_counts.run(
        input_path=workspace / "input.csv",
        output_path=counts_path,
        analysis_config_path=workspace / "analysis.yaml",
        lexicon_config_path=workspace / "lexicon.yaml",
    )
    assert "count_Exit" in counted.columns
    assert "count_Religious" in counted.columns
    assert "word_count" in counted.columns
    assert counts_path.exists()
    assert counts_path.with_suffix(".manifest.json").exists()

    results_path = workspace / "fisher.csv"
    results = fishers.run(
        input_path=counts_path,
        output_path=results_path,
        analysis_config_path=workspace / "analysis.yaml",
        lexicon_config_path=workspace / "lexicon.yaml",
    )
    assert len(results) > 0
    # No axis column any more; identities are compared pairwise.
    assert "axis" not in results.columns
    assert "identity_1" in results.columns
    # Every identity is compared against every other, including across the
    # former religion/gender split (e.g. christian vs male).
    pairs = set(zip(results["identity_1"], results["identity_2"]))
    assert ("christian", "male") in pairs or ("male", "christian") in pairs
    # Adjusted p-values are never smaller than raw p-values.
    valid = results["p_value"].notna()
    assert (results.loc[valid, "p_value_adj"] >= results.loc[valid, "p_value"] - 1e-12).all()


def test_fisher_detects_a_planted_difference(workspace):
    """A category present in every christian response and no jewish response
    must come out significant."""
    df = pd.read_csv(workspace / "input.csv")
    df["count_Exit"] = 0
    df["count_Religious"] = (df["identity"] == "christian").astype(int) * 5
    path = workspace / "planted.csv"
    df.to_csv(path, index=False)

    results = fishers.run(
        input_path=path,
        output_path=workspace / "planted_fisher.csv",
        analysis_config_path=workspace / "analysis.yaml",
        lexicon_config_path=workspace / "lexicon.yaml",
    )
    religious = results[
        (results["category"] == "Religious")
        & (results["identity_1"] == "christian")
        & (results["identity_2"] == "jewish")
    ]
    assert len(religious) > 0
    assert (religious["p_value"] < 0.05).all()
    assert religious["significant_adj"].any()


# --- Pipeline B -----------------------------------------------------------


def test_llm_pipeline_end_to_end_in_mock_mode(workspace):
    annotated_path = workspace / "annotated.csv"
    annotated = annotate_run(
        input_path=workspace / "input.csv",
        output_path=annotated_path,
        analysis_config_path=workspace / "analysis.yaml",
        rubric_config_path=workspace / "rubric.yaml",
        cache_path=workspace / "cache.jsonl",
        mock=True,
    )
    assert "Empathy" in annotated.columns
    assert "Religion" in annotated.columns
    assert annotated["Empathy"].between(0, 3).all()
    assert not annotated["annotation_failed"].any()

    results_path = workspace / "ordinal.csv"
    results = mann_whitney.run(
        input_path=annotated_path,
        output_path=results_path,
        analysis_config_path=workspace / "analysis.yaml",
        rubric_config_path=workspace / "rubric.yaml",
        distribution_output=workspace / "distribution.csv",
    )
    assert len(results) > 0
    assert "axis" not in results.columns
    assert "identity_1" in results.columns
    assert "word_count" in set(results["metric"])
    assert results["effect_size"].dropna().between(-1, 1).all()
    assert (workspace / "distribution.csv").exists()


def test_annotation_is_reproducible_in_mock_mode(workspace):
    """Two runs with the same inputs give identical scores."""
    first = annotate_run(
        input_path=workspace / "input.csv",
        output_path=workspace / "a1.csv",
        analysis_config_path=workspace / "analysis.yaml",
        rubric_config_path=workspace / "rubric.yaml",
        cache_path=workspace / "cache1.jsonl",
        mock=True,
    )
    second = annotate_run(
        input_path=workspace / "input.csv",
        output_path=workspace / "a2.csv",
        analysis_config_path=workspace / "analysis.yaml",
        rubric_config_path=workspace / "rubric.yaml",
        cache_path=workspace / "cache2.jsonl",
        mock=True,
    )
    pd.testing.assert_series_equal(first["Empathy"], second["Empathy"])


def test_ordinal_detects_a_planted_difference(workspace):
    """A dimension where christian scores 3 and jewish scores 0 must be
    flagged meaningful."""
    df = pd.read_csv(workspace / "input.csv")
    df["Empathy"] = 1
    df["Religion"] = df["identity"].map({"christian": 3, "jewish": 0}).fillna(1).astype(int)
    df["word_count"] = 20
    path = workspace / "planted_scores.csv"
    df.to_csv(path, index=False)

    results = mann_whitney.run(
        input_path=path,
        output_path=workspace / "planted_ordinal.csv",
        analysis_config_path=workspace / "analysis.yaml",
        rubric_config_path=workspace / "rubric.yaml",
    )
    religion = results[
        (results["metric"] == "Religion")
        & (results["identity_1"] == "christian")
        & (results["identity_2"] == "jewish")
    ]
    assert len(religion) > 0
    # christian is identity_1 and scores higher, so the effect size is positive.
    assert (religion["effect_size"] > 0.9).all()
    assert religion["meaningful_raw"].all()


def test_every_identity_is_compared_against_every_other(workspace):
    """With 5 identities there are 10 unordered pairs per (model, prompt,
    metric) cell."""
    annotate_run(
        input_path=workspace / "input.csv",
        output_path=workspace / "annotated.csv",
        analysis_config_path=workspace / "analysis.yaml",
        rubric_config_path=workspace / "rubric.yaml",
        cache_path=workspace / "cache.jsonl",
        mock=True,
    )
    results = mann_whitney.run(
        input_path=workspace / "annotated.csv",
        output_path=workspace / "ordinal.csv",
        analysis_config_path=workspace / "analysis.yaml",
        rubric_config_path=workspace / "rubric.yaml",
    )
    one_cell = results[
        (results["model"] == "model_a")
        & (results["prompt"] == 1)
        & (results["metric"] == "Empathy")
    ]
    pairs = set(zip(one_cell["identity_1"], one_cell["identity_2"]))
    assert len(pairs) == 10  # C(5, 2)


def test_identity_not_in_config_is_ignored(workspace):
    """An identity present in the data but absent from the config's identity
    list is never tested."""
    df = pd.read_csv(workspace / "input.csv")
    extra = df[df["identity"] == "male"].copy()
    extra["identity"] = "nonbinary"  # not listed in ANALYSIS_CONFIG["identities"]
    combined = pd.concat([df, extra], ignore_index=True)
    combined["count_Exit"] = 1
    combined["count_Religious"] = 1
    path = workspace / "with_extra.csv"
    combined.to_csv(path, index=False)

    results = fishers.run(
        input_path=path,
        output_path=workspace / "extra_fisher.csv",
        analysis_config_path=workspace / "analysis.yaml",
        lexicon_config_path=workspace / "lexicon.yaml",
    )
    all_identities = set(results["identity_1"]) | set(results["identity_2"])
    assert "nonbinary" not in all_identities


def test_missing_required_column_raises_readable_error(workspace):
    df = pd.read_csv(workspace / "input.csv").drop(columns=["identity"])
    path = workspace / "broken.csv"
    df.to_csv(path, index=False)
    with pytest.raises(KeyError, match="identity"):
        lexicon_counts.run(
            input_path=path,
            output_path=workspace / "out.csv",
            analysis_config_path=workspace / "analysis.yaml",
            lexicon_config_path=workspace / "lexicon.yaml",
        )
