"""Tests for the LLM annotation module (Pipeline B, step 1).

These tests never contact the API: they exercise prompt construction, reply
parsing, pronoun neutralisation, and the cache.
"""

import json

import pytest

from adviceaudit.annotate import (
    AnnotationCache,
    ParseFailure,
    annotate_texts,
    build_system_prompt,
    mock_scores,
    neutralize_pronouns,
    parse_scores,
)

DIMENSIONS = ["Empathy", "Praise", "Criticism"]

RUBRIC = {
    "role_description": "You are a text classifier.",
    "scale": {0: "Absent.", 1: "Low.", 2: "Moderate.", 3: "High."},
    "dimensions": {
        "Empathy": {
            "definition": "empathy and acknowledgment of feelings",
            "examples": [{"score": 3, "text": "That sounds really hard."}],
        },
        "Praise": {"definition": "praise for behavior", "examples": []},
        "Criticism": {"definition": "criticism of behavior", "examples": []},
    },
}


# --- pronoun neutralisation ----------------------------------------------


def test_neutralize_replaces_gendered_pronouns():
    assert neutralize_pronouns("he was late") == "your partner was late"
    assert neutralize_pronouns("she was late") == "your partner was late"


def test_neutralize_preserves_leading_capitalisation():
    assert neutralize_pronouns("He was late") == "Your partner was late"


def test_neutralize_replaces_relationship_nouns():
    assert neutralize_pronouns("my boyfriend called") == "my your partner called"


def test_neutralize_leaves_other_words_alone():
    assert neutralize_pronouns("the meeting ran late") == "the meeting ran late"


def test_neutralize_respects_word_boundaries():
    """'here' contains 'her' but must not be rewritten."""
    assert neutralize_pronouns("come here") == "come here"


# --- prompt construction --------------------------------------------------


def test_system_prompt_contains_scale_and_dimensions():
    prompt = build_system_prompt(RUBRIC)
    for level in ("0 = Absent.", "3 = High."):
        assert level in prompt
    for dimension in DIMENSIONS:
        assert dimension in prompt


def test_system_prompt_includes_scored_examples():
    prompt = build_system_prompt(RUBRIC)
    assert "That sounds really hard." in prompt
    assert "Score: 3" in prompt


def test_system_prompt_states_the_required_keys():
    prompt = build_system_prompt(RUBRIC)
    assert "Empathy, Praise, Criticism" in prompt


# --- reply parsing --------------------------------------------------------


def test_parse_valid_json():
    raw = json.dumps({"Empathy": 2, "Praise": 0, "Criticism": 3})
    assert parse_scores(raw, DIMENSIONS) == {"Empathy": 2, "Praise": 0, "Criticism": 3}


def test_parse_strips_markdown_fences():
    raw = '```json\n{"Empathy": 1, "Praise": 1, "Criticism": 1}\n```'
    assert parse_scores(raw, DIMENSIONS)["Empathy"] == 1


def test_parse_ignores_surrounding_prose():
    raw = 'Here you go: {"Empathy": 1, "Praise": 2, "Criticism": 0} Hope that helps.'
    assert parse_scores(raw, DIMENSIONS)["Praise"] == 2


def test_parse_rounds_floats():
    raw = '{"Empathy": 2.0, "Praise": 1.4, "Criticism": 0}'
    assert parse_scores(raw, DIMENSIONS)["Praise"] == 1


def test_parse_rejects_missing_dimension():
    with pytest.raises(ParseFailure, match="missing dimension"):
        parse_scores('{"Empathy": 1, "Praise": 1}', DIMENSIONS)


def test_parse_rejects_out_of_range_score():
    with pytest.raises(ParseFailure, match="out of range"):
        parse_scores('{"Empathy": 7, "Praise": 1, "Criticism": 0}', DIMENSIONS)


def test_parse_rejects_non_numeric_score():
    with pytest.raises(ParseFailure, match="Non-numeric"):
        parse_scores('{"Empathy": "high", "Praise": 1, "Criticism": 0}', DIMENSIONS)


def test_parse_rejects_reply_without_json():
    with pytest.raises(ParseFailure, match="No JSON object"):
        parse_scores("I cannot score this text.", DIMENSIONS)


def test_parse_failure_is_not_silently_zero():
    """A failed parse must raise, never return zeros: a spurious 0 is
    indistinguishable from a genuine 'Absent' judgment."""
    for bad in ["", "no json", '{"Empathy": 1}']:
        with pytest.raises(ParseFailure):
            parse_scores(bad, DIMENSIONS)


# --- mock scorer ----------------------------------------------------------


def test_mock_scores_are_deterministic():
    assert mock_scores("hello", DIMENSIONS) == mock_scores("hello", DIMENSIONS)


def test_mock_scores_are_in_range():
    scores = mock_scores("some text", DIMENSIONS)
    assert set(scores) == set(DIMENSIONS)
    assert all(0 <= v <= 3 for v in scores.values())


def test_mock_scores_differ_across_texts():
    assert mock_scores("a", DIMENSIONS) != mock_scores("completely different", DIMENSIONS)


# --- cache ----------------------------------------------------------------


def test_cache_round_trip(tmp_path):
    cache = AnnotationCache(tmp_path / "cache.jsonl")
    key = cache.make_key("model-x", "hash1", "some text")
    assert cache.get(key) is None
    cache.put(key, "some text", {"Empathy": 2})
    assert cache.get(key) == {"Empathy": 2}


def test_cache_persists_across_instances(tmp_path):
    path = tmp_path / "cache.jsonl"
    key = AnnotationCache.make_key("model-x", "hash1", "text")
    AnnotationCache(path).put(key, "text", {"Empathy": 1})
    assert AnnotationCache(path).get(key) == {"Empathy": 1}


def test_cache_key_depends_on_model_and_rubric():
    base = AnnotationCache.make_key("m", "h", "text")
    assert AnnotationCache.make_key("other", "h", "text") != base
    assert AnnotationCache.make_key("m", "other", "text") != base
    assert AnnotationCache.make_key("m", "h", "other") != base


def test_cache_tolerates_truncated_final_line(tmp_path):
    path = tmp_path / "cache.jsonl"
    path.write_text('{"key": "a", "scores": {"Empathy": 1}}\n{"key": "b", "sco')
    cache = AnnotationCache(path)
    assert cache.get("a") == {"Empathy": 1}
    assert len(cache) == 1


def test_annotate_texts_uses_cache_and_reports_failures(tmp_path):
    cache = AnnotationCache(tmp_path / "cache.jsonl")
    calls = []

    def scorer(text):
        calls.append(text)
        if text == "bad":
            raise RuntimeError("no valid annotation")
        return {d: 1 for d in DIMENSIONS}

    annotations, failed = annotate_texts(
        ["good", "bad"], scorer, cache, "m", "h", max_workers=2, verbose=False
    )
    assert annotations == {"good": {d: 1 for d in DIMENSIONS}}
    assert failed == ["bad"]

    # Second run: "good" is cached, only "bad" is retried.
    calls.clear()
    annotate_texts(["good", "bad"], scorer, cache, "m", "h", max_workers=2, verbose=False)
    assert calls == ["bad"]
