"""Tests for the lexicon counting logic (Pipeline A, step 1)."""

import pandas as pd
import pytest

from adviceaudit.lexicon_counts import (
    build_category_terms,
    compile_category_patterns,
    count_category,
    count_lexicon,
    word_count,
)

CATEGORIES = {
    "Exit": ["leave", "walk away", "you deserve better", "you deserve"],
    "Control": ["control", "controlling"],
}


@pytest.fixture
def patterns():
    return compile_category_patterns(build_category_terms(CATEGORIES))


def test_terms_are_normalised_and_deduped():
    terms = build_category_terms({"C": ["  Leave ", "leave", "LEAVE", "walk  away"]})
    assert terms["C"] == ["walk away", "leave"]


def test_terms_sorted_longest_phrase_first():
    terms = build_category_terms(CATEGORIES)
    word_lengths = [len(t.split()) for t in terms["Exit"]]
    assert word_lengths == sorted(word_lengths, reverse=True)


def test_matching_is_case_insensitive(patterns):
    assert count_category("LEAVE now", patterns["Exit"]) == 1


def test_whole_word_matching_only(patterns):
    """'control' must not match inside 'controller'."""
    assert count_category("the controller was broken", patterns["Control"]) == 0
    assert count_category("he is controlling", patterns["Control"]) == 1


def test_longer_phrase_masks_shorter_one(patterns):
    """'you deserve better' must not also be counted as 'you deserve'."""
    assert count_category("you deserve better", patterns["Exit"]) == 3


def test_words_mode_credits_phrase_length(patterns):
    assert count_category("walk away", patterns["Exit"]) == 2


def test_matches_mode_credits_one_per_hit(patterns):
    assert count_category("walk away", patterns["Exit"], count_mode="matches") == 1


def test_repeated_hits_accumulate(patterns):
    assert count_category("leave and leave again", patterns["Exit"]) == 2


def test_no_match_returns_zero(patterns):
    assert count_category("nothing relevant here", patterns["Exit"]) == 0


def test_whitespace_is_normalised(patterns):
    assert count_category("walk    away", patterns["Exit"]) == 2
    assert count_category("walk\naway", patterns["Exit"]) == 2


def test_invalid_count_mode_raises(patterns):
    with pytest.raises(ValueError):
        count_category("leave", patterns["Exit"], count_mode="nonsense")


def test_word_count():
    assert word_count("one two three") == 3
    assert word_count("") == 0


def test_count_lexicon_adds_one_column_per_category():
    df = pd.DataFrame({"text": ["you deserve better", "the controller broke"]})
    out = count_lexicon(df, "text", CATEGORIES)
    assert list(out["count_Exit"]) == [3, 0]
    assert list(out["count_Control"]) == [0, 0]
    assert list(out["word_count"]) == [3, 3]


def test_count_lexicon_handles_missing_text():
    df = pd.DataFrame({"text": ["leave", None]})
    out = count_lexicon(df, "text", CATEGORIES)
    assert list(out["count_Exit"]) == [1, 0]


def test_count_lexicon_rejects_missing_column():
    df = pd.DataFrame({"other": ["leave"]})
    with pytest.raises(KeyError):
        count_lexicon(df, "text", CATEGORIES)


def test_categories_are_independent():
    """A term in two categories counts toward both."""
    shared = {"A": ["sin"], "B": ["sin"]}
    df = pd.DataFrame({"text": ["that is a sin"]})
    out = count_lexicon(df, "text", shared)
    assert out["count_A"].iloc[0] == 1
    assert out["count_B"].iloc[0] == 1
