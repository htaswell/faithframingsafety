"""Pipeline A, step 1: dictionary-based keyword counting.

For every response text, counts how many words fall into each lexicon
category and writes one new column per category.

Counting rules
--------------
* Case-insensitive, whitespace-normalised.
* Whole-word matching only, so ``control`` does not match inside ``controller``.
* Multi-word phrases are matched as phrases. Longer phrases are matched first
  and then masked out, so ``you deserve better`` is not also counted as
  ``you deserve``.
* ``count_mode="words"`` credits a phrase with its own word count;
  ``count_mode="matches"`` credits each hit as 1.
* Categories are independent: a term listed in two categories counts toward
  both.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import pandas as pd

from .io_utils import (
    config_hash,
    load_config,
    read_table,
    require_columns,
    write_manifest,
    write_table,
)

CountMode = str  # "words" | "matches"


def build_category_terms(categories: dict[str, Iterable[str]]) -> dict[str, list[str]]:
    """Normalise and de-duplicate terms, sorted longest-phrase-first."""
    out: dict[str, list[str]] = {}
    for category, terms in categories.items():
        seen: set[str] = set()
        cleaned: list[str] = []
        for term in terms or []:
            normalised = re.sub(r"\s+", " ", str(term).strip().lower())
            if normalised and normalised not in seen:
                seen.add(normalised)
                cleaned.append(normalised)
        # Longest phrases first (by word count, then character length) so that
        # they are matched and masked before their shorter substrings.
        cleaned.sort(key=lambda s: (-len(s.split()), -len(s)))
        out[category] = cleaned
    return out


def compile_category_patterns(
    category_terms: dict[str, list[str]],
) -> dict[str, list[tuple[re.Pattern[str], int]]]:
    """Compile each term into a whole-word regex paired with its word count."""
    compiled: dict[str, list[tuple[re.Pattern[str], int]]] = {}
    for category, terms in category_terms.items():
        patterns = []
        for term in terms:
            body = r"\s+".join(re.escape(word) for word in term.split())
            pattern = re.compile(r"(?<!\w)" + body + r"(?!\w)")
            patterns.append((pattern, len(term.split())))
        compiled[category] = patterns
    return compiled


def count_category(
    text: str,
    patterns: list[tuple[re.Pattern[str], int]],
    count_mode: CountMode = "words",
) -> int:
    """Count lexicon hits for one category in one text."""
    if count_mode not in {"words", "matches"}:
        raise ValueError("count_mode must be 'words' or 'matches'")
    haystack = re.sub(r"\s+", " ", str(text).lower())
    total = 0
    for pattern, n_words in patterns:  # longest-first, so mask before shorter terms
        hits = list(pattern.finditer(haystack))
        if hits:
            total += len(hits) * (n_words if count_mode == "words" else 1)
            # Blank out the matched span so shorter terms cannot double-count it.
            haystack = pattern.sub(lambda m: " " * len(m.group()), haystack)
    return total


def word_count(text: str) -> int:
    """Whitespace-delimited word count for a response."""
    return len(str(text).split())


def count_lexicon(
    df: pd.DataFrame,
    text_col: str,
    categories: dict[str, Iterable[str]],
    count_mode: CountMode = "words",
    prefix: str = "count_",
) -> pd.DataFrame:
    """Add ``word_count`` and one count column per category to ``df``."""
    require_columns(df, [text_col], "Lexicon counting")
    out = df.copy()

    texts = out[text_col].fillna("").astype(str)
    unique_texts = pd.unique(texts)

    out["word_count"] = texts.map({t: word_count(t) for t in unique_texts})

    compiled = compile_category_patterns(build_category_terms(categories))
    for category, patterns in compiled.items():
        cache = {t: count_category(t, patterns, count_mode) for t in unique_texts}
        out[f"{prefix}{category}"] = texts.map(cache)

    return out


def run(
    input_path: str | Path,
    output_path: str | Path,
    analysis_config_path: str | Path,
    lexicon_config_path: str | Path,
    sheet: str | int = 0,
) -> pd.DataFrame:
    """Execute the counting step end to end and write results to disk."""
    analysis_cfg = load_config(analysis_config_path)
    lexicon_cfg = load_config(lexicon_config_path)

    columns = analysis_cfg["columns"]
    lexicon_opts = analysis_cfg.get("lexicon", {})
    count_mode = lexicon_opts.get("count_mode", "words")
    prefix = lexicon_opts.get("column_prefix", "count_")
    categories = lexicon_cfg["categories"]

    df = read_table(input_path, sheet=sheet)
    require_columns(
        df,
        [columns["id"], columns["text"], columns["model"], columns["prompt"], columns["group"]],
        f"Reading {input_path}",
    )

    counted = count_lexicon(
        df,
        text_col=columns["text"],
        categories=categories,
        count_mode=count_mode,
        prefix=prefix,
    )

    write_table(counted, output_path)
    write_manifest(
        Path(output_path).with_suffix(".manifest.json"),
        {
            "step": "lexicon_counts",
            "input": str(input_path),
            "output": str(output_path),
            "n_rows": int(len(counted)),
            "count_mode": count_mode,
            "column_prefix": prefix,
            "n_categories": len(categories),
            "categories": list(categories),
            "lexicon_config_hash": config_hash(lexicon_cfg),
        },
    )
    return counted
