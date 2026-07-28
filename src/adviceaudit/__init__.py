"""Reproducible analysis pipelines for auditing LLM advice responses.

Two independent pipelines operate on the same input table of responses:

Pipeline A (lexicon):  count  ->  fisher
    Dictionary keyword counts per category, then Fisher's exact tests on
    keyword presence/absence between demographic groups.

Pipeline B (LLM ordinal):  annotate  ->  ordinal
    Ordinal 0-3 annotation of each response by an LLM judge, then
    Mann-Whitney U tests with rank-biserial effect sizes between groups.

Neither pipeline depends on the other's output.
"""

__version__ = "1.0.0"

__all__ = [
    "annotate",
    "fishers",
    "io_utils",
    "lexicon_counts",
    "mann_whitney",
    "plots",
    "stats_utils",
]
