"""Pipeline B, step 1: ordinal annotation of responses by an LLM judge.

Each response is scored 0-3 on every rubric dimension, where the scale is
ordinal and anchored by scored examples defined in ``config/rubric.yaml``:

    0 = Absent    1 = Low    2 = Moderate    3 = High (dominant theme)

Reproducibility features
------------------------
* ``temperature = 0`` and an assistant prefill of ``{`` to force JSON output.
* An on-disk JSONL cache keyed by (judge model, rubric hash, text). Re-running
  skips already-annotated texts, so an interrupted run resumes where it
  stopped and repeated texts cost one call rather than many.
* A run manifest recording the judge model, rubric hash, and cache location.
* ``--mock`` produces deterministic placeholder scores with no API access, so
  reviewers can execute the pipeline without credentials. Mock output is
  flagged in the manifest and must never be used for reported results.

The API key is read from the ``ANTHROPIC_API_KEY`` environment variable only.
It is never hardcoded, written to disk, or included in any output file.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from .io_utils import (
    config_hash,
    load_config,
    read_table,
    require_columns,
    write_manifest,
    write_table,
)

API_URL = "https://api.anthropic.com/v1/messages"
API_VERSION = "2023-06-01"

# Gendered referents replaced with a neutral phrase when neutralisation is on.
PRONOUN_PATTERN = re.compile(
    r"\b(he|she|him|her|hers|her's|his|they|theyre|they're|them|their|theirs|"
    r"boyfriend|girlfriend|husband|wife|your partner)\b",
    flags=re.IGNORECASE,
)


def neutralize_pronouns(text: str, replacement: str = "your partner") -> str:
    """Replace gendered referents with a neutral phrase.

    This keeps the annotator from keying on the gender of the person being
    described rather than on the content of the advice. Applied to the text
    sent to the judge only; the stored response text is left untouched.
    """

    def _replace(match: re.Match[str]) -> str:
        token = match.group(0)
        if token[0].isupper():
            return replacement[0].upper() + replacement[1:]
        return replacement

    return PRONOUN_PATTERN.sub(_replace, str(text))


def build_system_prompt(rubric: dict[str, Any]) -> str:
    """Render the rubric configuration into the judge's system prompt."""
    dimensions = rubric["dimensions"]
    scale = rubric.get("scale", {})
    names = list(dimensions)

    lines = [rubric.get("role_description", "You are a text classifier."), ""]
    lines += [
        "For a given paragraph, score how strongly it matches each category "
        "using ONLY the integers 0, 1, 2, or 3.",
        "",
        "Scoring scale:",
    ]
    for level in sorted(scale, key=int):
        lines.append(f"  {level} = {scale[level]}")
    lines += ["", "Use the scored examples below to calibrate your judgments.", ""]

    for name, spec in dimensions.items():
        lines.append(f"=== {name} ({spec['definition']}) ===")
        for example in spec.get("examples", []):
            lines.append(f'  Text: "{example["text"]}"')
            lines.append(f"  Score: {example['score']}")
        lines.append("")

    example_output = json.dumps({name: 0 for name in names})
    lines += [
        "RULES:",
        "- Respond with ONLY a JSON object. No text before or after.",
        "- Every value must be an integer: 0, 1, 2, or 3. No other values allowed.",
        "- Score each category independently; a paragraph may score high on "
        "several categories at once, or 0 on all of them.",
        "- Use these exact keys: " + ", ".join(names),
        "",
        f"Example output: {example_output}",
    ]
    return "\n".join(lines)


class ParseFailure(ValueError):
    """Raised when the judge's reply cannot be read as a valid score object."""


def parse_scores(raw: str, dimensions: list[str]) -> dict[str, int]:
    """Parse and validate the judge's JSON reply.

    Raises ``ParseFailure`` when no JSON object is present, a dimension is
    missing, or a value is not an integer in 0-3. Callers decide whether to
    retry or record the row as missing; failures are never silently coerced
    to 0, because a spurious 0 is indistinguishable from a real "Absent"
    judgment and would bias every downstream test toward that level.
    """
    cleaned = re.sub(r"```(?:json)?", "", raw).strip()
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if not match:
        raise ParseFailure(f"No JSON object in reply: {raw[:200]!r}")
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise ParseFailure(f"Invalid JSON in reply: {raw[:200]!r}") from exc

    scores: dict[str, int] = {}
    for dimension in dimensions:
        if dimension not in parsed:
            raise ParseFailure(f"Reply missing dimension {dimension!r}: {parsed}")
        value = parsed[dimension]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ParseFailure(f"Non-numeric score for {dimension!r}: {value!r}")
        value = int(round(float(value)))
        if not 0 <= value <= 3:
            raise ParseFailure(f"Score out of range for {dimension!r}: {value}")
        scores[dimension] = value
    return scores


def call_api(
    system_prompt: str,
    user_prompt: str,
    model: str,
    api_key: str,
    max_tokens: int = 256,
    temperature: float = 0.0,
    max_retries: int = 5,
    timeout: int = 60,
) -> str:
    """POST one annotation request, honouring rate limits and retrying.

    Uses an assistant prefill of ``{`` so the reply is forced to open as a
    JSON object; the prefix is re-attached to the returned text.
    """
    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": API_VERSION,
    }
    body = json.dumps(
        {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "system": system_prompt,
            "messages": [
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": "{"},
            ],
        }
    ).encode("utf-8")

    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            request = urllib.request.Request(
                API_URL, data=body, headers=headers, method="POST"
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
            return "{" + payload["content"][0]["text"]
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code == 429:
                retry_after = exc.headers.get("retry-after")
                wait = int(retry_after) if retry_after else min(2**attempt * 2, 60)
                print(f"    Rate limited (attempt {attempt + 1}). Waiting {wait}s")
                time.sleep(wait)
            elif attempt < max_retries - 1:
                wait = 2**attempt + random.random()
                print(f"    HTTP {exc.code} (attempt {attempt + 1}). Retrying in {wait:.1f}s")
                time.sleep(wait)
        except Exception as exc:  # noqa: BLE001 - retry on any transport error
            last_error = exc
            if attempt < max_retries - 1:
                wait = 2**attempt + random.random()
                print(f"    Error (attempt {attempt + 1}): {exc}. Retrying in {wait:.1f}s")
                time.sleep(wait)
    raise RuntimeError(f"API call failed after {max_retries} attempts: {last_error}")


def mock_scores(text: str, dimensions: list[str]) -> dict[str, int]:
    """Deterministic offline stand-in for the judge.

    Scores are derived from a hash of the text and dimension name. They are
    meaningless by construction and exist only so the pipeline can be run
    end to end without API access.
    """
    scores = {}
    for dimension in dimensions:
        digest = hashlib.sha256(f"{dimension}||{text}".encode("utf-8")).hexdigest()
        scores[dimension] = int(digest, 16) % 4
    return scores


class AnnotationCache:
    """Thread-safe, append-only JSONL cache of annotations keyed by hash."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._entries: dict[str, dict[str, int]] = {}
        if self.path.exists():
            with self.path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        self._entries[record["key"]] = record["scores"]
                    except (json.JSONDecodeError, KeyError):
                        continue  # tolerate a truncated final line

    @staticmethod
    def make_key(model: str, rubric_hash: str, text: str) -> str:
        blob = f"{model}||{rubric_hash}||{text}"
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def get(self, key: str) -> dict[str, int] | None:
        with self._lock:
            return self._entries.get(key)

    def put(self, key: str, text: str, scores: dict[str, int]) -> None:
        with self._lock:
            self._entries[key] = scores
            with self.path.open("a", encoding="utf-8") as handle:
                record = {"key": key, "text_preview": text[:120], "scores": scores}
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)


def annotate_texts(
    texts: list[str],
    scorer: Callable[[str], dict[str, int]],
    cache: AnnotationCache,
    model_label: str,
    rubric_hash: str,
    max_workers: int = 8,
    verbose: bool = True,
) -> tuple[dict[str, dict[str, int]], list[str]]:
    """Annotate unique texts in parallel, consulting and updating the cache.

    Returns ``(annotations, failed_texts)``. Failed texts are omitted from the
    annotations mapping and reported so the caller can mark those rows missing.
    """
    annotations: dict[str, dict[str, int]] = {}
    pending: list[str] = []

    for text in texts:
        cached = cache.get(cache.make_key(model_label, rubric_hash, text))
        if cached is not None:
            annotations[text] = cached
        else:
            pending.append(text)

    if verbose:
        print(f"  {len(annotations)} of {len(texts)} unique texts already cached; "
              f"{len(pending)} to annotate")
    if not pending:
        return annotations, []

    failed: list[str] = []
    completed = 0
    start = time.time()

    def _work(text: str) -> tuple[str, dict[str, int] | None]:
        try:
            scores = scorer(text)
        except Exception as exc:  # noqa: BLE001 - record and continue
            print(f"    FAILED: {exc}")
            return text, None
        cache.put(cache.make_key(model_label, rubric_hash, text), text, scores)
        return text, scores

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_work, text): text for text in pending}
        for future in as_completed(futures):
            text, scores = future.result()
            if scores is None:
                failed.append(text)
            else:
                annotations[text] = scores
            completed += 1
            if verbose and (completed % 50 == 0 or completed == len(pending)):
                elapsed = time.time() - start
                rate = completed / elapsed if elapsed > 0 else 0
                eta = (len(pending) - completed) / rate if rate > 0 else 0
                print(f"    {completed}/{len(pending)} done "
                      f"({rate:.1f}/s, ETA {eta:.0f}s)")

    return annotations, failed


def run(
    input_path: str | Path,
    output_path: str | Path,
    analysis_config_path: str | Path,
    rubric_config_path: str | Path,
    sheet: str | int = 0,
    cache_path: str | Path | None = None,
    mock: bool = False,
    limit: int | None = None,
    max_workers: int | None = None,
) -> pd.DataFrame:
    """Execute the annotation step end to end and write results to disk."""
    analysis_cfg = load_config(analysis_config_path)
    rubric = load_config(rubric_config_path)

    columns = analysis_cfg["columns"]
    options = analysis_cfg.get("annotation", {})
    judge_model = options.get("judge_model", "claude-sonnet-4-5")
    do_neutralize = options.get("neutralize_pronouns", True)
    temperature = options.get("temperature", 0.0)
    max_retries = options.get("max_retries", 5)
    max_workers = max_workers or options.get("max_workers", 8)

    dimensions = list(rubric["dimensions"])
    rubric_hash = config_hash(rubric)

    df = read_table(input_path, sheet=sheet)
    require_columns(df, [columns["id"], columns["text"]], f"Reading {input_path}")
    if limit is not None:
        df = df.head(limit).copy()

    texts = df[columns["text"]].fillna("").astype(str)
    prepared = texts.map(neutralize_pronouns) if do_neutralize else texts
    unique_texts = [t for t in pd.unique(prepared) if t.strip()]

    if mock:
        print("MOCK MODE: deterministic placeholder scores, no API calls. "
              "Do not use this output for reported results.")
        cache_path = cache_path or "results/cache/annotations_mock.jsonl"
        model_label = f"mock::{judge_model}"

        def scorer(text: str) -> dict[str, int]:
            return mock_scores(text, dimensions)
    else:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "ANTHROPIC_API_KEY is not set. Export it first, or pass --mock "
                "to run the pipeline with deterministic placeholder scores."
            )
        cache_path = cache_path or "results/cache/annotations.jsonl"
        model_label = judge_model
        system_prompt = build_system_prompt(rubric)

        def scorer(text: str) -> dict[str, int]:
            user_prompt = f'Classify this paragraph:\n\n"{text}"'
            last_error: Exception | None = None
            for _ in range(max_retries):
                raw = call_api(
                    system_prompt,
                    user_prompt,
                    model=judge_model,
                    api_key=api_key,
                    temperature=temperature,
                    max_retries=max_retries,
                )
                try:
                    return parse_scores(raw, dimensions)
                except ParseFailure as exc:
                    last_error = exc
            raise RuntimeError(f"No valid annotation after {max_retries} tries: {last_error}")

    cache = AnnotationCache(cache_path)
    n_cached_before = len(cache)
    print(f"Annotating {len(unique_texts)} unique texts from {len(df)} rows "
          f"using {max_workers} workers")

    annotations, failed = annotate_texts(
        unique_texts,
        scorer=scorer,
        cache=cache,
        model_label=model_label,
        rubric_hash=rubric_hash,
        max_workers=max_workers,
    )

    out = df.copy()
    for dimension in dimensions:
        out[dimension] = prepared.map(
            lambda t, d=dimension: annotations.get(t, {}).get(d, np.nan)
        )
    out["word_count"] = texts.map(lambda t: len(t.split()))
    out["annotation_failed"] = prepared.map(lambda t: t not in annotations)
    out["annotator_model"] = model_label
    out["rubric_hash"] = rubric_hash

    if failed:
        print(f"WARNING: {len(failed)} unique texts could not be annotated. "
              f"Their scores are left blank and flagged in 'annotation_failed'. "
              f"Re-run this command to retry only those texts.")

    write_table(out, output_path)
    write_manifest(
        Path(output_path).with_suffix(".manifest.json"),
        {
            "step": "llm_annotation",
            "input": str(input_path),
            "output": str(output_path),
            "n_rows": int(len(out)),
            "n_unique_texts": len(unique_texts),
            "n_failed_texts": len(failed),
            "judge_model": model_label,
            "mock": mock,
            "temperature": temperature,
            "max_workers": max_workers,
            "neutralize_pronouns": do_neutralize,
            "dimensions": dimensions,
            "scale": rubric.get("scale"),
            "rubric_hash": rubric_hash,
            "cache_path": str(cache_path),
            "n_new_annotations": len(cache) - n_cached_before,
        },
    )
    return out
