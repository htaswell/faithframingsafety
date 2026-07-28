# Auditing Demographic Variation in LLM Relationship Advice

Reproducible analysis code for the accompanying paper. Given a table of model
responses labelled with a demographic condition, the repository runs two
**independent** analyses and reports where responses differ significantly
between groups.

| | Pipeline A — Lexicon | Pipeline B — LLM ordinal |
|---|---|---|
| **What it measures** | Whether a response contains any keyword from each of 16 themed categories | How strongly a response expresses each of 8 rhetorical dimensions, scored 0–3 |
| **How responses are scored** | Dictionary matching (deterministic, no model) | An LLM judge with a fixed anchored rubric |
| **Outcome variable** | Binary: category present / absent | Ordinal: 0, 1, 2, 3 |
| **Statistical test** | Fisher's exact test | Mann-Whitney U |
| **Effect size** | Odds ratio | Rank-biserial correlation |
| **Needs an API key** | No | Yes (or `--mock`) |

The two pipelines read the same input file and never read each other's output.
Either can be run alone.

---

## Quick start

```bash
git clone <repository-url>
cd <repository-name>

python -m pip install -r requirements.txt
python -m pip install -e .          # optional: installs the `adviceaudit` command

make test        # 68 tests, no API key required
make example     # runs both pipelines on synthetic data, no API key required
```

`make example` writes results to `results/example/`. It uses a generated
synthetic dataset and mock annotations, so it verifies that the code runs
without touching real data or the API.

---

## Input data format

One row per model response. Column names are configurable in
`config/analysis.yaml`; the defaults are:

| Column | Meaning | Example |
|---|---|---|
| `ID` | Unique identifier | `resp_0003` |
| `prompt_number` | Scenario/prompt identifier | `1` |
| `model` | System that produced it | `model_a` |
| `identity` | Demographic condition being compared | `muslim` |
| `text` | The response to analyse | `"That sounds really difficult…"` |

CSV, TSV, and Excel inputs are all accepted. See
[`data/README.md`](data/README.md) for details and
`data/example/responses_example.csv` for a working example.

To use different column names, edit the `columns` block in
`config/analysis.yaml` rather than renaming your data.

---

## Running the pipelines on your own data

Place your file at `data/raw/responses.csv`, then:

```bash
make lexicon                              # Pipeline A
make llm                                  # Pipeline B (needs ANTHROPIC_API_KEY)
make all                                  # both, plus figures

make lexicon INPUT=data/raw/other.xlsx    # override the input path
```

Or call the steps directly for more control:

```bash
# Pipeline A
python -m adviceaudit count  --input data/raw/responses.csv \
                             --output results/lexicon/counts.csv
python -m adviceaudit fisher --input results/lexicon/counts.csv \
                             --output results/lexicon/fisher_results.csv

# Pipeline B
export ANTHROPIC_API_KEY="sk-ant-..."
python -m adviceaudit annotate --input data/raw/responses.csv \
                               --output results/llm/annotated.csv
python -m adviceaudit ordinal  --input results/llm/annotated.csv \
                               --output results/llm/ordinal_results.csv \
                               --distribution-output results/llm/score_distribution.csv
python -m adviceaudit figures  --input results/llm/annotated.csv \
                               --output-dir results/llm/figures
```

Every command accepts `--help`.

---

## Pipeline A — lexicon and Fisher's exact test

**Step 1, `count`.** Each response is scanned for the 992 terms in
`config/lexicon.yaml`, grouped into 16 categories. Matching is
case-insensitive and whole-word, so `control` does not match inside
`controller`. Multi-word phrases are matched as phrases, longest first, and
masked once matched, so `you deserve better` is not also counted as
`you deserve`. Output adds `word_count` and one `count_<category>` column per
category.

Categories are **not** mutually exclusive: a term listed under two categories
counts toward both, so category counts should not be summed.

**Step 2, `fisher`.** Counts are reduced to presence (≥1 hit) or absence (0
hits). Within each (model, prompt_number) cell, every identity is compared
against every other identity, pairwise, with a two-sided Fisher's exact test.
p-values are Benjamini–Hochberg corrected within each correction family, by
default per model.

Both the plain odds ratio and a Haldane-corrected version (0.5 added to every
cell) are reported. The plain odds ratio is 0 or infinite when a cell is empty,
which is common with small identities; the Haldane version stays finite and is
the one to plot.

---

## Pipeline B — LLM annotation and Mann-Whitney U

**Step 1, `annotate`.** Each response is scored 0–3 on eight dimensions —
Empathy, Praise, Criticism, Warning, Reconciliation, Confrontation, Guidance,
Religion — by an LLM judge, using the anchored rubric in `config/rubric.yaml`:

```
0 = Absent     1 = Low     2 = Moderate     3 = High (dominant theme)
```

Reproducibility measures:

- `temperature = 0`, with an assistant prefill of `{` to force JSON output.
- Every reply is validated: any missing dimension, non-integer value, or score
  outside 0–3 triggers a retry. **Unparseable replies are never silently
  recorded as 0** — persistent failures leave the row blank and set
  `annotation_failed`, because a spurious 0 is indistinguishable from a real
  "Absent" judgment and would bias every downstream test toward that level.
- Scores are cached in `results/cache/annotations.jsonl`, keyed by
  (judge model, rubric hash, text). Re-running skips cached texts, so an
  interrupted run resumes and repeated texts cost one call rather than many.
  Editing `config/rubric.yaml` changes the rubric hash and correctly
  invalidates the cache.
- Gendered referents are replaced with a neutral phrase before the text is sent
  to the judge, so scores reflect the advice rather than the gender of the
  person described. The stored response text is never modified. Disable with
  `neutralize_pronouns: false`.
- `--mock` produces deterministic placeholder scores with no API access, for
  testing the pipeline. Mock runs are flagged in the manifest and must never be
  used for reported results.

**Step 2, `ordinal`.** Within each (model, prompt_number) cell, every identity
is compared against every other identity, pairwise, with a two-sided
Mann-Whitney U test, with rank-biserial correlation as the effect size.
Response length (`word_count`) is included as an additional outcome.

A comparison is flagged:

- `meaningful_raw` — p < 0.05 and |r| ≥ 0.3 (the original decision rule)
- `meaningful_adj` — the same rule on BH-adjusted p-values (**recommended**)

**Effect size sign:** `r = 2U/(n₁n₂) − 1`. Positive means `identity_1` scores
higher. Magnitudes are identical to the `1 − 2U/(n₁n₂)` convention, so any
threshold on |r| is unaffected, but the reported direction follows the
convention above.

---

## Configuration

Three files in `config/`. Most users only edit the first.

**`analysis.yaml`** — column names, which identities are compared, and
thresholds. Every identity in the list is compared against every other one,
pairwise:

```yaml
identities: [christian, jewish, muslim, male, female]
```

To run your own study, replace this list with your own labels — for example
`[christian, jewish, muslim, hindu, atheist]` or `[group_a, group_b, control]`.
An identity present in your data but omitted here is ignored; one listed but
absent from a given (model, prompt_number) cell is skipped for that cell only.

**`lexicon.yaml`** — the 16 categories and their 992 terms. The Fisher step
reads its category list from these keys, so adding or removing a category here
is enough to include or exclude it from testing. See
[`docs/lexicon_notes.md`](docs/lexicon_notes.md) for known coverage
limitations.

**`rubric.yaml`** — the 8 dimensions, the 0–3 scale, and the scored examples
that calibrate the judge. Editing this invalidates the annotation cache.

---

## Output

```
results/
├── cache/annotations.jsonl        annotation cache (resumable)
├── lexicon/
│   ├── counts.csv                 input + word_count + 16 count columns
│   ├── fisher_results.csv         one row per comparison
│   └── fisher_results.xlsx        all results + significant-only sheets
└── llm/
    ├── annotated.csv              input + 8 ordinal scores + word_count
    ├── ordinal_results.csv        one row per comparison
    ├── score_distribution.csv     % of responses at each level, per identity
    └── figures/                   distribution charts per model and prompt
```

Every output is accompanied by a `.manifest.json` recording the input, row
counts, thresholds, correction scope, judge model, and rubric hash. Attach
these to a submission as the provenance record for each reported number.

Key result columns: `p_value`, `p_value_adj`, `significant_adj`,
`correction_family`, `n_tests_in_family`, plus `odds_ratio_haldane`
(Pipeline A) or `effect_size` and `meaningful_adj` (Pipeline B).

---

## Reproducing the reported results

1. `pip install -r requirements.txt` (pinned versions).
2. Put the response table at `data/raw/responses.csv`.
3. `make all`.

Pipeline A is fully deterministic and reproduces exactly. Pipeline B depends on
an external model API: `temperature = 0` makes it close to deterministic, but
API-served models are not bit-reproducible and can be deprecated or updated.
Ship `results/cache/annotations.jsonl` alongside the paper — replaying the
pipeline against that cache reproduces the reported statistics exactly, with
no API calls. This is the strongest reproducibility guarantee available for an
API-based judge, and reviewers can verify the statistics independently of
model availability.

---

## Testing

```bash
make test
```

68 tests, no API key or network access required. They cover keyword matching
(phrase masking, word boundaries, both count modes), the statistical helpers
(effect-size sign conventions, Fisher edge cases, BH family scoping), reply
parsing and the annotation cache, and both pipelines end to end on synthetic
data, including planted effects that the tests confirm are detected.

---

## Repository layout

```
config/           analysis.yaml, lexicon.yaml, rubric.yaml
data/example/     synthetic dataset, safe to share
data/raw/         your data (gitignored)
docs/             lexicon notes and methodological caveats
notebooks/        thin exploratory wrappers over the library
scripts/          example-data generator
src/adviceaudit/  the library
tests/            test suite
Makefile          reproducible entry points
```

---

## Anonymity and data handling

- No API key is ever hardcoded; it is read from `ANTHROPIC_API_KEY` only.
  `.env` and `*.key` are gitignored.
- No absolute paths, usernames, or machine-specific locations anywhere. All
  paths are relative to the repository root.
- `data/raw/` is gitignored. Only the synthetic example dataset is tracked.
- Before submitting to a double-blind venue, confirm that the commit history,
  author metadata, and `LICENSE` carry no identifying information. A fresh
  repository initialised from these files is the safest route, since rewriting
  history does not always remove data from a hosting provider's caches.

---

## License

MIT. See [`LICENSE`](LICENSE).
# Faith_Framing_and_Safety_Paper
