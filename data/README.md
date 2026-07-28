# Data

## What belongs here

```
data/
├── example/    synthetic dataset, tracked in git, safe to share
└── raw/        your response data (gitignored, never committed)
```

`data/raw/` is gitignored. Put your real response table there; it will not be
committed. Only the synthetic example is tracked.

## Expected format

One row per model response. CSV, TSV, or Excel.

| Column | Type | Meaning |
|---|---|---|
| `ID` | string | Unique identifier for the response |
| `prompt_number` | int or string | Which prompt/scenario was used |
| `model` | string | Which system produced the response |
| `identity` | string | The demographic condition being compared |
| `text` | string | The response text to analyse |

Extra columns are carried through untouched.

Values in `identity` are the labels compared against each other. List the ones
you want compared in the `identities` block of `config/analysis.yaml`. With the
shipped configuration those are `christian`, `jewish`, `muslim`, `male`,
`female` — every one is compared against every other, pairwise.

If your columns are named differently, edit the `columns` block in
`config/analysis.yaml` rather than renaming your data.

## Example

```csv
ID,model,prompt_number,identity,text
resp_0001,model_a,1,christian,"That sounds difficult. Pray about it..."
resp_0002,model_a,1,muslim,"Consider speaking with your imam..."
```

Regenerate the synthetic example at any time:

```bash
python scripts/make_example_data.py
```

It is built from fixed templates with a fixed seed. It is not real model output
and contains no real user data; its only purpose is to let someone run both
pipelines before supplying their own data.

## Choosing your own identities

To run your own study, replace the `identities` list in
`config/analysis.yaml` with your labels — nothing else needs to change:

```yaml
identities: [christian, jewish, muslim, hindu, atheist]
```

or

```yaml
identities: [group_a, group_b, control]
```

Every listed identity is compared against every other one, within each
(model, prompt_number) cell. An identity present in your data but omitted from
this list is ignored; one listed but absent from a given cell is skipped for
that cell only.

## Sample size

Both pipelines skip any identity with too few responses in a
(model, prompt_number) cell: `fisher.min_n` (default 3) and `ordinal.min_n`
(default 2). Skipped comparisons appear in the output with a note rather than
being dropped silently.

Small cells also limit what multiple-comparison correction can detect. With 4
responses per identity, a perfectly separated category gives a Fisher p of
0.029, which does not survive Benjamini-Hochberg correction across a family of
tests. Aim for at least 6 responses per identity per cell, and more if you
expect modest effects.
