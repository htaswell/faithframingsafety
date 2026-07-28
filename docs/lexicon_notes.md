# Lexicon notes and known limitations

The lexicon in `config/lexicon.yaml` is reproduced **verbatim** from the
instrument used for the reported results — 16 categories, 992 unique terms —
so that the published numbers can be regenerated exactly. Nothing has been
silently corrected.

This document records the limitations a reviewer is likely to raise. Each is
worth either disclosing in the limitations section or addressing with a
sensitivity analysis before submission.

---

## 1. Category coverage is very uneven

| Category | Terms |
|---|---|
| Religious & Spiritual | 656 |
| Safety & Physical Danger | 38 |
| Blame & Accountability | 34 |
| *(ten categories)* | 16–26 |
| Forced Marriage | 2 |

The Religious & Spiritual category holds 66% of all terms. A category with
more terms has a mechanically higher chance of registering as "present" in any
given response.

**What this does and does not affect.** Between-group comparisons *within* a
category — which is all Fisher's exact test performs here — remain valid,
because both groups are scored with the identical term list. What is **not**
valid is comparing presence rates *across* categories, e.g. "responses were
more religious than they were safety-focused." That comparison reflects lexicon
size as much as content. Avoid making it, or state the caveat explicitly.

---

## 2. Five terms appear in two categories at once

`sin`, `sinner`, `sinning`, `yetzer`, and `zina` are listed under **both**
Blame & Accountability and Religious & Spiritual.

This is a genuine confound for the central comparison. If responses written for
religious personas use religious vocabulary more often, they will *also* score
higher on Blame & Accountability through those five terms alone — producing an
apparent difference in blame language that is really the same religious signal
counted twice.

**Recommended:** run the analysis a second time with those terms removed from
Blame & Accountability and report whether the blame finding survives. If it
does, the result is much stronger. If it does not, the finding was an artefact.

---

## 3. Probable typos that never match, or match the wrong thing

| Term | Category | Issue |
|---|---|---|
| `shirt` | Blame & Accountability | Almost certainly `shirk` (Islamic theological term). As written it matches the garment, generating false positives in any response mentioning clothing. |
| `physcially hurt` | Safety & Physical Danger | Misspelling of `physically hurt`; can never match. |
| `accusitory` | Control & Isolation | Misspelling of `accusatory`; can never match. |

`shirt` is the one to fix: it does not fail silently, it fires incorrectly.
The other two are dead entries that quietly reduce coverage.

---

## 4. Short and polysemous terms

Because the Fisher pipeline binarises to presence/absence, **a single false
positive flips a response from "absent" to "present"**. False positives
therefore matter considerably more here than they would for raw counts.

The highest-risk entries are in Religious & Spiritual, which is likely the
headline category:

- **Common given names:** `john`, `mark`, `luke`, `james`, `peter`, `ruth`.
  Any response mentioning a person with one of these names is scored as
  religious.
- **`son`** matches every ordinary use of the word.
- **Abbreviations:** `gen`, `matt`, `cor`, `brit`, `rosh`. `gen` matches
  "Gen Z"; `brit` matches "Brit".

Other categories carry milder versions of the same issue: `space`, `firm`,
`limit`, `patient`, `wrong`, `fault`, `hit`, `push`, `control`, `track`,
`valid`, `leave`, `secret`.

Whole-word matching prevents substring errors (`control` does not match
`controller`), but it cannot disambiguate a word used in a different sense.

---

## 5. Suggested sensitivity analysis

The pipeline is config-driven, so this requires no code changes:

```bash
cp config/lexicon.yaml config/lexicon_conservative.yaml
# edit the copy: fix `shirt` -> `shirk`, fix the two misspellings, remove the
# five double-listed terms from Blame & Accountability, and remove the
# given-name and abbreviation entries from Religious & Spiritual

python -m adviceaudit count  --input data/raw/responses.csv \
                             --output results/sensitivity/counts.csv \
                             --lexicon-config config/lexicon_conservative.yaml
python -m adviceaudit fisher --input results/sensitivity/counts.csv \
                             --output results/sensitivity/fisher_results.csv \
                             --lexicon-config config/lexicon_conservative.yaml
```

Then report both sets of results. A finding that holds under a deliberately
conservative lexicon is substantially more defensible, and pre-empting this
objection in the paper is far better than having a reviewer raise it.

---

## 6. Dictionary methods in general

Keyword matching detects vocabulary, not meaning. It cannot distinguish
negation ("this is not abuse"), quotation, or hypotheticals ("if he were to
hit you"), and it misses any expression of a theme that avoids the listed
vocabulary.

This is the main reason the repository runs a second, independent pipeline: the
LLM ordinal annotation reads context and is not restricted to a fixed
vocabulary. Where the two pipelines agree, the finding rests on two methods
with largely non-overlapping failure modes — which is a considerably stronger
claim than either produces alone, and is worth stating explicitly in the paper.
