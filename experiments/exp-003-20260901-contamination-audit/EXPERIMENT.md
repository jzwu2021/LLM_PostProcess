# Experiment exp-003-20260901: contamination and overlap audit

Audits `research/ai-infra-expert/corpus_v04/train.jsonl` (3000 records) against
`research/ai-infra-expert/benchmark.jsonl` (500 records) before any training run
uses the corpus or any evaluation uses the benchmark.

## Why this was run before training

The v0.2 evaluation work in this repository produced a repair dataset authored by
inspecting benchmark failures. That dataset was never promoted, because training
on it and evaluating on the benchmark it was derived from measures the return of
transferred information rather than capability. This audit exists so the same
mistake is not repeated silently with the v0.4 corpus.

## What was measured, and what cannot be

Contamination is four separate questions that are routinely conflated:

| # | Question | Measurable from artifacts |
| --- | --- | --- |
| 1 | Does a training item reproduce a benchmark item? | yes |
| 2 | Do training answers carry the benchmark's expected content? | yes |
| 3 | Does training cover the same subjects? | partly, and only as judgement |
| 4 | Did benchmark items inform how the training data was authored? | **no** |

Question 4 is the one that matters most and no string comparison can answer it.
It is a provenance fact, recorded below.

## Results

Command: `python audit_contamination.py`, full output in `logs/audit.txt`.

### 0. The benchmark's own composition

| metric | value |
| --- | --- |
| records | 500 |
| distinct questions | 500 |
| **distinct reference answers** | **140** |
| distinct topics | 41 |
| reference answers reused across records | 90 answers covering 450 records |

The benchmark has the same defect as the v0.3 corpus, at smaller scale: 500 items
carry 140 distinct expected answers, and 90% of records share an answer with at
least one other record. Scores on it are not 500 independent observations.

### 1. Item overlap — CLEAN

| metric | value |
| --- | --- |
| exact question matches | 0 |
| questions with 5-gram Jaccard >= 0.30 | 0 of 3000 |
| max question 5-gram Jaccard | 0.000 |
| max question content-term overlap | 0.167 |

Zero 5-gram overlap means no training question shares a five-word sequence with
any benchmark question. The term-level maximum of 0.167 is reported because a
zero n-gram score alone would be uninformative; at that level the two are talking
about the same subject in entirely different words, which is what a fair
benchmark requires.

### 2. Answer leakage — CLEAN

| metric | value |
| --- | --- |
| answers with 5-gram Jaccard >= 0.20 against a reference answer | 0 of 3000 |
| max answer 5-gram Jaccard | 0.000 |
| key-point verifier items | 95 |
| verbatim key phrases from those items found in training answers | 0 |

The 95 `contains_key_points` items are the ones where verbatim phrase reuse would
most directly inflate a score. No key phrase longer than 40 characters appears in
the training text.

### 3. Topic coverage — 38 of 41 topics covered (declared, not measured)

The lexical proxy is reported in the log and is weak: it requires every content
word of a topic phrase to co-occur in a single record, so it marks "design a safe
model rollout" absent even though three rollout mechanisms exist. The audit
therefore carries a hand-authored mapping from each benchmark topic to the v0.4
mechanisms addressing it, labelled in the output as judgement rather than
measurement, so a reviewer can disagree with a specific line.

Three benchmark topics have **no** corresponding training mechanism:

- CUDA streams
- speculative decoding
- detect duplicate request IDs

These are the only part of the benchmark that tests subject matter the corpus does
not teach.

**This is the finding that constrains what the benchmark can be used for.** With
38 of 41 topics covered, the benchmark measures whether training on these
mechanisms transfers to differently worded questions about the same mechanisms.
It does not measure generalisation to unseen subject areas, and it should not be
described as if it does.

### 4. Cross-check against the v0.3 corpus — CLEAN

| metric | value |
| --- | --- |
| v0.3 rows | 5399 |
| v0.4 questions also present in v0.3 | 0 |
| max v0.4-to-v0.3 question Jaccard (sampled) | 0.000 |

v0.4 shares no question text with v0.3, so a model trained on v0.4 is not being
evaluated on material it saw through the older corpus.

## Provenance statement for question 4

This is the part no measurement covers, so it is stated as fact rather than
derived.

- The 100 v0.4 mechanisms were authored from domain knowledge, not from reading
  `benchmark.jsonl`.
- `benchmark.jsonl` was opened twice during this work: once with `head -c 400`
  and once with a `json.loads` of the first record, both to determine the field
  schema for the contamination check. That exposed one question
  (`aiinfra-0001`, GPU memory hierarchy) and its reference answer.
- The benchmark's **topic list** was read during this audit, after all 100
  mechanisms were authored and committed. The mechanism set was not revised
  afterwards, and the declared mapping in section 3 was produced by matching an
  already-frozen mechanism list against the topic list.
- No benchmark item's failure was inspected to decide what any mechanism should
  teach. That is the specific process that contaminated the v0.2 repair dataset,
  and it did not occur here.

The residual exposure is one benchmark item seen for schema purposes. The
conservative response is to exclude `aiinfra-0001` from any evaluation used to
make a promotion decision about a model trained on this corpus.

## Conclusions

1. The corpus is clean on item overlap and answer leakage against the benchmark
   and against v0.3. It can be used for training.
2. The benchmark is a **same-subject, different-wording** test for this corpus,
   not a generalisation test. Any result must be described that way.
3. The benchmark's 500 records carry 140 distinct reference answers, so its
   effective resolution is far below its record count. Statistical claims about
   score differences must use the distinct-answer count, not 500.
4. Exclude `aiinfra-0001` from promotion decisions.
5. Before any capability claim is made, a benchmark whose topic list was never
   read during corpus authoring is still required. This audit establishes that
   the corpus did not copy the benchmark; it does not establish that the
   benchmark is an independent test of the corpus.

## Status

Provisional. This audit checks artifacts against artifacts. It is not an
assessment of whether the corpus content is technically correct, which requires
domain-expert review that has not been performed.
