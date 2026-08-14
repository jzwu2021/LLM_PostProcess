# Teacher-A corpus calibration

## Status

`IN_PROGRESS`. This directory stores provisional calibration produced by the current conversational model. It is not an expert-approved gold set and must not overwrite the source corpus.

## Lane separation

- Teacher-A: current GPT-5.6-Luna conversational model.
- Teacher-B: a later independent model, to be stored in a separate directory.
- Teacher-A output is provisional and will be compared against Teacher-B output after the model switch.

## Progress

```text
train processed: 40 / 5399
validation processed: 0 / 601
total processed: 40 / 6000
progress: 0.67%
```

## Batch 0001

Input: first 10 records of `research/ai-infra-expert/corpus/train.jsonl`.

Result:

```text
records processed: 10
source ID alignment: PASS
keep: 0
rewrite: 10
reject: 0
```

## Batch 0002

Input: records 11-20 of `research/ai-infra-expert/corpus/train.jsonl`.

Result:

```text
records processed: 10
source ID alignment: PASS
keep: 0
rewrite: 10
reject: 0
```

The first 20 records reused one KV-cache assistant answer across multiple distinct instructions. The original targets did not satisfy contrast, failure-mode, measurement-plan, or explicit-boundary-condition requests. All twenty were therefore rewritten while preserving original source text and recording risks/evidence requirements.

## Batch 0003

Input: records 21-30 of `research/ai-infra-expert/corpus/train.jsonl`.

Result:

```text
records processed: 10
source ID alignment: PASS
keep: 0
rewrite: 10
reject: 0
initial schema check: FAILED on one missing corrected_answer field
repair: PASS
final schema check: PASS
```

This batch covered KV-cache measurement plans, performance-claim assumptions, and training-versus-inference distinctions. The first write had one missing `corrected_answer`; it was repaired before final manifest generation and verification.

## Batch 0004

Input: records 31-40 of `research/ai-infra-expert/corpus/train.jsonl` (source IDs `corpus-00034` through `corpus-00043`).

Result:

```text
records processed: 10
source ID alignment: PASS
keep: 0
rewrite: 10
reject: 0
initial schema check: PASS
repair: not required
final schema check: PASS
```

This batch covered training-versus-inference cache behavior, misleading intuitions about cache cost/capacity, and controlled experiments. Corrections explicitly separated measured quantities and execution regimes, including memory-pressure and non-equivalent-baseline risks.

## Output schema

Each output record includes:

- `source_id`
- `teacher_lane`
- `teacher_model`
- `calibration_status`
- `decision`
- original user/assistant content
- `corrected_answer`
- dimension scores
- `risks`
- `evidence_required`
- `confidence`

All scores are provisional teacher judgments. They are not human expert scores.

## Verification

The batch was checked with a fresh JSON parse and source-ID alignment check:

```text
TEACHER_A_BATCH_VERIFY_PASS rows=10 id_alignment=pass all_decision=rewrite
```

## Next batches

Continue train-only calibration in immutable batch files (`train-batch-0005.jsonl`, etc.). Do not use validation or benchmark records as training targets. After the user switches models, write the second model's outputs under a separate `teacher-b-corpus-calibration/` directory and compare by source ID, decision, answer content, and disagreement type.
