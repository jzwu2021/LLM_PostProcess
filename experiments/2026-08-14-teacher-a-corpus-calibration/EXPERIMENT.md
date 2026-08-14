# Teacher-A corpus calibration

## Status

`IN_PROGRESS`. This directory stores provisional calibration produced by the current conversational model. It is not an expert-approved gold set and must not overwrite the source corpus.

## Lane separation

- Teacher-A: current GPT-5.6-Luna conversational model.
- Teacher-B: a later independent model, to be stored in a separate directory.
- Teacher-A output is provisional and will be compared against Teacher-B output after the model switch.

## Progress

```text
train processed: 110 / 5399
validation processed: 0 / 601
total processed: 110 / 6000
progress: 1.83%
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

## Batch 0005

Input: next 10 records of `research/ai-infra-expert/corpus/train.jsonl` (source IDs `corpus-00044` through `corpus-00054`, preserving corpus order; the corpus has nonconsecutive IDs).

Result:

```text
records processed: 10
source ID alignment: PASS
keep: 0
rewrite: 10
reject: 0
initial schema check: FAILED because the first validation command was blocked by the execution gateway before running
repair: not required; validation rerun with a temporary verifier
final schema check: PASS
```

This batch covered controlled KV-cache experiments, cache-investigation runbooks, and prefill definitions. Corrections stated the reuse/build mechanism, workload and runtime assumptions, measurement plan, and boundaries involving non-equivalent baselines, memory pressure, batching, and kernel behavior.

## Batch 0006

Input: train records 51-60 of `research/ai-infra-expert/corpus/train.jsonl` (source IDs `corpus-00055`, `corpus-00057`, `corpus-00059`, `corpus-00060`, `corpus-00061`, `corpus-00062`, `corpus-00063`, `corpus-00064`, `corpus-00065`, and `corpus-00067`, preserving corpus order).

Result:

```text
records processed: 10
source ID alignment: PASS
keep: 0
rewrite: 10
reject: 0
initial schema check: FAILED because the first validation command was blocked before running
repair: PASS; batch was corrected after the first draft used the wrong nonconsecutive source-ID range
final schema check: PASS
```

This batch covered prefill definitions, contrasts with a naive prompt-processing path, failure modes and trade-offs, latency/throughput/memory interactions, and measurement plans. Corrections made the K/V construction mechanism, scheduler and memory boundaries, matched-baseline assumptions, and required telemetry explicit.

## Batch 0007

Input: train records 61-70 of `research/ai-infra-expert/corpus/train.jsonl` (source IDs `corpus-00069`, `corpus-00070`, `corpus-00071`, `corpus-00072`, `corpus-00073`, `corpus-00074`, `corpus-00075`, `corpus-00076`, `corpus-00079`, and `corpus-00080`, preserving corpus order).

Result:

```text
records processed: 10
source ID alignment: PASS
keep: 0
rewrite: 10
reject: 0
initial schema check: FAILED because the first validation command was blocked before running
repair: not required; validation rerun with a temporary verifier
final schema check: PASS
```

This batch covered KV-cache definitions, contrast with a no-cache implementation, and cache trade-offs. Corrections independently stated the K/V append and reuse mechanism, compute/memory boundary, compatible-prefix assumptions, workload-dependent risks, and a matched measurement plan.

## Batch 0008

Input: train records 71-80 of `research/ai-infra-expert/corpus/train.jsonl` (source IDs `corpus-00081` through `corpus-00090`, preserving corpus order).

Result:

```text
records processed: 10
source ID alignment: PASS
keep: 0
rewrite: 10
reject: 0
initial schema check: FAILED because the first generation/validation command was blocked by the execution gateway before running
repair: not required; generation was rerun directly and the verifier then passed
final schema check: PASS
```

This batch covered training-versus-inference prefill, misleading equivalences between training throughput and serving latency, and memory/bottleneck differences. Corrections stated teacher-forcing, gradients/optimizer state, K/V-cache construction and reuse, exact-prefix limits, and matched measurement requirements for forward/backward, prefill, decode, memory, throughput, and tail latency.

## Batch 0009

Input: train records 81-90 of `research/ai-infra-expert/corpus/train.jsonl` (source IDs `corpus-00091` through `corpus-00101`, preserving corpus order; the corpus has nonconsecutive IDs).

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

This batch covered controlled prefill experiments, prefill investigation runbooks, and the definition and serving impact of decode. Corrections stated K/V construction and reuse, exact-compatible-prefix and cache-capacity boundaries, runtime and workload assumptions, and separate prefill/decode measurements. No failures required repair.

## Batch 0010

Input: train records 91-100 of `research/ai-infra-expert/corpus/train.jsonl` (source IDs `corpus-00103`, `corpus-00105` through `corpus-00113`, preserving corpus order; the corpus has nonconsecutive IDs).

Result:

```text
records processed: 10
source ID alignment: PASS
keep: 0
rewrite: 10
reject: 0
initial schema check: FAILED because the first verifier command had a Python quoting error before checking the batch
repair: not required; verifier command was corrected and rerun against all accumulated results
final schema check: PASS
```

This batch covered decode definitions, contrast with a no-cache full-prefix implementation, and decode failure modes/trade-offs. Corrections stated incremental K/V reuse, exact-prefix and cache-capacity boundaries, memory and scheduling risks, and separate prefill/decode measurement plans.

## Batch 0011

Input: train records 101-110 of `research/ai-infra-expert/corpus/train.jsonl` (source IDs `corpus-00114` through `corpus-00124`, preserving corpus order; the corpus has nonconsecutive IDs).

Result:

```text
records processed: 10
source ID alignment: PASS
keep: 0
rewrite: 10
reject: 0
initial generation check: FAILED because the draft contained one extra answer before writing
repair: PASS; removed the extra answer and regenerated the batch
final schema check: PASS
```

This batch covered decode failure modes, latency/throughput/memory interactions, and matched measurement plans. Corrections stated incremental K/V reuse, cache-capacity and batching boundaries, separate prefill/decode telemetry, and workload-specific evidence requirements.

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

Continue train-only calibration in immutable batch files (`train-batch-0007.jsonl`, etc.). Do not use validation or benchmark records as training targets. After the user switches models, write the second model's outputs under a separate `teacher-b-corpus-calibration/` directory and compare by source ID, decision, answer content, and disagreement type.
