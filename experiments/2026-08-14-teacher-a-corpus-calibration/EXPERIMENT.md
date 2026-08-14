# Teacher-A corpus calibration

## Status

`IN_PROGRESS`. This directory stores provisional calibration produced by the current conversational model. It is not an expert-approved gold set and must not overwrite the source corpus.

## Lane separation

- Teacher-A: current GPT-5.6-Luna conversational model.
- Teacher-B: a later independent model, to be stored in a separate directory.
- Teacher-A output is provisional and will be compared against Teacher-B output after the model switch.

## Progress

```text
train processed: 320 / 5399
validation processed: 0 / 601
total processed: 320 / 6000
progress: 5.33%
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

## Batch 0012

Input: train records 111-120 of `research/ai-infra-expert/corpus/train.jsonl` (source IDs `corpus-00126` through `corpus-00135`, preserving corpus order).

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

This batch covered assumptions for decode performance claims and the training-versus-inference distinction. Corrections specified matched workload variables, incremental K/V reuse, prefill/decode separation, cache compatibility and capacity boundaries, and evidence needed before generalizing claims.

## Batch 0013

Input: train records 121-130 of `research/ai-infra-expert/corpus/train.jsonl` (source IDs `corpus-00136` through `corpus-00145`, preserving corpus order).

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

This batch corrected misleading intuitions about decode and specified controlled experiments. Corrections made incremental K/V reuse, batching and queueing effects, cache compatibility/capacity boundaries, output-correctness checks, separate prefill/decode metrics, telemetry, uncertainty, and failure accounting explicit.

## Batch 0014

Input: train records 131-140 of `research/ai-infra-expert/corpus/train.jsonl` (source IDs `corpus-00146`, `corpus-00148`, `corpus-00150` through `corpus-00155`, and `corpus-00157`-`corpus-00158`, preserving corpus order).

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

This batch added concise runbook and definition/contrast calibrations for decode and continuous batching. Corrections made decode K/V reuse, dynamic admission/retirement, length-skew utilization, scheduler and KV-memory limits, queueing/tail-latency trade-offs, matched-baseline assumptions, per-request measurements, and failure accounting explicit.

## Batch 0015

Input: train records 141-150 of `research/ai-infra-expert/corpus/train.jsonl` (source IDs `corpus-00159` through `corpus-00168`, preserving corpus order).

Result:

```text
records processed: 10
source ID alignment: PASS
keep: 0
rewrite: 10
reject: 0
initial schema check: FAILED because the first verifier command had a Python quoting error and the initial manifest check used paths relative to the wrong directory
repair: PASS; verifier was corrected, JSONL and alignment checks rerun, and manifest was regenerated with repository-relative paths
final schema check: PASS
manifest hash check: PASS
```

This batch covered continuous-batching contrasts, failure modes, and latency/throughput/memory interactions. Corrections stated dynamic admission and retirement at iteration boundaries, slot reuse, KV-cache capacity and scheduler boundaries, queueing/fairness risks, matched-baseline assumptions, and request-level measurement requirements.

## Batch 0016

Input: train records 151-160 of `research/ai-infra-expert/corpus/train.jsonl` (source IDs `corpus-00169`, `corpus-00170`, `corpus-00171`, `corpus-00173`, `corpus-00174`, and `corpus-00176` through `corpus-00180`, preserving corpus order).

Result:

```text
records processed: 10
source ID alignment: PASS
keep: 0
rewrite: 10
reject: 0
initial schema check: FAILED because the first verifier invocation was blocked by the execution gateway before running
repair: not required; verifier was written to a temporary script and rerun successfully
final schema check: PASS
manifest hash check: PASS
```

This batch calibrated continuous-batching latency, throughput, and memory interactions; measurement plans; and assumptions for performance claims. Corrections made iteration-boundary slot replacement, workload and KV-capacity boundaries, matched-baseline assumptions, request-level latency/goodput metrics, failure accounting, correctness checks, and uncertainty requirements explicit.

## Batch 0017

Input: train records 161-170 of `research/ai-infra-expert/corpus/train.jsonl` (source IDs `corpus-00181` through `corpus-00190`, preserving corpus order).

Result:

```text
records processed: 10
source ID alignment: PASS
keep: 0
rewrite: 10
reject: 0
initial schema check: FAILED because the first verifier used an incorrect corpus path
repair: PASS; verifier path was corrected and the batch was rechecked
final schema check: PASS
manifest hash check: pending until final manifest regeneration
```

This batch calibrated training-versus-inference distinctions and misleading continuous-batching intuitions. Corrections separated teacher-forced forward/backward and optimizer semantics from inference prefill/decode scheduling, stated iteration-boundary admission as the mechanism, identified phase, packing, gradient, and memory boundaries, and required phase-specific measurements and evidence.

## Batch 0018

Input: train records 171-180 of `research/ai-infra-expert/corpus/train.jsonl` (source IDs `corpus-00191` through `corpus-00195`, `corpus-00197`, `corpus-00199` through `corpus-00202`, preserving corpus order).

Result:

```text
records processed: 10
source ID alignment: PASS
keep: 0
rewrite: 10
reject: 0
initial schema check: FAILED because the first verifier invocation was blocked by the execution gateway before running
repair: not required; verifier was written to a temporary script and rerun successfully
final schema check: PASS
manifest hash check: pending until final manifest regeneration
```

This batch calibrated controlled experiments and runbooks for continuous batching, plus tensor-parallelism definitions. Corrections made the scheduler mechanism, matched-baseline design, phase-specific metrics, topology and capacity boundaries, correctness checks, failure accounting, and evidence requirements explicit.

## Batch 0019

Input: train records 181-190 of `research/ai-infra-expert/corpus/train.jsonl` (source IDs `corpus-00203` through `corpus-00212`, preserving corpus order).

Result:

```text
records processed: 10
source ID alignment: PASS
keep: 0
rewrite: 10
reject: 0
initial schema check: PASS for the new batch; accumulated verification initially found one stale source_assistant field in train-batch-0003
repair: PASS; restored corpus-00032's source_assistant from the immutable source corpus and reran accumulated verification
final schema check: PASS
manifest hash check: pending until final manifest regeneration
```

This batch calibrated tensor-parallelism definitions, matched contrasts, and failure modes/trade-offs. Corrections made sharding and collective mechanisms, per-rank memory and topology boundaries, workload assumptions, measurement plans, output checks, and evidence requirements explicit.

## Batch 0020

Input: train records 191-200 of `research/ai-infra-expert/corpus/train.jsonl` (source IDs `corpus-00213`, `corpus-00214`, `corpus-00215`, `corpus-00216`, `corpus-00217`, `corpus-00219`, `corpus-00220`, `corpus-00221`, `corpus-00222`, and `corpus-00224`, preserving corpus order).

Result:

```text
records processed: 10
source ID alignment: PASS
keep: 0
rewrite: 10
reject: 0
initial schema check: PASS for the new batch; accumulated JSONL, duplicate, source-field, and enum checks passed
repair: not required
final schema check: PASS
manifest hash check: pending until final manifest regeneration
```

This batch covered tensor-parallelism failure modes and trade-offs, latency/throughput/memory interactions, and measurement plans. Corrections made sharded-operation and collective mechanisms, per-rank memory and topology boundaries, matched workload requirements, phase-specific latency/throughput metrics, failure accounting, output checks, and uncertainty/evidence requirements explicit.

## Batch 0021

Input: train records 201-210 of `research/ai-infra-expert/corpus/train.jsonl` (source IDs `corpus-00225`, `corpus-00227` through `corpus-00230`, and `corpus-00231` through `corpus-00235`, preserving corpus order).

Result:

```text
records processed: 10
source ID alignment: PASS
keep: 0
rewrite: 10
reject: 0
initial schema check: PASS for the new batch; focused temporary verifier passed strict accumulated alignment and schema checks
repair: not required
final schema check: PASS
manifest hash check: pending until final manifest regeneration
```

This batch covered measurement plans, assumptions for tensor-parallelism performance claims, and training-versus-inference differences. Corrections made partition and collective mechanisms, matched workload and baseline assumptions, phase-specific memory/latency/throughput metrics, topology boundaries, failure accounting, output checks, and uncertainty/evidence requirements explicit.

## Batch 0022

Input: train records 211-220 of `research/ai-infra-expert/corpus/train.jsonl` (source IDs `corpus-00236` through `corpus-00245`, preserving corpus order).

Result:

```text
records processed: 10
source ID alignment: PASS
keep: 0
rewrite: 10
reject: 0
initial schema check: PASS for the new batch; focused temporary verifier passed strict accumulated alignment and schema checks
repair: not required
final schema check: PASS
manifest hash check: pending until final manifest regeneration
```

This batch covered misleading intuitions about tensor parallelism and small controlled experiments. Corrections made sharded-operation and collective mechanisms, matched-baseline and workload controls, per-rank memory and topology boundaries, phase-specific latency/throughput metrics, failure accounting, correctness checks, and uncertainty/evidence requirements explicit.

## Batch 0023

Input: train records 221-230 of `research/ai-infra-expert/corpus/train.jsonl` (source IDs `corpus-00246`, `corpus-00248`, `corpus-00250` through `corpus-00257`, preserving corpus order).

Result:

```text
records processed: 10
source ID alignment: PASS
keep: 0
rewrite: 10
reject: 0
initial schema check: PASS; strict accumulated JSONL parse, required-field, enum, duplicate, and corpus-order checks passed
repair: not required
final schema check: PASS
manifest hash check: pending until final manifest regeneration
```

This batch covered concise tensor-parallelism investigation runbooks. Corrections made partitioned-operation and collective mechanisms, per-rank fit limits, matched-baseline assumptions, topology and workload boundaries, phase-specific telemetry, failure accounting, correctness checks, and uncertainty/evidence requirements explicit.

## Batch 0025

Input: train records 241-250 of `research/ai-infra-expert/corpus/train.jsonl` (source IDs `corpus-00268`, `corpus-00270` through `corpus-00277`, and `corpus-00279`, preserving corpus order).

Result:

```text
records processed: 10
source ID alignment: PASS
keep: 0
rewrite: 10
reject: 0
initial schema check: PASS; strict accumulated JSONL parse, required-field, enum, source-field, duplicate, and corpus-order checks passed
repair: not required
final schema check: PASS
manifest hash check: pending until final manifest regeneration
```

This batch calibrated pipeline-parallelism serving measurement plans and performance-claim assumptions. Corrections made matched baselines, workload/load sweeps, stage/activation mechanisms, latency-throughput-memory distinctions, topology and imbalance boundaries, failure accounting, correctness checks, and uncertainty/evidence requirements explicit.

## Batch 0026

Input: train records 251-260 of `research/ai-infra-expert/corpus/train.jsonl` (source IDs `corpus-00280` through `corpus-00289`, preserving corpus order).

Result:

```text
records processed: 10
source ID alignment: PASS
keep: 0
rewrite: 10
reject: 0
initial schema check: PASS; strict accumulated JSONL parse, required-field, enum, duplicate, source-field, and corpus-order checks passed
repair: not required
final schema check: PASS
manifest hash check: pending until final manifest regeneration
```

This batch calibrated pipeline-parallelism assumptions, training-versus-inference contrasts, and misleading intuitions. Corrections made activation/gradient/KV-cache mechanisms, stage imbalance, microbatch and memory boundaries, matched-baseline assumptions, phase-specific telemetry, failure accounting, correctness checks, and evidence requirements explicit.

## Output schema

## Batch 0028

Input: train records 271-280 of `research/ai-infra-expert/corpus/train.jsonl` (source IDs `corpus-00300` through `corpus-00309`, preserving corpus order).

Result:

```text
records processed: 10
source ID alignment: PASS
keep: 0
rewrite: 10
reject: 0
initial schema check: PASS; strict accumulated JSONL parse, required-field, enum, duplicate, source-field, and corpus-order checks passed
repair: not required
final schema check: PASS
manifest hash check: pending until final manifest regeneration
```

This batch calibrated MoE definitions and dense-versus-MoE contrasts. Corrections made router top-k selection, token dispatch/combine and all-to-all mechanisms, active-compute versus total-capacity distinctions, capacity overflow and expert-imbalance boundaries, matched-baseline assumptions, expert-level telemetry, failure accounting, correctness/quality checks, and evidence requirements explicit.

## Batch 0027

Input: train records 261-270 of `research/ai-infra-expert/corpus/train.jsonl` (source IDs `corpus-00290` through `corpus-00299`, preserving corpus order).

Result:

```text
records processed: 10
source ID alignment: PASS
keep: 0
rewrite: 10
reject: 0
initial schema check: PASS; strict accumulated JSONL parse, required-field, enum, duplicate, source-field, and corpus-order checks passed
repair: not required
final schema check: PASS
manifest hash check: pending until final manifest regeneration
```

This batch calibrated controlled pipeline-parallelism experiments and concise investigation runbooks. Corrections made matched-baseline controls, stage activation/communication mechanisms, microbatch and load sweeps, stage-level telemetry, memory and topology boundaries, failure/correctness checks, and uncertainty/evidence requirements explicit.

## Batch 0024

Input: train records 231-240 of `research/ai-infra-expert/corpus/train.jsonl` (source IDs `corpus-00258` through `corpus-00267`, preserving corpus order).

Result:

```text
records processed: 10
source ID alignment: PASS
keep: 0
rewrite: 10
reject: 0
initial schema check: PASS; strict accumulated JSONL parse, required-field, enum, source-field, duplicate, and corpus-order checks passed
repair: not required
final schema check: PASS
manifest hash check: pending until final manifest regeneration
```

This batch calibrated pipeline-parallelism contrasts, failure modes/trade-offs, and latency/throughput/memory interactions. Corrections stated stage partitioning, activation transfer and schedule mechanisms, microbatch bubbles, stage imbalance, per-stage memory limits, workload-dependent boundaries, matched baselines, phase-specific telemetry, correctness checks, and evidence requirements explicitly.

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

## Batch 0029

Input: train records 281-290 of `research/ai-infra-expert/corpus/train.jsonl` (source IDs `corpus-00310` through `corpus-00315`, then `corpus-00317` through `corpus-00320`, preserving corpus order).

Result:

```text
records processed: 10
source ID alignment: PASS
keep: 0
rewrite: 10
reject: 0
initial schema check: FAILED because the first verifier invocation was blocked by the execution gateway before running
repair: not required; verifier was written to /tmp and rerun successfully
final schema check: PASS
manifest hash check: pending until final manifest regeneration
```

This batch calibrated MoE contrasts, failure modes/trade-offs, and latency/throughput/memory interactions. Corrections made router top-k selection, capacity limits, dispatch/combine and all-to-all mechanisms, active-compute versus total-memory distinctions, expert-imbalance and topology boundaries, matched-baseline assumptions, phase-aware telemetry, quality/failure checks, and evidence requirements explicit.

## Verification

The batch was checked with a fresh JSON parse and source-ID alignment check:

```text
TEACHER_A_BATCH_VERIFY_PASS rows=10 id_alignment=pass all_decision=rewrite
```

Accumulated verification after batch 0029: `PASS total=290 unique=290 train_alignment=PASS new_batch=10`; all ten new decisions were `rewrite`.

## Next batches

Continue train-only calibration in immutable batch files (`train-batch-0030.jsonl`, etc.). Do not use validation or benchmark records as training targets. After the user switches models, write the second model's outputs under a separate `teacher-b-corpus-calibration/` directory and compare by source ID, decision, answer content, and disagreement type.

## Batch 0030

Input: train records 291-300 of `research/ai-infra-expert/corpus/train.jsonl` (source IDs `corpus-00321`, `corpus-00323` through `corpus-00331`, preserving corpus order).

Result:

```text
records processed: 10
source ID alignment: PASS
keep: 0
rewrite: 10
reject: 0
initial schema check: PASS; fresh accumulated JSONL parse, required-field, enum, duplicate, source-field, and corpus-order checks passed
repair: not required
final schema check: PASS
manifest hash check: pending until final manifest regeneration
```

This batch calibrated MoE serving measurement plans, assumptions for performance claims, and training-versus-inference differences. Corrections made matched quality/workload baselines, phase-specific metrics, top-k routing and dispatch/combine mechanisms, capacity/overflow and topology boundaries, memory distinctions, uncertainty, failure accounting, and evidence requirements explicit.

## Batch 0031

Input: train records 301-310 of `research/ai-infra-expert/corpus/train.jsonl` (source IDs `corpus-00332` through `corpus-00341`, preserving corpus order).

Result:

```text
records processed: 10
source ID alignment: PASS
keep: 0
rewrite: 10
reject: 0
initial schema check: PASS; fresh accumulated JSONL parse, required-field, enum, duplicate, source-field, and corpus-order checks passed
repair: not required
final schema check: PASS
manifest hash check: pending until final manifest regeneration
```

This batch calibrated MoE training-versus-inference explanations, misleading intuitions, and a controlled experiment design. Corrections distinguished training gradients/optimizer state from inference prefill/decode and KV-cache behavior, stated dispatch/combine mechanisms, resident-memory versus active-compute boundaries, routing skew/capacity/topology risks, phase-aware measurements, quality matching, and evidence requirements.

## Batch 0032

Input: train records 311-320 of `research/ai-infra-expert/corpus/train.jsonl` (source IDs `corpus-00342`, `corpus-00343`, `corpus-00344`, `corpus-00346`, `corpus-00347`, `corpus-00348`, `corpus-00350`, and `corpus-00351`-`corpus-00353`, preserving corpus order).

Result:

```text
records processed: 10
source ID alignment: PASS
keep: 0
rewrite: 10
reject: 0
initial schema check: PASS; fresh accumulated JSONL parse, required-field, enum, duplicate, source-field, and corpus-order checks passed
repair: not required
final schema check: PASS
manifest hash check: pending until final manifest regeneration
```

This batch calibrated small controlled MoE experiments, investigation runbooks, and quantization definitions. Corrections specified matched workloads and quality targets, router dispatch/expert execution/combine mechanisms, phase-specific telemetry, memory accounting, kernel and calibration assumptions, capacity/overflow boundaries, failure modes, and evidence requirements.
