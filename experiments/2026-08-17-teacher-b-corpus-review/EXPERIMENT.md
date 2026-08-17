# Experiment: teacher-B corpus review (blind, independent second opinion)

Started: 2026-08-17
Lane: teacher-B
Reviewer model: claude-opus-5 (provider: copilot), pinned explicitly so this lane
is NOT the same model that produced teacher-A (gpt-5.6-luna-current).

## Purpose

Produce a second, INDEPENDENT provisional calibration of the same 6000 corpus
records that teacher-A calibrated, so that inter-teacher agreement can be measured
afterwards. This is a review lane, not gold labels, and not evidence of model
domain capability.

## Mode: BLIND

For each record the reviewer sees ONLY:
  - source_user   (from research/ai-infra-expert/corpus/*.jsonl)
  - source_assistant (from the same raw corpus)

The reviewer MUST NOT read, open, or consult any file under
experiments/2026-08-14-teacher-a-corpus-calibration/ while producing a batch.
teacher-A's corrected_answer is deliberately withheld to avoid anchoring.
Agreement against teacher-A is computed only in a separate later analysis step.

## Isolation guarantees

- Outputs live ONLY under this directory's results/.
- teacher-A results are read-only for this lane and are never modified.
- Raw corpus (research/ai-infra-expert/corpus/) is never modified.
- Benchmark raw generations are never modified.
- teacher_lane is always "teacher-B"; teacher-A files are never overwritten.

## Scope

- train: 5399 records, in corpus order (prefix-aligned)
- validation: 601 records, in corpus order (prefix-aligned)
- total: 6000

## Output schema (per JSONL record)

Same 12 required fields as teacher-A so the two lanes are directly comparable:

  source_id           str, exact id from raw corpus, corpus order preserved
  teacher_lane        "teacher-B"
  teacher_model       "claude-opus-5-current"
  calibration_status  "provisional"
  decision            one of keep | rewrite | reject
  source_user         exact copy of raw corpus user content
  source_assistant    exact copy of raw corpus assistant content
  corrected_answer    non-empty str, teacher-B's independent answer
  quality_dimensions  {technical_correctness, instruction_coverage, operational_safety} ints 1-5
  risks               list[str]
  evidence_required   list[str]
  confidence          float in [0,1]

## Status

Progress: train 60/5399; validation 0/601; total 60/6000; remaining 5940.

Runs are appended below, newest first.

## Run log (newest first)

### 2026-08-17 — train-batch-0007.jsonl

- Batch file: results/train-batch-0007.jsonl
- Builder: scripts/gen_batch_0007.py
- Corpus range: research/ai-infra-expert/corpus/train.jsonl lines 61-70 (0-indexed 60-69)
- Source IDs: corpus-00069, corpus-00070, corpus-00071, corpus-00072, corpus-00073,
  corpus-00074, corpus-00075, corpus-00076, corpus-00079, corpus-00080
- Progress after this batch: train 70/5399, validation 0/601, total 70/6000,
  remaining 5930
- Decisions: keep=0, rewrite=10, reject=0
- Initial schema check: scripts/verify_batches.py → VERIFY_PASS on first run
  (train=70/5399 validation=0/601 total=70/6000). No repair actions were needed.
- Repair actions: none.
- Final schema check: VERIFY_PASS (JSONL line-parse, batch count 10, all 12 required
  fields, teacher_lane/teacher_model/calibration_status/decision value checks,
  byte-exact source_user and source_assistant equality against the raw corpus,
  non-empty corrected_answer, confidence within [0,1], globally unique source_id,
  and train sequence is a strict prefix of corpus order).
- Manifest: MANIFEST.sha256 regenerated over all 19 files in this directory except
  the manifest itself; `sha256sum -c MANIFEST.sha256 --quiet` → MANIFEST_OK.

#### Technical topics covered by this batch

All ten records are prefill-themed Knowledge/Concept items whose seed assistant text
is the same one-sentence stub ("Prefill processes the prompt and is generally parallel
across prompt tokens..."), which is directionally correct but non-actionable, so every
record was marked `rewrite`. The rewrites cover:

- Prefill vs decode roofline separation: prefill ≈ 2 · P_active · N FLOPs (compute-bound,
  weight reads amortised over N) versus decode ≈ 2 · P_active per token (HBM-bandwidth
  bound), plus the O(L · H · N² · d_head) attention term that dominates at long context.
- KV cache sizing: bytes/token = 2 · L · H_kv · d_head · dtype_bytes, with the GQA/MQA
  ratio and FP8 KV as the two largest capacity levers.
- Scheduler coupling: head-of-line blocking of decode by long prefills, chunked prefill
  as a TTFT-vs-ITL trade, and max_num_batched_tokens trading GEMM tile efficiency against
  KV-pool capacity (and its interaction with tensor-parallel degree).
- Prefix caching: block-hash reuse of prefix KV, the offline longest-common-prefix
  characterisation that must precede any experiment, template-ordering pitfalls, cache
  thrashing under load, and a temperature-0 output-equivalence correctness gate.
- Prefill/decode disaggregation (NVIDIA Dynamo, Mooncake style): KV transfer volume over
  the fabric, the RDMA/RoCE and GPUDirect prerequisites, perftest/ib_write_bw baselining,
  ECN/PFC and retransmit counters, and the transfer_time < ~0.3 · prefill_time break-even.
- Length-aware admission / shortest-job-first approximation, per-length-bucket metrics,
  and explicit starvation instrumentation for long requests.
- Measurement hygiene: assumptions that must be stated before any prefill performance
  claim (active vs total parameters for MoE, uncached N after prefix-cache matching, KV
  dtype, clock locking and throttle reasons, MIG/co-tenancy, open- vs closed-loop load
  generation, TTFT measurement boundary, percentile estimator and repetition count).

Each rewrite states an explicit mechanism, at least one boundary condition where the
claim stops holding, a falsifiable prediction where applicable, the evidence required to
check it, and a rollback gate.

#### Status caveats

These records are **provisional teacher-B output only**. They are NOT expert gold labels,
NOT validated by a human domain expert, and NOT evidence of any model's domain capability
(model domain capability and runtime/system capability remain strictly separate concerns).
Agreement with teacher-A was not computed and teacher-A artefacts were not read while
producing this batch — this lane is blind by construction.

### 2026-08-17 — train-batch-0006.jsonl

- Batch file: results/train-batch-0006.jsonl
- Builder: scripts/gen_train_batch_0006.py
- Corpus range: research/ai-infra-expert/corpus/train.jsonl lines 51-60 (0-indexed 50-59)
- Source IDs: corpus-00055, corpus-00057, corpus-00059, corpus-00060, corpus-00061,
  corpus-00062, corpus-00063, corpus-00064, corpus-00065, corpus-00067
- Records this run: 10
- Progress after run: train 60/5399; validation 0/601; total 60/6000; remaining 5940.
- Decisions: keep 0, rewrite 10, reject 0.
- Initial schema check: PASS on first run of scripts/verify_batches.py
  (`train=60/5399 validation=0/601 total=60/6000` then `VERIFY_PASS`).
- Repairs performed: none required this run.
- Final schema check: PASS (same invocation; no re-run needed since the first check passed).
- Manifest: MANIFEST.sha256 regenerated over all 17 files in this directory except
  the manifest itself; `sha256sum -c MANIFEST.sha256 --quiet` returned clean (MANIFEST_OK).

Technical topics covered by this batch. All ten records are `Knowledge/Concept`
items on the concept **prefill**, spanning four question shapes: define-and-motivate
(corpus-00055), contrast-with-naive-implementation (corpus-00057, 00059, 00060),
enumerate-two-failure-modes (corpus-00061 through 00065), and
interaction-with-latency/throughput/memory (corpus-00067). Every source answer was
the same single generic sentence ("prefill processes the prompt and is generally
parallel across prompt tokens..."), which is directionally true but answers none of
the four question shapes — hence 10/10 `rewrite` and an `instruction_coverage` score
of 1 on nine of ten records.

The rewritten answers develop: the prefill/decode roofline split (prefill ~2*P*N
FLOPs, compute-bound; decode ~2*P FLOPs/token but HBM-bandwidth-bound); the O(N^2)
attention term and the crossover N* that breaks linear TTFT capacity plans; KV cache
sizing arithmetic (2 * layers * kv_heads * head_dim * N * dtype_bytes, worked through
for a GQA 9B-class model on a 24 GB A30); head-of-line blocking by unchunked prefill
and the chunked-prefill chunk-size trade-off (U-shaped TTFT/ITL objective); admission
control on tail rather than mean input length and the self-amplifying
preemption/recompute loop; prefix-cache reuse validity conditions (byte-identical
prefix, tokenizer/model revision, K/V dtype, RoPE offset handling) as a silent
correctness hazard; prefill/decode disaggregation via NVIDIA Dynamo and Mooncake,
with the KV-transfer cost model and the GPUDirect RDMA over RoCE dependency
(including silent host-bounce fallback and PFC/ECN pause-frame counters as
invalidating conditions); and tensor-parallel prefill exposing heterogeneous
interconnect paths that NCCL will route around silently rather than fail on.

Each rewritten answer states an explicit falsifiable hypothesis, the evidence needed
to test it (phase-separated Nsight/DCGM traces, TTFT/ITL percentiles,
gpu_cache_usage_perc, num_preempted_requests, nccl-tests bus bandwidth,
NCCL_DEBUG=INFO transport selection), and a numeric rollback gate.

Caveat. These outputs are **provisional** teacher-B review labels. They are not
expert gold, they have not been validated against a running system, and they are not
evidence of any model's domain capability. Numeric examples are flagged in-text as
assumptions rather than measured platform facts. Agreement with teacher-A has not
been computed and was not consulted while producing this batch (blind lane).

### 2026-08-17 — train-batch-0005.jsonl

- Batch file: results/train-batch-0005.jsonl
- Builder: scripts/build_train_batch_0005.py
- Corpus range: research/ai-infra-expert/corpus/train.jsonl lines 41-50 (0-indexed 40-49)
- Source IDs: corpus-00044, corpus-00045, corpus-00046, corpus-00047, corpus-00048,
  corpus-00049, corpus-00050, corpus-00052, corpus-00053, corpus-00054
  (corpus order preserved verbatim; nothing skipped or reordered — the corpus itself
  has no corpus-00051 in this span)
- Progress after this run: train 50/5399; validation 0/601; total 50/6000; remaining 5950
- Decisions: keep=0, rewrite=10, reject=0
- Initial schema/ad-hoc check: PASS on first run (scripts/verify_batches.py →
  train=50/5399 validation=0/601 total=50/6000, VERIFY_PASS)
- Repairs performed: none required
- Final schema/ad-hoc check: PASS (identical output)
- Manifest: MANIFEST.sha256 regenerated over all files in this directory except itself;
  `sha256sum -c` reported all files OK
- Technical topics covered by this batch: KV cache sizing and growth
  (bytes = 2 * layers * kv_heads * head_dim * seq_len * batch * dtype_bytes, GQA/MQA
  effects, fp8 KV quantization trade-off), prefix-cache reuse and its invalidation
  boundaries, KV leak vs. legitimate growth triage, KV transfer under disaggregated
  prefill/decode (Mooncake- / NVIDIA Dynamo-style splits) over RDMA/RoCE with
  GPUDirect, and prefill semantics — compute-bound vs. bandwidth-bound regimes,
  quadratic attention scaling, chunked prefill interference with decode SLOs, and
  TTFT budgeting.
- Review rationale: all ten source assistant answers were one-sentence generic
  definitions that ignored the explicit instruction to give one concrete mechanism and
  one boundary condition, hence uniformly `rewrite` with instruction_coverage=1.
- Blind-mode compliance: no file under experiments/2026-08-14-teacher-a-corpus-calibration/
  was read, opened, grepped, or otherwise consulted while producing this batch. Only
  source_user / source_assistant from the raw corpus were visible.
- These outputs are PROVISIONAL teacher-B review, not expert gold labels, and are not
  evidence of any model's domain capability.

### 2026-08-17 — train-batch-0004.jsonl

- Batch file: results/train-batch-0004.jsonl
- Builder: scripts/build_train_batch_0004.py
- Corpus range: research/ai-infra-expert/corpus/train.jsonl lines 31-40 (0-indexed 30-39)
- Source IDs: corpus-00034, corpus-00035, corpus-00036, corpus-00037, corpus-00038,
  corpus-00039, corpus-00040, corpus-00041, corpus-00042, corpus-00043
  (corpus order preserved verbatim; nothing skipped or reordered)
- Progress after this run: train 40/5399; validation 0/601; total 40/6000; remaining 5960
- Decisions: keep=0, rewrite=10, reject=0
- Initial schema/ad-hoc check: PASS on first run (scripts/verify_batches.py →
  train=40/5399 validation=0/601 total=40/6000, VERIFY_PASS)
- Repairs performed: none required
- Final schema/ad-hoc check: PASS (identical output)
- Manifest: MANIFEST.sha256 regenerated over all files in this directory except itself;
  `sha256sum -c` reported all files OK
- Blind-mode compliance: no file under experiments/2026-08-14-teacher-a-corpus-calibration/
  was read, opened, grepped, or otherwise consulted while producing this batch. Only
  source_user / source_assistant from the raw corpus were visible.

Technical topics covered by this batch (all KV-cache themed, but three distinct
instruction shapes):
1. Training vs inference KV cache — parallelism/sharding view: per-GPU
   kv_bytes = 2 * layers * seq_len * (kv_heads / TP) * head_dim * bytes_per_elem, and the
   GQA/MQA regime where kv_heads < TP forces replication so extra TP stops reducing
   per-GPU KV; pipeline parallelism dividing by layers/PP at the cost of bubbles;
   pinned-cache implications for re-sharding and for disaggregated KV transfer
   (Mooncake / NVIDIA Dynamo) over RDMA.
2. Training vs inference KV cache — numerics view: why fp8 KV is a capacity lever
   (halves kv_bytes, halves the bandwidth-bound TPOT floor) but carries error forward
   for the life of a sequence, so equivalence must be tested at max served context with a
   pre-registered token-match / task-metric criterion rather than bitwise identity.
3. Misleading intuitions corrected: (a) "cache makes decode compute-cheap" — decode is
   bandwidth-bound at ~1 MAC/byte with TPOT_floor ≈ kv_bytes / achievable_HBM_bandwidth;
   (b) "KV is small next to weights" — per-sequence KV scales with concurrency × context
   and drives replica count; (c) "prefix caching always helps" — bounded by
   shared_prefix_tokens / total_prompt_tokens of prefill only, zero for decode, plus
   cross-tenant block-reuse isolation and timing-side-channel risk; (d) "TP always splits
   KV" — the kv_heads < TP replication knee; (e) "a cache hit is free" — hits convert
   compute cost into pool residency and, when disaggregated, into an RDMA transfer on the
   TTFT critical path, so admission control must key on occupancy and preemption rate,
   not hit rate.
4. Controlled experiment designs: decode bandwidth-floor sweep at batch size 1 against an
   independently measured achievable bandwidth; pre-registered concurrency-ceiling ramp to
   first preemption with fixed sequence length as the control; and an interleaved A/B for
   KV quantisation gated on quality first, then capacity, with shadow → canary → rollout
   staging.

All ten source_assistant values were the same generic KV-cache definition regardless of
what the prompt asked for, so every record scored instruction_coverage = 1 and was marked
`rewrite`: the source text is not factually wrong, it is a topic-shaped non-answer.
Confidence 0.72 reflects that the defect is unambiguous while the replacement answers
carry unverified platform-specific assumptions.

**These results are provisional.** They are one model's independent review pass, not
expert gold labels, and they are NOT evidence of model domain capability. Agreement with
teacher-A has not been computed and is out of scope for this lane.

### 2026-08-17 11:1x UTC — train-batch-0003.jsonl

- Batch file: results/train-batch-0003.jsonl
- Corpus range: research/ai-infra-expert/corpus/train.jsonl lines 21-30 (0-indexed 20-29)
- Source IDs: corpus-00024, corpus-00025, corpus-00026, corpus-00027, corpus-00028,
  corpus-00029, corpus-00030, corpus-00031, corpus-00032, corpus-00033
  (corpus order preserved verbatim; nothing skipped or reordered)
- Progress after this run: train 30/5399; validation 0/601; total 30/6000; remaining 5970
- Decisions: keep=0, rewrite=10, reject=0
- Initial schema/ad-hoc check: PASS on first run (scripts/verify_batches.py →
  train=30/5399 validation=0/601 total=30/6000, VERIFY_PASS)
- Repairs performed: none required
- Final schema/ad-hoc check: PASS (identical output)
- Manifest: MANIFEST.sha256 regenerated over all files in this directory except itself;
  `sha256sum -c` reported all files OK
- Blind-mode compliance: no file under experiments/2026-08-14-teacher-a-corpus-calibration/
  was read, opened, grepped, or listed while producing this batch.

Technical topics covered by this batch. All ten records are `Knowledge/Concept`,
difficulty medium, concept `kv_cache`, and split across three question families:
(a) measurement plans for validating whether the KV cache helps a serving workload
(variants 4-5), (b) the assumptions that must be declared before any KV-cache
performance claim (variants 1-5), and (c) how the KV cache differs between training
and inference (variants 1-3). The rewritten answers cover: the KV sizing identity
kv_bytes = 2 * layers * seq_len * kv_heads * head_dim * bytes_per_elem; paged
allocation and bounded internal fragmentation; the decode bandwidth roofline
TPOT_floor ≈ kv_bytes / achievable_HBM_bandwidth; prefix caching / RadixAttention
hit-rate mechanisms and multi-tenant reuse leakage; KV quantisation as a
throughput/accuracy trade with a greedy-token equivalence gate; tensor-parallel KV
sharding and the GQA case where kv_heads < TP forces replication; disaggregated
prefill/decode (Mooncake, NVIDIA Dynamo) with KV moved over RDMA/RoCE, GDR and
HCA-to-GPU affinity, and PFC/ECN pause-frame counters as the fault signal; and
open-loop vs closed-loop load generation as a statistical-validity assumption.
Every record carries an explicit falsifiable hypothesis, a boundary condition, an
evidence list, and a rollback gate.

Uniform assessment for this batch: every `source_assistant` is the same generic
KV-cache definition regardless of which question was asked, so
instruction_coverage is 1 across the batch while technical_correctness is 4 (the
statement itself is true, just not responsive). Decision is therefore `rewrite`,
not `reject`: the prompts are usable, the paired answers are not.

IMPORTANT: these results are PROVISIONAL teacher-B review output. They are not
expert gold labels, they have not been validated by a human domain expert, and
they say nothing about any model's domain capability. Agreement with teacher-A is
deliberately unknown at this point and is computed only in a separate later step.

### 2026-08-17 11:0x UTC — train-batch-0002.jsonl

- Batch file: results/train-batch-0002.jsonl
- Corpus range: research/ai-infra-expert/corpus/train.jsonl lines 11-20
- Source IDs: corpus-00013, corpus-00014, corpus-00015, corpus-00017, corpus-00018,
  corpus-00019, corpus-00020, corpus-00021, corpus-00022, corpus-00023
  (corpus ids remain non-contiguous in the raw file; line order preserved verbatim,
  nothing skipped or reordered)
- Progress after this run: train 20/5399; validation 0/601; total 20/6000; remaining 5980
- Decisions: keep=0, rewrite=10, reject=0
- Initial schema/ad-hoc check: PASS on first run (scripts/verify_batches.py)
- Repairs performed: none required
- Final schema/ad-hoc check: PASS (train=20/5399 validation=0/601 total=20/6000, VERIFY_PASS)
- Manifest: MANIFEST.sha256 regenerated over all files in this directory except itself;
  `sha256sum -c` reported all files OK
- Blind-mode compliance: no file under experiments/2026-08-14-teacher-a-corpus-calibration/
  was read, opened, or grepped while producing this batch. Only source_user and
  source_assistant from the raw corpus were consulted.

Technical topics covered by this batch: still the KV cache cluster (concepts=["kv_cache"],
task_type=explanation), now spanning three prompt intents — (a) two failure modes / trade-offs
(easy variants 3-5), (b) how KV cache interacts with latency, throughput or memory (medium
variants 2-5), (c) a measurement plan validating whether KV cache helps a serving workload
(medium variants 1-3). The rewrites cover: the per-sequence size model
2 * layers * seq_len * num_kv_heads * head_dim * dtype_bytes with a worked 1.07 GB example
stated as an assumption; decode as an HBM-bandwidth-bound regime and the
max(compute, (weight_bytes + batch*kv_bytes)/BW) roofline with its throughput knee; paged
allocation, internal vs. external fragmentation and block-table indirection; prefix-cache reuse
correctness hazards under LoRA swap, KV-dtype change or position-offset change (silent wrong
logits, no crash signal); preemption thrash with swap-over-PCIe vs. evict-and-recompute and the
crossover between them; per-tenant KV quotas and p99 tail unfairness; the mitigation lever set
(GQA/MQA, fp8/int8 KV, paging, prefix caching, disaggregated prefill/decode in the
Dynamo/Mooncake style with KV transfer over RDMA/RoCE and GPUDirect RDMA); and multi-GPU
behaviour where tensor parallelism shards the cache by TP degree but adds an NCCL collective
latency floor that makes cross-node TP for decode normally the wrong choice, while pipeline
parallelism does not shard KV within a stage at all. The measurement-plan records specify
production-trace replay rather than fixed-length synthetic load, one-variable-at-a-time arms,
closed-loop load generation, three repetitions with warm-up discarded, three-layer
instrumentation (client / engine / DCGM GPU counters), cold-vs-warm prefix-cache reporting, a
two-direction capacity perturbation to separate compute-bound from bandwidth-bound from
capacity-bound behaviour, and pre-registered falsifiable thresholds with explicit rollback gates.

Why every record was marked rewrite: the raw corpus again reuses one identical single-sentence
assistant answer verbatim across all ten records and all three distinct prompt intents. The
sentence is not false about caching and the memory-scaling factors, but it answers none of the
asked questions — no enumerated failure modes, no latency/throughput/memory interaction, and
no measurement plan whatsoever. It carries no units, no boundary condition, no falsifiable
hypothesis, no required evidence and no rollback gate. instruction_coverage was scored 1/5 for
all ten; technical_correctness 3/5 (non-false but under-specified); operational_safety 2/5
(gives an operator nothing actionable and no blast-radius control).

### 2026-08-17 10:58 UTC — train-batch-0001.jsonl

- Batch file: results/train-batch-0001.jsonl
- Corpus range: research/ai-infra-expert/corpus/train.jsonl lines 1-10
- Source IDs: corpus-00001, corpus-00003, corpus-00004, corpus-00005, corpus-00006,
  corpus-00007, corpus-00008, corpus-00009, corpus-00010, corpus-00012
  (note: corpus ids are non-contiguous in the raw file; line order was preserved verbatim,
  nothing skipped or reordered)
- Progress after this run: train 10/5399; validation 0/601; total 10/6000; remaining 5990
- Decisions: keep=0, rewrite=10, reject=0
- Initial schema/ad-hoc check: PASS on first run (scripts/verify_batches.py)
- Repairs performed: none required
- Final schema/ad-hoc check: PASS (train=10/5399 validation=0/601 total=10/6000, VERIFY_PASS)
- Manifest: MANIFEST.sha256 regenerated over all files in this directory except itself;
  `sha256sum -c` reported all files OK
- Blind-mode compliance: no file under experiments/2026-08-14-teacher-a-corpus-calibration/
  was read, opened, or grepped while producing this batch. Only source_user and
  source_assistant from the raw corpus were consulted.

Technical topics covered by this batch: all ten records sit in the KV cache cluster of the
Knowledge/Concept category (concepts=["kv_cache"], task_type=explanation, difficulty=easy),
split across three prompt intents — (a) define KV cache and why it matters, (b) contrast KV
cache against a naive no-cache decode path, (c) name two failure modes / trade-offs. The
rewrites cover the autoregressive caching mechanism itself, the per-sequence size model
bytes = 2 * layers * seq_len * num_kv_heads * head_dim * dtype_bytes (explicitly num_kv_heads,
so GQA/MQA reduces it), the shift of the decode bottleneck from GEMM FLOPs to HBM capacity and
bandwidth, paged/block KV allocation and fragmentation, capacity exhaustion under concurrency
with preemption / evict-and-recompute / host-memory swap over PCIe, and the mitigation
trade-space (GQA/MQA, fp8-int8 KV quantization, offload, sliding-window/eviction) with its
quality cost.

Why every record was marked rewrite: the raw corpus reuses one identical single-sentence
assistant answer across all three distinct prompt intents. That sentence is technically
non-false about caching and memory scaling factors, but it does not answer the contrast or
failure-mode variants at all, carries no units, no quantitative size model, no explicit
boundary condition tied to an HBM budget or concurrency level, no falsifiable claim, no
required evidence and no rollback gate. Instruction coverage was therefore scored 2/5 for the
definitional variants and 1/5 for the contrast and failure-mode variants.

Provisional status disclaimer: these teacher-B outputs are PROVISIONAL model-authored review
notes. They are NOT expert gold labels, have NOT been validated by a human domain expert or by
execution against real hardware, and they are NOT evidence of any model's domain capability.
Numeric examples in corrected answers are order-of-magnitude illustrations and must be
re-derived from the actual model config before use. Inter-teacher agreement against teacher-A
is deliberately NOT computed here; it is a separate, later, independent analysis step.
