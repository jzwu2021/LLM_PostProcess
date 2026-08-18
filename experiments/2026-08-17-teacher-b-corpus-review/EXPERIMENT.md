# Experiment: teacher-B corpus review (blind, independent second opinion)

Started: 2026-08-17
Lane: teacher-B
Reviewer model: claude-opus-5 (provider: copilot), pinned explicitly so this lane
is NOT the same model that produced teacher-A (gpt-5.6-luna-current).

## Run 2026-08-18 batch 0139

- Batch file: results/train-batch-0139.jsonl
- Corpus range: train.jsonl lines 1381-1390 (0-indexed 1380..1389)
- Source IDs: corpus-01525, 01526, 01527, 01528, 01529, 01530, 01533, 01534,
  01535, 01536 (corpus order preserved verbatim; the gap at 01531/01532 is in the
  source corpus itself and was NOT introduced here)
- Progress: train 1390/5399, validation 0/601, total 1390/6000, remaining 4610
- Decisions: keep=0, rewrite=10, reject=0
- Initial schema check: PASS (0 errors) on first run — JSONL line-parse, 10 records,
  12 required fields present, teacher_lane/teacher_model/calibration_status/decision
  values correct, byte-exact source_user and source_assistant vs corpus,
  non-empty corrected_answer, confidence in [0,1], globally unique source_id across
  all 1390 records, contiguous batch numbering, and the aggregated train sequence is
  an exact prefix of train.jsonl.
- Repairs: none required.
- Final schema check: PASS.
- Manifest: MANIFEST.sha256 regenerated over 237 files; `sha256sum -c` all OK.
- Technical topics covered: this is again the identical-rubric family "long-context
  workload intermittently hits OOM after several concurrent requests" (scenario
  variants 225-236). Because the prompts are textually near-identical, the ten
  corrected answers were deliberately diversified by primary mechanism hypothesis,
  measurement plan and rollback gate rather than by prose paraphrase:
  (1) KV high-water mark vs admission control, with the explicit
  2*n_layers*n_kv_heads*head_dim*dtype*seq_len KV formula and a halved-max_num_seqs
  A/B on a replayed fixed-seed trace; (2) allocator fragmentation, discriminated by
  the reserved-minus-allocated gap and an expandable_segments arm that must leave
  peak allocated bytes unchanged within 2%; (3) prefix-cache retention and
  multi-tenant sharing, predicting OOM tracks prefix-hit-rate collapse rather than
  QPS, tested by a shared-prefix vs randomised-prefix synthetic; (4) chunked prefill,
  arguing the failing allocation is prefill activation not KV and using the OOM
  requested-bytes field itself as the discriminator; (5) tensor-parallel per-rank
  skew including NCCL registered buffers outside the PyTorch allocator and MoE
  expert imbalance, requiring per-rank rather than averaged memory series;
  (6) disaggregated prefill/decode KV transfer buffers in the Mooncake / NVIDIA
  Dynamo architectural pattern, where RDMA-registered staging memory scales with
  in-flight transfers and RoCE congestion silently inflates residency, plus the
  GPUDirect-RDMA-vs-host-bounce-buffer verification requirement; (7) speculative
  decoding draft model, multi-LoRA adapters and structured-output logits-processor
  state as hidden co-tenants whose OOM threshold shifts linearly with speculation
  length; (8) capacity-model-first arithmetic deriving the theoretical concurrency
  ceiling before touching any knob, with the explicit caveat that naive KV formulas
  are invalid for MLA, sliding-window and hybrid-attention models; (9) failure-mode
  containment — making OOM survivable via preemption, router token-budget admission
  gates and drain-aware readiness probes before making it rare, validated by
  staging-only fault injection; (10) time-correlated drift across deploy SHAs,
  image digests, driver/CUDA/NCCL/engine versions and prompt-length p95 steps,
  discriminated by replaying today's trace against the last known-good image.
  Every record states assumptions, one falsifiable hypothesis, the measurements
  that would refute it, expected confounders, prioritised mitigations and an
  explicit numeric rollback gate.
- Status caveat: these outputs are PROVISIONAL teacher-B blind-review artifacts.
  They are not expert gold labels, they have not been adjudicated against teacher-A
  (which remained unread during this run, per the blind protocol), and they say
  nothing about any model's domain capability. Agreement analysis is a separate
  later step outside this worker's scope.

## Run 2026-08-17 batch 0138

- Batch file: results/train-batch-0138.jsonl
- Corpus range: train.jsonl lines 1371-1380 (0-indexed 1370..1379)
- Source IDs: corpus-01515 .. corpus-01524
- Progress: train 1380/5399, validation 0/601, total 1380/6000, remaining 4620
- Decisions: keep=0, rewrite=10, reject=0
- Initial schema check: PASS (0 errors) on first run — JSONL line-parse, 10 records,
  12 required fields, lane/model/status/decision values, byte-exact source_user and
  source_assistant vs corpus, non-empty corrected_answer, confidence in [0,1],
  globally unique source_id, aggregated train sequence is an exact corpus prefix.
- Repairs: none required.
- Final schema check: PASS.
- Manifest: MANIFEST.sha256 regenerated over 229 files; `sha256sum -c` all OK.
- Technical topics covered: long-context KV-cache OOM under concurrency. The ten
  records are diversified across distinct diagnostic lenses so identical rubric
  prompts do not collapse into identical answers: (1) admission control and the
  concurrency high-water mark, (2) reserved-vs-allocated split as the fragmentation
  discriminator, (3) KV-pool capacity sizing on p99 rather than mean context,
  (4) prefill transient spike vs decode steady state and chunked prefill,
  (5) block-release/leak accounting vs legitimate cache retention, (6) prefix/radix
  cache as capacity relief or added pressure, (7) KV quantization constant-factor
  relief with a pre-agreed quality gate, (8) tensor-parallel per-rank imbalance and
  the num_kv_heads < TP replication boundary, (9) context-limit policy with shadow-mode
  rejection-rate measurement, (10) reproduction discipline as a gate before any A/B.
  Each answer states assumptions, a falsifiable hypothesis, ordered measurements,
  a single-variable experiment design, confounders, an explicit boundary condition,
  rollback thresholds, and what would refute the framing.
- Status: these outputs are PROVISIONAL teacher-B second opinions produced blind
  (teacher-A artifacts were not read at any point). They are NOT expert gold labels,
  have not been validated against real hardware measurements, and say nothing about
  any model's domain capability. Agreement analysis is a separate later step.

## Run 2026-08-17 batch 0137

- Batch file: results/train-batch-0137.jsonl
- Corpus range: train.jsonl lines 1361-1370 (0-indexed 1360..1369)
- Source IDs: corpus-01505 .. corpus-01514
- Progress: train 1370/5399, validation 0/601, total 1370/6000, remaining 4630
- Decisions: keep=0, rewrite=10, reject=0
- Initial schema check: PASS (0 errors) on first run — JSONL line-parse, 10 records,
  12 required fields, lane/model/status/decision values, byte-exact source_user and
  source_assistant vs corpus, non-empty corrected_answer, confidence in [0,1],
  globally unique source_id, aggregated train sequence is an exact corpus prefix.
- Repairs: none required.
- Final schema check: PASS.
- Manifest: MANIFEST.sha256 regenerated over 232 files; `sha256sum -c` all OK.
- Topics covered: all ten items are the same rubric family — intermittent OOM in a
  long-context LLM serving workload under concurrency (variants 205-214). To avoid
  ten near-identical answers, each record was written through a distinct diagnostic
  lens: (1) KV high-water vs steady-state accounting, (2) allocator fragmentation vs
  true exhaustion, (3) scheduler over-subscription and preemption thrash, (4) prefix/
  paged cache reuse under heterogeneous prompts, (5) non-KV activation/workspace and
  CUDA-graph pools, (6) gpu_memory_utilization headroom vs co-resident consumers,
  (7) tensor-parallel per-rank imbalance, (8) weight vs KV quantization as distinct
  levers, (9) output-length tail driving the concurrency-weighted peak, (10) monotonic
  leak vs bounded oscillation. Every record states the KV byte arithmetic
  (2 * n_layers * n_kv_heads * d_head * bytes_per_elem), a named falsifiable
  hypothesis with a refutation condition, a controlled single-factor experiment with
  repeats and confounders, measurements with units, and explicit rollback thresholds.
- All source records were graded rewrite because the corpus assistant field contains a
  grading rubric ("Answer should state...", "Minimum technical points:") rather than an
  engineering answer: no units, no arithmetic budget, no stated hypothesis, no rollback
  gate. Fine-tuning on that text would teach meta-commentary about answers.
- Blind-review compliance: no file under experiments/2026-08-14-teacher-a-corpus-calibration/
  was read, opened, or searched during this run. Inputs were the corpus source_user and
  source_assistant only.
- Status: these outputs are PROVISIONAL teacher-B reviews produced by a general model.
  They are NOT expert gold labels, have not been validated by a human domain expert, and
  say nothing about any trained model's domain capability. Agreement analysis against
  teacher-A is a separate, later step outside this task.

## Run 2026-08-17 batch 0136

- Batch file: results/train-batch-0136.jsonl
- Corpus range: train.jsonl lines 1351-1360 (0-indexed 1350..1359)
- Source IDs: corpus-01495 .. corpus-01504
- Progress: train 1360/5399, validation 0/601, total 1360/6000, remaining 4640
- Decisions: keep=0, rewrite=10, reject=0
- Initial schema check: PASS (0 errors) on first run of /tmp/tb_verify.py
- Repairs: none required
- Final schema check: PASS — train=1360/5399, validation=0/601, total=1360, ERRORS: 0
- Manifest: MANIFEST.sha256 regenerated over 231 files, `sha256sum -c` all OK
- Technical topics covered: all ten items are the same rubric family — intermittent CUDA OOM in a long-context serving workload under concurrency. To avoid ten near-identical answers, each record was written from a distinct diagnostic axis: (1) KV-cache byte accounting vs free-HBM budget, (2) allocator fragmentation vs true capacity exhaustion (reserved-minus-allocated, expandable_segments), (3) token-budget admission control and scheduler backpressure, (4) preemption/recompute and swap-space path, (5) prefix/radix cache sharing as a capacity multiplier, (6) KV and weight quantization (fp8 KV) as a budget lever with an accuracy gate, (7) tensor/pipeline parallelism topology and per-GPU KV sharding vs NCCL buffer growth, (8) memory-utilization fraction vs non-KV consumers (CUDA graphs, activation workspace, comm buffers), (9) multi-tenant co-residency and noisy-neighbour HBM contention, (10) the decode tail — unbounded max_tokens as the binding variable. Every answer states assumptions, an explicit falsifiable hypothesis with a prediction, a single-variable controlled experiment on a replayed trace, named confounders, required evidence, and a rollback gate tied to p99 latency / goodput / accuracy / OOM recurrence.
- Rubric-source note: the corpus `assistant` field for this family is a grading rubric, not a model answer, so `decision=rewrite` throughout with instruction_coverage scored 2.
- Status: these results are PROVISIONAL teacher-B blind review output produced by the current conversation model. They are NOT expert gold labels, have not been validated by a human domain expert, and say nothing about any trained model's domain capability. Blind protocol held: no teacher-A artifact was read, opened, or searched during this batch.

## Run 2026-08-17 batch 0135

- Batch file: results/train-batch-0135.jsonl
- Corpus range: train.jsonl lines 1341-1350 (0-indexed 1340..1349)
- Source IDs: corpus-01484, corpus-01485, corpus-01486, corpus-01487, corpus-01488,
  corpus-01489, corpus-01490, corpus-01492, corpus-01493, corpus-01494
- Progress: train 1350/5399, validation 0/601, total 1350/6000, remaining 4650
- Decisions: keep=0, rewrite=10, reject=0
- Initial schema check: PASS (verify_batches.py, verify_teacher_b.py, verify_run.py — 10 lines parse
  as JSONL, all 12 required fields present, teacher_lane/teacher_model/calibration_status/decision
  values correct, source_user and source_assistant byte-identical to corpus, corrected_answer
  non-empty, confidence in [0,1], quality_dimensions 1-5 integers, risks/evidence_required arrays,
  source_id globally unique across all 135 batches, aggregated train sequence a strict prefix of
  train.jsonl, validation still empty)
- Repairs applied: none required this run.
- Final schema check: PASS (train 1350, validation 0, total 1350, unique_ids 1350)
- Manifest: MANIFEST.sha256 regenerated over 225 files; `sha256sum -c` 225/225 OK, no failures.

### Technical topics covered by this batch

Same rubric family as the preceding long-context OOM block, deliberately re-angled to avoid
template repetition. This batch is organised around three distinct analytical angles rather than a
shared checklist: (a) Troubleshooting — lifetime accounting, i.e. which bytes are actually released
at request completion versus pinned by slow-finishing unbounded generations, and how to tell
monotonic free-block decay (accumulation) from a single-interval collapse (one large allocation);
(b) Performance Analysis — failure-time attribution via a full per-device HBM decomposition
(weights / CUDA context / KV pool / activation workspace peak / unexplained remainder), with the
remainder used as the discriminator between a serving problem and a co-tenancy problem, plus a
CUDA memory snapshot to recover the size of the failing allocation instead of guessing it;
(c) System Design — making the unbounded generation tail a declared, enforceable contract
(mandatory max_tokens, token-budget admission on sum(P_i + max_tokens_i)) as the precondition on
which every other mitigation depends. Cross-cutting content: paged-KV capacity arithmetic with the
explicit P-known / G-unknown asymmetry, chunked prefill as a workspace-peak bound, expandable
segments for fragmentation, KV quantization gated on a fixed accuracy eval, and tensor parallelism
priced as a per-layer PCIe all-reduce on a no-NVLink 8x A30 node. Disaggregated prefill/decode
(NVIDIA Dynamo / Mooncake-style) is included with its boundary condition stated: without an
RDMA/GPUDirect path the KV transfer traverses host memory and PCIe and can dominate TTFT, so it is
not a free memory fix here. Every batch item carries guardrail metrics (p99 TTFT, inter-token
latency, throughput, 429 rate, truncation rate) and explicit rollback thresholds.

Status: these records are **provisional** teacher-B output produced by a general-purpose model in a
blind lane. They are not expert gold, have not been validated by a domain expert, and say nothing
about any trained model's domain capability. No teacher-A artifact was read while producing them.

## Run 2026-08-17 batch 0134

- Batch file: results/train-batch-0134.jsonl
- Corpus range: train.jsonl lines 1331-1340 (0-indexed 1330..1339)
- Source IDs: corpus-01474, corpus-01475, corpus-01476, corpus-01477, corpus-01478,
  corpus-01479, corpus-01480, corpus-01481, corpus-01482, corpus-01483
- Progress: train 1340/5399, validation 0/601, total 1340/6000, remaining 4660
- Decisions: keep=0, rewrite=10, reject=0
- Initial schema check: PASS (ad-hoc verifier /tmp/tb_verify.py — 10 lines parse as JSONL, all 12
  required fields present, teacher_lane/teacher_model/calibration_status/decision values correct,
  source_user and source_assistant byte-identical to corpus, corrected_answer non-empty, confidence
  in [0,1], quality_dimensions are 1-5 integers, risks/evidence_required are arrays, source_id
  globally unique across all 134 batches, aggregated train sequence is a strict prefix of train.jsonl)
- Repairs applied: none required this run.
- Final schema check: PASS (train 1340, validation 0, total 1340)
- Manifest: MANIFEST.sha256 regenerated over 228 files; `sha256sum -c` all OK.

### Technical topics covered by this batch

All ten items are scenario variants 174-183 of the same prompt family: intermittent OOM in a
long-context LLM serving workload under concurrency, with an explicit requirement for a falsifiable
hypothesis and a controlled experiment. The source assistant text is an identical grading rubric for
all ten, so all ten were marked `rewrite`: a rubric enumerates topics but supplies no mechanism, no
arithmetic, no thresholds and no rollback gate.

To avoid ten near-duplicate answers, each item was given a distinct primary analytical angle, each
with its own hypothesis, experiment design and dominant mitigation:

1. corpus-01474 — admission control and the concurrency ceiling as the binding constraint.
2. corpus-01475 — allocator external fragmentation vs true capacity exhaustion (largest-free-block test).
3. corpus-01476 — first-principles KV bytes-per-token arithmetic and GQA/MQA sizing errors.
4. corpus-01477 — prefix/prompt cache retention as a pinned, non-evicting consumer.
5. corpus-01478 — long-tail input length distribution and rare co-arrival of p99 requests.
6. corpus-01479 — transient prefill activation peak vs steady KV growth; chunked prefill.
7. corpus-01480 — co-resident processes, MIG/MPS neighbours and NCCL buffers stealing HBM.
8. corpus-01481 — KV quantization as a capacity lever with a pre-registered accuracy margin.
9. corpus-01482 — preemption / recompute / CPU swap as graceful degradation instead of hard OOM.
10. corpus-01483 — tensor-parallel rank imbalance from non-sharded state (vocab-parallel logits).

Shared across all ten: the KV demand equation, declared hardware assumptions (8x A30 24 GB, PCIe
Gen4, no NVLink), confounder control (interleaved A/B arms, fixed engine/driver/model revision, SM
clock logging because A30 down-clocks under sustained load), boundary conditions where the analysis
does not hold (host-RAM OOM, speculative decoding/beam search multiplying resident tokens, late block
free on client cancellation), and quantitative rollback gates (p99 TTFT +20%, throughput -10%, >1%
silent rejection, breach of the long-context non-inferiority margin).

**Status caveat:** these outputs are PROVISIONAL teacher-B review artifacts produced blind (teacher-A
outputs were not read at any point during this run). They are NOT expert gold labels, have not been
validated against a real deployment, and say nothing about any model's domain capability.

## Run 2026-08-17 batch 0133

- Batch file: results/train-batch-0133.jsonl
- Corpus range: train.jsonl lines 1321-1330 (0-indexed 1320..1329)
- Source IDs: corpus-01464, corpus-01465, corpus-01466, corpus-01467, corpus-01468,
  corpus-01469, corpus-01470, corpus-01471, corpus-01472, corpus-01473
- Progress: train 1330/5399, validation 0/601, total 1330/6000, remaining 4670
- Decisions: keep=0, rewrite=10, reject=0
- Initial schema check: PASS (ad-hoc verifier — 10 lines parse as JSONL, all 12 required fields present,
  teacher_lane/teacher_model/calibration_status/decision values correct, source_user and source_assistant
  byte-identical to corpus, corrected_answer non-empty, confidence in [0,1], quality_dimensions are
  1-5 integers, risks/evidence_required are string arrays, source_id globally unique across all batches,
  aggregated train sequence is a strict prefix of train.jsonl)
- Repair actions: none required this run (verification passed on first execution)
- Final schema check: PASS (train=1330/5399, validation=0/601, ERRORS 0)
- Manifest: MANIFEST.sha256 regenerated over 227 files, sha256sum -c --quiet → all OK, 0 mismatch
- Technical topics covered: intermittent OOM in long-context LLM serving under concurrency,
  scenario variants 164-173. Each rewrite shares a common capacity spine — per-token KV bytes
  (2 * num_layers * num_kv_heads * head_dim * dtype_bytes / TP) and max_concurrent_tokens, with the
  binding quantity being the in-flight sum of (prompt + generated) tokens rather than request count —
  but carries a distinct secondary falsifiable hypothesis H2 so the batch is not a template clone:
  caching-allocator fragmentation (reserved-minus-allocated plus alloc-retry signature,
  expandable_segments remediation); max_model_len / max_num_batched_tokens over-provisioning against the
  empirical p99.9 prompt-length CDF; preemption and recompute storms correlated at 1 s resolution;
  prefix-cache retention and eviction bounding; multi-tenant interference and per-tenant token budgets;
  weight and KV quantization trading HBM against a gated accuracy band; NCCL communicator and collective
  staging buffer overhead shrinking the pool at higher TP degree; CUDA-graph capture workspace invisible
  to naive weights+KV accounting; bursty arrival where p99.9/mean in-flight tokens exceeds 2 despite safe
  mean utilisation; and explicit leak-versus-saturation discrimination via a monotonically rising idle
  memory floor over >=6 idle points. Every item states a controlled single-variable replay experiment
  (>=3 repetitions, >=30 min per arm, OOM per 10k requests as primary metric), named confounders
  (autoscaling, warm cache asymmetry, client retries, GPU co-tenancy) and quantitative rollback gates
  (p99 TTFT >10%, ITL >15%, 5xx above baseline, quality outside pre-agreed band). Platform-specific
  constants are deliberately not asserted; they are listed as evidence to be measured.
- Status: these teacher-B judgements are PROVISIONAL. They are an independent blind second opinion
  produced without any access to teacher-A outputs. They are NOT expert gold labels, have not been
  validated against hardware, and say nothing about any model's domain capability.

## Run 2026-08-17 batch 0132

- Batch file: results/train-batch-0132.jsonl
- Corpus range: train.jsonl lines 1311-1320 (0-indexed 1310..1319)
- Source IDs: corpus-01452, corpus-01453, corpus-01454, corpus-01456, corpus-01457,
  corpus-01458, corpus-01460, corpus-01461, corpus-01462, corpus-01463
- Progress: train 1320/5399, validation 0/601, total 1320/6000, remaining 4680
- Decisions: keep=0, rewrite=10, reject=0
- Initial schema check: PASS (verify_batches.py — 10 lines, 12 required fields, lane/model/status/decision
  values correct, source_user and source_assistant byte-identical to corpus, corrected_answer non-empty,
  confidence in [0,1], source_id globally unique, train sequence a strict prefix of train.jsonl)
- Repair actions: none required this run
- Final schema check: PASS (train=1320/5399 validation=0/601 total=1320, VERIFY_PASS)
- Manifest: MANIFEST.sha256 regenerated over 221 files, sha256sum -c → 221 OK, 0 mismatch
- Technical topics covered: intermittent OOM in long-context LLM serving under concurrency.
  All 10 items are scenario variants 152-163 of the same long-context OOM prompt, split across
  Troubleshooting / Performance Analysis / System Design categories. Rewrites give a per-token KV
  capacity model (2 * layers * kv_heads * head_dim * dtype_bytes / TP) and the max_concurrent_tokens
  derivation; a category-specific falsifiable hypothesis (admission oversubscription vs caching-allocator
  fragmentation via reserved-minus-allocated vs missing length-aware admission contract); separation of
  device OOM from host cgroup OOM kill; prefill activation/logits workspace spikes as a non-KV allocation
  source; a 4-arm single-variable replay experiment (baseline / token budget / expandable_segments /
  clamped max_model_len) with OOM-per-10k as primary metric and p99 TTFT, ITL, throughput, 429 rate as
  guardrails; confounders (429s masking dropped traffic, synthetic prefix-cache inflation, warm-up,
  power/thermal capping on a dense 8x A30 chassis, co-resident processes); and explicit rollback gates.
  Interconnect boundary conditions are stated for the A30 PCIe-only topology: TP all-reduce cost per layer,
  and disaggregated prefill/decode (Dynamo / Mooncake style) requiring KV transfer that without
  RDMA/GPUDirect falls back to host memory over PCIe and can dominate TTFT.
- Status: PROVISIONAL. These are teacher-B second-opinion outputs from a blind review lane. They are
  NOT expert gold labels, have not been validated against ground truth or teacher-A, and say nothing
  about any model's domain capability. Agreement analysis with teacher-A is a separate later step.

## Run 2026-08-17 batch 0131

- Batch file: results/train-batch-0131.jsonl
- Corpus range: train.jsonl lines 1301-1310 (0-indexed 1300..1309)
- Source IDs: corpus-01441, corpus-01442, corpus-01443, corpus-01444, corpus-01445,
  corpus-01446, corpus-01447, corpus-01448, corpus-01449, corpus-01451
- Progress: train 1310/5399, validation 0/601, total 1310/6000, remaining 4690
- Decisions: keep=0, rewrite=10, reject=0
- Initial schema check: PASS (verify_batches.py — 10 lines, 12 required fields, lane/model/status/decision
  values correct, source_user and source_assistant byte-identical to corpus, corrected_answer non-empty,
  confidence in [0,1], source_id globally unique, train sequence is a strict prefix of train.jsonl)
- Repair actions: none required this run
- Final schema check: PASS (train=1310/5399 validation=0/601 total=1310, VERIFY_PASS)
- Manifest: MANIFEST.sha256 regenerated over 220 files; `sha256sum -c` all OK

Technical topics covered by this batch: intermittent out-of-memory failures in long-context
LLM serving under concurrency. Each rewritten answer grounds the diagnosis in an explicit
per-token KV capacity model (2 x n_layers x n_kv_heads x head_dim x dtype_bytes / TP), worked
against a 9B-class model on 24 GB A30 devices, and then advances one of four discriminating
falsifiable hypotheses — length-blind admission control, caching-allocator fragmentation,
KV block leakage on client disconnect/abort, and prefix-cache retention competing with live
requests — each paired with a controlled single-variable experiment on a replayed, byte-identical
production trace. Answers enumerate the required measurements (KV pool free-block time series,
prompt/max_tokens tail distribution, reserved-vs-allocated at failure, out-of-process nvidia-smi
cross-check, dmesg to exclude host OOM kill), the expected confounders (co-tenant HBM consumers,
CUDA graph and NCCL buffers outside framework accounting, warm-up allocations, retry storms),
a reversibility-ordered mitigation list, and explicit rollback thresholds. Tensor parallelism is
treated as a last resort with its PCIe-fabric all-reduce cost stated, since the assumed A30 node
has no NVLink; KV quantization is gated on a pre-registered accuracy check and barred from
in-incident use.

These results are PROVISIONAL model-generated review output. They are not expert gold labels,
they have not been validated by a human domain expert, and they say nothing about any model's
domain capability. They are a blind second opinion produced without any access to teacher-A
artifacts, intended solely as input to a later, separate agreement analysis.

## Run 2026-08-17 batch 0130

- Batch file: results/train-batch-0130.jsonl
- Corpus range: train.jsonl lines 1291-1300 (0-indexed 1290..1299)
- Source IDs: corpus-01430, corpus-01431, corpus-01433, corpus-01434, corpus-01435,
  corpus-01436, corpus-01437, corpus-01438, corpus-01439, corpus-01440
  (note: corpus-01432 does not appear in train.jsonl; the sequence is copied in
  exact corpus order, nothing skipped or reordered)
- Progress: train 1300/5399, validation 0/601, total 1300/6000, remaining 4700
- Decisions: keep=0, rewrite=10, reject=0
- Initial schema check: PASS (ad-hoc verifier /tmp/tb_verify.py, 0 errors across all
  1300 aggregated records; checks JSONL line parse, batch count, 12 required fields,
  fixed-value fields, source_user/source_assistant byte-equality against
  research/ai-infra-expert/corpus/train.jsonl, non-empty corrected_answer,
  confidence in [0,1], global source_id uniqueness, and strict corpus-prefix ordering)
- Repairs: none required this run; the batch verified clean on first write.
- Final schema check: PASS (0 errors)
- Manifest: MANIFEST.sha256 regenerated over all 218 files in this directory
  (excluding the manifest itself); `sha256sum -c` reports all OK.

### Technical topics covered by this batch

All ten records are variants of the same scenario family: a long-context serving
workload that intermittently OOMs after several concurrent requests, each asking for
a prioritized diagnosis plus an explicit falsifiable hypothesis and controlled
experiment. Every source `assistant` field is a grading rubric rather than an answer,
so all ten were marked `rewrite`.

To avoid ten near-identical rewrites, each record was given a distinct primary
hypothesis and a matching discriminating experiment: (1) length-blind admission
control, (2) allocator fragmentation from variable-length activation buffers,
(3) prefix-cache eviction thrash, (4) chunked prefill disabled so one long prompt
allocates a single huge activation, (5) CUDA-graph capture pools reserved per batch
shape, (6) mis-estimated per-token KV bytes from wrong kv_head count or dtype,
(7) host pinned-memory / CPU-offload pressure with the host OOM-killer as the real
terminator, (8) GPU co-tenancy from a stale worker or exporter holding a context,
(9) max_model_len arithmetically incompatible with the KV pool at any concurrency,
and (10) speculative-decoding draft-model memory. Shared across all ten: exact
per-token KV arithmetic (2 x layers x kv_heads x head_dim x bytes_per_elem, with GQA/MQA
called out as an order-of-magnitude factor), the reserved-vs-allocated gap as the
discriminator between fragmentation and genuine capacity exhaustion, confounders
(cold vs warm prefix cache, client retry amplification, mid-window autoscaling), and
mitigations ordered by reversibility with quantization gated behind a task-level eval
rather than an OOM fix alone.

- Status: PROVISIONAL. These are second-opinion reviews from a single model under
  blind conditions. They are not expert gold labels, have not been validated by a
  human domain expert, and say nothing about any model's domain capability. No
  teacher-A artifact was read, opened, or grepped while producing this batch.

## Run 2026-08-17 batch 0129

- Batch file: results/train-batch-0129.jsonl
- Corpus range: train.jsonl lines 1281-1290 (0-indexed 1280..1289)
- Source IDs: corpus-01419, corpus-01420, corpus-01421, corpus-01422, corpus-01424,
  corpus-01425, corpus-01426, corpus-01427, corpus-01428, corpus-01429
  (note: corpus-01423 does not exist in the corpus; the sequence is preserved
  exactly as it appears in train.jsonl, nothing was skipped or reordered)
- Progress: train 1290/5399, validation 0/601, total 1290/6000, remaining 4710
- Decisions: keep=0, rewrite=10, reject=0
- Initial schema check: PASS (ad-hoc verifier /tmp/tb_verify.py, 0 errors across all
  1290 aggregated records)
- Repairs: one generator-side fix before any output was accepted. The first build ran
  with an off-by-one corpus slice (rows[1290:1300], i.e. corpus-01430..01440) which
  would have left a 10-record gap and broken the strict-prefix property. This was
  caught before verification, the slice was corrected to rows[1280:1290], and the
  batch file was regenerated in place. No previously committed batch and no original
  corpus file was touched.
- Final schema check: PASS (JSONL line-parse, 10 records in batch, all 12 required
  fields, teacher_lane/teacher_model/calibration_status/decision values correct,
  source_user and source_assistant byte-identical to corpus, corrected_answer
  non-empty, confidence in [0,1], quality_dimensions integers in 1..5, source_id
  globally unique, train sequence a strict prefix of train.jsonl)
- Manifest: MANIFEST.sha256 regenerated over 217 files (all files in this directory
  except the manifest itself); `sha256sum -c` reports all OK.
- Blind-review compliance: no file under experiments/2026-08-14-teacher-a-corpus-calibration/
  was read, opened, or searched at any point during this run.
- Lock: /tmp/teacher-b-corpus-review.lock acquired atomically at run start (no
  pre-existing lock, no stale-lock cleanup needed); released at run end.

### Technical topics covered by this batch

All ten items are variants of the same scenario family: a long-context serving
workload that intermittently hits OOM only after several concurrent requests, split
across the Troubleshooting, Performance Analysis, and System Design categories. The
rewritten answers cover: separating device-side `torch.cuda.OutOfMemoryError` from a
host cgroup OOM-kill as the mandatory first discriminator; the per-token KV-cache
cost model (2 * layers * kv_heads * head_dim * dtype_bytes / TP) and why a heavy-tailed
prompt-length distribution makes admission a worst-case rather than average-case
problem; paged KV versus caching-allocator fragmentation for non-KV tensors, with
logprobs/top_logprobs materialisation called out as a concrete blow-up path; startup
KV-pool sizing via `gpu_memory_utilization` and the ways that assumption is silently
violated later (co-tenants, MIG/MPS neighbours, ECC page retirement). Each answer
states a falsifiable hypothesis H1 (length-blind admission) against a competing H0
(fragmentation / non-KV spike), and a single controlled replay experiment that
discriminates them, with pre-registered success criteria and an explicit confounder
list (co-tenancy, warm-up and CUDA-graph capture, prefix-cache hit rate, page
retirement, clock throttling). Mitigations are ordered P0..P3 with rollback cost:
server-enforced `max_model_len` and mandatory server-side `max_tokens`, reservation-based
admission, prefix caching gated on measured hit rate, fp8 KV quantization gated on a
non-inferiority quality eval, `expandable_segments:True` as an H0-specific treatment,
and finally prefill/decode disaggregation (NVIDIA Dynamo-style routing with a
Mooncake-class KV store over RDMA/RoCE and GPUDirect RDMA) as the structural fix,
including the requirement to verify GDR is actually engaged rather than silently
falling back to a host-memory bounce. The Performance Analysis variants additionally
frame the problem as a hard capacity inequality with an explicit token-in-flight
roofline, and use "was kv_cache_usage near 1.0 at failure time?" as the fastest
available discriminator between H1 and H0.

Source-side assessment: every source_assistant in this batch is a grading rubric
("Answer should state assumptions, ... Minimum technical points: ...") rather than an
answer. Training on that text would teach meta-commentary about answers instead of
engineering reasoning, which is why all ten are marked `rewrite` rather than `keep`.

**Status caveat**: these teacher-B outputs are PROVISIONAL. They are one model's
independent blind review, not expert gold labels, not validated against measured
system behaviour, and they say nothing about any trained model's domain capability.
Agreement analysis against teacher-A is a separate, later step and was deliberately
not performed here.

## Run 2026-08-17 batch 0128

- Batch file: results/train-batch-0128.jsonl
- Corpus range: train.jsonl lines 1271-1280 (0-indexed 1270..1279)
- Source IDs: corpus-01409, corpus-01410, corpus-01411, corpus-01412, corpus-01413,
  corpus-01414, corpus-01415, corpus-01416, corpus-01417, corpus-01418
- Progress: train 1280/5399, validation 0/601, total 1280/6000, remaining 4720
- Decisions: keep=0, rewrite=10, reject=0
- Initial schema check: PASS (ad-hoc verifier, first run, no repairs needed)
- Repairs: none. The generator and verifier scripts were kept under /tmp so they do
  not enter the experiment directory or the manifest.
- Lock: /tmp/teacher-b-corpus-review.lock acquired atomically at run start (no
  pre-existing lock, no stale-lock cleanup needed); released at run end.
- Final schema check: PASS (train 1280, validation 0, total 1280, unique ids 1280,
  strict-prefix check against train.jsonl OK, source_user/source_assistant
  byte-identical to corpus, 0 errors)
- Manifest: MANIFEST.sha256 regenerated over 216 files; `sha256sum -c` all OK
- Topics covered: long-context intermittent OOM under concurrent LLM serving,
  scenario variants 109-118 of the same base prompt, rotating across System Design,
  Troubleshooting and Performance Analysis. All ten source assistant fields are
  grading rubrics ("Answer should state assumptions...") rather than answers, so all
  ten are rewrites. Each rewrite gives the device-memory decomposition (weights +
  activation peak + KV pool + fragmentation + CUDA/NCCL context), the KV-bytes-per-token
  formula with the GQA num_key_value_heads correction, the distinction between device
  OOM and host oom-killer and between prefill activation spike and KV pool exhaustion,
  five ranked falsifiable hypotheses (capacity tail, leak, fragmentation, prefill spike,
  co-tenant) each with an explicit kill criterion, a fixed-trace replay experiment with
  controls and >=3 runs per arm, a nine-step reversible mitigation ladder (token-budget
  admission control, max_model_len/max_tokens ceilings, chunked prefill, lowering rather
  than raising gpu_memory_utilization, prefix caching, expandable segments, KV
  quantization gated on a quality eval, higher TP with its A30 PCIe-vs-NVLink cost, and
  prefill/decode disaggregation Dynamo/Mooncake-style as a last resort), and canary
  rollback gates on OOM count, TTFT p95, TPOT p95, throughput and preemption rate.
  Category-specific framing was added per record (capacity contract for System Design,
  a timed triage order for Troubleshooting, explicit pool arithmetic and
  tok/s-at-fixed-SLO reporting for Performance Analysis).
- Status: these outputs are PROVISIONAL teacher-B review artifacts. They are not
  expert gold, they have not been validated against any real incident or measurement,
  and they say nothing about any model's domain capability.

## Run 2026-08-17 batch 0127

- Batch file: results/train-batch-0127.jsonl
- Corpus range: train.jsonl lines 1261-1270 (0-indexed 1260..1269)
- Source IDs: corpus-01395, corpus-01396, corpus-01397, corpus-01398, corpus-01399,
  corpus-01400, corpus-01401, corpus-01403, corpus-01404, corpus-01408
- Progress: train 1270/5399, validation 0/601, total 1270/6000, remaining 4730
- Decisions: keep=0, rewrite=10, reject=0
- Initial schema check: PASS (verify_batches.py, first run, no repairs needed)
- Repairs: none. Housekeeping only: two temporary helper scripts (tmp_dump.py,
  tmp_gen.py) used to inspect the corpus slice and emit the batch were deleted
  before manifest regeneration so they do not enter the manifest.
- Lock: previous run's lock (/tmp/teacher-b-corpus-review.lock, owner
  cron-1787007193) was older than the 10-minute staleness threshold
  (age ~45785 s); stale lock cleanup was recorded, the lock removed, and a fresh
  lock acquired for this run.
- Final schema check: PASS (train 1270, validation 0, total 1270, unique_ids 1270,
  strict-prefix check against train.jsonl OK, VERIFY_PASS)
- Manifest: MANIFEST.sha256 regenerated; `sha256sum -c` all OK
- Topics covered: long-context intermittent OOM under concurrent serving,
  scenario variants 95-108 of the same base prompt, split across Troubleshooting,
  Performance Analysis and System Design. The source assistant text for all ten is
  a grading rubric ("Answer should state assumptions...") rather than an answer,
  which is why every item is a rewrite: training on a rubric teaches meta-commentary
  about answers instead of the domain reasoning itself. Rewrites give the device
  memory budget as an explicit sum (weights + activation/workspace peak + KV pool +
  fragmentation + CUDA/NCCL context), the KV bytes-per-token formula
  2 * n_layers * n_kv_heads * head_dim * dtype_bytes / TP with an explicit warning
  that using attention heads instead of KV heads under GQA is the most common
  sizing error, and the capacity inequality max_t sum(prompt_len + max_tokens) <= T.
  Three competing falsifiable hypotheses are separated by distinct predictions:
  H1 co-arrival KV pool exhaustion (falsified by a flat long-soak memory trend),
  H2 leak / unbounded non-paged cache (monotonic growth that never returns to
  baseline), H3 single pathological max-length prefill (falsified by replaying it
  alone at concurrency 1). Discriminating measurements include >=1 Hz engine
  metrics (aliasing warning for 1-minute Prometheus scrapes), reserved-vs-allocated
  gap as the fragmentation fingerprint, per-process nvidia-smi to exclude a
  co-tenant, and device-OOM vs host OOM-killer classification via dmesg. The
  controlled experiment fixes the replay trace and sweeps max_num_seqs /
  max_num_batched_tokens, with confounders named explicitly (warmup, CUDA graph
  capture, prefix-cache hit differences from trace reordering, and the memory cost
  of memory-history recording itself). Mitigations are ordered by reversibility:
  P0 chunked prefill, concurrency caps, max_model_len, expandable_segments;
  P1 token-aware admission control and gateway-enforced max_tokens ceilings;
  P2 FP8/INT8 KV, higher TP to shard KV, and prefill/decode disaggregation in the
  NVIDIA Dynamo / Mooncake style with the KV transfer costed against RDMA/RoCE NIC
  bandwidth, GPUDirect RDMA PCIe affinity and lossless-fabric prerequisites.
  Raising gpu_memory_utilization is called out as an anti-first-move. Rollback
  gates are numeric (p99 TTFT +20%, throughput -10%, any OOM, preemption >1% over a
  30-minute canary) and KV quantization additionally requires an offline quality
  gate before shipping.
- Status: PROVISIONAL. These are single-model teacher-B opinions produced blind
  (no teacher-A artifact was read while generating this batch). They are not expert
  gold labels, have not been human-verified, and say nothing about any model's
  domain capability.

## Run 2026-08-17 batch 0125

- Batch file: results/train-batch-0125.jsonl
- Corpus range: train.jsonl lines 1241-1250 (0-indexed 1240..1249)
- Source IDs: corpus-01373, corpus-01374, corpus-01375, corpus-01376, corpus-01377,
  corpus-01378, corpus-01379, corpus-01380, corpus-01381, corpus-01382
- Progress: train 1250/5399, validation 0/601, total 1250/6000, remaining 4750
- Decisions: keep=0, rewrite=10, reject=0
- Initial schema check: PASS (verify_batches.py, first run, no fixes needed)
- Repairs: none
- Final schema check: PASS (train 1250 records, validation 0, TOTAL 1250, VERIFY=PASS)
- Manifest: MANIFEST.sha256 regenerated over 211 files; `sha256sum -c` 211/211 OK
- Topics covered: long-context serving OOM under concurrency. All ten items are
  scenario variants 73-82 of the same base prompt, split across System Design,
  Troubleshooting and Performance Analysis. Rewrites cover the KV-cache memory
  model (2 * n_layers * n_kv_heads * head_dim * dtype_bytes * sum(seq_len)),
  paged-KV block granularity, prefill transient workspace as an unbudgeted term,
  admission overcommit (H1) vs allocator fragmentation (H2) as competing
  falsifiable hypotheses discriminated by the reserved-minus-allocated gap, a
  concurrency sweep 1-16 with repeats to locate first-OOM concurrency C*, and an
  A/B arm design (concurrency cap vs expandable_segments). Mitigations are ordered
  by reversibility: admission control, chunked prefill, prefix caching,
  expandable_segments, gpu_memory_utilization headroom, KV fp8 quantization,
  tensor parallelism. Explicit safety warning against raising
  gpu_memory_utilization toward 1.0, and against multi-knob changes that destroy
  attribution. Rollback gate: zero OOM over a 30-minute replay, p95 regression
  <10%, throughput regression <15%, preemption/recompute <2x baseline.
- Source answers in this range are rubric checklists rather than answers, so all
  ten were marked `rewrite` with instruction_coverage=2.

STATUS CAVEAT: these outputs are PROVISIONAL teacher-B model review. They are NOT
expert gold labels, have not been validated by a human domain expert, and say
nothing about any model's domain capability.

## Run 2026-08-17 batch 0124

- Batch file: results/train-batch-0124.jsonl
- Corpus range: train.jsonl lines 1231-1240 (0-indexed 1230..1239)
- Source IDs: corpus-01362, corpus-01363, corpus-01364, corpus-01366, corpus-01367,
  corpus-01368, corpus-01369, corpus-01370, corpus-01371, corpus-01372
- Progress: train 1240/5399, validation 0/601, total 1240/6000, remaining 4760
- Decisions: keep=0, rewrite=10, reject=0
- Initial schema check: PASS (1240 aggregate records over 124 batch files, 12 required
  fields per record, teacher_lane/teacher_model/calibration_status/decision values
  correct, source_user and source_assistant byte-identical to corpus, corrected_answer
  non-empty, confidence in [0,1], quality_dimensions integers in 1-5, risks and
  evidence_required string arrays, source_id globally unique, aggregate train sequence
  a strict prefix of train.jsonl, validation sequence empty)
- Repairs: none required; the first verification run passed. Note: this run used a
  freshly written verify_run.py because cron sessions block execute_code; the batch
  generator and verifier were written to disk and executed via terminal.
- Final schema check: PASS (same run as initial)
- Manifest: MANIFEST.sha256 regenerated over 209 files; `sha256sum -c` all OK
- Technical topics covered: long-context intermittent-OOM family, scenario variants
  62-72. The ten rewrites deliberately attack the problem from ten distinct angles
  rather than restating one template: (62) steady-state KV capacity with the explicit
  per-rank kv_bytes_per_token formula and a threshold-invariance hypothesis; (63) the
  reservation-vs-occupancy scheduling bug, tested by clamping max_new_tokens under
  closed-loop concurrency; (64) capacity-versus-leak discrimination via a two-hour
  flat-memory soak; (66) term-scaling analysis separating O(batch*seq) KV from
  O(batch*seq) fused-attention activations, with a superlinear knee as the refutation
  signal for a kernel fallback; (67) incident-response ordering where the failing
  allocation size in the OOM traceback discriminates fragmentation from capacity;
  (68) per-rank TP asymmetry from kv_heads not dividing tp_degree, plus NCCL registered
  buffers outside the framework allocator, with a failing-rank histogram as the test;
  (69) prefill/decode interference and disaggregated serving (Dynamo/Mooncake-style),
  including RDMA staging and pinned-host accounting and an explicit warning not to adopt
  disaggregation for a problem admission control solves; (70) a cause-to-measurement
  map that orders cheap tests before expensive ones; (71) cost/SLO framing of the three
  levers (reduce demand, reduce footprint, add supply) with retry-amplification as a
  named confounder; (72) an epistemic answer that refuses to name a root cause from the
  reported symptom and instead pre-registers a classification rule over 20 captured
  failures. Every record states assumptions, a falsifiable hypothesis with its explicit
  refutation condition, a one-variable controlled experiment, expected confounders,
  required evidence and rollback criteria with numeric thresholds.
- Status: PROVISIONAL. These are second-opinion reviewer outputs from a blind lane,
  produced without any access to teacher-A artifacts, and they are NOT expert gold
  labels. They say nothing about any model's domain capability.

## Run 2026-08-17 batch 0123

- Batch file: results/train-batch-0123.jsonl
- Corpus range: train.jsonl lines 1221-1230 (0-indexed 1220..1229)
- Source IDs: corpus-01350, corpus-01351, corpus-01352, corpus-01353, corpus-01355,
  corpus-01356, corpus-01357, corpus-01358, corpus-01359, corpus-01361
- Progress: train 1230/5399, validation 0/601, total 1230/6000, remaining 4770
- Decisions: keep=0, rewrite=10, reject=0
- Initial schema check: PASS (1230 aggregate records, 12 required fields per record,
  lane/model/status/decision values correct, source_user and source_assistant
  byte-identical to corpus, corrected_answer non-empty, confidence in [0,1],
  source_id globally unique, aggregate train sequence a strict prefix of train.jsonl,
  validation sequence empty)
- Repairs: none required; the first verification run passed.
- Final schema check: PASS (same run as initial)
- Manifest: MANIFEST.sha256 regenerated over 207 files; `sha256sum -c` all OK
- Technical topics covered: this block is the long-context intermittent-OOM family
  (scenario variants 50-61). Rewrites separate static weight memory from dynamic KV
  and activation memory; give the explicit KV byte formula with GQA kv_heads; treat
  prefill activation spikes, allocator fragmentation (reserved-vs-allocated gap) and
  per-rank TP/NCCL overhead as distinct mechanisms with distinct refutations. Design
  answers cover worst-case token-budget admission control with 429 shedding,
  short/long-context replica isolation and gateway routing, prefill/decode
  disaggregation (Mooncake-style shared KV pool, gated on measured interconnect
  bandwidth), fp8 KV and weight-quantization economics gated on long-context task
  evals rather than perplexity, 1s-resolution KV-pool-utilization telemetry with a
  falsifiable lead-time claim before enabling auto-shedding, and a written capacity
  model whose prediction error is itself the falsification test. Every item states
  assumptions, a falsifiable hypothesis with its refutation condition, a one-variable
  controlled experiment, expected confounders, required evidence and rollback
  criteria. Source assistant turns are rubric stubs, not answers, hence rewrite=10.
- Status: PROVISIONAL. These are second-opinion reviewer outputs from a blind lane,
  NOT expert gold labels, and they say nothing about any model's domain capability.

## Run 2026-08-17 batch 0122

- Batch file: results/train-batch-0122.jsonl
- Corpus range: train.jsonl lines 1211-1220 (0-indexed 1210..1219)
- Source IDs: corpus-01339, corpus-01341, corpus-01342, corpus-01343, corpus-01344,
  corpus-01345, corpus-01346, corpus-01347, corpus-01348, corpus-01349
- Progress: train 1220/5399, validation 0/601, total 1220/6000, remaining 4780
- Decisions: keep=0, rewrite=10, reject=0
- Initial schema check: PASS (1220 aggregate records, 12 required fields per record,
  lane/model/status/decision values correct, source_user and source_assistant
  byte-identical to corpus, corrected_answer non-empty, confidence in [0,1],
  source_id globally unique, aggregate train sequence a strict prefix of train.jsonl,
  validation sequence empty)
- Repair actions: none required for the batch itself. Lifecycle note: a stale lock
  (age ~11 minutes, above the 10-minute threshold) from a previous run was recorded
  and removed before this run acquired the lock atomically.
- Final schema check: PASS
- Manifest: MANIFEST.sha256 regenerated over all files in this experiment directory
  (excluding the manifest itself and __pycache__); `sha256sum -c` reported all OK.

### Technical themes covered by this batch

All ten items are variants (scenario 39, 41-49) of one seed: a long-context LLM
serving deployment that intermittently OOMs after several concurrent requests, with
the instruction explicitly demanding a falsifiable hypothesis and a controlled
experiment. The source assistant text is identical across all ten and is a grading
rubric rather than an answer, so every item was marked rewrite.

The ten rewrites deliberately attack the scenario from different angles so the batch
is not ten paraphrases: memory-headroom regression and sweep design; incident triage
with a device-OOM vs host-OOM-kill split; allocator fragmentation as the binding
constraint (allocated vs reserved gap, prompt-length variance); admission contract as
a design defect (L, C, P bounds making the worst case computable); intermittency as a
state-variable/uptime question with a restart-based discriminator; quantitative memory
budgeting with an explicit residual and unmodelled-term detection; degradation policy
(queue vs reject vs preempt) under overload; the false-confirmation trap of lowering
concurrency when the KV pool was never saturated; cost-of-fix accounting (headroom
gained per unit goodput lost, with a quality gate on KV quantization); and failure
isolation, where a single rank's OOM aborts an entire tensor-parallel collective so
headroom requirements scale with parallel degree.

Recurring technical content: the KV closed form
2 * layers * kv_heads * head_dim * dtype_bytes * resident_tokens under GQA/MQA; the
boundary condition that a preallocated paged KV pool should preempt or queue rather
than raise CUDA OOM, implying out-of-pool consumers (prefill activation spikes, logits
buffers scaling with batch x vocab, LoRA adapters, speculative draft state, allocator
fragmentation); chunked prefill, expandable segments, length bucketing, prefix caching
and KV quantization as mitigations ordered by reversibility; and multi-node topics
including NCCL comm buffers, RDMA/RoCE registered memory, PFC/ECN misconfiguration
coupling into latency and in-flight state, Dynamo-style prefill/decode disaggregation
and Mooncake-style hierarchical KV offload as structural options with named costs.

Every answer carries explicit assumptions, a numbered falsifiable hypothesis with its
refutation condition, a controlled experiment with repeats and randomized ordering,
named confounders, required evidence, and rollback gates.

**Status: these teacher-B outputs are PROVISIONAL.** They are one model's independent
blind second opinion, not expert gold labels, and they were produced without any
access to the teacher-A lane. They say nothing about any trained model's domain
capability; agreement analysis against teacher-A is a separate, later step.

## Run 2026-08-17 batch 0120

- Batch file: results/train-batch-0120.jsonl
- Corpus range: train.jsonl lines 1191-1200 (0-indexed 1190..1199)
- Source IDs: corpus-01317, corpus-01318, corpus-01319, corpus-01320, corpus-01321,
  corpus-01322, corpus-01323, corpus-01324, corpus-01326, corpus-01327
- Progress: train 1200/5399, validation 0/601, total 1200/6000, remaining 4800
- Decisions: keep=0, rewrite=10, reject=0
- Initial schema check: PASS (1200 aggregate records, exactly 12 required fields per
  record, lane/model/status/decision values correct, source_user and source_assistant
  byte-identical to corpus, corrected_answer non-empty, confidence in [0,1],
  source_id globally unique, aggregate train sequence is a strict prefix of
  train.jsonl, validation sequence empty)
- Repair actions: none required; the batch passed on first verification run.
- Final schema check: PASS
- Manifest: MANIFEST.sha256 regenerated over all 208 files in this experiment
  directory (excluding the manifest itself); `sha256sum -c` reported all OK.

### Technical themes covered by this batch

All ten items are variants of the same seed scenario: a long-context LLM serving
deployment that hits intermittent OOM only after several concurrent requests, with the
instruction requiring an explicit falsifiable hypothesis and a controlled experiment.
The source assistant fields are rubric stubs ("answer should state ..."), not answers,
so every item was marked `rewrite`.

Each corrected_answer shares a common mechanism derivation — GPU memory decomposed into
weights, CUDA/NCCL context, activation peak of the scheduled batch, and the paged KV
pool, with the KV byte formula made explicit and GQA's effect on kv_heads noted — and
then diverges into a distinct primary hypothesis so the batch spans the real diagnostic
space rather than repeating one answer ten times:

- 01317: concurrent long-prefill activation peaks vs KV exhaustion (chunked prefill arm)
- 01318: KV pool sized from an under-concurrency profiling pass (re-profile arm)
- 01319: unbounded client-supplied context length, i.e. no capacity model
- 01320: caching-allocator fragmentation (largest-contiguous-free-block discriminator)
- 01321: CUDA graph capture pools growing per newly observed batch shape
- 01322: loss of device-level memory isolation / colocated ranks and NCCL buffers
- 01323: preemption-with-recompute path itself allocating the failing peak
- 01324: logits/sampling tensor scaling as batch*vocab (many-short vs few-long arm)
- 01326: uptime-dependent leak outside the KV pool (fresh vs aged replica arm)
- 01327: memory-unaware request routing producing per-replica occupancy imbalance

Every answer carries mitigations ordered by cost and reversibility (admission control
and context caps before pool resizing before chunked prefill before quantization),
named confounders, the evidence artifacts that must exist before a permanent change,
and an explicit rollback gate (24h canary, zero OOM, <10% p99 TTFT/TPOT regression,
preemption rate <2%, unchanged frozen-eval quality for any numerics change).

**Status caveat:** these outputs are *provisional* teacher-B review artifacts produced
blind (teacher-A outputs were not read while generating this batch). They are not
expert gold labels, they have not been validated against hardware, and they are not
evidence of any model's domain capability. Agreement analysis against teacher-A is a
separate, later step outside this worker's scope.

## Run 2026-08-17 batch 0119

- Batch file: results/train-batch-0119.jsonl
- Corpus range: train.jsonl lines 1181-1190 (0-indexed 1180..1189)
- Source IDs: corpus-01305, corpus-01306, corpus-01307, corpus-01308, corpus-01309,
  corpus-01310, corpus-01311, corpus-01312, corpus-01313, corpus-01314
- Progress: train 1190/5399, validation 0/601, total 1190/6000, remaining 4810
- Decisions: keep=0, rewrite=10, reject=0
- Initial schema check: PASS (1190 aggregate records, exactly 12 fields per record,
  lane/model/status/decision values correct, source_user and source_assistant
  byte-identical to corpus, corrected_answer non-empty, confidence in [0,1],
  source_id globally unique, aggregate train sequence is a strict prefix of
  train.jsonl, validation sequence empty)
- Repairs performed: none. First verification pass succeeded; no original corpus,
  no prior batch file, and no teacher-A artifact was read or modified.
- Final schema check: PASS (identical to initial run)
- Manifest: MANIFEST.sha256 regenerated over all 207 files in the experiment
  directory except the manifest itself; `sha256sum -c` returned all-OK.

### Technical topics covered by this batch

All ten records are the same scenario family (long-context serving hits intermittent
OOM after several concurrent requests, variants 5-14), split across three category
lenses: Troubleshooting, Performance Analysis, and System Design. Every source
assistant turn is a grading rubric rather than an answer, so all ten were rated
technical_correctness=3 / instruction_coverage=2 / operational_safety=2 and marked
`rewrite`.

The rewritten answers make the mechanism explicit rather than listing knobs:
memory is partitioned into weights / KV / prefill activation working set /
allocator fragmentation / non-engine residents (NCCL buffers, CUDA graph pools),
with the KV formula stated as 2 x layers x num_kv_heads x head_dim x dtype_bytes x
cached_tokens and an explicit warning that using num_attention_heads instead of the
GQA-corrected num_kv_heads overestimates KV by the GQA ratio. Six prioritized
hypotheses are ranked (aggregate KV exhaustion, prefill spike, fragmentation,
missing admission control, non-engine residents, genuine capacity shortfall), each
with a distinguishing signature — notably reserved-minus-allocated as the
fragmentation discriminator, and preemption count as the leading indicator that
precedes OOM.

The falsifiable hypothesis H1 (failure is driven by aggregate live KV crossing the
analytically computed budget, not by fragmentation) carries quantitative
predictions (+/-10% boundary prediction, <10% vs >20% reserved-allocated gap) and an
order-permutation control arm A5 that holds total tokens fixed while shuffling
arrival order — the cheapest single experiment that separates KV exhaustion from
fragmentation. Six arms total (baseline, admission control, paged KV / prefix
caching, chunked prefill, KV quantization or reduced max_model_len, order
permutation), one variable each, >=3 repeats, frozen hashed trace, warmup discarded.
Mitigations are ordered by risk, ending with prefill/decode disaggregation
(Dynamo- / Mooncake-style) framed explicitly as an architecture change that moves KV
over RDMA/RoCE and buys new tail-latency and failure modes rather than as a knob.
Rollback gates are numeric and pre-registered: zero OOM/5xx, P99 TTFT regression
<= 20%, output throughput drop <= 15%, accuracy within pre-registered tolerance for
any quantization arm, canary first, two consecutive clean windows before fleet
rollout. Each category lens adds its own emphasis: incident ordering
(observe/bound/bisect, snapshot before restart) for Troubleshooting, the
memory-headroom-versus-load curve and goodput-at-fixed-SLO scoring for Performance
Analysis, and SLO-derived KV budgeting with pool isolation for System Design.

These outputs are provisional teacher-B review artifacts. They are NOT expert gold
labels, have NOT been validated by a human domain expert, and say nothing about any
model's domain capability.

## Run 2026-08-17 batch 0118

- Batch file: results/train-batch-0118.jsonl
- Corpus range: train.jsonl lines 1171-1180 (0-indexed 1170..1179)
- Source IDs: corpus-01294, corpus-01296, corpus-01297, corpus-01298, corpus-01299,
  corpus-01300, corpus-01301, corpus-01302, corpus-01303, corpus-01304
- Progress: train 1180/5399, validation 0/601, total 1180/6000, remaining 4820
- Decisions: keep=0, rewrite=10, reject=0
- Initial schema check: PASS (1180 aggregate records, exactly 12 fields per record,
  lane/model/status/decision values correct, source_user and source_assistant
  byte-identical to corpus, corrected_answer non-empty, confidence in [0,1],
  source_id globally unique, aggregate train sequence is a strict prefix of
  train.jsonl, validation sequence empty, batch numbering contiguous)
- Repairs performed: none. First verification pass succeeded; no batch was rewritten
  and no original corpus or prior batch file was touched.
- Final schema check: PASS (identical to initial run)
- Manifest: MANIFEST.sha256 regenerated over all 206 files in the experiment
  directory except the manifest itself; `sha256sum -c` returned all-OK.

### Technical topics covered by this batch

Two scenario families, both rated technical_correctness=3 / instruction_coverage=2 /
operational_safety=2 because every source answer is a grading rubric ("answer should
state...") rather than an answer, and is therefore unusable as a supervised target.

1. Mixed short-prompt / long-generation serving evaluation (corpus-01294 through
   corpus-01300). Rewrites supply explicit metric definitions where the rubric had
   none: TTFT measured from client send rather than engine admission (the common way
   queueing delay is hidden), TPOT normalised over output_tokens-1, queueing delay
   exported separately, and throughput split into output tok/s, total tok/s and
   completed req/s so arms with different length mixes stay comparable. Load
   generation is specified open-loop with Poisson arrivals to avoid coordinated
   omission, which structurally understates P99 under closed-loop clients. Each
   rewrite carries a pre-registered falsifiable hypothesis on chunked prefill with a
   decision rule fixed in advance (bootstrap CI on P99 TTFT reduction must exclude 0
   while the P99 TPOT regression CI upper bound stays under 10%), interleaved arms to
   de-alias thermal drift, and prefix-cache hit rate pinned or disabled since an
   uncontrolled hit-rate delta can fabricate a 2x TTFT "win". Confounder checks span
   DCGM clock-throttle reasons, allocator drift, tokenizer-induced token-count
   mismatch, and for multi-node arms NCCL algorithm selection plus RoCE PFC/ECN state,
   where a paused link masquerades as a decode-speed regression.

2. Long-context intermittent OOM under concurrency (corpus-01301 through
   corpus-01304). Rewrites start from KV arithmetic the rubric never performs:
   per-token KV bytes = 2 x layers x num_kv_heads x head_dim x dtype_bytes, with the
   GQA/TP caveat that KV heads divide by TP degree only when num_kv_heads is divisible
   by TP -- otherwise heads are replicated and the per-GPU footprint is silently 2-4x
   larger than assumed. The discriminating hypothesis pits prefill activation peak
   against KV-pool exhaustion: capping max_num_batched_tokens to 4096 must eliminate
   OOM and cut peak allocated memory by >=20% under H0, and must not help under H1.
   Mitigations are ordered by reversibility -- budget arithmetic and chunked prefill
   (config-only) before admission control (converts a crash into backpressure, the
   correct operational behaviour) before expandable_segments fragmentation work
   (meaningful only if the memory snapshot shows large inactive_split blocks) before
   max_model_len / max_num_seqs reduction and fp8 KV quantization, which change
   user-visible capability or output quality and require a separate accuracy gate.
   Reproduction rate over >=5 replays with a binomial CI is mandated so an intermittent
   fault cannot appear "fixed" by chance.

All rewrites specify rollback gates: <=5% canary traffic, auto-rollback on >10-15%
P99 regression or error-rate breach, config-only rollback path exercised in staging
first.

**Status: PROVISIONAL.** These are one model's blind, independent second-opinion
reviews. They are not expert gold labels, have not been validated against teacher-A
(deliberately unread during this batch to avoid anchoring), and say nothing about any
trained model's domain capability. Agreement analysis is a separate, later step.

## Run 2026-08-17 batch 0117

- Batch file: results/train-batch-0117.jsonl
- Corpus range: train.jsonl lines 1161-1170 (0-indexed 1160..1169)
- Source IDs: corpus-01283, corpus-01284, corpus-01285, corpus-01286, corpus-01287,
  corpus-01289, corpus-01290, corpus-01291, corpus-01292, corpus-01293
- Progress: train 1170/5399, validation 0/601, total 1170/6000, remaining 4830
- Decisions: keep=0, rewrite=10, reject=0
- Initial schema check: PASS (1170 aggregate records, 12 required fields present,
  lane/model/status/decision values correct, source_user and source_assistant
  byte-identical to corpus, corrected_answer non-empty, confidence in [0,1],
  source_id globally unique, aggregate train sequence is a strict prefix of
  train.jsonl, validation sequence empty)
- Repairs: none required this run.
- Final schema check: PASS (identical run, no intervening edits)
- Manifest: MANIFEST.sha256 regenerated over all 205 files in this experiment
  directory except the manifest itself; `sha256sum -c` reports all OK.
- Topics covered: serving-evaluation scenario variants 283-293 (note the corpus
  itself skips 288), spread across System Design, Troubleshooting and Performance
  Analysis. Each asks for an evaluation plan for a mixed short-prompt /
  long-generation LLM endpoint reporting TTFT, TPOT, throughput, queueing and P99,
  with a falsifiable hypothesis and a controlled experiment. The rewrites supply
  the mechanism the source answers omit: KV-cache capacity as the binding
  constraint on concurrency, TTFT measured from client arrival rather than engine
  admission (the usual way queueing is hidden), open-loop Poisson arrivals instead
  of a closed-loop client that suffers coordinated omission, a frozen hashed
  request trace replayed identically across arms, interleaved repeated trials with
  bootstrap CIs, chunked-prefill as a testable TTFT/TPOT tradeoff, and
  fabric-side evidence (NVLink/PCIe intra-node; RoCE/IB port, PFC pause and ECN
  counters inter-node) plus KV-transfer latency and GDR-actually-active checks for
  disaggregated prefill/decode (Dynamo, Mooncake-style) deployments. Explicit
  rollback gates and a canary window were added to every item.
- Status: PROVISIONAL. These are one model's blind second-opinion rewrites. They
  are NOT expert gold labels, have not been cross-checked against teacher-A (this
  lane is blind by construction), and say nothing about any trained model's
  domain capability.

## Run 2026-08-17 batch 0116

- Batch file: results/train-batch-0116.jsonl
- Corpus range: train.jsonl lines 1151-1160 (0-indexed 1150..1159)
- Source IDs: corpus-01273, corpus-01274, corpus-01275, corpus-01276, corpus-01277,
  corpus-01278, corpus-01279, corpus-01280, corpus-01281, corpus-01282
- Progress: train 1160/5399, validation 0/601, total 1160/6000, remaining 4840
- Decisions: keep=0, rewrite=10, reject=0
- Initial schema check: PASS (1160 aggregate records, 12 required fields present,
  lane/model/status/decision values correct, source_user and source_assistant
  byte-identical to corpus, corrected_answer non-empty, confidence in [0,1],
  source_id globally unique, aggregate train sequence is a strict prefix of
  train.jsonl)
- Repairs: none required this run.
- Final schema check: PASS (identical run, no intervening edits)
- Manifest: MANIFEST.sha256 regenerated over all 196 files in this experiment
  directory except the manifest itself; `sha256sum -c` reports all OK.
- Topics covered: this block is a homogeneous run of "serving capacity" scenario
  variants 273-282, spread across Performance Analysis, System Design and
  Troubleshooting categories. Every item asks for an evaluation plan on a mixed
  short-prompt / long-generation LLM serving endpoint reporting TTFT, TPOT,
  throughput, queueing delay and P99 latency, with an explicit falsifiable
  hypothesis and a controlled experiment. The rewrites give operational metric
  definitions (client-side vs server-side TTFT; per-request TPOT rather than a
  global token/s counter; prefill and decode throughput reported separately;
  per-class rather than pooled P99), a falsifiable hypothesis attributing P99 TTFT
  degradation to chunked-prefill budget contention under KV-cache pressure rather
  than raw compute saturation, with both confirming and refuting predictions; a
  factorial design over chunked-prefill budget, KV-cache/concurrency cap and
  long-generation mixture ratio with warmup discard, repeated independent runs and
  randomized run order; explicit confounders (prefix caching / KV reuse, forced
  max_tokens vs EOS, client-side tokenizer and connection limits, clock and thermal
  drift, shared-GPU interference, autoscaler churn); required evidence (per-request
  traces, engine scheduler and preemption counters, GPU memory/occupancy telemetry,
  pinned versions and frozen workload manifest); and rollback gates tied to the
  measured repeat-to-repeat spread, a bounded canary, and automatic revert on
  preemption or KV-eviction regressions. The source assistant texts are grading
  rubrics rather than answers, which is why all ten are marked rewrite.
- Status: PROVISIONAL teacher-B output. It is NOT expert gold, has not been
  adjudicated against teacher-A, and says nothing about any model's domain
  capability. Blind-review discipline held: no teacher-A artifact was read.

## Run 2026-08-17 batch 0115

- Batch file: results/train-batch-0115.jsonl
- Corpus range: train.jsonl lines 1141-1150 (0-indexed 1140..1149)
- Source IDs: corpus-01259, corpus-01262, corpus-01263, corpus-01264, corpus-01265,
  corpus-01266, corpus-01267, corpus-01269, corpus-01270, corpus-01271
- Progress: train 1150/5399, validation 0/601, total 1150/6000, remaining 4850
- Decisions: keep=0, rewrite=10, reject=0
- Initial schema check: PASS (1150 aggregate records, 12 required fields present,
  lane/model/status/decision values correct, source_user and source_assistant
  byte-identical to corpus, corrected_answer non-empty, confidence in [0,1],
  source_id globally unique, aggregate train sequence is a strict prefix of
  train.jsonl)
- Repairs: none required this run; verification passed on the first attempt
- Final schema check: PASS (VERIFY_PASS, train 1150 / validation 0 / total 1150)
- Manifest: MANIFEST.sha256 regenerated over all files except itself (198 entries),
  `sha256sum -c` all OK
- Technical topics covered: serving-capacity evaluation design for a mixed
  short-prompt / long-generation workload. All ten items are the same scenario
  family (variants 259-271) split across System Design, Troubleshooting and
  Performance Analysis. The rewrites make explicit what the rubric only gestures at:
  operational definitions with units for TTFT (queue-wait + prefill), TPOT
  (decode-only, excluding the first token), throughput split into output-tokens/s
  vs completed-requests/s, and instrumented queue-wait rather than inferred;
  a pre-registered falsifiable hypothesis on batch-size vs throughput/TTFT-p99
  trade-off with an explicit null; a controlled protocol fixing the workload trace,
  arrival process (open-loop Poisson vs closed-loop concurrency are called out as
  non-interchangeable), decoding params (ignore_eos on for throughput, off for
  latency, never mixed), warmup discard with a stationarity check, and >=3
  randomized repeated trials with a lambda sweep to the saturation knee;
  confounders including prefill/decode interference bounded by chunked prefill,
  KV-cache pressure causing preemption/recompute (bimodal TTFT), client-side
  bottlenecks, power/thermal drift, noisy neighbours, and completion bias in the
  percentile window; and pre-agreed rollback gates (TTFT p99 +20%, preemption >1%,
  error-rate rise, gain inside the trial CI) with rollback by config flag against a
  still-deployed previous build.
- Status: PROVISIONAL. This is a second-opinion machine review, not expert gold, and
  it says nothing about any model's domain capability. Agreement analysis against
  teacher-A is a separate later step; no teacher-A artifact was read during this run.

## Run 2026-08-17 batch 0114

- Batch file: results/train-batch-0114.jsonl
- Corpus range: train.jsonl lines 1131-1140 (0-indexed 1130..1139)
- Source IDs: corpus-01247, corpus-01248, corpus-01249, corpus-01252, corpus-01253,
  corpus-01254, corpus-01255, corpus-01256, corpus-01257, corpus-01258
- Progress: train 1140/5399, validation 0/601, total 1140/6000, remaining 4860
- Decisions: keep=0, rewrite=10, reject=0
- Initial schema check: PASS (10/10 records, 12 required fields present, lane/model/
  status/decision values correct, source_user and source_assistant byte-identical to
  corpus, corrected_answer non-empty, confidence in [0,1], source_id globally unique,
  aggregate train sequence is a strict prefix of train.jsonl)
- Repairs: none required this run; verification passed on the first attempt
- Final schema check: PASS (train 1140, validation 0, total 1140, 0 errors)
- Manifest: MANIFEST.sha256 regenerated over 195 files; `sha256sum -c` all OK
- Topics covered: LLM serving evaluation methodology for a mixed short-prompt /
  long-generation workload (variants 247-249, 252-258 across System Design,
  Troubleshooting and Performance Analysis framings). The source assistant text is a
  rubric stub, not an answer, so all ten were rewritten with a shared mechanism section
  (TTFT = queue + schedule + prefill; TPOT bounded by HBM bandwidth and KV length;
  throughput knee converting into queueing) but a distinct pre-registered falsifiable
  hypothesis per item: chunked prefill token budget; KV-cache utilization vs
  preemption/recompute rate; admission control and open-loop backpressure; tensor-parallel
  degree vs replica count at fixed GPU budget (with NCCL all-reduce microbenchmark and
  nvidia-smi topo -m evidence); speculative decoding acceptance rate and its inversion at
  saturation; automatic prefix caching with hit rate as covariate; prefill/decode
  disaggregation of the Mooncake / NVIDIA Dynamo class with RDMA KV transfer cost and
  RoCE PFC/ECN verification; long-context attention scaling at pinned batch size;
  multi-node collective interference on a shared RoCE fabric with PFC pause and
  out-of-sequence counters; and FP8 weight+KV quantization gated on a paired quality eval.
  Each rewrite carries an explicit measurement protocol (open-loop Poisson arrivals,
  warmup discard, 5 restarted trials, bootstrap CIs on p99), a confounder list, and
  rollback gates (>10% P99 regression, >1% preemption, >0.5% error rate during a 10%
  canary).
- Status: PROVISIONAL second-opinion review only. Not expert gold, not adjudicated
  against teacher-A (this lane is blind by construction), and not evidence of any model
  domain capability. Agreement analysis is a separate later step.

## Run 2026-08-17 batch 0113

- Batch file: results/train-batch-0113.jsonl
- Corpus range: train.jsonl lines 1121-1130 (0-indexed 1120..1129)
- Source IDs: corpus-01235, corpus-01236, corpus-01237, corpus-01238, corpus-01239,
  corpus-01240, corpus-01241, corpus-01243, corpus-01244, corpus-01246
- Progress: train 1130/5399, validation 0/601, total 1130/6000, remaining 4870
- Decisions: keep=0, rewrite=10, reject=0
- Initial schema check: PASS (10/10 records, 12 required fields present, lane/model/
  status/decision values correct, source_user and source_assistant byte-identical to
  corpus, corrected_answer non-empty, confidence in [0,1], source_id globally unique,
  aggregate train sequence is a strict prefix of train.jsonl)
- Repairs: none required this run; verification passed on the first attempt
- Final schema check: PASS (train 1130, validation 0, total 1130, 0 errors)
- Manifest: MANIFEST.sha256 regenerated over 194 files; `sha256sum -c` all OK
- Topics covered: LLM serving evaluation methodology for a mixed short-prompt /
  long-generation workload. All ten items reuse the same scenario template with variant
  numbers 235-241, 243, 244, 246 across System Design, Troubleshooting and Performance
  Analysis framings, so each rewrite was given a distinct falsifiable hypothesis and
  controlled experiment rather than a shared boilerplate: chunked prefill token budget
  vs default scheduling; head-of-line blocking caused by long prefills; KV-cache block
  pressure and preemption/recompute as the throughput-limiting resource rather than SM
  compute; open-loop Poisson load generation versus closed-loop coordinated omission and
  its effect on P99; automatic prefix caching as a confound that fabricates TTFT gains on
  repetitive synthetic prompts; tensor-parallel degree sweeps where decode becomes NCCL
  all-reduce latency bound and tokens/s/GPU falls; disaggregated prefill/decode pools
  (NVIDIA Dynamo / Mooncake style) with KV transfer over RDMA and a KV-transfer share of
  the TTFT budget as the go/no-go gate; warmup, CUDA graph capture and clock/thermal
  drift as stationarity requirements; two-class SLO-tiered admission control and batch
  starvation limits; and speculative decoding whose TPOT benefit inverts below a draft
  acceptance rate of roughly 0.6 at large batch sizes. Every rewrite states assumptions
  with units, separates prefill from decode mechanism, lists expected confounders,
  enumerates the required per-request traces and DCGM telemetry, and gives explicit
  canary/rollback thresholds.
- Status caveat: these teacher-B outputs are PROVISIONAL. They are one model's blind
  independent review, not expert gold labels, and they are not evidence of any model's
  domain capability. Agreement analysis against teacher-A is a separate later step and
  no teacher-A artifact was read while producing this batch.

## Run 2026-08-17 batch 0112

- Batch file: results/train-batch-0112.jsonl
- Corpus range: train.jsonl lines 1111-1120 (0-indexed 1110..1119)
- Source IDs: corpus-01224, corpus-01226, corpus-01227, corpus-01228, corpus-01229,
  corpus-01230, corpus-01231, corpus-01232, corpus-01233, corpus-01234
- Progress: train 1120/5399, validation 0/601, total 1120/6000, remaining 4880
- Decisions: keep=0, rewrite=10, reject=0
- Initial schema check: PASS (10/10 records, 12 required fields present, lane/model/
  status/decision values correct, source_user and source_assistant byte-identical to
  corpus, corrected_answer non-empty, confidence in [0,1], source_id globally unique,
  aggregate train sequence is a strict prefix of train.jsonl)
- Repairs: none required this run
- Final schema check: PASS (train 1120, validation 0, total 1120, 0 errors)
- Manifest: MANIFEST.sha256 regenerated over 193 files; `sha256sum -c` all OK
- Topics covered: LLM serving evaluation methodology for a mixed short-prompt /
  long-generation workload. All ten items are the same scenario template with different
  variant numbers (224, 226-234) spread across Troubleshooting, System Design and
  Performance Analysis framings. The rewrites cover: prefill being compute-bound versus
  decode being HBM-bandwidth-bound; decomposing TTFT into queue wait plus prefill time
  and computing TPOT over the decode phase only; the throughput-versus-offered-load
  saturation knee driven by KV-cache block exhaustion and preemption/recompute; a
  falsifiable hypothesis on chunked prefill (>=30% P99 TTFT reduction with <=10% TPOT
  regression and <5% throughput change) with explicit falsification thresholds; open-loop
  load generation, coordinated omission, seeded identical request traces, clock locking,
  warmup discard and >=5 repetitions with bootstrap CIs; confounders including prefix
  caching, differing realized output lengths, thermal throttling and CUDA graph warmup;
  and canary-based rollback gates on P99 end-to-end, timeout/5xx rate and KV preemption
  counts.
- Blind-review compliance: no file under experiments/2026-08-14-teacher-a-corpus-calibration/
  was read, opened or searched while producing this batch.
- Status: these results are PROVISIONAL teacher-B second opinions. They are NOT expert
  gold labels, they have not been validated against ground truth, and they say nothing
  about any model's domain capability.

## Run 2026-08-17 batch 0111

- Batch file: results/train-batch-0111.jsonl
- Corpus range: train.jsonl lines 1101-1110 (0-indexed 1100..1109)
- Source IDs: corpus-01212, corpus-01213, corpus-01214, corpus-01215, corpus-01217,
  corpus-01219, corpus-01220, corpus-01221, corpus-01222, corpus-01223
  (原始 corpus 顺序，无跳过无重排；corpus-01216/01218 在原始 train.jsonl 中不存在)
- Progress: train 1110/5399, validation 0/601, total 1110/6000, remaining 4890
- Decisions: keep=0, rewrite=10, reject=0
- Initial schema check: PASS（/tmp/tb_verify.py 首次运行即 ERRORS: 0）
- Repairs: 无。本轮未修改原始 corpus、未修改既有批次、未触碰 teacher-A 产物
- Final schema check: PASS — train=1110/5399 validation=0/601 total=1110/6000
  覆盖项：逐行 JSONL 解析、批内条数=10、12 字段齐全、lane/model/status/decision 取值、
  source_user/source_assistant 与原始 corpus 逐字符相等、corrected_answer 非空、
  confidence ∈ [0,1]、source_id 全局唯一、train 序列严格为 corpus 前缀
- Manifest: MANIFEST.sha256 重新生成（192 个文件），sha256sum -c 全部通过
- 技术主题：本批 10 条全部是 mixed short-prompt / long-generation 在线推理服务的评测方案设计
  （scenario variant 212-223），按 category 分为 Troubleshooting / Performance Analysis /
  System Design 三种切入框架。重写答案显式给出：TTFT / queue-wait / TPOT / 双口径 throughput
  （output tokens/s 与 requests/s）/ P99 的可操作定义；open-loop Poisson 到达与 closed-loop
  客户端会掩盖排队的失效模式；prefill-decode 干扰、KV cache 压力与 preemption 导致的双峰 TPOT；
  chunked prefill、disaggregated prefill/decode、TP 度数与 NVLink/PCIe 拓扑（nvidia-smi topo -m）
  对 NCCL 每步延迟的影响；每条给出单变量、单阈值、自带证伪条件的假设；>=5 trials、>=3000 条
  post-warmup 完成、bootstrap 95% CI、随机化执行顺序；以及 canary <=5% 流量 60 分钟、
  次要 SLO 回退 <=5%、第二台同型节点复现、保留旧配置 artifact hash 以便免重建回滚的门槛。
- 原始 source_assistant 是评分 rubric（"Answer should state ..."）而非答案，直接用于 SFT
  会教出 meta-commentary，因此本批 10 条全部判 rewrite。
- 结果性质：provisional teacher-B 盲审意见，非 expert gold，不代表任何模型领域能力；
  与 teacher-A 的一致率分析是后续独立步骤，本轮全程未读取 teacher-A 目录。

## Run 2026-08-17 batch 0110

- Batch file: results/train-batch-0110.jsonl
- Corpus range: train.jsonl lines 1091-1100 (0-indexed 1090..1099)
- Source IDs: corpus-01201, corpus-01202, corpus-01203, corpus-01205, corpus-01206,
  corpus-01207, corpus-01208, corpus-01209, corpus-01210, corpus-01211
  (原始 corpus 顺序，无跳过无重排；corpus-01204 在原始 train.jsonl 中不存在)
- Progress: train 1100/5399, validation 0/601, total 1100/6000, remaining 4900
- Decisions: keep=0, rewrite=10, reject=0
- Initial schema check: PASS (verify_teacher_b.py, 首次运行即通过)
- Repairs: 无。本轮未修改原始 corpus、未修改既有批次、未触碰 teacher-A 产物
- Final schema check: PASS — train=1100/5399 validation=0/601 total=1100/6000 VERIFY_PASS
  覆盖项：逐行 JSONL 解析、批内条数=10、12 字段齐全、lane/model/status/decision 取值、
  source_user 与 source_assistant 与原始 corpus 逐字符相等、corrected_answer 非空、
  confidence ∈ [0,1]、source_id 全局唯一、聚合序列严格是 train corpus 前缀
- Manifest: MANIFEST.sha256 已重新生成（191 个文件），sha256sum -c 全部通过

### 本批技术主题

本批 10 条全部是同一族的 "mixed short-prompt / long-generation 服务评测方案"
题目（scenario variant 201-211），按 category 分为三种评审框架：

- Performance Analysis (01201, 01207, 01210)：可证伪假设 H-P —— 开启 chunked
  prefill (chunk=512) 在固定 QPS 下将短 prompt 的 TTFT P99 降低 ≥25%，同时
  output-token throughput 退化 ≤5%。机制：分块限制单次 prefill 占用调度步的长度，
  减少对短请求的 head-of-line blocking；代价是调度迭代变多、prefill 算术强度下降。
  边界条件：prompt 全短时效应应趋近于零，若短 prompt-only cell 也改善则机制判定被证伪。
- System Design (01202, 01205, 01208, 01211)：可证伪假设 H-S —— 在等 GPU 数下把
  prefill / decode 拆到独立副本池（P:D 比按实测 token 比例调）使生产混合负载的
  端到端 P99 降低 ≥20%，且 KV 传输开销占 TTFT ≤15%。显式给出 KV 传输字节量
  公式 2 * layers * kv_heads * head_dim * dtype_bytes * prompt_tokens，并要求验证
  跨节点路径确实走 RDMA (RoCE/IB) + GPUDirect RDMA（NCCL_DEBUG=INFO 中的 GDRDMA、
  NCCL_NET_GDR_LEVEL、PCIe 亲和性），涉及 NIXL/Mooncake 类传输引擎与 NVIDIA Dynamo
  的 disaggregated router。边界条件：短 prompt 与窄互联下 disaggregation 会输，
  必须通过 prompt 长度扫描复现 crossover 点。
- Troubleshooting (01203, 01206, 01209)：按成本从低到高排列的四条机制假设
  H-T1 排队受限、H-T2 KV 耗尽/抢占、H-T3 prefill/decode 干扰、H-T4 硬件与集合通信
  (throttle reasons、TP>1 的 NCCL allreduce 方差、topo/功耗上限)，每条给出独立
  signature 与单变量证伪实验，并要求记录负结果。

三种框架共享同一套硬性内容：指标定义歧义消除（TTFT 含排队并可分解为
queue_wait + prefill_compute + network；TPOT 报分布不报全局均值；throughput 同时报
output-token / total-token / requests 三种）、prefill 计算受限 vs decode
显存带宽与 KV 容量受限的机制区分与干扰项 I = P99(C) - max(P99(A),P99(B))、
warmup 丢弃规则、开环 Poisson 与闭环并发双扫描、≥5 次独立试验且每 cell
≥3000 次完成、bootstrap CI、混杂因素（热/功耗降频、邻居噪声、客户端瓶颈、
tokenizer 差异、prefix caching 命中、输出长度方差）、所需证据清单，以及
promotion/rollback 门槛（P99 不退化 >5%、TTFT P95 不退化 >10%、吞吐 ≥+10%、
无抢占/超时增加、显存 headroom ≥5%；canary ≤5% 流量 ≥30 分钟）。

原始 source_assistant 全部是评分 rubric（"Answer should state ..."）而非答案本身，
因此 10 条一律判 rewrite；quality_dimensions 统一为 technical_correctness=3、
instruction_coverage=2、operational_safety=3，confidence=0.62。

**结果性质声明**：本批结果是 provisional teacher-B 盲审输出，由当前对话模型
(claude-opus-5-current) 独立写出，未查看 teacher-A 的任何产物。它不是 expert gold，
未经领域专家复核，也不代表任何模型的领域能力；其中所有性能数字阈值都是待验证的
假设门槛，而非实测结论。

## Run 2026-08-17 batch 0109

- Batch file: results/train-batch-0109.jsonl
- Corpus range: train.jsonl lines 1081-1090 (0-indexed 1080..1089)
- Source IDs: corpus-01191 .. corpus-01200 (连续，原始顺序保持，无跳过无重排)
- Progress: train 1090/5399, validation 0/601, total 1090/6000, remaining 4910
- Decisions: keep=0, rewrite=10, reject=0
- Initial schema check: PASS (scripts/verify_batches.py, 首次即通过，无修复动作)
- Fix actions: 无
- Final schema check: PASS (train 1090 records, validation 0, TOTAL 1090, VERIFY=PASS)
- Manifest: MANIFEST.sha256 重新生成（184 条），`sha256sum -c` 全部通过
- 技术主题: 混合短 prompt / 长生成流量的服务评估协议。本批 10 条同源模板，
  按 category 分成三种独立视角改写：Troubleshooting（排队受限 / KV 耗尽与抢占 /
  prefill-decode 干扰 / 硬件与 NCCL 四条可证伪假设，按成本从低到高排序）、
  Performance Analysis（prefill 计算受限与 decode 显存带宽受限的 roofline 容量模型、
  KV bytes/token = 2*n_layers*n_kv_heads*head_dim*dtype_bytes 的可核对推导、
  预测与实测偏差 >20% 即判模型失效）、System Design（可复现 trace 生成器与结果
  schema、容量口径定义为满足 SLO 的最大到达率而非最大吞吐、灰度与自动回滚门槛）。
  共同部分强制：开环 Poisson 到达（闭环会掩盖排队时延并低估 P99）、TTFT/TPOT 的
  精确测量端点与 queue_wait+prefill 分解、按 short/long 分类分别报告 P99 并给
  bootstrap 95% CI、warmup 截断与稳态判据、每臂 3-5 次独立启动、DCGM 1Hz 采样含
  throttle reasons、Little's Law 一致性校验，以及 abort/rollback 门槛。
- 原始 assistant 内容为评分要点清单而非可执行答案，故 10 条全部判为 rewrite；
  quality_dimensions 统一给 technical_correctness=3 / instruction_coverage=2 /
  operational_safety=2，confidence 0.70-0.71。
- 声明: 本结果为 provisional teacher-B 盲审产物，不是 expert gold，也不代表任何
  模型的领域能力。本批产出过程中未读取 teacher-A 目录下任何文件。

## Run 2026-08-17 batch 0108

- Batch file: results/train-batch-0108.jsonl
- Corpus range: train.jsonl lines 1071-1080 (0-indexed 1070..1079)
- Source IDs: corpus-01180, corpus-01181, corpus-01182, corpus-01184, corpus-01185,
  corpus-01186, corpus-01187, corpus-01188, corpus-01189, corpus-01190
  (corpus-01183 is absent from the corpus itself; original order preserved, nothing skipped by this worker)
- Progress: train=1080/5399, validation=0/601, total=1080/6000, remaining=4920
- Decisions: keep=0, rewrite=10, reject=0
- Initial schema check: PASS (verify_batches.py, first run, no repair needed)
- Repairs: none
- Final schema check: PASS
- Manifest: regenerated, sha256sum -c => 185/185 OK
- Technical topics covered: LLM serving evaluation methodology for mixed
  short-prompt / long-generation traffic — TTFT / TPOT / queueing-delay / goodput
  definitions frozen before measurement, open-loop Poisson load generation and
  coordinated-omission avoidance, prefill (compute-bound) vs decode
  (HBM-bandwidth-bound) decomposition, chunked prefill and prefill/decode
  disaggregation trade-offs including the KV-handoff interconnect boundary,
  KV-cache preemption/eviction as a P99 tail driver, clock pinning and thermal/power
  drift as confounders, per-step scheduler and per-request tracing as required
  evidence, and pre-registered mechanical rollback gates.
- Per-category framing: Performance Analysis items got a chunked-prefill goodput
  hypothesis, System Design items an equal-GPU disaggregation-vs-homogeneous-pool
  hypothesis with a stated sign-flip crossover, Troubleshooting items a
  queueing/preemption-vs-compute discriminating diagnosis with an explicit falsifier.
- Blind review discipline: no file under experiments/2026-08-14-teacher-a-corpus-calibration/
  was read, opened, or searched during this run.
- Status: PROVISIONAL. These are provisional teacher-B second opinions, not expert
  gold labels, and they are not evidence of any model's domain capability.

## Run 2026-08-17 batch 0107

- Batch file: results/train-batch-0107.jsonl
- Corpus range: train.jsonl lines 1061-1070 (source IDs corpus-01170 through corpus-01179, contiguous, corpus file order preserved exactly, no skips, no reordering)
- Progress: train 1070/5399, validation 0/601, total 1070/6000, remaining 4930
- Decisions: keep 0, rewrite 10, reject 0
- Initial schema/prefix check: PASS (verify_batches.py, first run, no repairs needed)
- Repairs performed: none
- Final schema/prefix check: PASS — train=1070/5399 validation=0/601 total=1070/6000, SCHEMA_CHECK=PASS; source_id global uniqueness re-checked independently (1070 records, 1070 unique)
- Manifest: MANIFEST.sha256 regenerated over all files except itself; `sha256sum -c` reports 184/184 OK, zero failures
- Technical topics covered: LLM serving evaluation methodology for mixed short-prompt / long-generation traffic — prefill vs decode separation (compute-bound vs HBM-bandwidth-bound), frozen TTFT/TPOT/queueing-delay/goodput definitions, open-loop Poisson arrival generation and why closed-loop harnesses truncate the P99 tail, clock pinning and DVFS/thermal confounds, KV cache preemption and fragmentation as a hidden regression, roofline-style attribution for the Performance Analysis items, symptom-class triage ordering (TTFT-only / TPOT-only / both) for the Troubleshooting items, and prefill/decode disaggregation with RDMA KV transfer (Mooncake-style, NVIDIA Dynamo-style) for the System Design items, framed as a measurable inequality rather than a default recommendation. Every rewrite carries a pre-registered falsifiable hypothesis with a kill condition, an explicit confounder list, an evidence-required list, and a canary rollback gate.
- All 10 source_assistant values were rubric checklists ("Answer should state...") rather than answers, hence decision=rewrite across the batch; source_user/source_assistant were copied byte-for-byte and the original corpus was not modified.
- Status: PROVISIONAL. These are single-model teacher-B second opinions produced blind (no teacher-A artifact was read, opened, or grepped during this batch). They are NOT expert gold labels, have NOT been human-verified, and say nothing about any model's domain capability. Agreement analysis against teacher-A is a separate, later step and was deliberately not performed here.

## Run 2026-08-17 batch 0106

- Batch file: results/train-batch-0106.jsonl
- Corpus range: train.jsonl lines 1051-1060 (source IDs corpus-01157, corpus-01158, corpus-01159, corpus-01160, corpus-01161, corpus-01162, corpus-01164, corpus-01165, corpus-01168, corpus-01169 — corpus file order preserved exactly, no skips, no reordering; corpus-01163, corpus-01166, corpus-01167 are absent from the corpus file itself, so the ID gaps are a source property, not an omission here)
- Progress: train 1060/5399, validation 0/601, total 1060/6000, remaining 4940
- Decisions: keep 0, rewrite 10, reject 0
- Initial schema/prefix check: PASS (verify_batches.py, first run, no repairs needed)
- Repairs performed: none
- Final schema/prefix check: PASS (1060 train records, 0 validation, TOTAL 1060, VERIFY=PASS)
- Manifest: MANIFEST.sha256 regenerated over 180 files (all files in this directory except MANIFEST.sha256 and scripts/__pycache__); `sha256sum -c` PASS
- Blindness: teacher-A directory (experiments/2026-08-14-teacher-a-corpus-calibration/) was not read, opened, or grepped at any point in this run.

Technical topics covered by this batch: all ten items are the "mixed short-prompt /
long-generation serving evaluation plan" template, so the rewrites differentiate by
mechanism rather than by prompt. Shared spine: pinned build/clock/tokenizer state,
open-loop Poisson arrivals (with an explicit argument for why closed-loop harnesses
delete the tail samples), frozen TTFT/TPOT/queue-delay/goodput definitions, per-stratum
P99 from raw records, load ladder past saturation, and pre-committed rollback gates.
Per-item hypotheses: 01157 chunked prefill vs head-of-line blocking (with a chunk=512
monotonicity check as the mechanism test); 01158 queueing-vs-service-time differential
diagnosis with four competing signatures; 01159 decode as an HBM-bandwidth roofline
claim tested by decoupled SM/memory clock manipulation; 01160 prefill/decode
disaggregation where the KV handover budget over achieved (not line-rate) RDMA/RoCE
bandwidth decides feasibility, including PFC/ECN evidence; 01161 KV-preemption vs code
regression separated by a mixture x commit 2x2 replay; 01162 speculative decoding with
the closed-form acceptance/break-even model and the win-to-loss batch crossover;
01164 rolling-deploy tail as cold prefix cache vs balancer imbalance, tested by
affinity routing; 01165 KV capacity arithmetic with FP8 KV vs matched-capacity context
reduction as two implementations of one mechanism, gated on output quality; 01168
TP=8 decode as communication-bound, checked against isolated all-reduce latency and
nvidia-smi topo -m; 01169 SLO-aware admission control under overload with per-stratum
starvation gates and an adversarial under-declared-length arm.

Status caveat: these outputs are PROVISIONAL teacher-B review artifacts produced by a
single model in one pass. They are not expert gold labels, they have not been validated
against hardware, and they are not evidence about any trained model's domain capability.
Agreement analysis against teacher-A is a separate, later step and is deliberately not
performed here.

## Run 2026-08-17 batch 0105

- Batch file: results/train-batch-0105.jsonl
- Corpus range: train.jsonl lines 1041-1050 (source IDs corpus-01145, corpus-01146, corpus-01147, corpus-01148, corpus-01149, corpus-01150, corpus-01151, corpus-01153, corpus-01154, corpus-01155 — corpus file order preserved exactly, no skips, no reordering; corpus-01152 is absent from the corpus file itself, so the ID gap is a source property, not an omission here)
- Progress: train 1050/5399, validation 0/601, total 1050/6000, remaining 4950
- Decisions: keep 0, rewrite 10, reject 0
- Initial schema/ad-hoc verification: PASS on first run (verify_batches.py over all 1050 aggregate train rows: physical-newline JSONL parse, 10 rows in this batch, all 12 required fields present and no extra fields, teacher_lane/teacher_model/calibration_status/decision values valid, source_user and source_assistant character-identical to corpus, corrected_answer non-empty, confidence in [0,1], quality_dimensions integers 1-5, source_id globally unique (1050 unique across 105 batches), aggregate train sequence a strict prefix of train.jsonl, validation 0/601)
- Repairs performed: none required
- Final schema/ad-hoc verification: PASS (SCHEMA_CHECK=PASS, exit 0)
- Manifest: MANIFEST.sha256 regenerated over all files in this directory except itself; `sha256sum -c` verified 180/180 entries OK with no failures
- Technical topics covered: serving-side evaluation methodology for mixed short-prompt / long-generation LLM traffic (scenario variants 145-151, 153-155). Shared frame: explicit non-claims (numbers valid only for a fixed engine-commit / weight-hash / quantization / tokenizer / GPU-SKU / driver / launch-flag tuple, and silent about model quality); a pinned two-population workload model with a declared mix ratio and output length pinned via max_tokens + ignore_eos so TPOT is not contaminated by a random length variable; open-loop Poisson arrivals from an off-host generator, with closed-loop harnesses restricted to an explicitly labelled saturation-capacity arm because they self-throttle and delete the tail under study; metric definitions that separate client-side TTFT from the server arrival-to-first-token timestamp (the delta being the queue+transport component, never folded in), per-request TPOT distributions rather than a global mean, output vs total token throughput, and SLO-filtered goodput as the actual decision metric; percentiles computed from pooled raw per-request records with bootstrap CIs and a rule that overlapping CIs mean "not distinguishable"; >=60 s warmup discard for CUDA-graph capture, autotuning, KV-pool growth and clock ramp; environment controls covering persistence mode, locked/recorded clocks, ECC, MPS/MIG, NUMA/CPU pinning and verified client headroom; measured confounders including thermal/power throttling, KV-cache pressure with preemption/recompute, chunked-prefill interference with in-flight decodes, tokenizer-induced token-count drift, and coordinated omission; pre-registered rollback gates (ship only on >=10% goodput gain with P99 TTFT/TPOT within 5%, roll back on >20% P99 TTFT regression, >2x preempt/OOM rate or >0.1% errors, 24 h canary at <=5% traffic with the previous image pinned). Category-specific falsifiable hypotheses: System Design items test whether prefill/decode disaggregation beats a co-located continuous-batching pool at equal GPU count, discriminated by per-step prefill/decode timelines and TPOT conditioned on co-scheduling, with the cross-node KV transfer budgeted against measured RDMA bandwidth; Troubleshooting items separate admission queueing from KV-exhaustion preemption from thermal throttling via three disjoint counter signatures in the queue-wait / decode-time / clock decomposition; Performance Analysis items test the memory-bandwidth-bound decode hypothesis by sweeping batch size and comparing achieved weight+KV bytes/s against measured HBM peak, refuting it when achieved bandwidth is <30% of peak at low SM occupancy (which relocates cost to launch overhead, scheduling or CPU detokenization), with prefill analysed separately as a compute-bound roofline/MFU question.
- Why all ten were marked `rewrite`: each source_assistant is a grading rubric ("Answer should state ...") rather than an answer, so training on it teaches meta-commentary about answers instead of engineering reasoning; it also omits metric definitions, the arrival model, output-length pinning, sample-size rules for percentiles, and any pre-registered numeric rollback threshold.
- Status caveat: these are PROVISIONAL model-generated second-opinion labels produced blind (no teacher-A artifact was read, opened or grepped during this batch). They are not expert gold and they say nothing about any model's domain capability.

## Run 2026-08-17 batch 0104

- Batch file: results/train-batch-0104.jsonl
- Corpus range: train.jsonl lines 1031-1040 (source IDs corpus-01134, corpus-01135, corpus-01136, corpus-01137, corpus-01138, corpus-01140, corpus-01141, corpus-01142, corpus-01143, corpus-01144 — corpus file order preserved exactly, no skips, no reordering; corpus-01139 is absent from the corpus file itself, so the ID gap is a source property, not an omission here)
- Progress: train 1040/5399, validation 0/601, total 1040/6000, remaining 4960
- Decisions: keep 0, rewrite 10, reject 0
- Initial schema/ad-hoc verification: PASS on first run (verify_batches.py over all 1040 aggregate train rows: physical-newline JSONL parse, 10 rows in this batch, all 12 required fields present, teacher_lane/teacher_model/calibration_status/decision values valid, source_user and source_assistant character-identical to corpus, corrected_answer non-empty, confidence in [0,1], quality_dimensions integers 1-5, source_id globally unique (1040 unique across 104 batches), aggregate train sequence a strict prefix of train.jsonl, validation 0/601)
- Repairs performed: none required
- Final schema/ad-hoc verification: PASS (VERIFY=PASS, exit 0)
- Manifest: MANIFEST.sha256 regenerated over all files in this directory except itself; `sha256sum -c` verified all entries OK with no failures
- Technical topics covered: serving-side evaluation methodology for mixed short-prompt / long-generation LLM traffic (scenario variants 134-138, 140-144). Shared frame: explicit non-claims (serving-system behaviour only, no transfer across engine commit, weights/quantization, sequence-length regime, batch policy or GPU SKU without a re-run); pinned clocks and persistence mode with fixed MIG/ECC state so DVFS and thermal drift cannot be read as a treatment effect; open-loop Poisson arrivals from an off-host generator because a closed-loop harness self-throttles and deletes the tail under study; PTP/NTP clock discipline below the smallest reportable difference; output length pinned via max_tokens + ignore_eos; prefix-sharing rate reported because prefix/KV cache warmth silently moves TTFT; metrics with mandatory units, client-side TTFT separated from the server first-decode timestamp (the gap is queue + transport, never folded in), per-request TPOT distributions, queue wait, throughput and SLO-filtered goodput, plus KV-cache utilization, running/waiting queue depth, preemption and recompute counters, achieved HBM bandwidth, SM occupancy and throttle reasons; every percentile reported with n and a confidence interval, sized so P99 rests on >= ~100 tail events. Falsifiable hypothesis H1 for this batch: at target arrival rate, median queue wait accounts for >= 50% of P99 end-to-end latency while TPOT P99 stays within 1.2x of TPOT P50, predicting that +1 replica at constant load cuts P99 by >= 30% with statistically unchanged TPOT P50; refuted if P99 moves < 10% or TPOT P99/P50 > 1.5, which relocates the bottleneck to per-step execution or KV capacity. Controlled design: interleaved A/B/A/B arms, 5 min discarded warmup, >= 15 min measurement, >= 3 repeats, paired comparison with intervals. Category depth: Troubleshooting items add an ordered cut set (admission vs execution via queue wait against first-decode latency; prefill vs decode via the TTFT/TPOT split; memory via KV utilization and preemption counters; host vs device via GPU busy% against tokenizer/scheduler CPU time; per-replica breakdown before aggregation, since one throttled or ECC-degraded device can own the whole P99). System Design items turn the measurements into decidable design choices (single pool vs prefill/decode disaggregation, where separation only wins if measured KV transfer time is small against the TTFT it protects; chunked-prefill and continuous-batching thresholds reported as a frontier rather than a best point; queue/priority isolation so a long generation cannot occupy the slot a short request needs; replica and TP/PP sizing from measured KV headroom rather than peak FLOPs; autoscaling on goodput and queue wait, never GPU utilization, which saturates long before the SLO breaks). Performance Analysis items enforce analysis discipline (per-request distributions instead of means over a bimodal mixture, breakdowns by class and replica, latency attribution into queue/prefill/decode/transport that must sum to end-to-end within tolerance, and comparison at equal offered and accepted load so a throughput win from dropped or truncated requests is rejected). Pre-committed rollback: revert the canary on > 10% P99 regression, goodput drop at equal offered load, error/timeout rate above baseline, preemption counters rising from zero, or any GPU OOM.
- Why all ten were marked `rewrite`: each source_assistant is a grading rubric ("Answer should state ...") rather than an answer, so training on it teaches meta-commentary about answers instead of engineering reasoning; it also omits units, the open-loop load-generation requirement, output-length pinning, the client-side vs server-side TTFT split, per-class reporting, sample-size requirements for percentiles, and any pre-registered numeric rollback threshold.
- Status caveat: these are PROVISIONAL model-generated second-opinion labels produced blind (no teacher-A artifact was read, opened or grepped during this batch). They are not expert gold and they say nothing about any model's domain capability.

## Run 2026-08-17 batch 0103

- Batch file: results/train-batch-0103.jsonl
- Corpus range: train.jsonl lines 1021-1030 (source IDs corpus-01123, corpus-01124, corpus-01125, corpus-01126, corpus-01127, corpus-01128, corpus-01130, corpus-01131, corpus-01132, corpus-01133 — corpus file order preserved exactly, no skips, no reordering; corpus-01129 is absent from the corpus file itself, so the ID gap is a source property, not an omission here)
- Progress: train 1030/5399, validation 0/601, total 1030/6000, remaining 4970
- Decisions: keep 0, rewrite 10, reject 0
- Initial schema/ad-hoc verification: PASS on first run (verify_batches.py over all 1030 aggregate train rows: physical-newline JSONL parse, 10 rows in this batch, all 12 required fields present, teacher_lane/teacher_model/calibration_status/decision values valid, source_user and source_assistant character-identical to corpus, corrected_answer non-empty, confidence in [0,1], quality_dimensions integers 1-5, source_id globally unique (1030 unique across 103 batches), aggregate train sequence a strict prefix of train.jsonl, validation 0/601)
- Repairs performed: none required
- Final schema/ad-hoc verification: PASS (VERIFY=PASS, exit 0)
- Manifest: MANIFEST.sha256 regenerated over all files in this directory except itself; `sha256sum -c` verified 176/176 OK with no failures
- Technical topics covered: serving-side evaluation methodology for mixed short-prompt / long-generation LLM traffic (scenario variants 123-128, 130-133). Shared frame: explicit non-claims (serving-system behavior only; no transfer across engine commit, weights/quantization, sequence-length regime, batch policy or GPU SKU without a re-run); pinned clocks, persistence mode, fixed MIG/ECC state so DVFS and thermal drift cannot be read as a treatment effect; open-loop Poisson arrivals from an off-host generator because closed-loop harnesses self-throttle and delete exactly the tail under study; output length pinned via max_tokens + ignore_eos; prefix-sharing rate reported explicitly because prefix/KV cache warmth silently moves TTFT; metrics with mandatory units, client-side TTFT separated from server first-decode timestamp (the gap is queue + transport and is never folded in), per-request TPOT distributions, queue wait, throughput and SLO-filtered goodput, plus KV utilization, queue depth, batch size per step, preemption/recompute counts and achieved HBM bandwidth; P50/P90/P99 always with n and a CI, with enough samples that P99 rests on >= ~100 tail events. Category-specific depth: Performance Analysis items test H1 that P99 is dominated by decode-phase queueing behind long generations rather than prefill cost, with a single-knob A/B (concurrency cap or chunked-prefill token budget), a load sweep from ~40% of saturation past the knee, and per-class rather than pooled tail analysis. System Design items add the measurement architecture itself: off-host generators with verified CPU headroom, request-scoped trace IDs propagated client -> gateway/router -> engine so per-hop queueing is attributable, one tenant per GPU/MIG slice during measurement, and for disaggregated prefill/decode or KV-transfer designs the requirement to report KV transfer bytes and latency as first-class metrics while naming the interconnect, since a design that looks good over NVLink can be fabric-bound over RDMA/RoCE without lossless/PFC and DCQCN tuning. Troubleshooting items decompose latency into client/transport, router queue, engine admission queue, prefill compute, decode steps and detokenize/stream, name the three standard misdiagnoses (queue wait misread as slow GPU, TTFT regression that is really a prefix-cache hit-rate change, throughput drop that is really KV-pressure preemption/recompute), and test H1 that the tail is KV-cache exhaustion with the explicit falsifier that zero preemption counters during tail events discards the hypothesis rather than patching it. All ten close with the same confounder list (prefix reuse, chunked-prefill/co-scheduling TTFT-vs-TPOT trade, background tenants and NUMA/CPU pinning, warmup discard with reported N, frozen replica count), the same evidence list (raw per-request traces, >= 1 Hz engine and GPU telemetry, exact versions/flags/seed, >= 3 interleaved repeats with CIs), and pre-registered rollback gates decided before seeing results.
- Why all ten were marked `rewrite`: each source_assistant is a grading rubric ("Answer should state ...") rather than an answer, so training on it teaches meta-commentary about answers instead of engineering reasoning; it also omits units, the open-loop load-generation requirement, output-length pinning, the client-side vs server-side TTFT split, per-class reporting, and any pre-registered numeric rollback threshold.
- Status caveat: these are PROVISIONAL model-generated second-opinion labels produced blind (no teacher-A artifact was read, opened or grepped during this batch). They are not expert gold and they say nothing about any model's domain capability.

## Run 2026-08-17 batch 0102

- Batch file: results/train-batch-0102.jsonl
- Corpus range: train.jsonl lines 1011-1020 (source IDs corpus-01112, corpus-01113, corpus-01114, corpus-01115, corpus-01116, corpus-01117, corpus-01118, corpus-01119, corpus-01121, corpus-01122 — corpus file order preserved exactly, no skips, no reordering; corpus-01120 is absent from the corpus file itself, so the ID gap is a source property, not an omission here)
- Progress: train 1020/5399, validation 0/601, total 1020/6000, remaining 4980
- Decisions: keep 0, rewrite 10, reject 0
- Initial schema/ad-hoc verification: PASS on first run (/tmp/tb_verify.py over all 1020 aggregate train rows: every line JSON-parseable via physical-newline split, 10 rows in this batch, exactly the 12 required fields and no extras, teacher_lane/teacher_model/calibration_status/decision values valid, source_user and source_assistant character-identical to corpus, corrected_answer non-empty, confidence in [0,1], quality_dimensions integers in 1-5, source_id globally unique across all 102 batches (1020 unique), aggregate train sequence a strict prefix of train.jsonl, validation 0/601)
- Repairs performed: none required
- Final schema/ad-hoc verification: PASS (VERIFY=PASS, exit 0)
- Manifest: MANIFEST.sha256 regenerated over all files in this directory except itself; `sha256sum -c` verified with no failures
- Technical topics covered: serving-side evaluation methodology for mixed short-prompt / long-generation LLM traffic (scenario variants 112-119, 121, 122). Shared frame across all ten: explicit non-claims (serving-system behavior only, no transfer across engine commit, model size, quantization, sequence-length regime or GPU SKU without a re-run); pinned clocks / persistence mode / fixed MIG and ECC state so DVFS and thermal drift cannot be read as a treatment effect; open-loop Poisson arrivals from an off-host generator because closed-loop harnesses self-throttle and delete the exact tail samples under study; output length frozen via max_tokens + ignore_eos so generation length cannot act as a hidden confounder; prefix-sharing rate reported explicitly because prefix-cache warmth silently moves TTFT; metrics defined with units and split into client-side TTFT vs server first-decode timestamp, per-request TPOT distribution, queue wait, throughput, and SLO-filtered goodput, plus GPU-side occupancy, achieved HBM bandwidth, KV-cache utilization and preemption/recompute counts. Falsifiable hypothesis per item: chunked prefill at chunk size C reduces P99 TPOT by >=20% relative at <=3% throughput loss, refuted if the effect is smaller or throughput drops more in >=2 of 5 paired repetitions, with a pre-registered secondary rejection if the longest-prompt-decile P99 TTFT regresses >15%. Category-specific depth: System Design items develop prefill/decode disaggregation (Dynamo-style prefill/decode workers, Mooncake-style KVCache-centric split) with the boundary condition that the added per-request KV transfer over NVLink/RDMA can exceed the saved decode jitter for short prompts, plus chunked prefill, paged KV with prefix/radix reuse gated on reported cache hit rate, and admission control (max_num_seqs, max_num_batched_tokens) as the actual queueing knobs. Troubleshooting items decompose latency into client->gateway, gateway queue, scheduler wait, prefill compute, decode steps, detokenize/stream and return path, then rank five hypotheses each with a discriminating test: queueing saturation (queue wait grows while per-step decode is flat), prefill interference (decode step time correlates with concurrent long prompts), KV pressure/preemption, NCCL collective stall in TP/PP with per-rank step-time divergence and a check that GDR is actually active via transport counters rather than a config flag, and host-side tokenizer/GIL/logging cost. Performance Analysis items require per-request records instead of bucketed exporter histograms (whose P99 is biased by bucket edges), >=5 fresh-process repetitions with median/IQR of per-run P99 and a bootstrap CI, and a decode roofline check that a claimed decode speedup must be accompanied by changed achieved HBM bandwidth since small-batch decode is memory-bandwidth bound. All ten close with the same evidence list and numeric rollout gates: 5% canary for >=30 min and >=10k requests, automatic rollback on >10% P99 end-to-end regression, +0.1 absolute percentage points of error/timeout rate, >5% goodput loss, or any preemption/OOM absent in control, with rollback as a config flip requiring no weight reload and a measured (not assumed) time-to-restore.
- Why all ten were marked `rewrite`: each source_assistant is a grading rubric ("Answer should state ...") rather than an answer, so training on it teaches meta-commentary about answers instead of the engineering reasoning itself; it also omits metric definitions with units, the open-loop load-generation requirement, output-length pinning, per-stratum reporting, mechanism-level attribution with a named refutation condition, confounder instrumentation, and the numeric abort/rollback gates the prompts explicitly demand.
- Status caveat: these are PROVISIONAL model-generated second-opinion labels produced blind (no teacher-A artifact was read, opened or grepped during this batch). They are not expert gold and they say nothing about any model's domain capability.

## Run 2026-08-17 batch 0101

- Batch file: results/train-batch-0101.jsonl
- Corpus range: train.jsonl lines 1001-1010 (source IDs corpus-01101, corpus-01103, corpus-01104, corpus-01105, corpus-01106, corpus-01107, corpus-01108, corpus-01109, corpus-01110, corpus-01111 — corpus file order preserved exactly, no skips, no reordering; note corpus-01102 is absent from the corpus file itself, so the ID sequence is non-contiguous by source, not by omission here)
- Progress: train 1010/5399, validation 0/601, total 1010/6000, remaining 4990
- Decisions: keep 0, rewrite 10, reject 0
- Initial schema/ad-hoc verification: PASS on first run (verify_batches.py over all 1010 aggregate train rows: every line JSON-parseable via physical-newline split, 10 rows in this batch, all 12 required fields present, teacher_lane/teacher_model/calibration_status/decision values valid, source_user and source_assistant character-identical to corpus, corrected_answer non-empty, confidence in [0,1], quality_dimensions integers in 1-5, source_id globally unique across all 101 batches (1010 unique), aggregate train sequence a strict prefix of train.jsonl, validation 0/601)
- Repairs performed: none required
- Final schema/ad-hoc verification: PASS (VERIFY=PASS, exit 0)
- Manifest: MANIFEST.sha256 regenerated over all files in this directory except itself (173 entries, __pycache__ excluded); `sha256sum -c` verified with no failures
- Technical topics covered: serving-side evaluation methodology for mixed short-prompt / long-generation LLM traffic (scenario variants 101, 103-111). Shared rigorous frame across all ten: explicit non-claims (serving behavior only; no transfer across engine commit, model size, quantization, sequence-length regime or GPU SKU without a re-run), pinned clocks and persistence mode with MIG off so DVFS/thermal drift cannot be read as a treatment effect, open-loop Poisson arrivals from an off-host generator, metric definitions frozen before any run (TTFT from server ingress to first flushed token, TPOT excluding single-token outputs, queueing delay isolated as ingress->admit, throughput split into req/s + prefill tok/s + decode tok/s because prefill is compute-bound and decode memory-bandwidth-bound), four-cell S/M/L/XL length stratification with a seeded mixture, warmup discard, 5 fresh-process repeats with randomized arm order, and a >=6-point load ladder to 130% of saturation. Distinct falsifiable hypotheses per item: chunked-prefill budget 2048 vs unbounded with a chunk=512 mechanism arm that must move S-stratum P99 TTFT monotonically or the head-of-line-blocking explanation is refuted (01101); KV-exhaustion-vs-network tail attribution with an explicit base-rate computation and a client-vs-ingress TTFT gap discriminator (01103); decode roofline test with bytes-per-token pre-registered before measurement and the memory-to-compute-bound crossover reported rather than assumed (01104); prefill/decode disaggregation gated on measured KV transfer cost < 15% of the TTFT budget, with nvidia-smi topo -m recorded, achieved KV transfer bandwidth measured end to end instead of quoting nameplate link numbers, and the RDMA/GDR path required to be shown active via transport counters or perftest rather than inferred from a config flag (01105); a Little's-law queueing-dominance test where an intervention that only shifts time between queue and execution is the signature of a saturated server (01106); a self-invalidating prefix-cache experiment (shared vs token-length-matched randomized preamble) whose confirmation relabels prior TTFT as best-case rather than rescaling it (01107); TP=2 vs TP=4 at equal GPU count where an out-of-band nccl-tests small-message all-reduce latency increase is required to sustain the collective explanation, with NCCL_DEBUG=INFO ring/tree selection and NVLink-vs-PCIe placement dumped rather than assumed (01108); a 6-hour fragmentation soak separating cumulative-request from wall-clock degradation via a low-RPS control arm, with any restart mitigation required to be drain-gated and rolled one replica at a time (01109); a coordinated-omission test using intended-arrival timestamps whose consequence, if confirmed, is relabeling prior P99 as lower bounds, explicitly not rescaling by a constant factor because the bias is load-dependent (01110); and a length-aware two-queue admission policy gated on three pre-registered criteria plus a required adversarial arm of clients mis-declaring max_tokens, since declared output length is untrusted input and priority schemes starve the heaviest stratum (01111). Confounders instrumented across all ten: prefix-cache hit-rate divergence (>2pp voids the comparison), continuous-batching batch-size distribution over time, preemption/recompute/swap counts, KV utilization and allocator OOM retries, sampling/EOS behavior (output length is itself a treatment), NUMA placement, CPU governor, generator RTT. Pre-committed stop/rollback gates: any thermal or power throttle reason marks the sample invalid rather than noisy and aborts the arm; error rate > 0.5% aborts the load point; promotion requires the pre-registered primary metric to beat a noise band of 2x the control arm's across-repeat spread with no stratum regressing > 5% on P99 E2E; rollback within one deploy cycle on > 10% P99 E2E regression, stratum starvation, or increased OOM/preemption. Honest limit stated in every item: P99 over a few thousand requests requires a bootstrap or binomial confidence interval, and an effect smaller than that interval is not a result.
- Why all ten were marked `rewrite`: each source_assistant is a grading rubric ("Answer should state ...") rather than an answer, so supervising on it teaches meta-commentary about answers instead of the engineering reasoning itself; it also omits metric definitions, the open-loop load-generation requirement, per-stratum reporting, mechanism-level attribution with a named refutation condition, confounder instrumentation, and the numeric abort/rollback gates the prompts explicitly demand.
- Status caveat: these are PROVISIONAL model-generated second-opinion labels produced blind (no teacher-A artifact was read during this batch). They are not expert gold and they say nothing about any model's domain capability.

## Run 2026-08-17 batch 0100

- Batch file: results/train-batch-0100.jsonl
- Corpus range: train.jsonl lines 991-1000 (source IDs corpus-01091, corpus-01092, corpus-01093, corpus-01094, corpus-01095, corpus-01096, corpus-01097, corpus-01098, corpus-01099, corpus-01100 — corpus file order preserved exactly, no skips, no reordering)
- Progress: train 1000/5399, validation 0/601, total 1000/6000, remaining 5000
- Decisions: keep 0, rewrite 10, reject 0
- Initial schema/ad-hoc verification: PASS on first run (verify_batches.py over all 1000 aggregate train rows: every line JSON-parseable via physical-newline split, 10 rows in this batch, all 12 required fields present and no extras, teacher_lane/teacher_model/calibration_status/decision values valid, source_user and source_assistant character-identical to corpus, corrected_answer non-empty, confidence in [0,1], quality_dimensions integers in 1-5, source_id globally unique across all 100 batches, aggregate train sequence a strict prefix of train.jsonl, validation 0/601)
- Repairs performed: none required
- Final schema/ad-hoc verification: PASS (VERIFY=PASS, 0 errors)
- Manifest: MANIFEST.sha256 regenerated over all files in this directory except itself (172 entries); `sha256sum -c` verified all entries OK with no failures
- Technical topics covered: serving-side evaluation methodology for mixed short-prompt / long-generation LLM traffic (scenario variants 91-100). All ten share a common rigorous frame: explicit non-claims (serving behavior only, no transfer across engine build / model size / quantization / GPU SKU without re-run), pinned clocks via `nvidia-smi -lgc` with persistence mode on and MIG off so DVFS and thermal drift cannot be read as a treatment effect, open-loop Poisson arrivals with an off-host generator (closed-loop harnesses self-throttle and delete the worst samples), frozen metric definitions before any run (TTFT from server ingress to first flushed token, TPOT excluding single-token outputs, queueing delay isolated as ingress->admit, throughput split into req/s + prefill tok/s + decode tok/s because prefill is compute-bound and decode memory-bandwidth-bound), four-cell length stratification S/M/L/XL with a fixed seeded mixture, warmup discard, 5 fresh-process repeats with randomized arm order, and a >=6-step load ladder to 130% of saturation so the knee is visible. P99 is computed per stratum from raw per-request records and never pooled across strata or repeats (Simpson's paradox and hidden variance). Each item carries a distinct numeric falsifiable hypothesis with a null and a mechanism check: chunked-prefill budget 2048 vs unbounded with a chunk=512 arm required to show the predicted throughput reversal or the mechanism is refuted; KV-exhaustion-vs-network spike attribution by time-joining outliers against preemption counters plus a client-vs-ingress TTFT gap test; a decode-phase roofline test with the bytes-per-token prediction pre-registered before measurement; prefill/decode disaggregation gated on measured KV transfer cost staying under 15% of the TTFT budget with no unmeasured interconnect capability asserted; a Little's-law queueing-dominance test where an intervention that only shifts time between queue and execution is the signature of a saturated server; a self-invalidating prefix-cache experiment (shared vs randomized system preamble) designed to kill our own benchmark; TP=2 vs TP=4 at equal GPU count with out-of-band NCCL small-message latency required to explain the decode regression and `nvidia-smi topo -m` recorded rather than assumed; a fragmentation soak vs periodic-restart arm distinguishing cumulative-request from wall-clock degradation, with any restart mitigation required to be drain-gated; a coordinated-omission test whose consequence, if confirmed, is relabeling prior P99 figures as lower bounds rather than rescaling them; and a length-aware admission policy gated on three pre-registered criteria plus an adversarial mis-declared-max_tokens arm, since client-declared output length is untrusted input and priority schemes starve the heaviest users. Confounders instrumented across all ten: prefix-cache hit-rate divergence (>2pp voids the comparison), continuous-batching batch-size distribution over time, preemption/recompute and swap events, KV utilization and allocator OOM retries, tokenizer/sampling/EOS behavior, NUMA placement, CPU governor and generator RTT. Stop and rollback gates are pre-committed: abort any arm showing a thermal or power throttle reason (invalid sample, not noise), abort above 0.5% error rate, promote only on a pre-registered primary metric beating a noise band defined as 2x the control arm's across-repeat spread with no stratum regressing >5% on P99 E2E, and roll back within one deploy cycle on >10% P99 E2E regression, starvation, or increased OOM/preemption. Honest limits are stated: P99 on a few thousand requests needs a bootstrap/binomial confidence interval quoted, and effects smaller than that interval are not results.
- Why all ten were marked `rewrite`: each source_assistant is a grading rubric ("Answer should state ...") rather than an answer, so supervising on it teaches meta-commentary about answers instead of the engineering reasoning itself; it also omits metric definitions, the open-loop load-generation requirement, per-stratum reporting, mechanism-level attribution, confounder instrumentation, and the numeric abort/rollback gates the prompts explicitly demand.
- Status caveat: these are PROVISIONAL model-generated second-opinion labels, not expert gold, and they say nothing about any model's domain capability.

## Run 2026-08-17 batch 0099

- Batch file: results/train-batch-0099.jsonl
- Corpus range: train.jsonl lines 981-990 (source IDs corpus-01081, corpus-01082, corpus-01083, corpus-01084, corpus-01085, corpus-01086, corpus-01087, corpus-01088, corpus-01089, corpus-01090 — corpus file order preserved exactly, no skips, no reordering)
- Progress: train 990/5399, validation 0/601, total 990/6000, remaining 5010
- Decisions: keep 0, rewrite 10, reject 0
- Initial schema/ad-hoc verification: PASS on first run (verify_batches.py over all 990 aggregate train rows: every line JSON-parseable via physical-newline split, 10 rows in this batch, all 12 required fields present, teacher_lane/teacher_model/calibration_status/decision values valid, source_user and source_assistant character-identical to corpus, corrected_answer non-empty, confidence in [0,1], quality_dimensions integers in 1-5, source_id globally unique across all 99 batches, aggregate train sequence a strict prefix of train.jsonl, validation 0/601)
- Repairs performed: none required
- Final schema/ad-hoc verification: PASS (VERIFY=PASS, 0 errors)
- Manifest: MANIFEST.sha256 regenerated over all files in this directory except itself; `sha256sum -c` verified all entries OK with no failures
- Technical topics covered: serving-side evaluation methodology for mixed short-prompt / long-generation LLM traffic (scenario variants 81-90). All ten share a rigorous common frame — pinned clocks (`nvidia-smi -lgc`) to prevent DVFS drift being read as a treatment effect, unambiguous metric definitions (TTFT measured from client submit and therefore inclusive of queue wait, with queue wait exported separately from engine scheduler counters; TPOT = (last_token - first_token)/(out_tokens-1); input tok/s, output tok/s and req/s reported separately), four-cell length stratification with >=3000 requests per cell, one frozen hashed replayable trace, open-loop Poisson arrivals to avoid coordinated omission, and stationarity-tested warmup discard. Each carries a distinct numeric falsifiable hypothesis with its own mechanism check: SROF vs FIFO admission control including a mis-declared-max_tokens robustness arm (client-declared output length is untrusted input); chunked prefill budget sweep {512, 2048, 8192} validated against per-scheduler-step prefill/decode token composition rather than endpoint metrics alone; KV-capacity vs fragmentation (gpu_memory_utilization 0.85->0.92) requiring the TPOT tail improvement to be concentrated in exactly the requests preempted in baseline, with any canary OOM an automatic abort; TP=2 vs TP=4 with the NCCL term isolated via all_reduce_perf at the true per-step message sizes plus NCCL_DEBUG=INFO transport confirmation to rule out silent NVLink->PCIe fallback; prefix caching under 40% shared system prompts with byte-identical trace ordering (hit rate is order-dependent) and a temperature-0 token-identity correctness gate because caching bugs present as latency wins; KV-aware vs round-robin multi-replica routing measured per-replica with an explicit burst-herding check; speculative decoding swept across 0.3/0.6/0.8/0.95 x saturation with per-stratum acceptance rates and a greedy token-identity gate; autoscaling cold-start where P99 spikes are attributed by joining per-request TTFT to replica lifecycle events and the warm pool is charged against a pre-registered idle-GPU-minute budget; disaggregated prefill/decode with the KV transfer path instrumented (payload bytes, achieved bandwidth, RDMA port counters and retransmits) under a capacity-neutral equal-GPU-count comparison; and SLO-bound capacity headroom showing closed-loop saturation QPS overstates deployable capacity, with the queue-delay curve required as mechanism evidence and a pre-registered scale-out trigger. Confounders enumerated include KV preemption inflating TPOT tails, prefix-cache hit divergence, non-independence of per-request TPOT under continuous batching (hence paired-run bootstrap CIs rather than per-request t-tests), coordinated omission, thermal drift, co-tenancy, NUMA/CPU pinning and detokenizer CPU cost. Rollback gates are pre-committed and automated: revert on >10% P99 E2E regression in ANY stratum, >5% output tok/s drop, any preemption-rate rise above baseline, or >0.1% error/timeout rate, gated by a <=5% canary over a peak window.
- Why all ten were marked `rewrite`: each source_assistant is a grading rubric ("Answer should state ...") rather than an answer, so supervising on it teaches meta-commentary about answers instead of the engineering reasoning, and it omits metric definitions, the open-loop load-generation requirement, per-stratum reporting, mechanism-level attribution, and the numeric abort/rollback gates the prompts demand.
- Status caveat: these are PROVISIONAL model-generated second-opinion labels, not expert gold, and they say nothing about any model's domain capability.
- Blind-review integrity: no file under experiments/2026-08-14-teacher-a-corpus-calibration/ was read, opened or searched during this run. Only research/ai-infra-expert/corpus/train.jsonl was consulted for source text.

## Run 2026-08-17 batch 0098

- Batch file: results/train-batch-0098.jsonl
- Corpus range: train.jsonl lines 971-980 (source IDs corpus-01070, corpus-01071, corpus-01072, corpus-01073, corpus-01074, corpus-01076, corpus-01077, corpus-01078, corpus-01079, corpus-01080 — corpus file order preserved exactly, no skips, no reordering; the gap at corpus-01075 is pre-existing in the corpus and was NOT introduced by this lane)
- Progress: train 980/5399, validation 0/601, total 980/6000, remaining 5020
- Decisions: keep 0, rewrite 10, reject 0
- Initial schema/ad-hoc verification: PASS on first run (ad-hoc verifier over all 980 aggregate train rows: every line JSON-parseable via physical-newline split, 10 rows in this batch, all 12 required fields present, teacher_lane/teacher_model/calibration_status/decision values valid, source_user and source_assistant character-identical to corpus, corrected_answer non-empty, confidence in [0,1], quality_dimensions integers in 1-5, source_id globally unique across all 98 batches, aggregate train sequence a strict prefix of train.jsonl, validation 0/601)
- Repairs performed: none required
- Final schema/ad-hoc verification: PASS (VERIFY_PASS, 0 errors)
- Manifest: MANIFEST.sha256 regenerated over all files in this directory except itself; `sha256sum -c` verified all 167 entries OK with no failures
- Technical topics covered: serving-side evaluation methodology for mixed short-prompt / long-generation LLM traffic (scenario variants 70-80). Every rewrite fixes unambiguous metric definitions (TTFT measured from client submit and therefore inclusive of queue wait, with queue wait exported separately from engine scheduler counters; TPOT = (last_token - first_token)/(out_tokens-1); input tok/s, output tok/s and req/s reported separately because output tok/s is the decode-capacity metric; P99 computed per stratum with >=3000 requests per cell). Each explains mechanistically that prefill is O(prompt_len) compute-bound and sets TTFT while decode is HBM-bandwidth and KV-capacity bound and sets TPOT, so pooled averages are meaningless. Load generation is mandated open-loop Poisson from one frozen replayable trace with its hash recorded, explicitly rejecting closed-loop-only harnesses because of coordinated omission; warmup discard is by stationarity test and reported. Each item carries a distinct numeric falsifiable hypothesis: prefill/decode disaggregation (>=30% P99 TTFT cut, <=10% P99 TPOT cost), head-of-line blocking from long generations via max_num_seqs cap, KV-capacity vs FLOP-bound throughput via gpu_memory_utilization 0.85->0.92, length-aware admission control vs FIFO, chunked prefill at 2048 tokens, TP=2 vs TP=4 decode communication cost, preemption/recompute as the source of P99 spikes, prefix caching under a 40% shared-system-prompt workload, KV-aware vs round-robin multi-replica routing, and scheduler starvation of short requests under continuous batching. Confounders enumerated include KV preemption inflating TPOT tails, prefix-cache hit divergence between arms, non-independence of per-request TPOT under continuous batching (hence paired-run bootstrap statistics rather than per-request t-tests), thermal/DVFS drift (clocks locked with nvidia-smi -lgc), co-tenancy, NUMA/CPU pinning and detokenizer CPU cost. Evidence requirements cover engine commit and launch flags, trace hash, per-request timestamp traces, scheduler counters (num_waiting, num_running, KV utilization, preemption counts) and DCGM telemetry aligned to the run window. Rollback gates are pre-committed and automated: revert on >10% P99 E2E regression in any stratum, >5% output tok/s drop, any preemption-rate rise above baseline, or >0.1% error/timeout rate, gated by a <=5% canary over a peak window.
- Why all ten were marked `rewrite`: each source_assistant is a grading rubric ("Answer should state ...") rather than an answer, so supervising on it teaches meta-commentary about answers instead of the engineering reasoning, and it omits metric definitions, the open-loop load-generation requirement, per-stratum reporting, and the numeric abort/rollback gates the prompts demand.
- Status caveat: these are PROVISIONAL model-generated second-opinion labels, not expert gold, and they say nothing about any model's domain capability.
- Blind-review integrity: no file under experiments/2026-08-14-teacher-a-corpus-calibration/ was read, opened or searched during this run. Only research/ai-infra-expert/corpus/train.jsonl was consulted for source text.

## Run 2026-08-17 batch 0097

- Batch file: results/train-batch-0097.jsonl
- Corpus range: train.jsonl lines 961-970 (source IDs corpus-01059, corpus-01060, corpus-01062, corpus-01063, corpus-01064, corpus-01065, corpus-01066, corpus-01067, corpus-01068, corpus-01069 — corpus file order preserved exactly, no skips, no reordering; the gaps at corpus-01058 and corpus-01061 are pre-existing in the corpus and were NOT introduced by this lane)
- Progress: train 970/5399, validation 0/601, total 970/6000, remaining 5030
- Decisions: keep 0, rewrite 10, reject 0
- Initial schema/ad-hoc verification: PASS on first run (verify_batches.py over 970 aggregate train rows: every line JSON-parseable, 10 rows in this batch, all 12 required fields present, teacher_lane/teacher_model/calibration_status/decision values valid, source_user and source_assistant byte-identical to corpus, corrected_answer non-empty, confidence in [0,1], quality_dimensions integers in 1-5, source_id globally unique across all batches, aggregate train sequence a strict prefix of train.jsonl, validation 0/601)
- Repairs performed: none required
- Final schema/ad-hoc verification: PASS (VERIFY=PASS, 0 errors)
- Manifest: MANIFEST.sha256 regenerated over all files in this directory except itself; `sha256sum -c` verified 173/173 entries OK with no failures
- Technical topics covered: serving-side evaluation methodology for mixed short-prompt / long-generation LLM traffic (scenario variants 59-69). Each rewrite states hardware/version/power-cap assumptions and warns that silent clock or power-cap drift invalidates cross-run comparison; explains mechanistically why prefill (O(prompt_tokens) dense GEMM, compute-bound, sets TTFT) and decode (per-step weight + KV reads, HBM-bandwidth-bound, sets TPOT) must be reported separately, and why batch-size and chunked-prefill changes trade one against the other; fixes metric definitions (TTFT including queue wait with queue wait reported separately, TPOT = (E2E - TTFT)/(out_tokens-1), decode vs prefill throughput, goodput at SLO, P99 requiring >=2000 completed requests per arm); mandates open-loop Poisson arrivals from one frozen seeded trace with explicit rejection of closed-loop harnesses, 60 s warmup exclusion, >=3 randomized-order repeats with reported inter-run spread, and a load ladder to the SLO knee where capacity is defined as requests/s at P99-TTFT SLO crossing rather than peak throughput. Falsifiable H1 is numeric (chunked prefill at 512-token chunks cuts P99 TTFT >=25% while degrading median TPOT <=10%) with pre-committed rejection thresholds. Confounders and controls: tokenizer/length mismatch, client-side bottleneck, thermal/power drift, GPU co-tenancy, prefix-cache hit inflation, differing KV pressure. Instrumentation covers per-request traces, KV-block occupancy and peak GPU memory, and >=1 Hz scheduler counters. Rollback gate is pre-committed (revert on >10% P99 TTFT regression, >5% goodput drop, or any preemption-rate rise) under canary at production load. Category emphasis: Troubleshooting items order diagnosis before tuning by splitting queue wait from compute time; Performance Analysis items add an explicit HBM roofline estimate for decode step time and treat a large measured-vs-bound gap as scheduling overhead rather than bandwidth; System Design items require deciding colocated vs disaggregated prefill/decode up front and measuring KV-transfer bytes and achieved link bandwidth before claiming benefit.
- Why all ten were marked `rewrite`: each source_assistant is a grading rubric rather than an answer, so supervising on it teaches meta-commentary about answers instead of engineering reasoning, and it omits metric definitions, the open-loop load-generation requirement, and the numeric abort/rollback gates the prompts demand.
- Status caveat: these are PROVISIONAL model-generated second-opinion labels, not expert gold, and they say nothing about any model's domain capability.
- Blind-review integrity: no file under experiments/2026-08-14-teacher-a-corpus-calibration/ was read, opened or searched during this run. Only research/ai-infra-expert/corpus/train.jsonl was consulted for source text.

## Run 2026-08-17 batch 0096

- Batch file: results/train-batch-0096.jsonl
- Corpus range: train.jsonl lines 951-960 (source IDs corpus-01048 through corpus-01057, contiguous, corpus file order preserved exactly, no skips, no reordering)
- Progress: train 960/5399, validation 0/601, total 960/6000, remaining 5040
- Decisions: keep 0, rewrite 10, reject 0
- Initial schema/ad-hoc verification: PASS on first run (verify_batches.py over 960 aggregate train rows: every line JSON-parseable, 10 rows in this batch, all 12 required fields present, teacher_lane/teacher_model/calibration_status/decision values valid, source_user and source_assistant byte-identical to corpus, corrected_answer non-empty, confidence in [0,1], quality_dimensions integers in 1-5, source_id globally unique across all batches, aggregate train sequence a strict prefix of train.jsonl, validation 0/601)
- Repairs performed: none required
- Final schema/ad-hoc verification: PASS (VERIFY=PASS, 0 errors)
- Manifest: MANIFEST.sha256 regenerated over all files in this directory except itself; `sha256sum -c` verified 171/171 entries OK with no failures
- Technical topics covered: serving-side evaluation methodology for mixed short-prompt / long-generation LLM traffic (scenario variants 48-57). All ten rewrites pin the config fingerprint (model revision, quantization, TP/PP degree, KV dtype, max_num_seqs, max_num_batched_tokens, chunked-prefill flag, sampling params, seeds); separate prefill (compute-bound, scaling with total batched prompt tokens) from decode (HBM-bandwidth and KV-capacity bound); define TTFT decomposed into queueing delay and prefill service time, TPOT per request rather than from a fleet average, throughput as both output-token goodput and total processed tokens/s, and P99 with stated N and bootstrap CIs; mandate open-loop seeded arrival generation with an explicit rejection of closed-loop harnesses (which self-throttle and hide queueing collapse), a load ladder through the knee, warmup exclusion with a steady-state criterion, and >=3 randomized-order repeats plus a reproducing negative control. Confounders enumerated with controls: output-length drift, prefix/radix cache hit-rate inflation, client-side coordinated omission, thermal and clock drift, NUMA/MIG placement and co-tenancy, mid-run replica changes. Falsifiable H1 is numeric (queueing-dominated P99 => scheduler tuning moves P99 >20% while median TPOT stays within 5%), with the stated failure branch pointing to chunked prefill / prefill-decode disaggregation. Rollback gates are pre-committed (P99 regression >10%, TPOT regression >5%, any rise in preemption or error/truncation rate, HBM headroom below declared peak concurrency) with shadow -> canary -> tested single-config-hash revert. Category-specific emphasis: Performance Analysis items add the load-vs-latency knee attribution and a utilization (rho) sanity check treating predicted-vs-measured knee mismatch as evidence of KV eviction or preemption; System Design items add the harness as a versioned deliverable and disaggregated prefill / KV-transfer topology (Dynamo/Mooncake-style) with KV-transfer bytes and time as first-class metrics plus explicit detection of a silent RDMA/RoCE-to-TCP fallback; Troubleshooting items add an outside-in discrimination ladder (client -> TTFT decomposition -> KV thrash counters -> NCCL collective time and transport confirmation -> model/kernel path).
- Why all ten were marked `rewrite`: each source_assistant is a grading rubric rather than an answer, so supervising on it teaches meta-commentary about answers instead of the engineering reasoning, and it omits metric definitions, the open-loop load-generation requirement, and the numeric abort/rollback gates the prompts demand.
- Status caveat: these are PROVISIONAL model-generated second-opinion labels, not expert gold, and they say nothing about any model's domain capability.
- Blind-review integrity: no file under experiments/2026-08-14-teacher-a-corpus-calibration/ was read, opened or searched during this run. Only research/ai-infra-expert/corpus/train.jsonl was consulted for source text.

## Run 2026-08-17 batch 0095

- Batch file: results/train-batch-0095.jsonl
- Corpus range: train.jsonl lines 941-950 (source IDs corpus-01036, corpus-01037, corpus-01038, corpus-01040, corpus-01041, corpus-01042, corpus-01043, corpus-01045, corpus-01046, corpus-01047 — corpus file order preserved exactly, no skips, no reordering; the gaps at corpus-01039 and corpus-01044 are pre-existing in the corpus and were NOT introduced by this lane)
- Progress: train 950/5399, validation 0/601, total 950/6000, remaining 5050
- Decisions: keep 0, rewrite 10, reject 0
- Initial schema/ad-hoc verification: PASS on first run (verify_batches.py over 950 aggregate train rows: every line JSON-parseable, 10 rows in this batch, all 12 required fields present, teacher_lane/teacher_model/calibration_status/decision values valid, source_user and source_assistant byte-identical to corpus, corrected_answer non-empty, confidence in [0,1], quality_dimensions integers in 1-5, source_id globally unique across all batches, aggregate train sequence a strict prefix of train.jsonl, validation 0/601)
- Repairs performed: none required
- Final schema/ad-hoc verification: PASS (VERIFY=PASS, 0 errors)
- Manifest: MANIFEST.sha256 regenerated over all 163 files in this directory except itself (excluding __pycache__); `sha256sum -c` verified 163/163 entries OK with no failures
- Technical topics covered: serving-side evaluation methodology for mixed short-prompt / long-generation LLM traffic, continuing the same scenario family. All ten rewrites separate prefill (compute-bound, scaling with total batched prompt tokens) from decode (HBM-bandwidth and KV-capacity bound, scaling with concurrent sequences); define TTFT with queueing delay split out from prefill service time, TPOT per request, throughput reported as both request goodput and output tokens/s, and P99 per request class with bootstrap CIs and reported sample counts; state a falsifiable H0/H1 pair with numeric thresholds (queueing component grows >3x vs median TPOT grows >20%) that the traces must decide between; specify an open-loop seeded replayed trace, single-variable arms, a load ladder around the empirically located saturation point including one deliberately over-saturated point, warmup exclusion and >=3 trials; enumerate confounders (closed-loop self-throttling, output-length drift, cold CUDA graphs, power/thermal capping on A30-class GPUs, prefix-cache hit-rate inflation, co-tenancy) each with a control; and define abort thresholds (error rate >1%, P99 >2x SLO for >60 s, any OOM or replica restart) plus a canary-first rollout with a single config-hash revert whose rollback path is exercised before use. Category-specific emphasis was added per item: Performance Analysis items carry a roofline/bandwidth expected-value model (weights+KV bytes per token over achievable HBM bandwidth) so measured-vs-expected ratios are the finding; System Design items carry harness topology and per-class percentile discipline; Troubleshooting items carry a discrimination matrix for head-of-line blocking vs KV preemption vs admission queueing vs clock capping.
- Why all ten were marked `rewrite`: each source_assistant is a grading rubric ("Answer should state assumptions, a falsifiable hypothesis, measurements...") rather than an answer. Supervising on rubric text teaches meta-commentary about answers instead of the engineering reasoning, and it omits the numeric falsifiable hypothesis, the abort/rollback gates and the per-request-class percentile discipline the prompts explicitly demand.
- Blind-review integrity: no file under experiments/2026-08-14-teacher-a-corpus-calibration/ was read, opened or searched during this run. Only research/ai-infra-expert/corpus/train.jsonl was consulted for source text.
- Status: these corrected_answer records are PROVISIONAL teacher-B output from a blind review. They are NOT expert gold labels, have NOT been validated against a running system, and say nothing about any model's domain capability.

## Run 2026-08-17 batch 0094

- Batch file: results/train-batch-0094.jsonl
- Corpus range: train.jsonl lines 931-940 (source IDs corpus-01023, corpus-01024, corpus-01025, corpus-01026, corpus-01027, corpus-01029, corpus-01030, corpus-01031, corpus-01032, corpus-01033 — corpus file order preserved exactly, no skips, no reordering; the gaps at corpus-01022 and corpus-01028 are pre-existing in the corpus and were NOT introduced by this lane)
- Progress: train 940/5399, validation 0/601, total 940/6000, remaining 5060
- Decisions: keep 0, rewrite 10, reject 0
- Initial schema/ad-hoc verification: PASS on first run (940 aggregate rows checked: JSONL line-parseable, 10 rows in this batch, all 12 required fields present, teacher_lane/teacher_model/calibration_status/decision values valid, source_user and source_assistant byte-identical to corpus, corrected_answer non-empty, confidence in [0,1], quality_dimensions integers in 1-5, source_id globally unique, train sequence is a strict prefix of train.jsonl)
- Repairs performed: none required
- Final schema/ad-hoc verification: PASS (0 errors)
- Manifest: MANIFEST.sha256 regenerated over all 168 files in this directory except itself; `sha256sum -c` verified all entries OK
- Technical topics covered: serving-side evaluation methodology for mixed short-prompt / long-generation traffic. All ten items ask for an evaluation plan reporting TTFT, TPOT, throughput, queueing delay and P99 latency with an explicit falsifiable hypothesis and a controlled experiment. The rewritten answers separate prefill (compute-bound) from decode (memory-bandwidth and KV-cache-capacity bound) regimes; define each metric operationally including separating queueing delay from TTFT; state a falsifiable hypothesis about head-of-line blocking from long-request prefill and a chunked-prefill intervention with quantified accept/reject thresholds; specify an open-loop Poisson load generator with a seeded replayed trace and a load ladder to capture the latency knee; require warmup exclusion, >=3 repeated trials and bootstrap 95% CIs; enumerate confounders (KV preemption, output-length drift under sampling, prefix caching, client-side saturation, GPU power/thermal capping, co-tenancy) with a control for each; and define canary blast radius plus concrete rollback triggers (P99 regression >10%, error rate >0.5%, preemption rate >2x baseline, any OOM) with the requirement that the scheduler flag be runtime-togglable.
- Why all ten were marked `rewrite`: each source_assistant is a grading rubric ("Answer should state assumptions...") rather than an answer. Training on rubric text teaches meta-commentary about answers instead of the engineering reasoning itself, and it omits the falsifiable hypothesis, the rollback gates and the per-request-class percentile discipline that the prompts explicitly demand.
- Status: these corrected_answer records are PROVISIONAL teacher-B output from a blind review. They are NOT expert gold labels, have NOT been validated against a running system, and say nothing about any model's domain capability.

## Run 2026-08-17 batch 0093

- Batch file: results/train-batch-0093.jsonl
- Corpus range: train.jsonl lines 921-930 (source IDs corpus-01011, corpus-01012, corpus-01014, corpus-01015, corpus-01016, corpus-01017, corpus-01018, corpus-01019, corpus-01020, corpus-01021 — corpus file order preserved exactly, no skips, no reordering; the gap at corpus-01013 is pre-existing in the corpus and was NOT introduced by this lane)
- Progress: train 930/5399, validation 0/601, total 930/6000, remaining 5070
- Decisions: keep 0, rewrite 10, reject 0
- Initial schema check: PASS (verify_batches.py — 930 train records, 930 unique source_ids, batch size 10, all 12 required fields present, enum fields correct, corrected_answer non-empty, confidence in [0,1], quality_dimensions integers in [1,5], risks/evidence_required string arrays, source_user/source_assistant byte-identical to corpus, aggregate train sequence a strict prefix of train.jsonl)
- Repairs applied: none required; the batch passed verification on first run.
- Final schema check: PASS (VERIFY=PASS)
- Manifest: MANIFEST.sha256 regenerated over 166 files (everything in this directory except MANIFEST.sha256); `sha256sum -c` reported 166/166 OK, exit 0
- Technical topics covered: all ten items are scenario variants 11-21 of the same serving-evaluation
  design prompt (mixed short prompts / long generations; report TTFT, TPOT, throughput, queueing,
  P99, with an explicit falsifiable hypothesis and a controlled experiment). The source assistants
  are rubric stubs listing what an answer "should" contain rather than an answer, so every item was
  marked rewrite (instruction_coverage 2, operational_safety 2). The rewrites make the mechanism
  explicit: open-loop vs closed-loop load generation and why closed-loop structurally under-reports
  P99; prefill as compute-bound vs decode as memory-bandwidth-bound and the roofline floor implied
  by KV bytes read per step; queueing delay taken from engine scheduler metrics rather than inferred
  from TTFT; chunked prefill stated as a testable hypothesis with quantified accept/reject bounds
  (>=25% P99 TTFT reduction, <=10% median TPOT regression, <=5% output-throughput loss); load-ladder
  sweeps past the saturation knee with fixed warmup, interleaved A/B repeats and bootstrap CIs;
  confounders (prefix/radix cache hits, CUDA-graph capture, thermal and power throttling, client-side
  bottlenecks, truncated tails from timeouts) each paired with a control; and explicit rollback
  thresholds plus a safety requirement that saturation sweeps run on a canary with an automatic abort,
  never against production traffic. Category-specific tails were added: Troubleshooting items get a
  three-way cause separation (queueing vs prefill interference vs decode slowdown), Performance
  Analysis items get the roofline check, System Design items get prefill/decode disaggregation
  (Mooncake / NVIDIA Dynamo style) framed as a measured trade against KV-transfer cost over the fabric.
- Status: PROVISIONAL. This is one model's blind second-opinion pass. It is not expert gold, has not
  been validated against measured hardware behaviour, and says nothing about any trained model's
  domain capability. Blind protocol held: no teacher-A artifact was read while producing this batch.

## Run 2026-08-17 batch 0092

- Batch file: results/train-batch-0092.jsonl
- Corpus range: train.jsonl lines 911-920 (source IDs corpus-00999, corpus-01000, corpus-01001, corpus-01002, corpus-01004, corpus-01005, corpus-01006, corpus-01007, corpus-01009, corpus-01010 — corpus file order preserved exactly, no skips, no reordering; the gaps at corpus-01003 and corpus-01008 are pre-existing in the corpus and were NOT introduced by this lane)
- Progress: train 920/5399, validation 0/601, total 920/6000, remaining 5080
- Decisions: keep 0, rewrite 10, reject 0
- Initial schema check: PASS (verify_batches.py — 920 train records, 920 unique source_ids, batch size 10, all 12 required fields present, enum fields correct, corrected_answer non-empty, confidence in [0,1], quality_dimensions integers in [1,5], risks/evidence_required string arrays, source_user/source_assistant byte-identical to corpus, aggregate train sequence a strict prefix of train.jsonl)
- Repairs applied: none required. Both KV-cache byte counts were recomputed independently from the parsed parameters (2×48×1536×6×64×1 = 56623104 B = 0.052734 GiB; 2×56×2048×8×96×2 = 352321536 B = 0.328125 GiB) and matched the source values before writing.
- Final schema check: PASS (VERIFY=PASS)
- Manifest: MANIFEST.sha256 regenerated over 164 files (everything in this directory except MANIFEST.sha256); `sha256sum -c` reported all OK, exit 0
- Technical topics covered: two per-request K/V cache sizing cases (INT8 and BF16/FP16 KV under GQA),
  and eight serving-evaluation design/diagnosis/perf-analysis items on TTFT, TPOT, throughput,
  queueing and P99 under a mixed short-prompt / long-generation workload. The eight evaluation items
  share a templated source answer that is a rubric ("Answer should state...") rather than an answer;
  each was rewritten into a concrete plan with an explicit falsifiable hypothesis (mixed traffic
  degrades short-prompt P99 TTFT by >30% via batch-slot contention), three-arm controlled experiment
  (short-only / long-only / mixed) with load sweep, warmup policy and randomized repeated trials,
  phase-separated metrics (prefill-dominated TTFT vs decode-dominated TPOT), SLO goodput, queue-depth
  and KV-utilization telemetry, named confounders (coordinated omission from closed-loop load
  generation, prefix caching, chunked prefill, clock throttling, tokenizer mismatch), and a rollback
  gate at >20% short-prompt P99 TTFT regression or >5% goodput loss.
- Status: PROVISIONAL. Single-model blind second opinion, not expert gold, not adjudicated against
  teacher-A (no teacher-A file was read during this run), and it says nothing about any model's
  domain capability — it is corpus review output only.

## Run 2026-08-17 batch 0091

- Batch file: results/train-batch-0091.jsonl
- Corpus range: train.jsonl lines 901-910 (source IDs corpus-00988, corpus-00989, corpus-00990, corpus-00991, corpus-00992, corpus-00993, corpus-00994, corpus-00995, corpus-00997, corpus-00998 — corpus file order preserved exactly, no skips, no reordering; the gap at corpus-00996 is pre-existing in the corpus and was NOT introduced by this lane)
- Progress: train 910/5399, validation 0/601, total 910/6000, remaining 5090
- Decisions: keep 0, rewrite 10, reject 0
- Initial schema check: PASS (scripts/verify_batches.py — 910 train records, 910 unique source_ids, batch size 10, all 12 required fields present, enum fields correct, corrected_answer non-empty, confidence in [0,1], quality_dimensions integers in [1,5], risks/evidence_required string arrays, source_user/source_assistant byte-identical to corpus, aggregate train sequence a strict prefix of train.jsonl)
- Repairs applied: none required. Every rewritten answer's arithmetic was recomputed independently from the parsed parameters and asserted equal to the byte count stated in the source record before the batch was written; all 10 assertions passed.
- Final schema check: PASS (VERIFY=PASS)
- Manifest: MANIFEST.sha256 regenerated over 163 files (everything in this directory except MANIFEST.sha256); `sha256sum -c` reported all OK, exit 0
- Technical topics covered: per-request K/V cache sizing under grouped-query / multi-query attention
  (Calculation generator cases 488-498), spanning layers 24/32/40/48/56, kv_heads 2/4/6/8,
  head_dim 64/96/128, sequence length 1024-4096, and both BF16/FP16 and INT8 KV dtypes. Each rewrite
  keeps the correct closed-form formula but adds the operational boundary conditions the source omits:
  PagedAttention block rounding, concurrency multiplication against the (GPU memory - weights -
  activations - overhead) budget, tensor-parallel KV-head replication when kv_heads < TP degree,
  prefix-cache and beam-search effects, speculative-decoding/chunked-prefill staging buffers, and for
  INT8 the per-block scale/zero-point overhead plus dequant cost. Each answer states a falsifiable
  doubling prediction, the specific telemetry needed to test it (engine KV block usage,
  torch.cuda.memory_allocated deltas, nvidia-smi, model config.json), and a ~15% over-estimate
  rollback gate before raising max_num_seqs / max_model_len.
- Status: PROVISIONAL. This is a single-model blind second opinion, not expert gold, not adjudicated
  against teacher-A (no teacher-A file was read during this run), and it says nothing about any
  model's domain capability.

## Run 2026-08-17 batch 0090

- Batch file: results/train-batch-0090.jsonl
- Corpus range: train.jsonl lines 891-900 (source IDs corpus-00976, corpus-00977, corpus-00979, corpus-00980, corpus-00981, corpus-00982, corpus-00983, corpus-00984, corpus-00985, corpus-00987 — corpus file order preserved exactly, no skips, no reordering; the gaps at corpus-00978 and corpus-00986 are pre-existing in the corpus and were NOT introduced by this lane)
- Progress: train 900/5399, validation 0/601, total 900/6000, remaining 5100
- Decisions: keep 0, rewrite 10, reject 0
- Initial schema check: PASS (ad-hoc verifier /tmp/tb_verify.py — 900 train records, 900 unique source_ids, batch size 10, all 12 required fields present, teacher_lane/teacher_model/calibration_status/decision enums correct, corrected_answer non-empty, confidence in [0,1], quality_dimensions integers in [1,5], risks/evidence_required string arrays, source_user/source_assistant byte-identical to corpus, aggregate train sequence a strict prefix of train.jsonl)
- Repairs applied: none required (first-run pass). Arithmetic in each rewritten answer was cross-checked against the byte count asserted in the source record before the batch was written.
- Final schema check: PASS
- Manifest: MANIFEST.sha256 regenerated over 161 files (everything in this experiment directory except MANIFEST.sha256 itself); `sha256sum -c` reported all OK, exit 0
- Technical topics covered: per-request K/V cache sizing under grouped-query / multi-query attention
  (Calculation generator cases 476-487), spanning layers 24/32/40/48/56, kv_heads 2/4/6/8,
  head_dim 64/96/128, sequence length 1024-4096, and both BF16/FP16 and INT8 KV dtypes. Each
  rewrite adds the GQA materialisation mechanism, a per-token planning constant, paged-allocator
  block-rounding effects, tensor-parallel sharding vs replication when kv_heads < TP, INT8 scale
  metadata overhead, a falsifiable linear-scaling prediction with a ~10% deviation tolerance,
  the evidence needed (config.json fields, engine KV-pool accounting, torch.cuda
  allocated-vs-reserved, actual kv_cache_dtype), and a rollback threshold (any KV OOM/preemption
  or >20% p99 TTFT regression reverts max_num_seqs / max_model_len).
- Status caveat: these teacher-B outputs are PROVISIONAL model-generated second opinions, not
  expert gold labels, and they say nothing about any trained model's domain capability.

## Run 2026-08-17 batch 0089

- Batch file: results/train-batch-0089.jsonl
- Corpus range: train.jsonl lines 881-890 (source IDs corpus-00965, corpus-00966, corpus-00967, corpus-00969, corpus-00970, corpus-00971, corpus-00972, corpus-00973, corpus-00974, corpus-00975 — corpus file order preserved exactly, no skips, no reordering; note the corpus itself has a gap at corpus-00968, which was NOT introduced by this lane)
- Progress: train 890/5399, validation 0/601, total 890/6000, remaining 5110
- Decisions: keep 0, rewrite 10, reject 0
- Initial schema check: PASS (ad-hoc verifier scripts/verify_batches.py — 890 train records, 890 unique source_ids, all 12 required fields present, teacher_lane/teacher_model/calibration_status/decision enums correct, corrected_answer non-empty, confidence in [0,1], source_user/source_assistant byte-identical to corpus, aggregate train sequence is a strict prefix of train.jsonl, batch numbering contiguous from 0001)
- Repairs applied: none required (first-run pass)
- Final schema check: PASS
- Manifest: MANIFEST.sha256 regenerated over 154 files (all files in this experiment directory except MANIFEST.sha256 itself and scripts/__pycache__); `sha256sum -c` reported 154/154 OK, exit 0
- Technical topics covered: per-request K/V cache sizing under grouped-query attention, same
  Calculation generator family (cases 465-475) as the previous batch, varying layers
  (24/32/40/48/56), kv_heads (2/4/6/8), head_dim (64/96/128), sequence length (1024-4096)
  and KV element width (BF16/FP16 = 2 B vs INT8 = 1 B). All ten source products were
  re-derived independently in this run (2 x L x S x H_kv x D x B) and every byte total and
  every GiB rounding matched the source answer exactly, so the rewrite decision is driven
  by incompleteness and operational risk, not by arithmetic error. Each corrected answer
  adds: the mechanism (only kv_heads are materialised under GQA/MQA because query heads are
  broadcast, hence linearity in S), the per-token KV cost as the real capacity-planning
  unit and the resident-token/concurrency budget it implies, paged-allocator block round-up
  (ceil(S/block)*block), prefix/radix cache sharing and speculative-decode draft branches as
  effects moving residency in opposite directions, the separate line items (weights,
  activations, CUDA graph pools, NCCL buffers, fragmentation) that must not be conflated
  with KV, and tensor-parallel behaviour including the replication case when kv_heads < TP.
  For the four INT8 items the answer quantifies the omitted scale/zero-point surcharge
  explicitly (2 x L x S x H_kv x 2 bytes for per-head-per-token FP16 scales, ~1.5-3% here)
  and marks the source figure as a lower bound. Every answer carries a falsifiable
  prediction (doubling S must double measured KV allocation), a concrete evidence list
  (config.json fields, engine KV dtype flag, engine startup KV-blocks log line, measured
  VRAM delta, TP degree) and a rollback gate (>20% overshoot at target concurrency stops
  the rollout and forces max_num_seqs / max_model_len reduction before KV-pool preemption
  wrecks tail latency).
- Status caveat: this output is PROVISIONAL teacher-B model review, produced blind without
  any access to the teacher-A lane. It is not expert gold, has not been validated against
  a running engine, and says nothing about any model's domain capability. Agreement
  analysis against teacher-A is a separate downstream step outside this lane.

## Run 2026-08-17 batch 0088

- Batch file: results/train-batch-0088.jsonl
- Corpus range: train.jsonl lines 871-880 (source IDs corpus-00955 … corpus-00964, contiguous; corpus file order preserved exactly, no skips, no reordering)
- Progress: train 880/5399, validation 0/601, total 880/6000, remaining 5120
- Decisions: keep 0, rewrite 10, reject 0
- Initial schema check: PASS (ad-hoc verifier — 880 train records, 880 unique source_ids, all 12 fields present, enum values correct, corrected_answer non-empty, confidence in [0,1], source_user/source_assistant byte-identical to corpus, aggregate train sequence is a strict prefix of train.jsonl)
- Repairs applied: none required (first-run pass)
- Final schema check: PASS
- Manifest: MANIFEST.sha256 regenerated over 158 files; `sha256sum -c` PASS
- Technical topics covered: per-request K/V cache sizing under grouped-query attention.
  All ten items are the same generator family (Calculation cases 455-464) varying layers
  (24/32/40/48/56), kv_heads (2/4/6/8), head_dim (64/96/128), sequence length
  (1024-4096) and KV element width (BF16/FP16 = 2 B vs INT8 = 1 B). Source arithmetic was
  re-derived independently for all ten and matched in every case, so the byte/GiB totals
  are correct; the rewrite decision is driven by incompleteness rather than error. Each
  corrected answer adds: the mechanism (why only kv_heads are materialised under GQA and
  why growth is strictly linear in sequence length), the per-token KV cost as the actual
  capacity-planning unit and the concurrency it implies, paged-allocator block rounding
  (ceil(S/block)*block plus block-table overhead), prefix-cache/copy-on-write sharing,
  transient KV from speculative decoding / beam search / chunked prefill, and for INT8 KV
  the per-block scale/zero-point surcharge plus the requirement for a separate accuracy
  eval gate. Each answer states a falsifiable check against engine KV block-occupancy
  telemetry, the evidence needed (model config, KV dtype actually in effect at runtime,
  memory telemetry), and a rollback gate (>10% deviation from estimate, or held-out
  accuracy regression beyond tolerance when enabling low-precision KV).
- Status: PROVISIONAL. These are one model's blind second-opinion reviews, not expert gold
  labels, and they say nothing about any trained model's domain capability. Produced blind:
  no teacher-A artifact was read while generating this batch.

## Run 2026-08-17 batch 0087

- Batch file: results/train-batch-0087.jsonl
- Corpus range: train.jsonl lines 861-870 (source IDs corpus-00945 … corpus-00954, contiguous; corpus file order preserved exactly, no skips, no reordering)
- Progress: train 870/5399, validation 0/601, total 870/6000, remaining 5130
- Decisions: keep 0, rewrite 10, reject 0
- Initial schema check: PASS (verify_batches.py — 870 train records, 870 unique source_ids, all 12 fields present, source_user/source_assistant byte-identical to corpus, prefix property holds)
- Repairs applied: none required
- Final schema check: PASS
- Manifest: MANIFEST.sha256 regenerated over 157 files; `sha256sum -c` PASS
- Technical topics covered: per-request K/V cache sizing for grouped-query attention —
  the 2 x layers x seq_len x kv_heads x head_dim x bytes_per_value formula, BF16/FP16 vs
  INT8 KV element width, per-token KV cost as the concurrency planning unit, paged-allocator
  block rounding and internal fragmentation, prefix/prompt-cache retention inflating
  steady-state occupancy, tensor-parallel KV replication when kv_heads < TP degree, and
  MLA / compressed-latent-KV as a case where the formula does not apply. Each rewrite adds a
  falsifiable occupancy prediction (within ~5-10% of N x bytes), the evidence needed to test it
  (KV utilisation gauge, torch/nvidia-smi memory summary, preemption/recompute counters,
  model config fields), and a rollback threshold (>1.25x predicted occupancy or preemption at
  target concurrency). All 10 source answers were arithmetically correct but were rewritten for
  missing boundary conditions and operational safety, not for numerical error.
- Blindness: this batch was produced without opening any file under
  experiments/2026-08-14-teacher-a-corpus-calibration/. teacher-A corrected answers were not read.
- Status: PROVISIONAL. This is a single-model second-opinion pass, not expert gold, and it is
  not evidence of any model's domain capability.

## Run 2026-08-17 batch 0086

- Batch file: results/train-batch-0086.jsonl
- Corpus range: train.jsonl lines 851-860 (source IDs corpus-00935 … corpus-00944, contiguous; corpus file order preserved exactly, no skips, no reordering)
- Progress: train 860/5399, validation 0/601, total 860/6000, remaining 5140
- This run: 10 items
- Decisions: keep 0, rewrite 10, reject 0
- Initial schema check: VERIFY_PASS on first attempt (train=860/5399, validation=0/601, total=860)
- Repairs performed: none required
- Final schema check: VERIFY_PASS (JSONL line-parseable, 10 records, all 12 required fields present, teacher_lane/teacher_model/calibration_status/decision values valid, source_user and source_assistant byte-identical to corpus, corrected_answer non-empty, confidence in [0,1], source_id globally unique, aggregated train sequence is a strict prefix of train.jsonl)
- Manifest: MANIFEST.sha256 regenerated over all files except itself; `sha256sum -c` → all OK, 0 failures
- Technical topics covered: KV-cache capacity sizing for grouped-query attention decoders — the 2 × layers × seq_len × kv_heads × head_dim × bytes_per_value identity, BF16/FP16 vs INT8 KV element width, per-token KV cost as the actual concurrency-planning quantity, paged-allocator block rounding and internal fragmentation, prefix-cache retention inflating steady-state occupancy, the kv_heads ≥ TP-degree limit on tensor-parallel KV sharding, MLA/latent-KV architectures invalidating the formula, and preemption/recompute counters plus HBM headroom as the rollback gate.
- Review note: every source answer's arithmetic was independently recomputed and matched; all 10 were marked `rewrite` (not `reject`) because the numbers are right but the answers stop at a raw byte count, omitting per-token cost, allocator overhead, TP sharding limits and any INT8 accuracy caveat — i.e. correct but operationally insufficient.
- Blind-review discipline: no file under experiments/2026-08-14-teacher-a-corpus-calibration/ was read, opened, grepped or listed while producing this batch. Only research/ai-infra-expert/corpus/train.jsonl was consulted.
- Status: PROVISIONAL. These corrected answers are a second-opinion model review, not expert gold labels, and they say nothing about any trained model's domain capability.

## Run 2026-08-17 batch 0085

- Batch file: results/train-batch-0085.jsonl
- Corpus range: train.jsonl lines 841-850 (source IDs corpus-00924, corpus-00925, corpus-00926, corpus-00927, corpus-00928, corpus-00930, corpus-00931, corpus-00932, corpus-00933, corpus-00934 — corpus file order preserved exactly, no skips, no reordering; note the corpus itself skips corpus-00929)
- Progress: train 850/5399, validation 0/601, total 850/6000, remaining 5150
- This run: 10 items
- Decisions: keep 0, rewrite 10, reject 0
- Initial schema check: `python3 experiments/2026-08-17-teacher-b-corpus-review/verify.py` → VERIFY_PASS on first attempt (train=850/5399, validation=0/601, total=850)
- Repairs performed: none required
- Final schema check: VERIFY_PASS (JSONL line-parseable, 10 records, all 12 required fields present, teacher_lane/teacher_model/calibration_status/decision values valid, source_user and source_assistant byte-identical to corpus, corrected_answer non-empty, confidence in [0,1], source_id globally unique, aggregated train sequence is a strict prefix of train.jsonl)
- Manifest: MANIFEST.sha256 regenerated over all files except itself; `sha256sum -c` → 153/153 OK, 0 failures
- Blind-review discipline: no file under experiments/2026-08-14-teacher-a-corpus-calibration/ was read, opened, grepped or listed while producing this batch. Only research/ai-infra-expert/corpus/train.jsonl was consulted.

### Technical topics covered by this batch

All ten items are `Calculation` / `numeric` cases on the same axis: per-request K/V cache
footprint for a grouped-query-attention decoder, spanning 24–56 layers, 2–8 KV heads,
head_dim 64/96/128, sequence lengths 1024–4096, and both BF16/FP16 (2 B) and INT8 (1 B)
KV element widths. The arithmetic in every source answer was re-derived independently and
matched, so all rewrites are additive rather than corrective: technical_correctness 5,
instruction_coverage 4, operational_safety 3.

The reason for `rewrite` rather than `keep` is that the source answers stop at the closed
form and a one-line caveat. For an infrastructure copilot that is not sufficient — a capacity
planner acting on the bare number will over-subscribe HBM. Each corrected_answer therefore adds:
the mechanism (why the product form holds under GQA and why it is strictly linear in sequence
length), the derived per-token cost `2 × layers × kv_heads × head_dim × bytes_per_value` as
the quantity that actually drives concurrency budgeting, and explicit boundary conditions —
PagedAttention block-size rounding and internal fragmentation, INT8/FP8 per-block scale
metadata (~1–3 % overhead and an accuracy-affecting change, not a free win), the tensor-parallel
sharding limit where kv_heads < TP degree causes replication so per-GPU KV stops shrinking,
and MLA/compressed-latent architectures where the formula does not apply at all.

Each answer states a falsifiable prediction (measured KV-pool occupancy at N concurrent
requests should land within ~5–10 % above N × bytes; a larger gap must be attributed to
fragmentation, prefix-cache retention or quantisation metadata by measurement, not assumption),
the evidence needed to check it (model config fields, vLLM `GPU KV cache size` / num_gpu_blocks
startup accounting, `nvidia-smi` or `torch.cuda.memory_summary()` steady-state HBM), and a
rollback gate (keep ≥10 % HBM headroom and zero scheduler preemption/recompute events, else
revert the max_num_seqs / max_model_len change before touching KV quantisation).

**Status caveat:** these outputs are *provisional* teacher-B second opinions produced by a
single model under blind conditions. They are not expert gold labels, they have not been
adjudicated against teacher-A, and they say nothing about the domain capability of any trained
model. Agreement analysis against teacher-A is a separate, later step outside this worker's scope.

## Run 2026-08-17 batch 0084

- Batch file: results/train-batch-0084.jsonl
- Corpus range: train.jsonl lines 831-840 (source IDs corpus-00914 … corpus-00923, corpus file order preserved exactly, no skips, no reordering)
- Progress: train 840/5399, validation 0/601, total 840/6000, remaining 5160
- Decisions: keep 0, rewrite 10, reject 0
- Initial schema check: PASS on first run (ad-hoc verifier scripts/verify_batches.py — per-line JSONL parse, batch count 10, the 12 required fields, enum values for teacher_lane / teacher_model / calibration_status / decision, quality_dimensions integers 1-5, risks and evidence_required as string arrays, character-exact source_user and source_assistant against research/ai-infra-expert/corpus/train.jsonl, non-empty corrected_answer, confidence in [0,1], global source_id uniqueness across all 84 batches, contiguous batch numbering, and strict train/validation corpus-prefix ordering)
- Repairs performed: none required. No corpus file, no earlier batch, no benchmark generation and no teacher-A artifact was read or modified.
- Final schema check: PASS (train 840 records prefix-checked, validation 0, total 840, VERIFY=PASS)
- Manifest: MANIFEST.sha256 regenerated over every file in this directory except itself; `sha256sum -c` reported 151 OK lines and 0 failures.
- Technical topics covered: single-request K/V cache sizing for GQA/MQA decoder stacks — layer depths 24/32/40/48/56, KV head counts 2/4/6/8, head dims 64/96/128, contexts 1024-4096 tokens; seven BF16/FP16 rows (2 B/value) and three INT8 rows (corpus-00915, corpus-00918, corpus-00921, 1 B/value). Every source byte total and GiB value was independently recomputed from 2 x layers x seq_len x kv_heads x head_dim x bytes_per_value before writing; all ten matched the source exactly, so no row was rejected. All ten are nonetheless `rewrite` because the source stops at formula-plus-number with one boilerplate caveat. Each rewrite adds: the meaning of the leading factor 2 (K and V tensors, not batch); per-token marginal bytes as the quantity that composes into paged-attention block-pool sizing and max_num_seqs; GiB (2^30) vs GB (10^9) disambiguation with both numbers shown; the GQA/MQA rule that kv_heads rather than query heads sets retained width; the architectures that break the uniform-layer assumption (sliding-window, MLA/latent-KV, cross-attention, hybrid SSM); the INT8 scale/zero-point metadata the payload-only formula omits; paged-allocator block round-up as ceil(S/block_size) x block_size x per-token bytes plus block-table metadata; the distinction between prompt length and total live context; the capacity composition (HBM minus weights, activations/workspace, fragmentation headroom, divided by per-request KV); an explicit evidence list (model config fields, engine startup log for cache dtype and block size, measured device memory at fixed concurrency, KV utilization and preemption counters); and a rollback gate — roll back max_num_seqs / max_model_len to the last validated values on preemption/recompute events, KV utilization above ~90%, or p99 TTFT SLO regression.
- Blind-review compliance: no file under experiments/2026-08-14-teacher-a-corpus-calibration/ was read, listed, grepped or searched at any point in this run. The only inputs were source_user and source_assistant from research/ai-infra-expert/corpus/train.jsonl.
- Status caveat: PROVISIONAL teacher-B model output. Not expert gold labels, not human-validated, and not evidence of any trained model's domain capability.

## Run 2026-08-17 batch 0083

- Batch file: results/train-batch-0083.jsonl
- Corpus range: train.jsonl lines 821-830 (source IDs corpus-00904, corpus-00905, corpus-00906, corpus-00907, corpus-00908, corpus-00909, corpus-00910, corpus-00911, corpus-00912, corpus-00913 — corpus file order preserved exactly, no skips, no reordering)
- Progress: train 830/5399, validation 0/601, total 830/6000, remaining 5170
- Decisions: keep 0, rewrite 10, reject 0
- Initial schema check: PASS on first run (ad-hoc verifier tools/verify.py — per-line JSONL parse, batch count 10, the 12 required fields, enum values for teacher_lane / teacher_model / calibration_status / decision, quality_dimensions integers in 1-5, risks and evidence_required as string arrays, character-exact source_user and source_assistant against research/ai-infra-expert/corpus/train.jsonl, non-empty corrected_answer, confidence in [0,1], global source_id uniqueness across all 83 batches, and strict train/validation corpus-prefix ordering)
- Repairs performed: none required. No corpus file, no earlier batch, no benchmark generation and no teacher-A artifact was read or modified.
- Final schema check: PASS (train 830 records prefix-checked, validation 0, total 830, VERIFY_PASS)
- Manifest: MANIFEST.sha256 regenerated over every file in this directory except itself; `sha256sum -c` reported all OK, 0 failures.
- Technical topics covered: single-request K/V cache sizing for GQA/MQA decoder stacks — layer depths 24/32/40/48/56, KV head counts 2/4/6/8, head dims 64/96/128, contexts 1024-4096 tokens, seven BF16/FP16 rows (2 B/value) and three INT8 rows (corpus-00906, corpus-00909, corpus-00912, 1 B/value). Every source byte total and GiB conversion was independently recomputed from 2 x layers x seq_len x kv_heads x head_dim x bytes_per_value before writing (tools/chk.py); all ten matched the source exactly, hence technical_correctness 5 and no rejections. All ten are nonetheless `rewrite` because the source stops at formula-plus-number with one boilerplate caveat and states no mechanism or validity domain. Each rewrite adds: the decode-time mechanism and why kv_heads rather than query heads sets the retained width under GQA/MQA; per-layer and per-token marginal byte cost (the quantity that actually composes into admission control and max_num_seqs); decimal-GB vs binary-GiB disambiguation; paged-attention block round-up with an explicit ceil(S/16) block count; the exclusion list (weights, activations/workspace, CUDA context ~300-600 MiB/GPU, NCCL buffers, allocator fragmentation, prefix-cache retention, speculative decoding and beam width); tensor-parallel sharding behaviour and the TP > kv_heads boundary where KV heads must be replicated and per-GPU KV stops shrinking; for the INT8 rows the per-token/per-head scale and zero-point metadata that the 1 B/value assumption omits, quantified against the reported total, plus the point that INT8 KV is an accuracy decision and not only a capacity win; a falsifiable prediction (single request of S tokens raises engine-reported KV usage by the computed GiB +/- one allocator block, scaling linearly with concurrency until pool exhaustion) with the exact evidence list needed to test it; and a rollback gate — size limits from measured steady state, not arithmetic, and roll back on KV pool utilization above ~90%, preemption/recompute events in engine logs, or p99 TTFT SLO regression, since KV exhaustion degrades as queueing rather than clean OOM.
- Blind-review compliance: no file under experiments/2026-08-14-teacher-a-corpus-calibration/ was read, listed, grepped or searched at any point in this run. The only inputs were source_user and source_assistant from research/ai-infra-expert/corpus/train.jsonl.
- Status caveat: PROVISIONAL teacher-B model output. Not expert gold labels, not human-validated, and not evidence of any trained model's domain capability.

## Run 2026-08-17 batch 0082

- Batch file: results/train-batch-0082.jsonl
- Corpus range: train.jsonl lines 811-820 (source IDs corpus-00892, corpus-00894, corpus-00895, corpus-00896, corpus-00898, corpus-00899, corpus-00900, corpus-00901, corpus-00902, corpus-00903 — corpus file order preserved exactly, no skips or reordering; the gaps at 00893 and 00897 are pre-existing gaps in the corpus ID sequence, not skipped rows)
- Progress: train 820/5399, validation 0/601, total 820/6000, remaining 5180
- Decisions: keep 0, rewrite 10, reject 0
- Initial schema check: PASS on first run (ad-hoc verifier /tmp/tb_verify.py — per-line JSONL parse, trailing-newline check, batch count 10, the 12 required fields, enum values for teacher_lane / teacher_model / calibration_status / decision, quality_dimensions integers in 1-5, risks and evidence_required as string arrays, character-exact source_user and source_assistant against research/ai-infra-expert/corpus/train.jsonl, non-empty corrected_answer, confidence in [0,1], global source_id uniqueness across all batches, and strict train/validation corpus-prefix ordering)
- Repairs performed: none required. No corpus file, no earlier batch, no benchmark generation and no teacher-A artifact was read or modified.
- Final schema check: PASS (train 820/5399 prefix-checked, validation 0/601, total 820, VERIFY_PASS)
- Manifest: MANIFEST.sha256 regenerated over every file in this directory except itself; `sha256sum -c` reported all OK, 0 failures.
- Technical topics covered: single-request K/V cache sizing for GQA/MQA decoder stacks — layer depths 24-56, KV head counts 2-8, head dims 64/96/128, contexts 1024-4096 tokens, BF16/FP16 (2 B/value) and INT8 (1 B/value) KV dtypes. Every source byte total and GiB conversion was independently recomputed from 2 x layers x seq_len x kv_heads x head_dim x bytes_per_value; all ten matched exactly, so technical_correctness is 4 and no row was rejected. All ten were nonetheless marked `rewrite` because the source answers stop at formula-plus-number with one boilerplate caveat and never state a validity domain. The rewrites add (a) the explicit GQA/MQA rule that the replication factor is num_key_value_heads rather than query heads; (b) per-token marginal byte cost and the derived tokens-per-GiB figure, which is the quantity that actually composes into max_num_seqs and concurrency planning; (c) the architectures that break the linear model — sliding-window/chunked local attention, MLA/latent-KV compression, cross-layer KV sharing, hybrid attention/SSM stacks; (d) tensor-parallel sharding behaviour including the case where kv_heads is not divisible by TP degree and ranks replicate; (e) paged-allocator block round-up with padding waste bounded at (block_size - 1) x per-token bytes; (f) for the three INT8 rows, the per-block scale/zero-point metadata overhead of roughly 1.5-3% that the raw formula omits; (g) explicit exclusions — allocator reserve and fragmentation, retained prefix-cache blocks, speculative-decoding draft state, CUDA-graph capture buffers, prefill activation workspace, NCCL buffers; (h) a three-step falsifiable measurement protocol (engine-reported KV block usage for one controlled request, torch.cuda.memory_allocated deltas across prefill, concurrency sweep to first preemption); and (i) a rollback gate requiring that a >15% analytic-vs-measured gap invalidates the sizing and forces max_num_seqs / max_model_len back to the last known-good value.
- Blind-review compliance: no file under experiments/2026-08-14-teacher-a-corpus-calibration/ was read, listed, grepped or searched at any point in this run. The only inputs were source_user and source_assistant from research/ai-infra-expert/corpus/train.jsonl.
- Status caveat: PROVISIONAL teacher-B model output. Not expert gold labels, not human-validated, and not evidence of any trained model's domain capability.

## Run 2026-08-17 batch 0081

- Batch file: results/train-batch-0081.jsonl
- Corpus range: train.jsonl lines 801-810 (source IDs corpus-00880, corpus-00881, corpus-00882, corpus-00883, corpus-00884, corpus-00886, corpus-00887, corpus-00888, corpus-00890, corpus-00891 — corpus file order preserved exactly, no skips or reordering; the gaps at 885/889 are pre-existing gaps in the corpus ID sequence, not skipped rows)
- Progress: train 810/5399, validation 0/601, total 810/6000, remaining 5190
- Decisions: keep 0, rewrite 10, reject 0
- Initial schema check: PASS on first run. Two independent checkers were run — an ad-hoc verifier written this round (/tmp/tb_verify.py) and the in-repo scripts/verify_batches.py. Both cover per-line JSONL parse, batch count 10, the 12 required fields, enum values for teacher_lane / teacher_model / calibration_status / decision, quality_dimensions integers in 1-5, risks and evidence_required as arrays, character-exact source_user and source_assistant against the corpus, non-empty corrected_answer, confidence in [0,1], global source_id uniqueness, and strict train/validation corpus-prefix ordering.
- Repairs performed: none required. No corpus file, no earlier batch, and no teacher-A artifact was read or modified.
- Final schema check: PASS (train_processed=810, validation_processed=0, total=810, ERRORS 0)
- Manifest: MANIFEST.sha256 regenerated over every file in this directory except itself; `sha256sum -c` reported all OK, 0 failures.
- Technical topics covered: single-request K/V cache sizing for GQA/MQA decoder stacks — layer depths 24-56, KV head counts 2-8, head dims 64/96/128, contexts 1024-3584 tokens, BF16/FP16 (2 B/value) and INT8 (1 B/value) KV dtypes. Every source byte total was independently recomputed from 2 x layers x seq_len x kv_heads x head_dim x bytes_per_value; all ten matched exactly, so technical_correctness is 4 and no row was rejected. All ten were nonetheless marked `rewrite`: the source answers give formula-plus-number with a single boilerplate caveat and never state the validity domain. The rewrites add (a) the explicit GQA/MQA rule that the replication factor is num_key_value_heads, not query heads; (b) per-token marginal byte cost, the quantity that actually composes into concurrency and max_num_seqs planning; (c) the architectures that break the linear formula — sliding-window/chunked local attention, MLA/latent-KV compression, cross-layer KV sharing, hybrid Mamba-attention stacks; (d) tensor-parallel sharding behaviour including the non-divisible kv_heads replication case; (e) paged-allocator block round-up with the padding waste bounded at (block_size - 1) x per-token bytes; (f) exclusions — allocator reserve and fragmentation, retained prefix-cache blocks, speculative-decoding draft state, CUDA-graph capture buffers, prefill activation workspace; (g) a falsifiable prediction that measured steady-state growth should match the padded footprint within allocator granularity, with a >15% gap attributed to prefix-cache retention or fragmentation rather than to the formula; and (h) a rollback gate forbidding concurrency increases once projected KV exceeds 85% of the runtime-reported KV pool, because the failure mode is preemption thrash or OOM under burst rather than graceful degradation.
- Blind-review compliance: no file under experiments/2026-08-14-teacher-a-corpus-calibration/ was read, listed, grepped or searched at any point in this run. The only inputs were source_user and source_assistant from research/ai-infra-expert/corpus/train.jsonl.
- Status caveat: PROVISIONAL teacher-B model output. Not expert gold labels, not human-validated, and not evidence of any trained model's domain capability.

## Run 2026-08-17 batch 0080

- Batch file: results/train-batch-0080.jsonl
- Corpus range: train.jsonl lines 791-800 (source IDs corpus-00870 … corpus-00879 — corpus file order preserved exactly, no skips or reordering)
- Progress: train 800/5399, validation 0/601, total 800/6000, remaining 5200
- Decisions: keep 0, rewrite 10, reject 0
- Initial schema check: PASS on first run (scripts/verify_batches.py — per-line JSONL parse, batch count 10, 12 required fields, enum values for teacher_lane/teacher_model/calibration_status/decision, quality_dimensions 1-5 integers, risks/evidence_required string arrays, character-exact source_user/source_assistant vs corpus, non-empty corrected_answer, confidence in [0,1], global source_id uniqueness, train/validation aggregates strict corpus prefixes)
- Repairs performed: none required; the batch verified clean on the first run. No corpus file, no earlier batch, and no teacher-A artifact was modified.
- Final schema check: PASS (train_processed=800, validation_processed=0, total=800, ERRORS 0)
- Manifest: MANIFEST.sha256 regenerated over every file in this directory except itself; `sha256sum -c` reported 139/139 OK, 0 failures
- Technical topics covered: single-request KV cache sizing for GQA/MQA decoder stacks — layer depths 24-56, KV head counts 2-8, head dims 64/96/128, contexts 1024-4096 tokens, BF16/FP16 (2 B/value) and INT8 (1 B/value) KV dtypes. Every source byte total was independently recomputed from 2 x layers x seq_len x kv_heads x head_dim x bytes_per_value; all ten matched, so technical_correctness is 4. All ten were still marked `rewrite` because the source answers state only formula-plus-number: they never require the head count to be post-GQA num_key_value_heads, never mention paged-allocator block round-up (PagedAttention block_size 16/32 makes measured usage a strict upper bound on the logical figure), never account for INT8 per-block scale/zero-point metadata or the accuracy regression that must be measured before adopting low-precision KV, never separate a per-request logical byte count from the HBM budget that must simultaneously hold weights, activations, CUDA graph pools and NCCL buffers, and never flag the architectures that break the linear formula (sliding-window attention, cross-layer KV sharing, MLA latent KV). The rewrites restate all of these as explicit falsifiable assumptions, add per-token byte cost as the quantity that actually composes to concurrency planning, enumerate required evidence (config.json num_hidden_layers / num_key_value_heads / head_dim / window settings, engine startup KV block report, steady-state nvidia-smi or torch.cuda.memory_summary), and set a rollback gate at a >15% analytic-vs-measured gap with preemption/recompute events held at zero.
- Blind-review compliance: no file under experiments/2026-08-14-teacher-a-corpus-calibration/ was read, listed, grepped or searched at any point in this run. Inputs were only source_user and source_assistant from research/ai-infra-expert/corpus/train.jsonl.
- Status caveat: PROVISIONAL teacher-B model output. Not expert gold labels, not human-validated, and not evidence of any trained model's domain capability.

## Run 2026-08-17 batch 0079

- Batch file: results/train-batch-0079.jsonl
- Corpus range: train.jsonl lines 781-790 (source IDs corpus-00860, corpus-00861, corpus-00862, corpus-00863, corpus-00864, corpus-00865, corpus-00866, corpus-00867, corpus-00868, corpus-00869 — corpus file order preserved exactly, no skips or reordering)
- Progress: train 790/5399, validation 0/601, total 790/6000, remaining 5210
- Decisions: keep 0, rewrite 10, reject 0
- Initial schema check: PASS on first run (scripts/verify_batches.py — JSONL per-line parse, batch count 10, 12 required fields, teacher_lane/teacher_model/calibration_status/decision enums, quality_dimensions integers in 1-5, risks/evidence_required string arrays, source_user/source_assistant character-exact vs corpus, non-empty corrected_answer, confidence in [0,1], global source_id uniqueness across all batches, train/validation aggregates strict prefixes of their corpora)
- Repairs performed: one self-initiated correction before the manifest step. The generator's first pass auto-assigned `keep` whenever the source arithmetic reproduced exactly, which scored the record on numeric correctness alone. That is the wrong rubric for this lane — instruction coverage and operational safety are graded separately — so the generator was amended to mark these `rewrite` and lower instruction_coverage/operational_safety to 2, then the batch was regenerated and re-verified. No corpus file and no earlier batch was touched.
- Final schema check: PASS (train_processed=790, validation_processed=0, total=790, ERRORS 0)
- Manifest: MANIFEST.sha256 regenerated over every file in this directory except itself; `sha256sum -c` all-pass, 0 failures
- Technical topics covered: single-request KV cache capacity arithmetic for GQA/MQA decoder stacks — layer depths 24-56, KV head counts 2-8, head dims 64-96-128, contexts 1024-4096 tokens, BF16/FP16 (2 B/value) and INT8 (1 B/value) KV dtypes. Every source byte total and GiB figure was independently recomputed from 2 x layers x seq_len x kv_heads x head_dim x bytes_per_value and matched to within 5e-6 GiB, so technical_correctness is 4. All ten were still marked `rewrite`: the source states the formula and the number but never says the head count must be post-GQA `num_key_value_heads` (substituting query heads silently overestimates by the GQA ratio), never mentions paged-allocator block-granularity round-up (vLLM/SGLang PagedAttention block_size 16/32), never accounts for INT8 per-block scale and zero-point metadata, never separates a per-request logical figure from an HBM capacity constraint that must also hold weights, activations, CUDA graphs and NCCL buffers, and never flags the architectures that break linear scaling (sliding-window attention, cross-layer KV sharing, MLA latent KV). The rewrites restate each of these as explicit falsifiable assumptions, add the per-token byte cost as the quantity that actually drives concurrency planning, enumerate the evidence needed (config.json num_hidden_layers / num_key_value_heads / head_dim, engine kv_cache_dtype, the startup GPU-KV-cache-size log line, nvidia-smi and torch.cuda.memory_summary under load) with a >10% analytic-vs-measured gap as the falsification trigger, and define a rollback gate on preemption/swap events or a p99 TTFT regression above 20% versus baseline.
- Blind-review compliance: no file under experiments/2026-08-14-teacher-a-corpus-calibration/ was read, opened, listed, grepped or searched at any point during this batch. The only inputs were source_user and source_assistant from research/ai-infra-expert/corpus/train.jsonl.
- Status caveat: these results are PROVISIONAL teacher-B model output. They are not expert gold labels, have not been validated by a human domain expert, and say nothing about any trained model's domain capability.

## Run 2026-08-17 batch 0078

- Batch file: results/train-batch-0078.jsonl
- Corpus range: train.jsonl lines 771-780 (source IDs corpus-00849, corpus-00850, corpus-00851, corpus-00852, corpus-00853, corpus-00854, corpus-00855, corpus-00856, corpus-00857, corpus-00859 — corpus file order preserved exactly, no skips or reordering; note corpus-00858 is absent from the corpus itself, the gap is in the source data, not a skip)
- Progress: train 780/5399, validation 0/601, total 780/6000, remaining 5220
- Decisions: keep 0, rewrite 10, reject 0
- Initial schema check: PASS on first run (ad-hoc verifier — JSONL line-parse, batch count 10, 12 required fields, teacher_lane/teacher_model/calibration_status/decision enums, quality_dimensions integers in 1-5, risks/evidence_required string arrays, source_user/source_assistant character-exact vs corpus, non-empty corrected_answer, confidence in [0,1], global source_id uniqueness across all batches, train/validation aggregates are strict prefixes of their corpora)
- Repairs performed: none required
- Final schema check: PASS (train 780/5399, validation 0/601, total 780, 0 errors)
- Manifest: MANIFEST.sha256 regenerated over all 140 files in this directory except itself; `sha256sum -c --quiet` all-pass, 0 failures
- Technical topics covered: per-request KV cache sizing for GQA/MQA decoder stacks — layer counts 24-56, KV head counts 2-8, head dims 64-128, contexts 1024-4096, mixed BF16/FP16 (2 B/value) and INT8 (1 B/value) KV dtypes. All 10 source byte totals were independently recomputed from 2 x layers x seq_len x KV_heads x head_dim x bytes_per_value and matched exactly, so technical_correctness scored 4. All 10 were nevertheless marked `rewrite` for thin instruction coverage and operational safety: the source never states that KV_heads must be the post-GQA `num_key_value_heads` (using query heads is a silent 4-8x overestimate), ignores paged-allocator block-granularity round-up (vLLM/SGLang PagedAttention, TRT-LLM paged KV), ignores INT8 per-block scale/zero-point metadata (~1-3% on top), does not distinguish a per-request figure from an HBM capacity constraint (weights + activations + concurrency), does not address tensor-parallel KV sharding when kv_heads is not divisible by TP, and does not flag the architectures that break linear scaling (sliding-window attention, cross-layer KV sharing, MLA latent KV). The rewrites state each of these as explicit falsifiable assumptions, give a +5%/-0% predicted band against engine-reported KV occupancy, enumerate the evidence needed (config.json num_hidden_layers / num_key_value_heads / head_dim / torch_dtype, engine kv_cache_dtype and block_size, startup KV-blocks log line, nvidia-smi and torch.cuda.memory_summary at steady state, tensor_parallel_size), and define a rollback gate on KV-cache preemption or p99 TTFT SLO breach over two consecutive 5-minute windows.
- Blind-review compliance: no file under experiments/2026-08-14-teacher-a-corpus-calibration/ was read, opened, listed or searched during this batch. Only research/ai-infra-expert/corpus/train.jsonl source_user/source_assistant were consulted.
- Status caveat: these results are PROVISIONAL teacher-B model output. They are not expert gold labels, have not been validated by a human domain expert, and say nothing about any trained model's domain capability.

## Run 2026-08-17 batch 0077

- Batch file: results/train-batch-0077.jsonl
- Corpus range: train.jsonl lines 761-770 (source IDs corpus-00839, corpus-00840, corpus-00841, corpus-00842, corpus-00843, corpus-00844, corpus-00845, corpus-00846, corpus-00847, corpus-00848 — corpus file order preserved exactly, no skips or reordering)
- Progress: train 770/5399, validation 0/601, total 770/6000, remaining 5230
- Decisions: keep 0, rewrite 10, reject 0
- Initial schema check: PASS on first run (scripts/verify_batches.py — JSONL line-parse, batch count, 12 required fields, teacher_lane/teacher_model/calibration_status/decision enums, source_user/source_assistant byte-exact vs corpus, non-empty corrected_answer, confidence in [0,1], global source_id uniqueness, train/validation aggregate is a strict prefix of each corpus)
- Repairs performed: none required
- Final schema check: PASS (train_processed=770, validation_processed=0, total=770, ERRORS 0)
- Manifest: MANIFEST.sha256 regenerated over all 138 files in this directory except itself; `sha256sum -c --quiet` all-pass, 0 failures
- Technical topics covered: per-request KV cache sizing for GQA/MQA decoder stacks, spanning BF16/FP16 (2 bytes/value) and INT8 (1 byte/value) KV dtypes, layer counts 24-56, KV head counts 2-8, head dims 64-128, and context lengths 1024-4096. Each source byte figure was independently recomputed from 2 x layers x seq_len x KV_heads x head_dim x bytes_per_value and all 10 matched exactly, so technical_correctness was scored 4. All 10 were still marked `rewrite` for insufficient instruction coverage and operational safety: the source omits that KV_heads must be the post-GQA key-value head count (not query heads, a 4-8x error mode), omits paged-allocator block-granularity round-up in vLLM/SGLang PagedAttention, omits INT8 per-block scale/zero-point metadata (~1-4%), omits the concurrency multiplier that turns a per-request figure into an HBM capacity constraint, omits tensor-parallel KV sharding vs replication when KV_heads is not divisible by TP, and omits architectures that break linear scaling (sliding-window, cross-layer KV sharing, MLA latent KV). Rewrites state these as explicit falsifiable assumptions, report both GiB and GB, list the evidence needed (config.json num_hidden_layers/num_key_value_heads/head_dim, engine kv_cache_dtype and block_size, startup KV-blocks log line, nvidia-smi / torch.cuda.memory_summary at steady state), and set a rollback threshold at >15% measured-vs-predicted KV divergence before any max_num_seqs/max_model_len rollout proceeds.
- Blind-review compliance: no file under experiments/2026-08-14-teacher-a-corpus-calibration/ was read, opened, or searched during this batch. Only research/ai-infra-expert/corpus/train.jsonl source_user/source_assistant were consulted.
- Status caveat: these results are PROVISIONAL teacher-B model output. They are not expert gold labels, have not been validated by a human domain expert, and say nothing about any trained model's domain capability.

## Run 2026-08-17 batch 0076

- Batch file: results/train-batch-0076.jsonl
- Corpus range: train.jsonl lines 751-760 (source IDs corpus-00828, corpus-00829, corpus-00831, corpus-00832, corpus-00833, corpus-00834, corpus-00835, corpus-00836, corpus-00837, corpus-00838 — corpus file order preserved exactly, no skips or reordering)
- Progress: train 760/5399, validation 0/601, total 760/6000, remaining 5240
- Decisions: keep 0, rewrite 10, reject 0
- Initial schema check: PASS on first run (ad-hoc verifier — JSONL line-parse, batch count 10, 12 required fields, teacher_lane/teacher_model/calibration_status/decision enums, source_user/source_assistant byte-exact vs corpus, non-empty corrected_answer, confidence in [0,1], global source_id uniqueness, train/validation aggregate is a strict prefix of each corpus)
- Repairs performed: none required
- Final schema check: PASS (train=760/5399, validation=0/601, total=760)
- Manifest: MANIFEST.sha256 regenerated over all 137 files in this directory; `sha256sum -c` all-pass
- Technical topics covered: per-request KV cache sizing for GQA/MQA decoders across BF16/FP16 and INT8 KV dtypes. Every source answer applies the correct formula 2 x layers x seq_len x KV_heads x head_dim x bytes_per_value and every arithmetic result was independently re-derived and confirmed exact. All 10 were nonetheless marked `rewrite` because the source stops at a raw lower bound: it omits paged-allocator block round-up (ceil(S/block_size) internal fragmentation), prefix/radix cache sharing and beam-width multiplication, INT8 per-block scale/zero-point metadata, and the tensor-parallel distinction between KV sharding (KV_heads divisible by TP) and KV replication (KV_heads < TP). Rewrites add explicit assumptions, binary-unit reporting, a falsifiable allocator-delta prediction with a ~15% tolerance band, the evidence needed to confirm it (model config fields, kv_cache_dtype, block_size, TP degree, before/after allocator snapshot), and a capacity-planning rollback threshold at 20% measured-vs-predicted concurrency shortfall.
- Status caveat: these results are PROVISIONAL teacher-B model output. They are not expert gold labels, have not been validated by a human domain expert, and say nothing about any trained model's domain capability.

## Run 2026-08-17 batch 0075

- Batch file: results/train-batch-0075.jsonl
- Corpus range: train.jsonl lines 741-750 (source IDs corpus-00818 … corpus-00827, contiguous, corpus file order preserved exactly, no skips or reordering)
- Progress: train 750/5399, validation 0/601, total 750/6000, remaining 5250
- Decisions: keep 0, rewrite 10, reject 0
- Initial schema check: PASS on first run (verify_batches.py — JSONL line-parse, batch count 10, 12 required fields, teacher_lane/teacher_model/calibration_status/decision enums, source_user/source_assistant byte-exact vs corpus, non-empty corrected_answer, confidence in [0,1], global source_id uniqueness, train/validation aggregate is a strict prefix of each corpus)
- Repairs: none required
- Final schema check: VERIFY_PASS, train=750/5399 validation=0/601 total=750/6000
- Manifest: MANIFEST.sha256 regenerated over all files in this directory except itself; `sha256sum -c` reports 132/132 OK, 0 failures
- Technical topics covered: single-request KV cache sizing for GQA/MQA models across BF16/FP16 and INT8 KV widths. All ten source answers use the same bare `2 × layers × S × kv_heads × head_dim × bytes_per_value` formula. The arithmetic in every source item was independently recomputed and is correct, so the rewrites are coverage rewrites, not correctness fixes. Each corrected answer adds: the assumption that exactly one K and one V tensor exists per layer (invalid for MLA latent KV in DeepSeek-V2/V3 and for hybrid SSM/attention stacks with fewer KV-bearing layers); the requirement to read num_key_value_heads rather than num_attention_heads, which is the dominant real-world overestimate; the effect of sliding-window attention capping effective length at min(S, W) and making the cache constant rather than linear in S; S as total context (prompt + generated) growing monotonically through decode; per-token KV bytes as the actual capacity-planning unit; paged-attention block rounding (ceil(S/block)*block) as an occupancy term the formula omits; and for the INT8 cases, that INT8 KV is lossy and needs per-token scales, kernel support, and accuracy validation rather than being a free 2x. Each answer states a falsifiable single-request measurement check against engine-reported KV block usage, the evidence set to collect (config.json fields, engine startup log block size and total blocks, controlled single-request KV utilization, allocator snapshot separating weights/activations/KV), and a rollback threshold of >15% deviation in measured per-token KV before the estimate may no longer be used for admission control.
- Status: PROVISIONAL. This is a blind, single-model second-opinion pass produced without any visibility into the teacher-A lane. It is not expert gold, has not been human-verified, and says nothing about the domain capability of any trained model. Agreement analysis against teacher-A is a separate, later step outside this task.

## Run 2026-08-17 batch 0074

- Batch file: results/train-batch-0074.jsonl
- Corpus range: train.jsonl lines 731-740 (source IDs corpus-00807, corpus-00808, corpus-00809, corpus-00811, corpus-00812, corpus-00813, corpus-00814, corpus-00815, corpus-00816, corpus-00817 — corpus-00810 is absent from the corpus file itself; corpus file order preserved exactly, no skips or reordering introduced by this lane)
- Progress: train 740/5399, validation 0/601, total 740/6000, remaining 5260
- Decisions: keep=0, rewrite=10, reject=0
- Initial schema check: pass (`verify.py` reported `train=740/5399 validation=0/601 total=740` / `VERIFY_PASS`; `verify_batches.py` reported `train: 740 records checked against corpus of 5399`, `unique source_ids: 740`, `VERIFY=PASS`). Checks covered line-by-line JSONL parse, exactly 10 records in this batch, all 12 required fields, teacher_lane=teacher-B, teacher_model=claude-opus-5-current, calibration_status=provisional, decision in {keep,rewrite,reject}, character-exact source_user/source_assistant equality against the corpus, non-empty corrected_answer, confidence in [0,1], quality_dimensions as three integers in 1-5, risks/evidence_required as string arrays, globally unique source_ids across all 74 batches, and the aggregated train sequence being an exact prefix of train.jsonl.
- Repairs performed: none required; verification passed on first execution.
- Final schema check: pass (740 records validated, 0 errors)
- Independent arithmetic re-derivation: the generator re-parsed (layers, kv_heads, head_dim, seq_len, dtype) from each user prompt and recomputed 2·L·S·H_kv·d·B from scratch, asserting both the byte total and the six-decimal GiB figure appear in source_assistant. All ten matched exactly (88080384 / 503316480 / 50331648 / 113246208 / 335544320 / 31457280 / 264241152 / 704643072 / 25165824 / 37748736 B). `rewrite` therefore reflects insufficiency of the answer, not arithmetic error; technical_correctness 4, instruction_coverage 3, operational_safety 2, confidence 0.82 on all ten.
- Manifest: MANIFEST.sha256 regenerated over every file in the experiment directory except itself; `sha256sum -c` returned all OK, 0 failures.
- Technical topics covered by this batch: single-request KV cache sizing for GQA/MQA transformers spanning 24-56 layers, 2-8 KV heads, head_dim 64/96/128, context 1024-4096, in BF16/FP16 and INT8 KV dtypes. Each rewrite states the five load-bearing assumptions (one K plus one V per layer giving the factor 2; num_key_value_heads not num_attention_heads; no sliding-window/cross-attention/MLA latent KV; sequence length as prompt plus generated context; dense rather than paged-rounded allocation), derives a per-token KV byte figure for admission control and extrapolation, and then states falsifiable predictions rather than assertions: a measured prefill memory delta at or above the estimate within one page-block per layer, and near-doubling of that delta when context doubles, with failure of either pointing at MLA, sliding-window, or an unaccounted quantized-KV path. Each record grounds the figure on the local A30 24 GiB host class by reporting the request's share of one GPU and an upper-bound sequence count for a 16 GiB KV pool, explicitly labelled a ceiling because weights, CUDA/NCCL context and comm buffers, activation and attention workspace, allocator fragmentation and retained prefix-cache blocks all draw on the same budget first. INT8 records additionally caveat that 1 byte/value is payload only, that per-head or per-token scale tensors add uncounted bytes, and that quantized KV is an accuracy-affecting change gated on engine kernel support. Every record closes with the evidence list needed before acting (config.json fields including sliding_window and architecture family, the engine's KV dtype and block size, one controlled single-request memory delta, and preemption/recompute counters) and an explicit rollback gate: measured peak KV exceeding the estimate by more than 15%, or any non-zero preemption/recompute under target concurrency, both observable inside one canary window without a restart.
- Status caveat: PROVISIONAL teacher-B model review, not expert gold labels, and no evidence whatsoever about any trained model's domain capability. Blind: no teacher-A artifact under experiments/2026-08-14-teacher-a-corpus-calibration/ was read, opened, or grepped while producing this batch. Agreement/consistency analysis against teacher-A remains a separate later step outside this task.

## Run 2026-08-17 batch 0073

- Batch file: results/train-batch-0073.jsonl
- Corpus range: train.jsonl lines 721-730 (source IDs corpus-00795, corpus-00796, corpus-00798, corpus-00799, corpus-00801, corpus-00802, corpus-00803, corpus-00804, corpus-00805, corpus-00806 — corpus-00797 and corpus-00800 are absent from the corpus file itself; corpus file order preserved exactly, no skips or reordering introduced by this lane)
- Progress: train 730/5399, validation 0/601, total 730/6000, remaining 5270
- Decisions: keep=0, rewrite=10, reject=0
- Initial schema check: pass (`scripts/verify_batches.py train-batch-0073.jsonl` reported `train=730/5399 validation=0/601 total=730/6000` and `VERIFY_PASS`). Checks covered line-by-line JSONL parse, exactly 10 records in this batch, all 12 required fields, teacher_lane=teacher-B, teacher_model=claude-opus-5-current, calibration_status=provisional, decision in {keep,rewrite,reject}, character-exact source_user/source_assistant equality against the corpus, non-empty corrected_answer, confidence in [0,1], quality_dimensions as three integers in 1-5, risks/evidence_required as string arrays, globally unique source_ids across all 73 batches, and the aggregated train sequence being an exact prefix of train.jsonl.
- Repairs performed: none required; verification passed on first execution.
- Final schema check: pass (730 records validated, 0 errors)
- Independent arithmetic re-derivation: the generator parsed (layers, kv_heads, head_dim, seq_len, dtype) out of each user prompt and recomputed 2·L·S·H_kv·d·B from scratch, asserting the byte total appears in source_assistant. All ten matched exactly (44040192 / 113246208 / 52428800 / 339738624 / 25165824 / 50331648 / 188743680 / 100663296 / 110100480 / 150994944 B). The `rewrite` decisions therefore reflect insufficiency of the answer, not arithmetic error; technical_correctness 4, instruction_coverage 3, operational_safety 2 on all ten.
- Manifest: MANIFEST.sha256 regenerated over every file in the experiment directory except itself; `sha256sum -c` returned all OK (129 entries, 0 failures).
- Technical topics covered by this batch: single-request KV cache sizing for GQA/MQA transformers spanning 24-56 layers, 2-8 KV heads, head_dim 64/96/128, context 1024-4096, in BF16/FP16 and INT8 KV dtypes. Each rewrite states the four load-bearing assumptions (one K plus one V per layer giving the factor 2; num_key_value_heads not num_attention_heads; no sliding-window/cross-attention/MLA latent compression; sequence length as prompt plus generated context), reports a per-token KV byte figure for admission control and extrapolation, then enumerates boundary conditions the arithmetic omits: PagedAttention block rounding, prefix-cache/radix retention, speculative-decoding and beam-search branch KV, engine-preallocated KV pools versus nvidia-smi readings, and disaggregated prefill/decode plus KV offload paths (Mooncake-style pooled KV store, NVIDIA Dynamo KV-aware routing) where the same byte count becomes a network transfer with an explicit 200 Gb/s RoCE line-rate floor computed per record. Batch-specific notes flag head_dim 96 kernel padding to 128 (up to 1.33x inflation), narrow num_kv_heads (2 and 6) as a hard TP-divisibility constraint where TP > H_kv replicates KV, depth-dominated cases (48-56 layers), short-context cases (1024-1536) where fixed allocator/CUDA-graph overheads dominate, and long-context cases (3072-4096) approaching the KV-over-weights crossover. INT8 records additionally caveat that 1 byte/value is payload only, with per-group scale/zero-point overhead of roughly 2/g bytes per value, and that INT8 KV is an accuracy-affecting change requiring a task-level quality gate. Every record closes with a falsifiable single-request test (max_num_seqs=1, compare the engine's reported KV cache size against the in-record per-token KV byte figure; >~10% deviation invalidates an assumption) and a rollback gate (>1% preemption/swap rate, >20% p99 TTFT regression, or any OOM in engine logs).
- Status caveat: PROVISIONAL teacher-B model review, not expert gold labels, and no evidence whatsoever about any trained model's domain capability. Blind: no teacher-A artifact under experiments/2026-08-14-teacher-a-corpus-calibration/ was read, opened, or grepped while producing this batch. Agreement/consistency analysis against teacher-A remains a separate later step outside this task.

## Run 2026-08-17 batch 0072

- Batch file: results/train-batch-0072.jsonl
- Corpus range: train.jsonl lines 711-720 (source IDs corpus-00785 through corpus-00794, contiguous; corpus file order preserved exactly, no skips or reordering introduced by this lane)
- Progress: train 720/5399, validation 0/601, total 720/6000, remaining 5280
- Decisions: keep=0, rewrite=10, reject=0
- Initial schema check: pass (`scripts/verify_batches.py train-batch-0072.jsonl` reported `train=720/5399 validation=0/601 total=720/6000` and `VERIFY_PASS`). Checks covered line-by-line JSONL parse, exactly 10 records in this batch, all 12 required fields, teacher_lane=teacher-B, teacher_model=claude-opus-5-current, calibration_status=provisional, decision in {keep,rewrite,reject}, character-exact source_user/source_assistant equality against the corpus, non-empty corrected_answer, confidence in [0,1], quality_dimensions as three integers in 1-5, risks/evidence_required as string arrays, globally unique source_ids across all 72 batches, and the aggregated train sequence being an exact prefix of train.jsonl.
- Repairs performed: none required; verification passed on first execution.
- Final schema check: pass (720 records validated, 0 errors)
- Independent arithmetic re-derivation: the generator parsed (layers, kv_heads, head_dim, seq_len, dtype) out of each user prompt and recomputed 2·L·S·H_kv·d·B from scratch, asserting the byte total appears in source_assistant. All ten matched exactly (176160768 / 44040192 / 301989888 / 167772160 / 18874368 / 176160768 / 188743680 / 100663296 / 110100480 / 402653184 B). The `rewrite` decisions therefore reflect insufficiency of the answer, not arithmetic error; technical_correctness 4, instruction_coverage 3, operational_safety 2 on all ten.
- Manifest: MANIFEST.sha256 regenerated over every file in the experiment directory except itself; `sha256sum -c` returned all OK (128 entries, 0 failures).
- Technical topics covered by this batch: single-request KV cache sizing for GQA/MQA transformers spanning 24-56 layers, 2-8 KV heads, head_dim 64/96/128, context 1024-4096, in BF16/FP16 and INT8 KV dtypes. Each rewrite states the four load-bearing assumptions (one K plus one V per layer giving the factor 2; num_key_value_heads not num_attention_heads; no sliding-window/cross-attention/MLA latent compression, each of which breaks linearity in sequence length; sequence length as prompt plus generated context), then enumerates boundary conditions the arithmetic omits: PagedAttention block rounding, prefix-cache/radix retention, speculative-decoding and beam-search branch KV, engine-preallocated KV pools versus nvidia-smi readings, and disaggregated prefill/decode plus KV offload paths (Mooncake-style pooled KV store, NVIDIA Dynamo KV-aware routing) where the same byte count becomes a network transfer with an explicit line-rate floor computed per record at 200 Gb/s RoCE. Batch-specific notes flag head_dim 96 kernel padding to 128 (up to 1.33x inflation), non-power-of-two or narrow num_kv_heads (2 and 6) as a hard TP-divisibility constraint where TP > H_kv replicates KV, depth-dominated cases (48-56 layers) where KV quantization at best halves the layer-linear term, short-context cases (1024-1536) where fixed allocator/CUDA-graph overheads dominate, and long-context cases (4096) approaching the KV-over-weights crossover. INT8 records additionally caveat that 1 byte/value is payload only, with per-group scale/zero-point overhead of roughly 2/g bytes per value (~12.5% at g=16, ~1.6% at g=128, doubled if asymmetric), and that INT8 KV is an accuracy-affecting change requiring a task-level quality gate. Every record closes with a falsifiable single-request test (max_num_seqs=1, compare the engine's reported KV cache size against a per-token KV byte figure derived in-record; >~10% deviation invalidates an assumption) and a rollback gate (>1% preemption/swap rate, >20% p99 TTFT regression, or any OOM in engine logs).
- Status caveat: PROVISIONAL teacher-B model review, not expert gold labels, and no evidence whatsoever about any trained model's domain capability. Blind: no teacher-A artifact under experiments/2026-08-14-teacher-a-corpus-calibration/ was read, opened, or grepped while producing this batch. Agreement/consistency analysis against teacher-A remains a separate later step outside this task.

## Run 2026-08-17 batch 0071

- Batch file: results/train-batch-0071.jsonl
- Corpus range: train.jsonl lines 701-710 (source IDs corpus-00774, corpus-00776, corpus-00777, corpus-00778, corpus-00779, corpus-00780, corpus-00781, corpus-00782, corpus-00783, corpus-00784 — corpus-00775 is absent from the corpus file itself; corpus file order preserved exactly, no skips or reordering introduced by this lane)
- Progress: train 710/5399, validation 0/601, total 710/6000, remaining 5290
- Decisions: keep=0, rewrite=10, reject=0
- Initial schema check: pass (`scripts/verify_batches.py train-batch-0071.jsonl` reported `train=710/5399 validation=0/601 total=710/6000` and `VERIFY_PASS`). Checks covered line-by-line JSONL parse, exactly 10 records in this batch, all 12 required fields, teacher_lane=teacher-B, teacher_model=claude-opus-5-current, calibration_status=provisional, decision in {keep,rewrite,reject}, character-exact source_user/source_assistant equality against the corpus, non-empty corrected_answer, confidence in [0,1], quality_dimensions as three integers in 1-5, risks/evidence_required as string arrays, globally unique source_ids across all 71 batches, and the aggregated train sequence being an exact prefix of train.jsonl.
- Repairs performed: none required; verification passed on first execution.
- Final schema check: pass (710 records validated, 0 errors)
- Independent arithmetic re-derivation: the generator parsed (layers, kv_heads, head_dim, seq_len, dtype) out of each user prompt and recomputed 2·L·S·H_kv·d·B from scratch, asserting the byte total appears in source_assistant. All ten matched exactly (25165824 / 201326592 / 20971520 / 188743680 / 528482304 / 234881024 / 18874368 / 100663296 / 62914560 / 377487360 B). The `rewrite` decisions therefore reflect insufficiency of the answer, not arithmetic error; technical_correctness 4, instruction_coverage 3, operational_safety 2 on all ten.
- Manifest: MANIFEST.sha256 regenerated over every file in the experiment directory except itself; `sha256sum -c` returned all OK (127 entries, 0 failures).
- Technical topics covered by this batch: single-request KV cache sizing for GQA/MQA transformers spanning 24-56 layers, 2-8 KV heads, head_dim 64/96/128, context 1024-4096, in BF16/FP16 and INT8 KV dtypes. Each rewrite states the four load-bearing assumptions (one K plus one V per layer giving the factor 2; num_key_value_heads not num_attention_heads; no sliding-window/cross-attention/MLA latent compression, each of which breaks linearity in sequence length; sequence length as prompt plus generated context), then enumerates boundary conditions the arithmetic omits: PagedAttention block rounding with waste bounded by (B-1)/seq_len, tensor-parallel KV sharding that only divides cleanly when TP divides num_key_value_heads (binding hard at 2 and non-power-of-two 6 KV heads, where TP>H_kv replicates KV and aggregate footprint grows with TP), prefix-cache/radix retention, speculative decoding and beam-search branch KV, and the pre-reserved KV pool being what nvidia-smi actually reports. Batch-specific notes flag head_dim 96 kernel padding to 128 (a potential 1.33x inflation), depth-dominated cases (56 layers x 4096 tokens) where quantization alone cannot reduce the layer-linear term, and the small-context case where fixed allocator/CUDA-graph overheads rather than KV are binding. INT8 records additionally caveat that 1 byte/value counts payload only and per-group scale/zero-point tensors add roughly 2/g bytes per value (~12.5% at g=16, ~1.6% at g=128, doubled if asymmetric), making the figure a lower bound. Every record closes with a falsifiable single-request test (max_num_seqs=1, compare against the engine's reported GPU KV cache size in tokens, >~10% deviation invalidates an assumption) and a rollback gate (>1% preemption/swap rate, >20% p99 TTFT regression, or any OOM in engine logs).
- Status caveat: PROVISIONAL teacher-B model review, not expert gold labels, and no evidence whatsoever about any trained model's domain capability. Blind: no teacher-A artifact under experiments/2026-08-14-teacher-a-corpus-calibration/ was read, opened, or grepped while producing this batch. Agreement/consistency analysis against teacher-A remains a separate later step outside this task.

## Run 2026-08-17 batch 0070

- Batch file: results/train-batch-0070.jsonl
- Corpus range: train.jsonl lines 691-700 (source IDs corpus-00763, corpus-00764, corpus-00765, corpus-00767, corpus-00768, corpus-00769, corpus-00770, corpus-00771, corpus-00772, corpus-00773 — corpus-00766 is absent from the corpus file itself; corpus file order preserved exactly, no skips or reordering introduced by this lane)
- Progress: train 700/5399, validation 0/601, total 700/6000, remaining 5300
- Decisions: keep=0, rewrite=10, reject=0
- Initial schema check: pass (`scripts/verify_batches.py train-batch-0070.jsonl` reported `train=700/5399 validation=0/601 total=700/6000` and `VERIFY_PASS`). Checks covered line-by-line JSONL parse, exactly 10 records in this batch, all 12 required fields present, teacher_lane=teacher-B, teacher_model=claude-opus-5-current, calibration_status=provisional, decision in {keep,rewrite,reject}, character-exact source_user/source_assistant equality against the corpus, non-empty corrected_answer, confidence in [0,1], quality_dimensions as three integers in 1-5, risks/evidence_required as string arrays, globally unique source_ids across all 70 batches, and the aggregated train sequence being an exact prefix of train.jsonl.
- Repairs performed: none required; verification passed on first execution.
- Final schema check: pass (700 records validated, 0 errors)
- Independent arithmetic re-derivation: the generator parsed (layers, kv_heads, head_dim, seq_len, dtype) directly out of each user prompt and recomputed 2·L·S·H_kv·d·B from scratch, then asserted equality against both the byte total and the GiB value quoted in source_assistant. All ten matched exactly (235929600 / 603979776 / 51380224 / 100663296 / 62914560 / 75497472 / 293601280 / 56623104 / 352321536 / 167772160 B). The `rewrite` decisions therefore reflect insufficiency of the answer, not arithmetic error; technical_correctness scored 4 on all ten, instruction_coverage 3, operational_safety 2.
- Manifest: MANIFEST.sha256 regenerated over every file in the experiment directory except itself; `sha256sum -c` returned all OK with 0 failures.
- Technical topics covered by this batch: single-request KV cache sizing for GQA/MQA transformers spanning 24-56 layers, 2-8 KV heads, head_dim 64-128, context 1024-4096, in BF16/FP16 and INT8 KV dtypes. Each rewrite states the four assumptions the formula silently makes (one K plus one V per layer giving the leading factor 2; num_key_value_heads rather than num_attention_heads; no sliding-window / cross-attention / MLA latent compression, each of which breaks linearity in sequence length; sequence length meaning prompt plus generated context), names the query-heads-for-KV-heads substitution as the dominant integer-factor error mode, and then enumerates the boundary conditions that push real allocation above the arithmetic: PagedAttention block rounding at block_size 16 with the per-sequence waste bounded by (B-1)/seq_len, tensor-parallel KV sharding that only divides cleanly when TP divides num_key_value_heads (binding hard at 2 KV heads), prefix-cache/radix retention keeping blocks resident after request completion, speculative decoding and beam search holding parallel branch KV, and the engine's pre-reserved KV pool (gpu_memory_utilization) being what nvidia-smi actually shows. The INT8 records additionally quantify the uncounted per-block scale/zero-point metadata (approximately 2·L·H_kv·ceil(S/16)·2 bytes, given as an explicit percentage of payload for each case) and flag INT8 KV as a quality-affecting change that must be gated on a task-level eval rather than on memory savings. Every record ends with a falsifiable single-request `torch.cuda.memory_allocated()` prefill-delta test plus a failure taxonomy (integer multiple implies wrong KV head count or factor-of-2 assumption; large non-integral excess implies block rounding, quantization metadata, or prefix-cache residency), the evidence list to collect, and a rollback gate at >20 percent measured overshoot or any scheduler preemption/recompute events at the derived concurrency.
- Status caveat: PROVISIONAL teacher-B model review, not expert gold labels, and no evidence whatsoever about any trained model's domain capability. Blind: no teacher-A artifact under experiments/2026-08-14-teacher-a-corpus-calibration/ was read, opened, or grepped while producing this batch. Agreement/consistency analysis against teacher-A remains a separate later step outside this task.

## Run 2026-08-17 batch 0069

- Batch file: results/train-batch-0069.jsonl
- Corpus range: train.jsonl lines 681-690 (source IDs corpus-00753 … corpus-00762, contiguous; corpus file order preserved exactly, no skips or reordering)
- Progress: train 690/5399, validation 0/601, total 690/6000, remaining 5310
- Decisions: keep=0, rewrite=10, reject=0
- Initial schema check: pass (`verify_batches.py` reported `train: 690 records checked against corpus of 5399`, `validation: 0 records checked against corpus of 601`, `unique source_ids: 690`, `VERIFY=PASS`). Checks covered line-by-line JSONL parse, 10 records in this batch, all 12 required fields, teacher_lane=teacher-B, teacher_model=claude-opus-5-current, calibration_status=provisional, decision domain, character-exact source_user/source_assistant against corpus, non-empty corrected_answer, confidence in [0,1], quality_dimensions three integers 1-5, globally unique source_ids, and aggregated train sequence being an exact prefix of train.jsonl.
- Repairs performed: none required; verification passed on first execution.
- Final schema check: pass (690 records validated, 0 errors)
- Independent arithmetic re-derivation: the generator parsed (layers, kv_heads, head_dim, seq_len, dtype) out of the user prompt and recomputed 2·L·S·H_kv·d·B from scratch instead of trusting the source string, then compared to the byte total asserted in source_assistant. All ten matched exactly (10485760 / 113246208 / 352321536 / 62914560 / 75497472 / 293601280 / 150994944 / 176160768 / 37748736 / 33554432 B). Decisions are therefore `rewrite` for insufficiency, not for arithmetic error; technical_correctness scored 4 on all ten.
- Manifest: MANIFEST.sha256 regenerated over every file in the experiment directory except itself (pycache excluded); `sha256sum -c` returned 122 OK lines and 0 failures.
- Technical topics covered by this batch: single-request KV cache sizing across GQA/MQA transformers with 24-56 layers, 2-8 KV heads, head_dim 64-128, context 1024-4096, in both BF16/FP16 and INT8 KV dtypes. Each rewrite makes the assumption set explicit and checkable against config.json and the engine KV-dtype flag, names the kv-heads-vs-query-heads substitution as the dominant integer-factor error mode, converts the per-request figure into the per-token byte rate and a tokens-per-GiB budget (the quantity that actually binds max_num_seqs × max_model_len), and then states the boundary conditions that push the real number up: PagedAttention block rounding at block_size 16/32, speculative decoding and beam/n>1 branch multiplication, and the second transient KV copy held in flight by disaggregated prefill/decode stacks (Mooncake, NVIDIA Dynamo) which can drive peak resident KV toward 2× during transfer. Every record states what the figure excludes (weights, activation/workspace, CUDA graph pools, NCCL comm buffers, allocator fragmentation and reserved-but-unallocated pool), the TP caveat that per-GPU KV only divides cleanly when TP divides num_key_value_heads, a falsifiable single-request `torch.cuda.memory_allocated()` prefill-delta test with an explicit failure taxonomy (integer factor ⇒ head/dtype assumption wrong; few percent ⇒ block rounding, expected), the evidence to collect, and a rollback gate at >15 percent measured overshoot. INT8 records additionally flag the uncounted per-block scale/zero-point tensors and that INT8 KV accuracy impact needs separate validation.
- Status caveat: PROVISIONAL teacher-B model review, not expert gold labels, and no evidence about any trained model's domain capability. Blind: no teacher-A artifact was read, opened, or grepped while producing this batch. Agreement analysis remains a separate later step.

## Run 2026-08-17 batch 0068

- Batch file: results/train-batch-0068.jsonl
- Corpus range: train.jsonl lines 671-680 (source IDs corpus-00742, corpus-00743, corpus-00745, corpus-00746, corpus-00747, corpus-00748, corpus-00749, corpus-00750, corpus-00751, corpus-00752 — corpus-00744 is absent from the corpus itself; corpus file order preserved exactly, no skips or reordering introduced by this lane)
- Progress: train 680/5399, validation 0/601, total 680/6000, remaining 5320
- Decisions: keep=0, rewrite=10, reject=0
- Initial schema check: pass (`verify_batches.py` reported `train: 680 records checked against corpus of 5399`, `validation: 0`, `unique source_ids: 680`, `VERIFY=PASS`). Checks covered line-by-line JSONL parse, 10 records in this batch, all 12 required fields, teacher_lane=teacher-B, teacher_model=claude-opus-5-current, calibration_status=provisional, decision domain, character-exact source_user/source_assistant against corpus, non-empty corrected_answer, confidence in [0,1], quality_dimensions three integers 1-5, globally unique source_ids, and aggregated train sequence being an exact prefix of train.jsonl.
- Repairs performed: none required; verification passed on first execution.
- Final schema check: pass (680 records validated, 0 errors)
- Independent arithmetic re-derivation: each byte total was recomputed in the generator from (layers, kv_heads, head_dim, seq_len, bytes_per_value) rather than copied from the source string; all ten matched the source values exactly (125829120 / 377487360 / 176160768 / 50331648 / 37748736 / 251658240 / 125829120 / 88080384 / 198180864 / 536870912 B). The rewrite decision therefore reflects insufficiency of the source answer, not arithmetic error.
- Manifest: MANIFEST.sha256 regenerated over all files except itself; `sha256sum -c` returned 120 OK lines and 0 failures.
- Technical topics covered by this batch: single-request KV cache sizing for GQA/MQA transformers spanning BF16/FP16 and INT8 KV dtypes, 24-56 layers, 2-8 KV heads, head_dim 64-128, context 1024-4096. Each rewrite states the mechanism (KV grows linearly in generated tokens and in concurrency while weights are a fixed cost), the derived per-token byte rate as the true capacity-planning quantity against max_num_seqs x max_model_len, the GQA/MQA kv-heads-not-query-heads trap, paged-attention block rounding as a lower-bound correction, exclusion of allocator fragmentation and CUDA-graph/activation/communication buffers, speculative decoding and beam/prefix-fork branch multiplication, MLA/latent-KV architectures as outside the formula, and the extra in-flight KV copy that disaggregated prefill/decode stacks (Mooncake, NVIDIA Dynamo) hold during transfer. INT8 cases additionally carry an explicit accuracy-A/B gate for quantized KV plus scale/zero-point overhead, and every record names the evidence to collect (config.json fields, engine-reported KV blocks/block_size, measured HBM, preemption counters) and a concrete rollback gate.
- Status caveat: PROVISIONAL teacher-B model review, not expert gold labels, and no evidence about any trained model's domain capability. Blind: no teacher-A artifact was read while producing this batch. Agreement analysis remains a separate later step.

## Run 2026-08-17 batch 0067

- Batch file: results/train-batch-0067.jsonl
- Corpus range: train.jsonl lines 661-670 (source IDs corpus-00732 … corpus-00741, contiguous; corpus order preserved exactly, no skips or reordering)
- Progress: train 670/5399, validation 0/601, total 670/6000, remaining 5330
- Decisions: keep=0, rewrite=10, reject=0
- Initial schema check: pass (`verify_batches.py` reported `train: 670 records checked against corpus of 5399`, `validation: 0`, `unique source_ids: 670`, `VERIFY=PASS`). Checks covered line-by-line JSONL parse, 10 records in this batch, all 12 required fields, teacher_lane=teacher-B, teacher_model=claude-opus-5-current, calibration_status=provisional, decision domain, character-exact source_user/source_assistant against corpus, non-empty corrected_answer, confidence in [0,1], quality_dimensions three integers 1-5, globally unique source_ids, and aggregated train sequence being an exact prefix of train.jsonl.
- Repairs performed: none required; verification passed on first execution.
- Final schema check: pass (670 records validated, 0 errors)
- Independent arithmetic re-derivation: all ten byte totals and GiB values were recomputed from (layers, kv_heads, head_dim, seq_len, dtype) and compared to the source strings; all ten matched exactly (33554432 / 47185920 / 201326592 / 110100480 / 226492416 / 117440512 / 83886080 / 113246208 / 352321536 / 12582912 B). The rewrite decision reflects insufficiency of the source answer, not arithmetic error.
- Manifest: MANIFEST.sha256 regenerated over all files except itself; `sha256sum -c` returned 118 OK lines and 0 failures.
- Technical topics covered by this batch: single-request KV cache sizing for GQA/MQA transformers spanning BF16/FP16 and INT8 KV dtypes, 24-56 layers, 2-8 KV heads, head_dim 64-128, context 1024-4096. Each rewrite states the mechanism (KV grows linearly in generated tokens and in concurrency), the derived per-token byte rate as the real capacity-planning quantity, PagedAttention block padding at block_size 16, preallocated KV pool vs gpu_memory_utilization, the TP <= kv_heads sharding limit, preemption/recompute as the observable failure signal instead of CUDA OOM, the lossy nature of FP8/INT8 KV with an explicit accuracy-A/B gate and rollback condition, and the concrete evidence needed (config.json fields, engine-reported KV dtype/block_size/block count, measured HBM under a controlled concurrency ramp).
- Status caveat: PROVISIONAL teacher-B model review, not expert gold labels, and no evidence about any trained model's domain capability. Blind: no teacher-A artifact was read while producing this batch. Agreement analysis remains a separate later step.

## Run 2026-08-17 batch 0066

- Batch file: results/train-batch-0066.jsonl
- Corpus range: train.jsonl lines 651-660 (source IDs corpus-00722 … corpus-00731, contiguous; corpus order preserved exactly, no skips or reordering)
- Progress: train 660/5399, validation 0/601, total 660/6000, remaining 5340
- Decisions: keep=0, rewrite=10, reject=0
- Initial schema check: pass (ad-hoc verifier `verify_batches.py` reported `train: 660 records checked against corpus of 5399`, `validation: 0`, `unique source_ids: 660`, `VERIFY=PASS`). Checks covered: line-by-line JSONL parse, exactly 10 records per batch file, all 12 required fields present and no extra fields, teacher_lane=teacher-B, teacher_model=claude-opus-5-current, calibration_status=provisional, decision in {keep,rewrite,reject}, source_user/source_assistant character-identical to the corpus, corrected_answer non-empty, confidence in [0,1], quality_dimensions an object of exactly three integers in 1-5, source_id globally unique across all 660 records, and the aggregated train sequence an exact prefix of train.jsonl.
- Repairs performed: none required (verification passed on first execution). The generator asserted each recomputed byte total and GiB value against the source string before writing, so any arithmetic divergence would have aborted the write rather than emitted a bad batch.
- Final schema check: pass (660 records validated, 0 errors)
- Independent arithmetic re-derivation: all ten source byte totals were recomputed from (layers, kv_heads, head_dim, seq_len, dtype) parsed from the prompt text and all ten matched the source exactly (201326592 / 110100480 / 603979776 / 58720256 / 18874368 / 150994944 / 419430400 / 37748736 / 308281344 / 301989888 B). The rewrite decision is therefore about insufficiency of the answer, not about arithmetic error.
- Manifest: MANIFEST.sha256 regenerated over all files in the experiment directory except MANIFEST.sha256 itself; `sha256sum -c` returned exit 0 with 116 OK lines and 0 failures.
- Technical topics covered by this batch: single-request KV cache sizing for GQA/MQA transformers across BF16/FP16 and INT8 KV dtypes, at layer counts 24-56, KV head counts 2-8, head_dim 64-128 and context lengths 1024-4096. Each rewrite adds the mechanism (per-token KV growth, linear in tokens and concurrency), the derived per-token byte rate as the actual capacity-planning quantity, PagedAttention block-padding waste at block_size 16, the preallocated KV pool interacting with gpu_memory_utilization, the TP <= kv_heads limit beyond which KV sharding stops reducing per-GPU footprint, preemption/recompute as the observable failure signal instead of OOM, the lossy nature of FP8/INT8 KV, the concrete evidence needed (config.json fields, engine-reported KV dtype and block count, measured HBM at known concurrency), and an explicit rollback gate (>15% divergence from the analytic value, or any nonzero preemption rate at target concurrency).
- Status caveat: these outputs are PROVISIONAL teacher-B model review, not expert gold labels, and they say nothing about the domain capability of any trained model. They are one blind second opinion pending later independent agreement analysis against teacher-A.
- Blind protocol: no file under experiments/2026-08-14-teacher-a-corpus-calibration/ was read, opened, listed, or grepped during this run; only research/ai-infra-expert/corpus/train.jsonl was consulted.

## Run 2026-08-17 batch 0065

- Batch file: results/train-batch-0065.jsonl
- Corpus range: train.jsonl lines 641-650 (source IDs corpus-00709, corpus-00710, corpus-00713, corpus-00714, corpus-00715, corpus-00716, corpus-00717, corpus-00719, corpus-00720, corpus-00721 — corpus order preserved exactly, no skips or reordering; the ID gaps are gaps in the corpus itself)
- Progress: train 650/5399, validation 0/601, total 650/6000, remaining 5350
- Decisions: keep=0, rewrite=10, reject=0
- Initial schema check: pass (ad-hoc verifier reported `train 650 validation 0 total 650` then `SCHEMA_CHECK_PASS`; all 12 required fields present, teacher_lane=teacher-B, teacher_model=claude-opus-5-current, calibration_status=provisional, decision in {keep,rewrite,reject}, source_user/source_assistant character-identical to corpus, corrected_answer non-empty, confidence in [0,1], quality_dimensions integers 1-5, source_id globally unique across all 650 records, aggregated train sequence an exact prefix of train.jsonl, validation still empty)
- Repairs performed: none required (verification passed on first execution; the generator asserted each recomputed byte total and GiB string against the source text before writing, so any arithmetic divergence would have aborted the write rather than produced a bad batch)
- Final schema check: pass (650 records validated, 0 errors)
- Independent arithmetic re-derivation: all ten source byte totals were recomputed from (layers, kv_heads, head_dim, seq_len, dtype) parsed from the prompt and all ten matched the source exactly (132120576 / 469762048 / 83886080 / 62914560 / 396361728 / 352321536 / 33554432 / 226492416 / 117440512 / 47185920 B). The rewrite decision is about insufficiency, not arithmetic error.
- Manifest: MANIFEST.sha256 regenerated over all 115 files in the experiment directory except MANIFEST.sha256 itself; `sha256sum -c --quiet` returned clean (0 failures)
- Blind protocol: no file under experiments/2026-08-14-teacher-a-corpus-calibration/ was read, opened, listed, or grepped during this run; only research/ai-infra-expert/corpus/train.jsonl was consulted.

Technical topics covered: single-request KV cache byte sizing for GQA/MQA serving
across BF16/FP16 and INT8 KV dtypes (Calculation cases 209-221). The source
answers state the correct closed form and the correct number, then stop, so each
was rewritten rather than kept. Each corrected answer adds explicit falsifiable
assumptions (standard non-MLA attention with one K and one V tensor per layer, KV
cost scaling with KV heads rather than query heads under GQA/MQA, 1 GiB = 2^30 B,
no prefix sharing), a derived per-token byte rate — the quantity actually used to
size max_num_seqs x max_model_len against free HBM — the mechanism by which KV and
not weights is the elastic term that caps concurrency (linear growth in generated
tokens and in batch, with pool exhaustion surfacing as preemption/recompute
throughput cliffs and TTFT/ITL tail spikes rather than a clean OOM), and the
boundary conditions the closed form omits: PagedAttention block padding with an
explicit block count at block_size 16, the preallocated KV pool under
gpu_memory_utilization masking per-request HBM movement, attention/CUDA-graph/NCCL
scratch, tensor-parallel KV-head sharding with replication once TP exceeds the KV
head count, and speculative-decoding/beam multipliers on live KV. INT8 cases
additionally flag the uncounted per-tensor scale/zero-point bytes and that INT8 KV
is a lossy accuracy/capacity trade rather than a free 2x; 16-bit cases state the
symmetric point. Every record closes with an evidence set (config.json shape
fields, engine KV dtype and block_size, runtime GPU KV cache usage and preemption
counters, steady-state nvidia-smi / torch.cuda.memory_summary()) and a rollback
gate (>~20% measured-over-estimate divergence or >~1% preemption rate means reduce
max_num_seqs / max_model_len or enable prefix caching, never raise
gpu_memory_utilization to absorb the gap).

These results are PROVISIONAL teacher-B model output. They are not expert gold
labels, they have not been validated by a human domain expert, and they say
nothing about the domain capability of any trained model.

## Run 2026-08-17 batch 0064

- Batch file: results/train-batch-0064.jsonl
- Corpus range: train.jsonl lines 631-640 (source IDs corpus-00699 through corpus-00708, contiguous)
- Progress: train 640/5399, validation 0/601, total 640/6000, remaining 5360
- Decisions: keep=0, rewrite=10, reject=0
- Initial schema check: pass (verify_batches.py reported `train 640 validation 0 total 640` then `PASS`; all 12 required fields present, teacher_lane/teacher_model/calibration_status/decision values valid, source_user and source_assistant character-identical to corpus, corrected_answer non-empty, confidence in [0,1], quality_dimensions integers 1-5, source_id globally unique across all 640 records, train sequence an exact prefix of train.jsonl, validation still empty)
- Repairs performed: none required (verification passed on first execution)
- Final schema check: pass (640 records validated, 0 errors)
- Independent arithmetic re-derivation: all ten source byte totals were recomputed from (layers, kv_heads, head_dim, seq_len, dtype) parsed out of the prompt; all ten matched the source figure exactly (75497472 / 440401920 / 75497472 / 58720256 / 377487360 / 201326592 / 22020096 / 75497472 / 251658240 / 125829120 B). The rewrite decision is therefore about insufficiency, not arithmetic error.
- Manifest: MANIFEST.sha256 regenerated over all files in the experiment directory except MANIFEST.sha256 itself; `sha256sum -c` reported all OK, 0 failures
- Blind protocol: no file under experiments/2026-08-14-teacher-a-corpus-calibration/ was read, opened, listed, or grepped during this run; only research/ai-infra-expert/corpus/train.jsonl was consulted.

Technical topics covered: single-request KV cache byte sizing for GQA/MQA serving
across BF16/FP16 and INT8 KV dtypes (Calculation cases 199-208). The source
answers give the right closed form and the right number but stop there, so each
was rewritten rather than kept: the corrected answers add explicit falsifiable
assumptions (standard non-MLA attention, KV cost driven by KV heads not query
heads, 1 GiB = 2^30 B, no prefix sharing), a derived per-token byte rate which is
the quantity actually used to size max_num_seqs x max_model_len against free HBM,
the mechanism by which KV rather than weights caps concurrency (linear growth in
generated tokens and in batch, with exhaustion surfacing as preemption/recompute
throughput cliffs and ITL tail spikes rather than clean OOM), boundary conditions
the closed form does not model (PagedAttention block-size rounding, preallocated
pool under gpu_memory_utilization, fragmentation, CUDA graph and prefill
workspace, MLA/latent-KV, sliding-window and hybrid SSM layers, speculative
decoding and beam search multiplying live KV copies), the uncounted INT8 scale/
zero-point metadata plus the accuracy cost of INT8 KV on long context, an
evidence list (config.json fields, engine startup KV/block log line, measured HBM
delta, long-context eval when KV is quantized) and an explicit rollback gate
(>=20% headroom; revert max_model_len / max_num_seqs if p99 ITL or preemption
counters regress in canary before touching quantization).

These results are provisional teacher-B second opinions produced by a single
model under blind review. They are NOT expert gold labels, have not been
validated against hardware measurements, and say nothing about any trained
model's domain capability.

## Run 2026-08-17 batch 0063

- Batch file: results/train-batch-0063.jsonl
- Corpus range: train.jsonl lines 621-630 (source IDs corpus-00689 through corpus-00698, contiguous)
- Progress: train 630/5399, validation 0/601, total 630/6000, remaining 5370
- Decisions: keep=10, rewrite=0, reject=0
- Initial schema check: pass (scripts/verify_batches.py reported `train=630/5399 validation=0/601 total=630/6000` then `VERIFY_PASS`; all 12 required fields present, teacher_lane/teacher_model/calibration_status/decision values correct, source_user and source_assistant character-identical to corpus, corrected_answer non-empty, confidence in [0,1], quality_dimensions integers in 1-5, source_id globally unique across all 630 records, train sequence an exact prefix of train.jsonl, validation still empty)
- Repairs performed: none required (verification passed on first execution)
- Final schema check: pass (630 records validated, 0 errors)
- Manifest: MANIFEST.sha256 regenerated over 117 files; `sha256sum -c` reported 117 OK and 0 failures
- Blind protocol: no file under experiments/2026-08-14-teacher-a-corpus-calibration/ was read, opened, or grepped during this run; only research/ai-infra-expert/corpus/train.jsonl was consulted.

Technical topics covered: single-request KV cache byte sizing for GQA/MQA models
across BF16/FP16 and INT8 KV dtypes (Calculation cases 189-198). Every source
arithmetic result was recomputed independently and all ten were exact, so the
decision is keep in each case; the corrected_answer adds what the terse source
answers omit — explicit assumptions (standard non-MLA attention, KV size driven
by KV heads not query heads, 1 GiB = 2^30 B), the linear-in-tokens and
linear-in-batch growth mechanism that makes KV rather than weights the
concurrency cap, boundary conditions the closed form does not model
(PagedAttention block padding rounding each sequence up to a whole block,
allocator fragmentation, CUDA graph/workspace reservations, prefix-cache sharing
that makes aggregate usage sublinear in N, MLA/latent-KV and sliding-window/SSM
layers where the formula does not apply, and INT8/FP8 scale+zero-point metadata),
a falsifiable doubling prediction on sequence length, the evidence needed to
confirm on real hardware (config.json fields, engine startup KV-block log,
measured HBM delta), and a rollback gate at ~20 percent measured-over-estimated
overshoot before admitting production traffic.

These results are PROVISIONAL model-generated second-opinion review output. They
are NOT expert gold labels, have not been validated by a human domain expert, and
say nothing about any model's domain capability.

## Run 2026-08-17 batch 0062

- Batch file: results/train-batch-0062.jsonl
- Corpus range: train.jsonl lines 611-620 (source IDs corpus-00678, corpus-00680, corpus-00681, corpus-00682, corpus-00683, corpus-00684, corpus-00685, corpus-00686, corpus-00687, corpus-00688)
- Progress: train 620/5399, validation 0/601, total 620/6000, remaining 5380
- Decisions: keep=0, rewrite=10, reject=0
- Initial schema check: pass (verify_batches.py reported `train 620 validation 0 total 620` then `PASS`; all 12 required fields present, teacher_lane/teacher_model/calibration_status/decision values correct, source_user and source_assistant byte-identical to corpus, corrected_answer non-empty, confidence in [0,1], quality_dimensions integers in 1-5, source_id globally unique across all 620 records, train sequence an exact prefix of train.jsonl, validation still empty)
- Repairs performed: none required (verification passed on first execution)
- Final schema check: pass (total 620 records validated, 0 errors)
- Manifest: MANIFEST.sha256 regenerated over 115 files; `sha256sum -c` reported 115 OK and 0 failures
- Blind protocol: no file under experiments/2026-08-14-teacher-a-corpus-calibration/ was read, opened or grepped during this run; only research/ai-infra-expert/corpus/train.jsonl was consulted.

Technical topics covered: single-request KV cache byte sizing under grouped-query
attention, layer counts 24-56, KV head counts 2-8, head dimensions 64-128, sequence
lengths 1024-4096, INT8 vs BF16/FP16 KV dtypes. Every source arithmetic result was
independently recomputed in the generator and matched (10/10), so the rewrites are not
corrections of numeric error but expansions of missing operational content: the leading
factor 2 is K+V and not tensor-parallel replication (so the figure is a cluster-wide
total, roughly divided by TP per GPU); the per-token marginal cost 2*L*H*D*bpv is the
quantity that should size max_num_seqs / max_num_batched_tokens; paged KV (vLLM
PagedAttention, SGLang radix cache) rounds allocations to whole blocks so real usage is
ceil(S/block_size)*block_size*per_token_bytes; INT8 KV carries per-group scales (and
zero-points if asymmetric) plus an accuracy obligation; prefix caching, speculative
decoding and beam search change the number of live KV copies; and weights, activations,
CUDA graphs and NCCL buffers sit outside this figure under gpu_memory_utilization. Each
record states a falsifiable prediction (measured KV delta within ~5% of the computed GiB,
falsified above ~15%), the evidence needed (model config fields, negotiated kv_cache_dtype,
engine KV-block startup log, torch.cuda.memory_reserved delta), and a rollback gate.

These outputs are PROVISIONAL teacher-B model review, not expert gold labels, and they do
not constitute evidence of any model's domain capability.

## Run 2026-08-17 batch 0061

- Batch file: results/train-batch-0061.jsonl
- Corpus range: train.jsonl lines 601-610 (source IDs corpus-00666, corpus-00667, corpus-00668, corpus-00669, corpus-00670, corpus-00672, corpus-00673, corpus-00674, corpus-00676, corpus-00677)
- Progress: train 610/5399, validation 0/601, total 610/6000, remaining 5390
- Decisions: keep=0, rewrite=10, reject=0
- Initial schema check: pass (10/10 new records; all 12 required fields present, teacher_lane/teacher_model/calibration_status/decision values correct, source_user and source_assistant byte-identical to corpus, corrected_answer non-empty, confidence in [0,1], source_id globally unique across all 610 records, train sequence an exact prefix of train.jsonl, validation still empty)
- Repairs performed: none required (verification passed on first execution)
- Final schema check: pass (total 610 records validated, 0 errors)
- Manifest: MANIFEST.sha256 regenerated over 113 files; `sha256sum -c` all OK

Technical topics covered: single-request KV cache byte sizing for grouped-query attention
transformers, layer counts 24-56, KV head counts 2-8, head dimensions 64-128, sequence
lengths 1024-4096, INT8 and BF16/FP16 KV element widths. All ten source byte totals were
independently recomputed from the stated parameters and every one matched, so each record is
marked rewrite for insufficiency rather than for a numeric error. The source answers give the
formula and the correct total but stop there. Each corrected answer restates the mechanism
(one K and one V vector per KV head per layer per token; the factor 2 is K plus V and is not
tensor-parallel replication), reports the per-token marginal cost that actually drives
admission control and max-num-seqs sizing, and then states the boundary conditions the
arithmetic does not cover: PagedAttention/radix-cache block rounding to whole blocks, INT8 and
FP8 scale and zero-point metadata, competition with weights, activation workspaces, CUDA graph
pools, NCCL buffers and allocator fragmentation, and the fact that MLA or cross-layer KV
sharing invalidates the per-layer independence assumption outright. Each record carries a
falsifiable prediction (measured steady-state KV for one request of the stated length should
land between the dense total and 1.10x that total), the evidence required to trust the figure
for capacity planning (num_hidden_layers, num_key_value_heads, head_dim, kv_cache_dtype,
engine block count and block_size, measured device-memory delta, TP degree), and a rollback
gate (revert gpu_memory_utilization / max_num_seqs / max_model_len to last known-good if
measured usage exceeds the upper bound by more than 10 percent or OOM occurs at target
concurrency). INT8 records additionally flag that an arithmetic saving is not an accuracy
result and needs a task-level evaluation.

These results are PROVISIONAL model-generated second-opinion labels. They are not expert gold,
have not been validated against hardware measurements, and say nothing about any model's
domain capability.

## Run 2026-08-17 batch 0060

- Batch file: results/train-batch-0060.jsonl
- Corpus range: train.jsonl lines 591-600 (source IDs corpus-00655, corpus-00656, corpus-00657, corpus-00658, corpus-00659, corpus-00661, corpus-00662, corpus-00663, corpus-00664, corpus-00665)
- Progress: train 600/5399, validation 0/601, total 600/6000, remaining 5400
- Decisions: keep=0, rewrite=10, reject=0
- Initial schema check: pass (10/10 new records; 12 required fields present, lane/model/status/decision values correct, source_user and source_assistant byte-identical to corpus, corrected_answer non-empty, confidence in [0,1], source_id globally unique across all 600 records, train sequence is an exact prefix of train.jsonl, validation empty)
- Repairs performed: none required (verification passed on first execution)
- Final schema check: pass (total 600 records validated, 0 errors)
- Manifest: MANIFEST.sha256 regenerated over 112 files; `sha256sum -c` all OK

Technical topics covered: per-request KV cache byte sizing for GQA transformers with layer
counts 24-56, KV head counts 2-8, head dimensions 64-128, sequence lengths 1024-4096, and
both BF16/FP16 and INT8 KV element widths. Every source arithmetic result was independently
recomputed and matched, so all ten were marked rewrite for incompleteness rather than for a
wrong number: the source answers give the formula and the correct byte total but state no
assumptions and omit the operationally decisive caveats. Each corrected answer adds the
explicit assumption set (dense non-MLA attention, separate K and V tensors, no allocator or
paged padding), the growth mechanism (KV is linear in sequence length and is the dominant
per-request term at long context while weights amortize across concurrency), and falsifiable
boundary conditions: PagedAttention block rounding to 16/32-token blocks makes measured
occupancy a strict upper bound on the analytic figure; tensor parallelism only divides
per-GPU KV when kv_heads % TP == 0, otherwise KV heads are replicated and aggregate KV
exceeds the estimate; MLA and cross-layer KV sharing invalidate the formula entirely;
speculative decoding and beam search multiply live KV by the branch count; INT8 KV adds
uncounted scale and zero-point storage. Each record also names the concrete evidence needed
(config.json attention fields, engine KV dtype and block_size, startup KV block count,
measured memory delta for one request) and a rollback threshold (halt rollout if measured
KV per request exceeds the estimate by more than 15% or free KV blocks drop below 10% at
target concurrency).

These results are PROVISIONAL model-generated second-opinion labels. They are not expert
gold, have not been validated against hardware measurements, and say nothing about any
trained model's domain capability.

## Run 2026-08-17 batch 0059

- Batch file: results/train-batch-0059.jsonl
- Corpus range: train.jsonl lines 581-590 (source IDs corpus-00644, corpus-00645, corpus-00646, corpus-00647, corpus-00649, corpus-00650, corpus-00651, corpus-00652, corpus-00653, corpus-00654)
- Progress: train 590/5399, validation 0/601, total 590/6000, remaining 5410
- Decisions: keep=0, rewrite=10, reject=0
- Initial schema check: pass (10/10 records, 12 required fields present, lane/model/status/decision values correct, source_user and source_assistant byte-identical to corpus, corrected_answer non-empty, confidence in [0,1], source_id globally unique, train sequence is an exact prefix of train.jsonl)
- Repairs performed: none required
- Final schema check: pass (total 590 records validated, 0 errors)
- Manifest: MANIFEST.sha256 regenerated over 111 files; `sha256sum -c` all OK

Technical topics covered: single-request KV cache byte sizing for GQA transformers across
varying layer counts (24-56), KV head counts (2-8), head dimensions (64-128), sequence
lengths (1536-4096), and BF16/FP16 vs INT8 KV element widths. All ten source arithmetic
results were independently recomputed and were numerically correct; every record was still
marked `rewrite` because the source answers stop at the raw number and omit the operational
context this corpus is supposed to teach: paged-attention block-size rounding (vLLM
PagedAttention block 16/32), INT8/FP8 KV scale and zero-point overhead, tensor-parallel KV
head sharding that only divides cleanly when kv_heads % TP == 0 (otherwise KV is replicated),
MLA / cross-layer-KV architectures that invalidate the formula outright, and speculative
decoding or beam search multiplying live KV by the branch count. Each corrected answer states
its assumptions explicitly, gives the formula and substitution, lists falsifiable boundary
conditions, names the evidence needed to confirm it (config.json fields, vLLM
num_gpu_blocks x block_size accounting, measured per-request GPU memory delta), and defines a
concrete rollback gate (reduce max_num_seqs / max_model_len if measured KV exceeds the
estimate by more than ~10 percent).

These results are PROVISIONAL model-generated second-opinion labels. They are NOT expert gold
data, they have not been validated by a human domain expert, and they say nothing about the
domain capability of any trained model. Agreement analysis against teacher-A is a separate,
later step; this batch was produced blind, without reading any teacher-A artifact.

## Run 2026-08-17 batch 0058

- Batch file: results/train-batch-0058.jsonl
- Corpus range: train.jsonl 1-indexed lines 571-580, source IDs corpus-00633 through corpus-00643 (contiguous corpus slice; corpus-00639 is absent from train.jsonl itself, so the ID sequence has a natural gap — strict corpus line order preserved, nothing skipped or reordered by this worker).
- Progress: train 580/5399, validation 0/601, total 580/6000, remaining 5420
- Decisions: keep 0, rewrite 10, reject 0
- Initial schema check: PASS (no repairs needed this run).
- Repairs: none. The original corpus, earlier batches, benchmark raw generations and all teacher-A artifacts were untouched; the teacher-A directory was not opened, read or grepped at any point (blind review).
- Final schema check: PASS (train=580 validation=0 total=580; 10/10 rows parse as physical-newline JSONL, all 12 required fields present, teacher_lane=teacher-B / teacher_model=claude-opus-5-current / calibration_status=provisional / decision in {keep,rewrite,reject}, source_user and source_assistant character-identical to corpus, corrected_answer non-empty, confidence in [0,1], quality_dimensions integers 1-5, 580 globally unique source_ids, aggregated train sequence an exact prefix of train.jsonl, validation still empty).
- Manifest: MANIFEST.sha256 regenerated over every file in this directory except the manifest itself; `sha256sum -c` reported OK with zero failures.

Technical topics covered by this batch: single-request KV-cache memory sizing for
transformer inference (Calculation cases 133-143). Parameter sweep: layers 24-56,
KV heads 2-8, head_dim 64-128, seq_len 1024-4096, KV dtype INT8 and BF16/FP16.
Each rewrite keeps the source arithmetic but adds the per-token planning constant,
the GQA num_kv_heads-vs-num_attention_heads failure mode, paged-attention block
rounding (vLLM/SGLang block_size=16), fixed startup KV-pool sizing under
gpu_memory_utilization, MLA / sliding-window invalidation of the closed-form
formula, tensor-parallel KV sharding divisibility, and disaggregated prefill/decode
KV handoff cost over RDMA/RoCE with and without GPUDirect RDMA (Mooncake, NVIDIA
Dynamo), plus falsifiable predictions, required evidence and an explicit rollback
gate (>15% deviation or non-zero preemption counters).

These results are PROVISIONAL model-generated second-opinion review output. They
are NOT expert gold labels, have not been validated by a human domain expert, and
say nothing about any model's domain capability.

## Run 2026-08-17 batch 0057

- Batch file: results/train-batch-0057.jsonl
- Corpus range: train.jsonl 0-indexed lines 560-569, source IDs corpus-00623 through corpus-00632 (contiguous, strict corpus order preserved, nothing skipped or reordered).
- Progress: train 570/5399, validation 0/601, total 570/6000, remaining 5430
- Decisions: keep 0, rewrite 10, reject 0
- Initial schema check: FAIL. The batch generator extracted `messages[0]`/`messages[1]` as user/assistant, but corpus rows carry a leading `system` message, so all 10 rows had `source_user`/`source_assistant` mismatched against the corpus (20 verifier failures).
- Repairs: fixed `scripts/gen_batch_0057.py` to select messages by `role` instead of positional index (this also corrected the parameter parsing, which had been reading layers/heads/dim/seq_len out of the wrong field), regenerated the batch, and re-ran verification. Only this run's own batch file and generator were changed. The original corpus, earlier batches, benchmark raw generations and all teacher-A artifacts were untouched; the teacher-A directory was not opened, read or grepped at any point (blind review).
- Final schema check: PASS (train=570 validation=0 total=570; 10/10 rows parse as physical-newline JSONL, all 12 required fields present, teacher_lane=teacher-B / teacher_model=claude-opus-5-current / calibration_status=provisional / decision in {keep,rewrite,reject}, source_user and source_assistant character-identical to corpus, corrected_answer non-empty, confidence in [0,1], quality_dimensions integers 1-5, 570 globally unique source_ids, aggregated train sequence an exact prefix of train.jsonl, validation still empty).
- Manifest: MANIFEST.sha256 regenerated over every file in this directory except the manifest itself; `sha256sum -c` reported OK with zero failures.

Technical topics covered by this batch: single-request KV-cache memory sizing for
transformer inference (Calculation cases 123-132). Parameter sweep: layers 24-56,
KV heads 2-8, head dim 64-128, seq_len 1024-4096, dtypes BF16/FP16 and INT8.
Each rewrite states the closed-form formula `2 x layers x seq_len x kv_heads x
head_dim x bytes_per_value`, derives the per-token planning constant, and then
makes the boundary conditions explicit: paged-attention block rounding (block_size
16, exact rounding cost computed per case), the GQA/MQA distinction between
num_kv_heads and num_attention_heads (the dominant real-world error mode),
MLA latent KV and sliding-window attention as cases where the formula
over-estimates, INT8/FP8 scale metadata that this arithmetic excludes, the fact
that the KV pool is sized once at engine startup and does not grow, tensor-parallel
sharding of KV heads (and replication when num_kv_heads is not divisible by TP),
and disaggregated prefill/decode (Mooncake, NVIDIA Dynamo) where the same KV bytes
exist on both sides during handoff and transfer time must be budgeted on measured
RDMA/RoCE goodput with GPUDirect RDMA rather than on link line rate. Each record
carries falsifiable predictions, the config/log/telemetry evidence needed to check
them, and a rollback gate (>15% deviation from the estimate, or non-zero
preemption/recompute counters, reverts max_model_len / max_num_seqs).

These results are PROVISIONAL teacher-B output from a general-purpose model. They
are not expert gold labels, they are not independently validated against running
hardware, and they say nothing about the domain capability of any trained model.

## Run 2026-08-17 batch 0056

- Batch file: results/train-batch-0056.jsonl
- Corpus range: train.jsonl 0-indexed lines 550-559, source IDs corpus-00612, corpus-00613, corpus-00614, corpus-00615, corpus-00616, corpus-00617, corpus-00618, corpus-00620, corpus-00621, corpus-00622 (strict corpus order preserved; the ID gap at corpus-00619 is a gap in the corpus itself, nothing was skipped or reordered).
- Progress: train 560/5399, validation 0/601, total 560/6000, remaining 5440
- Decisions: keep 0, rewrite 10, reject 0
- Initial schema check: PASS on first run (10/10 rows parsed as physical-newline JSONL, all 12 required fields present, teacher_lane=teacher-B / teacher_model=claude-opus-5-current / calibration_status=provisional / decision in {keep,rewrite,reject}, source_user and source_assistant character-identical to corpus, corrected_answer non-empty, confidence in [0,1], quality_dimensions integers 1-5, 560 globally unique source_ids, aggregated train sequence an exact prefix of train.jsonl, validation still empty).
- Repairs: none required. Original corpus, earlier batches, benchmark raw generations and all teacher-A artifacts untouched; the teacher-A directory was not opened, read or grepped during this run (blind review).
- Final schema check: PASS (train=560 validation=0 total=560).
- Manifest: MANIFEST.sha256 regenerated over every file in this directory except the manifest itself; `sha256sum -c` reported 106 entries OK, zero failures.

Technical topics covered by this batch: single-request KV-cache memory sizing for
transformer inference (Calculation cases 112-122). Parameter sweep: layers 24-56,
KV heads 2-8, head_dim 64/96/128, seq_len 1024-4096, INT8 vs BF16/FP16 KV dtype.
Independent recomputation of 2 x L x S x H x D x bytes confirmed all ten source
arithmetic results and GiB conversions are exact, so every rewrite is a coverage
rewrite rather than a numerical correction: the source answers stop at a raw byte
count. The teacher-B answers add the per-token planning constant, paged-attention
block rounding at block_size=16, INT8/FP8 scale metadata, the GQA-vs-query-head
trap, the MLA and sliding-window regimes where the closed form silently breaks,
static KV-pool sizing at engine startup, and the disaggregated prefill/decode
(Mooncake, NVIDIA Dynamo) case where the same KV bytes exist on both sides during
handoff and transfer time depends on measured RDMA/RoCE GPUDirect goodput rather
than line rate. Each record carries falsifiable predictions, the config/log/metric
evidence needed to check them, and a rollback gate at >15 percent measured-vs-
estimated divergence or non-zero preemption counters.

These results are PROVISIONAL teacher-B model output. They are not expert gold
labels, have not been validated by a human infrastructure engineer, and say
nothing about the domain capability of any trained model. Agreement analysis
against teacher-A is a separate later step and was deliberately not performed here.

## Run 2026-08-17 batch 0055

- Batch file: results/train-batch-0055.jsonl
- Corpus range: train.jsonl 0-indexed lines 540-549, source IDs corpus-00600, corpus-00601, corpus-00602, corpus-00603, corpus-00604, corpus-00605, corpus-00606, corpus-00608, corpus-00609, corpus-00611 (strict corpus order preserved; the ID gaps are gaps in the corpus itself, nothing was skipped or reordered).
- Progress: train 550/5399, validation 0/601, total 550/6000, remaining 5450
- Decisions: keep 0, rewrite 10, reject 0
- Initial schema check: PASS on first run (10/10 rows parsed as physical-newline JSONL, all 12 required fields present, teacher_lane=teacher-B / teacher_model=claude-opus-5-current / calibration_status=provisional / decision in {keep,rewrite,reject}, source_user and source_assistant character-identical to corpus, corrected_answer non-empty, confidence in [0,1], quality_dimensions integers 1-5, 550 globally unique source_ids, aggregated train sequence an exact prefix of train.jsonl, validation still empty).
- Repairs: none required. Original corpus, earlier batches, benchmark raw generations and all teacher-A artifacts untouched; the teacher-A directory was not opened, read or grepped during this run (blind review).
- Final schema check: PASS (train=550 validation=0 total=550).
- Manifest: MANIFEST.sha256 regenerated over every file in this directory except the manifest itself; `sha256sum -c` reported 105 entries OK, zero failures.

Technical topics covered by this batch: single-request KV-cache memory sizing for
transformer inference (Calculation cases 100-111). Parameter sweep: layers 24-56,
KV heads 2-8, head_dim 64/96/128, sequence length 1024-4096, dtype INT8 (1 B/value)
and BF16/FP16 (2 B/value). Every source arithmetic result was independently
recomputed and matched, so all rewrites are coverage/safety rewrites rather than
arithmetic corrections. Each corrected answer adds: explicit assumptions, the
per-token KV cost as the admission-control planning number, the GQA/MQA reason
kv_heads (not num_attention_heads) enters the formula, concurrency scaling, and
boundary conditions where the formula breaks (MLA latent KV, sliding-window
attention, paged-attention block rounding, INT8/FP8 scale metadata, prefix-cache
sharing). Each also states a falsifiable prediction (free-KV-block delta and a
5-10% memory agreement band), the evidence needed to confirm it, and a rollback
gate (>15% overshoot at target concurrency or <10% free-block headroom -> reduce
max_num_seqs / max_model_len or revert the KV dtype change).

These results are PROVISIONAL teacher-B model output. They are not expert gold
labels, have not been validated by a human domain expert, and say nothing about
any model's domain capability.

## Run 2026-08-17 batch 0054

- Batch file: results/train-batch-0054.jsonl
- Corpus range: train.jsonl 0-indexed lines 530-539, source IDs corpus-00590 through corpus-00599 (contiguous, strict corpus order preserved, nothing skipped or reordered).
- Progress: train 540/5399, validation 0/601, total 540/6000, remaining 5460
- Decisions: keep 0, rewrite 10, reject 0
- Initial schema check: PASS on first run (10/10 rows parsed as physical-newline JSONL, exactly the 12 required fields per record, teacher_lane=teacher-B / teacher_model=claude-opus-5-current / calibration_status=provisional / decision in {keep,rewrite,reject}, source_user and source_assistant character-identical to corpus, corrected_answer non-empty, confidence in [0,1], quality_dimensions integers 1-5, 540 globally unique source_ids, aggregated train sequence an exact prefix of train.jsonl, validation still empty).
- Repairs: none required. Original corpus, earlier batches, benchmark raw generations and all teacher-A artifacts untouched; the teacher-A directory was not opened, read or grepped during this run (blind review).
- Final schema check: PASS (train=540 validation=0 total=540).
- Manifest: MANIFEST.sha256 regenerated over every file in this directory except the manifest itself; `sha256sum -c` reported all entries OK, zero failures.

Technical topics covered by this batch: single-request KV-cache memory sizing for
transformer inference (Calculation cases 90-99). Parameter sweep: layers 24-56,
KV heads 2-8, head dimension 64-128, sequence length 1024-4096, KV dtype
BF16/FP16 (2 B) and INT8 (1 B). Each byte count was recomputed independently as
2 x layers x seq_len x kv_heads x head_dim x bytes_per_value with the GiB
conversion checked against 2^30; all ten source arithmetic results matched an
independent recomputation (asserted programmatically at generation time), so the
arithmetic itself is sound. The decision is `rewrite` rather than `keep` because
the source answers stop at the raw byte count and omit what an infrastructure
engineer needs to act: the per-token KV cost that governs admission control and
max_num_seqs, the GQA/MQA mechanism making the cache scale with kv_heads rather
than query heads, paged-attention block rounding (block_size 16 internal
fragmentation), INT8/FP8 KV scale/zero-point overhead making bytes_per_value=1 a
lower bound, prefix-caching/beam-search sharing effects, and the fact that the
formula does not hold for MLA latent KV or sliding-window attention. Each
rewrite adds a falsifiable prediction (linear KV growth in concurrency within
+/-5%), the evidence needed before use in capacity planning (config.json fields,
effective kv_cache_dtype, engine-reported block count/block_size, measured GPU
memory at known concurrency) and an explicit rollback threshold (>15% deviation
or preemption/recompute events -> reduce max_num_seqs/max_model_len and
re-measure).

These results are PROVISIONAL second-opinion review output from a single model
lane. They are NOT expert gold labels, have NOT been validated by a human domain
expert or by execution against real serving stacks, and they say nothing about
any trained model's domain capability.

## Run 2026-08-17 batch 0053

- Batch file: results/train-batch-0053.jsonl
- Corpus range: train.jsonl 0-indexed lines 520-529, source IDs corpus-00578 through corpus-00589 (corpus-00580 and corpus-00582 absent from the train split; strict corpus order preserved, nothing skipped or reordered by this worker).
- Progress: train 530/5399, validation 0/601, total 530/6000, remaining 5470
- Decisions: keep 0, rewrite 10, reject 0
- Initial schema check: PASS on first run (10/10 rows parsed as physical-newline JSONL, exactly the 12 required fields per record, teacher_lane=teacher-B / teacher_model=claude-opus-5-current / calibration_status=provisional / decision in {keep,rewrite,reject}, source_user and source_assistant character-identical to corpus, corrected_answer non-empty, confidence in [0,1], quality_dimensions integers 1-5, 530 globally unique source_ids, aggregated train sequence an exact prefix of train.jsonl, validation still empty).
- Repairs: none required. Original corpus, earlier batches, benchmark raw generations and all teacher-A artifacts untouched; the teacher-A directory was not opened, read or grepped during this run (blind review).
- Final schema check: PASS (train=530 validation=0 total=530).
- Manifest: MANIFEST.sha256 regenerated over every file in this directory except the manifest itself; `sha256sum -c` reported 103/103 OK, zero failures.

Technical topics covered by this batch: single-request KV-cache memory sizing for
transformer inference (Calculation cases 78-89). Parameter sweep: layers 24-56,
KV heads 2-8, head dimension 64-128, sequence length 1024-4096, KV dtype
BF16/FP16 (2 B) and INT8 (1 B). Every byte count was recomputed independently as
2 x layers x seq_len x kv_heads x head_dim x bytes_per_value with the GiB
conversion checked against 2^30; all ten source arithmetic results matched, so
the arithmetic itself is sound. The decision is nevertheless `rewrite` rather
than `keep` because the source answers stop at the raw byte count and omit the
quantities an infrastructure engineer actually needs: the per-token KV cost that
governs admission control and max_num_seqs, the GQA/MQA mechanism that makes the
cache scale with kv_heads rather than query heads, paged-attention block
rounding (block_size 16 internal fragmentation), INT8/FP8 KV scale/zero-point
overhead, prefix/radix cache sharing, and speculative-decoding or beam-search
branch multiplication. Each corrected answer adds a falsifiable concurrency
prediction (free KV bytes = N x per-request bytes should admit ~N sequences,
+/-1 block), the evidence needed to confirm it (config.json head counts, engine
startup '# GPU blocks' log line, torch.cuda.memory_summary / nvidia-smi
steady-state, load-test preemption counters), and an explicit rollback
threshold (>10% deviation of measured bytes/token, or any non-zero
preemption/swap counter at planned concurrency, blocks the config rollout).

These outputs are provisional teacher-B second opinions produced by an LLM
reviewer. They are NOT expert gold labels, have not been validated on hardware,
and say nothing about any model's domain capability.

## Run 2026-08-17 batch 0052

- Batch file: results/train-batch-0052.jsonl
- Corpus range: train.jsonl 0-indexed lines 510-519, source IDs corpus-00567 through corpus-00577 (corpus-00570 absent from train split; strict corpus order preserved, nothing skipped or reordered by this worker).
- Progress: train 520/5399, validation 0/601, total 520/6000, remaining 5480
- Decisions: keep 10, rewrite 0, reject 0
- Initial schema check: PASS on first run (10/10 rows parsed as physical-newline JSONL, exactly the 12 required fields per record, teacher_lane=teacher-B / teacher_model=claude-opus-5-current / calibration_status=provisional / decision in {keep,rewrite,reject}, source_user and source_assistant character-identical to corpus, corrected_answer non-empty, confidence in [0,1], quality_dimensions integers 1-5, 520 globally unique source_ids, aggregated train sequence an exact prefix of train.jsonl, validation still empty).
- Repairs: none required. Original corpus, earlier batches, benchmark raw generations and all teacher-A artifacts untouched; teacher-A directory not opened during this run (blind review).
- Final schema check: PASS (train=520 validation=0 total=520).
- Manifest: MANIFEST.sha256 regenerated over every file in this directory except the manifest itself; `sha256sum -c` reported 98/98 OK, zero failures.

Technical topics covered by this batch: single-request KV-cache memory sizing for
transformer inference (Calculation cases 67-77). Parameter sweep: layers 24-56,
KV heads 2-8, head dimension 64-128, sequence length 1024-4096, KV dtype
BF16/FP16 (2 B) and INT8 (1 B). Each byte count was recomputed independently as
2 x layers x seq_len x kv_heads x head_dim x bytes_per_value and its GiB
conversion checked against 2^30; all ten source values matched exactly, hence ten
keeps. The teacher-B corrected answers make explicit what the source leaves
implicit: the GQA/MQA assumption (kv_heads are key/value heads, not query heads),
paged-attention block-size rounding (allocated >= computed, up to block_size-1
tokens of waste per sequence per layer), uncounted INT8 scale/zero-point
overhead, weight dtype vs kv_cache_dtype, per-GPU behaviour under TP (sharded
when kv_heads % TP == 0, replicated when kv_heads < TP) and PP, the transfer-time
floor this byte count imposes on Mooncake-style prefill/decode disaggregation and
NVIDIA Dynamo KV routing over RDMA/RoCE or GDS, a falsifiable measurement check
(memory delta within ~1.15x of prediction), the evidence required (config.json
fields, kv_cache_dtype, block_size, measured KV-usage metric), and an explicit
rollback threshold (>90% KV utilisation or preemption/recompute events => reduce
max_num_seqs/max_model_len, do not raise gpu_memory_utilization on a live fleet).

These results are PROVISIONAL teacher-B second-opinion labels. They are NOT
expert gold, NOT verified ground truth, and they say nothing about any model's
domain capability. Agreement analysis against teacher-A is a separate, later step
and was deliberately not performed or consulted here.

## Run 2026-08-17 batch 0051

- Batch file: results/train-batch-0051.jsonl
- Corpus range: train.jsonl 0-indexed lines 500-509, source IDs corpus-00557 through corpus-00566 (contiguous) — strict corpus order, nothing skipped or reordered.
- Progress: train 510/5399, validation 0/601, total 510/6000, remaining 5490
- Decisions: keep 10, rewrite 0, reject 0
- Initial schema check: PASS on first run (10/10 rows parsed as physical-newline JSONL, exactly the 12 required fields per record, teacher_lane=teacher-B / teacher_model=claude-opus-5-current / calibration_status=provisional / decision in {keep,rewrite,reject}, source_user and source_assistant character-identical to corpus, corrected_answer non-empty, confidence in [0,1], quality_dimensions integers 1-5, 510 globally unique source_ids, aggregated train sequence an exact prefix of train.jsonl, validation still empty).
- Repairs: none required. Original corpus, earlier batches, benchmark raw generations and all teacher-A artifacts untouched.
- Final schema check: PASS (train=510 validation=0 total=510).
- Manifest: MANIFEST.sha256 regenerated over every file in this directory except the manifest itself; `sha256sum -c` reported 100/100 OK, zero failures.

Technical topics covered by this batch: single-request KV-cache memory sizing for
transformer inference. All ten items are the same numeric template with varying
layers (24-56), KV heads (2-8), head dimension (64-128), sequence length
(1024-4096) and KV dtype (BF16/FP16 vs INT8). Each byte count was recomputed
independently as 2 x layers x seq_len x kv_heads x head_dim x bytes_per_value
before reading the source answer's number; all ten source values and their GiB
conversions matched exactly, so all ten are keep. The teacher-B corrected answers
extend the source by making explicit what the source leaves implicit: the GQA/MQA
assumption that kv_heads are key/value heads and not query heads, paged-attention
block-size rounding (allocated >= computed), the uncounted INT8 scale/zero-point
overhead, the distinction between weight dtype and kv_cache_dtype, per-GPU
behaviour under TP (sharded when kv_heads % TP == 0, replicated when kv_heads <
TP) and PP (layer split), the implication for KV transfer cost on
Mooncake-style prefill/decode disaggregation and NVIDIA Dynamo KV routing over
RDMA/RoCE or GDS, a falsifiable measurement predicting the resident-memory delta
within one block per layer, and an operational rollback threshold on KV
utilisation / preemption events.

These teacher-B outputs are provisional second-opinion review artifacts. They are
NOT expert gold labels, have not been validated by a human domain expert, and say
nothing about any model's domain capability. This lane was produced blind: no
teacher-A artifact was read while generating this batch.

## Run 2026-08-17 batch 0050

- Batch file: results/train-batch-0050.jsonl
- Corpus range: train.jsonl lines 491-500 (0-indexed 490-499), source IDs corpus-00544, corpus-00545, corpus-00546, corpus-00547, corpus-00548, corpus-00549, corpus-00550, corpus-00554, corpus-00555, corpus-00556 — strict corpus order, nothing skipped or reordered (corpus-00551..corpus-00553 are absent from the corpus file itself).
- Progress: train 500/5399, validation 0/601, total 500/6000, remaining 5500
- Decisions: keep 10, rewrite 0, reject 0
- Initial schema check: PASS on first run (10/10 rows parsed as physical-newline JSONL, exactly 12 required fields per record, teacher_lane=teacher-B / teacher_model=claude-opus-5-current / calibration_status=provisional / decision in {keep,rewrite,reject}, source_user and source_assistant byte-identical to corpus, corrected_answer non-empty, confidence in [0,1], quality_dimensions integers 1-5, 500 globally unique source_ids, aggregated train sequence an exact prefix of train.jsonl).
- Repairs: none required. No batch was rewritten; the original corpus, earlier batches, benchmark generations and all teacher-A artifacts were left untouched.
- Final schema check: PASS (train=500 validation=0 total=500).
- Manifest: MANIFEST.sha256 regenerated over all 97 files in this directory except the manifest itself; `sha256sum -c` reported 97/97 OK, exit 0.

Independent arithmetic check: each of the ten byte counts was recomputed from
2 x layers x seq_len x kv_heads x head_dim x bytes_per_value before review; all
ten source figures matched exactly, so every item was a keep on correctness. The
corrected answers add what the source omits: the GQA reason the count uses
num_key_value_heads rather than query heads, the boundary conditions that make
the formula valid (dense causal attention, no sliding-window, no MLA latent
compression, uniform head_dim, no prefix sharing), the excluded terms that
actually decide concurrency (paged-KV block rounding, pre-reserved KV pool,
allocator fragmentation, TP KV-head replication when kv_heads is not divisible by
the TP degree, and beam/speculative branch multiplication), a GiB-vs-GB unit
disambiguation, a falsifiable single-request memory-delta prediction with the
measurement procedure, the evidence needed (config.json fields, engine startup KV
block log, TP topology, concurrency ramp), and an explicit rollback gate at
70-80% of derived max concurrency keyed on p99 latency, preemption counters and
OOM. The three INT8-KV cases (corpus-00546, corpus-00549, corpus-00555) were
additionally marked down on operational_safety to 2 because the source presents
INT8 as an exact 1 byte/value halving with no quantization scale/zero-point
overhead and no accuracy gate; those records carry an extra risk entry and an
extra evidence requirement for a fixed-eval accuracy comparison before the
quantized path serves traffic.

These outputs are PROVISIONAL teacher-B judgements produced blind, without any
access to teacher-A artifacts during generation. They are not expert gold labels,
they have not been validated against production telemetry, and they say nothing
about the trained model's domain capability.

## Run 2026-08-17 batch 0049

- Batch file: results/train-batch-0049.jsonl
- Corpus range: train.jsonl lines 481-490 (0-indexed 480-489), source IDs corpus-00532, corpus-00534, corpus-00535, corpus-00536, corpus-00537, corpus-00538, corpus-00539, corpus-00540, corpus-00541, corpus-00543 — contiguous in corpus order (corpus-00533 and corpus-00542 are absent from the corpus file itself; this run skipped and reordered nothing).
- Progress: train 490/5399, validation 0/601, total 490/6000, remaining 5510
- Decisions: keep 10, rewrite 0, reject 0
- Initial schema check: PASS on first run (10/10 rows parsed, 12 required fields present, teacher_lane/teacher_model/calibration_status/decision values correct, source_user and source_assistant byte-identical to corpus, corrected_answer non-empty, confidence in [0,1], 490 globally unique source_ids, aggregated train sequence is an exact prefix of train.jsonl).
- Repairs: none required. No batch was rewritten, and neither the original corpus nor any earlier batch was modified.
- Final schema check: PASS (train=490 validation=0 total=490 unique_ids=490).
- Manifest: MANIFEST.sha256 regenerated over all 96 files in this directory except the manifest itself; `sha256sum -c` verified all entries PASS.

Technical topics covered by this batch: all ten items are single-request KV-cache
sizing calculations for GQA/MQA transformer decoders, spanning 24-56 layers, 2-8 KV
heads, head dimensions 64/96/128, sequence lengths 1024-4096, in both BF16/FP16 and
INT8 KV dtypes. Each corrected answer states the closed-form
`2 x layers x seq_len x kv_heads x head_dim x bytes_per_value`, derives the
per-token byte cost used for admission control, and makes the assumption set
explicit (KV heads not query heads, dense attention, no sliding-window KV sharing,
no cross-attention). Boundary conditions added beyond the source answer: paged-KV
block rounding in vLLM-style engines and prefix-cache sharing, tensor-parallel KV
head sharding versus replication when kv_heads is not divisible by the TP degree,
INT8 per-group scale/zero-point overhead, and exclusion of weights, activation
workspaces, CUDA graph pools and allocator fragmentation. Each answer carries a
falsifiable prediction about KV-block occupancy growth under a concurrency ramp and
an explicit rollback gate forbidding max_num_seqs / max_model_len increases on
arithmetic alone. The source arithmetic was recomputed independently for all ten
items and matched exactly in both byte counts and GiB values, which is why all ten
are `keep` rather than `rewrite`.

These results are PROVISIONAL teacher-B output produced blind by the current
conversation model. They are not expert gold labels, they have not been checked
against teacher-A (deliberately, to avoid anchoring), and they say nothing about
any trained model's domain capability.

## Run 2026-08-17 batch 0048

- Batch file: results/train-batch-0048.jsonl
- Corpus range: train.jsonl lines 471-480 (0-indexed 470-479), source IDs corpus-00521, corpus-00522, corpus-00523, corpus-00525, corpus-00526, corpus-00527, corpus-00528, corpus-00529, corpus-00530, corpus-00531 — contiguous in corpus order (corpus-00524 is absent from the corpus itself; nothing was skipped or reordered by this run).
- Progress: train 480/5399, validation 0/601, total 480/6000, remaining 5520
- Decisions: keep 10, rewrite 0, reject 0
- Initial schema check: PASS on first run (scripts/verify_batches.py plus an inline per-record assertion pass — per-line JSONL parse, 10 records this batch, exactly the 12 required fields with no extras, lane/model/status/decision enum checks, quality_dimensions integer 1-5 on all three axes, risks/evidence_required string arrays, byte-exact source_user/source_assistant equality against research/ai-infra-expert/corpus/train.jsonl, non-empty corrected_answer, confidence in [0,1], global source_id uniqueness across all 48 batches, and strict train-prefix ordering).
- Repairs: none required this run.
- Final schema check: PASS (train 480, validation 0, total 480 unique source IDs, prefix OK).
- Manifest: MANIFEST.sha256 regenerated over all 91 files in the experiment directory (excluding the manifest itself and __pycache__); `sha256sum -c` returned OK for all 91 entries with zero failures.
- Technical topics covered: ten more per-request KV-cache sizing calculations (Calculation / medium, concepts kv_cache + memory), spanning 24–56 layers, 2–8 KV heads, head_dim 64–128, seq_len 1024–4096, in BF16/FP16 (7 cases) and INT8 (3 cases: corpus-00522, corpus-00525, corpus-00528, corpus-00531 INT8 subset). Every source byte count and GiB conversion was recomputed independently as 2 x layers x seq_len x kv_heads x head_dim x bytes_per_value and matched exactly, so all ten were graded keep on technical_correctness. The corrected_answer for each still adds the material the source omits: the factor 2 is K and V rather than bidirectionality; kv_heads (GQA/MQA) not query heads drives the term; paged allocators round the tail block up to ceil(seq_len/block_size)*block_size so real allocation exceeds the analytic figure; INT8 KV carries per-group scale/zero-point metadata that is not free and must be gated on an accuracy comparison; weights, activation workspace, CUDA-graph pools and fragmentation live outside the KV pool; and aggregate capacity is per-request bytes x in-flight sequences, which is the quantity that actually sets max_num_seqs. Each record states a falsifiable prediction (KV bytes scale exactly linearly in in-flight requests until the block pool saturates, then admission stalls rather than OOMs), the evidence required (served config.json layer/kv-head/head-dim fields, engine-reported KV block size and total GPU blocks, measured torch.cuda.memory_reserved deltas under a controlled concurrency ramp), and a rollback gate (>~15% measured-versus-estimate gap at steady state ⇒ stop raising max_num_seqs / gpu_memory_utilization and re-measure, since the gap implies unaccounted padding, quantization metadata, or prefix-cache retention).
- Status: PROVISIONAL. These are blind, single-pass teacher-B judgements, not expert gold labels, and they say nothing about any model's domain capability. No file under experiments/2026-08-14-teacher-a-corpus-calibration/ was read, opened, or grepped at any point during this batch.

## Run 2026-08-17 batch 0047

- Batch file: results/train-batch-0047.jsonl
- Corpus range: train.jsonl lines 461-470 (0-indexed 460-469), source IDs corpus-00511 … corpus-00520 — contiguous in corpus order, nothing skipped or reordered.
- Progress: train 470/5399, validation 0/601, total 470/6000, remaining 5530
- Decisions: keep 0, rewrite 10, reject 0
- Initial schema check: PASS on first run (scripts/verify_batches.py — per-line JSONL parse, 10 records this batch, all 12 required fields present with no extras, lane/model/status/decision enum checks, quality_dimensions integer 1-5 on all three axes, risks/evidence_required string arrays, byte-exact source_user/source_assistant equality against research/ai-infra-expert/corpus/train.jsonl, non-empty corrected_answer, confidence in [0,1], global source_id uniqueness across all 47 batches, and strict train-prefix ordering).
- Repairs: none required this run.
- Final schema check: PASS (train 470, validation 0, total 470 unique source IDs).
- Manifest: MANIFEST.sha256 regenerated over all 93 files in the experiment directory (excluding the manifest itself); `sha256sum -c` returned OK for every entry.
- Technical topics covered: ten per-request KV-cache sizing calculations (Calculation / medium, concepts kv_cache + memory) spanning 24–56 layers, 2–8 KV heads, head_dim 64–128, seq_len 1024–4096, across BF16/FP16 (7 cases) and INT8 (3 cases: corpus-00513, corpus-00516, corpus-00519). All ten source byte counts and GiB conversions were independently recomputed with the formula 2 x layers x seq_len x kv_heads x head_dim x bytes_per_value and matched exactly, so every rewrite is an enrichment rather than an arithmetic correction. Added mechanism and boundary material: why GQA/MQA makes kv_heads (not query heads) the driving term; the per-token marginal cost 2 x L x H x D x B as the quantity that actually governs how far a live request can grow; PagedAttention/RadixAttention block rounding (ceil(S/block)*block, up to block_size-1 tokens of internal fragmentation per request); exclusion of weights, activation/workspace buffers, CUDA-graph pools, NCCL buffers and allocator fragmentation from the KV pool; TP sharding of KV heads only when kv_heads % TP == 0 and the replication cliff once TP exceeds kv_heads, versus PP splitting the layer dimension; prefix/radix cache sharing breaking the naive concurrency x per-request aggregation; and for INT8 KV the scale/zero-point metadata overhead (~1-6%) plus the requirement to gate on an accuracy comparison, not memory alone. Each rewrite states a falsifiable prediction (measured single-request KV pool delta within ~5-10% of the analytic value, with >20% divergence indicating wrong config, MLA/latent-KV compression, or fixed pool preallocation), the evidence needed (model config fields, engine-reported KV block size and GPU block count, controlled nvidia-smi / torch.cuda.memory_summary deltas, TP/PP topology, INT8 quality comparison), and a rollback gate (>20% measured-vs-estimate gap or preemption/recompute at target concurrency ⇒ revert max_num_seqs / gpu_memory_utilization and re-derive from measured KV).
- Status: PROVISIONAL. These are blind, single-pass teacher-B judgements, not expert gold labels, and they say nothing about any model's domain capability. No file under experiments/2026-08-14-teacher-a-corpus-calibration/ was read, opened, or grepped at any point during this batch.

## Run 2026-08-17 batch 0046

- Batch file: results/train-batch-0046.jsonl
- Corpus range: train.jsonl lines 451-460 (0-indexed 450-459), source IDs corpus-00500, corpus-00501 … corpus-00508, corpus-00510 — contiguous in corpus order (the corpus itself has no corpus-00509 at this position), nothing skipped or reordered by this worker.
- Progress: train 460/5399, validation 0/601, total 460/6000, remaining 5540
- Decisions: keep 0, rewrite 10, reject 0
- Initial schema check: PASS on first run (verify_batches.py — JSONL line-parse, 10 records, all 12 required fields, lane/model/status/decision enums, byte-exact source_user/source_assistant equality against corpus, non-empty corrected_answer, confidence in [0,1], global source_id uniqueness across all 46 batches, and strict train-prefix ordering).
- Repairs: none required this run.
- Final schema check: PASS (train 460, validation 0, total 460).
- Manifest: MANIFEST.sha256 regenerated over all 89 files in the experiment directory (excluding the manifest itself); `sha256sum -c` returned OK for every entry.
- Technical topics covered: one speculative-decoding runbook entry (draft/target proposal + batched verification, modified rejection sampling and its distribution-preserving guarantee, expected-accepted-tokens (1-a^(k+1))/(1-a), and the break-even boundary where speculation loses at high batch size because the target step becomes compute-bound); and nine per-request KV-cache sizing calculations spanning INT8 and BF16/FP16 across 24–56 layers, 2–8 KV heads, head_dim 64–128 and seq_len 1024–4096. Source arithmetic was independently recomputed and found correct in all nine cases; rewrites add the GQA/MQA caveat that kv_heads (not query heads) is the driving factor, PagedAttention block-rounding and per-sequence internal fragmentation, INT8 scale/zero-point metadata overhead, the GiB vs GB unit ambiguity (~7% planning error), and explicit measurement evidence plus a >15% over-estimate rollback gate.
- Status: PROVISIONAL. These are blind, single-pass teacher-B judgements, not expert gold labels, and they say nothing about any model's domain capability. teacher-A outputs were not read at any point during this batch.

## Run 2026-08-17 batch 0045

- Batch file: results/train-batch-0045.jsonl
- Corpus range: train.jsonl lines 441-450 (0-indexed 440-449), source IDs corpus-00490 through corpus-00499 — contiguous, strict corpus order, nothing skipped or reordered.
- Progress: train 450/5399, validation 0/601, total 450/6000, remaining 5550
- Decisions: keep 0, rewrite 10, reject 0
- Initial schema check: PASS on first run (ad-hoc verifier — JSONL line-parse, batch count 10, all 12 required fields present and no extras, enum values for teacher_lane/teacher_model/calibration_status/decision, exact character-level source_user/source_assistant equality against the original corpus, non-empty corrected_answer, quality_dimensions integers in 1-5, risks/evidence_required string arrays, confidence in [0,1], globally unique source_id across train+validation, and both aggregates strictly a prefix of their corpus).
- Repairs: none required. No re-verification loop was needed this run.
- Manifest: MANIFEST.sha256 regenerated over every file in the experiment directory except the manifest itself; `sha256sum -c` returned OK for all 88 entries, 0 mismatches.
- Final schema check: PASS (train 450/5399, validation 0/601, total 450, 0 errors).

Technical topics covered by this batch: ten further speculative-decoding items,
sharing one identical stub assistant answer and differing only in the user
instruction, which splits them into three sub-clusters. (1) corpus-00490 asks for
a misleading intuition plus its correction; the rewrite targets the common belief
that speculation approaches draft-model latency, and shows why it cannot — the
target weight stream is read once per verification pass regardless of acceptance,
so the ceiling is (target step cost)/E[n] with E[n] <= k+1, and the drafter adds
resident weights and speculated KV slots that reduce admitted concurrency.
(2) corpus-00491..00495 ask for a small controlled experiment; the rewrites give a
factorial design over speculation on/off x k in {1,2,4,8} x offered concurrency,
with explicitly held-fixed build/weights-hash/TP-PP/clocks/scheduler, replayed
identical production traces rather than synthetic uniform prompts, warmup
discard, A/B/B/A interleaving against cluster drift, >=3 repeats per cell, and a
residual check of measured speedup against the model S ~= E[n]/(1 + k*c_draft/c_target
+ overhead) so that a numeric win is not shipped with a wrong mechanism story.
(3) corpus-00496..00499 ask for a runbook entry; the rewrites give an ordered
diagnostic that classifies the bandwidth-bound vs. compute-bound regime from a
profiler trace BEFORE any config change, measures the acceptance histogram and
c_draft/c_target instead of tuning k blindly, treats a TPOT win that costs
admitted concurrency as a capacity regression, and escalates tokenizer/vocab
mismatch between drafter and target as a correctness incident rather than a
performance issue.
Across all three sub-clusters the shared boundary condition is the crossover:
E[n] = (1 - a^(k+1))/(1 - a), and speculation turns negative once the server is
compute-bound at high concurrency, or when a is depressed by domain mismatch or
high sampling temperature, or when c_draft/c_target is not small. Rollback
thresholds are stated explicitly (p99 TPOT regression at production concurrency,
KV-pressure preemption increase, admitted concurrency below the queueing SLO, or
quality drift under any non-lossless acceptance rule). All ten source answers
were scored instruction_coverage 1 because the stub supplies neither the requested
mechanism nor a boundary condition; technical_correctness 4 because what it does
say is true but incomplete; operational_safety 2 because it invites unconditional
performance claims.

These results are PROVISIONAL teacher-B output from a blind review pass. They are
not expert gold, they have not been reconciled with teacher-A (whose outputs were
not read during this batch), and they say nothing about any model's domain
capability.

## Run 2026-08-17 batch 0044

- Batch file: results/train-batch-0044.jsonl
- Corpus range: train.jsonl lines 431-440 (0-indexed 430-439), source IDs corpus-00478, 00479, 00480, 00481, 00482, 00483, 00484, 00485, 00487, 00489 — strict corpus order, nothing skipped or reordered (the gaps at 00486/00488 are gaps in the corpus itself, not skips by this lane).
- Progress: train 440/5399, validation 0/601, total 440/6000, remaining 5560
- Decisions: keep 0, rewrite 10, reject 0
- Initial schema check: PASS on first run (ad-hoc verifier `verify_batches.py` — JSONL line-parse, batch count 10, all 12 required fields, enum values for teacher_lane/teacher_model/calibration_status/decision, exact source_user/source_assistant equality against the original corpus, non-empty corrected_answer, confidence in [0,1], globally unique source_id, and train/validation aggregates strictly a prefix of each corpus).
- Repairs: none required. No re-verification loop was needed this run.
- Manifest: MANIFEST.sha256 regenerated over every file in the experiment directory except the manifest itself (`__pycache__` excluded); `sha256sum -c` returned OK for all 84 entries, 0 mismatches.
- Final schema check: PASS (train 440/5399, validation 0/601, total 440, 0 errors).

Technical topics covered by this batch: ten more speculative-decoding items, in
three sub-clusters distinguished only by the user prompt. (1) corpus-00478..00480
— "what assumptions must be stated before making a performance claim about
speculative decoding"; the rewrite enumerates the hardware/topology, workload
distribution and concurrency, sampling and acceptance-rule, draft/target pairing
with measured E[a] and c_draft/c_target, and baseline-definition assumptions.
(2) corpus-00481..00485 — "how speculative decoding changes between training and
inference"; the rewrite makes the point that it is an inference-only execution
strategy that leaves loss and gradients untouched, that training is already
compute-bound under teacher forcing so there is no idle bandwidth to convert,
that the only training-side connection is draft distillation to raise acceptance,
and that a relaxed acceptance rule used in an RL rollout phase biases the
gradient estimator. (3) corpus-00487, corpus-00489 — "give one misleading
intuition and correct it"; the rewrite targets the belief that speculation is
strictly free and raises throughput, correcting it with the latency-vs-throughput
inversion at high concurrency, the speedup ≈ E[a]/(1 + gamma·c_draft/c_target)
relation showing acceptance rate alone is insufficient, and the memory cost of
resident draft weights plus speculated KV slots displacing admitted concurrency.
Every rewrite carries an explicit mechanism (t_step ≈ model_bytes_read /
effective_HBM_bandwidth in the memory-bound decode regime), at least one boundary
condition, a falsifiable hypothesis, the evidence needed to test it, and a
rollback gate.

All ten source answers were the same two-clause definition. They are technically
accurate but answer none of the three distinct questions, so the uniform decision
is `rewrite` with instruction_coverage=1, technical_correctness=4, and
operational_safety=2 (the source omits the memory/capacity cost and does not
distinguish the lossless rejection-sampling acceptance rule from relaxed
acceptance).

Blindness: this batch was produced without reading, opening, grepping or
otherwise inspecting any file under experiments/2026-08-14-teacher-a-corpus-calibration/.
No teacher-A corrected_answer was visible at any point. Agreement-rate analysis
is a separate, later step and is deliberately out of scope here.

These results are PROVISIONAL second-opinion review output. They are not expert
gold labels, they have not been validated against measurements on real hardware,
and they say nothing about any model's domain capability.

## Run 2026-08-17 batch 0043

- Batch file: results/train-batch-0043.jsonl
- Corpus range: train.jsonl lines 421-430 (0-indexed 420-429), source IDs corpus-00468 through corpus-00477 — strict corpus order, nothing skipped or reordered.
- Progress: train 430/5399, validation 0/601, total 430/6000, remaining 5570
- Decisions: keep 0, rewrite 10, reject 0
- Initial schema check: PASS (ad-hoc verifier — JSONL line-parse, batch count 10, all 12 required fields, enum values for teacher_lane/teacher_model/calibration_status/decision, exact source_user/source_assistant equality against corpus, non-empty corrected_answer, confidence in [0,1], globally unique source_id, train/validation aggregate strictly a prefix of each corpus).
- Repairs: none required for the batch data itself; it passed on first verification. Follow-up repair to the tooling: an ad-hoc negative control (7 injected defects — dropped field, wrong lane, invalid decision, blank corrected_answer, out-of-range confidence, one-character source_user drift, out-of-range quality dimension) showed `verify_batches.py` hard-coded an absolute results/ path, so a sandboxed copy silently re-verified the pristine originals. Fixed by resolving results/ relative to the script (corpus path stays canonical). After the fix all 7 mutations are caught and the real batches still PASS.
- Manifest: MANIFEST.sha256 regenerated over all files in the experiment directory (excluding the manifest itself); `sha256sum -c` returned OK for every entry.
- Final schema check: PASS (train 430/5399, validation 0/601, total 430, 0 errors), re-run after the verifier fix.

Technical topics covered by this batch: ten speculative-decoding items in three
sub-clusters. (1) corpus-00468..00470 — "how speculative decoding interacts with
latency, throughput, or memory": rewrites give the memory-bound decode mechanism
(t_step ~= model_bytes_read / effective_HBM_bandwidth, weights streamed once per
step regardless of verified positions), the break-even inequality
E[accepted] > 1 + gamma * c_draft/c_target, the latency-wins-vs-throughput-loses
inversion at saturation, and the memory cost of resident draft weights plus
speculated KV slots reducing admitted concurrency. (2) corpus-00471..00475 —
measurement plans: open-loop arrival-rate load generation rather than closed-loop
concurrency, per-segment acceptance logging, gamma x concurrency sweeps, fixed-seed
output-equivalence gating, clock pinning and warmup discard, interleaved A/B repeats
with confidence intervals. (3) corpus-00476..00477 — assumptions required before any
performance claim: regime (memory- vs compute-bound), concurrency, draft config,
per-segment acceptance, sampling parameters and whether the acceptance rule is exact
rejection sampling or a lossy relaxation, metric/percentile definition, memory
accounting, and environment/versions.

All ten source_assistant strings are the same 150-character generic one-liner, which
is not technically wrong but omits the mechanism, the boundary condition, the
throughput inversion, and all memory accounting — hence uniform `rewrite` with
technical_correctness 3, instruction_coverage 2, operational_safety 2, confidence 0.79.

These outputs are PROVISIONAL teacher-B second opinions from a blind review. They are
NOT expert gold labels, have not been validated against measurements on real hardware,
and say nothing about any model's domain capability. Agreement analysis against
teacher-A is a separate later step and was deliberately not performed here; no
teacher-A artifact was read while producing this batch.

## Run 2026-08-17 batch 0042

- Batch file: results/train-batch-0042.jsonl
- Corpus range: train.jsonl lines 411-420 (0-indexed 410-419), source IDs corpus-00458 through corpus-00467 — strict corpus order, nothing skipped or reordered.
- Progress: train 420/5399, validation 0/601, total 420/6000, remaining 5580
- Decisions: keep 0, rewrite 10, reject 0
- Initial schema check: PASS (ad-hoc verifier — JSONL line-parse, batch count 10, all 12 required fields, enum values for teacher_lane/teacher_model/calibration_status/decision, exact source_user/source_assistant equality against corpus, non-empty corrected_answer, confidence in [0,1], globally unique source_id, train/validation aggregate strictly a prefix of each corpus).
- Repairs: none required; the batch passed on first verification.
- Final schema check: PASS (VERIFY_PASS, train 420/5399, validation 0/601, total 420, 0 errors).
- Manifest: MANIFEST.sha256 regenerated over all 83 files in the experiment directory (excluding the manifest itself); `sha256sum -c` returned OK for all 83 entries.

Technical topics covered by this batch: all ten items are speculative-decoding
concept questions in three sub-clusters. (1) corpus-00458..00460 — contrast
against naive autoregressive decode: the memory-bandwidth-bound decode regime
where per-token latency is bytes_read / effective_HBM_bandwidth, the
draft-propose / single-pass-target-verify mechanism, the modified
rejection-sampling acceptance rule min(1, p_target/p_draft) with residual
resampling that makes the technique provably distribution-preserving rather
than an approximation, and the inversion boundary where a compute-bound target
step makes verification cost scale with block size so speculation becomes a net
loss. (2) corpus-00461..00465 — failure modes and trade-offs: negative speedup
above a measurable crossover batch size B*, acceptance-rate collapse under
distribution shift (code, non-English, structured output, high sampling
temperature) with the superlinear sensitivity of expected accepted tokens
(1-alpha^(gamma+1))/(1-alpha) - 1 to alpha, wasted draft+verify work below
break-even alpha, widened p99 inter-token latency variance even when mean
throughput improves, HBM taken from the KV pool by draft weights, and outright
breakage on tokenizer/vocabulary mismatch. (3) corpus-00466..00467 — the
latency / throughput / memory interaction: TPOT improves while TTFT is
essentially unchanged because prefill is already parallel and compute-bound,
aggregate throughput can regress at high concurrency because speculation
converts spare FLOPs into latency and there are none to convert, and memory
pressure arrives twice (resident draft weights reducing max concurrency, plus
gamma+1 speculative KV positions per sequence requiring correct rollback on
rejection — a silent-corruption surface rather than a crashing one).

All ten source answers were single-sentence definitional stubs that stated the
draft/verify mechanism but supplied no boundary condition, no acceptance-rule
detail, no measurement plan and no rollback gate, so every item was scored
instruction_coverage 2 and operational_safety 2 and marked `rewrite`.

These results are PROVISIONAL teacher-B second opinions produced blind (no
teacher-A artifact was read, opened, or grepped while producing this batch).
They are NOT expert gold labels, they are NOT adjudicated, and they say nothing
about any model's domain capability — they are corpus-review annotations only.

## Run 2026-08-17 batch 0041

- Batch file: results/train-batch-0041.jsonl
- Corpus range: train.jsonl lines 401-410 (0-indexed 400-409), source IDs corpus-00447 through corpus-00457 (corpus-00455 is absent from the source file) — strict corpus order, nothing skipped or reordered.
- Progress: train 410/5399, validation 0/601, total 410/6000, remaining 5590
- Decisions: keep 0, rewrite 10, reject 0
- Initial schema check: PASS (verify_batches.py — JSONL line-parse, batch count 10, all 12 required fields, enum values, exact source_user/source_assistant equality with corpus, non-empty corrected_answer, confidence in [0,1], globally unique source_id, train/validation prefix ordering).
- Repairs: none required; the batch passed on first verification.
- Final schema check: PASS (VERIFY_RESULT=PASS, train 410/5399, validation 0/601, ERRORS 0).
- Manifest: MANIFEST.sha256 regenerated over all 80 files in the experiment directory; `sha256sum -c` verified all entries OK.

Technical topics covered by this batch: (1) NCCL collective diagnosis — hang vs. throughput-regression triage, ring allreduce cost model 2*(N-1)/N*S bytes over 2*(N-1) synchronous steps and why the reporting rank is usually the victim rather than the cause, layered bisection across topology / transport / process group / workload, the watchdog-timeout boundary condition where a legitimately slow large-buffer collective is misread as deadlock, and the safety rule that NCCL_P2P_DISABLE / NCCL_IB_DISABLE are diagnostics and not production fixes. (2) Speculative decoding — draft-propose / target-verify mechanism, why memory-bandwidth-bound decode leaves FLOP headroom that batched verification converts into throughput, modified rejection sampling giving provable losslessness, the expected-speedup relation (1 + E[accepted]) / (1 + k*c), and the two collapse regimes (low acceptance from a domain-shifted draft, and high concurrency where continuous batching has already consumed the FLOP headroom so aggregate tokens/s can regress).

All ten source assistant answers in this batch were one-line generic stubs that restated a taxonomy or a definition without the requested concrete mechanism, boundary condition, quantities, or evidence, hence a uniform `rewrite` decision with instruction_coverage = 1.

Status caveat: these results are **provisional** teacher-B output produced by a general-purpose model under blind review. They are NOT expert gold labels, have not been validated by a human domain expert, and say nothing about any trained model's domain capability. Agreement analysis against teacher-A is a separate, later step and was deliberately not consulted here to avoid anchoring.

## Run 2026-08-17 batch 0040

- Batch file: results/train-batch-0040.jsonl
- Corpus range: train.jsonl lines 391-400 (0-indexed 390-399), source IDs corpus-00437 through corpus-00446 — strict corpus order, nothing skipped or reordered.
- Progress: train 400/5399, validation 0/601, total 400/6000, remaining 5600
- Decisions: keep 0, rewrite 10, reject 0
- Initial schema check: PASS on first run (ad-hoc verifier reported train=400, validation=0, total=400, VERIFY_PASS).
- Repairs performed: none required.
- Final schema check: PASS — line-by-line JSONL parse, batch count 10, all 12 required fields present, teacher_lane/teacher_model/calibration_status/decision values correct, source_user and source_assistant character-identical to corpus, corrected_answer non-empty, confidence in [0,1], quality_dimensions integers 1-5, risks/evidence_required string arrays, source_id globally unique across all 40 batches, aggregated train sequence a strict prefix of train.jsonl.
- Manifest: MANIFEST.sha256 regenerated over all files in this directory except itself (79 entries); `sha256sum -c` passed with zero failures.
- Technical topics covered: another all-NCCL block, all ten source answers being the same one-line taxonomy stub, hence rewrite across the board. (a) Misleading-intuition correction — the claim that allreduce bandwidth scales with rank count, corrected with the ring cost model (2*(N-1)/N*S bytes per rank, 2*(N-1) steps, per-step payload S/N), the consequence that added ranks buy latency and not throughput at fixed S, the latency-bound crossover where NCCL switches to Tree/CollNet/NVLS and scaling turns logarithmic, and a falsifiable busbw-flatness prediction testable with all_reduce_perf at N=2/4/8. (b) Controlled-experiment design separating transport/topology causes from workload causes — arms over default vs NCCL_P2P_DISABLE vs NCCL_IB_DISABLE vs NCCL_ALGO=Tree, 5 repeats with warm-up discard, the silent-downgrade mechanism (per-peer transport chosen at communicator init, visible in NCCL_DEBUG=INFO NET/IB vs NET/Socket lines), the boundary condition that sweep sizes must overlap the real DDP bucket size or results do not transfer, and a >=3% end-to-end step-time rollback threshold. (c) Runbook for collective hang/timeout — lock-step semantics meaning the watchdog reports victims rather than the culprit rank, evidence capture before mutation (py-spy dumps per rank, collective name and sequence number, NCCL_DEBUG_SUBSYS=INIT,NET,GRAPH), the topology/transport/process-group/rank/timeout/workload taxonomy, dmesg Xid and ECC checks, out-of-band nccl-tests reproduction, and the explicit rule that raising the timeout is not a fix unless a measured step-time distribution justifies it.
- Status: these outputs are PROVISIONAL teacher-B judgements from a single blind model pass. They are not expert gold labels, were produced without any access to teacher-A artifacts, have not been validated against ground truth, and say nothing about any trained model's domain capability.

## Run 2026-08-17 batch 0039

- Batch file: results/train-batch-0039.jsonl
- Corpus range: train.jsonl lines 381-390 (0-indexed 380-389), source IDs corpus-00423, corpus-00424, corpus-00425, corpus-00426, corpus-00428, corpus-00430, corpus-00431, corpus-00432, corpus-00434, corpus-00435 — strict corpus order, nothing skipped or reordered.
- Progress: train 390/5399, validation 0/601, total 390/6000, remaining 5610
- Decisions: keep 0, rewrite 10, reject 0
- Initial schema check: PASS on first run (ad-hoc verifier reported train=390, validation=0, TOTAL 390, VERIFY=PASS).
- Repairs performed: none required.
- Final schema check: PASS — line-by-line JSONL parse, batch count 10, all 12 required fields present, teacher_lane/teacher_model/calibration_status/decision values correct, source_user and source_assistant character-identical to corpus, corrected_answer non-empty, confidence in [0,1], quality_dimensions integers 1-5, risks/evidence_required string arrays, source_id globally unique across all 39 batches, aggregated train sequence a strict prefix of train.jsonl.
- Manifest: MANIFEST.sha256 regenerated over all files in this directory except itself (77 entries); `sha256sum -c` passed with zero failures.
- Technical topics covered: an all-NCCL block on measurement discipline and regime separation. (a) Measurement plans for validating whether an NCCL change helps a serving workload — deriving all-reduce message size from batch*hidden*dtype under tensor parallelism, the latency-bound (<~256 KB/rank) vs bandwidth-bound (>~4-16 MB/rank) crossover that decides which knob can matter, profiler attribution of ncclKernel share as the hard ceiling on end-to-end gain, one-variable A/B with a concurrency ladder, pre-registered effect sizes, >=5 repeats and second-node reproduction; plus the disaggregated prefill/decode (Mooncake / NVIDIA Dynamo style) KV-transfer variant covering GPUDirect RDMA preconditions (nvidia_peermem, GPU-NIC PCIe affinity), RoCE PFC/ECN counters as a co-tenant externality check, and NCCL_IB_HCA pinning. (b) The assumption set that must precede any NCCL performance claim — topology and per-peer transport selection, collective type/size/rank count, algbw vs busbw (a 2x reporting error if conflated), NCCL/CUDA/driver/NIC-firmware version matrix, GDR and NUMA affinity state, clock/power capping, fabric isolation, warmup and repeat discipline, and the overlap question that determines whether a collective-time reduction can produce any end-to-end gain at all. (c) Training-vs-inference divergence — gradient bucketing pushing training into the bandwidth-bound regime with comm/compute overlap, versus per-layer per-token decode all-reduces in the latency-bound regime where CUDA graph capture typically beats NCCL env tuning; communicator lifetime and the gang-scheduling constraint that makes a TP group an atomic scheduling and failure unit; NCCL buffer VRAM competing directly with KV cache and therefore with max concurrency; and opposite-signed timeout policy (patient watchdog plus checkpoint restart for training, fast detection plus TP-group eviction and N+1 headroom for serving), with SIGSTOP fault injection as the falsification test for detection latency.
- Status: these outputs are PROVISIONAL teacher-B judgements from a single blind model pass. They are not expert gold labels, were produced without any access to teacher-A artifacts, have not been validated against ground truth, and say nothing about any trained model's domain capability.

## Run 2026-08-17 batch 0038

- Batch file: results/train-batch-0038.jsonl
- Corpus range: train.jsonl lines 371-380 (0-indexed 370-379), source IDs corpus-00412, corpus-00413, corpus-00414, corpus-00415, corpus-00416, corpus-00417, corpus-00418, corpus-00419, corpus-00420, corpus-00421 — strict corpus order, nothing skipped or reordered.
- Progress: train 380/5399, validation 0/601, total 380/6000, remaining 5620
- Decisions: keep 0, rewrite 10, reject 0
- Initial schema check: PASS (verify_batches.py reported train=380 validation=0 total=380, VERIFY_PASS) on first run.
- Repairs performed: none required.
- Final schema check: PASS — line-by-line JSONL parse, batch count 10, all 12 required fields present, teacher_lane/teacher_model/calibration_status/decision values correct, source_user and source_assistant byte-identical to corpus, corrected_answer non-empty, confidence in [0,1], source_id globally unique across all 38 batches, aggregated train sequence a strict prefix of train.jsonl.
- Manifest: MANIFEST.sha256 regenerated over all files in this directory except itself; `sha256sum -c` passed with zero failures.
- Technical topics covered: an all-NCCL block. Failure modes and trade-offs (silent NVLink/P2P fallback to SHM/PCIe under IOMMU or cross-root-complex topology and its size-dependent visibility; collective/timeout coupling where a straggler rank makes a victim rank report the abort; buffer/channel memory-vs-bandwidth trade-off via NCCL_BUFFSIZE x channels and SM contention; ring-vs-tree crossover shifting with world size and fabric latency; GPUDirect RDMA preconditions — nvidia_peermem, PCIe ACS, NCCL_NET_GDR_LEVEL — and host-staging fallback; RoCE PFC/DCQCN misconfiguration surfacing only under incast; duplicate device binding from missing LOCAL_RANK; watchdog timeout floor derived from measured p99). Latency/throughput/memory interaction (ring cost model 2(N-1)/N*S, LL/LL128/Simple protocol crossover, NVLS/CollNet in-switch reduction, NCCL buffers competing with KV cache outside the framework allocator, TP decode being latency-exposed with no compute to hide behind). Plus a full measurement plan for validating an NCCL configuration change on a serving workload: message sizes derived from hidden_size, fixed request trace, one-variable A/B with >=3 repetitions, KV-cache capacity accounting, Nsight attribution, and an acceptance/rollback gate.
- Status: these outputs are PROVISIONAL teacher-B judgements from a single blind model pass. They are not expert gold labels, were produced without any access to teacher-A artifacts, have not been validated against ground truth, and say nothing about any trained model's domain capability.

## Run 2026-08-17 batch 0037

- Batch file: results/train-batch-0037.jsonl
- Corpus range: train.jsonl lines 361-370 (0-indexed 360-369), source IDs corpus-00399, corpus-00401, corpus-00402, corpus-00403, corpus-00405, corpus-00406, corpus-00407, corpus-00409, corpus-00410, corpus-00411 — strict corpus order, nothing skipped or reordered.
- Progress: train 370/5399, validation 0/601, total 370/6000, remaining 5630
- Decisions: keep 0, rewrite 10, reject 0
- Initial schema check: PASS (train=370, validation=0, total=370, ERRORS 0) on first run.
- Repairs performed: none required.
- Final schema check: PASS — JSONL line-by-line parse, batch count 10, all 12 required fields present, teacher_lane/teacher_model/calibration_status/decision values correct, source_user and source_assistant byte-identical to corpus, corrected_answer non-empty, confidence within [0,1], source_id globally unique, aggregated train sequence is a strict prefix of train.jsonl.
- Manifest: MANIFEST.sha256 regenerated over all files except itself; `sha256sum -c` reported 75/75 OK, zero failures.
- Technical topics covered: quantization diagnosis on a serving deployment (weight-only INT4 as a memory-traffic rather than FLOP optimization, the memory-bound-to-compute-bound crossover, absence of FP8 tensor cores on A30, outlier-driven accuracy loss); and a dense NCCL block — definition and role as the transport under DP/FSDP/TP/EP, ring all-reduce cost model 2*(R-1)/R*N versus O(R*N) coordinator funnels, ring vs tree vs NVLS algorithm and LL/LL128 protocol selection, per-peer transport selection (NVLink P2P > PCIe P2P > SHM > net), GPUDirect RDMA preconditions and silent host-staging fallback, NCCL_NET_GDR_LEVEL topology dependence, compute/communication overlap limits and SM contention, and collective-schedule-mismatch hangs presenting as 100% GPU utilization until watchdog abort.
- Status: these outputs are PROVISIONAL teacher-B judgements from a single model pass. They are not expert gold labels, have not been validated against ground truth or against teacher-A (this lane is blind), and say nothing about any trained model's domain capability.

## Run 2026-08-17 batch 0036

- Batch file: results/train-batch-0036.jsonl
- Corpus range: train.jsonl lines 351-360 (0-indexed 350-359), source IDs corpus-00389, corpus-00390, corpus-00391, corpus-00392, corpus-00393, corpus-00394, corpus-00395, corpus-00396, corpus-00397, corpus-00398 — strict corpus order, nothing skipped or reordered.
- Progress: train 360/5399, validation 0/601, total 360/6000, remaining 5640
- Decisions: keep 0, rewrite 10, reject 0
- Initial schema check: PASS (verify script reported VERIFY_PASS on first run; train=360 validation=0 total=360)
- Repairs performed: none
- Final schema check: PASS (JSONL parseable line-by-line, 10 records, all 12 required fields present, teacher_lane/teacher_model/calibration_status/decision values valid, source_user and source_assistant character-identical to the corpus, corrected_answer non-empty, confidence in [0,1], source_id globally unique across all 36 batches, aggregated train sequence a strict prefix of train.jsonl)
- Manifest: MANIFEST.sha256 regenerated over all files in this directory except itself; `sha256sum -c` re-verified, all entries OK.
- Technical topics covered: the quantization block continues with three instruction framings. (a) Misleading-intuition correction (2 items): "INT4 is 4x smaller so decode is 4x faster" corrected with the decode byte budget (weights + KV + activations), the concrete arithmetic for a 9B model (~18 GB FP16 weights vs ~5.3 GB W4A16-g128 including scale/zero overhead), the fact that FP16 KV grows with batch x sequence and can rival the weight term at batch 32 / 8k, and the fused-dequant-GEMM requirement (Marlin on SM80) without which naive NF4 kernels lose to FP16 at batch >= 8; and "quantization error is uniform noise" corrected with persistent activation outlier channels 20-100x the median collapsing per-tensor scales, making the loss concentrate in a few layers (first block, down_proj input, attention output projection) and making per-channel / group-wise scaling, SmoothQuant-style scale migration, and mixed-precision keep-lists produce discontinuous rather than smooth recovery. (b) Controlled experiment design (5 items): W4A16 accuracy-vs-throughput with clock locking, greedy decoding, warmup discard and a declared 1.0-point budget; KV-cache 8-bit quantization tested for concurrency headroom via the KV byte formula (2 x layers x kv_heads x head_dim x seq_len x bytes_per_elem) with long-context retrieval as the acceptance metric rather than perplexity; layer-wise ablation with a random-keep-list negative control to test whether outlier ranking has explanatory power; calibration-corpus and calibration-size sensitivity as a 3x3 factorial with 3 draws per cell and an exact-hash contamination check against the eval set; and a measurement-integrity protocol whose six checks (warmup, clock lock, equal output length, GPU isolation, nsight dram__bytes.sum mechanism confirmation, full operating-point grid) can each independently falsify a claimed speedup. (c) Runbook entries (3 items): accuracy-regression triage (revert first, rule out template/sampling drift before blaming weights, reproduce offline greedy, separate KV-quantization failures — long-context and exact-copy — from weight-quantization failures, then layer-localize and check calibration hygiene and shard hashes); load/OOM triage (shard sha256 vs manifest, scheme-to-kernel-to-SM mapping with the explicit note that A30/SM80 has no FP8 tensor-core path, the counterintuitive mechanism where smaller weights cause a LARGER KV pool under a utilization-fraction allocator and therefore MORE OOM under load, load-time vs run-time OOM separation, and a one-variable-at-a-time fallback ladder, plus the TP boundary condition that group size must divide the per-rank shard); and an adoption-review checklist (identify the binding constraint first — capacity-bound, decode-bandwidth-bound, or prefill-compute-bound, where weight-only quantization helps little — reject bit-width-only proposals, require a pre-declared accuracy budget, require production operating points rather than batch-1 numbers, and require a canary with an automatic revert threshold; the boundary condition is that "no" is often correct when the model already fits and the cluster is idle at peak).
- Every rewritten answer states an explicit mechanism, an explicit boundary condition, a falsifiable prediction with numeric thresholds where applicable, the evidence that would refute it, and a rollback/exit gate.
- Why all ten were marked `rewrite`: all ten source_assistant values are the same generic sentence reused verbatim ("Quantization reduces representation precision and often memory traffic, but accuracy, kernel support, calibration, and outlier handling must be measured"). It answers none of the three distinct requested tasks in this range and supplies neither the concrete mechanism nor the boundary condition the instruction explicitly demands.
- Provisional status: these are provisional teacher-B judgements produced blind (teacher-A outputs were not read during this run). They are NOT expert gold labels, have NOT been externally validated, and say nothing about any model's domain capability. All A30/SM80 and model-size figures in the rewritten answers are engineering estimates, not measurements.

## Run 2026-08-17 batch 0035

- Batch file: results/train-batch-0035.jsonl
- Corpus range: train.jsonl lines 341-350 (0-indexed 340-349), source IDs corpus-00377, corpus-00379, corpus-00380, corpus-00381, corpus-00383, corpus-00384, corpus-00385, corpus-00386, corpus-00387, corpus-00388 — strict corpus order, nothing skipped or reordered (the absent IDs simply do not occupy these positions in the corpus file).
- Progress: train 350/5399, validation 0/601, total 350/6000, remaining 5650
- Decisions: keep 0, rewrite 10, reject 0
- Initial schema check: PASS (verify script reported VERIFY=PASS on first run; train=350 validation=0 total=350)
- Repairs performed: none
- Final schema check: PASS (JSONL parseable line-by-line, 10 records, all 12 required fields present, teacher_lane/teacher_model/calibration_status/decision values valid, source_user and source_assistant character-identical to the corpus, corrected_answer non-empty, confidence in [0,1], source_id globally unique across all 35 batches, aggregated train sequence a strict prefix of train.jsonl)
- Manifest: MANIFEST.sha256 regenerated over all 73 files in this directory except itself; `sha256sum -c` re-verified, all entries OK.
- Technical topics covered: the quantization block continues, with three distinct instruction framings in this range. (a) Assumption hygiene before a performance claim (3 items): scheme specification beyond bit width (weight-only vs W8A8/FP8, group size, symmetric/asymmetric, KV quantization bundled separately, excluded layers such as lm_head/embeddings that make a nominal 4x compression closer to 3.3x), kernel and SM-version dependence (Marlin/Machete-class fused dequant-GEMM must exist; FP8 has no tensor-core path on SM80, so Hopper FP8 compute claims do not transfer to A30), operating point (concurrency, prefill vs decode, TTFT vs TPOT, percentile), calibration/eval protocol, and baseline tuning parity as the dominant source of inflated wins. (b) Training vs inference quantization (4 items): the constraint differs — training is bounded by gradient numerics (BF16 compute with FP32 accumulation and FP32/8-bit master weights, BF16 chosen over FP16 to avoid loss scaling, optimizer state rather than weights dominating memory at ~72 GB FP32 Adam vs ~18 GB BF16 weights for a 9B, so ZeRO/FSDP sharding and 8-bit optimizers are the real training-side levers), whereas inference is bounded by HBM traffic and single-pass output error (GPTQ layerwise reconstruction, AWQ salient-channel protection, SmoothQuant outlier migration, KV quantization as an inference-only lever). The compounding mechanism is stated explicitly: quantization error at inference perturbs one forward pass and is bounded by the group scale, while in training the same error enters every update and small gradients below the quantization step round to zero, producing silent plateau rather than a loud failure. QAT (straight-through estimator, biased gradient) and QLoRA (frozen NF4 base, BF16 adapters, memory win but a 20-40% step-time penalty) are placed as the boundary cases that break the clean split. (c) Misleading-intuition correction (3 items): "4-bit is 4x faster" corrected via the three conditions that must all hold before size becomes speed; "quantization loss is small and uniform" corrected via activation outlier channels 10-100x the median collapsing a per-tensor INT8 grid to roughly 1 effective bit for ordinary values, with the weight-vs-activation asymmetry noted as the limit of that correction; and "freed memory means proportional concurrency" corrected with the additive arithmetic (total - weight_bytes_new)/(total - weight_bytes_old), showing the same quantization gives ~3.5-4x KV headroom on a 24 GB A30 but only ~1.2x on an 80 GB H100, and evaporating entirely at 32k context where KV dominates — pointing to KV quantization, GQA/MLA, or Mooncake-style KV pooling and Dynamo-style prefill/decode disaggregation as the correct levers there.
- Every rewritten answer states an explicit mechanism, an explicit boundary condition (compute-bound crossover at batch * activation_bytes ≈ weight_bytes; KV traffic dominating weight traffic at long context; SM80 lacking FP8 tensor cores so the Hopper training-side FP8 win does not exist; weight distributions being well behaved so the outlier argument applies to activations not weights; QLoRA buying memory but not speed), a falsifiable prediction with numeric thresholds, the evidence that would refute it, and a rollback gate (≤1.0 absolute point held-out accuracy loss, ≤10% p99 TTFT regression, <1% preemption/recompute rate, BF16 checkpoint kept warm).
- Why all ten were marked `rewrite`: all ten source_assistant values are the same generic sentence reused verbatim ("Quantization reduces representation precision and often memory traffic, but accuracy, kernel support, calibration, and outlier handling must be measured"). It answers none of the three distinct requested tasks in this range, and supplies neither the concrete mechanism nor the boundary condition the instruction explicitly demands. instruction_coverage was scored 2 for the train-vs-inference and misleading-intuition items because the source does not even gesture at the requested contrast or correction, and 3 for the assumption items where the sentence at least names some of the required assumptions.
- Blind-review compliance: no file under experiments/2026-08-14-teacher-a-corpus-calibration/ was read, opened, grepped, or listed at any point during this batch.
- Status caveat: these are **provisional** teacher-B blind-review outputs produced by an LLM reviewer. They are NOT expert gold labels, have not been validated by a human domain expert, and constitute no evidence about any model's domain capability. Hardware generations are cited only as constraints on which kernels can run, never as a capability claim.

## Run 2026-08-17 batch 0034

- Batch file: results/train-batch-0034.jsonl
- Corpus range: train.jsonl lines 331-340 (0-indexed 330-339), source IDs corpus-00367 through corpus-00376 inclusive, strict corpus order, nothing skipped or reordered.
- Progress: train 340/5399, validation 0/601, total 340/6000, remaining 5660
- Decisions: keep 0, rewrite 10, reject 0
- Initial schema check: PASS (verify script reported VERIFY_PASS on first run; train=340 validation=0 total=340)
- Repairs performed: none
- Final schema check: PASS (JSONL parseable line-by-line, 10 records, all 12 required fields present, teacher_lane/teacher_model/calibration_status/decision values valid, source_user and source_assistant character-identical to corpus, corrected_answer non-empty, confidence in [0,1], source_id globally unique across all batches, aggregated train sequence a strict prefix of train.jsonl)
- Manifest: MANIFEST.sha256 regenerated over every file in this directory except itself; `sha256sum -c` re-verified, all entries OK.
- Technical topics covered: continuation of the quantization block, but shifting from definitional items to (a) the latency/throughput/memory interaction (4 items) and (b) measurement-plan and assumption-hygiene items (6 items). The rewrites for the interaction items supply four distinct mechanisms rather than repeating one: the bandwidth-bound decode floor TPOT ≈ (weight_bytes + kv_bytes_touched)/effective_HBM_bandwidth with worked A30-class arithmetic (933 GB/s, ~18 GB BF16 weights vs ~4.8 GB group-128 int4 for a 9B model); the three separable memory budgets (weight residency including scale/zero-point metadata, KV bytes = 2·layers·kv_heads·head_dim·dtype_bytes·seq_len, and activation/CUDA-graph workspace that quantization does not shrink) and the freed-HBM-to-KV-blocks-to-admitted-concurrency conversion path; kernel-availability discontinuity, where a missing fused W4A16 kernel (Marlin/Machete-class) forces a dequantize-then-GEMM fallback that is strictly slower than not quantizing, gated by architecture (int4/int8 tensor-core paths on SM80, FP8 compute only from SM90, NVFP4 from Blackwell) and by tensor-parallel divisibility of the group size; and the accuracy-side mechanism where activation outlier channels 10-100x the median force a coarse per-tensor scale, with per-channel/group scaling, SmoothQuant range migration, and GPTQ/AWQ calibration-aware rounding each addressing a different part of that failure. The six measurement-plan items are deliberately differentiated by framing rather than duplicated: latency-benchmark hygiene (clock pinning, warmup discard, concurrency sweep 1/4/16/64/128, SLO-constrained goodput instead of peak tokens/s, 3 repetitions with spread), capacity validation (predicted vs observed max admitted sequences at 2k/8k/32k, scheduler preemption and recompute counters, weight-quantization and KV-quantization varied one at a time), quality validation (checkpoint sha256 pinning, tiered eval with perplexity as smoke test only and tool-call schema validity plus long-context retrieval as the actual gate, enforced calibration/eval disjointness, confidence intervals against a same-day baseline), mechanism attribution (pre-registered predicted TPOT floor and ratio, ncu dram__bytes_read per decode step, resolved-runtime-config diff to catch attention-backend or max_num_batched_tokens changes shipped alongside the quantized checkpoint), production rollout (trace replay with production input/output length and arrival distributions, concurrent rather than sequential canary to avoid diurnal confounds, warm baseline deployment as a one-routing-change rollback), and assumption enumeration (hardware and compute capability, model geometry and checkpoint hash, full quantization scheme specification since a bit width alone underspecifies granularity and calibration, resolved runtime config, workload distribution, metric definition, measurement hygiene, and explicit labeling of each number as measured vs estimated vs vendor-quoted).
- Every rewritten answer states an explicit boundary condition (compute-bound crossover where W4A16 dequant overhead becomes pure loss; freed KV capacity ceasing to help once decode is scheduler- or compute-bound, whose signature is rising p99 with zero preemptions; kernel/architecture gating that makes an FP8 claim on SM80 a storage claim not a compute claim; weight-only quantization being accuracy-safe while activation quantization degrades long-context and multilingual traffic first; benchmark length distribution mismatch systematically misranking the arms; attribution breaking whenever more than one variable moved; non-stationary traffic mix invalidating a sequential A/B), a falsifiable prediction, the evidence that would refute it, and a rollback gate.
- Why all ten were marked `rewrite`: all ten source_assistant values are the identical generic sentence "Quantization reduces representation precision and often memory traffic, but accuracy, kernel support, calibration, and outlier handling must be measured," reused verbatim. It answers none of the three distinct requested tasks in this range (explain the latency/throughput/memory interaction; give a measurement plan; enumerate assumptions required before a performance claim), and supplies neither the concrete mechanism nor the boundary condition the instruction explicitly demands.
- Blind-review compliance: no file under experiments/2026-08-14-teacher-a-corpus-calibration/ was read, opened, grepped, or listed at any point during this batch.
- Status caveat: these are **provisional** teacher-B blind-review outputs produced by an LLM reviewer. They are NOT expert gold labels, have not been validated by a human domain expert, and constitute no evidence about any model's domain capability. Hardware generations are cited only as constraints on which kernels can run, never as a capability claim.

## Run 2026-08-17 batch 0033

- Batch file: results/train-batch-0033.jsonl
- Corpus range: train.jsonl lines 321-330 (0-indexed 320-329), source IDs corpus-00354, corpus-00356, corpus-00357, corpus-00358, corpus-00360, corpus-00361, corpus-00362, corpus-00363, corpus-00364, corpus-00366 (strict corpus order; nothing skipped or reordered — corpus-00355, corpus-00359 and corpus-00365 do not occupy these positions in the corpus)
- Progress: train 330/5399, validation 0/601, total 330/6000, remaining 5670
- Decisions: keep 0, rewrite 10, reject 0
- Initial schema check: PASS (verify script VERIFY_PASS on first run; train=330/5399 validation=0/601 total=330)
- Repairs performed: none
- Final schema check: PASS (JSONL parseable line-by-line, 10 records, all 12 required fields present, teacher_lane/teacher_model/calibration_status/decision values valid, source_user and source_assistant character-identical to corpus, corrected_answer non-empty, confidence in [0,1], source_id globally unique across all batches, aggregated train sequence a strict prefix of train.jsonl)
- Manifest: MANIFEST.sha256 regenerated over every file in this directory except itself; `sha256sum -c` re-verified.
- Technical topics covered: this batch is entirely quantization, spanning definition, contrast-with-naive-baseline, failure-mode enumeration, and the latency/throughput/memory interaction. The rewrites separate the three channels that the one-line source conflates: (a) capacity — bits/weight sets resident weight bytes (~18 GB BF16 vs ~5 GB group-128 int4 for a 9B model) and the freed HBM converts into KV-cache blocks, i.e. concurrency rather than single-stream latency; (b) bandwidth — decode step time has a floor of (weight bytes + KV bytes)/achieved HBM bandwidth, which is the only channel where weight-only quantization directly lowers latency, and only when dequant is fused into the GEMM rather than materialised back to HBM; (c) compute — low-precision tensor-core paths (int8/int4 on Ampere-class, FP8 and narrower on Hopper-class and newer), which W4A16 does not touch at all, so prefill throughput is unchanged or slightly worse. Additional mechanisms supplied: group-wise scales along the input-channel axis as the reason group int4 is usable while per-tensor int4 is not; SmoothQuant's exact scale migration X·W → (X/s)(diag(s)W) folded into the preceding layernorm, valid only when outlier channel indices are stable across tokens; FP8 e4m3 KV cache with per-head versus per-tensor scaling and the attention-sink/massive-activation outlier heads that break the per-tensor variant on long-context retrieval; GPTQ's inverse-Hessian error compensation and AWQ's activation-aware channel protection as the reason RTN int4 has a sharper cliff; calibration-distribution mismatch producing targeted in-domain regressions; kernel-specific packed layouts (interleaved nibbles, permuted groups matched to a CUTLASS/Marlin tile shape) making quantized artifacts non-portable and version-pinned; and error amplification over long autoregressive generations, which perplexity is close to blind to but exact-match structured-output and tool-call evals expose.
- Each rewritten answer states an explicit boundary condition (bandwidth-bound-only benefit with a predicted compute-bound crossover batch B*; FP8 on Ampere-class yielding memory savings but no GEMM acceleration; per-channel scale migration failing when outlier positions move per token; super-additive degradation when weight and KV quantization are shipped together), a falsifiable prediction with the measurement that would refute it, the evidence required, and a rollback gate (p99 TPOT regression at production batch, accuracy loss beyond a pre-registered budget, or profiling showing an unfused dequant path).
- Why all ten were marked `rewrite`: every source_assistant is the identical generic sentence "Quantization reduces representation precision and often memory traffic, but accuracy, kernel support, calibration, and outlier handling must be measured," reused verbatim across all ten case variants. It performs none of the four distinct requested tasks (define, contrast against a naive baseline, give two failure modes, explain the latency/throughput/memory interaction) and supplies neither the concrete mechanism nor the boundary condition the instruction explicitly demands.
- Blind-review compliance: no file under experiments/2026-08-14-teacher-a-corpus-calibration/ was read, opened, grepped, or listed at any point during this batch.
- Status caveat: these are **provisional** teacher-B blind-review outputs produced by an LLM reviewer. They are NOT expert gold labels, have not been validated by a human domain expert, and constitute no evidence about any model's domain capability. Hardware generations are cited only as constraints on which kernels can run, never as a capability claim.

## Run 2026-08-17 batch 0032

- Batch file: results/train-batch-0032.jsonl
- Corpus range: train.jsonl lines 311-320 (0-indexed 310-319), source IDs corpus-00342, corpus-00343, corpus-00344, corpus-00346, corpus-00347, corpus-00348, corpus-00350, corpus-00351, corpus-00352, corpus-00353 (strict corpus order; nothing skipped or reordered — corpus-00345 and corpus-00349 do not appear at these positions in the corpus)
- Progress: train 320/5399, validation 0/601, total 320/6000, remaining 5680
- Decisions: keep 0, rewrite 10, reject 0
- Initial schema check: PASS (verify.py VERIFY_PASS on first run; train=320/5399 validation=0/601 total=320)
- Repairs performed: none
- Final schema check: PASS (JSONL parseable line-by-line, 10 records, all 12 required fields, teacher_lane/teacher_model/calibration_status/decision values valid, source_user and source_assistant byte-identical to corpus, corrected_answer non-empty, confidence in [0,1], source_id globally unique, train sequence a strict prefix of train.jsonl)
- Manifest: MANIFEST.sha256 regenerated over every file in this directory except itself; `sha256sum -c` passed (70 entries, MANIFEST_OK)
- Technical topics covered: two families. Items 1-7 (corpus-00342 … corpus-00350) are Mixture-of-Experts, split between controlled-experiment design and runbook authoring. The rewrites supply the mechanisms the one-line source omits: top-k routing with capacity factor C and the dispatch/combine all-to-all as a *synchronizing* collective whose step time tracks the max-loaded expert rank rather than the mean; the training-versus-inference inversion from load-balanced-by-loss (auxiliary balance loss, token dropping allowed) to load-balanced-by-luck (frozen router, small correlated decode batches, drop-less serving); active-versus-total parameter accounting (active = shared + k·expert drives FLOPs, total = shared + E·expert drives HBM residency, so sizing on active params underestimates memory by ~E/k); router collapse as a positive-feedback failure mode with router z-loss and logit norms as the earliest actionable signal; expert replication versus expert parallelism as the lever that removes the collective from the decode path; and prefill/decode disaggregation (Dynamo/Mooncake-style) so all-to-all cost lands where batches amortize it. Items 8-10 (corpus-00351 … corpus-00353) are quantization: weight-only int4 with group-wise scales as an HBM-traffic (not arithmetic) optimization, hence large decode gains and small prefill gains; KV-cache quantization arithmetic (2·layers·kv_heads·head_dim·bytes/token) with the asymmetry that K error is amplified through the softmax exponential while V error averages out; and post-training calibration, where scales fitted on unrepresentative data cause targeted per-slice regressions that aggregate benchmark scores hide.
- Each rewritten answer states an explicit boundary condition (batch-1 decode being HBM-weight-load bound so routing skew is nearly free; validity only while the all-to-all is on the critical path; token-budget-per-expert floors below which one measures undertraining rather than capacity; KV quantization buying nothing when weights dominate HBM; missing fused kernels for a given format/architecture), a falsifiable hypothesis, the evidence required to believe it, and a rollback gate.
- Why all ten were marked `rewrite`: every source_assistant is one of two identical generic sentences (MoE routing/capacity/all-to-all, or quantization precision/accuracy) reused verbatim across case variants. None performs the requested task (design an experiment, write a runbook entry, define with one mechanism), and none supplies the boundary condition the instruction explicitly demands.
- Blind-review compliance: no file under experiments/2026-08-14-teacher-a-corpus-calibration/ was read, opened, or searched at any point during this batch.
- Status caveat: these are **provisional** teacher-B blind-review outputs produced by an LLM reviewer. They are NOT expert gold labels, have not been validated by a human domain expert, and constitute no evidence about any model's domain capability.

## Run 2026-08-17 batch 0031

- Batch file: results/train-batch-0031.jsonl
- Corpus range: train.jsonl lines 301-310 (0-indexed 300-309), source IDs corpus-00332 through corpus-00341 (strict corpus order; nothing skipped or reordered)
- Progress: train 310/5399, validation 0/601, total 310/6000, remaining 5690
- Decisions: keep 0, rewrite 10, reject 0
- Initial schema check: PASS (verify.py VERIFY_PASS on first run; no repairs needed)
- Repairs performed: none
- Final schema check: PASS (JSONL parseable, 10 records, 12 required fields, lane/model/status/decision values valid, source_user and source_assistant byte-identical to corpus, corrected_answer non-empty, confidence in [0,1], source_id globally unique, train sequence is a strict prefix of train.jsonl)
- Manifest: MANIFEST.sha256 regenerated over all files except itself; `sha256sum -c` MANIFEST_OK (67 entries)
- Technical topics covered: all ten items are Mixture-of-Experts (MoE) Knowledge/Concept items in three sub-families — (a) how MoE differs between training and inference, (b) misleading MoE intuitions to correct, (c) designing a controlled MoE experiment. The rewrites make explicit the mechanisms the one-line source answer omits: token-level top-k routing with an auxiliary load-balancing loss and capacity factor at training time versus a frozen router with no balancing pressure and drop-less serving; the inversion from active-FLOP-bound training to HBM-capacity- and weight-bandwidth-bound decode (all experts resident, only k read); the two synchronizing all-to-alls (dispatch/combine) per MoE layer under expert parallelism whose cost tracks the most-loaded rank rather than the mean; permute/un-permute and ragged grouped-GEMM overheads; prefill/decode asymmetry motivating disaggregated serving (Mooncake/Dynamo-style); expert-count scaling trading HBM and collective latency for quality; and the fallacy that experts are human-interpretable topical specialists. Each answer states a boundary condition (typically tokens-per-step >> num_experts/k, or expert replication removing the all-to-all entirely), a falsifiable prediction, required evidence (per-expert token histograms, comm-profiler all-to-all share, TTFT/TPOT percentiles, HBM breakdown, iso-quality held-out eval), and a rollback gate.
- Why all ten were marked `rewrite`: every source_assistant is the identical single generic sentence about routing/capacity/all-to-all, which does not answer the specific variant asked and supplies neither mechanism nor boundary condition despite the instruction demanding both.
- Status caveat: these results are **provisional** teacher-B blind review output produced by an LLM reviewer. They are NOT expert gold labels, have not been validated by a human domain expert, and say nothing about any model's domain capability. teacher-A artifacts were not read at any point during this batch.

## Run 2026-08-17 batch 0030

- Batch file: results/train-batch-0030.jsonl
- Corpus range: train.jsonl lines 291-300 (0-indexed 290-299), source IDs corpus-00321, corpus-00323, corpus-00324, corpus-00325, corpus-00326, corpus-00327, corpus-00328, corpus-00329, corpus-00330, corpus-00331 (strict corpus order; nothing skipped or reordered — corpus-00322 does not exist at this position in the corpus)
- Progress: train 300/5399, validation 0/601, total 300/6000, remaining 5700
- Decisions: keep 0, rewrite 10, reject 0
- Initial schema check: PASS (verify.py, VERIFY_PASS, train=300/5399 validation=0/601)
- Repairs: none required this run. One operational note: verify.py must be invoked from the repository root (paths are repo-relative); running it from inside the experiment directory raises FileNotFoundError on research/ai-infra-expert/corpus/train.jsonl. This is a harness invocation detail, not a data defect.
- Final schema check: PASS
- Manifest: MANIFEST.sha256 regenerated over all files in this directory except the manifest itself; `sha256sum -c` reports 66/66 OK, 0 failures.
- Technical topics covered: all ten items are Mixture-of-Experts (MoE) knowledge/concept items in three families — (a) a measurement plan for validating whether MoE helps a serving workload, (b) the assumptions that must be stated before making an MoE performance claim, and (c) how MoE behaves differently in training vs inference. The rewrites make explicit the active-vs-total parameter distinction, router top-k and capacity factor, dispatch/combine all-to-all under expert parallelism, expert load imbalance measured as a per-expert token histogram, the intra-node NVLink domain vs inter-node fabric boundary for the EP group, the decode-time asymmetry where total-parameter HBM residency is paid for active-parameter compute, iso-quality baseline selection, and explicit falsifiable hypotheses with SLO-based rollback gates (p95 TTFT/TPOT, HBM headroom for peak KV cache, expert max/mean load bound).
- Seed-answer assessment: every source_assistant in this batch is the same one-sentence MoE definition, which is technically defensible but does not answer any of the three asked tasks, carries no units, no measurement method and no rollback threshold. Hence decision=rewrite for all ten, with instruction_coverage scored 1.
- Status: PROVISIONAL. These are teacher-B second-opinion labels produced blind (teacher-A outputs were not read at any point during this batch). They are NOT expert gold, have not been validated by a human domain expert, and say nothing about any model's domain capability. Agreement analysis against teacher-A is a separate, later step.

## Run 2026-08-17 batch 0029

- Batch file: results/train-batch-0029.jsonl
- Corpus range: train.jsonl lines 281-290 (0-indexed 280-289), source IDs corpus-00310, corpus-00311, corpus-00312, corpus-00313, corpus-00314, corpus-00315, corpus-00317, corpus-00318, corpus-00319, corpus-00320 (contiguous in corpus order; nothing skipped or reordered — the corpus itself has no corpus-00316 at this position)
- Progress: train 290/5399, validation 0/601, total 290/6000, remaining 5710
- Decisions: keep 0, rewrite 10, reject 0
- Initial schema check: PASS (verify_batches.py, ERRORS=0) — no repair actions were required this run
- Repairs: none
- Final schema check: PASS (train 290/5399, validation 0/601, ERRORS=0, VERIFY_RESULT=PASS)
- Manifest: MANIFEST.sha256 regenerated over all files in this directory except the manifest itself; `sha256sum -c` reports 64/64 OK, 0 failures
- Topics covered: Mixture-of-Experts. All ten items are the same seed sentence about token routing, reduced active compute, and routing/capacity/all-to-all concerns, asked under three instruction shapes (contrast against a naive dense implementation, two failure modes/trade-offs, interaction with latency/throughput/memory). Rewrites supply what the source omits: the capacity-factor mechanism and silent token dropping; the auxiliary load-balancing loss and routing collapse; router z-loss and bf16 logit numerics; batch-dependent non-determinism at capacity saturation; dispatch/combine all-to-all as a bisection-bandwidth-bound critical path and its exposure of RoCE/PFC/ECN and GPUDirect RDMA fallback behaviour; the arithmetic-intensity crossover that makes MoE lose to dense in low-concurrency decode; the HBM partition between expert weights and KV cache and its effect on maximum concurrency; token permute/unpermute HBM traffic; prefill/decode disaggregation (Mooncake-style KV-centric and NVIDIA Dynamo-style) as the natural fit for MoE's phase asymmetry; and expert offload with GDS-style paths bounded by expert-reuse and prefetch-hit-rate conditions.
- Every rewrite states assumptions, names one concrete mechanism and one boundary condition, and ends with the evidence required plus an explicit rollback gate. No numeric threshold is asserted as a measured fact for any specific cluster.
- Status: PROVISIONAL. This is a single-model blind second opinion, not expert gold, not adjudicated against teacher-A, and it is not evidence of any model's domain capability. No teacher-A artifact was read, opened, or searched during this run.

## Run 2026-08-17 batch 0028

- Batch file: results/train-batch-0028.jsonl
- Corpus range: train.jsonl lines 271-280, source IDs corpus-00300 through corpus-00309 (contiguous in corpus order; nothing skipped or reordered)
- Progress: train 280/5399, validation 0/601, total 280/6000, remaining 5720
- Decisions: keep 0, rewrite 10, reject 0
- Initial schema/prefix check: PASS on first run (verify_batches.py, ERRORS: 0)
- Repairs performed: none required
- Final schema check: VERIFY_RESULT=PASS (train 280/5399 strict corpus prefix, validation 0/601)
- Manifest: MANIFEST.sha256 regenerated over all 61 files in this directory (excluding itself and __pycache__); `sha256sum -c` passed with no mismatches
- Technical topics covered: pipeline parallelism runbook diagnostics (1F1B bubble fraction (P-1)/(M+P-1), stage imbalance, P2P link placement, activation-memory ceiling on raising microbatch count, small-GEMM efficiency floor); Mixture-of-Experts definition and infrastructure implications (top-k routing decoupling capacity from per-token FLOPs, expert parallelism turning the FFN into two all-to-all collectives, capacity factor and silent token dropping under routing skew); MoE vs dense FFN contrast (all-reduce vs all-to-all fabric stress, NVLink vs cross-node RoCE/IB behaviour, regimes where dense wins on wall clock, decode-time HBM residency of all experts).
- Every record carries explicit falsifiable hypotheses, required evidence (per-expert token histograms, drop rate, profiler traces over >=3 steady-state steps, per-rank peak memory, NCCL topology dumps) and numeric rollback gates.
- These outputs are **provisional** teacher-B review only. They are NOT expert gold labels, have not been validated by a human domain expert, and say nothing about any model's domain capability. Blind review discipline held: no teacher-A artifact was read, opened or grepped during this batch.

## Run 2026-08-17 batch 0027

- Batch file: results/train-batch-0027.jsonl
- Corpus range: train.jsonl lines 261-270, source IDs corpus-00290 through corpus-00299 (contiguous in corpus order; nothing skipped or reordered)
- Progress: train 270/5399, validation 0/601, total 270/6000, remaining 5730
- Decisions: keep 0, rewrite 10, reject 0
- Initial schema check: the first generator run aborted before writing any output with `KeyError: 'corpus-00300'`, because the slice offset was set to lines 271-280 instead of the correct next-unprocessed window 261-270. No file was produced by that attempt, so no partial or misaligned batch ever reached results/.
- Repairs applied: the generator's corpus slice was corrected from `[270:280]` to `[260:270]` after confirming, from results/train-batch-0026.jsonl and the corpus index, that the last processed record was corpus-00289 at corpus index 259. The prefix boundary was verified against the corpus itself, not assumed. No corpus file, no prior batch, and no teacher-A artifact was modified. A fresh reusable verifier was written to verify.py for this run.
- Final schema check: PASS (verify.py -> VERIFY_PASS, train=270/5399, validation=0/601, total=270). Checks enforced: line-by-line JSONL parse; exactly 10 records per batch file; all 12 required fields present; teacher_lane=teacher-B, teacher_model=claude-opus-5-current, calibration_status=provisional, decision in keep/rewrite/reject; source_user and source_assistant character-exact against the corpus; corrected_answer non-empty; confidence in [0,1]; quality_dimensions three integers in 1-5; risks and evidence_required string arrays; source_id globally unique across all 270 records; and the aggregated train sequence is a strict prefix of corpus order.
- Manifest: MANIFEST.sha256 regenerated over all 59 files in this directory excluding itself; `sha256sum -c` reports all OK, 0 failures.
- Second repair (caught by post-commit ad-hoc verification): the aborted first-attempt generator `build_batch_0027.py` survived into the first commit, because its cleanup `rm` had been chained behind the command that failed. A throwaway verifier at /tmp/hermes-verify-teacher-b-0027.py re-checked the committed state from a clean process and flagged the stray file; it was removed from the index and the working tree and MANIFEST.sha256 was regenerated (60 -> 59 files). The removal ships as a separate follow-up commit rather than an amend: rewriting an already-pushed commit would have required a force-push, which is not appropriate for an unattended job. No batch data, corpus file, prior batch, or teacher-A artifact was touched by the repair.
- Lock: /tmp/teacher-b-corpus-review.lock acquired atomically at run start (mkdir, owner.timestamp written), released at run end.
- Blind protocol: no file under experiments/2026-08-14-teacher-a-corpus-calibration/ was read, opened, listed or grepped during this run. teacher-A corrected answers remain unseen by this lane.
- Technical topics covered: all ten items are pipeline-parallelism (PP) prompts in three structural families. (a) corpus-00290 asks for a misleading intuition and its correction: the rewrite targets the "enough microbatches gives near-linear PP speedup, so PP is a cheap way to cross a node boundary" belief, and corrects it in two parts — the synchronous bubble (P-1)/(M+P-1) never reaches zero and is a *lower* bound on the loss, and PP's real ceiling is max(slowest stage, exposed boundary communication). It works the arithmetic (P=8, M=8 gives a 47% bubble; roughly M >= 4P is needed to get under ~6%) and flags that raising M inflates global batch size, which is a training-dynamics change rather than a free throughput knob. (b) corpus-00291..00295 ask for a small controlled experiment: the rewrite sweeps M in {P, 2P, 4P, 8P} against a two-arm design — Arm A with all stage boundaries intra-node, Arm B with one boundary forced across the RDMA/RoCE fabric — because a single-arm sweep cannot distinguish a pipeline bubble from an exposed send, which is the load-bearing design decision. It fixes the control set (P, TP, DP, micro_bs, S, dtype, schedule, recomputation, NCCL env, driver/NCCL/framework versions, rank-to-node map), requires >=20 warmup and >=5 measured steps reported as median and p90, states the refutation conditions (no ~1/M decay in Arm A means stage imbalance, not bubble; Arm B divergence past M=4P means communication-bound), and pre-empts the global-batch-size confound by scoping the experiment to throughput only. (c) corpus-00296..00299 ask for a runbook entry: the rewrite is a six-step read-only-first triage — pin the config, compare predicted vs measured bubble, check per-stage balance (embedding and LM-head stages are the usual >10% outliers, and rebalancing is the largest win that costs no memory), measure the actual boundary link with ib_write_bw / nccl-tests instead of the nameplate and confirm GPUDirect RDMA is genuinely active via NCCL_DEBUG=INFO rather than silently host-staged, then handle OOM separately using the fact that peak activation memory scales with in-flight microbatches (~P under 1F1B) and not with M, so raising M is memory-neutral while raising P is not.
- Every rewrite in this batch states its mechanism explicitly, at least one boundary condition where the bubble formula stops holding (stage imbalance, inter-node activation payload micro_bs*S*H*dtype_bytes exceeding what the measured link can hide inside one stage's compute window, and the in-flight-microbatch memory model), a falsifiable prediction with the measurement that would refute it, the evidence required, and an explicit rollback gate (revert if tokens/s regresses >3%, if any rank's peak memory exceeds 90% of device memory, or if the loss curve leaves the seed-noise band within 200 steps). All ten source answers are again the identical single generic sentence ("Pipeline parallelism partitions layers across stages and can require microbatching to reduce pipeline bubbles"), which answers none of the three question forms — it proposes no experiment for the experiment prompts, no ordered steps or rollback gate for the runbook prompts, and for corpus-00290 it restates a mechanism in place of the correction the prompt asks for. Hence uniform decision=rewrite with instruction_coverage=1.
- Status caveat: these results are **provisional teacher-B review output produced by a general-purpose model under a blind protocol**. They are NOT expert gold labels, they have not been validated by a human domain expert or by execution against real hardware, and they say nothing about any trained model's domain capability. Every quantitative expression above (bubble fractions, memory scaling, payload arithmetic) is analytic or order-of-magnitude, not measured on this cluster.

## Run 2026-08-17 batch 0026

- Batch file: results/train-batch-0026.jsonl
- Corpus range: train.jsonl lines 251-260, source IDs corpus-00280 through corpus-00289 (contiguous in corpus order; nothing skipped or reordered)
- Progress: train 260/5399, validation 0/601, total 260/6000, remaining 5740
- Decisions: keep 0, rewrite 10, reject 0
- Initial schema check: PASS on first run (verify_batches.py -> VERIFY_PASS, train=260/5399, validation=0/601, total=260)
- Repairs applied: one pre-write correction inside the generator script only — the initial `quality_dimensions` table scored the *rewritten* answer (5/5/4) instead of the *source* assistant answer; it was corrected to (3/1/2) before the batch was emitted, to stay consistent with the convention used by batches 0001-0025. No corpus file, no prior batch, and no teacher-A artifact was modified.
- Final schema check: PASS (12 required fields per record; teacher_lane=teacher-B, teacher_model=claude-opus-5-current, calibration_status=provisional, decision in keep/rewrite/reject; source_user and source_assistant character-exact against corpus; corrected_answer non-empty; confidence in [0,1]; quality_dimensions three integers in 1-5; risks and evidence_required string arrays; source_id globally unique across all 260 records; aggregated train sequence is a strict prefix of corpus order)
- Manifest: MANIFEST.sha256 regenerated over all 56 files in this directory excluding itself; `sha256sum -c` reports all OK, 0 failures
- Lock: /tmp/teacher-b-corpus-review.lock acquired atomically at run start (owner.timestamp written), released at run end
- Blind protocol: no file under experiments/2026-08-14-teacher-a-corpus-calibration/ was read, opened, listed or grepped during this run. teacher-A corrected answers remain unseen by this lane.
- Technical topics covered: all ten items remain pipeline-parallelism (PP) prompts, in three structural families. (a) corpus-00280 asks what assumptions must precede a PP performance claim: the rewrite enumerates five assumption classes (model shape, topology and *measured* boundary-link bandwidth/latency rather than nameplate, schedule + microbatch count M, composition with TP/DP/ZeRO/recomputation, and measurement protocol including warm-up and whether the timer covers the optimizer step), then gives the mechanism — PP moves only [mb, S, H] boundary activations point-to-point, so bubble fraction is (P-1)/(M+P-1) and 1F1B bounds peak activation memory to ~P in-flight microbatches — and the boundary M < P where PP loses to TP or DP. (b) corpus-00281..00285 ask how PP differs between training and inference: the rewrites all hinge on one invariant (PP always ships only boundary hidden states) and one variable (what fills the pipe — M microbatches you choose in training versus exogenous request concurrency in serving), and derive from it the four concrete divergences: the bubble knob exists only in training; peak memory shifts from stashed activations + optimizer state to a layer-partitioned KV cache; payload shifts from bandwidth-bound [mb,S,H] to latency-bound [batch,1,H], which is why decode PP dies on TCP/Ethernet boundaries but survives on RDMA/RoCE with GDR; and the failure signature shifts from a uniform step-time increase to a p99 tail with a healthy p50. Each states the occupancy boundary (M < P in training, in-flight concurrency < P in serving) and the operational trap of reusing the training parallelism plan for serving. (c) corpus-00286..00289 ask for a misleading intuition and its correction, and the rewrites target four distinct real misconceptions: that PP scales like DP (it scales capacity, not throughput — headline benefit is memory, and interleaved 1F1B only buys (P-1)/(v*M+P-1) by multiplying boundary crossings by v); that PP lowers per-request latency (it strictly raises it by (P-1) hops, since only TP splits a single layer's math); that switching GPipe to 1F1B reduces the bubble (both have the same bubble fraction — 1F1B is a memory schedule that *unlocks* a larger M, which is what actually reduces it); and that PP is network-insensitive because its volume is low (low volume with per-message overhead on a synchronous critical path is exactly the GDR-versus-host-bounce regime, where a silent GPUDirect fallback adds two PCIe copies per hop and is detectable by comparing NCCL_DEBUG=INFO transport lines against ib_write_lat).
- Every rewrite in this batch states its mechanism, at least one boundary condition, a falsifiable prediction with the measurement that would refute it, the evidence required (per-stage timing histograms and send/recv wait fraction, ib_write_lat / ib_write_bw / nccl-tests on the exact boundary path, torch.cuda.max_memory_allocated per stage, concurrency sweeps separating p50 from p99), and an explicit rollback gate. All ten source answers are again the identical single generic sentence ("Pipeline parallelism partitions layers across stages and can require microbatching to reduce pipeline bubbles"), which answers none of the three question forms and, for corpus-00288 specifically, restates the very misconception the prompt asks to correct. Hence uniform decision=rewrite with instruction_coverage=1.
- Status caveat: these results are **provisional teacher-B review output produced by a general-purpose model under a blind protocol**. They are NOT expert gold labels, they have not been validated by a human domain expert or by execution against real hardware, and they say nothing about any trained model's domain capability. Every quantitative expression above (bubble fractions, payload sizes, hop-latency ranges) is analytic or order-of-magnitude, not measured on this cluster.

## Run 2026-08-17 batch 0025

- Batch file: results/train-batch-0025.jsonl
- Corpus range: train.jsonl lines 241-250, source IDs corpus-00268, corpus-00270 through corpus-00277, corpus-00279 (contiguous in corpus order; the ID gaps are gaps in the corpus itself, nothing was skipped or reordered by this run)
- Progress: train 250/5399, validation 0/601, total 250/6000, remaining 5750
- Decisions: keep 0, rewrite 10, reject 0
- Initial schema check: PASS on first run (scripts/verify_batches.py -> VERIFY_PASS, train=250/5399, validation=0/601)
- Repairs applied: none. No corpus file, no prior batch, and no teacher-A artifact was modified.
- Final schema check: PASS (12 required fields per record; teacher_lane=teacher-B, teacher_model=claude-opus-5-current, calibration_status=provisional, decision in keep/rewrite/reject; source_user and source_assistant character-exact against corpus; corrected_answer non-empty; confidence in [0,1]; quality_dimensions three integers in 1-5; risks and evidence_required string arrays; source_id globally unique across all 250 records; aggregated train sequence is a strict prefix of corpus order)
- Manifest: MANIFEST.sha256 regenerated over all files in this directory excluding itself; `sha256sum -c` reports all OK, 0 failures
- Lock: /tmp/teacher-b-corpus-review.lock acquired atomically at run start (owner.timestamp written), released at run end
- Blind protocol: no file under experiments/2026-08-14-teacher-a-corpus-calibration/ was read, opened, listed or grepped during this run. teacher-A corrected answers remain unseen by this lane.
- Technical topics covered: all ten items are pipeline-parallelism (PP) prompts, in three structural families that this batch answers with three distinct rewrites. (a) corpus-00268/00270 ask how PP interacts with latency, throughput and memory: the rewrite separates the three axes explicitly — PP does *not* reduce single-request latency and adds (P-1) stage-hop serializations to TTFT and inter-token latency; it raises aggregate throughput only when enough microbatches/requests are in flight; and it divides parameter+optimizer state by ~1/P while 1F1B still forces stage 0 to retain up to ~P in-flight microbatches of activations, which is the reason naive GPipe OOMs where 1F1B survives. The stated boundary conditions are M < P (bubble fraction (P-1)/(M+P-1) exceeds 50%, PP loses to TP/DP) and a slow stage-boundary link where per-hop activation transfer exceeds per-stage compute, making added stages monotonically worse. (b) corpus-00271..00275 ask for a measurement plan: the rewrite is a six-step protocol that fixes the decision metric and rollback gate *before* running (revert if the primary metric gains < 10% or p95 TTFT regresses > 15%), baselines with a production-derived request trace under a concurrency sweep, holds total GPU count and batching policy constant across the TP8 / TP4-PP2 / TP2-PP4 arms, and instruments the mechanism directly (per-stage NVTX busy fraction, measured bubble fraction, stage-boundary transfer time) rather than inferring it from end-to-end throughput. It requires confirming that the inter-node stage boundary genuinely rides RDMA/RoCE (NCCL_DEBUG=INFO, IB port counters) because a silent NCCL TCP fallback invalidates the result, and it names three competing falsifiable hypotheses — including H3, that PP's real throughput gain is often the *indirect* one of freeing memory for a larger KV-cache budget rather than the compute split itself. Confounder control (locked clocks, warm-up, pinned framework/CUDA/driver/NCCL versions, no co-tenants) and a <= 5% canary with automatic revert are mandatory. (c) corpus-00276/00277/00279 ask what assumptions must precede a PP performance claim: the rewrite enumerates seven — hardware/topology (NVLink domain vs PCIe vs IB vs RoCEv2, GDR active or not), the full (TP, PP, DP, EP, SP) tuple at matched GPU count, the schedule (GPipe vs 1F1B vs interleaved) with the microbatch count M, workload shape and arrival process, numerics/quantization/recompute policy and KV-cache sizing, the measurement protocol with dispersion, and an explicit measured-vs-estimated split — and closes with the rule that if M < P or concurrency < P, any claimed speedup is almost certainly a confounder rather than pipelining.
- Every rewrite in this batch states its mechanism, at least one boundary condition, a falsifiable prediction, the evidence required to test it, and a rollback gate, which is exactly what the source assistant answers omit: all ten source answers are the identical single generic sentence ("Pipeline parallelism partitions layers across stages and can require microbatching to reduce pipeline bubbles"), which is technically true but answers none of the three distinct question forms. Hence uniform decision=rewrite with instruction_coverage=1.
- Status caveat: these results are **provisional teacher-B review output produced by a general-purpose model under a blind protocol**. They are NOT expert gold labels, they have not been validated by a human domain expert or by execution against real hardware, and they say nothing about any trained model's domain capability. Every quantitative expression above (bubble fraction, memory arithmetic) is analytic, not measured on this cluster.

## Run 2026-08-17 batch 0024

- Batch file: results/train-batch-0024.jsonl
- Corpus range: train.jsonl lines 231-240, source IDs corpus-00258 through corpus-00267 (contiguous in corpus order; nothing skipped or reordered)
- Progress: train 240/5399, validation 0/601, total 240/6000, remaining 5760
- Decisions: keep 0, rewrite 10, reject 0
- Initial schema check: PASS on first run (scripts/verify_batches.py -> VERIFY_RESULT=PASS, train=240/5399, validation=0/601, ERRORS=0). An independent second verifier written this run reproduced the same result (24 batch files, 240 records, prefix-exact, no duplicate source_id).
- Repairs applied: one process defect, no data defect. The independent verifier was first written to experiments/2026-08-17-teacher-b-corpus-review/verify_batches.py, which overwrote an existing tracked copy of the canonical verifier at that path. The file was restored byte-exact with `git checkout -- verify_batches.py` and the canonical scripts/verify_batches.py was used as the authoritative check. No batch data, no corpus file, no prior batch, and no teacher-A artifact was modified.
- Final schema check: PASS (12 required fields per record; teacher_lane=teacher-B, teacher_model=claude-opus-5-current, calibration_status=provisional, decision in keep/rewrite/reject; source_user and source_assistant character-exact against corpus; corrected_answer non-empty; confidence in [0,1]; quality_dimensions three integers in 1-5; risks and evidence_required string arrays; source_id globally unique across all 240 records; aggregated train sequence is a strict prefix of corpus order)
- Manifest: MANIFEST.sha256 regenerated over all files in this directory excluding itself; `sha256sum -c` reports all OK, 0 failures
- Lock: /tmp/teacher-b-corpus-review.lock acquired atomically at run start (owner.timestamp written), released at run end
- Blind protocol: no file under experiments/2026-08-14-teacher-a-corpus-calibration/ was read, opened, listed or grepped during this run. teacher-A corrected answers remain unseen by this lane.
- Technical topics covered: all ten items are pipeline-parallelism (PP) explanation prompts in three structural families. corpus-00258/00259/00260 ask for a contrast against a naive non-PP implementation; the rewrite sizes the naive baseline concretely (9B bf16 + Adam ~= 18 GB weights + 18 GB fp32 master + 36 GB moments ~= 72 GB before activations, infeasible on a 24 GB A30), identifies PP's distinguishing mechanism as point-to-point send/recv of a single (microbatch, seq, hidden) boundary activation rather than a per-layer collective — hence O(S-1) messages independent of parameter count, which is why PP tolerates fabrics that would destroy tensor parallelism — and states the bubble boundary (S-1)/(M+S-1) with worked values (43% at S=4,M=4 versus 8.6% at S=4,M=32) together with the opposing arithmetic-intensity boundary where microbatches below ~1-2k tokens stop saturating the GEMMs. corpus-00261..00265 ask for two failure modes; the rewrite gives the bubble and stage-imbalance/activation-skew pair, notes that step time is set by the slowest stage because every boundary send/recv synchronizes, quantifies the LM-head logits tensor ((microbatch, seq, vocab) ~= 1.2 GB at vocab 150k, seq 4096, bf16) as the usual cause of tail-stage OOM, and gives two competing falsifiable hypotheses distinguished by whether throughput improves when M is raised. corpus-00266/00267 ask about the latency/throughput/memory interaction; the rewrite separates parameter-state memory (divides by S) from 1F1B activation memory (stage s holds ~(S-s) microbatches, so peak memory stops falling with deeper pipelines), and states that PP is actively hostile to single-request decode latency because one token traverses all S stages serially with no microbatch overlap available, making continuous batching, not microbatching, the thing that keeps a serving pipeline full. Every rewrite ends with a falsifiable prediction, the evidence required (per-stage timers, peak memory per stage, tokens/s/GPU swept over M, p50/p99 TTFT and TPOT versus a TP-only baseline, nvidia-smi topo -m) and an explicit rollback gate.
- Source quality note: the source_assistant string is byte-identical across all ten records ("Pipeline parallelism partitions layers across stages and can require microbatching to reduce pipeline bubbles.") while the prompts ask three structurally different things (contrast, two failure modes, latency/throughput/memory interaction) and every prompt explicitly demands one concrete mechanism and one boundary condition. One generic sentence supplies neither, so instruction_coverage=2 and decision=rewrite for all ten. The sentence is not technically wrong (technical_correctness=4); operational_safety=3 reflects the absence of any evidence requirement, threshold, or rollback gate rather than an active hazard.
- Status caveat: all 10 records are PROVISIONAL single-model teacher-B output. They are not expert gold, have not been validated by a human domain expert or by execution, and say nothing about any model's domain capability. Consistency analysis against teacher-A is a separate, later, out-of-scope step.

## Run 2026-08-17 batch 0023

- Batch file: results/train-batch-0023.jsonl
- Corpus range: train.jsonl lines 221-230, source IDs corpus-00246, corpus-00248, corpus-00250, corpus-00251, corpus-00252, corpus-00253, corpus-00254, corpus-00255, corpus-00256, corpus-00257 (contiguous in corpus line order; nothing skipped or reordered — the corpus itself skips 00247 and 00249)
- Progress: train 230/5399, validation 0/601, total 230/6000, remaining 5770
- Decisions: keep 0, rewrite 10, reject 0
- Initial schema check: PASS on first run (scripts/verify_batches.py -> VERIFY_RESULT=PASS, train=230/5399, validation=0/601)
- Repairs applied: none to the batch data — it passed every check on the first verification pass. One verifier defect was found afterwards by an ad-hoc negative test (/tmp/hermes-verify-tb0023-negative.py, 17 injected defects, since removed): the per-batch 10-record count check was absent, so a truncated batch would have passed silently. Fixed in scripts/verify_batches.py by asserting n==10 per batch, exempting only a final batch that actually exhausts the corpus. All 17 defects (tampered source_user, tampered source_assistant, whitespace-only corrected_answer, missing required field, out-of-range confidence, wrong teacher_lane, wrong teacher_model, wrong calibration_status, invalid decision, duplicate source_id, out-of-range and wrong-typed quality dimensions, reordered prefix, short batch, extra 13th field, non-list risks, and joined physical lines) are now detected; batch 0023 was restored byte-exact (sha256 389c8bf3…) and re-verified PASS. No corpus file, no prior batch, and no teacher-A artifact was modified.
- Final schema check: PASS (12 required fields per record and exactly 12 keys; teacher_lane=teacher-B, teacher_model=claude-opus-5-current, calibration_status=provisional, decision in keep/rewrite/reject; source_user and source_assistant character-exact against corpus; corrected_answer non-empty; confidence in [0,1]; quality_dimensions three integers in 1-5; risks and evidence_required string arrays; source_id globally unique across all 230 records; aggregated train sequence is a strict prefix of corpus order)
- Manifest: MANIFEST.sha256 regenerated over all files in this directory excluding itself; `sha256sum -c` reports 51/51 OK, 0 failures
- Lock: /tmp/teacher-b-corpus-review.lock acquired atomically at run start (owner.timestamp written), released at run end
- Blind protocol: no file under experiments/2026-08-14-teacher-a-corpus-calibration/ was read, opened, listed or grepped during this run. teacher-A corrected answers remain unseen by this lane.
- Technical topics covered: this block spans the tensor-parallelism to pipeline-parallelism boundary. corpus-00246/00248/00250 ask for a runbook entry on tensor parallelism; the rewrite supplies the column-parallel/row-parallel MLP pairing and head-wise attention sharding that produce exactly two blocking all-reduces per transformer block, sizes the decode all-reduce payload (batch*hidden*dtype_bytes, ~256 KB at batch=32/hidden=4096/bf16) to place it in the latency-dominated rather than bandwidth-dominated NCCL regime, and states two hard boundary conditions: interconnect-domain homogeneity (a PHB/SYS hop in the TP group inverts the expected gain) and the divisibility requirement that TP divide both num_attention_heads and num_key_value_heads, past which GQA KV heads are replicated and the memory argument for raising TP disappears. corpus-00251..00255 ask for a definition of pipeline parallelism; the rewrite distinguishes depth-partitioning with point-to-point activation transfer from TP's intra-layer collectives, derives the 1F1B bubble fraction (P-1)/(m+P-1) with worked values (75% at m=1, ~16% at m=16, ~4.5% at m=64 for P=4), explains why 1F1B caps in-flight activation memory at ~P rather than m microbatches versus GPipe, and gives three boundaries: stage imbalance making throughput slowest-stage-bound, microbatch shrinkage pushing GEMMs below GPU saturation so the optimum m must be measured, and the sequential-dependence argument for why PP does not help autoregressive decode latency. corpus-00256/00257 ask for a contrast against the naive non-PP implementation; the rewrite argues naive single-device placement is the correct default whose failure mode is binary (OOM, not slowdown), positions host offload as the other naive fallback costing a PCIe round trip per offloaded layer per step, and frames the honest comparison as bounded bubble loss versus infeasibility. Every rewrite ends with a falsifiable prediction and an explicit rollback gate.
- Source quality note: the source_assistant strings are again block-constant — one generic sentence for the TP items and one for all eight PP items — while the prompts ask three structurally different things (runbook entry, definition, contrast). A single sentence cannot satisfy any of them, so instruction_coverage=1 and decision=rewrite for all ten. The source sentences are not technically wrong (hence technical_correctness=4), they are merely non-responsive; operational_safety=3 reflects the absence of any evidence requirement or rollback gate rather than an active hazard.
- Status caveat: all 10 records are PROVISIONAL single-model teacher-B output. They are not expert gold, have not been validated by a human domain expert or by execution, and say nothing about any model's domain capability. Consistency analysis against teacher-A is a separate, later, out-of-scope step.

## Run 2026-08-17 batch 0022

- Batch file: results/train-batch-0022.jsonl
- Corpus range: train.jsonl lines 211-220, source IDs corpus-00236 through corpus-00245 (contiguous in corpus order; nothing skipped or reordered)
- Progress: train 220/5399, validation 0/601, total 220/6000, remaining 5780
- Decisions: keep 0, rewrite 10, reject 0
- Initial schema check: PASS on first run (experiments/2026-08-17-teacher-b-corpus-review/verify_batches.py -> VERIFY_PASS, train=220/5399, validation=0/601, total=220/6000)
- Repairs applied: none required on the batch data. One verifier defect was found and fixed post-hoc by an ad-hoc negative test (/tmp/hermes-verify-negative-tb.py, 9 injected defects): verify_batches.py unconditionally exempted the last batch file from the 10-record check, so a truncated final batch would have passed silently. The exemption is now conditional on that batch actually exhausting the corpus. All 9 injected defects (tampered source_user, empty corrected_answer, missing field, out-of-range confidence, wrong teacher_lane, duplicate source_id, out-of-range quality dimension, reordered prefix, short batch) are now detected; batch 0022 data was byte-restored and re-verified unchanged.
- Final schema check: PASS (12 required fields per record; teacher_lane=teacher-B, teacher_model=claude-opus-5-current, calibration_status=provisional, decision in keep/rewrite/reject; source_user and source_assistant character-exact against corpus; corrected_answer non-empty; confidence in [0,1]; quality_dimensions integers 1-5; risks and evidence_required string arrays; source_id globally unique across all batches; aggregated train sequence is a strict prefix of corpus order)
- Manifest: MANIFEST.sha256 regenerated over all files in this directory excluding itself; `sha256sum -c` reports 50/50 OK, 0 failures
- Generator script archived at /tmp/tb_build_0022.py (run-local); verifier committed at verify_batches.py
- Lock: /tmp/teacher-b-corpus-review.lock acquired atomically at run start (owner.timestamp written), released at run end
- Blind protocol: no file under experiments/2026-08-14-teacher-a-corpus-calibration/ was read, opened or grepped during this run
- Technical topics covered: tensor parallelism, two question framings, five case variants each. Items corpus-00236..00240 ask for a misleading intuition and its correction; the five rewrites each attack a distinct false belief: (1) "TP=8 gives 8x throughput" corrected by the column-parallel/row-parallel pairing that yields exactly 2 blocking all-reduces per transformer block, with the ring cost 2*M*(N-1)/N bytes per rank showing the bandwidth term does not shrink with N; (2) "raise TP whenever near OOM" corrected by the replicated per-rank floor (CUDA context, NCCL buffers, fragmentation) and the head-wise KV sharding bound 1/min(TP, num_key_value_heads); (3) "TP is a GPU-speed problem" corrected by interconnect-domain placement, with NCCL_DEBUG transport inspection, GPUDirect RDMA / nvidia-peermem checks, and the NET/Socket fallback as a hard stop; (4) "a training TP degree transfers to inference" corrected by the prefill compute-bound vs decode weight-read-bound asymmetry, decode all-reduce payload of batch*hidden*dtype_bytes landing in the latency-dominated NCCL regime, motivating Mooncake-style / NVIDIA Dynamo-style prefill-decode disaggregation; (5) "use every GPU in the box for TP" corrected by the (TP x PP x DP) budget, replica-count throughput scaling, and the correlated-failure blast radius of a wide TP group. Items corpus-00241..00245 ask for a small controlled experiment; the five rewrites give distinct designs: a pre-registered TP={1,2,4} sweep with a microbenchmark-derived predicted ITL penalty; a batch-pinned three-arm decomposition that separates communication cost from memory benefit; a fabric-validation protocol (topo matrix, NCCL transport lines, busbw curve) that must precede any TP conclusion; a prefill-arm/decode-arm split that refuses blended latency metrics and flags chunked prefill as a confounder; and a statistical-validity design with randomised interleaved repeats, warmup discard, pre-registered primary metric and a stated minimum detectable effect. Every rewrite carries a named mechanism, an explicit boundary condition, a falsifiable prediction, an evidence list and a rollback gate (typically revert on >10% p99 regression at matched offered load).
- Source quality note: all ten source_assistant strings in this block are byte-identical to each other and to the block reviewed in batch 0021 - a single generic sentence about TP. Because the ten prompts ask two structurally different things, that one sentence cannot answer any of them, hence instruction_coverage=1 and decision=rewrite for all ten.
- Status caveat: all 10 records are PROVISIONAL single-model teacher-B output. They are not expert gold, have not been validated by a human domain expert or by execution, and say nothing about any model's domain capability.

## Run 2026-08-17 batch 0021

- Batch file: results/train-batch-0021.jsonl
- Corpus range: train.jsonl lines 201-210, source IDs corpus-00225, corpus-00227, corpus-00228, corpus-00229, corpus-00230, corpus-00231, corpus-00232, corpus-00233, corpus-00234, corpus-00235 (contiguous in corpus order; the corpus itself has no corpus-00226, nothing was skipped or reordered)
- Progress: train 210/5399, validation 0/601, total 210/6000, remaining 5790
- Decisions: keep 0, rewrite 10, reject 0
- Initial schema check: PASS on first run (scripts/verify_batches.py -> VERIFY=PASS, train=210/5399, validation=0/601, total=210/6000)
- Repairs applied: none required this run
- Final schema check: PASS (12 required fields per record; teacher_lane=teacher-B, teacher_model=claude-opus-5-current, calibration_status=provisional, decision in keep/rewrite/reject; source_user and source_assistant character-exact against corpus; corrected_answer non-empty; confidence in [0,1]; quality_dimensions integers 1-5; risks and evidence_required string arrays; source_id globally unique; aggregated train sequence is a strict prefix of corpus order)
- Manifest: MANIFEST.sha256 regenerated over all files in this directory excluding itself; `sha256sum -c` reports 48/48 OK, 0 failures
- Generator script archived at scripts/gen_train_batch_0021.py; corpus peek helper at scripts/peek_0021.py
- Lock: /tmp/teacher-b-corpus-review.lock acquired atomically at run start (owner.timestamp written), released at run end
- Blind protocol: no file under experiments/2026-08-14-teacher-a-corpus-calibration/ was read, opened or grepped during this run
- Technical topics covered: tensor parallelism, three question framings. (1) A measurement plan for validating whether TP helps a serving workload: fix feasibility-vs-speed as the question first, characterise the fabric with nvidia-smi topo -m and nccl-tests in both the small-message decode regime and large-message prefill regime, sweep TP in {1,2,4,8} at concurrency 1/8/64 with TTFT and TPOT reported separately, and establish a five-repeat noise band before interpreting any delta. (2) The assumption set that must be stated before a TP performance claim is auditable: interconnect class and measured latency/bandwidth, prefill-vs-decode phase, metric and load point, model head counts and TP divisibility (num_key_value_heads), baseline plus variance, the unchosen alternative (N data-parallel replicas at equal GPU count), rank health and throttle state, and acceptance of TP's correlated-failure model. (3) How TP differs between training and inference: backward pass roughly doubles collective count in training; message-size regime shifts from bandwidth-bound (training/prefill, O(b*s*h)) to latency-bound (decode, s=1); KV cache exists only at inference and shards as 1/min(TP, num_kv_heads); optimizer-state reduction comes from the DP/ZeRO dimension not from TP; and restart semantics differ so serving should prefer smaller TP groups. All rewrites carry the falsifiable model TPOT(N) = C/N + L(N), explicit PCIe-vs-NVLink crossover claims for 8x A30 24GB, an evidence list, and a rollback gate.
- Status caveat: all 10 records are PROVISIONAL single-model teacher-B output. They are not expert gold, have not been validated by a human domain expert or by execution, and say nothing about any model's domain capability.

## Run 2026-08-17 batch 0020

- Batch file: results/train-batch-0020.jsonl
- Corpus range: train.jsonl lines 191-200, source IDs corpus-00213, corpus-00214, corpus-00215, corpus-00216, corpus-00217, corpus-00219, corpus-00220, corpus-00221, corpus-00222, corpus-00224 (contiguous in corpus order; ID gaps exist in the corpus itself, nothing was skipped or reordered)
- Progress: train 200/5399, validation 0/601, total 200/6000, remaining 5800
- Decisions: keep 0, rewrite 10, reject 0
- Initial schema check: PASS on first run (scripts/verify_batches.py -> VERIFY=PASS, train=200/5399, validation=0/601, total=200/6000)
- Repairs applied: none required this run
- Final schema check: PASS (12 required fields per record; teacher_lane=teacher-B, teacher_model=claude-opus-5-current, calibration_status=provisional, decision in keep/rewrite/reject; source_user and source_assistant character-exact against corpus; corrected_answer non-empty; confidence in [0,1]; quality_dimensions integers 1-5; risks and evidence_required string arrays; source_id globally unique; aggregated train sequence is a strict prefix of corpus order)
- Manifest: MANIFEST.sha256 regenerated over all files in this directory excluding itself; `sha256sum -c` reports 45/45 OK, 0 failures
- Generator script archived at scripts/build_batch_0020.py
- Lock: /tmp/teacher-b-corpus-review.lock acquired atomically at run start, released at run end
- Blind protocol: no file under experiments/2026-08-14-teacher-a-corpus-calibration/ was read, opened or grepped during this run
- Technical topics covered: tensor parallelism end-to-end - Megatron column/row-parallel sharding and the resulting 2 all-reduces per block per pass; prefill (compute-bound) vs decode (HBM-bandwidth-bound) asymmetry; per-token collective latency scaling with num_layers; KV-cache sharding limits under GQA when TP degree exceeds num_kv_heads; head-divisibility constraints; GEMM tile-size degradation at high TP; intra-node NVLink vs cross-node RoCE/IB transport cliffs and GPUDirect RDMA fallback; straggler/blast-radius effects of synchronizing collectives; and measurement methodology (nccl-tests busbw, nsys NCCL-vs-GEMM share, DCGM throttle counters, open-loop production-trace replay, SLO-constrained goodput per GPU, greedy-output equality as a correctness gate, canary + rollback thresholds).
- Status caveat: all 10 records are PROVISIONAL single-model teacher-B output. They are not expert gold, have not been validated by a human domain expert or by execution, and say nothing about any model's domain capability.

## Run 2026-08-17 batch 0019

- Batch file: results/train-batch-0019.jsonl
- Corpus range: train.jsonl lines 181-190, source IDs corpus-00203 .. corpus-00212 (contiguous, corpus order preserved exactly, nothing skipped or reordered)
- Progress: train 190/5399, validation 0/601, total 190/6000, remaining 5810
- Decisions: keep 0, rewrite 10, reject 0
- Initial schema check: PASS on first run (scripts/verify_batches.py -> VERIFY=PASS, train=190/5399, validation=0/601, total=190/6000)
- Repairs applied: none required this run
- Final schema check: PASS (exactly 12 fields per record; teacher_lane=teacher-B, teacher_model=claude-opus-5-current, calibration_status=provisional, decision in keep/rewrite/reject; source_user and source_assistant character-exact against corpus; corrected_answer non-empty; confidence in [0,1]; quality_dimensions integers 1-5; risks and evidence_required string arrays; source_id globally unique; aggregated train sequence is a strict prefix of corpus order)
- Manifest: MANIFEST.sha256 regenerated over all files in this directory excluding itself; `sha256sum -c` reports all OK, 0 failures
- Generator script archived at scripts/gen_train_batch_0019.py
- Lock: /tmp/teacher-b-corpus-review.lock acquired atomically at run start, released at run end
- Blind protocol: no file under experiments/2026-08-14-teacher-a-corpus-calibration/ was read, opened or grepped during this run

Technical topics covered by this batch: all ten items are tensor-parallelism (TP)
questions in three framings - define TP (00203-00205), contrast TP against a naive
non-TP implementation (00206-00210), and give two failure modes / trade-offs
(00211-00212). Every source answer is the identical single sentence ("shards
computation within layers and introduces collective communication; the best degree
depends on memory, topology, batch, and communication cost"), which is true but is
reused verbatim across all three question shapes, so it never actually contrasts
anything and never enumerates failure modes. All ten were rewritten with distinct
content: Megatron column-then-row sharding and the two all-reduces per block;
head-sharding of attention and the min(TP, num_key_value_heads) bound on KV savings;
the TPOT(N) = C/N + L(N) latency model and its measurable crossover on PCIe-attached
A30s versus NVLink; ring all-reduce cost 2(N-1)/N * S bytes in 2(N-1) steps and why
prefill is bandwidth-bound while decode is latency-bound; the non-shardable memory
remainder (full-shape residual stream, replicated norms) that makes real savings less
than 1/N; correlated failure and NCCL watchdog lockstep as an operational cost the
naive single-device path does not pay; and checkpoint/TP-degree coupling with a
generation-parity probe as the rollback gate.

These outputs are PROVISIONAL teacher-B model judgements. They are not expert gold
labels, they have not been validated against measurements on real hardware, and they
say nothing about any model's domain capability. They exist only to be compared later,
in a separate step, against the independently produced teacher-A lane.

## Run 2026-08-17 batch 0018

- Batch file: results/train-batch-0018.jsonl
- Corpus range: train.jsonl lines 171-180, source IDs corpus-00191, corpus-00192, corpus-00193, corpus-00194, corpus-00195, corpus-00197, corpus-00199, corpus-00200, corpus-00201, corpus-00202 (corpus order preserved exactly; the gaps at 00196 and 00198 are gaps in the corpus file itself, not skips by this worker)
- Progress: train 180/5399, validation 0/601, total 180/6000, remaining 5820
- Decisions: keep 0, rewrite 10, reject 0
- Initial schema check: PASS on first run (scripts/verify_batches.py -> VERIFY=PASS, train=180/5399, validation=0/601, total=180/6000)
- Repairs applied: none required this run
- Final schema check: PASS (12 required fields present and field count exactly 12; teacher_lane=teacher-B, teacher_model=claude-opus-5-current, calibration_status=provisional, decision in keep/rewrite/reject; source_user and source_assistant character-exact against the corpus; corrected_answer non-empty; confidence in [0,1]; quality_dimensions integers 1-5; risks and evidence_required string arrays; source_id globally unique; aggregated train sequence is a strict prefix of corpus order)
- Manifest: MANIFEST.sha256 regenerated over all 40 files in this directory (excluding itself); `sha256sum -c` reports 40 OK, 0 failures
- Generator script archived at scripts/gen_train_batch_0018.py
- Lock: /tmp/teacher-b-corpus-review.lock acquired atomically at run start, released at run end
- Blind protocol: no file under experiments/2026-08-14-teacher-a-corpus-calibration/ was read, opened or grepped during this run

Technical topics covered by this batch: three question shapes over two concepts.
(1) corpus-00191..00195 ask for a *small controlled experiment* on continuous batching.
Every source answer is the same single definitional sentence, which is technically true
but answers a different question than the one asked, so all five were rewritten with
distinct experimental designs: single-factor A/B at swept offered load; paired trace
replay with a signed-rank test to remove arrival-process variance; a 2x3 factorial
crossing scheduler with output-length dispersion (CV 0.1/0.8/2.0) so the straggler-
displacement mechanism becomes falsifiable via the interaction term; a saturation sweep
comparing goodput at a fixed p99 SLO instead of peak throughput; and an ablation
separating retire-early from chunked-prefill interference. Shared across all five: the
boundary condition that the benefit vanishes at near-uniform output lengths or once KV
cache is the binding constraint, an explicit rollback gate (revert on >20% p99 TTFT
regression, >1% preemption rate, or any OOM/KV-eviction failure), and a stated
falsification threshold (<5% throughput gain at matched p99 rejects the hypothesis).
(2) corpus-00197, 00199, 00200 ask for a *runbook entry*; the source answers are again
the definitional sentence and give an on-call engineer no ordered procedure. Rewrites
supply numbered diagnostics: confirm the scheduler config actually in force rather than
assuming defaults; sample /metrics at 1 s resolution; classify the regime as
scheduler-limited (waiting>0, KV<80%) versus memory-limited (KV>95% with rising
preemptions); decompose admission versus prefill versus decode latency, since blaming
"batching" without that split is the standard misdiagnosis; correlate TPOT spikes with
prefill admissions before lowering batch size; and check GPU exclusivity and clock
stability. (3) corpus-00201..00202 ask for a definition of tensor parallelism. The
source answers are directionally correct but abstract, so the rewrites add the
Megatron-style mechanism (column-parallel first MLP GEMM, row-parallel second, roughly
two all-reduces per transformer layer moving batch x seq x hidden x dtype_bytes), the
interconnect boundary condition (NVLink/NVSwitch amortizes the collective; PCIe or
cross-node RoCE can make TP=8 slower than TP=2 because the per-collective latency floor
lands on the decode critical path), the head-count divisibility constraint, an A30-24GB
sizing sketch marked explicitly as an estimate from parameter counts rather than a
measurement, and the required evidence (nvidia-smi topo -m, nccl-tests all_reduce_perf
at the real message sizes, per-token latency at TP in {1,2,4,8}).

These outputs are **provisional** teacher-B review artifacts. They are not expert gold
labels, they have not been validated by a human domain expert or by execution against
real hardware, and they say nothing about any model's domain capability. Every numeric
claim in the rewrites is labelled as an estimate unless it is accompanied by a stated
measurement procedure. Agreement analysis against teacher-A is a separate, later step
and was deliberately not performed here.

## Run 2026-08-17 batch 0017

- Batch file: results/train-batch-0017.jsonl
- Corpus range: train.jsonl lines 161-170, source IDs corpus-00181 through corpus-00190 (contiguous)
- Progress: train 170/5399, validation 0/601, total 170/6000, remaining 5830
- Decisions: keep 0, rewrite 10, reject 0
- Initial schema check: PASS on first run (scripts/verify_batches.py -> VERIFY=PASS, train=170/5399, validation=0/601, total=170/6000)
- Repairs applied: none required this run
- Final schema check: PASS (all 12 required fields present and field count exactly 12; teacher_lane/teacher_model/calibration_status/decision values valid; source_user and source_assistant character-exact against the corpus; corrected_answer non-empty; confidence in [0,1]; quality_dimensions integers 1-5; risks and evidence_required string arrays; source_id globally unique; aggregated train sequence is a strict prefix of corpus order)
- Manifest: MANIFEST.sha256 regenerated over all 37 files in this directory (excluding itself); `sha256sum -c` reports 0 failures
- Generator script archived at scripts/gen_batch_0017.py
- Lock: /tmp/teacher-b-corpus-review.lock acquired atomically at run start, released at run end
- Blind protocol: no file under experiments/2026-08-14-teacher-a-corpus-calibration/ was read, opened or grepped during this run

Technical topics covered by this batch: all ten records concern **continuous batching**
in two question shapes. (1) corpus-00181..00185 ask how continuous batching differs
between training and inference; answered around the scheduling-unit contrast (mutable
membership at iteration boundaries in inference versus membership frozen by the optimizer
barrier in training), with the boundary condition that continuous batching cannot be
ported to training because the backward pass needs the activation graph of exactly the
forward-pass sequences and mutable membership would make the effective batch size, hence
the gradient estimator and LR schedule, non-deterministic; the legitimate training analogue
is sequence packing with a block-diagonal mask. Variant axes: scheduling unit, memory
lifetime (activations/optimizer state known ahead of time versus KV occupancy drifting at
runtime), NCCL synchronisation and straggler cost versus preemption/recompute, padding
waste versus KV block fragmentation, and single-scalar MFU versus a two-sided
throughput-at-fixed-p99-TTFT/ITL SLO. (2) corpus-00186..00190 ask for a misleading
intuition plus correction; the five corrections cover: continuous batching does not reduce
single-request latency (at concurrency 1 the schedulers must be indistinguishable, which is
the falsifier); batch size did not disappear but moved to max_num_seqs plus the KV pool
implied by gpu_memory_utilization; it removes batch-level but not prefill-level
head-of-line blocking, so unchunked long prefills still spike ITL; it does not transfer to
training; and nvidia-smi utilisation is not proof of useful work, with achieved HBM
bandwidth against the ~933 GB/s A30 ceiling as the load-bearing metric. All answers carry an
explicit A30 assumption frame, label estimates versus measured values, state the evidence
required (separate TTFT/ITL percentile series, scheduler running/waiting/preempted counters,
KV block utilisation, profiler bandwidth, kv_bytes_per_token read from the served checkpoint
config, fixed-seed loss-curve diff for training-side changes), and end with an explicit
rollback gate (revert if p99 ITL regresses more than 10 percent or preemption counts rise).

These outputs are **provisional teacher-B review artifacts only**. They are not expert gold
labels, they have not been validated by a human domain expert or against measured hardware
data, and they say nothing whatsoever about the domain capability of any trained model. The
teacher-A/teacher-B agreement analysis is a separate, later step and was deliberately not
performed here.

## Run 2026-08-17 batch 0016

- Batch file: results/train-batch-0016.jsonl
- Corpus range: train.jsonl lines 151-160 (0-indexed 150-159), source IDs corpus-00169, 00170, 00171, 00173, 00174, 00176, 00177, 00178, 00179, 00180
- Progress: train 160/5399, validation 0/601, total 160/6000, remaining 5840
- Decisions: keep 0, rewrite 10, reject 0
- Initial schema check: PASS on first run (scripts/verify_batches.py -> VERIFY_PASS, train=160/5399, validation=0/601, total=160/6000)
- Repairs applied: none required this run
- Final schema check: PASS (VERIFY_PASS; all 12 required fields present, teacher_lane/teacher_model/calibration_status/decision values valid, source_user and source_assistant byte-equal to the corpus, corrected_answer non-empty, confidence in [0,1], source_id globally unique, aggregated train sequence is a strict prefix of corpus order)
- Manifest: MANIFEST.sha256 regenerated over all files in this directory (excluding itself); `sha256sum -c` reports 0 failures
- Generator script archived at scripts/gen_batch_0016.py
- Lock: /tmp/teacher-b-corpus-review.lock acquired atomically at run start, released at run end

Technical topics covered by this batch: all ten records again concern **continuous
batching** in LLM serving, in three question shapes. (1) corpus-00169..00170 ask how
continuous batching interacts with latency, throughput and memory, answered via the
iteration-boundary admission/retirement mechanism, the coupled budget in which each
admitted sequence simultaneously consumes KV blocks, adds bytes to every decode step and
contributes tokens/s, and the roofline estimate step_time ~= (weights + sum_i KV_i)/BW_eff
with BW_eff ~0.6-0.75 x 933 GB/s on A30. (2) corpus-00171, 00173, 00174 ask for a
measurement plan, answered with pre-registered hypotheses and SLOs, production trace
replay closed-loop over a QPS ladder, a three-arm design (static+contiguous KV, static+paged
KV, continuous+paged KV) that separates the memory-fragmentation win from the scheduler
win, goodput-at-SLO as the decision metric, locked clocks, warm-up discard and >=3 repeats.
(3) corpus-00176..00180 ask what assumptions must be stated before a performance claim,
answered with an explicit disclosure set (kv_bytes_per_token = 2 * n_layers * n_kv_heads *
head_dim * dtype_bytes from the served checkpoint, weight bytes, max_model_len,
gpu_memory_utilization, TP degree, chunked prefill, PCIe topology, clock-lock state, metric
definition, repeat count) and a claim template whose every slot is falsifiable. Recurring
boundary conditions: the gain is bounded by output-length dispersion (padded-slot waste
~= 1 - mean(L)/max(L)) and vanishes when KV bytes rather than slots are binding; recurring
failure mode: preemption/recompute thrash, where swap costs KV_bytes / ~25 GB/s realised
PCIe Gen4 x16 each way and goodput collapses while GPU utilisation still reads high.
Every source answer in this batch was the same single generic sentence about
iteration-boundary retirement, off-task for the measurement-plan and assumption-list
questions, hence 10 rewrites with instruction_coverage 1.

These results are **provisional** teacher-B second opinions produced by a general-purpose
model under blind review. They are NOT expert gold labels, have not been validated against
measured hardware data, and say nothing about any trained model's domain capability.

## Run 2026-08-17 batch 0015

- Batch file: results/train-batch-0015.jsonl
- Corpus range: train.jsonl lines 141-150 (0-indexed 140-149), source IDs corpus-00159 .. corpus-00168
- Progress: train 150/5399, validation 0/601, total 150/6000, remaining 5850
- Decisions: keep 0, rewrite 10, reject 0
- Initial schema check: PASS on first run (scripts/verify_batches.py -> VERIFY_PASS, train=150/5399, validation=0/601, total=150/6000)
- Repairs applied: none required this run
- Final schema check: PASS (VERIFY_PASS; all 12 required fields present, teacher_lane/teacher_model/calibration_status/decision values valid, source_user and source_assistant byte-equal to the corpus, corrected_answer non-empty, confidence in [0,1], source_id globally unique, aggregated train sequence is a strict prefix of corpus order)
- Manifest: MANIFEST.sha256 regenerated over all 34 files in this directory (excluding itself); `sha256sum -c` reports 34 OK and 0 failures
- Generator script archived at scripts/gen_batch_0015.py
- Lock: /tmp/teacher-b-corpus-review.lock acquired atomically at run start, released at run end

Technical topics covered by this batch: all ten records concern **continuous batching**
in LLM serving, split into three sub-themes. (1) corpus-00159..00160 contrast continuous
batching with naive request-level/static batching, covering iteration-boundary admission
and retirement, useful-token efficiency mean(L)/max(L), and the separate paged-KV memory
effect (contiguous max_seq_len reservation vs. per-page lazy allocation, internal
fragmentation bounded by one page). (2) corpus-00161..00165 enumerate failure modes and
trade-offs: preemption/recompute thrash under KV pressure, head-of-line interference and
the throughput/tail-latency trade, CUDA-graph shape instability, fairness/starvation of
long-context requests, prefill/decode interference and chunked prefill, goodput vs.
throughput accounting divergence, over-admission on current rather than projected KV
footprint, client retry amplification feedback loops, tensor-parallel collective coupling
(2 all-reduces per layer, PCIe-only topology without NVLink), and bf16 reduction-order
nondeterminism breaking eval reproducibility. (3) corpus-00166..00168 analyse the
latency/throughput/memory interaction quantitatively via the roofline relation
step_time = (W + sum_i KV_i)/BW_eff, the throughput knee at sum_i KV_i ~= W, the three
coupled budgets (bandwidth, memory, latency), the TTFT/ITL decomposition that opposing
effects can hide in an end-to-end average, and prefix-caching-induced benchmark inflation.

Every source_assistant in this batch was the same single-sentence stub, which is why all
ten decisions are `rewrite`: the stub states one mechanism but never supplies the boundary
condition the prompt explicitly requests, and for corpus-00161..00165 it does not answer
the asked question (failure modes) at all. instruction_coverage was scored 2-3 accordingly.

**Status caveat (mandatory):** these corrected_answer texts are *provisional* teacher-B
output produced blind by claude-opus-5-current. They are NOT expert gold, they have not
been validated against MEASURED data from this cluster, and every quantitative figure in
them (A30 933 GB/s peak, 0.65-0.75 achieved-bandwidth fraction, ~18 GB bf16 weights,
~25 GB/s PCIe Gen4 x16, 16-token pages) is an explicitly labelled estimate under a stated
assumption frame. This artifact says nothing about any model's domain capability; it is a
review corpus only. Agreement analysis against teacher-A is a separate later step and was
deliberately not performed here — no teacher-A file was read during this run, preserving
the blind-review condition.

## Run 2026-08-17 batch 0014

- Batch file: results/train-batch-0014.jsonl
- Corpus range: train.jsonl lines 131-140 (0-indexed 130-139), source IDs corpus-00146 .. corpus-00158
- Progress: train 140/5399, validation 0/601, total 140/6000, remaining 5860
- Decisions: keep 0, rewrite 10, reject 0
- Initial schema check: PASS on first run (scripts/verify_batches.py -> VERIFY_PASS, train=140/5399)
- Repairs applied: none required this run
- Final schema check: PASS (VERIFY_PASS, all 12 fields, source_user/source_assistant byte-equal to corpus, source_id globally unique, train sequence is a strict prefix of corpus order)
- Manifest: MANIFEST.sha256 regenerated over all 32 files in this directory; `sha256sum -c` reports 0 failures
- Lock: /tmp/teacher-b-corpus-review.lock acquired atomically at run start, released at run end

Technical topics covered by this batch: decode-phase runbook triage (3 records) and
continuous batching (7 records: 5 definition-style, 2 contrast-with-naive-static-batching).
The decode records were rewritten to supply the per-step byte model
(step_time ~= (weight_bytes + sum of resident KV bytes) / achieved_HBM_bandwidth), the
GEMV-vs-GEMM arithmetic-intensity argument that makes decode memory-bandwidth-bound rather
than FLOP-bound, the bandwidth-to-compute knee as an explicit boundary condition, and the
no-NVLink A30 caveat where PCIe-resident tensor-parallel all-reduce becomes the limiter
instead of HBM. The batching records were rewritten to add the admission half of the
iteration-level scheduling loop, paged KV block allocation as the enabling mechanism, the
mean_len/max_len slot-utilisation argument for why the gain tracks output-length skew, and
the failure boundary where KV-pool saturation causes evict-and-recompute preemption thrash
that can drive throughput below a conservative static batch. Every record carries an explicit
assumption frame (8x A30 24 GB, ~9B bf16 dense, paged KV), falsifiable hypotheses, an
evidence list, and a rollback gate keyed on p95 TTFT/ITL regression, preemption rate, and
greedy-decode output equivalence.

Status caveat: these outputs are PROVISIONAL teacher-B model review under blind conditions.
They are NOT expert gold labels, they have not been validated against measurements on this
hardware, and they say nothing about any model's domain capability. No teacher-A artifact was
read, opened, or grepped while producing this batch.

## Run 2026-08-17 batch 0013

- Batch file: results/train-batch-0013.jsonl
- Corpus range: train.jsonl lines 121-130 (0-indexed 120-129), source IDs corpus-00136 .. corpus-00145
- Progress: train 130/5399, validation 0/601, total 130/6000, remaining 5870
- Decisions: keep=0, rewrite=10, reject=0
- Initial schema check: FAIL at generation time - the batch script was first written with
  START=130 instead of START=120, so it read corpus-00146.. and tripped the in-script
  source_id assertion before writing anything. No bad output file was ever produced.
- Repair action: corrected START to 120 in scripts/gen_batch_0013.py and regenerated. The
  raw corpus and all previously committed batches were left untouched.
- Final schema check: PASS (scripts/verify_batches.py -> VERIFY_PASS; JSONL parses line by
  line, 10 records, all 12 required fields present, teacher_lane/teacher_model/
  calibration_status/decision values valid, source_user and source_assistant byte-identical
  to the raw corpus, corrected_answer non-empty, confidence in [0,1], source_id globally
  unique, and the aggregated train sequence is an exact prefix of train.jsonl).
- Manifest: MANIFEST.sha256 regenerated over every file in this directory except itself;
  sha256sum -c reports OK for all entries.

### Technical topics covered by this batch

All ten records are Knowledge/Concept items on the decode phase of LLM inference, split into
two task families. Records corpus-00136..00140 ask for a misleading intuition plus its
correction; the rewrites target five distinct misconceptions: (1) decode is compute-bound
and responds to more FLOPS, corrected via GEMV arithmetic intensity and the HBM weight-
streaming floor with the roofline ridge point as the boundary; (2) decode latency scales
linearly with context, corrected by separating the context-independent weight stream from
the context-dependent KV re-read and giving the crossover in aggregate cached tokens;
(3) tensor parallelism scales decode near-linearly, corrected with the per-layer NCCL
all-reduce latency term and the PCIe-vs-NVLink boundary; (4) weight quantisation gives a
proportional speedup, corrected with an Amdahl decomposition over weight/KV/fixed-overhead
bytes plus the Ampere-has-no-native-FP8 caveat; (5) higher aggregate tok/s is always better,
corrected by the throughput-versus-inter-token-latency tradeoff under continuous batching and
the preemption cliff.

Records corpus-00141..00145 ask for a small controlled experiment; the rewrites specify an
independent variable, held-fixed controls, warm-up and repetition policy, the falsifiable
prediction, the invalidating boundary condition, the evidence to capture, and a rollback or
stop gate. They cover, respectively: a batch-size sweep testing the memory-bound hypothesis;
a context-length sweep with prefix caching explicitly disabled to avoid silent prefill skip;
a tensor-parallel degree A/B with an independent nccl-tests latency calibration arm and a
topology control; an open-loop load sweep that locates the throughput/latency knee (with an
explicit warning that a closed-loop client hides the knee); and a speculative decoding
evaluation gated on measured acceptance rate and byte-identical output versus baseline.

The source_assistant text is identical boilerplate across all ten records ("Decode generates
one or a few tokens per step ..."), which is true but answers none of the ten distinct
prompts, so every record was marked rewrite with instruction_coverage 1-2.

### Status caveat

These are PROVISIONAL teacher-B outputs from a blind second-opinion lane. They are not expert
gold labels, they have not been checked against teacher-A (that comparison is a separate later
step), and they are not evidence of any model's domain capability. All numeric values in the
corrected answers are roofline estimates or worked examples under a stated assumption frame,
not measurements from this host.

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

Progress: train 120/5399; validation 0/601; total 120/6000; remaining 5880.

Runs are appended below, newest first.

## Run log (newest first)

### 2026-08-17 — train-batch-0012.jsonl

- Batch file: results/train-batch-0012.jsonl
- Generator: scripts/gen_batch_0012.py
- Corpus range: train.jsonl lines 111–120
- Source IDs: corpus-00126, corpus-00127, corpus-00128, corpus-00129, corpus-00130,
  corpus-00131, corpus-00132, corpus-00133, corpus-00134, corpus-00135
- Progress: train 120/5399, validation 0/601, total 120/6000, remaining 5880
- Decisions: keep=0, rewrite=10, reject=0
- Initial schema check: PASS (verify_batches.py → VERIFY_PASS, train=120 validation=0 total=120)
- Repairs: none required
- Final schema check: PASS
- Manifest: MANIFEST.sha256 regenerated over all 28 non-manifest files; `sha256sum -c`
  reported 28 OK / 0 FAILED.

Technical topics covered by this batch. All ten records are `Knowledge/Concept`,
difficulty `medium`, concept `decode`, and all ten share the same one-sentence
source_assistant ("Decode generates one or a few tokens per step ..."), which is
directionally true but answers neither prompt's actual instruction and supplies no
mechanism, no numbers and no boundary condition. Hence rewrite on all ten, with
instruction_coverage scored 1.

Records corpus-00126..00130 ask which assumptions must be stated before making a
decode performance claim. The rewrites split this across five distinct angles so the
variants are not near-duplicates: (1) throughput claims — workload shape, batching
regime, output-only vs total tok/s, precision, sampling and speculative-decoding
acceptance rate; (2) latency claims — TTFT vs ITL separation, offered load,
prefill interference, tenancy/isolation, warmup window, plus the PCIe-only TP=2
caveat; (3) memory and capacity claims — KV bytes/token arithmetic, fragmentation
and preemption policy, prefix-cache hit rate as a benchmark artifact, quantization,
steady state vs 60-second runs; (4) multi-GPU and disaggregated serving — TP/PP/EP
layout, topology, GPUDirect RDMA actually active vs merely compiled in, KV transfer
cost in Mooncake / NVIDIA Dynamo style prefill-decode split, KV-aware routing;
(5) comparison hygiene — equal tuning budget on both sides, identical artifacts and
clocks, multi-run statistics, and a mandatory correctness gate.

Records corpus-00131..00135 ask how decode differs between training and inference.
The rewrites cover: compute shape (GEMM→GEMV, ~1000x drop in weight reuse, why PP
is unsuitable for decode); memory and state (optimizer state ~16 B/param vs weights
plus a long-lived per-request KV cache, and KV as a transferable first-class object);
collectives (large bandwidth-bound gradient all-reduce vs ~80 tiny latency-bound
activation all-reduces per token, NVLink vs PCIe, why TP must not cross node
boundaries and why RDMA/RoCE with GDR is the multi-node KV path); scheduling
(static batching vs continuous/iteration-level batching, chunked prefill and
head-of-line blocking); and numerics (decode as a sequential amplifier of tiny
numerical differences, batch-dependent reduction order breaking bitwise
reproducibility, and why perplexity is an inadequate acceptance test for
decode-side quantization).

Every rewrite carries an explicit assumptions frame (A30 24 GB, ~933 GB/s peak HBM,
no NVLink on this host, PCIe Gen4, bf16 ~9B ≈ 18 GB weights, ~1.6 MB/token KV), at
least one falsifiable prediction, an evidence list and a rollback gate.

Status caveat: these outputs are PROVISIONAL teacher-B review labels produced blind
from source_user/source_assistant only. They are not expert gold, have not been
validated against teacher-A (that comparison is a separate later step), and say
nothing about any model's domain capability. The quantitative figures in the
rewrites are stated as assumption-framed estimates to be confirmed by the listed
evidence, not as measured results.

### 2026-08-17 — train-batch-0011.jsonl

- Batch file: results/train-batch-0011.jsonl
- Corpus range: train.jsonl lines 101–110
- Source IDs: corpus-00114, corpus-00115, corpus-00116, corpus-00117, corpus-00118,
  corpus-00119, corpus-00121, corpus-00122, corpus-00123, corpus-00124
- Progress: train 110/5399, validation 0/601, total 110/6000, remaining 5890
- Decisions: keep=0, rewrite=10, reject=0
- Initial schema check: PASS (verify_batches.py → VERIFY_PASS, train=110 validation=0 total=110)
- Repairs: none required
- Final schema check: PASS
- Manifest: MANIFEST.sha256 regenerated over all 26 non-manifest files; `sha256sum -c` → 26 OK, 0 failures
- Generator script preserved: scripts/gen_batch_0011.py

Technical topics covered by this batch: the decode phase of LLM inference. All ten
records share the same boilerplate source_assistant, so all ten were rewritten. The
replacement answers cover: (1) HBM-bandwidth starvation at low batch size with the
weight-streaming cost model and the GEMV degeneration mechanism; (2) KV-cache capacity
collapse, paged KV, GQA/MQA byte accounting, and preemption/swap thrash driven by the
length-distribution tail; (3) tensor-parallel decode becoming collective-latency bound
(~2 NCCL all-reduces per layer per token in the small-message latency regime) and the
U-shaped TPOT-vs-TP curve; (4) speculative decoding's dependence on the model being
bandwidth bound and its regression under load; (5) weight and KV quantization as
bandwidth optimizations, including the fused-dequant boundary condition; (6) chunked
prefill vs prefill/decode disaggregation (Dynamo- and Mooncake-style), with the KV
transfer byte/time budget over RDMA/RoCE and GPUDirect RDMA, plus the cross-root-complex
and PFC/ECN boundary conditions; and (7) four distinct measurement plans covering
open- vs closed-loop load models, A/A noise-floor quantification, clock/thermal
confounder control, output-token parity, mechanism-counter attribution, canary/overload
testing, and pre-registered rollback gates.

Status caveat: these corrected_answer values are PROVISIONAL teacher-B review output
produced blind (teacher-A artifacts were not read while producing this batch). They are
NOT expert gold labels, have not been verified against hardware measurements, and are
NOT evidence of any model's domain capability.

### 2026-08-17 — train-batch-0010.jsonl

- Batch file: results/train-batch-0010.jsonl
- Corpus range: train.jsonl lines 91–100 (strict corpus order, no skips, no reordering)
- Source IDs: corpus-00103, corpus-00105, corpus-00106, corpus-00107, corpus-00108,
  corpus-00109, corpus-00110, corpus-00111, corpus-00112, corpus-00113
- Progress after this run: train 100/5399; validation 0/601; total 100/6000; remaining 5900.
- Decisions: keep 0, rewrite 10, reject 0.
- Initial schema check: PASS on first run (JSONL line-parseable, 10 records, all 12
  required fields present, teacher_lane/teacher_model/calibration_status/decision values
  correct, source_user and source_assistant byte-identical to raw corpus, corrected_answer
  non-empty, confidence within [0,1], source_id globally unique across all batches,
  aggregated train sequence is an exact prefix of train.jsonl).
- Repairs performed: none required.
- Final schema check: PASS (train 100, validation 0, total 100, VERIFY_PASS).
- Manifest: MANIFEST.sha256 regenerated over all 24 files in this directory except the
  manifest itself; `sha256sum -c` reported 24 OK, 0 FAILED.

#### Technical topics covered by this batch

All ten records target the decode phase of LLM inference. The rewritten answers cover:
autoregressive decode as a memory-bandwidth-bound regime and the HBM-floor latency bound
t_step >= (W_bytes + KV_bytes)/BW_HBM instantiated on A30 (933 GB/s); KV cache sizing from
n_layers/n_kv_heads/head_dim/dtype under GQA and the resulting concurrency ceiling on a
24 GB card; KV cache versus naive O(N^2) attention recompute; static versus continuous
(in-flight) batching and the mean_len/max_len slot-utilization argument; prefill/decode
interference, chunked prefill token budgets, and prefill/decode disaggregation of the kind
used by Mooncake- and NVIDIA Dynamo-style architectures including the KV-transfer break-even
over NVLink or RDMA/RoCE with GPUDirect; speculative decoding acceptance-rate math and why
its gain decays as batch size grows; paged attention block allocation, fragmentation bounds,
and prefix sharing with copy-on-write; KV/weight quantization error compounding across
generation steps; tensor-parallel all-reduce latency dominating decode at small batch; and
straggler-induced tail amplification across TP ranks in multi-GPU/multi-node decode.

Every source answer in this range was the same templated one-sentence definition of decode
reused verbatim across ten different instructions (define / contrast-with-naive /
list-two-failure-modes), so instruction coverage was scored 1–2 throughout and every record
was marked `rewrite`. Each rewritten answer states an explicit mechanism, a boundary
condition delimiting the regime in which the mechanism holds, at least one falsifiable
claim, the measurements needed to test it, and a rollback gate.

#### Status caveats

These records are PROVISIONAL teacher-B review output produced blind by
claude-opus-5-current. They are NOT expert gold labels, have NOT been validated against
hardware measurements, and do NOT constitute evidence of any model's domain capability.
No file under experiments/2026-08-14-teacher-a-corpus-calibration/ was read while producing
this batch; inter-teacher agreement remains a separate later analysis step.

### 2026-08-17 — train-batch-0009.jsonl

- Batch file: results/train-batch-0009.jsonl
- Corpus range: research/ai-infra-expert/corpus/train.jsonl lines 81–90 (0-indexed 80–89)
- Source IDs: corpus-00091, corpus-00092, corpus-00093, corpus-00094, corpus-00095,
  corpus-00096, corpus-00097, corpus-00098, corpus-00100, corpus-00101
- Progress after this run: train 90/5399; validation 0/601; total 90/6000; remaining 5910
- Decisions: keep=0, rewrite=10, reject=0
- Initial schema/verification check: PASS on first run (scripts/verify_batches.py →
  train=90/5399 validation=0/601 total=90/6000 VERIFY_PASS)
- Fix actions: none required this run
- Final schema/verification check: PASS
- Manifest: MANIFEST.sha256 regenerated over all files in this directory except
  MANIFEST.sha256 itself; `sha256sum -c` verified OK for every entry
- Topics covered: prefill as a parallel prompt-processing phase and its
  compute/GEMM-bound behavior (controlled experiment design, P-sweep, arithmetic
  intensity, O(P^2) attention term, chunked prefill and prefix caching as
  confounders); prefill/TTFT regression triage runbook (queueing vs kernels vs
  clock throttling vs KV preemption); and decode as the memory-bandwidth-bound
  autoregressive phase (weight re-read per step, batch-size scaling, KV-read
  crossover at long context, A30 24 GB capacity limits).
- Why rewrite for all 10: every source_assistant is a single generic sentence with
  no units, no boundary condition, no falsifiable prediction, no evidence list and
  no rollback gate, while each source_user explicitly asks for one concrete
  mechanism and one boundary condition. Instruction coverage was scored 1.
- Caveat: these results are PROVISIONAL teacher-B review output. They are NOT
  expert gold labels, have NOT been validated by a human domain expert, and say
  nothing about any model's domain capability. Hardware figures quoted inside the
  corrected answers (A30 BF16 peak, HBM2 bandwidth, model byte sizes) are vendor
  spec or arithmetic estimates, not measurements on this cluster.

### 2026-08-17 — train-batch-0008.jsonl

- Batch file: results/train-batch-0008.jsonl
- Builder: scripts/gen_batch_0008.py
- Corpus range: research/ai-infra-expert/corpus/train.jsonl lines 71-80 (0-indexed 70-79)
- Source IDs: corpus-00081, corpus-00082, corpus-00083, corpus-00084, corpus-00085,
  corpus-00086, corpus-00087, corpus-00088, corpus-00089, corpus-00090
- Progress after this batch: train 80/5399, validation 0/601, total 80/6000,
  remaining 5920
- Decisions: keep=0, rewrite=10, reject=0
- Initial schema check: scripts/verify_batches.py → VERIFY_PASS on first run
- Repairs applied: none required
- Final schema check: VERIFY_PASS (train=80/5399 validation=0/601 total=80/6000)
- Manifest: MANIFEST.sha256 regenerated over all 21 files in this directory
  except itself; `sha256sum -c` reported 21 OK, 0 failures
- Technical topics covered: this block is two prompt families of five case
  variants each, both anchored on the prefill phase. Family 1 ("how prefill
  changes between training and inference") required contrasting the two regimes;
  the rewrite separates what is structurally identical (one parallel causal pass,
  dense GEMMs of shape (B*N x d) x (d x d'), compute-bound at large B*N) from
  what actually differs — activation retention for backward at O(L*B*N*d) versus
  immediate discard, the KV cache emitted by inference prefill
  (2*L*N*n_kv_heads*head_dim*dtype_bytes) versus training throwing it away,
  the ~3x forward cost of training with no decode phase at all, and static
  fixed-shape training schedules versus online heterogeneous prefill needing
  chunked prefill, continuous batching, or full P/D disaggregation
  (Mooncake / NVIDIA Dynamo). Each variant carries a distinct boundary
  condition: small-N underutilization crossover on A30 (~933 GB/s, ~165 TFLOPS
  bf16, crossover N ~100-200), the O(N^2) attention term overtaking the dense
  term past N ~8k-16k, activation checkpointing shifting the memory/compute
  tradeoff (~25-35% step-time cost), chunked-prefill-versus-decode SM contention
  producing a TTFT/p99-ITL frontier, and tensor-parallel all-reduce traffic of
  ~2*B*N*d bytes per layer making prefill communication-bound across RoCE/IB.
  Family 2 ("give one misleading intuition about prefill and correct it")
  required naming a false intuition, which the source answer never does; the
  five rewrites each pick a different one — "parallel means length is free",
  "one tokens/s number describes the server" (prefill compute-bound vs decode
  HBM-bandwidth-bound roofline split), "compute-bound means no memory problem"
  (KV cache is created during prefill and is the dominant OOM source on 24GB
  A30s), "TP scales prefill linearly" (per-layer collectives grow with N; GDR
  versus host-staged paths change effective busbw), and "disaggregation always
  wins" (full KV transfer sits on the TTFT critical path, so it loses for short
  prompts or weak fabrics). Every rewrite states assumptions, gives a
  falsifiable prediction, and names the measurement that would refute it;
  variants 4 and 5 of family 2 also state explicit rollback gates (revert TP
  width if the measured collective gap exceeds ~20% of step time; revert to
  colocated serving if p99 TTFT regresses more than 10%).
- Quality scoring rationale: all 10 source answers are the same single generic
  sentence, which restates a textbook definition without mechanism, units, or
  boundary conditions, and in both families fails to perform the instruction
  actually issued. Scored technical_correctness=2 (directionally true but too
  coarse to act on and silently wrong in the small-N and long-context regimes),
  instruction_coverage=1, operational_safety=3 (no unsafe advice, but no
  evidence requirements or rollback gates either). Decision was rewrite for all
  10 rather than reject, since the source sentence contains a usable kernel.
- IMPORTANT: these results are PROVISIONAL teacher-B review output. They are NOT
  expert gold labels, have NOT been validated by a human domain expert, and do
  NOT constitute evidence of any model's domain capability. They exist only to
  support a later independent inter-teacher agreement analysis. This batch was
  produced BLIND: no file under
  experiments/2026-08-14-teacher-a-corpus-calibration/ was read, opened, or
  grepped at any point while producing it.

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
