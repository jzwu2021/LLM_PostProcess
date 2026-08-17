# Experiment: teacher-B corpus review (blind, independent second opinion)

Started: 2026-08-17
Lane: teacher-B
Reviewer model: claude-opus-5 (provider: copilot), pinned explicitly so this lane
is NOT the same model that produced teacher-A (gpt-5.6-luna-current).

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
