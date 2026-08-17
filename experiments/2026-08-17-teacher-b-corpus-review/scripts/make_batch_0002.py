#!/usr/bin/env python3
"""Build teacher-B provisional review batch 0002 (train lines 11-20)."""
import json, os

ROOT = "/home/johnson/workspace/LLM_PostProcess"
CORP = os.path.join(ROOT, "research/ai-infra-expert/corpus/train.jsonl")
OUT = os.path.join(ROOT, "experiments/2026-08-17-teacher-b-corpus-review/results/train-batch-0002.jsonl")
LO, HI = 11, 20

KV_MEM = (
    "Sizing model (state the assumption explicitly): "
    "kv_bytes = 2 (K and V) * layers * seq_len * kv_heads * head_dim * bytes_per_elem * batch. "
    "Example, assumption only, not measured: 32 layers, 8 KV heads (GQA), head_dim 128, fp16 (2 B), "
    "one sequence of 8192 tokens -> 2*32*8192*8*128*2 B = 1.07 GB per sequence. "
    "Verify against the actual config (num_hidden_layers, num_key_value_heads, head_dim, kv dtype) "
    "before trusting any number."
)

FAILMODE = {
    "corpus-00013": (
        "Two failure modes / trade-offs of the KV cache.\n\n"
        "Mechanism 1 - memory-bound capacity collapse. " + KV_MEM + " Because the cache is per-sequence and "
        "grows linearly in sequence length, HBM, not FLOPs, becomes the binding constraint at high concurrency. "
        "In a paged allocator (vLLM PagedAttention style) the concrete symptom is that free KV blocks hit zero, "
        "the scheduler preempts or swaps running sequences, and goodput drops non-linearly while GPU SM utilisation "
        "stays low.\n\n"
        "Mechanism 2 - decode is bandwidth-bound, not compute-bound. Each decode step must re-read the entire "
        "cache for the attended context: bytes_read_per_step ~= kv_bytes(seq_len). Arithmetic intensity is ~1 MAC "
        "per loaded element, so time per token approaches kv_bytes / achievable_HBM_bandwidth. Caching removes "
        "recompute but converts the cost into memory traffic; past a certain context length, longer contexts slow "
        "decode even though the cache 'helped'.\n\n"
        "Boundary condition: the cache is a win only while (a) the sequence is long enough that recompute would "
        "dominate (prefill-like O(n^2) work avoided) and (b) kv_bytes still fits in HBM alongside weights and "
        "activation workspace. At batch=1 with very short sequences the cache is nearly free but also nearly "
        "pointless; at high batch with long context it is the first resource to exhaust.\n\n"
        "Falsifiable hypothesis: if decode is KV-bandwidth-bound, halving kv dtype (fp16 -> fp8/int8 KV) should "
        "reduce time-per-output-token by close to the same ratio at fixed batch and context. If TPOT barely moves, "
        "the bottleneck is elsewhere (kernel launch overhead, MoE routing, host-side scheduling) and the hypothesis "
        "is refuted.\n\n"
        "Evidence needed: model config (layers, kv_heads, head_dim, dtype), measured HBM free vs. allocated, "
        "KV block utilisation and preemption counters from the serving engine, TPOT vs. context-length sweep, "
        "and nvidia-smi / DCGM DRAM-active counters to confirm bandwidth saturation.\n\n"
        "Rollback gate: if a KV-quantisation or eviction change does not improve p95 TPOT by a pre-registered "
        "margin, or moves task-level quality beyond the accepted tolerance on a held-out set, revert to fp16 KV."
    ),
    "corpus-00014": (
        "Two failure modes / trade-offs of the KV cache (different pair from the memory/bandwidth framing).\n\n"
        "Mechanism 1 - fragmentation and over-reservation. If the cache is allocated as one contiguous "
        "max_seq_len buffer per sequence, every short request still reserves worst-case memory. With max_seq_len "
        "32768 but a median request of 900 tokens, ~97 percent of reserved KV is idle, so achievable concurrency "
        "is set by the reservation, not by real usage. Paged/block allocation (fixed block of e.g. 16 tokens) "
        "fixes the external fragmentation but introduces internal fragmentation of up to block_size-1 tokens per "
        "sequence and a block table indirection on every attention kernel.\n\n"
        "Mechanism 2 - correctness/consistency hazards on cache reuse. Prefix caching and cross-request reuse are "
        "only sound when the reused prefix is byte-identical in tokens AND produced under the same model weights, "
        "same kv dtype, same RoPE/position offsets, and same attention masking. Reusing a prefix across a LoRA "
        "swap, a quantisation change, or a different position offset silently yields wrong logits rather than an "
        "error - a failure mode with no crash signal.\n\n"
        "Boundary condition: prefix caching pays off only when the shared-prefix ratio is high (system prompts, "
        "few-shot blocks, agent scratchpads). With mostly unique inputs, the hash/lookup and block-refcount "
        "bookkeeping is pure overhead.\n\n"
        "Falsifiable hypothesis: enabling prefix caching on a workload with >=60 percent shared prefix tokens "
        "should cut mean TTFT roughly in proportion to the shared fraction. If TTFT improves by <5 percent, either "
        "the prefix is not actually being hit (check cache-hit counters) or prefill was not the bottleneck.\n\n"
        "Evidence needed: token-level prefix-overlap histogram of production traffic, engine cache-hit rate, "
        "block-allocator utilisation and internal-fragmentation stats, and a logit-equivalence test comparing "
        "cached vs. uncached generation for identical inputs.\n\n"
        "Rollback gate: any measurable divergence in the cached-vs-uncached logit equivalence test, or a cache-hit "
        "rate below the level that justifies the bookkeeping cost, triggers disabling reuse immediately."
    ),
    "corpus-00015": (
        "Two failure modes / trade-offs of the KV cache, framed around eviction and multi-tenancy.\n\n"
        "Mechanism 1 - preemption thrash under admission pressure. When KV blocks are exhausted, the scheduler "
        "must either swap a sequence's blocks to host memory over PCIe or recompute its prefill on resume. Swap "
        "cost ~= kv_bytes(seq) / PCIe_effective_bandwidth (order tens of GB/s on Gen4 x16, far below HBM); "
        "recompute cost scales with prefill FLOPs. If arrival rate keeps the pool saturated, sequences are evicted "
        "and restored repeatedly and the system spends its time on eviction bookkeeping instead of decode - "
        "throughput collapses while queue depth grows monotonically.\n\n"
        "Mechanism 2 - tail-latency unfairness across tenants. A single long-context request can pin a large share "
        "of the KV pool for its whole lifetime. Without per-tenant KV quotas, short interactive requests are "
        "starved by one batch job, so p99 TTFT degrades even though mean utilisation looks healthy.\n\n"
        "Boundary condition: eviction-by-recompute beats eviction-by-swap only while prefill FLOP time is less "
        "than the PCIe transfer time for the same sequence - i.e. for short sequences on a compute-rich, "
        "PCIe-poor node. The crossover must be measured per hardware generation, not assumed.\n\n"
        "Falsifiable hypothesis: capping max KV blocks per tenant at N should reduce p99 TTFT for short requests "
        "without reducing aggregate token throughput by more than a small pre-registered percentage. If aggregate "
        "throughput drops sharply, the pool was not actually contended and the quota is only adding rejection.\n\n"
        "Evidence needed: preemption and swap counters over time, queue-depth and admission-rejection series, "
        "per-tenant KV-block occupancy, measured PCIe host-transfer bandwidth, and a p50/p95/p99 TTFT and TPOT "
        "breakdown segmented by request class.\n\n"
        "Rollback gate: revert the quota if p99 TTFT for the protected class does not improve, or if aggregate "
        "goodput regresses past the pre-agreed threshold, in two consecutive measurement windows."
    ),
}

INTERACT = {
    "corpus-00017": (
        "How the KV cache interacts with latency, throughput and memory.\n\n"
        "Latency. Split the request into prefill and decode, because the cache affects them oppositely. Prefill "
        "computes attention over all input tokens at once (work grows roughly with n^2 in sequence length for the "
        "attention term) and it is what fills the cache; TTFT is therefore dominated by prefill compute and is "
        "*not* reduced by the cache on a cold request. Decode reuses the cache: without it, every new token would "
        "recompute all previous keys and values, making per-token cost grow with position. With it, per-token "
        "attention work is O(n) reads instead of O(n^2) recompute, so TPOT stays roughly flat until bandwidth "
        "limits bite.\n\n"
        "Throughput. Because decode re-reads the whole cache each step, decode is memory-bandwidth-bound: "
        "TPOT_floor ~= kv_bytes_read_per_step / achievable_HBM_bandwidth. Continuous batching raises throughput by "
        "amortising the weight read across many sequences, but the KV read is per-sequence and does not amortise. "
        "So throughput rises with batch size only until the KV pool or HBM bandwidth saturates, then it plateaus "
        "or falls as preemption starts.\n\n"
        "Memory. " + KV_MEM + " GQA/MQA reduces kv_heads and is the single largest structural lever; KV "
        "quantisation to fp8/int8 halves or quarters bytes_per_elem at some accuracy risk.\n\n"
        "Boundary condition: the cache turns a compute problem into a memory problem. Below the context length "
        "where kv_bytes is small relative to HBM, caching is nearly pure win; above it, every extra token of "
        "context costs both capacity and per-step bandwidth, and the effective max batch size falls "
        "hyperbolically with context length.\n\n"
        "Falsifiable hypothesis: at fixed context length, throughput vs. batch size should be near-linear until "
        "the KV pool saturates and then flatten. If the curve flattens well before the KV pool is full, the limit "
        "is compute or scheduling, not KV.\n\n"
        "Evidence needed: TTFT and TPOT percentiles separated by prefill/decode, batch-size sweep at fixed input "
        "and output lengths, KV-pool occupancy, DCGM HBM-bandwidth utilisation, and the model's actual KV config.\n\n"
        "Rollback gate: if a KV-related change (batch size, block size, dtype) does not move p95 TPOT or tokens/s "
        "beyond measurement noise across three runs, revert it rather than keeping unexplained configuration drift."
    ),
    "corpus-00018": (
        "KV cache interaction with latency, throughput and memory, expressed as a capacity/roofline argument.\n\n"
        "Mechanism. Per decode step the GPU must read the model weights once per batch and the KV cache once per "
        "sequence. Time per step is approximately "
        "max(compute_time, (weight_bytes + batch * kv_bytes_per_seq) / HBM_bandwidth). At small batch the weight "
        "term dominates and adding sequences is almost free (throughput scales close to linearly). Once "
        "batch * kv_bytes_per_seq exceeds weight_bytes, the KV term dominates, per-step time grows with batch, and "
        "tokens/s stops improving - this is the knee of the throughput curve.\n\n"
        "Memory capacity sets a hard ceiling before that: max_batch ~= (HBM_total - weights - activation_workspace "
        "- fragmentation_headroom) / kv_bytes_per_seq. " + KV_MEM + "\n\n"
        "Latency consequence. Larger batches raise throughput but lengthen each step, so TPOT and p99 latency rise "
        "monotonically with batch. There is no setting that maximises throughput and minimises latency "
        "simultaneously; the operating point must be chosen against an explicit SLO.\n\n"
        "Boundary condition: the analysis above assumes decode is bandwidth-bound and attention kernels achieve a "
        "reasonable fraction of peak HBM bandwidth. It breaks down for MoE models (weight reads become sparse and "
        "routing-dependent), for very small models where kernel-launch and Python overhead dominate, and under "
        "tensor parallelism where the cache is sharded across ranks so per-GPU kv_bytes falls by roughly the TP "
        "degree while collectives add fixed per-step latency.\n\n"
        "Falsifiable hypothesis: measured tokens/s vs. batch should show a knee at the batch where "
        "batch*kv_bytes_per_seq ~= weight_bytes. If the observed knee is far from that predicted point, the "
        "roofline assumption is wrong for this deployment and should be replaced by a profiled model.\n\n"
        "Evidence needed: measured weight footprint and kv_bytes_per_seq, achievable (not spec) HBM bandwidth from "
        "a microbenchmark, tokens/s and TPOT across a batch sweep, and TP degree with collective time per step.\n\n"
        "Rollback gate: if raising batch size pushes p95 TPOT past the SLO, drop back to the previous batch cap "
        "even if aggregate tokens/s was higher."
    ),
    "corpus-00019": (
        "KV cache vs. latency, throughput and memory, viewed through the levers that actually change the "
        "trade-off.\n\n"
        "Mechanism. The cache removes recompute from decode and replaces it with per-step memory traffic and "
        "persistent HBM occupancy. Every lever below moves one of those three axes:\n"
        "1. GQA/MQA - cuts kv_heads, so kv_bytes and per-step KV traffic drop by the query/KV head ratio. This is "
        "an architectural property of the checkpoint and cannot be changed at serving time.\n"
        "2. KV quantisation (fp8 or int8) - halves or quarters bytes_per_elem; raises max batch and lowers TPOT, "
        "at a quality risk that must be measured, not assumed.\n"
        "3. Paged allocation with small blocks - reduces reservation waste, raises achievable concurrency, adds a "
        "block-table indirection per attention kernel.\n"
        "4. Prefix caching - cuts prefill work and TTFT for shared prefixes; does nothing for TPOT.\n"
        "5. Disaggregated prefill/decode (Dynamo/Mooncake style) - lets prefill and decode scale on separate "
        "hardware pools, but requires shipping the KV cache between them, so the interconnect (NVLink, RDMA/RoCE "
        "with GPUDirect RDMA) becomes the new bottleneck and adds a transfer term to TTFT.\n\n"
        "Boundary condition for lever 5: KV transfer is worthwhile only when transfer_time = kv_bytes / "
        "effective_link_bandwidth is materially less than the prefill recompute time it avoids. Over a slow or "
        "congested fabric, or without GPUDirect RDMA so the transfer takes an extra host bounce, disaggregation "
        "can be net-negative on TTFT.\n\n"
        "Falsifiable hypothesis: moving from fp16 to fp8 KV should increase max concurrent sequences by close to "
        "2x at the same context length. If it does not, the limit was activation workspace or fragmentation "
        "headroom, not KV, and the quantisation quality risk was taken for nothing.\n\n"
        "Evidence needed: per-lever A/B with fixed traffic replay, measured max concurrency, TTFT/TPOT "
        "percentiles, KV-transfer bandwidth and whether GDR is actually in use (not just configured), plus a "
        "held-out quality eval for any quantisation change.\n\n"
        "Rollback gate: revert KV quantisation if held-out quality moves beyond the pre-registered tolerance; "
        "revert disaggregation if p95 TTFT does not improve versus the colocated baseline."
    ),
    "corpus-00020": (
        "KV cache vs. latency, throughput and memory in a multi-GPU / multi-node setting.\n\n"
        "Mechanism. Under tensor parallelism of degree TP, attention heads are sharded, so each rank holds roughly "
        "kv_bytes / TP. That relieves per-GPU capacity and per-GPU KV bandwidth roughly linearly, but every decode "
        "step now contains NCCL all-reduces whose cost is a fixed latency floor (order tens of microseconds "
        "intra-node over NVLink, materially higher across nodes over RoCE/InfiniBand) plus a bandwidth term. So TP "
        "trades a memory-bandwidth problem for a collective-latency problem, and there is a TP degree beyond which "
        "TPOT gets worse, not better. Pipeline parallelism does not shard the cache within a stage at all - it "
        "partitions layers, so each stage holds the full KV for its own layers and adds pipeline bubbles that hurt "
        "single-stream decode latency.\n\n"
        "Boundary condition: TP scaling of decode holds only while the per-step collective time stays small "
        "relative to the per-step KV read time. Crossing a node boundary usually breaks this: intra-node NVLink "
        "bandwidth is far above typical inter-node fabric bandwidth, so TP across nodes for decode is normally the "
        "wrong choice unless the model genuinely does not fit otherwise. " + KV_MEM + "\n\n"
        "Falsifiable hypothesis: doubling TP within one node should reduce per-GPU KV footprint by ~2x and improve "
        "TPOT sublinearly; doubling TP across a node boundary should show TPOT regression once collective time "
        "exceeds the KV-read saving. Measure both and compare against the predicted crossover.\n\n"
        "Evidence needed: per-rank memory breakdown, NCCL collective timings per step (NCCL_DEBUG plus a profiler "
        "trace, not inference from wall clock alone), confirmation of the actual transport in use (NVLink vs. "
        "PCIe vs. RoCE, and whether GPUDirect RDMA is active), fabric link utilisation and any retransmit or "
        "pause-frame counters on the RoCE side, and TPOT/tokens-per-second at each TP degree.\n\n"
        "Rollback gate: if the higher TP degree does not improve p95 TPOT at the target concurrency, or if RoCE "
        "counters show congestion or retransmits, return to the lower TP degree and re-measure before any further "
        "topology change."
    ),
}

PLAN = {
    "corpus-00021": (
        "Measurement plan: does the KV cache help this serving workload?\n\n"
        "Framing. 'Cache on vs. cache off' is not a real option in a modern engine - correctness of decode depends "
        "on it. The answerable question is whether the current KV *configuration* (dtype, block size, pool size, "
        "prefix reuse) is on the right operating point. Frame the experiment that way.\n\n"
        "Baseline and control. Pin engine version, model checkpoint, kv dtype, TP degree, max_num_seqs, block "
        "size, and driver/CUDA version; record them all in the run manifest. Replay a captured production trace "
        "(real input/output length distribution and real arrival process), not a fixed-length synthetic benchmark "
        "- the conclusion is highly sensitive to the length distribution.\n\n"
        "Design. Sweep one variable at a time against a fixed offered-load ladder (requests/s stepped until the "
        "SLO breaks). Primary metrics: p50/p95/p99 TTFT, p50/p95/p99 TPOT, goodput (requests completing within "
        "SLO), max concurrent sequences. Secondary: KV-pool occupancy, preemption and swap counters, prefix "
        "cache-hit rate, HBM DRAM-active percentage from DCGM. Three repetitions per point; report medians and "
        "spread, and discard the first run as warm-up (CUDA graph capture, allocator warm-up, autotuning).\n\n"
        "Concrete mechanism being tested: if decode is KV-bandwidth-bound, kv_bytes reduction should show up "
        "directly in TPOT. " + KV_MEM + "\n\n"
        "Boundary condition: results are valid only for the measured length distribution and concurrency range. "
        "Do not extrapolate to longer contexts, because the KV term grows linearly with sequence length while "
        "weight-read cost does not.\n\n"
        "Falsifiable hypothesis, pre-registered: reducing kv_bytes by 2x (fp8 KV) increases sustainable goodput "
        "at the SLO by at least 25 percent. Refuted if the gain is under 5 percent or within run-to-run spread.\n\n"
        "Evidence required for a go decision: the metric table above plus a held-out quality eval showing no "
        "regression beyond the agreed tolerance.\n\n"
        "Rollback gate: any SLO regression, any quality regression beyond tolerance, or any increase in preemption "
        "rate reverts the configuration; keep the baseline config in version control so the revert is one deploy."
    ),
    "corpus-00022": (
        "Measurement plan for validating a KV-cache change, with emphasis on isolating the variable.\n\n"
        "Step 1 - characterise the workload. Extract from production logs: input-length and output-length "
        "histograms, shared-prefix ratio, arrival-rate distribution, and concurrency profile. Every downstream "
        "number is conditional on these; record them as part of the result.\n\n"
        "Step 2 - build the roofline prediction before measuring. Compute kv_bytes_per_seq and weight_bytes, then "
        "predict the batch size at which the KV read term overtakes the weight read term. " + KV_MEM + " Writing "
        "the prediction down first is what makes the experiment falsifiable rather than a fishing expedition.\n\n"
        "Step 3 - controlled sweep. Vary exactly one of {kv dtype, block size, max_num_seqs, prefix caching "
        "on/off} per arm. Hold traffic replay, model, engine build, and TP degree fixed. Use a closed-loop load "
        "generator at fixed offered rate, not open-loop max-throughput blasting, so latency percentiles are "
        "meaningful.\n\n"
        "Step 4 - instrument at three layers. Engine level (KV occupancy, preemptions, cache-hit rate, batch "
        "composition per step), GPU level (DCGM DRAM-active, SM occupancy, achieved HBM bandwidth), and client "
        "level (TTFT, TPOT, end-to-end latency, error and timeout rate). A change that improves GPU counters but "
        "not client metrics is not a win.\n\n"
        "Boundary condition: with prefix caching enabled, the measured TTFT depends on cache warmth. Report cold "
        "and warm separately; mixing them makes the comparison meaningless.\n\n"
        "Falsifiable hypothesis: prefix caching reduces mean TTFT in proportion to the shared-prefix fraction "
        "measured in step 1. Refuted if TTFT improves materially less than that fraction while the reported "
        "cache-hit rate is high - which would point at prefill not being the TTFT bottleneck.\n\n"
        "Evidence required: the step-1 histograms, the step-2 written prediction, per-arm metric tables with "
        "repetitions, and the delta between predicted and observed knee.\n\n"
        "Rollback gate: revert if client-level p95 does not improve, if timeout/error rate rises at all, or if "
        "observed behaviour contradicts the roofline prediction without an explanation."
    ),
    "corpus-00023": (
        "Measurement plan focused on whether KV capacity, not compute, is the binding constraint.\n\n"
        "Hypothesis to test first: the serving tier is KV-capacity-bound. Concrete mechanism: if true, the "
        "scheduler's running batch is capped by free KV blocks rather than by max_num_seqs, GPU SM utilisation "
        "stays well below saturation while HBM is nearly full, and the queue grows even though the GPU looks idle.\n\n"
        "Discriminating measurements.\n"
        "1. Sample KV-pool free blocks and the running/waiting queue lengths at high frequency (at least 1 Hz) "
        "under peak load. KV-bound looks like free_blocks pinned near zero with a non-empty waiting queue.\n"
        "2. Compare against DCGM SM-active and DRAM-active. Compute-bound shows SM-active high; bandwidth-bound "
        "shows DRAM-active high with SM-active low; capacity-bound shows both moderate with preemptions non-zero.\n"
        "3. Do a controlled capacity perturbation: reduce gpu_memory_utilization (shrinking the KV pool) and "
        "confirm goodput falls; then reduce kv_bytes (fp8 KV, or lower max_model_len) and confirm goodput rises. "
        "A metric that responds to both directions is a real causal signal; one that responds to neither means the "
        "bottleneck is elsewhere.\n\n"
        "Sizing input for the perturbation. " + KV_MEM + "\n\n"
        "Boundary condition: this diagnosis is only valid at the offered load where it was taken. A tier can be "
        "compute-bound at low concurrency and KV-capacity-bound at high concurrency; always report the load point "
        "alongside the conclusion.\n\n"
        "Falsifiable hypothesis, pre-registered: a 2x reduction in kv_bytes_per_seq raises max concurrent running "
        "sequences by at least 1.7x. Refuted if concurrency rises by less than 1.2x, which would indicate that "
        "activation workspace, fragmentation headroom, or max_num_seqs is the real cap.\n\n"
        "Evidence required: the time series from measurements 1 and 2, the two-direction perturbation results, "
        "engine configuration dump, and preemption/swap counters for each arm.\n\n"
        "Rollback gate: restore the original gpu_memory_utilization and kv dtype immediately if error rate, "
        "timeout rate, or held-out quality regresses; run the perturbation in a canary replica, never on the "
        "whole fleet at once."
    ),
}

ANSWERS = {}
ANSWERS.update(FAILMODE)
ANSWERS.update(INTERACT)
ANSWERS.update(PLAN)

META = {
    "corpus-00013": (["source assistant only restates the caching mechanism and the memory-scaling factors; it "
                      "never enumerates two failure modes or trade-offs as instructed",
                      "no boundary condition, no units, no falsifiable claim",
                      "no measurement or rollback guidance, so it cannot guide an operator"],
                     ["model config: num_hidden_layers, num_key_value_heads, head_dim, kv dtype",
                      "measured achievable HBM bandwidth from a microbenchmark",
                      "engine KV-pool occupancy and preemption counters",
                      "TPOT vs. context-length sweep"], 0.72),
    "corpus-00014": (["source assistant is identical boilerplate reused across variants and does not answer the "
                      "requested two failure modes",
                      "omits cache-reuse correctness hazards, which are silent failures with no crash signal",
                      "no boundary condition on when prefix reuse is worthwhile"],
                     ["token-level prefix-overlap histogram of production traffic",
                      "engine prefix cache-hit rate",
                      "block-allocator internal/external fragmentation stats",
                      "cached-vs-uncached logit equivalence test"], 0.7),
    "corpus-00015": (["source assistant does not mention eviction, preemption, swap or multi-tenant fairness",
                      "no quantitative comparison between swap and recompute eviction",
                      "no tail-latency or per-tenant quota consideration"],
                     ["preemption and swap counters over time",
                      "queue-depth and admission-rejection time series",
                      "per-tenant KV-block occupancy",
                      "measured PCIe host-transfer bandwidth",
                      "p50/p95/p99 TTFT and TPOT segmented by request class"], 0.68),
    "corpus-00017": (["source assistant never separates prefill from decode, so it cannot explain why the cache "
                      "does not reduce TTFT on a cold request",
                      "no throughput or bandwidth-bound reasoning",
                      "no numeric sizing example and no units"],
                     ["TTFT and TPOT percentiles separated by prefill and decode phases",
                      "batch-size sweep at fixed input/output lengths",
                      "KV-pool occupancy",
                      "DCGM HBM-bandwidth utilisation",
                      "actual model KV configuration"], 0.73),
    "corpus-00018": (["source assistant gives no roofline or capacity model, so it cannot predict the throughput knee",
                      "does not state the latency/throughput tension - no operating point can be chosen from it",
                      "no MoE, small-model, or tensor-parallel caveats"],
                     ["measured weight footprint and kv_bytes_per_seq",
                      "achievable HBM bandwidth microbenchmark",
                      "tokens/s and TPOT across a batch sweep",
                      "TP degree and measured collective time per decode step"], 0.7),
    "corpus-00019": (["source assistant lists no levers (GQA, KV quantisation, paging, prefix caching, "
                      "disaggregation) so it gives an operator nothing actionable",
                      "no mention that disaggregated prefill/decode moves the bottleneck to the interconnect",
                      "no quality-risk caveat for KV quantisation"],
                     ["per-lever A/B under fixed traffic replay",
                      "measured max concurrency and TTFT/TPOT percentiles",
                      "KV-transfer bandwidth and confirmation that GPUDirect RDMA is actually active",
                      "held-out quality eval for any quantisation change"], 0.66),
    "corpus-00020": (["source assistant ignores multi-GPU entirely; it does not mention TP sharding of the cache",
                      "no NCCL collective-latency trade-off and no node-boundary caveat",
                      "could mislead an operator into scaling TP across nodes for decode"],
                     ["per-rank memory breakdown",
                      "NCCL collective timings per decode step from a profiler trace",
                      "confirmation of transport in use (NVLink vs. PCIe vs. RoCE) and GDR status",
                      "RoCE link utilisation, retransmit and pause-frame counters",
                      "TPOT and tokens/s at each TP degree"], 0.64),
    "corpus-00021": (["the instruction asks for a measurement plan; the source assistant gives a concept "
                      "restatement with no plan at all",
                      "no metrics, no controls, no repetitions, no acceptance criteria",
                      "no rollback gate, which is an operational-safety gap"],
                     ["captured production trace with real length and arrival distributions",
                      "pinned engine/model/driver versions in the run manifest",
                      "p50/p95/p99 TTFT and TPOT plus goodput at each load step",
                      "KV occupancy, preemption, prefix-hit and DCGM DRAM-active counters",
                      "held-out quality eval for any dtype change"], 0.75),
    "corpus-00022": (["no measurement plan is given despite the explicit instruction",
                      "no isolation-of-variables discipline, so any result would be confounded",
                      "no cold-vs-warm cache distinction, which would silently bias TTFT results"],
                     ["input/output length histograms and shared-prefix ratio from production logs",
                      "written roofline prediction recorded before measurement",
                      "per-arm client, engine and GPU metric tables with repetitions",
                      "closed-loop load generator configuration"], 0.72),
    "corpus-00023": (["no plan, and no way to distinguish compute-bound from bandwidth-bound from "
                      "capacity-bound behaviour",
                      "no perturbation design, so any conclusion would be correlational",
                      "no canary or blast-radius control for the proposed configuration changes"],
                     ["high-frequency KV free-block and queue-length time series at peak load",
                      "DCGM SM-active and DRAM-active counters",
                      "two-direction capacity perturbation results",
                      "engine configuration dump and preemption/swap counters per arm"], 0.7),
}


def main():
    recs = []
    with open(CORP, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if i < LO:
                continue
            if i > HI:
                break
            d = json.loads(line)
            sid = d["id"]
            u = next(m["content"] for m in d["messages"] if m["role"] == "user")
            a = next(m["content"] for m in d["messages"] if m["role"] == "assistant")
            risks, evid, conf = META[sid]
            recs.append({
                "source_id": sid,
                "teacher_lane": "teacher-B",
                "teacher_model": "claude-opus-5-current",
                "calibration_status": "provisional",
                "decision": "rewrite",
                "source_user": u,
                "source_assistant": a,
                "corrected_answer": ANSWERS[sid],
                "quality_dimensions": {
                    "technical_correctness": 3,
                    "instruction_coverage": 1,
                    "operational_safety": 2,
                },
                "risks": risks,
                "evidence_required": evid,
                "confidence": conf,
            })
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(recs)} records -> {OUT}")


if __name__ == "__main__":
    main()
