import json, os

CORPUS = 'research/ai-infra-expert/corpus/train.jsonl'
OUT = 'experiments/2026-08-17-teacher-b-corpus-review/results/train-batch-0004.jsonl'
LO, HI = 30, 40

SIZING = ("Sizing identity to state up front: kv_bytes = 2 * layers * seq_len * kv_heads * head_dim * "
          "bytes_per_elem per sequence (the 2 covers K and V). Read layers/num_key_value_heads/head_dim "
          "and the KV dtype from the served config; never assume them.")

ANS = {}

ANS[30] = f"""How the KV cache differs between training and inference (variant 4 - the parallelism and sharding view)

Mechanism. In training, K and V are transient activations. Their sharding follows whatever parallelism the trainer uses: with tensor parallelism the attention heads are split across ranks, with sequence/context parallelism the sequence dimension is split and partial attention results are combined with a ring or all-to-all exchange, and with ZeRO/FSDP the parameters and optimizer state are sharded while activations stay local. Nothing persists between steps, so re-sharding costs nothing beyond the step itself.

In inference the cache is persistent state that is pinned to whatever rank produced it. With tensor parallelism of degree TP, per-GPU footprint is kv_bytes_per_gpu = 2 * layers * seq_len * (kv_heads / TP) * head_dim * bytes_per_elem - but only while kv_heads >= TP. With GQA/MQA where kv_heads < TP, the KV heads cannot be split further and are replicated, so raising TP stops reducing per-GPU KV and only adds all-reduce latency. This is one of the most common wrong capacity claims in serving reviews. {SIZING}

Second structural difference: because the cache is pinned, a serving system cannot re-shard mid-request. Changing TP requires draining. Disaggregated prefill/decode (Mooncake, NVIDIA Dynamo) instead moves the cache across the fabric with RDMA/GPUDirect, which makes KV a network object with a transfer cost on the TTFT critical path - a concept with no training analogue.

Falsifiable hypothesis. H1: doubling TP from 2 to 4 reduces per-GPU KV bytes by ~2x if and only if kv_heads >= 4; if kv_heads < TP, measured per-GPU KV stays flat while decode latency rises from extra collectives. Test by reading pool occupancy at fixed concurrency across TP settings.

Boundary condition. The linear per-GPU model breaks (a) when kv_heads < TP (replication regime), and (b) with pipeline parallelism, where each stage holds only its own layers, so per-GPU KV falls by layers/PP but inter-stage bubbles now dominate decode latency at low batch.

Evidence required: config dump (layers, kv_heads, head_dim, dtype), TP/PP settings, per-GPU KV pool occupancy at fixed concurrency, nvidia-smi topo -m, NCCL collective timings per rank including the straggler, and for disaggregation the KV transfer latency distribution.

Rollback gate: revert a parallelism change if measured per-GPU KV does not move as predicted, if the straggler rank regresses, or if P99 TPOT rises while pool occupancy is unchanged."""

ANS[31] = f"""How the KV cache differs between training and inference (variant 5 - the numerics and correctness view)

Mechanism. Training computes K and V in the same pass that consumes them, under the trainer's autocast policy (typically bf16 compute with fp32 master weights and fp32 reductions). There is no storage format decision to make: the values live for one step and are consumed by the backward pass, so precision choices are governed by gradient stability, not by memory footprint.

Inference stores K and V for the whole request, which turns their dtype into a capacity lever. Halving the KV dtype (bf16 -> fp8) halves kv_bytes and therefore roughly doubles the number of concurrent sequences the pool holds, and also halves the bytes read per decode step, so it moves the bandwidth-bound TPOT floor down by ~2x. That is a real mechanism, but it injects quantisation error into every attention score for the remaining life of the sequence. {SIZING}

Why the error profile differs from training. A training step's numerical error is not carried forward across steps - the optimizer averages over many samples. A decode step's KV error is carried forward for every subsequent token in that sequence, and the number of attended quantised positions grows with context length. So the same nominal dtype is materially riskier at inference-time long context than in a training forward pass.

Falsifiable hypothesis. H1: with fp8 KV, greedy-decode outputs match the bf16 baseline on >= 99% of a held-out prompt set at the maximum context length actually served, and the task metric moves less than a pre-registered tolerance. H0: it does not, in which case the throughput gain must be reported as a quality/throughput trade, never as a pure win.

Boundary condition. Equivalence measured at 2k context does not transfer to 64k - error accumulates with attended positions. Kernel-level accumulation order also differs between attention backends, so bitwise identity is not a valid criterion; the acceptance criterion must be stated as a token-match rate or task-metric tolerance before the experiment runs.

Evidence required: KV dtype and scaling scheme, attention backend and kernel version, decoding config (greedy, seed), held-out prompt-set hash, token-level diff rate at max served context, task metric with confidence intervals, and the throughput/latency table.

Rollback gate: revert if greedy-token match falls below the pre-registered threshold at max context, or if the task metric regresses beyond tolerance, regardless of the measured throughput gain."""

ANS[32] = """One misleading intuition about the KV cache, corrected (variant 1 - "the KV cache makes decoding compute-cheap, so decode should be fast")

The misleading intuition. Because the cache removes O(L^2) recomputation, people conclude decode is now cheap and should run near peak FLOPS. Utilisation dashboards then look "broken": GPUs are busy, but tokens/s is low.

Correction and mechanism. The cache does not make decode cheap - it changes which resource is scarce. Each decode step must read the entire cache for that sequence and perform roughly one multiply-accumulate per element loaded, so arithmetic intensity is ~1 FLOP/byte, far below the ~100+ FLOP/byte needed to saturate modern tensor cores. Decode is therefore memory-bandwidth-bound, and the floor is:

  TPOT_floor ~= kv_bytes_per_step / achievable_HBM_bandwidth
  where kv_bytes = 2 * layers * seq_len * kv_heads * head_dim * bytes_per_elem per sequence.

The correct response to slow decode is not more FLOPS. It is (a) raise batch size so weight reads amortise across sequences, (b) shrink bytes per token via GQA/MQA or KV quantisation, (c) use paged allocation so higher concurrency is actually reachable, or (d) speculative decoding to produce several tokens per pass.

Falsifiable check. H1: measured TPOT at batch size 1 is within ~2x of kv_bytes/achievable_bandwidth, and raising batch size increases tokens/s roughly linearly until the pool or the SLO binds. H0: TPOT is far above the floor and does not improve with batch - then the bottleneck is elsewhere (kernel launch overhead, CPU-side scheduling, tokenisation, network) and none of the KV levers will pay.

Boundary condition. This holds for the decode phase only. Prefill is compute-bound and does behave like the naive intuition; a workload dominated by short outputs is prefill-bound, so KV-side optimisation will produce close to zero benefit there. Also, "achievable bandwidth" must come from a microbenchmark, not the datasheet peak - typically 70-85% of peak.

Evidence required: model config and KV dtype, measured achievable HBM bandwidth, batch-size sweep of TPOT and tokens/s, prefill vs decode token counts, and a profiler trace showing the attention kernel is memory-stalled rather than launch-bound.

Rollback gate: do not fund a compute-side change (bigger GPU, higher clocks) for a decode-latency problem unless the profile shows compute stalls rather than memory stalls; re-measure against the bandwidth floor first."""

ANS[33] = """One misleading intuition about the KV cache, corrected (variant 2 - "KV memory is small compared to the weights, so it can be ignored in capacity planning")

The misleading intuition. Weights are the headline number, so planners size a deployment by weights plus "some slack" and treat KV as rounding error. Under load the server then preempts, swaps, or OOMs at concurrency far below the plan.

Correction and mechanism. Weights are a fixed, one-time cost; KV is a per-sequence cost that scales with concurrency and context length:

  kv_bytes = 2 * layers * seq_len * kv_heads * head_dim * bytes_per_elem   (per sequence)

Illustrative arithmetic only, not a measurement: a model with 32 layers, 8 KV heads, head_dim 128, fp16 KV, at 8192 tokens gives 2*32*8192*8*128*2 B ~= 1.07 GB per sequence. Sixty-four such concurrent sequences require ~69 GB of KV - which can exceed the weight footprint of a mid-sized model several times over. Long context plus high concurrency, not model size, is what usually decides how many replicas you need.

The right planning identity is: usable_kv_bytes = HBM - weights - activation/workspace - allocator fragmentation - CUDA context; max_concurrency ~= usable_kv_bytes / kv_bytes(typical_seq_len), with >= 15% headroom held for burst.

Falsifiable check. H1: the concurrency at which the first preemption occurs matches the predicted max_concurrency within ~10%. H0: it does not - meaning one declared input is wrong (KV dtype, kv_heads under GQA, pool fraction, block size) and the capacity model must be fixed before it is used to size hardware.

Boundary condition. The linear model holds only under paged allocation, where internal fragmentation is bounded by about one block per sequence (~num_seqs * block_size * bytes_per_token). With contiguous per-request pre-allocation to max_model_len, usable capacity collapses to pool / kv_bytes(max_model_len) regardless of actual prompt lengths, and the linear estimate is simply invalid.

Evidence required: config dump, computed and measured KV pool bytes, block size, occupancy trace under load ramp, first-preemption concurrency, and prompt/output length distributions from a real trace.

Rollback gate: do not sign off a capacity plan whose predicted first-preemption concurrency is more than 10% away from the measured one, or that leaves less than 15% pool headroom at forecast peak."""

ANS[34] = """One misleading intuition about the KV cache, corrected (variant 3 - "prefix caching always helps, so turn it on everywhere")

The misleading intuition. Prefix/prompt caching skips prefill for shared leading tokens, so it is treated as free money and enabled globally by default.

Correction and mechanism. The saving is bounded and specific: with hash-keyed block reuse (vLLM automatic prefix caching, SGLang RadixAttention, Mooncake-style external KV stores), only the prefill work on the shared prefix is skipped. The upper bound on the win is roughly shared_prefix_tokens / total_prompt_tokens of prefill compute, and it is exactly zero for decode. In a decode-dominated workload (long outputs, short prompts) the measurable benefit is near noise, while the hashing, lookup, and reference-counting work is paid on every request.

Worse, reuse consumes pool capacity: retained blocks are blocks not available to in-flight sequences. If the reusable working set exceeds the pool, the evictor recycles blocks before they are hit again, hit rate collapses, and you have added overhead with no mechanism.

Operational safety point that is frequently missed: block reuse keyed only on token hashes can serve one tenant's cached content to another if namespaces are not isolated. In a multi-tenant deployment prefix caching is a data-isolation decision, not just a performance knob. It also creates a timing side channel - a cache hit is observably faster, which can leak whether a given prefix was seen before.

Falsifiable check. H1: on a replayed production trace, enabling the cache cuts P50 TTFT by >= 20% with cache hit rate >= 30% and no P99 TPOT regression beyond 5%. H0: hit rate is low or TTFT is unchanged - then the mechanism is absent and the feature should stay off.

Boundary condition. The win disappears when (a) prompts are near-unique, (b) concurrency * kv_bytes approaches pool size so eviction outruns reuse, or (c) outputs are long enough that decode dominates end-to-end time. Synthetic random prompt sets destroy prefix locality and will systematically understate it; production replay is mandatory for a valid answer either way.

Evidence required: replay trace hash and prefix-sharing rate, cache hit rate, TTFT/TPOT percentiles with CIs, pool occupancy and preemption counters, and the tenancy/namespace isolation configuration.

Rollback gate: disable if hit rate < 10%, if P99 TPOT regresses > 5%, if preemptions increase, or if tenant isolation for cached blocks cannot be demonstrated."""

ANS[35] = """One misleading intuition about the KV cache, corrected (variant 4 - "tensor parallelism splits the KV cache, so more GPUs always means proportionally more concurrency")

The misleading intuition. Tensor parallelism shards attention heads, so people assume per-GPU KV falls as 1/TP and that going from TP=2 to TP=8 multiplies serving concurrency by four.

Correction and mechanism. KV is sharded along the key/value head dimension, so:

  kv_bytes_per_gpu = 2 * layers * seq_len * (kv_heads / TP) * head_dim * bytes_per_elem

This only holds while kv_heads >= TP. Modern models use GQA/MQA with few KV heads - 8, 4, sometimes 1. Once TP exceeds kv_heads, the KV heads cannot be divided further and engines replicate them across ranks. From that point on, adding GPUs does not reduce per-GPU KV bytes at all; it only adds all-reduce traffic per layer, raising decode latency. The concurrency ceiling stops improving exactly where people expect the biggest gain.

There is a second, opposing effect: weights do keep sharding with TP, which frees HBM for the KV pool. So total capacity may still improve somewhat past kv_heads, but through a different mechanism and with a much smaller slope - and it is paid for with collective latency on every decode step, which is precisely the latency-sensitive path.

Falsifiable check. H1: per-GPU pool occupancy at fixed concurrency falls ~2x when TP doubles while TP <= kv_heads, and stays flat once TP > kv_heads. H0: it does not follow that shape - then the engine's sharding behaviour differs from the assumed one and capacity planning must be re-derived from measurement.

Boundary condition. Intra-node NVLink/NVSwitch makes the extra all-reduces relatively cheap; spanning TP across nodes over RoCE/InfiniBand puts a per-layer collective on the fabric for every decode step, and decode latency becomes dominated by network round trips. TP across nodes for a latency-SLO workload is almost always the wrong tool - use pipeline parallelism, data-parallel replicas, or disaggregation instead.

Evidence required: num_key_value_heads from the config, TP setting, per-GPU KV pool occupancy at fixed concurrency across TP values, nvidia-smi topo -m, per-rank collective timings including the straggler, and TPOT percentiles per configuration.

Rollback gate: revert the TP increase if per-GPU KV occupancy does not drop as predicted, if TP crosses a node boundary for a latency-SLO service, or if P99 TPOT regresses despite higher nominal capacity."""

ANS[36] = """One misleading intuition about the KV cache, corrected (variant 5 - "a KV cache hit means the request is basically free")

The misleading intuition. Dashboards show a high cache hit rate, so operators assume those requests cost nothing and admit more load, expecting flat latency.

Correction and mechanism. A hit removes prefill compute for the matched prefix. It does not remove: (a) decode, which still costs one full cache read per generated token at ~1 MAC per byte loaded, so TPOT_floor ~= kv_bytes/achievable_bandwidth remains; (b) pool residency, since the hit sequence still occupies blocks for its whole lifetime and grows one block per block_size tokens; (c) scheduling, admission, tokenisation, and detokenisation overhead; and (d) in a disaggregated setup (Mooncake, NVIDIA Dynamo), the KV transfer itself - a "hit" in a remote store still costs an RDMA transfer of the matched blocks, whose time sits directly on TTFT.

So the correct mental model is: a hit converts compute cost into memory-residency and possibly network cost. Admission control must be driven by pool occupancy and preemption rate, not by hit rate. A system at 90% hit rate can still collapse if concurrency * kv_bytes exceeds the pool.

Falsifiable check. H1: at fixed hit rate, raising offered load increases P99 TTFT non-linearly once pool occupancy crosses ~85%, and preemption count rises before the SLO breach. H0: latency stays flat with occupancy - which would mean the pool is not the binding constraint and the capacity story is different from the assumed one.

Boundary condition. In disaggregated serving the claim "hits are cheap" is only true while the KV transfer path is healthy and GPUDirect RDMA is actually engaged. If GDR is not active, or the HCA and GPU sit on different PCIe root complexes, transfers stage through host memory and a "hit" can be slower than recomputing prefill locally. Verify with an ib_write_bw baseline on the same path and per-request transfer timing, not with the hit-rate metric.

Evidence required: hit rate together with pool occupancy, preemption/swap counters, TTFT/TPOT percentiles across a load ramp, prefill vs decode token accounting, and for disaggregation the KV transfer latency distribution plus GDR/HCA-GPU affinity confirmation.

Rollback gate: raise admission limits only if a load ramp shows occupancy staying below ~85% and preemptions at zero at the new limit; revert immediately if preemptions appear or P99 TTFT crosses the SLO."""

ANS[37] = """A small controlled experiment for the KV cache (variant 1 - measure the decode bandwidth floor)

Question. Is decode on this deployment actually bounded by KV-cache reads, as the standard model claims?

Mechanism under test. Each decode step reads the whole per-sequence cache with ~1 MAC per element, so the predicted floor is TPOT_floor = kv_bytes_per_step / achievable_HBM_bandwidth, with kv_bytes = 2 * layers * seq_len * kv_heads * head_dim * bytes_per_elem.

Falsifiable hypothesis. H1: at batch size 1, measured TPOT grows linearly in context length with slope within 2x of kv_bytes_per_token / measured_bandwidth. H0: the curve is flat or superlinear, meaning something other than KV reads dominates (kernel launch overhead, CPU scheduling, or attention-kernel inefficiency).

Procedure.
1. Fix the independent variable: context length, swept at 1k, 4k, 16k, 32k tokens. Hold everything else constant - same server build, same engine flags, same model config, batch size 1, greedy decoding with a fixed seed, fixed 128 output tokens.
2. Measure achievable HBM bandwidth independently first (a simple copy/stream microbenchmark). Do not use the datasheet peak; achievable is typically 70-85% of it.
3. Discard a warmup window (at least 30 s or 3 requests) so allocator and clock state are in steady state; record SM/memory clocks and temperature to confirm no thermal throttling between arms.
4. Run >= 3 repetitions per point, interleaved rather than blocked, so drift does not align with the treatment.
5. Report per-point median TPOT with a confidence interval, plus the predicted floor, plus the ratio measured/predicted.

Control that makes it a controlled experiment. A batch-size-1 run isolates the mechanism from scheduling and batching effects; only after the batch-1 curve matches the model should the sweep be repeated at production concurrency.

Boundary condition. The linear relation holds only while the sequence fits comfortably in the pool and no preemption occurs; once occupancy nears capacity the curve breaks and points past that knee must not be fitted. At very short contexts, fixed per-step overhead (kernel launch, sampling, detokenisation) dominates and the measured slope will be misleadingly flat.

Evidence required: model config and KV dtype, engine version and flags, bandwidth microbenchmark result, the sweep table with CIs, clock/temperature logs, and a profiler trace confirming memory stalls in the attention kernel.

Rollback gate: if measured/predicted exceeds ~2x, do not pursue KV-size optimisations; profile for launch and scheduling overhead first, and re-run this experiment before any capacity change is approved."""

ANS[38] = """A small controlled experiment for the KV cache (variant 2 - find the true concurrency ceiling)

Question. At what concurrency does this deployment start preempting, and does it match the capacity model?

Mechanism under test. usable_kv_bytes = HBM - weights - activation/workspace - fragmentation - CUDA context; predicted_max_concurrency = usable_kv_bytes / kv_bytes(target_seq_len), with kv_bytes = 2 * layers * seq_len * kv_heads * head_dim * bytes_per_elem. Under paged allocation, internal fragmentation should be bounded by about one block per sequence.

Falsifiable hypothesis. H1: the concurrency at which the first preemption/swap event occurs is within 10% of predicted_max_concurrency. H0: it is materially lower, indicating a wrong assumption (KV dtype, kv_heads under GQA, pool fraction, block size) or an allocator problem such as retained blocks.

Procedure.
1. Compute the prediction first and write it down before running. Pre-registering the number is what makes this falsifiable rather than a post-hoc rationalisation.
2. Fix sequence length exactly (e.g. every request has a 4096-token prompt and 128-token output) so kv_bytes per sequence is a constant, not a distribution. This is the control that makes the ceiling identifiable.
3. Ramp concurrency in steps (e.g. 4, 8, 16, 24, 32, 40...), holding each step long enough (>= 60 s) to reach steady state.
4. At each step record: KV pool occupancy, free block count, preemption count, swap count, queue depth, TTFT and TPOT percentiles, and tokens/s.
5. Identify the first step where preemption count becomes non-zero. Repeat the ramp >= 3 times; report the ceiling as a range, not a point.

Boundary condition. The result is valid only for the fixed sequence length used. Real traffic has a length distribution, and the effective ceiling under mixed lengths is lower than the fixed-length ceiling because long sequences hold blocks for longer. Treat this number as an upper bound and re-measure with a replayed trace before using it for admission control. If swap-to-host is enabled, the failure mode is a latency cliff rather than preemption, so watch PCIe throughput as well.

Evidence required: the pre-registered prediction with its inputs, block size and pool bytes, per-step occupancy and preemption counters, latency percentiles per step, and >= 3 ramp repetitions.

Rollback gate: if measured and predicted ceilings differ by more than 10%, do not use the capacity model for hardware sizing until the discrepancy is explained; set admission limits from the measured ceiling minus 15% headroom, never from the prediction alone."""

ANS[39] = """A small controlled experiment for the KV cache (variant 3 - A/B a KV-quantisation change safely)

Question. Does switching the KV cache from bf16 to fp8 deliver the predicted capacity gain without unacceptable quality loss?

Mechanism under test. kv_bytes = 2 * layers * seq_len * kv_heads * head_dim * bytes_per_elem, so halving bytes_per_elem halves the per-sequence footprint - predicting ~2x concurrency at fixed pool size and ~2x lower bandwidth-bound TPOT floor. The cost is quantisation error injected into every stored key/value, which is attended to for the whole remaining life of the sequence.

Pre-registered hypotheses (both must be tested, and the quality one is the gate).
H1-perf: measured max concurrency before first preemption rises by >= 1.7x, and P50 TPOT at fixed concurrency falls by >= 20%.
H1-quality: greedy-decode outputs match the bf16 baseline on >= 99% of a held-out prompt set at the maximum context length actually served, and the task metric moves within a tolerance declared before the run.

Procedure.
1. Single-variable change only: identical engine build, flags, model weights, parallelism, and traffic - the KV dtype is the sole difference between arms. Any second change invalidates the attribution.
2. Interleave the arms (A/B/A/B) rather than running them in blocks, so clock drift, thermal state, and neighbour noise do not correlate with the treatment.
3. Quality first, on a held-out prompt set that was not used to tune the quantisation scales, with greedy decoding and a fixed seed. Measure at the longest context that will actually be served, since error accumulates with attended positions.
4. Then performance: concurrency ramp to first preemption, and a fixed-concurrency latency comparison with >= 3 repetitions and confidence intervals.
5. Stage the rollout: shadow traffic, then a small canary percentage with automatic comparison against the baseline arm, then full rollout - never a global flip.

Boundary condition. Quality equivalence measured at short context does not transfer to long context. Bitwise identity is not a valid criterion because attention backends differ in accumulation order; the criterion must be a token-match rate or task-metric tolerance stated in advance. Also, per-tensor scaling behaves differently from per-token/per-channel scaling - the scheme must be recorded, since the result does not generalise across schemes.

Evidence required: KV dtype and scaling scheme, engine/kernel versions, held-out prompt-set hash, token-level diff rate at max served context, task metric with CIs, concurrency ceiling per arm, latency percentiles per arm, and canary comparison output.

Rollback gate: revert immediately if the greedy-token match rate falls below the pre-registered threshold at max context, if the task metric regresses beyond tolerance, or if canary error rate or P99 latency exceeds baseline - regardless of the capacity gain."""

lines = open(CORPUS).read().splitlines()
out = []
for i in range(LO, HI):
    d = json.loads(lines[i])
    msgs = d['messages']
    su = [m['content'] for m in msgs if m['role'] == 'user'][0]
    sa = [m['content'] for m in msgs if m['role'] == 'assistant'][0]
    rec = {
        "source_id": d['id'],
        "teacher_lane": "teacher-B",
        "teacher_model": "claude-opus-5-current",
        "calibration_status": "provisional",
        "decision": "rewrite",
        "source_user": su,
        "source_assistant": sa,
        "corrected_answer": ANS[i],
        "quality_dimensions": {
            "technical_correctness": 4,
            "instruction_coverage": 1,
            "operational_safety": 3,
        },
        "risks": [
            "source_assistant is a generic KV-cache definition and does not answer the asked question (training-vs-inference contrast / misleading-intuition correction / controlled experiment design)",
            "no units, no sizing identity, no boundary condition, no falsifiable hypothesis, no rollback gate",
            "training on this pairing teaches topic-shaped non-answers and instruction drift",
        ],
        "evidence_required": [
            "served model config (num_hidden_layers, num_key_value_heads, head_dim, KV dtype) and engine version/flags",
            "benchmark trace hash, arrival process, and percentile latency table with confidence intervals over >=3 interleaved repetitions",
            "KV pool occupancy, cache hit rate, and preemption/swap counters",
            "independent achievable-HBM-bandwidth microbenchmark for the roofline cross-check",
        ],
        "confidence": 0.72,
    }
    out.append(json.dumps(rec, ensure_ascii=False))

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, 'w') as f:
    f.write("\n".join(out) + "\n")
print("wrote", OUT, len(out))
