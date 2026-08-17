import json

SRC = "research/ai-infra-expert/corpus/train.jsonl"
OUT = "experiments/2026-08-17-teacher-b-corpus-review/results/train-batch-0007.jsonl"
START, END = 60, 70

A = {}

A["corpus-00069"] = """Claim under review. "Prefill is parallel across prompt tokens and is compute-bound" is directionally right but too coarse to act on. Here is the interaction with latency, throughput and memory stated so it can be falsified.

Mechanism. Prefill runs one forward pass over all N prompt tokens. Per layer the projections and MLP are dense GEMMs of shape roughly (N x d) x (d x d'), so arithmetic intensity grows with N and the kernels reach a high fraction of peak FLOP/s once N is a few hundred. Total dense cost is about 2 * P * N FLOPs for P active parameters, plus attention scoring at O(L * H * N^2 * d_head). Decode instead issues one token at a time: ~2 * P FLOPs while streaming all P weights and the whole KV cache from HBM, i.e. bandwidth-bound.

Latency. Prefill sets TTFT. First order, TTFT ~= 2 * P * N / (FLOPs_effective) + queueing + scheduling. On an A30 (~165 TF/s dense BF16 peak, expect 30-50% achieved on real prefill kernels) a 9B dense model at N=4096 gives 2*9e9*4096 = 7.4e13 FLOPs, i.e. roughly 0.9-1.5 s of pure compute on one GPU. Decode latency (ITL) is untouched by prefill except through contention.

Throughput. Prefill and decode compete for the same SMs. In a continuous-batching server (vLLM, TensorRT-LLM, SGLang) admitting a long prefill into a step stalls every in-flight decode for the duration of that step, so p99 ITL degrades even though aggregate token throughput may rise. Chunked prefill caps the injected prefill tokens per step (e.g. 512-2048) and trades slightly worse TTFT for bounded ITL. Prefill/decode disaggregation (NVIDIA Dynamo, Mooncake) removes the contention entirely at the cost of shipping the KV cache across the fabric.

Memory. Prefill materialises KV for all N positions at once: bytes = 2 * L * N * H_kv * d_head * dtype_bytes. For a 9B-class model with L=48, H_kv=8, d_head=128, FP16: 2*48*8*128*2 = 196,608 B/token ~= 0.19 MiB/token, so N=4096 is ~0.77 GiB for one request. Peak transient activation for the attention scores is O(N^2) unless FlashAttention-style tiling is used; with FlashAttention the score matrix is never materialised and peak activation is O(N * d) per block. This is why an un-tiled attention path OOMs at long context while a tiled one does not.

Boundary condition. The "prefill is compute-bound" statement holds only while N is large enough to fill the GEMM tiles and while N^2 attention has not overtaken the dense term. At very small N (say N <= 32, common in chat turns after prefix caching) prefill degenerates to a bandwidth-bound weight sweep and behaves like decode. At very large N (>= 32k) the O(N^2) attention term dominates and TTFT stops scaling linearly, so linear extrapolation from a 2k measurement will underpredict a 32k TTFT.

Falsifiable prediction. Holding batch size at 1, TTFT vs N should be near-linear with slope 2*P/FLOPs_eff in the 512-8192 range and superlinear beyond; if measured TTFT is flat below N=512, the request is scheduler- or bandwidth-limited, not compute-limited.

Evidence needed. Per-step Nsight Compute or torch profiler traces separating GEMM from attention time; nvidia-smi / DCGM SM occupancy and HBM bandwidth during prefill; server-side TTFT and ITL histograms; KV cache bytes reported by the engine.

Rollback gate. If enabling chunked prefill or disaggregation does not improve p99 ITL by at least 20% at equal or better p99 TTFT over a 30-minute replay of production traffic, revert to the previous scheduler configuration."""

A["corpus-00070"] = """Restating precisely. Prefill is the single forward pass over the N prompt tokens that emits the first output token and populates the KV cache. Its interaction with the three resources is not uniform - it is compute-first, memory-second, and it perturbs latency of other requests through the scheduler.

Mechanism (scheduler-centric view). A continuous-batching engine builds a token budget per step. Decode contributes 1 token per active sequence; a prefill contributes up to N. When a 8k-token prefill is admitted whole, that step's kernel launch takes ~100x the wall time of a decode-only step, and every co-resident decode sequence waits. So a prefill's cost is charged to other users' inter-token latency, not only to its own TTFT. This is measurable: plot ITL over time and the prefill admissions appear as spikes aligned with new-request arrivals.

Quantitative anchors (state as estimates, not vendor facts).
- Compute: ~2 * P * N FLOPs dense + O(L*H*N^2*d_head) attention.
- KV memory: 2 * L * N * H_kv * d_head * dtype_bytes; with GQA (H_kv << H_q) this drops by the GQA ratio, typically 4-8x versus MHA.
- Bandwidth: prefill reads each weight once per step regardless of N, so weight traffic is amortised over N tokens - this is exactly why prefill is efficient and decode is not.

Boundary condition. The favourable amortisation collapses when prefix caching hits. If 95% of the prompt is served from a cached KV prefix, effective N drops to the uncached suffix; a 8k prompt with a 7.8k cache hit behaves like a 200-token prefill and is no longer compute-bound. Any capacity model that sizes GPUs from raw prompt length will over-provision in that regime. Symmetrically, cache misses after an eviction cause a sudden TTFT cliff with no change in request mix - that is a strong diagnostic signature.

Memory pressure interaction. KV cache and prefill activations draw from the same HBM pool. Raising max_num_batched_tokens improves prefill efficiency but shrinks the KV pool, lowering the number of concurrently resident sequences and therefore decode throughput. There is a real interior optimum; it must be found empirically per model, per context distribution, per GPU.

Falsifiable prediction. Doubling max_num_batched_tokens should reduce mean TTFT sublinearly (less than 2x) while increasing p99 ITL and reducing max concurrent sequences roughly in proportion to the KV bytes reclaimed. If TTFT improves 2x with no ITL cost, the server was scheduler-starved, not compute-bound, and the tuning conclusion does not generalise.

Evidence needed. Engine metrics for cache hit rate, KV blocks in use, running vs waiting sequence counts; TTFT/ITL percentiles; HBM occupancy from DCGM; a controlled A/B at fixed arrival rate with a fixed prompt-length distribution.

Rollback gate. Revert the batching-token change if p99 ITL regresses more than 15% or KV-pool exhaustion (preemption/recompute events) appears at any point in a 1-hour soak at peak arrival rate."""

A["corpus-00071"] = """The question as posed is slightly malformed and should be reframed before measuring: prefill is not an optional optimisation you can enable or disable, it is a mandatory phase of autoregressive inference. What can be validated is whether a specific prefill *policy* - chunked prefill, prefix caching, or prefill/decode disaggregation - helps a given serving workload. The plan below tests that.

Hypothesis (falsifiable). H1: enabling chunked prefill with a per-step budget of 2048 tokens reduces p99 inter-token latency by >= 20% at no worse than +10% p99 TTFT, at fixed arrival rate and fixed prompt-length distribution.

Mechanism being probed. Chunked prefill splits an N-token prefill into ceil(N/C) steps so decode sequences interleave, bounding head-of-line blocking to one chunk instead of one full prompt.

Measurement plan.
1. Workload capture. Replay a real trace, not synthetic uniform prompts. Record the joint distribution of prompt length, output length, and arrival timestamps; prefill behaviour is driven by the tail of prompt length, so a mean-only description is useless.
2. Fix the confounders. Same engine version, same model weights and quantisation, same tensor-parallel degree, same GPU clocks (lock with nvidia-smi -lgc to remove thermal drift), same KV dtype, warm caches disabled or explicitly controlled.
3. Arms. A = baseline scheduler, B = chunked prefill C=2048, C = chunked prefill C=512. Randomised interleaved blocks of 10 minutes each, at least 3 repetitions per arm to separate run-to-run variance from effect size.
4. Load points. Sweep arrival rate to find the knee: 25%, 50%, 75%, 90% of measured saturation throughput. Scheduling effects are invisible at low load; conclusions from a single load point do not transfer.
5. Metrics. Primary: p50/p95/p99 TTFT, p50/p95/p99 ITL, goodput under an SLO (e.g. TTFT <= 2 s and ITL <= 50 ms). Secondary: output tokens/s, GPU SM utilisation and HBM bandwidth via DCGM, KV blocks in use, preemption/recompute counts, queue depth.
6. Statistics. Report medians with bootstrap CIs over the repetitions, not single-run numbers. Reject H1 unless the CI for the ITL improvement excludes 0.

Boundary condition. This result is valid only for the captured prompt-length distribution and the tested concurrency band. If the production mix shifts toward short prompts (say median < 256 tokens), chunking overhead dominates and the measured benefit can invert; re-run whenever the p95 prompt length moves by more than 2x.

Evidence required. Trace file hash, engine version and full launch flags, GPU model and driver/CUDA versions, clock-lock confirmation, per-arm raw latency samples retained for re-analysis.

Rollback gate. Roll back to arm A if, in a 24-hour canary at 10% of traffic, SLO goodput is lower than baseline or preemption events exceed baseline by more than 2x."""

A["corpus-00072"] = """Reframing first: prefill always happens, so the testable object is a prefill *placement* decision. This plan validates whether prefill/decode disaggregation - running prefill on a dedicated GPU pool and streaming the KV cache to decode workers, as in NVIDIA Dynamo or Mooncake-style architectures - helps the workload versus a single colocated pool.

Hypothesis. H1: at equal total GPU count, disaggregation improves SLO goodput (fraction of requests meeting TTFT <= 2 s AND p99 ITL <= 40 ms) by >= 15% for a workload whose p95 prompt length exceeds 4k tokens.

Mechanism. Colocated serving forces prefill and decode to share SMs; a long prefill blocks decode steps. Disaggregation removes that coupling but introduces a KV transfer of 2 * L * N * H_kv * d_head * dtype_bytes per request over the fabric. For a 9B-class model at ~0.19 MiB/token, an 8k prompt moves ~1.5 GiB. Over 200 Gb/s RoCE with RDMA and GPUDirect that is ~60 ms of wire time at line rate; over TCP without GDR, expect several times worse plus host-memory bounce copies. The transfer must be overlapped with the first decode steps or it simply relocates the latency.

Measurement plan.
1. Baseline the fabric before the model. Run ib_write_bw / perftest between the prefill and decode nodes to establish achievable one-way bandwidth and latency, and nvbandwidth or a GDR-enabled microbenchmark to confirm GPU-to-GPU RDMA is actually taking the GPUDirect path (check nv_peer_mem / dmabuf module presence, and that the transfer does not show host bounce buffers in the profile). Without this, a negative result is unattributable.
2. Arms at equal cost. A = 8 GPUs colocated. B = 4 prefill + 4 decode. C = 2 prefill + 6 decode. The prefill:decode ratio is the main tunable and should track the workload's prompt:output token ratio.
3. Load sweep and replay as in any scheduler experiment: real trace, 25/50/75/90% of saturation, randomised interleaved blocks, >= 3 repetitions.
4. Metrics. SLO goodput (primary), TTFT and ITL percentiles, KV transfer time per request, fabric utilisation and RDMA retransmit/CNP counters (ECN/PFC events on RoCE), GPU utilisation per pool.

Boundary condition. Disaggregation only pays when KV transfer time is small relative to the prefill compute it de-conflicts. If prompts are short (transfer overhead dominates) or the fabric is not RDMA-capable, expect a regression. Rough break-even: transfer_time < 0.3 * prefill_time. Also, a lossy or misconfigured RoCE fabric (no PFC/ECN tuning) will show sporadic multi-hundred-millisecond tails that masquerade as scheduler problems.

Evidence required. perftest bandwidth/latency logs, confirmation of the GDR path, per-request KV transfer timings from the engine, switch-level ECN/PFC counters, raw latency samples per arm.

Rollback gate. Revert to colocated if canary SLO goodput is not above baseline after 24 hours, or if RoCE congestion counters rise above the pre-change baseline at any sustained load."""

A["corpus-00073"] = """Reframed target: validate whether *prefix caching* (reusing previously computed prefill KV for a shared prompt prefix) helps this serving workload. This is the prefill policy with the largest possible effect and the one most sensitive to workload shape.

Hypothesis. H1: with automatic prefix caching enabled, mean TTFT drops by >= 40% and prefill FLOPs per request drop by >= 40%, for a workload whose requests share a system prompt or document prefix.

Mechanism. Prefill KV for a token position depends only on the tokens at or before it, so an identical prefix yields identical KV. Block-level caching (e.g. paged KV blocks keyed by a rolling hash of the token prefix) lets a new request skip recomputation for the matched blocks and prefill only the divergent suffix. Effective prefill cost becomes 2 * P * N_uncached instead of 2 * P * N.

Measurement plan.
1. Characterise sharing first, offline. From the captured trace compute the distribution of longest-common-prefix length across requests within sliding windows of 1, 10 and 60 seconds. If median shared prefix is under a few hundred tokens, stop - the mechanism has no substrate and the experiment is not worth GPU time.
2. Arms. A = caching off. B = caching on, default cache size. C = caching on with the KV pool enlarged by 20% at the cost of max concurrent sequences.
3. Control tokenizer and templating. A single differing whitespace or a per-request timestamp injected into the system prompt destroys prefix matching. Verify by asserting cache hit rate > 0 in a smoke run before the real experiment.
4. Metrics. Cache hit rate (fraction of prompt tokens served from cache), TTFT percentiles, prefill FLOPs or prefill GPU-seconds per request, KV pool occupancy, eviction rate, end-to-end throughput, and an output-equivalence check.
5. Correctness gate. Run a fixed 500-prompt set with temperature 0 under arms A and B and compare outputs. Prefix caching must be numerically near-identical; systematic divergence indicates a hashing or block-boundary bug, not a tuning issue.

Boundary condition. The benefit vanishes when (a) prefixes are unique per request, (b) the cache is thrashed because working set exceeds the KV pool - visible as hit rate collapsing under load even though it was high at low load, or (c) requests carry per-request preambles ahead of the shared text, since matching is prefix-anchored, not substring-based. Reordering the template to put invariant text first is often the entire fix.

Evidence required. Offline LCP distribution, engine cache hit-rate metric time series, eviction counters, temperature-0 output diff report, TTFT samples per arm.

Rollback gate. Disable caching if the temperature-0 diff shows any non-trivial output change, or if hit rate under peak load falls below 20% while KV-pool pressure causes preemptions above baseline."""

A["corpus-00074"] = """Reframed target: validate whether increasing the prefill token budget per scheduler step (max_num_batched_tokens, i.e. how aggressively prefill work is packed) helps this workload. This is the cheapest prefill knob to test and the one most often mis-tuned.

Hypothesis. H1: raising max_num_batched_tokens from 2048 to 8192 reduces p50 TTFT by >= 25% while p99 ITL degrades by <= 15% and no KV-pool preemption occurs, at 75% of saturation load.

Mechanism. Larger per-step token budgets fill GEMM tiles better, raising achieved FLOP/s during prefill, and let several short prefills be fused into one step. The cost is twofold: a longer step blocks decode (ITL tail), and the larger activation working set plus scheduler headroom reduce the HBM left for KV blocks, cutting max concurrent sequences.

Measurement plan.
1. Establish saturation throughput per arm first with an open-loop load generator; all comparisons must be at the same *offered* rate, not the same achieved rate, otherwise a slower arm silently gets an easier test.
2. Arms: 1024 / 2048 / 4096 / 8192 batched tokens, all else identical (engine version, TP degree, KV dtype, quantisation, locked GPU clocks).
3. Replay the production trace with its real prompt-length tail; synthetic fixed-length prompts will systematically flatter large budgets.
4. Per arm record: TTFT p50/p95/p99, ITL p50/p95/p99, output tok/s, achieved prefill FLOP/s (from profiler or derived from 2*P*N over measured prefill time), max concurrent sequences, KV blocks in use, preemption and recompute counters, SM occupancy and HBM bandwidth from DCGM.
5. Repeat each arm >= 3 times in randomised order; report medians with bootstrap CIs.

Boundary condition. Returns to a larger budget saturate once the GEMMs are already tile-efficient - typically a few thousand tokens per step for a 9B-class model on a 24 GB-class GPU. Past that point you pay ITL and KV capacity for no compute gain. The knob also interacts with tensor parallelism: at TP>1 each rank sees a sharded d, so the token count needed to fill tiles rises, and the optimum shifts upward. A value tuned at TP=1 must not be copied to TP=4 without re-measuring.

Evidence required. Load generator configuration and offered-rate logs, per-arm raw latency samples, profiler traces for at least one prefill step per arm, DCGM time series, engine counters for preemption and KV occupancy.

Rollback gate. Revert to the previous budget if p99 ITL regresses more than 15%, if any preemption/recompute events appear that were absent at baseline, or if max concurrent sequences falls below the level required by the capacity plan."""

A["corpus-00075"] = """Reframed target: validate whether a prefill-aware *admission and queueing* policy helps - specifically, whether prioritising short prefills or capping concurrent long prefills improves SLO attainment versus FIFO admission.

Hypothesis. H1: an admission policy that admits at most one prompt longer than 8k tokens per scheduler step improves p99 TTFT for short requests by >= 30% while degrading p99 TTFT for long requests by <= 25%, holding throughput within 5% of baseline.

Mechanism. Prefill cost is roughly linear in N (superlinear once N^2 attention bites). Under FIFO, a single very long prompt occupies the compute budget and every short request queued behind it inherits that latency - classic head-of-line blocking, and the reason TTFT distributions in production are heavy-tailed even when the mean looks fine. Length-aware admission is shortest-job-first approximation: it reduces mean and tail waiting time for the majority at the cost of the minority of long jobs.

Measurement plan.
1. Segment the trace by prompt length before doing anything. Report metrics separately for short (< 1k), medium (1k-8k) and long (> 8k) buckets. Aggregate percentiles will hide the entire effect, since the policy deliberately trades between buckets.
2. Arms. A = FIFO baseline. B = length-aware admission with the cap above. C = B plus chunked prefill at 2048, to test whether the two mechanisms are redundant or additive.
3. Fairness instrumentation. Track per-bucket queue wait time and, critically, maximum observed wait for the long bucket. Any shortest-job-first scheme can starve long requests; measure it explicitly rather than assuming an aging term works.
4. Load: 50/75/90% of saturation, real arrival timestamps (bursty arrivals are where queueing policies matter; Poisson-smoothed replay will understate the benefit).
5. Repetitions >= 3 per arm, randomised interleaving, medians with bootstrap CIs.

Boundary condition. The policy only helps if the prompt-length distribution is genuinely heavy-tailed. If prompt lengths are near-uniform, length-aware admission reduces to FIFO plus overhead and should show no effect - which is itself a useful negative result. It also degrades under sustained overload: when offered load exceeds capacity, no scheduling policy fixes it, and the correct action is capacity or backpressure, not reordering.

Evidence required. Per-bucket latency samples and queue-wait histograms, maximum long-request wait per run, throughput per arm, scheduler decision logs sufficient to reconstruct admission order, saturation-point measurement.

Rollback gate. Revert to FIFO if any long-bucket request exceeds a hard 30 s TTFT ceiling, if long-bucket p99 TTFT degrades more than 25%, or if aggregate throughput drops more than 5% during a 24-hour canary."""

A["corpus-00076"] = """Assumptions that must be on the record before any performance claim about prefill is meaningful. A claim missing these is not falsifiable and should not be accepted.

1. Model and numerics. Parameter count and whether it is dense or MoE (for MoE, active parameters per token, not total, drive prefill FLOPs). Number of layers, hidden size, attention head count, KV head count (MHA vs GQA vs MQA - this changes KV bytes by 4-8x), weight dtype and quantisation scheme, KV cache dtype (FP16 vs FP8 halves KV bytes and changes accuracy).
2. Hardware and topology. GPU model and memory capacity/bandwidth, clock policy (boost vs locked - unlocked clocks alone produce 10-20% run-to-run drift), tensor/pipeline parallel degree, intra-node interconnect (NVLink vs PCIe), inter-node fabric (InfiniBand vs RoCE, link rate, whether GPUDirect RDMA is actually in the path), driver and CUDA versions.
3. Software. Inference engine and exact version, attention kernel (FlashAttention variant, paged attention), scheduler settings: max_num_batched_tokens, chunked prefill on/off and chunk size, prefix caching on/off, max concurrent sequences, CUDA graph usage.
4. Workload. The prompt-length distribution, not its mean - report p50, p95, p99 and max. Output length distribution. Arrival process (open vs closed loop; closed-loop clients self-throttle and understate tail latency). Concurrency or offered rate. Whether caches were warm or cold at measurement start.
5. Measurement definition. What TTFT means precisely: does it include tokenisation, queue wait, network time, and is it measured at the client or inside the engine? These differ by tens to hundreds of milliseconds and most disagreements about prefill performance are definitional, not physical.
6. Statistical framing. Number of runs, warm-up discarded, percentile estimator, confidence interval. A single run is an anecdote.

Concrete mechanism these assumptions govern. Prefill compute is about 2 * P_active * N FLOPs plus O(L * H * N^2 * d_head) attention, and KV allocation is 2 * L * N * H_kv * d_head * dtype_bytes. Every term above is a parameter of one of those two formulas, which is why omitting any of them makes a throughput or TTFT number non-reproducible.

Boundary condition. Even with all assumptions stated, results transfer only within the measured regime. A prefill efficiency figure measured at N=2048, TP=1, batch 1 does not predict N=32k (attention term dominates), TP=4 (per-rank tiles shrink, collectives appear), or high concurrency (scheduler contention dominates). Extrapolation across those boundaries must be labelled an estimate.

Evidence required. Full engine launch command, nvidia-smi -q topology and clock dump, driver/CUDA/engine versions, the trace or generator config with its hash, raw per-request latency samples, and a profiler trace for at least one representative prefill step.

Rollback gate. Reject the performance claim - and any capacity decision built on it - if the experiment cannot be re-run by a second operator from the recorded artefacts to within 10% of the reported numbers."""

A["corpus-00079"] = """Before a prefill performance claim can be defended, these assumptions must be explicit. Framing them as a checklist of what would invalidate the claim:

Configuration assumptions.
- Active parameter count (MoE: active != total), layer count, KV head count, head dim.
- Weight and KV cache dtypes, and quantisation method. FP8 KV halves cache bytes and changes both capacity and accuracy; a throughput claim without the KV dtype is unusable.
- Parallelism: TP/PP degree and placement. At TP>1, prefill includes an all-reduce per layer; on NVLink this is a few percent overhead, over PCIe it can be tens of percent. The claim is fabric-dependent.
- Attention kernel implementation, since a non-tiled kernel has O(N^2) memory and an entirely different long-context profile.

Workload assumptions.
- The full prompt-length distribution and the shared-prefix structure. If prefix caching is enabled, the reported prefill cost is the *uncached suffix* cost, and reporting it as though it were full-prompt prefill overstates efficiency by whatever the hit rate is. This is the single most common way prefill numbers are accidentally inflated.
- Whether the measurement was warm or cold, and whether cache state was reset between arms.
- Open-loop vs closed-loop load generation and the offered rate.

Environmental assumptions.
- GPU clock lock and thermal steady state; report whether the run was long enough to reach it.
- Exclusive GPU access (no other tenants, no MIG partitioning, no co-located jobs).
- ECC state and whether any GPU was throttling (check nvidia-smi throttle reasons; a single thermally capped GPU in a TP group drags the whole step).

Definitional assumptions.
- TTFT boundaries (client-side vs engine-side, inclusive of queueing or not).
- Percentile method and sample count.

Mechanism tying these together. TTFT ~= queue_wait + (2 * P_active * N_uncached / achieved_FLOPs) + attention_term(N) + collective_overhead(TP) + detokenise/network. Each assumption above pins one term. Leave a term unpinned and the number is not attributable.

Boundary condition. Claims are valid only inside the measured (N, concurrency, TP, cache-hit-rate) box. In particular, a prefill claim measured with a high prefix-cache hit rate collapses the moment the traffic mix changes to unique prompts - the same code, same hardware, and 3-5x worse TTFT. Any capacity plan derived from cached measurements must carry an explicit worst-case uncached scenario.

Evidence required. Engine launch flags, cache hit-rate time series for the measurement window, nvidia-smi throttle-reason and clock logs, topology dump, raw latency samples, and an independent re-run.

Rollback gate. Treat the claim as unproven and block the associated rollout if the uncached worst-case scenario has not been measured, or if a repeat run by another operator deviates by more than 10%."""

A["corpus-00080"] = """Assumptions required before asserting anything about prefill performance, organised by the failure mode each one prevents.

1. What is being counted. Prefill FLOPs ~= 2 * P_active * N plus attention O(L * H * N^2 * d_head). State P_active (MoE routing means total parameters overstate cost, often by 5-10x) and whether the attention term is included. Prevents: comparing a dense model's number against an MoE's and concluding the wrong thing.
2. What N actually is. Raw prompt tokens, or uncached tokens after prefix-cache matching? Prevents: reporting a cache-assisted TTFT as an intrinsic prefill capability.
3. KV cache geometry and dtype. bytes/token = 2 * L * H_kv * d_head * dtype_bytes. Prevents: capacity plans that are off by the GQA ratio or by 2x from FP8.
4. Hardware state. GPU SKU, HBM bandwidth, clocks locked or not, throttle reasons clear, MIG off, exclusive access, driver/CUDA version. Prevents: attributing a thermal or noisy-neighbour artefact to a code change.
5. Parallel topology. TP/PP degree, NVLink vs PCIe intra-node, IB vs RoCE inter-node, and whether GPUDirect RDMA is genuinely engaged rather than falling back to host bounce buffers. Prevents: a scaling claim that silently depends on a fabric the target environment lacks.
6. Scheduler configuration. Chunked prefill and chunk size, max_num_batched_tokens, max concurrent sequences, CUDA graphs. Prevents: attributing a scheduling effect to a kernel-level one.
7. Load model. Open-loop arrival process with real timestamps and burstiness, warm-up excluded, run length sufficient for steady state. Prevents: closed-loop self-throttling hiding the tail.
8. Metric definition and statistics. TTFT measurement boundary, percentile estimator, repetition count, confidence intervals. Prevents: single-run anecdotes presented as results.

Concrete mechanism worth stating explicitly. Prefill amortises weight reads over N tokens while decode re-reads all weights per token, so prefill is compute-bound and decode bandwidth-bound. Any claim that a change "made inference faster" must say which phase it moved, because optimisations for one routinely harm the other - larger prefill batches improve TTFT and worsen ITL, and FP8 KV improves decode capacity while adding quantisation error.

Boundary condition. All of the above holds within the regime where prefill kernels are tile-efficient and attention is not yet dominant, roughly N from a few hundred to a few thousand tokens for current 9B-class models. Below that range prefill behaves like decode (bandwidth-bound); above roughly 32k the N^2 term dominates and linear scaling assumptions fail. A claim is only valid inside the box it was measured in, and must be labelled as such.

Evidence required. Complete engine and hardware configuration dump, cache hit-rate series, clock/throttle logs, raw per-request samples, profiler trace of a representative prefill step, and a successful independent re-run.

Rollback gate. Do not act on the claim - and roll back any change justified by it - if an independent operator cannot reproduce the headline number within 10% from the recorded artefacts, or if the measurement regime does not cover the production prompt-length distribution's p95."""

Q = {
    "corpus-00069": ({"technical_correctness": 2, "instruction_coverage": 2, "operational_safety": 3},
        ["Source answer omits the O(N^2) attention term, so long-context TTFT is underpredicted",
         "No KV memory sizing formula; capacity planning cannot be derived",
         "'often compute utilization' is vague and not falsifiable"], 0.86),
    "corpus-00070": ({"technical_correctness": 2, "instruction_coverage": 2, "operational_safety": 3},
        ["Source answer ignores scheduler coupling: prefill degrades other requests' ITL",
         "No mention of prefix caching, which can invalidate compute-bound assumption",
         "No boundary condition given despite being requested"], 0.85),
    "corpus-00071": ({"technical_correctness": 1, "instruction_coverage": 1, "operational_safety": 2},
        ["Prompt is malformed: prefill is mandatory, not an optional feature to validate",
         "Source answer is a definition, not a measurement plan; instruction coverage is near zero",
         "Risk of teams running single-load-point benchmarks and generalising"], 0.83),
    "corpus-00072": ({"technical_correctness": 1, "instruction_coverage": 1, "operational_safety": 2},
        ["Source answer does not answer the measurement-plan instruction at all",
         "Disaggregation adds KV transfer cost that can regress latency on non-RDMA fabrics",
         "Unverified GPUDirect path can silently fall back to host bounce buffers"], 0.8),
    "corpus-00073": ({"technical_correctness": 1, "instruction_coverage": 1, "operational_safety": 2},
        ["Source answer is off-instruction",
         "Prefix caching can change outputs if block hashing is buggy; requires correctness gate",
         "Cache thrashing under load can invert measured benefit"], 0.82),
    "corpus-00074": ({"technical_correctness": 1, "instruction_coverage": 1, "operational_safety": 2},
        ["Source answer gives no plan, no metrics, no controls",
         "Raising batched-token budget shrinks KV pool and can trigger preemption in production",
         "Tuning at TP=1 does not transfer to TP>1"], 0.83),
    "corpus-00075": ({"technical_correctness": 1, "instruction_coverage": 1, "operational_safety": 2},
        ["Source answer does not address measurement",
         "Length-aware admission can starve long requests without an explicit aging/ceiling guard",
         "No scheduling policy helps under sustained overload; risk of masking a capacity shortfall"], 0.81),
    "corpus-00076": ({"technical_correctness": 1, "instruction_coverage": 1, "operational_safety": 2},
        ["Source answer lists no assumptions, which is the entire instruction",
         "Unstated TTFT measurement boundary is a common source of false performance claims",
         "Single-run numbers presented as results"], 0.85),
    "corpus-00079": ({"technical_correctness": 1, "instruction_coverage": 1, "operational_safety": 2},
        ["Source answer off-instruction",
         "Cache-warm measurements misreported as intrinsic prefill performance inflate capacity plans",
         "Thermal throttling or MIG/co-tenancy can be misattributed to code changes"], 0.84),
    "corpus-00080": ({"technical_correctness": 1, "instruction_coverage": 1, "operational_safety": 2},
        ["Source answer off-instruction",
         "MoE total-vs-active parameter confusion overstates prefill cost by 5-10x",
         "Assumed GPUDirect RDMA path may not exist in the target environment"], 0.84),
}

EV = {
    "corpus-00069": ["Profiler trace separating GEMM vs attention time during prefill",
                     "DCGM SM occupancy and HBM bandwidth during prefill steps",
                     "TTFT/ITL percentile histograms from the serving engine",
                     "Engine-reported KV cache bytes per request"],
    "corpus-00070": ["Prefix cache hit-rate time series", "KV blocks in use and preemption counters",
                     "TTFT and ITL percentiles at fixed offered rate",
                     "Controlled A/B with fixed prompt-length distribution"],
    "corpus-00071": ["Production trace hash and prompt/output length distributions",
                     "Engine version and full launch flags", "GPU clock-lock confirmation",
                     "Per-arm raw latency samples with bootstrap CIs"],
    "corpus-00072": ["perftest (ib_write_bw) bandwidth and latency between pools",
                     "Confirmation that GPUDirect RDMA path is active (nv_peer_mem/dmabuf)",
                     "Per-request KV transfer timings", "Switch ECN/PFC and RDMA retransmit counters"],
    "corpus-00073": ["Offline longest-common-prefix distribution over the trace",
                     "Engine cache hit-rate and eviction counters",
                     "Temperature-0 output diff between caching on/off",
                     "TTFT samples per arm"],
    "corpus-00074": ["Open-loop load generator config and offered-rate logs",
                     "Profiler trace of one prefill step per arm",
                     "DCGM SM/HBM time series",
                     "Engine preemption, recompute and KV occupancy counters"],
    "corpus-00075": ["Per-bucket (short/medium/long prompt) latency and queue-wait histograms",
                     "Maximum long-request wait time per run",
                     "Scheduler decision logs to reconstruct admission order",
                     "Saturation throughput measurement"],
    "corpus-00076": ["Full engine launch command and version",
                     "nvidia-smi topology, clock and throttle-reason dump",
                     "Trace/generator config hash", "Raw per-request latency samples",
                     "Independent re-run by a second operator"],
    "corpus-00079": ["Cache hit-rate series for the measurement window",
                     "nvidia-smi throttle-reason and clock logs",
                     "Topology dump including TP placement and fabric type",
                     "Independent re-run within 10%"],
    "corpus-00080": ["Complete hardware and engine configuration dump",
                     "Profiler trace of a representative prefill step",
                     "Cache hit-rate series", "Raw per-request samples",
                     "Reproduction by an independent operator"],
}

lines = open(SRC).readlines()
out = []
for i in range(START, END):
    d = json.loads(lines[i])
    sid = d["id"]
    msgs = d["messages"]
    su = [m for m in msgs if m["role"] == "user"][0]["content"]
    sa = [m for m in msgs if m["role"] == "assistant"][0]["content"]
    qd, risks, conf = Q[sid]
    rec = {
        "source_id": sid,
        "teacher_lane": "teacher-B",
        "teacher_model": "claude-opus-5-current",
        "calibration_status": "provisional",
        "decision": "rewrite",
        "source_user": su,
        "source_assistant": sa,
        "corrected_answer": A[sid],
        "quality_dimensions": qd,
        "risks": risks,
        "evidence_required": EV[sid],
        "confidence": conf,
    }
    out.append(json.dumps(rec, ensure_ascii=False))

with open(OUT, "w") as f:
    f.write("\n".join(out) + "\n")
print("WROTE", OUT, len(out))
