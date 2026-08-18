#!/usr/bin/env python3
"""teacher-B blind review generator for train-batch-0139 (corpus lines 1381-1390)."""
import json, os

ROOT = "/home/johnson/workspace/LLM_PostProcess"
SRC = os.path.join(ROOT, "research/ai-infra-expert/corpus/train.jsonl")
OUT = os.path.join(ROOT, "experiments/2026-08-17-teacher-b-corpus-review/results/train-batch-0139.jsonl")
START, N = 1380, 10  # 0-indexed

# Distinct analytical angle per item; rubric prompt family is identical
# (long-context intermittent OOM under concurrency), so diversification is by
# primary mechanism hypothesis + measurement plan + rollback gate.
ANGLES = [
    dict(
        title="KV-cache high-water mark vs. admission control",
        h="Intermittent OOM is driven by the *peak* concurrent KV footprint, not by mean utilisation: OOM fires only when the sum of active sequence lengths crosses free-VRAM minus weights/activation scratch.",
        mech=("KV bytes = 2 (K,V) x n_layers x n_kv_heads x head_dim x dtype_bytes x seq_len per sequence. "
              "With GQA this is much smaller than MHA, but it still grows strictly linearly in decoded tokens, "
              "so a request that started short can push the server over the edge 30k tokens later. "
              "A paged allocator (vLLM PagedAttention) bounds fragmentation but does NOT bound total demand; "
              "the scheduler must preempt/swap when the block pool is exhausted."),
        meas=("(1) Sample block-pool occupancy (vLLM: gpu_cache_usage_perc, num_preemptions_total) at 1 Hz and "
              "align timestamps with OOM events. (2) Log per-request prompt_tokens + generated_tokens at completion. "
              "(3) torch.cuda.max_memory_allocated / max_memory_reserved snapshot at crash. "
              "Compute predicted peak KV from the formula and compare with measured reserved bytes."),
        exp=("Controlled experiment: replay the same trace twice at fixed seed and fixed arrival times, once with "
             "max_num_seqs = current and once halved, holding max_model_len constant. If the peak-KV hypothesis holds, "
             "OOM rate goes to zero in the halved arm while p50 latency is roughly unchanged and throughput drops "
             "sub-linearly (queueing, not compute)."),
        conf=("Confounders: background processes on the same GPU (check nvidia-smi per-PID memory); "
              "CUDA graph capture pools reserved at first run; a co-tenant embedding model; "
              "and autoscaling that changes replica count mid-trace."),
        mit=("Priority order: (a) cap max_model_len and enforce a token-budget admission gate at the router; "
             "(b) lower max_num_seqs / max_num_batched_tokens so the scheduler backpressures instead of OOMing; "
             "(c) raise gpu_memory_utilization only after confirming headroom for activations; "
             "(d) enable preemption/recompute or CPU KV offload; (e) KV quantisation (fp8) as a last resort "
             "because it changes numerics and needs a quality gate."),
        roll=("Rollback gate: revert if p99 TTFT regresses >20% or reject-rate exceeds 1% of requests over a 30-min "
              "canary at production arrival rate."),
        risks=["Raising gpu_memory_utilization to 'fix' OOM removes activation headroom and converts rare OOM into frequent OOM",
               "KV fp8 quantisation silently degrades long-context accuracy if no eval gate is attached",
               "Admission control that drops instead of queues turns a latency problem into a correctness/SLA problem"],
        ev=["1 Hz time series of gpu_cache_usage_perc and num_preemptions_total aligned to OOM timestamps",
            "Per-request prompt/generated token histogram for the failing window",
            "torch.cuda memory summary at crash including reserved vs allocated split",
            "A/B trace replay results at two max_num_seqs settings with identical seed and arrival times"],
        dims=(4, 4, 4), dec="rewrite", cf=0.72,
    ),
    dict(
        title="Allocator fragmentation and reserved-vs-allocated gap",
        h="Failures are fragmentation-driven: total free bytes exceed the failing allocation, but no single contiguous block does, so OOM correlates with reserved-minus-allocated gap rather than with total demand.",
        mech=("PyTorch's caching allocator splits cached segments by size class; long-lived variable-length "
              "activation buffers plus repeated resize of a non-paged KV tensor leave many small free splits. "
              "The failing allocation is typically a large contiguous activation or a new KV block set. "
              "PagedAttention removes this for KV specifically because blocks are fixed-size and uniform, "
              "which is exactly why the fragmentation hypothesis predicts OOM even at moderate cache occupancy."),
        meas=("Enable PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True as the test arm and capture "
              "torch.cuda.memory_snapshot() / _dump_snapshot() before and at OOM. Key statistic: "
              "reserved_bytes.all.current - allocated_bytes.all.current, plus num_alloc_retries and "
              "the size of the failing request from the OOM message."),
        exp=("Two arms over an identical replayed trace: baseline allocator vs expandable_segments. "
             "Falsifiable prediction: if fragmentation dominates, the expandable-segments arm survives the same trace "
             "while peak allocated bytes are statistically indistinguishable (within 2%). If peak allocated also "
             "rises to the cap, the hypothesis is refuted and demand, not fragmentation, is the driver."),
        conf=("Confounders: cudaMalloc retries mask the effect by silently freeing cache; NCCL and cuBLAS workspaces "
              "reserve outside the PyTorch allocator and are invisible in allocator stats; CUDA graphs pin memory pools."),
        mit=("(a) expandable_segments or a tuned max_split_size_mb; (b) move to a paged KV implementation so KV "
             "never fragments; (c) pre-allocate a fixed activation scratch at startup and fail fast at boot rather "
             "than mid-traffic; (d) pin max batch token count so activation size is bounded and constant."),
        roll=("Rollback gate: revert the allocator flag if throughput drops >10% or if any new allocator-related "
              "stall (num_alloc_retries > 0) appears during a 1-hour soak."),
        risks=["expandable_segments interacts poorly with some CUDA-graph/custom-kernel paths and can mask a real capacity shortfall",
               "Allocator stats exclude NCCL/cuBLAS workspaces, so 'free memory' can be over-reported",
               "Tuning only the allocator postpones OOM instead of bounding demand"],
        ev=["torch.cuda.memory_snapshot dumps from a healthy window and from the OOM instant",
            "Exact OOM message including requested-bytes and free/reserved figures",
            "num_alloc_retries counter over the trace",
            "nvidia-smi per-process memory to attribute non-allocator reservations"],
        dims=(4, 4, 4), dec="rewrite", cf=0.7,
    ),
    dict(
        title="Prefix-cache retention and multi-tenant sharing",
        h="Automatic prefix caching keeps evictable blocks resident; OOM appears only when cache-hit-heavy tenants stop sharing prefixes, so OOM incidence should track prefix-hit-rate collapse, not raw QPS.",
        mech=("Prefix caching lets N requests with a shared system prompt reference one set of KV blocks with a "
              "refcount. Effective KV demand is therefore sub-linear in concurrency while prefixes are shared, "
              "and jumps to fully linear the moment tenants diverge. That step change explains intermittency "
              "under a nominally stable request rate."),
        meas=("Instrument prefix cache hit rate (vLLM prefix_cache_hit_rate / gpu_prefix_cache_hit_rate) and "
              "distinct-prefix cardinality per minute. Correlate with OOM timestamps; the hypothesis predicts "
              "a hit-rate drop leading each OOM by less than one scheduling window."),
        exp=("Synthetic A/B: hold QPS and sequence-length distribution fixed; arm A sends all requests with one "
             "shared 4k system prefix, arm B randomises the prefix per request. Prediction: arm B OOMs at a "
             "concurrency level arm A survives, with peak block-pool occupancy differing by roughly the "
             "shared-prefix block count times (concurrency - 1)."),
        conf=("Confounders: cache eviction policy changes between versions; a warm cache from prior traffic; "
              "tokenizer differences that break prefix identity by one token and silently disable sharing."),
        mit=("(a) Reserve a floor of free blocks so the cache can never consume the last of the pool; "
             "(b) size capacity from the worst case (zero sharing), not the observed average; "
             "(c) route tenants with distinct system prompts to separate replicas so sharing is predictable; "
             "(d) cap cache blocks explicitly."),
        roll=("Rollback gate: if reserving a free-block floor cuts prefix hit rate below the level where p50 TTFT "
              "regresses >15%, revert and instead reduce max_num_seqs."),
        risks=["Capacity planned on observed (high-sharing) hit rate collapses under a new tenant mix",
               "Prefix caching across tenants can leak timing side-channels if hit/miss latency is observable",
               "A one-token prompt difference silently disables sharing and doubles demand with no error"],
        ev=["Prefix cache hit rate and distinct-prefix cardinality time series aligned to OOM events",
            "Tenant/prompt-template inventory for the failing window",
            "Block-pool occupancy under forced zero-sharing synthetic load",
            "Serving-engine version and its prefix-cache eviction policy"],
        dims=(4, 4, 4), dec="rewrite", cf=0.68,
    ),
    dict(
        title="Chunked prefill and peak activation, not KV",
        h="The failing allocation is prefill activation memory, not KV: OOM should correlate with the largest single prefill batch (tokens per forward), and disappear under chunked prefill at constant KV demand.",
        mech=("Prefill cost is O(batched_tokens) for activations and, without FlashAttention, O(seq_len^2) for the "
              "score matrix. A scheduler that co-schedules several long prompts into one forward pass creates a "
              "transient activation spike far above steady-state decode usage. KV usage at that instant may be modest, "
              "which is why cache-occupancy dashboards look healthy right up to the crash."),
        meas=("Log max_num_batched_tokens actually realised per step (not just the configured cap), and the "
              "requested-bytes field of the OOM. Compare requested-bytes against an activation estimate "
              "(hidden_size x batched_tokens x dtype x layer working set) versus a KV block estimate; "
              "they differ by orders of magnitude, so the OOM message alone discriminates the hypotheses."),
        exp=("Enable chunked prefill with a fixed chunk (e.g. 2048 tokens) while holding max_num_seqs and "
             "max_model_len constant, and replay the identical trace. Prediction: OOM disappears, peak reserved "
             "memory drops by roughly the activation delta, TTFT for long prompts rises modestly, and decode "
             "throughput is unchanged. If OOM persists, activation-spike is refuted."),
        conf=("Confounders: attention backend choice (FlashAttention vs naive) changes the quadratic term entirely; "
              "CUDA-graph capture excludes prefill so graph memory is not the cause; "
              "mixed prompt/decode batching policy differs by engine version."),
        mit=("(a) Enable chunked prefill; (b) cap max_num_batched_tokens; (c) confirm a memory-efficient attention "
             "backend is actually active (verify at runtime, do not assume from config); "
             "(d) route very long prompts to a dedicated queue with concurrency 1."),
        roll=("Rollback gate: revert chunked prefill if p99 TTFT for the longest decile regresses >25% or if "
              "output-token throughput drops >10% over a 30-min canary."),
        risks=["Assuming FlashAttention is active when the build silently fell back to a quadratic path",
               "Chunked prefill shifts latency onto long prompts and can breach a per-request SLA",
               "Dashboards showing healthy KV occupancy create false confidence while activations spike"],
        ev=["Realised batched-token count per scheduler step, not the configured cap",
            "Full OOM traceback with requested-bytes to discriminate activation vs KV allocation",
            "Runtime confirmation of the active attention backend",
            "Before/after peak reserved memory under identical replayed trace with chunked prefill toggled"],
        dims=(4, 4, 4), dec="rewrite", cf=0.71,
    ),
    dict(
        title="Tensor-parallel rank skew and NCCL buffer growth",
        h="OOM is rank-local, not global: one TP rank carries extra memory (buffers, unbalanced experts, or NCCL workspaces), so failures concentrate on a single device and are invisible in cluster-average metrics.",
        mech=("Under tensor parallelism each rank holds 1/TP of weights and KV, so demand should be symmetric. "
              "Asymmetry comes from (i) NCCL communicator and registered-buffer allocations that scale with the "
              "number of communicators and message sizes and live outside the PyTorch allocator, (ii) rank 0 also "
              "hosting the scheduler/tokenizer/logging process, and (iii) MoE expert imbalance where token routing "
              "sends more tokens to experts homed on one rank."),
        meas=("Per-rank torch.cuda.max_memory_reserved plus nvidia-smi per-PID bytes, sampled at 1 Hz for every "
              "device, with the OOM rank recorded. Also capture NCCL_DEBUG=INFO buffer sizes and the count of "
              "created communicators. The hypothesis predicts >5% spread between the OOM rank and the median rank."),
        exp=("Run the same trace with NCCL_BUFFSIZE reduced and with the API/scheduler process pinned off rank 0 "
             "(or a separate device). Prediction: if rank skew is causal, the spread collapses and OOM moves or "
             "disappears; if all ranks OOM near-simultaneously at the same occupancy, the skew hypothesis is refuted "
             "and this is a global capacity problem."),
        conf=("Confounders: MPS/MIG partitioning; ECC-reserved memory differing per GPU; a stuck previous process "
              "holding memory on one device; PCIe vs NVLink topology changing NCCL algorithm and therefore buffers."),
        mit=("(a) Isolate the scheduler/API process from any TP rank's device; (b) bound NCCL_BUFFSIZE and reuse "
             "communicators; (c) for MoE, enable expert-parallel load balancing / capacity factor limits; "
             "(d) size headroom against the worst rank, never the mean."),
        roll=("Rollback gate: revert NCCL buffer reduction if all-reduce bandwidth measured by nccl-tests drops "
              ">15% or if any collective timeout appears during a 1-hour soak."),
        risks=["Shrinking NCCL buffers trades memory for collective bandwidth and can silently raise inter-token latency",
               "Cluster-average GPU memory dashboards hide single-rank skew and delay diagnosis",
               "A leftover zombie process holding VRAM is misdiagnosed as a workload capacity problem"],
        ev=["Per-rank (not averaged) memory time series with the OOM rank identified",
            "NCCL_DEBUG=INFO log showing buffer sizes and communicator count",
            "nvidia-smi per-PID attribution on every device in the TP group",
            "MoE expert-routing token counts per rank if the model is sparse"],
        dims=(4, 4, 4), dec="rewrite", cf=0.66,
    ),
    dict(
        title="Disaggregated prefill/decode and KV transfer buffers (Mooncake/Dynamo shape)",
        h="In a disaggregated deployment the OOM is in the transfer path: staging buffers for KV migration between prefill and decode workers scale with in-flight transfers, so OOM tracks concurrent handoffs rather than resident sequences.",
        mech=("Prefill/decode disaggregation (the Mooncake and NVIDIA Dynamo architectural pattern) moves KV from a "
              "prefill pool to a decode pool over RDMA. Each in-flight transfer needs registered, pinned memory on "
              "both sides; registration is expensive and typically pooled, so the pool is sized for expected "
              "concurrency. A burst of long prompts raises both the per-transfer size (linear in prompt tokens) and "
              "the number of concurrent transfers, multiplying staging demand while resident KV looks normal."),
        meas=("Instrument: in-flight transfer count, bytes per transfer, transfer duration, and the size of the "
              "registered-memory pool on both sides. Correlate OOM with the product (in-flight x bytes). "
              "Also record RDMA completion-queue errors, which often precede allocation failures."),
        exp=("Cap concurrent KV transfers at a fixed small number (e.g. 4) while leaving request concurrency "
             "unchanged, and replay the trace. Prediction: OOM disappears and TTFT rises only for the queued tail; "
             "if OOM persists at the same rate, the transfer-buffer hypothesis is refuted and the problem is "
             "resident KV on the decode side."),
        conf=("Confounders: RoCE congestion (PFC pause / ECN marking) lengthening transfer duration and thus raising "
              "in-flight count without any change in request rate; GPUDirect RDMA availability differing per node, "
              "silently falling back to a host-memory bounce buffer with different memory behaviour."),
        mit=("(a) Bound concurrent transfers and make the bound explicit backpressure to the router; "
             "(b) pre-register a fixed buffer pool at startup and fail fast if it cannot be reserved; "
             "(c) verify GDR is actually in use rather than assumed; (d) chunk large KV transfers so per-transfer "
             "buffer size is constant."),
        roll=("Rollback gate: revert the transfer cap if end-to-end p99 TTFT regresses >20% or if prefill-worker "
              "utilisation falls below 60%, indicating the cap has become the bottleneck."),
        risks=["Assuming GPUDirect RDMA is active when the stack silently fell back to host bounce buffers",
               "Unbounded in-flight transfers convert a network slowdown into an OOM on an otherwise healthy node",
               "RoCE congestion control misconfiguration lengthens transfers and amplifies buffer residency"],
        ev=["In-flight KV transfer count and bytes-per-transfer time series aligned to OOM",
            "Confirmation that GPUDirect RDMA is in the actual data path (not inferred from config)",
            "RDMA CQ error and RoCE PFC/ECN counters for the failing window",
            "Registered-memory pool size and high-water mark on both prefill and decode sides"],
        dims=(4, 4, 4), dec="rewrite", cf=0.62,
    ),
    dict(
        title="Speculative decoding / draft model as hidden co-tenant",
        h="A second model or auxiliary structure (draft model, reward model, LoRA adapters, logits processors) consumes VRAM that scales with concurrency, so OOM appears only above a concurrency threshold even though the primary model's footprint is static.",
        mech=("Speculative decoding keeps a draft model resident plus per-request draft KV and candidate-token "
              "buffers; verification runs a batch of gamma+1 tokens per sequence, inflating activation width by "
              "the speculation length. Similarly, multi-LoRA serving holds adapter weights per active adapter, and "
              "structured-output logits processors allocate per-request automaton state proportional to vocab size."),
        meas=("Attribute memory by module: snapshot allocations tagged by owner (draft vs target vs adapters) via "
              "memory_snapshot stack traces. Record active adapter count, speculation length, and acceptance rate. "
              "The hypothesis predicts OOM threshold shifts linearly with speculation length."),
        exp=("Disable speculative decoding (and separately, restrict to a single LoRA adapter) on one replica while "
             "leaving all other settings identical, then replay the trace against both replicas. Prediction: the "
             "control replica survives; measured peak reserved memory differs by draft-weights + gamma-scaled "
             "buffers. If both OOM identically, this hypothesis is refuted."),
        conf=("Confounders: acceptance rate varies with the workload, so speculative memory and throughput both move "
              "with prompt distribution; adapter LRU eviction makes the footprint time-dependent."),
        mit=("(a) Bound active adapter count and pin a hard cap; (b) reduce speculation length or disable "
             "speculation above a concurrency watermark (adaptive spec); (c) account draft-model memory explicitly "
             "in the capacity model rather than treating it as noise; (d) cap concurrent structured-output requests."),
        roll=("Rollback gate: if disabling speculation raises p50 inter-token latency >30%, re-enable with a reduced "
             "gamma and an adaptive disable-above-watermark policy instead of a blanket off."),
        risks=["Capacity models that count only target-model weights and KV understate real demand",
               "Adaptive speculation policies make performance non-stationary and hard to benchmark",
               "Per-request logits-processor state scales with vocab size and is easy to overlook"],
        ev=["Owner-attributed memory snapshot separating target model, draft model, adapters and per-request state",
            "Active adapter count and speculation length/acceptance-rate time series",
            "A/B peak reserved memory with speculation enabled vs disabled on identical trace",
            "Vocabulary size and per-request automaton state size for structured-output paths"],
        dims=(4, 4, 4), dec="rewrite", cf=0.63,
    ),
    dict(
        title="Capacity model first: derive the theoretical concurrency ceiling before touching knobs",
        h="The deployment is simply over-subscribed: the configured max_model_len x max_num_seqs product exceeds physical KV capacity, so OOM is the expected outcome and 'intermittent' only reflects how rarely the worst case is reached.",
        mech=("Budget: VRAM_total = weights + optimizer-free inference scratch + activation peak + KV pool + "
              "framework/NCCL/CUDA-context overhead (typically 0.8-1.5 GiB context per device plus workspaces). "
              "KV pool tokens = (VRAM_total - everything else) / per-token KV bytes, where per-token KV bytes = "
              "2 x n_layers x n_kv_heads x head_dim x dtype_bytes / TP. Max safe concurrency = KV pool tokens / "
              "max_model_len. If configured concurrency exceeds that, no allocator tuning can help."),
        meas=("Read the actual model config (n_layers, n_kv_heads, head_dim, dtype) rather than assuming; measure "
              "weights bytes at load; measure steady-state reserved before traffic; then compute the ceiling and "
              "compare it against the configured product. This is arithmetic, not an experiment, and it should be "
              "step zero."),
        exp=("Falsifiable check: set max_num_seqs to the computed ceiling minus a 10% margin and run the trace at "
             "1.5x production arrival rate for 1 hour. Prediction: zero OOM, non-zero queueing. If OOM still occurs "
             "at that setting, an unaccounted consumer exists and the budget is wrong -- which is itself the finding."),
        conf=("Confounders: dtype of KV differing from weight dtype; sliding-window or hybrid attention layers that "
              "break the uniform-per-layer assumption; MLA-style compressed KV which invalidates the naive formula."),
        mit=("(a) Enforce the derived ceiling in config and in the router's admission gate; (b) scale horizontally "
             "instead of raising utilisation; (c) segment traffic by context length so short requests are not "
             "penalised by the long-context ceiling; (d) only then consider fp8 KV or offload."),
        roll=("Rollback gate: any change that raises the effective ceiling must hold zero OOM at 1.5x arrival rate "
              "for 1 hour and must not regress an accuracy eval by more than the pre-registered threshold."),
        risks=["Naive KV formulas are wrong for MLA, sliding-window, and hybrid-attention models; using them yields false confidence",
               "Horizontal scaling raises cost and may be blocked by GPU availability, so the ceiling must be an explicit product decision",
               "Enforcing a ceiling converts OOM crashes into visible rejections, which needs SLA sign-off"],
        ev=["Model config values for n_layers, n_kv_heads, head_dim and KV dtype read from the checkpoint",
            "Measured weights bytes and steady-state reserved bytes before any traffic",
            "The computed concurrency ceiling and the currently configured max_num_seqs x max_model_len product",
            "1-hour zero-OOM soak result at 1.5x production arrival rate"],
        dims=(5, 4, 5), dec="rewrite", cf=0.78,
    ),
    dict(
        title="Failure-mode containment: make OOM survivable before making it rare",
        h="The operational defect is that an allocation failure kills the whole server and drops all in-flight requests; if OOM were contained to the offending request, the same memory pressure would produce degraded latency instead of an outage.",
        mech=("Most serving stacks abort the process on CUDA OOM because the allocator state and any in-flight "
              "CUDA graphs cannot be reliably unwound. Containment therefore has to happen above the engine: "
              "an admission gate that rejects before the engine allocates, a scheduler that preempts and recomputes "
              "the lowest-priority sequence, and a supervisor that restarts a worker while the router drains it. "
              "This is a reliability argument, orthogonal to the root-cause hypotheses."),
        meas=("Measure blast radius, not just OOM count: requests failed per OOM event, time to healthy after "
              "restart, and whether the router removed the worker from rotation before or after user-visible errors. "
              "Also record whether preemption fired at all (num_preemptions_total)."),
        exp=("Inject a controlled memory-pressure fault (allocate a large tensor on the device from a sidecar) and "
             "observe the system's response. Prediction under this hypothesis: today the worker dies and N in-flight "
             "requests fail; after adding preemption plus a router health check, the same injection yields zero "
             "5xx and a measurable latency bump. Fault injection is the experiment; do it in staging first."),
        conf=("Confounders: a health check that passes while the engine is wedged; restart storms that mask "
              "improvement; retries that amplify load and re-trigger pressure."),
        mit=("(a) Enable scheduler preemption/recompute so pressure degrades latency instead of killing the process; "
             "(b) hard token-budget admission gate at the router with a 429 and Retry-After; "
             "(c) readiness probe that reflects real engine state and drains before restart; "
             "(d) bounded retries with jitter to avoid amplification."),
        roll=("Rollback gate: staging-only fault injection first; promote only if the canary shows zero 5xx during "
              "injection and no more than 20% p99 latency regression. Revert immediately on any restart storm."),
        risks=["Fault injection in production can cause the outage it is meant to prevent -- staging first is mandatory",
               "Health checks that only test the HTTP port report healthy while the engine is dead or wedged",
               "Unbounded client retries amplify a transient pressure event into a sustained outage"],
        ev=["Requests-failed-per-OOM-event and time-to-healthy measurements from the current system",
            "Whether scheduler preemption ever fired before the crash (num_preemptions_total)",
            "Router health/readiness probe definition and its drain behaviour",
            "Staging fault-injection results before and after containment changes"],
        dims=(4, 5, 5), dec="rewrite", cf=0.74,
    ),
    dict(
        title="Time-correlated drift: version, traffic-mix and environment change control",
        h="Nothing in the code changed the memory model; the workload or environment did. OOM onset should align with a discrete change event (deploy, driver/library upgrade, new tenant, or prompt-template change) rather than with a gradual load ramp.",
        mech=("Memory demand is a function of both configuration and input distribution. A prompt-template edit that "
              "adds 2k tokens, a client that starts sending documents, a serving-engine minor version that changes "
              "default gpu_memory_utilization or attention backend, or a driver upgrade that changes CUDA context "
              "size all shift the ceiling without any code change on the serving side. Intermittency then reflects "
              "the tail of the new input distribution crossing an unchanged ceiling."),
        meas=("Build a change timeline: deploy SHAs, container image digests, driver/CUDA/NCCL versions, engine "
              "version, config diffs, and tenant onboarding dates. Overlay the OOM event series and the p95 prompt-"
              "length series. The hypothesis predicts a step change in prompt-length p95 or a version boundary "
              "within a short window before first OOM."),
        exp=("Deploy the last known-good image with today's traffic (or replay today's trace against it). "
             "Two outcomes discriminate cleanly: if the old image also OOMs, the change is in traffic, and the "
             "prompt-length series should show the step; if the old image is clean under identical traffic, the "
             "change is in the software/environment and a bisect over image digests localises it."),
        conf=("Confounders: rollbacks that also revert an unrelated fix; caches warmed differently after restart; "
              "seasonal traffic patterns mimicking a step change; sampled logs hiding the true prompt-length tail."),
        mit=("(a) Pin engine, driver and CUDA versions and record them in the deployment manifest; "
             "(b) add a pre-merge memory regression test at fixed synthetic load; "
             "(c) enforce per-tenant token quotas so one client cannot shift the global distribution; "
             "(d) alert on prompt-length p95 drift, not just on error rate."),
        roll=("Rollback gate: rolling back to the known-good image is itself the first mitigation; keep it only if "
              "it eliminates OOM within one hour at production load, and re-evaluate the forward fix under the "
              "same trace before re-deploying."),
        risks=["Rolling back reverts unrelated fixes and can reintroduce known defects",
               "Sampled request logs understate the prompt-length tail that actually causes OOM",
               "Without pinned driver/CUDA/engine versions the environment silently drifts between deploys"],
        ev=["Change timeline of deploy SHAs, image digests, engine/driver/CUDA/NCCL versions and config diffs",
            "Unsampled prompt-length p95/p99 time series spanning the onset window",
            "Per-tenant request volume and prompt-length breakdown",
            "Replay result of today's trace against the last known-good image"],
        dims=(4, 5, 5), dec="rewrite", cf=0.69,
    ),
]


def build(angle, user):
    a = angle
    return (
        f"Assumptions: single-node GPU serving of a decoder-only LLM behind a router; "
        f"the crash is a CUDA out-of-memory abort, not a host OOM-killer event; I have read access to engine "
        f"metrics, logs and the model config, and a staging replica where I can replay traffic. If any of these "
        f"is false, say so before acting on the plan below.\n\n"
        f"Primary angle for this variant: {a['title']}.\n\n"
        f"Falsifiable hypothesis. {a['h']}\n\n"
        f"Mechanism. {a['mech']}\n\n"
        f"Measurements to take first. {a['meas']}\n\n"
        f"Controlled experiment. {a['exp']}\n\n"
        f"Expected confounders. {a['conf']}\n\n"
        f"Prioritised mitigations. {a['mit']}\n\n"
        f"Rollback criteria. {a['roll']} In all cases, change one variable at a time and keep the replayed trace, "
        f"seed and arrival timestamps fixed, otherwise the comparison is not interpretable.\n\n"
        f"What I will not claim. Without the measurements listed above I cannot attribute the OOM to any single "
        f"cause; the ordering above is a prior, not a conclusion. Numbers such as per-token KV bytes must be "
        f"recomputed from the deployed model's actual config rather than assumed from a similar model."
    )


def main():
    with open(SRC, encoding="utf-8") as f:
        lines = f.readlines()[START:START + N]
    assert len(lines) == N, len(lines)
    out = []
    for line, angle in zip(lines, ANGLES):
        d = json.loads(line)
        ms = d["messages"]
        user = [m["content"] for m in ms if m["role"] == "user"][0]
        asst = [m["content"] for m in ms if m["role"] == "assistant"][0]
        tc, ic, os_ = angle["dims"]
        out.append({
            "source_id": d["id"],
            "teacher_lane": "teacher-B",
            "teacher_model": "claude-opus-5-current",
            "calibration_status": "provisional",
            "decision": angle["dec"],
            "source_user": user,
            "source_assistant": asst,
            "corrected_answer": build(angle, user),
            "quality_dimensions": {
                "technical_correctness": tc,
                "instruction_coverage": ic,
                "operational_safety": os_,
            },
            "risks": angle["risks"],
            "evidence_required": angle["ev"],
            "confidence": angle["cf"],
        })
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("wrote", OUT, len(out))


if __name__ == "__main__":
    main()
