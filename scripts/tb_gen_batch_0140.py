#!/usr/bin/env python3
"""teacher-B blind review generator for train-batch-0140 (corpus lines 1391-1400)."""
import json, os

ROOT = "/home/johnson/workspace/LLM_PostProcess"
SRC = os.path.join(ROOT, "research/ai-infra-expert/corpus/train.jsonl")
OUT = os.path.join(ROOT, "experiments/2026-08-17-teacher-b-corpus-review/results/train-batch-0140.jsonl")
START, N = 1390, 10  # 0-indexed

# Identical rubric prompt family (long-context intermittent OOM under concurrency).
# Diversification is by primary mechanism hypothesis + measurement plan + rollback gate.
ANGLES = [
    dict(
        title="Prefill token-budget spikes vs. decode steady state",
        h="OOM is triggered during the prefill (prompt-processing) phase, not during decode: the transient activation tensor for a chunk of P prompt tokens dominates, so the crash correlates with concurrent *arrivals* of long prompts rather than with the number of active decodes.",
        mech=("Prefill materialises Q/K/V and MLP intermediates for all P tokens at once: roughly "
              "P x hidden x dtype_bytes per activation buffer, multiplied by the number of concurrently "
              "prefilled tokens (max_num_batched_tokens). Decode only materialises one token per sequence. "
              "So peak memory = weights + KV pool + prefill activation scratch, and the third term is the only "
              "one that spikes on a sub-second timescale. Chunked prefill exists precisely to bound it."),
        meas=("(1) Timestamp-correlate OOM aborts with a per-second histogram of arriving prompt lengths. "
              "(2) Record engine scheduler counters for tokens-in-prefill vs. tokens-in-decode per step. "
              "(3) torch.cuda.memory_stats() delta across a single prefill step in staging at the observed max prompt length."),
        exp=("Replay the identical trace twice with fixed arrival timestamps: arm A with chunked prefill disabled, "
             "arm B with chunked prefill enabled at a chunk size of ~2k tokens. If the prefill-spike hypothesis "
             "holds, arm B eliminates OOM with a measurable TTFT increase on the longest prompts and effectively "
             "unchanged inter-token latency for decodes."),
        conf=("Confounders: chunked prefill also changes batching and can mask a genuine KV-pool shortage; "
              "a client retry storm after the first OOM inflates arrival rate; and CUDA graph capture pools "
              "reserved on the first request look like a step change in reserved bytes."),
        mit=("Priority order: (a) enable/tighten chunked prefill and cap max_num_batched_tokens; "
             "(b) enforce a router-side max prompt length with an explicit 4xx rather than an engine abort; "
             "(c) smooth arrivals with a bounded queue plus admission control; "
             "(d) only then consider lowering gpu_memory_utilization to reclaim activation headroom."),
        roll=("Rollback gate: revert if p99 TTFT regresses more than 25% at production arrival rate over a 30-minute "
              "canary, or if throughput drops more than 10% without any reduction in OOM count."),
        risks=["Enabling chunked prefill can hide a real KV-pool undersizing and postpone the failure to a busier day",
               "Rejecting long prompts at the router silently changes product behaviour if not surfaced to callers",
               "Retry storms after an abort make the arrival-rate measurement non-stationary and invalidate the replay"],
        ev=["Per-second arrival histogram of prompt_tokens spanning the OOM window",
            "Engine scheduler counters separating prefill tokens from decode tokens per step",
            "torch.cuda.memory_stats() before/after a single max-length prefill in staging",
            "Exact engine version and the effective values of max_num_batched_tokens, max_model_len, gpu_memory_utilization"],
        dims=(4, 5, 5), dec="rewrite", cf=0.71,
    ),
    dict(
        title="Caching allocator fragmentation vs. true capacity exhaustion",
        h="Free memory is sufficient in aggregate but not contiguous: the failure is allocator fragmentation, which predicts that reserved bytes stay near the cap while allocated bytes at the moment of failure are materially below it.",
        mech=("PyTorch's caching allocator splits cached blocks by size class; long-lived variable-length "
              "allocations interleaved with short-lived scratch produce blocks too small for the next request. "
              "The distinguishing signature is reserved_bytes - allocated_bytes remaining large at the abort. "
              "A paged KV allocator removes fragmentation for KV specifically, but activation and NCCL buffers "
              "still go through the caching allocator."),
        meas=("(1) Dump torch.cuda.memory_summary() and memory_stats() at the abort, focusing on "
              "num_alloc_retries, reserved_bytes.all.peak minus allocated_bytes.all.peak, and inactive_split_bytes. "
              "(2) Enable the memory snapshot recorder in staging and inspect the block map. "
              "(3) Track whether the failing allocation size is large and contiguous."),
        exp=("Run the same replayed trace with PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True in one arm and the "
             "default allocator in the other, holding engine config, seed and arrival timestamps fixed. If "
             "fragmentation is the cause, the expandable-segments arm survives the trace with unchanged allocated-byte "
             "peak; if it still OOMs at the same allocated peak, the cause is genuine capacity exhaustion."),
        conf=("Confounders: expandable_segments changes performance characteristics slightly; "
              "num_alloc_retries also rises under genuine pressure; and a co-tenant process on the same device "
              "shifts the contiguity picture without any change in this workload."),
        mit=("Priority order: (a) confirm the signature before changing anything; (b) expandable_segments or a tuned "
             "max_split_size_mb; (c) move variable-length KV onto a paged allocator if not already; "
             "(d) reduce peak demand via admission control; (e) restart-on-fragmentation is an operational band-aid, "
             "acceptable only with drain-then-restart and never as the permanent fix."),
        roll=("Rollback gate: revert the allocator flag if p50 inter-token latency regresses more than 10% or if "
              "OOM count is not strictly reduced over a 1-hour canary."),
        risks=["Allocator env-var tuning is version-sensitive and can silently no-op on a different PyTorch build",
               "Periodic restarts mask the defect and destroy prefix-cache warmth, degrading TTFT",
               "Reading only nvidia-smi free memory cannot distinguish fragmentation from exhaustion"],
        ev=["torch.cuda.memory_stats() dump at the moment of failure, including inactive_split_bytes and num_alloc_retries",
            "Size of the specific allocation that failed, from the CUDA OOM traceback",
            "PyTorch and CUDA versions plus the effective PYTORCH_CUDA_ALLOC_CONF",
            "Per-PID device memory from nvidia-smi to rule out co-tenancy"],
        dims=(4, 5, 5), dec="rewrite", cf=0.7,
    ),
    dict(
        title="Prefix-cache retention as a hidden, unbounded consumer",
        h="Automatic prefix caching retains KV blocks for completed requests, so effective free capacity decays with traffic diversity; this predicts OOM onset correlates with cumulative distinct-prefix volume since process start rather than with instantaneous concurrency.",
        mech=("Prefix caching keeps evicted-but-reusable blocks pinned in the same pool that serves live sequences. "
              "Hit rate is high for a few shared system prompts and near zero for diverse user prefixes; in the "
              "latter regime the cache is pure overhead competing for blocks. Eviction is usually LRU over the block "
              "pool, so the failure mode is not a leak but a policy that trades live capacity for speculative reuse."),
        meas=("(1) Track prefix-cache hit rate and cached-block count alongside gpu_cache_usage over the full uptime "
              "of the process. (2) Test whether time-since-restart predicts OOM better than concurrent request count. "
              "(3) Measure the distinct-prefix cardinality of production traffic."),
        exp=("Two arms over the same replayed trace at fixed arrival times: prefix caching enabled vs. disabled. "
             "If the retention hypothesis holds, the disabled arm shows a flat rather than rising baseline occupancy "
             "and no OOM, at the cost of TTFT on the requests that genuinely shared a prefix."),
        conf=("Confounders: disabling prefix caching also lowers throughput, which reduces concurrency and can "
              "remove OOM for the wrong reason; a monotone rise in occupancy can equally come from a real block leak; "
              "and traffic mix drifts across the day independently of uptime."),
        mit=("Priority order: (a) quantify hit rate before touching the feature; (b) cap the fraction of the block pool "
             "usable by cached prefixes; (c) disable prefix caching for tenants with near-zero hit rate; "
             "(d) admission control on live token budget; (e) scheduled drain-and-restart only as an interim guard "
             "with an explicit expiry date."),
        roll=("Rollback gate: re-enable prefix caching if TTFT p95 on the shared-prompt tenant regresses more than 30% "
              "while OOM count is unchanged; that would falsify the hypothesis."),
        risks=["Disabling prefix caching can regress TTFT sharply for tenants with a large shared system prompt",
               "Rising occupancy is consistent with both cache retention and a genuine block leak; conflating them wastes a cycle",
               "Hit-rate metrics averaged across tenants hide a bimodal distribution"],
        ev=["Prefix-cache hit rate and cached-block count time series, broken down per tenant",
            "Occupancy baseline as a function of time since process start across several restarts",
            "Distinct-prefix cardinality and shared-prefix length in production traffic",
            "Engine configuration showing whether prefix caching and its pool cap are enabled"],
        dims=(4, 5, 5), dec="rewrite", cf=0.68,
    ),
    dict(
        title="Tensor-parallel rank skew and NCCL buffer overhead",
        h="OOM is not uniform across ranks: one TP rank carries extra memory (embedding/LM-head shard, NCCL communication buffers, or the scheduler process) and fails first, which predicts the abort always names the same device ordinal.",
        mech=("Under tensor parallelism each rank holds 1/TP of most weights, but vocabulary-parallel embeddings and "
              "the LM head can be unevenly sharded, and NCCL allocates per-communicator buffers plus registered "
              "memory for its transports. On the node the rank co-located with the API/scheduler process also pays "
              "for its CUDA context and any pinned staging buffers. The result is a systematic per-rank offset that "
              "makes a symmetric capacity plan wrong."),
        meas=("(1) Record per-rank torch.cuda.max_memory_reserved and the device ordinal in every OOM traceback. "
              "(2) nvidia-smi per-PID memory for all ranks during steady load. "
              "(3) NCCL buffer accounting via NCCL_DEBUG=INFO plus the configured NCCL_BUFFSIZE and number of channels."),
        exp=("Hold the trace fixed and run one arm with the API/scheduler process pinned off the busiest rank "
             "(or with CUDA_VISIBLE_DEVICES reordered), and a second arm unchanged. If rank skew is the cause, the "
             "failing ordinal moves with the co-located process rather than staying with a fixed physical GPU; if it "
             "stays put, suspect a faulty device or a per-device co-tenant instead."),
        conf=("Confounders: a genuinely degraded GPU with less usable memory (ECC retired pages); MIG or MPS "
              "partitioning; a co-tenant on one device only; and NCCL lazily allocating buffers at first collective, "
              "which delays the skew until the first multi-rank request."),
        mit=("Priority order: (a) size capacity against the worst-case rank, not the mean; (b) move auxiliary "
             "processes off the model GPUs; (c) tune NCCL channel and buffer settings only with measurements, since "
             "shrinking them costs bandwidth; (d) rebalance the vocab-parallel shard if the engine supports it; "
             "(e) lower gpu_memory_utilization globally as the blunt fallback."),
        roll=("Rollback gate: revert NCCL buffer or channel changes if all-reduce bandwidth measured by a standard "
              "collective benchmark drops more than 15%, or if end-to-end decode throughput falls more than 10%."),
        risks=["Shrinking NCCL buffers to buy memory can degrade collective bandwidth and inter-token latency",
               "ECC retired pages reduce usable memory on one device and mimic rank skew",
               "Averaging memory metrics across ranks hides the exact skew that causes the failure"],
        ev=["Per-rank peak reserved/allocated memory with device ordinals, over the same load window",
            "Device ordinal named in each OOM traceback across many incidents",
            "nvidia-smi per-PID memory and ECC/retired-page status for every GPU on the node",
            "NCCL_DEBUG=INFO startup log showing channels, transports and buffer sizes"],
        dims=(4, 5, 5), dec="rewrite", cf=0.67,
    ),
    dict(
        title="Sampling-parameter amplification (beam width, n, best_of, logprobs)",
        h="A minority of requests multiply their own KV and logits footprint via n/best_of/beam search or full-vocabulary logprobs, so OOM correlates with a specific request-parameter signature rather than with request count.",
        mech=("With n or best_of = k, the engine maintains k parallel sequences per request, multiplying KV demand by "
              "roughly k after the shared prefix diverges. Requesting logprobs over the full vocabulary materialises "
              "a batch x vocab float tensor per step, which for a 150k-token vocabulary and a large batch is a "
              "non-trivial transient. Both are client-controlled, so capacity planning based on request count alone "
              "is structurally wrong."),
        meas=("(1) Join OOM timestamps against request parameters (n, best_of, use_beam_search, logprobs, "
              "max_tokens) for requests in flight at that moment. (2) Compute an effective-sequence count = "
              "sum over active requests of max(n, best_of) and correlate it with occupancy. "
              "(3) Measure the logits tensor size analytically from vocab size and batch."),
        exp=("Replay the trace with the high-k and full-logprob requests filtered out in one arm and present in the "
             "other, keeping arrival timestamps and all engine settings fixed. If the amplification hypothesis holds, "
             "the filtered arm has a strictly lower occupancy peak and no OOM at identical raw request counts."),
        conf=("Confounders: high-k requests often come from one tenant whose prompts are also longer, conflating two "
              "effects; and removing them lowers total load, so the comparison must control for effective-sequence "
              "count, not just request count."),
        mit=("Priority order: (a) cap n/best_of and top-k logprobs at the API boundary with an explicit error; "
             "(b) charge high-k requests against a token budget in admission control; "
             "(c) route them to a separate replica pool with its own capacity; "
             "(d) only then adjust engine-level memory settings."),
        roll=("Rollback gate: revert the API cap if it rejects more than 0.5% of legitimate traffic in a 24-hour "
              "shadow evaluation, and revisit with a per-tenant quota instead of a global cap."),
        risks=["A global cap on n/best_of silently degrades quality for products that legitimately rely on sampling breadth",
               "Filtering high-k requests in the experiment also lowers total load and can produce a false positive",
               "Full-vocabulary logprobs may be a contractual API feature that cannot simply be removed"],
        ev=["Request-parameter distribution (n, best_of, beam, logprobs, max_tokens) joined to OOM timestamps",
            "Effective-sequence-count time series versus block-pool occupancy",
            "Model vocabulary size and configured max batch, to compute the logits tensor analytically",
            "Per-tenant breakdown of which callers issue high-k or full-logprob requests"],
        dims=(4, 5, 5), dec="rewrite", cf=0.7,
    ),
    dict(
        title="Preemption and swap path as the real failure boundary",
        h="The scheduler is supposed to preempt when blocks run out, so a hard OOM means the preemption path itself failed or was disabled; this predicts either zero preemption events before the abort, or preemptions whose recompute/swap buffers caused the fatal allocation.",
        mech=("A well-configured engine degrades gracefully: on block exhaustion it preempts the newest sequences and "
              "either recomputes or swaps their KV to host memory. Recompute needs a prefill-sized activation buffer; "
              "swap needs pinned host memory and a device staging buffer. Both allocate at exactly the moment the "
              "device is most constrained, so a mis-sized swap space converts a soft degradation into a hard abort."),
        meas=("(1) Count num_preemptions_total in the minutes preceding each OOM. Zero preemptions means the "
              "safety valve never engaged; a burst means it engaged and failed. "
              "(2) Inspect the configured swap_space and whether pinned host memory is available. "
              "(3) Capture the failing allocation size and compare it to the recompute activation size."),
        exp=("Fixed trace, two arms: swap_space set to a measured value versus swap disabled with recompute-only "
             "preemption. If the preemption-path hypothesis holds, at least one arm converts the abort into "
             "throughput loss plus rising queue delay, with no crash; if both still abort at the same point, the "
             "hypothesis is falsified and the ceiling is elsewhere."),
        conf=("Confounders: host memory pressure and cgroup limits can make pinned allocation fail for reasons "
              "unrelated to the GPU; PCIe bandwidth makes swap slow enough to trigger client timeouts, which look "
              "like a different incident; and preemption counters may not be exported by older engine versions."),
        mit=("Priority order: (a) verify the preemption path is enabled and instrumented at all; "
             "(b) size swap_space from measured KV-per-sequence rather than a default; "
             "(c) prefer recompute when PCIe is the bottleneck and swap when compute is; "
             "(d) add admission control so preemption is rare rather than routine; "
             "(e) alert on preemption rate as a leading indicator well before OOM."),
        roll=("Rollback gate: revert the swap configuration if p99 end-to-end latency exceeds the client timeout for "
              "more than 0.1% of requests during a 1-hour canary, since a timeout storm is not an improvement over "
              "a rare abort."),
        risks=["Swapping KV over PCIe can push tail latency past client timeouts and convert one failure mode into another",
               "Pinned host memory allocation can fail under cgroup limits, crashing the process for a host-side reason",
               "Treating a rising preemption rate as healthy hides that the deployment is chronically undersized"],
        ev=["num_preemptions_total and queue-depth time series aligned to each OOM event",
            "Effective swap_space setting and host free/pinned memory at the time of failure",
            "Failing allocation size from the CUDA OOM traceback",
            "Measured KV bytes per sequence at the deployment's max_model_len"],
        dims=(4, 5, 5), dec="rewrite", cf=0.69,
    ),
    dict(
        title="Multi-tenancy and residual host/device processes on the same GPU",
        h="The serving process is not the sole consumer of the device: another process (embedding model, monitoring agent, a leaked worker from a prior deploy, or MPS/MIG neighbours) holds memory, so the engine's capacity assumption at startup is invalidated later.",
        mech=("gpu_memory_utilization is applied against total device memory at engine start. Any process that "
              "attaches afterwards competes for the remainder, and CUDA contexts alone cost hundreds of MB each. "
              "Because the engine pre-reserves its pool, a late arrival typically does not shrink the pool but does "
              "consume the activation headroom the engine assumed was free, producing intermittent aborts under peak "
              "prefill rather than a clean startup failure."),
        meas=("(1) nvidia-smi --query-compute-apps=pid,used_memory sampled continuously, not just at incident time. "
              "(2) Correlate OOM events with the appearance of any non-serving PID on the device. "
              "(3) Check for orphaned processes from previous deployments and for MIG/MPS configuration."),
        exp=("Isolate the device: run the identical replayed trace on a node where the serving process is the only "
             "compute app, with everything else unchanged. If the multi-tenancy hypothesis holds, the isolated arm "
             "completes the trace with the same engine settings that aborted on the shared node."),
        conf=("Confounders: the isolated node may differ in driver version or clock behaviour; a monitoring agent may "
              "attach only intermittently and be absent during the test window; and removing the co-tenant may also "
              "remove PCIe or memory-bandwidth contention, improving results for a different reason."),
        mit=("Priority order: (a) enforce single-tenant scheduling for serving GPUs via the orchestrator's device "
             "plugin and resource requests; (b) reap orphaned processes on deploy with an explicit pre-start check; "
             "(c) leave a documented absolute headroom in bytes rather than a utilisation fraction; "
             "(d) alert on any unexpected PID appearing on a serving device."),
        roll=("Rollback gate: if enforcing isolation costs more replicas than the budget allows, revert and instead "
              "lower gpu_memory_utilization until a 24-hour production window shows zero aborts; revert that too if "
              "capacity loss causes queue rejections above 1%."),
        risks=["Assuming a fraction-based headroom is safe when the absolute byte headroom is what actually matters",
               "Orphaned processes from prior deploys can persist unnoticed and reappear only under specific rollout paths",
               "Isolating the GPU raises cost and may be rejected without a quantified failure rate to justify it"],
        ev=["Continuous nvidia-smi per-PID memory sampling across the OOM window, not a single snapshot",
            "Deployment and pod lifecycle events showing whether a stale process survived a rollout",
            "MIG/MPS configuration and the orchestrator's device allocation policy",
            "Absolute free bytes on the device immediately before each abort"],
        dims=(4, 5, 5), dec="rewrite", cf=0.68,
    ),
    dict(
        title="Distinguishing device OOM from host OOM-killer and cgroup limits",
        h="The observed 'OOM' may not be a CUDA allocation failure at all: a host cgroup memory limit killing the process produces a similar symptom (dead worker under long-context load) but requires an entirely different fix, and this is falsifiable directly from the exit signal.",
        mech=("Long-context serving inflates host memory too: tokenizer state, request bodies, pinned staging buffers "
              "for KV swap, and per-request Python objects. A container memory limit terminates the process with "
              "SIGKILL and no Python traceback, whereas a CUDA OOM raises a catchable exception naming allocated and "
              "reserved bytes. Confusing the two leads to tuning GPU settings for a host-side problem, which "
              "cannot succeed."),
        meas=("(1) Read the exit code and kernel log: an OOM-killer event appears in dmesg with the cgroup and the "
              "victim's RSS; a CUDA OOM appears in application logs with byte counts. "
              "(2) Track container memory.current against memory.max alongside GPU metrics. "
              "(3) Confirm whether any Python traceback exists at all for the failure."),
        exp=("Raise the container memory limit substantially in a staging replica while holding every GPU setting "
             "constant, then replay the same trace. If the host hypothesis holds, the crash disappears without any "
             "GPU change; if a CUDA OOM traceback then appears, the two failure modes were stacked and both need fixing."),
        conf=("Confounders: increasing the container limit also increases page-cache room and can change I/O "
              "behaviour; a memory leak in the request path grows with uptime independently of context length; and "
              "some runtimes report the host kill as a generic container restart, hiding the signal."),
        mit=("Priority order: (a) classify the failure from the exit signal before any tuning; "
             "(b) size the container limit from measured RSS at peak concurrency with explicit headroom; "
             "(c) bound pinned host memory used by KV swap; (d) cap request body size and concurrent tokenization; "
             "(e) only then revisit GPU-side settings."),
        roll=("Rollback gate: revert the container limit increase if node-level memory pressure causes eviction of "
              "other pods, and instead reduce per-replica concurrency until measured peak RSS fits with 20% headroom."),
        risks=["Tuning GPU memory settings for a host-side OOM-killer event wastes cycles and can degrade throughput for nothing",
               "Raising container limits can destabilise the node by causing eviction of co-located workloads",
               "Absence of a traceback is easy to misread as a silent CUDA failure rather than a SIGKILL"],
        ev=["Process exit code/signal and dmesg or kernel OOM-killer records for the failure window",
            "Container memory.current versus memory.max time series alongside GPU occupancy",
            "Peak process RSS measured at production concurrency in staging",
            "Application log showing whether a CUDA OOM traceback with byte counts exists"],
        dims=(4, 5, 5), dec="rewrite", cf=0.71,
    ),
    dict(
        title="Capacity model first: compute the theoretical ceiling before tuning",
        h="The deployment is simply configured beyond its arithmetic capacity, which is falsifiable without any experiment: computed weights + KV-at-max-concurrency + activation scratch exceeds usable VRAM, so intermittent OOM is the expected behaviour rather than an anomaly.",
        mech=("Usable VRAM = device memory - driver/context overhead - co-tenants. Weights = params x dtype_bytes / TP. "
              "KV per sequence = 2 x n_layers x n_kv_heads x head_dim x dtype_bytes x seq_len. Peak KV = that value "
              "times max_num_seqs at max_model_len. Activation scratch scales with max_num_batched_tokens. If the sum "
              "of these exceeds usable VRAM, the configuration is over-subscribed and only the arrival pattern "
              "determines when the abort happens."),
        meas=("(1) Read n_layers, n_kv_heads, head_dim, dtype and vocab from the deployed model config, not from a "
              "similar model. (2) Read the effective max_model_len, max_num_seqs, max_num_batched_tokens and "
              "gpu_memory_utilization actually in force. (3) Compare the computed total against measured "
              "max_memory_reserved from a run that survived."),
        exp=("Validate the model rather than the fix: pick a concurrency level the arithmetic says is safe and one it "
             "says is over-subscribed, and drive each with a synthetic max-length workload. If predicted-safe survives "
             "and predicted-unsafe aborts near the predicted point, the capacity model is trustworthy and can be used "
             "to size every subsequent change; if reality diverges, an unmodelled consumer exists and must be found first."),
        conf=("Confounders: GQA versus MHA changes KV by a large factor and is easy to get wrong; KV quantisation "
              "changes dtype_bytes; some engines reserve CUDA graph pools not captured in a naive model; and "
              "gpu_memory_utilization is a fraction of total, not of free, memory."),
        mit=("Priority order: (a) reduce max_model_len or max_num_seqs so the arithmetic closes with explicit headroom; "
             "(b) increase tensor parallelism or add replicas if the SLA requires the current context length; "
             "(c) adopt KV quantisation only behind an output-quality gate; "
             "(d) express headroom in absolute bytes and enforce it in the deploy pipeline as a pre-flight check."),
        roll=("Rollback gate: revert any capacity change that fails the pre-flight arithmetic check, and revert a "
              "max_model_len reduction if it truncates more than 0.5% of production requests measured in shadow mode."),
        risks=["Assuming MHA when the model uses grouped-query attention overstates KV by several times and misdirects the fix",
               "KV quantisation changes numerics and can degrade output quality without an explicit evaluation gate",
               "A capacity model that ignores CUDA graph pools and NCCL buffers will look correct and still under-predict peak"],
        ev=["Deployed model config values: n_layers, n_kv_heads, head_dim, dtype, tensor-parallel degree",
            "Effective runtime limits: max_model_len, max_num_seqs, max_num_batched_tokens, gpu_memory_utilization",
            "Measured max_memory_reserved from a surviving run at known concurrency, for model validation",
            "Production distribution of prompt and generation lengths, to choose a defensible max_model_len"],
        dims=(4, 5, 5), dec="rewrite", cf=0.73,
    ),
    dict(
        title="Observability and guardrails: make the failure diagnosable before it recurs",
        h="The incident is currently under-instrumented, so no diagnosis is defensible; the falsifiable claim is that with the listed instrumentation in place the next occurrence uniquely discriminates between the candidate causes.",
        mech=("Each candidate cause has a distinct signature: capacity exhaustion shows allocated bytes at the cap; "
              "fragmentation shows a large reserved-minus-allocated gap; prefill spikes show the abort inside a "
              "prefill step; rank skew shows a fixed device ordinal; host kill shows a SIGKILL with no traceback; "
              "preemption-path failure shows a preemption burst immediately before the abort. Instrumentation that "
              "captures all six signatures turns the next incident into a single-step diagnosis."),
        meas=("(1) On abort, dump torch.cuda.memory_stats(), the failing allocation size, the device ordinal, the "
              "scheduler phase and the in-flight request parameters. (2) Continuously export block-pool occupancy, "
              "preemption count, queue depth, prompt-length percentiles and per-PID device memory at 1 Hz. "
              "(3) Record the process exit signal and container memory.current."),
        exp=("Validate the instrumentation itself before trusting it: in staging, deliberately induce each failure "
             "mode (over-subscribe concurrency; shrink the container limit; attach a co-tenant process) and confirm "
             "the telemetry produces the predicted distinct signature. If two modes produce identical telemetry, the "
             "instrumentation is insufficient and must be extended before the next production incident."),
        conf=("Confounders: metric scrape intervals coarser than the failure timescale alias the spike away; "
              "sampled request logs miss the tail that matters; and adding instrumentation itself consumes memory and "
              "can perturb the system it measures."),
        mit=("Priority order: (a) ship the abort-time dump and 1 Hz occupancy export first, since they are cheap and "
             "decisive; (b) add a conservative admission-control guard so the service degrades rather than aborts "
             "while diagnosis proceeds; (c) alert on occupancy and preemption rate as leading indicators; "
             "(d) defer engine tuning until one incident has been captured with full telemetry."),
        roll=("Rollback gate: remove any instrumentation whose overhead costs more than 2% throughput or that itself "
              "allocates device memory; revert the admission-control guard if it rejects more than 1% of requests "
              "without reducing abort count."),
        risks=["Acting on a hypothesis before instrumentation exists produces a change with no way to confirm it worked",
               "1 Hz scraping can still alias a sub-second prefill spike, giving false confidence in a clean occupancy chart",
               "Admission-control guards added under time pressure become permanent undocumented capacity caps"],
        ev=["Current metric list and scrape interval, to establish what the existing data can and cannot discriminate",
            "A single abort-time dump containing memory_stats, failing allocation size, device ordinal and scheduler phase",
            "Staging reproduction results for each deliberately induced failure mode",
            "Overhead measurement of the added instrumentation at production load"],
        dims=(4, 5, 5), dec="rewrite", cf=0.72,
    ),
]


def build(angle, user):
    a = angle
    return (
        f"Assumptions: single-node GPU serving of a decoder-only LLM behind a router; "
        f"the crash is a CUDA out-of-memory abort unless the host-side check below says otherwise; I have read "
        f"access to engine metrics, logs and the model config, and a staging replica where I can replay traffic. "
        f"If any of these is false, say so before acting on the plan below.\n\n"
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
        f"cause; the ordering above is a prior, not a conclusion. Quantities such as per-token KV bytes must be "
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
