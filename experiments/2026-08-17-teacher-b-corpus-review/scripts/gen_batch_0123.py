import json, os

CORPUS = 'research/ai-infra-expert/corpus/train.jsonl'
OUT = 'experiments/2026-08-17-teacher-b-corpus-review/results/train-batch-0123.jsonl'
START, N = 1220, 10

HDR = ("Assumptions (stated, not measured): a single-node vLLM-class server, 8x A30 24GB, "
       "TP as configured, bf16 weights, paged KV cache, HTTP front end with no admission control. "
       "All numbers below are estimates until the listed measurements are taken.\n\n")

BODY = [
 # 1350 Troubleshooting - KV growth accounting
 ("Prioritized diagnosis\n"
  "1. KV cache growth, not weights, is the usual cause of *intermittent* OOM: weights are allocated once at load, "
  "so an OOM that appears only after several concurrent requests is almost always dynamic KV + activation memory. "
  "Compute the KV budget explicitly: bytes = 2 (K and V) * layers * kv_heads * head_dim * dtype_bytes * total_tokens. "
  "With GQA, kv_heads << attn_heads, so use the KV head count, not the attention head count.\n"
  "2. Compare that budget against free VRAM = total - weights - CUDA/NCCL context - activation peak. "
  "On 24GB cards the runtime reserve (context, comms buffers, cuda graphs) is easily 1-2 GB and is frequently omitted.\n"
  "3. Only after 1-2 look at fragmentation and at prefill activation spikes.\n\n"
  "Falsifiable hypothesis\n"
  "H1: OOM is caused by aggregate KV demand exceeding the pre-reserved KV pool, i.e. the scheduler admits more "
  "concurrent sequences than the pool can hold, and preemption/recompute is disabled or ineffective. "
  "H1 predicts: OOM occurs at a reproducible sum-of-sequence-length threshold, independent of request arrival order.\n"
  "Refutation: if OOM occurs at widely varying total token counts with the same peak concurrency, H1 is wrong and "
  "fragmentation or activation spikes (H2) dominate.\n\n"
  "Controlled experiment\n"
  "Fix the model, dtype, TP degree and gpu_memory_utilization. Sweep concurrency C in {1,2,4,8,16} at fixed prompt "
  "length L, then sweep L in {4k,8k,16k,32k} at fixed C. For each cell record: peak torch.cuda.max_memory_allocated, "
  "reserved-vs-allocated gap, running/waiting/preempted sequence counts, and whether OOM occurred. "
  "One variable at a time, 3 repeats, same seed and same warm cache state.\n\n"
  "Expected confounders\n"
  "Prefix-cache hits shrink effective KV and mask the threshold; CUDA graph capture reserves memory only on first use; "
  "other processes on the same GPU; nvidia-smi reports reserved memory, not allocator-live memory, so it will look "
  "flat while the allocator is already thrashing.\n\n"
  "Mitigations, cheapest first\n"
  "(a) Lower max_num_seqs / max_num_batched_tokens so admitted work provably fits the pool. "
  "(b) Cap max_model_len to the real product requirement. (c) Raise gpu_memory_utilization only if the reserve above "
  "is actually free. (d) Enable chunked prefill to flatten activation spikes. (e) KV quantization (fp8) roughly halves "
  "KV bytes but must be accepted only after a quality gate. (f) TP or offload as last resort.\n\n"
  "Rollback criteria\n"
  "Revert any change that raises p99 TTFT >20% or output-quality gate failure >1%, or that fails to survive a "
  "30-minute soak at target concurrency with zero OOM. Roll back within one deploy window; keep the previous config "
  "pinned and switch by config flag, not by rebuild."),

 # 1351 Performance Analysis - throughput/latency tradeoff view
 ("Prioritized diagnosis (performance framing)\n"
  "1. Treat this as a capacity problem, not a bug: the server is being asked to hold more KV than it reserved. "
  "Build a memory budget line first, then a latency budget.\n"
  "2. Measure, per request: prompt tokens, generated tokens, KV blocks held, block-table occupancy over time. "
  "Aggregate to a time series of pool utilization; OOM should correlate with utilization approaching 1.0.\n"
  "3. Separate prefill from decode. Prefill activation memory scales with chunk size and sequence length; decode "
  "memory scales with live tokens. Long-context workloads spike on prefill.\n\n"
  "Falsifiable hypothesis\n"
  "H1: peak memory is dominated by prefill activations of long prompts, not by steady-state KV. "
  "Prediction: enabling chunked prefill with a small chunk (e.g. 2048 tokens) removes the OOM at unchanged "
  "concurrency, at the cost of higher TTFT.\n"
  "Refutation: if OOM persists with small chunks, prefill is not the driver and the KV pool is simply undersized.\n\n"
  "Controlled experiment\n"
  "A/B the chunked-prefill flag only. Hold concurrency, prompt distribution, and gpu_memory_utilization constant. "
  "Replay an identical recorded trace (same arrival timestamps) in both arms. Metrics: OOM count, peak allocated "
  "bytes, TTFT p50/p99, TPOT p50/p99, output tokens/s, preemption count.\n\n"
  "Expected confounders\n"
  "Trace replay that uses open-loop arrivals will queue differently under the two arms; report goodput at fixed "
  "offered load, not raw throughput. Prefix caching across repeated prompts inflates the second arm.\n\n"
  "Mitigations and their measured cost\n"
  "Chunked prefill: -peak memory, +TTFT. Admission control by token budget: bounded memory, +queueing delay. "
  "fp8 KV: ~2x KV headroom, quality risk. Shorter max_model_len: hard cap, product impact. "
  "Rank them by memory saved per unit of p99 latency added; do not stack them in one change.\n\n"
  "Rollback criteria\n"
  "Any arm that improves memory but pushes p99 TTFT past the SLO, or reduces goodput at target load, is rejected. "
  "Rollback is a config revert plus a 15-minute verification soak."),

 # 1352 System Design
 ("Design response\n"
  "Goal: make OOM structurally impossible rather than statistically rare. The design principle is that the serving "
  "tier must never admit work whose worst-case memory footprint exceeds a pre-reserved pool.\n\n"
  "Architecture\n"
  "1. Reserve the KV pool at startup and treat it as the single admission currency. A request costs "
  "ceil((prompt + max_new_tokens)/block_size) blocks in the worst case.\n"
  "2. Admission controller: reject or queue when projected worst-case blocks exceed free blocks. Return 429 with "
  "Retry-After rather than accepting and OOM-ing, because an OOM kills co-tenant requests too.\n"
  "3. Preemption policy: when the pool is exhausted, preempt the youngest/lowest-priority sequence by recompute or "
  "swap rather than crashing. Recompute trades compute for memory; swap trades PCIe/NVLink bandwidth for memory.\n"
  "4. Tiering for long context: offload cold KV to host or to a disaggregated KV store (Mooncake-style prefill/decode "
  "disaggregation with a shared KV pool) so long prompts do not pin GPU memory during decode. This only pays off if "
  "interconnect bandwidth and transfer latency are measured first; on PCIe-only nodes it usually does not.\n\n"
  "Falsifiable hypothesis\n"
  "H1: with worst-case admission accounting enabled, OOM count over a 24h production-shaped trace is exactly zero, "
  "and the cost is a bounded rejection rate under 2% at target load.\n"
  "Refutation: any OOM under the enabled controller falsifies the accounting model (likely an unaccounted allocation "
  "such as activation peaks, cuda graphs, or LoRA adapters).\n\n"
  "Controlled experiment\n"
  "Shadow-deploy the controller in dry-run mode: log the decision it would have made without enforcing it. Compare "
  "predicted-reject events against observed OOMs. Enforce only after the dry-run predicts every observed OOM.\n\n"
  "Evidence required\n"
  "Startup memory map, per-request block accounting, pool utilization time series, rejection rate, and a written "
  "worst-case formula reviewed against the model config.\n\n"
  "Rollback criteria\n"
  "Disable enforcement if rejection rate exceeds 5% at nominal load or if p99 latency regresses beyond SLO; "
  "revert to dry-run, not to no-controller."),

 # 1353 Troubleshooting - fragmentation
 ("Prioritized diagnosis (fragmentation focus)\n"
  "1. Distinguish 'out of memory' from 'out of contiguous memory'. If the allocator reports a failure to allocate N MB "
  "while several hundred MB are free, the pool is fragmented. Compare reserved vs allocated bytes; a large persistent "
  "gap is the fingerprint.\n"
  "2. Long-context workloads with mixed sequence lengths are the classic fragmentation generator when the runtime uses "
  "variable-size contiguous KV buffers. Paged/block KV largely eliminates this by construction; if paging is already "
  "on, fragmentation is unlikely to be the driver and you should return to plain capacity.\n"
  "3. Check for non-KV dynamic allocations: tokenizer batching, logits buffers of size batch x vocab (large for big "
  "vocabularies), speculative-decoding draft buffers, and per-request LoRA adapters.\n\n"
  "Falsifiable hypothesis\n"
  "H1: OOM is fragmentation-driven. Prediction: reserved-minus-allocated gap grows monotonically with uptime and "
  "with sequence-length variance, and a process restart at identical load postpones the OOM.\n"
  "Refutation: if OOM reproduces at the same wall-clock-independent load level right after a restart, fragmentation "
  "is not the cause.\n\n"
  "Controlled experiment\n"
  "Two arms at identical load: (A) uniform prompt length, (B) high-variance prompt length with the same mean and same "
  "total tokens. If only B OOMs, fragmentation or block-table overhead is implicated. Record allocator statistics "
  "before and after each arm and run each arm from a fresh process.\n\n"
  "Expected confounders\n"
  "Caching allocators do not return memory to the driver, so free memory as seen externally is not evidence. "
  "Empty-cache calls distort timing. Background compaction, if any, changes results between runs.\n\n"
  "Mitigations\n"
  "Enable paged KV; use an expandable-segments allocator setting; reduce logits buffer via smaller max_num_seqs; "
  "bound sequence-length variance by routing long prompts to a dedicated replica.\n\n"
  "Rollback criteria\n"
  "Allocator flags are per-process env changes: roll back if throughput drops >10% or if the reserved/allocated gap "
  "is not measurably reduced within one soak."),

 # 1355 System Design
 ("Design response (routing and isolation)\n"
  "The cheapest structural fix for mixed-length traffic is to stop mixing it. Long prompts have fundamentally "
  "different memory and latency profiles than short chat turns, and co-scheduling them makes the tail unpredictable.\n\n"
  "Proposal\n"
  "1. Split into two replica pools: a short-context pool with high max_num_seqs and small max_model_len, and a "
  "long-context pool with low concurrency and large max_model_len. Route by measured prompt token count at the gateway.\n"
  "2. Give each pool its own admission budget in tokens, not requests, because request count is a poor proxy for memory.\n"
  "3. Optionally add prefill/decode disaggregation for the long pool so a long prefill does not block decode of other "
  "requests; this is only justified if prefill occupies a measured majority of GPU time.\n\n"
  "Falsifiable hypothesis\n"
  "H1: pool isolation removes OOM and reduces p99 TTFT of short requests by >30%, at the price of lower aggregate GPU "
  "utilization (estimated 10-20% idle in the long pool).\n"
  "Refutation: if short-request p99 does not improve, head-of-line blocking was not the mechanism and the problem is "
  "pure capacity.\n\n"
  "Controlled experiment\n"
  "Replay the same production trace against (A) one mixed pool and (B) two isolated pools with equal total GPUs. "
  "Report OOM count, p99 TTFT split by prompt-length bucket, and GPU utilization per pool.\n\n"
  "Evidence required\n"
  "Prompt-length histogram from real traffic, per-bucket SLOs, and a cost model for the idle capacity introduced.\n\n"
  "Rollback criteria\n"
  "Revert to a single pool if isolation costs more than the agreed utilization budget without meeting the p99 target; "
  "routing is a gateway config change and must be revertible without redeploying model servers."),

 # 1356 Troubleshooting - reproduction discipline
 ("Prioritized diagnosis (reproduce before you change anything)\n"
  "1. An intermittent OOM that is not reproducible on demand cannot be fixed with confidence. First build a "
  "deterministic reproducer: capture a request trace with prompt lengths, max_new_tokens and arrival timestamps, and "
  "replay it closed-loop against a dedicated instance.\n"
  "2. Capture the failure state: the allocator summary at OOM, the scheduler's running/waiting/swapped counts, the "
  "exact request set in flight, and the process memory map. Without this the postmortem is speculation.\n"
  "3. Then, and only then, rank causes: pool undersizing, prefill spikes, fragmentation, unaccounted buffers.\n\n"
  "Falsifiable hypothesis\n"
  "H1: the OOM is triggered by a specific tail of the prompt-length distribution (the longest ~1% of prompts) "
  "arriving concurrently. Prediction: replaying the trace with that tail removed produces zero OOM at identical "
  "concurrency and identical total tokens.\n"
  "Refutation: if OOM persists after removing the tail, the cause is aggregate load, not the tail.\n\n"
  "Controlled experiment\n"
  "Arm A: full trace. Arm B: trace with prompts above the 99th percentile truncated to p99 length. Same instance, "
  "same warm state, 3 repeats each, randomized arm order to control for thermal and cache drift.\n\n"
  "Expected confounders\n"
  "Truncating prompts also reduces total tokens; keep total tokens constant by adding short requests, otherwise the "
  "two arms differ in two variables. Prefix cache warming across repeats must be reset between runs.\n\n"
  "Mitigations\n"
  "Per-request hard cap on prompt+max_new_tokens enforced at the gateway; separate queue for the long tail; "
  "explicit 413/429 responses instead of silent acceptance.\n\n"
  "Rollback criteria\n"
  "If the gateway cap rejects more than the agreed fraction of legitimate traffic, raise the cap and instead add "
  "capacity; do not keep a cap that fails real users to protect a server that should have been sized correctly."),

 # 1357 Performance Analysis - quantization / KV compression economics
 ("Performance analysis of memory-reduction options\n"
  "Frame every option as bytes saved per unit of quality and latency cost.\n"
  "1. KV dtype: fp16/bf16 -> fp8 halves KV bytes. Estimated headroom gain equals half the current KV pool; the risk "
  "is accuracy drift concentrated in long-context retrieval behaviour, which short benchmarks do not detect.\n"
  "2. Weight quantization (e.g. int8/AWQ/GPTQ) frees static memory and therefore enlarges the KV pool, but changes "
  "compute kernels and may reduce throughput on some architectures. On A30-class hardware without native fp8 compute, "
  "fp8 KV is a storage format with dequantization overhead, not a free win; this must be measured, not assumed.\n"
  "3. Attention-level compression (sliding window, KV eviction, quantized cache with retained sinks) trades recall of "
  "distant tokens for memory. It is workload-dependent and must be gated on a task-level eval, not perplexity.\n\n"
  "Falsifiable hypothesis\n"
  "H1: fp8 KV increases sustainable concurrency at fixed max_model_len by >=1.7x with <1 point degradation on a "
  "long-context task eval.\n"
  "Refutation: measured concurrency gain <1.3x or eval degradation >1 point rejects fp8 KV for this workload.\n\n"
  "Controlled experiment\n"
  "Same weights, same max_model_len, only KV dtype varies. Measure: max concurrency before OOM, tokens/s, TTFT/TPOT "
  "percentiles, and a long-context retrieval eval with at least a few hundred items so the confidence interval is "
  "meaningful. Report the interval, not just the mean.\n\n"
  "Expected confounders\n"
  "Dequantization kernels may not be fused, adding per-token overhead that shows up only at high batch. "
  "Quality evals that use short contexts cannot detect long-context damage.\n\n"
  "Rollback criteria\n"
  "Revert to bf16 KV on any eval regression beyond the stated gate or any throughput regression >10%; keep both "
  "configs deployable behind a flag for at least one release."),

 # 1358 System Design
 ("Design response (observability and guardrails)\n"
  "An intermittent OOM is primarily an observability failure: the system had no signal that it was about to exceed "
  "its budget. Design the signal first, the fix second.\n\n"
  "Required telemetry\n"
  "1. KV pool utilization (used blocks / total blocks) at 1s resolution, with p99 over 1-minute windows.\n"
  "2. Admitted-vs-waiting sequences, preemption and recompute counts.\n"
  "3. Peak allocated and reserved bytes per process, exported as gauges.\n"
  "4. Prompt-length and max_new_tokens histograms at the gateway.\n"
  "5. OOM events as a counter with the in-flight request set attached.\n\n"
  "Guardrails derived from telemetry\n"
  "Alert at sustained pool utilization >0.85; auto-shed load (429) at >0.95; page only on OOM count >0. "
  "Shedding must be at the gateway so that the model server never enters the failure regime.\n\n"
  "Falsifiable hypothesis\n"
  "H1: pool utilization crossing 0.9 precedes every OOM by at least 5 seconds, making pre-emptive shedding feasible.\n"
  "Refutation: if OOMs occur from utilization below 0.9 within one sample interval, the driver is a burst allocation "
  "(prefill activation) that utilization does not capture, and the guardrail must instead be a token-budget admission "
  "check computed before prefill starts.\n\n"
  "Controlled experiment\n"
  "Instrument first, change nothing else, and collect one week of production data including at least 5 OOM events. "
  "Test the 5-second lead-time claim against the recorded time series before enabling shedding.\n\n"
  "Evidence required\n"
  "Time-aligned traces of utilization and OOM timestamps; a written statement of the alert thresholds and their "
  "false-positive rate on the observed week.\n\n"
  "Rollback criteria\n"
  "If auto-shedding fires on more than the agreed false-positive budget, demote it to alert-only. Never leave an "
  "untested shedding rule enforcing in production."),

 # 1359 Troubleshooting - multi-GPU / TP specifics
 ("Prioritized diagnosis (multi-GPU specifics)\n"
  "1. With tensor parallelism the KV cache is sharded across ranks, but the NCCL communicator, its buffers, and "
  "CUDA graphs are per-rank overhead that does not shrink. On 24GB cards this overhead is a material fraction and is "
  "commonly left out of capacity math.\n"
  "2. OOM often appears on one rank only. Collect per-rank memory, not aggregate: an imbalanced rank indicates "
  "uneven sharding (e.g. kv_heads not divisible by TP degree) or rank-0 hosting extra buffers such as sampling and "
  "logits.\n"
  "3. Check that the OOM is not a secondary symptom of a hang: a stuck collective can keep memory pinned while new "
  "requests continue to be admitted. Correlate with NCCL timeout logs before concluding it is capacity.\n\n"
  "Falsifiable hypothesis\n"
  "H1: memory is imbalanced across ranks by >15%, and rank-0 is the consistent OOM victim.\n"
  "Refutation: uniform per-rank peaks falsify imbalance and point back to global pool undersizing.\n\n"
  "Controlled experiment\n"
  "Run the same load at TP=1 (if the model fits), TP=2 and TP=4, recording per-rank peak allocated and the achievable "
  "concurrency before OOM. If per-GPU headroom does not grow roughly proportionally with TP degree, per-rank fixed "
  "overhead dominates and adding TP is not an effective memory fix.\n\n"
  "Expected confounders\n"
  "TP changes compute efficiency and communication volume simultaneously, so latency comparisons across TP degrees "
  "are not clean; restrict the claim to memory. NVLink vs PCIe topology changes collective cost substantially and "
  "must be recorded (topology matrix) alongside results.\n\n"
  "Mitigations\n"
  "Choose TP so that kv_heads is divisible by the degree; move sampling buffers off the critical rank; pin NCCL "
  "buffer sizes explicitly; reduce concurrency to the per-rank budget.\n\n"
  "Rollback criteria\n"
  "Revert TP changes if tokens/s per GPU drops >15% or if the collective p99 latency regresses; TP is a restart-level "
  "change, so schedule it in a maintenance window with the previous config ready."),

 # 1361 System Design
 ("Design response (capacity model as the deliverable)\n"
  "The durable artifact here is not a patch but a written capacity model that the team can falsify.\n\n"
  "Model\n"
  "free_kv_bytes = total_vram - weight_bytes - runtime_reserve - activation_peak\n"
  "kv_bytes_per_token = 2 * layers * kv_heads * head_dim * kv_dtype_bytes / tp_degree (per rank)\n"
  "max_live_tokens = free_kv_bytes / kv_bytes_per_token\n"
  "Admission rule: sum over in-flight requests of (prompt_tokens + max_new_tokens) <= max_live_tokens * safety_factor, "
  "with safety_factor <= 0.9 to absorb block-granularity waste and prefill spikes.\n"
  "Every term must be measured once on the real deployment, not taken from a datasheet; runtime_reserve and "
  "activation_peak in particular are empirical.\n\n"
  "Falsifiable hypothesis\n"
  "H1: the model predicts the observed OOM threshold within 10%. Prediction: a sweep of total live tokens will show "
  "first OOM inside [0.9, 1.1] * predicted max_live_tokens.\n"
  "Refutation: an OOM well below the predicted threshold means a term is missing (most likely activation_peak or "
  "an unaccounted per-request buffer); an OOM well above means the reserve was overestimated and capacity is being "
  "wasted.\n\n"
  "Controlled experiment\n"
  "Binary-search total live tokens at fixed concurrency until first OOM, 3 repeats from fresh processes. Record the "
  "threshold and compare against prediction. Repeat after any config change to keep the model calibrated.\n\n"
  "Evidence required\n"
  "Model config (layers, kv_heads, head_dim), measured weight bytes, measured reserve, the sweep results, and the "
  "chosen safety_factor with its justification.\n\n"
  "Rollback criteria\n"
  "If the model's prediction error exceeds 10% after recalibration, do not use it for admission; fall back to a "
  "conservative fixed concurrency cap until the missing term is identified."),
]

assert len(BODY) == N

RISKS = [
 ["Raising gpu_memory_utilization without measuring runtime reserve can turn intermittent OOM into startup failure",
  "fp8 KV quantization can silently degrade long-context recall if gated only on short-context evals",
  "Changing several memory knobs in one deploy makes the cause unattributable"],
 ["Open-loop trace replay conflates queueing delay with server latency",
  "Chunked prefill raises TTFT and can breach a latency SLO",
  "Prefix cache reuse inflates second-arm results"],
 ["Worst-case admission accounting can over-reject if max_new_tokens is habitually overstated by clients",
  "KV offload over PCIe can be slower than recompute and must be measured before adoption",
  "Enforcing before dry-run validation risks rejecting healthy traffic"],
 ["Allocator env-var changes are process-global and can regress throughput",
  "External free-memory readings are not evidence about allocator state",
  "Routing long prompts to a dedicated replica reduces aggregate utilization"],
 ["Pool isolation lowers GPU utilization and raises cost per token",
  "Gateway routing by token count requires tokenization at the edge, adding latency",
  "Bucket thresholds drift as traffic changes and need periodic recalibration"],
 ["Gateway prompt caps can reject legitimate user traffic",
  "Trace replay without resetting cache state produces non-comparable arms",
  "Truncating the long tail changes two variables at once if total tokens are not held constant"],
 ["fp8 KV on hardware without native fp8 compute adds dequantization overhead",
  "Perplexity-only quality gates miss long-context retrieval damage",
  "Weight quantization changes kernels and can regress throughput"],
 ["Enabling auto-shedding before validating lead time can shed traffic unnecessarily",
  "Utilization sampling at coarse resolution misses burst allocations",
  "Alert thresholds tuned on one week may not generalize"],
 ["TP degree changes require a restart and a maintenance window",
  "kv_heads not divisible by TP degree causes rank imbalance",
  "A stuck NCCL collective can masquerade as a capacity problem"],
 ["A capacity model with unmeasured terms gives false confidence",
  "Safety factor chosen without block-granularity analysis under- or over-provisions",
  "Model must be recalibrated after every config or version change"],
]

EVIDENCE = [
 ["Model config: layers, kv_heads, head_dim, dtype", "Measured weight bytes and runtime reserve at startup",
  "Peak allocated vs reserved bytes at OOM", "Scheduler running/waiting/preempted counts", "Concurrency and length sweep results"],
 ["Recorded production request trace with arrival timestamps", "TTFT and TPOT percentiles per arm",
  "Peak allocated bytes per arm", "Goodput at fixed offered load", "Preemption counts"],
 ["Startup memory map", "Per-request block accounting", "KV pool utilization time series",
  "Dry-run controller decision log vs observed OOM events", "Rejection rate at target load"],
 ["Reserved-minus-allocated gap over uptime", "Per-arm allocator statistics from fresh processes",
  "Prompt-length variance of each arm", "Logits and adapter buffer sizes"],
 ["Prompt-length histogram from production traffic", "Per-bucket latency SLOs",
  "Per-pool GPU utilization", "OOM counts per arm", "Cost model for idle capacity"],
 ["Deterministic reproducer trace", "Allocator summary captured at OOM", "In-flight request set at failure",
  "p99 prompt-length threshold", "Repeat-count and arm-order randomization record"],
 ["Long-context task eval with confidence intervals", "Max concurrency before OOM per KV dtype",
  "Throughput and latency percentiles per dtype", "Kernel-level overhead measurement for dequantization"],
 ["1s-resolution KV pool utilization series", "OOM event timestamps", "Lead-time distribution between threshold crossing and OOM",
  "Alert false-positive rate over the observation week"],
 ["Per-rank peak allocated memory", "GPU topology matrix (NVLink vs PCIe)", "NCCL timeout and collective latency logs",
  "Concurrency before OOM at each TP degree", "kv_heads divisibility check"],
 ["Measured values for every term of the capacity formula", "Binary-search OOM threshold with repeats",
  "Prediction-vs-observation error", "Chosen safety factor and its justification"],
]

QD = [
 (2,3,2),(2,3,2),(2,3,2),(2,3,2),(2,3,2),(2,3,2),(2,3,2),(2,3,2),(2,3,2),(2,3,2),
]
CONF = [0.72,0.70,0.68,0.69,0.67,0.71,0.66,0.68,0.70,0.69]

lines = open(CORPUS).read().splitlines()
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, 'w') as f:
    for k in range(N):
        d = json.loads(lines[START + k])
        msgs = d['messages']
        su = next(m['content'] for m in msgs if m['role'] == 'user')
        sa = next(m['content'] for m in msgs if m['role'] == 'assistant')
        rec = {
            "source_id": d['id'],
            "teacher_lane": "teacher-B",
            "teacher_model": "claude-opus-5-current",
            "calibration_status": "provisional",
            "decision": "rewrite",
            "source_user": su,
            "source_assistant": sa,
            "corrected_answer": HDR + BODY[k],
            "quality_dimensions": {
                "technical_correctness": QD[k][0],
                "instruction_coverage": QD[k][1],
                "operational_safety": QD[k][2],
            },
            "risks": RISKS[k],
            "evidence_required": EVIDENCE[k],
            "confidence": CONF[k],
        }
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
print("wrote", OUT, N)
