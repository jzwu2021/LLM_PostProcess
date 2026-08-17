import json, os

OFF, N = 1210, 10
OUT = 'experiments/2026-08-17-teacher-b-corpus-review/results/train-batch-0122.jsonl'

rows = [json.loads(l) for l in open('research/ai-infra-expert/corpus/train.jsonl')]
sl = rows[OFF:OFF + N]

COMMON_MECH = (
"Mechanism first. Steady-state device memory = weights + CUDA/NCCL context and comm buffers + framework "
"runtime buffers + activation peak of the largest scheduled step + KV-cache pool. Only the last two scale "
"with concurrency, which is precisely why the failure is intermittent rather than deterministic: a plain "
"sizing error would fail on request one. KV bytes = 2 * layers * kv_heads * head_dim * dtype_bytes * "
"total_resident_tokens; under GQA/MQA the kv_heads factor is small, so resident context length x concurrent "
"sequences dominates. Important boundary condition: if the engine preallocates a paged KV pool, true KV "
"exhaustion should manifest as preemption, recompute or queueing -- not as a CUDA OOM. An actual OOM "
"therefore indicates memory consumed OUTSIDE the pool: prefill activation spikes on long prompts (chunked "
"prefill off or chunk too large), caching-allocator fragmentation from mixed-size transient buffers, or an "
"unaccounted co-resident consumer (LoRA adapters, speculative draft model, logits/logprob tensors "
"proportional to batch x vocab, a second process on the same device, metrics/profiler buffers)."
)

ANSWERS = [
# 1210 corpus-01339 Performance Analysis v39
"Performance-analysis framing (variant 39). The question is not 'why did it crash' but 'what is the memory "
"headroom curve as a function of concurrency, and where does it cross zero'.\n\n"
"Assumptions, stated because the whole priority ranking is conditional on them: single-node inference engine "
"with paged KV cache; weights loaded once and static; PyTorch caching allocator; no co-tenant on the GPUs; "
"the error is a device-side CUDA OOM, not a host RSS OOM-kill. If any assumption is false the ranking must be "
"re-derived, not patched.\n\n" + COMMON_MECH + "\n\n"
"Falsifiable hypothesis H1: peak device memory is a linear function of resident KV tokens with a bounded "
"residual, and OOM occurs when concurrency pushes the linear term past (total - reserved) memory. Prediction: "
"in a controlled sweep, peak allocated memory regressed on resident tokens gives R^2 > 0.9 and the fitted "
"intercept matches measured weights + context within 10 percent. Refutation: if peak memory jumps "
"discontinuously at a concurrency level, or reserved-minus-allocated (fragmentation gap) grows monotonically "
"while allocated stays flat, H1 is false and the driver is fragmentation or a transient allocation, not "
"steady KV growth.\n\n"
"Controlled experiment. Fix model, dtype, max_model_len, engine version and seed. Sweep concurrency 1,2,4,8,16 "
"at two prompt-length classes (short ~1k, long ~32k), 3 repeats, randomized order to avoid drift confounding. "
"Per run record torch.cuda.max_memory_allocated and max_memory_reserved, nvidia-smi used memory, engine "
"KV-utilization and preemption counters, and TTFT/ITL. Fragmentation gap = reserved - allocated is the key "
"derived metric.\n\n"
"Prioritized diagnosis, highest expected-information-per-minute first: (1) confirm it is device OOM and read "
"the requested block size from the CUDA OOM message -- a large request with ample free-but-fragmented memory "
"is diagnostic of fragmentation, a small request means genuine exhaustion; (2) check whether preemption "
"counters moved -- nonzero preemption with OOM means the pool worked and the spike is outside it; (3) verify "
"only one process holds the device; (4) confirm max_model_len x max concurrent sequences is actually bounded "
"by admission control rather than by luck.\n\n"
"Mitigations in order of reversibility, not of appeal: cap max_model_len and max concurrent sequences so the "
"worst case is bounded by construction; enable or shrink chunked prefill to flatten activation peaks; lower "
"the KV pool fraction to leave allocator headroom; set expandable_segments to reduce fragmentation; only then "
"consider KV quantization (FP8/INT8) which changes output quality and requires its own eval gate.\n\n"
"Expected confounders: request-length distribution drift between runs, prefix-cache hit-rate changes altering "
"effective KV footprint, background metrics scrapes, and thermal/clock variation affecting timing but not "
"memory. Control by replaying a fixed trace.\n\n"
"Evidence required before declaring root cause: the sweep table, the OOM message block size, the "
"fragmentation-gap trend, and a reproduction at the predicted concurrency threshold.\n\n"
"Rollback gates: revert any change if p99 TTFT regresses more than 20 percent, if throughput drops more than "
"10 percent at target concurrency, or if a quality eval on a held-out set moves beyond its pre-registered "
"tolerance. Roll forward only after 24h with zero OOM at 1.5x peak observed concurrency.",

# 1211 corpus-01341 Troubleshooting v41
"Troubleshooting framing (variant 41). Goal of the first 15 minutes is not a fix but a classification: device "
"OOM vs host OOM-kill, in-pool exhaustion vs out-of-pool spike, single-tenant vs contended device.\n\n"
"Assumptions: single-node engine with paged KV, static weights, caching allocator, no co-tenant, device-side "
"CUDA OOM. State them in the incident channel so anyone can contradict them cheaply.\n\n" + COMMON_MECH + "\n\n"
"Triage sequence with decision points. Step 1: pull the exact exception. 'CUDA out of memory. Tried to "
"allocate X' with X large and 'free' also large means fragmentation; X small with near-zero free means true "
"exhaustion; dmesg showing an oom-killer entry means it was host RAM and the entire GPU analysis is the wrong "
"tree. Step 2: nvidia-smi during load -- if two PIDs hold the device, stop and remove the co-tenant before "
"any tuning. Step 3: engine counters -- if preemption/recompute counters are zero while OOM fires, the pool "
"never became the binding constraint and the spike is out-of-pool (prefill activations, logits, adapters). "
"Step 4: correlate OOM timestamps with request length percentiles from the access log; if every OOM follows a "
"burst of >16k-token prompts within one scheduling window, admission is the missing control.\n\n"
"Falsifiable hypothesis H2: OOM is triggered by concurrent long-prompt PREFILL, not by decode-phase KV growth. "
"Prediction: forcing chunked prefill with a small token budget eliminates OOM at the same concurrency and "
"same total resident KV, at the cost of higher TTFT. Refutation: if OOM persists unchanged with chunked "
"prefill on and KV utilization at 100 percent, H2 is false and the constraint is steady-state pool sizing.\n\n"
"Controlled experiment: replay a captured production trace (same arrival times and prompt lengths) against two "
"identical replicas, A with chunked prefill enabled and a bounded prefill token budget, B unchanged, same "
"build and same GPUs, 3 repeats with A/B roles swapped to cancel node effects. Metric: OOM count, "
"max_memory_reserved, TTFT p50/p99, throughput.\n\n"
"Mitigations ordered by blast radius: (a) immediate and reversible -- lower max concurrent sequences and cap "
"max_model_len, add a request-size admission limit that rejects with 4xx instead of crashing the replica; "
"(b) enable/shrink chunked prefill; (c) set expandable_segments to reduce fragmentation; (d) reduce KV pool "
"fraction; (e) structural -- KV quantization or a disaggregated prefill/decode split so prefill spikes cannot "
"kill decode capacity. Never begin with (e) during an incident.\n\n"
"Confounders: autoscaling changing replica count mid-test, prefix-cache warmth, retry storms after the first "
"OOM inflating apparent load, and mixed model versions behind one endpoint.\n\n"
"Evidence required: exception text with block size, per-PID nvidia-smi snapshot, preemption counters, "
"prompt-length histogram at OOM time, and the A/B replay table.\n\n"
"Rollback gates: revert immediately if error rate from admission rejections exceeds the agreed budget, if p99 "
"TTFT doubles, or if OOM recurs at reduced concurrency (which falsifies the whole capacity model and demands "
"re-diagnosis rather than further tuning).",

# 1212 corpus-01342 Performance Analysis v42
"Performance-analysis framing (variant 42). Treat memory as a throughput resource: the metric that matters is "
"tokens served per GB-second, and OOM is what happens when the scheduler is allowed to overcommit it.\n\n"
"Assumptions: single node, paged KV engine, static weights, caching allocator, exclusive GPUs, device-side "
"OOM. Ranking is conditional on these.\n\n" + COMMON_MECH + "\n\n"
"Falsifiable hypothesis H3: the binding constraint is transient allocator fragmentation, not aggregate demand. "
"Prediction: at the OOM point, reserved memory is within a few percent of device capacity while allocated is "
"materially lower, and the gap grows with the diversity of prompt lengths; enabling expandable segments "
"shrinks the gap and raises the concurrency at which OOM occurs, with no change in steady-state allocated. "
"Refutation: if the reserved-allocated gap is small and stable while allocated tracks capacity, fragmentation "
"is not the cause and the answer is capacity, not allocator tuning.\n\n"
"Controlled experiment: two workloads at matched total token volume -- W1 uniform prompt length, W2 highly "
"variable lengths (heavy tail). If fragmentation drives the failure, W2 OOMs at strictly lower concurrency "
"than W1 despite equal resident tokens. Repeat each 3 times, alternate order, record allocated, reserved, gap, "
"OOM concurrency threshold, and goodput.\n\n"
"Prioritized diagnosis: (1) separate allocated vs reserved and compute the gap trend; (2) read the failing "
"allocation size; (3) check KV pool utilization and preemption to see whether the pool is even saturated; "
"(4) audit out-of-pool consumers -- logits buffers scale with batch x vocab and are easy to overlook at large "
"batch; (5) confirm no second process.\n\n"
"Mitigations, cheapest and most reversible first: expandable_segments; bound prompt-length variance by "
"bucketing requests into length classes so the allocator sees fewer distinct block sizes; cap concurrency and "
"max_model_len; chunked prefill; reduce KV pool fraction to leave allocator slack; KV quantization last, "
"behind a quality gate.\n\n"
"Confounders: prefix caching changing effective KV per request between arms; engine version differences in "
"allocator behaviour; measurement overhead from memory snapshotting; and warmup effects -- always discard the "
"first N requests.\n\n"
"Evidence required: allocated/reserved/gap time series per arm, OOM concurrency threshold per arm, failing "
"block size, and goodput at the new limits so the fix is not silently a throughput regression.\n\n"
"Rollback gates: revert if goodput at target concurrency falls more than 10 percent, if p99 latency regresses "
"more than 20 percent, or if the gap metric fails to improve -- the last case means the hypothesis was wrong "
"and further allocator tuning is unjustified.",

# 1213 corpus-01343 System Design v43
"System-design framing (variant 43). The defect is architectural: the service accepts work whose peak memory "
"demand is unbounded, so no amount of tuning removes the failure mode -- it only moves the threshold.\n\n"
"Assumptions: single-node replicas behind a router, paged KV engine, static weights, exclusive GPUs, "
"device-side OOM. Multi-tenant or fractional-GPU deployments invalidate the design below.\n\n" + COMMON_MECH + "\n\n"
"Design principle: make the worst case computable. Define an explicit admission contract -- max_model_len L, "
"max concurrent sequences C, max prefill tokens per step P -- and size the KV pool so that L*C tokens fit with "
"a reserved headroom for activations and allocator slack. Then the OOM becomes structurally impossible and "
"overload degrades as queueing or 429s, which are observable and SLO-negotiable, rather than as replica death.\n\n"
"Falsifiable hypothesis H4: with (L, C, P) enforced at admission and the KV pool sized to L*C plus headroom, "
"OOM count is zero across the full production trace at 1.5x peak load, and the only visible effect is queue "
"delay. Refutation: any OOM under those bounds proves an unaccounted out-of-pool consumer exists, and the "
"memory model must be corrected before shipping.\n\n"
"Controlled experiment: shadow-deploy the bounded configuration on one replica, mirror production traffic, run "
"a synthetic 1.5x overload for 30 minutes, and compare OOM count, rejection rate, queue depth, TTFT/ITL and "
"throughput against an unbounded control replica. Three repeats, replica roles swapped.\n\n"
"Structural options beyond bounds, with their costs: disaggregated prefill/decode (Dynamo-style) isolates "
"prefill spikes from decode pools but adds a KV transfer over the fabric and requires RDMA-class links to keep "
"transfer off the critical path; hierarchical/offloaded KV (Mooncake-style pooling to CPU or NVMe) raises "
"effective capacity but introduces bandwidth-bound stalls and a new failure domain; tensor parallelism reduces "
"per-GPU weights and KV but adds NCCL buffers and per-step collectives, so it is not a free win at low batch. "
"Each of these should be justified by a measured constraint, not adopted by default.\n\n"
"Confounders: router retry behaviour turning one rejection into three requests; heterogeneous replicas; "
"prefix-cache warmth differing between shadow and control.\n\n"
"Evidence required: the memory budget worksheet with every term measured (weights, context, activations at "
"P, pool at L*C, slack), the overload test table, and a demonstration that rejections are graceful.\n\n"
"Rollback gates: revert the bounded config if rejection rate exceeds the error budget at normal load, if p99 "
"TTFT regresses beyond the SLO, or if any OOM occurs -- an OOM under bounds means the model is wrong and the "
"design must not ship until the missing term is identified.",

# 1214 corpus-01344 Troubleshooting v44
"Troubleshooting framing (variant 44), focused on the intermittency itself. Intermittent means "
"state-dependent; the job is to find which state variable crosses a threshold.\n\n"
"Assumptions: single node, paged KV, static weights, caching allocator, exclusive device, CUDA OOM rather "
"than host OOM-kill.\n\n" + COMMON_MECH + "\n\n"
"Candidate state variables, each independently testable: resident KV tokens; number of simultaneous prefills; "
"prompt-length tail within one scheduling window; allocator fragmentation accumulated since process start; "
"number of loaded LoRA adapters; prefix-cache occupancy. Note that the fourth is time-dependent, which "
"predicts a distinctive signature -- OOM occurring at progressively lower concurrency as process uptime "
"increases.\n\n"
"Falsifiable hypothesis H5: OOM probability depends on process uptime, indicating accumulating fragmentation "
"or a leak, not on instantaneous concurrency alone. Prediction: after restarting the replica, the same trace "
"runs clean at a concurrency that OOMed before restart, and reserved memory at fixed load rises monotonically "
"with uptime. Refutation: if a freshly restarted replica OOMs at the identical concurrency and reserved is "
"flat over hours, uptime is irrelevant and the cause is instantaneous demand.\n\n"
"Controlled experiment: hold the trace fixed and vary only uptime -- measure OOM-free maximum concurrency at "
"t=5min, t=2h, t=8h on the same replica, with a restarted control replica measured at the same wall-clock "
"times to separate uptime from diurnal traffic effects. Record reserved, allocated, gap, and adapter count.\n\n"
"Prioritized actions: (1) capture the failing allocation size and free/reserved at failure; (2) check whether "
"a restart resets the threshold -- this single test cleanly splits leak/fragmentation from capacity; "
"(3) enumerate out-of-pool consumers, especially dynamically loaded adapters, which grow with traffic mix and "
"are a common hidden accumulator; (4) verify admission bounds exist at all.\n\n"
"Mitigations: if uptime-dependent -- enable expandable segments, bound the adapter cache with eviction, and as "
"a stopgap add a supervised rolling restart with connection draining (a mitigation, explicitly not a fix, and "
"it must be labelled as such so the leak is still hunted). If not uptime-dependent -- cap L and C, enable "
"chunked prefill, resize the pool.\n\n"
"Confounders: diurnal load, deploys during the observation window, prefix-cache warmth, and the restart itself "
"clearing caches and thereby changing the workload's effective cost.\n\n"
"Evidence required: the uptime-vs-threshold table with the restarted control, reserved-memory trend over "
"hours, adapter-count trend, and failing block size.\n\n"
"Rollback gates: remove the rolling restart the moment the underlying accumulator is fixed; revert allocator "
"changes if the gap does not shrink; escalate to capacity work if the restart test shows no uptime effect, "
"because then all leak-oriented remediation is wasted effort.",

# 1215 corpus-01345 Performance Analysis v45
"Performance-analysis framing (variant 45), quantitative sizing before any knob is touched.\n\n"
"Assumptions: single node, paged KV, static weights, caching allocator, exclusive GPUs, device OOM.\n\n"
+ COMMON_MECH + "\n\n"
"Build the budget explicitly. Measure, do not estimate: weights via allocated memory after load with zero "
"requests; CUDA/NCCL context via nvidia-smi used minus torch allocated at idle; activation peak by running a "
"single request at max_model_len and taking max_memory_allocated minus the idle baseline; KV per token by "
"the closed form above, then validated against measured pool growth at known token counts. The residual "
"between measured total and the sum of terms is the honest uncertainty and must be carried through, not "
"rounded away.\n\n"
"Falsifiable hypothesis H6: measured KV bytes per token equals the analytic 2*layers*kv_heads*head_dim*"
"dtype_bytes within 5 percent, and total peak at concurrency C equals baseline + C*L*kv_per_token + "
"activation_peak within 10 percent. Prediction: the model predicts the empirical OOM concurrency threshold "
"within one step of the sweep. Refutation: a systematic underestimate means an unmodelled consumer -- most "
"often logits/logprob buffers scaling with batch x vocab, speculative-decoding draft state, or block-level "
"padding waste in the paged allocator.\n\n"
"Controlled experiment: sweep concurrency at fixed prompt length, 3 repeats, and plot measured peak against "
"predicted peak. The residual slope identifies the missing per-sequence term; a residual that grows with "
"batch but not with tokens points at logits buffers specifically, which is a cleanly discriminating signature.\n\n"
"Prioritized diagnosis: (1) validate the budget; (2) identify the dominant term -- optimizing anything else "
"is theatre; (3) check block-size padding waste in the paged cache, which can be 10-20 percent with many "
"short sequences; (4) only then tune.\n\n"
"Mitigations chosen by which term dominates: KV-dominated -> cap L and C, KV quantization, prefix-cache reuse "
"to raise served tokens per resident token; activation-dominated -> chunked prefill and smaller prefill "
"budget; overhead-dominated -> reduce NCCL buffer sizing or reconsider the parallelism plan; "
"fragmentation-dominated -> expandable segments and length bucketing.\n\n"
"Confounders: dtype differences between weights and KV, prefix cache silently reducing measured KV growth, "
"engine-version changes to block size, and profiler overhead inflating peaks.\n\n"
"Evidence required: the budget worksheet with every measured term and its uncertainty, the "
"predicted-vs-measured curve, and the identified dominant term.\n\n"
"Rollback gates: do not ship any tuning whose predicted benefit is smaller than the budget residual -- that is "
"unfalsifiable at current measurement precision. Revert if throughput at target concurrency drops more than "
"10 percent or if the quality gate on a held-out eval fails after KV quantization.",

# 1216 corpus-01346 System Design v46
"System-design framing (variant 46), capacity contract and degradation policy.\n\n"
"Assumptions: replicated single-node serving behind a router, paged KV, static weights, exclusive GPUs, "
"device OOM. Fractional-GPU or MIG deployments change the accounting and are out of scope of this ranking.\n\n"
+ COMMON_MECH + "\n\n"
"The design question is what should happen when demand exceeds memory. Three legitimate answers, each with a "
"different SLO shape: queue (latency degrades, no errors, requires bounded queue and a deadline policy); "
"reject (errors are explicit and cheap, protects tail latency for admitted work); preempt and recompute "
"(throughput degrades, hides the pressure from clients but wastes compute). Crashing on OOM is the one answer "
"that is never acceptable, because it converts a capacity problem into an availability problem and drops "
"in-flight requests from other tenants.\n\n"
"Falsifiable hypothesis H7: with admission bounds (L, C, P) plus a bounded queue with request deadlines, "
"overload at 2x peak produces zero OOM, bounded p99 latency for admitted requests, and a rejection rate that "
"scales smoothly with excess load. Refutation: nonlinear latency blowup or any OOM at 2x indicates the queue "
"is unbounded in memory terms -- queued requests must not hold KV allocations, and if they do, the design is "
"wrong.\n\n"
"Controlled experiment: staircase load test at 0.5x, 1x, 1.5x, 2x peak, 10 minutes per step, on a shadow "
"replica with bounds enabled versus a control without. Record OOM count, rejection rate, queue depth, p50/p99 "
"TTFT and ITL, throughput, and whether the replica ever restarts. Three repeats.\n\n"
"Scaling options and their honest costs: horizontal replicas are simplest and linear in cost but do nothing "
"for a single oversized request; prefill/decode disaggregation isolates the spike source but adds KV transfer "
"and needs RDMA/RoCE-class links plus careful congestion control (PFC/ECN misconfiguration turns this into a "
"new incident class); KV offload to host or NVMe raises capacity but is bandwidth-bound and needs GDS-style "
"paths to be viable; larger tensor-parallel degree lowers per-GPU pressure but raises collective cost and "
"NCCL buffer footprint.\n\n"
"Confounders: router retries amplifying rejections into load, unequal replica warmth, and load generators that "
"close connections on timeout and hide true tail latency.\n\n"
"Evidence required: staircase table, proof that queued requests hold no device memory, and a memory budget "
"showing the bounds are consistent with pool size.\n\n"
"Rollback gates: revert if rejection rate at 1x load is nonzero, if p99 exceeds SLO at 1.5x, or if any OOM "
"occurs at any step. Promote only after a full-week soak at production traffic with zero OOM.",

# 1217 corpus-01347 Troubleshooting v47
"Troubleshooting framing (variant 47), with explicit attention to what would make the obvious answer wrong.\n\n"
"Assumptions: single node, paged KV engine, static weights, caching allocator, exclusive GPUs, device-side "
"CUDA OOM. If the process was instead killed by the kernel OOM-killer, everything below is misdirected and "
"the investigation moves to host memory -- check dmesg first, it costs seconds.\n\n" + COMMON_MECH + "\n\n"
"The obvious answer is 'too much KV, lower concurrency'. It is frequently wrong for a specific and checkable "
"reason: a correctly configured paged engine preallocates its pool, so the pool cannot grow into an OOM. "
"Therefore, before touching concurrency, establish whether KV utilization was actually at 100 percent and "
"whether preemption fired. If utilization was 70 percent when the OOM hit, lowering concurrency will "
"coincidentally reduce the out-of-pool spike and appear to fix the problem while leaving the real consumer "
"unidentified -- the classic false-confirmation trap here.\n\n"
"Falsifiable hypothesis H8: the OOM originates from out-of-pool transient allocations during prefill. "
"Prediction: OOM events correlate with simultaneous prefill count (not with total resident tokens), and "
"capping prefill token budget per step eliminates them while leaving KV utilization unchanged. Refutation: if "
"OOMs correlate with resident tokens and occur only at 100 percent pool utilization, H8 is false.\n\n"
"Controlled experiment: instrument to log, at 1-second resolution, simultaneous prefill count, resident KV "
"tokens, allocated, reserved. Replay a fixed trace three times. Compute point-biserial correlation of OOM "
"against each candidate. Then run one arm with a capped prefill token budget, everything else identical.\n\n"
"Prioritized mitigations: cap prefill tokens per step / enable chunked prefill; enforce a max prompt length at "
"the gateway with an explicit 4xx; enable expandable segments; reduce KV pool fraction to leave slack for "
"transients; reduce max concurrent sequences only after the above, since it costs throughput directly; "
"KV quantization last and behind a quality gate.\n\n"
"Confounders: prefix-cache hits making some long prompts cheap and breaking the naive length-to-cost mapping; "
"retry storms; speculative decoding adding draft-model memory that is invisible in KV metrics; mixed adapter "
"traffic.\n\n"
"Evidence required: 1-second telemetry around at least three OOM events, correlation table, failing block "
"size, KV utilization at failure, and the capped-prefill arm comparison.\n\n"
"Rollback gates: revert any change that raises p99 TTFT beyond SLO; treat 'no OOM for one hour' as "
"insufficient evidence -- require the event rate to drop by an order of magnitude over a period at least 10x "
"the historical mean inter-arrival time of OOMs before closing the incident.",

# 1218 corpus-01348 Performance Analysis v48
"Performance-analysis framing (variant 48), cost of the fix as a first-class metric.\n\n"
"Assumptions: single node, paged KV, static weights, caching allocator, exclusive GPUs, device OOM. The "
"ranking below assumes throughput matters; for a latency-only service the ordering changes.\n\n"
+ COMMON_MECH + "\n\n"
"Every mitigation buys memory headroom at a price, and the prices are not comparable by intuition. Capping "
"concurrency costs throughput roughly linearly in the memory-bound regime. Chunked prefill costs TTFT on long "
"prompts but is nearly free for decode-heavy traffic. Shrinking the KV pool costs effective batch size. KV "
"quantization to FP8 roughly halves KV bytes but perturbs outputs and demands a quality eval. Prefix caching "
"can be strongly negative-cost when the workload has shared system prompts, and worthless when it does not. "
"The correct decision therefore depends on measured workload structure, not on a general ranking.\n\n"
"Falsifiable hypothesis H9: headroom per unit of lost goodput is highest for chunked prefill in this workload, "
"because the traffic is decode-dominated. Prediction: chunked prefill raises OOM-free concurrency by at least "
"30 percent while costing under 10 percent p99 TTFT; capping concurrency achieves the same headroom only with "
"a proportional goodput loss. Refutation: if the trace is prefill-dominated (measure prefill token share), "
"chunked prefill costs far more and the hypothesis is false before the experiment even runs.\n\n"
"Controlled experiment: four arms -- baseline, chunked prefill, reduced pool fraction, KV FP8 -- on identical "
"hardware with an identical replayed trace, 3 repeats, randomized order. Metrics: OOM-free maximum "
"concurrency, goodput at fixed SLO, p99 TTFT and ITL, and for the FP8 arm a held-out quality score with "
"pre-registered tolerance. Report headroom gained per percent goodput lost.\n\n"
"Prioritized diagnosis before the sweep: measure prefill vs decode token share, prompt-length distribution, "
"and prefix-cache hit rate. These three numbers determine which arm can possibly win.\n\n"
"Confounders: quality eval variance for the FP8 arm (needs enough samples to resolve the tolerance), warmup, "
"trace drift, and engine-version differences between arms.\n\n"
"Evidence required: workload characterization, the four-arm table with the headroom-per-goodput ratio, and "
"the quality-eval confidence interval for any quantization arm.\n\n"
"Rollback gates: reject KV quantization if the held-out quality delta exceeds the pre-registered tolerance "
"regardless of memory savings; revert any arm whose goodput loss exceeds the headroom benefit; and do not "
"stack multiple mitigations in one deploy, since that makes attribution impossible if the OOM returns.",

# 1219 corpus-01349 System Design v49
"System-design framing (variant 49), failure isolation and the multi-node question.\n\n"
"Assumptions: paged KV engine, static weights, exclusive GPUs, device-side OOM, and a service that may be "
"scaled either by replication or by parallelism. These two paths have different failure semantics and the "
"choice must be deliberate.\n\n" + COMMON_MECH + "\n\n"
"Isolation principle: an OOM should destroy at most one request, not one replica, and never a whole "
"parallel group. This matters because under tensor or pipeline parallelism a single rank's OOM aborts the "
"collective and takes down every rank in the group -- the blast radius is the parallel degree, not one GPU. "
"Consequently, memory headroom requirements are strictly tighter for a TP group than for independent "
"replicas, and any capacity plan that ignores this understates risk by the parallel degree. NCCL communication "
"buffers and, on multi-node, RDMA/RoCE registered memory add a fixed per-rank term that must appear in the "
"budget; if RoCE is misconfigured (PFC/ECN thresholds wrong) you additionally get retransmit-driven latency "
"that inflates in-flight state and worsens memory pressure -- a coupling that surprises people.\n\n"
"Falsifiable hypothesis H10: at current model size, replication has strictly better memory headroom per served "
"token than raising tensor parallelism, because per-rank comm buffers and duplicated activation peaks offset "
"the weight savings at this batch size. Prediction: TP=2 does not double OOM-free concurrency; it yields less "
"than 1.6x. Refutation: if the model is weight-dominated (weights >> KV at target concurrency), TP wins "
"clearly and the hypothesis fails -- so measure the weight-to-KV ratio at target load first.\n\n"
"Controlled experiment: three configurations at equal total GPU count -- N independent replicas, N/2 replicas "
"at TP=2, N/4 at TP=4 -- same trace, 3 repeats. Record OOM-free concurrency per GPU, goodput, p99 latency, and "
"per-rank memory breakdown including NCCL buffers. On multi-node arms also record NIC counters and any "
"retransmits, since those confound latency measurements.\n\n"
"Mitigations for isolation specifically: per-request memory admission so one pathological prompt cannot take a "
"group down; health checks that detect a poisoned collective and restart the whole group deterministically "
"rather than leaving hung ranks; and circuit-breaking at the router on repeated group failures.\n\n"
"Confounders: NUMA and PCIe topology differences between nodes, unequal NIC/GPU affinity, and cold prefix "
"caches on freshly started groups.\n\n"
"Evidence required: weight-to-KV ratio at target load, per-rank budget including comm buffers, the "
"three-configuration table, and fabric counters for multi-node arms.\n\n"
"Rollback gates: revert a parallelism change if goodput per GPU falls, if p99 regresses beyond SLO, or if a "
"single-rank OOM is observed to hang rather than cleanly restart the group -- an unclean failure path is a "
"blocking defect independent of performance.",
]

assert len(ANSWERS) == N

recs = []
for r, ans in zip(sl, ANSWERS):
    m = r['messages']
    u = [x['content'] for x in m if x['role'] == 'user'][0]
    a = [x['content'] for x in m if x['role'] == 'assistant'][0]
    recs.append({
        "source_id": r['id'],
        "teacher_lane": "teacher-B",
        "teacher_model": "claude-opus-5-current",
        "calibration_status": "provisional",
        "decision": "rewrite",
        "source_user": u,
        "source_assistant": a,
        "corrected_answer": ans,
        "quality_dimensions": {
            "technical_correctness": 3,
            "instruction_coverage": 2,
            "operational_safety": 3,
        },
        "risks": [
            "source assistant text is a grading rubric, not an answer; training on it teaches meta-commentary instead of infrastructure reasoning",
            "no explicit falsifiable hypothesis or controlled experiment despite the prompt demanding both",
            "omits the boundary condition that a preallocated paged KV pool should preempt rather than OOM, so out-of-pool consumers go unconsidered",
            "no rollback gate or quality gate for KV quantization, which changes model outputs",
        ],
        "evidence_required": [
            "exact CUDA OOM message including attempted block size and free/reserved memory",
            "torch max_memory_allocated vs max_memory_reserved time series (fragmentation gap)",
            "engine KV-pool utilization and preemption/recompute counters at failure time",
            "per-PID nvidia-smi snapshot proving device exclusivity",
            "prompt-length distribution and prefill/decode token share from the replayed trace",
            "concurrency sweep table with predicted vs measured peak memory",
        ],
        "confidence": 0.62,
    })

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, 'w') as f:
    for x in recs:
        f.write(json.dumps(x, ensure_ascii=False) + '\n')
print('WROTE', OUT, len(recs))
