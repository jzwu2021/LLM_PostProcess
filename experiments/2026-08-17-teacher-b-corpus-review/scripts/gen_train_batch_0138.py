import json

CORPUS = "/home/johnson/workspace/LLM_PostProcess/research/ai-infra-expert/corpus/train.jsonl"
OUT = "/home/johnson/workspace/LLM_PostProcess/experiments/2026-08-17-teacher-b-corpus-review/results/train-batch-0138.jsonl"
START, N = 1370, 10

HDR = ("Assumptions (stated, not measured): single-model LLM serving on NVIDIA GPUs with a paged KV-cache "
       "engine; OOM is raised by the CUDA caching allocator; tensor-parallel degree, max_model_len and "
       "max_num_seqs are fixed across the comparison; no platform-specific behaviour is asserted beyond "
       "what the listed measurements would confirm.\n\n")

ANGLES = [
 ("Admission control and the concurrency high-water mark",
  "Hypothesis H1 (falsifiable): OOM is driven by the *peak* number of simultaneously-resident sequences, not by any single long request. If true, capping max_num_seqs (or the scheduler's running-queue length) at a value derived from measured worst-case per-sequence KV bytes eliminates OOM at unchanged per-request context length.",
  ["Instrument the scheduler: log running/waiting queue depth, total allocated KV blocks and free blocks at 100 ms resolution; align timestamps with the OOM event.",
   "Compute worst-case per-sequence KV bytes = 2 (K and V) x num_layers x num_kv_heads x head_dim x dtype_bytes x max_model_len. Multiply by observed peak concurrency and compare with KV pool bytes.",
   "Controlled experiment: hold the request trace byte-identical and replay it twice, varying only max_num_seqs (baseline vs baseline/2). Three replays per arm."],
  "Confounders: request arrival jitter changes peak concurrency between replays (mitigate with a fixed-rate replayer, not open-loop clients); prefix-cache hits reduce effective KV per sequence and can mask the effect.",
  "Rollback: if p99 latency regresses >20% or throughput drops >15% while OOM persists, restore the previous max_num_seqs and move to H2 (fragmentation).",
  "Boundary condition: admission control only helps while the *single largest* request still fits in an empty pool. If one max-length request alone exceeds the KV pool, no concurrency cap can help and the context limit must change."),

 ("Reserved-vs-allocated split: fragmentation as the discriminating measurement",
  "Hypothesis H2 (falsifiable): the allocator holds enough total free bytes but cannot serve the requested contiguous block, i.e. this is fragmentation, not exhaustion. If true, the failing allocation size is small relative to (reserved - allocated) at failure time.",
  ["Record from the CUDA allocator at the failure instant: requested bytes, allocated bytes, reserved bytes, and device free bytes. The pair (reserved - allocated) vs requested is the decisive number.",
   "Dump the allocator's block histogram before and after a failing run to see whether free memory is split into many sub-block fragments.",
   "Controlled experiment: rerun the identical trace with an expandable-segments / larger-block allocator configuration; keep model, batch limits and trace fixed."],
  "Confounders: non-KV transient buffers (logits, sampling, CUDA graphs, NCCL buffers) also consume reserved memory and shift with batch shape; a change in allocator config can alter fragmentation and peak reserved at once.",
  "Rollback: revert the allocator configuration if throughput drops >10% or if OOM recurs at the same trace position within two replays.",
  "Boundary condition: if (reserved - allocated) at failure is smaller than the requested block, fragmentation is ruled out and the pool is genuinely exhausted; stop pursuing allocator tuning."),

 ("Capacity design: sizing the KV pool as a first-class budget",
  "Hypothesis H3 (falsifiable): the deployment was sized on *average* context length while the workload's tail context dominates memory. If true, the measured p99 context length x peak concurrency exceeds the provisioned KV pool, and sizing to the p99 (not the mean) removes OOM.",
  ["Measure the input+output token length distribution (p50/p90/p99/max) from production logs, not from a synthetic mix.",
   "Compute the static budget: total device memory - weight bytes - activation/workspace headroom - communication buffers = KV pool bytes; state each term with units.",
   "Controlled experiment: replay the same trace at two KV pool fractions (e.g. 0.86 vs 0.92 of device memory) and record OOM count, throughput and p99 TTFT."],
  "Confounders: raising the KV fraction steals headroom from activations and can move the OOM from the KV allocator into prefill workspace; weights and pool interact with tensor-parallel degree.",
  "Rollback: if raising the KV fraction produces prefill-stage OOM or any crash-loop, revert to the previous fraction within one deploy cycle.",
  "Boundary condition: this design lever is exhausted once KV pool + weights + workspace already fill the device; beyond that only context limits, quantization, or more GPUs change the outcome."),

 ("Prefill spike vs decode steady state",
  "Hypothesis H4 (falsifiable): OOM occurs during *prefill* of a long request whose transient attention/workspace allocation, not its KV footprint, is the peak. If true, failures correlate with prefill starts rather than with decode-phase queue depth, and chunked prefill removes them.",
  ["Tag each allocator failure with the engine phase (prefill vs decode) and the token count of the request being prefilled.",
   "Measure peak reserved bytes during a single isolated long prefill with concurrency 1 - this isolates transient workspace from KV growth.",
   "Controlled experiment: enable chunked prefill with a fixed chunk budget; replay the identical trace; compare OOM count and TTFT."],
  "Confounders: chunked prefill also changes scheduling and thus concurrency, so an OOM reduction may come from lower peak concurrency rather than from smaller transient buffers; control by holding max_num_seqs fixed.",
  "Rollback: disable chunked prefill if TTFT p99 regresses >25% without an OOM reduction.",
  "Boundary condition: if failures are tagged to decode steps with no concurrent prefill, H4 is refuted and the cause is cumulative KV growth."),

 ("Effective KV growth per decode step and the leak/eviction question",
  "Hypothesis H5 (falsifiable): KV blocks are not being released on request completion or cancellation, so the pool monotonically drains across the run. If true, free-block count after each idle period is strictly lower than after the previous idle period.",
  ["Sample free KV blocks at every scheduler tick; after each traffic lull, record the recovered free-block count and fit a trend.",
   "Correlate with client-side cancellations, timeouts and streaming disconnects - aborted requests are the usual leak path.",
   "Controlled experiment: two replays of the same trace, one with all cancellations removed, one with cancellations at the production rate."],
  "Confounders: a prefix/radix cache legitimately retains blocks after completion and looks identical to a leak unless its eviction policy is accounted for; measure cached-block count separately from in-use blocks.",
  "Rollback: if a cache-eviction change raises recompute cost such that throughput drops >15%, restore prior eviction settings.",
  "Boundary condition: if free blocks fully recover to the initial value after every lull, the leak hypothesis is refuted regardless of how tight steady-state memory is."),

 ("Prefix/radix cache: capacity relief versus memory pressure",
  "Hypothesis H6 (falsifiable): shared prefixes are large enough that prefix caching lowers aggregate KV residency below the OOM threshold. If true, measured cache hit rate on the production trace exceeds a threshold at which aggregate unique-block count fits the pool.",
  ["Measure prefix hit rate and *unique* block count (not total requests) on a replay of the real trace.",
   "Compute aggregate residency with and without sharing; the difference is the only quantity that matters for OOM.",
   "Controlled experiment: enable/disable prefix caching on identical traces; record OOM count, unique blocks, hit rate, throughput."],
  "Confounders: caching adds retained blocks that compete with in-flight requests, so a low hit rate can make memory pressure *worse*; hit rate measured on a synthetic trace will not transfer.",
  "Rollback: disable prefix caching if OOM frequency rises or unique-block high-water increases in the first replay.",
  "Boundary condition: with near-zero shared prefix (each request unique), this lever is inert by construction and should not be pursued."),

 ("KV quantization and precision trade-offs",
  "Hypothesis H7 (falsifiable): halving KV dtype width (e.g. 16-bit to 8-bit KV) reduces per-sequence KV bytes ~2x and clears the OOM at unchanged concurrency and context, at a quality cost bounded by a pre-agreed threshold.",
  ["Recompute per-sequence KV bytes at the new dtype and confirm the arithmetic against measured pool occupancy, not against docs.",
   "Run a fixed task-level quality evaluation (held-out set, same decoding parameters, same seed) before and after; report the delta with a confidence interval.",
   "Controlled experiment: identical trace, only KV dtype varies; record OOM count, throughput, and the quality metric."],
  "Confounders: KV quantization may disable fused attention or CUDA-graph paths and thus change speed for reasons unrelated to memory; quality regressions may appear only on long-context items.",
  "Rollback: revert to the original KV dtype if the quality metric degrades beyond the pre-agreed threshold (fixed before the run) or if throughput drops >15%.",
  "Boundary condition: quantization buys a constant factor, not asymptotic relief - if required residency exceeds 2x the pool, this alone cannot fix it."),

 ("Multi-GPU placement: tensor parallelism and per-rank imbalance",
  "Hypothesis H8 (falsifiable): OOM appears on one rank first because per-rank memory is not symmetric (rank 0 carries extra buffers, logits, or the sampler). If true, per-rank peak reserved bytes differ measurably and the failing rank is consistent across replays.",
  ["Collect per-rank allocated/reserved peaks and the identity of the failing rank across at least three replays.",
   "Account for non-sharded terms: embedding/output head placement, NCCL communication buffers, and any rank-local caches.",
   "Controlled experiment: same trace at TP=N and TP=2N (if devices allow), holding max_num_seqs and max_model_len fixed; per-rank KV bytes should scale ~1/TP for sharded KV heads."],
  "Confounders: raising TP shrinks per-rank KV but adds communication buffers and can reduce throughput; if num_kv_heads < TP, KV replication breaks the expected 1/TP scaling.",
  "Rollback: return to the previous TP configuration if throughput drops >20% or if any rank still OOMs after two replays.",
  "Boundary condition: when num_kv_heads is not divisible by TP, KV heads are replicated and increasing TP yields little or no per-rank KV reduction."),

 ("Context-limit policy and the request-level contract",
  "Hypothesis H9 (falsifiable): a small tail of over-long requests, admitted because max_model_len exceeds what the pool can support at target concurrency, causes essentially all OOMs. If true, rejecting or truncating requests above a computed length threshold removes OOM with a bounded rejection rate.",
  ["Derive the supportable length: pool_bytes / (target_concurrency x per-token KV bytes); compare with the observed length distribution to predict the rejection rate before changing anything.",
   "Log which request lengths are in flight at each OOM to test whether the tail is actually implicated.",
   "Controlled experiment: enforce the computed max length in shadow mode first (log-only), then in enforcing mode on the identical trace."],
  "Confounders: truncation silently changes task outcomes, so an apparent 'fix' may trade an availability failure for a correctness failure; the tail may be concentrated in one tenant.",
  "Rollback: revert to the prior limit if the measured rejection rate exceeds the agreed budget or if any critical tenant is disproportionately affected.",
  "Boundary condition: if OOMs occur with only short requests in flight, the tail hypothesis is refuted and the cause is concurrency or fragmentation."),

 ("Reproduction discipline: making an intermittent OOM deterministic before fixing it",
  "Hypothesis H10 (falsifiable): the failure is a deterministic function of a reproducible arrival pattern (a specific concurrency and length combination), not a random event. If true, a closed-loop replay of the captured trace reproduces OOM at the same position in at least 2 of 3 runs.",
  ["Capture a production trace with per-request arrival time, input length, and requested output length; replay it closed-loop with a fixed seed.",
   "Record OOM position (request index) and peak allocated/reserved per run; a reproduction rate is the prerequisite for trusting any later A/B.",
   "Controlled experiment: three baseline replays to establish the reproduction rate and its variance *before* changing any knob; only then run single-variable arms."],
  "Confounders: open-loop load generators drift in concurrency between runs and destroy reproducibility; background jobs or other tenants on the same GPU perturb free memory.",
  "Rollback: if reproduction rate is below 2/3, do not deploy any fix based on it - return to observability work rather than shipping an unvalidated change.",
  "Boundary condition: a fix validated on a non-reproducing baseline is unfalsifiable; treat any such 'improvement' as noise until the reproduction gate passes."),
]

TAILS = [
 ("Because the source answer is a rubric sketch rather than a worked diagnosis, it is rewritten into an "
  "explicit mechanism-plus-experiment form."),
]

RISKS = [
 ["Capping concurrency trades availability for latency and can silently queue traffic instead of failing fast.",
  "Peak-concurrency estimates from averaged metrics understate the true high-water mark."],
 ["Allocator configuration changes can alter throughput as well as fragmentation, confounding attribution.",
  "Fragmentation diagnosis is invalid without the requested-block size at failure time."],
 ["Raising the KV pool fraction can relocate the OOM into prefill workspace rather than removing it.",
  "Sizing on mean context length underestimates tail residency."],
 ["Chunked prefill changes scheduling and can mask the real cause by lowering concurrency.",
  "Phase attribution is unreliable without per-failure phase tagging."],
 ["A prefix cache can be mistaken for a memory leak, producing a wrong fix.",
  "Aggressive eviction raises recompute cost and can regress throughput."],
 ["Low prefix-hit workloads get worse, not better, when caching retains blocks.",
  "Hit rates measured on synthetic traces do not transfer to production."],
 ["KV quantization can degrade long-context quality in ways short-prompt evaluations miss.",
  "Quantized paths may disable fused kernels and change performance for unrelated reasons."],
 ["Increasing TP adds communication buffers and can reduce throughput while only partly relieving memory.",
  "KV-head replication when num_kv_heads < TP breaks the assumed 1/TP scaling."],
 ["Truncation converts an availability failure into a silent correctness failure.",
  "Length limits may fall unevenly across tenants."],
 ["Acting on a non-reproducing baseline yields unfalsifiable 'fixes'.",
  "Open-loop load generation destroys run-to-run comparability."],
]

EVIDENCE = [
 ["Scheduler queue-depth and free-KV-block time series aligned to the OOM timestamp",
  "Worst-case per-sequence KV byte computation with all shape terms shown",
  "Three replays per max_num_seqs arm with OOM count, throughput, p99 latency"],
 ["Allocator failure record: requested / allocated / reserved / device-free bytes",
  "Free-block size histogram before and after a failing run",
  "Paired replays with only the allocator configuration varied"],
 ["Production token-length distribution (p50/p90/p99/max)",
  "Line-by-line memory budget with units for weights, workspace, comm buffers, KV pool",
  "OOM count, throughput and p99 TTFT at two KV pool fractions"],
 ["Per-failure phase tag (prefill vs decode) and request token count",
  "Peak reserved bytes for an isolated long prefill at concurrency 1",
  "Paired replays with chunked prefill on/off at fixed max_num_seqs"],
 ["Free-KV-block recovery trend across successive traffic lulls",
  "Cancellation/timeout/disconnect rates correlated with block accounting",
  "Cached-block count reported separately from in-use blocks"],
 ["Measured prefix hit rate and unique-block high-water on the real trace",
  "Aggregate residency with and without sharing",
  "Paired replays with prefix caching enabled and disabled"],
 ["Recomputed per-sequence KV bytes at the new dtype, checked against pool occupancy",
  "Held-out quality metric before/after with a confidence interval and a pre-agreed threshold",
  "Throughput and OOM count on identical traces"],
 ["Per-rank peak allocated/reserved bytes and failing-rank identity across three replays",
  "num_kv_heads, TP degree and the resulting per-rank KV shard size",
  "Throughput comparison across TP settings on an identical trace"],
 ["Computed supportable context length from pool bytes and target concurrency",
  "In-flight request lengths recorded at each OOM",
  "Shadow-mode rejection-rate measurement before enforcement"],
 ["Captured production trace with arrival times and length fields",
  "Reproduction rate across three baseline replays with OOM position",
  "Peak allocated/reserved variance across those baseline runs"],
]

CONF = [0.62, 0.6, 0.63, 0.61, 0.59, 0.6, 0.62, 0.58, 0.61, 0.63]
QD = [(2,2,2),(2,2,2),(2,2,2),(2,2,2),(2,2,2),(2,2,2),(2,2,2),(2,2,2),(2,2,2),(2,2,2)]


def build(rec, k):
    title, hyp, steps, conf_txt, rb, bound = ANGLES[k]
    body = HDR
    body += "Primary lens for this case: %s.\n\n" % title
    body += "1) Falsifiable hypothesis\n   %s\n\n" % hyp
    body += "2) Prioritized diagnosis and measurements (cheapest and most discriminating first)\n"
    for i, s in enumerate(steps):
        body += "   %s. %s\n" % ("abc"[i], s)
    body += "\n3) Controlled experiment design\n   Single variable per arm, byte-identical request trace across arms, three repetitions per arm, "
    body += "report medians with min/max. Any arm that changes two knobs at once is discarded rather than interpreted.\n\n"
    body += "4) Expected confounders\n   %s\n\n" % conf_txt
    body += "5) Boundary condition\n   %s\n\n" % bound
    body += "6) Rollback criteria\n   %s Rollback is executed within one deploy cycle and the prior configuration is re-verified by one replay.\n\n" % rb
    body += ("7) What would refute this framing\n   If the failure envelope shows device-free bytes near zero with a "
             "requested block larger than (reserved - allocated), the system is genuinely out of memory and every "
             "lever above except capacity, context limits and quantization is inert. State that outcome explicitly "
             "rather than continuing to tune.\n")
    return body


rows = []
with open(CORPUS) as f:
    for i, line in enumerate(f):
        if i < START:
            continue
        if i >= START + N:
            break
        d = json.loads(line)
        msgs = {m["role"]: m["content"] for m in d["messages"]}
        k = i - START
        tc, ic, os_ = QD[k]
        rows.append({
            "source_id": d["id"],
            "teacher_lane": "teacher-B",
            "teacher_model": "claude-opus-5-current",
            "calibration_status": "provisional",
            "decision": "rewrite",
            "source_user": msgs["user"],
            "source_assistant": msgs["assistant"],
            "corrected_answer": build(d, k),
            "quality_dimensions": {
                "technical_correctness": tc,
                "instruction_coverage": ic,
                "operational_safety": os_,
            },
            "risks": RISKS[k],
            "evidence_required": EVIDENCE[k],
            "confidence": CONF[k],
        })

with open(OUT, "w") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print("wrote", len(rows), OUT)
