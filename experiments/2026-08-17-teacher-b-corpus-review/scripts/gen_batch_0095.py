import json

IN = "/tmp/tb_batch_in.jsonl"
OUT = "/home/johnson/workspace/LLM_PostProcess/experiments/2026-08-17-teacher-b-corpus-review/results/train-batch-0095.jsonl"

EMPHASIS = {
    "Performance Analysis": (
        "Emphasis for this variant is quantitative performance attribution: every reported number must be tied to a "
        "roofline-style expectation. Decode throughput per replica is bounded by weight+KV traffic per token: "
        "t_decode >= (bytes_weights + bytes_KV_read) / achievable_HBM_BW. On an A30 (933 GB/s peak, expect 0.6-0.75 of peak "
        "achievable) a 9B model in bf16 (~18 GB of weights) implies a hard floor of roughly 18/0.65e3 s ~= 28 ms per decode step "
        "at batch 1, so a measured TPOT far above that floor means the bottleneck is scheduling, not bandwidth. Prefill is "
        "compute-bound: FLOPs ~= 2 * params * prompt_tokens, so prefill time should scale linearly in total batched prompt tokens "
        "until the batch fills the SMs. Report measured-vs-model ratios, not raw latencies alone; an unexplained gap is the finding."
    ),
    "System Design": (
        "Emphasis for this variant is the design of the harness itself so the measurement is decision-grade. Separate the "
        "load generator from the serving host, drive open-loop (Poisson or replayed arrivals) rather than closed-loop, because "
        "closed-loop load generators self-throttle under queueing and silently hide the very tail you are trying to measure. "
        "Define request classes up front (short-prompt/short-gen, short-prompt/long-gen, long-prompt/short-gen) and report "
        "percentiles per class as well as pooled, since a pooled P99 over a bimodal mixture is a mixing artifact and not a "
        "service-level fact. Pin one configuration hash per arm (model revision, quantization, TP/PP degree, KV dtype, "
        "max_num_seqs, max_num_batched_tokens, chunked-prefill on/off, scheduler policy) and treat any unpinned knob as a confound."
    ),
    "Troubleshooting": (
        "Emphasis for this variant is diagnostic discrimination: the plan must be able to tell competing failure hypotheses apart. "
        "If P99 degrades while median holds, the candidate causes are (a) head-of-line blocking from long prefills sharing steps "
        "with decode, (b) KV-cache pressure causing preemption and recompute, (c) admission queueing at saturation, (d) an "
        "infrastructure artifact such as power/clock capping or a noisy neighbour. These are separable: (a) is falsified by enabling "
        "chunked prefill and observing TTFT-of-short-requests improve while TPOT is unchanged; (b) is confirmed by nonzero "
        "preemption/recompute counters correlated in time with the tail; (c) is confirmed by queue-depth time series leading the "
        "latency spike; (d) is confirmed by SM-clock or power-limit telemetry. Collect the counters that discriminate before "
        "changing any knob."
    ),
}

BASE_HEAD = (
    "Scope and framing. The system under test is an online LLM serving deployment with continuous batching "
    "(vLLM/SGLang/Dynamo-class scheduler), serving a mixed workload of short prompts with long generations and short "
    "generations. The deliverable is an evaluation plan that reports TTFT, TPOT, throughput, queueing delay and P99 "
    "end-to-end latency such that another engineer can reproduce the numbers and, more importantly, falsify the conclusion.\n\n"
    "1. Stated assumptions (must be recorded with the results, not implied)\n"
    "- Prefill and decode are distinct regimes and must never be pooled into one 'latency' metric. Prefill is compute-bound and "
    "scales with total batched prompt tokens; decode is HBM-bandwidth- and KV-capacity-bound and scales with the number of "
    "concurrently decoding sequences.\n"
    "- All timestamps are taken from a single server-side clock source (arrival, admission, first token emitted, completion) so "
    "queueing delay = admission_time - arrival_time can be computed exactly rather than inferred.\n"
    "- Model revision, quantization, tensor-parallel degree, KV dtype and scheduler flags are pinned per arm and recorded as a "
    "config hash; sampling parameters (temperature, max_tokens, stop conditions) are fixed because output length is the dominant "
    "driver of end-to-end latency variance.\n"
    "- The network path and the load generator are not the bottleneck; this is verified, not assumed (see step 5).\n\n"
    "2. Metric definitions (ambiguity here is the most common source of non-reproducible results)\n"
    "- TTFT = first_token_time - arrival_time, and it is reported split into queueing delay and prefill service time. Reporting "
    "TTFT without that split hides head-of-line blocking.\n"
    "- TPOT = (completion_time - first_token_time) / (output_tokens - 1), reported per request and aggregated by request class.\n"
    "- Throughput is reported twice: goodput in completed requests/s and in output tokens/s. Output tokens/s alone can be raised "
    "by admitting more concurrency while every individual user gets slower, so it is not a standalone success metric.\n"
    "- P99 is reported per request class with bootstrap 95% confidence intervals, and the sample count per class is reported; a "
    "P99 from fewer than ~1000 samples per class is not a stable estimate and should be labelled as such.\n\n"
    "3. Falsifiable hypothesis (the point of the experiment)\n"
    "H0: at the target arrival rate, P99 end-to-end latency degradation relative to the unloaded baseline is driven mainly by "
    "queueing/admission delay, not by per-token decode slowdown. Operationally: under H0, as load rises from 0.5x to 1.0x of the "
    "measured saturation rate, the queueing component of TTFT grows by more than 3x while median TPOT grows by less than 20%.\n"
    "H1 (the alternative): decode-side contention dominates, i.e. median TPOT grows by more than 20% while the queueing component "
    "stays under 3x.\n"
    "The experiment is designed so that at least one of H0/H1 must be rejected by the collected traces; if neither is cleanly "
    "rejected, the result is reported as inconclusive rather than reinterpreted after the fact.\n\n"
    "4. Controlled experiment design\n"
    "- Arms: (A) baseline scheduler configuration; (B) one single-variable change (for example chunked prefill enabled, or "
    "max_num_seqs reduced). Exactly one variable changes per arm; multi-knob changes make attribution impossible.\n"
    "- Load levels: sweep open-loop arrival rate at approximately 0.25x, 0.5x, 0.75x, 0.9x and 1.1x of the empirically located "
    "saturation point (the rate at which queue depth grows without bound over a 5-minute window). The over-saturated point is "
    "included deliberately to characterise degradation mode, and is run last.\n"
    "- Workload: a fixed, seeded request trace replayed identically across all arms and load levels, with the prompt- and "
    "output-length distributions recorded. The same trace file, not a re-sampled one, is what makes arms comparable.\n"
    "- Repetition: at least 3 independent trials per (arm, load level), each with an explicit warmup (discard the first 60-120 s "
    "or the first N requests until KV-cache occupancy and step time stabilise) and warmup data excluded from all statistics.\n"
    "- Randomize or counterbalance the run order of arms to avoid confounding with thermal drift or background cluster activity.\n\n"
)

BASE_TAIL = (
    "\n5. Expected confounders and how each is controlled\n"
    "- Closed-loop load generation: self-throttles and hides the tail. Controlled by driving open-loop and by verifying the "
    "generator kept up (recorded scheduling lag of the generator itself).\n"
    "- Output-length drift between arms: a change that shortens generations will look like a throughput win. Controlled by "
    "greedy/fixed sampling and by asserting the output-token histogram is statistically indistinguishable across arms.\n"
    "- Cold caches and autoscaling: first-run CUDA graph capture, page-cache warmth, and replica scale-up all distort early "
    "samples. Controlled by warmup and by pinning replica count.\n"
    "- Power/thermal capping and noisy neighbours: an A30-class GPU under sustained load can drop clocks. Controlled by logging "
    "SM clocks, power draw and throttle reasons at >=1 Hz and discarding or flagging intervals with throttling.\n"
    "- Prefix caching: repeated prompts in a synthetic trace can be served from cached prefixes, inflating throughput relative to "
    "production. Controlled by reporting prefix-cache hit rate and, if it exceeds the production rate, by de-duplicating prompts.\n"
    "- Multi-tenant interference on the same host (other containers, other replicas) is controlled by exclusive scheduling for the "
    "duration of the run, or by reporting it as an uncontrolled factor.\n\n"
    "6. Evidence to collect (without these, conclusions are not supportable)\n"
    "- Per-request trace: arrival, admission, first-token and completion timestamps, prompt and output token counts, request class.\n"
    "- Scheduler counters per step: queue depth, running batch size, batched prompt tokens, KV-cache utilization, preemption and "
    "recomputation counts.\n"
    "- GPU telemetry at >=1 Hz: SM utilization, HBM used, power, clocks, throttle reasons.\n"
    "- The seeded trace file, the config hash per arm, the server version/commit, and the raw per-trial outputs.\n\n"
    "7. Safety, blast radius and rollback criteria\n"
    "- Run load tests on a dedicated replica or a mirrored environment; if production traffic is used, use shadow/mirrored traffic "
    "with responses discarded, never live user requests.\n"
    "- Pre-declare abort thresholds before the run: abort immediately if error rate exceeds 1%, if P99 exceeds the SLO by more "
    "than 2x for over 60 s, or if any OOM or replica restart occurs.\n"
    "- Any configuration change validated by this experiment is rolled out canary-first (single replica, >=30 minutes, compare "
    "against control replicas on the same traffic slice) with an automatic revert on SLO regression.\n"
    "- Rollback is a single config-hash revert plus replica restart; the revert path is exercised once in the test environment "
    "before the canary, so it is a measured procedure rather than an assumption.\n\n"
    "8. What would make me change this plan\n"
    "If the per-request traces show that queueing delay and TPOT rise together in lockstep at every load level, neither H0 nor H1 "
    "is cleanly separable with this instrumentation, and the next step is finer-grained per-step tracing (time in prefill vs decode "
    "per scheduler step) rather than more load levels. Reporting an inconclusive result with the counters that would resolve it is "
    "the correct outcome, not a narrative built from aggregate latency curves."
)

RISKS_BASE = [
    "source_assistant is a grading rubric describing what an answer should contain, not an answer; supervising on it teaches meta-commentary instead of engineering reasoning",
    "no concrete falsifiable hypothesis with numeric decision thresholds is stated, despite the prompt explicitly requiring one",
    "no rollback criteria, abort thresholds or blast-radius control, so a naive reader could run this load test against production replicas",
    "pooling P99 across a bimodal short/long request mixture yields a mixing artifact rather than a service-level fact",
    "queueing delay is not separated from prefill service time inside TTFT, hiding head-of-line blocking",
]

EVID_BASE = [
    "per-request trace with arrival, admission, first-token and completion timestamps plus prompt/output token counts and request class",
    "scheduler counters per step: queue depth, running batch size, batched prompt tokens, KV-cache utilization, preemption and recomputation counts",
    "GPU telemetry at >=1 Hz (SM utilization, HBM used, power, clocks, throttle reasons) to rule out power or thermal capping",
    ">=3 repeated trials per arm and load level with bootstrap 95% confidence intervals and reported per-class sample counts",
    "pinned config hash per arm covering model revision, quantization, TP degree, KV dtype, max_num_seqs, max_num_batched_tokens and chunked-prefill setting",
    "the seeded request trace file and the load generator's own scheduling lag, to prove the generator was not the bottleneck",
]

EXTRA_RISK = {
    "Performance Analysis": "no roofline or bandwidth-based expected value is given, so a measured TPOT cannot be judged as reasonable or anomalous",
    "System Design": "the harness topology is unspecified; a closed-loop generator co-located with the server would invalidate every tail-latency number",
    "Troubleshooting": "the plan does not name competing failure hypotheses, so the collected data may not discriminate head-of-line blocking from KV-cache preemption",
}

EXTRA_EVID = {
    "Performance Analysis": "an analytic bandwidth/FLOP model of the deployed model size so measured-vs-expected ratios can be computed per load level",
    "System Design": "the arrival-process specification (open-loop Poisson or replayed trace) and evidence the load generator ran on a separate host",
    "Troubleshooting": "time-aligned overlay of queue depth, preemption count and P99 latency so lead-lag relationships can be read directly",
}

rows = []
for line in open(IN):
    d = json.loads(line)
    msgs = {m["role"]: m["content"] for m in d["messages"]}
    cat = d["category"]
    ans = BASE_HEAD + "4b. Category-specific emphasis (" + cat + ")\n" + EMPHASIS[cat] + "\n" + BASE_TAIL
    rows.append({
        "source_id": d["id"],
        "teacher_lane": "teacher-B",
        "teacher_model": "claude-opus-5-current",
        "calibration_status": "provisional",
        "decision": "rewrite",
        "source_user": msgs["user"],
        "source_assistant": msgs["assistant"],
        "corrected_answer": ans,
        "quality_dimensions": {
            "technical_correctness": 3,
            "instruction_coverage": 2,
            "operational_safety": 2,
        },
        "risks": RISKS_BASE + [EXTRA_RISK[cat]],
        "evidence_required": EVID_BASE + [EXTRA_EVID[cat]],
        "confidence": 0.72,
    })

with open(OUT, "w") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print("wrote", len(rows), OUT)
