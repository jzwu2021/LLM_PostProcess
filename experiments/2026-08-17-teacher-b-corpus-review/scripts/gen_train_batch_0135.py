import json

CORPUS = "research/ai-infra-expert/corpus/train.jsonl"
OUT = "experiments/2026-08-17-teacher-b-corpus-review/results/train-batch-0135.jsonl"
START, N = 1340, 10

ANGLES = {
    "Troubleshooting": "Lifetime accounting: which bytes are freed at request completion and which are not",
    "Performance Analysis": "Failure-time attribution: splitting device HBM into weights / KV / workspace / foreign before tuning anything",
    "System Design": "Contract design: making the unbounded generation tail a declared, enforceable input",
}

COMMON_HEAD = """## Assumptions (declared, not measured)
- Single node, 8x NVIDIA A30 24 GB HBM2, PCIe Gen4 x16, no NVLink. One paged-KV serving replica per device unless tensor parallelism (TP) is stated.
- The KV pool is sized once at process start from HBM remaining after weights, CUDA context (~0.5-1.0 GB/device), activation workspace and communicator buffers. It does not grow later.
- "Intermittent after several concurrent requests" is read as load-correlated, not input-correlated: no single request reproduces it in isolation.
- Units: HBM GiB, bandwidth GB/s, latency ms, throughput tokens/s. Every number below is a measurement target, not a claimed result for this cluster.

## Byte budget that must be written down before any tuning
HBM_total = weights + CUDA_context + activation_workspace_peak + KV_pool + foreign_consumers
per_token_KV_bytes = 2 * num_layers * num_kv_heads * head_dim * dtype_bytes / TP_degree
max_concurrent_tokens = KV_pool_bytes / per_token_KV_bytes
A request holds (P + G) tokens for its entire lifetime, so the binding quantity is sum(P_i + G_i) over in-flight requests, not the in-flight request count. Note the asymmetry: P is known at admission, G is not. Any capacity claim that does not bound G is unfalsifiable.
"""

HYP = {
    "Troubleshooting": (
        "H1 (primary): nothing leaks; the resident set simply outlives your mental model of it. KV blocks for a request are held until the final token, and requests without an explicit max_tokens run to the model limit, so a handful of slow-finishing long generations pin the pool while new arrivals are still admitted.\n"
        "Falsifier: if H1 holds, at every failure timestamp the set of in-flight requests must contain at least one request older than the p95 request duration, and the free-block count must decay monotonically over the preceding window rather than dropping in a single step. Forcing a server-side max_tokens ceiling equal to the observed p99 of G must then eliminate OOM without any engine change. If failures occur while all in-flight requests are young and free blocks collapse in one sampling interval, H1 is refuted and the cause is a single large allocation, not accumulation."
    ),
    "Performance Analysis": (
        "H1 (primary): the failing allocation is not a KV block at all. It is the prefill attention/logits workspace for the longest prompt in the batch, whose peak is proportional to prompt length (and to vocab size for the final logits projection), and it must be found in free HBM *outside* the pre-reserved KV pool while decode traffic already holds that pool.\n"
        "Falsifier: if H1 holds, the CUDA memory snapshot at failure must show the failing allocation size roughly tracking the longest admitted prompt, and free KV blocks must be non-zero at that instant. Chunked prefill (bounding tokens processed per prefill step) must then remove the failures at unchanged admitted load. If free KV blocks are ~0 at failure, H1 is refuted and this is plain pool exhaustion."
    ),
    "System Design": (
        "H1 (primary): the system has no admission contract over generated length, so capacity is planned against a quantity the callers never declare. The deployment is therefore correct on average and undefined on the tail; OOM is the expected behaviour of an unbounded input, not a bug in the engine.\n"
        "Falsifier: if H1 holds, making max_tokens mandatory at the gateway and budgeting the scheduler on sum(P_i + max_tokens_i) <= 0.8 * max_concurrent_tokens must drive device OOM to zero, with the load shifting into a measurable queue-wait and 429 rate instead. If OOM persists under a hard, enforced token budget, the accounting itself is wrong (foreign consumer or workspace spike) and the design hypothesis is refuted."
    ),
}

FOCUS = {
    "Troubleshooting": """## Prioritized diagnosis (stop at the first hypothesis the data actually supports)
1. Establish which OOM you have. Device OOM (CUDA allocator) and host OOM (cgroup / oom-killer in dmesg) present identically to a caller behind a load balancer and have unrelated fixes. Check dmesg and the container memory cgroup before touching serving config.
2. Reconstruct the in-flight set at each failure timestamp from request logs: arrival time, P, max_tokens, completion time. If long-lived requests are present at every failure, accumulation is live; if not, it is a single-allocation spike.
3. Sample free KV blocks at >= 1 Hz. Monotonic decay before failure => lifetime/accumulation. Single-interval collapse => one big allocation.
4. Diff nvidia-smi HBM used against framework-accounted bytes (allocated + reserved). A persistent gap means a foreign consumer on the device: a second process, a stale training job holding context, MPS, or a profiling agent. This silently shrinks the pool the engine sized at startup and makes every earlier calculation wrong.
5. Only now test for a real leak: drain to idle and compare torch.cuda.memory_allocated against the cold baseline. If it returns to baseline, there is no leak and any "leak fix" would be a placebo.

## Mitigations ordered by reversibility and blast radius
1. Server-side max_tokens ceiling and mandatory client max_tokens (config/gateway only, revert in seconds). This is what makes G bounded and capacity computable. Cost: long generations get truncated; that must be a product decision, recorded, not a silent change.
2. Token-budget admission at 0.8 * max_concurrent_tokens. Converts a correlated replica-wide crash into isolated, retryable 429s. Strictly better operationally, but it moves the failure into a metric you must now watch.
3. Verify preemption/recompute (or KV swap) is enabled so an over-budget request yields instead of the process dying. Cost: latency spikes on preempted requests - measure them, do not assume they are small.
4. Chunked prefill to cap the workspace peak independently of the longest prompt.
5. expandable_segments allocator setting for fragmentation; config-only but re-baseline throughput because allocator behaviour changes.
6. KV quantization (FP8/INT8) last, because it changes numerics: gate on a fixed accuracy eval with a pre-agreed threshold, never on memory numbers alone.
7. TP across devices to enlarge effective KV per replica: on a PCIe-only A30 node this adds an all-reduce per layer over PCIe and can cost double-digit percent of decode throughput. It is a capacity purchase with a latency bill, not a free win.""",

    "Performance Analysis": """## Measurements required before changing anything (with units)
- per_token_KV_bytes from the served config, reported in KB/token, and the derived max_concurrent_tokens in tokens.
- Full HBM decomposition per device at steady state: weights GiB, CUDA context GiB, KV pool GiB, observed activation workspace peak GiB, unexplained remainder GiB. The remainder is the number that decides whether this is a serving problem or a co-tenancy problem.
- Free KV blocks and running/waiting queue depth at >= 1 Hz, timestamp-aligned with arrivals and failures.
- torch.cuda.memory_allocated vs memory_reserved at the same cadence, plus a CUDA memory snapshot captured at the failure, so the *size* of the failing allocation is known rather than guessed.
- P and G distributions (p50/p95/p99/max) from a captured production trace. The mean is irrelevant here; the tail is the load.
- Pre-change baseline of p50/p99 TTFT, inter-token latency, completed requests/s and 429 rate on the same replayed trace.

## Controlled experiment
Replay one fixed captured trace (identical content, identical arrival timestamps, fixed seed) against the current build. One variable per arm, each arm long enough to contain several p99 events:
- A: baseline.
- B: baseline + chunked prefill (workspace peak bounded).
- C: baseline + enforced max_tokens at observed p99 of G.
- D: baseline + token-budget admission at 0.8 * max_concurrent_tokens.
Primary metric: device OOM events per 10k requests. Guardrails: p99 TTFT, inter-token latency, completed requests/s, 429 rate, truncation rate. An arm counts as a fix only if OOM reaches zero AND no guardrail crosses its pre-agreed threshold.

## Confounders that can manufacture a false win
- Admission control and max_tokens ceilings convert crashes into 429s and truncations; scoring only OOM count will declare success while quietly dropping or shortening traffic.
- Prefix-cache benefit is entirely trace-dependent; a synthetic trace with shared system prompts overstates it by a wide margin.
- Warm-up, kernel autotuning and cache population make the first minutes unrepresentative - discard them explicitly and state the discard window.
- Thermal and power capping on a dense 8x A30 chassis shifts latency between arms for reasons unrelated to memory; log per-device SM clock and power draw per arm.
- A foreign process present in only some arms changes the pool size and invalidates the comparison; verify device occupancy at the start and end of every arm.""",

    "System Design": """## Design changes, each with the boundary condition that makes it valid
1. Make generated length a declared input: max_tokens mandatory at the gateway, with a server-side ceiling. Boundary: without this, sum(P + G) is unbounded and no admission policy is enforceable - every later item on this list depends on it.
2. Budget the scheduler on tokens, not requests: admit while sum(P_i + max_tokens_i) <= 0.8 * max_concurrent_tokens. Boundary: the 0.8 factor is a placeholder for measured workspace peak plus foreign-consumer slack; it must be derived from the HBM decomposition, not copied.
3. Segregate traffic classes: route long-context requests to dedicated replicas (or TP=2 replicas with larger effective KV), keep short interactive traffic on single-device replicas. Boundary: worth the fleet split only if long requests are a small share of volume but a large share of KV bytes - verify from the trace first.
4. Backpressure at the gateway with a bounded queue and explicit 429 + Retry-After, so overload degrades predictably instead of killing a replica and dumping all of its in-flight requests at once.
5. Disaggregated prefill/decode (NVIDIA Dynamo / Mooncake-style) isolates the prefill workspace spike from long-lived decode KV and lets each pool be sized for one job. Boundary condition: it requires shipping KV between prefill and decode workers. On a PCIe-only A30 node with no RDMA/GPUDirect path, that transfer traverses host memory and PCIe and can dominate TTFT. Measure KV bytes/request and achievable link bandwidth before adopting; disaggregation is not a memory fix you get for free.

## Rollback gates, agreed before deployment
- Revert if p99 TTFT regresses more than 15% against the recorded baseline on the same replayed trace.
- Revert if the 429 rate exceeds the agreed SLO error budget, even when OOM count is zero.
- Revert if the truncation rate caused by the max_tokens ceiling exceeds the product-agreed limit.
- Revert if any accuracy gate fails after KV quantization (fixed eval set, threshold fixed in advance).
- Automatic revert if OOM-caused replica restarts exceed the pre-change rate at any point during canary.
- Canary one replica for a full daily traffic cycle before fleet rollout; a change validated only at off-peak load is not validated.""",
}

RISKS = [
    "source_assistant is a grading rubric rather than an answer; training on it teaches the model to enumerate what a good answer would contain instead of producing one",
    "it never bounds generated length G, so the stated capacity checks are unfalsifiable and the OOM cannot be attributed",
    "no HBM decomposition, so KV pool exhaustion, prefill workspace spikes and a foreign process on the device are indistinguishable",
    "quantization and tensor parallelism are offered as mitigations with no accuracy gate and no PCIe interconnect cost, which is unsafe guidance on a no-NVLink A30 node",
    "no rollback threshold, so a change that removes OOM by mass-429ing or truncating traffic would be scored as a success",
]

EVIDENCE = [
    "served model config (num_layers, num_kv_heads, head_dim, dtype, TP degree) to compute per-token KV bytes",
    "per-device HBM decomposition: weights, CUDA context, KV pool, activation workspace peak, unexplained remainder",
    "free-KV-block and queue-depth time series at >= 1 Hz, timestamp-aligned to arrivals and failure events",
    "CUDA memory snapshot at failure giving the size of the failing allocation, plus memory_allocated vs memory_reserved",
    "P and G distributions (p50/p95/p99/max) from a captured production trace, not synthetic load",
    "nvidia-smi HBM time series as an out-of-process cross-check and dmesg to exclude host-side OOM kill",
    "pre-change baseline of p99 TTFT, inter-token latency, throughput, 429 rate and truncation rate on the same replayed trace",
]


def answer(cat, cid, variant):
    return (
        f"# Intermittent OOM under long-context concurrency ({cat}, {cid}, scenario variant {variant})\n"
        f"Primary analytical angle for this variant: {ANGLES[cat]}.\n\n"
        + COMMON_HEAD
        + "\n## Falsifiable hypothesis\n" + HYP[cat] + "\n\n"
        + FOCUS[cat]
        + "\n\n## What this answer deliberately does not claim\n"
          "No platform-specific constant (pool size, achieved PCIe bandwidth, prefix hit rate, safe utilisation factor) is asserted for this cluster. "
          "Every number above is a measurement target; using one as an input before measuring it turns this plan into a guess. "
          "Any mitigation deployed without its guardrail metric and rollback threshold is an unmeasured change, not a fix.\n"
    )


def main():
    recs = [json.loads(l) for l in open(CORPUS) if l.strip()][START:START + N]
    out = []
    for r in recs:
        msgs = {m["role"]: m["content"] for m in r["messages"]}
        cid = r["id"]
        cat = r["category"]
        su = msgs["user"]
        sa = msgs["assistant"]
        v = su.split("Scenario variant ")[1].split(":")[0]
        out.append({
            "source_id": cid,
            "teacher_lane": "teacher-B",
            "teacher_model": "claude-opus-5-current",
            "calibration_status": "provisional",
            "decision": "rewrite",
            "source_user": su,
            "source_assistant": sa,
            "corrected_answer": answer(cat, cid, v),
            "quality_dimensions": {
                "technical_correctness": 3,
                "instruction_coverage": 2,
                "operational_safety": 2,
            },
            "risks": RISKS,
            "evidence_required": EVIDENCE,
            "confidence": 0.62,
        })
    with open(OUT, "w") as f:
        for o in out:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    print("wrote", len(out), "->", OUT)
    print("ids:", [o["source_id"] for o in out])


main()
