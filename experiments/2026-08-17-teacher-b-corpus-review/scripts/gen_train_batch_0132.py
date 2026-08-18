import json

CORPUS = "research/ai-infra-expert/corpus/train.jsonl"
OUT = "experiments/2026-08-17-teacher-b-corpus-review/results/train-batch-0132.jsonl"
START, N = 1310, 10

COMMON_HEAD = """## Assumptions (declared, not measured)
- Single node, 8x NVIDIA A30 24 GB HBM2, PCIe Gen4 x16, no NVLink bridge assumed. One paged-KV serving replica per device unless tensor parallelism (TP) is stated.
- The engine sizes its KV pool once at startup from HBM left over after weights, CUDA context (~0.5-1.0 GB/device), activation workspace and NCCL/communicator buffers.
- "Intermittent after several concurrent requests" means the failure is correlated with concurrency and prompt length, not with cold start, and is reproducible only under load.

## Capacity model used throughout (compute before you tune)
per_token_KV_bytes = 2 (K and V) * num_layers * num_kv_heads * head_dim * dtype_bytes / TP_degree
max_concurrent_tokens = KV_pool_bytes / per_token_KV_bytes
A request of P prompt tokens and G generated tokens holds (P + G) tokens for its whole lifetime, so the binding quantity is the sum of (P+G) over in-flight requests, not the request count. Any claim of "we have enough memory" that is not this arithmetic is an opinion.
"""

def answer(cat, cid, variant):
    hyp = {
        "Troubleshooting": (
            "H1 (primary): the OOM is *admission* oversubscription, not a leak. Concurrent in-flight (P+G) token sum transiently exceeds max_concurrent_tokens, and the engine's preemption/recompute path is either disabled or too slow, so a cudaMalloc for a new block fails.\n"
            "Falsifier: if H1 holds, the free-KV-block time series must reach ~0 in the sampling window immediately preceding every failure, and clamping total admitted tokens to 0.8 * max_concurrent_tokens must drive OOM count to zero without any code change. If OOM still occurs while free blocks stay above ~10%, H1 is refuted and the cause is outside the KV pool (fragmentation, activation spike, or a non-KV allocation)."
        ),
        "Performance Analysis": (
            "H1 (primary): caching allocator *fragmentation* outside the KV pool. torch.cuda.memory_reserved keeps climbing while memory_allocated returns to baseline after each request, so a large contiguous activation buffer (prefill attention workspace for the longest prompt) eventually cannot be served even though total free bytes look sufficient.\n"
            "Falsifier: if H1 holds, (reserved - allocated) at failure must exceed the size of the failing allocation, and enabling expandable_segments (PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True) must reduce OOM rate by a large margin at identical admitted load. If reserved ~= allocated at failure, fragmentation is refuted and the workload is simply over capacity."
        ),
        "System Design": (
            "H1 (primary): the deployment has no *length-aware* admission contract. A single tail request (p99 prompt) can consume the KV budget of many median requests, so the system is designed for average load while failing on a mixture it never bounded.\n"
            "Falsifier: if H1 holds, failure timestamps must be dominated by windows containing at least one prompt above the p95 length, and a token-budget scheduler (bound sum of P+max_tokens, not request count) must eliminate OOM while leaving p99 TTFT for median requests within the pre-change baseline. If OOM persists under a hard token budget, the design hypothesis is refuted."
        ),
    }[cat]

    focus = {
        "Troubleshooting": """## Prioritized diagnosis (stop at the first that explains the data)
1. Exhaust the cheap explanations first: confirm it is a *device* OOM, not a host OOM. Check dmesg for oom-killer and the container memory cgroup; a killed worker looks like a serving failure but has a different fix.
2. Correlate every failure timestamp against free-KV-block count sampled at >= 1 Hz. Near-zero free blocks at failure => capacity/admission. Non-zero free blocks at failure => allocator or non-KV allocation.
3. Separate prefill from decode. Prefill for one long prompt allocates a large attention/logits workspace proportional to prompt length (and to vocab size for the final logits); this spike is *not* in the KV pool and is the classic cause of "OOM only when a long request arrives during steady decode".
4. Check for multi-tenancy on the same device: a second process, an idle-but-resident training job, MPS, or a monitoring/profiling agent silently reduces the pool the engine assumed at startup.
5. Only after 1-4 do you look for a leak. A real leak shows monotonically rising allocated bytes across a drain-to-idle cycle; if allocated returns to baseline when idle, there is no leak.

## Mitigations, ordered by risk (cheapest and most reversible first)
1. Admission control / token budget (config only, instantly revertible): cap concurrent scheduled tokens at 0.8 * max_concurrent_tokens and cap max_model_len to the p99 you actually serve. Cost: some requests queue or get 429 instead of crashing the replica. This trades a correlated multi-request failure for an isolated, retryable one, which is strictly better operationally.
2. Enable/verify preemption + recompute or KV swap so an over-budget request yields instead of killing the process. Cost: latency spikes on preempted requests; must be measured, not assumed.
3. expandable_segments allocator setting. Config-only, revertible, but re-baseline throughput because allocator behaviour changes.
4. Prefix caching if the traffic is genuinely prefix-heavy. Measure the actual prefix hit rate first; on low-sharing traffic it adds bookkeeping and evictions for no gain.
5. KV quantization (FP8/INT8) or GQA-friendly model variant: last, because it changes numerics. Gate on an accuracy check, not on memory numbers alone.
6. Tensor parallelism to split KV across devices: on a PCIe-only A30 node this adds all-reduce on every layer over PCIe and can cost double-digit percent of decode throughput. It is a capacity fix with a real latency bill; do not present it as free.""",

        "Performance Analysis": """## Measurements to take before changing anything (all with units)
- per_token_KV_bytes computed from the served config; state the number in KB/token and the derived max_concurrent_tokens in tokens.
- Free KV blocks and running/waiting queue depth, >= 1 Hz, timestamp-aligned with request arrivals and with failures.
- torch.cuda.memory_reserved vs memory_allocated (bytes) sampled at the same cadence, plus a memory snapshot captured at the failure.
- nvidia-smi HBM used (MiB) as an out-of-process cross-check; a gap between framework-accounted bytes and nvidia-smi means a second consumer on the device.
- Prompt/generation length distribution from a captured production trace: p50/p95/p99/max of P and of G. Averages are useless here; the tail is the load.
- Baseline p50/p99 TTFT, inter-token latency and error rate on the replayed trace, recorded *before* any mitigation.

## Controlled experiment
Replay one fixed captured trace (same seed, same arrival timestamps, same content) against the current build. Change exactly one variable per arm and run each arm long enough to cover several p99 events:
- A: baseline.
- B: baseline + token-budget admission at 0.8 * max_concurrent_tokens.
- C: baseline + expandable_segments.
- D: baseline + max_model_len clamped to observed p99 prompt length.
Primary metric: OOM events per 10k requests. Guardrail metrics: p99 TTFT, inter-token latency, completed-requests/s, 429 rate. A mitigation only "works" if OOM goes to zero *and* no guardrail regresses beyond its pre-agreed threshold.

## Expected confounders (each one can fake a win)
- Admission control converts crashes into 429s; if you only count OOM you will declare victory while dropping traffic.
- Prefix cache hit rate depends on trace content; a synthetic trace with shared prefixes overstates its benefit dramatically.
- Warm-up and autotuning make the first minutes unrepresentative; discard them explicitly.
- Thermal/power capping on a dense 8x A30 chassis shifts latency between arms independently of memory. Log per-device clocks and power to rule this out.
- A co-resident process appearing only in some arms changes the pool size; verify device occupancy per arm.""",

        "System Design": """## Design changes, with the boundary conditions that make each valid
1. Make capacity explicit: publish max_concurrent_tokens and enforce a scheduler budget on the sum of (P + max_tokens) for in-flight requests. Boundary: this only holds if max_tokens is required on every request; without it the tail is unbounded and no budget is enforceable.
2. Segregate traffic classes. Route long-context requests to a dedicated replica (or a TP=2 replica with a larger effective KV pool) and keep short interactive traffic on single-device replicas. Boundary: worth it only if long requests are a small fraction of volume but a large fraction of KV bytes; verify from the trace before splitting fleets.
3. Reserve headroom rather than maximizing utilization: size the pool for p99 concurrency, not mean. Boundary: headroom is wasted money if the arrival process is not bursty; justify with the measured burst factor.
4. Backpressure at the gateway with a queue and explicit 429 + Retry-After, so overload degrades predictably instead of killing a replica and dumping all its in-flight requests.
5. Disaggregated prefill/decode (Dynamo/Mooncake-style) separates the large prefill workspace spike from the long-lived decode KV, and lets the KV pool be sized for decode alone. Boundary condition: it requires moving KV between prefill and decode workers; on a PCIe-only A30 node without RDMA/GPUDirect this transfer runs over host memory and PCIe and can dominate TTFT. Do not adopt it here without first measuring KV transfer bytes/request and the achievable link bandwidth.

## Rollback gates (decide these before deploying)
- Revert if p99 TTFT regresses more than 15% versus the recorded baseline on the same replayed trace.
- Revert if the 429 rate exceeds the agreed SLO budget, even if OOM count is zero.
- Revert if any accuracy gate fails after KV quantization (fixed eval set, pre-agreed threshold).
- Revert automatically if replica restarts caused by device OOM exceed the pre-change rate at any point in the canary window.
- Canary on one replica for a full traffic cycle before fleet-wide rollout; a mitigation validated only at low load is not validated.""",
    }[cat]

    return (
        f"# Intermittent OOM under long-context concurrency ({cat}, {cid}, scenario variant {variant})\n\n"
        + COMMON_HEAD
        + "\n## Falsifiable hypothesis\n" + hyp + "\n\n"
        + focus
        + "\n\n## What this answer deliberately does not claim\nNo platform-specific constant (pool size, achieved bandwidth, hit rate) is asserted here. Every number above must come from the measurements listed, on this cluster, before it is used in a decision. Any mitigation adopted without its guardrail metric is an unmeasured change, not a fix.\n"
    )


def main():
    recs = [json.loads(l) for l in open(CORPUS) if l.strip()][START:START + N]
    out = []
    for r in recs:
        msgs = {m["role"]: m["content"] for m in r["messages"]}
        cid = r["id"]
        cat = r["category"]
        variant = cid.split("-")[-1].lstrip("0")
        su = msgs["user"]
        sa = msgs["assistant"]
        v = su.split("Scenario variant ")[1].split(":")[0]
        risks = [
            "source_assistant is a grading rubric, not an answer; supervised training on it teaches the model to describe what a good answer would contain instead of producing one",
            "no KV capacity arithmetic, so oversubscription cannot be distinguished from allocator fragmentation or a non-KV activation spike",
            "quantization and tensor parallelism are listed as mitigations with no accuracy gate and no interconnect cost, which is actively unsafe on a PCIe-only A30 node",
            "no rollback threshold, so admission control that converts OOM crashes into mass 429s would be scored as a success",
        ]
        ev = [
            "served model config (num_layers, num_kv_heads, head_dim, dtype, TP degree) to compute per-token KV bytes",
            "KV pool size and free-block time series at >= 1 Hz, timestamp-aligned to arrivals and failures",
            "prompt and generation length distribution (p50/p95/p99/max) from a captured production trace",
            "torch.cuda.memory_reserved vs memory_allocated at failure plus a CUDA memory snapshot",
            "nvidia-smi HBM time series as an out-of-process cross-check and dmesg to exclude host-side OOM kill",
            "pre-change baseline of p99 TTFT, inter-token latency, throughput and 429 rate on the same replayed trace",
        ]
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
            "risks": risks,
            "evidence_required": ev,
            "confidence": 0.62,
        })
    with open(OUT, "w") as f:
        for o in out:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    print("wrote", len(out), "->", OUT)
    print("ids:", [o["source_id"] for o in out])


main()
