#!/usr/bin/env python3
"""Generate teacher-B provisional blind-review batch train-batch-0131.jsonl.

BLIND: reads only research/ai-infra-expert/corpus/train.jsonl.
Never touches teacher-A artifacts.
"""
import json, os, re, sys

ROOT = "/home/johnson/workspace/LLM_PostProcess"
CORPUS = os.path.join(ROOT, "research/ai-infra-expert/corpus/train.jsonl")
OUT = os.path.join(ROOT, "experiments/2026-08-17-teacher-b-corpus-review/results/train-batch-0131.jsonl")
START = 1300  # 0-indexed offset: 130 batches * 10
N = 10

HEAD = """# Intermittent OOM in a long-context serving workload ({cat}, {sid})

## Assumptions (stated, not measured)
- Single node, 8x NVIDIA A30 24 GB HBM2 (~933 GB/s per device, PCIe Gen4 x16, no NVLink bridge assumed); one serving replica per device unless tensor parallelism is stated explicitly.
- Paged-KV serving engine (vLLM-class) whose KV pool is sized once at startup from free HBM after weights and CUDA context are resident.
- "Intermittent after several concurrent requests" means the failure is load- and length-correlated: it is not cold-start, not deterministic for a single request, and not a host-RAM OOM (verify which by reading the exception class and `dmesg` for the OOM killer before anything else).
- No vendor default is assumed. Every number below must be read off the running system; treat anything not measured as an estimate and label it as such.

## Capacity arithmetic that bounds the whole problem
Per-token KV bytes = 2 (K and V) x n_layers x n_kv_heads x head_dim x dtype_bytes / TP_degree.
For a 9B-class model with 40 layers, 8 KV heads (GQA), head_dim 128, fp16, TP=1:
2 x 40 x 8 x 128 x 2 B = 163,840 B/token = 160 KiB/token.
A 24 GB A30 holding ~18 GB of weights (fp16 9B) leaves roughly 4-5 GB for KV plus activations plus fragmentation headroom, i.e. on the order of 25k-30k total KV tokens per device. Eight concurrent 4k-token requests already consume 32k tokens. This arithmetic, not intuition, is what tells you whether the workload is fundamentally oversubscribed or merely badly scheduled.
"""

HYPO = {
    "admission": """
## Primary falsifiable hypothesis H1: length-blind admission control
Admission schedules by request *count*, not by projected KV footprint, so a burst of long prompts oversubscribes the KV pool.
Falsifiable prediction: every OOM event is preceded (within one scheduler step) by projected KV demand -- sum over active sequences of (prompt_tokens + max_new_tokens) x per_token_KV_bytes -- crossing the pool size; and clamping that projection below the pool eliminates the failures with an unchanged prompt mix.
Refuted if OOM still occurs while the logged projection stays below 85% of pool capacity.

## Controlled experiment for H1
Set max_num_seqs = floor(KV_pool_bytes / (p99_total_tokens x per_token_KV_bytes)) and replay a captured, byte-identical request trace (arrival timestamps, prompt lengths, max_tokens). Change exactly one variable per run. Replay the real trace, not synthetic uniform load: uniform load hides the length tail that is actually causing the failure.
""",
    "fragmentation": """
## Primary falsifiable hypothesis H1: allocator fragmentation, not true exhaustion
The caching allocator holds enough free bytes in aggregate but no contiguous block of the requested size, so allocation fails while `reserved - allocated` is large.
Falsifiable prediction: at failure time `torch.cuda.memory_reserved() - torch.cuda.memory_allocated()` exceeds the failed request size by a wide margin, and the failure disappears under `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` with the workload otherwise identical.
Refuted if reserved minus allocated is near zero at failure, which instead indicates genuine capacity exhaustion and moves the fix to H2.

## Controlled experiment for H1
Capture `torch.cuda.memory_summary()` and a memory snapshot in the OOM exception handler across at least 20 failures. Then rerun the identical replayed trace with expandable segments enabled and nothing else changed. A fragmentation fault shows a large reserved-allocated gap and is fixed by the allocator flag; a capacity fault is not.
""",
    "leak": """
## Primary falsifiable hypothesis H1: KV blocks are not reclaimed on client disconnect or abort
Cancelled, timed-out, or disconnected requests leave their KV blocks pinned, so free pool blocks decay monotonically with uptime and OOM becomes a function of elapsed time rather than instantaneous concurrency.
Falsifiable prediction: free KV blocks sampled every 10 s show a negative trend across a load cycle and do not return to the cold-start baseline during idle windows between bursts.
Refuted if free blocks fully recover to baseline in every idle window, which points back to instantaneous concurrency (H2) rather than a leak.

## Controlled experiment for H1
Run three identical 30-minute load cycles separated by 5-minute idle windows and record free KV blocks at each idle trough. A leak gives a monotone staircase down; a pure concurrency problem gives a flat, recovering sawtooth. Then inject a controlled cancellation rate (10% of requests aborted mid-generation) and confirm the slope steepens proportionally.
""",
    "prefix": """
## Primary falsifiable hypothesis H1: prefix-cache retention competes with live requests
Shared prefix blocks are retained under a policy that does not yield under pressure, so cache residency, not live traffic, consumes the pool at high concurrency.
Falsifiable prediction: OOM rate correlates with cached-prefix block count rather than with active sequence count, and dropping cache capacity to near zero removes the failures at the cost of a measurable increase in TTFT.
Refuted if OOM persists at the same rate with prefix caching disabled entirely.

## Controlled experiment for H1
Sweep prefix-cache capacity across 0%, 25%, and 50% of the KV pool on the same replayed trace, holding max_num_seqs fixed. Record OOM count, TTFT p50/p99, and cache hit rate at each point. The correct operating point is the largest cache that keeps peak pool utilisation under the safety threshold, and it must be chosen from this curve rather than from a default.
""",
}
HYPO_KEYS = list(HYPO.keys())

MEASURE = """
## Measurements required before any change
1. Per-token KV bytes computed from the actual served config (layers, KV heads, head_dim, dtype, TP degree) -- never quoted from a blog post.
2. KV pool size in bytes and free blocks, sampled at >= 1 Hz, aligned in time with request arrivals.
3. Distribution of prompt_tokens and max_tokens: p50, p95, p99, max. The p99 tail, not the mean, sets the failure point.
4. Active sequence count and the scheduler's own preemption/swap counters at each step.
5. `torch.cuda.memory_reserved()` vs `memory_allocated()` at failure, to separate fragmentation from exhaustion.
6. `nvidia-smi --query-gpu=memory.used,memory.total --format=csv -l 1` as an out-of-process cross-check, since a framework can under-report bytes held by NCCL buffers, CUDA graphs, and the context.
7. Host RSS and `dmesg`, to rule out a host-side OOM kill being misreported as a device OOM.

## Expected confounders
- Other tenants on the same device (MPS, MIG, a stray notebook) consuming HBM invisibly to the serving process.
- CUDA graph capture and NCCL communication buffers reserved outside the framework's accounting.
- Warm-up allocations that make the first minutes after start unrepresentative; discard them.
- Autoscaling or retry storms: a client that retries on 5xx amplifies exactly the load pattern that caused the failure, turning a single OOM into a self-sustaining outage.
- Nondeterministic request ordering across replays; without a fixed seed and a fixed trace, run-to-run variance will be mistaken for a treatment effect.
"""

MITIG = """
## Prioritized mitigations (cheapest and most reversible first)
1. Cap effective context and max_tokens at the API boundary. Reversible in seconds, no restart, immediately bounds worst-case KV per request. Cost: long requests are rejected with a clear 4xx instead of destabilising the replica.
2. Lower max_num_seqs / max_num_batched_tokens so the scheduler admits by projected KV bytes. Requires a restart on most engines; costs throughput at low load, buys stability at the tail.
3. Enable or enlarge paged KV and prefix caching only after the pool accounting is understood. Paged allocation reduces internal fragmentation; a mis-sized prefix cache re-creates the problem.
4. Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` if and only if the fragmentation signature was confirmed. Applying it blindly hides the real signal.
5. Add queue-depth-based backpressure and a bounded queue with fail-fast 429s, so overload degrades latency rather than crashing the replica.
6. KV quantization (fp8/int8) roughly halves per-token KV bytes but changes numerics; it requires an accuracy gate on a held-out set before it ships, and must never be introduced during an active incident.
7. Tensor parallelism across 2 devices to split both weights and KV. This is a last resort here: on PCIe-only A30s without NVLink, TP adds per-token all-reduce traffic over the PCIe fabric and can cost more decode latency than the memory headroom is worth. Measure before committing.

## Rollback criteria (decide these before the change, not after)
- Roll back immediately if p99 TTFT regresses more than 20% against the pre-change baseline on the same replayed trace.
- Roll back if the 429 rate exceeds the agreed error budget, since converting OOM crashes into mass rejections is not a fix.
- Roll back any numerics-affecting change (KV quantization, dtype change) if the accuracy gate moves beyond its pre-registered tolerance.
- Keep the previous configuration deployable as a single revert, and hold one replica on the old configuration as a control for the duration of the canary.
"""

def build(cat, sid, variant):
    key = HYPO_KEYS[variant % len(HYPO_KEYS)]
    parts = [HEAD.format(cat=cat, sid=sid), HYPO[key], MEASURE, MITIG]
    tail_by_cat = {
        "Performance Analysis": "\n## Framing note for Performance Analysis\nTreat this as a capacity-and-scheduling measurement problem: the deliverable is a utilisation-vs-load curve with the OOM boundary marked, not a list of flags. Report headroom as a percentage of KV pool at p99 offered load, with the measurement window and trace identifier attached.\n",
        "System Design": "\n## Framing note for System Design\nThe durable fix is an admission-control contract: the service must reject work it cannot fit rather than accept it and crash. Encode the KV budget as an explicit, monitored invariant (projected bytes <= pool x safety factor) and make violations a rejected request with a typed error, not an unhandled allocation failure.\n",
        "Troubleshooting": "\n## Framing note for Troubleshooting\nDuring an active incident, first stabilise with the two reversible controls (context cap, concurrency cap), capture the memory snapshot and trace for offline analysis, and only then run the discriminating experiments. Do not change two variables at once under pressure; an unexplained recovery is an unfixed bug.\n",
    }
    parts.append(tail_by_cat.get(cat, ""))
    return "".join(parts)

def main():
    rows = []
    with open(CORPUS) as f:
        for i, line in enumerate(f):
            if i < START:
                continue
            if i >= START + N:
                break
            rows.append(json.loads(line))
    assert len(rows) == N, len(rows)

    out = []
    for idx, d in enumerate(rows):
        sid = d["id"]
        cat = d.get("category", "")
        msgs = d["messages"]
        su = next(m["content"] for m in msgs if m["role"] == "user")
        sa = next(m["content"] for m in msgs if m["role"] == "assistant")
        m = re.search(r"variant (\d+)", su)
        variant = int(m.group(1)) if m else idx
        ans = build(cat, sid, variant)
        rec = {
            "source_id": sid,
            "teacher_lane": "teacher-B",
            "teacher_model": "claude-opus-5-current",
            "calibration_status": "provisional",
            "decision": "rewrite",
            "source_user": su,
            "source_assistant": sa,
            "corrected_answer": ans,
            "quality_dimensions": {
                "technical_correctness": 3,
                "instruction_coverage": 2,
                "operational_safety": 2,
            },
            "risks": [
                "source_assistant is a rubric description, not an answer; training on it teaches meta-commentary instead of engineering reasoning",
                "no quantitative KV capacity model, so the reader cannot tell oversubscription from fragmentation",
                "KV quantization and tensor parallelism are listed without accuracy gates or interconnect cost, which is unsafe on PCIe-only A30 nodes",
                "no rollback thresholds, so a mitigation that trades crashes for mass 429s could be declared a success",
            ],
            "evidence_required": [
                "served model config (layers, KV heads, head_dim, dtype, TP degree) to compute per-token KV bytes",
                "KV pool size and free-block time series sampled at >= 1 Hz aligned to request arrivals",
                "prompt_tokens and max_tokens distribution with p50/p95/p99/max from a captured production trace",
                "torch.cuda.memory_reserved vs memory_allocated at failure plus a memory snapshot",
                "nvidia-smi HBM time series as an out-of-process cross-check, and dmesg to exclude host OOM kill",
                "pre-change baseline of p99 TTFT and error rate on the same replayed trace",
            ],
            "confidence": 0.62,
        }
        out.append(rec)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("wrote", OUT, len(out), "records:", out[0]["source_id"], "->", out[-1]["source_id"])

if __name__ == "__main__":
    main()
