#!/usr/bin/env python3
"""Build teacher-B provisional blind-review batch 0142 (train lines 1411-1420)."""
import json, os

ROOT = "/home/johnson/workspace/LLM_PostProcess"
CORPUS = os.path.join(ROOT, "research/ai-infra-expert/corpus/train.jsonl")
OUT = os.path.join(ROOT, "experiments/2026-08-17-teacher-b-corpus-review/results/train-batch-0142.jsonl")
START, END = 1411, 1420  # 1-indexed inclusive

BASE = """Assumptions (state and verify before acting): single-node or TP-sharded LLM serving engine (vLLM/SGLang-class) with a paged KV cache; the OOM is CUDA device OOM, not host RAM OOM; "long context" means per-request prompt lengths in the 32k-128k token range; no MIG partitioning; the reported failures are reproducible under a fixed request mix. If any assumption is false (e.g. it is a host-side OOM killer event, or the engine uses a contiguous non-paged KV allocator), the priority order below changes and must be re-derived.

Mechanism. Steady-state device memory = weights + activation/workspace peak + KV cache pool + fragmentation overhead + allocator caching. Weights and the KV pool are usually pre-reserved; the part that scales with concurrency is (a) per-request KV blocks, (b) prefill activation peak, which scales roughly with (batch_tokens x hidden x layers-in-flight) and is dominated by the largest single prefill chunk, and (c) fragmentation from mixed-size transient allocations. KV bytes per token = 2 (K and V) x num_kv_heads x head_dim x num_layers x dtype_bytes; for GQA models this is far smaller than num_attention_heads suggests, so compute it from num_key_value_heads, not from the attention head count. "Intermittent after several concurrent requests" is the signature of a peak that is reached only when several long prefills overlap, not of a leak.

Prioritized diagnosis (cheapest and most discriminating first):
1. Capture the failing peak. Enable engine metrics and record, at 1 s resolution: gpu_cache_usage_perc (or equivalent KV utilization), num_running / num_waiting requests, and torch.cuda.memory_allocated / memory_reserved. Compare reserved minus allocated at failure time: a large gap (>10-15% of total) indicates fragmentation or allocator caching, a small gap indicates genuine demand.
2. Attribute the peak to prefill vs KV. If OOM coincides with several requests entering prefill simultaneously and KV utilization is well below 100%, the peak is activation/workspace, and the fix is chunked prefill plus a lower max_num_batched_tokens. If KV utilization saturates and preemption/recompute events spike just before the failure, the pool is undersized for the offered concurrency.
3. Check the configured envelope. Verify max_model_len, max_num_seqs, max_num_batched_tokens, and the memory-utilization fraction. A max_model_len set to the model maximum while clients send far shorter prompts still forces worst-case reservations in some code paths; conversely admission control that does not bound total in-flight tokens permits an unbounded peak.
4. Rule out non-serving consumers. Confirm no second process, no profiler, no NCCL buffer growth from a co-located job, and no leaked LoRA adapters or multimodal encoder caches on the same device.

Falsifiable hypothesis. H1: the OOM is caused by overlapping prefill activation peaks, not by KV pool exhaustion. Prediction: if H1 is true, enabling chunked prefill and capping max_num_batched_tokens at roughly one quarter of its current value will eliminate OOM at the same offered concurrency while KV utilization stays below 100% and p50 latency changes by less than 10%. If OOM persists at unchanged KV utilization, H1 is falsified and the peak lies in fragmentation or a non-serving consumer.

Controlled experiment. Fix model, dtype, parallelism, and driver version. Replay one deterministic trace (fixed prompt lengths, fixed arrival times, fixed seed) at concurrency levels 1, 2, 4, 8, 16, three repetitions each, arms A = current config, B = chunked prefill with the reduced token cap, C = A plus expandable_segments allocator setting. Primary metric: OOM occurrences per 1000 requests. Secondary: peak reserved bytes, KV utilization, preemption count, p50/p99 TTFT and TPOT, throughput in output tokens/s. Declare an arm a winner only if OOM count is zero across all repetitions and p99 TTFT regression is under 20%.

Expected confounders. Warm-up and allocator caching make the first run non-representative, so discard run 1. Prefix caching makes repeated identical prompts unrealistically cheap, so randomize prompt prefixes unless prefix reuse is genuinely representative. Background ECC retirement, other tenants, and clock throttling shift both memory and latency. Client-side queuing can silently reduce the true concurrency reaching the engine, so measure server-side in-flight counts rather than client intent.

Mitigations, in order of expected benefit per unit of risk: (1) chunked prefill plus an explicit max_num_batched_tokens cap; (2) admission control that bounds total in-flight tokens and rejects or queues past the bound with a clear 429 rather than accepting and failing; (3) enlarge the KV pool only after confirming the activation peak is bounded, and enable prefix/paged reuse; (4) reduce max_model_len to the actual p99.9 request length plus margin; (5) KV-cache quantization to FP8 or INT8, which roughly halves KV bytes per token but must be quality-gated; (6) increase tensor parallelism or move to a larger-memory GPU class, which is the most expensive and slowest option.

Evidence required before rollout: a reproducible OOM trace, the measured KV bytes per token derived from the model config, peak reserved vs allocated at failure, and an A/B on the replay harness showing zero OOM with bounded latency regression. For any quantization step, additionally a task-level quality comparison, not just perplexity.

Rollback criteria. Roll back immediately if, after the change, p99 TTFT regresses more than 20% versus baseline, throughput drops more than 10%, preemption/recompute rate exceeds 1% of requests, error rate exceeds baseline, or any quality gate on the accuracy suite regresses beyond its pre-registered threshold. Deploy behind a canary of at most 10% of traffic for a minimum of one full diurnal peak before fleet-wide rollout, and keep the previous engine configuration pinned and one command away."""

SLANT = {
    "Troubleshooting": "\n\nOperational note for on-call: do not restart the engine as a first action, because the restart destroys the very evidence (peak reserved bytes, in-flight counts) needed to attribute the peak. Capture metrics and the allocator snapshot first, then shed load via admission control, then restart only if the service is already failing all requests.",
    "Performance Analysis": "\n\nPerformance-analysis note: report memory as a distribution over time, not a single maximum, and always pair a memory change with its latency and throughput cost. A configuration that eliminates OOM by serializing prefills can look healthy on memory dashboards while quietly destroying p99 TTFT, so both must appear in the same comparison.",
    "System Design": "\n\nDesign note: the durable fix is an explicit memory budget enforced at admission time. Compute a per-replica token budget from measured KV bytes per token and the measured activation peak, expose it as a configuration invariant, and make the scheduler reject work beyond the budget. A design that relies on the OOM handler for backpressure is not a design; it is an outage with extra steps.",
}

def main():
    with open(CORPUS, encoding="utf-8") as f:
        lines = f.readlines()
    sel = lines[START - 1:END]
    assert len(sel) == 10, len(sel)
    out = []
    for ln in sel:
        rec = json.loads(ln)
        msgs = rec["messages"]
        su = next(m["content"] for m in msgs if m["role"] == "user")
        sa = next(m["content"] for m in msgs if m["role"] == "assistant")
        cat = rec.get("category", "")
        answer = BASE + SLANT.get(cat, "")
        out.append({
            "source_id": rec["id"],
            "teacher_lane": "teacher-B",
            "teacher_model": "claude-opus-5-current",
            "calibration_status": "provisional",
            "decision": "rewrite",
            "source_user": su,
            "source_assistant": sa,
            "corrected_answer": answer,
            "quality_dimensions": {
                "technical_correctness": 3,
                "instruction_coverage": 2,
                "operational_safety": 3,
            },
            "risks": [
                "source_assistant is a rubric checklist describing what an answer should contain, not an answer; training on it teaches meta-commentary instead of diagnosis",
                "no quantitative KV-cache sizing is given, so the model cannot learn to distinguish activation-peak OOM from KV-pool exhaustion",
                "no rollback threshold or canary gate is specified, which is unsafe for a production serving change",
                "KV-cache quantization is listed as a mitigation without a quality gate, inviting silent accuracy regression",
            ],
            "evidence_required": [
                "engine metrics at >=1 Hz: KV utilization, num_running/num_waiting, torch.cuda.memory_allocated vs memory_reserved at failure",
                "KV bytes per token computed from num_key_value_heads, head_dim, num_layers and dtype in the model config",
                "deterministic replay trace reproducing the OOM at fixed concurrency, >=3 repetitions per arm",
                "A/B comparison of OOM per 1000 requests plus p99 TTFT/TPOT and throughput between baseline and chunked-prefill arm",
                "task-level accuracy comparison if KV quantization is adopted",
            ],
            "confidence": 0.62,
        })
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("wrote", OUT, len(out))
    print("ids", out[0]["source_id"], "..", out[-1]["source_id"])

main()
