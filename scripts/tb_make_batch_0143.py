import json, os
os.chdir("/home/johnson/workspace/LLM_PostProcess")
SRC = "research/ai-infra-expert/corpus/train.jsonl"
OUT = "experiments/2026-08-17-teacher-b-corpus-review/results/train-batch-0143.jsonl"
START, N = 1421, 10

rows = []
with open(SRC) as f:
    for i, line in enumerate(f):
        if i < START - 1: continue
        if i >= START - 1 + N: break
        rows.append(json.loads(line))
assert len(rows) == N

CORE = """Assumptions (state and verify before acting): a paged-KV LLM serving engine (vLLM/SGLang class) on NVIDIA GPUs, tensor-parallel within a node; the failure is a CUDA device-side OOM, not a host RAM / cgroup OOM-kill; "long context" means per-request prompts in the 32k-128k token range; no MIG partitioning; the incident is reproducible under a fixed request mix. If any of these is false -- in particular if dmesg shows an OOM-killer event, or the engine uses a contiguous (non-paged) KV allocator -- the priority order below is invalid and must be re-derived.

Mechanism. Device memory at steady state = weights + activation/workspace peak + reserved KV pool + allocator fragmentation + non-engine residents (NCCL buffers, CUDA graphs, cuBLAS workspaces, any co-located process). Weights and the KV pool are normally pre-reserved at startup, so what actually scales with concurrency is (a) KV blocks consumed per running sequence, (b) the prefill activation peak, dominated by the single largest chunked-prefill token batch rather than by request count, and (c) transient fragmentation from mixed-size allocations. KV bytes per token = 2 (K and V) x num_key_value_heads x head_dim x num_layers x dtype_bytes -- compute this from num_key_value_heads, not num_attention_heads, or GQA models will be overestimated by 4-8x. "Intermittent, only after several concurrent requests" is the signature of a transient peak reached when long prefills overlap, not of a monotonic leak; a leak would show a floor that rises across the run and never returns.

Falsifiable hypothesis (H1). The OOM is caused by the concurrent prefill activation peak, not by KV capacity exhaustion. Concretely: if H1 holds, then capping the per-step prefill token budget (max_num_batched_tokens) to roughly half its current value, while leaving max_num_seqs, KV pool fraction and context limit unchanged, eliminates the OOM at identical offered load, and KV utilization at the moment of failure was below ~0.95. H1 is falsified if the OOM persists at the reduced prefill budget, or if telemetry shows KV utilization pinned at ~1.0 with a growing preemption/waiting queue immediately before the failure -- that pattern instead supports H2 (KV capacity + admission control) or H3 (fragmentation: reserved-minus-allocated bytes large and rising while allocated is flat).

Controlled experiment. Fixed arms, one variable each, same model build, same GPU SKU and driver, same seed and same replayed trace: A0 = current config (baseline, must reproduce the OOM at a stated rate, e.g. >=3 failures in 30 min); A1 = prefill token budget halved; A2 = max_num_seqs halved (isolates KV capacity from prefill width); A3 = allocator policy changed only (expandable_segments:True or equivalent) to isolate fragmentation. Run each arm >=3 times, >=20 minutes, to failure or clean completion. Record per second: gpu_cache_usage_perc, num_running / num_waiting / num_preempted, torch.cuda.memory_allocated and memory_reserved, TTFT and TPOT percentiles, and achieved output tokens/s. Primary endpoint: OOM occurrences per GPU-hour. Guardrail endpoints: p95 TTFT and throughput regression versus A0 must stay within a pre-declared budget (e.g. <=20% TTFT regression), otherwise the "fix" merely trades an availability failure for a latency failure.

Expected confounders. (1) Request-mix drift -- unless the trace is replayed deterministically, prompt-length distribution alone can move the peak; pin and report the p99 prompt length per arm. (2) Warm prefix cache -- a second run hits cached prefixes, lowers prefill work and hides the bug; either cold-start every arm or report cache hit rate as a covariate. (3) Co-tenancy: another process, a leftover profiler, or NCCL/CUDA-graph buffers can hold GB-scale memory; check nvidia-smi per-process usage before each arm. (4) CUDA graph capture memory appears only after the first steps at each captured batch size. (5) Fragmentation is history-dependent, so an arm that never reproduces may simply have had a luckier allocation order -- this is why arms are repeated, and why a single clean run is not evidence.

Prioritized mitigations, cheapest and most reversible first. (1) Bound admission: cap max_num_batched_tokens and max_num_seqs so the worst-case concurrent prefill fits with headroom; this is a config change, instantly revertible. (2) Enforce a hard max context / max prompt length at the gateway and reject over-limit requests with an explicit 4xx rather than letting them reach the engine. (3) Turn on chunked prefill so a single long prompt cannot create an unbounded activation spike. (4) Keep prefix caching on for shared system prompts, but treat its memory as capacity, not as free savings. (5) Change allocator policy to reduce fragmentation. (6) Only then consider heavier, less reversible changes: KV quantization (FP8/INT8 KV) with an accuracy gate, higher tensor parallelism or offload, or a disaggregated prefill/decode split so prefill spikes cannot evict decode-phase KV.

Evidence and rollback gates. Do not declare the fix effective on a single clean run. Ship only if: zero OOMs across >=3 repeats x >=20 minutes per arm at the target concurrency, KV utilization p99 <= 0.9, guardrail latency within the declared budget, and an accuracy check unchanged if KV quantization was used. Roll back automatically if, after deploy, OOM rate exceeds 0 per GPU-hour over any 1-hour window, or p95 TTFT regresses beyond the declared budget, or 5xx rate rises above the pre-incident baseline; keep the previous engine config pinned and revertible in one step. Explicitly out of scope of this evidence: none of the above measures model quality, and none of it should be read as a claim about capability -- it is a serving-capacity and reliability result only."""

FOCUS = {
    "System Design": "\n\nDesign-level framing. Treat per-GPU memory as a declared budget with named line items (weights, activation peak, KV pool, fragmentation reserve, safety headroom) that must sum below device capacity at the worst-case admitted mix -- an OOM is then a budget-violation bug, not a mystery. Size the KV pool from the target concurrency x p99 context length x KV bytes/token, and make the gateway's admission limits the enforcement point of that budget. For sustained long-context traffic, prefer disaggregated prefill/decode (separate pools, e.g. Dynamo-style routing, with a KV transfer path such as Mooncake-style KV store or NIXL/RDMA transfer) so prefill spikes are isolated from decode residency; the cost is added KV transfer bandwidth and tail latency, which must be measured, not assumed.",
    "Troubleshooting": "\n\nTriage-level framing. First separate device OOM from host OOM (dmesg / cgroup) and from a downstream 5xx that merely looks like OOM; then decide leak versus peak by checking whether post-drain memory returns to its startup floor. Preserve evidence before restarting: engine log with the allocator's requested-versus-free bytes line, nvidia-smi per-process snapshot, and the last 60 s of metrics; a restart destroys exactly the state that discriminates the hypotheses. Mitigate first by admission control, which is instantly revertible, and only then change allocator or quantization settings.",
    "Performance Analysis": "\n\nPerformance framing. Report the failure against a throughput-latency curve, not as a binary: sweep offered concurrency and plot output tokens/s and p95 TTFT/TPOT until the OOM or the knee appears, so the safe operating point is a measured number with units rather than a guess. Attribute time and memory separately -- prefill is compute-bound and drives the activation peak, decode is memory-bandwidth-bound and drives KV residency -- and state which one the mitigation actually targets. Any configuration change must be reported with both the availability endpoint (OOMs per GPU-hour) and the guardrail endpoints (p95 TTFT, tokens/s), since capping prefill width trades throughput for stability.",
}

def rec(d):
    u = [m for m in d["messages"] if m["role"] == "user"][0]["content"]
    a = [m for m in d["messages"] if m["role"] == "assistant"][0]["content"]
    cat = d.get("category", "Troubleshooting")
    ans = CORE + FOCUS.get(cat, FOCUS["Troubleshooting"])
    return {
        "source_id": d["id"],
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
            "source_assistant is a grading rubric, not an answer: it lists topics to mention without stating the memory-budget mechanism, so a model trained on it learns to enumerate keywords rather than diagnose",
            "no distinction between device OOM and host/cgroup OOM, which sends triage down the wrong path",
            "KV size guidance omits GQA (num_key_value_heads), a common 4-8x overestimate",
            "no guardrail metric: capping prefill width or concurrency can fix OOM while silently regressing p95 TTFT and throughput",
            "mitigations such as KV quantization are listed without an accuracy gate or rollback threshold",
        ],
        "evidence_required": [
            "engine allocator error line with requested vs free bytes, plus dmesg check to exclude host OOM-kill",
            "1 Hz telemetry at failure: gpu_cache_usage_perc, num_running/num_waiting/num_preempted, torch.cuda.memory_allocated vs memory_reserved",
            "nvidia-smi per-process memory snapshot to exclude co-tenant or leftover profiler residency",
            "model config num_key_value_heads, head_dim, num_layers, KV dtype for the KV bytes/token computation",
            "replayed fixed request trace with reported p99 prompt length and prefix-cache hit rate per arm",
            ">=3 repeats x >=20 min per arm reporting OOMs per GPU-hour plus p95 TTFT and tokens/s guardrails",
        ],
        "confidence": 0.62,
    }

with open(OUT, "w") as f:
    for d in rows:
        f.write(json.dumps(rec(d), ensure_ascii=False) + "\n")
print("wrote", OUT, len(rows), [d["id"] for d in rows])
