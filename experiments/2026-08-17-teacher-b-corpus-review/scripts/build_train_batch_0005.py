#!/usr/bin/env python3
"""Build teacher-B provisional review batch train-batch-0005 (corpus rows 41-50)."""
import json, os

ROOT = "/home/johnson/workspace/LLM_PostProcess"
EXP = os.path.join(ROOT, "experiments/2026-08-17-teacher-b-corpus-review")
SRC = os.path.join(ROOT, "research/ai-infra-expert/corpus/train.jsonl")
OUT = os.path.join(EXP, "results/train-batch-0005.jsonl")
START, END = 41, 50  # 1-indexed inclusive

KV_MEM = (
    "Mechanism: during autoregressive decoding each layer's attention re-reads the keys and values of all "
    "previous tokens. Caching them turns per-step attention from O(L^2) recompute into O(L) reads, so decode "
    "becomes memory-bandwidth bound rather than FLOP bound. Size model: "
    "bytes = 2 (K and V) * layers * kv_heads * head_dim * seq_len * batch * dtype_bytes. "
    "Example (assumption, not a measured platform fact): 32 layers, 8 KV heads (GQA), head_dim 128, fp16 "
    "-> 2*32*8*128*2 = 131072 B per token per sequence = 128 KiB/token; 8k context * 16 concurrent sequences "
    "= ~16 GiB, which on a 24 GB A30 leaves little room beyond weights."
)

KV_BOUND = (
    "Boundary condition: the cache is only reusable while the prefix is byte-identical and the positional "
    "encoding scheme is position-invariant with respect to reuse (e.g. RoPE re-application or offset handling). "
    "Any change in the prefix, sampling of a different branch, cache eviction, quantization of K/V to a lower "
    "dtype, or sliding-window/attention-sink truncation invalidates or changes the numerics of reuse."
)

EVID = [
    "vLLM/SGLang metrics: gpu_cache_usage_perc, num_preempted_requests, prefix cache hit rate",
    "nvidia-smi / DCGM memory used vs reserved, and HBM read bandwidth during decode",
    "measured TTFT and inter-token latency at fixed batch size and context length",
    "per-request context-length histogram from the serving access log",
]

RISKS_KV = [
    "source answer is a generic definition with no size formula, so it cannot be falsified or used for capacity planning",
    "omits GQA/MQA and KV quantization, which change memory by integer factors",
    "OOM / request preemption risk if KV budget is sized from average rather than tail context length",
]

RISKS_PF = [
    "source answer asserts prefill is compute-bound without stating the regime where it is not (very short prompts, small batch, low arithmetic intensity)",
    "no TTFT budget, no chunked-prefill interference discussion, so it gives no operational guidance",
    "running unbounded prefill alongside decode can starve decode and violate inter-token latency SLOs",
]

EVID_PF = [
    "TTFT p50/p95/p99 vs prompt length sweep at fixed concurrency",
    "GPU SM occupancy / achieved FLOPs during prefill (DCGM or Nsight)",
    "vLLM chunked-prefill chunk size setting and resulting decode inter-token latency delta",
    "prefix-cache hit rate to quantify prefill actually skipped",
]

PF_CORE = (
    "Definition: prefill is the forward pass over the whole prompt that produces the first output token and "
    "populates the KV cache for all prompt positions. Mechanism: because all prompt tokens are known up front, "
    "attention and MLP run as large batched GEMMs, so prefill has high arithmetic intensity and is normally "
    "compute (SM/tensor-core) bound, while decode is one token at a time and is memory-bandwidth bound. "
    "Cost scales roughly linearly with prompt tokens for the MLP/projection terms and quadratically with prompt "
    "length for the attention score term, so prefill dominates TTFT for long prompts."
)

PF_BOUND = (
    "Boundary condition: the compute-bound claim fails when the prompt is short (a few dozen tokens) or the batch "
    "is small, because kernel launch overhead and weight loading dominate and prefill becomes bandwidth/latency "
    "bound like decode. Falsifiable test: sweep prompt length 32 -> 8192 at fixed batch; if TTFT is flat in the "
    "low range and only becomes linear/superlinear above some threshold, the compute-bound regime starts at that "
    "threshold, not at zero."
)

ANSWERS = {
    "corpus-00044": (
        "Controlled experiment for KV cache growth.\n\n" + KV_MEM +
        "\n\nHypothesis (falsifiable): steady-state KV bytes are linear in generated tokens with slope "
        "2*layers*kv_heads*head_dim*dtype_bytes, and TTFT is unaffected while inter-token latency rises only "
        "when the cache crosses the allocator's high-water mark.\n\nDesign: single server, fixed model and dtype, "
        "one variable = max sequence length in {512, 2048, 8192}; three repeats, warm up 30 s, discard first run. "
        "Control: fixed batch=1, greedy decoding, prefix caching disabled so reuse does not confound the slope.\n\n"
        + KV_BOUND +
        "\n\nMeasure: gpu_cache_usage_perc, resident bytes from nvidia-smi, inter-token latency p50/p99, preemption "
        "count. Rollback gate: abort the sweep and revert the config if gpu_cache_usage_perc > 0.9 or any request "
        "preemption occurs, since past that point the measurement reflects eviction policy rather than cache growth."
    ),
    "corpus-00045": (
        "Controlled experiment for KV cache reuse (prefix caching).\n\n" + KV_MEM +
        "\n\nHypothesis (falsifiable): with a shared system prompt of N tokens, enabling prefix caching cuts TTFT by "
        "approximately the prefill time of those N tokens and leaves inter-token latency unchanged.\n\nDesign: A/B on "
        "the same binary and weights, single flag flipped (prefix caching on/off), identical request trace replayed "
        "twice, 3 repeats each, interleaved A/B/A/B to absorb thermal drift.\n\n" + KV_BOUND +
        "\n\nMeasure: prefix cache hit rate, TTFT p50/p95, cache usage, tokens/s. "
        "Rollback gate: if hit rate < 0.2 the experiment is invalid (trace lacks shared prefixes) — fix the trace "
        "before interpreting latency; revert the flag if p99 inter-token latency regresses more than 10%."
    ),
    "corpus-00046": (
        "Runbook: KV cache memory pressure.\n\nSymptom: rising request preemption, falling throughput, or OOM on the "
        "serving GPU.\n\n" + KV_MEM +
        "\n\nSteps: 1) read gpu_cache_usage_perc and num_preempted_requests; 2) compute the theoretical bytes/token "
        "from the formula and compare with observed growth — a large gap means fragmentation or a leak, not demand; "
        "3) check the context-length p99 of live traffic against the configured max_model_len; 4) reduce "
        "max_num_seqs or max_model_len, or enable KV quantization, one change at a time.\n\n" + KV_BOUND +
        "\n\nRollback gate: revert any change that raises p99 inter-token latency > 15% or lowers output quality "
        "checks; escalate rather than raising gpu_memory_utilization above 0.92, which trades preemption for OOM."
    ),
    "corpus-00047": (
        "Runbook: KV cache sizing before a capacity change.\n\n" + KV_MEM +
        "\n\nSteps: 1) record model config (layers, kv_heads, head_dim, dtype) from the served checkpoint, not from "
        "docs; 2) compute bytes/token; 3) multiply by target concurrency * p99 context length to get required KV "
        "bytes; 4) subtract weights and activation working set from device memory to get the actual budget; "
        "5) only then set max_num_seqs.\n\n" + KV_BOUND +
        "\n\nRollback gate: if computed demand exceeds 85% of the free budget, do not deploy — either shard with "
        "tensor parallelism, quantize KV, or cap max_model_len. Revert immediately on any preemption in canary."
    ),
    "corpus-00048": (
        "Runbook: diagnosing suspected KV cache leak vs legitimate growth.\n\n" + KV_MEM +
        "\n\nSteps: 1) drain traffic and confirm cache usage returns to baseline — if it does not, the allocator is "
        "holding blocks and the issue is a leak or fragmentation, not demand; 2) with traffic, plot cache bytes "
        "against sum of live sequence lengths — a linear fit whose slope matches the formula indicates healthy "
        "behaviour; 3) inspect aborted/cancelled request handling, the usual source of unfreed blocks.\n\n" + KV_BOUND +
        "\n\nRollback gate: restart the replica only after capturing metrics and a heap/block dump; a blind restart "
        "destroys the evidence needed to distinguish leak from demand."
    ),
    "corpus-00049": (
        "Runbook: KV cache and quantization trade-off.\n\n" + KV_MEM +
        "\n\nMechanism of the change: storing K/V in fp8 instead of fp16 halves bytes/token, roughly doubling "
        "concurrency at fixed memory, at the cost of dequantization work per attention step and small numerical "
        "drift in long contexts.\n\n" + KV_BOUND +
        "\n\nSteps: enable KV fp8 in canary only; measure concurrency, p99 inter-token latency, and a fixed-seed "
        "quality suite at long context. Rollback gate: revert if the quality suite moves beyond the pre-agreed "
        "tolerance, or if p99 latency regresses > 10%; memory savings alone are not sufficient justification."
    ),
    "corpus-00050": (
        "Runbook: KV cache interaction with disaggregated prefill/decode (e.g. Mooncake- or Dynamo-style splits).\n\n"
        + KV_MEM +
        "\n\nMechanism: when prefill and decode run on different workers, the KV blocks produced by prefill must be "
        "transferred to the decode worker, typically over RDMA/RoCE with GPUDirect so the payload never lands in "
        "host memory. Transfer time is approximately KV_bytes / achieved_link_bandwidth and is added to TTFT.\n\n"
        "Boundary condition: disaggregation only pays off when the prefill compute saved exceeds the transfer time; "
        "for short prompts the transfer dominates and TTFT gets worse. " + KV_BOUND +
        "\n\nEvidence and rollback: measure achieved RDMA bandwidth (perftest ib_write_bw) and KV transfer time "
        "separately from prefill compute; revert to colocated serving if transfer time exceeds ~30% of TTFT or if "
        "any RDMA retransmission/PFC pause counters are non-zero, which indicates the fabric, not the model, is the "
        "bottleneck."
    ),
    "corpus-00052": (
        "Prefill, variant focused on the batching mechanism.\n\n" + PF_CORE +
        "\n\nWhy it matters: prefill sets TTFT and competes with decode for the same SMs, so an unbounded prefill of "
        "a 32k-token prompt can stall every in-flight decode stream. Chunked prefill splits the prompt into fixed "
        "token chunks interleaved with decode steps to bound that stall.\n\n" + PF_BOUND +
        "\n\nEvidence and rollback gate: sweep chunk size and record TTFT and inter-token latency together; revert "
        "any chunk size that improves TTFT while pushing p99 inter-token latency past the SLO."
    ),
    "corpus-00053": (
        "Prefill, variant focused on cost scaling.\n\n" + PF_CORE +
        "\n\nWhy it matters: because the attention term grows with the square of prompt length, doubling prompt "
        "length more than doubles prefill cost past the point where attention dominates the projections; capacity "
        "planning that assumes linearity will under-provision.\n\n" + PF_BOUND +
        "\n\nEvidence and rollback gate: fit TTFT against prompt length and report both the linear and quadratic "
        "coefficients; if the quadratic term is not resolvable in the tested range, state that explicitly rather "
        "than asserting quadratic behaviour. Gate any max_model_len increase on the measured p99 TTFT staying "
        "inside budget."
    ),
    "corpus-00054": (
        "Prefill, variant focused on what removes prefill work.\n\n" + PF_CORE +
        "\n\nWhy it matters: prefill is the part of the request that prefix caching can eliminate outright. If N of "
        "M prompt tokens are an already-cached shared prefix, only M-N tokens are actually prefilled, so TTFT "
        "improvement is bounded above by N/M of the prefill time and never affects decode.\n\n" + PF_BOUND +
        "\n\nEvidence and rollback gate: report prefix cache hit rate alongside TTFT; a TTFT improvement claimed "
        "without a hit-rate change is unexplained and should not be accepted. Revert if cache lookup overhead "
        "raises TTFT for cache-miss traffic beyond 5%."
    ),
}

CONF = {
    "corpus-00050": 0.66,
    "corpus-00049": 0.7,
}


def main():
    with open(SRC, encoding="utf-8") as f:
        lines = f.readlines()[START - 1:END]
    out = []
    for line in lines:
        d = json.loads(line)
        sid = d["id"]
        u = next(m["content"] for m in d["messages"] if m["role"] == "user")
        a = next(m["content"] for m in d["messages"] if m["role"] == "assistant")
        is_kv = sid in ("corpus-00044", "corpus-00045", "corpus-00046",
                        "corpus-00047", "corpus-00048", "corpus-00049", "corpus-00050")
        rec = {
            "source_id": sid,
            "teacher_lane": "teacher-B",
            "teacher_model": "claude-opus-5-current",
            "calibration_status": "provisional",
            "decision": "rewrite",
            "source_user": u,
            "source_assistant": a,
            "corrected_answer": ANSWERS[sid],
            "quality_dimensions": {
                "technical_correctness": 3,
                "instruction_coverage": 1,
                "operational_safety": 2,
            },
            "risks": RISKS_KV if is_kv else RISKS_PF,
            "evidence_required": EVID if is_kv else EVID_PF,
            "confidence": CONF.get(sid, 0.74),
        }
        out.append(json.dumps(rec, ensure_ascii=False))
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    print(f"wrote {OUT} n={len(out)} ids={out and json.loads(out[0])['source_id']}..{json.loads(out[-1])['source_id']}")


if __name__ == "__main__":
    main()
