#!/usr/bin/env python3
"""Generate teacher-B provisional BLIND review batch 0067 (train rows 661-670, corpus-00732..00741).

BLIND: teacher-A artifacts are never read by this script.
"""
import json, os

ROOT = "/home/johnson/workspace/LLM_PostProcess"
CORPUS = os.path.join(ROOT, "research/ai-infra-expert/corpus/train.jsonl")
OUT = os.path.join(ROOT, "experiments/2026-08-17-teacher-b-corpus-review/results/train-batch-0067.jsonl")
START, END = 660, 670  # 0-based slice

rows = []
with open(CORPUS, encoding="utf-8") as f:
    for i, line in enumerate(f):
        if START <= i < END:
            rows.append(json.loads(line))
        elif i >= END:
            break

# (layers, kv_heads, head_dim, seq_len, bytes_per_value, dtype label)
PARAMS = {
    "corpus-00732": (32, 8, 64, 1024, 1, "INT8"),
    "corpus-00733": (40, 2, 96, 1536, 2, "BF16/FP16"),
    "corpus-00734": (48, 4, 128, 2048, 2, "BF16/FP16"),
    "corpus-00735": (56, 6, 64, 2560, 1, "INT8"),
    "corpus-00736": (24, 8, 96, 3072, 2, "BF16/FP16"),
    "corpus-00737": (32, 2, 128, 3584, 2, "BF16/FP16"),
    "corpus-00738": (40, 4, 64, 4096, 1, "INT8"),
    "corpus-00739": (48, 6, 96, 1024, 2, "BF16/FP16"),
    "corpus-00740": (56, 8, 128, 1536, 2, "BF16/FP16"),
    "corpus-00741": (24, 2, 64, 2048, 1, "INT8"),
}

GIB = 1 << 30
MIB = 1 << 20


def answer(cid):
    L, H, D, S, B, dt = PARAMS[cid]
    total = 2 * L * S * H * D * B
    per_tok = 2 * L * H * D * B
    blocks = -(-S // 16)
    if B == 1:
        dtype_para = (
            "KV is already quantized to 8-bit here. That is a lossy transform: INT8 KV halves capacity pressure "
            "versus BF16 but perturbs attention scores, and the error compounds with context length. Treat the "
            "capacity win as unproven until a long-context accuracy A/B against a BF16-KV control passes; "
            "rollback gate: revert to BF16 KV if the quality delta on the long-context eval exceeds the "
            "pre-agreed tolerance, even if throughput improved. Also confirm the engine actually stores INT8 "
            "and does not silently dequantize into a BF16 scratch buffer, and account for per-block scale/zero-point "
            "metadata, which this formula omits."
        )
    else:
        dtype_para = (
            "KV is stored at 16-bit here. Moving to FP8/INT8 KV would cut this to %d B, but that is a lossy change: "
            "gate it on a long-context accuracy A/B against a BF16-KV control, and roll back if the quality delta "
            "exceeds the pre-agreed tolerance regardless of the throughput gain." % (total // 2)
        )
    return (
        "Assumptions (each one is falsifiable and should be checked against the model config.json and the serving "
        "engine's KV dtype flag before this number is used for capacity planning): standard MHA/GQA attention with "
        "exactly one K and one V tensor per layer; %d KV heads (under GQA/MQA the KV footprint scales with KV heads, "
        "never with query heads); head_dim %d; one request holding its full %d-token context; KV dtype %s "
        "(%d B per value); 1 GiB = 2^30 B; no prefix sharing, no cross-request block reuse, no KV offload to host.\n\n"
        "Formula: bytes = 2 (K and V) x layers x seq_len x kv_heads x head_dim x bytes_per_value.\n"
        "Substituting: 2 x %d x %d x %d x %d x %d = %d B = %.6f GiB (%.3f MiB).\n"
        "Derived rate: %d B per token per request (%.3f KiB/token). The per-token rate, not the single-request "
        "total, is the quantity to multiply against max_num_seqs x max_model_len when sizing the KV pool against "
        "free HBM.\n\n"
        "Mechanism: every decode step appends one K and one V vector of size kv_heads x head_dim in each of the %d "
        "layers, so KV bytes grow linearly in generated tokens and linearly in concurrency. Weights are a fixed "
        "cost; KV is the elastic one, so on a fixed HBM budget KV is what caps batch size and therefore throughput. "
        "When the KV pool is exhausted a paged engine preempts and recomputes rather than OOMing, so the observable "
        "symptom is a throughput cliff with TTFT/ITL tail spikes and a rising preemption counter, not a CUDA "
        "out-of-memory error.\n\n"
        "%s\n\n"
        "Boundary conditions this estimate does not cover, all of which push the real footprint up: (1) PagedAttention "
        "block padding - at block_size 16 a %d-token request occupies %d blocks and the last partial block is wasted; "
        "(2) the KV pool is preallocated, so gpu_memory_utilization reserves it up front and per-request free HBM will "
        "not visibly move; (3) attention/workspace scratch, CUDA graph capture pools, activation buffers and NCCL "
        "communication buffers sit outside this formula; (4) under tensor parallelism of degree TP, KV heads are "
        "sharded, so per-GPU KV is about this figure / TP only while TP <= kv_heads (%d) - past that point KV heads "
        "must be replicated and per-GPU KV stops shrinking; (5) speculative decoding, beam search or n>1 sampling "
        "multiply live KV by the number of candidate sequences.\n\n"
        "Evidence required before treating this as a measured fact rather than an estimate: the model's config.json "
        "(num_hidden_layers, num_key_value_heads, head_dim), the engine's effective KV cache dtype from the startup "
        "log, the reported number of KV blocks and block_size, and nvidia-smi or torch.cuda.memory_summary() under a "
        "controlled concurrency ramp. Falsification test: hold seq_len fixed, sweep concurrency, and confirm measured "
        "KV growth tracks %d B/token/request within a few percent; a systematic gap means the assumed KV dtype, KV "
        "head count or block accounting is wrong. Rollback gate: if measured KV per token exceeds this estimate by "
        "more than the headroom left in gpu_memory_utilization, reduce max_model_len or max_num_seqs before admitting "
        "production traffic."
        % (H, D, S, dt, B, L, S, H, D, B, total, total / GIB, total / MIB, per_tok, per_tok / 1024.0, L,
           dtype_para, S, blocks, H, per_tok)
    )


recs = []
for r in rows:
    cid = r["id"]
    msgs = r["messages"]
    su = next(m["content"] for m in msgs if m["role"] == "user")
    sa = next(m["content"] for m in msgs if m["role"] == "assistant")
    recs.append({
        "source_id": cid,
        "teacher_lane": "teacher-B",
        "teacher_model": "claude-opus-5-current",
        "calibration_status": "provisional",
        "decision": "rewrite",
        "source_user": su,
        "source_assistant": sa,
        "corrected_answer": answer(cid),
        "quality_dimensions": {
            "technical_correctness": 4,
            "instruction_coverage": 3,
            "operational_safety": 3,
        },
        "risks": [
            "Arithmetic is correct but the answer omits GQA/MQA framing, so a reader may scale KV by query heads instead of KV heads.",
            "No PagedAttention block-padding or preallocated-pool accounting, so real HBM use is understated for capacity planning.",
            "No tensor-parallel sharding boundary (per-GPU KV stops shrinking once TP exceeds kv_heads).",
            "KV dtype choice is presented without an accuracy-regression gate.",
        ],
        "evidence_required": [
            "model config.json: num_hidden_layers, num_key_value_heads, head_dim",
            "serving engine startup log: effective KV cache dtype, block_size, number of KV blocks",
            "nvidia-smi / torch.cuda.memory_summary() under a controlled concurrency ramp",
            "long-context accuracy A/B versus a BF16-KV control before accepting any quantized-KV capacity claim",
        ],
        "confidence": 0.82,
    })

with open(OUT, "w", encoding="utf-8") as f:
    for rec in recs:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
print("wrote", OUT, len(recs))
