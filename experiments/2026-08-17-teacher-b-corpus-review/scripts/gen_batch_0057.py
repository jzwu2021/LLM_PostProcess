import json, re, os

ROOT = "/home/johnson/workspace/LLM_PostProcess"
SRC = os.path.join(ROOT, "research/ai-infra-expert/corpus/train.jsonl")
OUT = os.path.join(ROOT, "experiments/2026-08-17-teacher-b-corpus-review/results/train-batch-0057.jsonl")
START, END = 561, 570  # 1-indexed inclusive

rows = []
with open(SRC) as f:
    for i, line in enumerate(f, 1):
        if START <= i <= END:
            rows.append(json.loads(line))
        if i > END:
            break

GiB = 1024 ** 3


def build(rec):
    m = rec["messages"]
    u = [x["content"] for x in m if x["role"] == "user"][0]
    a = [x["content"] for x in m if x["role"] == "assistant"][0]
    L = int(re.search(r"(\d+) layers", u).group(1))
    H = int(re.search(r"(\d+) KV heads", u).group(1))
    D = int(re.search(r"head dimension (\d+)", u).group(1))
    S = int(re.search(r"sequence length (\d+)", u).group(1))
    if "INT8" in u:
        B, dt = 1, "INT8"
    else:
        B, dt = 2, "BF16/FP16"
    total = 2 * L * S * H * D * B
    per_tok = 2 * L * H * D * B
    gib = total / GiB
    blocks = (S + 15) // 16
    padded = blocks * 16 * per_tok
    rounding = padded - total
    quant_note = (
        "INT8 KV quantization scales/zero-points are NOT counted here; they add roughly 2-4 bytes per (block, head) group, a low-single-digit percent at high concurrency."
        if B == 1 else
        "KV dtype is BF16/FP16 (2 bytes/value) end to end; moving to INT8/FP8 halves this figure and adds small per-block scale metadata."
    )
    p3 = ("Switching INT8 back to BF16/FP16 changes KV bytes by exactly 2.0x."
          if B == 1 else
          "Switching BF16/FP16 to INT8 changes KV bytes by exactly 0.5x.")

    ans = f"""Answer: {total} bytes = {gib:.6f} GiB per request at seq_len={S}.

Formula and mechanism
kv_bytes = 2 (K and V) x num_layers x seq_len x num_kv_heads x head_dim x bytes_per_value
         = 2 x {L} x {S} x {H} x {D} x {B} = {total} bytes = {gib:.6f} GiB.
Per-token cost is the more useful planning constant: 2 x {L} x {H} x {D} x {B} = {per_tok} bytes/token ({per_tok*1024/1048576:.4f} MiB per 1024 tokens). Total pool demand = per_token_bytes x sum(seq_len over concurrent sequences).

Assumptions that must hold for this number to be valid
1. Standard attention with GQA/MQA; one K and one V tensor per layer; num_kv_heads={H} is already the post-GQA KV head count, not the query head count.
2. KV dtype is {dt} ({B} byte(s)/value). {quant_note}
3. Dense contiguous allocation, batch size 1, no paged-block rounding, no prefix-cache sharing, no speculative-decode draft KV.
4. Dense (non-MLA) architecture. MLA (DeepSeek-style latent KV) stores a compressed latent per token and this formula overestimates by roughly an order of magnitude. Sliding-window or interleaved-local attention caps seq_len at the window for those layers, so the formula overestimates there too.

Boundary conditions and corrections you should apply before provisioning
- Paged attention (vLLM/SGLang) rounds each sequence up to a whole block. With block_size=16 this sequence occupies {blocks} blocks = {padded} bytes; here the rounding cost is {rounding} bytes. Worst case is +(block_size-1) tokens per sequence, which matters for many short sequences, not for one long one.
- The KV pool is sized once at startup from gpu_memory_utilization minus weights minus activation/CUDA-graph reserve; it does not grow at runtime. Free memory reported at steady state is not available KV capacity.
- In disaggregated prefill/decode (Mooncake, NVIDIA Dynamo) the same {total} bytes exist on both prefill and decode side during handoff; transfer time = kv_bytes / effective_link_bandwidth. Over RDMA/RoCE with GPUDirect RDMA budget on measured NIC goodput, not line rate; without GDR the extra host bounce roughly doubles handoff latency.
- Multi-GPU: with tensor parallelism TP, KV heads are sharded, so per-GPU KV is about {total}/TP bytes provided num_kv_heads ({H}) is divisible by TP; if it is not, engines replicate KV heads and per-GPU cost rises.

Falsifiable predictions
P1: Serving at concurrency C with all sequences at {S} tokens requires at least C x {gib:.6f} GiB of KV pool; below that the engine queues or preempts/recomputes.
P2: Doubling seq_len to {2*S} doubles KV bytes exactly (linear in seq_len), unlike weights which stay constant.
P3: {p3}
P4: The engine's reported total KV blocks x block_size x {per_tok} bytes/token will match its advertised KV pool size within a few percent.

Evidence to collect
config.json (num_hidden_layers, num_key_value_heads, head_dim or hidden_size/num_attention_heads); engine effective kv_cache_dtype and block_size; startup log line reporting total KV blocks / KV cache size; nvidia-smi or torch.cuda.memory_summary at a known steady concurrency; engine preemption/recompute counters; attention variant from the model card.

Rollback gate
If measured KV usage at fixed concurrency exceeds this estimate by more than 15 percent, or if preemption/recompute counters are non-zero at the target concurrency, revert max_model_len / max_num_seqs to the previous known-good values and re-derive the per-token constant from the engine's own block accounting rather than from this closed-form formula."""

    risks = [
        "Arithmetic is correct but the figure is single-request and dense; used directly for capacity planning it under-provisions the KV pool because concurrency is not modeled.",
        f"Paged-attention block rounding is ignored (block_size=16 rounding costs {rounding} bytes for this sequence).",
        "Confusing num_kv_heads with num_attention_heads inflates or deflates the estimate by the GQA group factor and is the most common failure mode; it surfaces as OOM at startup, not as a wrong number.",
        "The formula silently fails for MLA latent KV and for sliding-window attention; blind reuse produces wrong sizing.",
        "Quantization metadata (INT8/FP8 scales, zero-points) and speculative-decode draft KV are excluded.",
        "No rollback threshold or measurement step in the source prompt, so a wrong assumption would go undetected until an OOM or preemption storm.",
    ]
    ev = [
        "model config.json: num_hidden_layers, num_key_value_heads, head_dim",
        "engine effective kv_cache_dtype and block_size",
        "engine startup log: total KV blocks and reported KV cache size",
        "measured GPU memory and preemption/recompute counters at a known concurrency (nvidia-smi, torch.cuda.memory_summary, engine metrics)",
        "attention variant (dense GQA/MQA vs MLA vs sliding-window) from model card or config",
        "tensor-parallel degree and whether num_kv_heads is divisible by it",
    ]
    return {
        "source_id": rec["id"],
        "teacher_lane": "teacher-B",
        "teacher_model": "claude-opus-5-current",
        "calibration_status": "provisional",
        "decision": "rewrite",
        "source_user": u,
        "source_assistant": a,
        "corrected_answer": ans,
        "quality_dimensions": {
            "technical_correctness": 4,
            "instruction_coverage": 3,
            "operational_safety": 3,
        },
        "risks": risks,
        "evidence_required": ev,
        "confidence": 0.9,
    }


with open(OUT, "w") as f:
    for r in rows:
        f.write(json.dumps(build(r), ensure_ascii=False) + "\n")
print("wrote", OUT, len(rows))
