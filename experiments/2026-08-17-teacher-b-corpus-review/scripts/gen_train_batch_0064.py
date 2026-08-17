import json

SRC = "/tmp/tb_src.jsonl"
OUT = "/media/home/johnson/workspace/LLM_PostProcess/experiments/2026-08-17-teacher-b-corpus-review/results/train-batch-0064.jsonl"

def parse(u):
    import re
    L = int(re.search(r"(\d+) layers", u).group(1))
    H = int(re.search(r"(\d+) KV heads", u).group(1))
    D = int(re.search(r"head dimension (\d+)", u).group(1))
    S = int(re.search(r"sequence length (\d+)", u).group(1))
    int8 = "INT8" in u
    return L, H, D, S, int8

recs = []
for line in open(SRC):
    d = json.loads(line)
    m = d["messages"]
    u = [x for x in m if x["role"] == "user"][0]["content"]
    a = [x for x in m if x["role"] == "assistant"][0]["content"]
    L, H, D, S, int8 = parse(u)
    b = 1 if int8 else 2
    dt = "INT8 (1 B/value)" if int8 else "BF16/FP16 (2 B/value)"
    total = 2 * L * S * H * D * b
    gib = total / (1024 ** 3)
    mib = total / (1024 ** 2)
    per_tok = 2 * L * H * D * b  # bytes per token

    quant = (
        "Because KV is INT8, the per-tensor/per-head scale (and zero-point, if asymmetric) is NOT counted above; "
        "it is small (order of kv_heads x layers x 4 B) but nonzero, and INT8 KV is lossy: it can degrade long-context "
        "retrieval and reasoning accuracy. Treat INT8 KV as an accuracy/capacity trade, not a free 2x."
        if int8 else
        "BF16/FP16 KV is the lossless-baseline case. Moving to FP8/INT8 KV would halve this number but adds "
        "quantization scale metadata and measurable accuracy risk on long-context retrieval, so it must be gated on eval, not assumed."
    )

    corrected = (
        "Assumptions (falsifiable, check against config.json): standard MHA/GQA attention with one K and one V tensor per layer; "
        f"{H} KV heads (KV cost scales with KV heads, not query heads, under GQA/MQA); head_dim {D}; full context {S} tokens materialised "
        f"for a single request; KV dtype {dt}; 1 GiB = 2^30 B; no cross-request prefix sharing.\n\n"
        "Formula: bytes = 2 (K and V) x layers x seq_len x kv_heads x head_dim x bytes_per_value.\n"
        f"Substituting: 2 x {L} x {S} x {H} x {D} x {b} = {total} B = {gib:.6f} GiB ({mib:.3f} MiB).\n"
        f"Useful derived rate: {per_tok} B per token per request ({per_tok/1024:.3f} KiB/token), which is the number to use when sizing "
        "max_num_seqs x max_model_len against free HBM.\n\n"
        "Mechanism: each decode step appends one K and one V vector of size kv_heads x head_dim in every layer, so KV bytes grow linearly "
        "in generated tokens and linearly in concurrency. On a fixed HBM budget, KV - not weights - is what caps batch size and therefore throughput; "
        "when the KV pool is exhausted the engine preempts or recomputes, which shows up as throughput cliffs and TTFT/ITL tail spikes rather than OOM.\n\n"
        f"{quant}\n\n"
        "Boundary conditions this estimate does NOT cover, and which make the real footprint larger: (1) PagedAttention block padding - vLLM rounds each "
        f"sequence up to a whole block (block_size 16 by default), so a {S}-token request occupies ceil({S}/16) blocks and any partial block is wasted; "
        "(2) allocator fragmentation and the preallocated KV pool (gpu_memory_utilization reserves the pool up front, so measured free HBM will not move "
        "per request); (3) attention workspace, CUDA graph capture pools, and activation buffers during prefill; (4) MLA / latent-KV models (DeepSeek-style), "
        "sliding-window or hybrid SSM layers, where this formula overestimates or is simply the wrong shape; (5) speculative decoding drafts and beam search, "
        "which multiply live KV copies.\n\n"
        "Rollback / gating: size the KV pool from the measured per-token rate above with >=20% headroom; if p99 ITL regresses or preemption counters "
        "(vllm num_preempted_requests / waiting queue depth) rise above baseline in a canary, revert max_model_len or max_num_seqs to the prior value "
        "before touching quantization."
    )

    recs.append({
        "source_id": d["id"],
        "teacher_lane": "teacher-B",
        "teacher_model": "claude-opus-5-current",
        "calibration_status": "provisional",
        "decision": "rewrite",
        "source_user": u,
        "source_assistant": a,
        "corrected_answer": corrected,
        "quality_dimensions": {
            "technical_correctness": 5,
            "instruction_coverage": 4,
            "operational_safety": 3,
        },
        "risks": [
            "Formula assumes MHA/GQA KV layout; it is invalid for MLA (latent KV), sliding-window attention, and SSM/hybrid layers.",
            "Ignores PagedAttention block padding, preallocated pool sizing, fragmentation and engine workspace, so real per-request HBM is higher.",
            ("INT8 KV quantization scale/zero-point metadata is uncounted and INT8 KV is lossy on long-context accuracy; source answer does not flag this."
             if int8 else
             "Single-request BF16/FP16 figure is often extrapolated linearly to a batch, ignoring prefix-cache sharing, preemption and recompute."),
            "Source answer states the number without boundary conditions or a verification path, so it can be over-trusted for capacity planning.",
        ],
        "evidence_required": [
            "Model config.json: num_hidden_layers, num_key_value_heads, head_dim (or hidden_size/num_attention_heads), and configured kv_cache_dtype.",
            "Serving engine startup log line reporting KV cache size, number of GPU blocks, and block_size.",
            "Measured HBM delta (nvidia-smi / torch.cuda.memory_allocated) before and after filling one request to the stated sequence length.",
            "Accuracy eval on a long-context task if INT8/FP8 KV is enabled, compared against the BF16 KV baseline.",
        ],
        "confidence": 0.9,
    })

with open(OUT, "w") as f:
    for r in recs:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print("wrote", len(recs), OUT)
