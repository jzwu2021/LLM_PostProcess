import json, os

CASES = [
    # (idx, layers, seq, kv_heads, head_dim, bytes_per_val, dtype_label)
    (880, 56, 2048, 2, 128, 2, "BF16/FP16"),
    (881, 24, 2560, 4, 64, 1, "INT8"),
    (882, 32, 3072, 6, 96, 2, "BF16/FP16"),
    (883, 48, 4096, 2, 64, 1, "INT8"),
    (884, 56, 1024, 4, 96, 2, "BF16/FP16"),
    (885, 24, 1536, 6, 128, 2, "BF16/FP16"),
    (886, 32, 2048, 8, 64, 1, "INT8"),
    (887, 40, 2560, 2, 96, 2, "BF16/FP16"),
    (888, 48, 3072, 4, 128, 2, "BF16/FP16"),
    (889, 56, 3584, 6, 64, 1, "INT8"),
]

GIB = 1073741824.0
MIB = 1048576.0

def fmt(x):
    s = f"{x:.6f}"
    return s

def build_answer(layers, seq, kvh, hd, bpv, dtype):
    total = 2 * layers * seq * kvh * hd * bpv
    per_tok = 2 * layers * kvh * hd * bpv
    gib = total / GIB
    mib = total / MIB
    int8_note = ""
    if bpv == 1:
        int8_note = (
            "\n\nINT8 KV caveat: bytes_per_value = 1 counts only the quantised payload. Real INT8 KV "
            "implementations also store per-group scales (and sometimes zero-points). With per-head-per-token "
            "FP16 scales for K and V the overhead is 2 x layers x seq x kv_heads x 2 bytes = "
            f"{2*layers*seq*kvh*2} bytes ({fmt(2*layers*seq*kvh*2/MIB)} MiB), i.e. about "
            f"{100.0*(2*layers*seq*kvh*2)/total:.2f}% on top of the payload. Finer group sizes cost more. "
            "Treat the number above as a lower bound until you read the actual quantiser config."
        )
    return (
        "Mechanism: every decoder layer caches one K vector and one V vector per token per KV head. Under "
        "grouped-query/multi-query attention only kv_heads are materialised — query heads are broadcast against "
        "them — so KV size depends on kv_heads, not on the attention head count. The cache is a product of "
        "independent factors and grows strictly linearly in sequence length.\n\n"
        "Formula:\n"
        "  bytes = 2 (K and V) x layers x sequence_length x kv_heads x head_dim x bytes_per_value\n\n"
        f"Substitution ({dtype} KV, bytes_per_value = {bpv}):\n"
        f"  bytes = 2 x {layers} x {seq} x {kvh} x {hd} x {bpv}\n"
        f"        = {total} bytes\n"
        f"        = {fmt(mib)} MiB = {fmt(gib)} GiB\n\n"
        f"Per-token cost is 2 x {layers} x {kvh} x {hd} x {bpv} = {per_tok} bytes/token. This is the number to "
        "plan capacity with: resident_token_budget = KV_pool_bytes / "
        f"{per_tok}, and dividing that budget by {seq} gives approximate concurrency at this context length."
        + int8_note +
        "\n\nBoundary conditions this figure deliberately excludes:\n"
        "- Paged allocators (vLLM/SGLang PagedAttention) round each sequence up to a block boundary, so actual "
        "occupancy is ceil(seq / block_size) x block_size; at block_size 16 the waste is up to 15 tokens per "
        "sequence, which matters only for very short requests.\n"
        "- Prefix/radix cache sharing and speculative-decode draft branches change residency in the opposite "
        "direction and are not captured by a single-request formula.\n"
        "- Weights, activation/workspace buffers, CUDA graph pools, NCCL communication buffers and allocator "
        "fragmentation are separate line items; do not subtract this number from total VRAM and call the "
        "remainder usable.\n"
        "- Under tensor parallelism of degree TP the KV heads are sharded, so per-GPU KV is roughly total/TP "
        "when kv_heads is divisible by TP; when kv_heads < TP the cache is replicated instead and per-GPU KV "
        "does not shrink.\n\n"
        "Falsifiable prediction: doubling sequence_length to "
        f"{seq*2} must approximately double the measured KV allocation to {total*2} bytes with no change in "
        "any other term. If measurement disagrees by more than the paging round-up, the deployed model's "
        "kv_heads/head_dim differ from the assumed config, or KV quantisation is not what was assumed.\n\n"
        "Evidence to collect before trusting this in a capacity plan: the served model's config.json "
        "(num_hidden_layers, num_key_value_heads, head_dim), the engine's KV dtype flag, the engine startup log "
        "line reporting KV cache blocks/GiB, and a torch.cuda.memory_allocated() or nvidia-smi delta measured "
        "across a controlled single-request run at this exact context length.\n\n"
        "Rollback gate: if measured KV per request exceeds this estimate by more than 20 percent at the target "
        "concurrency, stop the rollout and reduce max_num_seqs or max_model_len before the pool hits OOM — "
        "preemption/recompute under a saturated KV pool degrades tail latency far more than the lower "
        "concurrency setting would."
    )

def main():
    root = "/home/johnson/workspace/LLM_PostProcess"
    src = os.path.join(root, "research/ai-infra-expert/corpus/train.jsonl")
    lines = open(src).read().splitlines()
    out_path = os.path.join(root, "experiments/2026-08-17-teacher-b-corpus-review/results/train-batch-0089.jsonl")
    recs = []
    for (idx, layers, seq, kvh, hd, bpv, dtype) in CASES:
        d = json.loads(lines[idx])
        msgs = d["messages"]
        user = next(m["content"] for m in msgs if m["role"] == "user")
        assistant = next(m["content"] for m in msgs if m["role"] == "assistant")
        total = 2 * layers * seq * kvh * hd * bpv
        assert str(total) in assistant, (idx, total)
        risks = [
            "source answer gives the correct product but states no boundary conditions, so it can be read as a total-VRAM budget",
            "no mention of paged-allocator block round-up, prefix-cache sharing, or tensor-parallel sharding of KV heads",
        ]
        if bpv == 1:
            risks.append("INT8 KV is costed at 1 byte/value with no accounting for quantisation scales/zero-points, understating real footprint")
        rec = {
            "source_id": d["id"],
            "teacher_lane": "teacher-B",
            "teacher_model": "claude-opus-5-current",
            "calibration_status": "provisional",
            "decision": "rewrite",
            "source_user": user,
            "source_assistant": assistant,
            "corrected_answer": build_answer(layers, seq, kvh, hd, bpv, dtype),
            "quality_dimensions": {
                "technical_correctness": 4,
                "instruction_coverage": 3,
                "operational_safety": 2 if bpv == 1 else 3,
            },
            "risks": risks,
            "evidence_required": [
                "model config.json: num_hidden_layers, num_key_value_heads, head_dim",
                "engine KV dtype flag and startup log reporting KV cache blocks/GiB",
                "measured VRAM delta for one request at this exact sequence length",
                "tensor-parallel degree and whether kv_heads is divisible by it",
            ],
            "confidence": 0.86,
        }
        recs.append(rec)
    with open(out_path, "w") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("WROTE", out_path, len(recs))

main()
