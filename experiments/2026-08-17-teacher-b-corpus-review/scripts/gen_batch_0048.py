import json, re, os

ROOT = "/home/johnson/workspace/LLM_PostProcess"
SRC = os.path.join(ROOT, "research/ai-infra-expert/corpus/train.jsonl")
OUT = os.path.join(ROOT, "experiments/2026-08-17-teacher-b-corpus-review/results/train-batch-0048.jsonl")

START, COUNT = 470, 10  # 0-indexed corpus offset

rows = []
with open(SRC) as f:
    for i, line in enumerate(f):
        if i < START:
            continue
        if i >= START + COUNT:
            break
        rows.append(json.loads(line))

out = []
for r in rows:
    msgs = r["messages"]
    user = next(m["content"] for m in msgs if m["role"] == "user")
    asst = next(m["content"] for m in msgs if m["role"] == "assistant")

    L = int(re.search(r"(\d+) layers", user).group(1))
    H = int(re.search(r"(\d+) KV heads", user).group(1))
    D = int(re.search(r"head dimension (\d+)", user).group(1))
    S = int(re.search(r"sequence length (\d+)", user).group(1))
    dtype = "INT8" if "INT8" in user else "BF16/FP16"
    B = 1 if dtype == "INT8" else 2

    total = 2 * L * S * H * D * B
    gib = total / (1024 ** 3)

    # independent check against the source answer's stated bytes
    m = re.search(r"= (\d+) bytes", asst)
    src_bytes = int(m.group(1)) if m else None
    agree = (src_bytes == total)

    ca = (
        "Assumptions: one request, GQA/MQA with {H} KV heads (not query heads), "
        "contiguous per-token KV, no quantization scales/zero-points counted, no page padding.\n\n"
        "Mechanism: every decoder layer caches one K vector and one V vector per token per KV head. "
        "The factor 2 is K and V, not bidirectionality.\n\n"
        "Formula: bytes = 2 x layers x seq_len x kv_heads x head_dim x bytes_per_value\n"
        "        = 2 x {L} x {S} x {H} x {D} x {B} ({dtype}, {B} byte/value)\n"
        "        = {total} bytes = {gib:.6f} GiB (1 GiB = 2^30 bytes).\n\n"
        "Boundary conditions and what this number excludes:\n"
        "- Paged allocators (vLLM PagedAttention, block_size 16/32 tokens) round the last block up, so "
        "real allocation is ceil(seq_len/block_size)*block_size, adding up to one block of slack per sequence.\n"
        "- INT8/FP8 KV needs per-block scales (and zero-points if asymmetric); at group size g that adds "
        "roughly 2*L*S*H*ceil(D/g)*2-4 bytes, a few percent, and is not free.\n"
        "- CUDA graph pools, activation workspace, fragmentation, and the weights themselves are separate.\n"
        "- Multiply by concurrency: aggregate KV = per-request bytes x in-flight sequences; that product, "
        "not this single number, is what sets max batch size.\n\n"
        "Falsifiable prediction: at fixed seq_len, allocated KV bytes should scale exactly linearly in "
        "in-flight requests until the block pool saturates, then admission should stall rather than OOM.\n\n"
        "Evidence to collect before trusting this in capacity planning: engine-reported KV block count and "
        "block size, torch.cuda.memory_allocated/reserved deltas across a controlled ramp, and the served "
        "config.json (num_hidden_layers, num_key_value_heads, hidden_size/num_attention_heads) rather than "
        "assumed values.\n\n"
        "Rollback gate: if measured KV bytes exceed this estimate by more than ~15% at steady state, stop "
        "raising max_num_seqs / gpu_memory_utilization and re-measure, because the gap means padding, "
        "quantization metadata, or prefix-cache retention is unaccounted for."
    ).format(L=L, S=S, H=H, D=D, B=B, dtype=dtype, total=total, gib=gib)

    out.append({
        "source_id": r["id"],
        "teacher_lane": "teacher-B",
        "teacher_model": "claude-opus-5-current",
        "calibration_status": "provisional",
        "decision": "keep" if agree else "rewrite",
        "source_user": user,
        "source_assistant": asst,
        "corrected_answer": ca,
        "quality_dimensions": {
            "technical_correctness": 5 if agree else 2,
            "instruction_coverage": 4,
            "operational_safety": 3,
        },
        "risks": [
            "Estimate ignores paged-KV block rounding; naive capacity planning from it can OOM under load.",
            "INT8 KV quantization scale/zero-point overhead and accuracy loss are not accounted for.",
            "Uses KV heads, not query heads; applying it to an MHA model by mistake understates memory.",
        ],
        "evidence_required": [
            "Served model config.json: num_hidden_layers, num_key_value_heads, head_dim.",
            "Engine KV block size and total GPU KV blocks reported at startup.",
            "Measured torch.cuda.memory_reserved delta during a controlled concurrency ramp.",
        ],
        "confidence": 0.9 if agree else 0.6,
    })

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as f:
    for o in out:
        f.write(json.dumps(o, ensure_ascii=False) + "\n")

print("wrote", len(out), "decisions", [o["decision"] for o in out])
