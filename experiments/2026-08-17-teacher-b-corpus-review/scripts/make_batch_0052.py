import json, re, os

SRC = "research/ai-infra-expert/corpus/train.jsonl"
OUT = "experiments/2026-08-17-teacher-b-corpus-review/results/train-batch-0052.jsonl"
LO, HI = 510, 520

rows = []
with open(SRC) as f:
    for i, line in enumerate(f):
        if LO <= i < HI:
            rows.append(json.loads(line))
        if i >= HI:
            break
assert len(rows) == 10

pat = re.compile(
    r"(\d+) layers, (\d+) KV heads, head dimension (\d+), sequence length (\d+), and (BF16/FP16|INT8)"
)

out = []
for r in rows:
    msgs = {m["role"]: m["content"] for m in r["messages"]}
    su, sa = msgs["user"], msgs["assistant"]
    m = pat.search(su)
    L, H, D, S, dt = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4)), m.group(5)
    b = 2 if dt == "BF16/FP16" else 1
    total = 2 * L * S * H * D * b
    gib = total / (1024 ** 3)
    mib = total / (1024 ** 2)
    stated = int(re.search(r"= (\d+) bytes", sa).group(1))
    stated_gib = float(re.search(r"= ([\d.]+) GiB", sa).group(1))
    ok = (stated == total) and abs(stated_gib - round(gib, 6)) < 1e-6
    decision = "keep" if ok else "rewrite"

    ca = f"""Assumptions (stated explicitly, none platform-specific):
- Layout is the standard per-layer paged/contiguous KV tensor; K and V are stored separately, hence the leading factor 2.
- GQA/MQA is already reflected by KV_heads = {H} (not query heads); per-token per-layer KV width = KV_heads x head_dim = {H} x {D} = {H*D} elements per tensor.
- Element width follows the KV dtype only: {dt} -> {b} byte(s)/value. INT8 KV additionally needs per-block scales/zero-points, which are NOT counted below.
- Sequence length {S} is the fully materialised context (prefill + decode) for ONE request, batch size 1, no prefix sharing or cross-request reuse.

Formula:
  bytes = 2 (K and V) x layers x sequence_length x KV_heads x head_dim x bytes_per_value

Substitution:
  bytes = 2 x {L} x {S} x {H} x {D} x {b} = {total} bytes
        = {mib:.6f} MiB = {gib:.6f} GiB   (1 GiB = 2^30 bytes; this is GiB, not GB)

Boundary conditions and what this number excludes:
- Paged-attention block granularity (e.g. vLLM block_size 16/32 tokens) rounds each request up to a whole number of blocks, so allocated >= computed; expect up to (block_size - 1) tokens of waste per sequence per layer.
- Excludes: block tables / page metadata, allocator fragmentation, weights, activations, CUDA/NCCL communicator and workspace buffers, and any speculative-decoding draft KV.
- Under tensor parallelism TP=N, KV heads are sharded when KV_heads % N == 0, so per-GPU KV is roughly this value / N; if KV_heads < N, most serving stacks replicate KV per rank and per-GPU KV does NOT shrink. With pipeline parallelism, layers split instead, so per-stage KV ~ this value x (layers_on_stage / {L}).
- Under KV offload/disaggregation (e.g. Mooncake-style prefill/decode split or NVIDIA Dynamo KV routing), this byte count is what must cross the transfer path; at {gib:.6f} GiB per request the transfer time floor is bytes / achievable_link_bandwidth (RDMA/RoCE or GDS), and that floor, not capacity, is usually the binding constraint.

Falsifiable check (how to prove or disprove this estimate):
- Measure steady-state GPU memory with one request of exactly {S} tokens resident, minus the same measurement with zero requests resident (torch.cuda.memory_allocated or the server's KV-cache-usage metric). Prediction: delta = {total} bytes +/- paged rounding, i.e. within one block per layer.
- If the measured delta exceeds ~1.15x the prediction, the extra is metadata/fragmentation or a non-{dt} internal KV dtype, not this formula being wrong.

Evidence needed before trusting this in capacity planning: the model config (num_hidden_layers, num_key_value_heads, head_dim) read from config.json, the serving engine's actual KV dtype (kv_cache_dtype may differ from weight dtype), and block_size.

Rollback / safety threshold: if a deployment sizes max_num_seqs from this figure and observed KV utilisation exceeds 90% or preemption/recompute events appear in engine logs, reduce max_num_seqs or max_model_len and re-measure before continuing; do not raise gpu_memory_utilization above the tested value on a live serving fleet.

Verdict on the source answer: the formula and the arithmetic are correct ({total} bytes = {gib:.6f} GiB). It is under-specified rather than wrong: it does not state the GQA assumption, the paged-block rounding, the INT8 scale overhead, or the parallelism sharding behaviour."""

    out.append({
        "source_id": r["id"],
        "teacher_lane": "teacher-B",
        "teacher_model": "claude-opus-5-current",
        "calibration_status": "provisional",
        "decision": decision,
        "source_user": su,
        "source_assistant": sa,
        "corrected_answer": ca,
        "quality_dimensions": {
            "technical_correctness": 5 if ok else 2,
            "instruction_coverage": 4,
            "operational_safety": 3,
        },
        "risks": [
            "Answer omits the GQA/MQA assumption that KV_heads refers to key/value heads rather than query heads; a reader with an MHA mental model would under-count.",
            "No mention of paged-attention block rounding, so the figure understates actually allocated KV memory in vLLM/SGLang-style engines.",
            ("INT8 KV quantisation scales/zero-points are not counted, so real INT8 KV footprint is slightly higher than stated."
             if b == 1 else
             "Does not distinguish weight dtype from kv_cache_dtype; engines can run BF16 weights with a different KV dtype."),
            "Single-request figure could be mistaken for a total serving budget if multiplied without accounting for TP/PP sharding or prefix cache sharing.",
        ],
        "evidence_required": [
            "config.json fields num_hidden_layers, num_key_value_heads, head_dim (or hidden_size / num_attention_heads) for the actual model.",
            "Serving engine kv_cache_dtype and block_size settings.",
            "Measured GPU memory delta or engine KV-cache-usage metric for one resident request of the stated sequence length.",
        ],
        "confidence": 0.9 if ok else 0.6,
    })

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as f:
    for o in out:
        f.write(json.dumps(o, ensure_ascii=False) + "\n")
print("wrote", OUT, len(out))
print("decisions", [o["decision"] for o in out])
