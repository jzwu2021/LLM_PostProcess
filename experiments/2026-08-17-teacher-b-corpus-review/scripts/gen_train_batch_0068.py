#!/usr/bin/env python3
"""Teacher-B provisional BLIND review generator for train-batch-0068 (corpus rows 671-680)."""
import json, os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
CORPUS = os.path.join(ROOT, "research/ai-infra-expert/corpus/train.jsonl")
OUT = os.path.join(ROOT, "experiments/2026-08-17-teacher-b-corpus-review/results/train-batch-0068.jsonl")
START, END = 671, 680  # 1-indexed inclusive

# (layers, kv_heads, head_dim, seq_len, bytes_per_value, dtype_label)
PARAMS = {
    "corpus-00742": (32, 4, 96, 2560, 2, "BF16/FP16"),
    "corpus-00743": (40, 6, 128, 3072, 2, "BF16/FP16"),
    "corpus-00745": (56, 2, 96, 4096, 2, "BF16/FP16"),
    "corpus-00746": (24, 4, 128, 1024, 2, "BF16/FP16"),
    "corpus-00747": (32, 6, 64, 1536, 1, "INT8"),
    "corpus-00748": (40, 8, 96, 2048, 2, "BF16/FP16"),
    "corpus-00749": (48, 2, 128, 2560, 2, "BF16/FP16"),
    "corpus-00750": (56, 4, 64, 3072, 1, "INT8"),
    "corpus-00751": (24, 6, 96, 3584, 2, "BF16/FP16"),
    "corpus-00752": (32, 8, 128, 4096, 2, "BF16/FP16"),
}


def answer(sid):
    L, H, D, S, B, dt = PARAMS[sid]
    total = 2 * L * S * H * D * B
    gib = total / (1024 ** 3)
    mib = total / (1024 ** 2)
    per_tok = 2 * L * H * D * B
    per_tok_kib = per_tok / 1024
    int8_note = (
        "Because KV is INT8 here, the byte count is half the BF16 equivalent, but that saving is not free: "
        "INT8 KV requires a per-head or per-token scale/zero-point that the engine stores alongside the blocks "
        "(commonly a few tenths of a percent of overhead, engine-specific), and it perturbs attention logits. "
        "Falsifiable check before shipping INT8 KV: run the same eval suite at BF16 KV and INT8 KV and require the "
        "task metric delta to stay inside the pre-agreed tolerance; if it does not, INT8 KV is not a valid substitution "
        "for this model regardless of the memory win."
        if B == 1 else
        "KV here is BF16/FP16 at 2 B per value. Moving to FP8/INT8 KV would halve this number, but that is a "
        "quality-affecting change and must be gated on an A/B eval at identical decode settings, not on the memory saving alone."
    )
    return (
        f"Assumptions, each independently checkable against the model config.json and the serving engine's KV dtype flag "
        f"before this figure is used for capacity planning: standard MHA/GQA attention with exactly one K tensor and one V "
        f"tensor per layer; {L} layers; {H} KV heads (under GQA/MQA the KV footprint scales with KV heads, never with query "
        f"heads - using query heads here is the single most common way this estimate comes out wrong by an integer factor); "
        f"head_dim {D}; a single request holding its full {S}-token context; KV dtype {dt} at {B} B per value; 1 GiB = 2^30 B; "
        f"no prefix sharing, no cross-request block reuse, no KV offload to host memory.\n\n"
        f"Formula: bytes = 2 (K and V) x layers x seq_len x kv_heads x head_dim x bytes_per_value.\n"
        f"Substituting: 2 x {L} x {S} x {H} x {D} x {B} = {total} B = {gib:.6f} GiB ({mib:.3f} MiB).\n"
        f"Derived rate: {per_tok} B per token per request ({per_tok_kib:.3f} KiB/token). The per-token rate, not this "
        f"single-request total, is the quantity to multiply by max_num_seqs x max_model_len when sizing the KV pool against "
        f"free HBM.\n\n"
        f"Mechanism: each decode step appends one K and one V vector of size kv_heads x head_dim in every one of the {L} "
        f"layers, so KV bytes grow linearly in generated tokens and linearly in concurrency. Model weights are a fixed cost; "
        f"KV is the elastic one, so on a fixed HBM budget KV is what caps batch size and therefore throughput. {int8_note}\n\n"
        f"Boundary conditions this number does not cover: paged-attention block granularity rounds each sequence up to a whole "
        f"number of blocks, so real allocation is >= this value (waste is bounded by block_size x per-token-bytes per sequence); "
        f"allocator fragmentation and the CUDA graph / activation / communication buffers are excluded; speculative decoding, "
        f"beam search and prefix-cache forks multiply KV by the number of live branches; MLA-style latent-KV architectures "
        f"(DeepSeek-family) do not obey this formula at all and need their own compressed-latent sizing; disaggregated "
        f"prefill/decode stacks (Mooncake, NVIDIA Dynamo) additionally hold a transferred copy of the prefill KV in flight, so "
        f"cluster-wide KV bytes exceed the sum of per-instance steady-state estimates during the transfer window.\n\n"
        f"Evidence to collect rather than assume: num_hidden_layers, num_key_value_heads and head_dim read from config.json; "
        f"the engine's reported KV cache blocks / GPU-blocks number at startup; nvidia-smi or torch.cuda.memory_summary HBM "
        f"occupancy under a saturating load; and the engine's preemption/recompute counter.\n\n"
        f"Rollback gate: if measured KV occupancy at target concurrency exceeds this estimate by more than the block-rounding "
        f"bound, or if preemption events are non-zero at the intended max_num_seqs, revert to the previously known-good "
        f"max_num_seqs / max_model_len pair and re-measure before raising concurrency again."
    )


rows = []
with open(CORPUS, encoding="utf-8") as f:
    for i, line in enumerate(f, 1):
        if i < START:
            continue
        if i > END:
            break
        rows.append(json.loads(line))

assert len(rows) == 10, len(rows)

out = []
for r in rows:
    sid = r["id"]
    msgs = {m["role"]: m["content"] for m in r["messages"]}
    out.append({
        "source_id": sid,
        "teacher_lane": "teacher-B",
        "teacher_model": "claude-opus-5-current",
        "calibration_status": "provisional",
        "decision": "rewrite",
        "source_user": msgs["user"],
        "source_assistant": msgs["assistant"],
        "corrected_answer": answer(sid),
        "quality_dimensions": {
            "technical_correctness": 4,
            "instruction_coverage": 3,
            "operational_safety": 2,
        },
        "risks": [
            "Source answer states the arithmetic correctly but gives no GQA/MQA warning, so a reader may substitute query heads for KV heads and overestimate KV by the GQA group factor.",
            "No paged-attention block-rounding term, so the figure understates real allocation on vLLM/SGLang-style engines.",
            "Single-request framing invites naive multiplication by concurrency without accounting for prefix sharing, speculative branches or preemption.",
            "MLA/latent-KV architectures and disaggregated prefill/decode KV transfer are outside this formula but not flagged as such.",
        ],
        "evidence_required": [
            "config.json fields num_hidden_layers, num_key_value_heads, head_dim (or hidden_size/num_attention_heads) for the actual checkpoint.",
            "Serving engine startup log line reporting KV cache blocks / GPU blocks and block_size.",
            "Measured HBM occupancy under saturating load (nvidia-smi or torch.cuda.memory_summary).",
            "Engine preemption / recompute counters at the intended max_num_seqs.",
        ],
        "confidence": 0.83,
    })

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    for o in out:
        f.write(json.dumps(o, ensure_ascii=False) + "\n")
print("wrote", OUT, len(out))
