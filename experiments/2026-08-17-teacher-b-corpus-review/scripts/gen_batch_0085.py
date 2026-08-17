#!/usr/bin/env python3
"""Generate teacher-B provisional review batch 0085 (train lines 841-850).

Reviewer reasoning is human/model-authored (see EXPERIMENT.md); this script only
re-derives the arithmetic independently and renders the agreed answer template so
that corrected_answer contains no copy of the source assistant text.
"""
import json, os

ROOT = "/home/johnson/workspace/LLM_PostProcess"
CORPUS = os.path.join(ROOT, "research/ai-infra-expert/corpus/train.jsonl")
OUT = os.path.join(ROOT, "experiments/2026-08-17-teacher-b-corpus-review/results/train-batch-0085.jsonl")
START, END = 841, 850  # 1-indexed inclusive

# reviewer-authored per-item parameters (layers, kv_heads, head_dim, seqlen, bytes_per_value, dtype label)
PARAMS = {
    "corpus-00924": (48, 8, 64, 2560, 1, "INT8"),
    "corpus-00925": (56, 2, 96, 3072, 2, "BF16/FP16"),
    "corpus-00926": (24, 4, 128, 3584, 2, "BF16/FP16"),
    "corpus-00927": (32, 6, 64, 4096, 1, "INT8"),
    "corpus-00928": (40, 8, 96, 1024, 2, "BF16/FP16"),
    "corpus-00930": (56, 4, 64, 2048, 1, "INT8"),
    "corpus-00931": (24, 6, 96, 2560, 2, "BF16/FP16"),
    "corpus-00932": (32, 8, 128, 3072, 2, "BF16/FP16"),
    "corpus-00933": (40, 2, 64, 3584, 1, "INT8"),
    "corpus-00934": (48, 4, 96, 4096, 2, "BF16/FP16"),
}

RISKS = [
    "Formula covers only the K/V tensors; PagedAttention block padding, allocator fragmentation and CUDA-graph/workspace buffers are excluded, so a capacity planner using this number alone will over-subscribe HBM.",
    "INT8 KV requires per-block scale/zero-point metadata that this closed-form ignores; real footprint is a few percent higher and accuracy impact is workload dependent.",
    "Per-request figure does not bound the server: concurrency x this value must fit the KV pool after weights, activation peaks and TP sharding.",
]
EVIDENCE = [
    "config.json / model config: num_hidden_layers, num_key_value_heads, hidden_size/num_attention_heads for head_dim, and the serving max_model_len.",
    "Runtime KV pool accounting, e.g. vLLM startup log 'GPU KV cache size' / num_gpu_blocks x block_size, compared against this analytic estimate.",
    "nvidia-smi or torch.cuda.memory_summary() steady-state HBM at known concurrency to measure the gap attributable to metadata and fragmentation.",
]


def kv_answer(cid, layers, kvh, hd, sl, bpv, dtype):
    total = 2 * layers * sl * kvh * hd * bpv
    gib = total / (1024 ** 3)
    mib = total / (1024 ** 2)
    per_tok = 2 * layers * kvh * hd * bpv
    return (
        "Mechanism: with grouped-query attention every decoder layer stores one K and one V vector per token "
        "for each KV head, so the cache is a product of independent factors and grows strictly linearly in "
        "sequence length.\n\n"
        "Formula:\n"
        "  bytes = 2 (K and V) x layers x sequence_length x kv_heads x head_dim x bytes_per_value\n\n"
        f"Substitution ({dtype} KV, bytes_per_value = {bpv}):\n"
        f"  bytes = 2 x {layers} x {sl} x {kvh} x {hd} x {bpv}\n"
        f"        = {total} bytes\n"
        f"        = {mib:.3f} MiB = {gib:.6f} GiB\n\n"
        f"Per-token cost is 2 x {layers} x {kvh} x {hd} x {bpv} = {per_tok} bytes/token, which is the number to "
        "use when projecting a concurrency target: KV_pool_bytes / per_token_bytes gives the total token budget "
        "the server can hold at once.\n\n"
        "Boundary conditions and what this number does NOT include:\n"
        "- Block-quantised allocators round each sequence up to a page (vLLM block_size 16 tokens by default), so "
        "short or ragged requests pay up to one block of internal fragmentation per layer group.\n"
        "- INT8/FP8 KV adds per-block scale (and possibly zero-point) metadata, typically ~1-3% overhead, and is "
        "an accuracy-affecting change, not a free win.\n"
        "- Tensor parallelism divides kv_heads across ranks only while kv_heads >= TP degree; below that the heads "
        "are replicated and per-GPU KV does not shrink further.\n"
        "- MLA-style architectures (compressed latent KV) break this formula entirely and must be sized from the "
        "latent dimension instead.\n\n"
        "Falsifiable prediction: if you serve N concurrent requests each at this sequence length, measured KV pool "
        f"occupancy should land within roughly 5-10% above N x {total} bytes. A larger gap means fragmentation, "
        "prefix-cache retention or quantisation metadata that must be measured, not assumed.\n\n"
        "Rollback gate: only adopt a concurrency setting derived from this estimate if steady-state HBM headroom "
        "stays above ~10% and no preemption/recompute events appear in the scheduler logs; otherwise revert the "
        "max_num_seqs / max_model_len change before touching quantisation."
    )


def main():
    with open(CORPUS, encoding="utf-8") as f:
        lines = f.readlines()[START - 1:END]
    out = []
    for line in lines:
        d = json.loads(line)
        cid = d["id"]
        msgs = d["messages"]
        su = [m["content"] for m in msgs if m["role"] == "user"][0]
        sa = [m["content"] for m in msgs if m["role"] == "assistant"][0]
        layers, kvh, hd, sl, bpv, dtype = PARAMS[cid]
        # independent re-derivation must agree with the numbers quoted in source_user
        assert f"{layers} layers" in su and f"{kvh} KV heads" in su
        assert f"head dimension {hd}" in su and f"sequence length {sl}" in su
        out.append({
            "source_id": cid,
            "teacher_lane": "teacher-B",
            "teacher_model": "claude-opus-5-current",
            "calibration_status": "provisional",
            "decision": "rewrite",
            "source_user": su,
            "source_assistant": sa,
            "corrected_answer": kv_answer(cid, layers, kvh, hd, sl, bpv, dtype),
            "quality_dimensions": {
                "technical_correctness": 5,
                "instruction_coverage": 4,
                "operational_safety": 3,
            },
            "risks": RISKS,
            "evidence_required": EVIDENCE,
            "confidence": 0.9,
        })
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("wrote", OUT, len(out))


if __name__ == "__main__":
    main()
