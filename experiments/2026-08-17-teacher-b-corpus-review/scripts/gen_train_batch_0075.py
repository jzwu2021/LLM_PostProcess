import json

SRC = 'research/ai-infra-expert/corpus/train.jsonl'
OUT = 'experiments/2026-08-17-teacher-b-corpus-review/results/train-batch-0075.jsonl'
START, N = 740, 10

rows = [json.loads(l) for l in open(SRC)][START:START+N]

# (layers, kv_heads, head_dim, seqlen, bytes_per_value, dtype_label)
PARAMS = {
    'corpus-00818': (40, 4, 128, 2048, 2, 'BF16/FP16'),
    'corpus-00819': (48, 6, 64, 2560, 1, 'INT8'),
    'corpus-00820': (56, 8, 96, 3072, 2, 'BF16/FP16'),
    'corpus-00821': (24, 2, 128, 3584, 2, 'BF16/FP16'),
    'corpus-00822': (32, 4, 64, 4096, 1, 'INT8'),
    'corpus-00823': (40, 6, 96, 1024, 2, 'BF16/FP16'),
    'corpus-00824': (48, 8, 128, 1536, 2, 'BF16/FP16'),
    'corpus-00825': (56, 2, 64, 2048, 1, 'INT8'),
    'corpus-00826': (24, 4, 96, 2560, 2, 'BF16/FP16'),
    'corpus-00827': (32, 6, 128, 3072, 2, 'BF16/FP16'),
}


def answer(cid):
    L, H, D, S, B, dt = PARAMS[cid]
    total = 2 * L * S * H * D * B
    gib = total / (1024 ** 3)
    per_tok = 2 * L * H * D * B
    int8_note = (
        "This case uses INT8 KV. INT8 KV is a lossy quantization of the cache, not a free "
        "2x: it needs per-token or per-head scales (small extra bytes, ignored here), a "
        "kernel that reads quantized KV, and it shifts accuracy. Treat the 1 byte/value as "
        "the payload only, and validate quality before adopting."
        if B == 1 else
        "This case uses BF16/FP16 KV at 2 bytes per value, which is the default in most "
        "serving stacks and requires no accuracy validation step."
    )
    return f"""Answer with assumptions stated first, because the number is only meaningful under them.

Assumptions (verify against config.json and the serving engine before using this for capacity planning):
1. Standard attention: exactly one K and one V tensor per layer. That is the leading factor of 2. MLA-style latent KV (DeepSeek-V2/V3) stores a compressed latent instead and this formula does not apply.
2. The head count is num_key_value_heads (GQA/MQA), not num_attention_heads. Substituting the attention-head count is the most common overestimate in practice.
3. No sliding-window attention and no cross-attention KV. A window W caps effective length at min(S, W), which makes the cache constant in S rather than linear.
4. S is total context (prompt + generated tokens), not prompt alone. During decode S grows monotonically until the request finishes or is preempted.
5. All layers are attention layers with identical shapes. Hybrid stacks (Mamba/SSM or linear-attention layers interleaved) have fewer KV-bearing layers, so the true cache is smaller.

Formula:
bytes = 2 (K and V) x layers x S x kv_heads x head_dim x bytes_per_value

Substituting layers={L}, S={S}, kv_heads={H}, head_dim={D}, bytes_per_value={B} ({dt}):
bytes = 2 x {L} x {S} x {H} x {D} x {B} = {total} bytes
GiB = {total} / 1024^3 = {gib:.6f} GiB

Per-token cost, which is the number you actually plan with:
2 x {L} x {H} x {D} x {B} = {per_tok} bytes/token = {per_tok/1024:.4f} KiB/token.
Multiply by your p95 context length and target concurrency to size the KV pool.

{int8_note}

Boundary conditions this figure excludes:
- Paged-attention block padding. vLLM-style block sizes (commonly 16 tokens) round each sequence up to a block boundary, so real occupancy is ceil(S/block)*block, not S. At small S this rounding can be a double-digit percent overhead.
- Allocator and framework overhead: CUDA graph pools, fragmentation, and the reserved-vs-allocated gap in the caching allocator.
- Weights, activations, and workspace. The KV pool is only what is left after those.
- Prefix-cache sharing and speculative-decoding draft state, which move the number in opposite directions.

Falsifiable check (do this before trusting the number):
Serve the model, drive exactly one request to a known total length S, and read the engine's reported KV block usage (vLLM logs cache blocks and GPU KV utilization). Predicted blocks = ceil(S/block_size). If measured bytes exceed the {total}-byte prediction by more than roughly 10 percent after accounting for block rounding, one of assumptions 1, 2, or 5 is wrong for this model. Re-read config.json for num_key_value_heads and the layer type map.

Evidence to collect: config.json (num_hidden_layers, num_key_value_heads, head_dim or hidden_size/num_attention_heads, sliding_window, layer_types), engine startup log with block size and total KV blocks, and nvidia-smi memory during a controlled single-request run.

Rollback threshold: if measured per-token KV exceeds the predicted {per_tok} bytes/token by more than 15 percent, stop using this estimate for admission control and re-derive from measurement, otherwise you will oversubscribe the KV pool and take preemption or OOM under concurrency."""


with open(OUT, 'w') as f:
    for r in rows:
        cid = r['id']
        msgs = {m['role']: m['content'] for m in r['messages']}
        rec = {
            "source_id": cid,
            "teacher_lane": "teacher-B",
            "teacher_model": "claude-opus-5-current",
            "calibration_status": "provisional",
            "decision": "rewrite",
            "source_user": msgs['user'],
            "source_assistant": msgs['assistant'],
            "corrected_answer": answer(cid),
            "quality_dimensions": {
                "technical_correctness": 4,
                "instruction_coverage": 2,
                "operational_safety": 3,
            },
            "risks": [
                "Formula assumes standard per-layer K/V; silently wrong for MLA latent KV and for hybrid SSM/attention stacks",
                "Does not warn that kv_heads must be num_key_value_heads, not num_attention_heads",
                "Ignores paged-attention block rounding, so the figure understates real GPU occupancy",
                "INT8 KV is presented as a pure byte-width change without noting scales, kernel support, or accuracy validation" if PARAMS[cid][4] == 1 else "No note that BF16 KV is the default baseline against which quantized KV must be validated",
                "No per-token cost or concurrency scaling, so the number cannot be used directly for capacity planning",
            ],
            "evidence_required": [
                "model config.json: num_hidden_layers, num_key_value_heads, head_dim, sliding_window, layer_types",
                "serving engine startup log: KV block size and total allocated KV blocks",
                "single-request controlled run at known total context length with engine KV utilization readout",
                "nvidia-smi or torch allocator snapshot separating weights, activations, and KV pool",
            ],
            "confidence": 0.83,
        }
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

print("wrote", OUT, len(rows))
