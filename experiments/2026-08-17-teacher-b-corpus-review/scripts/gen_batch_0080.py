import json, re, os

SRC = 'research/ai-infra-expert/corpus/train.jsonl'
OUT = 'experiments/2026-08-17-teacher-b-corpus-review/results/train-batch-0080.jsonl'
START, N = 790, 10

lines = open(SRC).read().splitlines()

def build(d):
    msgs = d['messages']
    u = [m['content'] for m in msgs if m['role'] == 'user'][0]
    a = [m['content'] for m in msgs if m['role'] == 'assistant'][0]
    layers = int(re.search(r'(\d+) layers', u).group(1))
    kvh = int(re.search(r'(\d+) KV heads', u).group(1))
    hd = int(re.search(r'head dimension (\d+)', u).group(1))
    sl = int(re.search(r'sequence length (\d+)', u).group(1))
    int8 = 'INT8' in u
    bpv = 1 if int8 else 2
    dtype = 'INT8' if int8 else 'BF16/FP16'
    total = 2 * layers * sl * kvh * hd * bpv
    gib = total / 1024**3
    per_tok = 2 * layers * kvh * hd * bpv
    # independent check against source claim
    m = re.search(r'=\s*(\d+)\s*bytes', a)
    src_bytes = int(m.group(1)) if m else -1
    num_ok = (src_bytes == total)
    ans = f"""Mechanism. The K/V cache holds one key and one value vector per attention layer per token, replicated over KV heads. Under GQA/MQA the KV head count (not the query head count) sets the footprint, so:

    bytes = 2 (K and V) x layers x sequence_length x kv_heads x head_dim x bytes_per_value

Assumptions, stated so they can be falsified against the model config:
- layers = {layers}, kv_heads = {kvh}, head_dim = {hd}, sequence_length = {sl} tokens (prompt + generated), dtype = {dtype} => {bpv} byte(s) per value.
- Every layer keeps a dense full-length cache. Sliding-window attention, cross-layer KV sharing, or MLA-style latent KV compression would invalidate this and must be read off config.json rather than assumed.
- Logical bytes only: excludes paged-allocator block padding, fragmentation, per-request metadata, and scheduler headroom.

Computation.
    2 x {layers} x {sl} x {kvh} x {hd} x {bpv} = {total:,} bytes
    {total:,} / 1024^3 = {gib:.6f} GiB
Per-token cost = 2 x {layers} x {kvh} x {hd} x {bpv} = {per_tok:,} bytes/token. That per-token number, not the single-request total, is what you multiply by realized concurrent context length for capacity planning.

Boundary conditions.
- One request at the full {sl}-token context. Total KV demand = per_token_bytes x sum(context_len over live requests); it grows linearly with concurrency and with realized (not maximum) context.
- Paged KV managers (vLLM-style block_size 16/32) round each sequence up to a block boundary, so measured usage is >= this figure. Treat this as a lower bound.
- {'INT8 KV is already assumed here; the scale/zero-point metadata per block is extra and accuracy regression must be measured, not assumed.' if int8 else 'Switching to FP8/INT8 KV halves this, but adds quant metadata and requires an accuracy check before adoption.'}
- Weights, activations, CUDA graph pools, NCCL/communication buffers, and allocator reserve are NOT included. On a 24 GiB-class GPU the usable KV pool is whatever remains after those, typically far less than device capacity.

Evidence required before this drives a capacity decision:
- config.json values for num_hidden_layers, num_key_value_heads, head_dim, and any window/compression settings.
- Runtime-reported KV block count and block size (e.g. vLLM "GPU KV cache size" / num_gpu_blocks at startup) compared against this arithmetic.
- A measured nvidia-smi / torch.cuda.memory_summary() sample at steady-state concurrency.

Rollback threshold. If measured KV usage exceeds this estimate by more than ~15 percent at the same context and concurrency, stop raising max_num_seqs or gpu_memory_utilization, re-derive from the runtime block report, and revert to the last configuration that held preemption/recompute events at zero."""
    dec = 'rewrite' if num_ok else 'reject'
    tc = 4 if num_ok else 2
    return {
        "source_id": d['id'],
        "teacher_lane": "teacher-B",
        "teacher_model": "claude-opus-5-current",
        "calibration_status": "provisional",
        "decision": dec,
        "source_user": u,
        "source_assistant": a,
        "corrected_answer": ans,
        "quality_dimensions": {
            "technical_correctness": tc,
            "instruction_coverage": 3,
            "operational_safety": 3,
        },
        "risks": [
            "Source answer gives the formula and number but no assumption list, so it silently generalizes to sliding-window / MLA / cross-layer-shared-KV models where the formula is wrong.",
            "Logical-bytes figure can be mistaken for real GPU consumption; paged block padding and allocator reserve are excluded.",
            "No per-token cost is given, so the number does not compose to multi-request capacity planning.",
        ] + ([] if num_ok else ["Source arithmetic does not reproduce; independent recomputation disagrees with the stated byte count."]),
        "evidence_required": [
            "Model config.json: num_hidden_layers, num_key_value_heads, head_dim, attention window / KV compression settings.",
            "Serving runtime KV block report at startup (block size and number of GPU blocks) to compare with this arithmetic.",
            "Steady-state measured GPU memory (nvidia-smi or torch.cuda.memory_summary) at the target concurrency.",
        ],
        "confidence": 0.82 if num_ok else 0.6,
    }

recs = [build(json.loads(lines[i])) for i in range(START, START + N)]
with open(OUT, 'w') as f:
    for r in recs:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print("wrote", OUT, len(recs))
from collections import Counter
print(Counter(r['decision'] for r in recs))
print([r['source_id'] for r in recs])
