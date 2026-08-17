#!/usr/bin/env python3
"""Teacher-B blind review generator for train-batch-0079 (corpus rows 780..789).

Blind: only source_user / source_assistant from research/ai-infra-expert/corpus/train.jsonl
are read. No teacher-A artifact is opened.
"""
import json, re, os

ROOT = "/home/johnson/workspace/LLM_PostProcess"
SRC = os.path.join(ROOT, "research/ai-infra-expert/corpus/train.jsonl")
OUT = os.path.join(ROOT, "experiments/2026-08-17-teacher-b-corpus-review/results/train-batch-0079.jsonl")
START, END = 780, 790

rows = [json.loads(l) for l in open(SRC) if l.strip()][START:END]

PAT = re.compile(
    r"(\d+) layers, (\d+) KV heads, head dimension (\d+), sequence length (\d+), and (\S+) KV values")

def dtype_bytes(tok):
    t = tok.upper()
    if t.startswith("INT8") or t.startswith("FP8"):
        return 1, tok.rstrip('.')
    if "BF16" in t or "FP16" in t:
        return 2, tok.rstrip('.')
    if "FP32" in t:
        return 4, tok.rstrip('.')
    raise ValueError(tok)

def answer(L, H, D, S, b, dt):
    total = 2 * L * S * H * D * b
    gib = total / (1024 ** 3)
    per_tok = 2 * L * H * D * b
    return f"""Mechanism. The K/V cache stores one key and one value vector per attention layer per token, for each KV head (GQA/MQA share KV heads across query heads, so KV heads — not query heads — set the size). Bytes are therefore:

    bytes = 2 (K and V) x layers x sequence_length x kv_heads x head_dim x bytes_per_value

Assumptions (explicit, falsifiable):
- layers = {L}, kv_heads = {H}, head_dim = {D}, sequence_length = {S} tokens (prompt + generated), dtype = {dt} => {b} byte(s) per value.
- Dense per-layer attention: every layer keeps a full cache. Sliding-window, cross-layer KV sharing (e.g. YOCO/CLA-style), or MLA-style latent compression would break this and must be checked against the model config, not assumed.
- Logical bytes only: no paged-allocator block padding, no fragmentation, no per-request metadata, no scheduler headroom.

Computation.
    2 x {L} x {S} x {H} x {D} x {b} = {total:,} bytes
    {total:,} / 1024^3 = {gib:.6f} GiB
Per-token cost: 2 x {L} x {H} x {D} x {b} = {per_tok:,} bytes/token, which is the number to multiply by real batch x realized context length for capacity planning.

Boundary conditions and caveats.
- This is one request at full sequence length {S}. Concurrency multiplies it linearly; total KV budget = per_token_bytes x sum(context_len over live requests).
- Paged KV (vLLM-style block_size 16/32 tokens) rounds each sequence up to a block boundary, so measured usage is >= this figure; expect a few percent overhead at short contexts.
- {"INT8/FP8 KV quantization additionally needs per-block scales (and possibly zero-points), typically ~0.5-2% extra, and it trades memory for accuracy — validate with a task-level eval, not perplexity alone." if b == 1 else "If you move to FP8/INT8 KV, this figure halves, but scale/zero-point metadata and accuracy regression must be measured before adopting it."}
- Weights, activations, CUDA graphs, NCCL buffers and the allocator pool are NOT included; on a 24 GiB-class card the usable KV pool is what remains after those.

Evidence required before trusting this in a capacity plan.
- Model config.json: num_hidden_layers, num_key_value_heads, head_dim (or hidden_size / num_attention_heads), and the serving engine's kv_cache_dtype.
- Engine-reported KV block count / "GPU KV cache size" line at startup, plus nvidia-smi or torch.cuda.memory_summary() under load.
- A measured max-concurrency run: if observed capacity is more than ~10% below the analytic prediction, the gap is allocator/paging overhead or a wrong dtype assumption.

Rollback gate. If enabling this configuration causes preemption/swap events or p99 TTFT regression > 20% versus the current baseline, revert max_num_seqs / gpu_memory_utilization to the previous values and re-measure before retrying.

Answer: {total:,} bytes = {gib:.6f} GiB for one request."""

recs = []
for r in rows:
    msgs = {m["role"]: m["content"] for m in r["messages"]}
    u, a = msgs["user"], msgs["assistant"]
    m = PAT.search(u)
    L, H, D, S = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
    b, dt = dtype_bytes(m.group(5))
    total = 2 * L * S * H * D * b
    gib = total / (1024 ** 3)
    # independently check the source assistant's numbers
    nums = [int(x.replace(",", "")) for x in re.findall(r"= (\d+) bytes", a)]
    src_gib = float(re.search(r"= ([0-9.]+) GiB", a).group(1))
    num_ok = bool(nums) and nums[0] == total and abs(src_gib - gib) < 5e-6
    # Arithmetic is checked independently; but the source answer omits the
    # GQA num_key_value_heads caveat, paged-allocator block round-up, INT8
    # scale metadata, TP sharding, and any evidence/rollback framing, so
    # instruction coverage and operational safety are thin -> rewrite.
    decision = "rewrite"
    recs.append({
        "source_id": r["id"],
        "teacher_lane": "teacher-B",
        "teacher_model": "claude-opus-5-current",
        "calibration_status": "provisional",
        "decision": decision,
        "source_user": u,
        "source_assistant": a,
        "corrected_answer": answer(L, H, D, S, b, dt),
        "quality_dimensions": {
            "technical_correctness": 4 if num_ok else 2,
            "instruction_coverage": 2,
            "operational_safety": 2,
        },
        "risks": [
            "Formula assumes dense per-layer full-context KV; MLA, sliding-window attention or cross-layer KV sharing would make the estimate too high.",
            "Logical-byte estimate ignores paged-allocator block rounding and fragmentation, so measured GPU usage will be higher.",
            "Answer gives no concurrency or total-GPU-budget framing, so it can be misread as a deployment capacity number.",
        ] + ([
            "INT8 KV quantization scales/zero-points are unaccounted for and accuracy impact is not mentioned.",
        ] if b == 1 else []),
        "evidence_required": [
            "model config.json: num_hidden_layers, num_key_value_heads, head_dim",
            "serving engine kv_cache_dtype and reported GPU KV cache size at startup",
            "measured GPU memory under load (nvidia-smi / torch.cuda.memory_summary) to bound allocator overhead",
        ],
        "confidence": 0.9 if num_ok else 0.75,
    })

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as f:
    for rec in recs:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
print("wrote", OUT, len(recs))
print("decisions", {d: sum(1 for x in recs if x["decision"] == d) for d in ("keep", "rewrite", "reject")})
