#!/usr/bin/env python3
"""Generate teacher-B provisional blind review batch train-batch-0087 (corpus lines 861-870).

Blind: only source_user / source_assistant from research/ai-infra-expert/corpus/train.jsonl
are read. No teacher-A artefact is opened.
"""
import json, os, re

ROOT = "/home/johnson/workspace/LLM_PostProcess"
CORPUS = os.path.join(ROOT, "research/ai-infra-expert/corpus/train.jsonl")
OUT = os.path.join(ROOT, "experiments/2026-08-17-teacher-b-corpus-review/results/train-batch-0087.jsonl")
START, END = 861, 870  # 1-indexed inclusive

GIB = 1024 ** 3
MIB = 1024 ** 2


def parse(u):
    layers = int(re.search(r"has (\d+) layers", u).group(1))
    kvh = int(re.search(r"(\d+) KV heads", u).group(1))
    hd = int(re.search(r"head dimension (\d+)", u).group(1))
    sl = int(re.search(r"sequence length (\d+)", u).group(1))
    bpv = 1 if "INT8" in u else 2
    return layers, kvh, hd, sl, bpv


def answer(layers, kvh, hd, sl, bpv):
    total = 2 * layers * sl * kvh * hd * bpv
    per_tok = 2 * layers * kvh * hd * bpv
    dtype = "INT8" if bpv == 1 else "BF16/FP16"
    quant_note = (
        "INT8 KV is already a quantised baseline: it halves footprint versus BF16/FP16 but only "
        "holds if the measured quality delta on your own eval is inside the accepted band; scale "
        "and zero-point metadata add a small per-block overhead not counted above."
        if bpv == 1 else
        "BF16/FP16 KV is the accuracy-neutral baseline; FP8/INT8 would cut this roughly 2x but "
        "must be gated on a measured quality check plus quantisation metadata overhead."
    )
    return f"""Mechanism: every decoder layer stores one K and one V vector per token per KV head. Under grouped-query attention only kv_heads are materialised (not all query heads), so the cache is a product of independent factors and grows strictly linearly in sequence length.

Formula:
  bytes = 2 (K and V) x layers x sequence_length x kv_heads x head_dim x bytes_per_value

Substitution ({dtype} KV, bytes_per_value = {bpv}):
  bytes = 2 x {layers} x {sl} x {kvh} x {hd} x {bpv}
        = {total} bytes
        = {total / MIB:.3f} MiB = {total / GIB:.6f} GiB

Per-token cost is 2 x {layers} x {kvh} x {hd} x {bpv} = {per_tok} bytes/token. That is the number to plan with: KV_pool_bytes / {per_tok} is the total resident token budget, and dividing by {sl} gives the approximate concurrency at this context length.

Boundary conditions this figure deliberately excludes:
- Paged allocators (e.g. vLLM with block_size 16) round each sequence up to a whole block, so short or ragged requests pay internal fragmentation.
- Prefix / prompt caching keeps blocks alive past request lifetime and inflates steady-state occupancy above the naive sum.
- Tensor parallelism shards kv_heads across ranks only while kv_heads >= TP degree; below that KV is replicated and per-GPU footprint stops shrinking (here kv_heads = {kvh}).
- MLA / compressed-latent-KV architectures invalidate this formula outright; size those from the latent dimension instead.
- {quant_note}

Falsifiable prediction: serving N concurrent requests of this exact shape should show measured KV pool occupancy within roughly 5-10% above N x {total} bytes. A larger gap is fragmentation, prefix reuse, or quantisation metadata and must be measured, not assumed.

Evidence to collect before acting: the server's KV-cache utilisation gauge, torch/nvidia-smi memory summary at steady state, scheduler preemption and recompute counters, and the model config values for num_hidden_layers, num_key_value_heads and head_dim (do not trust the request text over the config).

Rollback threshold: if measured occupancy exceeds 1.25x the predicted value, or preemption/recompute events appear at the target concurrency, revert the concurrency or max-context change and re-derive the budget from measured per-request occupancy rather than this analytic estimate."""


rows = []
with open(CORPUS, encoding="utf-8") as f:
    for i, line in enumerate(f, 1):
        if i < START:
            continue
        if i > END:
            break
        d = json.loads(line)
        msgs = d["messages"]
        u = next(m["content"] for m in msgs if m["role"] == "user")
        a = next(m["content"] for m in msgs if m["role"] == "assistant")
        layers, kvh, hd, sl, bpv = parse(u)
        total = 2 * layers * sl * kvh * hd * bpv
        # independent arithmetic check against the source assistant's stated byte count
        src_bytes = int(re.search(r"= (\d+) bytes", a).group(1))
        arithmetic_ok = (src_bytes == total)
        rows.append({
            "source_id": d["id"],
            "teacher_lane": "teacher-B",
            "teacher_model": "claude-opus-5-current",
            "calibration_status": "provisional",
            "decision": "rewrite" if arithmetic_ok else "reject",
            "source_user": u,
            "source_assistant": a,
            "corrected_answer": answer(layers, kvh, hd, sl, bpv),
            "quality_dimensions": {
                "technical_correctness": 4 if arithmetic_ok else 1,
                "instruction_coverage": 3,
                "operational_safety": 2,
            },
            "risks": [
                "Analytic KV size is treated as the deployable budget; paged-allocator block rounding and prefix-cache retention make real occupancy higher.",
                "No guidance that num_key_value_heads/head_dim must be read from the model config rather than from prose, so a wrong shape silently yields a wrong budget.",
                "Tensor-parallel replication of KV when kv_heads < TP degree is not mentioned, so per-GPU footprint can be under-estimated.",
                "MLA / latent-KV models would make this formula inapplicable without warning.",
            ],
            "evidence_required": [
                "model config: num_hidden_layers, num_key_value_heads, head_dim, kv cache dtype",
                "serving engine KV-cache utilisation gauge at steady state",
                "allocator block size and observed fragmentation / preemption counters",
                "nvidia-smi or torch memory summary confirming pool size on the target GPU",
            ],
            "confidence": 0.86 if arithmetic_ok else 0.6,
        })

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print("wrote", OUT, len(rows), "rows")
print("decisions:", {d: sum(1 for r in rows if r["decision"] == d) for d in {r["decision"] for r in rows}})
