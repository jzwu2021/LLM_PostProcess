#!/usr/bin/env python3
"""teacher-B provisional BLIND review generator for train batch 0074 (corpus lines 731-740).

Blind: teacher-A artifacts are NOT read by this script or its author.
Corrected answers are recomputed independently from the stated parameters.
"""
import json, re, sys, os

ROOT = "/home/johnson/workspace/LLM_PostProcess"
CORPUS = os.path.join(ROOT, "research/ai-infra-expert/corpus/train.jsonl")
OUT = os.path.join(ROOT, "experiments/2026-08-17-teacher-b-corpus-review/results/train-batch-0074.jsonl")
START, END = 731, 740  # 1-indexed inclusive

PAT = re.compile(
    r"(\d+) layers, (\d+) KV heads, head dimension (\d+), sequence length (\d+), and (INT8|BF16/FP16|FP8|FP32)",
)


def parse(user):
    m = PAT.search(user)
    if not m:
        raise ValueError("unparsed: " + user[:200])
    L, H, D, S, dt = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4)), m.group(5)
    bpv = {"INT8": 1, "FP8": 1, "BF16/FP16": 2, "FP32": 4}[dt]
    return L, H, D, S, dt, bpv


def build(L, H, D, S, dt, bpv):
    per_tok = 2 * L * H * D * bpv
    total = per_tok * S
    gib = total / (1024 ** 3)
    mib = total / (1024 ** 2)
    a30_frac = total / (24 * 1024 ** 3)
    seqs_in_16gib = int((16 * 1024 ** 3) // total)
    dtype_note = (
        "INT8/FP8 KV quantization halves the footprint versus BF16 but is not free: it introduces "
        "quantization error in attention scores that is scale-sensitive at long context, and it requires "
        "engine support for a quantized KV path (per-head or per-token scaling factors, which themselves "
        "cost a small amount of extra memory not counted above)."
        if bpv == 1 else
        "BF16/FP16 is the reference dtype here; moving to FP8/INT8 KV would halve this number, at the cost "
        "of quantization error in attention and a dependency on engine-side quantized-KV kernels."
    )
    return f"""Answer with assumptions stated first, because the number is only meaningful under them.

Assumptions (verify against config.json and the serving engine before using this for capacity planning):
1. Standard attention: exactly one K and one V tensor per layer. This is the leading factor of 2. MLA-style latent KV (DeepSeek-V2/V3) stores a compressed latent instead and breaks this formula entirely.
2. The head count is num_key_value_heads (GQA/MQA), not num_attention_heads. Using the attention-head count is the single most common overestimate.
3. No sliding-window attention and no cross-attention KV. A sliding window of W caps effective sequence length at min(S, W), making the cache constant rather than linear in S.
4. Sequence length S = prompt + generated tokens (total context), not prompt alone. Decode grows this monotonically until the request finishes or is preempted.
5. Dense KV allocation. Paged allocators (vLLM PagedAttention, or block-based caches generally) round up to a block boundary, so real usage is >= this figure.

Formula:
  kv_bytes = 2 (K and V) x num_layers x seq_len x num_kv_heads x head_dim x bytes_per_value

Substitution ({dt}, {bpv} byte(s)/value):
  2 x {L} x {S} x {H} x {D} x {bpv} = {total} bytes
  = {mib:.4f} MiB = {gib:.6f} GiB   (GiB = bytes / 2^30)

Per-token KV cost = 2 x {L} x {H} x {D} x {bpv} = {per_tok} bytes/token. This is the number to carry into admission control, since it lets you extrapolate to any context length without redoing the derivation.

Boundary conditions and falsifiable predictions:
- Prediction: if you serve a single request at S = {S} and diff device memory before and after prefill, the KV delta should be >= {gib:.6f} GiB and within roughly one page-block per layer of it. A delta materially below this falsifies assumption 1 or 2 (likely MLA, sliding-window, or a KV-quantized path you did not account for).
- Prediction: doubling context to {2*S} tokens should roughly double the KV delta ({2*gib:.6f} GiB). If it does not, sliding-window attention is capping the cache.
- Scale check on this host class (A30 24 GiB, no TP): this request is about {a30_frac*100:.3f}% of one GPU's total memory; ignoring weights and activations entirely, an upper bound of only ~{seqs_in_16gib} such requests fits in a 16 GiB KV pool. Weights, CUDA context, activation workspace and fragmentation all come out of the same budget first, so treat that as a ceiling, never a target.
- {dtype_note}

What this figure excludes: model weights, CUDA/NCCL context and comm buffers, activation and attention workspace, allocator fragmentation and paging round-up, and any prefix-cache blocks retained after the request completes.

Evidence needed before acting on this: config.json (num_hidden_layers, num_key_value_heads, head_dim or hidden_size/num_attention_heads, sliding_window, architecture family), the engine's KV dtype and block size, and one measured nvidia-smi / torch.cuda.memory_allocated delta across a controlled single-request prefill.

Rollback gate: if a capacity change is made on the strength of this estimate, roll back when measured peak KV exceeds the estimate by more than 15%, or when preemption//recompute counters become non-zero under the target concurrency. Both are observable within one canary window and neither requires a restart to detect.
"""


def main():
    lines = open(CORPUS, encoding="utf-8").read().splitlines()
    rows = []
    for i in range(START - 1, END):
        d = json.loads(lines[i])
        msgs = {m["role"]: m["content"] for m in d["messages"]}
        user, asst = msgs["user"], msgs["assistant"]
        L, H, D, S, dt, bpv = parse(user)
        total = 2 * L * S * H * D * bpv
        # independent check of the source answer's arithmetic
        src_ok = str(total) in asst
        gib_ok = f"{total/(1024**3):.6f}" in asst
        rows.append({
            "source_id": d["id"],
            "teacher_lane": "teacher-B",
            "teacher_model": "claude-opus-5-current",
            "calibration_status": "provisional",
            "decision": "rewrite",
            "source_user": user,
            "source_assistant": asst,
            "corrected_answer": build(L, H, D, S, dt, bpv),
            "quality_dimensions": {
                "technical_correctness": 4 if (src_ok and gib_ok) else 2,
                "instruction_coverage": 3,
                "operational_safety": 2,
            },
            "risks": [
                "Source answer gives the formula and a correct number but states no assumptions; applying it to a GQA/MQA model with num_attention_heads, or to an MLA or sliding-window model, silently produces a large overestimate.",
                "No per-token KV cost is derived, so the result cannot be extrapolated to other context lengths for admission control.",
                "The caveat 'excludes allocator metadata and other runtime memory' is too weak: weights, activation workspace, paged-block round-up and retained prefix-cache blocks dominate and are unlisted, so the figure can be mistaken for a capacity budget.",
                ("KV quantization to 1 byte/value is treated as a pure memory win; accuracy impact and the engine's need for a quantized-KV kernel plus scale tensors are unstated."
                 if bpv == 1 else
                 "No dtype sensitivity is given, so a later switch to FP8/INT8 KV has no documented expected effect to check against."),
            ],
            "evidence_required": [
                "config.json fields: num_hidden_layers, num_key_value_heads, head_dim (or hidden_size/num_attention_heads), sliding_window, and architecture family to rule out MLA.",
                "Serving engine KV dtype and cache block size, to bound paged round-up above the dense estimate.",
                "Measured device-memory delta across one controlled single-request prefill at the stated sequence length.",
                "Engine preemption/recompute counters under the target concurrency, as the rollback signal.",
            ],
            "confidence": 0.82 if (src_ok and gib_ok) else 0.6,
        })
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("wrote", OUT, len(rows))
    print("arith_all_ok", all(r["quality_dimensions"]["technical_correctness"] == 4 for r in rows))


if __name__ == "__main__":
    main()
