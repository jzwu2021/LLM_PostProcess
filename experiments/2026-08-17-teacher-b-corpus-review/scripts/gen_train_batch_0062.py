#!/usr/bin/env python3
"""Generate teacher-B provisional BLIND review batch train-batch-0062 (corpus lines 611-620).

Blind: teacher-A artifacts are never read by this script or its author.
Source: research/ai-infra-expert/corpus/train.jsonl
"""
import json, re, os

ROOT = "/home/johnson/workspace/LLM_PostProcess"
SRC = os.path.join(ROOT, "research/ai-infra-expert/corpus/train.jsonl")
OUT = os.path.join(ROOT, "experiments/2026-08-17-teacher-b-corpus-review/results/train-batch-0062.jsonl")
START, END = 611, 620  # 1-indexed inclusive

PAT = re.compile(
    r"(\d+) layers, (\d+) KV heads, head dimension (\d+), sequence length (\d+), and (INT8|BF16/FP16) KV values"
)


def build(sid, user, assistant):
    m = PAT.search(user)
    L, H, D, S, dt = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4)), m.group(5)
    bpv = 1 if dt == "INT8" else 2
    total = 2 * L * S * H * D * bpv
    gib = total / (1024 ** 3)
    per_tok = 2 * L * H * D * bpv
    # independent check against the source-stated number
    src_bytes = int(re.search(r"= (\d+) bytes", assistant).group(1))
    src_ok = (src_bytes == total)
    blk = 16
    pad_worst = blk * per_tok

    quant_note = (
        "INT8 KV is not free: symmetric per-group INT8 adds one FP16 scale per group (asymmetric adds a zero-point too), "
        "so at group_size=64 the real cost is about 1.03x this figure, and accuracy on long-context retrieval must be "
        "measured, not assumed."
        if bpv == 1 else
        "BF16/FP16 KV needs no scale metadata, so the dense figure is tight; moving to FP8/INT8 KV would cut it ~2x but "
        "adds per-group scales and requires an accuracy regression run before rollout."
    )

    ans = f"""Mechanism. In a decoder-only transformer with grouped-query attention, every token writes one K vector and one V vector per KV head per layer, and those tensors are retained for the whole request. The dense lower bound is

  bytes = 2 (K and V) x layers x sequence_length x KV_heads x head_dim x bytes_per_value

The leading 2 is K plus V. It is not tensor-parallel replication: under TP the KV heads are sharded, so this number is the cluster-wide total for the request and the per-GPU share is roughly total / TP when KV_heads is divisible by TP.

Substitution. layers={L}, sequence_length={S}, KV_heads={H}, head_dim={D}, dtype={dt} -> bytes_per_value={bpv}.

  bytes = 2 x {L} x {S} x {H} x {D} x {bpv} = {total} bytes
  GiB   = {total} / 2**30 = {gib:.6f} GiB

Per-token marginal cost = 2 x {L} x {H} x {D} x {bpv} = {per_tok} bytes/token. That per-token number, not the per-request total, is what admission control, max_num_seqs and max_num_batched_tokens should actually be sized against.

Independent check of the source answer: recomputed {total} bytes vs stated {src_bytes} bytes -> {"match" if src_ok else "MISMATCH"}; the source formula and arithmetic are {"correct" if src_ok else "incorrect"}.

Boundary conditions this figure does not cover.
1. Paged KV. vLLM PagedAttention / SGLang allocate whole blocks, so real usage is ceil(S / block_size) x block_size x per_token_bytes. With block_size={blk} the worst-case tail padding is {pad_worst} bytes per request, which matters when many short requests are in flight.
2. Quantization metadata. {quant_note}
3. Single request only. Steady-state pool need is roughly concurrency x per-request bytes; with prefix caching, shared system prompts are stored once, so the effective total is lower than the naive product.
4. Speculative decoding, beam search and prefix forking multiply live KV copies beyond one per request.
5. Weights, activations, CUDA graphs, NCCL buffers and the allocator's fragmentation headroom sit outside this number; gpu_memory_utilization must leave room for all of them.

Falsifiable prediction. Serving one request at S={S} on this model with the KV pool isolated, nvidia-smi / torch.cuda.memory_reserved deltas attributable to KV should land within about 5 percent above {gib:.6f} GiB (block padding accounts for the excess). If the observed delta is more than ~15 percent higher, the hypothesis "KV is dense and unquantized as specified" is falsified; look for an unapplied KV-quantization flag, a larger block size, or duplicated KV under TP.

Evidence to collect before trusting this in capacity planning.
- Model config: num_hidden_layers, num_key_value_heads, head_dim (hidden_size / num_attention_heads), and the actual kv_cache_dtype the server negotiated.
- Engine startup log line reporting KV cache blocks and GPU blocks, converted back to bytes.
- torch.cuda.memory_summary or memory_reserved before and after a controlled single-request run at the target length.

Rollback gate. If measured KV per token exceeds {per_tok} bytes by more than 15 percent, or if enabling KV quantization moves a fixed long-context eval set by more than the pre-agreed accuracy budget, revert to the previous dtype and pool size before increasing concurrency limits."""
    return {
        "source_id": sid,
        "teacher_lane": "teacher-B",
        "teacher_model": "claude-opus-5-current",
        "calibration_status": "provisional",
        "decision": "rewrite",
        "source_user": user,
        "source_assistant": assistant,
        "corrected_answer": ans,
        "quality_dimensions": {
            "technical_correctness": 5 if src_ok else 2,
            "instruction_coverage": 4,
            "operational_safety": 3,
        },
        "risks": [
            "Dense formula ignores paged-KV block rounding, so capacity planning from it under-provisions for many short requests",
            "Source does not state that the leading factor 2 is K+V rather than TP replication, inviting double counting under tensor parallelism",
            ("INT8 KV scale/zero-point metadata and accuracy regression are not mentioned" if bpv == 1
             else "No note that switching to FP8/INT8 KV changes both size and accuracy"),
            "No prefix-caching or speculative-decoding effect on live KV copies",
        ],
        "evidence_required": [
            "Model config values: num_hidden_layers, num_key_value_heads, head_dim, negotiated kv_cache_dtype",
            "Engine startup KV block count converted to bytes",
            "torch.cuda.memory_reserved delta for a controlled single-request run at the target sequence length",
            "Long-context accuracy eval if KV quantization is enabled",
        ],
        "confidence": 0.9 if src_ok else 0.6,
    }


def main():
    recs = []
    with open(SRC) as f:
        for i, line in enumerate(f, 1):
            if i < START:
                continue
            if i > END:
                break
            d = json.loads(line)
            msgs = d["messages"]
            user = next(m["content"] for m in msgs if m["role"] == "user")
            asst = next(m["content"] for m in msgs if m["role"] == "assistant")
            recs.append(build(d["id"], user, asst))
    with open(OUT, "w") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(recs)} records to {OUT}")


if __name__ == "__main__":
    main()
