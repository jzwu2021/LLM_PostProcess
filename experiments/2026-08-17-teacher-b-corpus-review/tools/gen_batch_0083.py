import json

START, N = 820, 10
OUT = 'experiments/2026-08-17-teacher-b-corpus-review/results/train-batch-0083.jsonl'

params = {
 'corpus-00904': (48, 8, 96, 3072, 2, 'BF16/FP16'),
 'corpus-00905': (56, 2, 128, 3584, 2, 'BF16/FP16'),
 'corpus-00906': (24, 4, 64, 4096, 1, 'INT8'),
 'corpus-00907': (32, 6, 96, 1024, 2, 'BF16/FP16'),
 'corpus-00908': (40, 8, 128, 1536, 2, 'BF16/FP16'),
 'corpus-00909': (48, 2, 64, 2048, 1, 'INT8'),
 'corpus-00910': (56, 4, 96, 2560, 2, 'BF16/FP16'),
 'corpus-00911': (24, 6, 128, 3072, 2, 'BF16/FP16'),
 'corpus-00912': (32, 8, 64, 3584, 1, 'INT8'),
 'corpus-00913': (40, 2, 96, 4096, 2, 'BF16/FP16'),
}

def answer(cid):
    L, H, D, S, B, dt = params[cid]
    total = 2 * L * S * H * D * B
    gib = total / 2**30
    per_layer = 2 * S * H * D * B
    per_tok = 2 * L * H * D * B
    kv_dim = H * D
    q8 = ''
    if dt == 'INT8':
        q8 = (" Because this configuration stores INT8 KV, the 1 byte/value assumption is only valid if the "
              "runtime keeps per-token or per-head scale/zero-point metadata outside the tensor; if scales are "
              "stored inline (a common fp16 scale per head per token) add 2 bytes x %d layers x %d heads x %d tokens "
              "= %d bytes (%.4f GiB), roughly %.1f%% overhead, and the reported figure is an underestimate. "
              "INT8 KV also changes accuracy, not just capacity, so it must be gated on an eval, not on a memory win alone."
              % (L, H, S, 2 * L * H * S, (2 * L * H * S) / 2**30, 100.0 * (2 * L * H * S) / total))
    else:
        q8 = (" With %s KV the 2 bytes/value assumption holds exactly; no scale metadata is stored, so the only "
              "discrepancy against nvidia-smi will come from allocator granularity and non-KV memory." % dt)

    return (
"Formula and result. The KV cache for a single request is bytes = 2 x layers x sequence_length x kv_heads x head_dim x bytes_per_value. "
"The leading 2 is the K and V tensors, not a batch or bidirectional factor. Substituting layers=%d, sequence_length=%d, kv_heads=%d, "
"head_dim=%d, bytes_per_value=%d gives 2 x %d x %d x %d x %d x %d = %d bytes = %.6f GiB (binary GiB = 2^30 bytes; in decimal GB that is %.6f GB). "
"Useful derived quantities: %d bytes per layer for the full sequence, and %d bytes per token across all layers, i.e. %.2f KiB/token. "
"The per-token figure is the number to use for admission control, because it is independent of the 4096-vs-3072 context assumption.\n\n"
"Mechanism. During autoregressive decode, attention at step t must score the current query against every previous key and read every previous "
"value. Recomputing them would make decode O(t^2) in FLOPs per step, so the K and V projections are materialized once and retained. Under "
"grouped-query or multi-query attention the retained width is kv_heads x head_dim = %d, not num_query_heads x head_dim: query heads are "
"replicated onto KV groups at attention time, which is why kv_heads and not the attention head count appears in the formula. This is the single "
"most common error in capacity planning for GQA models.\n\n"
"Boundary conditions and what this number excludes.%s Paged attention allocators (vLLM/SGLang block managers) round each sequence up to a whole "
"block, so a sequence of %d tokens with block_size 16 costs ceil(%d/16) = %d blocks and any partially filled final block is still charged; "
"expect low single-digit percent internal fragmentation, worse for many short sequences. The figure also excludes model weights, activation and "
"workspace buffers, CUDA context (~300-600 MiB/GPU), NCCL and communication buffers, and fragmentation in the caching allocator. Under tensor "
"parallelism of degree TP the KV cache is sharded across KV heads, so per-GPU cost is this value divided by TP only while TP <= %d; beyond that "
"KV heads must be replicated and per-GPU KV stops shrinking. Speculative decoding, beam search with width k, and prefix-cache retention all "
"multiply the live KV footprint above this single-sequence number.\n\n"
"Falsifiable prediction and evidence to collect. Prediction: on an idle server, serving one request of exactly %d tokens should raise reported "
"KV-cache usage by %.6f GiB +/- one allocator block, and concurrency C should scale it linearly to C x %.6f GiB until the KV pool is exhausted. "
"Evidence: the served model config (num_hidden_layers, num_key_value_heads, head_dim or hidden_size/num_attention_heads), the runtime's KV dtype "
"flag, the engine's reported total KV blocks and block size, torch.cuda.memory_summary or the engine's gpu_cache_usage metric before and after, "
"and nvidia-smi as a cross-check. If measured growth exceeds the prediction by more than ~10%%, the likely causes in order are: KV dtype is not "
"what was assumed, kv_heads was confused with attention heads, or block-level rounding.\n\n"
"Rollback gate. Do not size max_num_seqs or gpu_memory_utilization from this arithmetic alone. Set the limit from measured steady-state usage at "
"target concurrency plus a headroom margin, and roll back any capacity increase if KV-cache utilization exceeds ~90%% of the pool, if preemption "
"or recompute events appear in engine logs, or if p99 TTFT regresses beyond the SLO, since KV exhaustion degrades as queueing and preemption "
"rather than as a clean OOM."
    ) % (L, S, H, D, B, L, S, H, D, B, total, gib, total / 1e9,
         per_layer, per_tok, per_tok / 1024.0, kv_dim, q8, S, S, (S + 15) // 16, H, S, gib, gib)


rows = [json.loads(l) for l in open('research/ai-infra-expert/corpus/train.jsonl')][START:START + N]
recs = []
for r in rows:
    cid = r['id']
    u = [m['content'] for m in r['messages'] if m['role'] == 'user'][0]
    a = [m['content'] for m in r['messages'] if m['role'] == 'assistant'][0]
    L, H, D, S, B, dt = params[cid]
    risks = [
        "Source answer gives only arithmetic with no mechanism, so a learner cannot generalize to GQA/MQA or paged allocators.",
        "kv_heads may be confused with attention head count, which overestimates KV by the GQA replication factor.",
        "Number is per-request and excludes weights, activations, CUDA context, NCCL buffers and allocator fragmentation; using it directly for capacity planning risks OOM or preemption.",
    ]
    if dt == 'INT8':
        risks.append("INT8 KV at 1 byte/value ignores quantization scale/zero-point metadata and the accuracy regression that INT8 KV can cause.")
    else:
        risks.append("Assumes no KV quantization and no speculative/beam expansion of the live KV set.")
    recs.append({
        "source_id": cid,
        "teacher_lane": "teacher-B",
        "teacher_model": "claude-opus-5-current",
        "calibration_status": "provisional",
        "decision": "rewrite",
        "source_user": u,
        "source_assistant": a,
        "corrected_answer": answer(cid),
        "quality_dimensions": {"technical_correctness": 5, "instruction_coverage": 3, "operational_safety": 3},
        "risks": risks,
        "evidence_required": [
            "Served model config: num_hidden_layers, num_key_value_heads, head_dim (or hidden_size/num_attention_heads).",
            "Runtime KV dtype flag (e.g. --kv-cache-dtype) and whether quantization scales are stored inline.",
            "Engine KV pool report: total KV blocks, block_size, gpu_cache_usage before and after a single %d-token request." % S,
            "torch.cuda.memory_allocated / memory_summary and nvidia-smi cross-check on an otherwise idle GPU.",
            "Tensor-parallel degree and whether TP exceeds kv_heads=%d (point where KV heads are replicated)." % H,
        ],
        "confidence": 0.9,
    })

with open(OUT, 'w') as f:
    for rec in recs:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
print("wrote", OUT, len(recs))
