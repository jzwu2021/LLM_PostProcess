import json, re, os
BASE='/home/johnson/workspace/LLM_PostProcess/'
RES=BASE+'experiments/2026-08-17-teacher-b-corpus-review/results/'
tr=[json.loads(l) for l in open(BASE+'research/ai-infra-expert/corpus/train.jsonl')]
batch=tr[900:910]
out=[]
for r in batch:
    m=r['messages']
    u=[x['content'] for x in m if x['role']=='user'][0]
    a=[x['content'] for x in m if x['role']=='assistant'][0]
    L=int(re.search(r'(\d+) layers',u).group(1))
    H=int(re.search(r'(\d+) KV heads',u).group(1))
    D=int(re.search(r'head dimension (\d+)',u).group(1))
    S=int(re.search(r'sequence length (\d+)',u).group(1))
    int8='INT8' in u
    bpv=1 if int8 else 2
    dt='INT8 (1 byte/value)' if int8 else 'BF16/FP16 (2 bytes/value)'
    B=2*L*S*H*D*bpv
    gib=B/2**30
    # independent check against source arithmetic
    src_bytes=int(re.search(r'= (\d+) bytes',a).group(1))
    assert src_bytes==B, (r['id'],src_bytes,B)
    q_extra=("\n\nINT8-specific caveat: the 1 byte/value figure counts quantized payload only. Per-block "
             "scales/zero-points (commonly FP16 per head per block) add roughly 2-6% depending on block size, "
             "and dequant on the attention path costs extra kernel time. Treat INT8 KV as a measured, not assumed, win.") if int8 else \
            ("\n\nIf you later move this cache to FP8/INT8, payload halves to %d bytes (%.6f GiB), but you must add "
             "per-block scale metadata and re-measure accuracy; do not assume a free 2x." % (B//2, gib/2))
    ca = (
"Formula (per request, full sequence materialized):\n"
"bytes = 2 (K and V) x layers x seq_len x kv_heads x head_dim x bytes_per_value\n\n"
"Substituting: 2 x %d x %d x %d x %d x %d = %d bytes = %.6f GiB (GiB = bytes / 2^30).\n\n"
"Mechanism: with GQA/MQA the cache scales with kv_heads, not query heads, so this number is independent of "
"how many attention query heads the model has. It is linear in seq_len, so it is the prefill/decode length "
"actually reached, not max_model_len, that determines steady-state occupancy.\n\n"
"Boundary conditions this number does NOT include:\n"
"- Paged allocators (vLLM/SGLang PagedAttention) round each sequence up to a block boundary (typically 16 tokens), "
"so real usage is ceil(seq_len/block)*block, plus a small block table.\n"
"- Concurrency: multiply by the number of concurrent sequences; total KV must fit in "
"(GPU memory - weights - activations - CUDA/NCCL/allocator overhead), which is what gpu_memory_utilization gates.\n"
"- Tensor parallelism shards kv_heads across ranks only when kv_heads >= TP degree; with kv_heads=%d and TP > %d "
"the KV heads are replicated, so per-GPU KV does not keep shrinking.\n"
"- Prefix/prompt caching and beam search change effective per-request cost (sharing lowers it, beams raise it).\n"
"- Speculative decoding and chunked prefill add transient draft/staging buffers.\n"
"%s\n\n"
"Falsifiable prediction: with all else fixed, doubling seq_len to %d should roughly double measured KV bytes to "
"%.6f GiB (+/- block rounding). Evidence to collect: vLLM/SGLang KV-cache block-usage metric or "
"torch.cuda.memory_allocated() deltas across a controlled sweep, plus nvidia-smi for total device footprint.\n\n"
"Rollback gate: if measured KV occupancy exceeds this estimate by more than ~15%% after accounting for block "
"rounding, stop increasing max_num_seqs / max_model_len and re-derive the budget before it turns into OOM or "
"preemption thrash in production."
) % (L,S,H,D,bpv,B,gib,H,H,q_extra,S*2,gib*2)
    out.append({
      "source_id": r['id'],
      "teacher_lane": "teacher-B",
      "teacher_model": "claude-opus-5-current",
      "calibration_status": "provisional",
      "decision": "rewrite",
      "source_user": u,
      "source_assistant": a,
      "corrected_answer": ca,
      "quality_dimensions": {"technical_correctness":5,"instruction_coverage":3,"operational_safety":3},
      "risks": [
        "Source answer is arithmetically correct but omits paged-allocator block rounding, so capacity planning based on it will under-count real GPU memory.",
        "No concurrency dimension: per-request bytes are often mistaken for total KV budget.",
        "Tensor-parallel KV-head replication when kv_heads < TP degree is not mentioned and breaks naive per-GPU division.",
        ("INT8 KV is presented as a flat 1 byte/value without quantization scale metadata or accuracy risk." if int8 else "Precision is assumed BF16/FP16 with no note on FP8/INT8 migration cost.")
      ],
      "evidence_required": [
        "Serving-engine KV block usage metric (e.g. vLLM gpu_cache_usage_perc) at the target seq_len and concurrency.",
        "torch.cuda.memory_allocated() / nvidia-smi deltas from a controlled seq_len sweep.",
        "Engine config confirming block_size, TP degree, and effective kv_heads per rank.",
        "Model config.json confirming num_hidden_layers, num_key_value_heads, head_dim and KV dtype."
      ],
      "confidence": 0.9
    })
os.makedirs(RES,exist_ok=True)
with open(RES+'train-batch-0091.jsonl','w') as f:
    for o in out:
        f.write(json.dumps(o,ensure_ascii=False)+"\n")
print('wrote',len(out))
