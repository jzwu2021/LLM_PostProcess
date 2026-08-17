import json, re, os

SRC="/media/home/johnson/workspace/LLM_PostProcess/research/ai-infra-expert/corpus/train.jsonl"
OUT="/media/home/johnson/workspace/LLM_PostProcess/experiments/2026-08-17-teacher-b-corpus-review/results/train-batch-0069.jsonl"
lines=open(SRC,encoding="utf-8").read().splitlines()[680:690]

recs=[]
for l in lines:
    d=json.loads(l)
    m={x["role"]:x["content"] for x in d["messages"]}
    u,a=m["user"],m["assistant"]
    L=int(re.search(r"(\d+) layers",u).group(1))
    KV=int(re.search(r"(\d+) KV heads",u).group(1))
    HD=int(re.search(r"head dimension (\d+)",u).group(1))
    S=int(re.search(r"sequence length (\d+)",u).group(1))
    int8 = "INT8" in u
    B=1 if int8 else 2
    dt="INT8 (1 B/value)" if int8 else "BF16/FP16 (2 B/value)"
    total=2*L*S*KV*HD*B
    gib=total/2**30
    mib=total/2**20
    per_tok=2*L*KV*HD*B
    # arithmetic check against source
    src_bytes=int(re.search(r"=\s*(\d+) bytes",a).group(1))
    ok = (src_bytes==total)
    tok_per_gib = int(2**30//per_tok)
    ca = (
"Assumptions to verify against the model config.json and the serving engine's KV dtype flag before any capacity decision uses this number: "
"standard MHA/GQA attention with exactly one K and one V tensor per layer; %d layers; %d KV heads (under GQA/MQA the KV footprint scales with KV heads, not query heads - substituting query heads is the most common integer-factor error here); head_dim %d; one request holding its full %d-token context; KV dtype %s; 1 GiB = 2^30 B; no prefix/block sharing, no KV offload to host, no quantization scale/zero-point side tensors counted.\n\n"
"Formula: bytes = 2 (K and V) x layers x seq_len x kv_heads x head_dim x bytes_per_value.\n"
"Substituting: 2 x %d x %d x %d x %d x %d = %d B = %.6f GiB (%.3f MiB).\n"
"Derived rate: %d B per token per request, so roughly %d tokens of KV per GiB of KV budget.\n\n"
"Boundary conditions that make the real number larger: PagedAttention/vLLM block allocation rounds each sequence up to a whole number of blocks (block_size 16 or 32 tokens), so short or ragged sequences waste up to block_size-1 tokens each; %s"
"speculative decoding, beam search, or n>1 sampling multiply live KV by the number of active branches; a prefill/decode-disaggregated stack (NVIDIA Dynamo, Mooncake) holds a second transient copy of the same KV while it is transferred over the fabric, so peak resident KV across both tiers can approach 2x this figure during the transfer window.\n\n"
"What this figure does NOT include and must not be used to size a GPU alone: model weights, activation/workspace buffers, CUDA graph pools, NCCL communication buffers, the allocator's fragmentation headroom, and the framework's reserved-but-unallocated pool. Under tensor parallelism TP=N the KV is sharded across KV heads, so per-GPU KV is this value divided by N only while N divides %d evenly; otherwise heads are replicated or padded and per-GPU KV is higher than the naive division suggests.\n\n"
"Falsifiable check: run one request at exactly %d tokens on an idle GPU and diff torch.cuda.memory_allocated() (or the engine's reported KV blocks x block bytes) before and after prefill. The measured delta should match %d B to within allocator block granularity. If it is off by an integer factor, the KV-head vs query-head assumption or the dtype assumption is wrong; if it is off by a few percent, that is block rounding and is expected.\n\n"
"Evidence required: config.json (num_hidden_layers, num_key_value_heads, hidden_size/num_attention_heads for head_dim), the engine's KV cache dtype setting, block_size, and one measured memory delta. Rollback gate: if measured per-request KV exceeds this estimate by more than 15 percent at the target max sequence length, cut max_num_seqs (or max_num_batched_tokens) until measured peak KV plus weights plus a 10 percent allocator margin fits in device memory, and re-measure before restoring the previous concurrency."
) % (L,KV,HD,S,dt,L,S,KV,HD,B,total,gib,mib,per_tok,tok_per_gib,
     ("INT8 KV additionally requires per-block scale (and possibly zero-point) tensors that this formula omits, typically a low single-digit percent overhead; " if int8 else "switching KV to FP8/INT8 halves this figure but adds per-block scale tensors and can shift accuracy, which must be validated separately; "),
     KV,S,total)

    recs.append({
      "source_id": d["id"],
      "teacher_lane": "teacher-B",
      "teacher_model": "claude-opus-5-current",
      "calibration_status": "provisional",
      "decision": "rewrite",
      "source_user": u,
      "source_assistant": a,
      "corrected_answer": ca,
      "quality_dimensions": {
        "technical_correctness": 4 if ok else 2,
        "instruction_coverage": 3,
        "operational_safety": 2,
      },
      "risks": [
        "Source states the arithmetic but not the GQA/MQA caveat, so a reader may substitute query heads and undersize KV by an integer factor.",
        "Omits PagedAttention block-rounding, so per-sequence KV is understated for short or ragged sequences.",
        "No mention that TP sharding of KV heads only divides cleanly when TP divides num_key_value_heads.",
      ] + ([ "INT8 KV scale/zero-point side tensors are unaccounted, and INT8 KV accuracy impact is not flagged." ] if int8 else [
        "No note that peak memory during prefill/decode disaggregation transfer can approach twice the resident KV."
      ]),
      "evidence_required": [
        "config.json: num_hidden_layers, num_key_value_heads, head_dim",
        "Serving engine KV cache dtype and block_size settings",
        "Measured torch.cuda.memory_allocated() delta for one full-length request",
        "Tensor-parallel degree and resulting KV head sharding map",
      ],
      "confidence": 0.9 if ok else 0.55,
    })

with open(OUT,"w",encoding="utf-8") as f:
    for r in recs:
        f.write(json.dumps(r,ensure_ascii=False)+"\n")
print("wrote",len(recs),OUT)
print("arith_all_ok", all(r["quality_dimensions"]["technical_correctness"]==4 for r in recs))
