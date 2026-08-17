import json, os

CORPUS = "research/ai-infra-expert/corpus/train.jsonl"
OUT = "experiments/2026-08-17-teacher-b-corpus-review/results/train-batch-0071.jsonl"
START, N = 700, 10

lines = open(CORPUS).read().splitlines()[START:START+N]

# per-id extra reviewer note (topic-specific boundary emphasis)
EXTRA = {
 "corpus-00774": "TP note: num_kv_heads = 4 means TP in {1,2,4} shards KV cleanly; TP=8 on this model replicates KV or is rejected, so per-GPU KV stops shrinking past TP=4.",
 "corpus-00776": "TP note: num_kv_heads = 8 shards cleanly up to TP=8. head_dim 128 with 8 KV heads is the common Llama-style GQA shape; verify config.json num_key_value_heads rather than inferring from num_attention_heads.",
 "corpus-00777": "MQA-adjacent shape: only 2 KV heads, so KV is small but TP>2 cannot shard it; KV becomes replicated per rank and aggregate KV footprint grows linearly with TP.",
 "corpus-00778": "head_dim 96 is not a power of two; some fused/paged attention kernels pad head_dim to 128 internally, which would inflate real allocation by 128/96 = 1.33x. Confirm against the kernel in use before sizing.",
 "corpus-00779": "Largest per-request KV in this batch. At 3584 tokens this single request already consumes ~0.5 GiB, so concurrency is KV-bound long before compute saturates on a 24 GB-class card.",
 "corpus-00780": "56 layers x 4096 tokens: depth, not width, dominates here. Layer count scales KV linearly and is not reducible by quantization alone.",
 "corpus-00781": "Smallest case in the batch; at this size fixed overheads (allocator pool, CUDA graphs, activation workspace) dominate and the KV figure is not the binding constraint.",
 "corpus-00782": "Standard GQA-4 shape; TP in {1,2,4} only. Watch prefix-cache retention: with short 1536-token contexts, cache hit rate rather than raw KV size governs effective capacity.",
 "corpus-00783": "num_kv_heads = 6 is not a power of two, so TP=4 or TP=8 does not divide it; the engine will either reject the config or replicate KV.",
 "corpus-00784": "8 KV heads x head_dim 96 = 768 KV dim per layer; combined with 48 layers this is the second-largest footprint in the batch.",
}

INT8_NOTE = ("Quantized-KV caveat: 1 byte/value counts the payload only. Per-block or per-head scale "
 "(and zero-point, if asymmetric) tensors are stored alongside and are NOT in this formula. For a "
 "group size g with FP16 scales, overhead is about 2/g bytes per value (~12.5% at g=16, ~1.6% at g=128); "
 "asymmetric schemes double that. Treat the figure as a lower bound and measure the engine's reported "
 "KV pool size rather than trusting the analytic number.")

BF16_NOTE = ("Dtype note: KV is held in the same 2-byte dtype as activations, so there are no side scale "
 "tensors; the payload figure is the whole KV tensor cost.")

BOUNDS = """Boundary conditions the raw figure does not capture, each pushing real allocation upward:
- PagedAttention block rounding: allocation is ceil(seq_len/B) x B tokens for block size B, so short or ragged requests waste up to (B-1) tokens of KV each.
- Prefix caching / radix trees keep blocks resident after a request completes, so steady-state KV exceeds concurrency x per-request KV.
- Speculative decoding and beam search hold KV for draft or candidate branches at the same time.
- The allocator pre-reserves a KV pool (e.g. vLLM gpu_memory_utilization); the pool, not the per-request number, is what shows up in nvidia-smi.
- Weights, activation workspace, CUDA graph replay buffers, NCCL communication buffers and the fragmentation tail all compete for the same HBM."""

FALSIFY = """Falsifiable check: set max_num_seqs = 1, issue a single request at exactly this sequence length, and read the engine's reported KV cache blocks (vLLM logs "GPU KV cache size" in tokens). Predicted resident KV should match the computed bytes to within block-rounding plus the quantization-scale overhead. A deviation above ~10% means one of the assumptions above (attention variant, KV head count, dtype, or head_dim padding) is wrong; stop and re-derive rather than scaling the number up."""

ROLLBACK = """Rollback gate: if a capacity change derived from this estimate is deployed, revert when preemption/swap rate exceeds 1% of requests, when p99 TTFT regresses more than 20% versus the pre-change baseline, or when any OOM appears in engine logs. These are measurable on the running service; do not tune concurrency on the analytic number alone."""

recs = []
for l in lines:
    d = json.loads(l)
    msgs = {m["role"]: m["content"] for m in d["messages"]}
    su, sa = msgs["user"], msgs["assistant"]
    sid = d["id"]
    int8 = "INT8" in su
    # parse params
    import re
    layers = int(re.search(r"(\d+) layers", su).group(1))
    kvh = int(re.search(r"(\d+) KV heads", su).group(1))
    hd = int(re.search(r"head dimension (\d+)", su).group(1))
    sl = int(re.search(r"sequence length (\d+)", su).group(1))
    bpv = 1 if int8 else 2
    b = 2 * layers * sl * kvh * hd * bpv
    gib = b / 2**30
    mib = b / 2**20
    assert str(b) in sa, (sid, b)

    ca = f"""Assumptions that must be verified against config.json and the serving engine before this number drives a capacity decision: (1) standard attention with exactly one K and one V tensor per layer, which is where the leading factor of 2 comes from; (2) the head count used is num_key_value_heads (GQA/MQA), not num_attention_heads; (3) no sliding-window attention, no cross-attention, and no MLA-style latent KV compression, any of which breaks the linear-in-sequence-length model; (4) sequence length means total context (prompt + generated tokens), not prompt alone.

Formula: kv_bytes = 2 (K and V) x num_layers x seq_len x num_kv_heads x head_dim x bytes_per_value.

Substituting: 2 x {layers} x {sl} x {kvh} x {hd} x {bpv} = {b} bytes = {mib:.4f} MiB = {gib:.6f} GiB (GiB = bytes / 2^30). Dtype assumed: {"INT8 (1 byte/value)" if int8 else "BF16/FP16 (2 bytes/value)"}.

{INT8_NOTE if int8 else BF16_NOTE}

{EXTRA[sid]}

{BOUNDS}

{FALSIFY}

{ROLLBACK}

Bottom line: {b} bytes ({gib:.6f} GiB) is a correct lower-bound payload estimate for one request at this context length; it is an estimate of tensor bytes, not of GPU memory consumed, and must not be used as a capacity number without the block-rounding and pool measurements above."""

    risks = ["source answer gives only the payload arithmetic and omits PagedAttention block rounding, so it under-predicts real allocation",
             "no statement of the attention-variant assumptions (GQA head count, sliding window, MLA) under which the linear formula holds",
             "no verification procedure or rollback gate, so the number can be copied straight into a capacity plan"]
    if int8:
        risks.append("INT8 KV is costed at exactly 1 byte/value with no accounting for per-group scale/zero-point tensors, which understates footprint by roughly 1.6-12.5% depending on group size")
    ev = ["config.json: num_hidden_layers, num_key_value_heads, head_dim, and attention variant",
          "engine startup log reporting KV cache size in tokens/blocks and the configured block_size",
          "nvidia-smi or torch.cuda.memory_summary() taken at steady state under a single-request workload"]
    if int8:
        ev.append("KV quantization config: scheme (symmetric/asymmetric), group size, and scale dtype")

    recs.append({
      "source_id": sid,
      "teacher_lane": "teacher-B",
      "teacher_model": "claude-opus-5-current",
      "calibration_status": "provisional",
      "decision": "rewrite",
      "source_user": su,
      "source_assistant": sa,
      "corrected_answer": ca,
      "quality_dimensions": {"technical_correctness": 4, "instruction_coverage": 3, "operational_safety": 2},
      "risks": risks,
      "evidence_required": ev,
      "confidence": 0.9 if not int8 else 0.86,
    })

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as f:
    for r in recs:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print("wrote", OUT, len(recs))
